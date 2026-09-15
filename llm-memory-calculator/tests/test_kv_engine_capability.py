"""The sliding-window KV saving is an ENGINE property, not a model one.

Measured in-cluster on an H100XM-80C vGPU, vLLM at tp=1 serving gpt-oss-20b:
**49,254 B/token** of KV. The model has 24 layers, 8 KV heads, head_dim 64, and
12 of those 24 layers declare a 128-token window, so the two candidate answers
are far apart:

    every layer full      2 * 24 * 8 * 64 * 2 = 49,152 B/token   (0.2% off measured)
    12 full + 12 clamped  2 * 12 * 8 * 64 * 2 = 24,576 B/token   (2.00x UNDER)

The engine took no window saving whatsoever: vLLM rewrites a windowed layer's
cache spec to a full one whenever it cannot manage a heterogeneous KV layout,
and only its hybrid KV-cache allocator makes the windowed layers cheaper. The
calculator used to assume the saving unconditionally. Under-prediction is the
dangerous direction -- it under-sizes the GPU slice and the pod either starves
for blocks or OOMs -- so the saving is now opt-in per engine and OFF by default.

A stack whose attention layers ALL share one window (Mistral-style) is not
heterogeneous: there is nothing for the engine to reconcile and every engine
allocates the window, so those keep the saving with no capability at all.
"""

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.calculator import (
    EngineKVCapabilities,
    ModelMemoryCalculator,
)

HYBRID_ALLOCATOR = {"heterogeneous_kv_layout": True}

# openai/gpt-oss-20b, verbatim from the checkpoint served in the measurement
# (/data/models-registry/openai_gpt-oss-20b_3cada3df/config.json). The shipped
# 24-entry layer_types alternates, starting windowed.
GPT_OSS_20B = dict(
    model_type="gpt_oss",
    hidden_size=2880,
    intermediate_size=2880,
    num_hidden_layers=24,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    sliding_window=128,
    num_local_experts=32,
    num_experts_per_tok=4,
    vocab_size=201088,
    max_position_embeddings=131072,
    layer_types=[
        "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(24)
    ],
)

MEASURED_GPT_OSS_20B_KV_PER_TOKEN = 49_254  # vLLM, H100XM-80C vGPU, tp=1

PER_LAYER_TOKEN = 2 * 8 * 64 * 2  # K+V, 8 kv heads, head_dim 64, bf16 = 2048 B


def kv_bytes(config, batch_size, seq_length, precision="bf16", engine=None):
    calc = ModelMemoryCalculator()
    calc.model_type = calc.detect_model_type(config)
    calc.attention_type = calc.detect_attention_type(config)
    return (
        calc.calculate_kv_cache(
            config, batch_size, seq_length, precision, engine_capabilities=engine
        )
        * 1e9
    )


# ------------------------------------------------------- the measured default


def test_gpt_oss_charges_every_layer_full_kv_by_default():
    """49,152 B/token: all 24 layers, no window saving. Matches the measurement."""
    assert kv_bytes(GPT_OSS_20B, 1, 1) == 24 * PER_LAYER_TOKEN == 49_152
    # and it stays linear in context, which is what the clamp used to break
    assert kv_bytes(GPT_OSS_20B, 1, 8192) == 24 * PER_LAYER_TOKEN * 8192
    assert kv_bytes(GPT_OSS_20B, 1, 131072) == 24 * PER_LAYER_TOKEN * 131072


def test_default_prediction_matches_the_measured_engine_within_a_percent():
    predicted = kv_bytes(GPT_OSS_20B, 1, 8192) / 8192
    assert predicted == pytest.approx(MEASURED_GPT_OSS_20B_KV_PER_TOKEN, rel=0.01)


def test_taking_the_window_saving_would_under_predict_by_two_x():
    """The old behavior, pinned as the failure it is."""
    windowed = kv_bytes(GPT_OSS_20B, 1, 8192, engine=HYBRID_ALLOCATOR) / 8192
    assert MEASURED_GPT_OSS_20B_KV_PER_TOKEN / windowed == pytest.approx(2.0, rel=0.02)


def test_the_saving_is_available_when_the_engine_is_declared_capable():
    seq = 8192
    expected = PER_LAYER_TOKEN * (12 * seq + 12 * 128)
    assert kv_bytes(GPT_OSS_20B, 1, seq, engine=HYBRID_ALLOCATOR) == expected


def test_below_the_window_the_capability_changes_nothing():
    """A window longer than the context clamps nothing, whatever the engine."""
    seq = 64  # shorter than the 128-token window
    assert kv_bytes(GPT_OSS_20B, 1, seq) == kv_bytes(
        GPT_OSS_20B, 1, seq, engine=HYBRID_ALLOCATOR
    )


# ---------------------------------------------- homogeneous windows are not gated

MISTRAL_LIKE = dict(
    model_type="mistral",
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    sliding_window=4096,
    max_position_embeddings=32768,
)


def test_a_uniform_window_keeps_its_saving_with_no_capability():
    """Every attention layer shares one window, so nothing is heterogeneous.

    Charging the full context here would over-predict 8x at 32k and make a model
    that fits look unschedulable. Engines have honored a uniform window since
    Mistral-7B; the gate exists for stacks that MIX window sizes.
    """
    per_layer_token = 2 * 8 * 128 * 2
    assert kv_bytes(MISTRAL_LIKE, 1, 32768) == per_layer_token * 32 * 4096
    assert kv_bytes(MISTRAL_LIKE, 1, 32768, engine=HYBRID_ALLOCATOR) == kv_bytes(
        MISTRAL_LIKE, 1, 32768
    )


def test_a_plain_model_is_untouched_by_the_gate():
    llama = dict(
        model_type="llama",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    assert kv_bytes(llama, 1, 8192) == kv_bytes(llama, 1, 8192, engine=HYBRID_ALLOCATOR)


# ------------------------------------------------------------ capability plumbing


def test_capability_accepts_bool_mapping_and_instance_alike():
    seq = 8192
    by_bool = kv_bytes(GPT_OSS_20B, 1, seq, engine=True)
    by_mapping = kv_bytes(GPT_OSS_20B, 1, seq, engine=HYBRID_ALLOCATOR)
    by_instance = kv_bytes(
        GPT_OSS_20B, 1, seq, engine=EngineKVCapabilities(heterogeneous_kv_layout=True)
    )
    assert by_bool == by_mapping == by_instance
    assert kv_bytes(GPT_OSS_20B, 1, seq, engine=False) == kv_bytes(GPT_OSS_20B, 1, seq)


def test_a_misspelled_capability_raises_instead_of_being_ignored():
    """Silently defaulting is how a confident wrong number gets shipped."""
    with pytest.raises(ValueError, match="Unknown engine capability"):
        kv_bytes(GPT_OSS_20B, 1, 8192, engine={"sliding_window_kv": True})
    with pytest.raises(TypeError):
        kv_bytes(GPT_OSS_20B, 1, 8192, engine="vllm")


def test_default_capabilities_are_all_off():
    assert EngineKVCapabilities() == EngineKVCapabilities(
        heterogeneous_kv_layout=False
    )
    assert EngineKVCapabilities.resolve(None).heterogeneous_kv_layout is False


# -------------------------------------------------------------- end to end wiring


def _report_kv(engine=None):
    return calculate_memory(
        GPT_OSS_20B,
        batch_size=1,
        seq_length=8192,
        precision="bf16",
        engine_capabilities=engine,
    )


def test_capability_threads_through_calculate_memory():
    seq = 8192
    conservative = _report_kv().kv_cache_bytes
    capable = _report_kv(HYBRID_ALLOCATOR).kv_cache_bytes
    assert conservative == 24 * PER_LAYER_TOKEN * seq
    assert capable == PER_LAYER_TOKEN * (12 * seq + 12 * 128)
    # ~1.97x here; it tends to 2x as the context grows past the window.
    assert conservative / capable == pytest.approx(1.97, rel=0.01)


def test_the_report_says_when_the_saving_was_withheld():
    withheld = _report_kv().notes
    assert any("windowed layers are charged the full" in n for n in withheld)
    assert not any(
        "windowed layers are charged the full" in n
        for n in _report_kv(HYBRID_ALLOCATOR).notes
    )


def test_breakdown_shows_the_layers_and_the_engine_decision():
    calc = ModelMemoryCalculator()
    default = calc.kv_cache_breakdown(GPT_OSS_20B, 1, 8192)
    assert (default["num_full_layers"], default["num_sliding_layers"]) == (12, 12)
    assert default["num_no_kv_layers"] == 0
    assert default["sliding_window"] == 128
    assert default["heterogeneous_attention"] is True
    assert default["sliding_window_saving_withheld"] is True
    assert default["marginal_bytes_per_token"] == 24 * PER_LAYER_TOKEN

    capable = calc.kv_cache_breakdown(
        GPT_OSS_20B, 1, 8192, engine_capabilities=HYBRID_ALLOCATOR
    )
    assert capable["sliding_window_saving_applied"] is True
    # Once the window is saturated only the 12 full layers keep growing.
    assert capable["marginal_bytes_per_token"] == 12 * PER_LAYER_TOKEN
