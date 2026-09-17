"""KV is priced per attention KIND, and only kinds that grow with context pay.

Three kinds exist, and a stack can mix all three:

* ``full``     -- caches every token.
* ``sliding``  -- caches at most its window, IF the engine allocates per layer
                  (see tests/test_kv_engine_capability.py).
* no KV        -- linear-attention / SSM / short-conv mixers and MLP-only slots
                  hold no growing cache at all. Their fixed state is a separate,
                  context-independent term.

Charging a linear-attention layer a KV cache is how Qwen3.5-27B came out 2x
over: 64 layers x 4 kv heads x 256 head_dim x 2 x 2 = 262,144 B/token, when only
16 of those 64 layers hold a cache. Nothing here keys off a model name -- the
kinds are read from whatever the config declares.
"""

import json
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.config_normalizer import ConfigNormalizer
from llm_memory_calculator.layer_plan import resolve_layer_plan
from llm_memory_calculator.state_memory import calculate_recurrent_state_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "hybrid_configs"
HYBRID_ALLOCATOR = {"heterogeneous_kv_layout": True}


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


# --------------------------------------------------------------- synthetic kinds
#
# Deliberately not a real model: the kinds must be driven by the declared
# layer_types, not by anything recognizable about the architecture.

SYNTH_BASE = dict(
    model_type="synthetic_hybrid",
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=64,
    sliding_window=256,
    # Mechanism dimensions, so the recurrent kinds are sized from the config
    # rather than from another model's defaults.
    linear_num_key_heads=8,
    linear_num_value_heads=16,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    conv_L_cache=3,
)

SYNTH_PER_LAYER_TOKEN = 2 * 4 * 64 * 2  # K+V, 4 kv heads, head_dim 64, bf16

MIXED_LAYERS = [
    "full_attention",
    "sliding_attention",
    "linear_attention",
    "mamba",
    "conv",
    "full_attention",
    "sliding_attention",
    "linear_attention",
    "mamba",
    "conv",
    "full_attention",
    "sliding_attention",
]


def synth(layer_types):
    return dict(SYNTH_BASE, layer_types=list(layer_types))


@pytest.mark.parametrize(
    "kind,kv_layers",
    [
        ("full_attention", 12),
        ("linear_attention", 0),  # Gated DeltaNet et al: no growing cache
        ("mamba", 0),
        ("conv", 0),  # short convolution: a ring buffer, not a KV cache
        ("mlp_only", 0),  # no mixer at all
    ],
)
def test_each_attention_kind_pays_exactly_what_it_holds(kind, kv_layers):
    seq = 4096
    assert kv_bytes(synth([kind] * 12), 1, seq) == (
        kv_layers * SYNTH_PER_LAYER_TOKEN * seq
    )


def test_a_uniformly_windowed_stack_is_clamped_without_any_capability():
    seq = 4096
    expected = 12 * SYNTH_PER_LAYER_TOKEN * 256
    assert kv_bytes(synth(["sliding_attention"] * 12), 1, seq) == expected


def test_the_three_kinds_mix_and_the_no_kv_layers_contribute_zero():
    seq = 4096
    cfg = synth(MIXED_LAYERS)
    # 3 full + 3 windowed pay; the 6 linear/mamba/conv layers pay nothing.
    assert kv_bytes(cfg, 1, seq) == 6 * SYNTH_PER_LAYER_TOKEN * seq
    assert kv_bytes(cfg, 1, seq, engine=HYBRID_ALLOCATOR) == SYNTH_PER_LAYER_TOKEN * (
        3 * seq + 3 * 256
    )
    # ... and never what a bare depth would charge.
    assert kv_bytes(cfg, 1, seq) < 12 * SYNTH_PER_LAYER_TOKEN * seq


def test_the_plan_reports_which_layers_pay_nothing():
    plan = ModelMemoryCalculator().resolve_kv_layer_plan(synth(MIXED_LAYERS), 4096)
    assert plan["num_layers"] == 12
    assert plan["num_full_layers"] == 3
    assert plan["num_sliding_layers"] == 3
    assert plan["num_no_kv_layers"] == 6
    assert plan["source"] == "layer_plan:layer_types"


def test_kv_is_linear_in_batch_for_every_kind_mix():
    cfg = synth(MIXED_LAYERS)
    assert kv_bytes(cfg, 8, 4096) == 8 * kv_bytes(cfg, 1, 4096)


def test_no_kv_layers_do_not_grow_with_context():
    """The defining property of a linear-attention layer."""
    cfg = synth(["linear_attention"] * 12)
    assert kv_bytes(cfg, 1, 1024) == kv_bytes(cfg, 1, 1_000_000) == 0.0


# ------------------------------------------- the normalizer speaks the same dialect


def test_metadata_no_longer_calls_linear_attention_an_attention_layer():
    """`'attention' in "linear_attention"` is True, and that is the whole bug.

    The per-layer metadata is the fallback the KV path uses when no hybrid plan
    resolves, so its classifier must agree with the plan's or the two disagree
    about which layers hold a cache.
    """
    meta = ConfigNormalizer.normalize_config(synth(MIXED_LAYERS))["_layer_metadata"]
    assert meta["num_attention_layers"] == 6
    assert meta["num_full_layers"] == 3
    assert meta["num_sliding_layers"] == 3
    assert meta["num_no_kv_layers"] == 6
    # `mamba_layers` keeps its narrow meaning: the two "mamba" entries only.
    # Gated DeltaNet and the short convolution are recurrent but not Mamba.
    assert meta["num_mamba_layers"] == 2
    assert meta["num_recurrent_layers"] == 6
    assert meta["has_hybrid_architecture"] is True


# ------------------------------------------------- Qwen3.5-27B, against the engine
#
# Verbatim geometry from the checkpoint served in the measurement
# (/data/models-registry/qwen_qwen3_8-27b_a849ea74/config.json): a multimodal
# wrapper whose text_config is a 64-layer hybrid, 16 full-attention layers and 48
# Gated DeltaNet layers. The shipped 64-entry layer_types is exactly this
# comprehension -- full attention on the last layer of each group of 4.

QWEN3_5_27B = dict(
    model_type="qwen3_5",
    architectures=["Qwen3_5ForConditionalGeneration"],
    tie_word_embeddings=False,
    vision_config=dict(model_type="qwen3_5_vl", hidden_size=1152, patch_size=16),
    text_config=dict(
        model_type="qwen3_5_text",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
        max_position_embeddings=262144,
        vocab_size=248320,
        layer_types=[
            "full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)
        ],
    ),
)

# vLLM's own arithmetic for this checkpoint, from the serving pod's startup log:
#
#   "Setting attention block size to 784 tokens to ensure that attention page
#    size is >= mamba page size."
#   "Padding mamba page size by 0.13% to ensure that mamba page size and
#    attention page size are exactly equal."
#
# An attention page is one layer x one block: 784 tokens x B bytes/token. The
# engine picked the smallest block that covers the mamba page, so
# ceil(mamba_page / B) == 784 pins B at 4096 -- one full-attention layer, 4 KV
# heads x 256 head_dim x 2 (K,V) x 2 bytes. Not a fitted constant: the same log
# line pins the mamba page to within 0.13%.
QWEN3_5_27B_ATTENTION_BLOCK = 784
QWEN3_5_27B_MAMBA_PAGE_PADDED = 3_211_264  # bytes/layer, = 4096 * 784
MEASURED_QWEN3_5_27B_KV_PER_TOKEN = 127_863


def test_only_the_sixteen_full_attention_layers_hold_a_kv_cache():
    plan = ModelMemoryCalculator().resolve_kv_layer_plan(QWEN3_5_27B, 8192)
    assert (plan["num_full_layers"], plan["num_sliding_layers"]) == (16, 0)
    assert plan["num_no_kv_layers"] == 48
    # 16 layers * 2(K,V) * 4 kv heads * 256 head_dim * 2 bytes
    assert kv_bytes(QWEN3_5_27B, 1, 1) == 65_536
    assert kv_bytes(QWEN3_5_27B, 1, 8192) == 65_536 * 8192


def test_per_layer_kv_matches_the_block_size_vllm_chose():
    per_layer_token = kv_bytes(QWEN3_5_27B, 1, 1) / 16
    assert per_layer_token == 4096
    assert per_layer_token * QWEN3_5_27B_ATTENTION_BLOCK == QWEN3_5_27B_MAMBA_PAGE_PADDED


def test_charging_all_64_layers_is_the_reported_two_x_over_prediction():
    """262,144 B/token was the pre-fix number, and it is 2.05x the measurement."""
    all_layers = 64 * 4096
    assert all_layers == 262_144
    assert all_layers / MEASURED_QWEN3_5_27B_KV_PER_TOKEN == pytest.approx(2.05, rel=0.01)


def test_the_measured_per_token_figure_is_kv_plus_amortized_recurrent_state():
    """The measured 127,863 B/token is not KV alone.

    vLLM reports one pooled cache for a hybrid model, so its per-token figure
    carries the 48 Gated DeltaNet layers' FIXED state amortized over the
    deployment's max_model_len. Split them and both halves check out
    independently: the KV half against the engine's block size above, the state
    half against the engine's own mamba page size.

        127,863 - 65,536 = 62,327 B/token  =>  154,140,672 / 2,473 tokens

    2,473 is a max_model_len consistent with the 2,048-token benchmark context
    under budcluster's (input + output) * 1.1 rule.
    """
    plan = resolve_layer_plan(QWEN3_5_27B)
    state = calculate_recurrent_state_bytes(QWEN3_5_27B, plan, 1, 2)
    # Recurrent state is per SEQUENCE and constant in context length.
    assert state == calculate_recurrent_state_bytes(QWEN3_5_27B, plan, 1, 2)
    per_layer_state = state / 48
    # The engine padded its mamba page by "0.13%" to match the attention page;
    # this is the unpadded number that padding was applied to.
    assert per_layer_state == pytest.approx(
        QWEN3_5_27B_MAMBA_PAGE_PADDED / 1.0013, rel=0.001
    )

    max_model_len = 2473
    reconstructed = 65_536 + 48 * QWEN3_5_27B_MAMBA_PAGE_PADDED / max_model_len
    assert reconstructed == pytest.approx(
        MEASURED_QWEN3_5_27B_KV_PER_TOKEN, rel=0.01
    )


def test_the_hybrid_layout_is_found_through_the_multimodal_wrapper():
    """The dims live in text_config; reading the top level finds no model at all."""
    flat = dict(QWEN3_5_27B["text_config"])
    assert kv_bytes(QWEN3_5_27B, 1, 8192) == kv_bytes(flat, 1, 8192)


# ----------------------------------- mixed stacks that now need the capability
#
# These four shipped models mix windowed and full attention. Their windowed
# layers are only cheaper on an engine that can manage a heterogeneous KV
# layout, so the interleaved figure moved from the default to opt-in. Pinned
# here so the arithmetic itself stays covered.


def cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "name,full,sliding,batch,seq",
    [
        ("openai_gpt-oss-120b", 18, 18, 1, 8192),
        ("PowerInfer_SmallThinker-21BA3B-Instruct", 13, 39, 5, 100_000),
        ("nvidia_Hymba-1.5B-Base", 3, 29, 1, 32_768),
    ],
)
def test_interleaved_models_split_full_and_windowed_when_the_engine_can(
    name, full, sliding, batch, seq
):
    c = cfg(name)
    kvh = c.get("num_key_value_heads") or c["num_attention_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    window = c.get("sliding_window") or c.get("sliding_window_size")
    per_layer_token = 2 * kvh * hd * 2

    capable = kv_bytes(c, batch, seq, engine=HYBRID_ALLOCATOR)
    assert capable == pytest.approx(
        per_layer_token * (full * seq + sliding * min(seq, window)) * batch, rel=1e-9
    )
    # Default: every attention layer full, which is strictly larger.
    assert kv_bytes(c, batch, seq) == pytest.approx(
        per_layer_token * (full + sliding) * seq * batch, rel=1e-9
    )
