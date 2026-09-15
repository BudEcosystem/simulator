"""Decoder geometry must be found wherever a checkpoint puts it.

``huggingface_config_to_model_config`` reads every dimension with
``hf_config.get(key, DEFAULT)``. Composite checkpoints -- any vision/audio/omni LLM,
and some text-only ones -- keep the decoder's geometry in a nested block and leave the
top level holding only routing metadata, so reading the top level silently returns the
DEFAULTS: hidden_size 4096, 32 layers, full MHA. That is a generic ~7B LLaMA, whatever
the real model is.

This is not hypothetical. Qwen3.8-27B (25.9B, dims under ``text_config``) resolved that
way to roughly 5.5B, which made its modelled decode ~5x too fast and drove the deployment
planner to escalate to tensor parallelism the workload did not need.

These tests are deliberately written against SYNTHETIC configs rather than a list of
known models: the contract is "find the decoder wherever it is declared", not "know about
Qwen". The two checkpoint-backed tests at the end only run where the real files exist.
"""

import warnings

import pytest

from llm_memory_calculator.genz.Models.get_language_model import (
    huggingface_config_to_model_config,
    resolve_decoder_config,
)


def _decoder(**overrides):
    """A minimal but complete decoder block."""
    block = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
    }
    block.update(overrides)
    return block


# --------------------------------------------------------------- the flat fast path


def test_a_flat_config_is_returned_unchanged():
    """The overwhelmingly common case must not be perturbed at all.

    Identity matters, not just equality: a flat config should not even be copied, so
    there is no chance of the hoist reordering or dropping a key.
    """
    flat = _decoder(model_type="llama")
    assert resolve_decoder_config(flat) is flat


def test_flat_configs_keep_producing_identical_model_configs():
    flat = _decoder(model_type="llama", architectures=["LlamaForCausalLM"])
    cfg = huggingface_config_to_model_config(dict(flat), "flat")
    assert cfg.num_decoder_layers == 64
    assert cfg.hidden_size == 5120


# ------------------------------------------------------------------ nested variants


@pytest.mark.parametrize(
    "block_name",
    ["text_config", "llm_config", "language_config", "decoder_config", "text_model", "language_model"],
)
def test_decoder_is_found_under_any_conventional_block_name(block_name):
    """Nesting conventions differ per model family; none of them may be special-cased."""
    composite = {
        "model_type": "some_new_omni_model",
        "architectures": ["SomethingForConditionalGeneration"],
        block_name: _decoder(),
        "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152},
    }
    resolved = resolve_decoder_config(composite)
    assert resolved["num_hidden_layers"] == 64
    assert resolved["hidden_size"] == 5120
    assert resolved["num_key_value_heads"] == 4


def test_an_unconventional_block_name_is_still_found():
    """The fallback must not depend on the block being conventionally named."""
    composite = {"model_type": "x", "the_actual_decoder": _decoder()}
    assert resolve_decoder_config(composite)["num_hidden_layers"] == 64


def test_the_vision_tower_is_never_mistaken_for_the_decoder():
    """A vision tower carries the same geometry keys and is usually listed first.

    Picking it would silently model a 27-layer/1152-wide encoder as the LLM.
    """
    composite = {
        "model_type": "x",
        "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152, "intermediate_size": 4304},
        "text_config": _decoder(),
    }
    resolved = resolve_decoder_config(composite)
    assert resolved["hidden_size"] == 5120, "picked the vision tower instead of the decoder"


@pytest.mark.parametrize("encoder_name", ["vision_config", "audio_config", "speech_encoder", "video_config"])
def test_no_modality_encoder_is_ever_selected(encoder_name):
    composite = {"model_type": "x", encoder_name: {"num_hidden_layers": 8, "hidden_size": 64}}
    with pytest.raises(ValueError):
        resolve_decoder_config(composite)


def test_a_decoder_nested_two_levels_deep_is_found():
    """Omni-style configs nest the text decoder inside another wrapper block."""
    composite = {"model_type": "omni", "thinker_config": {"text_config": _decoder()}}
    assert resolve_decoder_config(composite)["num_hidden_layers"] == 64


# ------------------------------------------------------------------ merge semantics


def test_nested_dims_win_but_top_level_keys_survive():
    composite = {
        "model_type": "wrapper",
        "architectures": ["WrapperForConditionalGeneration"],
        "quantization_config": {"quant_method": "mxfp4"},
        "hidden_size": 4096,  # a stale/unrelated top-level value
        "text_config": _decoder(),
    }
    resolved = resolve_decoder_config(composite)
    assert resolved["hidden_size"] == 5120, "the decoder block must win for dimensions"
    assert resolved["architectures"] == ["WrapperForConditionalGeneration"]
    assert resolved["quantization_config"] == {"quant_method": "mxfp4"}


def test_hoisting_carries_hybrid_attention_markers_through():
    """Downstream KV/layer-plan code reads these; they are useless left nested."""
    composite = {
        "model_type": "hybrid",
        "text_config": _decoder(layer_types=["linear", "full"] * 32, full_attention_interval=2),
    }
    resolved = resolve_decoder_config(composite)
    assert resolved["full_attention_interval"] == 2
    assert len(resolved["layer_types"]) == 64


# ------------------------------------------------------------- refuse, do not invent


def test_a_composite_config_with_no_decoder_raises_rather_than_defaulting():
    """The whole point: never silently substitute a different model.

    Before this guard, this config produced a confident 4096-wide/32-layer model.
    """
    composite = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3P5ForConditionalGeneration"],
        "vision_config": {"depth": 27, "out_hidden_size": 1152},
    }
    with pytest.raises(ValueError) as excinfo:
        resolve_decoder_config(composite)
    message = str(excinfo.value)
    assert "num_hidden_layers" in message, "the error must name what it looked for"
    assert "vision_config" in message, "the error must show what it actually saw"


def test_a_partial_flat_config_still_falls_back_quietly():
    """Only COMPOSITE configs raise.

    Callers legitimately pass small partial dicts (MODEL_DICT overrides, synthetic
    fixtures); tightening those is a separate decision and would break them here.
    """
    resolved = resolve_decoder_config({"num_attention_heads": 8})
    assert resolved == {"num_attention_heads": 8}


# ------------------------------------------------------- real checkpoints, if present

_REGISTRY = "/data/models-registry"
_NESTED_CHECKPOINT = f"{_REGISTRY}/qwen_qwen3_8-27b_a849ea74"


def _load(path):
    import json
    import os

    if not os.path.isfile(os.path.join(path, "config.json")):
        pytest.skip(f"checkpoint not mounted: {path}")
    with open(os.path.join(path, "config.json")) as handle:
        return json.load(handle)


def test_real_nested_checkpoint_resolves_to_its_true_geometry():
    cfg = huggingface_config_to_model_config(_load(_NESTED_CHECKPOINT), "nested-checkpoint")
    # Read straight out of text_config; before the fix these were 32 and 4096.
    assert cfg.num_decoder_layers == 64
    assert cfg.hidden_size == 5120


def test_real_nested_checkpoint_decode_is_no_longer_five_times_too_fast():
    """Regression guard tied to a measurement, not to a golden number.

    Measured on an H100XM-80C vGPU, tp=1, batch 1, 2048-token context: TPOT 20.197 ms.
    The old top-level read gave 3.906 ms (5.2x optimistic) because it modelled ~5.5B
    instead of 25.9B. The remaining gap is the roofline's missing achieved-bandwidth
    derating, so this asserts the order of magnitude is right rather than an exact match.
    """
    pytest.importorskip("pandas")
    from llm_memory_calculator import HardwareManager, estimate_end_to_end_performance

    _load(_NESTED_CHECKPOINT)  # skips when the checkpoint is absent
    result = estimate_end_to_end_performance(
        model=_NESTED_CHECKPOINT,
        batch_size=1,
        input_tokens=2048,
        output_tokens=2048,
        system_name=HardwareManager().get_hardware_config("H100_GPU"),
        bits="bf16",
        tensor_parallel=1,
    )
    measured_tpot_ms = 20.197
    assert result["average_tpot"] > 0.5 * measured_tpot_ms, (
        f"decode still wildly optimistic: {result['average_tpot']:.3f} ms vs {measured_tpot_ms} ms measured"
    )
    assert result["average_tpot"] < 1.5 * measured_tpot_ms


def test_a_config_whose_geometry_cannot_be_found_warns_loudly():
    """The last generic hole: a FLAT config using unfamiliar key names.

    Nesting is handled by the hoist and an unresolvable composite raises, but a flat
    dialect that simply spells its fields differently would still fall through to the
    ~7B defaults. That must at least be visible rather than silent.
    """
    with pytest.warns(UserWarning, match="falling back to defaults"):
        huggingface_config_to_model_config({"model_type": "some_new_dialect"}, "unknown-dialect")


def test_a_fully_specified_config_warns_about_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        huggingface_config_to_model_config(_decoder(model_type="llama"), "complete")
