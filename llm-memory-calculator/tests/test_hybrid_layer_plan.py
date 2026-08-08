"""Hybrid attention/recurrent architectures, against real shipped configs.

Every fixture in ``tests/fixtures/hybrid_configs/`` is the verbatim
``config.json`` from the HuggingFace repo named by the filename. Expected values
are hand-derived from those configs and, where the geometry was checkable, from
the shipped ``model.safetensors`` headers -- not from what this code currently
returns.

The bug these guard against: eleven families each invented their own config key
for "which layers carry a KV cache", the calculator read one of them, and every
other hybrid was charged full KV on all layers. The over-count ran from 3.75x
(LFM2) to 14x (Nemotron-H), and it grows with context length -- precisely the
regime these architectures exist to serve.
"""

import json
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.layer_plan import resolve_layer_plan
from llm_memory_calculator.state_memory import calculate_recurrent_state_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "hybrid_configs"


def cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ---------------------------------------------------------------- layer plans
# (fixture, full, sliding, recurrent, dialect)
PLANS = [
    # `layer_types` spelling GDN as "linear_attention" -- the substring
    # "attention" is why these 48 layers used to be charged full KV.
    ("Qwen_Qwen3.6-27B", 16, 0, 48, "layer_types"),
    # Same family, but ships NO layer_types -- only the interval.
    ("Qwen_Qwen3-Next-80B-A3B-Instruct", 12, 0, 36, "full_attention_interval"),
    ("MiniMaxAI_MiniMax-Text-01", 10, 0, 70, "attn_type_list"),
    ("ibm-granite_granite-4.0-h-small", 4, 0, 36, "layer_types"),
    ("nvidia_NVIDIA-Nemotron-Nano-9B-v2", 4, 0, 27, "hybrid_override_pattern"),
    ("nvidia_Nemotron-H-8B-Base-8K", 4, 0, 24, "hybrid_override_pattern"),
    ("ibm-ai-platform_Bamba-9B-v2", 3, 0, 29, "attn_layer_indices"),
    ("Zyphra_Zamba2-2.7B", 9, 0, 54, "layers_block_type"),
    (
        "moonshotai_Kimi-Linear-48B-A3B-Instruct",
        7,
        0,
        20,
        "linear_attn_config.full_attn_layers",
    ),
    ("LiquidAI_LFM2-2.6B", 8, 0, 22, "layer_types"),
    ("ai21labs_Jamba-v0.1", 4, 0, 28, "attn_layer_period"),
    # Parallel hybrids: attention AND recurrence in the same layer, so the two
    # counts overlap rather than partition the depth.
    ("nvidia_Hymba-1.5B-Base", 3, 29, 32, "global_attn_idx"),
    ("tiiuae_Falcon-H1-34B-Instruct", 72, 0, 72, "parallel_hybrid"),
    # Pure SSM: no attention anywhere.
    ("mistralai_Mamba-Codestral-7B-v0.1", 0, 0, 64, "pure_ssm"),
    ("state-spaces_mamba-2.8b-hf", 0, 0, 64, "pure_ssm"),
    # Sliding/full, the one mix that already worked. Must not regress.
    ("openai_gpt-oss-120b", 18, 18, 0, "layer_types"),
    # A 0/1 mask rather than a list of strings or indices.
    ("PowerInfer_SmallThinker-21BA3B-Instruct", 13, 39, 0, "sliding_window_layout"),
]


@pytest.mark.parametrize("name,full,sliding,rec,dialect", PLANS)
def test_layer_plan_matches_shipped_config(name, full, sliding, rec, dialect):
    plan = resolve_layer_plan(cfg(name))
    assert plan is not None, f"{name}: no plan resolved"
    assert plan["dialect"] == dialect
    assert (
        plan["num_full_layers"],
        plan["num_sliding_layers"],
        plan["num_recurrent_layers"],
    ) == (full, sliding, rec)


@pytest.mark.parametrize(
    "name", ["MiniMaxAI_MiniMax-M2", "unsloth_gemma-3-27b-it", "kyutai_helium-1-2b"]
)
def test_non_hybrid_models_resolve_to_no_plan(name):
    """A plan must not be invented for ordinary transformers.

    MiniMax-M2 is the trap: it ships `attn_type_list` like MiniMax-Text-01 but
    dropped lightning attention, so every entry is 1. Treating it as hybrid would
    under-count its KV by the interleave ratio -- the opposite error, and a far
    more dangerous one since it silently under-provisions.
    """
    assert resolve_layer_plan(cfg(name)) is None


def test_kimi_layer_lists_are_one_indexed():
    """Kimi's full_attn_layers/kda_layers cover 1..n, not 0..n-1.

    Read as 0-indexed they drop layer n and shift every boundary, which changes
    the full-attention count and silently moves which layers pay KV.
    """
    c = cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct")
    n = c["num_hidden_layers"]
    lac = c["linear_attn_config"]
    assert min(lac["full_attn_layers"] + lac["kda_layers"]) == 1
    assert max(lac["full_attn_layers"] + lac["kda_layers"]) == n

    plan = resolve_layer_plan(c)
    # Layer index 26 (0-based) is layer 27 (1-based) and IS full attention.
    assert plan["attn"][n - 1] == "full"
    assert plan["recurrent"][n - 1] is None


# ------------------------------------------------------------------ KV cache


def _kv(name, batch, seq, precision="bf16"):
    calc = ModelMemoryCalculator()
    c = cfg(name)
    calc.model_type = calc.detect_model_type(c)
    return calc.calculate_kv_cache(c, batch_size=batch, seq_length=seq, precision=precision)


def test_qwen36_kv_counts_only_full_attention_layers():
    """Qwen3.6-27B at the workload that exposed this: 5 x (80k in + 20k out).

    Geometry confirmed against the safetensors headers: k_proj and v_proj are
    each [1024, 5120] = 4 kv heads x 256 head_dim, on the 16 full-attention
    layers only. 2 (K and V) * 4 * 256 * 2 bytes = 4 KiB per token per layer.
    """
    per_token_per_layer = 2 * 4 * 256 * 2
    expected = per_token_per_layer * 16 * 100_000 * 5 / 1e9
    assert _kv("Qwen_Qwen3.6-27B", 5, 100_000) == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(32.768, rel=1e-9)


def test_qwen36_kv_is_four_times_below_charging_every_layer():
    """The exact defect ratio: 64 layers charged where only 16 hold a cache."""
    got = _kv("Qwen_Qwen3.6-27B", 5, 100_000)
    all_layers = got * 64 / 16
    assert all_layers == pytest.approx(131.072, rel=1e-9)


def test_nemotron_h_kv_drops_by_thirteen_x():
    """Nemotron-H-8B: 4 attention layers in a 52-layer stack (M/*/- pattern)."""
    c = cfg("nvidia_Nemotron-H-8B-Base-8K")
    n_layers = c["num_hidden_layers"]
    kv = _kv("nvidia_Nemotron-H-8B-Base-8K", 1, 32_768)
    per_layer = kv / 4
    assert per_layer * n_layers / kv == pytest.approx(13.0, rel=1e-9)


def test_pure_ssm_has_no_kv_cache_at_all():
    assert _kv("mistralai_Mamba-Codestral-7B-v0.1", 4, 65_536) == 0.0
    assert _kv("state-spaces_mamba-2.8b-hf", 4, 65_536) == 0.0


def test_hymba_windowed_layers_are_clamped_not_dropped():
    """Hymba is a PARALLEL hybrid: all 32 layers carry KV.

    Three are global and 29 are windowed at 1024, so the answer is neither "32
    full layers" (the old over-count) nor "3 layers" (what a sequential reading
    of global_attn_idx would wrongly give).
    """
    seq, window = 32_768, 1024
    # head_dim is absent; it falls out as hidden_size // n_heads = 1600 // 25 = 64.
    per_token_per_layer = 2 * 5 * 64 * 2
    expected = per_token_per_layer * (3 * seq + 29 * window) / 1e9
    assert _kv("nvidia_Hymba-1.5B-Base", 1, seq) == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(0.16384, rel=1e-9)
    # For contrast: charging all 32 layers at full length is 8.192x larger, and
    # charging only the 3 global layers would be 3.9x too small.
    assert per_token_per_layer * 32 * seq / 1e9 == pytest.approx(1.342177, rel=1e-5)


def test_smallthinker_layout_polarity_is_one_means_sliding():
    """1 = SWA, 0 = normal attention -- per PowerInfer's own config docstring
    ("0 for normal attention, 1 for SWA") and modeling_smallthinker.py, which
    applies the sliding mask when `sliding_window_layout[layer_idx] == 1`.

    Inverting this is not a small error: it would mark the 13 genuinely-global
    layers as windowed and the 39 windowed ones as global, which UNDER-counts
    KV. Every other mistake in this file over-counts.
    """
    c = cfg("PowerInfer_SmallThinker-21BA3B-Instruct")
    layout = c["sliding_window_layout"]
    plan = resolve_layer_plan(c)
    for i, flag in enumerate(layout):
        assert plan["attn"][i] == ("sliding" if flag == 1 else "full")
    # The global layers are the sparse ones: 13 of 52, at indices 0, 4, 8, ...
    assert [i for i, a in enumerate(plan["attn"]) if a == "full"][:3] == [0, 4, 8]


def test_smallthinker_window_size_alias_is_resolved():
    """The window ships as `sliding_window_size`, not `sliding_window`.

    Without the alias the per-layer path finds no window and charges all 39
    windowed layers the full context -- the plan would be right and the answer
    still wrong.
    """
    from llm_memory_calculator.config_normalizer import ConfigNormalizer

    c = cfg("PowerInfer_SmallThinker-21BA3B-Instruct")
    assert "sliding_window" not in c and c["sliding_window_size"] == 4096
    assert ConfigNormalizer.normalize_config(c)["sliding_window"] == 4096


def test_smallthinker_kv_at_long_context():
    """13 full layers at full length + 39 windowed at 4096, 4 kv heads x 128."""
    seq, window = 100_000, 4096
    per_token_per_layer = 2 * 4 * 128 * 2
    expected = per_token_per_layer * (13 * seq + 39 * window) * 5 / 1e9
    got = _kv("PowerInfer_SmallThinker-21BA3B-Instruct", 5, seq)
    assert got == pytest.approx(expected, rel=1e-9)
    # Charging all 52 layers the full context is 3.56x larger.
    assert per_token_per_layer * 52 * seq * 5 / 1e9 / expected == pytest.approx(3.56, rel=1e-2)


def test_smallthinker_below_the_window_matches_uniform_attention():
    """A sanity property: when the context fits inside the window, a windowed
    layer and a full layer cost the same, so the plan must not change anything."""
    seq = 4096
    per_token_per_layer = 2 * 4 * 128 * 2
    uniform = per_token_per_layer * 52 * seq / 1e9
    assert _kv("PowerInfer_SmallThinker-21BA3B-Instruct", 1, seq) == pytest.approx(
        uniform, rel=1e-9
    )


def test_gpt_oss_sliding_full_mix_is_unchanged():
    """The one interleave that already worked -- guard against regression."""
    c = cfg("openai_gpt-oss-120b")
    seq, window = 8192, c["sliding_window"]
    kv = _kv("openai_gpt-oss-120b", 1, seq)
    kvh, hd = c["num_key_value_heads"], c["head_dim"]
    expected = 2 * kvh * hd * 2 * (18 * seq + 18 * min(seq, window)) / 1e9
    assert kv == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------- state memory


def _state(name, batch, precision="bf16"):
    c = cfg(name)
    return calculate_recurrent_state_bytes(
        c, resolve_layer_plan(c), batch, {"bf16": 2, "fp16": 2, "fp32": 4}[precision]
    )


def test_qwen36_gdn_state_matches_shipped_conv1d_geometry():
    """Gated DeltaNet state, hand-derived from Qwen3.6-27B's own tensors.

    The delta-rule matrix is per VALUE head (48), not per key head (16): keys are
    shared GQA-style across value heads. Held in fp32 per `mamba_ssm_dtype`.
    The conv width is checkable -- the shipped linear_attn.conv1d.weight is
    [10240, 1, 4], and 2*16*128 + 48*128 = 10240.
    """
    recurrent = 48 * 128 * 128 * 4
    conv = (2 * 16 * 128 + 48 * 128) * (4 - 1) * 2
    per_seq = (recurrent + conv) * 48
    assert _state("Qwen_Qwen3.6-27B", 5) == pytest.approx(per_seq * 5, rel=1e-9)
    assert per_seq * 5 / 1e9 == pytest.approx(0.76972032, rel=1e-6)


def test_gdn_state_is_constant_in_context_length():
    """The defining property, and what the old zero-valued term erased.

    A same-width full-attention stack would need 4 GB more KV going from 1k to
    262k; the GDN layers need exactly the same bytes at both.
    """
    assert _state("Qwen_Qwen3.6-27B", 1) == _state("Qwen_Qwen3.6-27B", 1)
    one, eight = _state("Qwen_Qwen3.6-27B", 1), _state("Qwen_Qwen3.6-27B", 8)
    assert eight == pytest.approx(one * 8, rel=1e-12)


def test_mamba2_state_uses_head_geometry_and_group_conv():
    """Mamba-Codestral-7B, a pure Mamba-2 stack.

    n_heads x d_head x d_state for the recurrent term, and a conv ring buffer
    widened by the B/C group projections: d_inner + 2*n_groups*d_state.
    """
    recurrent = 128 * 64 * 128 * 4
    conv = (8192 + 2 * 8 * 128) * (4 - 1) * 2
    per_seq = (recurrent + conv) * 64
    assert _state("mistralai_Mamba-Codestral-7B-v0.1", 1) == pytest.approx(
        per_seq, rel=1e-9
    )


def test_falcon_h1_state_reads_the_mamba_prefixed_keys():
    """Falcon-H1 ships mamba_d_state=256, not state_size.

    The old code read only `state_size`/`d_state` and fell through to a
    hardcoded 16 -- a 16x under-count on this model.
    """
    c = cfg("tiiuae_Falcon-H1-34B-Instruct")
    assert c["mamba_d_state"] == 256 and "state_size" not in c
    recurrent = 32 * 128 * 256 * 4
    conv = (32 * 128 + 2 * 2 * 256) * (4 - 1) * 2
    assert _state("tiiuae_Falcon-H1-34B-Instruct", 1) == pytest.approx(
        (recurrent + conv) * 72, rel=1e-9
    )


def test_every_recurrent_family_reports_nonzero_state():
    """The headline regression: all of these used to report exactly 0.00 GB."""
    for name, _f, _s, rec, _d in PLANS:
        if rec:
            assert _state(name, 1) > 0, f"{name} still reports zero state"


def test_lfm2_has_conv_state_but_no_recurrent_matrix():
    """LFM2's `conv` layers are a short convolution only -- no delta-rule state."""
    c = cfg("LiquidAI_LFM2-2.6B")
    expected = c["hidden_size"] * c["conv_L_cache"] * 2 * 22
    assert _state("LiquidAI_LFM2-2.6B", 1) == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------ end-to-end wiring


def test_state_memory_no_longer_needs_a_hand_set_model_type():
    """Detection must reach the state path on its own.

    The two pre-existing tests assigned `calc.model_type = "hybrid"` before
    calling, which asserted the arithmetic while concealing that a real
    falcon_h1/qwen3_next/nemotron_h config never got there.
    """
    for name in (
        "tiiuae_Falcon-H1-34B-Instruct",
        "Qwen_Qwen3-Next-80B-A3B-Instruct",
        "nvidia_NVIDIA-Nemotron-Nano-9B-v2",
        "ibm-granite_granite-4.0-h-small",
    ):
        calc = ModelMemoryCalculator()
        c = cfg(name)
        calc.model_type = calc.detect_model_type(c)  # no manual override
        assert calc.calculate_state_memory(c, batch_size=1, precision="bf16") > 0


def test_mamba2_is_no_longer_detected_as_unknown():
    """`mamba2` used to miss every list and land in the generic-transformer
    fallback: hidden_size 768, 12 layers, vocab 50257 -- for a 7B model."""
    calc = ModelMemoryCalculator()
    assert calc.detect_model_type(cfg("mistralai_Mamba-Codestral-7B-v0.1")) == "state-space"


def test_multimodal_no_longer_preempts_hybrid_handling():
    """Qwen3.6 is multimodal AND hybrid.

    KV used to be gated on `self.model_type == "multimodal"` picking text_config,
    while the hybrid layout was only reachable from a "hybrid" model_type -- so a
    model that is both got neither.
    """
    calc = ModelMemoryCalculator()
    c = cfg("Qwen_Qwen3.6-27B")
    assert calc.detect_model_type(c) == "multimodal"
    assert calc.calculate_kv_cache(c, 5, 100_000, "bf16") == pytest.approx(32.768, rel=1e-9)
    assert calc.calculate_state_memory(c, batch_size=5, precision="bf16") > 0


def test_phi4flash_is_flagged_rather_than_silently_approximated():
    """SambaY shares one global KV across its YOCO layers, which a per-layer plan
    cannot express. It must announce that instead of quietly reporting a number
    that looks as trustworthy as the others."""
    plan = resolve_layer_plan(cfg("microsoft_Phi-4-mini-flash-reasoning"))
    assert plan["approximate"] is True
    assert plan["notes"] and "OVER-estimate" in plan["notes"][0]


def test_phi4flash_approximation_keeps_its_sliding_window():
    """An approximation must not be *worse* than the behavior it replaces.

    Phi-4-flash uses a 512-token window. Marking its layers `full` -- the obvious
    "be conservative, over-count" reflex -- discards that clamp and inflates KV
    by 195x at 100k context. Regression caught by diffing every model's report
    against the pre-refactor tree.
    """
    c = cfg("microsoft_Phi-4-mini-flash-reasoning")
    window = c["sliding_window"]
    plan = resolve_layer_plan(c)
    assert plan["num_sliding_layers"] == c["num_hidden_layers"]
    assert plan["num_full_layers"] == 0

    seq = 100_000
    kv = _kv("microsoft_Phi-4-mini-flash-reasoning", 5, seq)
    expected = 2 * 20 * (2560 // 40) * 2 * 32 * window * 5 / 1e9
    assert kv == pytest.approx(expected, rel=1e-9)
    # Unclamped it would be seq/window = 195x larger.
    assert seq / window == pytest.approx(195.3125, rel=1e-6)
