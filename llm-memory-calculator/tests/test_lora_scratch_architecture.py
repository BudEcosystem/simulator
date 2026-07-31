"""The LoRA prefill scratch: measured anchors, the mechanistic envelope, and devices.

The term is an artifact of vLLM's CPU LoRA path (`lora/ops/torch_ops`): fancy-indexing
the stacked weight tensor with a per-token index materializes one copy of the LoRA
matrix PER TOKEN -- `(T, out_features, max_lora_rank)` per wrapped module per layer.
See docs/lora-scratch-root-cause.md for the full derivation and probe evidence.

That mechanism fixes what each input is allowed to do:

  * configured `max_lora_rank` -- linear with a floor (probe: rank 32 -> 0.573x)
  * `max_num_batched_tokens`   -- convex measured shape
  * `max_loras`                -- nearly nothing (probe: 2 slots -> +7.7%)
  * device                     -- CPU only; CUDA's Triton kernels never copy
  * architecture / width       -- via the wrapped modules' dimensions

Measured (arch, hidden) pairs reproduce their anchors exactly. Everything else gets
the mechanistic envelope `C x T x R x S_max x 2B + base` with C=7 (measured 2.3-6.5)
and a note saying so. A five-architecture campaign showed no config dimension predicts
the term across families, so unmeasured cases must err high and say why.
"""

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator


GIB = 1024**3


class Lora:
    enabled = True
    max_loras = 1
    max_lora_rank = 64


def cfg(
    arch, hidden, layers=32, inter=None, model_type=None, heads=None, head_dim=None
):
    c = {
        "architectures": [arch] if arch else None,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
    }
    if inter is not None:
        c["intermediate_size"] = inter
    if model_type is not None:
        c["model_type"] = model_type
    if heads is not None:
        c["num_attention_heads"] = heads
    if head_dim is not None:
        c["head_dim"] = head_dim
    return c


def scratch_gib(config, tokens, lora=None, tp=1, device=None):
    calc = ModelMemoryCalculator()
    gb, notes = calc.calculate_lora_prefill_scratch(
        config,
        lora or Lora(),
        batched_tokens=tokens,
        tensor_parallel=tp,
        target_device=device,
    )
    return gb * 1e9 / GIB, notes


# ------------------------------------------------------------------ measured anchors

MEASURED = [
    # arch, hidden, T_meas, measured GiB (control-subtracted memory.peak)
    ("Qwen3ForCausalLM", 1024, 4273, 6.10),
    ("Qwen3ForCausalLM", 2560, 4273, 24.07),
    ("LlamaForCausalLM", 4096, 4273, 20.21),
    ("Gemma4UnifiedForConditionalGeneration", 3840, 2048, 8.63),
    ("Qwen3_5ForConditionalGeneration", 2560, 4273, 0.21),
]


@pytest.mark.parametrize("arch,hidden,tokens,measured", MEASURED)
def test_reproduces_every_measured_anchor(arch, hidden, tokens, measured):
    """Consistency at each anchor's own (T, rank 64, 1 lora, TP 1) point."""
    got, _ = scratch_gib(cfg(arch, hidden), tokens)
    assert got == pytest.approx(measured, rel=0.02), f"{arch}@{hidden}"


def test_measured_dense_models_carry_no_notes():
    """Dense anchor + calibrated rank + in-envelope tokens = no caveats.

    The hybrid is excluded on purpose: its anchor always warns (see below)."""
    for arch, hidden, tokens, _ in MEASURED:
        if arch == "Qwen3_5ForConditionalGeneration":
            continue
        _, notes = scratch_gib(cfg(arch, hidden), tokens)
        assert notes == [], f"{arch}@{hidden}: {notes}"


def test_out_of_sample_token_budget_errs_high():
    """Qwen3-4B at T=2048 -- the one point never fitted. Measured 15.46 GiB.

    The convex shape predicts ~+15% high, the safe direction; a linear-in-T form
    under-predicted this point by 19%, which is the error that OOMKills."""
    got, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 2048)
    measured = 15.46
    assert got > measured, "under-predicting is the failure mode that crash-loops pods"
    assert got == pytest.approx(measured, rel=0.25)


def test_model_type_alias_resolves_without_architectures_key():
    """Some configs carry only model_type; a measured model must not fall to the envelope."""
    got, notes = scratch_gib(cfg(None, 1024, model_type="qwen3"), 4273)
    assert got == pytest.approx(6.10, rel=0.02)
    assert notes == []


# ------------------------------------------------------------------ the device gate


def test_cuda_pays_nothing_and_says_why():
    """The term is a CPU torch_ops artifact; Triton kernels never materialize copies."""
    for device in ("cuda", "CUDA", "gpu", "rocm"):
        got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, device=device)
        assert got == 0.0, device
        assert any("artifact" in n for n in notes), device


def test_cpu_and_unspecified_devices_pay_the_term():
    """None must behave as CPU: a caller that does not know its device stays safe."""
    for device in ("cpu", "cpu_high", None, ""):
        got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, device=device)
        assert got == pytest.approx(24.07, rel=0.02), device
        assert notes == [], device


def test_unknown_device_is_conservative_with_a_note():
    """HPU (or anything unrecognised) gets the CPU-derived term plus a caveat --
    over-reserving is recoverable, an OOM at warmup is not."""
    got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, device="hpu")
    assert got == pytest.approx(24.07, rel=0.02)
    assert any("UNVALIDATED on device 'hpu'" in n for n in notes)


# ------------------------------------------------------------------ rank and slots


def test_rank_scales_linearly_with_a_floor():
    """Probe-measured: rank 32 -> 0.573x of rank 64, NOT 0.5x."""

    class R32(Lora):
        max_lora_rank = 32

    base, _ = scratch_gib(cfg("Qwen3ForCausalLM", 1024), 4273)
    r32, _ = scratch_gib(cfg("Qwen3ForCausalLM", 1024), 4273, lora=R32())
    assert r32 / base == pytest.approx(0.573, abs=0.01)


def test_rank_above_64_is_plain_linear_and_flagged():
    """Above the characterised rank the floor form would under-predict; use plain
    linear and keep the LOWER BOUND warning (a rank-256 pod has OOMKilled past it)."""

    class R256(Lora):
        max_lora_rank = 256

    base, _ = scratch_gib(cfg("Qwen3ForCausalLM", 1024), 4273)
    r256, notes = scratch_gib(cfg("Qwen3ForCausalLM", 1024), 4273, lora=R256())
    assert r256 == pytest.approx(4 * base, rel=0.01)
    assert any("UNVALIDATED at rank 256" in n for n in notes)


def test_extra_lora_slots_cost_percent_not_multiples():
    """Probe-measured: a second slot cost +7.7%. The old formula doubled the term."""

    class L2(Lora):
        max_loras = 2

    base, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273)
    two, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, lora=L2())
    assert two == pytest.approx(1.10 * base, rel=0.01)
    assert notes == []  # 2 slots is measured; only >2 is extrapolation

    class L4(Lora):
        max_loras = 4

    four, notes4 = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, lora=L4())
    assert four == pytest.approx(1.30 * base, rel=0.01)
    assert any("max_loras=4 is extrapolated" in n for n in notes4)


# ------------------------------------------------------------------ the envelope


def test_unmeasured_architecture_uses_the_mechanism_and_says_so():
    """Mistral-7B geometry: envelope = 7 x T x R x S_max x 2B + 1GB, S_max from config."""
    got, notes = scratch_gib(cfg("MistralForCausalLM", 4096, inter=14336), 4273)
    expect = (7 * 4273 * 64 * 14336 * 2 + 1e9) / GIB
    assert got == pytest.approx(expect, rel=0.01)
    assert any("UNMEASURED" in n and "largest wrapped slice" in n for n in notes)


def test_envelope_scales_with_width_unlike_the_old_table():
    """The old fallback borrowed Qwen3-4B's multiplier regardless of size -- a wider
    model was under-sized. The envelope must grow with the model's own dimensions."""
    small, _ = scratch_gib(cfg("MistralForCausalLM", 2048, inter=5632), 4273)
    large, _ = scratch_gib(cfg("MistralForCausalLM", 8192, inter=28672), 4273)
    assert large > 4 * small


def test_unmeasured_width_in_a_measured_family_gets_the_envelope():
    """Qwen3 at an unmeasured width: the rate tripled between measured widths, so
    neither anchor transfers; the mechanism (which knows the new width) does."""
    got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 5120, inter=25600), 4273)
    expect = (7 * 4273 * 64 * 25600 * 2 + 1e9) / GIB
    assert got == pytest.approx(expect, rel=0.01)
    assert any("hidden_size 5120" in n for n in notes)


def test_envelope_covers_every_measured_model():
    """C=7 must sit above every observation (measured live-concurrency was 2.3-6.5),
    so an unknown architecture with the same dims as a measured one errs high."""
    dims = {
        1024: 3072,
        2560: 9728,
        4096: 14336,
        3840: 15360,
    }
    for arch, hidden, tokens, measured in MEASURED:
        if arch == "Qwen3_5ForConditionalGeneration":
            continue  # warmup-skip suspect; the envelope intentionally dwarfs it
        got, _ = scratch_gib(
            cfg("UnknownForCausalLM", hidden, inter=dims[hidden]), tokens
        )
        assert got > measured, f"envelope under {arch}@{hidden}"


def test_config_without_dimensions_refuses_loudly():
    got, notes = scratch_gib({"architectures": ["NoDimsForCausalLM"]}, 4273)
    assert got == 0.0
    assert any("no usable module dimensions" in n for n in notes)


# ------------------------------------------------------------------ the hybrid landmine


def test_hybrid_anchor_always_warns_about_serving_time():
    """Qwen3.5's near-zero was measured at WARMUP, which is suspected to skip the
    copy path for GDN hybrids. Budgeting from it is allowed -- it is the
    measurement -- but never silently: a real adapter request may OOM a pod that
    passed warmup."""
    got, notes = scratch_gib(cfg("Qwen3_5ForConditionalGeneration", 2560), 4273)
    assert got == pytest.approx(0.21, rel=0.03)
    assert any("SKIP" in n and "after passing warmup" in n for n in notes)


def test_hybrid_is_two_orders_below_dense_at_the_same_width():
    """The observation that killed every width-based law; pinned so a refactor that
    regresses to predicting from geometry fails loudly."""
    hybrid, _ = scratch_gib(cfg("Qwen3_5ForConditionalGeneration", 2560), 4273)
    dense, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273)
    assert dense / hybrid > 50


# ------------------------------------------------------------------ mechanics


def test_depth_does_not_drive_the_term():
    """The peak is per-call live memory, not a per-layer accumulation: Llama (32
    layers) measured BELOW Qwen3-4B (36) and Gemma (48) below both. Layer count
    must not move the prediction."""
    a, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560, layers=16), 4273)
    b, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560, layers=64), 4273)
    assert a == b


def test_disabled_lora_costs_nothing():
    class Off:
        enabled = False

    got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, lora=Off())
    assert got == 0.0 and notes == []


def test_unset_rank_refuses():
    class NoRank(Lora):
        max_lora_rank = 0

    got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, lora=NoRank())
    assert got == 0.0
    assert any("max_lora_rank is unset" in n for n in notes)


def test_tensor_parallel_shards_the_term():
    one, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, tp=1)
    two, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560), 4273, tp=2)
    assert two == pytest.approx(one / 2, rel=1e-6)


def test_largest_slice_prefers_intermediate_but_not_blindly():
    """S_max is a max over candidate slices, not hardcoded to the FFN: a model with
    attention wider than its FFN sizes from the attention slice."""
    calc = ModelMemoryCalculator()
    ffn_wide = {"hidden_size": 1024, "intermediate_size": 8192}
    attn_wide = {
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_attention_heads": 64,
        "head_dim": 128,
    }
    assert calc._lora_largest_slice(ffn_wide) == 8192
    # MHA (kv_heads defaults to heads): the fused kv slice, 2 x 64 x 128, is the
    # widest projection in this geometry -- wider than q itself.
    assert calc._lora_largest_slice(attn_wide) == 2 * 64 * 128
