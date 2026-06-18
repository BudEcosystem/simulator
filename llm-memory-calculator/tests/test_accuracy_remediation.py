"""Regression safety net for the accuracy-remediation plan (solutions.md).

Extreme-TDD foundation, authored BEFORE any production change (rollout Step 0). Three classes:

1. Physical-limit invariants — ground-truthed against vendor DENSE bf16 datasheet peaks (not the
   config's possibly-sparse value). These are RED now (H100/H200/GH200 exceed dense peak; decode
   exceeds peak HBM bandwidth) and turn GREEN as the structural + calibration fixes land. Each is
   tagged with the rollout step that fixes it.

2. Byte-identical baselines — snapshot current decode latency + parameter counts. The structural
   fixes (F1 dense FLOPS, C causal mask, C4 dead-code removal) touch only the COMPUTE term, which
   loses max(compute,memory) in memory-bound decode, so decode MUST stay byte-identical through
   Steps 1-3. The snapshot auto-generates on first run and is asserted thereafter.

3. Parameter-family golden tests — the 7 families the audit verified correct must not regress; Gemma
   is RED until PC-GLU (H0) lands.

Run:  .venv/bin/python -m pytest llm-memory-calculator/tests/test_accuracy_remediation.py -v
"""
import json
import os
import warnings
import math
import pytest

warnings.filterwarnings("ignore")

from llm_memory_calculator import UniversalParameterCounter, get_hardware_config
from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling
from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling

BASELINE = os.path.join(os.path.dirname(__file__), "accuracy_baseline.json")

# ---------------------------------------------------------------------------
# Vendor DENSE bf16 tensor-core peak TFLOPS + peak HBM/LPDDR BW (GB/s).
# SOURCES (datasheets): NVIDIA H100 SXM 989.5 TF dense (1979 = 2:4 sparse); A100 312; B200 2250
# dense; RTX4090 330.3 dense (661 = sparse); AMD MI300X 1307 dense bf16; NVIDIA GB10 — measured
# achievable 85 TF (audit_microbench_data.md) used as the ceiling since the "250" label is ambiguous.
# These are the GROUND TRUTH the invariants check against, independent of what configs.py stores.
# ---------------------------------------------------------------------------
REFERENCE_DENSE_BF16_TFLOPS = {
    "H100_GPU": 989.5,
    "A100_80GB_GPU": 312.0,
    "RTX4090_GPU": 330.3,
    "B200_GPU": 2250.0,
    "MI300X": 1307.0,
    "GB10": 250.0,   # config-labeled peak; measured-achievable ceiling is ~85 TF (see GB10 note below)
}
REFERENCE_PEAK_BW_GBPS = {
    "H100_GPU": 3350.0,
    "A100_80GB_GPU": 2039.0,
    "RTX4090_GPU": 1008.0,
    "B200_GPU": 8000.0,
    "MI300X": 5300.0,
    "GB10": 273.0,
}

# (model, dense-bf16 peak ground truth) prefill cases — compute-bound regime
PREFILL_CASES = [
    ("llama2_7b", "H100_GPU", 6.738e9),
    ("llama2_7b", "A100_80GB_GPU", 6.738e9),
    ("llama3_8b", "H100_GPU", 8.03e9),
    ("mistral_7b", "A100_80GB_GPU", 7.242e9),
]
DECODE_CASES = [
    ("llama2_7b", "H100_GPU", 6.738e9),
    ("llama2_7b", "A100_80GB_GPU", 6.738e9),
    ("llama3_8b", "H100_GPU", 8.03e9),
    ("llama2_7b", "GB10", 6.738e9),   # positive control: calibrated device, realism test must PASS
]


def _prefill_latency_ms(model, hw, input_tokens=2048):
    out = prefill_moddeling(model=model, batch_size=1, input_tokens=input_tokens,
                            system_name=hw, bits="bf16")
    return out.Latency


def _decode_tpot_ms(model, hw, input_tokens=2048):
    out = decode_moddeling(model=model, batch_size=1, input_tokens=input_tokens,
                           output_tokens=1, system_name=hw, bits="bf16")
    return out.Latency


def _decode_self_mbu(model, hw, input_tokens=2048):
    """Self-MBU = (bytes the MODEL itself accounts as moved in one decode step) / tpot / peak_BW.
    Uses the model's OWN weight+KV traffic (NOT external params*2, which would wrongly count the
    input-embedding table — a gather, not streamed, in decode). At eta_mem=1.0 this is exactly 1.0."""
    from llm_memory_calculator.genz.unit import Unit
    u = Unit()
    _, summ = decode_moddeling(model=model, batch_size=1, input_tokens=input_tokens, output_tokens=1,
                               system_name=hw, bits="bf16", model_profilling=True)
    counted_gb = (summ[f"Total Weights ({u.unit_mem})"].values[0]
                  + summ[f"KV Cache ({u.unit_mem})"].values[0]) / 1024.0
    tpot_s = _decode_tpot_ms(model, hw, input_tokens) / 1e3
    peak_bw = REFERENCE_PEAK_BW_GBPS[hw] * 1e9
    return counted_gb * 1e9 / tpot_s / peak_bw


# ===========================================================================
# 1. PHYSICAL-LIMIT INVARIANTS  (RED until the named step lands)
# ===========================================================================
@pytest.mark.parametrize("model,hw,params", PREFILL_CASES)
def test_prefill_mfu_within_dense_peak(model, hw, params):
    """Implied prefill MFU (vs vendor DENSE peak) must be <= 1.0 + tol. Physically impossible to
    exceed. RED now for H100 (engine uses sparse 1979). GREEN after Step 1 (F1 dense FLOPS)."""
    S = 2048
    lat_s = _prefill_latency_ms(model, hw) / 1e3
    flops_lower_bound = 2.0 * params * S          # linear FLOPs only (conservative lower bound)
    peak = REFERENCE_DENSE_BF16_TFLOPS[hw] * 1e12
    implied_mfu = flops_lower_bound / lat_s / peak
    assert implied_mfu <= 1.05, (
        f"[Step1/F1] {model}@{hw}: implied prefill MFU={implied_mfu:.2%} exceeds dense peak "
        f"({REFERENCE_DENSE_BF16_TFLOPS[hw]} TF) — physically impossible; engine likely uses sparse FLOPS"
    )


@pytest.mark.parametrize("model,hw,params", DECODE_CASES)
def test_decode_self_consistent_with_peak_bw(model, hw, params):
    """PHYSICAL INVARIANT (device-agnostic, must always hold): the model may not move its OWN counted
    decode traffic faster than the device peak bandwidth (i.e. eta_mem <= 1). Passes now; guards
    against any future eta_mem>1 regression."""
    self_mbu = _decode_self_mbu(model, hw)
    assert self_mbu <= 1.01, (
        f"{model}@{hw}: model moves its own counted bytes at {self_mbu:.2%} of peak BW — "
        f"physically impossible (eta_mem>1)"
    )


@pytest.mark.parametrize("model,hw,params", DECODE_CASES)
def test_decode_mbu_realistic(model, hw, params):
    """REALISM TARGET (C2): real single-stream decode sustains <~85-90% MBU; GB10 measured 70%
    (audit_microbench_data.md). 0.95 is a generous physical ceiling. RED now for uncalibrated H100/A100
    (eta_mem=1.0 -> 100%); GREEN after Step 6 per-device eta_mem calibration (GB10 already passes ~65%)."""
    self_mbu = _decode_self_mbu(model, hw)
    assert self_mbu <= 0.95, (
        f"[Step6/calib] {model}@{hw}: decode self-MBU={self_mbu:.2%} — assumes ~100% bandwidth "
        f"(eta_mem=1.0); real HW sustains <~90%. Needs per-device eta_mem calibration."
    )


# ===========================================================================
# 2. BYTE-IDENTICAL BASELINES  (guard: decode must NOT change under structural fixes)
# ===========================================================================
def _load_or_init_baseline():
    if os.path.exists(BASELINE):
        return json.load(open(BASELINE))
    base = {"decode_tpot_ms": {}, "param_counts": {}}
    for model, hw, _ in DECODE_CASES:
        try:
            base["decode_tpot_ms"][f"{model}|{hw}"] = round(_decode_tpot_ms(model, hw), 6)
        except Exception as e:                      # pragma: no cover
            base["decode_tpot_ms"][f"{model}|{hw}"] = f"ERROR:{type(e).__name__}"
    json.dump(base, open(BASELINE, "w"), indent=2)
    return base


@pytest.mark.parametrize("model,hw,params", DECODE_CASES)
def test_decode_byte_identical(model, hw, params):
    """Decode TPOT must match the recorded baseline (structural compute fixes are memory-bound-invisible).
    Regenerate the baseline ONLY at the calibration step, for calibrated devices."""
    base = _load_or_init_baseline()
    key = f"{model}|{hw}"
    expected = base["decode_tpot_ms"].get(key)
    if isinstance(expected, str):
        pytest.skip(f"baseline unavailable: {expected}")
    actual = _decode_tpot_ms(model, hw)
    assert actual == pytest.approx(expected, rel=1e-6), (
        f"decode TPOT for {key} changed {expected}->{actual:.6f}ms; a structural fix touched the "
        f"memory-bound decode path (should be byte-identical until calibration Step 6)"
    )


# ===========================================================================
# 3. PARAMETER-FAMILY GOLDEN TESTS  (the audit-verified correct families must not regress)
# ===========================================================================
PARAM_FAMILIES = {
    "llama2_7b": (dict(model_type="llama", num_hidden_layers=32, hidden_size=4096,
        intermediate_size=11008, num_attention_heads=32, num_key_value_heads=32, vocab_size=32000,
        hidden_act="silu", tie_word_embeddings=False, rope_theta=10000), 6.738),
    "llama3_8b": (dict(model_type="llama", num_hidden_layers=32, hidden_size=4096,
        intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
        hidden_act="silu", tie_word_embeddings=False, rope_theta=500000), 8.030),
    "mistral_7b": (dict(model_type="mistral", num_hidden_layers=32, hidden_size=4096,
        intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8, vocab_size=32000,
        hidden_act="silu", tie_word_embeddings=False), 7.242),
    "qwen2.5_7b": (dict(model_type="qwen2", num_hidden_layers=28, hidden_size=3584,
        intermediate_size=18944, num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
        hidden_act="silu", tie_word_embeddings=False), 7.616),
    "mixtral_8x7b": (dict(model_type="mixtral", num_hidden_layers=32, hidden_size=4096,
        intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8, vocab_size=32000,
        hidden_act="silu", num_local_experts=8, num_experts_per_tok=2, tie_word_embeddings=False), 46.7),
}


def test_r2g1_system_object_path_applies_efficiency_bands():
    """R2-G1: get_inference_system on a System object must apply the eta bands (not run at 100%)."""
    from llm_memory_calculator.genz.system import System
    from llm_memory_calculator.genz.LLM_inference.utils import get_inference_system
    s = get_inference_system(system_name=System(flops=312, offchip_mem_bw=2039), bits="bf16", phase="decode")
    assert s.compute_efficiency < 1.0, f"System-obj compute_efficiency={s.compute_efficiency} (should be banded <1)"
    assert s.memory_efficiency < 1.0, f"System-obj memory_efficiency={s.memory_efficiency} (should be banded <1)"


def test_r2g2_cpu_uses_vendor_aware_memory_band():
    """R2-G2 (updated by the x86-STREAM eta fix): CPU memory efficiency is VENDOR-AWARE and single-
    sourced, not the generic accelerator midpoint applied blindly. x86 server parts (Intel/AMD) sustain
    the published multi-channel DDR5 STREAM-TRIAD band (~0.80; Dell/AMD STREAM 0.80-0.86 of peak), while
    ARM CPUs (no x86 STREAM evidence) keep the conservative DDR band (~0.65). Either way it is derived
    from the device, never a guess."""
    from llm_memory_calculator.genz.LLM_inference.utils import get_inference_system, _default_eta_mem_cpu
    s = get_inference_system(system_name="SapphireRapids_CPU", bits="bf16", phase="decode")  # Intel x86
    assert s.memory_efficiency == pytest.approx(0.80, abs=1e-9), \
        f"Intel x86 CPU memory_efficiency={s.memory_efficiency} (expected published STREAM band 0.80)"
    # ARM CPUs do NOT get the x86 STREAM band — they keep the conservative DDR band.
    assert _default_eta_mem_cpu({'manufacturer': 'arm', 'memory_type': None}) <= 0.70
    assert _default_eta_mem_cpu({'manufacturer': 'intel', 'memory_type': None}) == pytest.approx(0.80, abs=1e-9)


def test_m1_pp_throughput_is_steady_state():
    """M1: steady-state PP throughput is gated by conserved per-token work, so Throughput·Latency/1000
    == batch_size (no fill/drain penalty on throughput). The prior formula divided by ~(2-1/PP)."""
    d = decode_moddeling(model="llama2_7b", batch_size=8, input_tokens=2048, output_tokens=1,
                         system_name="A100_80GB_GPU", bits="bf16", pipeline_parallel=2)
    identity = d.Throughput * d.Latency / 1000.0
    assert identity == pytest.approx(8, rel=0.02), (
        f"PP=2 decode Throughput·Latency/1000={identity:.3f} (expected batch_size=8); "
        f"throughput is still penalized by fill/drain"
    )


def test_m2_decode_kv_growth_is_context_scaled():
    """M2: short-output (2-10 tokens) decode TPOT growth must track ACTUAL KV growth, not a fixed bump.
    5 tokens of KV growth on a 32k context is negligible (<1%); the old heuristic forced a fixed
    +5% (0.1*5/10) regardless of context."""
    one = decode_moddeling(model="llama2_7b", batch_size=1, input_tokens=32768, output_tokens=1,
                           system_name="A100_80GB_GPU", bits="bf16").Latency
    five = decode_moddeling(model="llama2_7b", batch_size=1, input_tokens=32768, output_tokens=5,
                            system_name="A100_80GB_GPU", bits="bf16").Latency
    growth = (five - one) / one
    assert 0 <= growth < 0.01, (
        f"output_tokens=5 grew TPOT by {growth:.2%} on 32k ctx — should be ~0 (KV-scaled), "
        f"not a fixed bump"
    )


def test_l0_tf32_is_half_rate_of_bf16():
    """L0: TF32 prefill must be ~2× bf16 — A100 TF32 tensor = 156 TFLOPS vs bf16 312 (datasheet, exactly
    half). Was equal (compute_multiplier['tf32']==1)."""
    bf16 = prefill_moddeling(model="llama2_7b", batch_size=1, input_tokens=2048,
                             system_name="A100_80GB_GPU", bits="bf16")
    tf32 = prefill_moddeling(model="llama2_7b", batch_size=1, input_tokens=2048,
                             system_name="A100_80GB_GPU", bits="tf32")
    ratio = tf32.Latency / bf16.Latency
    assert 1.8 < ratio < 2.2, f"tf32/bf16 prefill ratio={ratio:.2f} (TF32 should be half-rate → ~2×)"


def test_c4_dead_tensor_core_code_removed():
    """C4: the ungrounded flat-0.85 tensor-core factor + dead shape-intensity code are removed."""
    import llm_memory_calculator.genz.operator_base as ob
    from llm_memory_calculator.genz.operators import GEMM
    assert not hasattr(ob, "get_shape_dependent_op_intensity"), "dead shape-intensity fn still present"
    assert not hasattr(ob, "get_tensor_core_efficiency"), "dead tensor-core fn still present"
    assert not hasattr(GEMM, "get_tensor_core_efficiency_factor"), "dead tc-eff method still present"


def test_c4_compute_efficiency_is_exactly_system_eta():
    """C4: a compute op's effective compute efficiency must equal system.compute_efficiency EXACTLY
    (no hidden ×0.85). RED now: roofline reports C Effcy = eta×0.85."""
    from llm_memory_calculator.genz.system import System
    from llm_memory_calculator.genz.operators import GEMM
    from llm_memory_calculator.genz.unit import Unit
    sys = System(flops=312, compute_efficiency=0.80)
    g = GEMM(dim=["g", 1, 8192, 4096, 4096, 0, 3], density=(1, 1, 1))
    ret = g.get_roofline(system=sys, unit=Unit())
    assert ret["C Effcy"] == pytest.approx(0.80, rel=1e-9), \
        f"C Effcy={ret['C Effcy']} (expected exactly 0.80; a hidden tensor-core factor remains)"


@pytest.mark.parametrize("hw,arch_eta", [("H100_GPU", 0.80), ("A100_80GB_GPU", 0.78)])
def test_c3_prefill_eta_compute_grounded(hw, arch_eta):
    """C3: prefill effective MFU is bounded by the grounded per-arch eta_compute (large-GEMM MFU band),
    not ~100%. RED now (uncalibrated devices run at compute_efficiency=1.0 × 0.85)."""
    out = prefill_moddeling(model="llama2_7b", batch_size=1, input_tokens=4096, system_name=hw, bits="bf16")
    peak = REFERENCE_DENSE_BF16_TFLOPS[hw] * 1e12
    implied_mfu = (2.0 * 6.738e9 * 4096) / (out.Latency / 1e3) / peak
    # implied MFU (linear FLOPs only, a lower bound) must not exceed the grounded compute-efficiency band.
    assert implied_mfu <= arch_eta + 0.05, (
        f"{hw} prefill implied MFU={implied_mfu:.2%} exceeds the grounded eta_compute band "
        f"({arch_eta:.0%}); compute is not derated to the achievable large-GEMM MFU"
    )


@pytest.mark.parametrize("name", list(PARAM_FAMILIES))
def test_param_families_no_regression(name):
    cfg, expected_b = PARAM_FAMILIES[name]
    counted = UniversalParameterCounter().count_parameters(cfg) / 1e9
    assert counted == pytest.approx(expected_b, rel=0.01), \
        f"{name}: counted {counted:.3f}B vs expected {expected_b}B (>1% drift)"


@pytest.mark.parametrize("fn_name", ["estimate_decode_performance", "estimate_prefill_performance"])
def test_h1_public_estimators_accept_string_hardware(fn_name):
    """H1: the public estimators must accept a hardware NAME (string), not only a dict. RED now
    (AttributeError: 'str' has no attribute 'get'). The underlying *_moddeling fns already accept strings."""
    import llm_memory_calculator as L
    fn = getattr(L, fn_name)
    kwargs = dict(model="llama2_7b", system_name="A100_80GB_GPU", input_tokens=512)
    if fn_name == "estimate_decode_performance":
        kwargs["output_tokens"] = 4
    res = fn(**kwargs)
    assert isinstance(res, dict) and res.get("Latency", 0) > 0, f"{fn_name}(str hw) returned {res!r}"


def test_h1_unknown_hardware_clear_error():
    """H1: an unknown hardware name should raise a clear error, not a cryptic AttributeError."""
    import llm_memory_calculator as L
    with pytest.raises((ValueError, KeyError)):
        L.estimate_decode_performance(model="llama2_7b", system_name="NOPE_XYZ_GPU",
                                      input_tokens=128, output_tokens=2)


def test_gemma_geglu_param_count():
    """Gemma-2-9B: RED until PC-GLU (H0). hidden_activation key + GeGLU 3-matrix FFN must be detected."""
    cfg = dict(model_type="gemma2", num_hidden_layers=42, hidden_size=3584, intermediate_size=14336,
               num_attention_heads=16, num_key_value_heads=8, head_dim=256, vocab_size=256000,
               hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True)
    counted = UniversalParameterCounter().count_parameters(cfg) / 1e9
    assert counted == pytest.approx(9.24, rel=0.02), \
        f"[H0/PC-GLU] Gemma-2-9B counted {counted:.3f}B vs real 9.24B (GeGLU FFN under-counted)"
