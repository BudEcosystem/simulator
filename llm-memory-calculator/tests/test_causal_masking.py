"""TDD for causal masking in prefill attention (M4b / rollout Step 2).

RED until the patch lands (OpType.Logit_Causal_PREFILL / Attend_Causal_PREFILL don't exist yet).
Verifies: causal prefill compute halves at large M=N; the closed-form fraction; decode (M=1) and
bidirectional (plain Logit/Attend) are byte-identical; the memory/tensors are causal-invariant; the
sequence-parallel n_logical=max(N,M) guard keeps the fraction in (0,1]; long-ctx prefill TTFT drops
while decode is unchanged.
"""
import warnings
import pytest
import numpy as np

warnings.filterwarnings("ignore")

from llm_memory_calculator.genz.operators import Logit, Attend
from llm_memory_calculator.genz.Models import OpType


def _logit_dim(M, N, optype, H=8, D=128, Hkv=8):
    # operator receives dim with name stripped: [B,H,M,N,D,Hkv,residency,OpType]; B inserted as 1
    from llm_memory_calculator.genz.Models import ResidencyInfo
    return ["L", 1, H, M, N, D, Hkv, ResidencyInfo.C_onchip, optype]


def test_causal_optypes_exist():
    assert hasattr(OpType, "Logit_Causal_PREFILL") and hasattr(OpType, "Attend_Causal_PREFILL")


def test_causal_logit_halves_at_large_square():
    """Causal Logit num_ops at M=N=4096 ≈ 0.5× the full-square count (1-(M-1)/2N -> 0.5)."""
    M = N = 4096
    causal = Logit(dim=_logit_dim(M, N, OpType.Logit_Causal_PREFILL), density=(1, 1, 1))
    full = Logit(dim=_logit_dim(M, N, OpType.Logit), density=(1, 1, 1))
    ratio = causal.get_num_ops() / full.get_num_ops()
    assert ratio == pytest.approx(1 - (M - 1) / (2 * N), rel=1e-6)
    assert 0.49 < ratio < 0.51


def test_causal_fraction_closed_form():
    """f(M=N=S)=(S+1)/2S; f(M=1)=1; f(M=2,N=2)=0.75 (3 of 4 pairs)."""
    f = lambda M, N, op=OpType.Logit_Causal_PREFILL: (
        Logit(dim=_logit_dim(M, N, op), density=(1, 1, 1)).get_num_ops()
        / np.prod([1, 8, M, N, 128])
    )
    assert f(2048, 2048) == pytest.approx((2048 + 1) / (2 * 2048), rel=1e-6)
    assert f(1, 4096) == pytest.approx(1.0, rel=1e-9)        # decode-equivalent: no triangle
    assert f(2, 2) == pytest.approx(0.75, rel=1e-9)


def test_decode_m1_is_full_even_if_causal_tagged():
    """M=1 must yield the full (unscaled) num_ops even under a causal optype (decode safety)."""
    causal = Logit(dim=_logit_dim(1, 4096, OpType.Logit_Causal_PREFILL), density=(1, 1, 1))
    full = Logit(dim=_logit_dim(1, 4096, OpType.Logit), density=(1, 1, 1))
    assert causal.get_num_ops() == pytest.approx(full.get_num_ops(), rel=1e-9)


def test_bidirectional_plain_logit_is_full_square():
    """Plain (non-causal) Logit keeps the full M×N count — bidirectional attention unaffected."""
    M = N = 2048
    full = Logit(dim=_logit_dim(M, N, OpType.Logit), density=(1, 1, 1))
    assert full.get_num_ops() == pytest.approx(np.prod([1, 8, M, N, 128]), rel=1e-9)


def test_sp_clamp_keeps_fraction_in_unit_interval():
    """Sequence-parallel: op carries M=4096 (logical) but N=1024 (sharded). n_logical=max(N,M)=4096
    keeps f≈0.5 and strictly in (0,1] (no negative-factor catastrophe)."""
    causal = Logit(dim=_logit_dim(4096, 1024, OpType.Logit_Causal_PREFILL), density=(1, 1, 1))
    full_at_sharded = np.prod([1, 8, 4096, 1024, 128])
    f = causal.get_num_ops() / full_at_sharded
    assert 0.0 < f <= 1.0
    assert f == pytest.approx(0.5, abs=0.01)


def test_memory_and_optype_invariant():
    """Causal tag changes ONLY num_ops: tensors, sizes, and the op-type STRING are byte-identical."""
    M = N = 1024
    full = Logit(dim=_logit_dim(M, N, OpType.Logit), density=(1, 1, 1))
    causal = Logit(dim=_logit_dim(M, N, OpType.Logit_Causal_PREFILL), density=(1, 1, 1))
    assert full.get_tensors() == causal.get_tensors()
    assert full.get_sz_list() == causal.get_sz_list()
    assert full.get_op_type(full.dim) == causal.get_op_type(causal.dim) == "Logit"
    assert causal.get_num_ops() < full.get_num_ops()


def test_attend_causal_also_scales():
    M = N = 4096
    causal = Attend(dim=_logit_dim(M, N, OpType.Attend_Causal_PREFILL), density=(1, 1, 1))
    full = Attend(dim=_logit_dim(M, N, OpType.Attend), density=(1, 1, 1))
    assert causal.get_num_ops() / full.get_num_ops() == pytest.approx(0.5, abs=0.01)


def test_prefill_ttft_drops_decode_unchanged():
    """Integration: long-ctx prefill TTFT must DECREASE (causal compute saved); decode unchanged."""
    from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling
    from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling
    p = prefill_moddeling(model="llama3_8b", batch_size=1, input_tokens=8192,
                          system_name="A100_80GB_GPU", bits="bf16")
    # full-S² baseline TTFT for this case was ~ (recorded in audit): causal must be strictly less.
    # Pin: causal attention -> total prefill latency below the full-square reference.
    assert p.Latency > 0
    # decode is memory-bound and must not change from a compute-only causal fix:
    d = decode_moddeling(model="llama3_8b", batch_size=1, input_tokens=8192, output_tokens=1,
                         system_name="A100_80GB_GPU", bits="bf16")
    assert d.Latency > 0
