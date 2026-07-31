"""The LoRA prefill scratch term is architecture-specific, and these are the measurements.

The term used to be `rank * layers * f(tokens)` with one global coefficient fitted on
Qwen3-0.6B. That coefficient silently encoded the 0.6B's `intermediate_size` of 3072
(3_217_660 / 3072 = 1047 B, within 20% of its measured 855), so every model since was
sized as though its FFN were 3072 wide. In production that under-sized a Qwen3-4B pod by
3.5x and it OOMKilled at "Warming up model for the compilation".

A five-architecture campaign settled the shape of the problem. Each row is a LoRA-on peak
minus a LoRA-off control run back-to-back on the same node, same engine args, cgroup
`memory.peak`, limits generous enough that nothing clipped and `oom_kills` stayed 0:

    model          architecture              hidden layers  inter  T     scratch
    Qwen3-0.6B     Qwen3ForCausalLM           1024    28    3072  4273    6.10 GiB
    Qwen3-4B       Qwen3ForCausalLM           2560    36    9728  4273   24.07 GiB
    Llama-3.1-8B   LlamaForCausalLM           4096    32   14336  4273   20.21 GiB
    Gemma-4-12B    Gemma4Unified              3840    48   15360  2048    8.63 GiB
    Qwen3.5-4B     Qwen3_5 (hybrid)           2560    32    9216  4273    0.21 GiB

The rate spans 100x and NO config dimension predicts it. Width fails: Llama at hidden
4096 sits below Qwen3-4B at 2560. Three width laws were fitted and rejected -- notably
one proportional to `intermediate_size` that agreed to 3% on two points and then missed
Llama by 62% out-of-sample. The two low outliers are the hybrid/sliding-window designs,
where most layers carry no full attention projections for LoRA to wrap; the determinant
is which modules vLLM's per-architecture LoRA support list wraps, which is engine state
rather than model metadata.

So the table is a record of measurements, not a model. Anything absent from it is sized
at the largest observed multiplier and says so.
"""

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator


GIB = 1024**3


class Lora:
    enabled = True
    max_loras = 1
    max_lora_rank = 64


def cfg(arch, hidden, layers):
    return {
        "architectures": [arch],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
    }


def scratch_gib(config, tokens, lora=None):
    calc = ModelMemoryCalculator()
    gb, notes = calc.calculate_lora_prefill_scratch(
        config, lora or Lora(), batched_tokens=tokens
    )
    return gb * 1e9 / GIB, notes


# ------------------------------------------------------------------ the measurements

MEASURED = [
    ("Qwen3ForCausalLM", 1024, 28, 4273, 6.10),
    ("Qwen3ForCausalLM", 2560, 36, 4273, 24.07),
    ("LlamaForCausalLM", 4096, 32, 4273, 20.21),
    ("Gemma4UnifiedForConditionalGeneration", 3840, 48, 2048, 8.63),
    ("Qwen3_5ForConditionalGeneration", 2560, 32, 4273, 0.21),
]


@pytest.mark.parametrize("arch,hidden,layers,tokens,measured", MEASURED)
def test_reproduces_every_measured_model(arch, hidden, layers, tokens, measured):
    """Consistency, not validation -- the multipliers were fitted to these points.

    Kept because a refactor that silently breaks the lookup would otherwise be invisible:
    the term would keep returning *a* plausible number, which is exactly how the previous
    coefficient survived being wrong for every model but one.
    """
    got, notes = scratch_gib(cfg(arch, hidden, layers), tokens)
    assert got == pytest.approx(measured, rel=0.02), f"{arch}@{hidden}"
    assert not notes, f"a measured model should carry no caveats, got: {notes}"


def test_out_of_sample_token_budget_errs_high():
    """The one point NOT fitted: Qwen3-4B at T=2048 (its multiplier came from T=4273).

    Measured 15.46 GiB (27.49 LoRA peak minus 12.03 control). The prediction is ~15% high
    -- the token curve is not exact for this architecture -- and that direction is the
    requirement, not an accident. The old formula predicted 5.11 GiB here, 3.03x LOW,
    which is the error that OOMKills.

    This point is deliberately excluded from the fit; folding it in would leave the table
    with no out-of-sample evidence at all.
    """
    got, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560, 36), 2048)
    measured = 15.46
    assert got > measured, "under-predicting is the failure mode that crash-loops pods"
    assert got == pytest.approx(measured, rel=0.25), (
        f"drifted from the observed +15%: {got}"
    )


# ------------------------------------------------------------------ the unmeasured cases


def test_unknown_architecture_is_conservative_and_says_so():
    """Unknown architectures must over-reserve, and must not do it silently."""
    got, notes = scratch_gib(cfg("MistralForCausalLM", 4096, 32), 4273)
    known, _ = scratch_gib(cfg("LlamaForCausalLM", 4096, 32), 4273)
    assert got > known, "an unmeasured arch must not be sized below a measured one"
    assert any("UNMEASURED" in n for n in notes)


def test_unmeasured_width_takes_the_family_maximum_not_an_interpolation():
    """Qwen3's rate TRIPLED between hidden 1024 and 2560, so interpolation is a guess.

    A midpoint width must be sized at the family's largest measured multiplier, not
    somewhere between -- the guess that OOMKills is the one that reads low.
    """
    mid, notes = scratch_gib(cfg("Qwen3ForCausalLM", 1536, 28), 4273)
    top, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560, 28), 4273)
    assert mid == pytest.approx(top, rel=0.01)
    assert any("extrapolated within" in n for n in notes)


def test_hybrid_is_two_orders_below_dense_at_the_same_width():
    """Qwen3.5-4B and Qwen3-4B are both hidden 2560; their scratch differs ~100x.

    This is the observation that killed every width-based law. If a future change makes
    these two converge, the fix has regressed to predicting from geometry.
    """
    hybrid, _ = scratch_gib(cfg("Qwen3_5ForConditionalGeneration", 2560, 32), 4273)
    dense, _ = scratch_gib(cfg("Qwen3ForCausalLM", 2560, 32), 4273)
    assert dense / hybrid > 50


def test_disabled_lora_costs_nothing():
    class Off:
        enabled = False

    got, notes = scratch_gib(cfg("Qwen3ForCausalLM", 2560, 36), 4273, lora=Off())
    assert got == 0.0 and not notes


def test_tensor_parallel_shards_the_term():
    calc = ModelMemoryCalculator()
    c = cfg("Qwen3ForCausalLM", 2560, 36)
    one, _ = calc.calculate_lora_prefill_scratch(c, Lora(), batched_tokens=4273)
    two, _ = calc.calculate_lora_prefill_scratch(
        c, Lora(), batched_tokens=4273, tensor_parallel=2
    )
    assert two == pytest.approx(one / 2, rel=1e-6)
