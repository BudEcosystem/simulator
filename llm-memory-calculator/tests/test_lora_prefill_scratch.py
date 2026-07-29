"""LoRA prefill-scratch regression tests, pinned to measured vLLM CPU pods.

Serving a LoRA-enabled model allocates two very different things:

  * the adapter A/B weights, which are small (0.22 GiB at rank 256 on a 0.6B
    model) and were the only thing this library reported, and
  * Punica's prefill working buffers, which at the same rank are ~16-21 GiB and
    were reported as nothing at all.

A control plane sizing a pod off the first number under-sizes it by ~75x and the
pod OOMKills at ``Warming up model for the compilation``. These tests pin the
second term against pods that were actually run.

Measurement campaign (Qwen3-0.6B, 28 layers, rank 256, ``max_num_seqs=1``, vLLM
CPU backend, cgroup limit 40 GiB so nothing was clipped -- an earlier pass read
capped pods and mistook the cap for the peak):

    batched tokens | LoRA off  | LoRA on   | scratch
    ---------------|-----------|-----------|-----------
    2394           |  5143 MiB | 22311 MiB | 16.77 GiB
    4273           |  5113 MiB | 27119 MiB | 21.49 GiB
    8192           |  5096 MiB | OOM @ 50G | > 45.02 GiB

Two facts fall out and are pinned below: base memory is *flat* in the token
budget (47 MiB across a 3.4x range -- all the context sensitivity is LoRA), and
the scratch term is *convex*, so the linear fit is honest only up to the
calibrated ceiling and must say so beyond it.
"""

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.lora.config import LoraConfig


GIB = 2**30

# The measured pod, verbatim from its config.json.
QWEN3_0_6B = dict(
    model_type="qwen3",
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=3072,
    vocab_size=151936,
    max_position_embeddings=40960,
    torch_dtype="bfloat16",
)


def scratch_gib(batched_tokens, rank=256, max_loras=1, tensor_parallel=1, config=None):
    """Scratch for one configuration, in GiB, discarding notes."""
    calc = ModelMemoryCalculator()
    lora = LoraConfig(enabled=True, max_loras=max_loras, max_lora_rank=rank)
    gb, _ = calc.calculate_lora_prefill_scratch(
        config or QWEN3_0_6B, lora, batched_tokens, tensor_parallel
    )
    return gb * 1e9 / GIB


def notes_for(batched_tokens, rank=256, config=None):
    calc = ModelMemoryCalculator()
    lora = LoraConfig(enabled=True, max_loras=1, max_lora_rank=rank)
    _, notes = calc.calculate_lora_prefill_scratch(
        config or QWEN3_0_6B, lora, batched_tokens
    )
    return notes


# ------------------------------------------------------------------ the measured points


@pytest.mark.parametrize(
    "batched_tokens, measured_gib",
    [
        (4273, 5.14),   # 10.85 GiB anon - 5.70 GiB same-series LoRA-off control
        (7310, 10.39),  # 16.08 GiB anon - 5.70 GiB same-series LoRA-off control
    ],
)
def test_reproduces_measured_scratch(batched_tokens, measured_gib):
    """The two same-series rank-64 points, straddling the knee.

    Both are cgroup ``anon`` deltas against a LoRA-off control run minutes apart on
    the same node, at limits generous enough that nothing clipped. A repeat of the
    7310 run landed within 1 MiB, so the term is deterministic.

    The model is allowed to sit ABOVE measurement (it is a pod budget, and short
    kills the pod) but only by a few percent -- 5.37 vs 5.14 and 10.62 vs 10.39.
    """
    got = scratch_gib(batched_tokens, rank=64)
    assert got >= measured_gib, "a budget below measurement OOMKills the pod"
    assert got == pytest.approx(measured_gib, rel=0.06)


def test_the_token_slope_has_a_knee():
    """A single slope fitted below 4273 under-predicts 7310 by 43%.

    That is not academic: it is what left a production pod pinned against its
    cgroup ceiling, hitting the limit 196 times and surviving only by evicting
    page cache. The steep segment is 2.75x the shallow one.
    """
    calc = ModelMemoryCalculator()
    below = scratch_gib(4273, rank=64)
    above = scratch_gib(7310, rank=64)
    shallow = (below - scratch_gib(2394, rank=64)) / (4273 - 2394)
    steep = (above - below) / (7310 - 4273)
    assert steep > 2.5 * shallow
    assert calc.LORA_SCRATCH_KNEE_TOKENS == 4273

    # a single-slope model would have said this, and it is 3 GiB short
    single = scratch_gib(4273, rank=64) + shallow * (7310 - 4273)
    assert above - single > 3.0


def test_scratch_dwarfs_adapter_storage():
    """The term this library already had is ~1.3% of the term it was missing.

    This is the whole reason the scratch is reported as its own field: summed
    into ``lora_adapter_memory_bytes`` it would look like a rounding error on a
    number a reader already believes is small.
    """
    report = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=2394,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=2394,
    )
    storage = report.lora_adapter_memory_bytes / GIB
    scratch = report.lora_prefill_scratch_bytes / GIB
    assert storage == pytest.approx(0.22, abs=0.05)
    assert scratch == pytest.approx(16.77, abs=0.05)
    assert scratch > 50 * storage


# ------------------------------------------------------------------ what drives it


def test_sized_by_batch_not_context():
    """A 8192-context deployment with chunked prefill pays for the chunk.

    vLLM's CPU backend does support chunked prefill, so a long-context pod need
    not allocate a long-context transient. Sizing this off ``seq_length`` would
    charge 31 GiB for a pod that measured 20631 MiB in total.
    """
    long_context_small_batch = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=8192,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=2048,
    )
    unchunked = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=8192,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=8192,
    )
    assert (
        long_context_small_batch.lora_prefill_scratch_bytes
        < unchunked.lora_prefill_scratch_bytes / 1.9
    )


def test_defaults_to_seq_length_when_batch_unset():
    """Callers that never learned about the batch budget get the safe reading.

    ``max_num_batched_tokens`` unset means "unknown", and an engine that has not
    been told to chunk prefills the whole context in one pass.
    """
    explicit = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=2394,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=2394,
    )
    implied = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=2394,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
    )
    assert implied.lora_prefill_scratch_bytes == explicit.lora_prefill_scratch_bytes


def test_scales_with_rank():
    """Rank is the lever that makes this term affordable: 256 -> 64 is 4x off."""
    assert scratch_gib(2394, rank=64) == pytest.approx(
        scratch_gib(2394, rank=256) / 4, rel=1e-6
    )


def test_shards_across_tensor_parallel():
    """Per-rank working memory, so tp=2 halves the per-process figure."""
    assert scratch_gib(2394, tensor_parallel=2) == pytest.approx(
        scratch_gib(2394) / 2, rel=1e-6
    )


def test_zero_without_lora():
    report = calculate_memory(
        model_id_or_config=QWEN3_0_6B, batch_size=1, seq_length=2394, precision="bf16"
    )
    assert report.lora_prefill_scratch_bytes == 0.0
    assert report.notes == []


# ------------------------------------------------------------------ honesty about limits


def test_past_the_ceiling_gets_margin_and_says_so():
    """A bare warning was not enough, so past the ceiling the term carries margin.

    The previous revision returned the un-margined figure and labelled it a LOWER
    BOUND. A T=17600 pod shipped on it and crash-looped -- its KV check found 8.8 GiB
    free where it needed 10.0, short by 1.21 GiB across 5 identical restarts. The
    note had fired; the pod died regardless.
    """
    calc = ModelMemoryCalculator()
    notes = notes_for(8192, rank=64)
    assert any("UNCALIBRATED" in n and "margin is applied" in n for n in notes)

    # margin is arithmetic, not just wording
    knee, ceil = calc.LORA_SCRATCH_KNEE_TOKENS, calc.LORA_SCRATCH_CALIBRATED_MAX_TOKENS
    units = 1 * 64 * 28
    raw = units * (
        calc.LORA_SCRATCH_BYTES_PER_UNIT
        + calc.LORA_SCRATCH_BYTES_PER_TOKEN_UNIT * min(8192, knee)
        + calc.LORA_SCRATCH_BYTES_PER_TOKEN_UNIT_STEEP * max(0, 8192 - knee)
    ) / 1e9
    assert 8192 > ceil
    assert scratch_gib(8192, rank=64) == pytest.approx(
        raw * 1e9 / GIB * calc.LORA_SCRATCH_EXTRAPOLATION_MARGIN, rel=1e-6
    )


def test_the_failing_production_config_would_now_fit():
    """T=17600 crash-looped at 28.41 GiB planned; >=29.62 GiB was needed."""
    assert scratch_gib(17600, rank=64) > 29.62


def test_no_note_inside_the_measured_envelope():
    assert notes_for(4273, rank=64) == []
    assert notes_for(7310, rank=64) == []


def test_flags_depth_extrapolation():
    """Depth has never been measured at a second value, so the direction is unknown."""
    deeper = dict(QWEN3_0_6B, num_hidden_layers=80)
    note = next(n for n in notes_for(2048, config=deeper) if "extrapolated in depth" in n)
    assert "assumed, not" in note


def test_only_rank_64_is_treated_as_characterised():
    """Rank 256 is NOT safe to size, and the model must say so.

    The rank-256 numbers this library shipped came from max_num_seqs=1 pods. At rank
    64 the scratch is seqs-independent (1 vs 2 differ by 54 MiB), which made that look
    generalisable -- but a rank-256 pod at T=4273 with seqs=2 OOMKilled at a 55 GiB
    limit, where linear-in-rank predicts 20.6 GiB. So the ceiling sits at the only
    rank actually characterised.

    An earlier revision claimed rank^1.29 from a cross-campaign comparison; the
    same-series data puts rank 64 and 256 within 4.5% of linear at T=4273. That claim
    is withdrawn, and this test exists so it is not quietly reintroduced.
    """
    calc = ModelMemoryCalculator()
    assert calc.LORA_SCRATCH_CALIBRATED_RANK == 64
    assert notes_for(4273, rank=64) == []
    assert any("UNVALIDATED at rank" in n and "LOWER BOUND" in n for n in notes_for(4273, rank=256))
    import inspect
    assert "rank^1.29" not in inspect.getsource(calc.calculate_lora_prefill_scratch)


def test_notes_reach_the_report():
    """A note dropped between the calculator and the report is a note that does not exist."""
    report = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=8192,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=8192,
    )
    assert any("UNCALIBRATED" in n for n in report.notes)


# ------------------------------------------------------------------ refusals


def test_unsized_rather_than_guessed_when_geometry_is_missing():
    """No layer count means no answer -- and the caller is told, not left at 0."""
    calc = ModelMemoryCalculator()
    lora = LoraConfig(enabled=True, max_loras=1, max_lora_rank=256)
    gb, notes = calc.calculate_lora_prefill_scratch({"model_type": "qwen3"}, lora, 2048)
    assert gb == 0.0
    assert any("no layer count" in n for n in notes)

    gb, notes = calc.calculate_lora_prefill_scratch(
        QWEN3_0_6B, LoraConfig(enabled=True, max_loras=1, max_lora_rank=0), 2048
    )
    assert gb == 0.0
    assert any("max_lora_rank" in n for n in notes)


def test_counted_in_the_total():
    """The total is what a pod gets sized against; an uncounted 16 GiB is an OOM."""
    with_lora = calculate_memory(
        model_id_or_config=QWEN3_0_6B,
        batch_size=1,
        seq_length=2394,
        precision="bf16",
        max_loras=1,
        max_lora_rank=256,
        max_num_batched_tokens=2394,
    )
    without = calculate_memory(
        model_id_or_config=QWEN3_0_6B, batch_size=1, seq_length=2394, precision="bf16"
    )
    delta = (with_lora.total_memory_bytes - without.total_memory_bytes) / GIB
    assert delta == pytest.approx(16.77 + 0.22, abs=0.1)
