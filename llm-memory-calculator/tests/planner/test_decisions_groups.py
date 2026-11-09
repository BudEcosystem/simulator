"""Bud spec 031 DM3 (FRD D-30): a decision call is a group of one-token reads over one shared state.

TC-SZ-01: an ungrouped workload predicts exactly what it did before groups existed.
TC-SZ-02: a group's shared prefix KV is counted once (with the prefix cache on), and every read's is counted with it off.
TC-SZ-03: the group hit rate: ``fresh`` computes the state once per group, ``repeated`` hits it on every read.
"""

import json
import math

import pytest

from llm_memory_calculator.device import catalogue
from llm_memory_calculator.engine import Workload
from llm_memory_calculator.engine.vllm import group_hit_rate, group_kv_capacity

from .decisions_baseline import BASELINE, cases, engine, model, predict

L40S = catalogue.get("l40s-48gb")
RTX3050 = catalogue.get("rtx3050-laptop-4gb")
Q8 = model("qwen3-8b")
Q27 = model("qwen3.8-27b")


def call(q, s=1024, question=48, calls=1, reuse="fresh", **extra):
    return Workload(
        input_tokens=s + question,
        output_tokens=1,
        concurrency=q * calls,
        prefix_tokens=s,
        arrival="wave",
        group_size=q,
        prefix_reuse=reuse,
        **extra,
    )


# ---------------------------------------------------------------- TC-SZ-01


@pytest.mark.parametrize("key, m, d, w", list(cases()))
def test_ungrouped_predictions_are_unchanged(key, m, d, w):
    baseline = json.loads(BASELINE.read_text())[key]
    assert predict(m, d, w) == baseline
    assert predict(m, d, w, group_size=1) == baseline, "group_size = 1 is the ungrouped workload"


def test_the_digest_of_an_ungrouped_workload_is_unchanged():
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=16)
    assert w.digest_key() == Workload(input_tokens=1024, output_tokens=256, concurrency=16, group_size=1).digest_key()
    assert len(w.digest_key()) == 12, "the pre-group key shape: caches keyed on it stay valid"
    assert call(64).digest_key() != call(32, calls=2).digest_key()
    assert call(64).digest_key() != call(64, reuse="repeated").digest_key()


def test_workload_validates_its_group_fields():
    with pytest.raises(ValueError):
        call(0)
    with pytest.raises(ValueError):
        call(8, reuse="sometimes")


# ---------------------------------------------------------------- TC-SZ-02


def test_a_group_counts_its_shared_prefix_once():
    lim = engine().limits({}, Q8, L40S)
    assert lim.block_tokens == 16
    # 64 reads of 1,024 + 48 tokens, one output token each
    grouped = group_kv_capacity(lim, 1072, 1, prefix_tokens=1024, group_size=64, prefix_cache=True)
    assert grouped == lim.kv_tokens // (1024 + 64 * (48 + 1))
    uncached = group_kv_capacity(lim, 1072, 1, prefix_tokens=1024, group_size=64, prefix_cache=False)
    assert uncached == lim.kv_tokens // (64 * (1072 + 1))
    # 16.5x the calls at these lengths (whole calls, so the floors move the ratio of small counts)
    assert grouped / uncached == pytest.approx(64 * 1073 / (1024 + 64 * 49), rel=0.25)
    assert grouped > 10 * uncached


def test_only_whole_blocks_of_the_prefix_are_shared():
    lim = engine().limits({}, Q8, L40S)
    # a 1,030-token prefix shares 64 blocks (1,024 tokens); each read holds its last 6 prefix tokens itself
    got = group_kv_capacity(lim, 1030 + 48, 1, prefix_tokens=1030, group_size=8, prefix_cache=True)
    assert got == lim.kv_tokens // (1024 + 8 * (6 + 48 + 1))
    # a prefix shorter than one block shares nothing
    got = group_kv_capacity(lim, 10 + 48, 1, prefix_tokens=10, group_size=8, prefix_cache=True)
    assert got == lim.kv_tokens // (8 * (10 + 48 + 1))


def test_a_hybrid_group_keeps_its_per_read_state():
    lim = engine().limits({"batch.max_running": 256}, Q27, catalogue.get("h100xm-80c-vgpu"))
    assert lim.state_bytes_per_sequence > 0
    block = lim.block_tokens
    groups = group_kv_capacity(lim, 1072, 1, prefix_tokens=1024, group_size=16, prefix_cache=True)
    shared = (1024 // block) * block
    per_read = lim.attn_groups * math.ceil((1072 - shared + 1) / block) + lim.state_bytes_per_sequence / lim.block_bytes
    group_blocks = lim.attn_groups * (shared // block) + 16 * per_read
    assert groups == int(lim.num_blocks // group_blocks)


def test_the_prediction_reports_how_many_calls_a_replica_holds():
    eng = engine()
    p = eng.predict({}, call(64), Q8, L40S)
    lim = p.limits
    kv_groups = lim.kv_tokens // (1024 + 64 * 49)
    batch_groups = eng.resolve({}, Q8, L40S).max_running // 64
    assert p.group_ceiling == min(kv_groups, batch_groups)
    # a batch limit below one call holds no call as one batch: the planner refuses it (D-31)
    p = eng.predict({"batch.max_running": 32}, call(64), Q8, L40S)
    assert p.group_ceiling == 0
    assert any("does not fit one batch" in n for n in p.notes)
    assert eng.predict({}, Workload(input_tokens=1024, output_tokens=8, concurrency=4), Q8, L40S).group_ceiling is None


def test_running_reads_are_whole_groups():
    eng = engine()
    p = eng.predict({"batch.max_running": 100}, call(64, calls=4), Q8, L40S)
    assert p.running == 64, "one whole call runs; the second does not fit beside it"
    assert p.queueing


# ---------------------------------------------------------------- TC-SZ-03


def test_group_hit_rates():
    cached = 1024 / 1072
    assert group_hit_rate(1072, 1024, 64, "fresh", block_tokens=16) == pytest.approx((1 - 1 / 64) * cached)
    assert group_hit_rate(1072, 1024, 64, "repeated", block_tokens=16) == pytest.approx(cached)
    assert group_hit_rate(1072, 1024, 1, "fresh", block_tokens=16) == 0.0, "a lone fresh read computes its state"
    assert group_hit_rate(58, 10, 64, "repeated", block_tokens=16) == 0.0, "under one block nothing is cached"


def test_the_hit_rate_reaches_the_prefill():
    eng = engine()
    fresh = eng.predict({}, call(64), Q8, L40S)
    repeated = eng.predict({}, call(64, reuse="repeated"), Q8, L40S)
    cold = eng.predict({"prefix_cache.enabled": False}, call(64), Q8, L40S)
    assert repeated.group_latency_ms.value < fresh.group_latency_ms.value < cold.group_latency_ms.value
    # the spike: without prefix caching a 64-question fresh call was ~21x slower (RTX 3050); the model keeps the order
    assert cold.group_latency_ms.value > 5 * fresh.group_latency_ms.value


def test_call_latency_covers_the_whole_group():
    eng = engine()
    p = eng.predict({}, call(64), Q8, L40S)
    assert p.group_latency_ms.value >= p.ttft_ms.value, "a call returns with its last read, not its median"
    one = eng.predict({}, call(1), Q8, L40S)
    assert one.group_latency_ms is None and one.group_ceiling is None, "a one-question call is an ungrouped read"
    more = eng.predict({}, call(64, calls=4), Q8, L40S)
    assert more.group_latency_ms.value > p.group_latency_ms.value, "calls in flight share the engine"


# ---------------------------------------------------------------- the step-profile predictor (R2)


def test_the_profile_predictor_sizes_groups_the_same_way():
    from llm_memory_calculator.perf.store import ProfileStore

    from .test_perf import VGPU_SKU, _engine, _profile, _q8

    vgpu = catalogue.get(VGPU_SKU)
    eng = _engine(ProfileStore([_profile()]))
    fresh = eng.predict({}, call(64, calls=2), _q8(), vgpu)
    assert any(n.startswith("predictor: step profile") for n in fresh.notes)
    lim = fresh.limits
    kv_groups = group_kv_capacity(lim, 1072, 1, prefix_tokens=1024, group_size=64, prefix_cache=True)
    assert fresh.group_ceiling == min(kv_groups, eng.resolve({}, _q8(), vgpu).max_running // 64)
    assert fresh.group_latency_ms is not None
    assert fresh.group_latency_ms.value >= fresh.ttft_ms.value
    repeated = eng.predict({}, call(64, calls=2, reuse="repeated"), _q8(), vgpu)
    assert repeated.group_latency_ms.value < fresh.group_latency_ms.value
    tight = eng.predict({"batch.max_running": 32}, call(64), _q8(), vgpu)
    assert tight.group_ceiling == 0


# ---------------------------------------------------------------- the wave's last read


def test_a_burst_just_past_one_chunk_is_not_two_chunks_of_work():
    """The call latency grows with the burst's own tokens: crossing a prefill chunk adds the remainder's time, not a
    whole second chunk (before the fix 80 reads on the RTX 3050 predicted 1.48 s against 0.79 s measured)."""
    eng = engine()
    previous = None
    for calls in range(1, 13):
        lat = eng.predict({}, call(16, calls=calls), Q8, L40S).group_latency_ms.value
        if previous is not None:
            assert lat >= previous * 0.999, "more calls in flight never answer sooner"
            # at most proportional to the burst (plus a little), never a whole extra chunk at a boundary
            bound = previous * calls / (calls - 1) * 1.1
            assert lat <= bound, f"a step at {calls} calls: {previous:.0f} -> {lat:.0f} ms"
        previous = lat


# ---------------------------------------------------------------- TC-SZ-05 on the RTX 3050 (measured)


def test_measured_rtx3050_calls_are_within_25_percent():
    """The DM3 e2e sweep on the lab's RTX 3050: median call latency at each calls-in-flight level, predicted from
    priors only. With several calls in flight (what a plan sizes) it is within +-25% (FRD §5.15 acceptance); every
    point is inside the prediction's band. A lone call is under-predicted (0.65-0.75x): a fixed per-call cost of
    ~60-100 ms the priors do not carry, which WP-DM3.2's decisions grounding points are for."""
    from .decisions_baseline import FIXTURES

    measured = json.loads((FIXTURES / "decisions" / "rtx3050_qwen3_0.6b.json").read_text())["rows"]
    q06 = model("qwen3-0.6b")
    for row in measured:
        w = Workload(
            input_tokens=row["state_tokens"] + row["question_tokens"],
            output_tokens=1,
            concurrency=row["questions_per_call"] * row["calls_in_flight"],
            prefix_tokens=row["state_tokens"],
            arrival="wave",
            group_size=row["questions_per_call"],
        )
        p = engine().predict({"batch.max_running": 256}, w, q06, RTX3050)
        est = p.group_latency_ms
        ratio = est.value / row["median_ms"]
        assert est.low <= row["median_ms"] <= est.high, (row, est.to_dict())
        if row["calls_in_flight"] > 1:
            assert 0.75 <= ratio <= 1.25, (row, round(est.value, 1))
        else:
            assert 0.6 <= ratio < 1.0, "the known lone-call gap; tighten once grounded"
