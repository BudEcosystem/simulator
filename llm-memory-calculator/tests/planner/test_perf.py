"""FRD-027 Revision 2 (§20): step profiles (L2/L3) and the scheduler simulation (L4)."""

import math

import pytest

from llm_memory_calculator.perf.profile import StepProfile


def _steps_file(decode, prefill_rate=0.02, chunk=8192, first_step_ms=4.0):
    """A synthetic profiler output: decode(b, ctx) -> ms, and a prefill phase at ``prefill_rate`` ms per token. As on
    the engine, async scheduling returns the first step after launching the first chunk (no tokens yet), so that
    chunk's time is split between the first two steps (TCS smoke run: 14 + 168 ms for an 8,192-token chunk)."""
    points = []
    for (b, length), ms in decode.items():
        n_chunks = -(-b * length // chunk)
        steps = [{"ms": first_step_ms, "decoding": 0, "waiting": b, "new_tokens": 0}]
        done = 0
        for k in range(n_chunks):
            tokens = min(chunk, b * length - k * chunk)
            finished = min(b, ((k + 1) * chunk) // length) if length <= chunk else (b if k == n_chunks - 1 else 0)
            cost = prefill_rate * tokens - (first_step_ms if k == 0 else 0.0)
            steps.append({"ms": cost, "decoding": done, "waiting": b - done, "new_tokens": finished - done})
            done = finished
        steps += [{"ms": ms, "decoding": b, "waiting": 0, "new_tokens": b} for _ in range(40)]
        points.append({"b": b, "len": length, "steps": steps, "decode_ms": ms, "decode_ctx": length + 20,
                       "preemptions": 0.0})
    return {"schema": "bud-perf-steps/1", "name": "synthetic", "model": "/m", "vllm": "0.30.0", "load_format": "dummy",
            "chunk": chunk, "max_num_seqs": 256, "max_model_len": 33280, "gpu_util": 0.9, "extra": {},
            "kv_blocks": 20000, "block_tokens": 16, "decode_steps": 40, "points": points}


def _decode(b, ctx):
    return 6.0 + 0.02 * b + 0.00002 * b * ctx


GRID = {(b, length): _decode(b, length + 20) for b in (1, 4, 16, 64) for length in (256, 1024, 4096)}


def test_a_profile_returns_its_measurements_at_the_grid():
    """T-STEP-03: at a measured (batch, context) the decode step is the measurement itself."""
    prof = StepProfile.from_steps(_steps_file(GRID), sku="h100xm-80c-vgpu", engine="vllm@0.30.0", model="m")
    for (b, length), ms in GRID.items():
        assert prof.decode_ms(b, length + 20) == pytest.approx(ms, rel=1e-6)


def test_between_grid_points_the_step_is_interpolated_in_batch_and_context():
    """T-STEP-03: piecewise-linear in batch and linear in context (attention reads grow with the context), so a step
    that is linear in both is recovered exactly between grid points, and extrapolated linearly past them."""
    prof = StepProfile.from_steps(_steps_file(GRID), sku="h100xm-80c-vgpu", engine="vllm@0.30.0", model="m")
    for b, ctx in ((8, 2000), (32, 600), (2, 3000), (48, 4116)):
        assert prof.decode_ms(b, ctx) == pytest.approx(_decode(b, ctx), rel=0.02), (b, ctx)
    assert prof.decode_ms(128, 1044) == pytest.approx(_decode(128, 1044), rel=0.05)  # past the largest batch


def test_the_profile_reports_its_own_interpolation_error():
    """T-PROF-02: a leave-one-out pass over the decode grid gives the band's interpolation term; a curved step (a
    batch knee the grid is too coarse to place) shows up as a larger error."""
    linear = StepProfile.from_steps(_steps_file(GRID), sku="s", engine="e", model="m")
    assert linear.interpolation_error() < 0.02
    knee = {k: (v if k[0] < 64 else 3 * v) for k, v in GRID.items()}
    assert StepProfile.from_steps(_steps_file(knee), sku="s", engine="e", model="m").interpolation_error() > 0.2


def test_prefill_is_priced_per_token_from_the_measured_phase():
    """T-STEP-04: the prefill phase (every step until the last prompt's first token) over its tokens gives the cost
    per token, by prompt length; a chunk costs that times its tokens."""
    prof = StepProfile.from_steps(_steps_file(GRID, prefill_rate=0.02), sku="s", engine="e", model="m")
    assert prof.chunk_ms(8192, 1024) == pytest.approx(8192 * 0.02, rel=1e-6)
    assert prof.prefill_ms(256, 256) == pytest.approx(256 * 0.02, rel=1e-6)
    assert prof.prefill_ms(64 * 4096, 4096) == pytest.approx(64 * 4096 * 0.02, rel=1e-6)  # 32 chunks


def test_a_preempted_point_is_kept_out_of_the_decode_table():
    """T-STEP-03: a point the engine preempted measured recompute, not a decode step (R0: the 27B at 96-128
    sequences); it is recorded for the scheduler's validation, never interpolated."""
    data = _steps_file(GRID)
    bad = dict(data["points"][-1])
    bad.update({"b": 128, "len": 4096, "decode_ms": 999.0, "preemptions": 18.0})
    data["points"].append(bad)
    prof = StepProfile.from_steps(data, sku="s", engine="e", model="m")
    assert prof.decode_ms(128, 4116) < 100
    assert (128, 4096) in prof.preempted


# ---------------------------------------------------------------- L4 the scheduler simulation


class _Linear:
    """A step model with known costs: decode a + b*B + c*B*ctx ms; prefill r ms per token."""

    def __init__(self, a=6.0, b=0.02, c=0.00002, r=0.02):
        self.a, self.b, self.c, self.r = a, b, c, r

    def decode_ms(self, n, ctx):
        return self.a + self.b * n + self.c * n * ctx

    def chunk_ms(self, tokens, request_len):
        return self.r * tokens


def _sim(model=None, *, c=1, n=1, prompt=1024, out=256, seqs=256, blocks=100000, block=16, chunk=8192, api=0.0, **kw):
    from llm_memory_calculator.perf.sched import SimConfig, simulate

    cfg = SimConfig(max_num_seqs=seqs, chunk=chunk, kv_blocks=blocks, block_tokens=block, api_ms=api, **kw)
    return simulate(model or _Linear(), cfg, [(prompt, out)] * n, concurrency=c)


def test_a_single_request_prefills_once_and_decodes_every_step():
    """T-SIM-04: one request, one chunk: its first token after the prefill step (plus the API time), then one token
    per decode step at its growing context."""
    m = _Linear()
    res = _sim(m, c=1, n=1, prompt=1024, out=256, api=5.0)
    assert res.ttft_ms == pytest.approx(m.chunk_ms(1024, 1024) + 5.0)
    expected_tpot = sum(m.decode_ms(1, 1024 + i) for i in range(1, 256)) / 255
    assert res.tpot_ms == pytest.approx(expected_tpot, rel=1e-6)
    assert res.throughput_tok_s == pytest.approx(256 / ((res.ttft_ms - 5.0 + 255 * expected_tpot) / 1000), rel=1e-3)
    assert res.preemptions == 0 and res.max_running == 1


def test_a_wave_s_early_prompts_decode_through_its_later_chunks():
    """T-SIM-03: 16 identical 1,024-token prompts fill two 8,192-token chunks, and decode tokens count against the
    budget, as the TCS smoke profile shows (8B, b=16: steps of 168 / 185 / 7.6 ms with 8 / 15 / 16 new tokens). The
    first 8 emit after chunk 1 and decode through chunk 2; chunk 2 holds their 8 decode tokens and 8,184 prompt tokens,
    so the 16th prompt finishes one step later. The early group's TPOT carries the mixed step (R0: the 8B at c64
    measured 12.7 ms against a 10.2 ms decode step)."""
    m = _Linear()
    res = _sim(m, c=16, n=16, prompt=1024, out=256)
    chunk = m.chunk_ms(8192, 1024)
    assert res.ttft_samples[:8] == pytest.approx([chunk] * 8)
    assert res.ttft_samples[8:15] == pytest.approx([2 * chunk] * 7)
    assert res.ttft_samples[15] > res.ttft_samples[14]
    early, late = res.tpot_samples[0], res.tpot_samples[15]
    assert early - late == pytest.approx((chunk - m.decode_ms(8, 1025)) / 255, rel=0.1)


def test_the_closed_loop_submits_the_next_request_when_one_finishes():
    """T-SIM-04: `vllm bench serve --max-concurrency c`: at most c requests in flight, the next one submitted when one
    completes; throughput is all output tokens over the run's duration."""
    res = _sim(c=4, n=12, prompt=256, out=64)
    assert res.max_running == 4 and len(res.ttft_samples) == 12
    assert res.throughput_tok_s == pytest.approx(12 * 64 / (res.duration_ms / 1000))


def test_max_num_seqs_queues_the_rest():
    """T-SIM-01: with `max-num-seqs` below the client concurrency the overflow waits in the queue: running never
    exceeds the cap and the median TTFT carries the wait (the study's capped plan: 37.7 s median at c16)."""
    free = _sim(c=16, n=16, prompt=256, out=256, seqs=16)
    capped = _sim(c=16, n=16, prompt=256, out=256, seqs=4)
    assert capped.max_running == 4
    assert capped.ttft_ms > 10 * free.ttft_ms


def test_a_full_kv_pool_preempts_the_newest_request():
    """T-SIM-02: KV pages bound admission; a growing sequence that finds no free page preempts the most recently
    admitted one, which recomputes its prompt and generated tokens later (the 27B reference at c128: 125 preemptions,
    TPOT 35.5 ms against a 56 ms 128-wide step)."""
    roomy = _sim(c=16, n=16, prompt=1024, out=512, blocks=100000)
    tight = _sim(c=16, n=16, prompt=1024, out=512, blocks=16 * 70)  # 70 pages per request would fit 16 at admission
    assert roomy.preemptions == 0 and tight.preemptions > 0
    assert tight.mean_running < roomy.mean_running
    assert len(tight.ttft_samples) == 16 and all(t > 0 for t in tight.tpot_samples)


def test_a_hybrid_sequence_also_holds_its_state_pages():
    """T-SIM-02: a recurrent state takes pages per sequence, independent of its length (vLLM's hybrid allocator
    unifies the page size), so fewer sequences fit than the attention pages alone allow."""
    plain = _sim(c=32, n=32, prompt=256, out=64, blocks=40 * 32, block=16)
    hybrid = _sim(c=32, n=32, prompt=256, out=64, blocks=40 * 32, block=16, state_blocks=30)
    assert plain.max_running == 32 and hybrid.max_running < 32


def test_the_simulation_is_deterministic():
    a = _sim(c=8, n=24, prompt=700, out=100)
    b = _sim(c=8, n=24, prompt=700, out=100)
    assert (a.ttft_ms, a.tpot_ms, a.throughput_tok_s) == (b.ttft_ms, b.tpot_ms, b.throughput_tok_s)


# ---------------------------------------------------------------- the profile path in the engine (opt-in)


def _profile():
    import json

    from .conftest import FIXTURES

    return StepProfile.from_dict(json.loads((FIXTURES / "profiles" / "qwen3-8b.vllm-0.30.0.json").read_text()))


def _engine(profiles=None):
    from llm_memory_calculator.engine import EngineRecord, engine_for
    from llm_memory_calculator.grounding import default_store

    from .conftest import load_record

    return engine_for(EngineRecord.from_dict(load_record("vllm-0.30-upstream")), default_store(), profiles)


def _q8():
    from llm_memory_calculator.model import ModelFacts

    from .conftest import load_model

    return ModelFacts.from_config(load_model("qwen3-8b"), family="Qwen3-8B")


def _chat16():
    from llm_memory_calculator.engine import Workload

    return Workload(1024, 256, 16, workload_class="synthetic", sampling="greedy", requests=48)


VGPU_SKU = "h100xm-80c-vgpu"
FP8_ONLINE = {"weights.format": "fp8", "weights.online": True}


def test_a_profiled_configuration_is_predicted_from_its_measured_steps():
    """T-STEP-05: with a profile for (SKU, engine record, family, kernel class), the engine predicts time from the
    measured steps through the scheduler simulation; memory and limits still come from the engine layer. The 8B's
    stock chat c16 on TCS measured TPOT 7.83 ms and 1,794 tok/s."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.store import ProfileStore

    vgpu = catalogue.get(VGPU_SKU)
    pred = _engine(ProfileStore([_profile()])).predict({}, _chat16(), _q8(), vgpu)
    assert any(n.startswith("predictor: step profile") for n in pred.notes)
    assert pred.tpot_ms.value == pytest.approx(7.83, rel=0.08)
    assert pred.throughput_tok_s.value == pytest.approx(1794, rel=0.08)
    assert pred.limits.kv_pool_bytes == _engine().limits({}, _q8(), vgpu).kv_pool_bytes


def test_another_kernel_class_or_a_shared_slice_keeps_the_fitted_predictor():
    """T-STEP-05: a profile is exact: online FP8 runs other kernels, and a HAMi slice shares its compute with
    co-tenants, so neither is predicted from a bf16 dedicated profile (NFR-5: the fitted predictor serves)."""
    from llm_memory_calculator.device import HardwareMode, catalogue
    from llm_memory_calculator.perf.store import ProfileStore

    vgpu = catalogue.get(VGPU_SKU)
    with_profiles = _engine(ProfileStore([_profile()]))
    fp8 = with_profiles.predict(FP8_ONLINE, _chat16(), _q8(), vgpu)
    assert not any(n.startswith("predictor: step profile") for n in fp8.notes)
    assert fp8.tpot_ms.value == _engine().predict(FP8_ONLINE, _chat16(), _q8(), vgpu).tpot_ms.value
    shared = vgpu.with_mode(HardwareMode(kind="shared", slice_bytes=40000 * 1024**2))
    assert not any(n.startswith("predictor: step profile") for n in with_profiles.predict({}, _chat16(), _q8(), shared).notes)


def test_the_band_combines_interpolation_and_measured_residual():
    """T-RES-05: the band is hypot(the profile's leave-one-out interpolation error, its residual against deployments);
    an unvalidated profile carries a wide default residual, so a plan does not lean on it."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.store import ProfileStore

    prof = _profile()
    prof.residual = {"tpot": 0.05, "throughput": 0.04, "ttft": 0.2}
    pred = _engine(ProfileStore([prof])).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    assert pred.tpot_ms.width == pytest.approx(math.hypot(prof.interpolation_error(), 0.05), rel=1e-6)
    unvalidated = _profile()
    wide = _engine(ProfileStore([unvalidated])).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    assert wide.ttft_ms.width > pred.ttft_ms.width


def test_the_r2_switch_reads_the_package_profiles(monkeypatch):
    """T-STEP-05: FRD027_PREDICTOR_VERSION=r2 turns the profile path on with the package's profiles; without it the
    engine keeps the fitted predictor (the switch stays off until the §20.7 acceptance)."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf import store as pstore

    monkeypatch.setattr(pstore, "_DEFAULT", pstore.ProfileStore([_profile()]))
    vgpu = catalogue.get(VGPU_SKU)
    assert not any(n.startswith("predictor: step profile") for n in _engine().predict({}, _chat16(), _q8(), vgpu).notes)
    monkeypatch.setenv("FRD027_PREDICTOR_VERSION", "r2")
    assert any(n.startswith("predictor: step profile") for n in _engine().predict({}, _chat16(), _q8(), vgpu).notes)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, {"weights": "bf16", "kv": "auto", "state": "auto", "spec": "none", "lora": "0"}),
        ({"weights.format": "fp8", "weights.online": True}, {"weights": "fp8-online"}),
        ({"kv.dtype": "fp8"}, {"kv": "fp8"}),
        ({"lora.max_adapters": 2}, {"lora": "1"}),
    ],
)
def test_a_profile_is_keyed_by_the_settings_that_change_kernels(settings, expected):
    """T-PROF-01: capacity settings (sequences, context, utilization) do not key a profile; the served format, KV and
    state dtype, speculation and LoRA do."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.store import kernel_class

    r = _engine().resolve({**settings, "batch.max_running": 64}, _q8(), catalogue.get(VGPU_SKU))
    got = kernel_class(r, _q8())
    assert {k: got[k] for k in expected} == expected


def test_a_checkpoint_in_another_format_does_not_match_the_profile():
    """T-PROF-01: the key is the served format itself, not "native": an FP8 checkpoint of a family runs FP8 kernels,
    so a bf16 profile of that family must not predict it (the R3 replay found the 8B's FP8 checkpoint 24% too fast
    on throughput through exactly this match). A profile that does not state its format matches nothing."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.model import ModelFacts
    from llm_memory_calculator.perf.store import ProfileStore

    from .conftest import load_model

    fp8 = ModelFacts.from_config(load_model("qwen3-8b-fp8"), family="Qwen3-8B")
    vgpu = catalogue.get(VGPU_SKU)
    with_profiles = _engine(ProfileStore([_profile()]))
    assert any(n.startswith("predictor: step profile") for n in with_profiles.predict({}, _chat16(), _q8(), vgpu).notes)
    assert not any(n.startswith("predictor: step profile") for n in with_profiles.predict({}, _chat16(), fp8, vgpu).notes)
    unstated = _profile()
    unstated.settings = {}
    assert ProfileStore([unstated]).find(VGPU_SKU, "vllm@0.30.0", "Qwen3-8B", {"weights": "bf16"}) is None


# ---------------------------------------------------------------- per-request lengths of a load


def test_a_synthetic_load_has_one_length():
    """T-SIM-05: random-token benchmark prompts are fixed-length, and so are their outputs (ignore_eos)."""
    from llm_memory_calculator.engine import Workload
    from llm_memory_calculator.perf.lengths import lengths_for

    got = lengths_for(Workload(1024, 256, 16, workload_class="synthetic"), 48)
    assert got == [(1024, 256)] * 48


def test_real_text_lengths_vary_around_the_workload_s_means():
    """T-SIM-05: real text desynchronizes a load: prompts follow the Spec-Bench shape (median 224 characters, mean
    1,202, p90 3,303) and outputs stop at end-of-sequence below the cap. Simulated at their means instead, the 27B's
    real-text runs came out 12-20% too fast with TTFTs several times too long (waves that never form in serving)."""
    import statistics

    from llm_memory_calculator.engine import Workload
    from llm_memory_calculator.perf.lengths import lengths_for

    w = Workload(271.2, 358.4, 8, workload_class="real", output_tokens_max=512)
    got = lengths_for(w, 48)
    prompts, outputs = [p for p, _ in got], [o for _, o in got]
    assert statistics.mean(prompts) == pytest.approx(271.2, rel=0.05)
    assert statistics.mean(outputs) == pytest.approx(358.4, rel=0.03)
    assert max(outputs) <= 512 and min(outputs) >= 1
    assert sorted(prompts)[-5] > 4 * sorted(prompts)[5]  # heavy-tailed prompts
    assert lengths_for(w, 48) == got  # deterministic


def test_a_mean_at_the_cap_runs_every_output_to_the_cap():
    from llm_memory_calculator.engine import Workload
    from llm_memory_calculator.perf.lengths import lengths_for

    got = lengths_for(Workload(300, 512, 8, workload_class="real", output_tokens_max=512), 16)
    assert all(o == 512 for _, o in got)


@pytest.mark.parametrize("c, n, prompt, out, blocks", [(64, 160, 1024, 256, 100000), (128, 256, 256, 1024, 100000),
                                                       (16, 48, 1024, 256, 16 * 70), (8, 24, 8192, 256, 100000)])
def test_fast_forwarding_decode_windows_is_exact(c, n, prompt, out, blocks):
    """T-SIM-06 (NFR-9): running a decode-only window to its first completion in one go gives the step-by-step
    result, KV-limited runs included."""
    from llm_memory_calculator.perf.sched import SimConfig, simulate

    def run(ff):
        cfg = SimConfig(max_num_seqs=256, chunk=8192, kv_blocks=blocks, block_tokens=16, arrive_ms=4.0, return_ms=4.0,
                        async_scheduling=True, fast_forward=ff)
        return simulate(_Linear(), cfg, [(prompt, out)] * n, concurrency=c)

    slow, fast = run(False), run(True)
    assert fast.ttft_samples == pytest.approx(slow.ttft_samples, rel=1e-9)
    assert fast.tpot_samples == pytest.approx(slow.tpot_samples, rel=1e-9)
    assert (fast.duration_ms, fast.preemptions, fast.steps) == (pytest.approx(slow.duration_ms, rel=1e-9), slow.preemptions, slow.steps)


# ---------------------------------------------------------------- R2: risk-aware ranking (FR-SEL-1)


def test_the_probability_of_a_target_follows_the_band():
    """T-SEL-01: a band is a p90 interval of a log-normal error, so a target at the band's edge holds with 90%
    probability, one at the prediction holds half the time, and one far past the band almost surely."""
    from llm_memory_calculator.engine import Estimate
    from llm_memory_calculator.perf.rank import p_at_least, p_at_most

    est = Estimate.around(100.0, 0.2)
    assert p_at_most(est, 120.0) == pytest.approx(0.9, abs=0.005)
    assert p_at_most(est, 100.0) == pytest.approx(0.5, abs=1e-9)
    assert p_at_most(est, 300.0) > 0.999 and p_at_most(est, 50.0) < 0.001
    assert p_at_least(est, 80.0) == pytest.approx(0.9, abs=0.005)  # the band's lower edge is value x (1 - h)
    exact = Estimate.around(100.0, 0.0)
    assert p_at_most(exact, 100.0) == 1.0 and p_at_most(exact, 99.0) == 0.0


def test_the_slo_probability_is_the_least_likely_target():
    """T-SEL-01: TTFT, end-to-end time and per-user throughput come from one step model, so their errors move together;
    the plan's probability is the least likely target's, not the product. An unset target does not count, and an
    ungrounded estimate gives no probability (the search keeps its point rule there, NFR-5)."""
    from llm_memory_calculator.engine import Estimate
    from llm_memory_calculator.perf.rank import slo_probability

    pred = {
        "ttft_ms": Estimate.around(400.0, 0.2).to_dict(),
        "e2e_s": Estimate.around(4.0, 0.1).to_dict(),
        "throughput_tok_s": Estimate.around(1600.0, 0.1).to_dict(),
    }
    p = slo_probability(pred, concurrency=16, ttft_ms=480.0, e2e_s=None, throughput_per_user=None)
    assert p == pytest.approx(0.9, abs=0.005)
    both = slo_probability(pred, concurrency=16, ttft_ms=480.0, throughput_per_user=100.0)
    assert both == pytest.approx(min(0.9, 0.5), abs=0.005)  # 1,600 / 16 = 100 tok/s per user: a coin flip
    assert slo_probability(pred, concurrency=16) == 1.0  # nothing to miss
    wide = dict(pred, ttft_ms=Estimate.around(400.0, 2.0, ceiling=0.2).to_dict())
    assert slo_probability(wide, concurrency=16, ttft_ms=480.0) is None


# ---------------------------------------------------------------------- L5: output grounding (R3)


def _res(family="F", quantity="tpot", bucket="c5-32/synthetic", ln_r=0.1, config="cfg", engine="vllm@0.30.0", n=1):
    from llm_memory_calculator.perf.residual import Residual

    return [
        Residual("h100xm-80c-vgpu", engine, family, quantity, bucket, 10.0, 10.0 * math.exp(ln_r), f"{config}{i}")
        for i in range(n)
    ]


def _spread(values, **kw):
    out = []
    for i, v in enumerate(values):
        out += _res(ln_r=v, config=f"{kw.get('config', 'c')}{i}-", **{k: x for k, x in kw.items() if k != "config"})
    return out


def test_corrections_are_per_family_so_adding_a_family_changes_no_other():
    """T-RES-01: a family's correction depends on its own points and fixed priors only; adding another family's runs
    leaves it, and its predictions, unchanged (the §19.5 coupling through a shared term cannot recur)."""
    from llm_memory_calculator.perf.residual import fit

    a = _spread([0.10, 0.12, 0.08, 0.11, 0.09], family="A")
    b = _spread([-0.20, -0.25, -0.22], family="B", config="b")
    alone, both = fit(a), fit(a + b)
    for key, corr in alone.entries.items():
        assert both.entries[key] == corr
    assert both.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "B", "tpot", "c5-32/synthetic").factor < 1.0


def test_a_small_slice_is_shrunk_toward_its_parent_and_a_large_one_keeps_its_own():
    """T-RES-02: a bucket with two points sits between its family's correction and its own mean; a bucket with fifty
    keeps its own; and the noisier the slice, the stronger the pull."""
    from llm_memory_calculator.perf.residual import fit

    many = _spread([0.10 + 0.08 * ((i % 5) - 2) / 2 for i in range(50)], family="A", bucket="c5-32/synthetic")
    few = _spread([0.25, 0.29], family="A", bucket="c33+/synthetic", config="f")
    corr = fit(many + few)
    fam = math.log(corr.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c1-4/real").factor)
    big = math.log(corr.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic").factor)
    small = math.log(corr.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c33+/synthetic").factor)
    assert big == pytest.approx(0.10, abs=0.01)  # its own mean
    assert fam < small < 0.27  # pulled toward the family from its own 0.27
    noisy = fit(_spread([0.10 + 0.30 * ((i % 5) - 2) / 2 for i in range(50)], family="A") + few)
    small_noisy = math.log(noisy.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c33+/synthetic").factor)
    assert abs(small_noisy - 0.27) > abs(small - 0.27)


def test_a_correction_past_the_cap_is_a_model_gap():
    """T-RES-03 (FR-RES-3): a slice needing |ln r| > ln 1.3 is a gap: the prediction keeps the simulation's value,
    is reported ungrounded, and names the gap; a slice inside the cap is corrected."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.residual import fit
    from llm_memory_calculator.perf.store import ProfileStore

    family = "Qwen3-8B"
    gap = fit(_spread([0.40, 0.42, 0.38, 0.41], family=family, bucket="c5-32/synthetic"))
    entry = gap.lookup(VGPU_SKU, "vllm@0.30.0", family, "tpot", "c5-32/synthetic")
    assert entry.gap and entry.needed > 1.3
    plain = _engine(ProfileStore([_profile()])).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    pred = _engine_with(gap).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    assert pred.tpot_ms.ungrounded and pred.tpot_ms.value == pytest.approx(plain.tpot_ms.value)
    assert any(n.startswith("model gap: tpot") for n in pred.notes)
    ok = fit(_spread([0.10, 0.12, 0.08, 0.11], family=family, bucket="c5-32/synthetic"))
    fixed = _engine_with(ok).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    assert not fixed.tpot_ms.ungrounded
    assert fixed.tpot_ms.value == pytest.approx(plain.tpot_ms.value * math.exp(0.1), rel=0.02)
    assert fixed.levels["l5.tpot"] == "bucket"
    # the end-to-end time moves with the corrected decode
    assert fixed.e2e_s.value > plain.e2e_s.value


def _engine_with(corrections):
    from llm_memory_calculator.engine import EngineRecord, engine_for
    from llm_memory_calculator.grounding import default_store
    from llm_memory_calculator.perf.store import ProfileStore

    from .conftest import load_record

    record = EngineRecord.from_dict(load_record("vllm-0.30-upstream"))
    return engine_for(record, default_store(), ProfileStore([_profile()]), corrections)


def test_corrections_are_per_engine_build_with_a_wider_band_for_a_new_build_of_the_minor():
    """T-RES-04: a new build starts from its minor's corrections with a wider band, and never inherits another
    minor's."""
    from llm_memory_calculator.perf.residual import MINOR_WIDEN, fit, minor_of

    assert minor_of("vllm@0.30.0") == "vllm@0.30" and minor_of("bud-vllm@0.9.0-481e481be") == "bud-vllm@0.9"
    corr = fit(_spread([0.10, 0.12, 0.08, 0.11], family="A"))
    own = corr.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic")
    patch = corr.lookup("h100xm-80c-vgpu", "vllm@0.30.1", "A", "tpot", "c5-32/synthetic")
    assert patch.factor == own.factor and patch.band == pytest.approx(own.band * MINOR_WIDEN)
    assert patch.level == "minor:vllm@0.30.0"
    assert corr.lookup("h100xm-80c-vgpu", "vllm@0.31.0", "A", "tpot", "c5-32/synthetic") is None


def test_an_unseen_family_on_a_known_build_gets_the_build_s_band_and_no_factor():
    """FR-RES-1/2: with no points of its own a family is not corrected; its band is what L1–L4 alone earned across the
    build's families (NFR-6's measurement), so a new family is not over-trusted."""
    from llm_memory_calculator.perf.residual import fit

    corr = fit(_spread([0.10, 0.12, 0.08], family="A") + _spread([-0.2, -0.18], family="B", config="b"))
    new = corr.lookup("h100xm-80c-vgpu", "vllm@0.30.0", "Z", "tpot", "c5-32/synthetic")
    assert new.level == "build" and new.factor == 1.0 and new.band >= 0.18


def test_the_band_covers_new_points_after_correction():
    """FR-RES-2: the band is the split-conformal 90% quantile of residuals each corrected without its configuration,
    so on a stream of new points from the slice ~90% or more fall inside it."""
    import random

    from llm_memory_calculator.perf.residual import fit

    rng = random.Random(7)
    train = _spread([0.15 + rng.gauss(0, 0.05) for _ in range(40)], family="A")
    corr = fit(train).lookup("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic")
    new = [0.15 + rng.gauss(0, 0.05) for _ in range(2000)]
    inside = sum(abs(v - math.log(corr.factor)) <= corr.band for v in new) / len(new)
    assert 0.88 <= inside <= 0.97


def test_corrections_round_trip_and_load_from_a_directory(tmp_path):
    import json

    from llm_memory_calculator.perf.residual import Corrections, fit

    corr = fit(_spread([0.10, 0.12, 0.08, 0.11], family="A") + _spread([0.4, 0.41, 0.39], family="G", config="g"))
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "corrections.json").write_text(json.dumps(corr.to_dict()))
    back = Corrections.load(tmp_path)
    assert back.entries.keys() == corr.entries.keys() and back.builds.keys() == corr.builds.keys()
    for k, c in corr.entries.items():
        assert back.entries[k].gap == c.gap and back.entries[k].factor == pytest.approx(c.factor, rel=1e-4)


def test_the_corpus_replay_grounds_the_8b_out_of_sample():
    """R3 gate (NFR-6/7) on the fixture corpus: every profiled Qwen3-8B point replays through the product path. Before
    correction TPOT and throughput are within NFR-6 (15%); after a correction fitted without each point's
    configuration they are within NFR-7 (10%), and TTFT within 15% (25% queueing)."""
    from llm_memory_calculator.grounding.seed import build_inputs
    from llm_memory_calculator.perf.replay import grade, replay
    from llm_memory_calculator.perf.store import ProfileStore

    from .conftest import FIXTURES

    inputs = build_inputs(FIXTURES / "corpus", FIXTURES / "models", FIXTURES / "records", version="t")
    residuals = replay(inputs, ProfileStore([_profile()]), families=["Qwen3-8B"])
    assert len({r.config for r in residuals}) >= 5
    rows = {(r["quantity"], r["class"], r["queueing"]): r for r in grade(residuals)}
    for (q, cls, queueing), row in rows.items():
        if q in ("tpot", "throughput"):
            assert row["before"]["mape"] <= 0.15, row
            assert row["after"]["mape"] <= 0.10, row
        else:
            assert row["after"]["mape"] <= (0.25 if queueing else 0.15), row


# ---------------------------------------------------------------------- FR-SEL-2: the simulation-backed chunk search


def _validated_engine():
    from llm_memory_calculator.perf.store import ProfileStore

    prof = _profile()
    prof.residual = {"tpot": 0.03, "throughput": 0.03, "ttft": 0.10}
    return _engine(ProfileStore([prof]))


def _long64():
    from llm_memory_calculator.engine import Workload

    return Workload(4096, 128, 64, workload_class="real")


def _choose(engine, intent=None, workload=None):
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.runtime.modules.capacity import choose_chunk

    w = workload or _long64()
    return choose_chunk(engine, {"batch.max_running": 256}, w, _q8(), catalogue.get(VGPU_SKU), 8192, intent=intent)


def test_without_targets_the_search_takes_a_chunk_only_when_its_throughput_separates():
    """T-SEL-02 / FR-GND-2: on the profile path every chunk size is simulated, for any input length. The 8B at c64
    with 4,096-token prompts serves ~13% more with 2,048-token chunks (each step's prefill interferes less with decode),
    and the bands separate, so 2,048 wins when no target constrains TTFT."""
    chunk, why = _choose(_validated_engine())
    assert chunk == 2048 and "separates above 8192" in why


def test_a_ttft_target_rules_out_the_chunk_that_would_miss_it():
    """T-SEL-01/02: 2,048-token chunks quadruple TTFT (1.2 s against 0.33 s): with a 500 ms TTFT target 2,048 cannot
    qualify; 4,096 qualifies but its throughput band overlaps the default's, so the default stays."""
    from llm_memory_calculator.runtime.context import Intent

    chunk, why = _choose(_validated_engine(), Intent(target_ttft_ms=500.0))
    assert chunk == 8192 and "overlap" in why


def test_when_the_default_misses_the_targets_a_qualifying_chunk_wins():
    """T-SEL-01/02: a per-user throughput target of 15 tok/s is missed at the default (~14) and met at 2,048 (~15.8)
    with P >= 0.9, so 2,048 is chosen even without its band separating."""
    from llm_memory_calculator.runtime.context import Intent

    chunk, why = _choose(_validated_engine(), Intent(target_throughput_per_user=15.0))
    assert chunk == 2048 and "meets the targets" in why


def test_the_fitted_predictor_keeps_today_s_chunk_rule():
    """NFR-5: without a profile only long inputs are compared, between the default and 16,384, as before."""
    from llm_memory_calculator.engine import Workload

    chunk, why = _choose(_engine(), workload=Workload(1024, 256, 64, workload_class="real"))
    assert chunk == 8192 and "record default" in why


# ---------------------------------------------------------------------- FR-PROF-2: profiles from a cluster's profiling job


@pytest.mark.parametrize(
    "fmt, extra, expected",
    [
        ("bf16", {}, {"weights": "bf16"}),
        ("bf16", {"quantization": "fp8"}, {"weights": "fp8-online"}),
        ("fp8", {"quantization": "fp8"}, {"weights": "fp8"}),  # an FP8 checkpoint is served as itself
        ("bf16", {"kv_cache_dtype": "fp8_e4m3"}, {"weights": "bf16", "kv": "fp8"}),
        ("bf16", {"enable_lora": True}, {"lora": "1"}),
        ("bf16", {"speculative_config": {"method": "eagle3", "num_speculative_tokens": 3}}, {"spec": "eagle3:3"}),
    ],
)
def test_a_job_s_engine_arguments_give_its_profile_s_kernel_class(fmt, extra, expected):
    """FR-PROF-2: the profiling job records the engine kwargs it ran with; the profile is keyed by the kernel class
    they select, so a plan finds it exactly when it would run the same kernels (T-PROF-01/03)."""
    from llm_memory_calculator.perf.store import kernel_class_of_args

    got = kernel_class_of_args(fmt, extra)
    assert {k: got[k] for k in expected} == expected


def test_a_cluster_profile_takes_precedence_over_the_package_s(monkeypatch):
    """FR-PROF-2 / open question 1: a profile measured on the cluster replaces the package's for the same key, and
    the plan names where its profile came from."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf import store as pstore

    monkeypatch.setattr(pstore, "_DEFAULT", pstore.ProfileStore([_profile()]))
    monkeypatch.setenv("FRD027_PREDICTOR_VERSION", "r2")
    vgpu = catalogue.get(VGPU_SKU)
    before = _engine().predict({}, _chat16(), _q8(), vgpu)
    mine = _profile()
    mine.decode = {k: v * 1.2 for k, v in mine.decode.items()}
    mine.provenance = {**mine.provenance, "source": "cluster", "image_digest": "sha256:abc"}
    pstore.register_profile(mine)
    after = _engine().predict({}, _chat16(), _q8(), vgpu)
    assert after.tpot_ms.value == pytest.approx(before.tpot_ms.value * 1.2, rel=0.03)
    assert any("cluster" in n and "sha256:abc" in n for n in after.notes)


def test_a_cold_first_use_stall_is_recorded_and_reported():
    """R1 finding 5: a build that compiles kernels per shape on first use stalls cold requests. The profiler times
    each shape's warm-up run; the profile keeps the stall, and a plan reports it as a cold-start note."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.store import ProfileStore

    steps = _steps_file(GRID)
    steps["points"][0]["warm_max_ms"] = 5400.0  # the first decode c32 step compiled the fused-MoE kernel
    prof = StepProfile.from_steps(steps, sku=VGPU_SKU, engine="vllm@0.30.0", model="Qwen3-8B", settings={"weights": "bf16"})
    assert prof.cold_stall_ms() >= 5000.0
    back = StepProfile.from_dict(prof.to_dict())
    assert back.cold_stall_ms() == prof.cold_stall_ms()
    full = _profile()
    full.cold_stall = {(32, 1024): 5400.0}
    pred = _engine(ProfileStore([full])).predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    assert any(n.startswith("cold start:") for n in pred.notes)


# ---------------------------------------------------------------------- FR-RES-4: the telemetry refit


def test_telemetry_refits_its_slices_and_ages_the_corpus_out():
    """FR-RES-4 / open question 4: telemetry points join the corpus residuals. A slice with AGE_OUT or more of them
    is fitted from telemetry alone (production supersedes the study), and every other slice keeps its corpus fit."""
    from llm_memory_calculator.perf.residual import AGE_OUT, Residual, fit, refit

    corpus = _spread([0.10, 0.12, 0.08, 0.11], family="A") + _spread([0.05, 0.06, 0.04], family="B", config="b")
    few = [Residual("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic", 10.0, 10.0 * math.exp(0.20),
                    f"telemetry:{i}") for i in range(3)]
    many = [Residual("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic", 10.0, 10.0 * math.exp(0.20),
                     f"telemetry:{i}") for i in range(AGE_OUT)]
    key = ("h100xm-80c-vgpu", "vllm@0.30.0", "A", "tpot", "c5-32/synthetic")
    base = fit(corpus).entries[key].factor
    mixed = refit(corpus, few).entries[key].factor
    aged = refit(corpus, many).entries[key].factor
    assert base < mixed < aged
    assert aged == pytest.approx(math.exp(0.20), rel=0.02)
    other = ("h100xm-80c-vgpu", "vllm@0.30.0", "B", "tpot", "c5-32/synthetic")
    assert refit(corpus, many).entries[other] == fit(corpus).entries[other]


def test_the_package_ships_the_corpus_residuals_its_corrections_were_fitted_from():
    """The refit needs the corpus points, not only their fit: the package carries both, and they agree."""
    from llm_memory_calculator.perf.residual import Corrections, corpus_residuals, fit

    residuals = corpus_residuals()
    assert len(residuals) > 1000
    shipped = Corrections.load()
    again = fit(residuals)
    for key, corr in shipped.entries.items():
        assert again.entries[key].factor == pytest.approx(corr.factor, rel=1e-3)


def test_a_telemetry_observation_becomes_residuals_against_the_uncorrected_simulation():
    """A deployment's measured TTFT and throughput at its observed load are replayed through the profile path with
    no correction, so the residual is what L1–L4 got wrong there."""
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.perf.store import ProfileStore
    from llm_memory_calculator.perf.telemetry import Observation, residuals_of

    engine = _engine(ProfileStore([_profile()]))
    plain = engine.predict({}, _chat16(), _q8(), catalogue.get(VGPU_SKU))
    obs = Observation(settings={}, concurrency=16, input_tokens=1024, output_tokens=256, workload_class="synthetic",
                      ttft_ms=plain.ttft_ms.value * 1.25, throughput_tok_s=plain.throughput_tok_s.value * 0.9,
                      requests=48, source="telemetry:ep1:w1")
    out = residuals_of(engine, obs, _q8(), catalogue.get(VGPU_SKU))
    by_q = {r.quantity: r for r in out}
    assert set(by_q) == {"ttft", "throughput"}  # no TPOT measured
    assert by_q["ttft"].ln_r == pytest.approx(math.log(1.25), abs=1e-6)
    assert by_q["throughput"].ln_r == pytest.approx(math.log(0.9), abs=1e-6)
    assert by_q["ttft"].config == "telemetry:ep1:w1" and by_q["ttft"].family == "Qwen3-8B"
    fp8 = Observation(**{**obs.__dict__, "settings": FP8_ONLINE})
    assert residuals_of(engine, fp8, _q8(), catalogue.get(VGPU_SKU)) == []  # no profile: L5 does not apply
