"""FRD-027 §9, T-GND-01…07, NFR-1/3/4: the grounding store, its ladders and the accuracy gate."""

import math

import pytest

from llm_memory_calculator.device import catalogue
from llm_memory_calculator.engine import EngineRecord, Estimate, Workload, draft_key, engine_for
from llm_memory_calculator.grounding import GroundingStore, default_store
from llm_memory_calculator.grounding.fit import fit_store
from llm_memory_calculator.grounding.gate import replay
from llm_memory_calculator.grounding.seed import build_inputs
from llm_memory_calculator.model import ModelFacts
from llm_memory_calculator.predict import EvaluationBudgetExceeded, Predictor, versions

from .conftest import FIXTURES, load_model, load_record

GIB = 1024**3
VGPU = catalogue.get("h100xm-80c-vgpu")


@pytest.fixture(scope="module")
def inputs():
    return build_inputs(
        FIXTURES / "corpus", FIXTURES / "models", FIXTURES / "records", version="test"
    )


@pytest.fixture(scope="module")
def store():
    return default_store()


@pytest.fixture(scope="module")
def report(inputs, store):
    return replay(inputs, store)


def rec(name="vllm-0.30-upstream"):
    return EngineRecord.from_dict(load_record(name))


def q8(family="Qwen3-8B"):
    return ModelFacts.from_config(load_model("qwen3-8b"), family=family)


# ---------------------------------------------------------------- T-GND-01 ladders


def test_full_key_is_measured(store):
    found = store.find("decode.kappa", {"bucket": "1"}, VGPU, rec(), q8(), prior=1.0)
    assert found.levels == {"hardware": "sku", "engine": "record", "model": "family"}


def test_unmeasured_family_falls_to_class(store):
    llama = ModelFacts.from_config(load_model("llama-3.1-8b"), family="Llama-3.1-8B")
    assert llama.arch_class == "dense-GQA"
    measured = store.find("decode.kappa", {"bucket": "1"}, VGPU, rec(), q8(), prior=1.0)
    found = store.find("decode.kappa", {"bucket": "1"}, VGPU, rec(), llama, prior=1.0)
    assert found.levels["model"] == "class"
    assert found.residual > measured.residual


def test_new_build_falls_to_engine_minor(store):
    found = store.find(
        "decode.kappa", {"bucket": "1"}, VGPU, rec("vllm-0.31-upstream"), q8(), prior=1.0
    )
    assert found.levels["engine"] == "family"  # 0.31 is a new minor: the engine-family level
    # a fork build on the 0.30 base that no run has measured
    on_030 = EngineRecord.from_dict(
        {**load_record("bud-vllm-0.10.0"), "id": "bud-vllm@0.99.0", "version": "0.99.0", "upstream_vllm": "0.30.0"}
    )
    fork = store.find("decode.kappa", {"bucket": "1"}, VGPU, on_030, q8(), prior=1.0)
    assert fork.levels["engine"] == "minor"  # upstream 0.30 base: inherits the 0.30 minor entries


def test_a_harness_image_key_names_its_engine_record(tmp_path):
    """T-GND-16: round 7 ran BudConnect dev's new build (image key ``bud010``); its runs ground bud-vllm@0.10.0."""
    import json

    from llm_memory_calculator.grounding.corpus import RECORDS, read_config
    from llm_memory_calculator.grounding.seed import RECORD_FILES

    assert RECORDS["bud010"] == "bud-vllm@0.10.0"
    assert RECORD_FILES["bud-vllm@0.10.0"] == "bud-vllm-0.10.0"
    run = tmp_path / "c-8b-010"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"name": run.name, "image_key": "bud010", "model": "8b-bf16"}))
    assert read_config(run).record == "bud-vllm@0.10.0"


def test_the_fixture_records_agree_with_the_vendored_ones():
    """T-GND-17: the fitter reads the fixture records; a stale copy (0.10.0 once claimed the 0.30 base) would file a
    build's runs under the wrong upstream minor."""
    import json

    from llm_memory_calculator.runtime.records import vendored

    for path in sorted((FIXTURES / "records").glob("*.json")):
        fixture = json.loads(path.read_text())
        shipped = vendored().get(fixture["id"])
        if shipped is None:
            continue
        for key in ("version", "upstream_vllm", "images", "features"):
            assert fixture.get(key) == shipped.get(key), (path.name, key)


def test_bud_vllm_0_10_0_is_grounded_on_its_own_record(store):
    """T-GND-18: round 7 measured the new build's speculative runs on TCS (8B EAGLE3, 27B MTP), so its acceptance
    comes from its own record, and queue F's stock 8B run (engine defaults, 13 points) grounds its base decode."""
    new = rec("bud-vllm-0.10.0")
    q27 = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    mtp = {"method": "mtp", "k": 3, "sampling": "greedy", "workload": "synthetic"}
    assert store.find("spec.acceptance", mtp, VGPU, new, q27, prior=1.0).levels["engine"] == "record"
    eagle = {
        "method": "eagle3",
        "k": 3,
        "sampling": "greedy",
        "workload": "synthetic",
        "draft": draft_key("RedHatAI/Qwen3-8B-speculator.eagle3"),
    }
    assert store.find("spec.acceptance", eagle, VGPU, new, q8(), prior=1.0).levels["engine"] == "record"
    assert store.find("decode.kappa", {"bucket": "1"}, VGPU, new, q8(), prior=1.0).levels["engine"] == "record"


def test_a_build_without_base_runs_keeps_the_family_s_ttft_terms(inputs):
    """T-GND-20: a speculative prefill ratio and the wave's survival are fitted relative to the build's own base prefill
    fit. Fitted on a build with no base runs (bud-vllm 0.10.0 before queue F) they absorbed the gap to the family's base
    terms instead: spec.prefill 3.27 against the family's 1.21, and 17.6 s predicted for a 1.8 s long-prompt TTFT. An
    empty grid search also wrote prefill.mix 0.25 with no samples. Such a build is fitted after the engine-family level
    is aggregated and keeps the family's terms."""
    import dataclasses

    keep = [
        c
        for c in inputs.corpus
        if c.family == "Qwen3-8B"
        and (
            (c.record == "vllm@0.30.0" and c.name.startswith(("8b-vanilla", "8b-s-", "8b-f-wbf16-spnone")))
            or c.name.startswith("c-8b-010-eagle3")
        )
    ]
    assert any(c.record == "bud-vllm@0.10.0" for c in keep)
    small = fit_store(dataclasses.replace(inputs, corpus=keep, version="t-gnd-20"))
    new = rec("bud-vllm-0.10.0")
    mine = [e for e in small.entries if e.key.engine == "record:bud-vllm@0.10.0"]
    assert mine and all(e.samples > 0 for e in mine)
    for kind in ("spec.prefill", "arrival.sync", "prefill.mix", "prefill.kappa", "decode.kappa"):
        assert not any(e.kind == kind for e in mine), kind
    found = small.find("prefill.mix", {}, VGPU, new, q8(), prior=1.0)
    assert found.levels["engine"] == "family" and found.value != 0.25
    report = replay(dataclasses.replace(inputs, corpus=[c for c in keep if c.record == "bud-vllm@0.10.0"]), small)
    pts = report.select("ttft", family="Qwen3-8B", kind="spec")
    assert pts and report.mape(pts) <= 0.50, report.mape(pts)


def test_bare_metal_falls_to_architecture(store):
    found = store.find(
        "decode.kappa", {"bucket": "1"}, catalogue.get("h100-sxm5-80gb"), rec(), q8(), prior=1.0
    )
    assert found.levels["hardware"] == "arch"
    a100 = store.find(
        "decode.kappa", {"bucket": "1"}, catalogue.get("a100-80gb"), rec(), q8(), prior=1.0
    )
    assert a100.levels["hardware"] == "prior"


def test_prediction_reports_levels_per_dimension(store):
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=16)
    p = engine_for(rec(), store).predict({}, w, q8(), catalogue.get("h100-sxm5-80gb"))
    assert p.levels["hardware"] == "arch" and p.levels["model"] == "family"


# ---------------------------------------------------------------- T-GND-02 bands decide


def test_bands_must_separate():
    a = Estimate.around(100, 0.10, ceiling=0.15)
    b = Estimate.around(105, 0.10, ceiling=0.15)
    c = Estimate.around(140, 0.10, ceiling=0.15)
    assert not b.separates_above(a) and c.separates_above(a)
    wide = Estimate.around(200, 0.40, ceiling=0.15)
    assert wide.ungrounded and not wide.separates_above(a)


def test_prior_level_is_ungrounded(store):
    unknown = ModelFacts.from_config(load_model("unknown-arch"))
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=4)
    p = engine_for(rec(), store).predict({}, w, unknown, catalogue.get("a100-80gb"))
    assert p.ungrounded and p.levels["model"] == "prior"
    assert "architecture: unrecognised" in p.notes


# ---------------------------------------------------------------- T-GND-03 accuracy gate (NFR-1)


def test_memory_within_ceiling(report):
    assert report.memory_mape() <= 0.03


def test_a_shared_slice_s_kv_pool_is_reported_in_absolute_bytes(report):
    """A HAMi slice leaves a small pool, so the 3% ceiling (a whole-GPU pool) does not apply to it: the replay reports
    its error in bytes against the memory band, and keeps it out of the dedicated memory MAPE (TCS 10-18: the 27B on
    69,888 MiB 10.81 vs 10.80 GiB; the corpus' 66,956 MiB slices 0.3-0.55 GiB, 4-7%)."""
    shared = report.memory_shared
    assert shared and {m["config"] for m in shared} >= {"bud-render", "s-slice66956-seqs64-l64k"}
    assert not {m["config"] for m in shared} & {m["config"] for m in report.memory}
    for m in shared:
        assert m["slice_mib"] and m["band"] > 0
        assert abs(m["predicted"] - m["measured"]) <= m["band"], m


@pytest.mark.parametrize("family", ["Qwen3.8-27B", "Qwen3-8B"])
@pytest.mark.parametrize("kind", ["base", "precision", "lora"])
def test_decode_within_nfr1_ceilings(report, family, kind):
    tpot = report.select("tpot", family=family, kind=kind)
    thr = report.select("throughput", family=family, kind=kind)
    assert tpot and thr
    assert report.mape(tpot) <= 0.15 and report.mape(thr) <= 0.15


def test_mtp_decode_within_nfr1_ceilings(report):
    """T-GND-09, 27B MTP (A-08): acceptance is grounded where the wave cannot move TPOT, the wave's survival on TPOT at c > 1,
    so the interference a wave point carries is not folded into the acceptance of every load."""
    tpot = report.select("tpot", family="Qwen3.8-27B", kind="spec")
    thr = report.select("throughput", family="Qwen3.8-27B", kind="spec")
    assert tpot and thr
    assert report.mape(tpot) <= 0.10 and report.mape(thr) <= 0.15


def test_mtp_decode_bands_within_twice_the_ceiling(report):
    from llm_memory_calculator.grounding.gate import ceiling

    pts = report.select("tpot", family="Qwen3.8-27B", kind="spec") + report.select(
        "throughput", family="Qwen3.8-27B", kind="spec"
    )
    wide = [
        p
        for p in pts
        if max(p.high / p.predicted - 1, 1 - p.low / p.predicted)
        > 2 * ceiling(p.quantity, p.queueing or p.concurrency >= 64) + 1e-9
    ]
    assert len(wide) <= 0.10 * len(
        pts
    ), f"{len(wide)}/{len(pts)} 27B MTP bands wider than 2x the ceiling"


def test_a_fixed_count_run_drains_and_a_plan_does_not(store):
    """T-GND-10: a benchmark of N prompts at concurrency c ends with its last request: as the run drains, fewer than c run, and
    the throughput over its duration falls below the open-ended rate. A plan's workload names no request count, so it
    is predicted open-ended."""
    eng = engine_for(rec(), store)
    model, settings = q8(), {}
    steady = dict(input_tokens=200, output_tokens=355, concurrency=8, arrival="steady")
    open_ended = eng.predict(settings, Workload(**steady), model, VGPU)
    fixed = eng.predict(settings, Workload(**steady, requests=48), model, VGPU)
    assert fixed.throughput_tok_s.value < open_ended.throughput_tok_s.value
    assert fixed.tpot_ms.value == open_ended.tpot_ms.value
    # identical synthetic requests in waves have no variable-length tail: the wave model covers them
    wave = dict(input_tokens=1024, output_tokens=256, concurrency=16, workload_class="synthetic")
    assert eng.predict(
        settings, Workload(**wave, requests=48), model, VGPU
    ).throughput_tok_s.value == pytest.approx(
        eng.predict(settings, Workload(**wave), model, VGPU).throughput_tok_s.value
    )
    single = Workload(input_tokens=200, output_tokens=355, concurrency=1, requests=8)
    assert eng.predict(settings, single, model, VGPU).throughput_tok_s.value == pytest.approx(
        eng.predict(
            settings, Workload(input_tokens=200, output_tokens=355, concurrency=1), model, VGPU
        ).throughput_tok_s.value
    )


def test_a_prefix_run_s_first_wave_is_cold(store):
    """T-GND-10: a fixed-count run of identical-prefix waves finds nothing cached in its first wave. Its measured
    run-average hit rate (0.65 over 4 waves) is the warm waves' hit spread over all four: the median request sees the
    warm hit (lower TTFT than the average hit implies), and the run's throughput carries the cold wave once.
    """
    eng = engine_for(rec(), store)
    model = q8()
    base = dict(
        input_tokens=4608,
        output_tokens=256,
        concurrency=16,
        workload_class="synthetic",
        prefix_hit_rate=0.65,
    )
    averaged = eng.predict({}, Workload(**base), model, VGPU)
    fixed = eng.predict({}, Workload(**base, requests=64), model, VGPU)
    assert fixed.ttft_ms.value < averaged.ttft_ms.value
    warm_only = eng.predict({}, Workload(**{**base, "prefix_hit_rate": 0.65 * 4 / 3}), model, VGPU)
    assert fixed.ttft_ms.value == pytest.approx(warm_only.ttft_ms.value)
    assert fixed.throughput_tok_s.value < warm_only.throughput_tok_s.value


def test_a_warm_prefix_hit_never_exceeds_the_shared_prefix(store):
    """T-GND-10: the warm waves' hit is capped at the prefix's share of the prompt (block-aligned). The study's prefix
    point shares 4,096 of 4,608 tokens; at a run-average hit of 0.76 over 4 waves the uncapped warm hit saturated at
    1.0 and the base 27B was predicted at 42 ms against 905 ms measured."""
    eng = engine_for(rec(), store)
    model = q8()
    base = dict(
        input_tokens=4608,
        output_tokens=256,
        concurrency=16,
        workload_class="synthetic",
        prefix_tokens=4096,
        requests=64,
    )
    share = (4096 // 16 * 16) / 4608
    over = eng.predict({}, Workload(**base, prefix_hit_rate=0.76), model, VGPU)
    at_cap = eng.predict({}, Workload(**base, prefix_hit_rate=share * 3 / 4), model, VGPU)
    assert over.ttft_ms.value == pytest.approx(at_cap.ttft_ms.value)
    cold = eng.predict({}, Workload(**{**base, "requests": None}, prefix_hit_rate=0.0), model, VGPU)
    assert over.ttft_ms.value > 0.05 * cold.ttft_ms.value  # 512 new tokens a request, not none


@pytest.mark.parametrize(
    "workload, shape",
    [
        (dict(input_tokens=4608, output_tokens=256, prefix_tokens=4096), "prefix"),
        (dict(input_tokens=1024, output_tokens=256, prefix_hit_rate=0.5), "prefix"),
        (dict(input_tokens=8192, output_tokens=256), "long"),
        (dict(input_tokens=256, output_tokens=1024), "decode"),
        (dict(input_tokens=1024, output_tokens=256), "chat"),
        (dict(input_tokens=300, output_tokens=350, workload_class="real"), "real"),
    ],
)
def test_a_workload_has_a_shape(workload, shape):
    """The speculative TTFT terms are keyed by the workload's shape, derived from the workload itself."""
    assert Workload(concurrency=16, **{"workload_class": "synthetic", **workload}).shape == shape


def test_real_text_throughput_without_speculation_within_ceiling(report):
    """Real-text points at c > 1 (variable output lengths, fixed prompt counts) were 14-24% high on throughput while
    their TPOT matched: the drain was not modelled."""
    for family in ("Qwen3.8-27B", "Qwen3-8B"):
        pts = [
            p
            for p in report.select("throughput", family=family, kind="base")
            if p.point in ("real", "realS") and p.concurrency > 1
        ]
        assert pts and report.mape(pts) <= 0.10, family


@pytest.mark.parametrize("fmt", ["bf16", "fp8"])
def test_advised_speculative_gain_holds_for_both_weight_formats(report, fmt):
    """A-08 on the study factorial (27B MTP k = 1..3 over no speculation): the measured gain is inside the advised
    band at >= 90% of the shared points. The band of a ratio of two predictions combines their relative bands in
    quadrature; the spec band alone ignored the base prediction's error, which store 19.1's calibrated bands no longer
    hid (bf16 26/30, FP8 21/30). FP8 was 26/30 on store .6, whose residuals measured |predicted / measured - 1|."""
    by_cfg = {}
    for p in report.points:
        if (
            p.quantity == "throughput"
            and p.config.endswith("-kva-ss16")
            and p.config.startswith("f-w")
        ):
            by_cfg.setdefault(p.config, {})[(p.point, p.concurrency)] = p
    base = by_cfg[f"f-w{fmt}-m0-kva-ss16"]
    inside = total = 0
    for k in (1, 2, 3):
        for key, p in by_cfg[f"f-w{fmt}-m{k}-kva-ss16"].items():
            b = base.get(key)
            if b is None or not b.predicted or not b.measured:
                continue
            measured = p.measured / b.measured
            advised = p.predicted / b.predicted
            r = math.hypot(_half_width(p), _half_width(b))
            inside += advised * (1 - r) <= measured <= advised * (1 + r)
            total += 1
    assert total and inside / total >= 0.90, (fmt, inside, total)


def test_ttft_within_ceiling_for_dense_and_hybrid_base(report):
    for family in ("Qwen3.8-27B", "Qwen3-8B"):
        for kind in ("base", "precision", "lora"):
            assert (
                report.mape(report.select("ttft", family=family, kind=kind, queueing=False)) <= 0.20
            ), (family, kind)
    assert report.mape(report.select("ttft", queueing=True)) <= 0.30


def test_overall_corpus_regression_guard(report):
    """Every family and setting together. gpt-oss (MoE) and speculative decoding are above the NFR-1 ceiling
    today (FRD-027 §16 known gaps); this guard keeps them from regressing while they are improved.
    """
    s = report.summary()
    assert s["tpot"]["mape"] <= 0.15
    assert s["throughput"]["mape"] <= 0.18
    assert s["ttft"]["mape"] <= 0.35
    assert s["ttft_queueing"]["mape"] <= 0.30


def test_speculative_ttft_terms_are_keyed_by_workload_shape(store):
    """T-GND-12: the speculative TTFT terms (arrival.sync, spec.prefill) carry an entry per workload shape, fitted on
    TTFT alone; TPOT and throughput keep the pooled value. 27B MTP on vLLM 0.30: the decode wave keeps half its burst,
    the long one none of it, where one pooled value had made every TTFT band +-68%."""
    shapes = {
        e.detail.get("shape")
        for e in store.entries
        if e.kind == "arrival.sync"
        and e.detail.get("method") == "mtp"
        and e.key.model == "family:Qwen3.8-27B"
        and e.key.hardware.startswith("sku:")
    }
    assert {"chat", "decode", "long", "prefix"} <= shapes
    eng = engine_for(rec(), store)
    model = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    settings = {"spec.method": "mtp", "spec.tokens": 2}
    w = Workload(input_tokens=256, output_tokens=1024, concurrency=32, workload_class="synthetic", requests=64)
    assert w.shape == "decode"
    pred = eng.predict(settings, w, model, VGPU)
    p = eng.params(model, VGPU, eng.resolve(settings, model, VGPU), w, running=32, decode_running=32)
    assert p.found["arrival.sync.ttft"].value != p.found["arrival.sync"].value
    # the shape's value moves TTFT only: without the shape entries, TTFT changes and TPOT and throughput do not
    pooled_only = GroundingStore(
        [
            e
            for e in store.entries
            if not (e.kind in ("arrival.sync", "spec.prefill") and "shape" in e.detail)
        ],
        version=store.version,
    )
    pooled = engine_for(rec(), pooled_only).predict(settings, w, model, VGPU)
    assert pred.ttft_ms.value != pytest.approx(pooled.ttft_ms.value)
    assert pred.tpot_ms.value == pytest.approx(pooled.tpot_ms.value)
    assert pred.throughput_tok_s.value == pytest.approx(pooled.throughput_tok_s.value)


def test_a_prefix_load_without_prefix_caching_takes_its_own_shape(store, report):
    """A shared-prefix load cannot hit on an engine whose prefix cache is off (MTP plans turn it off, vLLM #53912), but it
    is not a random long load either: the repeated text decodes with higher MTP acceptance, and the requests desynchronize.
    0.30 27B MTP k=2, cache off, 4,608 tokens at c16: the prefix load measured 671 ms and the random one 1,031 ms. Taken as
    a long load it was predicted at 1,235 ms (TCS k=2/k=3: 639/698 ms measured), so it has its own shape,
    ``prefix-cold`` (T-GND-25)."""
    w = Workload(
        input_tokens=4608,
        output_tokens=256,
        concurrency=16,
        workload_class="synthetic",
        prefix_tokens=4096,
        requests=64,
    )
    assert w.shape_for(True) == "prefix" and w.shape_for(False) == "prefix-cold"
    long_w = Workload(input_tokens=4608, output_tokens=256, concurrency=16, workload_class="synthetic")
    assert long_w.shape_for(False) == "long"
    pts = [
        p
        for p in report.select("ttft", family="Qwen3.8-27B", kind="spec")
        if p.point == "prefix" and p.config in ("g-27b-mtp1-pc-off", "g-27b-mtp2-pc-off", "g-27b-mtp3-pc-off")
    ]
    assert len(pts) >= 4 and all(p.inside for p in pts), [(p.config, p.predicted, p.measured) for p in pts]


@pytest.mark.parametrize(
    "hub_id, corpus_path",
    [
        ("RedHatAI/Qwen3-8B-speculator.eagle3", "/drafts/eagle3-RedHatAI_Qwen3-8B-speculator.eagle3"),
        ("AngelSlim/Qwen3-8B_eagle3", "/drafts/eagle3-AngelSlim_Qwen3-8B_eagle3"),
        ("zhuyksir/EAGLE3-gpt-oss-20b-bf16", "/drafts/eagle3-zhuyksir_EAGLE3-gpt-oss-20b-bf16"),
    ],
)
def test_a_draft_has_one_identity_by_hub_id_or_staged_path(hub_id, corpus_path):
    """T-SPEC-08: the draft's grounding key is the same whether a plan names it by hub id (budsim's draft_path) or the
    study staged it under /drafts (the corpus)."""
    from llm_memory_calculator.engine import draft_key

    assert draft_key(hub_id) == draft_key(corpus_path) == draft_key(hub_id + "/")


def test_eagle3_acceptance_is_keyed_by_draft(report):
    """T-SPEC-08: EAGLE3 acceptance is grounded per draft. Keyed without it, the 8B's c1 TPOT was 9% off with the
    RedHat draft (11 points), 45% with AngelSlim's and 40% with Tengyunw's (one k=3 config each); per draft it is 5%,
    15% and 16%. NFR-1's 15% is a MAPE: the c1 points meet it together, and a draft measured by a single
    configuration stays within the relaxed 1.5x ceiling."""
    by_draft = {}
    for p in report.select("tpot", family="Qwen3-8B", kind="spec"):
        if "eagle3" in p.config and p.concurrency == 1:
            draft = "angelslim" if "angel" in p.config else "tengyunw" if "teng" in p.config else "redhat"
            by_draft.setdefault(draft, []).append(p)
    assert {"redhat", "angelslim", "tengyunw"} <= set(by_draft)
    assert report.mape([p for pts in by_draft.values() for p in pts]) <= 0.15
    assert report.mape(by_draft["redhat"]) <= 0.15
    for draft, pts in by_draft.items():
        assert report.mape(pts) <= 1.5 * 0.15, (draft, round(report.mape(pts), 3))


def test_mtp_ttft_bands_are_narrow_and_calibrated(report):
    """A-05 on the corpus: with the TTFT terms per shape and bands calibrated around the prediction (a residual is
    p90 |measured / predicted - 1|), the 27B's MTP TTFT points are >= 90% in band and about a tenth of their bands are
    wider than twice the ceiling (store .6: 168 of 260; 19.1: 29 of 284; 19.3: 33 of 292)."""
    from llm_memory_calculator.grounding.gate import ceiling

    # bud-vllm 0.10.0 ran no base suite, so its TTFT terms are the engine family's (inflated, T-GND-20)
    pts = [
        p
        for p in report.select("ttft", family="Qwen3.8-27B", kind="spec")
        if not p.queueing and not p.config.startswith("c-27b-010-")
    ]
    assert pts and report.coverage(pts) >= 0.90
    wide = [
        p
        for p in pts
        if max(p.high / p.predicted - 1, 1 - p.low / p.predicted)
        > 2 * ceiling("ttft", p.concurrency >= 64)
    ]
    # the wide ones are the prefix c16 and cache-off long loads (§19.2): with the cache off k=2/3 measured 0.67-0.68 s
    # where k=1 with the same arguments measured 1.10-1.17 s, one point per k=2/3, so no per-k value is grounded (19.3:
    # 33 of 292)
    assert len(wide) <= 0.12 * len(pts), len(wide)



def _half_width(p):
    return max(p.high / p.predicted - 1, 1 - p.low / p.predicted)


def test_speculative_decode_bands_are_calibrated_per_stratum(report):
    """T-GND-19 (A-05): a speculative TPOT / throughput band is the p90 error of its stratum -- method (and draft),
    workload shape, batch bucket -- not the pooled acceptance residual, which carries the worst stratum everywhere. On
    store .7 the 8B's EAGLE3 bands were +-58% (18 of 21 TCS bands past twice the ceiling) at errors of 1-19%, and its
    n-gram bands +-85%. Coverage stays >= 90%. Store 19.1: 45% of the 8B's speculative bands are wider than twice the
    ceiling (77% on .7; the rest are strata the corpus measured fewer than four times, which keep the terms' band) and
    6% of the 27B's (7%)."""
    from llm_memory_calculator.grounding.gate import ceiling

    for family, share in (("Qwen3-8B", 0.50), ("Qwen3.8-27B", 0.10)):
        for quantity in ("tpot", "throughput"):
            pts = report.select(quantity, family=family, kind="spec")
            assert pts and report.coverage(pts) >= 0.90, (family, quantity, report.coverage(pts))
            wide = [p for p in pts if _half_width(p) > 2 * ceiling(quantity, False) + 1e-9]
            assert len(wide) <= share * len(pts), (family, quantity, len(wide), len(pts))
    # the 8B's EAGLE3 long c32 TPOT, 53-66% under-predicted on store .7, replays within 10%
    long32 = [
        p
        for p in report.select("tpot", family="Qwen3-8B", kind="spec")
        if p.point == "long" and p.concurrency == 32 and "eagle3" in p.config
    ]
    assert long32 and report.mape(long32) <= 0.10, report.mape(long32)


def test_an_unmeasured_stratum_keeps_the_pooled_band(store):
    """T-GND-19: the stratum's TPOT bias and bands apply only where the stratum was measured; elsewhere (a shape the
    corpus never ran at that batch) the pooled terms decide, as before."""
    eng = engine_for(rec(), store)
    draft = ModelFacts.from_config(
        load_model("eagle3-qwen3-8b-redhatai"), family="draft:" + draft_key("RedHatAI/Qwen3-8B-speculator.eagle3")
    )
    s = {"spec.method": "eagle3", "spec.tokens": 3, "spec.draft": "/drafts/eagle3-RedHatAI_Qwen3-8B-speculator.eagle3"}
    r = eng.resolve(s, q8(), VGPU)
    chat = Workload(input_tokens=1024, output_tokens=256, concurrency=16)
    p = eng.params(q8(), VGPU, r, workload=chat, draft=draft, running=16, decode_running=16)
    for name in ("spec.scale", "band.tpot", "band.throughput"):
        assert name in p.found and p.found[name].levels["model"] != "prior", name
    long = Workload(input_tokens=8192, output_tokens=256, concurrency=32)
    p = eng.params(q8(), VGPU, r, workload=long, draft=draft, running=32, decode_running=32)
    assert "spec.scale" in p.found and 0.8 <= p.found["spec.scale"].value <= 1.8
    odd = Workload(input_tokens=30000, output_tokens=256, concurrency=128)  # no 30K-token c128 EAGLE3 run
    p = eng.params(q8(), VGPU, r, workload=odd, draft=draft, running=128, decode_running=128)
    assert not {"spec.scale", "band.tpot", "band.throughput"} & set(p.found)


def test_a_build_s_own_speculative_ttft_terms_need_three_configurations(inputs):
    """T-GND-22: fitted from two MTP configurations (bud-vllm 0.10.0's 27B MTP3 at online FP8, the cache on and off) a
    build's spec.prefill came out at 2.12 against the family's 1.21 and predicted a bf16 MTP2 deployment's long-prompt
    TTFT at 2.5x, and its TPOT interference sat at the solver's bound (2.0). With fewer than three configurations of the
    method the engine family's terms serve; round 8's MTP2 run is the third. A replicate does not count again."""
    import dataclasses

    base = [
        c
        for c in inputs.corpus
        if c.family == "Qwen3.8-27B"
        and (
            (c.record == "vllm@0.30.0" and c.name.startswith(("vanilla", "s-mtp", "f-wbf16-m0", "f-wbf16-m1")))
            or c.name in ("c-27b-010-base256", "c-27b-010-mtp3-pc-on", "c-27b-010-mtp3-pc-off")
        )
    ]
    new = rec("bud-vllm-0.10.0")
    q27 = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    two = fit_store(dataclasses.replace(inputs, corpus=base, version="t-gnd-22a"))
    for kind in ("spec.prefill", "arrival.sync", "spec.mix"):
        assert two.find(kind, {"method": "mtp"}, VGPU, new, q27, prior=1.0).levels["engine"] != "record", kind
    third = [c for c in inputs.corpus if c.name == "c-27b-010-mtp2-bf16"]
    three = fit_store(dataclasses.replace(inputs, corpus=base + third, version="t-gnd-22b"))
    assert three.find("spec.prefill", {"method": "mtp"}, VGPU, new, q27, prior=1.0).levels["engine"] == "record"
    # a replicate is the same configuration again, not a third one (round 9 repeated 0.10.0's MTP plans)
    again = [dataclasses.replace(c, name=c.name + "-again") for c in base if c.name == "c-27b-010-mtp3-pc-on"]
    replicated = fit_store(dataclasses.replace(inputs, corpus=base + again, version="t-gnd-22c"))
    for kind in ("spec.prefill", "arrival.sync", "spec.mix"):
        assert replicated.find(kind, {"method": "mtp"}, VGPU, new, q27, prior=1.0).levels["engine"] != "record", kind


def test_a_warm_repeat_still_prefills_its_last_block(store):
    """T-GND-23: vLLM caches full blocks only and always computes a prompt's last token, so a request re-sent whole hits
    at most floor((input - 1) / block) blocks. The first-wave correction pushed a repeat run's warm hit (0.984 of 1,024
    tokens) to 1.0, and every warm wave at c > 1 predicted the API time alone: gpt-oss -r2 at c16/c64 12.4 ms against
    37-60 ms measured."""
    model = ModelFacts.from_config(load_model("gpt-oss-20b"), family="gpt-oss-20b")
    eng = engine_for(rec(), store)
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=16, requests=48, prefix_hit_rate=0.9844, arrival="wave")
    pred = eng.predict({}, w, model, VGPU)
    api = eng.last_components["api_ms"]
    assert pred.ttft_ms.value > api + 2.0, (pred.ttft_ms.value, api)
    cold = eng.predict({}, Workload(input_tokens=1024, output_tokens=256, concurrency=16, requests=48, arrival="wave"), model, VGPU)
    assert pred.ttft_ms.value < cold.ttft_ms.value



def test_grounded_points_are_inside_their_band(report):
    grounded = [p for p in report.points if not p.ungrounded]
    assert grounded
    assert report.coverage(grounded) >= 0.90


def test_no_band_wider_than_twice_the_ceiling_passes(report):
    from llm_memory_calculator.grounding.gate import ceiling

    for p in report.points:
        half_width = max(p.high / p.predicted - 1, 1 - p.low / p.predicted)
        if half_width > 2 * ceiling(p.quantity, p.queueing or p.concurrency >= 64) + 1e-9:
            assert p.ungrounded, (p.config, p.quantity, half_width)


def test_family_hold_out(inputs):
    """Fit without Qwen3-8B; predict it from the other families: within 1.5x the ceilings, and reported
    ungrounded so no module acts on it."""
    held = build_inputs(
        FIXTURES / "corpus",
        FIXTURES / "models",
        FIXTURES / "records",
        version="holdout",
        exclude_families=["Qwen3-8B"],
    )
    store = fit_store(held)
    rep = replay(inputs, store, families=["Qwen3-8B"], kinds=["base"])
    assert rep.mape(rep.select("tpot")) <= 1.5 * 0.15
    assert rep.memory_mape() <= 1.5 * 0.03
    assert all(p.ungrounded for p in rep.select("tpot"))


@pytest.mark.skip(
    reason="needs a second measured SKU (FRD-027 FR-GND-3); written so it runs once one exists"
)
def test_sku_hold_out():
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- ranking (Phase 0 exit)


@pytest.mark.parametrize(
    "family, configs",
    [
        ("Qwen3.8-27B", ["s-base-rep", "s-fp8w-ckpt", "f-wfp8-m3-kv8-ss16", "bud-render"]),
        ("Qwen3-8B", ["8b-s-base-rep", "8b-s-fp8w-ckpt", "8b-bud-render"]),
    ],
)
def test_ranking_matches_measured_order(inputs, store, family, configs):
    """The predicted order of the study's configurations equals the measured order; pairs measured within 3% of
    each other are ties (TEST_CASES T-GND-03: "ties within band allowed")."""
    from llm_memory_calculator.grounding.fit import (
        device_for,
        draft_for,
        model_for,
        predict_settings,
        workload_for,
    )

    for point, conc in (("chat", 1), ("chat", 16), ("chat", 64)):
        measured, predicted = [], []
        for cfg in inputs.corpus:
            if cfg.name not in configs:
                continue
            pts = [p for p in cfg.points if p.point == point and p.concurrency == conc]
            if not pts:
                continue
            eng = engine_for(inputs.records[cfg.record], store)
            pred = eng.predict(
                predict_settings(cfg),
                workload_for(pts[0]),
                model_for(inputs, cfg),
                device_for(cfg),
                draft=draft_for(inputs, cfg),
            )
            measured.append(pts[0].output_tok_s)
            predicted.append(pred.throughput_tok_s.value)
        pairs = concordant = 0
        for i in range(len(measured)):
            for j in range(i + 1, len(measured)):
                if abs(measured[i] / measured[j] - 1) <= 0.03:
                    continue
                pairs += 1
                concordant += (measured[i] > measured[j]) == (predicted[i] > predicted[j])
        assert pairs and concordant == pairs, (point, conc, measured, predicted)


# ---------------------------------------------------------------- T-GND-06 boot outcomes


def test_boot_outcome_widens_the_memory_band(store):
    s = GroundingStore(list(store.entries), [], version="t")
    model = q8()
    eng = engine_for(rec(), s)
    before = eng.limits({}, model, VGPU).memory_band_bytes
    s.record_boot_outcome(
        VGPU, rec(), model, {"memory.device_fraction": 0.95}, 1.2 * GIB, "oom", "CUDA out of memory"
    )
    after = eng.limits({}, model, VGPU).memory_band_bytes
    assert after - before >= 1.2 * GIB - 1
    s.record_boot_outcome(VGPU, rec(), model, {"memory.device_fraction": 0.95}, 0.8 * GIB, "oom")
    assert eng.limits({}, model, VGPU).memory_band_bytes == pytest.approx(
        after
    )  # no double widening
    assert s.outcomes[0].settings == {"memory.device_fraction": 0.95}


def test_seeded_boot_outcomes_carry_provenance(store):
    names = {o.provenance.get("config") for o in store.outcomes}
    assert {"s-util097", "s-seqs512-cg256", "8b-s-util024"} <= names
    for o in store.outcomes:
        assert o.outcome in ("oom", "refused", "error") and o.provenance["run_id"]


def test_predictor_agrees_with_the_seeded_refusals(store, inputs):
    """The predictor refuses the configurations vLLM refused (T-ENG-02..04): the falsification record is clean."""
    agreed = [
        o
        for o in store.outcomes
        if o.outcome in ("oom", "refused") and not o.provenance.get("predicted_boots", True)
    ]
    assert len(agreed) >= 6


# ---------------------------------------------------------------- T-GND-07, provenance, versions


def test_vgpu_entries_never_keyed_bare_metal(store):
    assert not [e for e in store.entries if e.key.hardware == "sku:h100-sxm5-80gb"]
    vgpu = {
        (e.kind, e.key.engine, e.key.model, tuple(sorted(e.detail.items()))): e
        for e in store.entries
        if e.key.hardware == "sku:h100xm-80c-vgpu"
    }
    for e in store.entries:
        if e.key.hardware == "arch:hopper":
            src = vgpu.get((e.kind, e.key.engine, e.key.model, tuple(sorted(e.detail.items()))))
            if src is not None:
                assert e.residual > src.residual


def test_every_entry_has_provenance(store):
    for e in store.entries:
        assert (
            e.provenance.get("run_id") and e.provenance.get("harness") and e.provenance.get("date")
        )


def test_fit_is_deterministic(inputs):
    a = fit_store(inputs)
    b = fit_store(inputs)
    assert a.digest() == b.digest()


def test_versions_and_memo(store):
    v = versions(store)
    assert v["grounding"] == store.version and v["version"].startswith("lmc-")
    pred = Predictor(rec(), store, max_evaluations=2)
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=1)
    a = pred.predict({}, w, q8(), VGPU)
    b = pred.predict({}, w, q8(), VGPU)
    assert a is b and pred.evaluations == 1 and pred.calls == 2
    pred.predict({"batch.max_running": 64}, w, q8(), VGPU)
    with pytest.raises(EvaluationBudgetExceeded):
        pred.predict({"batch.max_running": 32}, w, q8(), VGPU)


def test_store_round_trip(tmp_path, store):
    copy_ = GroundingStore(store.entries, store.outcomes, version="rt", metadata=store.metadata)
    copy_.save(tmp_path)
    again = GroundingStore.load("rt", tmp_path)
    assert again.digest() == copy_.digest()


def test_previous_store_versions_still_load():
    """NFR-4: a new store version replays the previous version's fixtures; every shipped version loads."""
    from llm_memory_calculator.grounding.store import DATA_DIR

    versions_ = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    assert versions_
    for v in versions_:
        assert GroundingStore.load(v).entries


def test_a_slow_boot_outlier_does_not_stretch_the_fallback_budget(store):
    """The boot band sets the fallback's per-rung budget (3x its top). Its residual keeps |predicted / measured - 1|:
    the band-consistent metric made one 1,238 s boot (median 240 s) a residual of 4.2 and the 27B's online-FP8 budget an
    hour; on this metric the budget is ~22 min and still holds the slowest boot measured (1,254 s)."""
    eng = engine_for(rec(), store)
    model = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    boot = eng.boot_time({"weights.online": "fp8"}, model, VGPU)
    assert 1254 <= 3 * boot.high <= 1800, boot


def test_boot_time_within_ceiling(report):
    """NFR-1: boot time within 25% (FR-ENG-7; the boot_failed budget is derived from it)."""
    assert report.summary()["boot"]["mape"] <= 0.25


def test_bands_carry_the_replicate_noise(inputs, store):
    """A fitted constant matches the run it was fitted on; a new run also differs by run-to-run noise.

    TCS 10-18: the 8B's real-text c8 TTFT band was +-5% while identical study runs differ by ~12% at that load, so a
    correct prediction landed outside its band. Out-of-sample band = fit residual (+) replicate noise.
    """
    from llm_memory_calculator.grounding.fit import replicate_noise

    noise = replicate_noise(inputs.corpus)
    ttft8 = noise[("Qwen3-8B", "ttft", "5-16")]
    assert 0.05 < ttft8 < 0.30
    assert noise[("Qwen3-8B", "tpot", "5-16")] < 0.05  # decode is steady run to run
    for record in (
        rec(),
        rec("bud-vllm-0.9.0-481e481be"),
    ):  # the fork has fewer runs: no narrower for that
        for workload in (
            Workload(1024, 256, 8, workload_class="synthetic", sampling="greedy"),
            Workload(229.4, 498.8, 8, workload_class="real", sampling="greedy"),  # the TCS miss
        ):
            pred = engine_for(record, store).predict(
                {"batch.max_running": 256}, workload, q8(), VGPU
            )
            assert pred.ttft_ms.width >= ttft8 - 1e-9, (
                record.id,
                workload.workload_class,
                pred.ttft_ms.width,
            )
    # identical command lines only: a flag the predictor does not model is not noise
    assert all(n < 1.5 for n in noise.values())


def test_lora_boots_are_their_own_boot_class(store):
    """A LoRA engine captures graphs for the adapter path and warms the adapters: on the fork the 27B boots in 149 s
    stock and 239-258 s with LoRA. Mixing them put a stock TCS boot (144 s) at a predicted 241 s."""
    from llm_memory_calculator.engine.vllm import boot_kind

    fork = engine_for(rec("bud-vllm-0.9.0-481e481be"), store)
    q27 = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    stock = fork.boot_time({"batch.max_running": 256}, q27, VGPU)
    lora = fork.boot_time(
        {"batch.max_running": 256, "lora.max_adapters": 1, "lora.max_rank": 64}, q27, VGPU
    )
    assert stock.value == pytest.approx(149, rel=0.10)
    assert lora.value == pytest.approx(243, rel=0.10)
    r = fork.resolve({"lora.max_adapters": 1, "lora.max_rank": 64}, q27, VGPU)
    assert boot_kind(r, q27) == "lora"


# ---------------------------------------------------------------- T-GND-27 noise-limited points (A-05 amendment)


def _cfg(name, args, points, record="vllm@0.30.0", family="gpt-oss-20b"):
    from llm_memory_calculator.grounding.corpus import CorpusConfig, Startup

    return CorpusConfig(
        name=name,
        family=family,
        checkpoint="mxfp4",
        record=record,
        sku="h100xm-80c-vgpu",
        hardware_mode="dedicated",
        slice_mib=None,
        args=list(args),
        settings={},
        unmodelled=[],
        guarded=False,
        startup=Startup(ready=True),
        points=list(points),
    )


def _pt(ttft, tpot=5.0, thr=1000.0, hit=0.0, point="chat", c=16):
    from llm_memory_calculator.grounding.corpus import Point

    return Point(
        point=point,
        input_tokens=1024.0,
        output_tokens=256.0,
        concurrency=c,
        num_prompts=48,
        completed=48,
        failed=0,
        ttft_ms=ttft,
        tpot_ms=tpot,
        output_tok_s=thr,
        sampling="greedy",
        workload_class="synthetic",
        prefix_hit_rate=hit,
    )


def test_replicate_spread_is_the_least_error_any_one_prediction_reaches():
    """T-GND-27: replicates are identical deployments (record, checkpoint, hardware mode, the whole command line) at the
    same prefix-cache state: a warm repeat of a cold run is another workload (gpt-oss's TCS -r2/-r3 repeats hit 98% of
    each prompt), and a run with another flag is not noise. A point's spread is (max - min) / (max + min), the smallest
    error any one prediction reaches on every replicate (gpt-oss chat c16 TTFT, 257 and 1,537 ms: 0.71)."""
    from llm_memory_calculator.grounding.fit import point_spread, spread_key

    args = ["--max-num-seqs=256"]
    a = _cfg("a", args, [_pt(257.0)])
    b = _cfg("b", args, [_pt(1537.0)])
    warm = _cfg("warm", args, [_pt(40.0, hit=0.98)])
    other = _cfg("other", ["--max-num-seqs=64"], [_pt(5000.0)])
    spread = point_spread([a, b, warm, other])
    got = spread[spread_key(a, a.points[0])]
    assert got["ttft"] == pytest.approx((1537 - 257) / (1537 + 257))
    assert got["tpot"] == 0.0 and got["throughput"] == 0.0
    best = 2 / (1 / 257 + 1 / 1537)  # the minimax prediction, the extremes' harmonic mean
    assert max(abs(best / 257 - 1), abs(best / 1537 - 1)) == pytest.approx(got["ttft"])
    assert spread_key(b, b.points[0]) == spread_key(a, a.points[0])
    assert spread_key(warm, warm.points[0]) not in spread  # one warm run has no replicate
    assert spread_key(other, other.points[0]) not in spread
    # a failed point is not a measurement
    broken = _cfg("broken", args, [_pt(0.0, tpot=0.0, thr=0.0)])
    assert point_spread([a, broken]) == {}


def test_a_warm_repeat_is_not_a_replicate_of_its_cold_run():
    """T-GND-27: the run-to-run floor pools replicates only at the same cache state; a cold run and its warm repeat
    (TTFT 828 and 37 ms) would otherwise put a 0.9 floor on every band of the bucket."""
    from llm_memory_calculator.grounding.fit import replicate_noise

    args = ["--max-num-seqs=256"]
    cold = [_cfg(f"c{i}", args, [_pt(800.0 + 10 * i)]) for i in range(6)]
    warm = [_cfg(f"w{i}", args, [_pt(37.0 + 0.2 * i, hit=0.98)]) for i in range(6)]
    noise = replicate_noise(cold + warm)
    assert noise[("gpt-oss-20b", "ttft", "5-16")] < 0.05


@pytest.fixture(scope="module")
def moe_store(inputs):
    """A store fitted on the two MoE families alone (gpt-oss and Qwen1.5-MoE): it carries the replicate noise entries
    and the two-family MoE class, whatever store the package defaults to (store 19.3 predates both, FRD §19.5)."""
    import dataclasses

    from llm_memory_calculator.grounding.corpus import load_corpus

    # round 9's runs (Qwen1.5-MoE, the cold TCS gpt-oss repeats) are a fixture of their own: the main corpus is the one
    # the default store was fitted on (FRD §19.5)
    round9 = load_corpus(FIXTURES / "corpus_round9")
    corpus = [c for c in inputs.corpus + round9 if c.family in ("gpt-oss-20b", "Qwen1.5-MoE-A2.7B")]
    return fit_store(dataclasses.replace(inputs, corpus=corpus, version="t-moe"))


def test_a_stratum_whose_replicates_disagree_is_noise_limited(moe_store):
    """T-GND-27 (A-05 amendment, FRD-027 §16): the store carries each stratum's replicate spread (noise.*) and a
    prediction reads it at its own family's level only. gpt-oss's base chat c16 TTFT differs by up to 0.7 between
    identical study runs, more than the ceiling, so its estimate is noise-limited; its TPOT is steady and is not. The flag
    is reporting: a band past twice the ceiling stays ungrounded, so modules still keep engine defaults (NFR-5)."""
    from llm_memory_calculator.engine.vllm import CEILINGS

    oss = ModelFacts.from_config(load_model("gpt-oss-20b"), family="gpt-oss-20b")
    eng = engine_for(rec(), moe_store)
    chat16 = Workload(1024, 256, 16, workload_class="synthetic", sampling="greedy", requests=48)
    pred = eng.predict({}, chat16, oss, VGPU)
    assert pred.ttft_ms.noise > CEILINGS["ttft"] and pred.ttft_ms.noise_limited
    assert not pred.tpot_ms.noise_limited
    assert pred.ttft_ms.ungrounded == (pred.ttft_ms.width > 2 * CEILINGS["ttft"])
    assert pred.to_dict()["ttft_ms"]["noise_limited"] is True
    calm = eng.predict({}, Workload(1024, 256, 1, workload_class="synthetic", sampling="greedy", requests=5), oss, VGPU)
    assert not calm.tpot_ms.noise_limited


def test_noise_is_never_borrowed_from_another_family(moe_store):
    store = moe_store
    """T-GND-27: a family the store lacks has no measured noise; another family's (gpt-oss's, in the MoE class) would
    excuse a wide band the hold-out earned on its own. Nor does the vGPU's noise describe a bare-metal H100."""
    oss = ModelFacts.from_config(load_model("gpt-oss-20b"), family="gpt-oss-20b")
    stranger = ModelFacts.from_config(load_model("gpt-oss-20b"), family="held-out-moe")
    chat16 = Workload(1024, 256, 16, workload_class="synthetic", sampling="greedy", requests=48)
    for model, device in ((stranger, VGPU), (oss, catalogue.get("h100-sxm5-80gb"))):
        pred = engine_for(rec(), store).predict({}, chat16, model, device)
        for est in (pred.ttft_ms, pred.tpot_ms, pred.throughput_tok_s):
            assert est.noise == 0.0 and not est.noise_limited, (model.family, device.sku)
    assert not [e for e in store.entries if e.kind.startswith("noise.") and not e.key.model.startswith("family:")]


def test_a_noise_limited_point_is_graded_on_coverage_only():
    """T-GND-27 (A-05 amendment): a point whose measured replicate noise exceeds its ceiling is graded on coverage only;
    its band's width is reported, not failed. Coverage still counts."""
    from llm_memory_calculator.grounding.gate import PointResult

    def point(noise_limited, measured=100.0, queueing=False):
        return PointResult("c", "f", "base", "chat", 16, "ttft", 100.0, 20.0, 180.0, measured, queueing, True, noise_limited)

    assert point(False).too_wide and point(False).inside
    assert not point(True).too_wide and point(True).inside
    assert not point(True, measured=200.0).inside
    assert point(False).half_width == pytest.approx(0.8)


def _run_dir(root, name, ttft, key="oss-mxfp4", hit=0.0):
    import json as _json

    d = root / name
    d.mkdir()
    (d / "config.json").write_text(_json.dumps({"name": "acc-oss-frd027", "image_key": "bud", "model": key, "args": []}))
    (d / "startup.json").write_text(_json.dumps({"ready": True, "wait_s": 114, "available_kv_gib": "58.75"}))
    row = {"point": "chat", "concurrency": 16, "num_prompts": 48, "completed": 48, "rc": 0, "input_len": 1024,
           "output_len": 256, "median_ttft_ms": ttft, "median_tpot_ms": 5.6, "output_throughput": 1774.7,
           "engine_delta": {"vllm:prefix_cache_queries_total": 1000, "vllm:prefix_cache_hits_total": 1000 * hit}}
    (d / "summary.jsonl").write_text(_json.dumps(row) + "\n")
    return d


def test_tcs_repeat_runs_join_the_corpus_as_replicates(tmp_path):
    """T-GND-27: the A-05 amendment's replicates are TCS repeat runs. Each is named by its directory (the repeats of one
    deployment share a config.json name), a later suite pass on the pod of an earlier one is not another boot, and a run
    whose checkpoint the corpus does not know is refused (the reader would take it for the 27B), as is a warm repeat."""
    from llm_memory_calculator.grounding.corpus import read_replicates
    from llm_memory_calculator.grounding.fit import point_spread

    _run_dir(tmp_path, "acc-oss-frd027", 829.0)
    _run_dir(tmp_path, "acc-oss-frd027-b", 1223.0)
    got = read_replicates(tmp_path, {"acc-oss-frd027": None, "acc-oss-frd027-b": "acc-oss-frd027"})
    assert [c.name for c in got] == ["acc-oss-frd027", "acc-oss-frd027-b"]
    assert got[0].startup.api_up_s == 114 and got[1].startup.api_up_s is None
    assert got[0].family == "gpt-oss-20b" and got[0].record == "bud-vllm@0.9.0-481e481be"
    assert len(point_spread(got)) == 1
    _run_dir(tmp_path, "acc-granite", 30.0, key="granite-3.3-8b")
    with pytest.raises(ValueError):
        read_replicates(tmp_path, {"acc-granite": None})
    with pytest.raises(FileNotFoundError):
        read_replicates(tmp_path, {"acc-missing": None})
    # a second suite pass on the same engine is served from its prefix cache (study harness caveat): not a first run,
    # and its 98%-cached points would widen the cold strata's bands (gpt-oss -r2/-r3: 9 bands past twice the ceiling)
    _run_dir(tmp_path, "acc-oss-frd027-warm", 37.0, hit=0.98)
    with pytest.raises(ValueError, match="warm"):
        read_replicates(tmp_path, {"acc-oss-frd027-warm": "acc-oss-frd027"})


# ---------------------------------------------------------------- T-GND-28 parallelism is ungrounded until measured (A-10)


def _with_parallel_point(store, sku="h100xm-80c-vgpu", tp=2):
    from llm_memory_calculator.grounding.store import Entry, Key

    point = Entry(
        kind="parallel.measured",
        key=Key(f"sku:{sku}", "record:vllm@0.30.0", "family:Qwen3.8-27B"),
        detail={"tp": tp, "pp": 1},
        value=1.0,
        residual=0.0,
        samples=1,
        provenance={},
    )
    return GroundingStore(list(store.entries) + [point], store.outcomes, version="t-gnd-28")


def test_tensor_parallel_is_ungrounded_until_measured(store):
    """T-GND-28 (A-10, FRD-027 §16): no TP > 1 point is measured (TCS has no bare-metal NVLink node), so a TP=2
    prediction is reported ungrounded, with a note saying why, on the vGPU and on bare metal; TP=1 is not. A measured
    TP=2 point grounds TP=2 on its SKU only: the parallel cost is the interconnect's (no NVLink on the vGPU)."""
    m27 = ModelFacts.from_config(load_model("qwen3.8-27b"), family="Qwen3.8-27B")
    w = Workload(1024, 256, 16, workload_class="synthetic", sampling="greedy")
    bare = catalogue.get("h100-sxm5-80gb")
    eng = engine_for(rec(), store)
    one = eng.predict({"parallel.tensor": 1}, w, m27, VGPU)
    two = eng.predict({"parallel.tensor": 2}, w, m27, VGPU)
    assert not one.ungrounded and not any("parallel" in n for n in one.notes)
    assert all(e.ungrounded for e in (two.ttft_ms, two.tpot_ms, two.throughput_tok_s))
    assert any(n.startswith("parallelism: TP=2 PP=1 ungrounded") for n in two.notes), two.notes
    assert any(n.startswith("parallelism:") for n in eng.predict({"parallel.pipeline": 2}, w, m27, VGPU).notes)
    on_bare = eng.predict({"parallel.tensor": 2}, w, m27, bare)
    assert on_bare.ungrounded and any(n.startswith("parallelism:") for n in on_bare.notes)
    measured = engine_for(rec(), _with_parallel_point(store))
    grounded = measured.predict({"parallel.tensor": 2}, w, m27, VGPU)
    assert not any(n.startswith("parallelism:") for n in grounded.notes)
    assert not grounded.tpot_ms.ungrounded
    assert any(n.startswith("parallelism:") for n in measured.predict({"parallel.tensor": 2}, w, m27, bare).notes)
    assert any(n.startswith("parallelism:") for n in measured.predict({"parallel.tensor": 4}, w, m27, VGPU).notes)


def test_the_fitter_records_each_measured_parallelism(inputs):
    """T-GND-28: a corpus configuration at TP > 1 leaves a parallel.measured entry on its SKU (none at the architecture
    level); the study's TP=1 corpus leaves none."""
    import dataclasses

    from llm_memory_calculator.grounding.fit import parallel_entries

    base = next(c for c in inputs.corpus if c.name == "8b-vanilla")
    assert parallel_entries(inputs.corpus) == []
    tp2 = dataclasses.replace(
        base, name="8b-tp2", args=base.args + ["--tensor-parallel-size=2"], settings={**base.settings, "parallel.tensor": 2}
    )
    got = parallel_entries([base, tp2])
    assert [(e.key.hardware, e.key.model, e.detail) for e in got] == [
        ("sku:h100xm-80c-vgpu", "family:Qwen3-8B", {"tp": 2, "pp": 1})
    ]


def test_a_default_valued_or_logging_flag_does_not_split_replicates():
    """T-GND-27: Bud's render spells out TP=1 and PP=1 and adds --trust-remote-code and a uvicorn log level; a bare
    pod on the same plan omits them. Neither changes the engine (round 9: the 0.10.0 8B EAGLE3 product plan and its
    two bare-pod repeats), so they are the same deployment. Any other flag, or TP=2, is another one."""
    from llm_memory_calculator.grounding.fit import replicate_ident

    bare = _cfg("bare", ["--max-num-seqs=256", "--quantization=fp8"], [_pt(100.0)])
    rendered = _cfg(
        "rendered",
        [
            "--trust-remote-code",
            "--tensor-parallel-size=1",
            "--pipeline-parallel-size",
            "1",
            "--max-num-seqs=256",
            "--quantization=fp8",
            "--uvicorn-log-level",
            "warning",
        ],
        [_pt(100.0)],
    )
    assert replicate_ident(bare) == replicate_ident(rendered)
    for extra in (["--tensor-parallel-size=2"], ["--enforce-eager"], ["--max-num-batched-tokens=4096"]):
        other = _cfg("other", bare.args + extra, [_pt(100.0)])
        assert replicate_ident(other) != replicate_ident(bare), extra


# ---------------------------------------------------------------- T-GND-26 the MoE class is grounded by two families (A-09)


def _class_entry(engine, value, families, kind="decode.kappa"):
    from llm_memory_calculator.grounding.store import Entry, Key

    return Entry(
        kind=kind,
        key=Key("sku:h100xm-80c-vgpu", engine, "class:MoE"),
        detail={"bucket": "5-16"},
        value=value,
        residual=0.2,
        samples=4,
        provenance={"families": list(families)},
    )


def test_a_family_the_store_lacks_reads_a_class_of_two_families_first():
    """T-GND-26: a class entry of one family has no between-family spread, so its band does not hold the class transfer
    (NFR-5, as for pooled entries). A family the store lacks reads a two-family class entry before a one-family one,
    even at a less specific engine level (the fork's MoE class is gpt-oss alone; upstream's has Qwen1.5-MoE too), and a
    one-family class entry still before a pooled one. A measured family keeps its own entries."""
    from llm_memory_calculator.grounding.store import Entry, Key

    fork = rec("bud-vllm-0.9.0-481e481be")
    minor = f"minor:{fork.minor_key}"
    hold = ModelFacts.from_config(load_model("qwen3-30b-a3b-fp8"), family="qwen3-30b-a3b")
    oss = ModelFacts.from_config(load_model("gpt-oss-20b"), family="gpt-oss-20b")
    own = Entry(
        kind="decode.kappa",
        key=Key("sku:h100xm-80c-vgpu", f"record:{fork.id}", "family:gpt-oss-20b"),
        detail={"bucket": "5-16"},
        value=0.45,
        residual=0.2,
        samples=3,
        provenance={},
    )
    one = _class_entry(f"record:{fork.id}", 0.45, ["gpt-oss-20b"])
    two = _class_entry(minor, 0.9, ["Qwen1.5-MoE-A2.7B", "gpt-oss-20b"])
    pooled = Entry(
        kind="decode.kappa",
        key=Key("sku:h100xm-80c-vgpu", f"record:{fork.id}", "pooled"),
        detail={"bucket": "5-16"},
        value=0.7,
        residual=0.2,
        samples=9,
        provenance={"families": ["Qwen3-8B", "gpt-oss-20b"]},
    )
    store = GroundingStore([own, one, two, pooled], version="t-gnd-26")
    found = store.find("decode.kappa", {"bucket": "5-16"}, VGPU, fork, hold, prior=1.0)
    assert (found.value, found.levels["model"], found.levels["engine"]) == (0.9, "class", "minor")
    assert store.find("decode.kappa", {"bucket": "5-16"}, VGPU, fork, oss, prior=1.0).value == 0.45
    without_two = GroundingStore([own, one, pooled], version="t-gnd-26b")
    found = without_two.find("decode.kappa", {"bucket": "5-16"}, VGPU, fork, hold, prior=1.0)
    assert (found.value, found.levels["model"]) == (0.45, "class")


def test_the_moe_class_is_grounded_by_two_families(moe_store):
    store = moe_store
    """T-GND-26 (A-09): round 9 measured a second MoE family, Qwen1.5-MoE-A2.7B (bf16, upstream 0.30). The MoE class's
    decode entries on upstream pool both families, and the Qwen3-30B-A3B hold-out on the fork reads them there."""
    from llm_memory_calculator.grounding.store import Key

    fork = rec("bud-vllm-0.9.0-481e481be")
    hold = ModelFacts.from_config(load_model("qwen3-30b-a3b-fp8"), family="qwen3-30b-a3b")
    assert hold.arch_class == "MoE"
    for bucket in ("1", "5-16", "17-64"):
        entry = [
            e
            for e in store.entries
            if e.kind == "decode.kappa"
            and e.key == Key("sku:h100xm-80c-vgpu", "record:vllm@0.30.0", "class:MoE")
            and e.detail == {"bucket": bucket}
        ]
        assert entry and entry[0].provenance["families"] == ["Qwen1.5-MoE-A2.7B", "gpt-oss-20b"], bucket
        found = store.find("decode.kappa", {"bucket": bucket}, VGPU, fork, hold, prior=1.0)
        assert (found.levels["model"], found.levels["engine"]) == ("class", "minor"), bucket




def test_another_family_s_runs_do_not_move_a_family_s_decode_terms(inputs):
    """T-GND-29: the prefill interference (prefill.mix) was one grid-searched value per build shared by every family,
    so another family's runs moved this one's decode terms: round 9's three gpt-oss TCS repeats flipped the fork's mix
    from 1.25 to 1.0, the fork 27B's decode kappas re-solved around it, and four of its TCS bands went past twice the
    ceiling. A family with MIX_MIN_SAMPLES base points has its own mix; the pooled one serves the rest."""
    import dataclasses

    from llm_memory_calculator.grounding.fit import kind_of
    from llm_memory_calculator.grounding.store import Key

    fork = "bud-vllm@0.9.0-481e481be"
    on_fork = [c for c in inputs.corpus if c.record == fork and kind_of(c) == "base"]
    dense = [c for c in on_fork if c.family in ("Qwen3.8-27B", "Qwen3-8B")]
    oss = [c for c in on_fork if c.family == "gpt-oss-20b"]
    assert dense and oss
    without = fit_store(dataclasses.replace(inputs, corpus=dense, version="t-gnd-29a"))
    with_oss = fit_store(dataclasses.replace(inputs, corpus=dense + oss, version="t-gnd-29b"))

    def terms(store, family):
        key = Key("sku:h100xm-80c-vgpu", f"record:{fork}", f"family:{family}")
        return {
            (e.kind, tuple(sorted(e.detail.items()))): round(e.value, 6)
            for e in store.entries
            if e.key == key and e.kind in ("decode.kappa", "prefill.mix")
        }

    for family in ("Qwen3.8-27B", "Qwen3-8B"):
        got = terms(without, family)
        assert ("prefill.mix", ()) in got, family
        assert got == terms(with_oss, family), family
