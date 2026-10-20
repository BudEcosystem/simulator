"""Fit grounding entries from a harness corpus (FRD-027 FR-GND-4, IMPLEMENTATION_PLAN 0.4).

The fit is deliberately simple and auditable -- every entry is a median of per-point inversions of the engine
model, so a reviewer can recompute any number from the corpus:

1. memory: per (record, family, format) the non-KV overhead at the reference settings, from
   ``fraction x usable - weights - measured KV pool``; pooled per-captured-sequence and per-batched-token
   coefficients and per-method speculative overheads by least squares;
2. prefill: ``prefill.kappa`` per family from TTFT at concurrency 1;
3. decode: ``decode.kappa`` per family and batch bucket, by inverting the closed-loop TPOT for each point (the
   mixed-prefill factor ``prefill.mix`` is chosen pooled to minimise the throughput and TPOT error);
4. precision ratios per (family, format, bucket) and speculative acceptance per (family, method, k, sampling,
   workload class), each relative to the base fit;
5. boot time (base, CUDA-graph capture per size, load bandwidth) pooled per record;
6. residuals: every point is replayed through the fitted store; an entry's residual is the p90 relative error
   of the points that used it (memory: absolute bytes);
7. ladder levels: class, pooled (model), Hopper architecture (hardware, FR-GND-7: inflated by the vGPU
   residual) and engine-minor copies.
"""

from __future__ import annotations

import datetime as _dt
import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..device import HardwareMode, catalogue
from ..engine.base import Workload
from ..engine.priors import GIB
from ..engine.record import EngineRecord
from ..engine.vllm import REF_CAPTURE, REF_CHUNK, VLLMEngine
from ..model import ModelFacts
from .corpus import PREFIX_SHARED_TOKENS, WARM_HIT, CorpusConfig, Point, parse_args
from .store import BootOutcome, Entry, GroundingStore, Key, OverrideStore, bucket_of

HARNESS_VERSION = "study-harness@2026-10-16"
VGPU_RESIDUAL = (
    0.05  # added to Hopper-architecture entries fitted on the H100XM-80C vGPU (FR-GND-7)
)
RESIDUAL_FLOOR = 0.03
#: Which measured quantity's replicate noise widens each residual entry (TTFT for every other kind).
NOISE_QUANTITIES = {
    "decode.kappa": ("tpot", "throughput"),
    "precision.ratio": ("tpot", "throughput"),
    "lora.ratio": ("tpot", "throughput"),
    "spec.acceptance": ("tpot",),
    "spec.acceptance_mean": ("throughput",),
    "band.tpot": ("tpot",),
    "band.throughput": ("throughput",),
}
#: A band is its stratum's own error (``engine.vllm.band_strata``) once the stratum has this many points; a term's
#: pooled residual carries its worst stratum everywhere (T-GND-19).
BAND_MIN_SAMPLES = 4
TIME_SETTINGS = {
    "batch.max_running",
    "batch.prefill_chunk_tokens",
    "context.max_tokens",
    "memory.device_fraction",
    "kv.dtype",
    "state.dtype",
    "prefix_cache.enabled",
    "parallel.tensor",
    "parallel.pipeline",
    "weights.format",
}


@dataclass
class FitInputs:
    corpus: Sequence[CorpusConfig]
    records: Mapping[str, EngineRecord]
    models: Mapping[Tuple[str, str], ModelFacts]  # (family, checkpoint format) -> facts
    drafts: Mapping[str, ModelFacts] = field(default_factory=dict)  # path substring -> facts
    exclude_families: Iterable[str] = ()
    version: str = "unversioned"
    run_id: str = "study-2026-10-15"


@dataclass
class Sample:
    cfg: CorpusConfig
    point: Point
    model: ModelFacts
    draft: Optional[ModelFacts]
    settings: Dict[str, Any]
    device: Any


# ====================================================================== helpers


def device_for(cfg: CorpusConfig):
    facts = catalogue.get(cfg.sku)
    if cfg.hardware_mode == "shared" and cfg.slice_mib:
        facts = facts.with_mode(
            HardwareMode(kind="shared", slice_bytes=int(cfg.slice_mib) * 1024**2)
        )
    return facts


def model_for(inputs: FitInputs, cfg: CorpusConfig) -> Optional[ModelFacts]:
    return inputs.models.get((cfg.family, cfg.checkpoint))


def draft_for(inputs: FitInputs, cfg: CorpusConfig) -> Optional[ModelFacts]:
    path = str(cfg.settings.get("spec.draft") or "")
    for needle, facts in inputs.drafts.items():
        if needle.lower() in path.lower():
            return facts
    return None


def predict_settings(cfg: CorpusConfig) -> Dict[str, Any]:
    settings = {k: v for k, v in cfg.settings.items() if k != "spec.draft"}
    if not settings.get("weights.online"):
        settings.pop("weights.format", None)  # the checkpoint's own format (the facts carry it)
    return settings


def workload_for(point: Point) -> Workload:
    return Workload(
        input_tokens=point.input_tokens,
        output_tokens=point.output_tokens,
        concurrency=point.concurrency,
        prefix_hit_rate=point.prefix_hit_rate,
        prefix_tokens=PREFIX_SHARED_TOKENS if point.point == "prefix" else 0,
        sampling=point.sampling,
        workload_class=point.workload_class,
        requests=point.num_prompts or None,
    )


def usable_point(cfg: CorpusConfig, point: Point) -> bool:
    return (
        not cfg.guarded
        and not cfg.invalid_points
        and cfg.startup.ready
        and point.failed == 0
        and point.completed > 0
        and point.tpot_ms > 0
        and point.output_tok_s > 0
    )


def conformal_p90(values: Sequence[float]) -> float:
    """The split-conformal 90% quantile: the ceil((n + 1) x 0.9)-th smallest value (the largest when n < 9), so a new
    point from the same stratum falls inside with probability >= 0.9. A plain p90 covers its own points and about
    90% of new ones, which A-05's out-of-sample >= 90% would miss by chance half the time (T-GND-19)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil((len(ordered) + 1) * 0.9)) - 1)
    return float(ordered[idx])


def p90(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(0.9 * len(ordered))) - 1))
    return float(ordered[idx])


#: Replicate cells with fewer samples than this carry no noise floor (too few to estimate a p90).
REPLICATE_MIN_SAMPLES = 10
#: A speculative point whose TPOT moves by more than this between no and full prefill interference (spec.mix 0 and 1)
#: grounds spec.mix, not the acceptance.
MIX_SENSITIVE = 0.10


QUANTITY_ATTRS = (("ttft", "ttft_ms"), ("tpot", "tpot_ms"), ("throughput", "output_tok_s"))


#: Flags that change no engine behaviour (logging, config loading for architectures vLLM implements natively).
NO_EFFECT_FLAGS = frozenset(
    {"trust-remote-code", "uvicorn-log-level", "disable-log-requests", "disable-log-stats", "served-model-name"}
)
#: Flags spelled out at their engine default (Bud's render writes them; a bare pod omits them).
DEFAULT_FLAGS = {"tensor-parallel-size": "1", "pipeline-parallel-size": "1"}


def replicate_ident(cfg: CorpusConfig) -> Tuple[Any, ...]:
    """What makes two configurations replicates: everything the corpus records about the deployment (family,
    checkpoint, record, hardware mode, slice, the argument list). A flag the predictor does not model is a real effect,
    not noise; only a flag that changes nothing (``NO_EFFECT_FLAGS``) or restates its default (``DEFAULT_FLAGS``) is
    left out of the identity."""
    flags = {
        key: str(value)
        for key, value in parse_args(cfg.args).items()
        if key not in NO_EFFECT_FLAGS and DEFAULT_FLAGS.get(key) != str(value)
    }
    return (
        cfg.family,
        cfg.checkpoint,
        cfg.record,
        cfg.hardware_mode,
        cfg.slice_mib,
        tuple(sorted(flags.items())),
    )


def point_ident(point: Point) -> Tuple[Any, ...]:
    """The same load point at the same cache state: a warm repeat hits nearly every prompt block (gpt-oss's TCS -r2/-r3
    at 98%), so it is not a replicate of its cold run."""
    return (
        point.point,
        point.concurrency,
        point.sampling,
        point.workload_class,
        point.input_tokens,
        (point.prefix_hit_rate or 0.0) >= WARM_HIT,
    )


def spread_key(cfg: CorpusConfig, point: Point) -> Tuple[Any, ...]:
    return (replicate_ident(cfg), point_ident(point))


def _replicates(corpus: Iterable[CorpusConfig]) -> Dict[Tuple[Any, ...], List[Point]]:
    """Points measured by at least two replicate configurations, by :func:`spread_key`."""
    groups: Dict[Tuple[Any, ...], List[CorpusConfig]] = {}
    for cfg in corpus:
        if cfg.guarded or cfg.invalid_points or not cfg.startup.ready:
            continue
        groups.setdefault(replicate_ident(cfg), []).append(cfg)
    out: Dict[Tuple[Any, ...], List[Point]] = {}
    for ident, cfgs in groups.items():
        if len(cfgs) < 2:
            continue
        for cfg in cfgs:
            for point in cfg.points:
                out.setdefault((ident, point_ident(point)), []).append(point)
    return {key: points for key, points in out.items() if len(points) >= 2}


def point_spread(corpus: Iterable[CorpusConfig]) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    """Each replicated point's spread per quantity, (max - min) / (max + min): the smallest error any one prediction
    reaches on every replicate (the extremes' harmonic mean reaches it on both). A point whose spread exceeds its ceiling
    is noise-limited (FRD-027 §16, the A-05 amendment): no predictor meets the ceiling on it."""
    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for key, points in _replicates(corpus).items():
        cell = {}
        for quantity, attr in QUANTITY_ATTRS:
            values = [getattr(p, attr) for p in points if getattr(p, attr) > 0]
            if len(values) >= 2:
                cell[quantity] = (max(values) - min(values)) / (max(values) + min(values))
        if cell:
            out[key] = cell
    return out


def replicate_noise(corpus: Iterable[CorpusConfig]) -> Dict[Tuple[str, str, str], float]:
    """Run-to-run noise: p90 of |run / median - 1| per (family, quantity, load bucket) over replicates
    (:func:`replicate_ident`, :func:`point_ident`).

    A fitted constant matches the run it was fitted on; a new run of the same deployment also differs by this much, so
    the out-of-sample band adds it to the fit residual (``fit_residuals``).
    """
    deviations: Dict[Tuple[str, str, str], List[float]] = {}
    for (ident, pident), points in _replicates(corpus).items():
        concurrency = pident[1]
        for quantity, attr in QUANTITY_ATTRS:
            values = [getattr(p, attr) for p in points if getattr(p, attr) > 0]
            if len(values) < 2:
                continue
            mid = statistics.median(values)
            cell = (ident[0], quantity, bucket_of(concurrency))
            deviations.setdefault(cell, []).extend(abs(v / mid - 1) for v in values)
    return {
        cell: p90(devs) for cell, devs in deviations.items() if len(devs) >= REPLICATE_MIN_SAMPLES
    }


def parallel_entries(corpus: Iterable[CorpusConfig]) -> List[Entry]:
    """The parallelism beyond one GPU each family was measured at, per (SKU, record): the engine reports a TP or PP > 1
    prediction ungrounded where none is (FRD-027 §16, A-10). The study ran TP=1 only, so there are none yet."""
    counts: Dict[Tuple[str, str, str, int, int], int] = {}
    for cfg in corpus:
        if cfg.guarded or not cfg.startup.ready or not cfg.points:
            continue
        tp = int(cfg.settings.get("parallel.tensor") or 1)
        pp = int(cfg.settings.get("parallel.pipeline") or 1)
        if tp * pp > 1:
            cell = (cfg.sku, cfg.record, cfg.family, tp, pp)
            counts[cell] = counts.get(cell, 0) + 1
    return [
        Entry(
            kind="parallel.measured",
            key=Key(f"sku:{sku}", f"record:{record}", f"family:{family}"),
            detail={"tp": tp, "pp": pp},
            value=1.0,
            residual=0.0,
            samples=n,
            provenance={"fit": "measured-parallelism"},
        )
        for (sku, record, family, tp, pp), n in sorted(counts.items())
    ]


def band_error(predicted: float, measured: float) -> float:
    """How wide a band around ``predicted`` must be to hold ``measured`` (|measured / predicted - 1|)."""
    return abs(measured / predicted - 1) if predicted > 0 else float("inf")


def _spec_keys(s: Sample, k: int) -> List[Tuple[Any, ...]]:
    """The acceptance keys a speculative point grounds: (family, method, k, sampling, workload) and, for a draft
    model, the same with the draft's identity."""
    from ..engine.vllm import _draft_id

    base = (s.cfg.family, str(s.settings["spec.method"]), k, s.point.sampling, s.point.workload_class)
    did = _draft_id(s.draft) if s.settings.get("spec.method") in ("eagle3", "draft") else None
    return [base, (*base, did)] if did else [base]


def _spec_detail(key: Tuple[Any, ...]) -> Dict[str, Any]:
    detail: Dict[str, Any] = {"method": key[1], "k": key[2], "sampling": key[3], "workload": key[4]}
    if len(key) > 5:
        detail["draft"] = key[5]
    return detail


#: A workload shape needs this many points to get its own speculative TTFT entry; below it the pooled entry serves.
SHAPE_MIN_SAMPLES = 4
#: A build's own speculative TTFT terms (prefill ratio, wave survival) need this many configurations of the method: from
#: two (bud-vllm 0.10.0's 27B MTP3 with the cache on and off, at online FP8) spec.prefill came out at 2.12 against the
#: family's 1.21 and predicted a bf16 MTP2 deployment's long-prompt TTFT at 2.5x (T-GND-22). Below it the engine
#: family's serve.
TERM_MIN_CONFIGS = 3
#: The prefill interference's grid, and the base points a family needs for its own value (T-GND-29). Shared by every
#: family on a build, round 9's three gpt-oss TCS repeats flipped the fork's from 1.25 to 1.0 and the fork 27B's decode
#: kappas re-solved around it.
MIX_GRID = (0.25, 0.5, 0.75, 1.0, 1.25)
MIX_MIN_SAMPLES = 10


def shape_of(record: EngineRecord, s: Sample) -> str:
    """The point's workload shape on its engine (a prefix load with the prefix cache off is a long or chat one)."""
    r = VLLMEngine(record, GroundingStore()).resolve(s.settings, s.model, s.device)
    return workload_for(s.point).shape_for(r.prefix_cache)


def _by_shape(record: EngineRecord, rows: Sequence[Sample]) -> Dict[str, List[Sample]]:
    out: Dict[str, List[Sample]] = {}
    for s in rows:
        out.setdefault(shape_of(record, s), []).append(s)
    return out


def _median(values: Sequence[float], default: float) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(statistics.median(vals)) if vals else float(default)


def kind_of(cfg: CorpusConfig) -> str:
    s = cfg.settings
    if s.get("spec.method"):
        return "spec"
    if s.get("lora.max_adapters"):
        return "lora"
    if (
        s.get("weights.online")
        or cfg.checkpoint not in ("bf16", "fp16")
        and cfg.family != "gpt-oss-20b"
    ):
        return "precision"
    if cfg.family == "gpt-oss-20b" and cfg.checkpoint != "mxfp4":
        return "precision"
    if cfg.hardware_mode == "shared":
        return "shared"
    return "base"


def fmt_of(cfg: CorpusConfig) -> str:
    return str(cfg.settings.get("weights.format") or cfg.checkpoint)


# ====================================================================== the fitter


class Fitter:
    def __init__(self, inputs: FitInputs, noise_corpus: Optional[Sequence[CorpusConfig]] = None):
        self.inputs = inputs
        # every run, replicates apart: the noise floors and spreads (fit_store fits the collapsed corpus)
        self.noise_corpus = list(noise_corpus if noise_corpus is not None else inputs.corpus)
        self.excluded = set(inputs.exclude_families)
        self._noise: Optional[Dict[Tuple[str, str, str], float]] = None
        self._spread: Optional[Dict[Tuple[Any, ...], Dict[str, float]]] = None
        self.today = _dt.date(2026, 10, 18).isoformat()
        self.entries: Dict[Tuple[str, Key, Tuple[Tuple[str, Any], ...]], Entry] = {}
        self._store_cache: Optional[GroundingStore] = None
        self.residual_log: Dict[Tuple[str, Key, Tuple[Tuple[str, Any], ...]], List[float]] = {}

    def replicate_noise(self) -> Dict[Tuple[str, str, str], float]:
        if self._noise is None:
            self._noise = replicate_noise(self.noise_corpus)
        return self._noise

    def point_spread(self) -> Dict[Tuple[Any, ...], Dict[str, float]]:
        if self._spread is None:
            self._spread = point_spread(self.noise_corpus)
        return self._spread

    # ------------------------------------------------------------------ entry bookkeeping

    def key(self, sku: str, record: str, family: str) -> Key:
        return Key(f"sku:{sku}", f"record:{record}", f"family:{family}")

    def put(
        self,
        kind: str,
        key: Key,
        detail: Mapping[str, Any],
        value: float,
        *,
        samples: int,
        residual: float = 0.0,
    ) -> None:
        dkey = tuple(sorted(detail.items()))
        self._store_cache = None
        self.entries[(kind, key, dkey)] = Entry(
            kind=kind,
            key=key,
            detail=dict(detail),
            value=float(value),
            residual=float(residual),
            samples=int(samples),
            provenance={
                "run_id": self.inputs.run_id,
                "harness": HARNESS_VERSION,
                "date": self.today,
                "fit": "median-inversion",
            },
        )

    def store(self) -> GroundingStore:
        # cached until an entry is written: every solve and grid trial reads it, and building the index is the cost
        if self._store_cache is None:
            self._store_cache = GroundingStore(
                list(self.entries.values()), version=self.inputs.version
            )
        return self._store_cache

    def supported(self, cfg: CorpusConfig) -> bool:
        """A configuration whose settings the record does not list is evidence for a future record, not this one."""
        try:
            VLLMEngine(self.inputs.records[cfg.record], GroundingStore()).check_supported(
                predict_settings(cfg)
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    def samples(self, record: str) -> List[Sample]:
        out = []
        for cfg in self.inputs.corpus:
            if cfg.record != record or cfg.family in self.excluded:
                continue
            model = model_for(self.inputs, cfg)
            if model is None or not self.supported(cfg):
                continue
            for point in cfg.points:
                if usable_point(cfg, point):
                    out.append(
                        Sample(
                            cfg,
                            point,
                            model,
                            draft_for(self.inputs, cfg),
                            predict_settings(cfg),
                            device_for(cfg),
                        )
                    )
        return out

    # ------------------------------------------------------------------ 1. memory

    def fit_memory(self, record_id: str) -> None:
        record = self.inputs.records[record_id]
        rows = []
        for cfg in self.inputs.corpus:
            if (
                cfg.record != record_id
                or cfg.family in self.excluded
                or not cfg.startup.ready
                or not cfg.startup.kv_gib
            ):
                continue
            if cfg.guarded or cfg.hardware_mode == "shared" or not self.supported(cfg):
                continue
            model = model_for(self.inputs, cfg)
            if model is None:
                continue
            engine = VLLMEngine(record, GroundingStore())
            settings = predict_settings(cfg)
            device = device_for(cfg)
            r = engine.resolve(settings, model, device)
            weights = engine._weights(r, model, draft_for(self.inputs, cfg))
            measured = r.fraction * device.usable_memory_bytes - weights - cfg.startup.kv_gib * GIB
            measured -= (
                4.0 * model.vocab_size * (min(r.max_running, 1024) - REF_CAPTURE)
            )  # sampler term (physics)
            captured = min(r.max_running, record.capture_max())
            rows.append((cfg, model, r, measured, captured))
        if not rows:
            return
        groups = sorted({(c.family, fmt_of(c)) for c, *_ in rows})
        specs = sorted(
            {
                (c.family, c.settings["spec.method"])
                for c, *_ in rows
                if c.settings.get("spec.method")
            }
        )
        cols = len(groups) + len(specs) + 2
        a = np.zeros((len(rows), cols))
        b = np.zeros(len(rows))
        for i, (cfg, model, r, measured, captured) in enumerate(rows):
            a[i, groups.index((cfg.family, fmt_of(cfg)))] = 1.0
            if cfg.settings.get("spec.method"):
                a[i, len(groups) + specs.index((cfg.family, cfg.settings["spec.method"]))] = (
                    1.0 + 0.1 * max(r.spec_k - 1, 0)
                )
            a[i, -2] = model.hidden_size * (captured - REF_CAPTURE)
            a[i, -1] = (r.chunk - REF_CHUNK) * (model.hidden_size * 5)
            b[i] = measured
        # scale columns for conditioning
        scale = np.maximum(np.abs(a).max(axis=0), 1e-12)
        sol, *_ = np.linalg.lstsq(a / scale, b, rcond=None)
        sol = sol / scale
        capture = float(np.clip(sol[-2], 0.0, 5e-6 * GIB))
        chunk = float(np.clip(sol[-1], 0.0, 1e-4))
        sku = rows[0][0].sku
        for (family, fmt), value in zip(groups, sol[: len(groups)]):
            n = sum(1 for c, *_ in rows if c.family == family and fmt_of(c) == fmt)
            self.put(
                "memory.overhead",
                self.key(sku, record_id, family),
                {"format": fmt},
                float(value),
                samples=n,
            )
        for (family, method), value in zip(specs, sol[len(groups) : len(groups) + len(specs)]):
            n = sum(
                1
                for c, *_ in rows
                if c.family == family and c.settings.get("spec.method") == method
            )
            self.put(
                "memory.spec",
                self.key(sku, record_id, family),
                {"method": method},
                max(float(value), 0.0),
                samples=n,
            )
        pooled = Key(f"sku:{sku}", f"record:{record_id}", "pooled")
        self.put("memory.capture", pooled, {}, capture, samples=len(rows))
        self.put("memory.chunk", pooled, {}, chunk, samples=len(rows))

    # ------------------------------------------------------------------ 2-4. time

    def _engine(
        self, record_id: str, overrides: Optional[Mapping[str, float]] = None
    ) -> VLLMEngine:
        store = self.store()
        if overrides:
            store = OverrideStore(store, overrides)
        return VLLMEngine(self.inputs.records[record_id], store)

    def _solve(
        self, record_id: str, s: Sample, kind: str, quantity: str, lo: float, hi: float
    ) -> Optional[float]:
        """The value of grounding ``kind`` for which the predicted ``quantity`` of point ``s`` equals the measured
        one (bisection; the quantity is monotonic in each kind). None when the target is out of range.
        """
        target = {
            "tpot": s.point.tpot_ms,
            "ttft": s.point.ttft_ms,
            "throughput": s.point.output_tok_s,
        }[quantity]
        if target <= 0:
            return None
        w = workload_for(s.point)

        def value(x: float) -> Optional[float]:
            try:
                p = self._engine(record_id, {kind: x}).predict(
                    s.settings, w, s.model, s.device, draft=s.draft
                )
            except Exception:  # noqa: BLE001
                return None
            return {
                "tpot": p.tpot_ms.value,
                "ttft": p.ttft_ms.value,
                "throughput": p.throughput_tok_s.value,
            }[quantity]

        f_lo, f_hi = value(lo), value(hi)
        if f_lo is None or f_hi is None:
            return None
        increasing = f_hi > f_lo
        if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
            return None
        a, b = lo, hi
        for _ in range(40):
            mid = math.sqrt(a * b) if a > 0 else (a + b) / 2
            f = value(mid)
            if f is None:
                return None
            if (f < target) == increasing:
                a = mid
            else:
                b = mid
        return math.sqrt(a * b) if a > 0 else (a + b) / 2

    def _running(self, record_id: str, s: Sample) -> Tuple[int, bool, float]:
        p = self._engine(record_id).predict(
            s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
        )
        new = s.point.input_tokens * (1.0 - (s.point.prefix_hit_rate or 0.0))
        return p.running, p.queueing, new

    def _mix_sensitive(self, record_id: str, s: Sample) -> bool:
        """Whether the point's TPOT carries other requests' prefills (c > 1 with prompts large against the output)."""
        if s.point.concurrency <= 1:
            return False
        w = workload_for(s.point)
        try:
            t0, t1 = (
                self._engine(record_id, {"spec.mix": v})
                .predict(s.settings, w, s.model, s.device, draft=s.draft)
                .tpot_ms.value
                for v in (0.0, 1.0)
            )
        except Exception:  # noqa: BLE001
            return False
        return bool(t0) and abs(t1 / t0 - 1.0) > MIX_SENSITIVE

    def fit_prefill(self, record_id: str, samples: List[Sample]) -> None:
        from ..engine.vllm import prefill_bucket

        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, str], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != "base":
                continue
            m, queueing, tokens = self._running(record_id, s)
            if queueing:
                continue
            if workload_for(s.point).arrival_model != "wave":
                # steady (real-text) TTFTs mix queueing and request handling into the prefill; they ground
                # api.tau instead (fit_api), so the prefill efficiency comes from clean synthetic points only
                continue
            k = self._solve(record_id, s, "prefill.kappa", "ttft", 0.05, 40.0)
            if k is not None:
                by_key.setdefault((s.cfg.family, prefill_bucket(tokens), bucket_of(m)), []).append(
                    k
                )
        for (family, tbucket, bbucket), values in by_key.items():
            self.put(
                "prefill.kappa",
                self.key(sku, record_id, family),
                {"tokens": tbucket, "bucket": bbucket},
                _median(values, 1.6),
                samples=len(values),
            )

    def fit_api(self, record_id: str, samples: List[Sample]) -> None:
        """Request handling around TTFT (tokenize, schedule, detokenize the first token), per family, from the
        steady real-text points whose prefill is short."""
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_family: Dict[str, List[float]] = {}
        for s in samples:
            if (
                kind_of(s.cfg) != "base"
                or workload_for(s.point).arrival_model == "wave"
                or s.point.concurrency != 1
            ):
                continue
            v = self._solve(record_id, s, "api.tau", "ttft", 0.5, 400.0)
            if v is not None:
                by_family.setdefault(s.cfg.family, []).append(v)
        for family, values in by_family.items():
            self.put(
                "api.tau",
                self.key(sku, record_id, family),
                {},
                _median(values, 12.0),
                samples=len(values),
            )

    def fit_queue(self, record_id: str, samples: List[Sample]) -> None:
        """Decode steps a steady arrival waits for, per (family, batch bucket) and pooled per family, from base runs'
        TTFT at c > 1. Real text queues longer at c32 than at c8 (0.30 8B: 146 ms against ~42 ms predicted from one
        pooled value), and one value let the c32 miss widen every c8 band (T-GND-21). A key needs BAND_MIN_SAMPLES
        points: a median of two or three, whose p90 residual is their largest error, does not ground it, and the
        engine family's serves instead (fork 8B: real c8 and c32 only). Precision runs stay out: with them the 0.30 8B's
        queue values moved and its speculative TTFT MAPE went from 38% to 65% (trial, 10-19)."""
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, Optional[str]], List[float]] = {}
        for s in samples:
            if (
                kind_of(s.cfg) != "base"
                or workload_for(s.point).arrival_model == "wave"
                or s.point.concurrency <= 1
            ):
                continue
            m, queueing, _ = self._running(record_id, s)
            if queueing:
                continue
            v = self._solve(record_id, s, "arrival.queue", "ttft", 0.0, 50.0)
            if v is not None:
                by_key.setdefault((s.cfg.family, bucket_of(m)), []).append(v)
                by_key.setdefault((s.cfg.family, None), []).append(v)
        for (family, bucket), values in by_key.items():
            if len(values) < BAND_MIN_SAMPLES:
                continue
            self.put(
                "arrival.queue",
                self.key(sku, record_id, family),
                {} if bucket is None else {"bucket": bucket},
                _median(values, 0.5),
                samples=len(values),
            )

    def fit_decode(self, record_id: str, samples: List[Sample]) -> None:
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        base = [s for s in samples if kind_of(s.cfg) == "base"]
        if not base:
            # nothing to fit: an empty grid search would write its first value (bud-vllm 0.10.0: prefill.mix 0.25 with
            # no samples, which then pinned spec.mix and the mean acceptance at their bounds)
            return
        # the build's pooled interference, for a family without enough base points of its own and for one the store
        # lacks; then each family's own, so another family's runs never move this one's decode terms (T-GND-29)
        self._fit_mix(record_id, sku, base, Key(f"sku:{sku}", f"record:{record_id}", "pooled"))
        by_family: Dict[str, List[Sample]] = {}
        for s in base:
            by_family.setdefault(s.cfg.family, []).append(s)
        for family, rows in sorted(by_family.items()):
            if len(rows) >= MIX_MIN_SAMPLES:
                self._fit_mix(record_id, sku, rows, self.key(sku, record_id, family))

    def _fit_mix(self, record_id: str, sku: str, base: List[Sample], key: Key) -> None:
        best = None
        for mix in MIX_GRID:
            self.put("prefill.mix", key, {}, mix, samples=len(base))
            self._fit_kappas(record_id, sku, base)
            err = self._error(record_id, base)
            if best is None or err < best[0]:
                best = (err, mix)
        self.put("prefill.mix", key, {}, best[1], samples=len(base))
        self._fit_kappas(record_id, sku, base)

    def _fit_kappas(self, record_id: str, sku: str, samples: List[Sample]) -> None:
        by_key: Dict[Tuple[str, str], List[float]] = {}
        for s in samples:
            m, _, _ = self._running(record_id, s)
            k = self._solve(record_id, s, "decode.kappa", "tpot", 0.05, 30.0)
            if k is not None:
                by_key.setdefault((s.cfg.family, bucket_of(m)), []).append(k)
        for (family, bucket), values in by_key.items():
            self.put(
                "decode.kappa",
                self.key(sku, record_id, family),
                {"bucket": bucket},
                _median(values, 1.1),
                samples=len(values),
            )

    def _error(self, record_id: str, samples: List[Sample]) -> float:
        engine = self._engine(record_id)
        errs = []
        for s in samples:
            try:
                p = engine.predict(
                    s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
                )
            except Exception:  # noqa: BLE001
                continue
            errs.append(abs(math.log(p.throughput_tok_s.value / s.point.output_tok_s)))
            errs.append(abs(math.log(p.tpot_ms.value / s.point.tpot_ms)))
        return float(np.mean(errs)) if errs else float("inf")

    def fit_precision(self, record_id: str, samples: List[Sample]) -> None:
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, str, str], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != "precision":
                continue
            m, _, _ = self._running(record_id, s)
            ratio = self._solve(record_id, s, "precision.ratio", "tpot", 0.1, 10.0)
            if ratio is not None:
                by_key.setdefault((s.cfg.family, fmt_of(s.cfg), bucket_of(m)), []).append(ratio)
        for (family, fmt, bucket), values in by_key.items():
            self.put(
                "precision.ratio",
                self.key(sku, record_id, family),
                {"format": fmt, "bucket": bucket},
                _median(values, 1.0),
                samples=len(values),
            )

    def fit_prefill_ratio(self, record_id: str, samples: List[Sample], *, kind: str) -> None:
        """Prefill efficiency of a weight format (kind='precision') or with a speculative method (kind='spec'),
        relative to the base prefill fit, from the TTFT of points that are not queueing."""
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        entry = "precision.prefill" if kind == "precision" else "spec.prefill"
        if kind == "spec":
            self._fit_spec_prefill(record_id, samples, sku)
            return
        by_key: Dict[Tuple[str, str], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != kind:
                continue
            m, queueing, _ = self._running(record_id, s)
            if queueing:
                continue
            v = self._solve(record_id, s, entry, "ttft", 0.1, 10.0)
            if v is None:
                continue
            detail = fmt_of(s.cfg) if kind == "precision" else str(s.settings["spec.method"])
            by_key.setdefault((s.cfg.family, detail), []).append(v)
        for (family, detail), values in by_key.items():
            if not self._has_base(sku, record_id, family):
                continue
            d = {"format": detail} if kind == "precision" else {"method": detail}
            self.put(
                entry,
                self.key(sku, record_id, family),
                d,
                _median(values, 1.0),
                samples=len(values),
            )

    def fit_lora(self, record_id: str, samples: List[Sample]) -> None:
        values = []
        for s in samples:
            if kind_of(s.cfg) != "lora":
                continue
            v = self._solve(record_id, s, "lora.ratio", "tpot", 0.5, 3.0)
            if v is not None:
                values.append(v)
        if values:
            sku = samples[0].cfg.sku
            self.put(
                "lora.ratio",
                Key(f"sku:{sku}", f"record:{record_id}", "pooled"),
                {},
                _median(values, 1.02),
                samples=len(values),
            )

    def fit_drain(self, record_id: str, samples: List[Sample]) -> None:
        """Rounds lost to the tail of a fixed-count run of steady, variable-length requests, per (family, workload
        class): solved on the throughput of such runs without speculation at c > 1, whose TPOT the decode fit already
        matches."""
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, str], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) not in ("base", "shared") or s.point.concurrency <= 1:
                continue
            if workload_for(s.point).arrival_model != "steady":
                continue
            m, _, _ = self._running(record_id, s)
            if m <= 1:
                continue
            v = self._solve(record_id, s, "arrival.drain", "throughput", 0.0, 10.0)
            if v is None:
                continue
            by_key.setdefault((s.cfg.family, s.point.workload_class), []).append(v)
        for (family, workload), values in by_key.items():
            self.put(
                "arrival.drain",
                self.key(sku, record_id, family),
                {"workload": workload},
                _median(values, 0.0),
                samples=len(values),
            )

    def fit_sync(self, record_id: str, samples: List[Sample]) -> None:
        """How much of the benchmark's synchronized wave survives speculative decoding, per (family, method): the
        median inversion TPOT and throughput use. Per (family, method, workload shape), the TTFT value: the one whose
        p90 TTFT band is narrowest, since that band is what A-05 grades."""
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, str], List[float]] = {}
        rows: Dict[Tuple[str, str], List[Sample]] = {}
        for s in samples:
            if (
                kind_of(s.cfg) != "spec"
                or workload_for(s.point).arrival_model != "wave"
                or s.point.concurrency <= 1
            ):
                continue
            _, queueing, _ = self._running(record_id, s)
            if queueing:
                continue
            v = self._solve(record_id, s, "arrival.sync", "ttft", 0.0, 1.0)
            if v is None:
                lo = (
                    self._engine(record_id, {"arrival.sync": 0.0})
                    .predict(s.settings, workload_for(s.point), s.model, s.device, draft=s.draft)
                    .ttft_ms.value
                )
                v = 0.0 if s.point.ttft_ms <= lo else 1.0
            group = (s.cfg.family, str(s.settings["spec.method"]))
            by_key.setdefault(group, []).append(v)
            rows.setdefault(group, []).append(s)
        for (family, method), values in by_key.items():
            if not self._has_base(sku, record_id, family):
                continue
            if len({replicate_ident(s.cfg) for s in rows[(family, method)]}) < TERM_MIN_CONFIGS:
                continue
            key = self.key(sku, record_id, family)
            pooled = _median(values, 0.5)
            self.put("arrival.sync", key, {"method": method}, pooled, samples=len(values))
            self._fit_shapes(
                record_id, rows[(family, method)], "arrival.sync", key, {"method": method},
                [i / 20 for i in range(21)], pooled,
            )

    def _has_base(self, sku: str, record_id: str, family: str) -> bool:
        """Whether the build has its own base prefill fit for the family. The prefill ratios and the wave's survival are
        relative to it; on a build without base runs they would absorb the gap to the family level's base terms
        instead (bud-vllm 0.10.0: spec.prefill 3.27 against 1.21), so such a build keeps the family's (T-GND-20)."""
        key = self.key(sku, record_id, family)
        return any(kind == "prefill.kappa" and k == key for kind, k, _ in self.entries)

    def _fit_spec_prefill(self, record_id: str, samples: List[Sample], sku: str) -> None:
        """A speculative method's prefill cost relative to the base fit, per (family, method): the median inversion
        over c = 1 points. Per (family, method, workload shape), the TTFT value over the points no wave sets (c = 1 and
        steady arrivals), at the narrowest p90 TTFT band."""
        by_key: Dict[Tuple[str, str], List[float]] = {}
        rows: Dict[Tuple[str, str], List[Sample]] = {}
        for s in samples:
            if kind_of(s.cfg) != "spec" or s.point.ttft_ms <= 0:
                continue
            m, queueing, _ = self._running(record_id, s)
            if queueing or (m > 1 and workload_for(s.point).arrival_model == "wave"):
                continue  # a c>1 wave grounds arrival.sync instead
            group = (s.cfg.family, str(s.settings["spec.method"]))
            rows.setdefault(group, []).append(s)
            if m > 1:
                continue
            v = self._solve(record_id, s, "spec.prefill", "ttft", 0.1, 10.0)
            if v is not None:
                by_key.setdefault(group, []).append(v)
        for (family, method), values in by_key.items():
            if not self._has_base(sku, record_id, family):
                continue
            if len({replicate_ident(s.cfg) for s in rows[(family, method)]}) < TERM_MIN_CONFIGS:
                continue
            key = self.key(sku, record_id, family)
            pooled = _median(values, 1.0)
            self.put("spec.prefill", key, {"method": method}, pooled, samples=len(values))
            self._fit_shapes(
                record_id, rows[(family, method)], "spec.prefill", key, {"method": method},
                [0.4 * (3.0 / 0.4) ** (i / 30) for i in range(31)], pooled,
            )

    def _fit_shapes(
        self,
        record_id: str,
        rows: List[Sample],
        kind: str,
        key: Key,
        detail: Mapping[str, Any],
        grid: Sequence[float],
        pooled: float,
    ) -> None:
        for shape, group in _by_shape(self.inputs.records[record_id], rows).items():
            if len(group) < SHAPE_MIN_SAMPLES:
                continue
            value = self._minimax(record_id, group, kind, [*grid, pooled])
            if value is not None:
                self.put(kind, key, {**detail, "shape": shape}, value, samples=len(group))

    def _minimax(
        self, record_id: str, rows: List[Sample], kind: str, grid: Sequence[float]
    ) -> Optional[float]:
        """The grid value of ``kind`` with the least p90 |log(measured / predicted)| TTFT over ``rows`` (ties: the
        first)."""
        best: Optional[Tuple[float, float]] = None
        base = self.store()
        record = self.inputs.records[record_id]
        for value in grid:
            engine = VLLMEngine(record, OverrideStore(base, {kind: value}))
            errs = []
            for s in rows:
                try:
                    pred = engine.predict(
                        s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
                    )
                except Exception:  # noqa: BLE001
                    continue
                # symmetric in log space: the band error alone favours over-prediction, which is bounded by 1
                errs.append(abs(math.log(max(s.point.ttft_ms, 1e-9) / max(pred.ttft_ms.value, 1e-9))))
            if not errs:
                continue
            err = p90(errs)
            if best is None or err < best[0] - 1e-9:
                best = (err, value)
        return None if best is None else best[1]

    def fit_spec_mix(self, record_id: str, samples: List[Sample]) -> None:
        """Prefill interference on TPOT with a speculative method, per (family, method, batch bucket) and pooled per
        method: solved on TPOT at the points that carry it, with the acceptance grounded where it does not. The study's 27B MTP runs carry 20-60% less
        than the base fit predicts at chat c16, long c8 and prefix c16 (2026-10-18).
        """
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        by_key: Dict[Tuple[str, str, Optional[str]], List[float]] = {}
        configs: Dict[Tuple[str, str], set] = {}
        for s in samples:
            if kind_of(s.cfg) != "spec" or not self._mix_sensitive(record_id, s):
                continue
            _, queueing, _ = self._running(record_id, s)
            if queueing:
                continue
            m, _, _ = self._running(record_id, s)
            v = self._solve(record_id, s, "spec.mix", "tpot", 0.0, 2.0)
            if v is None:
                t0, t2 = (
                    self._engine(record_id, {"spec.mix": x})
                    .predict(s.settings, workload_for(s.point), s.model, s.device, draft=s.draft)
                    .tpot_ms.value
                    for x in (0.0, 2.0)
                )
                v = 0.0 if abs(s.point.tpot_ms - t0) <= abs(s.point.tpot_ms - t2) else 2.0
            method = str(s.settings["spec.method"])
            by_key.setdefault((s.cfg.family, method, bucket_of(m)), []).append(v)
            by_key.setdefault((s.cfg.family, method, None), []).append(v)
            configs.setdefault((s.cfg.family, method), set()).add(replicate_ident(s.cfg))  # replicates count once
        for (family, method, bucket), values in by_key.items():
            if len(configs.get((family, method), ())) < TERM_MIN_CONFIGS:
                # two configurations pinned bud-vllm 0.10.0's 27B MTP interference at the solver's bound (2.0)
                continue
            self.put(
                "spec.mix",
                self.key(sku, record_id, family),
                {"method": method} if bucket is None else {"method": method, "bucket": bucket},
                _median(values, 1.0),
                samples=len(values),
            )

    def fit_spec(self, record_id: str, samples: List[Sample]) -> None:
        """Acceptance per (family, method, k, sampling, workload) from TPOT, and its mean from throughput; for a draft
        model also per draft (a draft's acceptance is its own: the 8B's RedHat and AngelSlim drafts sit 67% and -20%
        off one pooled value at chat c1, k=3).

        TPOT at c > 1 also carries the other requests' prefills, so the median acceptance is taken over the points
        that interference cannot move (c = 1, decode-heavy loads) and falls back to every point only for a key that
        has none of them; spec.mix is grounded on the rest. The mean keeps every point: fitted on the same subset it
        moved throughput MAPE from 12.8% to 13.7% over the corpus (2026-10-18).
        """
        sku = samples[0].cfg.sku if samples else "h100xm-80c-vgpu"
        free: Dict[Tuple[Any, ...], List[float]] = {}
        waved: Dict[Tuple[Any, ...], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != "spec":
                continue
            k = int(s.settings.get("spec.tokens") or 1)
            alpha = self._solve(record_id, s, "spec.acceptance", "tpot", 0.0, float(k))
            if alpha is None:
                # out of range: slower than no acceptance (a net loss) -> 0; faster than all accepted -> k
                try:
                    zero = (
                        self._engine(record_id, {"spec.acceptance": 0.0})
                        .predict(
                            s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
                        )
                        .tpot_ms.value
                    )
                except Exception:  # noqa: BLE001
                    continue
                alpha = 0.0 if s.point.tpot_ms >= zero else float(k)
            target = waved if self._mix_sensitive(record_id, s) else free
            for key in _spec_keys(s, k):
                target.setdefault(key, []).append(min(alpha, float(k)))
        by_key = {key: free.get(key) or waved[key] for key in set(free) | set(waved)}
        means: Dict[Tuple[Any, ...], List[float]] = {}
        means_by_bucket: Dict[Tuple[Tuple[Any, ...], str], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != "spec":
                continue
            k = int(s.settings.get("spec.tokens") or 1)
            keys = [key for key in _spec_keys(s, k) if key in by_key]
            if not keys:
                continue
            mean = self._solve(record_id, s, "spec.acceptance_mean", "throughput", 0.0, float(k))
            m, _, _ = self._running(record_id, s)
            for key in keys:
                means.setdefault(key, []).append(mean if mean is not None else 0.0)
                means_by_bucket.setdefault((key, bucket_of(m)), []).append(
                    mean if mean is not None else 0.0
                )
        for key, values in by_key.items():
            family, detail = key[0], _spec_detail(key)
            self.put(
                "spec.acceptance",
                self.key(sku, record_id, family),
                detail,
                _median(values, 1.0),
                samples=len(values),
            )
        for key, values in means.items():
            self.put(
                "spec.acceptance_mean",
                self.key(sku, record_id, key[0]),
                _spec_detail(key),
                _median(values, 0.85),
                samples=len(values),
            )
        for (key, bucket), values in means_by_bucket.items():
            self.put(
                "spec.acceptance_mean",
                self.key(sku, record_id, key[0]),
                {**_spec_detail(key), "bucket": bucket},
                _median(values, 0.85),
                samples=len(values),
            )

    def fit_spec_scale(self, record_id: str, samples: List[Sample]) -> None:
        """The reported TPOT's bias per stratum (method[, draft], workload shape, batch bucket): the median measured /
        predicted TPOT of the stratum's points, once it has ``BAND_MIN_SAMPLES``. A benchmark's per-request TPOT carries
        the other requests' prefill stalls, which the decode cycle (and so throughput) does not: the 8B's EAGLE3 TPOT
        ran 1.25x the prediction at chat c256 and 1.6x at long c32 with its throughput within 1-37% (T-GND-19). It
        scales the reported TPOT only.
        """
        engine = self._engine(record_id)
        ratios: Dict[Tuple[Key, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        for s in samples:
            if kind_of(s.cfg) != "spec" or s.point.tpot_ms <= 0:
                continue
            try:
                pred = engine.predict(
                    s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
                )
            except Exception:  # noqa: BLE001
                continue
            if pred.tpot_ms.value <= 0:
                continue
            k = int(s.settings.get("spec.tokens") or 1)
            stratum = {
                "method": s.settings["spec.method"],
                "shape": shape_of(engine.record, s),
                "bucket": bucket_of(pred.running),
            }
            drafted = _spec_keys(s, k)
            strata = [stratum] + ([{**stratum, "draft": drafted[1][5]}] if len(drafted) > 1 else [])
            fam_key = self.key(s.cfg.sku, record_id, s.cfg.family)
            for st in strata:
                ratios.setdefault((fam_key, tuple(sorted(st.items()))), []).append(
                    s.point.tpot_ms / pred.tpot_ms.value
                )
        for (key, dkey), values in ratios.items():
            if len(values) >= BAND_MIN_SAMPLES:
                self.put("spec.scale", key, dict(dkey), _median(values, 1.0), samples=len(values))

    def fit_boot(self, record_id: str) -> None:
        """Boot time (FR-ENG-7) per (family, kind): the median harness ``api_up`` of base, precision-converted and
        speculative configurations; the residual is the p90 relative error of that median over its runs.
        """
        from ..engine.vllm import boot_kind

        record = self.inputs.records[record_id]
        groups: Dict[Tuple[str, str], List[float]] = {}
        for cfg in self.inputs.corpus:
            if (
                cfg.record != record_id
                or cfg.family in self.excluded
                or not cfg.startup.ready
                or not cfg.startup.api_up_s
            ):
                continue
            if cfg.guarded or not self.supported(cfg):
                continue
            model = model_for(self.inputs, cfg)
            if model is None:
                continue
            r = VLLMEngine(record, GroundingStore()).resolve(
                predict_settings(cfg), model, device_for(cfg)
            )
            groups.setdefault((cfg.family, boot_kind(r, model)), []).append(
                float(cfg.startup.api_up_s)
            )
        sku = self.inputs.corpus[0].sku if self.inputs.corpus else "h100xm-80c-vgpu"
        for (family, kind), values in groups.items():
            med = _median(values, 120.0)
            # |predicted / measured - 1|, not the band-consistent error: this band only sets the fallback's per-rung
            # budget (3x its top), whose margin assumes it; a slow outlier (1,238 s against a 240 s median) would
            # otherwise give a residual of 4.2 and an hour-long budget
            residual = p90([abs(med / v - 1) for v in values if v > 0]) if len(values) > 1 else 0.25
            self.put(
                "boot.time",
                self.key(sku, record_id, family),
                {"kind": kind},
                med,
                samples=len(values),
                residual=max(residual, RESIDUAL_FLOOR),
            )

    # ------------------------------------------------------------------ 6. residuals

    def fit_residuals(self, record_id: str, samples: List[Sample]) -> None:
        engine = self._engine(record_id)
        logs: Dict[Tuple[str, Key, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        floors: Dict[Tuple[str, Key, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        noise = self.replicate_noise()
        spread = self.point_spread()
        spreads: Dict[Tuple[str, Key, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        current: Dict[str, Any] = {}

        def note(kind: str, key: Key, detail: Mapping[str, Any], err: float) -> None:
            slot = (kind, key, tuple(sorted(detail.items())))
            logs.setdefault(slot, []).append(err)
            quantities = NOISE_QUANTITIES.get(kind, ("ttft",))
            cell = lambda q: (current["family"], q, current["bucket"])  # noqa: E731
            floors.setdefault(slot, []).append(max(noise.get(cell(q), 0.0) for q in quantities))

        for s in samples:
            try:
                pred = engine.predict(
                    s.settings, workload_for(s.point), s.model, s.device, draft=s.draft
                )
            except Exception:  # noqa: BLE001
                continue
            fam_key = self.key(s.cfg.sku, record_id, s.cfg.family)
            current.update(family=s.cfg.family, bucket=bucket_of(s.point.concurrency))
            # the band is built around the prediction, so its residual is the error relative to it: a point the
            # prediction undershoots needs meas / pred - 1, more than |pred / meas - 1| (coverage fell to 86% on the
            # 27B's MTP TTFT once its bands were narrow enough for the difference to show)
            e_tpot = band_error(pred.tpot_ms.value, s.point.tpot_ms)
            e_thr = band_error(pred.throughput_tok_s.value, s.point.output_tok_s)
            e_ttft = band_error(pred.ttft_ms.value, s.point.ttft_ms) if s.point.ttft_ms > 0 else 0.0
            dec = max(e_tpot, e_thr)
            kind = kind_of(s.cfg)
            bucket = bucket_of(pred.running)
            # every quantity's error in each stratum the point belongs to (the TPOT one after a speculative stratum's
            # bias, spec.scale): the bands the engine reads where a stratum has BAND_MIN_SAMPLES points
            from ..engine.vllm import _draft_id, band_strata

            res = engine.resolve(s.settings, s.model, s.device)
            did = _draft_id(s.draft) if s.settings.get("spec.method") in ("eagle3", "draft") else None
            measured_spread = spread.get(spread_key(s.cfg, s.point), {})
            for st in band_strata(res, s.model, s.device, shape_of(engine.record, s), bucket, did):
                note("band.tpot", fam_key, st, e_tpot)
                note("band.throughput", fam_key, st, e_thr)
                # the stratum's replicate spread, from the points identical runs measured (the A-05 amendment)
                for quantity, value in measured_spread.items():
                    spreads.setdefault((f"noise.{quantity}", fam_key, tuple(sorted(st.items()))), []).append(value)
            if kind in ("base", "shared"):
                note("decode.kappa", fam_key, {"bucket": bucket}, dec)
                from ..engine.vllm import prefill_bucket

                new = s.point.input_tokens * (1.0 - (s.point.prefix_hit_rate or 0.0))
                if not pred.queueing and workload_for(s.point).arrival_model == "wave":
                    pb = bucket if pred.running > 1 else "1"
                    note(
                        "prefill.kappa",
                        fam_key,
                        {"tokens": prefill_bucket(new), "bucket": pb},
                        e_ttft,
                    )
                elif not pred.queueing and pred.running > 1:
                    note("arrival.queue", fam_key, {}, e_ttft)
                    note("arrival.queue", fam_key, {"bucket": bucket}, e_ttft)
                elif not pred.queueing:
                    note("api.tau", fam_key, {}, e_ttft)
            elif kind == "precision":
                note("precision.ratio", fam_key, {"format": fmt_of(s.cfg), "bucket": bucket}, dec)
                # a steady load at c > 1 carries the queue term, whose band (arrival.queue) the point's TTFT band already
                # takes; its error is not the format's prefill (fork FP8 8B: real c32 at 3x, the base run's miss too)
                steady_queue = pred.running > 1 and workload_for(s.point).arrival_model != "wave"
                if not pred.queueing and not steady_queue:
                    note("precision.prefill", fam_key, {"format": fmt_of(s.cfg)}, e_ttft)
            elif kind == "spec":
                k = int(s.settings.get("spec.tokens") or 1)
                detail = {
                    "method": s.settings["spec.method"],
                    "k": k,
                    "sampling": s.point.sampling,
                    "workload": s.point.workload_class,
                }
                note("spec.acceptance", fam_key, detail, e_tpot)
                drafted = _spec_keys(s, k)
                if len(drafted) > 1:
                    with_draft = {**detail, "draft": drafted[1][5]}
                    note("spec.acceptance", fam_key, with_draft, e_tpot)
                    note("spec.acceptance_mean", fam_key, with_draft, e_thr)
                    note(
                        "spec.acceptance_mean",
                        fam_key,
                        {**with_draft, "bucket": bucket},
                        e_thr,
                    )
                if pred.running > 1:
                    method = s.settings["spec.method"]
                    note("spec.mix", fam_key, {"method": method, "bucket": bucket}, e_tpot)
                    note("spec.mix", fam_key, {"method": method}, e_tpot)
                note("spec.acceptance_mean", fam_key, detail, e_thr)
                note("spec.acceptance_mean", fam_key, {**detail, "bucket": bucket}, e_thr)
                if not pred.queueing:
                    target = (
                        "arrival.sync"
                        if (pred.running > 1 and workload_for(s.point).arrival_model == "wave")
                        else "spec.prefill"
                    )
                    note(target, fam_key, {"method": s.settings["spec.method"]}, e_ttft)
                    note(
                        target,
                        fam_key,
                        {"method": s.settings["spec.method"], "shape": shape_of(engine.record, s)},
                        e_ttft,
                    )
        for (kind, key, dkey), values in spreads.items():
            # the typical replicated point's spread: one disagreeing pair does not mark a stratum of steady ones
            self.put(kind, key, dict(dkey), float(statistics.median(values)), samples=len(values))
        for (kind, key, dkey), errs in logs.items():
            if kind.startswith("band.") and len(errs) >= BAND_MIN_SAMPLES:
                # a band-only entry: its value is unused, its residual is the stratum's
                self.put(kind, key, dict(dkey), 1.0, samples=len(errs))
            entry = self.entries.get((kind, key, dkey))
            if entry is not None:
                # out of sample: the fit's error and the run-to-run noise are independent, so they add in quadrature
                fit_err = max(conformal_p90(errs) if kind.startswith("band.") else p90(errs), RESIDUAL_FLOOR)
                entry.residual = math.hypot(fit_err, p90(floors.get((kind, key, dkey), [])))
        # memory residual: absolute bytes, per (family, format)
        for cfg in self.inputs.corpus:
            if (
                cfg.record != record_id
                or cfg.family in self.excluded
                or cfg.guarded
                or not cfg.startup.kv_gib
            ):
                continue
            model = model_for(self.inputs, cfg)
            if model is None or cfg.hardware_mode == "shared" or not self.supported(cfg):
                continue
            lim = engine.limits(
                predict_settings(cfg), model, device_for(cfg), draft=draft_for(self.inputs, cfg)
            )
            err = abs(lim.kv_pool_bytes - cfg.startup.kv_gib * GIB)
            logs.setdefault(
                (
                    "memory.overhead",
                    self.key(cfg.sku, record_id, cfg.family),
                    (("format", fmt_of(cfg)),),
                ),
                [],
            ).append(err)
        for (kind, key, dkey), errs in logs.items():
            if kind == "memory.overhead":
                entry = self.entries.get((kind, key, dkey))
                if entry is not None:
                    entry.residual = max(p90(errs), 0.15 * GIB)
        for entry in self.entries.values():
            if entry.residual <= 0:
                entry.residual = (
                    RESIDUAL_FLOOR if not entry.kind.startswith("memory") else 0.3 * GIB
                )

    # ------------------------------------------------------------------ 7. ladder levels

    def aggregate(self, family_class: Mapping[str, str]) -> None:
        # a stratum's replicate spread is its own family's on the measured SKU (read there only, the A-05 amendment): no
        # class, pooled or architecture copy
        measured = [
            e for e in self.entries.values() if e.key.model.startswith("family:") and not e.kind.startswith("noise.")
        ]
        groups: Dict[Tuple[str, str, str, str, Tuple[Tuple[str, Any], ...]], List[Entry]] = {}
        for e in measured:
            family = e.key.model.split(":", 1)[1]
            klass = family_class.get(family)
            dkey = tuple(sorted(e.detail.items()))
            if klass:
                groups.setdefault(
                    (e.kind, e.key.hardware, e.key.engine, f"class:{klass}", dkey), []
                ).append(e)
            groups.setdefault((e.kind, e.key.hardware, e.key.engine, "pooled", dkey), []).append(e)
        # memory overhead across formats, for a format no family measured
        for e in measured:
            if e.kind == "memory.overhead":
                groups.setdefault((e.kind, e.key.hardware, e.key.engine, "pooled", ()), []).append(
                    e
                )
        for (kind, hw, en, model_level, dkey), items in groups.items():
            values = [e.value for e in items]
            value = float(statistics.median(values))
            spread = (max(values) - min(values)) / abs(value) if value and len(values) > 1 else 0.0
            residual = max(
                max(e.residual for e in items), spread if not kind.startswith("memory") else 0.0
            )
            if kind.startswith("memory") and len(values) > 1:
                residual = max(residual, max(values) - min(values))
            existing = self.entries.get((kind, Key(hw, en, model_level), dkey))
            if (
                existing is not None
                and model_level == "pooled"
                and not existing.key.model.startswith("family")
            ):
                continue
            self._store_cache = None
            self.entries[(kind, Key(hw, en, model_level), dkey)] = Entry(
                kind=kind,
                key=Key(hw, en, model_level),
                detail=dict(dkey),
                value=value,
                residual=residual,
                samples=sum(e.samples for e in items),
                provenance={
                    "run_id": self.inputs.run_id,
                    "harness": HARNESS_VERSION,
                    "date": self.today,
                    "fit": f"aggregate:{model_level}",
                    # the member families: a pooled entry of one family does not stand for a family the store lacks
                    "families": sorted({e.key.model.split(":", 1)[1] for e in items}),
                },
            )
        # hardware: the Hopper architecture level from the vGPU SKU (FR-GND-7)
        arch_entries = []
        for e in list(self.entries.values()):
            if not e.key.hardware.startswith("sku:") or e.kind.startswith("noise.") or e.kind == "parallel.measured":
                continue  # measured on the SKU, read there only
            sku = e.key.hardware.split(":", 1)[1]
            arch = catalogue.get(sku).architecture if sku in catalogue.skus() else None
            if not arch:
                continue
            bump = VGPU_RESIDUAL * (
                e.value if e.kind.startswith("memory") and e.kind != "memory.capture" else 1.0
            )
            if e.kind in ("memory.capture", "memory.chunk"):
                bump = VGPU_RESIDUAL
            arch_entries.append(
                replace(
                    e,
                    key=Key(f"arch:{arch}", e.key.engine, e.key.model),
                    residual=e.residual + abs(bump),
                    provenance={**e.provenance, "fit": "architecture-from-vgpu"},
                )
            )
        for e in arch_entries:
            self._store_cache = None
            self.entries[(e.kind, e.key, tuple(sorted(e.detail.items())))] = e
        # engine: the minor level from the upstream record of that minor
        minor_entries = []
        for record_id, record in self.inputs.records.items():
            if record.build != "upstream":
                continue
            for e in list(self.entries.values()):
                if e.key.engine == f"record:{record_id}":
                    minor_entries.append(
                        replace(
                            e,
                            key=Key(e.key.hardware, f"minor:{record.minor_key}", e.key.model),
                            provenance={**e.provenance, "fit": "engine-minor"},
                        )
                    )
                    if e.kind.startswith("noise."):
                        continue  # read on this build or its minor only
                    minor_entries.append(
                        replace(
                            e,
                            key=Key(e.key.hardware, f"family:{record.engine}", e.key.model),
                            residual=e.residual * 1.5,
                            provenance={**e.provenance, "fit": "engine-family"},
                        )
                    )
        for e in minor_entries:
            self.entries.setdefault((e.kind, e.key, tuple(sorted(e.detail.items()))), e)

    # ------------------------------------------------------------------ boot outcomes

    def boot_outcomes(self) -> List[BootOutcome]:
        out = []
        for cfg in self.inputs.corpus:
            if cfg.startup.ready or cfg.family in self.excluded:
                continue
            model = model_for(self.inputs, cfg)
            record = self.inputs.records.get(cfg.record)
            if model is None or record is None:
                continue
            engine = VLLMEngine(record, self.store())
            lim = engine.limits(
                predict_settings(cfg), model, device_for(cfg), draft=draft_for(self.inputs, cfg)
            )
            margin = min((c.margin_bytes for c in lim.checks), default=0.0)
            out.append(
                BootOutcome(
                    key=self.key(cfg.sku, cfg.record, cfg.family),
                    settings=dict(cfg.settings),
                    predicted_margin_bytes=float(margin) if lim.boots else 0.0,
                    outcome=cfg.startup.failure or "error",
                    message=cfg.startup.message or "",
                    provenance={
                        "run_id": self.inputs.run_id,
                        "config": cfg.name,
                        "harness": HARNESS_VERSION,
                        "date": self.today,
                        "predicted_boots": lim.boots,
                        "predicted_failures": [c.check for c in lim.failures],
                    },
                )
            )
        return out


def _fit_record(fitter: "Fitter", record_id: str, samples: List[Sample]) -> None:
    fitter.fit_api(record_id, samples)
    fitter.fit_prefill(record_id, samples)
    fitter.fit_decode(record_id, samples)
    fitter.fit_queue(record_id, samples)
    fitter.fit_drain(record_id, samples)
    fitter.fit_precision(record_id, samples)
    fitter.fit_prefill_ratio(record_id, samples, kind="precision")
    fitter.fit_precision(record_id, samples)  # re-fit with the format's prefill ratio in place
    fitter.fit_lora(record_id, samples)
    fitter.fit_spec(record_id, samples)
    fitter.fit_prefill_ratio(record_id, samples, kind="spec")
    fitter.fit_spec(record_id, samples)  # re-fit acceptance with the method's prefill ratio in place
    # TPOT interference and the wave's survival on TTFT each move the other's fit: alternate until both hold
    for _ in range(2):
        fitter.fit_spec_mix(record_id, samples)
        fitter.fit_sync(record_id, samples)
    fitter.fit_spec(record_id, samples)  # the mean acceptance (throughput) with the interference in place
    fitter.fit_spec_scale(record_id, samples)
    fitter.fit_boot(record_id)
    fitter.fit_residuals(record_id, samples)


def fit_store(inputs: FitInputs) -> GroundingStore:
    fitter = Fitter(inputs)
    family_class = {}
    for (family, _fmt), facts in inputs.models.items():
        family_class[family] = facts.arch_class
    order = sorted(inputs.records, key=lambda rid: (inputs.records[rid].build != "upstream", rid))
    work = []
    for record_id in order:
        if not any(c.record == record_id for c in inputs.corpus):
            continue
        fitter.fit_memory(record_id)
        samples = fitter.samples(record_id)
        if samples:
            work.append((record_id, samples))
    # a build with no base runs (bud-vllm 0.10.0: speculative runs only) is fitted against the engine-family level,
    # which aggregation writes; fitted before it, its terms were solved against the bare priors (T-GND-20)
    with_base = [w for w in work if any(kind_of(s.cfg) in ("base", "shared") for s in w[1])]
    without = [w for w in work if w not in with_base]
    for record_id, samples in with_base:
        _fit_record(fitter, record_id, samples)
    if without:
        fitter.aggregate(family_class)
        for record_id, samples in without:
            _fit_record(fitter, record_id, samples)
    for e in parallel_entries(c for c in inputs.corpus if c.family not in fitter.excluded):
        fitter.entries[(e.kind, e.key, tuple(sorted(e.detail.items())))] = e
    fitter.aggregate(family_class)
    store = fitter.store()
    store.outcomes = fitter.boot_outcomes()
    store.metadata = {
        "version": inputs.version,
        "harness": HARNESS_VERSION,
        "run_id": inputs.run_id,
        "excluded_families": sorted(fitter.excluded),
        "configs": len(inputs.corpus),
        "entries": len(store.entries),
    }
    return GroundingStore(
        store.entries, store.outcomes, version=inputs.version, metadata=store.metadata
    )
