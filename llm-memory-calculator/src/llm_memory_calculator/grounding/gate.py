"""The accuracy gate (FRD-027 FR-GND-3, NFR-1): replay a corpus through the predictor and score it.

For every usable point the replay records the predicted value, the band, the measured value and whether the
measured value falls inside the band; for every configuration with a measured KV pool, the memory error. The
summary is what the CI test asserts: per-quantity MAPE under the NFR-1 ceiling, at least 90% of points inside
their band, and no band wider than twice its ceiling (a wider one is reported ungrounded, not a pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..engine.priors import GIB
from ..engine.vllm import CEILINGS, VLLMEngine
from .fit import (
    FitInputs,
    device_for,
    draft_for,
    kind_of,
    model_for,
    predict_settings,
    usable_point,
    workload_for,
)
from .store import GroundingStore


@dataclass
class PointResult:
    config: str
    family: str
    kind: str
    point: str
    concurrency: int
    quantity: str
    predicted: float
    low: float
    high: float
    measured: float
    queueing: bool
    ungrounded: bool
    #: the stratum's measured replicate spread exceeds the ceiling (``Estimate.noise_limited``)
    noise_limited: bool = False

    @property
    def error(self) -> float:
        return abs(self.predicted / self.measured - 1.0) if self.measured else 0.0

    @property
    def inside(self) -> bool:
        return self.low <= self.measured <= self.high

    @property
    def half_width(self) -> float:
        if not self.predicted:
            return 0.0
        return max(self.high - self.predicted, self.predicted - self.low) / self.predicted

    @property
    def too_wide(self) -> bool:
        """A band past twice its ceiling fails A-05, unless the point is noise-limited: its replicates disagree by more
        than the ceiling, so it is graded on coverage only (FRD-027 §16, the A-05 amendment)."""
        return self.half_width > 2 * ceiling(self.quantity, self.queueing) + 1e-9 and not self.noise_limited


@dataclass
class GateReport:
    points: List[PointResult] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    #: a HAMi slice's KV pool, in bytes against the memory band: the relative ceiling is for a whole-GPU pool
    memory_shared: List[Dict[str, Any]] = field(default_factory=list)
    boot: List[Dict[str, Any]] = field(default_factory=list)

    def select(
        self,
        quantity: str,
        *,
        family: Optional[str] = None,
        kind: Optional[str] = None,
        queueing: Optional[bool] = None,
    ) -> List[PointResult]:
        out = [p for p in self.points if p.quantity == quantity]
        if family:
            out = [p for p in out if p.family == family]
        if kind:
            out = [p for p in out if p.kind == kind]
        if queueing is not None:
            out = [p for p in out if p.queueing == queueing]
        return out

    @staticmethod
    def mape(points: Iterable[PointResult]) -> float:
        pts = list(points)
        return sum(p.error for p in pts) / len(pts) if pts else 0.0

    @staticmethod
    def coverage(points: Iterable[PointResult]) -> float:
        pts = list(points)
        return sum(1 for p in pts if p.inside) / len(pts) if pts else 1.0

    def memory_mape(self) -> float:
        return sum(m["error"] for m in self.memory) / len(self.memory) if self.memory else 0.0

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for q in ("tpot", "throughput", "ttft"):
            pts = self.select(q)
            out[q] = {
                "mape": round(self.mape(pts), 4),
                "coverage": round(self.coverage(pts), 4),
                "n": len(pts),
                "max_band": round(
                    max(
                        (max(p.high / p.predicted - 1, 1 - p.low / p.predicted) for p in pts),
                        default=0,
                    ),
                    3,
                ),
            }
        out["ttft_queueing"] = {"mape": round(self.mape(self.select("ttft", queueing=True)), 4)}
        out["ttft_free"] = {"mape": round(self.mape(self.select("ttft", queueing=False)), 4)}
        out["memory"] = {"mape": round(self.memory_mape(), 4), "n": len(self.memory)}
        out["boot"] = {
            "mape": (
                round(sum(b["error"] for b in self.boot) / len(self.boot), 4) if self.boot else 0.0
            ),
            "n": len(self.boot),
        }
        return out


def replay(
    inputs: FitInputs,
    store: GroundingStore,
    *,
    families: Optional[Iterable[str]] = None,
    kinds: Optional[Iterable[str]] = None,
) -> GateReport:
    fams = set(families) if families else None
    want_kinds = set(kinds) if kinds else None
    report = GateReport()
    for cfg in inputs.corpus:
        if fams and cfg.family not in fams:
            continue
        record = inputs.records.get(cfg.record)
        model = model_for(inputs, cfg)
        if record is None or model is None:
            continue
        kind = kind_of(cfg)
        if want_kinds and kind not in want_kinds:
            continue
        engine = VLLMEngine(record, store)
        settings = predict_settings(cfg)
        try:
            engine.check_supported(settings)
        except Exception:  # noqa: BLE001 - not a configuration this record may render
            continue
        device = device_for(cfg)
        draft = draft_for(inputs, cfg)
        if cfg.startup.ready and cfg.startup.kv_gib and not cfg.guarded:
            lim = engine.limits(settings, model, device, draft=draft)
            measured = cfg.startup.kv_gib * GIB
            row = {
                "config": cfg.name,
                "family": cfg.family,
                "predicted": lim.kv_pool_bytes,
                "measured": measured,
                "error": abs(lim.kv_pool_bytes / measured - 1),
            }
            if cfg.hardware_mode == "dedicated":
                report.memory.append(row)
            else:
                report.memory_shared.append(
                    {**row, "slice_mib": cfg.slice_mib, "band": lim.memory_band_bytes}
                )
        if cfg.startup.ready and cfg.startup.api_up_s and not cfg.guarded:
            boot = engine.boot_time(settings, model, device)
            report.boot.append(
                {
                    "config": cfg.name,
                    "predicted": boot.value,
                    "measured": cfg.startup.api_up_s,
                    "error": abs(boot.value / cfg.startup.api_up_s - 1),
                }
            )
        for point in cfg.points:
            if not usable_point(cfg, point):
                continue
            try:
                pred = engine.predict(settings, workload_for(point), model, device, draft=draft)
            except Exception:  # noqa: BLE001 - a refused configuration is not graded on time
                continue
            for quantity, est, measured in (
                ("tpot", pred.tpot_ms, point.tpot_ms),
                ("throughput", pred.throughput_tok_s, point.output_tok_s),
                ("ttft", pred.ttft_ms, point.ttft_ms),
            ):
                if measured <= 0:
                    continue
                report.points.append(
                    PointResult(
                        cfg.name,
                        cfg.family,
                        kind,
                        point.point,
                        point.concurrency,
                        quantity,
                        est.value,
                        est.low,
                        est.high,
                        measured,
                        pred.queueing,
                        est.ungrounded,
                        est.noise_limited,
                    )
                )
    return report


def ceiling(quantity: str, queueing: bool = False) -> float:
    if quantity == "ttft":
        return CEILINGS["ttft_queueing"] if queueing else CEILINGS["ttft"]
    return CEILINGS[quantity]
