"""Judge the planner against the measured runs (FRD-023 NFR-9).

    python -m llm_memory_calculator.kv.validate

Decision quality first, then the parts of the break-even:

* **decision accuracy and regret** on the load fixtures: the planner's enable/disable choice for the
  tier against the configuration with the higher measured throughput, and the throughput it gives up
  (for a CPU engine, growing its KV space past demand versus not);
* **enable precision / recall** over every fixture;
* **reload error** (predicted single-load time vs tier TTFT minus GPU-hit TTFT), **recompute error**
  (GenZ vs measured prefill), and **bytes** (predicted vs logged transfer sizes).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .calibration import Fixture, fixtures, geometry_checks
from .constants import BREAK_EVEN_RATIO
from .geometry import KVGeometry
from .plan import KVTierPlan
from .planner import plan_kv


@dataclass
class FixtureResult:
    fixture: Fixture
    plan: KVTierPlan
    planned: bool  # the planner's decision for the fixture's tier
    oracle: bool
    reload_pred_ms: Optional[float] = None
    recompute_pred_ms: Optional[float] = None
    bytes_pred: Optional[float] = None
    regret: Optional[float] = None

    @property
    def correct(self) -> bool:
        return self.planned == self.oracle or self.fixture.tie


def run_fixture(fixture: Fixture) -> FixtureResult:
    plan = plan_kv(
        fixture.deployment,
        fixture.engine,
        fixture.group,
        fixture.infra,
        recompute_fn=fixture.recompute,
        workload=fixture.workload,
    )
    verdict = plan.evaluations.get(fixture.tier)
    if fixture.kind == "single":
        # One request alone: the decision the planner's rule makes for an idle tier.
        planned = bool(
            verdict is not None
            and verdict.recompute_ms is not None
            and verdict.reload_idle_ms < BREAK_EVEN_RATIO * verdict.recompute_ms
        )
    elif fixture.tier == "CPU":
        # A CPU engine's lever (FR-T0-4): grow the KV space past budsim's demand toward the working
        # set.
        demand_gib = (fixture.group.kv_cache_memory_gb or 0.0) * 1e9 / (1 << 30)
        planned = bool(plan.cpu_kv_gib and plan.cpu_kv_gib > demand_gib + 1e-6)
    else:
        planned = plan.planned(fixture.tier)
    result = FixtureResult(fixture=fixture, plan=plan, planned=planned, oracle=fixture.oracle_keep)
    if verdict is not None:
        result.reload_pred_ms = verdict.reload_idle_ms
        result.recompute_pred_ms = verdict.recompute_ms
        result.bytes_pred = verdict.bytes
    if fixture.kind == "load":
        best = max(fixture.goodput_with or 0.0, fixture.goodput_without or 0.0)
        chosen = fixture.goodput_with if planned else fixture.goodput_without
        result.regret = (best - (chosen or 0.0)) / best if best > 0 else 0.0
    return result


def _ape(predicted: Optional[float], measured: Optional[float]) -> Optional[float]:
    if predicted is None or not measured or not math.isfinite(predicted):
        return None
    return abs(predicted - measured) / measured


@dataclass
class Report:
    results: List[FixtureResult]
    geometry: List[Dict[str, float]] = field(default_factory=list)

    def _load(self) -> List[FixtureResult]:
        return [r for r in self.results if r.fixture.kind == "load"]

    @property
    def decision_accuracy(self) -> float:
        load = self._load()
        return sum(r.correct for r in load) / len(load) if load else 1.0

    @property
    def regrets(self) -> List[float]:
        return [r.regret for r in self._load() if r.regret is not None]

    def enable_counts(self) -> Dict[str, int]:
        decided = [r for r in self.results if not r.fixture.tie]
        tp = sum(1 for r in decided if r.planned and r.oracle)
        fp = sum(1 for r in decided if r.planned and not r.oracle)
        fn = sum(1 for r in decided if not r.planned and r.oracle)
        tn = sum(1 for r in decided if not r.planned and not r.oracle)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ties": len(self.results) - len(decided)}

    @property
    def enable_precision(self) -> float:
        c = self.enable_counts()
        return c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0

    @property
    def enable_recall(self) -> float:
        c = self.enable_counts()
        return c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0

    def reload_errors(self) -> List[float]:
        return [
            e
            for r in self.results
            if r.fixture.kind == "single" and r.fixture.gate_measured
            for e in [_ape(r.reload_pred_ms, r.fixture.reload_ms)]
            if e is not None
        ]

    def recompute_errors(self) -> List[float]:
        seen, errors = set(), []
        for r in self.results:
            f = r.fixture
            key = (f.deployment.model_id, f.workload.input_tokens)
            if f.genz_ms is None or key in seen:
                continue
            seen.add(key)
            errors.append(abs(f.genz_ms - f.recompute_ms) / f.recompute_ms)
        return errors

    def misses(self) -> List[str]:
        return [r.fixture.name for r in self.results if not r.correct]


def geometry_report() -> List[Dict[str, float]]:
    rows = []
    for check in geometry_checks():
        geometry = KVGeometry.from_model(check.config, seq_length=check.prompt_tokens + 32)
        block = check.block_size or geometry.block_size
        _, nbytes = geometry.prefix_bytes(check.prompt_tokens, block_size=block, scope="total")
        rows.append(
            {
                "name": check.name,
                "block": block,
                "predicted": nbytes,
                "measured": check.measured_bytes,
                "error": abs(nbytes - check.measured_bytes) / check.measured_bytes,
            }
        )
    return rows


def run() -> Report:
    return Report(results=[run_fixture(f) for f in fixtures()], geometry=geometry_report())


def _fmt_ms(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:,.0f}"


def render(report: Report) -> str:
    lines = []
    lines.append(
        f"{'fixture':40} {'tier':4} {'kind':6} {'planner':8} {'oracle':7} {'ok':3}"
        f" {'regret':>7} {'reload ms pred/meas':>22} {'recompute ms pred/meas':>24}"
    )
    for r in report.results:
        f = r.fixture
        regret = f"{r.regret:.0%}" if r.regret is not None else "-"
        reload = f"{_fmt_ms(r.reload_pred_ms)}/{_fmt_ms(f.reload_ms)}"
        recompute = f"{_fmt_ms(r.recompute_pred_ms)}/{_fmt_ms(f.recompute_ms)}"
        lines.append(
            f"{f.name:40} {f.tier:4} {f.kind:6} {'keep' if r.planned else 'drop':8}"
            f" {'tie' if f.tie else 'keep' if r.oracle else 'drop':7}"
            f" {'yes' if r.correct else 'NO':3}"
            f" {regret:>7}"
            f" {reload:>22} {recompute:>24}"
        )
    lines.append("")
    regrets = report.regrets
    counts = report.enable_counts()
    lines.append(
        f"decision accuracy (load fixtures): {report.decision_accuracy:.0%} of {len(regrets)};"
        f" regret median {statistics.median(regrets):.0%}, max {max(regrets):.0%}"
        if regrets
        else "no load fixtures"
    )
    lines.append(
        f"enable precision {report.enable_precision:.2f}, recall {report.enable_recall:.2f}"
        f" (tp {counts['tp']}, fp {counts['fp']}, fn {counts['fn']}, tn {counts['tn']};"
        f" {counts['ties']} ties excluded)"
    )
    reload = report.reload_errors()
    if reload:
        lines.append(
            f"reload error (gate-measured single fixtures): MAPE {statistics.mean(reload):.0%},"
            f" max {max(reload):.0%} over {len(reload)}"
        )
    recompute = report.recompute_errors()
    if recompute:
        lines.append(
            f"recompute error (GenZ vs measured): MAPE {statistics.mean(recompute):.0%},"
            f" max {max(recompute):.0%} over {len(recompute)}"
        )
    worst = max((g["error"] for g in report.geometry), default=0.0)
    lines.append(
        f"bytes per prefix load: {len(report.geometry)} logged transfers, worst error {worst:.2%}"
    )
    misses = report.misses()
    lines.append("misses: " + (", ".join(misses) if misses else "none"))
    return "\n".join(lines)


def main() -> None:
    print(render(run()))


if __name__ == "__main__":
    main()
