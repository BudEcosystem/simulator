"""SLO (Service Level Objective) tracking for the serving simulation.

Tracks TTFT, TPOT, E2E, and ITL metrics against configurable targets
and computes violation rates, percentiles, and goodput.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math

from .constants import NS_PER_MS
from .request import Request, RequestStatus


@dataclass
class SLOTargets:
    """Configurable SLO target thresholds in nanoseconds."""
    ttft_target_ns: int = 500_000_000       # 500 ms
    tpot_target_ns: int = 100_000_000       # 100 ms
    e2e_target_ns: int = 30_000_000_000     # 30 s

    @classmethod
    def from_ms(cls, ttft_ms: float, tpot_ms: float, e2e_ms: float) -> 'SLOTargets':
        """Create targets from millisecond values."""
        return cls(
            ttft_target_ns=int(ttft_ms * NS_PER_MS),
            tpot_target_ns=int(tpot_ms * NS_PER_MS),
            e2e_target_ns=int(e2e_ms * NS_PER_MS),
        )


class SLOTracker:
    """Accumulates latency metrics from completed requests and evaluates SLO compliance.

    Only processes requests with status == COMPLETE. Non-complete requests
    (FAILED, QUEUED, etc.) are silently ignored.
    """

    def __init__(self, targets: Optional[SLOTargets] = None):
        self.targets = targets or SLOTargets()
        self._ttft_values: List[int] = []
        self._tpot_values: List[int] = []
        self._e2e_values: List[int] = []
        self._itl_values: List[int] = []
        self._total_completed: int = 0
        self._slo_met_count: int = 0

    def record_completed_request(self, request: Request) -> None:
        """Record metrics from a completed request. Non-COMPLETE requests are ignored."""
        if request.status != RequestStatus.COMPLETE:
            return
        self._total_completed += 1
        self._ttft_values.append(request.ttft_ns)
        self._tpot_values.append(request.tpot_ns)
        self._e2e_values.append(request.e2e_ns)
        self._itl_values.extend(request.itl_ns)

        ttft_ok = request.ttft_ns <= self.targets.ttft_target_ns
        tpot_ok = request.tpot_ns <= self.targets.tpot_target_ns
        e2e_ok = request.e2e_ns <= self.targets.e2e_target_ns
        if ttft_ok and tpot_ok and e2e_ok:
            self._slo_met_count += 1

    def ttft_violation_rate(self) -> float:
        """Fraction of requests whose TTFT exceeds the target."""
        if not self._ttft_values:
            return 0.0
        violations = sum(1 for v in self._ttft_values if v > self.targets.ttft_target_ns)
        return violations / len(self._ttft_values)

    def tpot_violation_rate(self) -> float:
        """Fraction of requests whose TPOT exceeds the target."""
        if not self._tpot_values:
            return 0.0
        violations = sum(1 for v in self._tpot_values if v > self.targets.tpot_target_ns)
        return violations / len(self._tpot_values)

    def e2e_violation_rate(self) -> float:
        """Fraction of requests whose E2E latency exceeds the target."""
        if not self._e2e_values:
            return 0.0
        violations = sum(1 for v in self._e2e_values if v > self.targets.e2e_target_ns)
        return violations / len(self._e2e_values)

    def _percentiles(self, values: List[int]) -> Dict[str, float]:
        """Compute p50, p95, p99 percentiles from a list of values."""
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def pct(p: float) -> float:
            idx = min(int(math.ceil(p / 100.0 * n)) - 1, n - 1)
            return float(sorted_vals[max(idx, 0)])

        return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}

    def ttft_percentiles(self) -> Dict[str, float]:
        """Return p50/p95/p99 for TTFT values."""
        return self._percentiles(self._ttft_values)

    def tpot_percentiles(self) -> Dict[str, float]:
        """Return p50/p95/p99 for TPOT values."""
        return self._percentiles(self._tpot_values)

    def e2e_percentiles(self) -> Dict[str, float]:
        """Return p50/p95/p99 for E2E latency values."""
        return self._percentiles(self._e2e_values)

    def itl_percentiles(self) -> Dict[str, float]:
        """Return p50/p95/p99 for inter-token latency values."""
        return self._percentiles(self._itl_values)

    def goodput(self) -> float:
        """Fraction of completed requests that met ALL SLO targets."""
        if self._total_completed == 0:
            return 0.0
        return self._slo_met_count / self._total_completed

    def summary(self) -> Dict[str, Any]:
        """Return a comprehensive summary dict of all tracked metrics."""
        return {
            "total_completed": self._total_completed,
            "goodput": self.goodput(),
            "ttft": {
                "violation_rate": self.ttft_violation_rate(),
                "percentiles_ns": self.ttft_percentiles(),
            },
            "tpot": {
                "violation_rate": self.tpot_violation_rate(),
                "percentiles_ns": self.tpot_percentiles(),
            },
            "e2e": {
                "violation_rate": self.e2e_violation_rate(),
                "percentiles_ns": self.e2e_percentiles(),
            },
            "itl": {
                "percentiles_ns": self.itl_percentiles(),
            },
        }
