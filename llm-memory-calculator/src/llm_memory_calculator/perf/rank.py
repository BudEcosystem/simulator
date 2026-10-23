"""FRD-027 Revision 2, §20.4: risk-aware ranking (FR-SEL-1).

A prediction's band is a p90 interval (FR-RES-2), read here as a log-normal error: the band's upper edge is the
value times (1 + h), so ln(1 + h) is 1.2816 standard deviations. The probability that a configuration meets the plan's
targets is the least likely target's: TTFT, end-to-end time and throughput come from one step model and their errors
move together, so a product would count the same error several times. An ungrounded estimate gives no probability;
callers keep their point rule for it (NFR-5).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

Z90 = 1.2815515655446004  # the standard normal's 90th percentile


def _parts(est: Any):
    if isinstance(est, Mapping):
        return float(est["value"]), float(est["low"]), float(est["high"]), bool(est.get("ungrounded"))
    return float(est.value), float(est.low), float(est.high), bool(getattr(est, "ungrounded", False))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigma(value: float, edge: float) -> float:
    return abs(math.log(edge / value)) / Z90 if value > 0 and edge > 0 else 0.0


def p_at_most(est: Any, limit: float) -> float:
    """P(the measured value <= limit), from the band's upper edge."""
    value, _, high, _ = _parts(est)
    s = _sigma(value, high)
    if s == 0.0:
        return 1.0 if value <= limit else 0.0
    return _phi(math.log(limit / value) / s) if limit > 0 else 0.0


def p_at_least(est: Any, limit: float) -> float:
    """P(the measured value >= limit), from the band's lower edge."""
    value, low, _, _ = _parts(est)
    s = _sigma(value, low)
    if s == 0.0:
        return 1.0 if value >= limit else 0.0
    return _phi(math.log(value / limit) / s) if limit > 0 else 1.0


def slo_probability(
    prediction: Mapping[str, Any],
    *,
    concurrency: int,
    ttft_ms: Optional[float] = None,
    e2e_s: Optional[float] = None,
    throughput_per_user: Optional[float] = None,
) -> Optional[float]:
    """The probability that every set target holds, or None when an estimate it needs is ungrounded."""
    probs = []
    checks = (
        ("ttft_ms", ttft_ms, p_at_most, 1.0),
        ("e2e_s", e2e_s, p_at_most, 1.0),
        ("throughput_tok_s", throughput_per_user, p_at_least, float(max(int(concurrency), 1))),
    )
    for key, target, prob, per in checks:
        if target is None or target <= 0:
            continue
        est = prediction[key]
        value, low, high, ungrounded = _parts(est)
        if ungrounded:
            return None
        scaled = {"value": value / per, "low": low / per, "high": high / per}
        probs.append(prob(scaled, float(target)))
    return min(probs) if probs else 1.0


def slo_confidence() -> float:
    """The probability a plan's targets must hold with: PLANNER_SLO_CONFIDENCE, default 0.9, within [0.5, 0.999]."""
    from ..planner_env import planner_env

    try:
        value = float(planner_env("SLO_CONFIDENCE", "0.9"))
    except ValueError:
        return 0.9
    return min(max(value, 0.5), 0.999)
