"""FRD-027 Revision 2, FR-RES-4: production telemetry as L5 grounding data.

A deployment's measured medians at its observed load are compared with the simulation's uncorrected prediction for
the same settings and load: the residual is what L1–L4 got wrong in production, exactly as a corpus point's is
(``perf.replay``). ``perf.residual.refit`` then joins those residuals to the corpus. Probe-before-commit results
(FR-SEL-3) arrive the same way, with their own source.

Telemetry has no TPOT today (budmetrics serves TTFT, latency and throughput), so an observation grounds the quantities
it carries and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from .residual import Residual, load_bucket

#: real-text outputs stop at end of sequence; the suite's cap, as the corpus replay uses
OUTPUT_CAP = 512


@dataclass
class Observation:
    """One deployment's measured medians over one window, at its observed load."""

    settings: Dict[str, Any] = field(default_factory=dict)  # the rendered engine settings (the spec's)
    concurrency: int = 1  # mean requests in flight per replica over the window
    input_tokens: float = 1024.0  # mean prompt tokens
    output_tokens: float = 256.0  # mean output tokens
    workload_class: str = "real"
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    throughput_tok_s: Optional[float] = None  # output tokens per second per replica
    requests: Optional[int] = None  # requests completed in the window (None: open-ended traffic)
    source: str = "telemetry"  # names the observation: telemetry:<endpoint>:<window> or probe:<plan>:<candidate>


def residuals_of(engine: Any, obs: Observation, model: Any, device: Any, *, draft: Any = None) -> List[Residual]:
    """The observation's residuals against the uncorrected simulation, or none when no profile predicts it (L5 then
    has nothing to correct: the fitted predictor serves that configuration)."""
    from ..engine import Workload
    from .predict import find_profile, predict

    profiles = engine.profiles()
    if profiles is None:
        return []
    profile = find_profile(engine, profiles, obs.settings, model, device)
    if profile is None:
        return []
    w = Workload(
        input_tokens=float(obs.input_tokens),
        output_tokens=float(obs.output_tokens),
        concurrency=max(int(obs.concurrency), 1),
        workload_class=obs.workload_class,
        requests=obs.requests,
    )
    if w.workload_class != "synthetic":
        w = replace(w, output_tokens_max=OUTPUT_CAP)
    pred = predict(engine, profile, obs.settings, w, model, device, draft=draft)
    measured = {"ttft": obs.ttft_ms, "tpot": obs.tpot_ms, "throughput": obs.throughput_tok_s}
    predicted = {"ttft": pred.ttft_ms.value, "tpot": pred.tpot_ms.value, "throughput": pred.throughput_tok_s.value}
    bucket = load_bucket(w.concurrency, w.workload_class)
    return [
        Residual(
            sku=profile.sku,
            engine=profile.engine,
            family=profile.model,
            quantity=q,
            bucket=bucket,
            predicted=float(predicted[q]),
            measured=float(measured[q]),
            config=obs.source,
            queueing=bool(pred.queueing),
        )
        for q in ("tpot", "throughput", "ttft")
        if measured[q] and measured[q] > 0 and predicted[q] > 0
    ]
