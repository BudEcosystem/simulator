"""FRD-027 Revision 2: a prediction from a measured step profile and the scheduler simulation (§20.2 L2-L4).

Memory, limits and refusals still come from the engine layer (its KV pool is within 1.6%); the time comes from the
profile's measured steps run through ``perf.sched``.

L5 (``perf.residual``) then grounds each quantity: its slice's correction multiplies the simulation's value, and the
band is the slice's conformal residual after correction (never narrower than the interpolation error). A slice that
would need more than the cap is a model gap: uncorrected, and the prediction is ungrounded. Without corrections for
the slice the band is the profile's interpolation error combined with its measured residual, as in R1.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional, Tuple

from ..engine.base import Estimate, Prediction
from .profile import StepProfile
from .sched import SimConfig, simulate

#: the band's residual term before a profile has been validated against deployments: wide, so a plan does not lean on
#: it (NFR-5)
DEFAULT_RESIDUAL = {"tpot": 0.15, "throughput": 0.15, "ttft": 0.35}
#: a first-use stall past this is reported with the plan
COLD_STALL_NOTE_MS = 500.0


def sim_config(engine: Any, settings: Any, model: Any, device: Any, profile: StepProfile, draft: Any = None) -> SimConfig:
    lim = engine.limits(settings, model, device, draft=draft)
    r = engine.resolve(settings, model, device)
    state = (lim.state_bytes_per_sequence / lim.block_bytes) if (lim.state_bytes_per_sequence and lim.block_bytes) else 0.0
    blocks = int(lim.num_blocks or (lim.kv_tokens // max(lim.block_tokens, 1)))
    half = profile.api_ms / 2.0
    return SimConfig(
        max_num_seqs=int(min(r.max_running, lim.sequence_ceiling or r.max_running)),
        chunk=int(r.chunk),
        kv_blocks=max(blocks, 1),
        block_tokens=max(int(lim.block_tokens), 1),
        attn_groups=int(lim.attn_groups or 1),
        state_blocks=state,
        arrive_ms=half,
        return_ms=half,
        async_scheduling="no_async_scheduling" not in getattr(engine.record, "features", ()),
    )


def predict(
    engine: Any,
    profile: StepProfile,
    settings: Any,
    workload: Any,
    model: Any,
    device: Any,
    *,
    draft: Any = None,
    corrections: Any = None,
) -> Prediction:
    from ..engine.vllm import CEILINGS

    lim = engine.limits(settings, model, device, draft=draft)
    cfg = sim_config(engine, settings, model, device, profile, draft=draft)
    from .lengths import lengths_for

    c = max(int(workload.concurrency), 1)
    n = int(workload.requests) if workload.requests else max(8 * c, 32)
    # a prefix-cache hit spares that share of every prompt's prefill
    new_tokens = float(workload.input_tokens) * (1.0 - float(workload.prefix_hit_rate or 0.0))
    res = simulate(profile, cfg, lengths_for(replace(workload, input_tokens=max(new_tokens, 1.0)), n), concurrency=c)
    interp = profile.interpolation_error()

    def band(q: str) -> float:
        return math.hypot(interp, profile.residual.get(q, DEFAULT_RESIDUAL[q]))

    queueing = res.max_running < c
    ceiling_ttft = CEILINGS["ttft_queueing"] if queueing or c >= 64 else CEILINGS["ttft"]
    notes = list(lim.notes) + [
        f"predictor: step profile {profile.engine} on {profile.sku} ({profile.weights} weights) with the scheduler "
        f"simulation (FRD-027 §20)"
    ]
    if workload.workload_class != "synthetic":
        notes.append("real-text lengths: Spec-Bench prompt shape, outputs capped-exponential around the mean (§20.2 L4)")
    if res.preemptions:
        notes.append(f"KV-limited: {res.preemptions} preemptions in the simulated run")
    if profile.provenance.get("source") == "cluster":
        notes.append(f"step profile measured on this cluster (image {profile.provenance.get('image_digest') or '?'})")
    stall = profile.cold_stall_ms()
    if stall >= COLD_STALL_NOTE_MS:
        notes.append(
            f"cold start: the first use of a batch shape stalled up to {stall / 1000.0:.1f} s in profiling (kernels "
            f"compiled on first use); the first requests at each new load run slower than predicted"
        )
    from .residual import load_bucket

    bucket = load_bucket(c, workload.workload_class)
    levels = {"hardware": "sku", "engine": "record", "model": "family"}

    def ground(q: str, value: float, ceiling: Optional[float]) -> Tuple[Estimate, float]:
        """The estimate of one quantity and the factor L5 applied to it."""
        corr = corrections.lookup(profile.sku, profile.engine, profile.model, q, bucket) if corrections else None
        if corr is None:
            return Estimate.around(value, band(q), ceiling=ceiling), 1.0
        levels[f"l5.{q}"] = corr.level
        if corr.gap:
            notes.append(
                f"model gap: {q} of {profile.model} on {profile.engine} at {bucket} needs x{corr.needed:.2f}, past the "
                f"x1.3 cap; uncorrected and ungrounded (FR-RES-3)"
            )
            est = Estimate.around(value, max(corr.rel, band(q)), ceiling=ceiling)
            return replace(est, ungrounded=True), 1.0
        return Estimate.around(value * corr.factor, max(corr.rel, interp), ceiling=ceiling), corr.factor

    ttft, f_ttft = ground("ttft", res.ttft_ms, ceiling_ttft)
    tpot, f_tpot = ground("tpot", res.tpot_ms, CEILINGS["tpot"])
    throughput, _ = ground("throughput", res.throughput_tok_s, CEILINGS["throughput"])
    if f_ttft != 1.0 or f_tpot != 1.0:
        notes.append(f"grounded (L5): TTFT x{f_ttft:.3f}, TPOT x{f_tpot:.3f} at {bucket}")
    # the end-to-end time moves with the corrected first token and decode
    e2e = res.percentile(res.e2e_samples, 0.5) / 1000.0
    decode_tokens = max(float(workload.output_tokens) - 1.0, 0.0)
    e2e += (ttft.value - res.ttft_ms) / 1000.0 + (tpot.value - res.tpot_ms) * decode_tokens / 1000.0

    def rel(est: Estimate) -> float:
        return est.high / est.value - 1.0 if est.value > 0 else 0.0

    return Prediction(
        ttft_ms=ttft,
        tpot_ms=tpot,
        throughput_tok_s=throughput,
        e2e_s=Estimate.around(max(e2e, 1e-6), max(rel(tpot), rel(ttft))),
        running=int(round(res.mean_running)),
        queueing=queueing,
        limits=lim,
        levels=levels,
        notes=notes,
    )


def find_profile(engine: Any, profiles: Any, settings: Any, model: Any, device: Any) -> Optional[StepProfile]:
    from .store import kernel_class

    sku = getattr(device, "sku", None)
    family = getattr(model, "family", None)
    if not (sku and family) or getattr(device, "hardware_mode", None) and device.hardware_mode.kind != "dedicated":
        return None  # a HAMi slice shares its compute with co-tenants: no profile measured it
    r = engine.resolve(settings, model, device)
    return profiles.find(sku, engine.record.id, family, kernel_class(r, model))
