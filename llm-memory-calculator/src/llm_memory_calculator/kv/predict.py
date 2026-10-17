"""Predicted hit rate and hit-aware TTFT across the released tiers (FR-PLAN-8).

Informational: nothing is sized from it. Checked against hardware in IMPLEMENTATION_PLAN 1.9 D1/D2
(Bud's ``spikes/reuse-traces``):

- **Hit rate.** The cache holds what its largest tier holds: T1 holds the GPU pool's blocks and T2
  holds T1's (``PlanContext.added_tokens``), and what T1 and T2 add over the pool counts at
  ``offload_share`` (whole blocks only). Within 0.01 of measured on TCS (Qwen3-8B) and accubits01
  (Qwen3-0.6B), Mooncake windows.
- **TTFT** is the request's own prefill of what misses, plus its tier loads, plus T1's write-through
  stores on its path, plus queueing for the GPU: Kingman's G/G/1 approximation of the prefill work
  at the deployment's request rate, with the work's spread (``prompt_cv2``) and the arrivals'
  burstiness (``arrival_cv2``) from the reuse curve's trace. Production arrivals are bursty (the
  Mooncake traces: 8 and 15 times Poisson's variance per second), and the burstiness, not the load,
  set most of TCS D1's wait. Decode work isn't counted in the queue: long outputs wait longer than
  this. The prefill estimate itself (GenZ) ran about 15% low on TCS's H100, and queueing amplifies
  that.
"""

from __future__ import annotations

from . import constants
from .constants import GB
from .context import PlanContext


def predict(ctx: PlanContext) -> None:
    plan = ctx.plan
    working = ctx.working_set_tokens
    if working <= 0:
        return
    # only a prefix's repeat uses can hit: its first use misses in every tier. A measured curve
    # already counts that, and its coverage is relative to its own ceiling.
    if ctx.curve is not None:
        share = ctx.curve.ceiling
    else:
        share = ctx.workload.reusable_share * ctx.workload.repeat_share
    released = {t["tier"] for t in plan.tiers}
    cumulative = ctx.pool_tokens()
    total_hit = share * ctx.coverage(cumulative)
    usable = ctx.offload_share()
    load_ms = 0.0
    for tier in ("T1", "T2", "T3"):
        if tier not in released:
            continue
        before = ctx.coverage(cumulative)
        cumulative += ctx.added_tokens(tier)
        tier_share = share * (ctx.coverage(cumulative) - before)
        if tier in ("T1", "T2"):
            tier_share *= usable
        total_hit += tier_share
        verdict = plan.evaluations.get(tier)
        if verdict is not None and verdict.link.gbps > 0:
            per_token = verdict.bytes / verdict.cached_tokens if verdict.cached_tokens else 0.0
            tokens = tier_share * ctx.workload.input_tokens
            load_ms += verdict.link.floor_ms * (tier_share / share if share else 0.0)
            load_ms += tokens * per_token / (verdict.link.gbps * GB) * 1e3
    plan.predicted_hit_rate = round(total_hit, 3)
    computed = int(ctx.workload.input_tokens * (1 - total_hit))
    prefill_ms, _ = ctx.recompute.ms(computed)
    if prefill_ms is None:
        return
    store_ms = ctx.t1_store_ms_per_token * computed if "T1" in released else 0.0
    alone_ms = prefill_ms + load_ms + store_ms
    # The GPU serves every request's prefill and the blocking part of its stores; loads run on the
    # copy engine.
    service_s = (prefill_ms + store_ms) / 1e3
    rate = ctx.traffic_rate()
    busy = rate * service_s
    if rate <= 0 or service_s <= 0:
        plan.predicted_ttft_ms = round(alone_ms, 1)
        return
    if busy >= constants.TTFT_MAX_UTILISATION:
        plan.predicted_ttft_ms = None
        plan.decisions.append(
            f"predicted TTFT: none; prefill alone keeps the GPU {busy:.0%} busy at {rate:.2f} req/s"
            f" ({service_s * 1e3:.0f} ms a request), so queues grow without bound"
        )
        return
    curve = ctx.curve
    cs2 = curve.prompt_cv2 if curve is not None else constants.DEFAULT_PROMPT_CV2
    ca2 = constants.DEFAULT_ARRIVAL_CV2
    if curve is not None:
        # a deployment seeing a fraction of the trace's rate sees its bursts thinned
        share = min(1.0, rate / curve.arrival_rate) if curve.arrival_rate > 0 else 1.0
        ca2 = 1.0 + share * (curve.arrival_cv2 - 1.0)
    # Kingman's G/G/1 approximation; with Poisson arrivals (ca2 = 1) it is Pollaczek-Khinchine
    wait_ms = busy / (1.0 - busy) * (ca2 + cs2) / 2.0 * service_s * 1e3
    plan.predicted_ttft_ms = round(alone_ms + wait_ms, 1)
    plan.decisions.append(
        f"predicted TTFT: {alone_ms + wait_ms:.0f} ms ({alone_ms:.0f} ms alone"
        + (f", {store_ms:.0f} of it T1 stores" if store_ms >= 1 else "")
        + f"; {wait_ms:.0f} ms queueing with prefill keeping the GPU {busy:.0%} busy"
        f" at {rate:.2f} req/s,"
        f" work spread cv² {cs2:g}, arrivals {ca2:.2g}x Poisson's variance)"
    )
