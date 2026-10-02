"""Predicted hit rate and hit-aware TTFT across the released tiers (FR-PLAN-8).

Informational: nothing is sized from it until spike S6 checks it against measurements. The hit
model is proportional (a tier holding x% of the working set serves x% of its reuse), which
overstates hits when a working set larger than the cache is re-scanned in order (an LRU then misses
every time; accubits01 1.2 finding).
"""

from __future__ import annotations

from .constants import GB
from .context import PlanContext


def predict(ctx: PlanContext) -> None:
    plan = ctx.plan
    working = ctx.working_set_tokens
    if working <= 0:
        return
    # only a prefix's repeat uses can hit: its first use misses in every tier
    share = ctx.workload.reusable_share * ctx.workload.repeat_share
    released = {t["tier"] for t in plan.tiers}
    cumulative = ctx.pool_tokens()
    gpu_hit = share * ctx.coverage(cumulative)
    total_hit = gpu_hit
    load_ms = 0.0
    for tier in ("T1", "T2", "T3"):
        if tier not in released:
            continue
        before = ctx.coverage(cumulative)
        cumulative += ctx.tokens_per_replica(tier)
        tier_share = share * (ctx.coverage(cumulative) - before)
        total_hit += tier_share
        verdict = plan.evaluations.get(tier)
        if verdict is not None and verdict.link.gbps > 0:
            per_token = verdict.bytes / verdict.cached_tokens if verdict.cached_tokens else 0.0
            tokens = tier_share * ctx.workload.input_tokens
            load_ms += verdict.link.floor_ms * (tier_share / share if share else 0.0)
            load_ms += tokens * per_token / (verdict.link.gbps * GB) * 1e3
    plan.predicted_hit_rate = round(total_hit, 3)
    prefill_ms, _ = ctx.recompute.ms(int(ctx.workload.input_tokens * (1 - total_hit)))
    if prefill_ms is None:
        return
    plan.predicted_ttft_ms = round(prefill_ms + load_ms, 1)
