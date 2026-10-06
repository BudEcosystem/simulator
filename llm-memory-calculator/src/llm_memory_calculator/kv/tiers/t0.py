"""T0, GPU memory: the dedicated-card fraction (FR-T0-1), a shared slice's KV growth toward the
working set (FR-T0-3), and a CPU engine's KV space (FR-T0-4)."""

from __future__ import annotations

import math
from typing import Optional

from .. import constants as c
from ..context import PlanContext


def _gib(value: Optional[float]) -> float:
    return float(value) if value and value > 0 else 0.0


def _round_gib(value: float) -> float:
    return math.ceil(value * 100) / 100


def plan_t0(ctx: PlanContext) -> float:
    """Decide the T0 KV pool per rank, in GiB, and record how."""
    plan, decisions = ctx.plan, ctx.plan.decisions
    demand_gib = ctx.demand_gib
    working_set_gib = ctx.working_set_gib
    if ctx.device_type in ("cpu", "cpu_high"):
        # FR-T0-4: a CPU engine's KV pool is host memory.
        host_free = ctx.host_free_gib
        cap = c.CPU_KV_SHARE_MAX * _gib(host_free) / (ctx.replicas * ctx.tp) if host_free else 0.0
        if working_set_gib > demand_gib and cap > 0:
            grown = demand_gib + min(working_set_gib - demand_gib, cap)
            plan.cpu_kv_gib = _round_gib(grown)
            decisions.append(
                f"T0 (CPU): KV space {demand_gib:.2f} -> {plan.cpu_kv_gib} GiB"
                " toward the working set"
            )
            return grown
        if working_set_gib > demand_gib:
            decisions.append("T0 (CPU): KV space stays at demand (free host memory unknown)")
        return demand_gib
    if ctx.device_type != "cuda":
        decisions.append(f"T0: no headroom on '{ctx.device_type}'")
        return demand_gib
    if ctx.hardware_mode == "shared":
        return _plan_slice(ctx, demand_gib, working_set_gib)
    # FR-T0-1: a dedicated card's leftover memory goes to the KV pool.
    plan.gpu_memory_utilization = c.T0_GPU_UTIL
    pool = demand_gib
    if ctx.device_memory_gib:
        pool = max(
            demand_gib,
            c.T0_GPU_UTIL * ctx.device_memory_gib
            - ctx.weight_gib
            - ctx.other_gib
            - c.DEVICE_OVERHEAD_GIB,
        )
    decisions.append(
        f"T0 (dedicated): --gpu-memory-utilization {c.T0_GPU_UTIL},"
        f" KV pool ~{pool:.1f} GiB per rank"
    )
    return pool


def _plan_slice(ctx: PlanContext, demand_gib: float, working_set_gib: float) -> float:
    """FR-T0-3: grow the slice's KV part toward the working set, by at most a share of the GPU's
    free memory, and never past what still places one slice per replica."""
    plan, decisions = ctx.plan, ctx.plan.decisions
    if working_set_gib <= demand_gib:
        decisions.append("T0 (slice): the working set fits the demand-sized slice")
        return demand_gib
    free = ctx.device_free_gib
    if not free or free <= 0:
        decisions.append("T0 (slice): no growth (the GPU's free memory is unknown)")
        return demand_gib
    # Replicas pinned to the node may land on the same card, so they share the cap.
    growth = min(working_set_gib - demand_gib, c.SLICE_KV_SHARE_MAX * free / ctx.replicas)
    room = (
        c.GPU_UTIL_MAX * ctx.device_memory_gib
        - ctx.weight_gib
        - ctx.other_gib
        - c.DEVICE_OVERHEAD_GIB
        - demand_gib
    )
    growth = max(0.0, min(growth, room)) if ctx.device_memory_gib else growth
    if growth <= 0:
        decisions.append("T0 (slice): no room on the card to grow the slice")
        return demand_gib
    cap = ctx.gpu_kv_cap_gib
    if cap is not None and demand_gib + growth > cap:
        # The cap comes in 0.01 GiB steps; rounding up past it would build a slice that doesn't fit.
        capped = math.floor(cap * 100 + 1e-6) / 100
        if capped <= demand_gib:
            decisions.append(
                "T0 (slice): no growth (a grown slice per replica doesn't fit the free cards)"
            )
            return demand_gib
        plan.gpu_kv_gib = capped
        decisions.append(
            f"T0 (slice): KV {demand_gib:.2f} -> {capped} GiB, as much as still fits one slice"
            f" per replica ({ctx.replicas}) on the free cards"
        )
        return capped
    plan.gpu_kv_gib = _round_gib(demand_gib + growth)
    decisions.append(
        f"T0 (slice): KV {demand_gib:.2f} -> {plan.gpu_kv_gib} GiB,"
        f" at most {c.SLICE_KV_SHARE_MAX:.0%} of "
        f"the GPU's {free:.1f} GiB free, shared by {ctx.replicas} replica(s)"
    )
    return demand_gib + growth
