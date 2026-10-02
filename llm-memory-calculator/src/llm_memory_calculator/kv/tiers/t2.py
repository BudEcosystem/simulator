"""T2, the node: vLLM's ``fs`` secondary tier on the cluster's node-local KV storage class
(FR-T2-*).

The fs tier sits behind the CPU tier (only the CPU tier touches the GPU), so T2 needs T1. Every
replica of the deployment shares it, because they all run on one node. The tier never evicts, so
budcluster enforces the planned ``size_gib`` (FR-T2-3).
"""

from __future__ import annotations

import math

from .. import constants as c
from ..context import PlanContext
from ..cost import TierLink, break_even, short_prefix
from .t1 import hybrid_offload_blocked, keep_on_throughput, staged_t1_ms, tiering_blocked

T2_PROFILES = ("agentic", "chat", "interactive")


def plan_t2(ctx: PlanContext) -> float:
    """Plan T2: its size in GiB for the deployment, or 0 with the reason."""
    plan, decisions = ctx.plan, ctx.plan.decisions

    def drop(reason: str, **numbers: float) -> float:
        decisions.append(f"T2 dropped: {reason}")
        plan.drop("T2", reason, **numbers)
        return 0.0

    if ctx.device_type != "cuda":
        return drop(f"vLLM's offloading connector isn't used on '{ctx.device_type}'")
    blocked = hybrid_offload_blocked(ctx)
    if blocked:
        return drop(blocked)
    if c.T2_FEATURE not in ctx.features:
        return drop(f"the engine doesn't list '{c.T2_FEATURE}'")
    if not ctx.tier_gib.get("T1"):
        return drop(
            "the node tier sits behind T1 (the offloading connector's CPU tier), which isn't"
            " planned"
        )
    blocked = tiering_blocked(ctx)
    if blocked:
        return drop(blocked)
    if ctx.replicas < 2 and ctx.workload.profile not in T2_PROFILES:
        return drop(
            f"one replica and a {ctx.workload.profile} profile: nothing to share across replicas"
        )
    storage = ctx.cluster.kv_storage()
    if storage is None:
        return drop("the cluster has no KV storage class (the setting is off or names no class)")
    if storage.node_local is not True:
        return drop(f"storage class '{storage.name}' isn't known to be node-local")
    if not storage.read_gbps:
        return drop(f"storage class '{storage.name}' has no measured read rate")
    capacity = storage.capacity_gib or ctx.node.local_nvme_gib
    if not capacity:
        return drop("the node's local disk capacity is unknown")

    per_token = ctx.geometry.bytes_per_token_rank * ctx.tp
    wanted = ctx.replicas * ctx.working_set_gib * ctx.tp
    cap = c.T2_DISK_SHARE * capacity
    size = math.floor(min(max(wanted, ctx.tier_gib["T1"]), cap) * 4) / 4
    if size <= 0 or per_token <= 0:
        return drop("no disk room for the working set")

    link = TierLink(
        tier="T2",
        gbps=storage.read_gbps,
        floor_ms=c.T2_FLOOR_MS,
        scope="total",
        source=f"class {storage.name} measured",
    )
    block = ctx.geometry.offload_block(c.T1_BLOCK_SIZE)
    tokens, nbytes = ctx.geometry.prefix_bytes(
        ctx.workload.reusable_prefix_tokens, block_size=block, scope="total"
    )
    if tokens <= 0:
        return drop(short_prefix(ctx.workload.reusable_prefix_tokens, block))
    recompute_ms, recompute_source = ctx.recompute.ms(tokens)
    if recompute_ms is None:
        return drop(f"break-even can't be checked: {recompute_source}")
    node_rate = ctx.request_rate_per_replica() * ctx.replicas
    verdict = break_even(
        tier="T2",
        cached_tokens=tokens,
        nbytes=nbytes,
        link=link,
        recompute_ms=recompute_ms,
        recompute_source=recompute_source,
        load_rate_per_s=node_rate * ctx.served_share("T2", size),
        write_bytes_per_s=ctx.write_bytes_per_s("T2", size, node_rate),
        staged_ms=staged_t1_ms(ctx),
    )
    plan.evaluations["T2"] = verdict
    if not keep_on_throughput(
        ctx, "T2", verdict, ctx.recompute.batched_ms(tokens), recompute_source
    ):
        return 0.0
    entry = {
        "tier": "T2",
        "backend": "fs",
        "size_gib": size,
        "storage_class": storage.name,
        "transport": None,
        "params": {"block_size": c.T1_BLOCK_SIZE},
    }
    ctx.tier_gib["T2"] = size
    limit = "the working set" if size >= wanted else f"{c.T2_DISK_SHARE:.0%} of the local disk"
    decisions.append(
        f"T2: {size:g} GiB on '{storage.name}' for {ctx.replicas} replica(s), sized by {limit}"
    )
    ctx.accept(entry)
    return size
