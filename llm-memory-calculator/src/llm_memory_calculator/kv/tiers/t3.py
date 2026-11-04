"""T3, the cluster tier: a Mooncake Store pool, or a fast shared filesystem (FR-T3-*).

Planned only for a cross-node need: since revision 3, a deployment whose replicas spread over more
nodes than one holds (``PlanContext.cross_node_need``). The pool's bandwidth is the rate its health gate measured
through Mooncake, scaled by how much of it vLLM's connector reaches at this model's per-layer
segment size, and shared by the deployment's loads and write-through. Spike S10 on TCS: the pool
worked over TCP but lost to recompute for Qwen3-8B by ~10x, and for the Qwen3.8-27B hybrid under
load; the fixtures in :mod:`..calibration` hold the planner to those outcomes.
"""

from __future__ import annotations

import math

from .. import constants as c
from ..context import PlanContext
from ..cost import TierLink, break_even, connector_efficiency, short_prefix
from .t1 import hybrid_offload_blocked, staged_t1_ms, tiering_blocked


def plan_t3(ctx: PlanContext) -> float:
    """Plan T3: the share of the pool it expects to use, in GiB, or 0 with the reason."""
    plan, decisions = ctx.plan, ctx.plan.decisions

    def drop(reason: str, **numbers: float) -> float:
        decisions.append(f"T3 dropped: {reason}")
        plan.drop("T3", reason, **numbers)
        return 0.0

    if not ctx.cross_node_need:
        return drop(f"no cross-node need: {ctx.cross_node_reason or 'every replica runs on one node'}")
    pool = ctx.cluster.t3
    if pool is None:
        return drop("the cluster has no T3 backend")
    if ctx.device_type != "cuda":
        return drop(f"not on '{ctx.device_type}'")
    # FR-CAP-10: a spread deployment plans on the slowest pool rate among the nodes its replicas can land on
    read_gbps, rate_note = pool.rate_for(ctx.spread_nodes)

    geometry = ctx.geometry
    if pool.backend == "mooncake":
        if c.T3_FEATURE not in ctx.features:
            return drop(f"the engine doesn't list '{c.T3_FEATURE}'")
        transport = pool.transport or "tcp"
        if transport == "rdma" and not ctx.node.gpudirect():
            return drop("an RDMA pool needs GPUDirect RDMA on this node, which isn't verified")
        block = geometry.block_size
        segment = geometry.segment_bytes(block)
        efficiency = (
            c.T3_RDMA_EFFICIENCY
            if transport == "rdma"
            else connector_efficiency(segment, c.T3_CONNECTOR_EFFICIENCY)
        )
        floor_ms = pool.floor_ms if pool.floor_ms is not None else c.T3_FLOOR_MS[transport]
        source = (
            f"pool {read_gbps:g} GB/s measured{rate_note} x {efficiency:.2f} connector efficiency at"
            f" {segment / 1024:.0f} KiB segments, {transport}"
            if read_gbps
            else "pool rate unmeasured"
        )
    elif pool.backend == "fs":
        blocked = hybrid_offload_blocked(ctx)
        if blocked:
            return drop(blocked)
        if c.T2_FEATURE not in ctx.features or not ctx.tier_gib.get("T1"):
            return drop("a shared-filesystem T3 sits behind T1, which isn't planned")
        blocked = tiering_blocked(ctx)
        if blocked:
            return drop(blocked)
        transport, efficiency = None, 1.0
        block = geometry.offload_block(c.T1_BLOCK_SIZE)
        floor_ms = pool.floor_ms if pool.floor_ms is not None else c.T2_FLOOR_MS
        source = f"shared filesystem {read_gbps:g} GB/s measured{rate_note}" if read_gbps else ""
    else:
        return drop(f"unknown T3 backend '{pool.backend}'")
    if not read_gbps:
        return drop(rate_note or "the pool's read rate was never measured")
    per_token = geometry.bytes_per_token_rank * ctx.tp
    # FR-PLAN-19: the pool holds every replica the autoscaler may run
    wanted = ctx.peak * ctx.working_set_gib * ctx.tp
    size = wanted
    if pool.capacity_gib:
        size = min(size, c.T3_POOL_SHARE_MAX * pool.capacity_gib)
    size = math.floor(size * 4) / 4
    if size <= 0 or per_token <= 0:
        return drop("no pool room for the working set")

    link = TierLink(
        tier="T3",
        gbps=round(read_gbps * efficiency, 3),
        floor_ms=floor_ms,
        scope="total",
        source=source,
        efficiency=efficiency,
    )
    tokens, nbytes = geometry.prefix_bytes(
        ctx.workload.reusable_prefix_tokens, block_size=block, scope="total"
    )
    if tokens <= 0:
        return drop(short_prefix(ctx.workload.reusable_prefix_tokens, block))
    recompute_ms, recompute_source = ctx.recompute.ms(tokens)
    if recompute_ms is None:
        return drop(f"break-even can't be checked: {recompute_source}")
    node_rate = ctx.request_rate_per_replica() * ctx.peak
    # FR-T3-6: the pool already moves other deployments' traffic over the same link
    others = (pool.busy_gbps or 0.0) * 1e9
    verdict = break_even(
        tier="T3",
        cached_tokens=tokens,
        nbytes=nbytes,
        link=link,
        recompute_ms=recompute_ms,
        recompute_source=recompute_source,
        load_rate_per_s=node_rate * ctx.served_share("T3", size),
        write_bytes_per_s=ctx.write_bytes_per_s("T3", size, node_rate) + others,
        staged_ms=staged_t1_ms(ctx) if pool.backend == "fs" else 0.0,
    )
    plan.evaluations["T3"] = verdict
    if not verdict.keep:
        return drop(
            f"loses break-even: {verdict.describe()}",
            reload_ms=verdict.reload_ms,
            recompute_ms=verdict.recompute_ms,
            needed_gbps=verdict.needed_gbps,
        )
    else:
        decisions.append(f"T3 break-even: {verdict.describe()}")
    if pool.backend == "mooncake":
        # FR-FAIL-7: a get that outlives the master's read lease never finishes
        lease_ms = pool.lease_ms or c.T3_DEFAULT_LEASE_MS
        longest, longest_bytes = geometry.prefix_bytes(ctx.workload.input_tokens, block_size=block, scope="total")
        alone_s = link.floor_ms / 1e3 + longest_bytes / (link.gbps * 1e9)
        worst_ms = alone_s * 1e3 / (1.0 - verdict.utilisation) if verdict.utilisation < 1.0 else math.inf
        if worst_ms > c.T3_LEASE_SHARE * lease_ms:
            return drop(
                f"a {longest}-token prompt ({longest_bytes / 1e6:.0f} MB) reloads in {worst_ms:.0f} ms under load,"
                f" over {c.T3_LEASE_SHARE:.0%} of the pool's {lease_ms:.0f} ms read lease: a get that outlives it"
                " fails, and vLLM retries it until the client gives up",
                reload_ms=worst_ms,
            )
    entry = {
        "tier": "T3",
        "backend": pool.backend,
        "size_gib": size,
        "storage_class": pool.storage_class if pool.backend == "fs" else None,
        "transport": transport,
        "params": {"block_size": block, "load_failure_policy": "recompute"},
    }
    if pool.backend == "mooncake":
        entry["params"].update(
            master_server_address=pool.master or c.T3_DEFAULT_MASTER,
            protocol=transport,
            local_buffer_size_mib=c.T3_LOCAL_BUFFER_MIB,
        )
    ctx.tier_gib["T3"] = size
    decisions.append(
        f"T3: {pool.backend} ({transport or 'shared filesystem'}), planning on {size:g} GiB of it"
    )
    accepted = ctx.accept(entry)
    if accepted and pool.backend == "mooncake":
        # each rank's store client pins its buffers in host memory, beside T1 (FR-RENDER-4)
        plan.host_memory_gib = round(plan.host_memory_gib + c.T3_HOST_GIB_PER_RANK * ctx.tp, 2)
    if accepted and transport == "rdma":
        plan.rdma = True
    return size
