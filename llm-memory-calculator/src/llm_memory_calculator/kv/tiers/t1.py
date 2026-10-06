"""T1, the engine pod's host memory: vLLM's OffloadingConnector CPU tier (FR-T1-*)."""

from __future__ import annotations

import math
from typing import FrozenSet, Optional, Tuple

from .. import constants as c
from ..constants import BREAK_EVEN_RATIO, MAX_LINK_UTILISATION
from ..context import PlanContext
from ..cost import BreakEven, TierLink, break_even, is_moe, short_prefix
from ..facts import NodeKVFacts

# An engine feature that would declare hybrid-model support in the offloading connector. vLLM 0.30's
# OffloadingConnector doesn't implement SupportsHMA (only the Mooncake store connector does), so no
# engine lists it yet.
HYBRID_OFFLOAD_FEATURE = "offload_hybrid"


def t1_mode(facts: NodeKVFacts, features: FrozenSet[str]) -> Tuple[Optional[str], str]:
    """Pick T1's host-memory mode for the node, or None with the reason (FR-T1-2)."""
    if facts.host_pointer_for_registered_mem is True:
        mode, why = "register", "the GPU uses registered host memory at its host address"
    elif facts.host_pointer_for_registered_mem is False and c.T1_PER_COPY_FEATURE in features:
        mode, why = (
            "register",
            "the GPU can't use registered host memory at its host address, and the engine copies it"
            f" one transfer at a time ('{c.T1_PER_COPY_FEATURE}')",
        )
    elif facts.host_pointer_for_registered_mem is False:
        mode, why = (
            "alloc",
            "the GPU can't use registered host memory at its host address (vGPU with UVM off)",
        )
    elif facts.vgpu is False:
        mode, why = "register", "bare-metal GPU (registered-memory fact not probed)"
    else:
        mode, why = "alloc", "registered-memory fact not probed on a vGPU or unknown node"
    feature = c.T1_MODE_FEATURES[mode]
    if feature not in features:
        return None, f"T1 dropped: {mode} mode ({why}) needs the engine feature '{feature}'"
    return mode, f"T1 mode: {mode} ({why})"


def staged_t1_ms(ctx: PlanContext) -> float:
    """The T1 load a tiered hit pays after its chunks reach T1 (vLLM's staged promotion): T1's own
    idle reload for the same prefix, or 0 when T1 wasn't evaluated."""
    verdict = ctx.plan.evaluations.get("T1")
    return (
        verdict.reload_idle_ms
        if verdict is not None and math.isfinite(verdict.reload_idle_ms)
        else 0.0
    )


def tiering_blocked(ctx: PlanContext) -> Optional[str]:
    """Why vLLM's tiering (a disk tier, or a shared-filesystem pool behind T1) can't run, or None.

    Its manager reads and writes T1 through the shared registered /dev/shm region, so T1 must be in
    register mode. alloc mode's private per-rank memory can't carry it."""
    if ctx.t1_mode == "alloc":
        return (
            "vLLM's tiering moves KV through T1's shared registered /dev/shm region, but T1 is in"
            " alloc mode here (private per rank) because the GPU can't use registered memory at its"
            " host address; it needs an engine that copies that region safely"
            f" ('{c.T1_PER_COPY_FEATURE}')"
        )
    return None


def hybrid_offload_blocked(ctx: PlanContext) -> Optional[str]:
    """Why the native offloading connector can't serve this model, or None."""
    if ctx.geometry.hybrid and HYBRID_OFFLOAD_FEATURE not in ctx.features:
        return (
            "the model has linear-attention or Mamba layers, and vLLM's offloading connector"
            " doesn't support hybrid models (no SupportsHMA in vLLM 0.30; the Mooncake connector"
            " does)"
        )
    return None


def keep_on_throughput(
    ctx: PlanContext, tier: str, verdict: BreakEven, saved: Optional[float], source: str
) -> bool:
    """Throughput decides T1 and T2 (the accuracy rounds' oracle): the link must have room, and a
    hit must save enough batched GPU work to pay for the tier's own copies. Loads run on the copy
    engine
    (and the disk) alongside the GPU's work, so a load no faster than one recompute still pays under
    load (Qwen3-1.7B T1 at 4K: +4%, +18%; Qwen3-8B T2 at 4K c16: +14% with a disk hit 1.6x slower
    than recompute). One load must beat one recompute only where TTFT binds: an interactive
    deployment, or when the batched saving can't be estimated (no device FLOPs)."""
    plan, decisions = ctx.plan, ctx.plan.decisions
    if verdict.recompute_ms is None:
        decisions.append(f"{tier} break-even not checked: {source}")
        return True

    def drop(reason: str, needed: bool = False) -> bool:
        decisions.append(f"{tier} dropped: {reason}")
        plan.drop(
            tier,
            reason,
            reload_ms=verdict.reload_ms,
            recompute_ms=verdict.recompute_ms,
            needed_gbps=verdict.needed_gbps if needed else None,
        )
        return False

    if verdict.utilisation >= MAX_LINK_UTILISATION:
        return drop(f"the link would saturate: {verdict.describe()}", needed=True)
    latency_ok = verdict.reload_idle_ms < BREAK_EVEN_RATIO * verdict.recompute_ms
    ttft_bound = ctx.workload.profile == "interactive" or saved is None
    if ttft_bound and not latency_ok:
        return drop(
            f"loading costs about as much as recomputing: {verdict.describe()}", needed=True
        )
    if saved is not None:
        moe = is_moe(ctx.model_config)
        floor = c.T1_MIN_RECOMPUTE_SAVED_MS_MOE if moe else c.T1_MIN_RECOMPUTE_SAVED_MS
        if saved < floor:
            return drop(
                f"a hit saves {saved:.0f} ms of batched GPU work (under {floor:g} ms"
                f"{' for a MoE' if moe else ''}): the tier's own copies on every request cost more"
            )
        if not latency_ok:
            decisions.append(
                f"{tier} kept for throughput: a hit saves {saved:.0f} ms of batched GPU work,"
                f" though one load takes about as long as one recompute ({verdict.describe()});"
                " hit TTFT doesn't improve"
            )
            return True
    decisions.append(f"{tier} break-even: {verdict.describe()}")
    return True


def plan_t1(ctx: PlanContext) -> float:
    """Plan T1: its size in GiB, total across ranks, or 0 with the reason."""
    plan, decisions = ctx.plan, ctx.plan.decisions
    if ctx.device_type != "cuda":
        decisions.append(
            f"T1 dropped: vLLM's offloading connector isn't used on '{ctx.device_type}'"
        )
        plan.drop("T1", f"not used on '{ctx.device_type}'")
        return 0.0
    if ctx.pp > 1:
        decisions.append(
            "T1 dropped: a multi-node (Ray) deployment can't share one host-memory region"
        )
        plan.drop("T1", "pipeline-parallel deployment")
        return 0.0
    blocked = hybrid_offload_blocked(ctx)
    if blocked:
        decisions.append(f"T1 dropped: {blocked}")
        plan.drop("T1", "hybrid model: the offloading connector doesn't support it")
        return 0.0
    facts = ctx.node
    if facts.pinned_offload_ok is not True:
        decisions.append(
            f"T1 dropped: pinned host offload not verified on this node ({facts.pinned_offload_ok})"
        )
        plan.drop("T1", "pinned host offload not verified on this node")
        return 0.0
    mode, why = t1_mode(facts, ctx.features)
    decisions.append(why)
    if mode is None:
        plan.drop("T1", why.replace("T1 dropped: ", ""))
        return 0.0
    host_free = ctx.host_free_gib
    if not host_free or host_free <= 0:
        decisions.append("T1 dropped: the node's free host memory is unknown")
        plan.drop("T1", "free host memory unknown")
        return 0.0

    pool_total = ctx.pool_gib * ctx.tp
    working_total = ctx.working_set_gib * ctx.tp
    floor_gib = c.T1_MIN_POOL_MULTIPLE * pool_total
    ceiling_gib = c.T1_MAX_POOL_MULTIPLE * pool_total
    wanted = min(max(working_total, floor_gib), ceiling_gib)
    available = c.T1_HOST_RAM_SHARE * host_free / ctx.replicas
    size = min(wanted, available)
    if available < wanted:
        limit = "the host memory"
    elif working_total > ceiling_gib:
        limit = f"{c.T1_MAX_POOL_MULTIPLE:g}x the GPU pool"
    elif working_total < floor_gib:
        limit = f"{c.T1_MIN_POOL_MULTIPLE:g}x the GPU pool"
    else:
        limit = "the working set"
    if size < pool_total:
        decisions.append(
            f"T1 dropped: {available:.1f} GiB of host memory per replica is less than"
            " the GPU pool it would back"
        )
        plan.drop("T1", "host memory per replica is less than the GPU pool")
        return 0.0
    size = math.floor(size * 4) / 4  # quarter-GiB steps keep plans stable

    # Break-even: one rank's share of the cached prefix over that GPU's pinned-copy link.
    bandwidth, source = facts.pinned_gbps()
    if mode == "alloc":
        efficiency, floor = c.T1_ALLOC_LOAD_EFFICIENCY, c.T1_ALLOC_FLOOR_MS
    elif (
        facts.host_pointer_for_registered_mem is False
    ):  # register mode only via per-copy transfers
        efficiency, floor = c.T1_PER_COPY_LOAD_EFFICIENCY, c.T1_PER_COPY_FLOOR_MS
    else:
        efficiency, floor = c.T1_LOAD_EFFICIENCY, c.T1_FLOOR_MS
    link = TierLink(
        tier="T1",
        gbps=round(bandwidth * efficiency, 3),
        floor_ms=floor,
        scope="rank",
        source=f"pinned copy {bandwidth:g} GB/s {source} x {efficiency}",
        efficiency=efficiency,
    )
    block = ctx.geometry.offload_block(c.T1_BLOCK_SIZE)
    tokens, nbytes = ctx.geometry.prefix_bytes(
        ctx.workload.reusable_prefix_tokens, block_size=block, scope="rank"
    )
    if tokens <= 0:
        reason = short_prefix(ctx.workload.reusable_prefix_tokens, block)
        decisions.append(f"T1 dropped: {reason}")
        plan.drop("T1", reason)
        return 0.0
    recompute_ms, recompute_source = ctx.recompute.ms(tokens)
    share = ctx.served_share("T1", size)
    verdict = break_even(
        tier="T1",
        cached_tokens=tokens,
        nbytes=nbytes,
        link=link,
        recompute_ms=recompute_ms,
        recompute_source=recompute_source,
        load_rate_per_s=ctx.request_rate_per_replica() * share,
    )
    plan.evaluations["T1"] = verdict
    saved = ctx.recompute.batched_ms(tokens)
    if not keep_on_throughput(ctx, "T1", verdict, saved, recompute_source):
        return 0.0

    entry = {
        "tier": "T1",
        "backend": "cpu",
        "size_gib": size,
        "storage_class": None,
        "transport": None,
        "params": {"host_memory": mode, "block_size": c.T1_BLOCK_SIZE},
    }
    ctx.tier_gib["T1"] = size
    ctx.t1_mode = mode
    decisions.append(f"T1: {size:g} GiB of pod host memory ({mode} mode), sized by {limit}")
    if ctx.accept(entry):
        plan.host_memory_gib = size
        plan.shm_gib = size if mode == "register" else 0.0
    return size
