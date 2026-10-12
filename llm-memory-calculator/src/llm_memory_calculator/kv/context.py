"""Everything one planning run knows, derived once and passed to every step."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional

from . import constants
from .constants import DECODE_MS_PER_TOKEN, GIB
from .cost import RecomputeModel
from .facts import ClusterKVFacts, NodeKVFacts
from .geometry import KVGeometry
from .plan import KVTierPlan
from .reuse import CURVES, ReuseCurve
from .workload import KVWorkload


@dataclass
class PlanContext:
    """The inputs of one plan, plus what earlier steps decided."""

    plan: KVTierPlan
    model_config: Mapping[str, Any]
    workload: KVWorkload
    features: FrozenSet[str]
    geometry: KVGeometry
    recompute: RecomputeModel
    node: NodeKVFacts
    cluster: ClusterKVFacts
    device_type: str
    hardware_mode: str
    device_memory_gib: float
    weight_gib: float
    other_gib: float
    demand_gib: float
    tp: int
    pp: int
    replicas: int
    host_free_gib: Optional[float]
    device_free_gib: Optional[float]
    gpu_kv_cap_gib: Optional[float]
    cross_node_need: bool = False
    e2e_latency_s: Optional[float] = None
    model_id: Optional[str] = None
    # -- filled in as the plan proceeds
    pool_gib: float = 0.0  # T0 KV pool per rank
    working_set_gib: float = 0.0  # per rank
    tier_gib: Dict[str, float] = field(default_factory=dict)  # planned offload size, all ranks
    t1_mode: Optional[str] = None  # register | alloc, once T1 is planned
    # ms of a request's critical path per stored token: T1 writes every computed block through
    # (``t1.store_ms``)
    t1_store_ms_per_token: float = 0.0
    t1_store_copies: str = ""  # how T1 copies, for the decision text

    # ---------------------------------------------------------------- reuse coverage

    @property
    def curve(self) -> Optional[ReuseCurve]:
        """The workload's measured reuse curve, if it has one and the request rate is known (1.9
        A2)."""
        if not self.workload.curve or self.traffic_rate() <= 0 or self.workload.input_tokens <= 0:
            return None
        return CURVES.get(self.workload.curve)

    def traffic_rate(self) -> float:
        """Requests per second one replica receives: concurrency over the end-to-end latency
        (Little's law), with budsim's latency when given, else the prefill estimate plus
        ``DECODE_MS_PER_TOKEN`` per output token.
        """
        e2e = self.e2e_latency_s
        if not e2e or e2e <= 0:
            prefill_ms, _ = self.recompute.ms(self.workload.input_tokens)
            e2e = (
                prefill_ms or 0.0
            ) / 1e3 + self.workload.output_tokens * DECODE_MS_PER_TOKEN / 1e3
        if not e2e or e2e <= 0 or not math.isfinite(e2e):
            return 0.0
        return self.workload.concurrency / e2e

    def retained_seconds(self, tokens: float) -> float:
        """Seconds of one replica's traffic a cache of ``tokens`` holds (the curves' x axis)."""
        per_second = self.traffic_rate() * self.workload.input_tokens
        return tokens / per_second if per_second > 0 else 0.0

    @property
    def working_set_tokens(self) -> int:
        """Distinct tokens one replica should keep. With a curve: what holds ``REUSE_TARGET`` of its
        reachable hit at this deployment's request rate. Without one: the analytic estimate
        (``KVWorkload``)."""
        curve = self.curve
        if curve is None:
            return self.workload.working_set_tokens
        seconds = curve.seconds_for(constants.REUSE_TARGET)
        return int(seconds * self.traffic_rate() * self.workload.input_tokens)

    def curve_hit(self, tokens: float) -> float:
        """Share of prompt tokens a cache of ``tokens`` serves, from the curve (callers check
        ``curve``)."""
        return self.curve.hit(self.retained_seconds(tokens))

    def tokens_per_replica(self, tier: str, gib: Optional[float] = None) -> float:
        """Tokens of one replica's working set a tier of ``gib`` holds. T1 is per replica; T2 and T3
        are shared by every replica, each holding its own sessions."""
        size = self.tier_gib.get(tier, 0.0) if gib is None else gib
        per_token = self.geometry.bytes_per_token_rank * self.tp
        if size <= 0 or per_token <= 0:
            return 0.0
        tokens = size * GIB / per_token
        return tokens / self.replicas if tier in ("T2", "T3") else tokens

    def pool_tokens(self) -> float:
        per = self.geometry.bytes_per_token_rank
        return self.pool_gib * GIB / per if per > 0 else 0.0

    def coverage(self, tokens: float) -> float:
        curve = self.curve
        if curve is not None:
            return min(1.0, self.curve_hit(tokens) / curve.ceiling) if curve.ceiling > 0 else 1.0
        working = self.working_set_tokens
        return min(1.0, tokens / working) if working > 0 else 1.0

    def added_tokens(self, tier: str, gib: Optional[float] = None) -> float:
        """Tokens of one replica's working set a tier adds beyond the faster tiers.

        vLLM's offloading connector writes every block a request computes through to its CPU tier
        (T1), and its fs secondary tier (T2) stores every block T1 stores (accubits01, 2026-10-06:
        the disk tier wrote exactly the GPU-to-CPU bytes). So each holds the faster tiers' blocks
        too and adds only what is beyond them: T1 beyond the GPU pool, T2 beyond T1. TCS D1
        (2026-10-09, Qwen3-8B, a 55 GiB pool and a 114 GiB T1) hit what an LRU of 114 GiB hits, not
        169; accubits01's pool, T1 and T2 together held what T2 alone holds. T3 is a separate
        connector and adds its whole size.
        """
        tokens = self.tokens_per_replica(tier, gib)
        if tier == "T1":
            return max(0.0, tokens - self.pool_tokens())
        if tier == "T2":
            return max(0.0, tokens - max(self.tokens_per_replica("T1"), self.pool_tokens()))
        return tokens

    def offload_share(self) -> float:
        """Share of what T1 and T2 hold that a hit can use: the offloading connector moves whole
        blocks, so every prefix it serves loses its partial last block, half a block on average,
        counted against the prompt. TCS D1 and accubits01 D2 lost 3-7% of the tiers' hits (half a
        block of the prompt: 4-14%)."""
        block = self.geometry.offload_block(constants.T1_BLOCK_SIZE)
        prompt = self.workload.input_tokens
        return max(0.0, 1.0 - block / 2.0 / prompt) if prompt > 0 else 0.0

    def cached_share(self, tokens: float) -> float:
        """Share of prompt tokens a cache of ``tokens`` distinct tokens serves, before block
        granularity."""
        if self.curve is not None:
            return self.curve_hit(tokens)
        return self.workload.reusable_share * self.coverage(tokens)

    def tokens_through(self, tier: Optional[str], gib: Optional[float] = None) -> float:
        """Tokens held by the GPU pool and every planned offload tier up to and including
        ``tier`` (None: the pool alone), with ``gib`` as that tier's size."""
        held = self.pool_tokens()
        for t in ("T1", "T2", "T3"):
            if tier is None or t > tier:
                break
            held += self.added_tokens(t, gib if t == tier else None)
        return held

    def hit_share_with(self, tier: str, gib: float) -> float:
        """Share of prompt tokens served from cache by every tier up to and including ``tier``. What
        T1 and T2 add over the GPU pool counts at ``offload_share``; T3 (its own connector) counts
        in full."""
        pool = self.cached_share(self.pool_tokens())
        node = self.tokens_through(tier if tier != "T3" else "T2", gib if tier != "T3" else None)
        share = pool + self.offload_share() * (self.cached_share(node) - pool)
        if tier == "T3":
            share += self.cached_share(self.tokens_through("T3", gib)) - self.cached_share(node)
        return share

    def write_bytes_per_s(self, tier: str, gib: float, rate_per_s: float) -> float:
        """Write-through traffic into a node-shared tier: every request writes the blocks it
        computed (all ranks' shares)."""
        computed = 1.0 - self.hit_share_with(tier, gib)
        per_token = self.geometry.bytes_per_token_rank * self.tp
        return rate_per_s * self.workload.input_tokens * computed * per_token

    def served_share(self, tier: str, gib: float) -> float:
        """Share of requests whose reusable prefix this tier serves: the reusable share times the
        part of the working set it holds beyond the faster tiers already planned."""
        previous = {"T1": None, "T2": "T1", "T3": "T2"}[tier]
        before = self.tokens_through(previous)
        after = before + self.added_tokens(tier, gib)
        usable = self.offload_share() if tier in ("T1", "T2") else 1.0
        if self.curve is not None:  # the curve already counts every prefix's first use as a miss
            return usable * (self.curve_hit(after) - self.curve_hit(before))
        share = self.workload.reusable_share * self.workload.repeat_share
        return usable * share * (self.coverage(after) - self.coverage(before))

    # --------------------------------------------------------------------- load rate

    def running_requests(self) -> float:
        """Requests one replica runs at once: the deployment's concurrency, but no more than its GPU
        KV pool holds (each running request keeps its whole prompt and output there; vLLM queues the
        rest). Round 4: Qwen1.5-MoE at 16K and c16 had ~2.6 requests in an 8 GiB pool, so its T1
        link ran ~40% busy where 16 in flight predicted over 80%, and T1 (+54%) was dropped."""
        concurrency = float(self.workload.concurrency)
        per_request = self.workload.input_tokens + self.workload.output_tokens
        pool = self.pool_tokens()
        if pool > 0 and per_request > 0:
            concurrency = min(concurrency, max(1.0, pool / per_request))
        return concurrency

    def request_rate_per_replica(self) -> float:
        """Requests per second one replica serves: concurrency over the end-to-end latency (budsim's
        prediction when given, else the prefill estimate plus ``DECODE_MS_PER_TOKEN`` per token),
        and never more than the GPU can prefill: the part of each prompt no cache can serve takes
        GPU time on every request."""
        e2e = self.e2e_latency_s
        if not e2e or e2e <= 0:
            prefill_ms, _ = self.recompute.ms(self.workload.input_tokens)
            decode_s = self.workload.output_tokens * DECODE_MS_PER_TOKEN / 1e3
            e2e = (prefill_ms or 0.0) / 1e3 + decode_s
        if not e2e or e2e <= 0 or not math.isfinite(e2e):
            return 0.0
        rate = self.running_requests() / e2e
        unique = int(self.workload.input_tokens * (1.0 - self.workload.reusable_share))
        unique_ms, _ = self.recompute.ms(unique)
        if unique_ms and unique_ms > 0:
            rate = min(rate, 1e3 / unique_ms)
        return rate

    # ------------------------------------------------------------------ release ceiling

    def accept(self, entry: Dict[str, Any]) -> bool:
        """Add a planned tier to the plan if this package release renders it (``RELEASED_TIERS``),
        else keep it as withheld and say so. Later tiers are sized with it either way."""
        tier = entry["tier"]
        if tier in constants.RELEASED_TIERS:
            self.plan.tiers.append(entry)
            return True
        self.plan.withheld.append(entry)
        self.plan.decisions.append(
            f"{tier} planned but withheld: this package release returns only"
            f" {', '.join(sorted(constants.RELEASED_TIERS)) or 'T0'}"
        )
        return False
