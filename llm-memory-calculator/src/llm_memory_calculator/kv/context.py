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

    # ---------------------------------------------------------------- reuse coverage

    @property
    def working_set_tokens(self) -> int:
        return self.workload.working_set_tokens

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
        working = self.working_set_tokens
        return min(1.0, tokens / working) if working > 0 else 1.0

    def hit_share_with(self, tier: str, gib: float) -> float:
        """Share of prompt tokens served from cache by every tier up to and including ``tier``."""
        tokens = self.pool_tokens() + sum(
            self.tokens_per_replica(t) for t in ("T1", "T2", "T3") if t < tier
        )
        tokens += self.tokens_per_replica(tier, gib)
        return self.workload.reusable_share * self.coverage(tokens)

    def write_bytes_per_s(self, tier: str, gib: float, rate_per_s: float) -> float:
        """Write-through traffic into a node-shared tier: every request writes the blocks it
        computed (all ranks' shares)."""
        computed = 1.0 - self.hit_share_with(tier, gib)
        per_token = self.geometry.bytes_per_token_rank * self.tp
        return rate_per_s * self.workload.input_tokens * computed * per_token

    def served_share(self, tier: str, gib: float) -> float:
        """Share of requests whose reusable prefix this tier serves: the reusable share times the
        part of the working set it holds beyond the faster tiers already planned."""
        before = self.pool_tokens() + sum(
            self.tokens_per_replica(t) for t in ("T1", "T2", "T3") if t < tier
        )
        after = before + self.tokens_per_replica(tier, gib)
        share = self.workload.reusable_share * self.workload.repeat_share
        return share * (self.coverage(after) - self.coverage(before))

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
