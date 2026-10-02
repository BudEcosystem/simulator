"""The plan: what budsim ships on each node group as ``kv_plan`` (FRD-023 §8.1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .constants import EVENTS_ENDPOINT, EVENTS_REPLAY, PLAN_VERSION


def _rounded(value: Optional[float], digits: int = 1) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


@dataclass
class KVTierPlan:
    """A deployment's KV plan. :meth:`to_wire` is the FRD-023 §8.1 shape budcluster parses."""

    profile: str = "off"
    workload_profile: str = "off"
    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: Optional[float] = None
    gpu_kv_gib: Optional[float] = None
    cpu_kv_gib: Optional[float] = None
    tiers: List[Dict[str, Any]] = field(default_factory=list)
    events_enabled: bool = False
    routing: Dict[str, Any] = field(default_factory=dict)
    predicted_hit_rate: Optional[float] = None
    predicted_ttft_ms: Optional[float] = None
    host_memory_gib: float = 0.0
    shm_gib: float = 0.0
    working_set_gib: float = 0.0
    gpu_pool_gib: float = 0.0
    decisions: List[str] = field(default_factory=list)
    # -- added with the full planner (all additive on the wire)
    namespace: Optional[str] = None
    hash_algo: str = "sha256"
    pd: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "role": None})
    rdma: bool = False
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    # -- not on the wire: for tests, validation and the KV tab's "what if"
    withheld: List[Dict[str, Any]] = field(
        default_factory=list
    )  # planned, above the release ceiling
    evaluations: Dict[str, Any] = field(default_factory=dict)  # tier -> BreakEven
    block_size: Optional[int] = None

    def tier(self, name: str) -> Optional[Dict[str, Any]]:
        """The planned (released) entry for ``name``, or None."""
        return next((t for t in self.tiers if t["tier"] == name), None)

    def planned(self, name: str) -> bool:
        """Whether the planner chose ``name``, released or withheld."""
        return self.tier(name) is not None or any(t["tier"] == name for t in self.withheld)

    def drop(
        self,
        tier: str,
        reason: str,
        *,
        reload_ms: Optional[float] = None,
        recompute_ms: Optional[float] = None,
        needed_gbps: Optional[float] = None,
    ) -> None:
        self.dropped.append(
            {
                "tier": tier,
                "reason": reason,
                "reload_ms": _rounded(reload_ms),
                "recompute_ms": _rounded(recompute_ms),
                "needed_gbps": _rounded(needed_gbps, 2),
            }
        )

    def to_wire(self) -> Dict[str, Any]:
        """Return the plan as budsim ships it on ``NodeGroupConfiguration.kv_plan``."""
        return {
            "version": PLAN_VERSION,
            "namespace": self.namespace,
            "profile": self.profile,
            "kv_cache_dtype": self.kv_cache_dtype,
            "skip_layers_sliding_window": False,
            "hash_algo": self.hash_algo,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "gpu_kv_gib": self.gpu_kv_gib,
            "cpu_kv_gib": self.cpu_kv_gib,
            "tiers": [dict(t) for t in self.tiers],
            "events": {
                "enabled": self.events_enabled,
                "endpoint": EVENTS_ENDPOINT,
                "replay": EVENTS_REPLAY,
            },
            "routing": dict(self.routing),
            "pd": dict(self.pd),
            "predicted": {"hit_rate": self.predicted_hit_rate, "ttft_ms": self.predicted_ttft_ms},
            "resources": {
                "host_memory_gib": self.host_memory_gib,
                "shm_gib": self.shm_gib,
                "ipc_lock": self.rdma,
                "rdma": self.rdma,
            },
            "dropped": [dict(d) for d in self.dropped],
            "decisions": list(self.decisions),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return every field, for logging."""
        evaluations, self.evaluations = self.evaluations, {}
        try:
            data = asdict(self)
        finally:
            self.evaluations = evaluations
        data["evaluations"] = {k: v.describe() for k, v in evaluations.items()}
        return data
