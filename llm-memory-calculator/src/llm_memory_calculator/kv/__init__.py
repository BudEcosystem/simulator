"""KV cache tier planning for Bud deployments (FRD-023).

:func:`plan_kv` makes every KV decision for the configuration budsim chose: GPU headroom (T0), pod
host memory (T1), the node's disk (T2), the cluster pool (T3) and prefill/decode, each kept only
where the node can run it and reloading a prefix beats recomputing it. ``plan.to_wire()`` is the
``kv_plan`` budcluster renders.
"""

from .constants import RELEASED_TIERS
from .cost import BreakEven, RecomputeModel, TierLink, break_even, prefill_seconds
from .facts import ClusterKVFacts, EngineSpec, NodeKVFacts, PoolFacts, StorageClassFacts
from .geometry import KVGeometry, kv_bytes_per_token_per_rank
from .group import NodeGroup, genz_hardware
from .infra import InfraSnapshot
from .namespace import kv_namespace
from .plan import KVTierPlan
from .planner import plan_kv, plan_kv_tiers
from .workload import DeploymentSpec, KVWorkload

__all__ = [
    "BreakEven",
    "ClusterKVFacts",
    "DeploymentSpec",
    "EngineSpec",
    "InfraSnapshot",
    "KVGeometry",
    "KVTierPlan",
    "KVWorkload",
    "NodeGroup",
    "NodeKVFacts",
    "PoolFacts",
    "RELEASED_TIERS",
    "RecomputeModel",
    "StorageClassFacts",
    "TierLink",
    "break_even",
    "genz_hardware",
    "kv_bytes_per_token_per_rank",
    "kv_namespace",
    "plan_kv",
    "plan_kv_tiers",
    "prefill_seconds",
]
