"""The planner's view of the target node and cluster, derived from the raw ``cluster_info`` records.

This is the input derivation budsim did in ``simulator/kv_plan.py`` and ``shared_slice.py``, moved
here so all KV logic lives in one package (FRD-023 FR-PLAN-1). The slice arithmetic mirrors
budcluster's ``deployment/memory_plan.py`` byte for byte; budsim's parity test compares the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .facts import ClusterKVFacts, NodeKVFacts

GIB = 2**30
MIB = 2**20
GB_DECIMAL = 10**9
# Mirrors of budcluster memory_plan constants. Change both in one PR; the parity tests fail
# otherwise.
DEVICE_CTX_BYTES_CUDA = 1536 * MIB
SLICE_OUTSIDE_RESERVE_BYTES_CUDA = 1536 * MIB
SLICE_TP_COMM_RESERVE_BYTES_CUDA = 1024 * MIB
GPU_UTIL_MAX_CENTI = 94
# Engines that take no fraction of the device: their HAMi slice is their demand, which already carries
# the CUDA context. Bud spec 031 DM4: Bud-Decision-Runtime is one.
SLICE_DEMAND_ONLY_ENGINES = ("latentbud", "bud-decision")
# Engines with no KV cache: no KV commit and no sampler logits in their demand (LatentBud's encoders,
# Bud-Decision-Runtime's decision models, Kev included: its kept states are its state term, not KV).
NO_KV_ENGINES = ("latentbud", "bud-decision")
# Host memory an engine pod uses that its node's `utilized_memory_gb` doesn't show: that figure sums
# pod memory REQUESTS, and a GPU engine pod without a T1 tier requests none. Spike S2 measured
# 2.7-3.9 GiB anonymous memory for 0.6B-8B models. [MEASURE]
ENGINE_RSS_ALLOWANCE_GIB = 6.0
# budcluster's transient headroom on a CPU pod: max(1 GiB, 10% of the model bytes).
CPU_TRANSIENT_RATIO = 0.10
CPU_TYPES = ("cpu", "cpu_high")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


# ------------------------------------------------------------------------------ slice arithmetic


def gb_to_bytes(value_gb: Optional[float]) -> int:
    """Read a *_memory_gb wire field the way budcluster does (decimal GB, 0 if absent)."""
    if not value_gb:
        return 0
    return int(float(value_gb) * GB_DECIMAL)


def wire_gib(value_gb: Optional[float]) -> float:
    """A *_memory_gb wire term, in GiB."""
    return gb_to_bytes(value_gb) / GIB


def kv_committed_bytes(kv_cache_memory_gb: Optional[float], is_encoder: bool) -> int:
    """KV bytes budcluster commits per rank: the demand rounded up to whole GiB, at least one."""
    return 0 if is_encoder else max(1, _ceil_div(gb_to_bytes(kv_cache_memory_gb), GIB)) * GIB


def device_demand_bytes(
    weight_memory_gb: Optional[float],
    kv_cache_memory_gb: Optional[float],
    activation_memory_gb: Optional[float],
    state_memory_gb: Optional[float],
    sampler_logits_gb: Optional[float],
    is_encoder: bool,
) -> int:
    """The per-rank device demand exactly as budcluster computes it from the same wire fields."""
    logits = 0 if is_encoder else gb_to_bytes(sampler_logits_gb)
    return (
        gb_to_bytes(weight_memory_gb)
        + kv_committed_bytes(kv_cache_memory_gb, is_encoder)
        + gb_to_bytes(activation_memory_gb)
        + logits
        + gb_to_bytes(state_memory_gb)
        + DEVICE_CTX_BYTES_CUDA
    )


def slice_mib(device_demand: int, engine: str, tp_size: int = 1) -> Tuple[int, Optional[int]]:
    """(slice MiB, util hundredths or None) budcluster renders for one shared cuda rank."""
    if engine in SLICE_DEMAND_ONLY_ENGINES:
        return _ceil_div(device_demand, MIB), None
    reserve = SLICE_OUTSIDE_RESERVE_BYTES_CUDA + (
        SLICE_TP_COMM_RESERVE_BYTES_CUDA if tp_size > 1 else 0
    )
    mib = max(
        _ceil_div(device_demand * 100, GPU_UTIL_MAX_CENTI * MIB),
        _ceil_div((device_demand + reserve) * 100, 99 * MIB),
    )
    return mib, _ceil_div(device_demand * 100, mib * MIB)


def grown_slice_mib(
    device_demand: int,
    kv_committed: int,
    gpu_kv_gib: Optional[float],
    engine: str,
    tp_size: int = 1,
    device_total_gib: Optional[float] = None,
) -> Tuple[int, Optional[int], bool]:
    """(slice MiB, util hundredths or None, grown) once a plan's ``gpu_kv_gib`` grows the slice.

    Mirrors compute_budget: the slice's KV part becomes ``gpu_kv_gib`` when that is more than the
    committed KV, and a grown slice larger than the card is not used. A demand-only engine never grows.
    """
    base_mib, base_centi = slice_mib(device_demand, engine, tp_size)
    if engine in SLICE_DEMAND_ONLY_ENGINES or not gpu_kv_gib or gpu_kv_gib <= 0:
        return base_mib, base_centi, False
    grown_kv = int(gpu_kv_gib * GIB)
    if grown_kv <= kv_committed:
        return base_mib, base_centi, False
    mib, centi = slice_mib(device_demand - kv_committed + grown_kv, engine, tp_size)
    if device_total_gib and mib * MIB > int(float(device_total_gib) * GIB):
        return base_mib, base_centi, False
    return mib, centi, True


def slice_fits(slice_size_mib: int, free_gib: Optional[float]) -> bool:
    """Whether a slice fits the HAMi-free memory (binary GiB, as HAMi reports)."""
    if free_gib is None:
        return True
    return slice_size_mib * MIB <= float(free_gib) * GIB


def wire_slice_mib(terms: Mapping[str, Any], engine: str, tp_size: int = 1) -> int:
    """The slice for a result dict whose *_memory fields budsim divides by 1024**3 on the wire."""

    def wire_gb(key: str) -> float:
        return (terms.get(key) or 0) / (1024**3)

    demand = device_demand_bytes(
        wire_gb("weight_memory"),
        wire_gb("kv_cache_memory"),
        wire_gb("activation_memory"),
        wire_gb("state_memory"),
        wire_gb("sampler_logits_memory") or None,
        is_encoder=engine in NO_KV_ENGINES,
    )
    return slice_mib(demand, engine, tp_size)[0]


def placeable_configs(
    configs: Sequence[Mapping[str, Any]], engine: str, free_gib: Optional[float]
) -> List[Mapping[str, Any]]:
    """The shared cuda configs whose slice fits ``free_gib`` (all of them when it is unknown)."""
    if free_gib is None:
        return list(configs)
    kept = []
    for config in configs:
        tp_size = int((config.get("config") or {}).get("tensor_parallel_size", 1) or 1)
        if slice_fits(wire_slice_mib(config, engine, tp_size), free_gib):
            kept.append(config)
    return kept


def _slices_that_fit(size_mib: int, free_gib: Sequence[float]) -> int:
    return sum(int(free * GIB // (size_mib * MIB)) for free in free_gib) if size_mib > 0 else 0


def fit_grown_slice(
    gpu_kv_gib: float,
    *,
    device_demand: int,
    kv_committed: int,
    engine: str,
    device_total_gib: Optional[float],
    replicas: int,
    free_gib: Sequence[float],
) -> Optional[float]:
    """The most KV per rank (0.01 GiB steps, at most ``gpu_kv_gib``) a grown slice may hold here.

    Every replica's slice has to fit the node's cards as HAMi reports them free now, or budcluster's
    pre-flight falls back to the demand-sized slice at deploy time. None when not even the smallest
    growth fits.
    """
    if not free_gib:
        return gpu_kv_gib

    def fits(centi: int) -> bool:
        size, _, grown = grown_slice_mib(
            device_demand, kv_committed, centi / 100, engine, 1, device_total_gib
        )
        return grown and _slices_that_fit(size, free_gib) >= replicas

    low, high = kv_committed * 100 // GIB + 1, int(gpu_kv_gib * 100)
    if high < low or not fits(low):
        return None
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if fits(mid) else (low, mid - 1)
    return low / 100


# --------------------------------------------------------------------------------- node record


def _mem_gb(device: Mapping[str, Any]) -> float:
    for key in ("mem_per_GPU_in_GB", "mem_per_gpu_in_gb", "memory_gb"):
        n = _num(device.get(key))
        if n:
            return n
    return 0.0


def free_gb(device: Mapping[str, Any], dev_type: str) -> float:
    """A device's free memory in GB: host RAM left for CPUs, card memory HAMi hasn't allocated."""
    if dev_type in CPU_TYPES:
        avail = _num(device.get("available_memory_gb"))
        if avail is not None:
            return max(0.0, avail)
        return max(0.0, _mem_gb(device) - (_num(device.get("utilized_memory_gb")) or 0.0))
    return max(0.0, _mem_gb(device) - (_num(device.get("memory_allocated_gb")) or 0.0))


def device_model_key(device: Mapping[str, Any]) -> str:
    """The card-model key of an inventory device: raw_name, else name, else device_name."""
    return str(device.get("raw_name") or device.get("name") or device.get("device_name") or "")


def _gpu_devices(node: Mapping[str, Any], model_key: str) -> List[Mapping[str, Any]]:
    return [
        d
        for d in node.get("devices") or []
        if str(d.get("type") or "").lower() not in CPU_TYPES and device_model_key(d) == model_key
    ]


def node_pool(node: Mapping[str, Any], model_key: str, devices_per_replica: int) -> Optional[int]:
    """Replicas of ``devices_per_replica`` cards of ``model_key`` the node has free for, or None when
    it reports no such card."""
    devices = _gpu_devices(node, model_key)
    if not devices:
        return None
    free = 0
    for device in devices:
        count = _num(device.get("available_count"))
        free += max(int(count), 0) if count is not None else 1
    return free // max(int(devices_per_replica), 1)


def conservative_capabilities(records: Sequence[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    """One ``kv_capabilities`` record that every node of ``records`` satisfies (FRD-023 FR-CAP-10).

    A spread deployment's single pod template lands on any of them, so it is planned on the worst
    case: a flag is True only when every node reports True, a number is the smallest, a text is kept
    only when every node agrees. Any node that doesn't report a fact makes it unknown (None), which
    never enables a tier.
    """
    records = [r or {} for r in records]
    merged: Dict[str, Any] = {}
    for key in sorted({k for r in records for k in r}):
        values = [r.get(key) for r in records]
        if any(v is None for v in values):
            merged[key] = None
        elif all(isinstance(v, bool) for v in values):
            merged[key] = all(values)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            merged[key] = min(values)
        else:
            merged[key] = values[0] if all(v == values[0] for v in values) else None
    return merged


@dataclass(frozen=True)
class InfraSnapshot:
    """The target node and its cluster as budcluster last reported them (``cluster_info``).

    A deployment whose replicas spread over several nodes (FR-PLACE) has no target node: ``spread``
    then holds every node with its card, and ``node`` is one synthetic record that all of them
    satisfy (``conservative_capabilities``).
    """

    node: Optional[Mapping[str, Any]] = None
    cluster: Optional[Mapping[str, Any]] = None
    spread: Tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_cluster_info(
        cls,
        cluster_info: Optional[Sequence[Mapping[str, Any]]],
        cluster_id: Any,
        node_name: Optional[str],
        *,
        spread_model_key: Optional[str] = None,
    ) -> "InfraSnapshot":
        """Find the plan's cluster and target node; a missing node gives no facts at all.

        With no ``node_name`` and a ``spread_model_key``, plan on every node that has that card.
        """
        cluster = next((c for c in cluster_info or [] if str(c.get("id")) == str(cluster_id)), None)
        node = None
        if cluster is not None and node_name:
            node = next(
                (
                    n
                    for n in cluster.get("nodes") or []
                    if node_name in (n.get("name"), n.get("id"))
                ),
                None,
            )
        elif cluster is not None and spread_model_key:
            nodes = tuple(
                n for n in cluster.get("nodes") or [] if _gpu_devices(n, spread_model_key)
            )
            if nodes:
                merged = {
                    "name": "spread:" + ",".join(str(n.get("name") or n.get("id")) for n in nodes),
                    "kv_capabilities": conservative_capabilities(
                        [n.get("kv_capabilities") for n in nodes]
                    ),
                    "devices": [],
                }
                return cls(node=merged, cluster=cluster, spread=nodes)
        return cls(node=node, cluster=cluster)

    def node_pool(self, model_key: str, devices_per_replica: int) -> Optional[int]:
        """How many replicas one node holds: the target node's pool, or for a spread deployment the
        largest pool among its nodes. None when no node reports the card."""
        nodes = self.spread or ((self.node,) if self.node else ())
        pools = [p for p in (node_pool(n, model_key, devices_per_replica) for n in nodes) if p is not None]
        return max(pools) if pools else None

    def card_nodes(self, model_key: str) -> Optional[int]:
        """How many nodes have the card, busy or not: the spread's nodes, or 1 for a target node that reports it.
        None when no node is known."""
        nodes = self.spread or ((self.node,) if self.node else ())
        if not nodes:
            return None
        return sum(1 for n in nodes if _gpu_devices(n, model_key))

    def node_facts(self) -> NodeKVFacts:
        return NodeKVFacts.from_capabilities((self.node or {}).get("kv_capabilities"))

    def cluster_facts(self) -> ClusterKVFacts:
        return ClusterKVFacts.from_record((self.cluster or {}).get("kv_capability"))

    def engine_pods_without_requests(self) -> int:
        """GPU engine pods on the node, which request no host memory (HAMi's container count)."""
        pods = 0
        for device in (self.node or {}).get("devices") or []:
            if str(device.get("type") or "").lower() in CPU_TYPES:
                continue
            containers = _num(device.get("shared_containers_count"))
            if containers is not None:
                pods += max(int(containers), 0)
            elif (_num(device.get("memory_allocated_gb")) or 0) > 0:
                pods += 1
        return pods

    def host_free_gib(self) -> Optional[float]:
        """The node's host memory free for a KV tier, in GiB, or None when it reports no CPU entry.

        The CPU entry's free memory (total minus requests), less what GPU engine pods use without
        requesting it (``ENGINE_RSS_ALLOWANCE_GIB`` each) and the cluster's reserved tier RAM.
        """
        if self.spread:
            frees = [InfraSnapshot(node=n, cluster=self.cluster).host_free_gib() for n in self.spread]
            return None if any(f is None for f in frees) else min(frees)
        if self.node is None:
            return None
        cpus = [
            d
            for d in self.node.get("devices") or []
            if str(d.get("type") or "").lower() in CPU_TYPES
        ]
        if not cpus:
            return None
        free = max(free_gb(d, str(d.get("type")).lower()) for d in cpus)
        reserved = self.cluster_facts().reserved_host_ram_gib_per_node
        free -= ENGINE_RSS_ALLOWANCE_GIB * self.engine_pods_without_requests() + max(reserved, 0.0)
        return max(free, 0.0)

    def gpu_free_list(self, model_key: str) -> List[float]:
        """The free memory (GiB) of every card of this model on the node, one entry per card. For a
        spread deployment, the node whose best card has the least free memory."""
        if self.spread:
            lists = [InfraSnapshot(node=n, cluster=self.cluster).gpu_free_list(model_key) for n in self.spread]
            lists = [f for f in lists if f]
            return min(lists, key=max) if lists else []
        free: List[float] = []
        for device in (self.node or {}).get("devices") or []:
            dev_type = str(device.get("type") or "").lower()
            if dev_type in CPU_TYPES or device_model_key(device) != model_key:
                continue
            count = _num(device.get("available_count"))
            count = 1 if count is None else int(count)
            if count > 0:
                free.extend([free_gb(device, dev_type)] * count)
        return free
