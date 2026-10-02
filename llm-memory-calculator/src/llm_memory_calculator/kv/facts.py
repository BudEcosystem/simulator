"""What the planner knows about the node, the cluster and the engine (FRD-023 §8.2).

Every fact is parsed from the record budcluster reports. Anything missing or malformed becomes
``None``, and the planner never reads ``None`` as "available".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from .constants import DEFAULT_PINNED_GBPS, PCIE_GBPS_BY_GEN


def _flag(record: Mapping[str, Any], key: str) -> Optional[bool]:
    value = record.get(key)
    return value if isinstance(value, bool) else None


def _number(record: Mapping[str, Any], key: str, *, positive: bool = True) -> Optional[float]:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if positive and value <= 0:
        return None
    return float(value)


def _text(record: Mapping[str, Any], key: str) -> Optional[str]:
    value = record.get(key)
    return str(value) if isinstance(value, str) and value else None


@dataclass(frozen=True)
class NodeKVFacts:
    """A node's capability facts, from its ``kv_capabilities``. None is unknown, never available."""

    pinned_offload_ok: Optional[bool] = None
    host_pointer_for_registered_mem: Optional[bool] = None
    vgpu: Optional[bool] = None
    pinned_copy_gbps: Optional[float] = None  # GB/s, best of 3 host-to-device copies
    pcie_gen: Optional[int] = None
    rdma_nics: Optional[int] = None
    rdma_link_gbps: Optional[float] = None  # Gbit/s, the highest active port's rate
    gpudirect_rdma: Optional[bool] = None
    dma_buf: Optional[bool] = None
    nvidia_peermem: Optional[bool] = None
    local_nvme_count: Optional[int] = None
    local_nvme_gib: Optional[float] = None
    host_dram_gib: Optional[float] = None

    @classmethod
    def from_capabilities(cls, caps: Optional[Mapping[str, Any]]) -> "NodeKVFacts":
        """Read the facts from a node's ``kv_capabilities``; anything malformed is unknown."""
        caps = caps or {}
        gen = _number(caps, "pcie_gen")
        nics = _number(caps, "rdma_nics", positive=False)
        nvme = _number(caps, "local_nvme_count", positive=False)
        return cls(
            pinned_offload_ok=_flag(caps, "pinned_offload_ok"),
            host_pointer_for_registered_mem=_flag(caps, "host_pointer_for_registered_mem"),
            vgpu=_flag(caps, "vgpu"),
            pinned_copy_gbps=_number(caps, "pinned_copy_gbps"),
            pcie_gen=int(gen) if gen else None,
            rdma_nics=int(nics) if nics is not None else None,
            rdma_link_gbps=_number(caps, "rdma_link_gbps"),
            gpudirect_rdma=_flag(caps, "gpudirect_rdma"),
            dma_buf=_flag(caps, "dma_buf"),
            nvidia_peermem=_flag(caps, "nvidia_peermem"),
            local_nvme_count=int(nvme) if nvme is not None else None,
            local_nvme_gib=_number(caps, "local_nvme_gib"),
            host_dram_gib=_number(caps, "host_dram_gib"),
        )

    def pinned_gbps(self) -> Tuple[float, str]:
        """Return the host-to-device bandwidth (GB/s) to plan with, and where it came from."""
        if self.pinned_copy_gbps:
            return self.pinned_copy_gbps, "measured"
        if self.pcie_gen in PCIE_GBPS_BY_GEN:
            return PCIE_GBPS_BY_GEN[self.pcie_gen], f"PCIe Gen{self.pcie_gen} default"
        return DEFAULT_PINNED_GBPS, "unknown link, conservative default"

    def gpudirect(self) -> bool:
        """Whether GPUDirect RDMA is known to work here: probed, or an RDMA NIC plus DMA-BUF or
        ``nvidia_peermem``."""
        if self.gpudirect_rdma is not None:
            return self.gpudirect_rdma
        return bool(self.rdma_nics) and (self.dma_buf is True or self.nvidia_peermem is True)


@dataclass(frozen=True)
class StorageClassFacts:
    """A storage class as the cluster's KV capability record lists it."""

    name: str
    node_local: Optional[bool] = None
    read_gbps: Optional[float] = None  # GB/s, measured by the T2 health gate
    capacity_gib: Optional[float] = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Optional["StorageClassFacts"]:
        name = _text(record, "name")
        if not name:
            return None
        return cls(
            name=name,
            node_local=_flag(record, "node_local"),
            read_gbps=_number(record, "kv_read_gbps") or _number(record, "read_gbps"),
            capacity_gib=_number(record, "capacity_gib"),
        )


@dataclass(frozen=True)
class PoolFacts:
    """The cluster's T3 backend, as its health gate recorded it."""

    backend: str  # mooncake | fs | none
    transport: Optional[str] = None  # tcp | rdma (mooncake)
    read_gbps: Optional[float] = None  # GB/s, a timed get through Mooncake from a GPU node
    capacity_gib: Optional[float] = None
    floor_ms: Optional[float] = None  # fixed cost of one get, when the gate measured it
    storage_class: Optional[str] = None  # fs backend

    @classmethod
    def from_record(cls, record: Optional[Mapping[str, Any]]) -> Optional["PoolFacts"]:
        if not record:
            return None
        backend = _text(record, "backend")
        if not backend or backend == "none":
            return None
        transport = _text(record, "transport")
        return cls(
            backend=backend,
            transport=transport.lower() if transport else None,
            read_gbps=_number(record, "read_gbps"),
            capacity_gib=_number(record, "capacity_gib"),
            floor_ms=_number(record, "floor_ms"),
            storage_class=_text(record, "storage_class"),
        )


@dataclass(frozen=True)
class ClusterKVFacts:
    """A cluster's KV capability record (FRD-023 §8.2, per cluster)."""

    storage_classes: Tuple[StorageClassFacts, ...] = ()
    kv_storage_class: Optional[str] = None
    kv_storage_enabled: Optional[bool] = None
    t3: Optional[PoolFacts] = None
    reserved_host_ram_gib_per_node: float = 0.0

    @classmethod
    def from_record(cls, record: Optional[Mapping[str, Any]]) -> "ClusterKVFacts":
        """Read the record; a missing or malformed one has no tiers to offer."""
        record = record or {}
        classes = []
        for entry in record.get("storage_classes") or []:
            if isinstance(entry, Mapping):
                parsed = StorageClassFacts.from_record(entry)
                if parsed:
                    classes.append(parsed)
        storage = record.get("kv_storage") if isinstance(record.get("kv_storage"), Mapping) else {}
        reserved = _number(record, "reserved_host_ram_gib_per_node") or 0.0
        t3 = record.get("t3") if isinstance(record.get("t3"), Mapping) else None
        return cls(
            storage_classes=tuple(classes),
            kv_storage_class=_text(storage, "class") or _text(storage, "class_name"),
            kv_storage_enabled=_flag(storage, "enabled"),
            t3=PoolFacts.from_record(t3),
            reserved_host_ram_gib_per_node=reserved,
        )

    def kv_storage(self) -> Optional[StorageClassFacts]:
        """The class T2 uses, or None when the setting is off or names no listed class."""
        if self.kv_storage_enabled is False or not self.kv_storage_class:
            return None
        return next((c for c in self.storage_classes if c.name == self.kv_storage_class), None)


@dataclass(frozen=True)
class EngineSpec:
    """The engine record BudConnect returns for the node group, plus budeval's dtype verdicts."""

    name: str
    version: Optional[str] = None
    kv_features: FrozenSet[str] = field(default_factory=frozenset)
    # (model_type, GPU generation) pairs budeval validated for fp8 KV (FR-QUANT-2)
    validated_kv_dtypes: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)
    capabilities: Any = None  # EngineKVCapabilities, for kv_cache_breakdown

    @classmethod
    def of(cls, name: str, kv_features: Any = None, **kwargs: Any) -> "EngineSpec":
        return cls(name=str(name or ""), kv_features=frozenset(kv_features or ()), **kwargs)
