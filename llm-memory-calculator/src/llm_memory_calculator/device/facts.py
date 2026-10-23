"""Layer 0 of the FRD-027 predictor: what a device is, as the planner may rely on it (FRD-027 §6).

Every layer above reads hardware only through :class:`DeviceFacts` (FR-DEV-1). The fields are the ones the
study found decide a configuration's fate on a card -- visible (not nameplate) memory, dtype support,
virtualization, interconnect, the hardware mode -- next to the roofline numbers (bandwidth, FLOPs per dtype).
``sources`` says where each field came from (``live`` from the cluster, ``catalogue``, or ``derived`` from an
architecture default), so a plan can say how much of the device it actually knows (FR-DEV-2, FR-DEV-3).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

GIB = 1024**3
MIB = 1024**2

#: Virtualization classes a record's ``limits.max_tp_by_virtualization`` is keyed by.
VIRTUALIZATION_CLASSES = ("bare_metal", "vgpu", "vgpu_no_vmm", "mig")


@dataclass(frozen=True)
class Interconnect:
    """How devices on one node talk: ``nvlink``, ``pcie`` or ``none`` (no peer access, e.g. a vGPU)."""

    kind: str = "none"
    link_bandwidth: float = 0.0  # bytes/s, one direction, between two devices
    pcie_gen: Optional[int] = None
    max_devices_per_node: int = 1


@dataclass(frozen=True)
class Virtualization:
    """vGPU and MIG facts. A vGPU without VMM cannot run TP > 1 and crashes on ``cudaHostRegister``
    (FRD-023 Phase 0 on TCS)."""

    vgpu: bool = False
    vmm: bool = True
    host_register: bool = True
    mig: bool = False
    profile: Optional[str] = None

    @property
    def klass(self) -> str:
        if self.mig:
            return "mig"
        if self.vgpu and not self.vmm:
            return "vgpu_no_vmm"
        if self.vgpu:
            return "vgpu"
        return "bare_metal"


@dataclass(frozen=True)
class HardwareMode:
    """``dedicated`` (the whole device) or ``shared`` (a HAMi memory slice of ``slice_bytes``)."""

    kind: str = "dedicated"
    slice_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("dedicated", "shared"):
            raise ValueError(f"hardware mode must be dedicated or shared, not {self.kind!r}")
        if self.kind == "shared" and not self.slice_bytes:
            raise ValueError("a shared hardware mode needs slice_bytes")


@dataclass(frozen=True)
class DeviceFacts:
    """One device as the predictor sees it (FRD-027 §6)."""

    sku: str
    architecture: str  # hopper | ampere | ada | blackwell | volta | cpu | unknown
    device_class: str  # cuda | cpu | hpu | rocm
    memory_visible_bytes: int
    hbm_bandwidth: float  # bytes/s
    peak_flops: Mapping[str, float]  # dense FLOP/s per compute dtype
    dtype_support: Mapping[str, str]  # weight/KV format -> native | weight_only | emulated
    kernel_support: FrozenSet[str] = frozenset()
    interconnect: Interconnect = Interconnect()
    virtualization: Virtualization = Virtualization()
    hardware_mode: HardwareMode = HardwareMode()
    compute_capability: Optional[Tuple[int, int]] = None
    memory_nameplate_bytes: Optional[int] = None
    vendor: str = "nvidia"
    derived: bool = False
    sources: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ views

    @property
    def usable_memory_bytes(self) -> int:
        """The memory the engine is given: the HAMi slice in shared mode, the visible memory otherwise."""
        if self.hardware_mode.kind == "shared" and self.hardware_mode.slice_bytes:
            return min(int(self.hardware_mode.slice_bytes), int(self.memory_visible_bytes))
        return int(self.memory_visible_bytes)

    def flops(self, dtype: str) -> float:
        """Peak dense FLOP/s for a compute dtype, falling back to bf16 (weight-only formats compute in
        the activation dtype)."""
        value = self.peak_flops.get(dtype)
        if value:
            return float(value)
        return float(self.peak_flops.get("bf16") or self.peak_flops.get("fp16") or 0.0)

    def with_mode(self, mode: HardwareMode) -> "DeviceFacts":
        sources = dict(self.sources)
        sources["hardware_mode"] = "intent"
        return replace(self, hardware_mode=mode, sources=sources)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kernel_support"] = sorted(self.kernel_support)
        data["peak_flops"] = dict(sorted(self.peak_flops.items()))
        data["dtype_support"] = dict(sorted(self.dtype_support.items()))
        data["sources"] = dict(sorted(self.sources.items()))
        data["compute_capability"] = (
            list(self.compute_capability) if self.compute_capability else None
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceFacts":
        cc = data.get("compute_capability")
        return cls(
            sku=str(data["sku"]),
            architecture=str(data["architecture"]),
            device_class=str(data["device_class"]),
            memory_visible_bytes=int(data["memory_visible_bytes"]),
            hbm_bandwidth=float(data["hbm_bandwidth"]),
            peak_flops={str(k): float(v) for k, v in dict(data.get("peak_flops") or {}).items()},
            dtype_support={
                str(k): str(v) for k, v in dict(data.get("dtype_support") or {}).items()
            },
            kernel_support=frozenset(data.get("kernel_support") or ()),
            interconnect=Interconnect(**dict(data.get("interconnect") or {})),
            virtualization=Virtualization(**dict(data.get("virtualization") or {})),
            hardware_mode=HardwareMode(**dict(data.get("hardware_mode") or {})),
            compute_capability=tuple(cc) if cc else None,  # type: ignore[arg-type]
            memory_nameplate_bytes=data.get("memory_nameplate_bytes"),
            vendor=str(data.get("vendor") or "nvidia"),
            derived=bool(data.get("derived", False)),
            sources={str(k): str(v) for k, v in dict(data.get("sources") or {}).items()},
        )

    def digest(self) -> str:
        """A stable digest of every fact: the plan's ``inputs_digest`` and the memo key use it."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]

    def __hash__(self) -> int:  # Mapping fields are not hashable; the digest is.
        return hash(self.digest())
