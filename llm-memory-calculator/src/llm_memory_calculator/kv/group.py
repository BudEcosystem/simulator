"""The node group budsim chose: device, parallelism, replicas and the memory terms it sized."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .infra import device_demand_bytes, kv_committed_bytes, wire_gib

GIB = 2**30


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NodeGroup:
    """One node group of the configuration that deploys (budsim's ``NodeGroupConfiguration``).

    Memory terms are budsim's wire fields: decimal GB per rank, read the way budcluster reads them.
    """

    device_type: str
    tp: int = 1
    pp: int = 1
    replicas: int = 1
    hardware_mode: str = "dedicated"  # dedicated | shared (a HAMi slice)
    engine: str = "vllm"
    weight_memory_gb: Optional[float] = None
    kv_cache_memory_gb: Optional[float] = None
    activation_memory_gb: Optional[float] = None
    state_memory_gb: Optional[float] = None
    sampler_logits_gb: Optional[float] = None
    device_total_memory_gb: Optional[float] = None  # the card's memory, GiB as budsim reports it
    device_model_key: str = ""  # matches the node inventory's raw_name / name
    device_generation: Optional[str] = None
    device_tflops: Optional[float] = None  # peak dense fp16/bf16 per device
    hardware: Optional[Mapping[str, Any]] = None  # GenZ hardware entry, when budsim matched one
    hardware_name: Optional[str] = None  # its GenZ name (e.g. H100_GPU), for P/D analysis
    is_embedding: bool = False
    precision: str = "bf16"
    data_parallel_attention: bool = False
    target_node: Optional[str] = None

    @classmethod
    def from_object(cls, group: Any, row: Any = None, **overrides: Any) -> "NodeGroup":
        """Read budsim's ``NodeGroupConfiguration`` (and its simulation row, for device FLOPs)."""

        def get(obj: Any, name: str, default: Any = None) -> Any:
            if obj is None:
                return default
            if isinstance(obj, Mapping):
                return obj.get(name, default)
            return getattr(obj, name, default)

        values = dict(
            device_type=str(get(group, "type") or "").lower(),
            tp=max(int(get(group, "tp_size") or 1), 1),
            pp=max(int(get(group, "pp_size") or 1), 1),
            replicas=max(int(get(group, "replicas") or 1), 1),
            hardware_mode=str(get(group, "hardware_mode") or "dedicated"),
            engine=str(get(group, "engine_type") or ""),
            weight_memory_gb=_num(get(group, "weight_memory_gb")),
            kv_cache_memory_gb=_num(get(group, "kv_cache_memory_gb")),
            activation_memory_gb=_num(get(group, "activation_memory_gb")),
            state_memory_gb=_num(get(group, "state_memory_gb")),
            sampler_logits_gb=_num(get(group, "sampler_logits_gb")),
            device_total_memory_gb=_num(get(group, "device_total_memory_gb")),
            device_model_key=str(
                get(row, "raw_name") or get(row, "device_name") or get(group, "raw_name") or ""
            ),
            device_generation=get(group, "device_model") or get(group, "raw_name"),
            device_tflops=_num(get(row, "peak_fp16_tflops")),
            is_embedding=bool(get(group, "is_embedding")),
            target_node=get(group, "target_node_name"),
        )
        values.update(overrides)
        return cls(**values)

    @property
    def shared(self) -> bool:
        return self.hardware_mode == "shared"

    @property
    def weight_gib(self) -> float:
        return wire_gib(self.weight_memory_gb)

    @property
    def kv_committed_gib(self) -> float:
        """The KV budcluster commits per rank (demand rounded up to whole GiB)."""
        return kv_committed_bytes(self.kv_cache_memory_gb, self.is_embedding) / GIB

    @property
    def other_gib(self) -> float:
        return sum(
            wire_gib(v)
            for v in (self.activation_memory_gb, self.state_memory_gb, self.sampler_logits_gb)
        )

    def pod_host_gib(self) -> float:
        """Host memory a CPU group's own pods take, as budcluster sizes them (before KV growth)."""
        kv = self.kv_committed_gib
        model = wire_gib(self.weight_memory_gb) + kv
        rank = model + self.other_gib
        return self.replicas * self.tp * rank + max(1.0, 0.10 * self.tp * model)

    def device_demand_bytes(self) -> int:
        """Per-rank device demand exactly as budcluster computes it."""
        return device_demand_bytes(
            self.weight_memory_gb,
            self.kv_cache_memory_gb,
            self.activation_memory_gb,
            self.state_memory_gb,
            self.sampler_logits_gb,
            self.is_embedding,
        )


def genz_hardware(
    device_model: Optional[str], device_type: str
) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """The GenZ hardware entry budsim's heuristic would use for this device, and its GenZ name;
    ``(None, None)`` when GenZ doesn't know it."""
    if not device_model or device_type in ("cpu", "cpu_high"):
        # GenZ's matcher fuzzy-matches a bare "CPU" to a GPU entry; CPU engines use their own FLOPs.
        return None, None
    try:
        from ..hardware import HardwareManager

        specs = HardwareManager().get_cluster_hardware_specs(
            {"raw_name": device_model, "type": device_type}
        )
    except Exception:  # noqa: BLE001 - an unknown device simply has no GenZ entry
        return None, None
    if not specs or not specs.get("matched"):
        return None, None
    config = {
        "Flops": specs["flops_fp16"],
        "Memory_size": specs["memory_size_gb"],
        "Memory_BW": specs["memory_bandwidth_gbs"],
        "ICN": specs["interconnect_bandwidth_gbs"],
        "real_values": specs.get("real_values", True),
    }
    return config, specs.get("device_name")
