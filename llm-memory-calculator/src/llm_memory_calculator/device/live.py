"""Device facts from what the cluster reports (FR-DEV-2: live over catalogue; FR-DEV-3: unknown SKUs).

Accepts the shapes budsim holds: a ``cluster_info`` device row (``name``/``raw_name``/``device_model``,
``mem_per_gpu_in_gb``, ``type``, counts) and the GFD/NFD labels budcluster collects
(``nvidia.com/gpu.product``, ``nvidia.com/gpu.memory``, ``nvidia.com/gpu.compute.major/minor``,
``nvidia.com/vgpu.present``). Nothing here raises on a missing field: an empty record still yields facts
(``derived``, architecture ``unknown``), so a plan never fails on coverage (NFR-5).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

from . import catalogue
from .facts import GIB, MIB, DeviceFacts, HardwareMode, Interconnect, Virtualization

_PRODUCT_KEYS = ("product", "raw_name", "device_model", "gpu_product", "name", "model")
_CPU_TYPES = {"cpu", "cpu_high", "cpu-high"}


def _labels(node: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = node.get("labels") or node.get("node_labels") or {}
    return labels if isinstance(labels, Mapping) else {}


def _product(node: Mapping[str, Any]) -> str:
    labels = _labels(node)
    if labels.get("nvidia.com/gpu.product"):
        return str(labels["nvidia.com/gpu.product"])
    for key in _PRODUCT_KEYS:
        if node.get(key):
            return str(node[key])
    return ""


def _memory_bytes(node: Mapping[str, Any]) -> Optional[int]:
    labels = _labels(node)
    for key in ("gpu_memory_mb", "memory_mb", "gpu_mem_mb"):
        if node.get(key):
            return int(float(node[key]) * MIB)
    for key in ("mem_per_gpu_in_gb", "memory_gb", "gpu_memory_gb"):
        if node.get(key):
            return int(round(float(node[key]) * GIB))
    if labels.get("nvidia.com/gpu.memory"):
        return int(float(labels["nvidia.com/gpu.memory"]) * MIB)
    return None


def _cc(node: Mapping[str, Any]):
    labels = _labels(node)
    if node.get("compute_capability"):
        return catalogue.parse_cc(node["compute_capability"])
    major, minor = labels.get("nvidia.com/gpu.compute.major"), labels.get(
        "nvidia.com/gpu.compute.minor"
    )
    if major is not None:
        return int(major), int(minor or 0)
    return None


def _vgpu(node: Mapping[str, Any]) -> Optional[bool]:
    labels = _labels(node)
    flag = node.get("vgpu", labels.get("nvidia.com/vgpu.present"))
    if flag is None:
        return None
    return str(flag).lower() in ("1", "true", "yes")


def _devices_per_node(node: Mapping[str, Any]) -> Optional[int]:
    labels = _labels(node)
    for key in ("max_devices_per_node", "total_count", "device_count", "gpu_count"):
        if node.get(key):
            return int(node[key])
    if labels.get("nvidia.com/gpu.count"):
        return int(labels["nvidia.com/gpu.count"])
    return None


def facts_from_node(
    node: Mapping[str, Any],
    *,
    hardware_mode: Optional[HardwareMode] = None,
) -> DeviceFacts:
    """Build :class:`DeviceFacts` for one device of a node, live values first."""
    node = node or {}
    device_type = str(node.get("type") or node.get("device_type") or "").lower()
    product = _product(node)
    memory = _memory_bytes(node)
    cc = _cc(node)

    base: Optional[DeviceFacts] = None
    if device_type not in _CPU_TYPES:
        base = catalogue.lookup(product)
    if base is None:
        if device_type in _CPU_TYPES:
            arch = "cpu"
        else:
            arch = (
                catalogue.architecture_from_cc(cc)
                or catalogue.architecture_from_product(product)
                or "unknown"
            )
        base = catalogue.architecture_defaults(arch, memory)
        if cc:
            base = replace(base, compute_capability=cc)

    sources = dict(base.sources)
    updates: dict = {}
    if memory and memory != base.memory_visible_bytes:
        updates["memory_visible_bytes"] = memory
    if memory:
        sources["memory_visible_bytes"] = "live"
    per_node = _devices_per_node(node)
    if per_node:
        updates["interconnect"] = replace(
            base.interconnect or Interconnect(), max_devices_per_node=per_node
        )
        sources["max_devices_per_node"] = "live"
    vgpu = _vgpu(node)
    if vgpu is not None and vgpu != base.virtualization.vgpu:
        # A vGPU the catalogue doesn't know: assume the TCS profile class (no VMM, no host register) --
        # the safe side for TP and FRD-023's register mode.
        updates["virtualization"] = (
            Virtualization(vgpu=True, vmm=False, host_register=False, profile=product or None)
            if vgpu
            else Virtualization()
        )
        sources["virtualization"] = "live"
    facts = replace(base, sources=sources, **updates)
    if hardware_mode is not None:
        facts = facts.with_mode(hardware_mode)
    return facts
