"""The device catalogue: what the cluster cannot report (bandwidth, peak FLOPs, interconnect) per SKU,
plus per-architecture defaults for SKUs it does not know (FRD-027 §6, FR-DEV-1…5).

A vGPU profile is its own SKU (FR-DEV-5): ``h100xm-80c-vgpu`` shares Hopper with ``h100-sxm5-80gb`` for the
grounding fallback, but never stands in for it -- the bare part has NVLink, more visible memory and no
hypervisor. Every ``hardware/configs.py`` GPU record is also reachable as a catalogue entry
(:func:`from_hardware_config`), so the old records keep working through the new type.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from .facts import GIB, MIB, DeviceFacts, Interconnect, Virtualization

NATIVE, WEIGHT_ONLY, EMULATED = "native", "weight_only", "emulated"

_COMMON_16 = {"bf16": NATIVE, "fp16": NATIVE, "int8": NATIVE, "int4": WEIGHT_ONLY}

#: Per-architecture defaults. ``ref_memory_gib`` is the part the bandwidth figure belongs to; an unknown
#: SKU of the architecture scales bandwidth by its reported memory against it (FR-DEV-3).
_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "hopper": {
        "cc": (9, 0),
        "ref_memory_gib": 80,
        "hbm_bandwidth": 3.35e12,
        "flops": {"bf16": 989.5e12, "fp16": 989.5e12, "fp8": 1979e12, "int8": 1979e12},
        "dtype": {**_COMMON_16, "fp8": NATIVE, "mxfp4": EMULATED},
        "kernels": {"flash_attn_3", "flashinfer", "fused_gdn", "cutlass_fp8", "marlin"},
    },
    "blackwell": {
        "cc": (10, 0),
        "ref_memory_gib": 180,
        "hbm_bandwidth": 8.0e12,
        "flops": {
            "bf16": 2250e12,
            "fp16": 2250e12,
            "fp8": 4500e12,
            "int8": 4500e12,
            "fp4": 9000e12,
        },
        "dtype": {**_COMMON_16, "fp8": NATIVE, "mxfp4": NATIVE, "nvfp4": NATIVE},
        "kernels": {
            "flashinfer",
            "trtllm_attn",
            "fused_gdn",
            "cutlass_fp8",
            "cutlass_fp4",
            "marlin",
        },
    },
    "ada": {
        "cc": (8, 9),
        "ref_memory_gib": 48,
        "hbm_bandwidth": 864e9,
        "flops": {"bf16": 362e12, "fp16": 362e12, "fp8": 733e12, "int8": 733e12},
        "dtype": {**_COMMON_16, "fp8": NATIVE, "mxfp4": EMULATED},
        "kernels": {"flash_attn_2", "flashinfer", "fused_gdn", "cutlass_fp8", "marlin"},
    },
    "ampere": {
        "cc": (8, 0),
        "ref_memory_gib": 80,
        "hbm_bandwidth": 2.039e12,
        "flops": {"bf16": 312e12, "fp16": 312e12, "int8": 624e12},
        "dtype": {**_COMMON_16, "fp8": WEIGHT_ONLY, "mxfp4": EMULATED},
        "kernels": {"flash_attn_2", "flashinfer", "fused_gdn", "marlin"},
    },
    "turing": {
        "cc": (7, 5),
        "ref_memory_gib": 16,
        "hbm_bandwidth": 320e9,
        "flops": {"fp16": 65e12, "int8": 130e12},
        "dtype": {"fp16": NATIVE, "int8": NATIVE, "bf16": EMULATED},
        "kernels": {"xformers"},
    },
    "volta": {
        "cc": (7, 0),
        "ref_memory_gib": 32,
        "hbm_bandwidth": 900e9,
        "flops": {"fp16": 125e12},
        "dtype": {"fp16": NATIVE, "bf16": EMULATED},
        "kernels": {"xformers"},
    },
    "cpu": {
        "cc": None,
        "ref_memory_gib": 256,
        "hbm_bandwidth": 200e9,
        "flops": {"bf16": 10e12, "fp32": 5e12},
        "dtype": {"bf16": NATIVE, "fp32": NATIVE, "int8": NATIVE},
        "kernels": set(),
    },
    # Nothing known: a conservative accelerator, so a plan never fails on coverage (NFR-5).
    "unknown": {
        "cc": None,
        "ref_memory_gib": 80,
        "hbm_bandwidth": 1.0e12,
        "flops": {"bf16": 200e12, "fp16": 200e12},
        "dtype": {"bf16": NATIVE, "fp16": NATIVE},
        "kernels": set(),
    },
}

_NVLINK_H100 = Interconnect(kind="nvlink", link_bandwidth=450e9, max_devices_per_node=8)
_PCIE5 = Interconnect(kind="pcie", link_bandwidth=64e9, pcie_gen=5, max_devices_per_node=8)
_PCIE4 = Interconnect(kind="pcie", link_bandwidth=32e9, pcie_gen=4, max_devices_per_node=8)

#: Exact SKUs Bud runs or plans for. ``memory_visible_mib`` is what the driver exposes, not the nameplate.
_SKUS: Dict[str, Dict[str, Any]] = {
    "h100xm-80c-vgpu": {
        "architecture": "hopper",
        "aliases": ["h100xm 80c", "h100xm80c", "grid h100xm 80c", "h100xm 80c vgpu"],
        "memory_visible_mib": 81582,  # 79.67 GiB measured by vLLM on TCS
        "memory_nameplate_gib": 80,
        "virtualization": Virtualization(
            vgpu=True, vmm=False, host_register=False, profile="H100XM-80C"
        ),
        "interconnect": Interconnect(kind="none", max_devices_per_node=4),
    },
    "h100-sxm5-80gb": {
        "architecture": "hopper",
        "aliases": [
            "h100 80gb hbm3",
            "h100 sxm5 80gb",
            "h100 sxm5",
            "h100 sxm",
            "h100",
            "h100 gpu",
            "h100 sxm 80gb",
        ],
        "memory_visible_mib": 81559,
        "memory_nameplate_gib": 80,
        "interconnect": _NVLINK_H100,
    },
    "h100-pcie-80gb": {
        "architecture": "hopper",
        "aliases": ["h100 pcie", "h100 pcie 80gb", "h100 pcie gpu"],
        "memory_visible_mib": 81559,
        "memory_nameplate_gib": 80,
        "hbm_bandwidth": 2.0e12,
        "flops": {"bf16": 756e12, "fp16": 756e12, "fp8": 1513e12, "int8": 1513e12},
        "interconnect": _PCIE5,
    },
    "h200-141gb": {
        "architecture": "hopper",
        "aliases": ["h200", "h200 141gb", "h200 nvl", "h200 gpu"],
        "memory_visible_mib": 143771,
        "memory_nameplate_gib": 141,
        "hbm_bandwidth": 4.8e12,
        "interconnect": _NVLINK_H100,
    },
    "a100-80gb": {
        "architecture": "ampere",
        "aliases": [
            "a100 sxm4 80gb",
            "a100 80gb",
            "a100 80gb pcie",
            "a100 pcie 80gb",
            "a100 80gb gpu",
        ],
        "memory_visible_mib": 81920,
        "memory_nameplate_gib": 80,
        "interconnect": Interconnect(kind="nvlink", link_bandwidth=300e9, max_devices_per_node=8),
    },
    "a100-40gb": {
        "architecture": "ampere",
        "aliases": ["a100 sxm4 40gb", "a100 pcie 40gb", "a100 40gb", "a100", "a100 40gb gpu"],
        "memory_visible_mib": 40960,
        "memory_nameplate_gib": 40,
        "hbm_bandwidth": 1.555e12,
        "interconnect": Interconnect(kind="nvlink", link_bandwidth=300e9, max_devices_per_node=8),
    },
    "l40s-48gb": {
        "architecture": "ada",
        "aliases": ["l40s", "l40s 48gb", "l40s 48gb gpu"],
        "memory_visible_mib": 46068,
        "memory_nameplate_gib": 48,
        "interconnect": _PCIE4,
    },
    "rtx3050-laptop-4gb": {
        "architecture": "ampere",
        "compute_capability": (8, 6),
        "aliases": [
            "geforce rtx 3050 laptop gpu",
            "rtx 3050 laptop gpu",
            "rtx 3050 laptop",
            "rtx3050 laptop",
        ],
        "memory_visible_mib": 3768,
        "memory_nameplate_gib": 4,
        "hbm_bandwidth": 192e9,
        "flops": {"bf16": 16e12, "fp16": 16e12, "int8": 32e12},
        "interconnect": Interconnect(kind="none", max_devices_per_node=1),
    },
    "gb10": {
        "architecture": "blackwell",
        "compute_capability": (12, 1),
        "aliases": ["gb10", "nvidia gb10", "dgx spark"],
        "memory_visible_mib": 122572,
        "memory_nameplate_gib": 128,
        "hbm_bandwidth": 273e9,
        "flops": {"bf16": 125e12, "fp16": 125e12, "fp8": 250e12, "int8": 250e12, "fp4": 500e12},
        "interconnect": Interconnect(kind="none", max_devices_per_node=1),
    },
    "b200-180gb": {
        "architecture": "blackwell",
        "aliases": ["b200", "b200 180gb", "b200 gpu"],
        "memory_visible_mib": 183359,
        "memory_nameplate_gib": 180,
        "interconnect": Interconnect(kind="nvlink", link_bandwidth=900e9, max_devices_per_node=8),
    },
}

_CONFIG_ARCH = {
    "HOPPER": "hopper",
    "AMPERE": "ampere",
    "ADA": "ada",
    "ADA_LOVELACE": "ada",
    "ADA LOVELACE": "ada",
    "BLACKWELL": "blackwell",
    "VOLTA": "volta",
    "TURING": "turing",
}

_FAMILY_KEYWORDS = (
    ("blackwell", ("b100", "b200", "b300", "gb200", "gb300", "gb10", "rtx 50", "rtx pro 6000")),
    ("hopper", ("h100", "h200", "h800", "h20", "gh200")),
    ("ada", ("l40", "l4 ", "rtx 40", "rtx 6000 ada", "l20")),
    (
        "ampere",
        ("a100", "a10", "a30", "a40", "a16", "a2 ", "a6000", "a5000", "a4000", "rtx 30", "a800"),
    ),
    ("turing", ("t4", "rtx 20", "quadro rtx")),
    ("volta", ("v100",)),
)


def normalise(product: str) -> str:
    """Lower-case, vendor prefixes dropped, separators collapsed: ``NVIDIA-H100XM-80C`` -> ``h100xm 80c``."""
    text = re.sub(r"[_\-/]+", " ", str(product or "").lower())
    text = re.sub(r"\b(nvidia|grid|geforce|tesla)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def architecture_from_cc(cc: Optional[Tuple[int, int]]) -> Optional[str]:
    if not cc:
        return None
    major, minor = cc
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8:
        return "ada" if minor >= 9 else "ampere"
    if major == 7:
        return "turing" if minor >= 5 else "volta"
    return None


def architecture_from_product(product: str) -> Optional[str]:
    text = normalise(product) + " "
    for arch, keys in _FAMILY_KEYWORDS:
        if any(k in text for k in keys):
            return arch
    return None


def _arch(name: str) -> Dict[str, Any]:
    return _ARCHITECTURES.get(name) or _ARCHITECTURES["unknown"]


def _device_class(arch: str) -> str:
    return "cpu" if arch == "cpu" else "cuda"


def _build(sku: str, spec: Mapping[str, Any]) -> DeviceFacts:
    arch_name = spec["architecture"]
    arch = _arch(arch_name)
    sources = {
        f: "catalogue"
        for f in ("memory_visible_bytes", "hbm_bandwidth", "peak_flops", "interconnect")
    }
    return DeviceFacts(
        sku=sku,
        architecture=arch_name,
        device_class=_device_class(arch_name),
        memory_visible_bytes=int(spec["memory_visible_mib"]) * MIB,
        memory_nameplate_bytes=int(spec["memory_nameplate_gib"]) * GIB,
        hbm_bandwidth=float(spec.get("hbm_bandwidth", arch["hbm_bandwidth"])),
        peak_flops=dict(spec.get("flops", arch["flops"])),
        dtype_support=dict(spec.get("dtype", arch["dtype"])),
        kernel_support=frozenset(spec.get("kernels", arch["kernels"])),
        interconnect=spec.get("interconnect", Interconnect()),
        virtualization=spec.get("virtualization", Virtualization()),
        compute_capability=tuple(spec.get("compute_capability") or arch["cc"] or ()) or None,  # type: ignore[arg-type]
        sources=sources,
    )


_BUILT: Dict[str, DeviceFacts] = {sku: _build(sku, spec) for sku, spec in _SKUS.items()}
_ALIASES: Dict[str, str] = {}
for _sku, _spec in _SKUS.items():
    _ALIASES[normalise(_sku)] = _sku
    for _alias in _spec.get("aliases", ()):
        _ALIASES[normalise(_alias)] = _sku


def skus() -> Tuple[str, ...]:
    return tuple(sorted(_BUILT))


def get(sku: str) -> DeviceFacts:
    """A catalogue SKU by id. Raises KeyError for an unknown id (use :func:`lookup` for product strings)."""
    return _BUILT[sku]


def architecture_defaults(architecture: str, memory_bytes: Optional[int] = None) -> DeviceFacts:
    """Facts for an unknown SKU of ``architecture``, scaled to ``memory_bytes`` (FR-DEV-3)."""
    name = architecture if architecture in _ARCHITECTURES else "unknown"
    arch = _arch(name)
    ref = int(arch["ref_memory_gib"]) * GIB
    memory = int(memory_bytes or ref)
    scale = memory / ref if ref else 1.0
    gib = int(round(memory / GIB))
    return DeviceFacts(
        sku=f"derived:{name}:{gib}gb",
        architecture=name,
        device_class=_device_class(name),
        memory_visible_bytes=memory,
        memory_nameplate_bytes=None,
        hbm_bandwidth=float(arch["hbm_bandwidth"]) * scale,
        peak_flops=dict(arch["flops"]),
        dtype_support=dict(arch["dtype"]),
        kernel_support=frozenset(arch["kernels"]),
        compute_capability=arch["cc"],
        derived=True,
        sources={
            "memory_visible_bytes": "derived",
            "hbm_bandwidth": "derived",
            "peak_flops": "derived",
            "interconnect": "derived",
        },
    )


def from_hardware_config(name: str) -> DeviceFacts:
    """A ``hardware/configs.py`` record as :class:`DeviceFacts` (FR-DEV-1: the old keys map onto the fields)."""
    from ..hardware.configs import HARDWARE_CONFIGS

    record = HARDWARE_CONFIGS[name]
    arch_name = _CONFIG_ARCH.get(str(record.get("architecture") or "").upper())
    cc = _parse_cc(record.get("compute_capability"))
    arch_name = (
        arch_name
        or architecture_from_cc(cc)
        or ("cpu" if str(record.get("type")) == "cpu" else "unknown")
    )
    arch = _arch(arch_name)
    bf16 = float(record.get("Flops") or 0.0) * 1e12
    flops = {"bf16": bf16, "fp16": bf16}
    if arch["dtype"].get("fp8") == NATIVE:
        flops["fp8"] = 2 * bf16
    if "int8" in arch["dtype"]:
        flops["int8"] = 2 * bf16
    if arch["dtype"].get("mxfp4") == NATIVE:
        flops["fp4"] = 4 * bf16
    icn_kind = str(record.get("interconnect") or "").lower()
    link = float(record.get("ICN") or 0.0) * 1e9
    interconnect = Interconnect(
        kind="nvlink" if "nvlink" in icn_kind else "pcie" if "pcie" in icn_kind else "none",
        link_bandwidth=link,
        max_devices_per_node=8,
    )
    memory = int(float(record.get("Memory_size") or 0) * GIB)
    return DeviceFacts(
        sku="cfg:" + normalise(name).replace(" ", "-"),
        architecture=arch_name,
        device_class=_device_class(arch_name),
        memory_visible_bytes=memory,
        memory_nameplate_bytes=memory,
        hbm_bandwidth=float(record.get("Memory_BW") or 0.0) * 1e9,
        peak_flops=flops,
        dtype_support=dict(arch["dtype"]),
        kernel_support=frozenset(arch["kernels"]),
        interconnect=interconnect,
        compute_capability=cc or arch["cc"],
        sources={
            k: "catalogue"
            for k in ("memory_visible_bytes", "hbm_bandwidth", "peak_flops", "interconnect")
        },
    )


def _parse_cc(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    match = re.match(r"^\s*(\d+)\.(\d+)", str(value))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _config_aliases() -> Dict[str, str]:
    from ..hardware.configs import HARDWARE_CONFIGS

    table: Dict[str, str] = {}
    for name, record in HARDWARE_CONFIGS.items():
        if str(record.get("type")) not in ("gpu", "accelerator"):
            continue
        for alias in [name, *(record.get("aliases") or [])]:
            table.setdefault(normalise(alias), name)
    return table


def lookup(product: str) -> Optional[DeviceFacts]:
    """The catalogue entry for a product string as GFD, NFD or nvidia-smi spell it, or None."""
    key = normalise(product)
    if not key:
        return None
    if key in _ALIASES:
        return _BUILT[_ALIASES[key]]
    stripped = re.sub(r"\b(gpu|hbm3|hbm3e|hbm2e|sxm\d?|pcie)\b", " ", key)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped in _ALIASES:
        return _BUILT[_ALIASES[stripped]]
    configs = _config_aliases()
    if key in configs:
        return from_hardware_config(configs[key])
    return None


def parse_cc(value: Any) -> Optional[Tuple[int, int]]:
    return _parse_cc(value)


def known_architectures() -> FrozenSet[str]:
    return frozenset(_ARCHITECTURES)
