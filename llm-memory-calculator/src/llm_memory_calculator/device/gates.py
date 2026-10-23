"""Capability gates (FR-DEV-4): a module cannot choose what the device cannot run.

``allows_dtype`` answers whether a weight/KV format can run and by which path. A ``native`` path is always
allowed. A fallback path (``weight_only`` FP8 on Ampere via Marlin, ``emulated`` MXFP4 on Hopper via Triton)
has a different speed profile, so it is allowed only when the engine record lists the kernel or the
grounding store holds an entry for that path -- otherwise the planner would price it with the native ratio.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .facts import DeviceFacts

#: The engine-record feature that implements a fallback path.
FALLBACK_FEATURES = {
    "mxfp4:emulated": "mxfp4_triton",
    "int4:weight_only": "int4_marlin",
    "fp8:weight_only": "fp8_marlin",
}

_PATH_NAMES = {"emulated": "triton", "weight_only": "marlin", "native": "native"}

#: Formats that are not a precision choice (the model's own 16-bit dtypes).
_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "auto": "bf16",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
}


def canonical_format(fmt: str) -> str:
    text = str(fmt or "bf16").lower()
    return _ALIASES.get(text, text)


def allows_dtype(
    facts: DeviceFacts,
    fmt: str,
    *,
    record_features: Iterable[str] = (),
    grounded_paths: Iterable[str] = (),
) -> Optional[str]:
    """The path a format runs by on this device (``native``, ``weight_only``, ``emulated``), or None."""
    fmt = canonical_format(fmt)
    path = facts.dtype_support.get(fmt)
    if path is None:
        return None
    if path == "native":
        return path
    key = f"{fmt}:{path}"
    if key in set(grounded_paths) or FALLBACK_FEATURES.get(key) in set(record_features):
        return path
    return None


def kernel_path(facts: DeviceFacts, record_features: Iterable[str], fmt: str) -> Optional[str]:
    """``<format>:<kernel family>`` for the path :func:`allows_dtype` grants, or None."""
    fmt = canonical_format(fmt)
    path = allows_dtype(
        facts, fmt, record_features=record_features, grounded_paths=_all_grounded(facts, fmt)
    )
    if path is None:
        return None
    return f"{fmt}:{_PATH_NAMES.get(path, path)}"


def _all_grounded(facts: DeviceFacts, fmt: str) -> Iterable[str]:
    # kernel_path reports the path for a format the caller already gated; it does not re-check grounding.
    path = facts.dtype_support.get(fmt)
    return [f"{fmt}:{path}"] if path and path != "native" else []


def max_tp(facts: DeviceFacts, limits: Optional[Mapping[str, Any]] = None) -> int:
    """TP ceiling: the record's ``max_tp``, the record's per-virtualization override, the node's device count,
    and the device's own peer-access facts (a vGPU without VMM has none)."""
    limits = limits or {}
    ceiling = int(limits.get("max_tp") or 8)
    by_virt = limits.get("max_tp_by_virtualization") or {}
    klass = facts.virtualization.klass
    if klass in by_virt:
        ceiling = min(ceiling, int(by_virt[klass]))
    if klass in ("vgpu_no_vmm", "mig") or facts.interconnect.kind == "none":
        ceiling = 1
    per_node = int(facts.interconnect.max_devices_per_node or 1)
    return max(1, min(ceiling, per_node if facts.interconnect.kind != "none" else 1))


def host_register_allowed(facts: DeviceFacts) -> bool:
    """FRD-023's register-mode T1 needs ``cudaHostRegister``; refused on a vGPU that crashes on it."""
    return bool(facts.virtualization.host_register)
