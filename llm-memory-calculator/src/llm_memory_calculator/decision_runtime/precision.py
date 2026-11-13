"""The precision Bud-Decision-Runtime will run in on a device, from what the cluster reports (Bud spec 031 D-35).

The runtime chooses on the node by timing multiplies (``basal_engine.hardware``); budsim cannot time anything, so this
mirrors the rule from the facts NFD and the hardware table give:

- CUDA, compute capability 8.0 or later: ``mixed16`` (bfloat16 tensor cores);
- CUDA 7.x with tensor cores (T4, V100): ``half16``;
- CUDA without tensor cores (GTX 1650): ``float32``: its float16 path is no faster;
- CPU with AMX: ``mixed16``; any other CPU: ``float32``.

``int8`` is never sized: it changes answers (D-35). When the runtime finds a faster mode on the node, the plan stays
conservative.

The runtime serves CUDA GPUs and CPUs only: a ROCm, Gaudi, Intel or other accelerator is not a device it runs on
(``device_kind`` is None for it, and the plan refuses it).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

MIXED16, HALF16, FLOAT32 = "mixed16", "half16", "float32"
PRECISIONS = (MIXED16, HALF16, FLOAT32)
#: bytes of a weight held in each precision. ``mixed16`` assumes the runtime holds its weights in bfloat16; Laya,
#: Julia and GLiNER store theirs in float32, so if it autocasts over the stored weights instead, the term is half the
#: real one (about 0.8 GiB short for Laya). Unverified until the benchmark records peak memory (runtime v0.1.1,
#: ``tools/decision_calibration.py``); ``tests/test_decision_runtime.py::test_memory_terms`` pins the assumption.
WEIGHT_BYTES = {MIXED16: 2.0, HALF16: 2.0, FLOAT32: 4.0}

CUDA, CPU = "cuda", "cpu"


def _type(device: Mapping[str, Any]) -> str:
    return str(device.get("type") or device.get("device_type") or "").lower()


def is_cpu(device: Mapping[str, Any]) -> bool:
    return _type(device).startswith("cpu")


def device_kind(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """``cuda`` | ``cpu`` | None: what the runtime would run on, None for a device it does not serve.

    ``cuda`` is a device reported as CUDA, or a GPU reported generically (``gpu``) that the hardware table or its
    name says is NVIDIA's. ROCm, HPU, XPU and every other accelerator are None.
    """
    kind = _type(device)
    if kind.startswith("cpu"):
        return CPU
    if kind == CUDA:
        return CUDA
    if kind in ("gpu", ""):
        vendor = str((hardware or {}).get("manufacturer") or "").lower()
        name = str(device.get("raw_name") or device.get("name") or device.get("device_name") or "").lower()
        if vendor == "nvidia" or "nvidia" in name or str(device.get("pci_vendor") or "").lower() == "10de":
            return CUDA
    return None


def compute_capability(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> Optional[float]:
    for source in (device, hardware or {}):
        value = source.get("compute_capability")
        if value in (None, ""):
            continue
        try:
            return float(str(value).strip())
        except ValueError:
            continue
    return None


def _flag(device: Mapping[str, Any], key: str, feature: str) -> Optional[bool]:
    """NFD's boolean (``cpu_amx``, ``cpu_avx512``), else the CPU's feature list (``amx_bf16``, ``avx512f``)."""
    value = device.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    features = device.get("cpu_features") or device.get("cpu_flags")
    if isinstance(features, (list, tuple, set)):
        return feature in {str(f).lower() for f in features}
    return None


def cpu_isa(device: Mapping[str, Any]) -> str:
    """``amx`` | ``avx512`` | ``avx2``: the instruction set the CPU's multiplies run on."""
    if _flag(device, "cpu_amx", "amx_bf16"):
        return "amx"
    if _flag(device, "cpu_avx512", "avx512f"):
        return "avx512"
    return "avx2"


def choose(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    """(precision, reason) for ``device`` (budsim's device dict); ``hardware`` is its hardware-table entry."""
    kind = device_kind(device, hardware)
    if kind is None:
        return FLOAT32, f"Bud-Decision-Runtime does not run on {_type(device) or 'this'} devices: it serves CUDA and CPU"
    if kind == CPU:
        if cpu_isa(device) == "amx":
            return MIXED16, "bfloat16 on the CPU's AMX units"
        return FLOAT32, "no AMX on this CPU: bfloat16 would be emulated, 3.3x slower than float32"
    cc = compute_capability(device, hardware)
    cores = str((hardware or {}).get("tensor_cores") or "").lower()
    if cc is not None and cc >= 8.0:
        return MIXED16, f"bfloat16 tensor cores (compute capability {cc:g})"
    if cc is not None and cc >= 7.0 and cores not in ("", "none"):
        return HALF16, f"float16 tensor cores; no bfloat16 below compute capability 8.0 (this GPU: {cc:g})"
    if cc is not None:
        return FLOAT32, f"no tensor cores (compute capability {cc:g}): float16 is no faster than float32 here"
    if cores.startswith("gen") and cores not in ("gen1", "gen2"):
        return MIXED16, "bfloat16 tensor cores (hardware table)"
    return FLOAT32, "the GPU's compute capability is not known: sized in float32"
