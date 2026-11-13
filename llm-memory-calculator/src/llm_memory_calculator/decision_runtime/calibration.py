"""Measured efficiency of Bud-Decision-Runtime per device, precision and family (Bud spec 031 D-45, C10.1).

``calibration.json`` is written by ``tools/decision_calibration.py`` from the runtime's decision benchmark
(``engine/bench/decision_bench/baselines/*.json``). An entry's ``eff`` is the share of the device's peak multiply rate
the runtime reached at saturation: measured tokens/s x FLOPs per token / peak FLOP/s. Sizing another workload on the
same device scales the measured throughput by the workload's FLOPs per token, so the entry carries over to other row
lengths and, through the device's own peak, to devices never measured.

A device class is a GPU's hardware-table name or a CPU's model (``cpu-xeon-6767p``): only the part measured is
``calibrated``. Lookup, most specific first (FRD §5.16.3):

1. the exact device class and precision: the family, else the lowest of its kind (``calibrated``);
2. the same vendor architecture (an Ampere entry for an A10) or CPU instruction set (``same-architecture``), again
   the family, else the lowest of its kind;
3. the lowest efficiency measured on that device type in that precision, then in any precision, x 0.8 (``estimated``);
   only here does an encoder's entry stand in for a decoder or the other way round.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple

from .precision import CPU, FLOAT32, compute_capability, cpu_isa, device_kind, is_cpu

_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
#: what an estimate keeps of the lowest efficiency measured
ESTIMATE_FACTOR = 0.8
#: the nominal all-core clock a CPU's peak is computed at. A calibrated entry was measured against the same figure, so
#: on the CPU model it was measured on the figure cancels out; on any other CPU (``same-architecture`` or
#: ``estimated``) the plan assumes that CPU sustains the measured one's clock, which a laptop's turbo overstates
CPU_NOMINAL_GHZ = 2.0
#: multiply FLOPs per cycle per core: AMX bfloat16 tiles, AVX-512 (two FMA ports) and AVX2 float32
_CPU_FLOPS_PER_CYCLE = {("amx", "mixed16"): 1024.0, "avx512": 64.0, "avx2": 32.0}
#: float32 (no TF32) over the dense 16-bit tensor rate the hardware table holds, by compute capability: the
#: data-center dies keep far fewer float32 lanes per tensor FLOP than the GA10x / AD10x parts (0.25).
#: V100 15.7 of 125, T4 8.1 of 65, A100 19.5 of 312, H100 SXM 67 of 989.5, B200 80 of 2250 TFLOPS.
_FP32_SHARE_BY_CC = {7.0: 15.7 / 125, 7.5: 8.1 / 65, 8.0: 19.5 / 312, 9.0: 67 / 989.5, 10.0: 80 / 2250}
_FP32_SHARE_BY_ARCH = {"VOLTA": _FP32_SHARE_BY_CC[7.0], "HOPPER": _FP32_SHARE_BY_CC[9.0]}
#: GA10x / AD10x (RTX 3090 35.6 of 142, L40S 91.6 of 362) and a card without tensor cores (its float16 is 2x)
_FP32_SHARE_TENSOR, _FP32_SHARE_NO_TENSOR = 0.25, 0.5
#: words a CPU's name carries that do not tell one model from another
_CPU_NOISE = re.compile(r"\(r\)|\(tm\)|@.*$|\b\d+(st|nd|rd|th)\s+gen\b|\b(intel|amd|cpu|processor|\d+-core)\b")

CALIBRATED, SAME_ARCHITECTURE, ESTIMATED = "calibrated", "same-architecture", "estimated"


@dataclass(frozen=True)
class Entry:
    key: str                       # "<device class>/<precision>/<family>"
    device_class: str
    device_type: str               # "cuda" | "cpu"
    architecture: str
    precision: str
    family: str
    kind: str
    eff: float
    t_pass_min_ms: float
    act_bytes_per_token: Optional[float]
    context_bytes: Optional[float]
    threads: Optional[int]
    runtime: str
    source: str


@lru_cache(maxsize=4)
def load(path: str = _PATH) -> Tuple[Entry, ...]:
    with open(path) as f:
        raw = json.load(f)
    entries = []
    for key, e in (raw.get("entries") or {}).items():
        entries.append(Entry(
            key=key, device_class=e["device_class"], device_type=e["device_type"], architecture=e["architecture"],
            precision=e["precision"], family=e["family"], kind=e["kind"], eff=float(e["eff"]),
            t_pass_min_ms=float(e["t_pass_min_ms"]), act_bytes_per_token=e.get("act_bytes_per_token"),
            context_bytes=e.get("context_bytes"), threads=e.get("threads"), runtime=str(e.get("runtime") or ""),
            source=str(e.get("source") or ""),
        ))
    return tuple(entries)


def runtime_version(path: str = _PATH) -> str:
    """The newest runtime release the calibration was measured on."""
    versions = sorted({e.runtime for e in load(path) if e.runtime}, key=_version_key)
    return versions[-1] if versions else ""


def _version_key(v: str) -> Tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def device_type(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]] = None) -> str:
    """``cuda`` | ``cpu``; ``unsupported`` for a device the runtime does not run on (no entry ever matches it)."""
    return device_kind(device, hardware) or "unsupported"


def cpu_model(device: Mapping[str, Any]) -> str:
    """A CPU's model from its name: ``Intel(R) Xeon(R) 6767P`` -> ``xeon-6767p``; ``unknown`` when it has none."""
    name = str(device.get("raw_name") or device.get("name") or device.get("cpu") or device.get("model_name") or "")
    words = re.findall(r"[a-z0-9]+", _CPU_NOISE.sub(" ", name.lower()))
    return "-".join(words) or "unknown"


def device_class(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> str:
    """A GPU's hardware-table name (``RTX3050_LAPTOP_GPU``); a CPU's model (``cpu-xeon-6767p``)."""
    if is_cpu(device):
        return f"cpu-{cpu_model(device)}"
    return str((hardware or {}).get("name") or device.get("name") or "unknown-gpu")


def architecture(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> str:
    if is_cpu(device):
        return f"cpu-{cpu_isa(device)}"
    return str((hardware or {}).get("architecture") or "unknown").upper()


def peak_flops(
    device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]], precision: str, threads: Optional[int]
) -> Optional[float]:
    """The device's peak multiply rate in the precision, FLOP/s; None when the hardware table does not know it or the
    runtime does not run on the device.

    GPU: the table's ``Flops`` (dense 16-bit tensor, TFLOPS); float32 runs at half of it on a card without tensor
    cores (its float16 rate is twice its float32) and at the die's float32 share on one with them
    (``_FP32_SHARE_BY_CC``). CPU: compute threads x the nominal clock x the instruction set's multiply FLOPs per cycle.
    """
    kind = device_kind(device, hardware)
    if kind is None:
        return None
    if kind == CPU:
        isa = cpu_isa(device)
        per_cycle = _CPU_FLOPS_PER_CYCLE.get((isa, precision)) or _CPU_FLOPS_PER_CYCLE["avx512" if isa != "avx2" else "avx2"]
        return max(int(threads or 1), 1) * CPU_NOMINAL_GHZ * 1e9 * per_cycle
    tflops = (hardware or {}).get("Flops")
    try:
        tflops = float(tflops)
    except (TypeError, ValueError):
        return None
    if tflops <= 0:
        return None
    if precision == FLOAT32:
        return tflops * 1e12 * fp32_share(device, hardware)
    return tflops * 1e12


def fp32_share(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> float:
    """A GPU's float32 rate over its dense 16-bit tensor rate."""
    cores = str((hardware or {}).get("tensor_cores") or "none").lower()
    if cores == "none":
        return _FP32_SHARE_NO_TENSOR
    cc = compute_capability(device, hardware)
    if cc is not None and cc in _FP32_SHARE_BY_CC:
        return _FP32_SHARE_BY_CC[cc]
    if cc is None:
        arch = str((hardware or {}).get("architecture") or "").upper()
        return _FP32_SHARE_BY_ARCH.get(arch, _FP32_SHARE_TENSOR)
    return _FP32_SHARE_TENSOR


def lookup(
    device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]], precision: str, family: str, kind: str,
    *, path: str = _PATH,
) -> Tuple[Optional[Entry], float, str]:
    """(entry, efficiency to use, basis). The entry is None only when the calibration holds nothing for the device
    type at all."""
    entries = load(path)
    dtype = device_type(device, hardware)
    dclass, arch = device_class(device, hardware), architecture(device, hardware)

    def best(cands, *, lowest: bool = False, any_kind: bool = False) -> Optional[Entry]:
        cands = list(cands)
        same = [e for e in cands if e.family == family]
        if same and not lowest:
            return same[0]
        of_kind = [e for e in cands if e.kind == kind] or (cands if any_kind else [])
        return min(of_kind, key=lambda e: e.eff) if of_kind else None

    exact = best(e for e in entries if e.device_type == dtype and e.device_class == dclass and e.precision == precision)
    if exact:
        return exact, exact.eff, CALIBRATED
    arch_match = best(e for e in entries if e.device_type == dtype and e.architecture == arch and e.precision == precision)
    if arch_match:
        return arch_match, arch_match.eff, SAME_ARCHITECTURE
    for cands in (
        [e for e in entries if e.device_type == dtype and e.precision == precision],
        [e for e in entries if e.device_type == dtype],
    ):
        low = best(cands, lowest=True, any_kind=True)
        if low:
            return low, low.eff * ESTIMATE_FACTOR, ESTIMATED
    return None, 0.0, ESTIMATED
