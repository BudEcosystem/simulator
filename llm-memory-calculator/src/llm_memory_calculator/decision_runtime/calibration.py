"""Measured efficiency of Bud-Decision-Runtime per device, precision and family (Bud spec 031 D-45, C10.1).

``calibration.json`` is written by ``tools/decision_calibration.py`` from the runtime's decision benchmark
(``engine/bench/decision_bench/baselines/*.json``). An entry's ``eff`` is the share of the device's peak multiply rate
the runtime reached at saturation: measured tokens/s x FLOPs per token / peak FLOP/s. Sizing another workload on the
same device scales the measured throughput by the workload's FLOPs per token, so the entry carries over to other row
lengths and, through the device's own peak, to devices never measured.

Lookup, most specific first (FRD §5.16.3):

1. the exact device class and precision: the family, else the lowest of its kind (``calibrated``);
2. the same vendor architecture (an Ampere entry for an A10) or CPU instruction set (``same-architecture``);
3. the lowest efficiency measured on that device type in that precision, then in any precision, x 0.8 (``estimated``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple

from .precision import FLOAT32, cpu_isa, is_cpu

_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
#: what an estimate keeps of the lowest efficiency measured
ESTIMATE_FACTOR = 0.8
#: the nominal all-core clock a CPU's peak is computed at: the calibration measured against the same figure, so only
#: an estimate on an unmeasured CPU depends on it
CPU_NOMINAL_GHZ = 2.0
#: multiply FLOPs per cycle per core: AMX bfloat16 tiles, AVX-512 (two FMA ports) and AVX2 float32
_CPU_FLOPS_PER_CYCLE = {("amx", "mixed16"): 1024.0, "avx512": 64.0, "avx2": 32.0}

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


def device_type(device: Mapping[str, Any]) -> str:
    return "cpu" if is_cpu(device) else "cuda"


def device_class(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> str:
    """A GPU's hardware-table name (``RTX3050_LAPTOP_GPU``); a CPU's instruction set (``cpu-amx``)."""
    if is_cpu(device):
        return f"cpu-{cpu_isa(device)}"
    return str((hardware or {}).get("name") or device.get("name") or "unknown-gpu")


def architecture(device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]]) -> str:
    if is_cpu(device):
        return f"cpu-{cpu_isa(device)}"
    return str((hardware or {}).get("architecture") or "unknown").upper()


def peak_flops(
    device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]], precision: str, threads: Optional[int]
) -> Optional[float]:
    """The device's peak multiply rate in the precision, FLOP/s; None when the hardware table does not know it.

    GPU: the table's ``Flops`` (dense 16-bit tensor, TFLOPS); float32 runs at half of it on a card without tensor
    cores (its float16 rate is twice its float32) and at a quarter on one with them. CPU: compute threads x the
    nominal clock x the instruction set's multiply FLOPs per cycle.
    """
    if is_cpu(device):
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
        cores = str((hardware or {}).get("tensor_cores") or "none").lower()
        return tflops * 1e12 * (0.5 if cores == "none" else 0.25)
    return tflops * 1e12


def lookup(
    device: Mapping[str, Any], hardware: Optional[Mapping[str, Any]], precision: str, family: str, kind: str,
    *, path: str = _PATH,
) -> Tuple[Optional[Entry], float, str]:
    """(entry, efficiency to use, basis). The entry is None only when the calibration holds nothing for the device
    type at all."""
    entries = load(path)
    dtype, dclass, arch = device_type(device), device_class(device, hardware), architecture(device, hardware)

    def best(cands, *, lowest: bool = False) -> Optional[Entry]:
        cands = list(cands)
        if not cands:
            return None
        same = [e for e in cands if e.family == family]
        if same and not lowest:
            return same[0]
        of_kind = [e for e in cands if e.kind == kind] or cands
        return min(of_kind, key=lambda e: e.eff)

    exact = best(e for e in entries if e.device_class == dclass and e.precision == precision)
    if exact:
        return exact, exact.eff, CALIBRATED
    arch_match = best(e for e in entries if e.device_type == dtype and e.architecture == arch and e.precision == precision)
    if arch_match:
        return arch_match, arch_match.eff, SAME_ARCHITECTURE
    for cands in (
        [e for e in entries if e.device_type == dtype and e.precision == precision],
        [e for e in entries if e.device_type == dtype],
    ):
        low = best(cands, lowest=True)
        if low:
            return low, low.eff * ESTIMATE_FACTOR, ESTIMATED
    return None, 0.0, ESTIMATED
