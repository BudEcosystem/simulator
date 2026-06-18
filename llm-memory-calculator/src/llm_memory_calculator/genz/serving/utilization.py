"""Roofline utilization (MFU / MBU) derived from a GenZ modeling result.

SV4 (solutions_round2.md §4): the serving power path must scale power with the *real* compute/memory
utilization the engine already computes, not a frozen ``0.5`` default (router) or a batch-slot-occupancy
proxy (sim loop). ``ModdelingOutput`` carries the per-operator ``model_df`` (FLOPs, bytes moved, latency);
the device peaks come from the single unified hardware source (``get_hardware_config``). This module turns
those into:

    MFU = achieved_FLOPs_per_s / (num_devices x peak_FLOPs_per_s)
    MBU = achieved_bytes_per_s / (num_devices x peak_bytes_per_s)

which vary correctly with batch size and phase (prefill -> high MFU; batch=1 decode -> low MFU, high MBU),
instead of a constant. No tuned constants: every quantity is the engine's own output or a datasheet peak.
"""
from typing import Tuple

# scale factors to base units (FLOP, byte, second) for the unit suffix in a model_df column header
_FLOP_SCALE = {'FLOP': 1.0, 'KFLOP': 1e3, 'MFLOP': 1e6, 'GFLOP': 1e9, 'TFLOP': 1e12, 'PFLOP': 1e15}
_MEM_SCALE = {'B': 1.0, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12}


def _sum_column(df, prefix: str, scale_map: dict) -> float:
    """Sum a model_df column whose header starts with ``prefix`` (e.g. 'Num ops', 'Total Data'),
    converting its unit suffix '(XB)'/'(XFLOP)' to base units. Returns 0.0 if the column is absent."""
    for col in df.columns:
        if col.startswith(prefix) and '(' in col and col.endswith(')'):
            unit = col[col.rindex('(') + 1:-1].strip()
            scale = scale_map.get(unit)
            if scale is None:
                continue
            try:
                return float(df[col].fillna(0).sum()) * scale
            except Exception:
                return 0.0
    return 0.0


def roofline_utilization(result, hardware, num_devices: int = 1) -> Tuple[float, float]:
    """Return ``(mfu, mbu)`` in [0, 1] for a GenZ ``ModdelingOutput`` on ``hardware``.

    Uses ``result.summary_table`` — the repeat-corrected per-model totals ('MACs (...)' which the GenZ
    roofline path already expresses as 2xMACs FLOPs, and 'Total Data (...)' the bytes moved across all
    layers). The raw ``model_df`` stores only ONE transformer block with a Repeat multiplier, so summing
    it directly undercounts an N-layer model ~Nx; the summary applies that multiplier. ``result.Latency``
    is the achieved latency (ms); device peak FLOPs/BW come from ``get_hardware_config``. Returns
    ``(0.0, 0.0)`` when data is unavailable (e.g. a mocked result) so callers degrade to idle power.
    """
    table = getattr(result, 'summary_table', None)
    latency_ms = getattr(result, 'Latency', 0) or 0
    if table is None or latency_ms <= 0:
        return 0.0, 0.0

    try:
        from llm_memory_calculator.hardware import get_hardware_config
        hw = get_hardware_config(hardware) or {}
    except Exception:
        hw = {}
    peak_flops = float(hw.get('Flops', 0) or 0) * 1e12      # TFLOPS -> FLOP/s (dense bf16 peak, post-C1)
    peak_bw = float(hw.get('Memory_BW', 0) or 0) * 1e9       # GB/s   -> byte/s
    nd = max(int(num_devices), 1)

    latency_s = latency_ms / 1000.0
    total_flops = _sum_column(table, 'MACs', _FLOP_SCALE)
    total_bytes = _sum_column(table, 'Total Data', _MEM_SCALE)

    mfu = (total_flops / latency_s / (nd * peak_flops)) if peak_flops > 0 else 0.0
    mbu = (total_bytes / latency_s / (nd * peak_bw)) if peak_bw > 0 else 0.0
    # Physical-limit clamp: utilization cannot exceed 1.0 (a value >1 would indicate a peak mismatch).
    return max(0.0, min(mfu, 1.0)), max(0.0, min(mbu, 1.0))
