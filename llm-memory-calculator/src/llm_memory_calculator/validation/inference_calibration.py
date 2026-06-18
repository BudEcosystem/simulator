"""Inference latency calibration for GenZ — the *systematic* way to make any hardware accurate.

The GenZ inference roofline (`max(compute, memory)`, efficiency=1, no launch/per-stream overhead) is
optimistic on real hardware. This module fits a small per-hardware constant set — the same one the
`inference_calibration` block on a hardware config carries — against MEASURED benchmarks, by running
the REAL `prefill_moddeling`/`decode_moddeling` (the production path), so the fitted constants match
runtime exactly. It complements the training-only `calibration_engine` and reuses the same
`differential_evolution` optimizer and JSON-friendly style.

Equation (per phase), applied inside GenZ given the block:
    decode  TPOT(B,T) = raw(B,T)/eta_mem      + N_ops·t_launch + c_stream·layers·(B-1)
    prefill TTFT(T)   = raw(T)/eta_compute|mem + N_ops·t_launch
GenZ supplies raw/N_ops/layers from the model architecture → generalizes across GQA/MQA/MLA/MoE.

Calibrate a new hardware:
    python -m llm_memory_calculator.validation.inference_calibration GB10
prints the ready-to-paste `inference_calibration` block + per-point residuals (config edits stay
human-reviewed). Add the hardware's measured grid to `INFERENCE_BENCHMARKS` first.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InferenceBenchmark:
    """One measured inference latency point on a real serving stack (e.g. llama.cpp)."""
    hardware: str
    model_name: str
    model_config: dict          # HF-style config dict (so GenZ resolves the architecture offline)
    phase: str                  # 'decode' | 'prefill'
    batch_size: int
    input_tokens: int
    output_tokens: int
    measured_ms: float          # decode: per-token TPOT; prefill: TTFT
    model_params_b: float = 0.0
    notes: str = ""


# Qwen2.5 architectures used for the GB10 grid (q4 weights → modelled at int8 for the raw roofline).
_QWEN_05B = dict(model_type="qwen2", hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
                 num_key_value_heads=2, intermediate_size=4864, vocab_size=151936,
                 max_position_embeddings=32768, name="qwen2.5-0.5b-cal")
_QWEN_3B = dict(model_type="qwen2", hidden_size=2048, num_hidden_layers=36, num_attention_heads=16,
                num_key_value_heads=2, intermediate_size=11008, vocab_size=151936,
                max_position_embeddings=32768, name="qwen2.5-3b-cal")


def _dec(cfg, params_b, B, it, ot, ms):
    return InferenceBenchmark(hardware="GB10", model_name=cfg["name"], model_config=cfg,
                              phase="decode", batch_size=B, input_tokens=it, output_tokens=ot,
                              measured_ms=ms, model_params_b=params_b, notes="llama.cpp ngl=999 q4")


def _pre(cfg, params_b, it, ms):
    return InferenceBenchmark(hardware="GB10", model_name=cfg["name"], model_config=cfg,
                              phase="prefill", batch_size=1, input_tokens=it, output_tokens=1,
                              measured_ms=ms, model_params_b=params_b, notes="llama.cpp ngl=999 q4")


# Measured GB10 grid (real llama.cpp, qwen2.5 0.5B & 3B q4_k_m): batch B∈{1,2,4,8} × context.
INFERENCE_BENCHMARKS: Dict[str, List[InferenceBenchmark]] = {
    "GB10": [
        _dec(_QWEN_05B, 0.5, 1, 504, 160, 2.90), _dec(_QWEN_05B, 0.5, 2, 504, 160, 3.27),
        _dec(_QWEN_05B, 0.5, 4, 504, 160, 4.06), _dec(_QWEN_05B, 0.5, 8, 504, 160, 6.02),
        _dec(_QWEN_05B, 0.5, 1, 4004, 160, 2.93), _dec(_QWEN_05B, 0.5, 1, 12005, 160, 3.59),
        _dec(_QWEN_3B, 3.0, 1, 504, 160, 11.95), _dec(_QWEN_3B, 3.0, 2, 504, 160, 12.90),
        _dec(_QWEN_3B, 3.0, 4, 504, 160, 13.71), _dec(_QWEN_3B, 3.0, 8, 504, 160, 18.04),
        _dec(_QWEN_3B, 3.0, 1, 4004, 160, 14.18), _dec(_QWEN_3B, 3.0, 1, 12005, 160, 16.16),
        _pre(_QWEN_05B, 0.5, 900, 54.0), _pre(_QWEN_05B, 0.5, 1800, 90.0),
        _pre(_QWEN_3B, 3.0, 900, 155.0), _pre(_QWEN_3B, 3.0, 1800, 300.0),
    ],
}


@dataclass
class InferenceCalibration:
    """The per-hardware inference calibration block + the fit residuals it was derived from."""
    decode: dict = field(default_factory=dict)
    prefill: dict = field(default_factory=dict)
    decode_rms_pct: float = 0.0
    prefill_rms_pct: float = 0.0

    def block(self) -> dict:
        """The `inference_calibration` block to paste into the hardware config."""
        return {"decode": self.decode, "prefill": self.prefill}


def _base_config(hardware: str) -> dict:
    """The hardware config WITHOUT any inference_calibration block (so we fit against raw)."""
    from llm_memory_calculator import get_hardware_config
    cfg = get_hardware_config(hardware)
    if not cfg:
        raise ValueError(f"Unknown hardware: {hardware}")
    return {k: v for k, v in cfg.items() if k != "inference_calibration"}


def _register(bench: List[InferenceBenchmark]):
    from llm_memory_calculator.genz.Models.default_models import MODEL_DICT
    from llm_memory_calculator.genz.Models.get_language_model import huggingface_config_to_model_config
    names = {}
    for b in bench:
        mc = huggingface_config_to_model_config(b.model_config, b.model_config["name"])
        MODEL_DICT.add_model(mc)
        names[b.model_name] = mc.model
    return names


def fit_inference_calibration(hardware: str, bits: str = "int8",
                              benchmarks: Optional[List[InferenceBenchmark]] = None) -> InferenceCalibration:
    """Fit the `inference_calibration` block for `hardware` against its measured benchmarks by
    running the real GenZ inference path. Decode (memory-bound) is a closed-form least-squares;
    prefill (mixed compute/memory) uses bounded differential_evolution over the production path."""
    from scipy.optimize import differential_evolution
    from llm_memory_calculator import estimate_decode_performance, estimate_prefill_performance
    from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling
    from llm_memory_calculator.genz.analyse_model import count_repeat_aware_ops

    bench = benchmarks if benchmarks is not None else INFERENCE_BENCHMARKS.get(hardware, [])
    if not bench:
        raise ValueError(f"No inference benchmarks for {hardware}; add them to INFERENCE_BENCHMARKS.")
    base = _base_config(hardware)
    names = _register(bench)
    dec = [b for b in bench if b.phase == "decode"]
    pre = [b for b in bench if b.phase == "prefill"]

    # ---- decode: linear closed-form  real = raw·(1/eta_mem) + N_ops·t_launch + L·(B-1)·c_stream ----
    A, y = [], []
    for b in dec:
        raw = estimate_decode_performance(model=names[b.model_name], batch_size=b.batch_size,
                                          input_tokens=b.input_tokens, output_tokens=b.output_tokens,
                                          system_name=base, bits=bits)["Latency"]
        df, _ = decode_moddeling(model=names[b.model_name], batch_size=b.batch_size,
                                 input_tokens=b.input_tokens, output_tokens=b.output_tokens,
                                 system_name=base, bits=bits, model_profilling=True)
        n_ops = count_repeat_aware_ops(df)
        reps = [df.loc[i, "Dimension"] for i in range(len(df)) if df.loc[i, "Op Type"] == "Repeat"]
        n_layers = max(reps) if reps else 1
        A.append([raw, n_ops, n_layers * max(0, b.batch_size - 1)])
        y.append(b.measured_ms)
    A, y = np.array(A), np.array(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    inv_eta_mem, t_launch_d, c_stream = (max(c, 0.0) for c in coef)
    decode = {"eta_mem": round(1.0 / inv_eta_mem, 4) if inv_eta_mem > 0 else 1.0,
              "t_launch_ms": round(t_launch_d, 5), "c_stream_ms_per_layer": round(c_stream, 5)}
    dec_pred = A @ np.array([1.0 / decode["eta_mem"], decode["t_launch_ms"], decode["c_stream_ms_per_layer"]])
    decode_rms = float(np.sqrt(np.mean(((dec_pred - y) / y) ** 2)) * 100)

    # ---- prefill: bounded production-path differential_evolution over (eta_compute, eta_mem, t_launch) ----
    def pre_err(x):
        ec, em, tl = x
        h = copy.deepcopy(base)
        h["inference_calibration"] = {"prefill": {"eta_compute": ec, "eta_mem": em, "t_launch_ms": tl}}
        e = [(estimate_prefill_performance(model=names[b.model_name], batch_size=1,
                                           input_tokens=b.input_tokens, system_name=h, bits=bits)["Latency"]
              - b.measured_ms) / b.measured_ms for b in pre]
        return float(np.sqrt(np.mean(np.square(e))))

    prefill, prefill_rms = {}, 0.0
    if pre:
        r = differential_evolution(pre_err, [(0.03, 0.5), (0.3, 1.0), (0.0, 0.3)],
                                   seed=1, maxiter=25, popsize=12, tol=1e-3)
        prefill = {"eta_compute": round(r.x[0], 4), "eta_mem": round(r.x[1], 4),
                   "t_launch_ms": round(r.x[2], 5)}
        prefill_rms = float(r.fun * 100)

    return InferenceCalibration(decode=decode, prefill=prefill,
                                decode_rms_pct=round(decode_rms, 1), prefill_rms_pct=round(prefill_rms, 1))


def _main(argv):
    import json
    hardware = argv[1] if len(argv) > 1 else "GB10"
    cal = fit_inference_calibration(hardware)
    print(f"# Fitted inference_calibration for {hardware} "
          f"(decode RMS {cal.decode_rms_pct}%, prefill RMS {cal.prefill_rms_pct}%)")
    print(f"# Paste into HARDWARE_CONFIGS['{hardware}']:")
    print("'inference_calibration': " + json.dumps(cal.block()))


if __name__ == "__main__":
    import sys
    _main(sys.argv)
