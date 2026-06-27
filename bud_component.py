#!/usr/bin/env python3
"""Bud Simulator component — the single estimation primitive other Bud OS components call.

Given a model (a local GGUF, a HuggingFace id, or an inline HF-style config) plus the serving shape
(batch = concurrent requests, seq = prompt+output context) and a hardware target (e.g. GB10), emit a
JSON report with the architecture-aware MEMORY footprint (weights / KV cache / activations / total)
and the PERFORMANCE SLOs (TTFT, TPOT, decode tok/s, throughput) — powered by llm-memory-calculator
/ GenZ. ALL model-architecture knowledge (GQA vs MQA vs MLA vs MoE vs Mamba, per-token KV, layer
shapes) lives here in the simulator, never in the Rust callers.

The estimator is ~80% accurate, so callers should apply a safety margin (the FCSP cap does).

Usage:
    python bud_component.py --gguf /path/model.gguf --hardware GB10 --batch 4 --seq 4096
    python bud_component.py --hf-id Qwen/Qwen2.5-3B --hardware GB10 --batch 1 --seq 2048
Output: a single JSON object on stdout. Non-zero exit on failure (caller falls back).
"""
import argparse
import json
import os
import sys

from llm_memory_calculator import get_hardware_config
from llm_memory_calculator import estimate_end_to_end_performance
from llm_memory_calculator.inference_engines import estimate_memory_for_engine
from llm_memory_calculator.genz.Models.default_models import MODEL_DICT
from llm_memory_calculator.genz.Models.get_language_model import (
    huggingface_config_to_model_config,
)

# llama.cpp serves the KV cache + compute buffers in f16 by default, independent of the (quantized)
# weight precision — so KV/activations are always sized at f16 for a realistic runtime budget.
RUNTIME_KV_PRECISION = "fp16"
# Fixed CUDA runtime context + driver + allocator fragmentation reserve (bytes) — not modelled by
# the analytic estimator but always present on a CUDA backend.
CUDA_RUNTIME_RESERVE = 1024 * 1024 * 1024

# NOTE: GB10 inference calibration now lives INSIDE GenZ (hardware/configs.py 'inference_calibration'
# block + genz/LLM_inference). estimate_end_to_end_performance(system_name='GB10') already returns
# CALIBRATED TTFT/TPOT, so this component no longer post-processes them — it just reads them. This
# makes the accuracy available to EVERY BudSimulator consumer (PyPI, API, Streamlit) with no API
# change, and Bud Gaia's external JSON schema is unchanged.


def gguf_to_hf_config(path):
    """Read a GGUF's architecture metadata and map it to an HF-style config dict (arch-aware here,
    not in Rust). Returns (config_dict, weight_bytes_on_disk)."""
    import gguf

    reader = gguf.GGUFReader(path)

    def val(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        try:
            return f.contents()  # recent gguf
        except Exception:
            # Fallback: single-value scalar fields
            if f.data is not None and len(f.data) > 0:
                return f.parts[f.data[0]].tolist() if hasattr(f.parts[f.data[0]], "tolist") else f.parts[f.data[0]]
            return None

    arch = val("general.architecture")
    if isinstance(arch, (bytes, bytearray)):
        arch = arch.decode("utf-8", "ignore")
    if not arch:
        raise ValueError("GGUF: no general.architecture")

    def a(suffix, default=None):
        v = val(f"{arch}.{suffix}")
        return v if v is not None else default

    n_layers = a("block_count")
    n_heads = a("attention.head_count")
    n_kv_heads = a("attention.head_count_kv", n_heads)
    n_embd = a("embedding_length")
    n_ff = a("feed_forward_length")
    head_dim_k = a("attention.key_length")
    ctx = a("context_length")
    vocab = a("vocab_size")
    if vocab is None:
        toks = reader.fields.get("tokenizer.ggml.tokens")
        vocab = len(toks.data) if toks is not None and toks.data is not None else 32000

    cfg = {
        "name": os.path.basename(path),
        "model_type": str(arch),
        "num_hidden_layers": int(n_layers),
        "num_attention_heads": int(n_heads),
        "num_key_value_heads": int(n_kv_heads) if n_kv_heads else int(n_heads),
        "hidden_size": int(n_embd),
        "intermediate_size": int(n_ff) if n_ff else 4 * int(n_embd),
        "vocab_size": int(vocab),
        "max_position_embeddings": int(ctx) if ctx else 8192,
    }
    if head_dim_k:
        cfg["head_dim"] = int(head_dim_k)
    weight_bytes = os.path.getsize(path)
    precision = _precision_from_gguf_file_type(val("general.file_type"))
    return cfg, weight_bytes, precision


# GGUF general.file_type (LLAMA_FTYPE) -> the perf-relevant weight precision. The on-disk byte sum is
# the EXACT memory weight; this only sets the GenZ perf `bits` (decode is memory-bound, so the per-weight
# byte size drives TPOT). Map to the nearest of {fp32, fp16, int8, int4}; unknown -> int4 (the dominant
# served case). Q5/Q6/Q8 round UP to int8 so we never UNDER-predict decode time (the unsafe SLO direction).
_GGUF_FTYPE_PRECISION = {
    0: "fp32",   # ALL_F32
    1: "fp16",   # MOSTLY_F16
    7: "int8",   # MOSTLY_Q8_0
    8: "int8",   # MOSTLY_Q5_0
    9: "int8",   # MOSTLY_Q5_1
    16: "int8",  # MOSTLY_Q5_K_S
    17: "int8",  # MOSTLY_Q5_K_M
    18: "int8",  # MOSTLY_Q6_K
}


def _precision_from_gguf_file_type(file_type):
    """Map a GGUF general.file_type enum to a weight precision for the perf model. None/unknown -> int4."""
    try:
        return _GGUF_FTYPE_PRECISION.get(int(file_type), "int4")
    except (TypeError, ValueError):
        return "int4"


def _precision_from_name(name):
    """Map an ONNX model dir/name to a weight quant tag (int4/int8/fp16) by whole-segment match — so
    a real quant subfolder like 'cuda-int4-rtn-block-32' hits but an incidental substring does not."""
    import re
    segs = set(s for s in re.split(r"[^a-z0-9]+", name.lower()) if s)
    if {"int4", "uint4", "q4", "awq", "gptq"} & segs:
        return "int4"
    if {"int8", "q8"} & segs:
        return "int8"
    return "fp16"  # fp16/bf16/fp32 and unknown → the runtime f16 budget


def onnx_dir_to_hf_config(path):
    """Read an ONNX-GenAI model directory (genai_config.json) and map its decoder shape to an HF-style
    config dict — ALL ONNX-architecture knowledge lives here in the simulator, not in the Rust caller.
    Returns (config_dict, weight_bytes_on_disk = summed directory files, dominated by model.onnx.data).
    `intermediate_size` is usually ABSENT in genai_config; it is left unset so the GenZ loader derives a
    model-type-aware default (the exact weights come from the on-disk byte sum, not this field)."""
    gc_path = os.path.join(path, "genai_config.json")
    gc = json.load(open(gc_path))
    m = gc.get("model", {})
    d = m.get("decoder", {})
    n_layers = d.get("num_hidden_layers")
    hidden = d.get("hidden_size")
    n_heads = d.get("num_attention_heads")
    if not (n_layers and hidden and n_heads):
        raise ValueError(f"genai_config missing decoder shape (layers/hidden/heads): {gc_path}")
    n_kv = d.get("num_key_value_heads", n_heads)
    head_dim = d.get("head_size") or (hidden // n_heads)
    vocab = m.get("vocab_size") or d.get("vocab_size") or 32000
    cfg = {
        "name": os.path.basename(path.rstrip("/")),
        "model_type": str(m.get("type", "llama")),
        "num_hidden_layers": int(n_layers),
        "hidden_size": int(hidden),
        "num_attention_heads": int(n_heads),
        "num_key_value_heads": int(n_kv),
        "head_dim": int(head_dim),
        "vocab_size": int(vocab),
        "max_position_embeddings": int(m.get("context_length", 8192)),
    }
    inter = d.get("intermediate_size") or m.get("intermediate_size")
    if inter:
        cfg["intermediate_size"] = int(inter)
    weight_bytes = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    return cfg, weight_bytes


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="path to a local .gguf model")
    src.add_argument("--onnx-dir", help="path to a local ONNX-GenAI model directory")
    src.add_argument("--hf-id", help="HuggingFace model id (fetches config)")
    src.add_argument("--config", help="inline HF-style config JSON")
    ap.add_argument("--hardware", default="GB10")
    ap.add_argument("--batch", type=int, default=1, help="concurrent requests (batch size)")
    ap.add_argument("--seq", type=int, default=4096, help="context tokens per request (prompt+output)")
    ap.add_argument("--input-tokens", type=int, default=None, help="prompt tokens for SLO (default seq*3/4)")
    ap.add_argument("--output-tokens", type=int, default=None, help="output tokens for SLO (default seq/4)")
    ap.add_argument("--weight-precision", default=None,
                    help="weight quant: fp16/int8/int4 (GGUF q4≈int4). Default: derived per source.")
    args = ap.parse_args()

    out = {"hardware": args.hardware, "batch": args.batch, "seq": args.seq}

    # ---- resolve the model config + actual weight bytes -------------------------------------
    weight_bytes_override = None
    if args.gguf:
        cfg, weight_bytes_override, gguf_precision = gguf_to_hf_config(args.gguf)
        model_spec = cfg
        # Use the GGUF's ACTUAL quant (from general.file_type) for the perf bits — an f16 GGUF must not
        # be modelled as int4 (which under-predicts decode time ~2x). Weights are the exact on-disk sum.
        if args.weight_precision is None:
            args.weight_precision = gguf_precision
    elif args.onnx_dir:
        cfg, weight_bytes_override = onnx_dir_to_hf_config(args.onnx_dir)
        model_spec = cfg
        # Derive the weight quant from the ONNX dir name (the precision heuristic lives HERE in the
        # simulator, never in the Rust caller) when not explicitly given. Only affects the perf `bits`
        # (weights come from the exact on-disk byte sum, independent of this).
        if args.weight_precision is None:
            args.weight_precision = _precision_from_name(args.onnx_dir)
    elif args.config:
        cfg = json.loads(args.config)
        model_spec = cfg
    else:
        cfg = args.hf_id
        model_spec = args.hf_id

    # Per-source default weight precision when not derived above / explicitly given.
    if args.weight_precision is None:
        args.weight_precision = "int4" if args.gguf else "fp16"

    hw = get_hardware_config(args.hardware)
    if hw is False or hw is None:
        # Fail-closed (NO silent fallback): an unrecognized hardware would otherwise return a memory-only
        # result with NO latency SLO, silently hiding a typo'd $BUD_SIMULATOR_HARDWARE behind a serve-
        # without-SLO. The simulator is the single source of truth for hardware-dependent perf, so make
        # the misconfig LOUD — bud-gaia then refuses the route + discloses why (rather than serving blind).
        json.dump(
            {
                "error": f"unknown hardware '{args.hardware}': not in the simulator hardware database",
                "hint": "set BUD_SIMULATOR_HARDWARE to a supported device (e.g. GB10, H100, A100)",
            },
            sys.stdout,
        )
        sys.exit(2)

    # ---- MEMORY (arch- AND runtime-aware) via the inference-engine LAYER -----------------------
    # The serving runtime sets the activation model: ONNX models run on ONNX Runtime (one non-freeing
    # arena → all-layers prefill working set); GGUF runs on llama.cpp (layer-freeing == the base). The
    # layer (llm_memory_calculator.inference_engines) overrides ONLY the runtime-specific activation on
    # top of the calibrated base — weights (exact on-disk for a GGUF/ONNX dir) + KV are runtime-agnostic.
    engine_name = "onnxruntime" if args.onnx_dir else ("llamacpp" if args.gguf else None)
    em = estimate_memory_for_engine(
        model_spec, engine_name,
        batch_size=args.batch, seq_length=args.seq,
        weight_precision=args.weight_precision, runtime_precision=RUNTIME_KV_PRECISION,
        weight_bytes_override=weight_bytes_override,
    )
    total_runtime = em.total_bytes + CUDA_RUNTIME_RESERVE
    out["memory"] = {
        "attention_type": em.attention_type,
        "engine": em.engine,
        "parameter_count": int(em.parameter_count),
        "weights_bytes": float(em.weights_bytes),
        "kv_cache_bytes": float(em.kv_cache_bytes),
        "activations_bytes": float(em.activation_bytes),
        "cuda_reserve_bytes": float(CUDA_RUNTIME_RESERVE),
        "total_runtime_bytes": float(total_runtime),
        "total_runtime_gb": total_runtime / 1e9,
        "kv_bytes_per_token": em.kv_cache_bytes / max(1, args.batch * args.seq),
    }

    # ---- SLOs (perf): register the config under a name so GenZ's perf path accepts it ---------
    in_tok = args.input_tokens if args.input_tokens is not None else max(1, args.seq * 3 // 4)
    out_tok = args.output_tokens if args.output_tokens is not None else max(1, args.seq // 4)
    try:
        if isinstance(model_spec, dict):
            mc = huggingface_config_to_model_config(model_spec, model_spec.get("name", "bud_custom"))
            MODEL_DICT.add_model(mc)
            perf_name = mc.model
        else:
            perf_name = model_spec
        perf_bits = "int8" if args.weight_precision in ("int4", "int8") else "bf16"
        perf = estimate_end_to_end_performance(
            model=perf_name, batch_size=args.batch, input_tokens=in_tok,
            output_tokens=out_tok, system_name=hw, bits=perf_bits)
        # TTFT/TPOT are already hardware-calibrated INSIDE GenZ (the hardware config's
        # `inference_calibration` block) — no post-processing here. Other hardware passes through
        # GenZ's raw roofline unchanged.
        tpot = perf.get("average_tpot") or 0.0
        ttft = perf.get("ttft") or 0.0
        total_latency = ttft + out_tok * tpot
        throughput = (1000.0 / tpot * args.batch) if tpot else None
        out["slo"] = {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "decode_tok_s": (1000.0 / tpot) if tpot else None,
            "total_latency_ms": total_latency,
            "throughput_tok_s": throughput,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "calibration_source": "genz_internal",
        }
    except Exception as e:  # SLOs are best-effort; memory still returned
        out["slo_error"] = f"{type(e).__name__}: {e}"

    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
