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

from llm_memory_calculator import calculate_memory, get_hardware_config
from llm_memory_calculator import estimate_end_to_end_performance
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
    return cfg, weight_bytes


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="path to a local .gguf model")
    src.add_argument("--hf-id", help="HuggingFace model id (fetches config)")
    src.add_argument("--config", help="inline HF-style config JSON")
    ap.add_argument("--hardware", default="GB10")
    ap.add_argument("--batch", type=int, default=1, help="concurrent requests (batch size)")
    ap.add_argument("--seq", type=int, default=4096, help="context tokens per request (prompt+output)")
    ap.add_argument("--input-tokens", type=int, default=None, help="prompt tokens for SLO (default seq*3/4)")
    ap.add_argument("--output-tokens", type=int, default=None, help="output tokens for SLO (default seq/4)")
    ap.add_argument("--weight-precision", default="int4", help="weight quant: fp16/int8/int4 (GGUF q4≈int4)")
    args = ap.parse_args()

    out = {"hardware": args.hardware, "batch": args.batch, "seq": args.seq}

    # ---- resolve the model config + actual weight bytes -------------------------------------
    weight_bytes_override = None
    if args.gguf:
        cfg, weight_bytes_override = gguf_to_hf_config(args.gguf)
        model_spec = cfg
    elif args.config:
        cfg = json.loads(args.config)
        model_spec = cfg
    else:
        cfg = args.hf_id
        model_spec = args.hf_id

    hw = get_hardware_config(args.hardware)

    # ---- MEMORY (arch-aware): weights at quant, KV+activations at the runtime f16 ------------
    # Weights: prefer the real on-disk size (exact for a GGUF); else the analytic quant estimate.
    rep_w = calculate_memory(model_spec, batch_size=args.batch, seq_length=args.seq,
                             precision=args.weight_precision)
    rep_rt = calculate_memory(model_spec, batch_size=args.batch, seq_length=args.seq,
                              precision=RUNTIME_KV_PRECISION)
    weights = float(weight_bytes_override) if weight_bytes_override else rep_w.weight_memory_bytes
    kv = rep_rt.kv_cache_bytes
    activations = rep_rt.activation_memory_bytes
    total_runtime = weights + kv + activations + CUDA_RUNTIME_RESERVE
    out["memory"] = {
        "attention_type": rep_rt.attention_type,
        "parameter_count": int(rep_rt.parameter_count),
        "weights_bytes": float(weights),
        "kv_cache_bytes": float(kv),
        "activations_bytes": float(activations),
        "cuda_reserve_bytes": float(CUDA_RUNTIME_RESERVE),
        "total_runtime_bytes": float(total_runtime),
        "total_runtime_gb": total_runtime / 1e9,
        "kv_bytes_per_token": kv / max(1, args.batch * args.seq),
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
