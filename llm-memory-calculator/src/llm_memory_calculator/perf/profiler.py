#!/usr/bin/env python3
"""FRD-027 Revision 2, L2: the step profiler (FR-PROF-1/3).

Runs inside the target engine image (it imports only vLLM and the standard library) and times every engine step of a
grid of (batch B, prompt length L) loads: the prefill chunks, the mixed steps where finished prompts decode while
others still prefill, and the decode-only steps. Each step is recorded with its composition, so the simulator's step
model (``perf.profile``) and scheduler (``perf.sched``) are built from measurements, not fitted terms (FRD-027 §20).

    python3 profiler.py --model <dir> --name <label> --out <dir> [--load-format auto] [--trust-remote-code]
        [--max-model-len 33280] [--max-num-seqs 256] [--gpu-util 0.9] [--extra '{"kv_cache_dtype": "fp8"}']

MoE models need ``--load-format auto`` (real weights): random router weights spread tokens almost uniformly over the
experts (R0: the FP8 MoE hold-out ran 25-30% slow at c16-c64, gpt-oss 23-40% fast). Dense and hybrid models are
profiled with random weights from their ``config.json`` alone.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the engine core runs in this process

SCHEMA = "bud-perf-steps/1"
BATCHES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256)
LENGTHS = (256, 1024, 4096, 8192, 16384, 32768)
DECODE_STEPS = 40  # decode-only steps timed per point
CHUNK = 8192  # max-num-batched-tokens, as `vllm serve` sets it on H100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=33280)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--extra", default="{}")
    # serving detokenizes and streams every token; the default leaves it out, so the steps are the engine's alone
    ap.add_argument("--detokenize", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.steps.json")

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        load_format=args.load_format,
        enable_prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.chunk,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=False,
        seed=0,
        **json.loads(args.extra),
    )
    init_s = time.perf_counter() - t0
    engine = llm.llm_engine
    cache = engine.vllm_config.cache_config
    kv_blocks = int(getattr(cache, "num_gpu_blocks", 0) or 0)
    block = int(getattr(cache, "block_size", 0) or 0)
    vocab = engine.model_config.get_vocab_size()
    rng = random.Random(0)
    counter = [0]

    def preemptions() -> float:
        try:
            for m in llm.get_metrics():
                if "preempt" in m.name:
                    return float(getattr(m, "value", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            pass
        return -1.0

    def run(b: int, length: int, decode_steps: int):
        """Add b prompts of ``length`` tokens at once and time every step until they finish."""
        max_tokens = decode_steps + -(-b * length // args.chunk) + 8
        sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0, detokenize=args.detokenize)
        ids = []
        for _ in range(b):
            counter[0] += 1
            rid = f"p{counter[0]}"
            ids.append(rid)
            prompt = [rng.randrange(1000, max(vocab - 1000, 2000)) for _ in range(length)]
            engine.add_request(rid, TokensPrompt(prompt_token_ids=prompt), sp)
        produced = {rid: 0 for rid in ids}
        steps = []
        # the interval between successive step returns: async scheduling overlaps the next step's scheduling with this
        # one's GPU work, as in serving, so no synchronize between steps (it would add the CPU time to every step)
        torch.cuda.synchronize()
        last = time.perf_counter()
        while engine.has_unfinished_requests():
            before_dec = sum(1 for v in produced.values() if 0 < v < max_tokens)
            before_wait = sum(1 for v in produced.values() if v == 0)
            outs = engine.step()
            now = time.perf_counter()
            dt, last = now - last, now
            new = 0
            for o in outs:
                n = len(o.outputs[0].token_ids) if o.outputs else 0
                new += max(n - produced.get(o.request_id, 0), 0)
                produced[o.request_id] = n
            steps.append({"ms": round(1000 * dt, 4), "decoding": before_dec, "waiting": before_wait, "new_tokens": new})
        return steps

    run(1, 256, 8)  # warm-up
    points = []
    batches = [int(x) for x in args.batches.split(",") if x]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    for length in lengths:
        for b in batches:
            need = b * (length + DECODE_STEPS + 16)
            if length + DECODE_STEPS + 16 > args.max_model_len or b > args.max_num_seqs:
                continue
            if kv_blocks and block and need > 0.85 * kv_blocks * block:
                points.append({"b": b, "len": length, "skipped": "kv"})
                continue
            # warm: the first step at a shape JIT-compiles Triton kernels inside the timing; its slowest step is the
            # shape's cold first-use stall, recorded for the plan's cold-start note
            warm = run(b, length, 4)
            p0 = preemptions()
            steps = run(b, length, DECODE_STEPS)
            p1 = preemptions()
            decode = [s["ms"] for s in steps if s["waiting"] == 0 and s["decoding"] == b and s["new_tokens"] == b]
            decode = decode[2:-2] if len(decode) > 8 else decode  # async scheduling shifts outputs by a step
            points.append({
                "b": b, "len": length, "steps": steps,
                "decode_ms": round(statistics.median(decode), 4) if decode else None,
                "decode_ctx": length + DECODE_STEPS // 2,
                "preemptions": (p1 - p0) if p0 >= 0 and p1 >= 0 else None,
                "warm_max_ms": round(max((s["ms"] for s in warm), default=0.0), 4),
            })
            print(json.dumps({"model": args.name, "b": b, "len": length, "decode_ms": points[-1]["decode_ms"],
                              "steps": len(steps), "preemptions": points[-1]["preemptions"]}), flush=True)
            _save(path, args, vllm.__version__, init_s, kv_blocks, block, points)
    _save(path, args, vllm.__version__, init_s, kv_blocks, block, points, total_s=time.perf_counter() - t0)
    return 0


def _save(path, args, version, init_s, kv_blocks, block, points, total_s=None):
    with open(path, "w") as f:
        json.dump({
            "schema": SCHEMA, "name": args.name, "model": args.model, "vllm": version, "load_format": args.load_format,
            "chunk": args.chunk, "max_num_seqs": args.max_num_seqs, "max_model_len": args.max_model_len,
            "gpu_util": args.gpu_util, "extra": json.loads(args.extra), "detokenize": args.detokenize,
            "init_s": round(init_s, 1),
            "kv_blocks": kv_blocks, "block_tokens": block, "decode_steps": DECODE_STEPS, "points": points,
            "total_s": None if total_s is None else round(total_s, 1),
        }, f)


if __name__ == "__main__":
    sys.exit(main())
