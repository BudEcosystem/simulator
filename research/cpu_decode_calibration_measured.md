# CPU decode calibration — measured provenance (Intel Xeon 6767P)

This documents the real vLLM measurements behind the CPU decode calibration added in
`genz/LLM_inference/llm_decode.py` (`_ETA_MEM_DECODE_CPU_X86`, `_CPU_TP_ALLREDUCE_LAT_MS`) and the
`Xeon6767P_CPU` / `Xeon6767P_1socket_CPU` hardware profiles. All constants are fit against these runs.

## Hardware under test
- 2× Intel Xeon 6767P (Granite Rapids-AP), 64 cores/socket = 128 cores, **2 sockets / 2 NUMA nodes**, 503 GB RAM
- AMX (bf16/int8) + AVX-512; 8-channel **DDR5-6400** per socket → **409.6 GB/s/socket, 819 GB/s aggregate** (datasheet peak)
- No GPU (CPU backend only)

## Measurement setup (ground truth)
- Engine: **vLLM 0.24.0+cpu**, torch-inductor (AVX-512+AMX), `--dtype bfloat16`
- TP=1 = single socket (vLLM CPU default binds to NUMA node 0); TP=2 = `-tp 2 --distributed-executor-backend mp` (one rank/socket, gloo all-reduce over UPI)
- Load gen: `vllm bench serve --dataset-name random --ignore-eos --max-concurrency C` (output length fixed ⇒ decode work comparable)
- Metric: mean TPOT (ms/token) and aggregate output tok/s

## Model A — Qwen/Qwen3-8B (calibration source)
8B params, 36 layers, GQA (32 q / 8 kv heads), bf16. Single-stream (batch=1) decode:

| config | TPOT | note |
|---|---|---|
| TP=1 (1 socket) | **102 ms** | anchor for η vs 409.6 GB/s |
| TP=2 (2 sockets) | **60 ms** | anchor for η vs 819 GB/s + all-reduce latency |

- Socket scaling TP=2/TP=1 = **1.70×** (not 2× → inter-socket all-reduce cost).
- GenZ decode resident bytes ≈ 11.0 GiB (matmul weights it streams per token).
- ⇒ `η_decode = 11.0 GiB / (60 ms × 819 GB/s) ≈ 0.219` (2S); `= 11.0/(102 ms×409.6) ≈ 0.257` (1S);
  **joint geomean = 0.237** (the shipped constant).
- ⇒ NUMA: ideal 2× floor = 55.4 ms, real 60 ms ⇒ `+4.6 ms` over 2×36 all-reduces ⇒
  **`L ≈ 0.064 ms/all-reduce`** (the shipped constant).

Full 8B grid (output tok/s, concurrency {1,20,50,100} × input/output):

| in/out | TP=1 (1→100) | TP=2 (1→100) |
|---|---|---|
| 200/300   | 9.6 / 158 / 318 / 472 | 16.5 / 259 / 537 / 749 |
| 512/512   | 9.7 / 150 / 285 / 402 | 16.3 / 247 / 467 / 642 |
| 1024/1024 | 9.7 / 139 / 246 / 329 | 16.6 / 231 / 423 / 551 |
| 4096/1024 | 9.4 /  94 / 139 / 166 | 16.0 / 163 / 243 / 301 |
| 8192/2048 | 9.0 /  70 /  93 /  89 | 15.5 / 127 / 167 / 174 |

(TP=2 is 1.6–1.96× over TP=1 across the whole grid — the CPU default uses only 1 of 2 sockets.)

## Model B — Qwen/Qwen3-4B (out-of-sample validation, constants UNCHANGED)
4B params, 36 layers, bf16. 12 cells (TP=1+TP=2, in/out ∈ {512/512,1024/1024,8192/2048}, conc ∈ {1,50}).
Single-stream socket scaling = **1.45×** (smaller model amortizes the fixed sync cost over less work).

## Calibration accuracy (simulator vs measured, output tok/s)
| | TP=1 geomean / median|err| | TP=2 geomean / median|err| |
|---|---|---|
| baseline (generic profile, no floor) | — | **2.25×** systematic / 123% |
| **8B** (in-sample) | 0.87× / 13% | 0.97× / 11% |
| **4B** (out-of-sample, no re-fit) | 0.86× / 7% | 1.05× / 16% |

The same two constants hold across **2 model sizes × 2 socket counts × several seq/concurrency points**
within ~7–16% median error ⇒ they are hardware/category-level, not per-model.

## Known residuals (future levers)
- Short-seq high-concurrency: sim optimistic (memory-controller contention, unmodeled).
- Long-seq (8k): sim ~0.7× (per-op KV-streaming efficiency > weight-streaming).
- Smaller models: TP=2 slightly optimistic (fixed NUMA latency; a `1/model_work` scaling would refine).
- TTFT: untouched here — needs a prefill compute floor + chunked-prefill serving cap.

Reproduce: `bud_component.py --hf-id <model> --hardware Xeon6767P_1socket_CPU --tensor-parallel {1,2}
--batch C --input-tokens IN --output-tokens OUT`.
