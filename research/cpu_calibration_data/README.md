# CPU calibration — raw benchmark data

Ground-truth vLLM measurements backing the CPU decode calibration (see
`../cpu_decode_calibration_measured.md` for provenance and how the constants were fit).

## Hardware / engine
2× Intel Xeon 6767P (Granite Rapids-AP, 128 cores / 2 sockets / 2 NUMA, DDR5-6400),
vLLM 0.24.0+cpu, `--dtype bfloat16` (AMX). No GPU.

## Contents
- **`cpu_best_config_per_workload.csv`** — the single best-throughput vLLM config per workload
  (input/output × concurrency) with the exact serve flags. TP=2 wins nearly everywhere; the lone
  exception is 8192/2048 @ conc50 (fp8 KV cache edges throughput, but note its 53 s TTFT).
- **`cpu_calibration_summary.csv`** — readable TP=1-vs-TP=2 pivot (out tok/s, speedup, TPOT, TTFT).
- `qwen3-8b_measured.csv`, `qwen3-4b_measured.csv` — consolidated per-cell metrics
  (output tok/s, total tok/s, mean TTFT, mean/p99 TPOT, duration).
- `qwen3-8b/*.json`, `qwen3-4b/*.json` — raw per-cell output of `vllm bench serve`
  (unmodified). Filenames: `<config>__<in>_<out>__c<concurrency>.json`.
- `scripts/` — reproduction: `bench.py` (server+load driver), `aggregate.py`,
  `compare_sim.py` / `compare_sim_4b.py` (simulator-vs-real).

## Config legend (which real config each cell maps to)
| config | meaning | maps to simulator |
|---|---|---|
| `c0_baseline` | vLLM default = **1 socket** (TP=1) | `Xeon6767P_1socket_CPU`, `--tensor-parallel 1` |
| `c1_tp2_numa` | `-tp 2` = **2 sockets** (TP=2) | `Xeon6767P_1socket_CPU`, `--tensor-parallel 2` |
| `c6_tp2_chunked`, `c7_tp4`, `c8/c9_*bt*`, `c10_*int8kv*` | vLLM feature sweep (not calibration inputs) | — |
| `prefixdemo_*` | prefix-cache A/B (shared-prefix dataset) | — |

The calibration was fit/validated against `c0_baseline` (TP=1) and `c1_tp2_numa` (TP=2) only;
the other 8B configs are the vLLM best-config study and are included for completeness.

## Methodology
`vllm bench serve --dataset-name random --ignore-eos --max-concurrency C --random-input-len IN
--random-output-len OUT` (fixed output length ⇒ decode work is comparable). One run per cell.
