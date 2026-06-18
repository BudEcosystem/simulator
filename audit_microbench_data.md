# GB10 microbenchmark grounding data (measured on this box, NVIDIA GB10, bf16, torch CUDA)

Empirical anchor for the accuracy-fix solutions. **All numbers measured, not assumed.**
GB10 config in repo: `Flops=250` (TOPS), `Memory_BW=273` (GB/s, LPDDR5X), `Memory_size=128` (GB).
GB10 is also the ONLY device with an existing `inference_calibration` block — so these measurements
both ground the methodology and validate that block.

## 1. Memory bandwidth (grounds `eta_mem` / MBU)
| workload | GB/s | % of 273 peak |
|---|---|---|
| copy 64 MB | 207.8 | 76.1% |
| copy 256 MB | 205.7 | 75.4% |
| copy 1024 MB | 210.8 | 77.2% |
| M=1 8192² matmul (weight-stream) | 200.1 | 73.3% |

**Achievable MBU ceiling ≈ 0.73–0.77.** Existing GB10 `decode.eta_mem=0.659` is slightly conservative
vs this (real llama.cpp decode carries extra KV/scheduler traffic → effective < pure-copy). Consistent.

## 2. GEMM bf16 achievable TFLOPS (grounds `eta_compute` / MFU) — roofline sweep at N=K=8192
| M | TFLOPS | Arithmetic Intensity (flop/byte) |
|---|---|---|
| 1 | 0.24 | 1.0 |
| 2 | 0.48 | 2.0 |
| 4 | 0.93 | 4.0 |
| 8 | 1.84 | 8.0 |
| 16 | 3.71 | 15.9 |
| 32 | 6.71 | 31.8 |
| 64 | 14.17 | 63.0 |
| 128 | 26.26 | 124.1 |
| 256 | 43.90 | 240.9 |
| 512 | 72.08 | 455.1 |
| 1024 | 77.15 | 819.2 |
| 2048 | 80.57 | 1365.3 |
| 4096 | 83.80 | 2048.0 |
| 8192³ (square) | 85.9 | 2731 |

**Achievable compute ceiling ≈ 85 TFLOPS** = **34% of the configured 250**. (If GB10's *true* bf16
dense peak is ~125 TFLOPS — i.e. 250 is an FP8/sparse-flavored figure — then MFU ≈ 69%. The peak label
is ambiguous; what is unambiguous is the **measured achievable 85 TFLOPS**.)

## 3. THE DECISIVE FINDING: the measured curve IS a clean roofline
`throughput(AI) = min(AI × eff_BW, eff_peak_compute)` with **eff_BW ≈ 205 GB/s, eff_peak ≈ 85 TFLOPS**,
ridge at AI = 85000/205 ≈ **419 flop/byte**. Verification of the model against measurements:
- M=1, AI=1.0 → predict 1.0×205 = 0.205 TF, measured 0.24 ✓
- M=128, AI=124 → predict 124×0.205 = 25.4 TF, measured 26.3 ✓
- M=512, AI=455 (just past ridge) → ~85 capped, measured 72 (transition) ✓
- M=4096, AI=2048 → ~85 capped, measured 83.8 ✓

**Implication for the fix:** GenZ already computes per-op `max(compute_time, memory_time)` — the SAME
roofline. The shape/M dependence (small-M decode = memory-bound, large-M prefill = compute-bound)
**emerges automatically from per-op arithmetic intensity**. Therefore:
- The ad-hoc `get_tensor_core_efficiency_factor` (flat 0.85, dead shape code) and
  `get_shape_dependent_op_intensity` (C4) are **redundant and should be removed**, not patched —
  the physical roofline already encodes shape dependence.
- The ONLY per-device knobs needed are **`eta_mem` (≈ measured MBU)** and **`eta_compute`
  (= measured_achievable_TFLOPS / configured_peak)**, both **fit from measurement**, plus a small
  **`t_launch`** and per-stream term for the launch-bound small-op regime. This is exactly the
  `inference_calibration` block shape — so the fix is: calibrate ALL devices the way GB10 was, with
  the existing `differential_evolution` fitter, against measured or literature MBU/MFU. No magic
  constants — every value is a measured ratio.

## 4. Kernel launch floor (grounds `t_launch`)
Tiny 64² matmul: **8.26 µs/op** raw launch+dispatch floor. Existing GB10 `decode.t_launch=4.56µs`
(same order ✓). `prefill.t_launch=206µs` absorbs the small-model launch-heavy regime (more than pure
launch). A per-op launch term on the order of 5–10 µs is empirically justified.

## 5. How to use this for the other devices (generalizable, not hardcoded)
- `eta_mem`: per memory technology, anchored to published streaming-BW efficiency (HBM2e/3 ~0.8–0.9,
  LPDDR5X ~0.75 as measured here, GDDR ~0.7–0.8). Source per device from a vendor/STREAM-style number.
- `eta_compute`: `achievable_dense_bf16 / configured_peak`. Published achievable GEMM MFU (cuBLAS/CUTLASS
  large-GEMM) is ~0.7–0.85 of TRUE dense peak on HBM datacenter GPUs — so once C1 sets the **dense**
  peak (not sparse), `eta_compute` ≈ 0.75–0.85 for big GEMMs, and the roofline handles small ops.
- The decode/prefill split, `t_launch`, and per-stream term are FIT per device by the calibration
  harness against a small measured grid (the GB10 procedure), never hand-typed.

---
## 6. KEYSTONE end-to-end validations (real models on GB10 — the regression anchors)

Built a 1.498B Llama (random weights, exact config, no download): hidden 2048, 16 layers, 32 heads,
8 KV heads (GQA), inter 8192, vocab 128256. Measured real decode & prefill, compared to the roofline.

### Decode (memory-bound) — confirms C2, validates the eta_mem fix
| ctx | measured TPOT | measured MBU | sim eff=1.0 (default) | eta_mem=0.659 (GB10 calib) | eta_mem=0.73 (measured) |
|---|---|---|---|---|---|
| 512  | 15.72 ms | 70.2% | **−30%** (11.04ms) | +7% | −4% |
| 2048 | 15.97 ms | 70.3% | **−30%** (11.22ms) | +7% | −4% |
**The default eff=1.0 overpredicts decode throughput by 30% on real hardware. Calibrated eta_mem brings
it to ±7%.** Decode MBU (70%) < copy MBU (76%) because per-layer launch/attention/sampling overhead —
hence the calibration's t_launch + per_stream terms close the residual.

### Prefill (compute-bound) — confirms C3, reveals the C×B coupling
| ctx | measured TTFT | achieved TFLOPS | sim now (250×0.85) | eta_compute=0.34 |
|---|---|---|---|---|
| 512  | 26.1 ms  | 60.1 | **−72%** (7.4ms) | −29% |
| 2048 | 100.6 ms | 66.5 | **−69%** (31.5ms)| −22% |
| 8192 | 490.7 ms | 68.0 | **−68%** (156.9ms)| −20% |
**The current simulator predicts prefill ~3× too fast on GB10** (achievable MFU is ~27%, not the
modeled 85%). With eta_compute≈0.27 (fit to prefill, not the 0.34 pure-GEMM ceiling) it matches.

### The coupling proof (decisive for rollout order)
At ctx=8192: full-S² attention + eta=0.34 → −20%; switching to causal-S²/2 → −31%. I.e. the calibrated
eta_compute ABSORBS the attention FLOP-counting convention. Conclusion (plan of record):
**structural FLOP fixes (M4b causal mask, C1 dense FLOPS, C4/L3 dead-code removal) MUST land first;
THEN eta_{compute,mem}/t_launch are fit by the differential_evolution harness against the corrected
structure.** Fixing efficiency before structure, or fixing one structural piece in isolation, double-
counts. Every eta is a measured/fit ratio against this kind of grid — never a hand-typed magic number.

## 7. Power (grounds M14) — measured GB10 power.draw
Idle ≈ 7.2 W; sustained GEMM @96% util ≈ 67–73 W; util 35% ≈ 44 W. Power scales ~linearly with
utilization: `P ≈ P_idle + util·(P_max − P_idle)`. This refutes the current fixed `0.4125·TDP`
(workload-insensitive). The util term must come from the roofline (compute_util / memory_util), and
P_idle/P_max from the device's vendor idle/TDP — both sourced, not invented.

## 8. CPU microbenchmark (Grace Cortex-X925, 20-core ARM, unified LPDDR5X) — grounds Round-2 CPU fixes
- bf16 GEMM (4096³, torch): **2.73 TFLOPS** ; fp32 GEMM: **0.76 TFLOPS** (20 cores).
- Streaming copy BW (numpy, partially threaded): **~54 GB/s** (single-stream floor; multi-threaded STREAM higher;
  peak LPDDR5X 273 GB/s is shared CPU+GPU, CPU can't saturate it).
- Implication: a 7B bf16 (13.5GB weights) on this CPU class decodes at weights/effective_BW ≈ 13.5/(~50-80 GB/s)
  ≈ **~2.5-6 tok/s**, consistent with published llama.cpp 7B on server CPUs (~3-10 tok/s). Confirms R2-CPU2
  (CPUSystem 0.84 tok/s = ~10× too slow) and R2-CPU3 (static path 18.6 tok/s = ~3× too fast). CPU decode must be
  modeled as DRAM-bandwidth-bound at a realistic CPU MBU (~0.45-0.65 of peak, lower than GPU HBM), NOT a cache-sim
  cycle count. x86 Xeon/EPYC DDR5 peaks (SPR 8ch DDR5-4800 ≈ 307 GB/s; Genoa 12ch ≈ 460) from datasheets.
