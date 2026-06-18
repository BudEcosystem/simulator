# BudSimulator — Brutal Accuracy & Correctness Audit

**Audit start:** 2026-06-16
**Branch:** `fix/serving-hardware-validation`
**Scope:** Every line of BudSimulator + llm-memory-calculator (GenZ engine), correlated with
real-world hardware/model/inference-engine numbers from vendor datasheets and published benchmarks.
**Method:** Static read of every core file → empirical reproduction with the installed package
(`.venv`) → cross-check predicted values against physical limits (peak FLOPS / HBM bandwidth) and
published real-world MFU/MBU/throughput. Every finding below is **reproduced**, not theoretical.

## Severity legend
- 🔴 **CRITICAL** — predictions are wrong by ≥1.5× or physically impossible (>100% of a hardware limit), or a feature is broken.
- 🟠 **HIGH** — systematic accuracy error 1.2–1.5×, or a correctness bug with a real blast radius.
- 🟡 **MEDIUM** — accuracy error <1.2×, edge-case crash, or a modeling simplification that misleads.
- 🔵 **LOW** — cosmetic, dead code, minor inconsistency, or missing-metric/UX gap.

Each finding: **Evidence** (reproduction) → **Root cause** (file:line) → **Impact** → **Fix**.

---

## EXECUTIVE SUMMARY

**39 distinct issues (7 critical, 10 high, 15 medium, 7 low), all reproduced.** Two systemic themes dominate:

1. **The inference engine is systematically optimistic** — it reports performance at or above physical
   hardware limits. Decode runs at **>100% of peak memory bandwidth** (impossible); prefill at **~85%
   MFU** (real ~50%); H100/H200 are configured with **sparse** FLOPS (2× the achievable dense rate).
   Net: decode/serving throughput is ~1.3–1.7× too fast, prefill/TTFT ~1.5–2× too fast on Hopper. The
   authors' own `inference_calibration.py` calls the roofline "optimistic" and ships a fitter to fix
   it — but it's applied to **1 of 92 devices**. The *training* path already uses a realistic ~46% MFU;
   inference does not.

2. **Hardware has no single source of truth.** The engine (`get_hardware_config`→static configs), the
   API catalog (DB, 92 rows), and the training selector (static + its own cost table) disagree. **18
   of 92 catalog devices can't be simulated at all** (404). The UI shows H100=989 TFLOPS while the
   engine computes with 1979. Duplicate rows (MI300X 1307 vs 1600) silently change answers 2×.

Plus: a popular-model parameter-counter bug for **Gemma (−23% memory → OOM-risk recommendations)**, a
**power endpoint that fabricates numbers for nonexistent models (HTTP 200)**, **dead "Phase 2/8"
accuracy code** (tensor-core efficiency + comm-efficiency tables never run), a **usecase-create 500**,
and **UI elements presenting fabricated data** (relative price index shown as "$/hr", hardcoded model-
compatibility table, constant "70% utilization").

### Top fixes by ROI (do these first)
1. **C1/C5/C7** — one hardware source of truth; set H100/H200/GH200/H100_PCIe to **dense** FLOPS
   (989/756); make every listed device simulatable. *(Fixes 2× error on the most common GPUs + 18 dead
   options + UI/engine split, all at once.)*
2. **C2/C3** — calibrate `eta_mem`/`eta_compute` per device (the fitter exists); stop shipping 1.0.
   *(Removes the >100% MBU/MFU impossibility across all decode/prefill/serving.)*
3. **C4** — fix `get_tensor_core_efficiency_factor` so the shape-aware model actually runs (or delete
   it and fold into eta_compute); add a test asserting it's not a constant.
4. **H0/M4** — Gemma/GeGLU param count (read `hidden_activation`, drive GLU from `model_type`).
5. **H2/H4** — stop fabricating power numbers; fix the null-SLO 500.

### Coverage & test volume
Audited: GenZ engine (system/roofline/operators/prefill/decode/analyse), memory+parameter counter,
collective/parallelism modeling, all 6 API router groups + services, training subsystem, CPU path,
hardware configs+DB (92 devices), React frontend (source). **>200 individual test executions**: 58
live-API tests (`api_audit_tests.py`, re-runnable), ~50 engine probes, 92 hardware-resolution checks,
~15 training scenarios, parameter counts for 9 model families. **Not live-tested:** the React UI in a
real browser — the connected Chrome is on macOS and can't reach this Linux box's localhost (UI findings
are from source audit cross-checked against live API responses).

---

## 🔴 CRITICAL findings

### C1. H100 / H200 use the *sparse* TFLOPS (1979) as if it were dense → compute-bound predictions ~2× too fast
**Evidence (reproduced):**
```
H100_GPU   cfg Flops=1979  (real dense bf16 = 989.5)   <-- 2.0× too high
H200_GPU   cfg Flops=1979  (real dense bf16 = 989.5)   <-- 2.0× too high
A100_80GB  cfg Flops=312   (real 312) ✓
B200_GPU   cfg Flops=2250  (real ~2250 dense) ✓
A6000_GPU  cfg Flops=155   (real 155) ✓
```
A prefill of llama2_7b @2048 tok on `H100_GPU` yields an **implied MFU of 162%** — i.e. the model
"achieves" 1.6× the GPU's *peak dense* FLOPS, which is physically impossible. On A100 the same run
implies 82% MFU (also too high — see C3 — but at least <100%). The 162/82 ≈ 2.0 ratio is exactly the
sparse-vs-dense factor.

**Root cause:** `llm-memory-calculator/src/llm_memory_calculator/hardware/configs.py` — `H100_GPU`
and `H200_GPU` set `'Flops': 1979`. NVIDIA's 1979 TFLOPS figure is FP16/BF16 **with 2:4 structured
sparsity**; dense tensor-core throughput is **989.5 TFLOPS** (H100 SXM). LLM inference GEMMs are dense.

**Impact:** Every compute-bound metric on the two most common datacenter GPUs is ~2× optimistic:
prefill latency / TTFT, prefill throughput, large-batch decode, and any hardware *recommendation* or
*cost-per-token* that ranks H100/H200 against correctly-specced parts (A100/B200). Cross-device
comparisons are skewed.

**Fix:** Set `H100_GPU` and `H200_GPU` `Flops` to `989` (dense). If a sparse mode is ever desired,
model it explicitly via a precision/sparsity flag, never as the default dense peak. Audit *every*
device for dense-vs-sparse consistency (B200 used dense 2250 — inconsistent treatment across the file).

**UPDATE (verified): the engine and the UI use DIFFERENT numbers for the same GPU.** The performance
engine resolves specs via `get_hardware_config` → the **static** `HARDWARE_CONFIGS`, while the
`/api/hardware` catalog serves the **DB** (`prepopulated.db`). They disagree:
| device | engine (`get_hardware_config`) | DB / UI catalog | real dense bf16 |
|---|---|---|---|
| H100_GPU | **1979** (sparse) | 989 (correct) | 989 |
| H100_PCIe_GPU | **1513** (sparse) | — | 756 |
| H200_GPU | **1979** (sparse) | 989 | 989 |
| GH200_GPU | **1979** (sparse) | 1979 | 989 |
So a user reads "H100 = 989 TFLOPS" in the UI but every H100 latency/throughput is **computed with
1979** (2× too fast). Fixing the static configs to dense (989/756) corrects both the engine *and*
realigns it with the DB. Hopper-family (H100/H200/GH200/H100_PCIe/H800) is the affected set; Blackwell
(B100=1750, B200=2250, GB200=2250/GPU) and MI300X (1307) are dense-correct.

### C5. 18 of 92 catalog devices are listed by the API/UI but **cannot be simulated** (404 "Unknown hardware")
**Evidence (reproduced):** the DB serves 92 devices; `get_hardware_config` (the engine entry point)
resolves only **74**. The other **18** — including the generic **`A100_GPU`**, **`MI300X_GPU`**,
**`H100_SXM_GPU`**, **`MI250X_GPU`**, **`Gaudi2`**, `L40_GPU`, `H800_GPU`, `V100_GPU`,
`Tesla_P100_GPU`, `TPU_v2/v3`, `PonteVecchio_GPU`, `Trainium_Trn1`, `Inferentia2_Inf2`,
`Graphcore_IPU`, `MI100_GPU`, `Tesla_K80_GPU`, `TPUv6_Trillium` — are absent from the static configs.
`POST /api/v2/power/estimate {hardware:"H100_SXM_GPU"}` → **404 "Unknown hardware: H100_SXM_GPU"**;
`simulate/serving {hardware:"V100_GPU"}` → **404**.
**Root cause:** the recent fix made `BudHardware` *serve* all 92 DB rows, but the GenZ engine still only
knows the ~73 static `HARDWARE_CONFIGS`. The catalog and the simulator draw from different sets.
**Impact:** the hardware picker offers ~20% of its options as dead ends; users selecting a common name
like "A100_GPU" or "MI300X_GPU" get a hard 404. **Fix:** make `get_hardware_config`/`get_inference_system`
fall back to the DB row (build a `System` from the DB's Flops/Memory_BW/Memory_size/ICN), so anything
listed is simulatable; or stop listing what can't be simulated.

### C6. Intel Data Center GPU Max (PonteVecchio / MAX1550 / MAX1100) use the FP32 peak as bf16 → ~18× too slow
**Evidence (verified):** `get_hardware_config('MAX1550').Flops = 45.2`; real Max 1550 bf16 (XMX) ≈ **839
TFLOPS dense** — 45/52 TFLOPS is the **FP32/FP64** peak. `PonteVecchio_GPU=52`, `MAX1100=32.7` likewise.
**Impact:** any prefill/compute-bound estimate on Intel Max GPUs is ~16–18× too slow → they'll never be
recommended and any latency shown is nonsense. **Fix:** set bf16/XMX dense TFLOPS (Max 1550 ≈ 839,
Max 1100 ≈ 362).

### C7. Duplicate/conflicting device rows disagree on core specs
**Evidence (DB dump):** `MI300X=1307` vs `MI300X_GPU=1600` (1600 is not a real MI300X bf16 figure);
`V100*=125` vs `V100_GPU=157`; `TPU_v2=180`/`TPU_v3=420` vs per-chip reality (v2≈46, v3≈123 — these
look like 4-chip board totals while `TPU_v4=275` is per-chip → mixed conventions); `RTX4090=661`
(sparse) vs A100=312 (dense). **Impact:** which row a user picks silently changes the answer by up to
2×; cross-device ranking is unreliable. **Fix:** dedupe to one canonical row per accelerator with a
consistent (dense bf16, per-chip) convention; add a unit test asserting engine specs == DB specs.

---

### C2. No memory-bandwidth derating → decode modeled at >100% MBU (physically impossible)
**Evidence (reproduced):** llama2_7b decode, batch 1, on A100_80GB (bf16):
```
DECODE TPOT = 6.559 ms  => 152.5 tok/s
weights-only roofline (14 GB / 2039 GB/s) = 6.87 ms => 146 tok/s  (100% MBU ceiling)
implied decode MBU = 104.7%   <-- exceeds the bandwidth ceiling, before even counting KV-cache reads
```
The model returns a decode step *faster than reading the weights once at peak HBM bandwidth* — and a
real decode step also streams the KV cache and activations, so the true implied MBU is even higher.
Real-world single-stream decode runs at ~60–85% MBU (HF/vLLM measurements). Decode throughput is
therefore overstated by roughly **1.3–1.7×**.

**Root cause:** `memory_efficiency` defaults to `1.0` (system.py:112, get_inference_system) and is
**only ever lowered for one device (GB10)** via its `inference_calibration` block. `get_memory_time`
divides byte-traffic by `system.memory_efficiency` (operator_base.py:472), so with eff=1.0 the model
assumes 100% of peak HBM bandwidth is usable. No GPU achieves that.

**Impact:** All decode latency/TPOT/ITL and decode/serving throughput numbers are too fast for every
GPU except GB10. Serving simulations (req/s, concurrent users for an SLO) inherit the error.

**Fix:** Give every hardware config a realistic `eta_mem` (MBU) — typically 0.65–0.85 depending on
arch — sourced from measured decode runs, mirroring the GB10 calibration block. Do **not** ship 1.0.

**Corroboration (the authors know):** `validation/inference_calibration.py`'s own docstring states the
roofline *"(max(compute, memory), efficiency=1, no launch/per-stream overhead) is optimistic on real
hardware"* and provides a `differential_evolution` fitter to derive per-device `eta_mem/eta_compute/
t_launch` from measured benchmarks. **It has been applied to exactly 1 of 92 devices (GB10).** The
systematic fix exists but is dormant for 91 devices. Real published single-stream Llama-3-8B decode is
~110–160 tok/s on H100 and ~60–90 tok/s on A100 (vLLM) — the sim returns ~250 and ~150. The CPU path
is affected too: SapphireRapids llama2_7b decode = 14.3 tok/s vs a 12.9 tok/s weights-only ceiling at
the configured 180 GB/s (again >100% of DDR bandwidth).

---

### C3. No compute-efficiency derating + dead tensor-core model → prefill modeled at ~82% MFU (real ~40–55%)
**Evidence (reproduced):** llama2_7b prefill @2048 on A100_80GB → implied **MFU 81.8%**. Published
dense-prefill MFU for 7–8B LLaMA-class models on A100/H100 is ~40–55%. Prefill latency / TTFT is
therefore ~1.5–1.6× too fast (on top of C1 for H100).

**Root cause:** Two compounding issues:
1. `compute_efficiency` defaults to `1.0` and is only lowered for GB10 (same gap as C2).
2. The roofline multiplies compute time by a **tensor-core efficiency that is hard-wired to 0.85**
   regardless of matrix shape — see C4. So effective MFU ≈ 1.0 × 0.85 = 0.85.

**Impact:** prefill/TTFT, prefill throughput, and the prefill portion of end-to-end and serving sims
are all ~1.5× optimistic for every GPU except GB10.

**Fix:** Calibrate `eta_compute` per device (measured prefill MFU), and fix C4 so shape-dependent
efficiency actually runs (decode M=1 GEMMs should get ~0.4, large prefill GEMMs ~0.9).

---

### C4. "Phase 2" shape-dependent tensor-core efficiency is **dead code** — always returns 0.85
**Evidence (reproduced):**
```
get_tensor_core_efficiency(8192,4096,4096) = 0.95   # the intended scalar API works
get_tensor_core_efficiency(1,4096,4096)    = 0.40   # decode (M=1) penalty works
BUT:
GEMM.get_tensor_core_efficiency_factor()   = 0.85   # <-- the value actually used; default, never shape-aware
Logit.get_tensor_core_efficiency_factor()  = 0.85   # <-- default
FC.get_tensor_core_efficiency_factor()     -> TypeError: '>' not supported between 'tuple' and 'int'
```
**Root cause:** `operator_base.py:397 get_tensor_core_efficiency_factor`:
- For `GEMM`, `get_dimensions()` (operators.py:102) returns `[self.get_tensors()]` — a **list**, so
  `isinstance(dims, tuple)` is False → falls through to `return 0.85`.
- For `Logit`/`Attend`, `get_dimensions()` returns a **tuple of 3 tensor-shape tuples** (len 3), and
  the branch requires `len(dims) >= 4` → falls through to `return 0.85`.
- For `FC`, the same len-3 tuple makes `m,n,k = dims[0],dims[1],dims[2]` each a *shape tuple*, then
  `get_tensor_core_efficiency` does `tuple / int` → **TypeError** (latent crash; avoided only because
  LLM models emit `GEMM`, never `FC`).

So the elaborate tile-padding + wave-quantization + small-M-decode penalty in
`get_tensor_core_efficiency()` (operator_base.py:60–107) **never executes**. Every op gets a flat
0.85 multiplier. Likewise `get_shape_dependent_op_intensity()` (op_base:110) only feeds the displayed
'Op Intensity' column and, for GEMM, isn't even used there (the `isinstance(...tuple)` check fails too).

**Impact:** The headline "Phase 2 accuracy improvement" does nothing for accuracy; it's a constant.
The decode-specific compute penalty (which would *raise* decode compute time and is the physically
correct behavior for M=1) is silently disabled. The FC path is a latent crash.

**Fix:** Make the operators expose scalar `(M,N,K)` to the efficiency factor (normalize
`get_dimensions()` contract; GEMM/Logit/Attend each pass real scalar dims). Add a unit test asserting
`factor != 0.85` for at least one shape. Fix FC dim extraction. Decide whether shape-aware TC
efficiency should *replace* or *compose with* the per-device `eta_compute` (C3) to avoid double-count.

---

## 🟠 HIGH findings

### H0. Gemma family memory under-counted ~23% — GeGLU FFN treated as 2-matrix
**Evidence (reproduced):** with a full, correct Gemma-2-9B config the parameter counter returns
**7.084B vs the real 9.24B (−23.3%)**. Forcing `hidden_act='geglu'` returns exactly 9.242B ✓.
By contrast Llama-2-7B, Llama-3-8B, Llama-3.2-1B, Mistral-7B, Qwen2.5-7B, Mixtral-8x7B all count to
within **0.1%** — so the counter core is sound; the failure is Gemma-specific.

**Root cause (two compounding bugs):** `parameter_counter.py:280` reads the activation as
`config.get('activation_function', config.get('hidden_act', 'gelu'))`, but **Gemma configs store it
under `hidden_activation`** — neither key matches, so it defaults to `'gelu'` and uses the 2-matrix
FFN branch (op:285-287). Even if the value were read, `'gelu_pytorch_tanh'` is **not in
`glu_variants`** (op:18, which only has `geglu`, not the pytorch/tanh spellings). `ConfigNormalizer`
does **not** map `hidden_activation → hidden_act` (confirmed). Gemma uses GeGLU = 3 FFN matrices, so
the missing matrix is `intermediate_size·hidden·layers` ≈ 2.16B params for the 9B.

**Impact:** Every Gemma / Gemma-2 / Gemma-3 / CodeGemma / PaliGemma memory estimate (weights, total,
min-GPU-count, batch-fit, hardware recommendation) is ~20–23% **too small** → the tool will recommend
hardware that cannot actually hold the model, risking real OOM. This flows through the whole API/UI.

**Fix:** (1) Read `hidden_activation` too (add to normalizer alias map). (2) Expand `glu_variants` to
include `gelu_pytorch_tanh`, `gelu_tanh`, `gelu_new`, `gelu_approx`, `gated-gelu`. (3) Best: drive
GLU/3-matrix detection from `model_type` (gemma*, llama, mistral, qwen2/3, phi3, yi, deepseek… are all
gated) instead of fragile activation-string matching. Add regression tests for Gemma/Phi/Yi.

### H1. Public `estimate_prefill/decode_performance` crash on a string `system_name`
**Evidence (reproduced):**
```python
L.estimate_decode_performance(model='llama2_7b', system_name='A100_80GB_GPU', ...)
# AttributeError: 'str' object has no attribute 'get'   (performance_estimator.py:222)
```
**Root cause:** the public wrapper does `system_name.get('type')` assuming a dict, while the
underlying `decode_moddeling`/`prefill_moddeling` *do* resolve string names via
`get_inference_system` → `get_hardware_config`. The wrapper is strictly less capable than what it
wraps, and the error is opaque.

**Impact:** The most natural call (pass a hardware *name*) fails for the documented top-level API.
Anyone using the package API (not the internal moddeling fns) hits this immediately.

**Fix:** In `estimate_*_performance`, if `system_name` is a str, resolve it via
`get_hardware_config(name)` (or pass it straight through to the moddeling fn, which already handles
str). Guard the `.get('type')` with `isinstance(system_name, dict)`.

---

## 🟠 HIGH findings (API / service layer — all reproduced against the running server)

### H2. `POST /api/v2/power/estimate` returns fabricated power numbers (HTTP 200) for a nonexistent model
**Evidence (reproduced):** `{"model":"nonexistent_model_xyz","hardware":"H100_GPU"}` →
**HTTP 200** with `base_power_w=282.86`, `estimated_power.avg_power_w=523.9`, full energy breakdown.
**Root cause:** `serving.py:144-168` wraps `get_modeling_functions`/prefill/decode in a broad
`try/except` that, on *any* failure, substitutes hand-tuned magic latencies
(`input_tokens*0.01*batch_size/tp`, …) and continues. A typo'd or broken model silently yields
believable garbage — the exact failure mode the team fixed for `simulate/serving` (see comment at
serving.py:255-258) but left in `power/estimate`.
**Impact:** Power/energy/cost figures are untrustworthy; callers can't distinguish a real estimate
from a fallback. **Fix:** remove the fabricating fallback; surface the modeling error as 4xx/502.

### H3. `ModelMemoryCalculator.calculate_parameters()` does not exist → param count silently 0
**Evidence (reproduced):** grep confirms no `def calculate_parameters` anywhere; the real API is
`UniversalParameterCounter.count_parameters()`. Called at `usecases.py:711` and `:821`, both inside
bare `except Exception` → `model_params = 0`.
**Impact:** The CPU-exclusion rule `model_params > 14e9` (usecases.py:716-719) **never fires** via this
path → CPUs get recommended for >14B models; category `parameter_count` stays 0. A real logic bug
hidden by a swallowed `AttributeError`. **Fix:** call `UniversalParameterCounter().count_parameters(cfg)`.

### H4. Creating a usecase without TTFT/E2E bounds crashes with HTTP 500 (`None > None`)
**Evidence (reproduced):** `POST /api/usecases` with required token fields but no `ttft_min/max`,
`e2e_min/max` → **500** `{"detail":"Internal error: '>' not supported between instances of 'NoneType'
and 'NoneType'"}`. The SLO bound fields are `Optional` (schema), but a min≤max validation compares
them without a null guard. **Impact:** can't create a usecase that omits optional SLOs; 500 (not 422)
on a normal input. Any prepopulated usecase with null SLOs will also crash the optimizer
(`hardware_optimizer.py:234` `perf['ttft'] <= usecase['ttft_max']`). **Fix:** guard min/max
comparisons for None; return 422 on genuine validation errors.

## 🟡 MEDIUM findings

### M4b. Prefill attention ignores causal masking → computes full S², ~2× too slow on long context
**Evidence (reproduced):** for llama2_7b prefill @4096, the `Logit` op output dimension is
`(1, 32, 4096, 4096)` — the **full** M×N=S² score matrix. Causal LM prefill (FlashAttention
`causal=True`) computes only the lower triangle ≈ S²/2. Measured attention share of prefill: **13.7% at
4k ctx, 56.0% at 32k ctx**. Halving the (over-counted) attention cuts 32k-ctx prefill latency ~28%.
**Root cause:** the `Logit`/`Attend` operators (`operators.py:161,189` `num_ops = B·H·M·N·D`) and the
attention memory use the full M×N with no causal factor; nothing applies the ½ triangle. **Impact:**
long-context prefill/TTFT attention is ~2× too slow; partially offsets the MFU optimism (C3), so the
net TTFT error is context-dependent and the two bugs mask each other at medium context. **Fix:** apply
a causal factor (~0.5 for the score/softmax/AV work) to prefill `Logit`/`Attend`, matching how real
kernels skip masked blocks.

### M5. `GET /api/models/popular?limit=` unbounded & accepts negatives → wrong results + slow fan-out
**Evidence (reproduced):** `?limit=-5` → HTTP 200, **17.8 s**, returns a (mis-sliced) list;
`?limit=100000` → HTTP 200, **23.9 s**. No `Query(ge=1, le=N)` bound (models.py:715), unlike every
other list endpoint. Each model triggers a HF metadata fetch, so large limits are a latency/DoS
amplifier. **Fix:** `limit: int = Query(10, ge=1, le=100)`.

### M6. Serving workload length sampling inverts min/max for small means → degenerate workload, silent
**Evidence (reproduced):** function-level repro — `input_length_mean=4` ⇒ `std=1, min=32, max=16`
(min>max). `np.clip(samples, 32, 16)` collapses **every** sampled length to 16 (NumPy applies a_max
last). `POST /api/v2/simulate/serving` with `input_length_mean=4` returns HTTP 200 with an all-identical
workload far below the stated `min`. **Root cause:** serving.py:282-291 hardcodes `min:32/8` while
`max = mean*4`. **Fix:** `max = max(mean*4, min_floor)`; validate mean ≥ min.

### M7. Training-simulator endpoints can't resolve GenZ model names; HF-fetch failure returned as 404
**Evidence (reproduced):** `/api/simulator/{check-fit,estimate-time,recommend-cluster,estimate-training}`
with `model="llama2_7b"` → **404** "Could not load model config for llama2_7b: 401 Client Error" (it
tries to fetch `llama2_7b` from HuggingFace). The *same* name works for `/serving` and `/power` (GenZ
registry). With a HF-fetchable id (`gpt2`) training works (HTTP 200, fits=true). **Impact:** (1) model
identifier space is inconsistent across endpoints — a name that works in one 404s in another;
(2) an upstream 401/network error is mis-reported as 404 (model-not-found). **Fix:** resolve via the
GenZ registry first (as serving does), and map upstream auth/network failures to 502/400, not 404.

### M1. Pipeline-parallel throughput uses fill-latency, not steady-state bottleneck → understates PP throughput
**Root cause:** `llm_decode.py:175-179` and `llm_prefill.py:105-110`:
```python
stage_latency = latency / pipeline_parallel
total_latency = latency + (pipeline_parallel - 1) * stage_latency
thrpt = 1000 * batch_size / total_latency
```
`total_latency` is the latency to push **one** batch through a filling pipeline (= latency·(2−1/PP)),
which is *larger* than single-device latency. So adding pipeline stages **reduces** modeled throughput
— the opposite of the real steady-state behavior, where throughput is gated by the slowest stage
(≈ `1000·batch / stage_latency`) once the pipeline is full with enough micro-batches.

**Impact:** PP configs look worse than they are; the parallelism optimizer will under-rank PP.
(Lower blast radius because PP is rarely optimal for inference, but the formula is still wrong.)

**Fix:** Model steady-state PP throughput from the bottleneck stage and micro-batch count; keep the
fill/drain term only for *latency*, not throughput.

### M2. Decode KV-growth heuristic for 2–10 output tokens ignores context size
**Root cause:** `llm_decode.py:131-133`: for `2 < output_tokens ≤ 10`,
`decode_latency = initial_latency * (1 + 0.1*output_tokens/10)` — a fixed ≤10% bump independent of
input context length. For a 100k-token prompt generating 5 tokens, KV growth is negligible yet the
formula still adds up to ~5%; for tiny prompts it may under/over-count. Minor magnitude but arbitrary.

**Fix:** Scale the growth term by actual KV bytes (∝ context length), or just always use the
two-point average path (already implemented for >10).

### M3. `FC` operator path is a latent crash (TypeError) — see C4. Filed separately because it also
breaks any future model that emits FC ops, not just the efficiency factor.

### M4. Parameter counter is fragile to missing `hidden_act` and mis-marks `llama` as always-tied
**Evidence (reproduced):**
- Llama-2-7B config **without** `hidden_act` → counter returns 5.296B vs 6.738B (**−21.4%**): the
  GLU detection falls back to `gelu` (2-matrix FFN) instead of SwiGLU (3-matrix).
- Llama-2-7B config **without** `tie_word_embeddings` → 6.607B vs 6.738B (−1.9%): `llama` is in
  `always_tied_models` (parameter_counter.py:14) so the counter assumes tied embeddings, but
  Llama-1/2 and Llama-3-8B/70B do **not** tie (they have a separate `lm_head`).

**Impact:** Lower than H0 because well-formed HF configs carry both fields, but custom/partial configs
(common in the "paste a config" UI path and for niche models) silently lose ~20% of FFN params or one
embedding matrix. Marking `llama` as always-tied is factually wrong and a footgun if the flag is absent.

**Fix:** Infer the gated-FFN default from `model_type` (see H0 fix #3). Remove `llama` from
`always_tied_models` (respect the explicit flag, which real configs always set; default untied for
llama if absent). Add the same for `mistral`, `falcon` (Falcon-7B/40B do not tie).

---

## 🟠/🟡 Communication & collective-cost modeling (collective_times.py)

### H5. Two large, well-engineered collective models are **dead code**; production uses the cruder path
**Evidence (grep-verified):**
- `get_hierarchical_AR_time` (collective_times.py:1063-1144) — the properly-parameterized 2-level
  (intra-NVLink / inter-IB) AllReduce with explicit 100 GB/s + 2 µs IB defaults — has **no callers**.
- `get_comm_efficiency` + `COMM_EFFICIENCY_TABLES` (collective_times.py:191-339) — 150 lines of
  per-hardware (H100/A100/TPU/MI300X/GB200) message-size efficiency tables — are referenced **only**
  inside `get_comm_efficiency` itself (self-ref at :319); no collective function calls them.
**Impact:** The "Phase 8 hardware-calibrated comm efficiency" is non-functional — an MI300X and an
H100 get identical hardcoded bucket efficiencies (`0.3/0.6/0.85`). Hardware choice doesn't affect
modeled collective efficiency at all. **Fix:** wire `get_comm_efficiency` into AR/AG/RS, or delete the
tables and stop advertising hardware calibration.

### H6. Inter-node bandwidth hardcoded at 40% of the intra-node link → multi-node collectives ~3× optimistic
**Evidence (verified at collective_times.py:384, :511, :919):**
`inter_node_bw = system.interchip_link_bw * 0.4`. A real 8-GPU H100 node has ~450 GB/s/GPU NVLink but
only ~50 GB/s/GPU of InfiniBand — a true inter:intra ratio of **~11%, not 40%** → inter-node AllReduce
/AllGather/ReduceScatter bandwidth terms are **~3.6× too fast**. Compounds: `inter_node_latency =
interchip_link_latency * 10` derives IB latency by scaling the NVLink number rather than using a real
IB value. (Only affects multi-node, i.e. TP/EP/PP that spans nodes; single-node TP≤8 is unaffected.)
**Fix:** source inter-node BW/latency from real IB/RoCE specs (use the already-written
`get_hierarchical_AR_time`), not a fraction of NVLink.

### M8. Intra-node TP collectives ~1.15–1.4× pessimistic — efficiency de-rating applied twice
**Evidence (collective_times.py:413 + :459-473):** the bandwidth term is divided by `bw_efficiency`
(0.85 for large msgs) **and** the whole result is then multiplied by `compute_scale_aware_overhead`
(≥1.08). Both model the same protocol/effective-BW loss → double-count. Single-node TP=8 large-message
AllReduce comes out ≈1.40× the bandwidth-optimal time. **Note:** this pushes opposite to H6, so for
multi-node the two errors partially mask in aggregate but diverge per-component. **Fix:** apply one
de-rating, not both.

### M9. Non-multiple-of-`gpus_per_node` parallel degrees collapse to a single fictional NVLink domain
**Evidence (collective_times.py:404, :526, :935):** `num_nodes = max(1, numNodes // gpus_per_node)`
with `gpus_per_node` defaulting to 8 on the inference path. `TP=12` → `12//8 = 1` → 12 GPUs modeled as
one all-NVLink ring (physically impossible); `TP=20` → 2 nodes (drops the partial third). Any
non-power-of-2 / non-multiple-of-8 degree silently gets wrong topology, and the default 8 ignores the
device's real NVLink width (GB200 NVL72 = 72). **Fix:** ceil division and source `gpus_per_node` from
the hardware config. *(MoE A2A also skips the load-imbalance/congestion factors on the inference path —
the authors' own `get_A2A_time_moe` docstring notes a "160% error" on DeepSeek-V2 EP=8; the inference
graph calls the plain `get_A2A_time`.)*

## 🟠 HIGH findings (Web UI — frontend, code-verified)

> **UI live-test limitation:** the connected Chrome runs on macOS (the user's laptop) while the
> servers run on this remote Linux GPU box; the Mac browser cannot reach the server's `localhost`
> (verified — both :3000 and :8000 error in-browser while `curl` succeeds locally). UI findings below
> are from source audit cross-checked against the live API responses, not a live click-through.

### H7. Relative price *index* is displayed as real dollars — "~$X.XX USD/hr"
**Evidence (verified):** `src/hardware.py calculate_price_indicator` docstring: *"This is NOT actual
pricing, only for relative comparison … Returns: Approximate price indicator (relative scale, not
actual dollars)."* The `RecommendationResponse` field doc repeats "NOT actual pricing." Yet
`AIMemoryCalculator.tsx:1797` and `:1914` render `~${hw.price_approx.toFixed(2)} USD/hr`. So a
dimensionless log-scale index (`10^(-3.46 -0.05·log10(flops) +0.49·log10(mem) +1.03·log10(bw))`) is
shown to users as an hourly dollar cost. (The `HardwareCard` component does it right — "Relative Price"
+ disclaimer.) **Fix:** label it "Relative price index" with the disclaimer, or replace with the real
`cost.*_on_demand` values that exist in the hardware configs.

### H8. "Model Compatibility" tab shows hardcoded fake memory numbers on every hardware page
**Evidence (verified):** `HardwareDetail.tsx:66-92` — `// Mock model compatibility data - in real
implementation, this would come from API`; a fixed Llama-2-7B / Mistral-7B table with invented memory
(14.2/16.8/22.1/32.7 GB) and "batch size: 10". Only the green/yellow/red status is computed live
against the real `hardware.memory_size`; the memory figures are fake and identical for every device.
**Fix:** call `/api/models/calculate` (or `/analyze`) per model×seq-len and render real numbers; or
remove the tab until wired.

### H9. Usecase optimizer "Utilization" is a hardcoded 70% and "cost/hr" a static table — presented as modeled
**Evidence (verified):** `src/optimization/hardware_optimizer.py:345` and `:573`:
`compute_util = 0.7  # Default 70% utilization` returned as the config's `utilization` for every
result; `cost_per_hour` from a hardcoded per-GPU dict (lines 74-85) × nodes; `cost_per_request =
cost_per_hour/(throughput·3600)`. `OptimizedHardwareRecommendations.tsx:423` prints
`{(config.utilization*100).toFixed(1)}%` → always "70.0%", and the component footer claims it uses
"advanced performance modeling to predict actual latencies and throughput." **Fix:** compute real
utilization from the roofline result; source cost from the per-cloud pricing already in configs.

## 🟡 MEDIUM findings (Web UI)

### M10. Hardware catalog Sort-by / Order / FLOPS-range controls silently do nothing
`hardwareAPI.ts:94-101` sends `sort_by/sort_order/min_flops/max_flops` to `GET /api/hardware`, but that
endpoint (hardware.py:203-211) only accepts `type/manufacturer/min_memory/max_memory/limit/offset` —
FastAPI drops unknown params. The controls render as functional but never change the list. (The
`/filter` endpoint does support them; the main list doesn't.) **Fix:** add the params to the list
endpoint or point the UI at `/filter`.

### M11. Hardware catalog pagination count is fabricated
`HardwareList.tsx:61-66` invents a total ("+1 more page" whenever a full page returns) because the list
endpoint returns no count → wrong `totalPages`, navigable empty/short final page. **Fix:** return a
`total_count` from the API (the DB query already knows it) and use it.

## 🟡 Training subsystem

### M12. `check-fit` returns `memory_per_gpu_gb: 0.0` when the config doesn't fit (correct value only in free text)
**Evidence (reproduced):** `check-fit Qwen-7B full FT, num_gpus=1` → `{fits:false, memory_per_gpu_gb:0.0,
utilization_percent:0.0, reason:"Insufficient memory: 152.9GB required, 80.0GB available across 1
GPUs"}`. The typed field (schema doc: "Memory per GPU required") is 0.0; the real 152.9 GB is only in
the prose `reason`. **Root cause:** `cluster_selector._find_best_parallelism` returns the default
`memory_per_gpu_gb=0.0` when no fitting parallelism is found. **Fix:** populate the field with the
unsharded/required per-GPU number even on the not-fit path.

### M13. Training uses a THIRD hardware source + its own cost table (fragmentation, ties into C5/C7)
`cluster_selector.check_fit` reads `HARDWARE_CONFIGS` (static) directly and carries its own
`GPU_COST_PER_HOUR` dict, separate from (a) the inference engine's `get_hardware_config` and (b) the
served DB. Three independent hardware sources → specs/cost can silently diverge across the
training / inference / catalog surfaces. **Fix:** single hardware source of truth.

**Training VALIDATED CORRECT:** time/cost estimate is sound — Qwen-7B, 1B tokens, 8×A100 → 11.15 h,
MFU **0.4625** (realistic for training), cost $327 (= 11.15h·8·$3.67 ✓). Full-FT ZeRO-3 memory
(13.38 GB/gpu for 7B on 8 GPUs) is correct. **Note the contrast:** the *training* calculator uses a
realistic ~46% MFU while the *inference* roofline uses ~85% (C3) — the same codebase knows the right
number on one path and not the other.

### M14. Power model: accelerator power is workload-insensitive (a coarse fixed fraction of TDP)
**Evidence (reproduced):** `power/estimate llama3_8b H100` gives `accelerator_w = 288.75 W` (exactly
**0.4125 × 700 W TDP**) for bs=1/in=128/out=8, **and** bs=1/in=4096/out=256, **and** bs=32/in=2048 —
identical. Only at bs=128/in=8192 does it rise (469.9 W, 0.671×TDP). A bs=32 decode does far more work
per step than bs=1 yet draws the same modeled accelerator power. **Impact:** energy-per-token and
power/cost figures don't track utilization across the batch/context range (the regime users compare).
**Fix:** drive accelerator power from the roofline's actual compute/memory utilization
(idle + util·(TDP−idle)), not a step constant. (Separate from H2, which is the fabricated-fallback bug.)

## 🔵 Engine behaviors VALIDATED CORRECT (not bugs — recorded to bound the audit & avoid false positives)
- **GQA KV cache:** llama3_8b (kv_heads=8) KV growth = ¼ of llama2_7b (kv_heads=32) — measured 4.0×
  ratio (Δ7.36 ms vs Δ1.85 ms over 2k→32k ctx). Correct.
- **Batch scaling:** decode throughput rises (153→1446 tok/s) then saturates when KV-bandwidth-bound;
  OOM raised correctly at bs=128/80GB. Qualitatively correct roofline behavior.
- **Quantization memory scaling:** bf16→int8/fp8 halves decode TPOT (2×), int4 quarters it (4×).
  Correct *for the memory term*. (Caveats in L5.)
- **Parameter counter** is accurate to <0.1% for Llama-2/3, Llama-3.2, Mistral, Qwen2.5, Mixtral-8x7B
  (MoE), and Llava-1.5-7B (multimodal: 7.04B vs 7.06B). Only Gemma/GeGLU (H0) and missing-field
  robustness (M4) fail.
- **MoE decode routing:** Mixtral-8x7B TP=4 decode (5.12 ms) ≈ Llama-2-13B TP=4 (4.69 ms), i.e. it
  reads only the top-2 active experts for latency while requiring all 8 resident for fit. Correct.
- **Speculative decode:** `expected_tokens(N=4, x=0.7)=2.77` matches Leviathan et al. `(1−α^(γ+1))/(1−α)`;
  per-step latency = `N·draft_decode + full_verify`, throughput = `expected_tokens/latency`. Correct.
- **`estimate_max_batch_size`:** 40/80/141 GB → 13/37/74 for Qwen-7B@2k. Sane (slightly superlinear as
  fixed weight overhead amortizes). Correct.
- **Training time/cost:** Qwen-7B/1B-tokens/8×A100 → 11.15 h, MFU 0.4625 (realistic), $327 (= h·8·$3.67).
  Full-FT ZeRO-3 memory (13.38 GB/gpu) correct. (Only M12/M13 are issues.)
- **Power magnitude:** within TDP for H100/A100/B200 at bs=1 (607/347/868 W). Plausible (M14 is the
  workload-insensitivity, not the magnitude).
- **Serving simulator (policy level):** Qwen-7B/A100 — TTFT rises with offered load (43→53→163→792 ms
  for rps 1→5→20→100) so queueing is modeled; achieved throughput saturates (~10.4 rps ≈ 1330 tok/s,
  matching the decode ceiling) so backpressure/contention is modeled. Scheduling/batching logic is
  sound; the absolute TPOT/TTFT inherit the C2/C3 optimism but the request-level dynamics are correct.

## 🔵 LOW / dead-code / quantization nuances

### L5. int4/int8 decode is over-optimistic (compounds C2)
int4 llama2_7b decode = **613 tok/s** on A100; real AWQ/GPTQ int4 single-stream is ~150–250 tok/s
(dequant overhead; A100 has no int4 tensor path, so the only win is memory traffic, and even that sits
on the >100% MBU baseline of C2). Also `bits='int8'/'int4'` quantizes the **KV cache too** (the
`mem_multiplier` applies to all tensors), but real weight-only quant (AWQ/GPTQ/bnb) keeps KV in fp16 —
so long-context quantized decode is doubly optimistic. **Fix:** separate weight precision from KV-cache
precision; add a dequant-overhead term for weight-only int4 on non-native hardware.

### L6. `parameter_count` not null-guarded in two UI spots (`AIMemoryCalculator.tsx:1667`, `:2092`)
→ renders "0.0B" / "NaNB" when the backend returns null param count.

## 🔵 (original LOW list)

### L4. `compare` partial-failure & nonexistent-model→"MODEL_GATED": `POST /api/models/calculate`
with a nonexistent id returns **403 "MODEL_GATED / requires authentication"** rather than 404 — a
nonexistent model is mis-described as gated (misleading error). LOW.

### L0. `tf32` compute multiplier = 1 (same as bf16) but A100 tf32 is half-rate
**Evidence (reproduced):** llama2_7b prefill on A100 — bits=`bf16` and bits=`tf32` both give 112.36 ms;
`fp32` correctly gives 224.72 ms (2×). Real A100 tf32 tensor cores run at **156 TFLOPS = half** of bf16
(312). **Root cause:** `system.py:24` `compute_multiplier['tf32'] = 1` (should be 2). LOW — tf32 is
rarely used for LLM inference, but it's a clear spec error (tf32 prefill modeled 2× too fast).

### L1. FP8 warning fires on *all* hardware including H100/MI300 (system.py:130) — should check arch.
### L2. `OFFLOAD_BW = 128` GB/s default (utils.py:10) is optimistic for PCIe CPU-offload (PCIe5 x16 ≈ 64 GB/s).
### L3. `get_shape_dependent_op_intensity` + `OPERATION_CHARACTERISTICS` are effectively dead for timing.

---

## Resolved leads (checked — NOT bugs)
- Offload trigger `memory_parallelism = PP*EP` (excludes TP): **correct.** TP sharding is already
  baked into the summary weights (verified: TP=1→12.55GB, TP=2→6.28, TP=4→3.14, TP=8→1.57), so total
  per-device division = TP·PP·EP as intended.

## Pending verification (leads, not yet confirmed)
- KV-cache memory math for GQA/MQA and MoE expert weight counting.
- CPU inference path (`cpu_aware_decode_moddeling`) accuracy.
- Whether `compute_time==0 → memory_time=0` (op_base:477) zeroes legitimate memory-bound elementwise ops.
- Hardware DB (`prepopulated.db`, 92 devices) Flops/BW vs vendor specs (only static configs checked so far).

---
---

# REVIEW ROUND 2 — Full-system component deep-review (2026-06-17)

After the 39 findings above were remediated (branch `fix/accuracy-remediation`, 783/24 tests + 58/58 API
harness green), a brutal review of **every other subsystem** (5 parallel review agents + cross-hardware
validation + live API). All findings below are **reproduced** (agent ran code/API), then independently
spot-verified. Cross-hardware sweep result: the Round-1 efficiency fixes (C2 `eta_mem`, C3 `eta_compute`)
hold across **all 84 simulatable devices** — 0 exceed configured peak BW or FLOPS. The 8 "errors" are
correct (a 13GB model genuinely doesn't fit on 8-12GB consumer GPUs).

> **✅ IMPLEMENTED (2026-06-18).** All five Round-2 units (CPU, Serving, Optimizers+router, Training,
> BudEvolve) are now fixed on `fix/accuracy-remediation` via extreme TDD + per-unit regression + live API
> verification. Per-unit status table and verification numbers: `solutions_round2.md` §"IMPLEMENTATION
> STATUS". Not committed.
>
> **⚠️ LIVE-STATE CORRECTIONS (2026-06-18, during Round-2 solution design — `solutions_round2.md`).**
> Re-verifying the findings against the live `fix/accuracy-remediation` tree turned up stale citations.
> Honor these before implementing:
> - **R2-OP1 is ALREADY LANDED** — `hardware_optimizer._get_hardware_types()` already returns `_GPU`-suffixed
>   resolvable names. Do not re-fix.
> - **R2-OP2 fabrication is GONE** — `budevolve/evaluator.py` already catches GenZ failures into
>   `EvalResult(feasible=False)`; there is no `throughput=2000` invention. The residual defect is *silent-empty
>   success* on the 3 `optimize/*` v2 endpoints (no up-front 404 guard), not fabrication.
> - **R2-TR1 / R2-TR3 mis-attributed** — the API path is `training/calculator.py:84`
>   (`TrainingMemoryCalculator`), NOT `distributed.py`/`advanced_calculator.py` (those are **dead to the API**;
>   grep confirms zero `apis/` references). PP/EP-blindness lives in `calculator.py`. The optimizer-table
>   consolidation goes ONTO the sourced `OPTIMIZER_CONFIGS`, not the GenZ `OPTIMIZER_STATE_BYTES`.
> - **R2-BE2 == R2-G1**, already fixed at the `utils.get_inference_system` System-object branch (verified
>   empirically: BudEvolve `evaluate_hardware` inherits eta<1). No BudEvolve-local change.
> - **R2-SV3 RCA corrected** — GenZ offload does not raise; it returns a larger valid latency with
>   `is_offload=True`. The defect is the scheduler never reading that flag (not the bare-except).

## R2 — Gaps in the Round-1 fixes (close these to make the landed work consistent)

### R2-G1 🟠 The `System`-object path bypasses the C2/C3 efficiency bands → 100% MFU/MBU on that path
`get_inference_system` (utils.py) applies `_default_eta_mem`/`_default_eta_compute` only in the **dict**
branch; the **`System`-object** branch sets `compute_efficiency=ceff(1)/memory_efficiency=meff(1)` and
returns — no bands. **BudEvolve `HardwareExplorer`/`evaluate_hardware` builds a `System` directly**, so it
runs at 100% efficiency: measured Llama-3.1-8B on matched-A100 specs → System-obj tpot=8.02ms vs dict
`A100_80GB_GPU` 10.03ms (=1/0.8) and ttft 850 vs 1090ms (=1/0.78) — **~25-28% optimistic**. `compare_vs_real`
then pits the System-obj hypothetical (100%) against named real GPUs (banded) — a rigged comparison.
**Fix:** apply the bands in the System-object branch too (default by a generic HBM band when the System
carries no memory_type/arch), or have callers set efficiency from the bands.

### R2-G2 🟡 CPU configs lack `memory_type` → decode uses 0.80 (generic) not the DDR5 band 0.65
Verified: `get_hardware_config('SapphireRapids_CPU'/'EmeraldRapids_CPU'/'Genoa_CPU').memory_type == None`.
The C2 table HAS `ddr5:0.65/ddr4:0.62`, but every CPU dict in `hardware/cpu_specs.py` omits `memory_type`,
so `_default_eta_mem(None)=0.80` applies → CPU decode ~20% optimistic. **Fix:** add `memory_type:'ddr5'`
(or ddr4) to each CPU config.

## R2-CPU — CPU inference modeling (genz/cpu/) — TWO inconsistent paths, both wrong

- **R2-CPU1 🔴 AMX prefill crashes** `KeyError: 'tile_k'` on EVERY Sapphire/Emerald/Granite Rapids CPU.
  `isa_model.py:239` reads `constraints['tile_k']` but the AMX dict only defines `tile_k_bf16`/`tile_k_int8`.
  Verified: `cpu_aware_prefill_moddeling(..., system_name='intel_xeon_8592_plus')` → KeyError. The AMX
  prefill path has **never run**; tests dodge it (only test Ice Lake, which has no AMX).
- **R2-CPU2 🔴 CPUSystem decode ~10× too SLOW + non-deterministic.** `cpu_operator.py:26-48` uses
  `max(sampled_cycle_time, bandwidth_time)` where the sampled cache term (cap ~8000 samples, DRAM=lat+200
  cyc) dominates and **doesn't scale with data volume**; 7B bf16 decode on EMR → 0.84 tok/s (real ~5-9).
  `cache_model.py` samples with **no RNG seed** → same input varies 62% run-to-run (6.5–12ms). A sim that
  returns different latencies for identical inputs can't rank hardware.
- **R2-CPU3 🟠 Two disjoint CPU paths disagree ~20×:** CPUSystem (cache-sim) 0.84 tok/s vs static
  `CPU_CONFIGS`→roofline 18.6 tok/s for the same chip class. No single source of truth (mirrors C7).
- **R2-CPU4 🟠 ARM/SVE/NEON omit `bf16` in `vector_width`** (`isa_model.py`) → bf16 falls to scalar
  (width 1). Graviton3 default bf16 decode → 19.3 s/token. AVX512 also excludes bf16 in GEMM ISA select
  (`:246`) → bf16 prefill mis-picks AVX2 (~half rate).
- **R2-CPU5 🟡 `peak_int8_tops` is the bf16 rate (2× off); NUMA/multi-socket effectively unmodeled**
  (`get_effective_bandwidth` is dead code; dual-socket modeled as perfect 2× BW, no remote penalty).
- **R2-CPU-meta:** the 62-test CPU suite passes 100% yet never asserts end-to-end tok/s — which is why a
  hard crash, a 10× error, and 62% nondeterminism all stayed green.

## R2-TRAIN — Training simulations (3 memory paths; the GenZ one is right, two are broken)

- **R2-TR1 🔴 Pipeline parallelism never shards grad/opt/weights** in `distributed.py:476-518` &
  `advanced_calculator.py:486-531` (PP passed into config then dropped before memory math). 70B full-FT
  TP8·PP8 → **280 GB gradients, 560 GB optimizer per GPU** (physically impossible). The GenZ path
  (`training_modeling._calculate_training_memory`) shards by PP correctly — proving it's a bug.
- **R2-TR2 🔴 `cluster_selector._find_best_parallelism` divides total by tp·pp·dp blindly + ×0.7,
  ignoring ZeRO stage** (`cluster_selector.py:269-274`; no `zero_stage` param exists). 7B full-FT,
  `zero_stage=0` (plain DDP), 8×A100 → claims **fits at 11 GB/GPU** when DDP needs the full ~126 GB/GPU
  (doesn't fit). **(This corrects my earlier M12 self-assessment — the 13.38 GB I'd called "correct
  ZeRO-3" is actually this blind division.)**
- **R2-TR3 🟠 `recommend_distributed_strategy` sizes clusters on weights only** (`distributed.py:580-635`)
  — ignores grad+opt+activations → recommends ZeRO-1 for a 70B that needs offload.
- **R2-TR4 🟠 `cluster_selector._estimate_throughput` ~100× low** (`:343` models "read every param in
  fp32 per token" as the ceiling) → 7B/8×A100 = 279 tps vs the GenZ path's 29,752 tps.
- **R2-TR5 🟡 Two divergent optimizer-byte tables** (optimizers.py vs training_modeling) disagree on
  sgd/adafactor; **GaLore memory is rank-independent/self-inconsistent**. Silent `except: continue`
  drops infeasible configs in scaling-curve/optimal-GPU search.
- **VALIDATED CORRECT:** `estimate_training_time` (GenZ path) — Llama-2-7B/1B-tok/8×A100 → 9.34 h,
  MFU 44.3%, 62.6 GB/GPU. Activation+grad-checkpointing math sound. Adam=8B/param correct.

## R2-SERVE — Serving simulator internals (queueing scaffold sound; corrupt under memory pressure)

- **R2-SV1 🔴 Requests complete MULTIPLE times under memory pressure.** `batch_scheduler.py:118-139`
  evict-and-requeue resets only `status` (not `tokens_generated`/KV) → evicted requests re-prefill and
  `complete_batch` completes them again. Repro: 50 requests → **583 completions**. Every aggregate
  (throughput/goodput/percentiles) inflates whenever eviction fires.
- **R2-SV2 🔴 No admission control / failure accounting.** `set_failed()`/`total_requests_failed` are
  dead code — an un-admittable request loops forever via eviction instead of failing.
- **R2-SV3 🔴 Scheduler fabricates decode latency `0.5*bs/TP` ms** when GenZ KV exceeds HBM
  (`batch_scheduler.py:299-302`, bare except). The serving MemoryModel and GenZ disagree on KV budget;
  the fallback is physically meaningless.
- **R2-SV4 🟠 Power model not workload-proportional (the M14 root, now located):** `serving.py:167-168`
  reads `getattr(result,'compute_utilization',0.5)` / `memory_utilization` — **those attributes don't
  exist on the GenZ result** → always 0.5 → accel power identical (165W) for bs=1..32. (Plus DB-only
  devices lack `tdp_watts` → default 300W → identical across devices.)
- **R2-SV5 🟠 Prefix-cache analyzer fabricates ~99% hit rate** for every workload (`prefix_cache.py:223`
  integer-divides position by ~1M so all tokens hash to bucket 0 → near-total false match). Repro:
  `/api/v2/cache/analyze` on 100 unique prompts → hit_rate 0.99.
- **R2-SV6 🟠 Cluster/disaggregation throughput assumes batch=1 serial serving** (`cluster.py`,
  `disaggregation.py`) → 8×A100 cluster = 3.85 rps (1-2 orders of magnitude too low); KV-transfer cost
  uses (in+out) tokens instead of input-only at the prefill→decode handoff.
- **VALIDATED CORRECT:** SLO percentile math (nearest-rank, no off-by-one), TTFT-grows-with-load +
  throughput-saturation curve shape, workload RNG seeding. (M6 clamp guarded in the router; `WorkloadGenerator`
  presets still vulnerable to min>max collapse.)

## R2-OPT — Optimizers

- **R2-OP1 🔴 `hardware_optimizer` cannot recommend A100 or H100** (the two most common GPUs).
  `_get_hardware_types()` returns `['A100_40GB','A100_80GB','H100_80GB',...]` but `get_hardware_config`
  needs the `_GPU` suffix → returns `False` → `find_optimal_configuration` returns `None`. Verified.
  Usecase-optimization API is biased to AMD/TPU by a naming accident. **Fix:** use resolvable names
  (`A100_80GB_GPU`,`H100_GPU`) or the unified resolver.
- **R2-OP2 🔴 v2 `optimize/config|pareto|sensitivity` fabricate results for OOM/unknown hardware**
  (`config_optimizer.py:338-343` invents `throughput=2000` on GenZ OOM; `_check_feasibility` has NO
  memory check; the 3 endpoints skip the hardware-404 validation that `simulate/serving` has). Repro:
  `optimize/config` with `hardware="TOTALLY_FAKE_GPU"` → HTTP 200, best_score 2000.
- **R2-OP3 🟠 `get_best_parallelization_strategy` search is degenerate:** `best_parallelization.py:45`
  uses `TP*PP < total_nodes` (never uses all N nodes), generates **no EP/DP** (MoE never gets expert
  parallelism), aborts the whole search on one infeasible combo (no per-combo try/except), ranks by
  Tokens/s only (no latency/cost objective despite the docstring).
- **R2-OP4 🟠 ConfigOptimizer `latency` objective mixes units** (`config_optimizer.py:376`:
  `prefill_total + tpot` — drops the `×output_tokens`, understating decode ~128×) → wrong "lowest-latency"
  config. (`hardware_optimizer.py:587` does it right — inconsistent.)
- **R2-OP5 🟡 fp8≡int8 (identical 8-bit multiplier) pads the Pareto front with duplicates**; Morris
  sensitivity treats categorical `precision` as ordinal; `throughput_rps` is mislabeled tokens/sec
  (off by `output_tokens`); ConfigOptimizer grid truncates in product order (budget bias to low TP);
  `estimate_max_batch_size` hard-caps at 1024.
- **VALIDATED CORRECT:** `get_minimum_system_size` (sizes by weights+KV, power-of-2, feasibility loop);
  `estimate_max_batch_size` memory bounding + KV growth; H9 utilization (real, not 0.7); ConfigOptimizer
  throughput genuinely varies across configs.

## R2-BUDEVOLVE — BudEvolve (2 of 3 modes substantially broken)

- **R2-BE1 🔴 Algorithm-evolution fitness is independent of the evolved algorithm.** `evaluator_bridge.py:66-95`
  builds a `ServingConfig` from fixed attributes and never executes/feeds the candidate program → throughput
  identical for all candidates; only a static AST-shape `quality_bonus` varies. Repro: a scheduler returning
  `[]` scores the same throughput as a working one. `evolve-scheduler/cache-policy/cpu-scheduler` are degenerate.
- **R2-BE2 🟠 = R2-G1** (HardwareExplorer System-object path runs 100% MFU/MBU).
- **R2-BE3 🟠 Hardware-DSE cost & power objectives are constant** (`hardware_explorer.py:306-360` builds
  `HardwareSpec` without `tdp_watts`/`estimated_cost_usd` → defaults 700/25000) → multi-objective search is
  degenerate on cost/power; `what_if` on cost is a no-op; `_to_dict` omits cost.
- **R2-BE4 🟡 pymoo not installed** → DSE & config optimizer silently fall back to a coarse grid
  (warning-only); tests mock past it. **R2-BE5 🟡 `throughput_rps == token_throughput_tps`** (off by
  `output_tokens`). **VALIDATED:** roofline_analyzer still reads valid columns post-Round-1; NumericOptimizer
  config search is sound.
