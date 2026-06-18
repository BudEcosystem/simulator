# BudSimulator — Round-2 Solutions (RCA-grounded, regression-safe fix plan)

> Companion to `bugs.md` (Round-2 findings) and `solutions.md` (Round-1). Every fix here traces to a
> **full root-cause analysis**, is **grounded** in a datasheet value / measured microbenchmark
> (`audit_microbench_data.md`) / published benchmark / closed-form derivation (no magic numbers), and
> ships with a **no-regression proof** (byte-identical guard + golden test). Produced via a
> design → critique → revise multi-agent loop with a final cross-coupling synthesis.

This Round-2 plan covers the subsystems the Round-1 engine fixes did not touch: **CPU inference modeling**,
**Training simulation**, **Serving (continuous batching)**, **Optimizers / v2 endpoints**, and
**BudEvolve**. The unifying discovery — same as Round-1, now in the code paths Round-1 didn't reach — is
that each subsystem is a **second source of truth** that re-derives, discards, or fabricates what the GenZ
roofline engine and the unified hardware source already produce.

---

## ✅ IMPLEMENTATION STATUS (2026-06-18) — all 5 units landed on `fix/accuracy-remediation`

Implemented via extreme TDD (red→green) + per-unit regression + live API verification. Not committed.

| Unit | Fixes implemented | Targeted tests | Live verification |
|---|---|---|---|
| **CPU** | R2-CPU1 tile_k; R2-CPU2 cache-cycle term removed + seeded RNG + dead `framework_overhead_ms` deleted; R2-CPU3 per-socket/aggregate BW + DDR-eff derate + configs reconciled; R2-CPU4-partial decode threading (all cores) + operand-aware DRAM volume | `test_cpu_round2.py` 7/7; CPU suite 78; **full lmc regression 0 failures** | 7B bf16 decode **0.85→15.7 tok/s** (cpu_aware) ≈ 18.5 (static), deterministic; AMX prefill no crash (263 ms) |
| **Serving** | SV1/SV2 multi-tier admission+failure; SV3 `is_offload` propagation; SV4 sim-loop power from `roofline_utilization`; SV5 request-id-salted prefix cache; SV6 `B_eff` cluster/disagg throughput + magic fallbacks removed | `test_serving_round2.py` 19/19; **`tests/serving/` 346 passed, 0 failed** | offloaded flag set; power bs1 384W vs bs16 518W; prefix hit-rate 0.99→0.0; cluster 0.98→212 rps (B_eff=216); 5 oversized → 5 failed |
| **Optimizers + router** | R2-OP-A honest-failure + 404 guards on 3 v2 endpoints; SV4 router power; R2-OP-B full-cluster parallelism; R2-OP-C E2E latency + true rps + precision dedup + categorical Morris | `test_optimizer_round2.py` 20/20; `test_config_optimizer.py` 11 | `optimize/config` FAKE_GPU → **HTTP 404**; power uses real MFU/MBU (≈0.78/0.82) not 0.5 |
| **Training** | TR1 PP/EP + full-FT TP grad/opt sharding; TR2 rank-aware GaLore; TR3 component-aware cluster selector (0.7/0.6 fudge removed); TR4 single sourced optimizer table in GenZ path; TR5 resolved-by-TR4 | `test_training_round2.py` 9/9; **full training suite 0 failures** | **70B full-FT TP8·PP8 → 19.8 GB/GPU** (was impossible); API TP8·PP8 weight/grad/opt all ÷64 |
| **BudEvolve** | R2-BE1 candidate executed (sandboxed policy-efficiency); R2-BE3 cost/power swept decision vars + perf/W ceiling; R2-BE5 true request rps; improvement_pct same-scale; R2-BE2 guard test (R2-G1 already fixed) | **`tests/budevolve/` 82 passed** (15 new) | working vs degenerate scheduler score differently; swept-cost designs differ |

**Shared `roofline_utilization` helper** (`genz/serving/utilization.py`) derives MFU/MBU from a result's
repeat-corrected `summary_table` + unified device peaks; verified decode bs1 MFU 0.022/MBU 0.820, prefill
MFU 0.780/MBU 0.165 (≈ the landed eta bands). **TR5 candor:** the single-optimizer-table consolidation is
achieved (TR4 routes the GenZ path onto `OPTIMIZER_CONFIGS`, which the API + advanced paths already use;
`distributed.py` takes optimizer bytes as a parameter). The remaining advanced/distributed memory-math
duplication is non-behavioral and dead-to-API; left documented rather than rewritten to avoid regression.
**BudEvolve R2-BE1 candor:** the candidate is truly executed as a pure function (per-call policy efficiency),
surfaced via `fitness_basis` — not a full continuous-batching lifecycle (`BatchScheduler` has no hook).

---

## ✅ BRUTAL RE-REVIEW PASS (2026-06-18) — adversarial review of the implemented code

After the units landed, 5 fresh **adversarial** reviewers (one per unit) hunted for bugs the authors' own
tests missed, verifying each finding with a live experiment. **Optimizers+router: clean (0 bugs).** The
other four surfaced **7 real bugs**, all now FIXED and re-verified:

| # | Sev | Unit | Bug | Fix | Verified |
|---|-----|------|-----|-----|----------|
| 1 | HIGH | Serving | SV1/SV2 admission check sat *after* the `max_num_batched_tokens` guard → a prompt > 8192 (default) was re-queued forever (neither completed nor failed) | moved capacity check to the front of the loop + added a permanent-failure path for un-batchable prompts (input > budget, chunking off) | `completed+failed==N` holds; un-batchable prompt → FAILED |
| 2 | MED | Serving | SV4 `roofline_utilization` divided the device peak by `tp` a 2nd time, but `summary_table` is already per-device → MFU/MBU understated by `tp×` | `num_devices=1` in the sim loop AND the router | per-device MFU == `tp×` the old buggy value |
| 3 | CRIT | BudEvolve | sandbox copied all builtins (`open`/`eval`/`exec` + `__subclasses__` gadget live) → arbitrary file I/O + RCE from an evolved candidate | minimal-builtins whitelist + guarded `getattr` (blocks dunder names) + AST validator rejects forbidden calls/dunder attrs | `open()`/gadget rejected at validation AND blocked at runtime (no file written) |
| 4 | HIGH | BudEvolve | timed-out candidate leaked a GIL-spinning daemon thread forever | execute the candidate in a **killable subprocess** (terminate on timeout) | infinite-loop candidate terminated at 2s, **0 leaked threads** |
| 5 | MED | BudEvolve | fitness gameable: `[queue[0]]*100000` → E=1.0 (re-introduces R2-BE1's defect) | dedup batch by identity + enforce evict count | duplicate-exploit E≈0.07, not 1.0 |
| 6 | HIGH | CPU | `bits` ignored on the cpu_aware path (preset default bf16 always used) → int8/fp32 byte-identical to bf16; **and** int8 decode fell to SCALAR ISA (17× too slow) | pass `bits` into `create_cpu_system`; add `int8` to the decode vector-ISA selection | int8 decode = **½** bf16, fp32 = **2×** bf16 (exact byte-ratio) |
| 7 | HIGH | Training | activation memory sharded by TP but **not PP** → per-GPU over-stated for PP>1, disagreed with the cluster selector | divide activation by `pipeline_parallel` (1F1B: bottleneck stage ≈ full/PP) | activation ÷PP; calculator now consistent with the selector |

**Deferred / documented (not bugs):** CPU `_is_decode_op` relies on GenZ's decode dim layout — correct for
every op the engine emits (documented in-code); batched-decode single-socket BW is the §7.4-deferred NUMA
item; the disagg `B_eff` legitimately raises KV-transfer-link queuing (TTFT rises with concurrency — correct
physics, not a cancellation bug); BudEvolve hardware-DSE `power` objective value/field mismatch is cosmetic
(monotone, same argmin). Live API re-test: 25/25 effective (training ÷64, optimizer 404, batch-varying power,
cluster `B_eff` 212 rps/instance, CPU serving end-to-end, honest upstream-error on unknown model).

---

## ✅ REMAINING-ITEMS PASS (2026-06-18) — RCA + full fix + adversarial verify of the deferred items

The previously-deferred items were taken to full fix via a dynamic workflow (RCA → fix → independent
adversarial verify). **All three verified `fixed`:**

1. **CPU ISA + NUMA + x86 eta** (R2-CPU4/CPU5 + eta): `isa_model.py` now selects **AVX512 for bf16**
   prefill (was excluded → fell to AVX2 at half-rate; AVX512-BF16 VDPBF16PS, live 1.83× speedup) and
   carries **bf16 vector widths for ARM SVE/NEON** (was scalar-1). `cpu_operator._is_decode_op` is now
   **batch-aware**: batch=1 → per-socket bandwidth (single NUMA stream), batched → aggregate (verified:
   8×batch-1 211ms > batch-8 140ms). CPU memory efficiency is **vendor-aware** (`utils._default_eta_mem_cpu`):
   x86 Intel/AMD = 0.80 (published Dell/AMD STREAM-TRIAD 0.80–0.86 of peak), ARM = 0.65, ARM+LPDDR5X = 0.73 —
   GPU bands byte-identical. 17 cpu tests pass.
2. **Serving power per-device TDP** (R2-SV4/M14): `power_model.py` resolves TDP `cost.tdp_watts → tdp_watts
   → Power → class-derive` instead of a flat 300W. Power now spans **105W–630W** across 14 devices (was a
   device-blind 210W active). All 12 explicit TDPs are datasheet-sourced; Trainium1/Inferentia2 use a
   documented 350W HBM-accelerator class derivation (AWS publishes no per-chip TDP). 55 power tests pass.
3. **BudEvolve pymoo** (R2-BE4): `pymoo>=0.6.0` promoted to the engine's **core** dependencies (was an
   optional extra → silent grid fallback). NSGA-II now runs by default (non-degenerate Pareto front,
   distinct cost/power/throughput); grid fallback retained + disclosed. 86 budevolve tests pass.

**Residuals (flagged, not silently dropped):** (a) Trainium/Inferentia TDP is a derived class-band, not a
vendor datasheet figure; (b) `hardware_explorer.py:283` wraps the whole NSGA-II body in
`except ImportError`, so a deep ImportError could be misattributed as "pymoo absent" (latent masking, not
triggered today); (c) **35 DB-only catalog entries use display-name strings that the hardware resolver
doesn't match** (F2/C5 name-resolution area, pre-existing) — they resolve by canonical key but not by
display name. These are noted for a follow-on, separate from the CPU/power/pymoo scope.

## 📦 Packaging & automation (no more missing deps; one-command setup/start)

- **`BudSimulator/requirements.txt`** + **`llm-memory-calculator/pyproject.toml`** now list every imported
  third-party dep that was silently missing: `scipy`, `PyYAML` (both module-level — a fresh install used to
  crash), `packaging`, `pymoo`, `slowapi`, `openai`; optional/private extras (`openevolve`, `bud_models`)
  are noted but not hard deps. (`flask` in `src/api/hardware_routes.py` is dead — imported nowhere — left out.)
- **`setup.sh`** (new, portable, idempotent): picks Python ≥3.9, creates `.venv`, installs the **core engine
  editable** (the gap the old `setup.py` missed), backend requirements, BudSimulator, runs npm install +
  DB init, and a **fail-loud import-verify gate**. `./setup.sh [--no-frontend] [--python pythonX.Y]`.
- **`start.sh`** (new, portable): frees the ports, launches backend + frontend, **polls `/api/health`
  (no fixed sleeps)**, prints URLs, and cleanly tears both down on Ctrl+C. `./start.sh [--backend-only]`.
  Replaces the old macOS-hardcoded `start_app.sh` / `start_servers.sh`.

---

## 0. Live-state corrections (load-bearing — verified against `fix/accuracy-remediation`)

Several Round-2 findings cite line numbers from an **older tree**. Re-verified live before planning:

- `hardware_optimizer._get_hardware_types()` **already** returns `_GPU`-suffixed resolvable names —
  **R2-OP1 is already landed.** Do not re-fix.
- `serving.py` `simulate/serving`, `power/estimate`, `memory/tiers` **already** have up-front 404 hardware
  guards (Round-1 H2 landed there). The **`optimize/config|pareto|sensitivity`** endpoints do **not** —
  that is the residual gap (§Optimizers RC-A / D).
- `budevolve/evaluator.py` **already** catches GenZ exceptions into `EvalResult(feasible=False)` — the
  R2-OP2 `throughput=2000` fabrication is **gone**. The residual defect is re-scoped (silent-empty success,
  not fabrication).
- `batch_scheduler.py` reads **only** `result.Latency`; `is_offload`/`model_df` are discarded (ROOT-A).
- CPU `tile_k` `KeyError` confirmed live; CPU BW divergence confirmed (716.8 aggregate vs 358.4 per-socket
  vs 180 in configs.py).

---

## 1. Shared root causes (fix once)

| Root | Round-1 instance (landed) | Round-2 instance (this plan) | Consolidating fix |
|---|---|---|---|
| **A. Multiple sources of truth** | GPU FLOPS triple → F1/F2/F4 | CPU BW ×4 sources (R2-CPU3); serving overlay re-derives engine output (R2-SV*); 3 training-memory paths (R2-TR*) | Extend the single-source resolver to CPU + serving; route training-memory APIs onto the correct GenZ `_calculate_training_memory` |
| **B. Cache-sim vs roofline** | dead GPU TC-efficiency double-correction (F3/C4) | CPU cache-cycle term wins `max()`, unseeded RNG (R2-CPU2) | Delete the second timing path; keep only `max(compute, memory)` |
| **C. Fabricated fallback hides failure** | serving 404 + power honest-fail (H2) | same magic heuristic copy-pasted into 5 serving sites + optimizer (R2-SV/OP) | One honest-failure policy at the engine/serving boundary |
| **D. Metric decoupled from the searched variable** | — | v2 endpoints silent-empty on unknown hw; cluster batch=1 serial throughput; BudEvolve candidate never executed; constant cost/power objectives | Up-front 404 guard; memory-budget `B_eff`; execute the candidate; sweep cost/power as decision variables |

---

## 2. Unit: CPU inference modeling

*(R2-CPU1 AMX crash · R2-CPU2 non-deterministic 10×-slow decode · R2-CPU3 four-source BW divergence)*

### Root causes
1. **Stale-key typo (R2-CPU1):** `isa_model.py:239` reads `constraints['tile_k']`; the dict defines only
   `tile_k_bf16=32` / `tile_k_int8=64`. Crashes every AMX chip on prefill. Reproduced:
   `cpu_aware_prefill_moddeling(... 'intel_xeon_8592_plus')` → `KeyError('tile_k')`.
2. **Cache-sim cycle term is a fictional second timing model winning `max()` (R2-CPU2):**
   `get_cpu_memory_time` takes `max(total_cycles/freq, bandwidth_time)`; `total_cycles` comes from an
   **unseeded** per-line cache simulator (`np.random.randint`, sample caps, flat +200-cycle DRAM penalty).
   For 7B decode it yields ~1181 ms (0.85 tok/s) and beats the real roofline — and is non-deterministic
   (live: 0.8471 vs 0.8468 tok/s on identical calls).
3. **Four disjoint CPU BW/FLOPS sources spanning 4× + wrong NUMA aggregation (R2-CPU3):**
   `SapphireRapids_CPU`=180 (configs.py) / `8480PLUS`=307.2 / `8592PLUS`=358.4 per-socket (cpu_specs.py) /
   `intel_xeon_8592_plus`=716.8 2-socket aggregate (cpu_configs.py). The `cpu_aware_*` path feeds the 716.8
   aggregate to the GPU roofline as if a single batch-1 decode stream saw both sockets. **Physics:** a
   batch=1 decode reads the whole weight tensor once per token; with default first-touch NUMA the weights
   live in ONE node, so a single stream is bound by ONE socket (358.4), not 2×. Closed form
   `TPOT = W_bytes / (per_socket_BW · eta_ddr5)`.

### Fixes
- **R2-CPU1:** select `tile_k = constraints['tile_k_int8'] if data_type=='int8' else constraints['tile_k_bf16']`
  (both keys exist, Intel ISA manual: BF16 K=32, INT8 K=64). No new constant.
- **R2-CPU2:** `get_cpu_memory_time` returns ONLY the bandwidth roofline; cache-cycle term removed from
  timing. `simulate_data_access` kept for informational hit-rate columns only and **seeded**
  (`np.random.default_rng` keyed on op dims). **Delete** the dead `framework_overhead_ms` field + its 6
  preset assignments (verified never read in any timing path; was an unsourced "initial diagnosis" 1250/1038ms).
- **R2-CPU3:** consolidate to one physically-correct rule, no magic derate. CPUSystem exposes
  `per_socket_mem_bw` (decode, M==1 ops) and `aggregate_mem_bw = per_socket·sockets` (prefill/batched).
  Reconcile the other 3 sources to the datasheet per-socket value (configs.py `SapphireRapids_CPU` 180→307.2,
  `EmeraldRapids_CPU` 350→358.4, Flops to the cpu_specs.py AMX peaks). The "NUMA derate" = 1/sockets is a
  **derivation from first-touch placement**, not a tuned number. Reuse the **landed** DDR5 eta 0.65 (R2-G2).

### Grounded constants
- AMX K=32/64: Intel ISA Extensions Manual (already in code).
- DDR5 per-channel BW (JEDEC): 4800→38.4, 5600→44.8, 6400→51.2 GB/s/channel; channel counts from Intel ARK /
  AMD EPYC datasheets → 8592+ 358.4, 8480+ 307.2, Genoa 460.8 GB/s per socket.
- `eta_ddr5 = 0.65`: landed Round-1 DDR5 streaming-MBU band (`utils.py:27`).
- NUMA single-stream factor = 1/sockets: derived (first-touch placement).

### Net effect & regression
- AMX prefill no longer crashes; decode becomes **deterministic**; 7B bf16 decode converges across all three
  CPU paths onto `W/(per_socket_BW·0.65)` ≈ 17 tok/s (8592+), 15 (8480+), 22 (Genoa) — `cpu_aware` DOWN ~2.6×
  (44.5→17), serving-path UP ~1.8× (9.3→17), cache-sim path UP ~20× (0.85→17).
- **Byte-identical:** GPU roofline untouched; `get_cpu_roofline`'s `max(compute,memory)` source line
  preserved (`test_cpu_roofline.py:116-125` stays green — the change is confined to what `memory_time`
  *contains*). **Golden tests:** `test_amx_prefill_no_keyerror` (bf16+int8), `test_decode_deterministic`,
  `test_decode_is_bandwidth_bound` (7B/8592+ ∈ [12,22]), `test_cpu_decode_path_convergence` (within 25%),
  `test_cpu_bw_single_source` (per_socket == dram_bw·channels).

---

## 3. Unit: Training simulation

*(R2-TR1 PP/EP-blind weight sharding · R2-TR2 rank-independent GaLore · R2-TR3 cluster-selector magic fudge ·
R2-TR4 two divergent optimizer tables · R2-TR5 three divergent memory paths)*

### Scope correction (load-bearing — verified by grep)
The user-facing path is **`TrainingMemoryCalculator.calculate_training_memory` (`training/calculator.py:84`)**,
`TrainingClusterSelector`, and `TrainingTimeEstimator`. `BudSimulator/apis` has **zero** references to
`calculate_distributed_memory` / `calculate_advanced_training_memory` / `recommend_distributed_strategy` —
so the original R2-TR1/R2-TR3 attribution to `distributed.py`/`advanced_calculator.py` was **wrong** (that
code is dead to the API). The defects live in `calculator.py` and `cluster_selector.py`.

### Root cause
One physical quantity — per-GPU training memory + per-param optimizer state — is computed by **three** memory
functions and read from **two** optimizer-byte tables, and the API uses the **least-shared** path:
1. **API path (`calculator.py`):** signature lacks `pipeline_parallel`/`expert_parallel`/numeric `zero_stage`
   → PP>1 and EP>1 are **inexpressible**; weights never divided by PP/EP. `_calculate_optimizer_memory`
   reads `total_bytes_per_param` directly, **bypassing** the rank-aware `OptimizerConfig.calculate_memory_gb`
   → GaLore rank 16 == rank 256. Full-FT grad+opt under pure TP are over-counted (not divided by TP).
2. **GenZ path (`training_modeling.py:4190`):** parallelism sharding is **correct** (TP·PP·EP + ZeRO/DP) but
   reads optimizer bytes from the **inferior** `OPTIMIZER_STATE_BYTES` (sgd=0, adafactor=4, galore=2 flat).
3. **Advanced/distributed path:** test-only, dead to the API.
4. **Cluster selector:** re-shards `total_memory_gb/(tp·pp·dp)` × a magic **0.7** (and `_calculate_min_gpus`
   × **0.6**) — unsourced fudge factors approximating ZeRO-2, a second source of truth that disagrees with
   `calculator.py`.

### Consolidation direction (rebuttal — important)
Consolidate **onto the sourced `OPTIMIZER_CONFIGS`** (`optimizers.py:102`: adamw=8, sgd=4 momentum, adafactor=1
factorized, rank-aware GaLore) and **delete the unsourced `OPTIMIZER_STATE_BYTES`** — *not* the reverse. So
the feared regressions (sgd 4→0, adafactor 1→4, galore flat) **cannot occur**; every optimizer moves toward
its sourced value.

### Fixes
| ID | Fix | File | Default behavior |
|----|-----|------|------------------|
| **TR1** | Add `pipeline_parallel`/`expert_parallel` to the API signature; `weight /= TP·PP·EP`; grad+opt `/= TP` only when `method=='full'` (mirrors `training_modeling.py:4237-4242`) | `calculator.py:298-406`, `types.py`, `training.py:30/321` | **byte-identical** (÷1; full-FT-TP branch skipped) |
| **TR2** | Route LOW_RANK optimizers through `OptimizerConfig.calculate_memory_gb(trainable, rank)` so GaLore/APOLLO scale by `rank/default_rank=16` (GaLore paper, Zhao et al. 2024) | `calculator.py:372-400` | **byte-identical at rank 16** |
| **TR3** | Selector consumes the **component breakdown** already on the estimate dict (`weight/gradient/optimizer/activation_memory_gb`) and re-derives per-GPU footprint with the **same** sharding contract; drop the 0.7/0.6 fudge. Fallback to no-0.7 `total/(tp·pp·dp)` when the breakdown is absent | `cluster_selector.py:101,269-274,307-314` | fallback preserves fixture tests |
| **TR4** | GenZ path reads `get_optimizer_config(opt).total_bytes_per_param` (rank-aware for LOW_RANK); delete `OPTIMIZER_STATE_BYTES` (only line 4257 + tests reference it) | `training_modeling.py:816,4257` | sgd 0→4, adafactor 4→1 (toward sourced) |
| **TR5** | Delegate `advanced_calculator.py`/`distributed.py` per-component math to the shared corrected helpers (non-behavioral; API never calls them) — collapses the 3-source root | `advanced_calculator.py`, `distributed.py` | API unaffected; advanced tests are directional |

### Grounded constants
- `TP·PP·EP` weight divisor: derived (Megatron 3D parallelism; matches `training_modeling.py:4237-4242`).
- Grad/optimizer fp32 = 4 B/param (IEEE-754). Full-FT TP-division of grad+opt: derived (DeepSpeed/Megatron
  co-locate states with the sharded weight slice; LoRA adapters are TP-replicated → not divided).
- GaLore rank-scaling: `effective_params = trainable × (rank/16)`, capped at 100% (`optimizers.py:74`).
- `OPTIMIZER_CONFIGS`: adamw=8, sgd=4, adafactor=1.0, 8-bit-adam=2 — each cited in the dataclass.
- 0.9 memory safety margin (`cluster_selector.py:245`) **retained** (documented 10% fragmentation headroom).

### Net effect & regression
- **Byte-identical:** every default API request (`method='lora'`, TP=DP=1, PP=EP=1, no ZeRO) end-to-end; the
  GenZ parallelism math (only its optimizer-byte *source* changes); non-LOW_RANK optimizers and GaLore at
  rank 16.
- **Intentional changes:** PP/EP>1 weight memory ÷ PP·EP (e.g. Mixtral EP=8 → /8); full-FT pure-TP grad+opt
  ÷ TP; GaLore rank>16 rises; selector configs that only fit via the 0.7 fudge now need more GPUs; GenZ
  sgd 0→4, adafactor 4→1.
- **One enumerated at-risk golden:** `test_cluster_selector.py::test_suggests_min_gpus` — expected `min_gpus`
  updated to the no-fudge physics (310GB / 72GB-usable → 8, not 4). All other selector/optimizer goldens stay
  green (their fixtures lack the component breakdown → no-0.7 fallback). New goldens: Mixtral EP ratio == 64,
  GaLore rank256 > rank16, selector per-GPU == calculator within 1e-6, GenZ uses `OPTIMIZER_CONFIGS`.

---

## 4. Unit: Serving simulator (continuous-batching correctness)

*(SV1/SV2 completion & admission · SV3 silent offload · SV4 fabricated/batch-invariant power · SV5 fake
prefix-cache · SV6 batch=1 serial cluster throughput)*

### Root causes
The serving overlay is a **second source of truth** that discards or re-fabricates the engine's
authoritative signals.
- **ROOT-A — discarded engine signals:** `ModdelingOutput` carries `is_offload`, `model_df`,
  `summary_table`; `batch_scheduler.py:202-305` reads **only** `result.Latency`.
- **ROOT-B — fabricated fallbacks duplicated:** the same heuristic (`input_tokens*0.01`, `0.5/tp`,
  `131072 bytes/token`, `getattr(...,0.5)`) is copy-pasted into 5 sites; Round-1's "fail honestly" hit
  only the router.
- **ROOT-C — batch=1 serial throughput:** `cluster.py:38` / `disaggregation.py:46-54` model an instance as
  `1000/time_per_request` (one request at a time).

### Fixes
- **SV3 (RCA corrected):** GenZ OFFLOAD does **not** raise — it sets `is_offloaded=True` and returns a
  *larger valid latency* (`llm_decode.py:78-82`); the bare-except only fires on the hard `ValueError`. True
  defect: the scheduler never reads `result.is_offload`. **Fix:** propagate `is_offload` through the latency
  cache to `ServingSimulationResult.offloaded/offload_steps`. No latency number changes; offloading is now
  flagged. Pure signal propagation.
- **SV4 (two roots separated):** router `getattr(prefill_result,'compute_utilization',0.5)` reads a
  **non-existent** attribute → always 0.5 → bs-invariant power (the symptom); sim loop uses a scheduling
  **occupancy** proxy (different defect). **Fix:** **build** MFU/MBU from `result.model_df`
  (`Num ops`→MACs, `Total Data`→bytes, `Latency`) against the **unified** peaks
  `get_hardware_config(hw)['Flops']*1e12` / `['Memory_BW']*1e9` (present for all 92 DB devices). New
  `roofline_utilization(result, hardware) → (mfu, mbu)`. Remove the 0.5 default and 0.8 occupancy factor.
  Grounded in GB10 measured linear power-vs-util (`audit_microbench_data.md:108-112`).
- **SV5:** `prefix_cache.py:220-224` hashes `(model, position)` only → every same-model request gets the
  identical token stream → ~100% hit (mislabeled "20-40%"). **Fix:** delete that branch; always use the
  `request_id`-salted construction. Default `shared_prefix_tokens=0` → ~0% overlap; any nonzero overlap is a
  **caller-supplied** measured system-prompt length, not a substituted constant.
- **SV6:** `B_eff` is **not** a free knob — it is the steady-state concurrency the memory budget already
  pins: `floor((tier_capacity − weight_bytes) / (bytes_per_token_kv × context))`, capped by
  `max_batch_size`, using the canonical `bytes_per_token_kv` (`memory_model.py:81`). Prefill vs decode differ
  by context but **cancel** in the disagg inequality tests (same `B_eff` both pools). Replace the 3 magic
  fallbacks with honest failure / derived footprint. `B_eff` defaults to 1 when the MemoryModel is
  unbuildable (preserves mocked tests).
- **SV1+SV2 (multi-tier corrected):** completion is already idempotent. The real gap: a request whose KV
  exceeds **all** tiers combined stays forever in `_pending`. `allocate_kv_blocks` is already the
  authoritative multi-tier budget (HBM→DRAM→HOST_DDR→CXL→NVME), so only a request larger than the summed
  capacity is `set_failed` (not appended to `raw_requests`, preserving `test_simulator.py:110`). The existing
  evict→QUEUED path is untouched (no `_preempted` queue).

### Net effect & regression
- On-chip-fitting workloads: **byte-identical** latency/throughput (SV1/SV2 add accounting; SV3 adds a flag).
- Cluster/disagg throughput **↑ by `B_eff`** (tens-to-hundreds×) toward realistic continuous-batching — the
  single largest correction, now memory-budget-bounded. Power on power-tracking runs shifts O(10-40%) and
  now varies with batch size / tracks the real roofline. Prefix-cache hit-rate **↓ 0.99→~0** for independent
  prompts. Offloaded/failed batches now surfaced.
- **Byte-identical guards:** on-chip latency math; `PowerConfig` fraction math; RadixCache mechanics;
  `small_workload` fully fits; disagg inequalities cancel; tps derived in-expression; mocked tests get
  `B_eff=1`. Golden tests pin every corrected value.

---

## 5. Unit: Optimizers (parallelism + config + v2 endpoints)

*(R2-OP-A fabricated-fallback + missing 404 · R2-OP-B mis-bounded parallelism search · R2-OP-C E2E latency /
throughput label / dedup / categorical Morris)*

### Root causes & fixes
- **RC-A (fabricated fallback, API-reachable):** `_evaluate_config` (config_optimizer.py:338-343) swallows
  engine exceptions and substitutes `prefill=input_tokens*0.01*bs/tp`, `decode=0.5*bs/tp`,
  `throughput=1000*bs/decode` with `feasible=True`; the three v2 optimizer endpoints skip the up-front 404
  guard the other endpoints already have. **Fix:** on engine exception set `feasible=False` + zeroed metrics
  + `error`; `_check_feasibility` returns False when the engine failed; delegate the OOM verdict to the
  engine's existing raise (single source); add a shared `_require_known_hardware` helper (reusing the Round-1
  resolver) to all three endpoints + `except HTTPException: raise`. **Missed-regression fix:**
  `analyze_sensitivity` must `continue` past infeasible points so the score-0 floor doesn't manufacture
  spurious Morris jumps.
- **RC-B (mis-bounded search, API-dead):** `get_various_parallization` uses `TP*PP < total_nodes`, never a
  full-cluster split. **Verified API-dead** (zero call sites in BudSimulator; production uses
  `_generate_parallelism_strategies`, which already enforces `tp*pp==num_nodes`). Fix the math at source
  (`TP*PP == total_nodes`, +EP for MoE) to remove the latent foot-gun; net effect scoped to direct-import /
  `platform_size` / docs callers, with a zero-diff assertion on the v2 harness.
- **RC-C (E2E latency / labels / dedup / Morris):**
  - **C1:** latency objective scores `e2e = prefill + tpot·output_tokens` in both `_compute_score` and
    `_compute_pareto_front` (was `prefill + per-token TPOT`, understating E2E by ~127× for output_tokens=128).
  - **C2:** `throughput_rps = token_tps / output_tokens` (true request-completion rate) so it is no longer
    byte-identical to `token_throughput_tps`.
  - **C3:** dedupe precisions by the `(compute_multiplier, mem_multiplier)` signature from `system.py` so
    fp8/int8 are evaluated once (frees budget, fixes Morris double-count).
  - **C4:** replace the ordinal Morris `delta=j-i` for categorical `precision` with a range-based effect
    `max(scores)−min(scores)` (mu=sigma=0), per standard Morris treatment of qualitative factors.

### Net effect & regression
- Unknown hw on the 3 v2 endpoints → **404** (was 200 with fabricated/empty); failed/OOM configs become
  `feasible=False` and excluded from best/Pareto/Morris. Latency reports true E2E; `throughput_rps` becomes a
  genuine request rate; fp8/int8 collapse to one row; Morris precision becomes a meaningful range.
- **Byte-identical:** all-feasible A100/Llama-3.1-8B grid → best_config/best_score/feasible_count unchanged;
  SearchSpace defaults unchanged; `TestOptimizeNoFeasible` unchanged; v2 harness zero-diff for RC-B.
  `TestParetoDominance` updated to E2E (deliberate test change). New goldens: per-endpoint 404,
  failed-config-infeasible, full-cluster-utilization, E2E relation, request-rps relation, precision dedup,
  categorical Morris.

---

## 6. Unit: BudEvolve (algorithm evolution + hardware DSE)

*(R2-BE1 candidate-independent fitness · R2-BE2 = R2-G1 [already fixed] · R2-BE3 constant cost/power ·
R2-BE5 mislabeled throughput · improvement_pct scale mismatch)*

### Meta-root
**The optimized/reported metric is decoupled from the variable being searched.** The evolved candidate never
executes (BE1); cost/power objectives are constant dataclass defaults (BE3); request throughput is mislabeled
decode-token throughput (BE5); the OpenEvolve `improvement_pct` mixes two scales.

### Fixes
- **R2-BE1:** the candidate (a pure function) is never executed — fitness varies only with AST shape
  (verified: two distinct candidates → identical `throughput_rps=100`/`combined=80.1`). `BatchScheduler` has
  **no injection point**. **Fix:** execute the candidate in a sandboxed thread (timeout 2.0s, derived bound)
  over the single-source synthetic workload (`WorkloadConfig`, seed=42) and fold an **executed
  policy-efficiency factor `E∈[ε,1]`** into the score (`perf = genz_ceiling × slo × E`). Per-family contracts
  (scheduler/cpu_scheduler → batch-fill efficiency; cache_policy → retained-hot fraction). Result gains
  `fitness_basis` so the UI never implies full-lifecycle execution. ε floor keeps the 4 mocked tests green.
- **R2-BE2 — REBUTTED (already fixed by R2-G1):** `evaluate_hardware` passes a `System` object into
  `prefill/decode_moddeling`; the R2-G1 branch at `utils.py:239-261` assigns the grounded GPU bands
  (0.75/0.80) when eta==1. Verified empirically (A100-class 2507 tok/s vs B200-class 12353 tok/s — only
  possible with eta<1). No BudEvolve-local change; add a guard test.
- **R2-BE3:** `hardware_explorer` builds `HardwareSpec` without `tdp_watts`/`estimated_cost_usd` → constant
  defaults (700/25000) → degenerate cost/power objectives. A 3-coef fit to 4-5 collinear anchors is
  ill-conditioned. **Fix:** promote `tdp_watts`/`estimated_cost_usd` to **swept decision variables**
  (co-design), bounds from the in-repo `_REAL_HARDWARE_SPECS` anchor table (tdp [400,1000] W, cost
  [10000,40000] USD), plus one monotone feasibility ceiling (`flops/tdp > 2.25 TFLOPS/W` rejected, B200
  anchor). Zero fitted coefficients. CPU targets leave cost/power `None` (honest "not modeled").
- **R2-BE5:** `Throughput == Throughput_tokens_per_sec` (llm_decode.py:181-183) → `throughput_rps == tps`.
  **Fix at the evaluator boundary** (engine byte-identical): `throughput_rps = token_throughput_tps /
  max(output_tokens,1)`.
- **improvement_pct (OpenEvolve branch only):** `(best_score − baseline.throughput_rps)/...` mixes
  combined_score with raw rps. **Fix:** evaluate the baseline through the same bridge and compute
  `(best_score − baseline_combined)/baseline_combined·100`.

### Net effect & regression
- Evolution fitness becomes candidate-dependent (removes a systematic false-positive "evolution works"
  signal); cost/power objectives vary over the real anchor band; `throughput_rps` becomes true request/s
  (~output_tokens× smaller); `improvement_pct` becomes same-scale.
- **Byte-identical:** no engine numbers change for named hardware; no unmocked GenZ/ServingSimulator call
  enters a default test; ε floor keeps `combined_score>0`. Re-baseline `test_bounds` n_var 7→9. New goldens
  for candidate-dependence, cost/power variation, perf/W rejection, rps=tps/genlen, OpenEvolve same-scale.

---

## 7. Cross-coupling synthesis (plan of record)

> _Spliced from the workflow's lead-architect synthesis (covers the interaction matrix, master regression
> suite, rollout order, and production-readiness checklist), with the Training unit's coupling folded in._

### 7.1 Interaction matrix (double-correction guards)

| Pair | Sign / magnitude | Type | Resolution |
|---|---|---|---|
| CPU cache-sim delete × landed DDR5 eta 0.65 | +~20× on 7B decode (0.85→17 tok/s) | Compound, both required | Deleting the cycle term exposes the roofline; the 0.65 band derates it. Without the band, ~26 tok/s (too fast). |
| CPU per-socket aggregation × cache-sim delete | cpu_aware −2.6×; serving +1.8×; cache-sim +20× | Convergence | All three CPU paths land on `W/(per_socket_BW·0.65)`. |
| Serving `is_offload` flag × Round-1 honest-fail | none numeric (flag) | Additive | Surfaces the silent larger offload latency the bare-except never caught. |
| Serving `B_eff` × cluster/disagg throughput | + tens-to-hundreds× | Largest correction | Memory-budget-bounded; cancels in disagg inequalities. |
| Serving power MFU/MBU from model_df × F3/C4 dead-code removal | power ±O(10-40%), bs-dependent | Coupled, re-snapshot | MFU/MBU must use the **post-F3** clean roofline (F3 already landed → safe). |
| v2 404 guard × Round-1 serving 404 guards | none | Consistency | Same `get_hardware_config or is_cpu_hardware` check; uniform across 4 endpoints. |
| Training selector × landed Round-1 hardware fix (65ffe10) | none | Single-source dependency | `cluster_selector` pulls `Memory_size` from the consolidated `HARDWARE_CONFIGS`, so a training fit-check and an inference lookup agree (H100=80GB). |
| Training optimizer-table consolidation (TR4) × GenZ `/estimate-time` | sgd 0→4, adafactor 4→1 | Source swap | Both training paths read one sourced table; the parallelism math is untouched. |

**Three guards to watch:** (1) do **not** add a CPU eta on top of the landed DDR5 0.65 — one band only;
(2) serving power MFU/MBU must use the post-F3 roofline (no residual 0.85); (3) `B_eff` divides the *memory*
budget only — never multiply the *latency* roofline by it.

### 7.2 Master regression suite

Round-1 stays green: **785-pass suite + 58/58 API harness** + its invariants (decode bs=1 byte-identical;
MFU/MBU ≤ 1.0 on all 84 devices; memory-param families byte-identical; single-node collectives
byte-identical; GB10 keystone ±7% decode / ±10% prefill). New Round-2 invariants:

- **CPU:** 7B bf16 decode matches `W/(per_socket_BW·0.65)` within 10% (quantized scales by byte ratio);
  decode deterministic; AMX prefill does not raise; the three CPU paths agree within 15%.
- **Serving:** completions ≤ submitted count; offload flagged not silent; cluster throughput bounded by
  `B_eff·single_stream` (orders of magnitude above the serial artifact); prefix-cache hit-rate → ~0 for
  independent prompts; `optimize/*` returns 404 for fake hw and honest empty-front for OOM hw.
- **Training:** default LoRA request byte-identical end-to-end; Mixtral EP=8 weight memory = 1/8 of EP=1;
  GaLore rank-256 > rank-16 (== at 16); cluster-selector per-GPU footprint == `calculator.py` breakdown
  within 1e-6 (single-source proof); GenZ optimizer bytes read `OPTIMIZER_CONFIGS`. One enumerated golden
  update: `test_suggests_min_gpus` → no-fudge physics (310GB → 8 GPUs).

### 7.3 Rollout order (dependency-correct, byte-identical guard per step)

0. **Arm new golden tests first** (record today's failing behavior as the merge gate). No production change.
1. **CPU `tile_k` typo** — isolated, unlocks AMX prefill.
2. **CPU single-source BW + per-socket aggregation** — reuse landed DDR5 eta 0.65.
3. **CPU cache-sim delete** — lands the CPU band + determinism (must ship with Step 2).
4. **Serving honest-failure consolidation** — one `_get_latencies` helper that raises; remove 5 magic fallbacks.
5. **Serving `is_offload` flag + offload accounting** — adds fields only.
6. **Serving `B_eff` throughput** — largest intended change; stage + snapshot carefully.
7. **Serving prefix-cache fix + power MFU/MBU from model_df** — power uses post-F3 roofline (safe).
8. **v2 endpoint 404 guard** — 3-line guard; makes 4 endpoints uniform.

**Training track (independent of the engine roofline; can land in parallel):**

9. **TR1+TR2 — API path sharding + rank-aware GaLore** (`calculator.py`): add PP/EP + full-FT TP grad/opt
   division; route LOW_RANK through `calculate_memory_gb`. Byte-identical for default LoRA/TP=1/rank=16.
10. **TR4 — consolidate optimizer table** (`training_modeling.py`): GenZ path reads `OPTIMIZER_CONFIGS`;
    delete `OPTIMIZER_STATE_BYTES`. Update any GenZ-direct optimizer-byte literal test.
11. **TR3 — selector consumes component breakdown** (`cluster_selector.py`): drop 0.7/0.6 fudge; no-0.7
    fallback when breakdown absent. Update `test_suggests_min_gpus` to the no-fudge min.
12. **TR5 — delegate advanced/distributed to shared helpers** (non-behavioral; collapses the 3-source root).

### 7.4 Production-readiness / integration checklist

**Fully functional end-to-end after Round-2:** CPU AMX chips no longer 500; CPU decode in the published band
and deterministic (stable rankings for the recommender); serving returns realistic continuous-batching
throughput, flags offload, honest prefix-cache; `optimize/*` returns 404 for unknown hw.

**Residual data to source (gates accuracy, not correctness):** x86 STREAM per-socket effective BW (to set the
x86 analog of `eta_ddr5` — 0.65 is the honest interim); per-device TDP / P_idle for serving power on DB-only
devices; `pymoo` install (else documented coarse-grid fallback).

**Genuinely structural / staged:** the `B_eff` continuous-batching model (Step 6); SV1 multiple-completion +
SV2 admission as a dedicated serving-scheduler unit if out of scope; CPU NUMA remote-access penalty for
batched/prefill; ARM SVE/NEON + AVX512 bf16 ISA selection.

**Anti-gaming guarantee:** every Round-2 number is a measured ratio, a datasheet value, a published band, a
closed-form derivation, or the engine's own output. The DDR5 eta is **reused** from Round-1, not re-invented.
No new tuned per-case constants.
