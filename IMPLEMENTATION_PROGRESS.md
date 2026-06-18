# Accuracy-remediation implementation progress

Branch: `fix/accuracy-remediation`. Plan: `solutions.md`. Grounding: `audit_microbench_data.md`.
Regression suite (extreme-TDD safety net): `llm-memory-calculator/tests/test_accuracy_remediation.py`
(physical invariants + byte-identical decode/param baselines). Run:
`.venv/bin/python -m pytest llm-memory-calculator/tests/test_accuracy_remediation.py -q -o addopts=""`

Rollout order (solutions.md §4): Step0 tests → F1 → C(causal) → C4(dead-code)+refit → F2 → F4 → calibration → power/UI.

**STATUS: 17 fixes implemented, all regression-clean.** llm-memory-calculator full regression held at
**721 passed / 0 failed** through F1, H0/M4, H1, C/M4b, C2, C4+C3, L0/L2/M2 (each re-run). Remediation
suite: 40 pass, 0 fail. API harness: **54/58 pass** (was 49; H2/M5/M7 fixed; 4 residual = harness
artifacts + L4 low + unverified compare). Training cluster tests 21/21.

Done: Step0, F1/C1, H0/M4, H1, C/M4b, C2, C4+C3, L0, L2, M2, H2, H3, H4, M5, M6, M7, M12.
Remaining: F2/F4 (hardware single-source + C5/C6/C7; risky RTX4090 inversion — DB needs F1 correction
first), comm E (H5/H6/M8/M9), frontend G (H7/H8/H9/M10/M11/L6), L1, L4, L5, M1.

## DONE (TDD green + full regression)
- **Step 0 — regression safety net** ✅ `test_accuracy_remediation.py`. Physical invariants (prefill
  MFU≤dense-peak, decode self-consistency eta_mem≤1, decode realism≤0.95), byte-identical decode +
  param-family goldens. Decode test corrected to use the model's OWN counted bytes (input-embedding is
  a gather, not streamed — params*2 over-counted). GB10 = calibrated positive control (65% MBU, passes).
- **F1 / C1 — dense FLOPS** ✅ `hardware/configs.py`: H100_GPU 1979→989.5, H100_PCIe 1513→756,
  H200 1979→989.5, GH200 1979→989.5 (datasheet dense bf16; sparse mislabeled). Engine reads static
  configs, so this corrects the engine. Prefill MFU invariant now GREEN (H100 162%→78.7%). Decode
  byte-identical preserved. Fixed 8 pre-existing tests that hardcoded the sparse values
  (`tests/unit/test_verified_fixes_core.py`, `tests/training/test_verified_fixes_training.py`) — their
  own docstrings said "989 dense", confirming the prior value was wrong. Full suite: 717 pass, 0 new breaks.
- **H0 / M4 — Gemma GeGLU param count** ✅ `parameter_counter.py`: added `_is_gated_ffn()` driven by
  `model_type` (gated_ffn_model_types set) + expanded activation spellings + reads Gemma's
  `hidden_activation` key. Removed `llama` from `always_tied_models` (Llama-2/3 don't tie; respect
  explicit flag). Gemma-2-9B 7.08→9.24B ✅; Gemma-2-2B, Phi-3-mini correct; 5 prior families unregressed.

- **H1 — public estimators accept string hardware** ✅ `performance_estimator.py`: added
  `_coerce_system_name` (None→A100 default, str→get_hardware_config dict, dict/System passthrough) +
  `_is_cpu_system` guard. str result byte-equivalent to dict; unknown name → clear ValueError. 3 sites.
  Tests: `test_accuracy_remediation.py::test_h1_*` green.
- **C / M4b — causal masking in prefill attention** ✅ (Step 2; design+review workflow wf_12f1cc01).
  Added `OpType.Logit_Causal_PREFILL=22 / Attend_Causal_PREFILL=23` (utils.py) mapped to 'Logit'/'Attend'
  in op_type_dicts (operator_base.py, byte-identical memory/strings). `_causal_fraction(M,N)=1-(M-1)/
  (2*max(N,M))` clamped (operators.py), gated in Logit/Attend.get_num_ops on dim[-1]. Re-tagged the 4
  causal-prefill builders (attention.py: mha prefill, prefill_local, mla prefill, chunked-prefill loop).
  VERIFIED: attn compute -50.0% exact; llama3_8b prefill@8k -10.5%; decode byte-identical; 9 operator
  tests green (`test_causal_masking.py`). Review-confirmed (chunked f exact; sp guarded; decode/bidir safe).
  NOTE: assumes decoder-only MHA prefill (codebase only builds decoder graphs); documented in enum.
  The full-square `Op Intensity` DISPLAY column is now stale for causal ops — resolved by C4 (Step 3)
  which removes get_shape_dependent_op_intensity. budevolve attention-FLOP fixtures may need re-pin.

- **C2 — decode realism via per-memory-tech eta_mem** ✅ (Step 6a, separable from C4). `utils.py`:
  `_ETA_MEM_BY_MEMORY_TYPE` (HBM3 0.85, HBM2e 0.80, LPDDR5X 0.73 [measured GB10], GDDR 0.75, DDR 0.62-0.65)
  applied as the default `memory_efficiency` in `get_inference_system` when no measured calibration block
  set eta_mem and meff==1. Grounded (documented memory-tech MBU + GB10 measurement), generalizable
  (keyed off `memory_type`), per-device measured calibration overrides. RESULT: H100 7B decode 252→214
  tok/s (85% MBU), A100 153→123 (80%), GB10 unchanged 13 (its 0.659 block). All 3 `decode_mbu_realistic`
  reds now GREEN; decode byte-identical baseline regenerated to calibrated values (intended Step-6 change).
  Full regression triage in progress (broad: changes decode for all devices). Remaining: eta_compute (C3)
  for prefill MFU is NOT yet applied (coupled with C4 dead-code removal — next).

- **C4 + C3 — remove dead tensor-core/shape code + grounded eta_compute** ✅ (Step 3, same change).
  `operator_base.py`: deleted `OPERATION_CHARACTERISTICS`, `get_tensor_core_efficiency`,
  `get_shape_dependent_op_intensity`, `get_tensor_core_efficiency_factor`, the flat-0.85 multiply, and
  the `TC Effcy` column (verified zero external consumers). `get_roofline` now: compute_time /=
  system.compute_efficiency (exact), op_intensity = num_ops/num_data. `utils.py`: added
  `_ETA_COMPUTE_BY_TENSOR_CORE_GEN` (gen5 0.82, gen4 0.80, gen3 0.78, CDNA3 0.78, CPU 0.65) applied as
  default ceff when no calibration + ceff==1 — grounded in published large-GEMM MFU, generalizable
  per-arch, calibration overrides. Net prefill: fake 0.85 → grounded ~0.80 (small, now principled).
  Tests: `test_c4_*` (dead code gone, C Effcy exact), `test_c3_prefill_eta_compute_grounded` green;
  decode byte-identical preserved. Full regression in progress (budevolve Op-Intensity may need re-pin).

- **L0 / L2 / M2 — engine nuances** ✅ (721-pass regression). L0: `system.py` tf32 compute_multiplier
  1→2 (A100 156 vs 312 TFLOPS datasheet; tf32 prefill now 2× bf16). L2: `utils.py` OFFLOAD_BW 128→64
  (PCIe5 x16 spec). M2: `llm_decode.py` removed the fixed 2-10-token KV bump; always uses the two-point
  KV-scaled average (5 tok on 32k ctx now ~0% growth, not fixed +5%). Tests `test_l0_*`/`test_m2_*` green.
- **API batch (F): H2, H3, H4, M5, M6** ✅ (verified live + no API regressions). H2: `serving.py`
  power/estimate fails honestly (400) instead of fabricating numbers for a bad model. H3: `usecases.py`
  uses `UniversalParameterCounter().count_parameters` (was a non-existent `calculate_parameters` →
  params=0 → dead CPU-exclusion). H4: `src/usecases.py` `_validate_usecase` guards None SLO bounds
  (null-SLO create now 200, was 500). M5: `models.py` popular limit `Query(10, ge=1, le=100)` (−5 → 422).
  M6: `serving.py` length clamp min≤mean≤max by construction (no degenerate collapse for small means).

- **M7 / M12 — training endpoints** ✅. M7: `training.py _get_model_config` now falls back to the GenZ
  registry (via `_genz_config_to_hf_dict`, num_decoder_layers→num_hidden_layers) so 'llama2_7b' resolves
  like serving; HF auth/network failures → 502, genuine not-found → 404 (was blanket 404). M12:
  `cluster_selector._find_best_parallelism` reports the minimal achievable per-GPU footprint on the
  not-fit path (was 0.0; now matches the prose `reason`).

- **Comm E (M9/H6/M8/H5)** ✅ (regression-clean). collective_times.py: M9 ceil node-count, H6 inter-node
  BW 0.4→0.11 (DGX IB:NVLink), M8 concurrent_ops 2→1 (no implicit DP overlap), H5 dead-code documented,
  + fixed a self-introduced `import math` shadow. Tests: test_comm_modeling.py 3/3.
- **L0/L1/M1/M2/L5/H9** ✅. L0 tf32 2× (test). L1 fp8 warning gated by arch (Hopper/Ada/Blackwell/CDNA3
  no-warn; Ampere warns). M1 PP throughput = steady-state (no fill/drain penalty; test identity). M2 decode
  KV two-point always. L5 System.kv_bits capability (default byte-identical; weight-only quant supported).
  H9 optimizer utilization computed from runtime breakdown (TP1=1.0/TP2=0.69/TP4=0.49, was flat 0.7).
- **F2/F4 + C5/C6/C7** ✅ (workflow-designed + adversarially reviewed; review caught H800→989.5 & Intel-Max
  defer). Migration BudSimulator/scripts/migrate_hardware_dense_bf16.py corrected 12 DB FLOPS rows to dense
  bf16 (idempotent). manager.py: STATIC-WINS merge (F1 values immune to DB drift — RTX4090 inversion guard)
  + env-var default-DB. apis/main.py startup points the engine at the DB. serving.py validations use the
  unified get_hardware_config (static+DB). Tests: test_hardware_source.py 18/18 — all 18 C5 GPU targets
  simulate, F1 statics intact, RTX4090 sentinel=330 (not sparse 661), DB-only devices dense. Intel Max /
  PonteVecchio bf16 deferred (no citable datasheet value; left real_values=0/estimated).
- **Frontend G (H7/H8/M10/M11/L6)** ✅ (agent-implemented, diffs reviewed; UI not live-testable — Chrome
  remote). H7 price index (not $/hr) + disclaimer; H8 mock compat table → live /api/models/calculate fetch;
  M10 list repointed to /filter (sort/flops honored); M11 real pagination; L6 param_count NaN guard.

- **L4** ✅ — `/api/models/calculate` gated/not-found message broadened (HF returns 401/403 for both;
  message now says "gated … or does not exist", not the misleading "gated" only). Harness accepts 403.

## FINAL STATUS — remediation complete & regression-clean
**llm-memory-calculator: 783 passed / 0 failed** (incl. all 4 new remediation suites + the 721 existing).
**BudSimulator/tests: 24/0.  API harness (`api_audit_tests.py`): 58/58.** Engine full-regression held at
720-783 pass / 0 fail across every step (decode byte-identical guards, budevolve unbroken). ~37 of 39
findings implemented with extreme TDD + workflow design/review for the coupled ones + every constant
datasheet/measured/closed-form (no magic numbers).

### The 2 NOT fully closed (documented, justified):
- **M14 (power realism)** — DEFERRED (the design synthesis itself scoped it out). Two roots: (1) DB-only
  devices lack `cost.tdp_watts` → PowerConfig defaults to 300 W (hence identical power across devices);
  (2) the power model's utilization isn't workload-proportional. Proper fix = per-device vendor idle/TDP
  sourcing + `P = idle + util·(TDP−idle)` driven by the roofline MFU/MBU. Separate effort; flagged in bugs.md.
- **Frontend live-verification** — BLOCKED: the connected Chrome runs on macOS and cannot reach this Linux
  box's localhost. The 5 frontend source fixes (G) are implemented + diff-reviewed but not click-tested.
- **C4 / L3 / M3 — remove dead tensor-core/shape-intensity code** (Step 3): delete
  `get_tensor_core_efficiency_factor`/`get_shape_dependent_op_intensity`/`get_tensor_core_efficiency`
  + the flat 0.85 multiply (operator_base.py); fold into eta_compute. Pin budevolve `Op Intensity`
  (Logit/Attend currently get constant 100 — NOT dead). Same-commit GB10 eta refit (0.096→0.082).
- **C2 / C3 — per-device efficiency calibration** (Step 6): currently 3 RED
  (`test_decode_mbu_realistic[llama2_7b-H100, llama2_7b-A100, llama3_8b-H100]`). Plan: memory-tech eta_mem
  (HBM3 ~0.8-0.9, HBM2e ~0.8, LPDDR5X 0.75 measured, GDDR ~0.75) + published/measured MFU, via the
  differential_evolution harness, AFTER structural fixes. tf32 multiplier 2 (L0). Power util-driven (M14).
- **F2 / F4 — single hardware source of truth**: engine resolve from DB (after F1); RTX4090 inversion
  guard (DB=661 sparse vs static=330). C5 (gate real_values=0 as estimated, don't 404), C6 (Intel Max
  dense), C7 (dedup).
- **E — comm modeling**: wire hierarchical AR; inter-node BW from IB spec not 0.4×NVLink; remove double
  de-rating; ceil node division.
- **F — API correctness**: H1 (str system_name), H2 (fabricated power), H3 (calculate_parameters missing),
  H4 (null-SLO 500), M5 (popular limit), M6 (serving clamp), M7 (training names), M12 (0.0 mem field), L4.
- **G — frontend**: H7 (price $/hr label), H8 (mock compat table), H9 (hardcoded util), M10 (filters),
  M11 (pagination), L6 (NaN guard).
- **H — engine nuances**: M1 (PP throughput), M2 (decode KV heuristic), L1 (fp8 warning gate),
  L2 (offload BW), L5 (weight vs KV precision).
