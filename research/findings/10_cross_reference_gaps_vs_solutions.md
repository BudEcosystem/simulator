# Cross-Reference Analysis: vllm-tuner Gaps vs Research-Backed Solutions

**Source findings:**
- Finding 01: Core source code analysis (9 source files in `src/`)
- Finding 02: Benchmarks, profiling, reporting, CLI analysis (15 files)
- Finding 03: SCOOT paper review and related work research
- Finding 04: Web research, best practices, production patterns
- Finding 06: LLMServingSim 2.0 deep analysis

---

## Part A: Code-Level Bug-to-Solution Matrix

This section maps every specific bug from findings 01 and 02 to the research technique that addresses it.

### A.1 Critical Bugs (3 from Finding 01)

| # | Bug | File | Research Solution | Source |
|---|-----|------|-------------------|--------|
| C1 | Duplicate field definitions in `TuningConfig` -- 7 fields defined twice, second definition silently overwrites first | `config/models.py` | No direct paper solution. Code quality issue. Fix: deduplicate fields. However, the field set itself should be expanded per SCOOT's parameter inventory (Finding 03 Section 1: 9 parameters including `scheduler-delay-factor`, `disable-custom-all-reduce`, `use-v2-block-manager`). | SCOOT |
| C2 | "ignore" direction filtering creates objective count mismatch with Optuna study -- optimizer returns wrong number of values | `tuner/optimizer.py` | SCOOT (Finding 03) uses explicit lambda weights `(lambda_T, lambda_l, lambda_phi, lambda_theta)` to control single vs multi-objective, never filtering directions mid-trial. vllm-tuner should adopt this: all 4 metrics always computed, weights determine which to optimize. | SCOOT |
| C3 | `BaselineConfig.enabled` has `ge=True` (Pydantic constraint), making it impossible to set to `False` | `config/models.py` | No paper solution. Pure code bug. Fix: remove `ge=True` from boolean field. |  |

### A.2 High-Severity Bugs (9 from Finding 01, cross-cutting from Finding 02)

| # | Bug | File | Research Solution | Source |
|---|-----|------|-------------------|--------|
| H1 | Constraint-violated trial returns `-inf` for minimize objectives, causing Optuna to SELECT it as best | `optimizer.py` | SCOOT (Finding 03, Section 1) handles infeasible trials via POF (Probability of Feasibility): infeasible configs are classified as such by the random forest, not returned as extreme metric values. The correct approach: return `None` for infeasible trials (Optuna's `FAIL` state) or use SCOOT's dedicated feasibility classifier. | SCOOT |
| H2 | OOM metrics return 3 values regardless of active objective count | `optimizer.py` | Same root cause as C2. SCOOT always returns all metrics; lambda weights control optimization. Fix: always return a fixed-length tuple matching study directions. | SCOOT |
| H3 | Async/sync mismatch -- async benchmark called from sync optimizer via ThreadPoolExecutor | `study_manager.py` | SCOOT (Finding 03) uses process-level parallelism (parallel suggestion with PD=k), not async threading. Each benchmark evaluation runs in its own process. vllm-tuner should adopt process isolation. | SCOOT |
| H4 | All exceptions in `_run_benchmark` reported as OOM, poisoning the BO search with false OOM signals | `study_manager.py` | SCOOT (Finding 03, Section 1) distinguishes infeasible (crashes, OOM) from errors (transient failures) via the random forest. The RF only trains on genuine observations, not misclassified errors. Fix: classify exceptions by type (OOM vs timeout vs connection error) and only mark true OOM as infeasible. | SCOOT |
| H5 | `batch_size` in search space is not a valid vLLM parameter -- wastes search dimension | `search_space.py` | SCOOT (Finding 03, Section 1) defines exactly 9 vLLM parameters validated against vLLM docs. Finding 04 (Section 1.1) provides the authoritative parameter list with correct names. Fix: replace `batch_size` with `max_num_seqs` (the actual vLLM parameter). | SCOOT, Web Research |
| H6 | Log file handle leaked -- `with` block closes file before subprocess finishes writing | `launcher.py` | No direct paper solution. Infrastructure bug. Fix: keep file handle open for subprocess lifetime, close in `stop()`. |  |
| H7 | Cache hit/miss rate calculation divides by `total * 100`, wrong by factor of 100x | `telemetry.py` | LLMServingSim 2.0 (Finding 06, Section 4.3) validates prefix cache hit rate with 0.93% error against real vLLM systems. The correct calculation: `hit_rate = hits / (hits + misses)`. Use LLMServingSim's validated metric definitions as reference. | LLMServingSim 2.0 |
| H8 | **Token counting counts HTTP chunks, not tokens** -- ALL throughput/TPOT metrics are wrong | `baseline/runner.py` (also `request_generator.py`) | This is the most impactful bug across the codebase. SCOOT (Finding 03) measures metrics via the vLLM engine's own reporting (100-second stress tests with proper metric collection). LLMServingSim 2.0 (Finding 06) computes tokens analytically from model config. Fix: parse the `usage` field from vLLM's final response chunk (contains `prompt_tokens`, `completion_tokens`, `total_tokens`), or use vLLM's Prometheus `/metrics` endpoint which reports `vllm:num_generation_tokens_total`. | SCOOT, LLMServingSim 2.0 |
| H9 | Any single request failure aborts entire baseline generation | `baseline/runner.py` | SCOOT (Finding 03) runs stress tests with fault tolerance -- individual request failures are recorded but don't abort the evaluation. Fix: catch per-request errors, record them, continue with remaining requests, report error rate as a metric. | SCOOT |

### A.3 Cross-Cutting Bugs (from Finding 02)

| # | Bug | Files | Research Solution | Source |
|---|-----|-------|-------------------|--------|
| X1 | **Latency = TTFT + TPOT** (wrong formula; should be TTFT + TPOT * output_tokens) | `vllm_metrics.py` | SCOOT (Finding 03, Section 1) defines metrics precisely: T(x) = request throughput, L(x) = P95 tail latency, Phi(x) = avg TTFT, Theta(x) = avg TPOT. End-to-end latency is not TTFT + TPOT. LLMServingSim 2.0 (Finding 06, Section 1.3) tracks per-request end-to-end latency correctly as the time from request arrival to last token generation. | SCOOT, LLMServingSim 2.0 |
| X2 | **Streaming JSON parsing counts chunks not tokens** -- SSE boundary mishandling | `request_generator.py`, `generate_baseline.py` | vLLM uses Server-Sent Events (SSE) format. Fix: use a proper SSE parser (e.g., `sseclient` library) or parse `data:` lines. vLLM's `/metrics` Prometheus endpoint provides authoritative token counts without parsing streaming output. Finding 04 (Section 5.1) documents vLLM production stack metrics. | Web Research |
| X3 | **XSS vulnerability in HTML reports** -- Jinja2 template without auto-escape | `html.py` | No paper solution. Security bug. Fix: use `jinja2.Environment(autoescape=True)`. |  |
| X4 | **No signal handling** -- orphan vLLM processes on Ctrl+C | `main.py`, `generate_baseline.py` | SCOOT (Finding 03) manages subprocess lifecycle properly since it runs multiple parallel evaluations. Fix: register SIGINT/SIGTERM handlers that call `launcher.stop()`. |  |
| X5 | **Global random seed side effect** -- `random.seed(2026)` at import pollutes global state | `alpaca.py` | SCOOT (Finding 03) uses Sobol sequences for initial sampling (quasi-Monte Carlo), not `random.seed()`. Fix: use `random.Random(2026)` instance. | SCOOT |
| X6 | **GPU history grows unboundedly** in collector | `gpu_collector.py` | LLMServingSim 2.0 (Finding 06, Section 1.4) tracks metrics as running statistics, not unbounded lists. Fix: use a circular buffer or rolling window. | LLMServingSim 2.0 |
| X7 | **Blocking NVML calls in async context** without `run_in_executor()` | `gpu_collector.py` | No direct paper solution. Fix: wrap NVML calls in `loop.run_in_executor()`. |  |
| X8 | **`peak_memory_mb` is just final memory**, not actual peak | `study_manager.py` | LLMServingSim 2.0 (Finding 06, Section 1.4) tracks memory usage over time, reporting true peaks. vLLM's Prometheus endpoint exposes `vllm:gpu_cache_usage_perc` which can be sampled at high frequency. Fix: sample GPU memory at 1Hz during benchmark, report max. | LLMServingSim 2.0 |

### A.4 Medium-Severity Bugs Mapped to Solutions

| # | Bug | Research Solution | Source |
|---|-----|-------------------|--------|
| M3 | `pipeline_parallel_size` not filtered by GPU count | SCOOT known constraints: `TP * PP <= num_gpus`. LLMServingSim 2.0 enforces valid parallelism in Execution Planner. | SCOOT, LLMServingSim 2.0 |
| M4 | No joint constraint on `TP * PP <= num_gpus` | Same as M3. SCOOT Section 1: known constraint pruning. | SCOOT |
| M9 | No port conflict detection in launcher | No paper solution. Infrastructure fix. |  |
| M10 | Swap counts overwritten instead of accumulated in telemetry | LLMServingSim 2.0 memory model tracks cumulative eviction counts. Fix: accumulate, don't overwrite. | LLMServingSim 2.0 |
| M11 | "initial_memory_mb" collected after benchmark, not before | LLMServingSim 2.0 tracks pre/post memory states explicitly in its memory model. Fix: collect before launching benchmark. | LLMServingSim 2.0 |

---

## Part B: Structural Gap-to-Solution Analysis

### B.1 Optimization Algorithm Gaps

#### GAP-01: No Hidden Constraint Learning -- Wasted Evaluations on Crashing Configs

- **Source:** Finding 01 (optimizer.py -- no crash prediction; H4: all exceptions classified as OOM)
- **Severity:** Critical
- **Description:** vllm-tuner's `VLLMOptimizer` uses Optuna's TPE sampler without any mechanism to predict whether a configuration will crash the vLLM engine. Many parameter combinations cause OOM, CUDA errors, or hangs. Each failed trial wastes 5-10 minutes of GPU time. Worse, bug H4 misclassifies all errors as OOM, poisoning the feasibility signal.
- **Research Solution:** SCOOT (Finding 03, Section 1) uses a **random forest classifier** trained on past observations to learn POF(x) -- the Probability of Feasibility for configuration x. It uses a dynamic threshold Delta (starting at 0.5, adjusted by +/- 0.05 based on consecutive feasible/infeasible results) to balance exploration near constraint boundaries.
- **Implementation Path:**
  1. Fix H4 first: properly classify exceptions (OOM vs timeout vs connection error vs unknown)
  2. After each trial, record (config, feasible/infeasible) as training data
  3. Train sklearn `RandomForestClassifier` on accumulated observations (minimum 5 observations before activating)
  4. Before suggesting a new config, predict POF and reject if below threshold
  5. Implement adaptive Delta: infeasible -> `Delta = min(0.75, Delta + 0.05)`; 5+ consecutive feasible -> `Delta = max(0.25, Delta - 0.05)`

#### GAP-02: No Known Constraint Pruning

- **Source:** Finding 01 (search_space.py -- M3, M4: no TP*PP constraint), Finding 03 (SCOOT known constraints)
- **Severity:** High
- **Description:** Bugs M3 and M4 show that `pipeline_parallel_size` is not filtered by GPU count and there is no joint `TP * PP <= num_gpus` constraint. Additionally, H5 shows `batch_size` is included but is not a valid vLLM parameter.
- **Research Solution:** SCOOT (Finding 03, Section 1) encodes known constraints for vLLM-0.4.2: `max-num-batched-tokens >= max-num-seqs`, `enable-chunked-prefill AND enable-prefix-caching` cannot both be True, when `enable-chunked-prefill` is False then `max-num-batched-tokens >= max_model_length`.
- **Implementation Path:**
  1. Add a `Constraint` class with `validate(params) -> bool`
  2. Encode: `TP * PP <= num_gpus`, `max_num_batched_tokens >= max_num_seqs`
  3. Version-specific: prefix caching + chunked prefill conflict (vLLM <0.6 only; compatible in V1)
  4. In sampler, reject samples violating constraints before evaluation (zero-cost filtering)
  5. Make constraints configurable per vLLM version via a constraint registry

#### GAP-03: Single Acquisition Function

- **Source:** Finding 01 (optimizer.py uses TPE only), Finding 03 (SCOOT MACE ensemble, HEBO)
- **Severity:** Medium
- **Description:** TPE is less sample-efficient than GP-based BO for the small evaluation budgets typical of vLLM benchmarking (~30 trials).
- **Research Solution:** SCOOT uses **MACE** -- runs UCB, PI, and EI simultaneously and selects from their Pareto frontier. HEBO (Finding 03, Section 3.1) adds input/output warping. LLAMBO (Finding 03, Section 3.8) provides LLM-guided zero-shot warmstarting as an Optuna sampler.
- **Implementation Path:**
  1. **Quick win:** Add OptunaHub's LLAMBO sampler for initial 5 trials
  2. **Medium:** Implement composite sampler running TPE + CMA-ES + GP, selecting best
  3. **Full:** Replace Optuna with HEBO backend (SCOOT's validated choice)

#### GAP-04: Broken Multi-Objective Optimization

- **Source:** Finding 01 (C2: direction filtering mismatch; design issue 1: weights as binary flags; design issue 6: no Pareto front selection), Finding 03 (SCOOT EHVI)
- **Severity:** High
- **Description:** Three compounding bugs make multi-objective optimization non-functional: (C2) objective count mismatch crashes Optuna, (design issue 1) weights are used as on/off flags instead of actual weights, (design issue 6) no mechanism to present Pareto frontier to users.
- **Research Solution:** SCOOT uses **EHVI (Expected Hypervolume Improvement)** with the default vLLM config as reference point. Lambda weights control which metrics are active in a principled way: `(1,0,0,0)` = throughput-only, `(0,0,-1,-1)` = TTFT+TPOT joint optimization.
- **Implementation Path:**
  1. Fix C2: always return all 4 metrics (TPS, P95 latency, TTFT, TPOT), use Optuna's full multi-objective study
  2. Replace binary weights with SCOOT's lambda-weight formulation
  3. Use `NSGAIISampler` (Optuna built-in) or implement EHVI
  4. Return full Pareto frontier via `study.best_trials`, not just `study.best_trial`
  5. Use default vLLM config (trial 0) as the EHVI reference point

#### GAP-05: No Parallel Suggestion

- **Source:** Finding 01 (optimizer.py runs trials serially), Finding 03 (SCOOT PD=k)
- **Severity:** Medium
- **Description:** Serial evaluation wastes GPU resources when multiple GPU sets are available.
- **Research Solution:** SCOOT suggests k configs simultaneously. PD=2 halves time with same quality.
- **Implementation Path:** Use Optuna's `ask()`/`tell()` API to generate multiple suggestions, partition GPUs across parallel trials, use process-level isolation.

### B.2 Parameter Space Gaps

#### GAP-06: Missing Critical Modern vLLM Parameters

- **Source:** Finding 01 (design issue 5: missing parameters; H5: invalid `batch_size`), Finding 03 (SCOOT param list), Finding 04 (vLLM v0.6+ parameters)
- **Severity:** High
- **Description:** vllm-tuner's search space is missing several high-impact parameters and includes an invalid one (`batch_size`).
- **Research Solution:** SCOOT defines 9 parameters. Finding 04 extends this with modern vLLM additions.
- **Implementation Path:** Replace search space with:

  | Parameter | Type | Range | Impact | Source |
  |---|---|---|---|---|
  | `tensor_parallel_size` | int | [1, num_gpus] | High | SCOOT |
  | `max_num_seqs` | int | [32, 512] | High | SCOOT (replaces invalid `batch_size`) |
  | `max_num_batched_tokens` | int | [512, 32768] | High | SCOOT |
  | `enable_chunked_prefill` | bool | {T, F} | Medium | SCOOT |
  | `enable_prefix_caching` | bool | {T, F} | Medium | SCOOT |
  | `block_size` | cat | {8, 16, 32} | Low | SCOOT |
  | `num_scheduler_steps` | int | [1, 15] | High | vLLM v0.6+ (Finding 04) |
  | `gpu_memory_utilization` | float | [0.80, 0.95] | High | Finding 04 |
  | `max_model_len` | int | [256, model_max] | Medium | Finding 04 |
  | `kv_cache_dtype` | cat | {auto, fp8} | Medium | Finding 04 |
  | `quantization` | cat | {None, fp8, awq, gptq} | High | Finding 04 |

#### GAP-07: No Workload-Aware Space Adaptation

- **Source:** Finding 01 (fixed search space), Finding 03 (SCOOT workload-specific objectives), Finding 06 (LLMServingSim workload config)
- **Severity:** Medium
- **Description:** Same search space regardless of workload. Chat vs batch vs long-context have different optimal parameter regions.
- **Research Solution:** GPTuner (Finding 03, Section 3.2) uses LLM-based knowledge extraction for workload-aware pruning. LLMServingSim takes workload configuration (arrival rates, sequence distributions) as a first-class input.
- **Implementation Path:** Add workload profiling phase: analyze traces for avg prompt/output lengths, shared prefix ratio, arrival pattern. Auto-adjust: if `shared_prefix_ratio > 0.5`, fix `enable_prefix_caching=True`; if prefill-heavy, prioritize chunked prefill tuning.

#### GAP-08: No Speculative Decoding Tuning

- **Source:** Finding 03 (SCOOT excludes spec decode), Finding 04 (vLLM spec decode params), Finding 06 (LLMServingSim does not model spec decode)
- **Severity:** Medium
- **Description:** Speculative decoding is a gap in both SCOOT and LLMServingSim 2.0, yet for latency-sensitive low-concurrency workloads it provides 2-5x speedup (Finding 03, Section 2.5).
- **Research Solution:** Finding 04 (Section 6.3) documents the parameter space. TurboSpec and Online Speculative Decoding (Finding 03) show adaptive approaches.
- **Implementation Path:** Add as conditional parameters active only in latency optimization mode. Include `speculative_model`, `num_speculative_tokens` [1-7]. Auto-disable if measured acceptance rate < 50%.

### B.3 Benchmarking and Measurement Gaps

#### GAP-09: Every Config Requires Expensive Real Benchmark -- No Pre-Screening

- **Source:** Finding 01 (all trials launch real vLLM; design issue 3: full model reload per trial), Finding 06 (LLMServingSim as surrogate)
- **Severity:** High
- **Description:** vllm-tuner evaluates every candidate by launching real vLLM (5-10 min per trial, 30 trials = 2.5-5 hours). Design issue 3 notes full model reload per trial makes this even slower. Many configs could be pre-screened via simulation.
- **Research Solution:** LLMServingSim 2.0 (Finding 06, Section 6.1) simulates configs in minutes with 0.97% accuracy. Vidur (Finding 06, Section 4.4) is even faster (~seconds) though less accurate. BudSimulator's GenZ framework provides millisecond analytical estimates.
- **Implementation Path:**
  1. **Phase 1 -- Analytical pre-filter:** Use GenZ-style memory calculation to reject OOM configs in milliseconds (also fixes GAP-16)
  2. **Phase 2 -- Simulation screening:** Simulate 50-100 configs via Vidur (seconds each) to rank them
  3. **Phase 3 -- Real benchmarks:** Only run top-10 configs on real hardware
  4. **Warm-start BO:** Inject simulation results as Optuna trials via `study.add_trial()`

  Directly simulatable parameters:
  | Parameter | LLMServingSim Feasibility |
  |---|---|
  | `tensor_parallel_size` | High -- MSG parallelism |
  | `pipeline_parallel_size` | High -- MSG parallelism |
  | `max_num_seqs` | High -- batch scheduler |
  | `max_num_batched_tokens` | High -- batch scheduler |
  | `gpu_memory_utilization` | High -- memory model |
  | `enable_prefix_caching` | High -- prefix cache model |
  | `block_size` | Medium -- cache granularity |
  | `enable_chunked_prefill` | Low -- not modeled |
  | `num_scheduler_steps` | Low -- framework-specific |

#### GAP-10: Token Counting is Fundamentally Broken (All Metrics Wrong)

- **Source:** Finding 02 (Cross-cutting issue 1: token counting counts chunks; H8 same bug; Cross-cutting issue 2: latency formula wrong)
- **Severity:** Critical (affecting all downstream analysis)
- **Description:** Two compounding bugs: (1) HTTP chunks counted as tokens, making throughput/TPOT wrong, (2) latency = TTFT + TPOT instead of TTFT + TPOT * num_output_tokens. Since token count is also wrong, latency is doubly incorrect.
- **Research Solution:** SCOOT (Finding 03) measures via vLLM's native metrics during 100-second stress tests. LLMServingSim 2.0 (Finding 06) computes tokens analytically. Finding 04 (Section 5.1) documents vLLM Prometheus metrics: `vllm:num_generation_tokens_total`, `vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`.
- **Implementation Path:**
  1. **Primary fix:** Use vLLM's Prometheus `/metrics` endpoint for authoritative metrics (this already exists in vllm-tuner's telemetry.py but has bug H7)
  2. **Secondary fix:** Parse `usage` field from vLLM's final streaming chunk (`completion_tokens`, `prompt_tokens`)
  3. **Latency formula:** `e2e_latency = TTFT + TPOT * (output_tokens - 1)` (first token counted in TTFT)
  4. Fix H7 (cache hit rate 100x error) simultaneously

#### GAP-11: No Statistical Rigor -- Single Run Per Config

- **Source:** Finding 02 (single benchmark per config), Finding 03 (SCOOT SLO robustness)
- **Severity:** Medium
- **Description:** vLLM performance varies 5-15% across runs. A single measurement may not represent true performance.
- **Research Solution:** SCOOT (Finding 03, Section 1) implements SLO robustness: after finding best config, run stress tests multiple times with varying arrival orders and report worst-case.
- **Implementation Path:** Run each promising config 3x minimum. For final top-3, run 5x with varying arrival patterns. Use median for BO objective. Report P50/P95 across runs.

#### GAP-12: Benchmark Workload Not Representative

- **Source:** Finding 02 (workload.py -- flat prompt list), Finding 03 (SCOOT uses ShareGPT + Poisson), Finding 06 (LLMServingSim workload config)
- **Severity:** Medium
- **Description:** Simple prompt list without arrival rate modeling, sequence length distribution, or prefix sharing patterns.
- **Research Solution:** SCOOT uses ShareGPT traces with Poisson arrival. LLMServingSim supports configurable arrival rates and per-request execution traces.
- **Implementation Path:** Add arrival rate config (Poisson lambda), support ShareGPT/MT-Bench as preset workloads, add shared prefix injection for prefix caching evaluation.

#### GAP-13: No Cost-Aware Trial Selection

- **Source:** Finding 01 (all trials treated as equal cost), Finding 04 (Section 3.3: cost-aware BO)
- **Severity:** Medium
- **Description:** Configs with higher TP or larger batches take longer to evaluate. Early exploration should prefer cheap configs.
- **Research Solution:** Finding 04 surveys EIpu, cost-cooling, and CArBO. BoTorch provides native cost-aware acquisition.
- **Implementation Path:** Track wall-clock time per trial, build cost predictor from (config -> evaluation time), modify acquisition to use `EI(x) / predicted_cost(x)`.

### B.4 Architecture and Infrastructure Gaps

#### GAP-14: Async/Sync Bridge is Fragile

- **Source:** Finding 01 (H3: async/sync mismatch), Finding 02 (Cross-cutting issue 8: deprecated asyncio patterns)
- **Severity:** Medium
- **Description:** Multiple async/sync issues: H3 (async benchmark from sync optimizer), deprecated `asyncio.get_event_loop()`, potential nested `asyncio.run()`.
- **Research Solution:** SCOOT uses process-level parallelism, avoiding async complexity entirely. Each benchmark evaluation is an independent process.
- **Implementation Path:** Restructure to use `subprocess` or `multiprocessing` for benchmark execution, eliminating async/sync bridge entirely. This also enables GAP-05 (parallel suggestions) naturally.

#### GAP-15: No Transfer Learning Across Sessions

- **Source:** Finding 01 (study_manager.py -- session-local state), Finding 03 (Restune meta-learning)
- **Severity:** Medium
- **Description:** Each tuning session starts from scratch. Results from tuning LLaMA-7B cannot inform LLaMA-13B tuning.
- **Research Solution:** Restune (Finding 03, Section 3.6) uses meta-learning warm-start. LLMServingSim 2.0 could provide simulated transfer across hardware.
- **Implementation Path:** Store completed studies with metadata (model, hardware, workload). For new studies, search for similar past studies, inject past results via `study.add_trial()` with similarity discount.

#### GAP-16: No OOM Pre-Check -- Crashes Discovered Only After Full Launch

- **Source:** Finding 01 (H1: -inf on constraint violation, H4: all errors classified as OOM), Finding 06 (LLMServingSim memory model, GenZ memory calculation)
- **Severity:** High
- **Description:** vllm-tuner launches a full vLLM instance and runs a complete benchmark before discovering OOM. Combined with H4 (misclassified errors), the optimizer wastes many trials on impossible configs.
- **Research Solution:** LLMServingSim 2.0's memory model (Finding 06, Section 2.3) computes memory requirements analytically. BudSimulator's GenZ framework has well-tested memory formulas. Finding 04 (Section 4.1) provides the KV cache formula.
- **Implementation Path:**
  1. Before launching: `model_memory = params * bytes_per_param / TP`
  2. `kv_per_token = layers * 2 * kv_heads * head_dim * dtype_bytes / TP`
  3. `max_tokens = (gpu_memory * gpu_memory_util - model_memory - activation_overhead) / kv_per_token`
  4. If `max_num_seqs * avg_seq_len > max_tokens`, reject immediately (zero cost)
  5. This eliminates ~30-50% of trial space that would OOM

#### GAP-17: Pruning Configured But Non-Functional

- **Source:** Finding 01 (design issue 2: pruner reports constant 0 before trial runs)
- **Severity:** Medium
- **Description:** Optuna's `MedianPruner` is configured but the intermediate value reported is always 0 (before the benchmark completes), so pruning never triggers.
- **Research Solution:** Finding 04 (Section 3.4: multi-fidelity) suggests using shorter benchmarks as cheap approximations. SCOOT's parallel suggestion avoids the need for pruning by evaluating fewer configs more thoroughly.
- **Implementation Path:**
  1. Report intermediate values during benchmark execution (e.g., throughput measured at 25%, 50%, 75% completion)
  2. Or: replace trial-level pruning with simulation-based pre-screening (GAP-09), which is more effective

#### GAP-18: Full Model Reload Per Trial

- **Source:** Finding 01 (design issue 3: full model reload makes optimization slow)
- **Severity:** High
- **Description:** vllm-tuner restarts the entire vLLM server for each trial, including model loading (which takes 30-120 seconds for large models). For parameters that do not require model reload (e.g., `max_num_seqs`, `max_num_batched_tokens`), this is wasteful.
- **Research Solution:** No direct paper solution, but SCOOT's approach of running 100-second stress tests (Finding 03) amortizes reload cost. LLMServingSim 2.0 avoids this entirely through simulation.
- **Implementation Path:**
  1. Classify parameters as "reload-required" (TP, PP, quantization, model_len) vs "runtime-adjustable" (max_num_seqs, max_num_batched_tokens, scheduler steps)
  2. When only runtime-adjustable params change between trials, use vLLM's `/reset` or hot-reconfigure API instead of full restart
  3. Group trials: first tune reload-required params (fewer iterations), then tune runtime params within each reload group

### B.5 Reporting and Output Gaps

#### GAP-19: No Power/Energy Metrics

- **Source:** Finding 02 (reporter.py -- no power metrics), Finding 06 (LLMServingSim 7-component power model)
- **Severity:** Medium
- **Description:** No power/energy analysis in tuning results despite TCO importance.
- **Research Solution:** LLMServingSim 2.0 provides 7-component power model with 1.34% accuracy (Finding 06, Section 2.4). nvidia-smi provides real-time GPU power.
- **Implementation Path:** Sample `nvidia-smi --query-gpu=power.draw` at 1Hz during benchmarks. Compute `watts_per_token = avg_power / throughput`. Add as optional optimization objective.

#### GAP-20: No Baseline Comparison in Reports

- **Source:** Finding 02 (no default config comparison), Finding 03 (SCOOT uses default as reference)
- **Severity:** Low
- **Description:** Users cannot see improvement from tuning without a baseline.
- **Research Solution:** SCOOT always benchmarks default config as reference point for EHVI.
- **Implementation Path:** Always run default vLLM config as trial 0. Report improvement percentages.

#### GAP-21: XSS in HTML Reports

- **Source:** Finding 02 (Cross-cutting issue 5: Jinja2 without auto-escape)
- **Severity:** Medium (security)
- **Description:** User-controlled values (study name, parameters) rendered as raw HTML.
- **Research Solution:** No paper solution. Standard security practice.
- **Implementation Path:** Use `jinja2.Environment(autoescape=True)`.

#### GAP-22: No Deployment-Ready Output

- **Source:** Finding 02 (Python dict output), Finding 04 (production deployment patterns)
- **Severity:** Low
- **Description:** Output is a Python dict, not usable for deployment.
- **Research Solution:** Finding 04 documents production patterns: vLLM serve commands, Docker compose, K8s Helm.
- **Implementation Path:** Add output formatters for `vllm-cmd`, `docker-compose`, `helm-values`.

### B.6 Simulation-Enabled Improvements

#### GAP-23: Parallelism Exploration Too Expensive

- **Source:** Finding 01 (TP/PP in search space), Finding 06 (LLMServingSim parallelism, GenZ analytical)
- **Severity:** High
- **Description:** Each TP/PP combination requires full vLLM restart with different GPU allocation (5-10 min each). For 8 GPUs, there are ~8 valid TP/PP combos.
- **Research Solution:** LLMServingSim 2.0 directly models parallelism with communication overhead. GenZ provides millisecond analytical estimates. APEX (Finding 06) specifically targets automated parallel execution search.
- **Implementation Path:** Use GenZ analytical model to rank TP/PP/EP combos in milliseconds. Only benchmark top-2 on real hardware. Reduces parallelism exploration from ~80 minutes to ~20 minutes.

#### GAP-24: Prefix Cache Impact Unpredictable

- **Source:** Finding 01 (no prefix cache modeling), Finding 06 (LLMServingSim multi-tier prefix caching: 0.93% error)
- **Severity:** Medium
- **Description:** Benefit of `enable_prefix_caching` depends on workload prefix sharing, only observable during full benchmark. Short benchmarks underestimate cache benefits.
- **Research Solution:** LLMServingSim 2.0 models prefix caching with radix-tree caches across device/host/CXL tiers.
- **Implementation Path:** Analyze workload for prefix sharing ratio before benchmark. Ensure benchmarks run long enough for cache steady-state (100+ requests with shared prefixes).

#### GAP-25: No P/D Disaggregation Support

- **Source:** Finding 01 (no disaggregation params), Finding 03 (DistServe: 7.4x more requests), Finding 06 (LLMServingSim M:N P/D)
- **Severity:** Medium (growing)
- **Description:** Disaggregated prefill-decode is increasingly adopted in production. New parameters: P/D GPU ratios, separate TP per phase, KV transfer bandwidth.
- **Research Solution:** LLMServingSim 2.0 (Finding 06, Section 3.3) models P/D as first-class feature. DistServe (Finding 03, Section 2.4) co-optimizes resources per phase.
- **Implementation Path:** Add `disaggregated_mode`, `prefill_tp`, `decode_tp`, `prefill_gpu_count`. Use LLMServingSim to explore P/D space. Wait for vLLM disaggregation API to stabilize.

---

## Part C: Summary Matrix

| Gap | Severity | Category | Primary Research Fix | Effort | Phase |
|-----|----------|----------|---------------------|--------|-------|
| GAP-01 | Critical | Algorithm | SCOOT random forest | Medium | 2 |
| GAP-02 | High | Algorithm | SCOOT known constraints | Low | 1 |
| GAP-03 | Medium | Algorithm | MACE / HEBO / LLAMBO | Medium | 4 |
| GAP-04 | High | Algorithm | SCOOT EHVI | Medium | 2 |
| GAP-05 | Medium | Algorithm | SCOOT parallel PD | Medium | 4 |
| GAP-06 | High | Params | SCOOT + Web research | Low | 1 |
| GAP-07 | Medium | Params | GPTuner / LLMServingSim | Medium | 4 |
| GAP-08 | Medium | Params | Web research | Low | 4 |
| GAP-09 | High | Benchmark | LLMServingSim / Vidur | High | 3 |
| GAP-10 | Critical | Metrics | SCOOT / Prometheus | Low | 1 |
| GAP-11 | Medium | Benchmark | SCOOT SLO robustness | Low | 1 |
| GAP-12 | Medium | Benchmark | SCOOT / LLMServingSim | Medium | 4 |
| GAP-13 | Medium | Benchmark | Cost-aware BO (EIpu) | Medium | 2 |
| GAP-14 | Medium | Code | SCOOT process model | Medium | 2 |
| GAP-15 | Medium | Code | Restune meta-learning | Medium | 4 |
| GAP-16 | High | Code | LLMServingSim / GenZ | Low | 1 |
| GAP-17 | Medium | Code | Multi-fidelity / simulation | Medium | 3 |
| GAP-18 | High | Code | Parameter classification | Medium | 2 |
| GAP-19 | Medium | Report | LLMServingSim power model | Low | 2 |
| GAP-20 | Low | Report | SCOOT baseline reference | Low | 1 |
| GAP-21 | Medium | Security | Standard practice | Low | 1 |
| GAP-22 | Low | Report | Web research | Low | 5 |
| GAP-23 | High | Simulation | LLMServingSim / GenZ | Medium | 3 |
| GAP-24 | Medium | Simulation | LLMServingSim | Medium | 3 |
| GAP-25 | Medium | Simulation | LLMServingSim / DistServe | High | 4 |

---

## Part D: Implementation Roadmap

### Phase 1: Critical Fixes and Quick Wins (Week 1-2)

**Goal:** Fix broken metrics, add basic constraints, plug the most impactful gaps.

1. **GAP-10 (Critical):** Fix token counting -- use Prometheus `/metrics` or `usage` field from response. Fix latency formula. Fix H7 cache hit rate 100x error.
2. **GAP-02 (High):** Add known constraint pruning -- `TP * PP <= num_gpus`, `max_batched_tokens >= max_num_seqs`.
3. **GAP-06 (High):** Expand parameter space -- add `num_scheduler_steps`, `gpu_memory_utilization`, `kv_cache_dtype`. Remove invalid `batch_size`.
4. **GAP-16 (High):** Add analytical OOM pre-check using GenZ-style memory formula.
5. **GAP-11 (Medium):** Add multi-run benchmarks (3x minimum for final configs).
6. **GAP-20 (Low):** Always benchmark default config as trial 0 for comparison.
7. **GAP-21 (Medium):** Fix XSS: `jinja2.Environment(autoescape=True)`.
8. Fix bugs C1, C3, H5, H6, H9, X3, X4, X5 (pure code fixes, no research dependency).

### Phase 2: Core Algorithm Improvements (Week 3-5)

**Goal:** Upgrade the optimization algorithm to research-quality.

9. **GAP-01 (Critical):** Implement random forest hidden constraint learning (fix H4 first).
10. **GAP-04 (High):** Implement multi-objective optimization with EHVI/NSGA-II (fix C2 first).
11. **GAP-13 (Medium):** Add cost-aware trial selection (EIpu).
12. **GAP-14 (Medium):** Restructure to process-level benchmark execution (fixes H3 and async issues).
13. **GAP-18 (High):** Classify params as reload-required vs runtime-adjustable, group trials.
14. **GAP-19 (Medium):** Add nvidia-smi power monitoring during benchmarks.

### Phase 3: Simulation Integration (Week 6-9)

**Goal:** Add simulation-based pre-screening to dramatically reduce tuning time.

15. **GAP-09 (High):** Integrate Vidur or GenZ as fast pre-screening surrogate.
16. **GAP-23 (High):** Use analytical models for parallelism strategy ranking.
17. **GAP-24 (Medium):** Add prefix cache impact prediction from workload analysis.
18. **GAP-17 (Medium):** Replace non-functional pruning with simulation-based filtering.

### Phase 4: Advanced Features (Week 10-16)

**Goal:** Bring vllm-tuner to research-competitive feature parity.

19. **GAP-03 (Medium):** Implement MACE ensemble or switch to HEBO backend.
20. **GAP-05 (Medium):** Add parallel trial suggestion (PD=2-3).
21. **GAP-07 (Medium):** Workload-aware search space adaptation.
22. **GAP-15 (Medium):** Transfer learning across tuning sessions.
23. **GAP-12 (Medium):** Realistic workload generation with arrival patterns.
24. **GAP-08 (Medium):** Speculative decoding parameter tuning.
25. **GAP-25 (Medium):** P/D disaggregation parameter support.

### Phase 5: Polish (Ongoing)

26. **GAP-22 (Low):** Deployment-ready output formats.
27. Remaining code quality fixes from findings 01/02 bug lists.
