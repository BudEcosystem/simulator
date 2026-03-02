# Master Analysis: vllm-tuner Research & Code Audit

**Date**: 2026-02-28
**Scope**: Comprehensive synthesis of 7 research findings across code audit, paper review, web research, and competitive analysis
**Target Audience**: BudSimulator development team

---

## 1. Executive Summary

### What is vllm-tuner?

vllm-tuner is an Optuna-based Bayesian optimization tool for automatically tuning vLLM inference engine parameters. It uses Tree-structured Parzen Estimator (TPE) sampling to search over parameters like `max_num_seqs`, `gpu_memory_utilization`, `tensor_parallel_size`, and others, seeking to maximize throughput and/or minimize latency for a given model and hardware configuration.

The tool manages the full lifecycle: start a vLLM server with candidate parameters, run benchmark requests against it, collect metrics (throughput, latency, memory), report results to Optuna, and repeat. It supports multi-objective optimization, baseline comparison, HTML report generation, and GPU telemetry collection.

### Overall Assessment: Severe Issues

The codebase is in **early prototype / pre-alpha** quality with multiple critical bugs that **invalidate optimization results**:

| Category | Count |
|----------|-------|
| Critical bugs (results are wrong) | 3 |
| High-severity bugs | 9 |
| Medium-severity bugs | 22+ |
| Low-severity bugs | 15+ |
| Conceptual/design flaws | 7 |
| Security vulnerabilities | 4 |
| Missing features (research-backed) | 20+ |
| Documentation errors | 17 |
| Actual test coverage (critical paths) | 15-25% |

**Bottom line**: The tool cannot produce reliable optimization results in its current state. Token counting is broken, latency formulas are wrong, the objective direction system has a critical mismatch with Optuna, and constraint penalties are inverted for minimization objectives. Any configurations recommended by vllm-tuner should not be trusted without independent validation.

---

## 2. Architecture Overview

### vllm-tuner Module Map

```
vllm-tuner/
+-- src/
|   +-- cli/
|   |   +-- main.py              CLI entry point (Typer)
|   |
|   +-- config/
|   |   +-- models.py            Pydantic config models (CRITICAL BUGS)
|   |   +-- validation.py        YAML loading, study name validation
|   |
|   +-- tuner/
|   |   +-- optimizer.py         Optuna TPE optimizer (CRITICAL BUGS)
|   |   +-- study_manager.py     Trial orchestration (HIGH BUGS)
|   |
|   +-- optimization/
|   |   +-- search_space.py      Parameter space definition (HIGH BUGS)
|   |
|   +-- vllm/
|   |   +-- launcher.py          vLLM server lifecycle management
|   |   +-- telemetry.py         Log parsing for metrics (HIGH BUGS)
|   |
|   +-- benchmarks/
|   |   +-- workload.py          Abstract workload interface
|   |   +-- alpaca.py            Alpaca dataset workload
|   |   +-- request_generator.py HTTP benchmark client (CRITICAL BUGS)
|   |
|   +-- profiling/
|   |   +-- gpu_collector.py     NVML-based GPU monitoring
|   |   +-- vllm_metrics.py      Latency/throughput tracking (HIGH BUGS)
|   |
|   +-- baseline/
|   |   +-- runner.py            Baseline performance generation (HIGH BUGS)
|   |
|   +-- reporting/
|       +-- dashboard.py         Rich terminal UI
|       +-- html.py              Plotly HTML report generator (XSS vuln)
|       +-- export.py            JSON/YAML config export
|
+-- config/
|   +-- default.yaml             Default tuning configuration
|
+-- examples/
|   +-- simple_tune.yaml
|   +-- latency_optimized.yaml
|   +-- multi_gpu_tune.yaml
|
+-- scripts/
|   +-- generate_baseline.py     Standalone baseline script
|   +-- regenerate_trials.py     Study data export utility
|
+-- tests/unit/                  70 tests, 15-25% effective coverage
+-- docs/                        17 documentation errors
```

### Data Flow

```
YAML Config
     |
     v
+------------------+     +-----------------+     +------------------+
|  CLI (main.py)   | --> | StudyManager    | --> | VLLMOptimizer    |
|  - parse args    |     | - orchestrate   |     | - TPE sampler    |
|  - load config   |     |   trial loop    |     | - suggest params |
|  - validate      |     |                 |     | - record results |
+------------------+     +-+----------+----+     +------------------+
                           |          |
              +------------+          +------------+
              v                                    v
   +------------------+                 +------------------+
   | VLLMLauncher     |                 | BenchmarkRunner  |
   | - build vLLM cmd |                 | - send requests  |
   | - start server   |                 | - stream parse   |
   | - health check   |                 | - collect metrics|
   | - stop server    |                 +------------------+
   +------------------+                          |
              |                                  v
              v                       +------------------+
   +------------------+               | VLLMMetrics      |
   | TelemetryParser  |               | - TTFT, TPOT     |
   | - parse vLLM logs|               | - throughput      |
   | - cache stats    |               | - percentiles    |
   +------------------+               +------------------+
              |                                  |
              +------------+  +------------------+
                           |  |
                           v  v
                  +------------------+
                  | HTML Report Gen  |
                  | - Plotly charts  |
                  | - baseline diff  |
                  | - export configs |
                  +------------------+
```

### Key Dependencies

- **Optuna 3.5+**: Bayesian optimization backend (TPE sampler, study management)
- **httpx**: Async HTTP client for vLLM API calls
- **pynvml**: NVIDIA GPU monitoring
- **datasets + transformers**: HuggingFace Alpaca dataset loading
- **Plotly + Jinja2**: HTML report generation
- **Rich**: Terminal dashboard
- **PyYAML**: Configuration parsing
- **Pydantic v2**: Configuration validation

---

## 3. Critical Bug Report (Priority Order)

### CRITICAL -- Results Are Invalid

| # | Severity | Source | File:Line | Bug Description | Impact | Fix Approach |
|---|----------|--------|-----------|-----------------|--------|--------------|
| 1 | CRITICAL | F01 | `config/models.py:126-146` | Duplicate field definitions in `TuningConfig` -- 7 fields defined twice | Second definitions overwrite first; confusing and error-prone class structure | Remove duplicate field block (lines 137-144) |
| 2 | CRITICAL | F01 | `tuner/optimizer.py:30-36,54,169-183` | "ignore" direction filtering creates objective count mismatch with Optuna study | `evaluate_trial` returns N values but study expects M (where M != N), causing Optuna exceptions | Filter objectives consistently: use same filtered list everywhere |
| 3 | CRITICAL | F01,F05 | `config/models.py:109` | `BaselineConfig.enabled` has `ge=True` on a bool field | Impossible to disable baseline generation via config (`False >= True` fails validation) | Remove `ge=True` from the bool field |

### HIGH -- Significant Functional Errors

| # | Severity | Source | File:Line | Bug Description | Impact | Fix Approach |
|---|----------|--------|-----------|-----------------|--------|--------------|
| 4 | HIGH | F02 | `benchmarks/request_generator.py:129` | Token counting counts HTTP chunks, not tokens (`output_tokens += 1` per chunk) | All throughput metrics (tokens/sec) are wrong; TPOT is wrong | Use `response["usage"]["completion_tokens"]` from API response |
| 5 | HIGH | F02 | `profiling/vllm_metrics.py:46` | Latency = TTFT + TPOT (should be TTFT + TPOT * num_output_tokens) | All latency statistics (avg, p50, p95, p99) are incorrect | Fix formula to `TTFT + (TPOT * output_tokens)` |
| 6 | HIGH | F01 | `tuner/optimizer.py:169-172` | Constraint-violated trial returns `-inf` for all directions, but `-inf` is optimal for minimize objectives | Constraint-violating trials could be selected as "best" for latency/memory minimization | Return `+inf` for minimize directions, `-inf` for maximize |
| 7 | HIGH | F01 | `tuner/optimizer.py:86-115` | OOM metrics always return 3 values regardless of how many objectives are active | Optuna raises exception when value count mismatches study directions | Return only active objectives, matching filtered directions list |
| 8 | HIGH | F01 | `tuner/study_manager.py:54-69` | Async/sync mismatch -- async benchmark called from sync optimizer via ThreadPoolExecutor | Shared mutable state (launcher, gpu_collector) accessed from multiple threads without synchronization | Make optimizer fully async, or make benchmark fully sync |
| 9 | HIGH | F01 | `tuner/study_manager.py:153-161` | All exceptions in `_run_benchmark` report as `oom_detected: True` | Network timeouts, loading errors, etc. all poison the optimizer's understanding of the parameter space | Distinguish OOM from other errors; use separate error categories |
| 10 | HIGH | F01 | `optimization/search_space.py:15` | `batch_size` is not a valid vLLM parameter | Wastes search space on a parameter that has no effect; generated params silently ignored by launcher | Remove `batch_size`; use `max_num_seqs` instead |
| 11 | HIGH | F01 | `vllm/launcher.py:100-107` | Log file handle leaked -- `with` block closes file immediately after Popen creation | File descriptor leaks in parent process; fails on Windows | Keep file object alive for process lifetime |
| 12 | HIGH | F01 | `vllm/telemetry.py:71-77` | Cache hit/miss rate divided by `total * 100` instead of `total`; wrong by factor of 100 | Telemetry reports 0.85% hit rate instead of 85% | Fix normalization: divide by `total_cache_ops` only (already percentages) |
| 13 | HIGH | F01 | `baseline/runner.py:296-309` | Same broken token counting as request_generator (chunks not tokens) | All baseline metrics are wrong; baseline comparison is meaningless | Use `usage` field from response |
| 14 | HIGH | F01 | `baseline/runner.py:362-363` | Any single request failure aborts entire baseline generation | One transient network error kills the whole run | Tolerate configurable failure rate (e.g., <5%) |
| 15 | HIGH | F01 | `config/validation.py:40-48` | `validate_study_name` sanitizes instead of rejecting; allows `.._` traversal-like names | Path traversal partially possible; name collisions between different inputs | Reject invalid names with clear error; validate length, reserved names |

### MEDIUM -- Incorrect Behavior

| # | Severity | Source | File:Line | Bug Description | Impact | Fix Approach |
|---|----------|--------|-----------|-----------------|--------|--------------|
| 16 | MEDIUM | F01 | `optimization/search_space.py:63-64` | `pipeline_parallel_size` not filtered by GPU count; no TP*PP<=num_gpus constraint | Invalid parallelism configs waste trials | Add cross-parameter constraint validation |
| 17 | MEDIUM | F01 | `tuner/study_manager.py:122-147` | `peak_memory_mb` is just final memory snapshot, not actual peak | Memory reporting is misleading | Track max across GPU collector samples |
| 18 | MEDIUM | F01 | `vllm/launcher.py:117-139` | `wait_ready` elapsed time uses counter instead of `time.monotonic()` | Actual wait could far exceed configured timeout | Use `time.monotonic()` for elapsed tracking |
| 19 | MEDIUM | F01 | `vllm/launcher.py:31` | No port conflict detection | Multiple trials or instances fail without clear error | Check port availability before starting server |
| 20 | MEDIUM | F02 | `benchmarks/request_generator.py:119-133` | Streaming JSON parsing assumes one JSON object per HTTP chunk | Chunks can be split across TCP segments; JSONDecodeError silently swallowed | Use SSE parser or buffer-and-split on newlines |
| 21 | MEDIUM | F02 | `profiling/vllm_metrics.py:46` | TTFT and TPOT lists may have different lengths; `zip` silently truncates | Some TTFT measurements dropped from latency calculation | Pair measurements by request_id, not positional zip |
| 22 | MEDIUM | F02 | `profiling/vllm_metrics.py:95` | Percentile calculation has off-by-one error | p50, p95, p99 values are slightly wrong | Use standard nearest-rank or interpolation method |
| 23 | MEDIUM | F01 | `vllm/telemetry.py:105-109` | Swap counts overwritten instead of accumulated on each log match | Only last swap event captured; total swap activity lost | Accumulate: `+= swap_out` instead of `= swap_out` |
| 24 | MEDIUM | F01 | `baseline/runner.py:533-538` | `initial_memory_mb` collected after benchmark completes | "Initial" memory is actually post-benchmark memory | Collect before benchmark starts |
| 25 | MEDIUM | F01 | `baseline/runner.py:228` | `\\n` in f-string produces literal `\n` instead of newline | Prompts contain literal backslash-n instead of actual line breaks | Use `\n` (single backslash) in the f-string |
| 26 | MEDIUM | F02 | `benchmarks/alpaca.py:19` | Global `random.seed(2026)` at module import time | Seeds all random operations process-wide on import | Use local `random.Random(2026)` instance |
| 27 | MEDIUM | F05 | `config/default.yaml` | `gpu-memory-utilization: 0.6` in vllm_args conflicts with search space override | Undefined behavior: which value wins depends on command construction order | Remove from vllm_args or exclude from search space |
| 28 | MEDIUM | F05 | `examples/multi_gpu_tune.yaml` | TP=4, PP=2 combination requires 8 GPUs but only 4 configured | Invalid config would crash vLLM | Validate TP*PP <= gpu.count at config time |
| 29 | MEDIUM | F02 | `reporting/html.py` | Jinja2 template does not auto-escape; study names with HTML/JS execute as code | XSS vulnerability in generated HTML reports | Use `jinja2.Environment(autoescape=True)` |
| 30 | MEDIUM | F05 | `config/validation.py:52-65` | `create_study_dirs` returns paths for `logs/` and `configs/` but never creates them | Subsequent code that writes to these dirs fails with FileNotFoundError | Create all returned directories |
| 31 | MEDIUM | F02 | `reporting/html.py:210` | `valid_latencies` computed but never used; chart uses original list with infinities | Infinite latency values appear on charts | Use `valid_latencies` in chart data |

### LOW -- Minor Issues

| # | Severity | Source | File:Line | Bug Description | Impact | Fix Approach |
|---|----------|--------|-----------|-----------------|--------|--------------|
| 32 | LOW | F01 | `tuner/optimizer.py:347-369` | `get_top_n_results` sorts by trial number, not objective value | Returns first N trials, not best N | Sort by objective value |
| 33 | LOW | F01 | `tuner/optimizer.py:50` | Hardcoded random seed `2026` | Not configurable for reproducibility experiments | Make seed a config parameter |
| 34 | LOW | F01 | `vllm/launcher.py:49-50` | Dead variable `visible_devices` computed but never used | Code confusion; suggests incomplete implementation | Remove dead code |
| 35 | LOW | F01 | `vllm/launcher.py:153-159` | Blocking `process.wait(timeout=30)` in async `stop()` | Blocks event loop for up to 30 seconds | Use `asyncio.to_thread()` |
| 36 | LOW | F01 | `vllm/telemetry.py:37-38` | `kv_cache_utilization` and `slot_occupancy` initialized but never populated | Metrics always report 0.0 | Add regex patterns or remove fields |
| 37 | LOW | F02 | `benchmarks/alpaca.py:109-113` | O(n*m) fill-up loop: `remaining_prompts` uses list membership check | Extremely slow for large datasets | Use set for `sampled` |
| 38 | LOW | F02 | `profiling/gpu_collector.py:76-77` | Invalid GPU device IDs not removed from `self.device_ids` | Subsequent `collect()` calls fail with NVMLError | Remove invalid IDs after warning |
| 39 | LOW | F01 | `baseline/runner.py:86` | `BaselineConfig.num_requests` ignored; uses `workload.sample_size` instead | Config documentation is wrong; config field is dead | Use baseline's own `num_requests` |
| 40 | LOW | F05 | `docs/installation.md:68` | CUDA URL uses `https://download.pydantic.org/whl/cu124` (should be pytorch.org) | Users cannot install CUDA dependencies following docs | Fix URL to pytorch.org |

---

## 4. Conceptual & Theoretical Issues

### 4.1 Optimization Algorithm Correctness

**Objective weights are binary flags, not weights** (F01: optimizer.py:30-36)
The `WeightedObjectives` model requires weights summing to 100 (e.g., throughput=60, latency=30, memory=10), strongly implying weighted multi-objective optimization. However, the optimizer only checks `weight > 0` to decide whether to include each objective. The actual weight values are completely ignored. This means `(60, 30, 10)` and `(1, 1, 1)` produce identical optimization behavior. Either implement proper weighted scalarization (weighted sum of normalized objectives) or change to boolean flags.

**Pruning is configured but non-functional** (F01: optimizer.py:253-257)
The `MedianPruner` is created but `trial.report(0, 0)` reports a constant value of 0 at step 0 before the trial runs. This provides no useful information for pruning decisions. Pruning requires intermediate performance values reported during trial execution.

**No Pareto front selection for multi-objective** (F01: optimizer.py:311)
`study.best_trials[0]` returns the first element of the Pareto front arbitrarily. There is no mechanism for users to specify preferences among Pareto-optimal solutions.

**TPE vs. GP-based BO** (F03: SCOOT paper)
SCOOT demonstrates that Gaussian Process-based surrogates with MACE acquisition function ensembles outperform TPE (used by Optuna/vllm-tuner) for vLLM parameter tuning. TPE is less sample-efficient for the small evaluation budgets typical of infrastructure tuning (20-50 trials).

### 4.2 Benchmarking Methodology Flaws

**Token counting is fundamentally broken** (F01, F02)
Both `BenchmarkClient` and `VLLMBaselineRunner` count HTTP response chunks as tokens. A single chunk can contain the entire response (non-streaming) or multiple tokens. All token-based metrics are wrong:
- `throughput_tokens_per_sec` is wrong
- `tpot_times` is wrong
- Any optimization based on token throughput is unreliable

**Latency formula is mathematically incorrect** (F02: vllm_metrics.py:46)
`latency = TTFT + TPOT` treats TPOT (time *per* output token) as total decode time. The correct formula is `latency = TTFT + (TPOT * num_output_tokens)`. Combined with broken token counting, latency is doubly incorrect.

**Streaming JSON parsing is fragile** (F02: request_generator.py:119-133)
HTTP chunked transfer does not respect JSON boundaries. The code attempts `json.loads()` on each byte chunk, silently swallowing `JSONDecodeError` when objects span chunks. This leads to undercounted tokens and potentially missed TTFT measurements.

### 4.3 Statistical Rigor Gaps

**No warmup phase for benchmarking** (F04)
The benchmark sends all requests immediately. No warmup period is provided for vLLM to populate KV caches, warm up CUDA graphs, or reach steady-state scheduling. Early requests will have artificially high TTFT due to cold caches.

**No confidence intervals or error bars** (F04)
Each configuration is evaluated once. No repeated measurements, no standard deviation, no confidence intervals. Benchmark results are noisy (especially at low request counts), so single measurements can be misleading.

**No percentile reporting for SLO compliance** (F04)
The metrics tracker computes p50/p95/p99 internally but these are computed incorrectly (off-by-one in percentile calculation, based on wrong latency formula). Production SLO compliance requires correct tail latency measurement.

**No request arrival pattern modeling** (F04)
All requests fire as fast as possible up to the concurrency limit. Real-world traffic follows Poisson or bursty patterns. SCOOT uses 100-second stress tests with Poisson arrival at configurable rates -- a much more realistic evaluation methodology.

### 4.4 Multi-Objective Handling Issues

**Weights as binary flags** (described above): The 100-point weight system is cosmetic.

**No proper Pareto front**: The HTML report labels a chart "Pareto Front" but it is just a throughput-vs-latency scatter plot of all trials. A true Pareto front identifies and highlights non-dominated solutions.

**Constraint penalty inversion**: Violated constraints return `-inf` for all objectives, which is the *best* possible value for minimization objectives (latency, memory). This can cause the optimizer to prefer constraint-violating configurations.

---

## 5. Missing Features (Research-Backed)

### 5.1 From SCOOT (WWW 2025)

| Missing Feature | SCOOT Implementation | Impact | Complexity |
|-----------------|---------------------|--------|------------|
| Hidden constraint learning | Random forest classifier to predict P(feasibility) from crash history | Critical -- prevents wasting evaluations on crashing configs | Medium |
| MACE acquisition ensemble | UCB + PI + EI ensemble; select from Pareto frontier of trade-offs | High -- more robust than single TPE sampler | Medium |
| GP surrogate model | Gaussian Process with Matern 5/2 kernel instead of TPE | High -- better sample efficiency for small budgets | Medium |
| Parallel suggestion | Suggest k configs simultaneously; evaluate in parallel on separate GPUs | High -- cuts tuning time proportionally (PD=2 nearly halves time) | Low |
| Known constraint pruning | Encode vLLM parameter constraints (e.g., `max_num_batched_tokens >= max_num_seqs`) | High -- eliminates invalid configurations from search space | Low |
| Dynamic feasibility threshold | Adaptive delta that prevents both over-conservative and over-aggressive exploration | Medium -- fine-tunes near constraint boundaries | Low |
| SLO robustness testing | Post-optimization stress tests with varying arrival orders; report worst-case | Medium -- ensures worst-case guarantees | Low |
| Sobol initialization | Quasi-Monte Carlo sampling for initial exploration (better space coverage than random) | Medium -- better initial coverage | Low |
| Multi-metric MOBO (EHVI) | Expected Hypervolume Improvement for true multi-objective optimization | High -- proper TTFT+TPOT joint optimization | Medium |

### 5.2 From vLLM Docs & Community

| Missing Parameter | Default Range | Impact | Source |
|-------------------|--------------|--------|--------|
| `num_scheduler_steps` | [1, 15] | 1.8-2.7x throughput (vLLM v0.6+) | vLLM v0.6.0 blog |
| `gpu_memory_utilization` | [0.80, 0.95] | Primary memory/concurrency lever | vLLM docs |
| `max_model_len` | [256, model_max] | Memory savings for short-form tasks | vLLM docs |
| `kv_cache_dtype` | [auto, fp8] | 50% KV cache memory reduction | vLLM docs |
| `enforce_eager` | [True, False] | Debug vs. production mode | vLLM docs |
| `enable_chunked_prefill` | [True, False] | Critical for mixed prefill/decode workloads | Sarathi-Serve paper |
| `block_size` | [8, 16, 32] | PagedAttention granularity | vLLM docs |
| `speculative_model` + `num_speculative_tokens` | [None, draft] + [1, 7] | Up to 3x latency improvement at low batch sizes | ICML 2023, vLLM docs |
| Quantization method | [None, fp8, awq, gptq] | 30-50% throughput gain (FP8), 75% memory savings (AWQ/GPTQ) | vLLM docs |
| `compilation_level` | [0, 3] | torch.compile optimization; level 3 recommended for production | vLLM docs |
| `scheduler_delay_factor` | [0, 2] | Controls batch accumulation delay | SCOOT paper |

### 5.3 From LLMServingSim 2.0

| Missing Capability | LLMServingSim Implementation | BudSimulator Relevance |
|--------------------|-----------------------------|----------------------|
| Simulation-as-surrogate | Use simulator to pre-filter configs before expensive real benchmarks | Could reduce tuning time by 5-10x |
| Power modeling | 7-component model (accelerator idle/active/standby, CPU, DRAM, link, NIC, storage) | Essential for TCO analysis; GenZ has no power model |
| P/D disaggregation awareness | Separate prefill and decode MSGs with independent parallelism and KV transfer modeling | Growing deployment pattern; GenZ doesn't support this |
| Prefix caching effects | Radix-tree based multi-tier (device/CPU/CXL) prefix cache with hit-rate tracking | Affects performance significantly; GenZ doesn't model this |
| Runtime-driven simulation | Discrete-event simulation with dynamic batching, queue evolution, memory state tracking | Captures temporal effects (contention, feedback loops) that static models miss |
| MoE expert routing | Per-token expert assignment with offloading, load balancing, All-to-All communication | GenZ has limited MoE support |

---

## 6. Improvement Opportunities (Tiered)

### Tier 1 -- Critical Fixes (Results Currently Invalid)

These must be fixed before vllm-tuner results can be trusted:

1. **Fix token counting** (F01, F02): Use `response["usage"]["completion_tokens"]` from the vLLM API response instead of counting HTTP chunks. This is a 5-line fix that corrects all throughput metrics.

2. **Fix latency formula** (F02): Change `latency = TTFT + TPOT` to `latency = TTFT + (TPOT * output_tokens)`. Requires correct token counting first.

3. **Fix objective direction mismatch** (F01): The `directions` list must be filtered consistently. Use the same filtered list for study creation and `evaluate_trial` return values.

4. **Fix constraint penalty values** (F01): Return `float("inf")` for minimize directions and `float("-inf")` for maximize directions when constraints are violated.

5. **Fix duplicate field definitions** (F01): Remove the duplicate field block in `TuningConfig` (lines 137-144).

6. **Fix BaselineConfig.enabled** (F01, F05): Remove `ge=True` from the boolean field to allow disabling baseline.

7. **Fix OOM metrics count** (F01): Return only active objectives when OOM is detected, matching the filtered directions count.

8. **Remove `batch_size` from search space** (F01): This is not a vLLM parameter. Replace with additional tuning of `max_num_seqs` or `max_num_batched_tokens`.

### Tier 2 -- Algorithm Improvements

These improve optimization quality:

9. **Replace TPE with GP+MACE** (F03): Adopt HEBO as the optimization backend (SCOOT's approach) or implement GP surrogate with MACE acquisition ensemble on top of Optuna. HEBO is a pip-installable library with a clean API.

10. **Add hidden constraint learning** (F03): Train a random forest classifier on trial outcomes (success/crash) to predict feasibility of untried configurations. Use probability of feasibility (POF) to gate acquisition function suggestions.

11. **Add proper multi-objective optimization** (F03): Implement EHVI (Expected Hypervolume Improvement) for multi-objective optimization with proper Pareto front discovery. Optuna supports NSGA-II, or use BoTorch for EHVI.

12. **Implement weighted scalarization** (F01): If keeping the 100-point weight system, implement proper weighted-sum scalarization with normalized objectives. Otherwise, change to boolean objective flags.

13. **Add parallel suggestion** (F03): Allow evaluating k configurations simultaneously on different GPU sets. SCOOT shows PD=2 nearly halves tuning time.

14. **Add known constraint pruning** (F03): Encode vLLM's documented constraints (e.g., `max_num_batched_tokens >= max_num_seqs`, chunked prefill + prefix caching conflict in older versions) to prune invalid regions before BO.

15. **Add Sobol initialization** (F03): Replace random initial trials with Sobol quasi-random sampling for better initial space coverage.

### Tier 3 -- Feature Additions

16. **Add missing vLLM parameters** (F04): Expand search space with `num_scheduler_steps`, `gpu_memory_utilization`, `max_model_len`, `kv_cache_dtype`, `enforce_eager`, `enable_chunked_prefill`, `block_size`.

17. **Add proper benchmarking methodology** (F04):
    - Warmup phase (N requests before measurement)
    - Poisson or configurable arrival rate
    - Multiple runs per configuration for statistical significance
    - Confidence intervals and standard deviation
    - Correct p50/p95/p99 percentile calculation

18. **Add workload diversity** (F04):
    - Multiple workload profiles (chat, batch, long-context)
    - Workload characterization phase (prefix sharing ratio, average prompt/output lengths)
    - Workload-aware parameter recommendations

19. **Add speculative decoding tuning** (F03, F04): Include draft model selection, speculation length, and acceptance threshold as tunable parameters for latency-focused optimization.

20. **Add cost-aware BO** (F04): Track evaluation cost per trial; use cost-aware acquisition functions (EIpu) to prefer cheap evaluations early.

21. **Add transfer learning / warmstarting** (F03): Store results from previous tuning sessions; use them to warmstart new sessions on similar hardware/model combinations (inspired by Restune, SIGMOD 2021).

### Tier 4 -- Architecture Redesign

22. **Simulation-in-the-loop** (F06, F07): Use LLMServingSim 2.0 or Vidur as a fast surrogate model in the optimization loop. Phase 1: simulate to prune parameter space (minutes/config). Phase 2: run real benchmarks on top-K candidates.

23. **LLM-guided search space pruning** (F03): Use an LLM (via LLAMBO/OptunaHub or GPTuner approach) to read vLLM documentation and suggest promising initial configurations, reducing cold-start exploration.

24. **Dynamic workload adaptation** (F03): Online re-tuning when workload distribution shifts, using change detection on incoming request patterns.

25. **Co-tuning of quantization and serving params** (F03): Jointly optimize quantization method (FP8, AWQ, GPTQ) with serving parameters, since they interact significantly.

---

## 7. Research-Backed Recommendations

| # | Recommendation | Supporting Source | Key Evidence |
|---|---------------|------------------|--------------|
| 1 | Replace TPE with GP+MACE acquisition ensemble | SCOOT (WWW 2025) | GP outperforms TPE for small budgets (20-50 trials); MACE provides robustness via 3-function ensemble |
| 2 | Add random forest for hidden constraint learning | SCOOT (WWW 2025) | vLLM crashes on many invalid configs; SCOOT's POF mechanism eliminates wasted evaluations |
| 3 | Use Sobol quasi-random for initialization | SCOOT (WWW 2025) | Better initial space coverage than random sampling; SCOOT uses Sobol sequence-based QMC |
| 4 | Support EHVI for multi-objective | SCOOT (WWW 2025), BoTorch | EHVI measures potential improvement to Pareto front hypervolume; proper multi-objective optimization |
| 5 | Add `num_scheduler_steps` to search space | vLLM v0.6.0 blog | 1.8-2.7x throughput improvement; most impactful new parameter since SCOOT's evaluation |
| 6 | Use Prometheus /metrics endpoint instead of log parsing | vLLM docs | Log formats change across versions; /metrics is stable API contract |
| 7 | Implement warmup + Poisson arrival benchmarking | SCOOT (WWW 2025) | SCOOT uses 100-second stress tests with Poisson arrival; cold-start measurements are misleading |
| 8 | Add SLO robustness testing | SCOOT (WWW 2025) | Stress test best config with varying arrival orders; report worst-case metrics for SLO guarantees |
| 9 | Consider HEBO as alternative backend | HEBO (NeurIPS 2020), SCOOT | NeurIPS 2020 BO challenge winner; SCOOT built on HEBO; input/output warping handles heteroscedasticity |
| 10 | Add workload characterization before tuning | Anyscale docs, Databricks best practices | Profile workload to set informed initial parameter ranges (prefill-heavy vs decode-heavy, shared prefix ratio) |
| 11 | Use cost-aware acquisition functions | Amazon Science, BoTorch | Evaluation costs vary 10x+ across configs; EIpu weights improvement by predicted cost |
| 12 | Add transfer learning across model/hardware combos | Restune (SIGMOD 2021) | Meta-learning from previous sessions can dramatically reduce tuning time for similar setups |
| 13 | Use simulation for pre-screening | LLMServingSim 2.0 (arXiv 2602.23036), Vidur (MLSys 2024) | 0.97% error rate; can evaluate configs in minutes instead of real benchmarks taking 5-10 min each |
| 14 | Add power/energy modeling | LLMServingSim 2.0 | 7-component model with 1.34% error; essential for TCO-aware optimization |
| 15 | Support P/D disaggregation tuning | DistServe (OSDI 2024), LLMServingSim 2.0 | 7.4x more requests at same SLO; separate tuning for prefill and decode pools is a growing deployment pattern |
| 16 | Consider LLM-based warmstarting | LLAMBO (ICLR 2024), GPTuner (PVLDB 2024) | Competitive with BO in early search stages; available as OptunaHub sampler |

---

## 8. LLMServingSim 2.0 Analysis Summary

### 8.1 Architecture Highlights

**Paper**: "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure"
**Authors**: Jaehong Cho, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)
**Date**: February 2026, arXiv 2602.23036

LLMServingSim 2.0 is a discrete-event simulator for LLM serving that takes three inputs:
1. **Workload configuration** -- LLM description, request patterns, execution traces
2. **Cluster configuration** -- nodes, devices, memory hierarchy, serving policies
3. **Hardware performance profiles** -- operator-level latency/power measurements per model-device pair

Core abstractions:
- **Model Serving Group (MSG)**: Logical unit bundling {device pool + parallelism + policies + model}. Supports heterogeneous devices (GPU + PIM), separate prefill/decode MSGs, and per-operator device assignment.
- **Execution Graph**: DAG of operators with device assignments, latency estimates, and communication/memory operations. Generated per batch iteration.
- **System Simulator**: Modified ASTRA-SIM + Chakra for evaluating execution graphs with network contention, memory access latency, and bandwidth modeling.
- **Runtime Loop**: Iterative simulation where scheduling decisions depend on evolving system state (queues, memory, cache).

**Validation**: 0.97% average error vs. real systems (A6000, H100, TPU-v6e). 1.34% error in power modeling. Simulation time: 5-10 minutes per experiment.

### 8.2 Design Patterns Worth Borrowing (11)

**High Priority:**

1. **Block-based KV cache with PagedAttention semantics**: `MemoryModel.get_block_kv()` accurately models vLLM's memory management with configurable block sizes and eviction.

2. **Radix tree prefix caching**: SGLang-derived RadixCache with LRU eviction and lock_ref protection. State-of-the-art data structure for prefix sharing.

3. **7-component power model**: Accelerator (idle/active/standby) + CPU + DRAM + link + NIC + storage + other. Novel feature not found in GenZ.

4. **P/D disaggregation**: M:N prefill:decode MSG mapping with automatic layer-wise KV transfer operations.

5. **Per-layer weight calculation with TP sharding**: Comprehensive reference for tensor parallelism weight partitioning (ColumnParallel, RowParallel, VocabParallel).

**Medium Priority:**

6. **Continuous batching simulation**: Iteration-by-iteration scheduling matching real vLLM behavior.
7. **FlashAttention split heuristics**: Faithful port of FA2 `num_splits_heuristic` for analytical attention performance prediction.
8. **MoE expert routing policies**: Configurable routing (round-robin, random, proportional-load) with expert offloading.
9. **Multi-tier memory cascade**: NPU -> CPU -> CXL -> storage hierarchy with eviction/loading costs.

**Low Priority:**

10. **Component logging with node/instance context**: `ComponentLoggerAdapter` for multi-instance observability.
11. **Hierarchical placement rules**: Default -> block -> layer memory placement overrides.

### 8.3 Bugs Found in LLMServingSim 2.0 (19)

**High-Severity (7):**

| # | File | Description |
|---|------|-------------|
| 1 | main.py:401 | CXL metrics `enumerate` unpacking error (3 values from 2-tuple) |
| 2 | controller.py:14-21 | `read_wait` can hang forever (no EOF/timeout detection) |
| 3 | controller.py:23-30 | `check_end` can hang forever (same issue) |
| 4 | graph_generator.py:35 | `subprocess.run` without `check=True` -- silent failures |
| 5 | config_builder.py:367-368 | `total_npu` calculated inside per-node loop with accumulating state |
| 6 | pim_model.py:126 | PIM latency model hardcoded for Llama-3.1-8B only |
| 7 | trace_generator.py:350-352 | Hardcoded magic numbers (700,000 ns) for KV prepare overhead |

**Medium-Severity (12):**

| # | File | Description |
|---|------|-------------|
| 1 | main.py:244 | Division by zero if `log_interval` is 0 |
| 2 | scheduler.py:156-160 | O(n^2) request deletion from pool |
| 3 | scheduler.py:488 | Implicit `batch_id - 1` convention with ASTRA-SIM |
| 4 | memory_model.py:363-381 | `lock_prefix`/`unlock_prefix` raise on valid None node states |
| 5 | memory_model.py:430 | `evict_prefix_cache` uses `and` instead of `or` in guard condition |
| 6 | radix_tree.py:523-525 | `_record_store_event` parent hash logic inverted |
| 7 | radix_tree.py:420-433 | `_split_node` does not update hash values |
| 8 | graph_generator.py:34 | `cmd.split()` breaks on paths with spaces |
| 9 | config_builder.py:31 | `exit(1)` instead of raising exception on JSON error |
| 10 | config_builder.py:231-240 | Only first node's CPU memory config used in multi-node setups |
| 11 | pim_model.py:128-148 | Only 4 DRAM technologies supported; others raise KeyError |
| 12 | power_model.py:116 | Base power accumulation causes floating-point precision loss over time |

### 8.4 LLMServingSim 2.0 vs. GenZ/BudSimulator

| Aspect | GenZ (BudSimulator) | LLMServingSim 2.0 |
|--------|--------------------|--------------------|
| Approach | Analytical roofline model | Runtime-driven simulation with profiled operators |
| Accuracy | Roofline-bounded (optimistic/pessimistic) | 0.97% average error vs. real systems |
| Speed | Milliseconds | Minutes |
| Hardware required | None (analytical) | One device for profiling |
| Temporal dynamics | None (single-pass analysis) | Full: queues, batching evolution, memory state |
| Power model | None | 7-component, 3-state accelerator |
| Memory model | Static (total weight + KV cache) | Dynamic multi-tier with eviction/migration/prefix caching |
| P/D disaggregation | Not modeled | First-class citizen (M:N MSG mapping) |
| Prefix caching | Not modeled | Multi-tier radix-tree with hit-rate tracking |
| MoE | Basic parallelism | Full expert routing, offloading, dynamic load |
| Heterogeneous HW | Single device type | Mixed device pools with operator-granular offloading |
| Extensibility | Add analytical formulas | Just profile the new hardware |
| Setup complexity | Low (pip install) | High (ASTRA-SIM build, profiling) |

---

## 9. Relevance to BudSimulator

### 9.1 Features BudSimulator Should Adopt

**Priority 1 -- High Impact, Feasible:**

1. **Power Model (7-component)**
   BudSimulator has zero power/energy modeling. The 7-component breakdown from LLMServingSim 2.0 is well-defined:
   - Accelerator: 3-state (idle/active/standby) with utilization-dependent active power
   - CPU: constant + utilization-based active
   - DRAM: energy-per-bit data access
   - Link: energy-per-bit collective communication
   - NIC, Storage, Other: constant power

   Even a simplified 3-component version (accelerator + DRAM + other) would enable TCO analysis and power-aware hardware recommendations. The power model validated at 1.34% error.

2. **P/D Disaggregation Modeling**
   BudSimulator analyzes prefill and decode as separate phases on the same hardware. Adding support for M:N prefill:decode configurations with KV transfer costs would enable analysis of disaggregated architectures (DistServe, Splitwise). This is becoming a standard deployment pattern -- almost every production framework now supports it.

3. **Prefix Caching Effect on Performance**
   BudSimulator could add a prefix cache hit-rate parameter that reduces effective compute and memory access for cached prefixes. Even without runtime dynamics, a steady-state prefix hit-rate model would improve analysis accuracy for workloads with shared system prompts.
   - Production impact: Anthropic reports 90% cost reduction, 85% latency reduction for long prompts
   - vLLM V1: near-zero overhead for prefix caching (safe to enable always)

4. **Multi-tier Memory Hierarchy**
   BudSimulator's System class models on-chip and off-chip memory. Extending to device HBM / host DRAM / CXL / storage tiers with associated bandwidth parameters would enable memory offloading analysis and more accurate KV cache sizing.

**Priority 2 -- Medium Impact:**

5. **MSG Abstraction Pattern**
   Adopt an MSG-like abstraction that bundles {model + hardware + parallelism + serving policies} into a single analyzable unit. This makes the API cleaner and supports multi-instance analysis.

6. **Execution Graph Approach**
   Instead of summing operator latencies with roofline analysis, construct an explicit execution DAG with dependencies. This enables more accurate parallelism and communication overlap modeling.

7. **MoE Expert Offloading Analysis**
   Extend BudSimulator's MoE support with expert placement and offloading cost modeling, including latency of loading evicted experts from host memory.

**Priority 3 -- Nice to Have:**

8. **Profile-Based Operator Modeling (Optional Mode)**
   Support both analytical (current GenZ) and profile-based (LLMServingSim 2.0 style) operator modeling. Profiles provide higher accuracy for supported hardware; analytical remains the default for unsupported hardware.

9. **Runtime Dynamics (Future)**
   A lightweight runtime simulation mode modeling queue evolution and dynamic batching. Major engineering effort but would be a significant differentiator.

### 9.2 Critical Gap: Speculative Decoding

Neither LLMServingSim 2.0 nor BudSimulator currently models speculative decoding, which is becoming a standard serving optimization. Mirror Speculative Decoding (Apple, 2025) achieves 2.8-5.8x speedups. This is a high-priority modeling gap for both projects.

---

## 10. Priority Action Items

### Immediate (Week 1-2): Fix Results-Invalidating Bugs

1. **Fix token counting in `request_generator.py` and `baseline/runner.py`**: Use `response["usage"]["completion_tokens"]` from the vLLM API. 5-line fix per file. All throughput metrics currently wrong.

2. **Fix latency formula in `vllm_metrics.py`**: Change `TTFT + TPOT` to `TTFT + (TPOT * output_tokens)`. Requires fix #1 first.

3. **Fix objective direction mismatch in `optimizer.py`**: Use consistently filtered directions list for both study creation and value return. Currently causes Optuna exceptions.

4. **Fix constraint penalty values in `optimizer.py`**: Return `+inf` for minimize, `-inf` for maximize when constraints are violated. Currently selects constraint-violating trials as "best" for minimization.

5. **Fix duplicate fields in `config/models.py`**: Remove lines 137-144 (duplicate block).

6. **Fix `BaselineConfig.enabled` in `config/models.py`**: Remove `ge=True` from boolean field.

7. **Remove `batch_size` from search space**: Not a vLLM parameter; wastes optimization budget.

### Short-term (Week 2-4): Improve Correctness

8. **Fix streaming JSON parsing**: Use SSE parser or buffer-and-split on newlines instead of raw `json.loads()` per chunk.

9. **Fix OOM metrics to match active objective count**: Return only active objectives in OOM case.

10. **Add `num_scheduler_steps` to search space**: Single highest-impact missing parameter (1.8-2.7x throughput).

11. **Add `gpu_memory_utilization` to search space**: Primary memory/concurrency lever.

12. **Fix async/sync mismatch in `study_manager.py`**: Either make optimizer async or make benchmark sync with internal event loop.

13. **Add TP*PP <= num_gpus constraint validation**: Prevent invalid parallelism configurations.

14. **Fix HTML report XSS vulnerability**: Use `jinja2.Environment(autoescape=True)`.

### Medium-term (Month 2-3): Algorithm Improvements

15. **Add proper warmup and statistical benchmarking**: Warmup phase + multiple runs + Poisson arrival + confidence intervals.

16. **Implement weighted scalarization or change to boolean flags**: End the misleading weight system.

17. **Add known constraint pruning**: Encode vLLM parameter constraints to prune search space.

18. **Add hidden constraint learning**: Random forest classifier on trial outcomes to predict feasibility.

19. **Evaluate HEBO as alternative backend**: Run comparison on same benchmark suite; HEBO is pip-installable.

20. **Add Prometheus `/metrics` integration**: Replace fragile log parsing with stable metrics API.

### Long-term (Month 3-6): Feature Additions

21. **Add remaining vLLM parameters**: `kv_cache_dtype`, `enable_chunked_prefill`, `block_size`, speculative decoding, quantization.

22. **Implement simulation-as-surrogate**: Integrate Vidur or LLMServingSim 2.0 for fast pre-screening of configurations.

23. **Add power model to BudSimulator**: Start with 3-component (accelerator + DRAM + other), extend to full 7-component.

24. **Add P/D disaggregation to BudSimulator**: Support M:N prefill:decode analysis with KV transfer modeling.

25. **Add prefix caching to BudSimulator**: Steady-state hit-rate model with compute/memory reduction.

26. **Add parallel suggestion**: Allow evaluating multiple configurations simultaneously.

27. **Add transfer learning**: Store and reuse results across model/hardware combinations for warmstarting.

28. **Add LLM-guided initialization**: Use LLAMBO OptunaHub sampler for intelligent first guesses.

---

## Appendix A: Test Coverage Analysis

### Current State: 15-25% Effective Coverage

| Module | Test File | Tests | Tests Core Logic | Missing Critical Tests |
|--------|-----------|-------|-----------------|----------------------|
| Config Models | test_config.py | 16 | Yes | Edge cases, BaselineConfig, duplicate fields, invalid ranges |
| Search Space | test_search_space.py | 7 | Partial | Inter-param constraints, edge bounds, trial integration |
| Study Manager | test_study_manager.py | 10 | No (duplicates logic) | run_study, _run_trial, error handling, server lifecycle |
| HTML Report | test_html_report.py | 15 | Partial | XSS, empty data, markdown, chart content |
| Baseline Runner | test_baseline.py | 13 | Output structure only | Server lifecycle, benchmarking, error recovery, token counting |
| Telemetry | test_telemetry.py | 9 | Yes | Realistic log formats, file I/O, version compatibility |
| **CLI** | **None** | 0 | N/A | All commands, error handling, signal handling |
| **Request Generator** | **None** | 0 | N/A | Token counting, streaming, concurrency |
| **VLLMMetrics** | **None** | 0 | N/A | Latency calculation, percentiles |
| **GPU Collector** | **None** | 0 | N/A | NVML lifecycle, error states |
| **Export** | **None** | 0 | N/A | File I/O, format handling |
| **Dashboard** | **None** | 0 | N/A | Rich rendering |
| **Workload/Alpaca** | **None** | 0 | N/A | Dataset loading, sampling strategies |
| **Optimizer** | **None** | 0 | N/A | Optuna integration, direction handling, constraints |
| **VLLMLauncher** | **None** | 0 | N/A | Server management, health checks, cleanup |

**Key observation**: The 6 most critical modules (Optimizer, Request Generator, VLLMMetrics, VLLMLauncher, CLI, Alpaca) have **zero tests**. The existing tests primarily verify data structure shapes, not algorithmic correctness.

---

## Appendix B: Documentation Errors (17)

| # | File | Error |
|---|------|-------|
| 1 | README.md | Claims "Pareto front" in reports but chart is just a scatter plot |
| 2 | README.md | Claims "plugin system" but none exists |
| 3 | AGENTS.md | Says "prefer relative imports" but code uses absolute imports |
| 4 | AGENTS.md | Safe event loop handling example not implemented anywhere |
| 5 | TESTING.md | Claims ~53 tests but table totals 70; contradictory numbers |
| 6 | TESTING.md | Claims 60% coverage; actual critical-path coverage is 15-25% |
| 7 | cli-commands.md | `--study-name` listed as "Required" but defaults to "default_tune" |
| 8 | cli-commands.md | `--no-progress` documented but code uses `--with-progress` |
| 9 | examples/README.md | `vllm-tune` command typos (should be `vllm-tuner tune`) on lines 213, 236 |
| 10 | examples/README.md | Model name mismatches between docs and actual example files |
| 11 | examples/README.md | "multi-GPO" typo (should be "multi-GPU") on line 203 |
| 12 | reports/html-reports.md | "Locality" should be "Latency" in Pareto front description |
| 13 | reports/baseline-comparison.md | "Regenerating Baseline" command doesn't work |
| 14 | installation.md | CUDA URL points to pydantic.org instead of pytorch.org |
| 15 | index.md | Output artifact names don't match actual filenames |
| 16 | index.md | "Parameter importance analysis" mentioned but not implemented |
| 17 | custom-workload.md | Describes `dataset_path`, `prompt_template`, JSONL format -- **NONE of these exist in the code**. Entirely aspirational documentation for unimplemented features. |

---

## Appendix C: Security Vulnerabilities

| # | Severity | Location | Vulnerability | Fix |
|---|----------|----------|--------------|-----|
| 1 | MEDIUM | `reporting/html.py` | XSS via Jinja2 template without auto-escaping | Use `jinja2.Environment(autoescape=True)` |
| 2 | LOW | `config/models.py` | `vllm_args` dict allows arbitrary CLI arguments to vLLM subprocess | Validate against known vLLM parameter allowlist |
| 3 | LOW | `config/validation.py` | Study name sanitization insufficient (hidden files, reserved names, length) | Reject invalid names; add length/pattern validation |
| 4 | LOW | `reporting/html.py` | CDN dependency for Plotly JS (supply-chain risk) | Bundle Plotly or use `include_plotlyjs=True` for inline JS |

---

## Appendix D: Key References

### Primary Papers
1. Cheng et al. "SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines." WWW 2025.
2. Cho et al. "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure." arXiv 2602.23036, Feb 2026.

### LLM Serving Systems
3. Kwon et al. "PagedAttention." SOSP 2023.
4. Yu et al. "Orca: Continuous Batching." OSDI 2022.
5. Agrawal et al. "Sarathi-Serve: Chunked Prefill." OSDI 2024.
6. Zhong et al. "DistServe: P/D Disaggregation." OSDI 2024.
7. Leviathan et al. "Speculative Decoding." ICML 2023.

### Bayesian Optimization
8. Cowen-Rivers et al. "HEBO." JAIR 2022 / NeurIPS 2020 BO Challenge Winner.
9. Lao et al. "GPTuner: LLM-Guided Database Tuning." PVLDB 2024.
10. Liu et al. "LLAMBO: LLMs to Enhance BO." ICLR 2024.
11. Zhang et al. "Restune: Meta-Learning for Cloud Database Tuning." SIGMOD 2021.

### Simulators
12. Vidur: "Large-Scale LLM Serving Simulation." MLSys 2024.
13. APEX: "Extensible Dynamics-Aware Simulator." arXiv 2025.
14. LLMServingSim v1: IISWC 2024.

### vLLM Resources
15. vLLM Optimization and Tuning: https://docs.vllm.ai/en/latest/configuration/optimization/
16. vLLM v0.6.0 Performance Update: https://blog.vllm.ai/2024/09/05/perf-update.html
17. vLLM V1 Alpha Release: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
