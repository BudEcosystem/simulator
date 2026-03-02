# Analysis: Configuration, Tests, Documentation, and Edge Cases

## Part 1: Configuration Files Analysis

---

### 1.1 `config/default.yaml`

**What it covers:**
- Model specification (Qwen1.5-0.5B-Chat-AWQ)
- Single GPU setup
- Balanced objective weights (60/30/10)
- Alpaca workload with 100 samples
- Baseline generation enabled
- Optuna study settings with 25 trials, 24h timeout
- Extra vllm_args for max-model-len and gpu-memory-utilization

**Issues found:**

1. **BUG: `gpu-memory-utilization: 0.6` conflicts with search space.** The `vllm_args` sets `gpu-memory-utilization: 0.6`, but the search space has `gpu_memory_utilization: null` (use defaults). The `vllm_args` value is passed directly to the vLLM server command. If the optimizer also sets `gpu_memory_utilization` as a tunable parameter, there will be a conflict -- which value wins? The `_build_vllm_command` in the launcher likely appends both, and the last `--gpu-memory-utilization` flag wins in vLLM's argparser. This is **undefined behavior** depending on command construction order.

2. **`storage_backend: "sqlite:///studies/optuna.db"`** uses a relative path. This means the database location depends on the working directory when the CLI is invoked. Could create databases in unexpected locations.

3. **No `max_tokens` in workload** -- actually present (256). Fine.

4. **`timeout_minutes: 1440` (24 hours)** is very aggressive for a default config. Most users doing quick testing would want much shorter. The documentation shows this as the default, which could surprise users.

5. **Missing `baseline.num_requests` and `baseline.max_tokens`** -- these exist in the model with defaults (1000 and 256), but the default config doesn't explicitly set them. The simple_tune.yaml does set them. Inconsistency.

### 1.2 `examples/simple_tune.yaml`

**What it covers:**
- GPT-2 model for quick iteration
- Fewer trials (10), shorter timeout (30 min)
- Smaller workload (50 samples, 5 concurrent)
- Baseline with explicit num_requests/max_tokens
- Empty vllm_args

**Issues:**

1. **Model mismatch with docs.** The docs examples README (docs/user-guide/examples/README.md) shows the simple example using `Qwen/Qwen1.5-0.5B-Chat-AWQ`, but the actual `examples/simple_tune.yaml` uses `gpt2`. Inconsistency between docs and actual example files.

2. **`n_startup_trials: 3` but only `min_trials: 10`.** With 3 startup trials (random exploration) and only 10 total, the optimizer has only 7 trials for Bayesian optimization. This is extremely few for meaningful optimization.

3. **`constraints.max_memory_utilization: 0.9`** is set, but the search space doesn't override `gpu_memory_utilization`. The default search space goes up to 0.95 or 0.99 (depending on code), which could conflict with the constraint.

### 1.3 `examples/latency_optimized.yaml`

**What it covers:**
- Latency-first objectives (30/70/0)
- Strict latency constraint (100ms)
- Smaller search ranges for low-latency operation
- Lower concurrency (5 concurrent)

**Issues:**

1. **Missing `max_tokens` in workload section.** Unlike other configs, this doesn't set `max_tokens`. It defaults to 256 from the model. Should be explicit for a latency-focused config since `max_tokens` directly impacts latency.

2. **Missing `baseline` section entirely.** Will use defaults (enabled=true, num_requests=1000). But for a quick latency test, 1000 baseline requests is excessive. Should override.

3. **`search_space.batch_size: [1, 64]` but `search_space.max_num_seqs: [16, 128]`.** For latency optimization, having `max_num_seqs` up to 128 while `batch_size` caps at 64 creates an imbalance. If `max_num_seqs > batch_size`, vLLM may not be able to use all sequence slots.

### 1.4 `examples/multi_gpu_tune.yaml`

**What it covers:**
- Llama-2-7b-hf on 4 GPUs
- Throughput-heavy objectives (70/20/10)
- Full multi-GPU search space (TP 1-4, PP 1-2)
- Weighted prompt distribution
- Extra vllm_args for dtype and trust-remote-code

**Issues:**

1. **`tensor_parallel_size: [1, 2, 4]` and `pipeline_parallel_size: [1, 2]`.** These are list-type search space overrides (categorical). But `TP * PP` must not exceed `gpu.count` (4). The combination `TP=4, PP=2` requires 8 GPUs but only 4 are available. No cross-parameter constraint validation exists in the config model.

2. **`trust-remote-code: true` in vllm_args.** This is a security concern -- it enables arbitrary code execution from the model repository. Should have a warning in docs.

3. **`dtype: float16` is vllm_args key.** The hyphen convention (`trust-remote-code`) is inconsistent with the underscore convention elsewhere. This is because vLLM uses hyphens in CLI args. Correct but potentially confusing.

4. **`sample_size: 200` with `concurrent_requests: 20`.** For multi-GPU, this is a reasonable ratio, but the benchmark runs all 200 requests. With 4 GPUs, the effective throughput per request is lower due to parallelism overhead for small models. No guidance on sizing.

---

## Part 2: `pyproject.toml` Analysis

**What it covers:**
- Build system (setuptools)
- Python 3.10+ requirement
- All runtime and dev dependencies
- Entry point `vllm-tuner = "src.cli.main:app"`
- Code quality tools (black, ruff, mypy) configuration

**Issues:**

1. **`vllm` is NOT in dependencies.** It's expected to be installed separately. This should be either an optional dependency or explicitly documented as a prerequisite. Currently it's only mentioned in the README and installation docs.

2. **Entry point `src.cli.main:app`** -- the entry point references a Typer app object. But `cli/main.py` uses absolute imports (`from src.config.validation import ...`). This works when installed as a package but not when running directly as `python src/cli/main.py`.

3. **Dependency version ranges are lower-bounded only** (e.g., `optuna>=3.5.0`). No upper bounds. A breaking change in optuna 4.x or pydantic 3.x could break the tool silently.

4. **`pynvml>=11.5.0`** is a required dependency, not optional. On systems without NVIDIA GPUs (e.g., CI runners), this makes the entire package fail to install or import. Should be optional.

5. **`datasets>=2.16.0` and `transformers>=4.37.0`** are heavy dependencies (several GB with models). They're required even if the user never uses the Alpaca workload. Should be optional.

6. **No `pytest-asyncio` mode configuration.** The tests use async fixtures but there's no `asyncio_mode = "auto"` in pyproject.toml's `[tool.pytest.ini_options]`. Tests may need explicit `@pytest.mark.asyncio` decorators.

7. **Missing `[tool.pytest.ini_options]` section.** No test markers, test paths, or asyncio mode defined.

---

## Part 3: Unit Test Analysis

### 3.1 `tests/unit/test_config.py` (16 tests)

**What it covers:**
- GPUConfig default values and device_ids validation
- WeightedObjectives valid weights, invalid sum, single objective
- Constraints validation (valid, invalid memory >1)
- SearchSpaceOverride basic usage
- WorkloadConfig defaults and custom values
- StudySettings defaults
- TuningConfig defaults and GPU validation
- YAML config loading from `config/default.yaml`
- Study name validation (valid, slashes, empty)
- TunerSettings defaults

**Critical gaps NOT tested:**

1. **No test for `SearchSpaceOverride` with invalid ranges** (e.g., min > max, negative values)
2. **No test for `WorkloadConfig` with `sample_size=0`** -- Pydantic should reject (ge=1), but untested
3. **No test for `Constraints` with `max_latency_ms` negative** -- Pydantic has `ge=0`, but untested
4. **No test for `TuningConfig` duplicate field definitions** -- the actual `models.py` has `gpu`, `objectives`, `constraints`, `search_space`, `workload`, `study`, and `vllm_args` defined TWICE in `TuningConfig` (lines 126-146). This is a BUG in the source code. The second definitions overwrite the first. The `baseline` field is only in the first block (line 132) and could be shadowed.
5. **No test for WeightedObjectives with negative weights** -- Pydantic has `ge=0`, but no explicit test
6. **No test for `load_yaml_config` with missing file** -- should raise `FileNotFoundError`
7. **No test for `load_yaml_config` with invalid YAML** -- malformed YAML content
8. **No test for `load_yaml_config` with YAML that has extra/unknown fields** -- Pydantic behavior with extra fields
9. **No test for `create_study_dirs`** -- directory creation behavior
10. **No test for `validate_study_name` with only invalid chars** -- e.g., `"///"` becomes `"___"` which is valid
11. **No test for the config's `vllm_args` field** -- arbitrary dict accepted without validation
12. **`test_load_yaml_config` uses hardcoded path `"config/default.yaml"`** -- test depends on CWD being the repo root. Fragile.
13. **No test for `BaselineConfig`** model validation (enabled, num_requests, max_tokens bounds)

**Bugs in tests:**

1. **`test_tuning_config_defaults` (line 127):** Tests `config.study.min_trials == 20` but the `default.yaml` sets `min_trials: 25`. Since this test creates a default `TuningConfig()` (not from YAML), the default is 20. This is correct but confusing -- the "default" in code differs from the "default" in config file.

### 3.2 `tests/unit/test_search_space.py` (7 tests)

**What it covers:**
- Default search space parameters for single GPU (batch_size, max_num_seqs, gpu_memory_utilization present; TP/PP absent)
- Multi-GPU search space (TP and PP present)
- Parameter bounds retrieval
- Categorical parameter retrieval
- Valid parameter validation
- Invalid parameter validation (batch_size=1000 out of range)
- Search space override from config

**Critical gaps NOT tested:**

1. **No test for inter-parameter constraints** (e.g., `TP * PP <= num_gpus`)
2. **No test for search space with 0 GPUs**
3. **No test for search space with negative GPU count**
4. **No test for edge values at bounds** (e.g., batch_size=1, batch_size=256)
5. **No test for `gpu_memory_utilization` bounds** (should be 0.0-1.0)
6. **No test for what happens when override specifies reversed range** (e.g., `batch_size: (256, 1)`)
7. **No test for trial parameter suggestion** (integration with Optuna trial)
8. **No test for the actual parameter names list** -- only checks membership, not completeness
9. **No test for `max_num_batched_tokens`** parameter

### 3.3 `tests/unit/test_study_manager.py` (10 tests)

**What it covers:**
- StudyManager initialization
- Average memory calculation from GPU history (multi-GPU, single GPU, empty, mixed)
- Study summary retrieval
- Best config retrieval (with and without results)
- Combined metrics include average_memory_mb

**Critical gaps NOT tested:**

1. **No test for `run_study()`** -- the core async method that orchestrates everything. The actual study execution workflow is completely untested.
2. **No test for trial execution** -- `_run_trial()` is not tested at all
3. **No test for vLLM server lifecycle** -- server start, health check, shutdown
4. **No test for benchmark execution** -- request sending, metrics collection
5. **No test for error handling** -- what happens when vLLM server crashes, OOM, timeout
6. **No test for prompt loading** -- dataset loading and sampling
7. **No test for Optuna integration** -- study creation, trial parameter suggestion, objective reporting
8. **No test for pruning behavior** -- early trial termination
9. **No test for concurrent request handling** -- semaphore behavior
10. **Tests duplicate logic instead of testing actual methods.** Tests like `test_average_memory_calculation` manually reproduce the memory averaging logic rather than calling the actual method. If the implementation changes, the test won't catch regressions.

### 3.4 `tests/unit/test_html_report.py` (15 tests)

**What it covers:**
- Generator initialization with and without baseline
- Baseline data loading from JSON file and missing file
- Improvement calculations (positive/negative throughput, latency, P95/P99, memory delta)
- Report file creation (HTML exists, has .html extension)
- Baseline comparison section presence/absence
- Failed trial filtering in charts
- Convenience function `generate_html_report`

**Critical gaps NOT tested:**

1. **No test for XSS vulnerability** -- study names or parameter values containing HTML/JS are not tested for proper escaping
2. **No test for empty trials data** -- what does the report look like with `trials_data=[]`?
3. **No test for trials with inf/NaN metric values** in the report rendering
4. **No test for Plotly chart content** -- tests verify charts dict keys exist but not chart content
5. **No test for very large number of trials** -- performance/rendering
6. **No test for `_load_baseline_data()` with YAML baseline file** -- only JSON tested
7. **No test for baseline with zero values** -- division by zero in improvement calculations
8. **No test for report with all trials failed** -- should produce report without charts
9. **No test for markdown report generation** via `_generate_markdown_summary`
10. **No test verifying HTML is valid/parseable** -- only string checks

### 3.5 `tests/unit/test_baseline.py` (13 tests)

**What it covers:**
- BaselineMetrics initialization and to_dict (empty and with samples)
- VLLMBaselineRunner initialization with vllm_params verification
- Prompt slicing with warmup (non-overlapping warmup/main prompts)
- GPU collector initialization check
- Division by zero in text summary
- YAML and JSON output structure

**Critical gaps NOT tested:**

1. **No test for `_build_vllm_command()`** -- the command construction is untested
2. **No test for `_start_vllm_server()`** -- server startup is untested
3. **No test for `_wait_server_ready()`** -- health check polling is untested
4. **No test for `_send_request()`** -- the actual HTTP request sending is untested
5. **No test for `_run_benchmark()`** -- concurrent benchmark execution is untested
6. **No test for `_stop_server()`** -- graceful/forced shutdown is untested
7. **No test for `run()` end-to-end workflow** -- the full workflow is untested
8. **No test for `generate_baseline_from_config()`** -- the public API function
9. **No test for baseline when vLLM fails to start**
10. **No test for baseline with request failures** (the RuntimeError("Aborting") path)
11. **`test_prompt_slicing_with_warmup` modifies runner attributes directly** (lines 100-102) instead of using config, which tests implementation not interface
12. **`test_division_by_zero_in_text_summary`** tests that `total_memory_mb: 0` doesn't crash, which is good. But the actual code in `src/baseline/runner.py` handles this correctly with a guard.

**Bugs in tests:**

1. **`test_vllm_params_full_construction` (line 93):** Asserts `gpu_memory_utilization` is a `str`. This is because the actual runner reads it from `config.vllm_args.get("gpu-memory-utilization", "0.6")` which returns a string. But this is inconsistent -- the vLLM CLI parameter should be a float. The test documents this bug rather than catching it.

### 3.6 `tests/unit/test_telemetry.py` (9 tests)

**What it covers:**
- Empty log parsing
- Throughput extraction from logs
- Decode latency extraction
- Prefill latency extraction
- OOM error detection
- Multiple metrics from combined logs
- Parser reset
- Convenience function `parse_vllm_logs`
- OOM detection function `detect_oom_from_logs`

**Critical gaps NOT tested:**

1. **No test for malformed log lines** -- partially matching patterns
2. **No test for multiple throughput values** -- which one wins?
3. **No test for log parsing with actual vLLM log format** -- the test strings are simplified; real vLLM logs have timestamps, logger names, etc.
4. **No test for preemption detection** from logs
5. **No test for KV cache utilization** parsing (mentioned in README features)
6. **No test for log file reading** -- `parse_vllm_logs(Path("/nonexistent.log"))` doesn't test actual file reading
7. **`test_detect_oom_from_logs` (line 89-96) is misleading.** It creates a parser, sets OOM on the parser, then calls `detect_oom_from_logs(Path("/nonexistent.log"))` which returns `False`. The test asserts `False` for the function but `True` for the parser. This tests that the function handles non-existent files, but the name suggests it tests OOM detection.

---

## Part 4: Source Code `models.py` Deep Issues

### 4.1 Duplicate Field Definitions (CRITICAL BUG)

In `src/config/models.py` lines 120-146, `TuningConfig` has fields defined TWICE:

```python
class TuningConfig(BaseModel):
    model: str = ...
    gpu: GPUConfig = ...          # Line 126
    objectives: WeightedObjectives = ...  # Line 127
    constraints: Constraints = ...   # Line 128
    search_space: SearchSpaceOverride = ...  # Line 129
    workload: WorkloadConfig = ...   # Line 130
    study: StudySettings = ...       # Line 131
    baseline: BaselineConfig = ...   # Line 132

    vllm_args: dict = ...           # Line 134

    gpu: GPUConfig = ...            # Line 137 (DUPLICATE)
    objectives: WeightedObjectives = ...  # Line 138 (DUPLICATE)
    constraints: Constraints = ...   # Line 139 (DUPLICATE)
    search_space: SearchSpaceOverride = ...  # Line 140 (DUPLICATE)
    workload: WorkloadConfig = ...   # Line 141 (DUPLICATE)
    study: StudySettings = ...       # Line 142 (DUPLICATE)

    vllm_args: dict = ...           # Line 144 (DUPLICATE)
```

The second `gpu` through `vllm_args` definitions overwrite the first. Importantly, `baseline` is only in the first block. In Pydantic v2, the last field definition wins. Since `baseline` isn't redeclared, it should survive. But this is clearly a copy-paste error that could cause confusion and unexpected behavior if the two blocks diverge.

### 4.2 `BaselineConfig.enabled` has `ge=True`

```python
enabled: bool = Field(default=True, ge=True, ...)
```

`ge=True` on a `bool` field makes no sense. Booleans are not comparable with `>=`. In Python, `True >= True` is `True` and `False >= True` is `False`, so this effectively prevents setting `enabled=False`. This is a **BUG** -- baseline cannot be disabled via config.

### 4.3 `validate_study_name` Allows Path Traversal

The function replaces `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|` with `_`. But it doesn't prevent:
- Names starting with `.` (hidden directories)
- Names that are `..` (parent directory traversal after replacement)
- Names with spaces or null bytes
- Very long names exceeding filesystem limits

---

## Part 5: Documentation Analysis

### 5.1 `README.md`

**Accuracy issues:**
1. Claims "Extensibility: Custom workloads and plugins" but no plugin system exists in the code.
2. Claims "Pareto front" in reports but the chart is just a scatter plot (not a true Pareto front).
3. Installation uses `uv venv` and `uv pip` but this is uv-specific. Standard `pip install -e .` also works but requires Python 3.10+.
4. CLI example `vllm-tuner tune --config config/default.yaml --study-name my_study` -- the `--study-name` is technically optional (defaults to "default_tune"), but the docs say it's required.

### 5.2 `AGENTS.md`

**Accuracy issues:**
1. Import style guideline says "Prefer relative imports within packages" but `cli/main.py` uses absolute imports (`from src.config.validation import ...`). Inconsistency between guideline and actual code.
2. Type annotations guideline says "Prefer modern style `list[str]` over legacy `List[str]`" but actual code mixes both styles (e.g., `gpu_collector.py` uses `List[int]`, `vllm_metrics.py` uses `list[float]`).
3. String quotes guideline says "Single quotes unless string contains single quote" but actual code predominantly uses double quotes.
4. Safe event loop handling example is shown but **not implemented anywhere** in actual code. `cli/main.py` uses `asyncio.run()` without the recommended check.

### 5.3 `TESTING.md`

**Accuracy issues:**
1. Claims 70 total tests but actual test files contain: test_config (16) + test_search_space (7) + test_telemetry (9) + test_baseline (13) + test_html_report (15) + test_study_manager (10) = 70. This matches, but the individual file counts listed in TESTING.md are slightly off (says 16, 7, 9, 13, 15, 10 = 70).
2. Coverage table shows "Current" percentages but these appear to be estimates, not measured. The CLI coverage of 20% is likely accurate given no CLI tests exist.
3. CI workflow section suggests GitHub Actions but no `.github/workflows/` directory exists in the repo.
4. TESTING.md says `Expected: ~53 tests, all passing` under "Run All Unit Tests" but the table says 70. Contradictory.

### 5.4 `docs/user-guide/cli-commands.md`

**Accuracy issues:**
1. `--study-name` is listed as "Required" for the `tune` command, but actual code defaults to "default_tune" if not provided.
2. `--gpu-count` short flag is listed as `-c` which conflicts with `--config` which is also `-c`. Typer should catch this conflict but the docs don't mention it.
3. Lists `--no-progress` but actual code uses `--with-progress` (default False). The docs describe the opposite flag.
4. No mention of `--baseline/--no-baseline` flags.

### 5.5 `docs/user-guide/configuration.md`

**Accuracy issues:**
1. Shows `baseline.num_requests` and `baseline.max_tokens` in config example, but doesn't explain interaction with `workload.sample_size`.
2. Missing documentation for `workload.name` field.
3. Missing documentation for `study.n_startup_trials` field.

### 5.6 `docs/user-guide/examples/README.md`

**Accuracy issues:**
1. **Command typos:** Line 213: `vllm-tune --config ...` should be `vllm-tuner tune --config ...`.
2. **Command typos:** Line 236: `vllm-tune --config ...` should be `vllm-tuner tune --config ...`.
3. **Multi-GPU example uses `device_ids: [0, 1]` with `count: 2`** but references `meta-llama/Llama-2-7b-chat-hf` -- different model than `examples/multi_gpu_tune.yaml` which uses `meta-llama/Llama-2-7b-hf` (no chat suffix).
4. Simple tune example shows `Qwen/Qwen1.5-0.5B-Chat-AWQ` but actual `examples/simple_tune.yaml` uses `gpt2`.
5. Configuration patterns show `batch_size: [128, 512]` but don't explain whether this is inclusive/exclusive bounds.
6. Objectives example omits the `memory` weight, showing only throughput+latency summing to 100. This is valid but inconsistent with the "must sum to 100" documentation.
7. **Typo on line 203:** "More trials for multi-GPO" should be "multi-GPU".

### 5.7 `docs/user-guide/reports/html-reports.md`

**Accuracy issues:**
1. "Pareto Front (Throughput vs. Locality)" -- "Locality" should be "Latency".
2. Report location says `reports/<study_name>/report.html` which matches code.

### 5.8 `docs/user-guide/reports/baseline-comparison.md`

**Accuracy issues:**
1. "Regenerating Baseline" section suggests `python -m src.baseline.runner` but this module doesn't have a `__main__.py` and wouldn't work as a standalone command without arguments.
2. Memory improvement labeling: says "+6.25%" for memory going up, which is actually **worse**. The docs need to clarify that for memory, positive percentage means regression.

### 5.9 `docs/user-guide/installation.md`

**Accuracy issues:**
1. Line 68: CUDA 12.4 URL uses `https://download.pydantic.org/whl/cu124` -- this is clearly wrong. Should be `https://download.pytorch.org/whl/cu124`. Typo: `pydantic` instead of `pytorch`.
2. Shows `pip install -e ".[dev]"` alongside `uv pip install`. Users might mix package managers.

### 5.10 `docs/user-guide/index.md`

**Accuracy issues:**
1. Output artifacts section says `best_params.json` and `study_summary.json` but actual output uses `best_config.yaml`, `best_config.json`, `summary.json`, `trials.json`. Names don't match.
2. HTML report location: `<study_name>_report.html` but actual is `report.html` inside a directory.
3. Export example has space in study name: `--study-name my Study` -- this would fail unless validated (replaced with underscore).
4. "Parameter importance analysis" mentioned in report features but this is NOT implemented in the code.

### 5.11 `docs/architecture/tuning-engine.md`

**Accuracy issues:**
1. Shows `direction="maximize"` but the actual optimizer uses multi-objective with multiple directions.
2. Shows `batch_size: [1, 256]` in search space but actual default range in code is unverified (need to check `search_space.py`).
3. Mentions `MedianPruner` but doesn't explain when pruning occurs or its impact.
4. Latency formula `avg_latency_ms = total_latency_ms / num_completed` -- the actual code calculates latency as `TTFT + TPOT` per request (which is wrong, as documented in findings 02).

### 5.12 `docs/troubleshooting/oom-errors.md`

**Accuracy issues:**
1. Line 12: Missing number prefix. "4 -" instead of "4." (formatting error).
2. Swap advice (creating 16GB swapfile) is questionable for GPU OOM errors -- swap doesn't help with GPU VRAM exhaustion.

### 5.13 `docs/user-guide/examples/custom-workload.md`

**Accuracy issues:**
1. Shows `dataset_path`, `dataset_dir`, `dataset_file`, `min_prompt_length`, `max_prompt_length`, `prompt_template` config fields -- **NONE of these exist in the actual `WorkloadConfig` model.** The model only supports `name`, `dataset_name`, `sample_size`, `prompt_length_distribution`, `warmup_requests`, `concurrent_requests`, `max_tokens`. This entire doc page describes features that DON'T EXIST.
2. "Option 2: Custom YAML Format" with inline `prompts:` list -- not implemented.
3. JSONL format section -- not implemented in any loader.
4. This is the most misleading documentation page. It describes a completely aspirational feature set.

### 5.14 `docs/developer-guide/api-reference.md`

**Issues:**
1. Extremely sparse. Only lists class names with no method signatures, parameters, return types, or examples.
2. Missing entire modules: profiling, benchmarks, reporting, CLI, optimization.
3. No actual API documentation -- just a list of class names.

---

## Part 6: Cross-Cutting Issues

### 6.1 Config-to-Code Inconsistencies

| Config Key | Config File | Pydantic Model | Code Usage | Issue |
|---|---|---|---|---|
| `search_space.batch_size` | `[1, 256]` (list) | `Optional[tuple[int, int]]` | Used as range bounds | YAML list parsed as tuple -- works via Pydantic coercion |
| `search_space.tensor_parallel_size` | `[1, 2, 4]` (list) | `Optional[List[int]]` | Categorical choices | Correct |
| `study.storage_backend` | `sqlite:///studies/optuna.db` | `str` | Passed to Optuna | Relative path -- CWD dependent |
| `baseline.enabled` | `true` | `bool` with `ge=True` | Checked in CLI | **BUG**: `ge=True` prevents `false` |
| `objectives` weights | Must sum to 100 | Validated by `check_sum` | Used as weights | Correct |
| `vllm_args` keys | Hyphenated (`max-model-len`) | Plain `dict` | Prepended with `--` | Correct but no validation |
| `workload.max_tokens` | 256 | `int` with `ge=1` | Passed to request | No upper bound validation |
| `gpu.count` | Integer | `int` with `ge=1` | Auto-derived from device_ids | Redundant -- always overwritten |

### 6.2 Default Value Concerns

| Parameter | Default | Concern |
|---|---|---|
| `study.min_trials` | 20 (code) / 25 (default.yaml) | Inconsistency. Which is the "real" default? |
| `study.timeout_minutes` | 60 (code) / 1440 (default.yaml) | 24 hours is excessive for default config |
| `workload.sample_size` | 100 | Too few for statistical significance |
| `workload.concurrent_requests` | 10 | May overwhelm small GPUs |
| `search_space` all `null` | Use code defaults | User has no visibility into what ranges are actually used |
| `gpu_memory_utilization` search range | Likely [0.6, 0.95] in code | Upper bound of 0.95+ is dangerous for stability |
| `baseline.num_requests` | 1000 | Very long for quick testing; takes significant time |

### 6.3 Validation Gaps

Things that SHOULD be validated but are NOT:

1. **`TP * PP <= gpu.count`** -- No cross-parameter constraint
2. **`model` name validity** -- No check if model exists on HuggingFace
3. **`vllm_args` key validity** -- No check against known vLLM parameters
4. **`vllm_args` value types** -- All values stored as-is (could be wrong type)
5. **`search_space` range validity** -- No check that min < max for tuple ranges
6. **`sample_size <= dataset size`** -- Only checked at runtime, not config time
7. **`concurrent_requests > 0` and reasonable** -- `ge=1` exists but no upper bound warning
8. **Port conflicts** -- No check if port 8000 is available before starting server
9. **Disk space** -- No check before creating potentially large study databases

### 6.4 Test Coverage Gap Summary

| Module | Has Tests | Tests Actually Test Core Logic | Missing Tests |
|---|---|---|---|
| Config Models | Yes | Yes | Edge cases, BaselineConfig, duplicate fields |
| Search Space | Yes | Partially | Inter-param constraints, edge bounds |
| Study Manager | Yes | No (tests duplicate logic) | run_study, _run_trial, error handling |
| HTML Report | Yes | Partially | XSS, empty data, markdown |
| Baseline Runner | Yes | Output structure only | Server lifecycle, benchmarking, errors |
| Telemetry | Yes | Yes | Realistic log formats, file I/O |
| CLI | No | N/A | All commands, error handling, signals |
| Request Generator | No | N/A | Token counting, streaming parsing |
| VLLMMetrics | No | N/A | Latency calculation, percentiles |
| GPU Collector | No | N/A | NVML lifecycle, error states |
| Export | No | N/A | File I/O, format handling |
| Dashboard | No | N/A | Rich rendering |
| Workload/Alpaca | No | N/A | Dataset loading, sampling |
| Optimizer | No | N/A | Optuna integration |
| VLLMLauncher | No | N/A | Server management |

**Estimated actual test coverage for critical paths: 15-25%** (versus the 60% claimed in TESTING.md). The existing tests mostly verify data structures and output formats, not actual behavior.

---

## Part 7: Edge Cases Summary

### Configuration Edge Cases

1. **Empty YAML file** -- `yaml.safe_load` returns `None`, `TuningConfig(**None)` raises `TypeError`
2. **YAML with only `model` field** -- All other fields use defaults. Works but may produce surprising behavior
3. **`objectives` summing to 0** -- Not caught by `ge=0` individual validators but caught by `check_sum != 100`
4. **`search_space` with single-element tuple** -- e.g., `batch_size: [64]` -- Pydantic expects tuple of 2
5. **`vllm_args` with boolean values** -- YAML `true` becomes Python `True`, then `str(True)` becomes `"True"` in command line, which may not be valid for vLLM
6. **Study name with unicode characters** -- Allowed by `validate_study_name` (only ASCII special chars filtered). Could cause filesystem issues on some systems
7. **Multiple studies with same name** -- Creates/reuses same database. Could corrupt previous study data

### Runtime Edge Cases

1. **No internet access** -- `load_dataset("tatsu-lab/alpaca")` fails. No offline fallback.
2. **vLLM version incompatibility** -- No version check. API endpoints may differ.
3. **GPU driver mismatch** -- pynvml may fail with driver/library version mismatches.
4. **Disk full during study** -- SQLite write failures, JSON dump failures. No pre-checks.
5. **Keyboard interrupt during baseline** -- vLLM server orphaned (no signal handler).
6. **Multiple vllm-tuner instances** -- Port 8000 conflict. No process locking.
7. **System clock change during benchmark** -- Negative latency values possible.
