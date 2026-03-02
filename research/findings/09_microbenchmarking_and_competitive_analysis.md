# Microbenchmarking Methodology & Competitive Tuning Tools Analysis

Comprehensive findings on LLM inference benchmarking methodology, competitive inference tuning tools, vLLM-related issues, and workload characterization techniques.

---

## 1. Microbenchmarking Methodology for LLM Inference

### 1.1 Core Metrics Definitions

Source: [BentoML LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics), [Anyscale Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)

| Metric | Definition | Phase | Measurement |
|--------|-----------|-------|-------------|
| **TTFT** (Time to First Token) | Wall-clock time from request submission to first token received | Prefill | Single event per request |
| **TPOT** (Time Per Output Token) | Mean time between consecutive output tokens | Decode | `(total_decode_time) / (output_tokens - 1)` |
| **ITL** (Inter-Token Latency) | Exact pause between two consecutive tokens | Decode | Per-token-pair measurement |
| **E2E Latency** | Total time from request to last token | Both | Single event per request |
| **Throughput** | Total tokens generated per second (system-wide) | Both | `total_output_tokens / total_time` |
| **Goodput** | Throughput of requests meeting SLA constraints | Both | `SLA_compliant_requests / total_time` |
| **Request Throughput** | Requests completed per second | Both | `completed_requests / total_time` |
| **Normalized Latency** | E2E latency / output token count | Both | Per-request, accounts for varying lengths |

**Critical distinction:** TPOT = mean(ITL) for a single request, but the distributions differ. ITL captures variance and stalls that TPOT averages away. Always report both.

### 1.2 Percentile Reporting

Source: [OptyxStack Latency Distributions](https://optyxstack.com/performance/latency-distributions-in-practice-reading-p50-p95-p99-without-fooling-yourself), [LLM Benchmarking from Scratch](https://phillippe.siclait.com/blog/llm-benchmarking-from-scratch)

**Required percentiles to report:**

| Percentile | Purpose | SLA Relevance |
|------------|---------|---------------|
| P50 (Median) | Typical user experience | Baseline; less sensitive to outliers than mean |
| P90 | Experience for bottom 10% of users | Warning threshold |
| P95 | Common SLA threshold | Standard production SLA |
| P99 | Near worst-case experience | Strict SLA for critical services |
| Mean | Overall average | Misleading alone; include only alongside percentiles |
| Std Dev | Variability measure | Indicates consistency |

**Anti-pattern: Reporting only mean values.** Averages hide tail latency. In LLM inference, tail latency is caused by long-sequence requests, preemption, and scheduling decisions. A system with mean TPOT of 30ms might have P99 TPOT of 200ms due to occasional prefill interference.

**Anti-pattern: Averaging TPOT across requests.** Different requests have different output lengths, so averaging TPOT gives disproportionate weight to short responses. Report TPOT distribution across all tokens, not across requests.

Source: [On Evaluating Performance of LLM Inference Systems (arXiv 2507.09019)](https://arxiv.org/abs/2507.09019)

### 1.3 Warmup Methodology

**Why warmup matters:**
1. **CUDA compilation**: First inference triggers JIT compilation of CUDA kernels
2. **CUDA graph capture**: First few iterations build and cache CUDA graphs
3. **KV cache allocation**: Memory pools are lazily initialized
4. **Prefix cache priming**: APC needs initial requests to populate cache
5. **Model loading**: Weights may still be loading/moving to GPU

**Recommended warmup protocol:**
```
1. Start server / initialize engine
2. Wait for "ready" signal (model loaded, KV cache allocated)
3. Send 10-50 warmup requests (matching target workload profile)
4. Use torch.cuda.synchronize() or equivalent to ensure GPU operations complete
5. Clear any accumulated metrics/counters
6. Begin measurement phase
```

**Warmup request considerations:**
- Warmup requests should match the input/output length distribution of the measurement phase
- For prefix caching evaluation: warmup with the shared prefix first, then measure with varied suffixes
- For multi-turn benchmarks: establish session context in warmup
- Discard warmup measurements entirely (do not include in statistics)

Source: [Measuring Inference Latency and Throughput](https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/measuring-inference-latency-throughput)

### 1.4 Statistical Rigor

**Minimum requirements for reporting:**

1. **Sample size**: Run at least 1000 requests for percentile stability (100 requests minimum for P50; 10,000+ for reliable P99)
2. **Multiple runs**: Conduct 3-5 independent runs per configuration to assess variability
3. **Confidence intervals**: Report 95% confidence intervals for key metrics using bootstrap resampling
4. **Steady-state measurement**: Ensure the system has reached steady state (queue length stable, throughput stable) before collecting measurements
5. **Report distribution shape**: Histograms or kernel density plots reveal multi-modal distributions that summary statistics hide

**Common statistical pitfalls:**

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Too few samples | Unstable percentiles | Min 1000 requests; 10,000+ for P99 |
| No repeat runs | Cannot distinguish signal from noise | 3-5 independent runs |
| Including warmup | Inflated tail latencies | Discard first 10-50 requests |
| Averaging across configs | Hides per-config variability | Report per-configuration distributions |
| Ignoring throughput state | Latency depends on load | Report latency at specific request rates |
| Cherry-picking metrics | Selective reporting | Always report TTFT, TPOT, ITL, throughput together |

### 1.5 Benchmarking Anti-Patterns (from Research)

Source: [On Evaluating Performance of LLM Inference Systems (arXiv 2507.09019)](https://arxiv.org/abs/2507.09019), [Meta-Metrics and Best Practices (arXiv 2508.10251)](https://arxiv.org/html/2508.10251)

**Anti-Pattern 1: Conflating Implementation and Algorithm**
LLM inference systems are complex engineering artifacts. CPU overhead alone contributes >50% of latency in some systems. When comparing systems, differences in scheduling, communication, and low-level optimization dominate over algorithmic differences. Always use the same engine version and similar engineering effort for fair comparison.

**Anti-Pattern 2: Unrepresentative Workloads**
Using only synthetic constant-length inputs hides real-world heterogeneity. Production workloads have varying input/output lengths, bursty arrivals, and shared prefixes. Always benchmark with realistic workload distributions.

**Anti-Pattern 3: Metric Normalization Hiding Variability**
Normalizing throughput by "tokens/second" can hide generation stalls. A system that generates 100 tokens, pauses for 500ms (preemption), then generates 100 more tokens has the same aggregate throughput as one that generates uniformly -- but the user experience is vastly different. Report ITL distributions to capture this.

**Anti-Pattern 4: Speculative Decoding Metrics**
Speculative decoding produces bursty, non-uniform token generation. Standard TPOT metrics are misleading because accepted speculations produce multiple tokens "instantly" while rejected ones cause visible stalls. Must report acceptance rate and adjusted ITL.

### 1.6 MLPerf Inference Methodology

Source: [MLPerf Inference v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/), [MLPerf Inference v5.0](https://mlcommons.org/2025/04/llm-inference-v5/)

**Key methodological standards from MLPerf:**

| Aspect | MLPerf Standard |
|--------|----------------|
| TPOT threshold | P99 TPOT <= 40ms (25 tok/s) for Server scenario |
| TTFT threshold | P99 TTFT constraint (model-dependent) |
| Measurement phase | 10 minutes minimum measurement after warmup |
| Output validation | Verify output quality against reference (ROUGE score) |
| Scenario types | Server (online), Offline (batch), SingleStream |
| Reproducibility | Full configuration disclosure required |

**2025 benchmarks include:** DeepSeek-R1 (up to 20,000 output tokens, avg 3,880), Llama 3.1 8B, Whisper Large V3.

**Actionable insight for vllm-tuner:** The tuner's benchmarking module should adopt MLPerf-style methodology: minimum measurement duration (not just request count), P99 thresholds for SLA validation, and output quality verification.

### 1.7 vLLM Benchmark Tooling

Source: [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/), [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)

**`vllm bench serve` parameters for load simulation:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--request-rate` | inf | Target request rate (req/s). `inf` = send all immediately |
| `--burstiness` | 1.0 | Traffic burstiness. 1.0=Poisson, <1=bursty, >1=uniform |
| `--max-concurrency` | None | Max concurrent outstanding requests |
| `--num-prompts` | 1000 | Total number of requests to send |
| `--dataset-name` | None | Dataset (sharegpt, sonnet, random, hf, custom) |
| `--random-input-len` | - | Input length for random dataset |
| `--random-output-len` | - | Output length for random dataset |
| `--sharegpt-output-len` | - | Override output length for ShareGPT |
| `--streaming` | False | Use streaming API |

**Common benchmark patterns:**

| Pattern | Parameters | Use Case |
|---------|-----------|----------|
| Max throughput | `--request-rate inf` | Find system ceiling |
| Latency at load | `--request-rate <RPS>` | Measure latency at target load |
| Concurrency sweep | `--max-concurrency 1,2,4,8,...` | Find optimal concurrency |
| SLA validation | `--request-rate <target> --burstiness 1.0` | Verify P99 meets SLA |

**What vLLM benchmarks report:**
- Mean/Median/P99 TTFT
- Mean/Median/P99 TPOT
- Mean/Median/P99 ITL
- Request throughput (req/s)
- Output token throughput (tok/s)
- Input token throughput (tok/s)

**What vLLM benchmarks do NOT report (gap for vllm-tuner):**
- Confidence intervals across multiple runs
- Bootstrap statistics
- Standard deviation of metrics across runs
- GPU utilization during benchmark
- Memory utilization during benchmark
- Preemption count and recomputation overhead

---

## 2. Competitive Analysis: Inference Tuning & Optimization Tools

### 2.1 TensorRT-LLM Auto-Tuning

Source: [NVIDIA Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html), [NVIDIA Blog: Benchmarking with TRT-LLM](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)

**Architecture:**
TensorRT-LLM provides `trtllm-bench` for benchmarking and `trtllm-serve` for deployment, with direct configuration transfer between them.

**Auto-Tuning Capabilities:**
- `trtllm-bench` automatically configures "optimal settings that generally provide good performance"
- Multiple optimization profiles can be enabled for broader kernel selection
- The `multiple_profiles` build-time flag allows TensorRT to select the most efficient kernels across different batch sizes

**Tuning Workflow:**
```
1. GPU environment setup (power limits, clock management)
2. Dataset preparation (JSON Lines format)
3. Run trtllm-bench with concurrency sweep (1 to 512)
4. Plot throughput vs. per-user speed curves
5. Identify operating point meeting SLO
6. Transfer settings to trtllm-serve
7. Verify with GenAI-Perf or benchmark_serving.py
```

**Key Tunable Parameters:**

| Parameter | Description | Tuning Approach |
|-----------|-------------|-----------------|
| `max_batch_size` | Maximum requests in flight | Sweep with trtllm-bench |
| `max_num_tokens` | Max tokens per batch iteration | Tied to memory budget |
| `tp_size` | Tensor parallelism | Based on model size vs GPU memory |
| `pp_size` | Pipeline parallelism | For multi-node |
| `ep_size` | Expert parallelism | For MoE models |
| `enable_chunked_context` | Chunked context processing | Enable for long inputs |
| FP8 quantization | Weight/activation quantization | 2.5-3x inference speedup on Hopper |
| `multiple_profiles` | Multiple TRT optimization profiles | Better kernel selection at cost of build time |

**Strengths vs. vllm-tuner:**
- Tight integration between benchmarking and deployment
- Automatic kernel selection through TRT engine optimization
- GPU-level profiling with Nsight Systems
- Well-documented workflow from benchmark to production

**Weaknesses vs. vllm-tuner:**
- No Bayesian optimization (manual sweep only)
- NVIDIA-only
- Requires model compilation (slow iteration)
- No multi-objective optimization support

### 2.2 SGLang Tuning

Source: [SGLang Hyperparameter Tuning Docs](https://docs.sglang.io/advanced_features/hyperparameter_tuning.html), [SGLang + Optuna + Hydra (Medium)](https://medium.com/@kimdoil1211/optimizing-llm-serving-speed-with-sglang-optuna-hydra-51a8eb450ef8)

**Key Tunable Parameters:**

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `--mem-fraction-static` | Heuristic | Memory fraction for model + KV cache | Reserve 5-8 GB for activations. Increment by 0.01 until OOM. |
| `--chunked-prefill-size` | - | Max tokens per prefill chunk | Decrease to 4096/2048 if prefill OOM. Trades memory for speed. |
| `--max-running-requests` | - | Max concurrent requests | Lower if decoding OOM |
| `--cuda-graph-max-bs` | - | Max batch size for CUDA graphs | Increase to 512-768 for large models |
| `--schedule-policy` | lpm | Scheduling policy | `lpm` for shared prefixes; `fcfs` otherwise |
| `--schedule-conservativeness` | 1.0 | Request acceptance aggressiveness | Decrease to ~0.3 if token usage <0.9 with queued requests |
| `--dp-size` | 1 | Data parallelism size | Prefer over TP for throughput when memory permits |
| `--tp-size` | 1 | Tensor parallelism size | Use when memory-constrained |
| `--enable-torch-compile` | False | PyTorch compilation | Enable for small models on small batch sizes |
| `--quantization` | None | Quantization method | Try `fp8` for performance gains |

**Monitoring for Tuning:**
SGLang provides real-time log output showing:
- `#running-req`: Currently processing requests
- `#token`: Total tokens in flight
- `token usage`: KV cache utilization (target >0.9)
- `#queue-req`: Waiting requests (healthy range: 100-2000)
- `gen throughput`: Generation tokens/second

**SGLang + Optuna Integration (Community):**
A community implementation exists that combines SGLang with Optuna for automated hyperparameter tuning:
- Uses Hydra for configuration management
- Each Optuna trial starts a Docker container with different SGLang arguments
- Runs benchmarks and returns latency as the optimization target
- Demonstrates feasibility of automated tuning for SGLang

**Strengths vs. vllm-tuner:**
- Built-in monitoring metrics for guided tuning
- `schedule-conservativeness` parameter provides fine-grained control
- Community Optuna integration exists (validates the approach)
- Data parallelism as first-class tuning dimension

**Weaknesses vs. vllm-tuner:**
- No built-in auto-tuning (manual or community tools)
- Optuna integration is a separate community project, not official
- Less documentation on parameter interactions

### 2.3 TGI (Text Generation Inference) Configuration

Source: [TGI GitHub](https://github.com/huggingface/text-generation-inference), [HuggingFace TGI Docs](https://huggingface.co/docs/text-generation-inference/en/index)

**Status: TGI is in maintenance mode as of December 2025.** Only minor bug fixes accepted. HuggingFace recommends vLLM or SGLang.

**Key Configuration Parameters:**

| Parameter | Description | Auto-Config |
|-----------|-------------|-------------|
| `MAX_INPUT_LENGTH` | Maximum input tokens | Auto-detected in v3 zero-config |
| `MAX_TOTAL_TOKENS` | Max total tokens (input + output) | Auto-detected |
| `MAX_BATCH_PREFILL_TOKENS` | Max tokens in prefill batch | Auto-detected |
| `MAX_BATCH_TOTAL_TOKENS` | Max tokens across all batches | Auto-detected |
| `QUANTIZE` | Quantization method | Manual |
| `NUM_SHARD` | Number of GPU shards | Manual |

**TGI v3 Zero-Config Mode:**
- Automatically selects maximal values for all batch/token parameters based on hardware
- Eliminates manual configuration for common use cases
- Focus on long-prompt optimization (3x-13x faster than vLLM on long prompts)

**Tuning approach:** TGI's philosophy is "zero-config" rather than "tunable." The system auto-detects optimal settings. This is the opposite approach to vllm-tuner's philosophy.

**Relevance to vllm-tuner:** TGI's zero-config approach validates the idea of intelligent defaults. The tuner could provide a "zero-config" mode that uses heuristics for initial configuration before Bayesian optimization refines further.

### 2.4 LMDeploy Tuning

Source: [LMDeploy GitHub](https://github.com/InternLM/lmdeploy), [LMDeploy Docs](https://lmdeploy.readthedocs.io/)

**Architecture:**
LMDeploy provides two inference engines:
- **TurboMind**: C++ engine focused on maximum performance
- **PyTorch**: Pure Python engine for developer accessibility

**Key Tunable Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cache_max_entry_count` | 0.8 | Fraction of FREE GPU memory for KV cache (after model weights) |
| `quant_policy` | 0 | KV cache quantization (0=off, 4=4-bit, 8=8-bit) |
| `max_batch_size` | - | Maximum concurrent requests |
| `tp` | 1 | Tensor parallelism |
| `session_len` | - | Maximum sequence length |

**Performance Highlights (2025):**
- Up to 1.8x higher request throughput than vLLM
- 4-bit inference is 2.4x faster than FP16
- Near-identical throughput to SGLang on H100 (16,132 vs 16,215 tok/s)
- Optimized GEMM kernels: average 19.2% performance gain over baseline
- Supports KV Cache Quant + AWQ + APC simultaneously

**Tuning approach:** LMDeploy focuses on engine-level optimization (kernel fusion, GEMM tuning) rather than user-facing parameter tuning. The `cache_max_entry_count` is the primary user-tunable parameter, analogous to vLLM's `gpu_memory_utilization`.

### 2.5 Competitive Summary

| Feature | vLLM | TensorRT-LLM | SGLang | TGI | LMDeploy |
|---------|------|-------------|--------|-----|----------|
| **Auto-tuning** | None built-in | trtllm-bench auto-config | None (community Optuna) | Zero-config v3 | None |
| **External tuning** | vllm-tuner, SCOOT | Manual sweep | Optuna+Hydra (community) | N/A | Manual |
| **Tunable params** | ~15+ | ~10+ | ~12+ | ~6 (auto) | ~5 |
| **Monitoring** | Prometheus metrics | Nsight profiling | Built-in log metrics | Basic | Basic |
| **BO-based tuning** | SCOOT (research) | None | Community project | None | None |
| **Multi-objective** | SCOOT | None | None | None | None |
| **Hardware support** | NVIDIA, AMD, CPU, TPU | NVIDIA only | NVIDIA, AMD | NVIDIA, AMD | NVIDIA |
| **Status (2025)** | Active, V1 | Active | Active, growing fast | Maintenance mode | Active |

**Key Insight:** No inference engine provides built-in Bayesian optimization or multi-objective tuning. SCOOT (for vLLM/TRT-LLM) and the SGLang+Optuna community project are the closest to what vllm-tuner aims to be. This confirms vllm-tuner fills a genuine gap in the ecosystem.

---

## 3. vLLM Tuning Ecosystem & GitHub Activity

### 3.0 jranaraki/vllm-tuner (The Subject Repository)

Source: [jranaraki/vllm-tuner](https://github.com/jranaraki/vllm-tuner)

**Repository Profile:**
- **Author**: Javad Anaraki, PhD
- **Created**: February 26, 2026
- **Stars**: 6 | **Forks**: 1 | **Issues**: 0 open | **PRs**: 0
- **Commits**: 12 total | **Contributors**: 1
- **License**: Apache-2.0 | **Language**: Python 100%
- **Status**: Early-stage, single developer

**Core Functionality:**
Intelligent parameter optimization tool for vLLM using Bayesian optimization (Optuna) with GPU metrics monitoring. Automatically tunes vLLM configuration parameters to "maximize throughput while minimizing latency and balancing memory" within user-defined constraints.

**Tunable Parameters:**
| Parameter | Default Range |
|-----------|--------------|
| `batch_size` | 1-256 |
| `gpu_memory_utilization` | 0.6-0.99 |
| `max_num_batched_tokens` | Configurable |
| `max_num_seqs` | Configurable |
| `tensor_parallel_size` | Multi-GPU |

**Architecture:**
1. YAML configuration specifying optimization goals (config/default.yaml)
2. Objective weights (throughput vs latency vs memory, must total 100)
3. Launches vLLM server with candidate parameters per trial
4. Bayesian optimization iteratively samples promising regions
5. GPU metrics and vLLM telemetry collected during each trial
6. SQLite database (optuna.db) stores trial history
7. Plotly interactive HTML report with trial progression and Pareto fronts

**Key Features:**
- Multi-objective optimization with weighted objectives
- GPU monitoring (memory utilization, vLLM-specific metrics)
- vLLM log parsing for KV cache utilization and preemption tracking
- Multi-GPU support (data-parallel and model-parallel)
- Extensible workload and plugin support

**Known Limitations (inferred from repository analysis):**
- Very early stage (12 commits, single contributor)
- No test suite visible
- No CI/CD pipeline
- Limited documentation beyond README
- No community engagement (0 issues, 0 PRs)
- No version-awareness for vLLM V0 vs V1 differences
- No crash recovery mechanism documented
- No cost-aware optimization
- Search space is limited to ~5 parameters

### 3.0.1 Competing Projects: openshift-psap/auto-tuning-vllm

Source: [auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm)

**Repository Profile:**
- **Maintainer**: Red Hat OpenShift PSAP team
- **Created**: June 9, 2025
- **Stars**: 42 | **Forks**: 14 | **Issues**: 38 open | **PRs**: 4
- **Commits**: 93 | **Version**: 0.0.1-alpha
- **License**: Apache 2.0

**Architecture:**
Distributed hyperparameter optimization framework combining:
- **vLLM** (inference engine)
- **Optuna** (optimization) with **BoTorch** (Bayesian backend)
- **Ray** (distributed computing)
- **GuideLLM** (benchmarking)
- **PostgreSQL** (persistent storage)

**CLI Interface:**
```bash
auto-tune-vllm optimize --config config.yaml --max-concurrent 2
auto-tune-vllm logs --study-id 42 --trial-number 15
auto-tune-vllm resume --study-name study_35884
auto-tune-vllm check-env --ray-cluster
```

**Key Differentiators vs jranaraki/vllm-tuner:**
| Feature | jranaraki/vllm-tuner | auto-tuning-vllm |
|---------|---------------------|------------------|
| Distributed execution | No | Yes (Ray) |
| Storage backend | SQLite | PostgreSQL |
| Benchmark integration | Custom | GuideLLM + plugins |
| Multi-objective | Weighted sum | Pareto frontier |
| BO backend | Optuna TPE | Optuna + BoTorch |
| Resume capability | Via Optuna DB | Explicit resume command |
| Community | 6 stars, 0 issues | 42 stars, 38 issues |
| Maturity | 12 commits | 93 commits |

**Known Limitation (auto-tuning-vllm):**
`--max-concurrent` parameter is not validated against available Ray cluster resources. Setting `--max-concurrent 10` on a 4-GPU cluster provides no warning; trials queue silently.

### 3.0.2 vLLM Built-In Auto-Tuning

Source: [vLLM auto_tune README](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md)

**vLLM's own benchmarks/auto_tune/ directory provides a built-in auto-tuning script:**

**Parameters Tuned:**
- `max-num-seqs`
- `max-num-batched-tokens`
- `gpu-memory-utilization` (determined by binary search starting from 0.98)

**How It Works:**
1. Memory optimization: finds highest safe `gpu-memory-utilization` (starting 0.98, stepping down on OOM)
2. Grid search: tests every combination from configured lists of `max-num-seqs` and `max-num-batched-tokens`
3. Latency-aware: for each combination, runs benchmark at infinite request rate; if P99 E2E latency exceeds threshold, iteratively reduces request rate until constraint is met
4. Tracks maximum sustainable throughput per configuration

**Configuration:**
```bash
MODEL="meta-llama/Llama-3.1-8B"
TP=1
INPUT_LEN=1024
OUTPUT_LEN=128
MAX_LATENCY_ALLOWED_MS=10000  # P99 constraint
NUM_SEQS_LIST="1 2 4 8 16 32 64 128 256"
NUM_BATCHED_TOKENS_LIST="512 1024 2048 4096 8192 16384"
```

**Outputs:**
- Detailed logs per configuration
- Summary file with best parameters and throughput
- Profiler traces (`.xplane.pb` for TPU, `.json` for GPU)

**Known Limitations:**
- Grid search only (no Bayesian optimization)
- Returns zero values if server fails to start
- Script path cannot contain "vllm" keyword (self-termination via process killing)
- No multi-objective optimization
- No interactive reporting

**Comparison: All Three Approaches:**
| Feature | jranaraki/vllm-tuner | auto-tuning-vllm | vLLM built-in auto_tune |
|---------|---------------------|------------------|------------------------|
| Search method | Bayesian (Optuna) | Bayesian (Optuna+BoTorch) | Grid search |
| Parameters | 5 | ~5+ | 3 |
| Multi-objective | Weighted sum | Pareto frontier | Single (throughput) |
| Distributed | No | Yes (Ray) | No |
| Latency constraints | Via objective | Via objective | P99 threshold |
| Reporting | Plotly HTML | PostgreSQL + logs | Text summary |
| GPU profiling | Monitoring | GuideLLM | Nsight/xprof traces |
| Crash recovery | Limited | Resume command | Skips failed configs |

### 3.1 Performance-Related Issues

Source: [vLLM Issues](https://github.com/vllm-project/vllm/issues), [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)

### 3.1 Performance-Related Issues

**AutoTune Bug (Issue #30905):**
- vLLM 0.11.2 AutoTune feature with `compile_sizes > 1` fails due to missing import in TorchInductor
- Indicates vLLM is developing internal auto-tuning capabilities
- Relevant to vllm-tuner: must handle engine-level auto-tune interactions

**Multi-Step Scheduling (Issues #6854, #9158, #7528):**
- `num_scheduler_steps` recommended at 10-15
- Known issue: bumpy ITL due to batched token returns
- V1 engine addresses this architecturally

**Performance at Scale:**
- Various issues around OOM with large batch sizes
- Preemption causing latency spikes
- CPU overhead dominating at high request rates

### 3.2 Common Troubleshooting Patterns

Source: [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)

**OOM Errors:**
- Reduce `gpu_memory_utilization` (0.85-0.90)
- Reduce `max_model_len`
- Reduce `max_num_seqs`
- Enable KV cache quantization (`--kv-cache-dtype fp8`)
- Reduce `max_num_batched_tokens`

**Slow Performance:**
- Enable CUDA graphs (`enforce_eager=False`)
- Enable prefix caching
- Increase `num_scheduler_steps` (V0)
- Use FP8 quantization
- Check CPU core count (minimum 2 + N_gpus)

**Preemption Issues:**
- Increase `gpu_memory_utilization`
- Reduce `max_model_len`
- Monitor `num_requests_waiting` and `gpu_cache_usage_perc`

### 3.3 Implications for vllm-tuner

1. **Safety bounds**: The tuner must enforce parameter constraints that prevent OOM. Conservative initial exploration, expanding bounds as safe regions are identified.
2. **Version awareness**: Parameters and their effects differ between V0 and V1. The tuner should detect vLLM version and adjust search space.
3. **Crash recovery**: Invalid configurations can crash the vLLM engine. The tuner needs robust error handling and process isolation per trial.
4. **Interaction effects**: Parameters like `gpu_memory_utilization`, `max_num_seqs`, and `max_model_len` interact non-linearly. Bayesian optimization handles this better than grid search.

---

## 4. Workload Characterization Techniques

### 4.1 Standard Benchmark Datasets

Source: [LMSYS Chatbot Arena](https://lmsys.org/projects/), [BurstGPT (arXiv 2401.17644)](https://arxiv.org/abs/2401.17644), [ServeGen (arXiv 2505.09999)](https://arxiv.org/abs/2505.09999)

| Dataset | Type | Scale | Characteristics |
|---------|------|-------|-----------------|
| **ShareGPT** | Real user conversations | ~90K conversations | Moderate prefill (user turns), short decode (assistant responses). Most commonly used for chat benchmarks. |
| **LMSYS-Chat-1M** | Real conversations | 1M conversations, 25 LLMs | Diverse models, real user behavior. Includes multi-turn sessions. |
| **BurstGPT** | Production traces | 10.31M traces, 213 days | Azure OpenAI GPT traces. Captures burstiness, failure patterns, conversation patterns. Anonymized. |
| **ServeGen** | Production traces | Billions of requests | Alibaba Cloud traces. Language, multimodal, reasoning models. Per-client workload composition. |
| **OpenThoughts** | Reasoning traces | - | Shortest prefill, longest decode. Stresses generation phase. |
| **LooGLE** | Long-context | - | Longest prefill, shortest decode. Stresses prefill phase. |
| **Sonnet** | Synthetic (poetry) | - | Fixed-length inputs. Useful for controlled experiments. |
| **Random** | Synthetic | Configurable | Fully controlled input/output lengths. For parameter sweeps. |

**Three representative workload profiles for benchmarking:**

| Profile | Prefill Load | Decode Load | Real-World Analog |
|---------|-------------|-------------|-------------------|
| **Chat** (ShareGPT) | Moderate (500-2000 tokens) | Moderate (100-500 tokens) | Chatbots, assistants |
| **Long-context** (LooGLE) | Heavy (4000-32000 tokens) | Light (50-200 tokens) | Document analysis, RAG |
| **Reasoning** (OpenThoughts) | Light (100-500 tokens) | Heavy (1000-20000 tokens) | Code generation, reasoning |

### 4.2 Workload Generation Techniques

#### 4.2.1 Input/Output Length Distribution

Source: [ServeGen (arXiv 2505.09999)](https://arxiv.org/html/2505.09999v1), [NVIDIA NIM Benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html)

**Distribution options for synthetic workloads:**

| Distribution | Use Case | Parameters |
|-------------|----------|------------|
| Constant | Controlled experiments | `mean_input_len`, `mean_output_len` |
| Normal | Typical chat workloads | `mean`, `std` for both input/output |
| Uniform | Wide coverage sweep | `min`, `max` for both |
| Log-normal | Production-realistic | `mu`, `sigma` (heavy tail) |
| Empirical (from dataset) | Production-matching | Sample from ShareGPT/LMSYS/BurstGPT |

**Key finding from ServeGen:** Using naive workload generation (e.g., constant lengths) leads to 50% under-provisioning compared to production-realistic workloads. Workload heterogeneity is critical.

#### 4.2.2 Request Arrival Patterns

Source: [LLM Inference Workload Characterization](https://arxiv.org/html/2505.09999), [BurstGPT](https://arxiv.org/html/2401.17644v3)

| Pattern | Model | Parameters | Best For |
|---------|-------|------------|----------|
| Constant rate | Deterministic | `rate` (req/s) | Baseline/controlled |
| Poisson | Exponential inter-arrivals | `lambda` (mean rate) | Standard traffic model |
| Gamma | Bursty inter-arrivals | `shape`, `rate` | Large model serving |
| Weibull | Variable burstiness | `shape`, `scale` | Mid-sized model serving |
| Trace replay | From production logs | Timestamp sequence | Most realistic |

**Key finding:** The best-fit arrival distribution varies by model/service type:
- Large models: Gamma distribution
- Mid-sized models: Weibull distribution
- Small models: Exponential distribution

**vLLM's burstiness parameter** maps to Gamma distribution shape: burstiness=1.0 is Poisson (exponential), <1.0 is bursty (Gamma with shape <1), >1.0 is more uniform.

#### 4.2.3 Session and Multi-Turn Patterns

Source: [AIBrix Workload Generator](https://aibrix.readthedocs.io/latest/features/benchmark-and-generator.html)

**Session-based workload generation:**
- Supports multi-turn conversations grouped by session ID
- Controlled prompts per session
- Session interleaving simulates real concurrent users
- Important for prefix caching evaluation (shared context across turns)

**AIBrix dataset formats:**
```json
// Plain (single-turn)
{"prompt": "XXXX"}

// Session (multi-turn)
{"session_id": 0, "prompts": ["turn1", "turn2", "turn3"]}
```

**AIBrix workload types:**
1. **Constant**: Steady QPS with controllable fluctuation
2. **Synthetic**: QPS varies based on configurable parameters
3. **Azure Trace Replay**: Replays production traces with original timestamps
4. **Mooncake Trace Simulation**: Simulates with cache block IDs

### 4.3 Workload Characterization for Tuning

**What the tuner should characterize about a workload before optimization:**

| Dimension | Why It Matters | How to Measure |
|-----------|---------------|----------------|
| Input length distribution | Determines prefill cost | Histogram of prompt token counts |
| Output length distribution | Determines decode cost, KV cache pressure | Histogram of completion token counts |
| Shared prefix ratio | Determines prefix caching benefit | Longest common prefix analysis |
| Request arrival rate | Determines batch formation | QPS measurement or trace analysis |
| Burstiness | Determines queue depth variance | Coefficient of variation of inter-arrival times |
| Session structure | Determines multi-turn context reuse | Session length distribution |
| Concurrent users | Determines max batch size needs | Peak concurrency measurement |

**Actionable insight for vllm-tuner:** The tuner should include a workload profiling step that analyzes a sample dataset or trace file to extract these dimensions. This profile then informs:
1. Initial parameter ranges (e.g., high shared prefix ratio -> enable prefix caching)
2. Benchmark configuration (matching production characteristics)
3. Objective function weights (e.g., long outputs -> weight TPOT higher)

### 4.4 Benchmark Reproducibility Checklist

**Environment documentation required:**

| Category | What to Record |
|----------|---------------|
| Hardware | GPU model, count, interconnect (NVLink/PCIe), CPU model, RAM |
| Software | vLLM version, CUDA version, driver version, PyTorch version, OS |
| Model | Model name, quantization, dtype, max_model_len |
| Configuration | All engine arguments passed |
| Workload | Dataset name, input/output length distribution, request rate, burstiness |
| Measurement | Warmup requests, measurement duration, total requests, repeat count |
| Results | Per-metric P50/P90/P95/P99, throughput, GPU utilization, memory usage |

Source: [Benchmarking LLM Serving Performance Guide](https://medium.com/@kimdoil1211/benchmarking-llm-serving-performance-a-comprehensive-guide-db94b1bfe8cf), [Anyscale Benchmarking Guide](https://docs.anyscale.com/llm/serving/benchmarking/benchmarking-guide)

---

## 5. Synthesis: Recommendations for vllm-tuner

### 5.1 Benchmarking Module Improvements

| Current Gap | Recommendation | Priority |
|-------------|---------------|----------|
| No confidence intervals | Add bootstrap CI (95%) across 3+ runs per config | High |
| No warmup protocol | Add configurable warmup (default 50 requests) with discard | High |
| Limited percentiles | Report P50/P90/P95/P99 for all metrics | High |
| No workload profiling | Add pre-tuning workload analysis step | Medium |
| No ITL distribution | Capture per-token ITL, report distribution | Medium |
| No GPU metrics | Collect GPU utilization and memory during benchmark | Medium |
| No preemption tracking | Monitor preemption events as constraint signal | Medium |
| No quality validation | Verify output correctness (ROUGE, exact match) | Low |

### 5.2 Workload Generation Improvements

| Current Gap | Recommendation | Priority |
|-------------|---------------|----------|
| Fixed workloads only | Support ShareGPT, LMSYS, custom datasets | High |
| No arrival pattern control | Support Poisson, Gamma, trace replay | High |
| No length distribution control | Support configurable input/output distributions | Medium |
| No multi-turn support | Support session-based workloads | Low |
| No workload characterization | Add dataset analysis/profiling tool | Medium |

### 5.3 Competitive Feature Gaps

| Feature | Status in vllm-tuner | Competitor Reference |
|---------|---------------------|---------------------|
| Bayesian optimization | Core feature (Optuna) | Shared with auto-tuning-vllm |
| Multi-objective optimization | Weighted sum | auto-tuning-vllm (Pareto), SCOOT |
| Distributed execution | Not supported | auto-tuning-vllm (Ray) |
| Auto-config defaults | Needed | TGI zero-config, TRT-LLM auto-config |
| Built-in monitoring | GPU metrics | SGLang log metrics |
| Benchmark-to-deployment | Needed | TRT-LLM bench->serve pipeline |
| Version-aware tuning | Needed | None (unique opportunity) |
| Crash recovery per trial | Critical | auto-tuning-vllm (resume command) |
| Cost-aware optimization | Needed | BoTorch CArBO |
| Persistent storage | SQLite | auto-tuning-vllm (PostgreSQL) |
| Benchmark framework | Custom | auto-tuning-vllm (GuideLLM) |

### 5.4 Recommended Benchmark Methodology for vllm-tuner

```
TUNING TRIAL METHODOLOGY:
1. SETUP
   - Record hardware/software environment
   - Apply trial parameters to vLLM config
   - Start vLLM engine (with process isolation)
   - Wait for engine ready signal

2. WARMUP
   - Send 50 warmup requests (matching workload profile)
   - Discard warmup metrics
   - Verify engine stability (no errors, queue empty)

3. MEASUREMENT
   - Send N requests (default 1000, configurable)
   - Use Poisson arrival (default) or configured pattern
   - Measure per-request: TTFT, TPOT, E2E latency
   - Measure per-token: ITL
   - Measure system: throughput, GPU util, memory, preemptions

4. ANALYSIS
   - Compute P50/P90/P95/P99 for all latency metrics
   - Compute mean throughput + std across 3 measurement windows
   - Check for preemption events (constraint violation)
   - Check for OOM events (infeasible config)
   - Return composite objective to optimizer

5. CLEANUP
   - Stop vLLM engine
   - Record trial results with full metadata
   - Report to Optuna study
```

---

## Key Sources Referenced

### Benchmarking Methodology
- [BentoML LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [BentoML LLM Performance Benchmarks](https://bentoml.com/llm/inference-optimization/llm-performance-benchmarks)
- [Anyscale LLM Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)
- [Anyscale Benchmarking Guide](https://docs.anyscale.com/llm/serving/benchmarking/benchmarking-guide)
- [LLM Benchmarking from Scratch](https://phillippe.siclait.com/blog/llm-benchmarking-from-scratch)
- [Benchmarking LLM Inference (Roey BC)](https://www.roeybc.com/blog/llm_inference_benchmark)
- [DigitalOcean LLM Benchmarking](https://www.digitalocean.com/blog/llm-inference-benchmarking)
- [Latency Distributions (OptyxStack)](https://optyxstack.com/performance/latency-distributions-in-practice-reading-p50-p95-p99-without-fooling-yourself)
- [Measuring Inference Latency (APXML)](https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/measuring-inference-latency-throughput)

### Research Papers
- [On Evaluating Performance of LLM Inference Systems (arXiv 2507.09019)](https://arxiv.org/abs/2507.09019)
- [Meta-Metrics and Best Practices for System-Level Inference Benchmarking (arXiv 2508.10251)](https://arxiv.org/html/2508.10251)
- [LLM-Inference-Bench (arXiv 2411.00136)](https://arxiv.org/pdf/2411.00136)
- [BurstGPT: Real-World Workload Dataset (arXiv 2401.17644)](https://arxiv.org/abs/2401.17644)
- [ServeGen: Workload Characterization (arXiv 2505.09999)](https://arxiv.org/abs/2505.09999)
- [LMSYS-Chat-1M Dataset (arXiv 2309.11998)](https://arxiv.org/html/2309.11998v4)
- [Efficient Mixed-Precision LLM Inference with TurboMind (arXiv 2508.15601)](https://arxiv.org/pdf/2508.15601)
- [Systematic Characterization of LLM Inference on GPUs (arXiv 2512.01644)](https://arxiv.org/html/2512.01644v1)

### MLPerf
- [MLPerf Inference v5.0](https://mlcommons.org/2025/04/llm-inference-v5/)
- [MLPerf Inference v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- [MLPerf Small LLM Benchmark](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
- [MLPerf Inference GitHub](https://github.com/mlcommons/inference)

### Competitive Tools
- [TensorRT-LLM Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [TensorRT-LLM Benchmarking Blog (NVIDIA)](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang Hyperparameter Tuning](https://docs.sglang.io/advanced_features/hyperparameter_tuning.html)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang + Optuna + Hydra (Medium)](https://medium.com/@kimdoil1211/optimizing-llm-serving-speed-with-sglang-optuna-hydra-51a8eb450ef8)
- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [LMDeploy GitHub](https://github.com/InternLM/lmdeploy)
- [LMDeploy Documentation](https://lmdeploy.readthedocs.io/)

### vLLM Benchmarking
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
- [vLLM Benchmark Suites](https://docs.vllm.ai/en/latest/benchmarking/)
- [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)

### Workload Generation
- [AIBrix Benchmark and Generator](https://aibrix.readthedocs.io/latest/features/benchmark-and-generator.html)
- [BurstGPT GitHub](https://github.com/HPMLL/BurstGPT)
- [ServeGen GitHub](https://github.com/alibaba/ServeGen)
- [NVIDIA NIM Benchmarking Parameters](https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html)
- [Benchmarking vLLM Serving (Medium)](https://medium.com/@kimdoil1211/benchmarking-vllm-inference-performance-measuring-latency-throughput-and-more-1dba830c5444)
- [Benchmarking LLM Serving Performance (Medium)](https://medium.com/@kimdoil1211/benchmarking-llm-serving-performance-a-comprehensive-guide-db94b1bfe8cf)

### vLLM Tuning Ecosystem
- [jranaraki/vllm-tuner](https://github.com/jranaraki/vllm-tuner) - The subject repository under analysis
- [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm) - Red Hat's distributed auto-tuning framework
- [vLLM Built-In Auto-Tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md) - vLLM's own grid-search auto-tuner
