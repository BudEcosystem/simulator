# Serving Simulation Validation Plan

Comprehensive validation methodology for the BudSimulator serving simulation subsystem,
synthesized from 40+ published papers, benchmarks, and production traces.

---

## Table of Contents

1. [Validation Framework Overview](#1-validation-framework-overview)
2. [KV Cache Memory Model](#2-kv-cache-memory-model)
3. [Power Model](#3-power-model)
4. [Workload Generator](#4-workload-generator)
5. [Batch Scheduler & Latency Estimation](#5-batch-scheduler--latency-estimation)
6. [Serving Simulator (End-to-End)](#6-serving-simulator-end-to-end)
7. [Config Optimizer](#7-config-optimizer)
8. [Prefix Cache](#8-prefix-cache)
9. [Disaggregation Analyzer](#9-disaggregation-analyzer)
10. [Cluster Analyzer](#10-cluster-analyzer)
11. [What is Hardest to Simulate](#11-what-is-hardest-to-simulate-error-source-analysis)
12. [Accuracy Targets](#12-accuracy-targets-summary)
13. [Identified Gaps & Calibration Needs](#13-identified-gaps--calibration-needs)
14. [Sources](#14-sources)

---

## 1. Validation Framework Overview

### Methodology (from Vidur, LLMCompass, TokenSim)

Every production-quality LLM serving simulator follows the same validation pattern:

1. **Select 3-5 models** spanning size range (7B, 13B, 70B, 175B/405B)
2. **Select 2-3 hardware platforms** (A100-80GB, H100-80GB, H200)
3. **Use real-world traces** for workload (ShareGPT, LMSYS-Chat-1M, Arxiv-Summarization)
4. **Compare at multiple load levels** (75%, 85%, 95% capacity)
5. **Report P50 and P95/P99 error rates** for throughput and latency metrics
6. **Show throughput vs latency tradeoff curves** vs real system measurements

### Statistical Methods

| Method | Use Case | When to Apply |
|--------|----------|---------------|
| MAPE (Mean Absolute Percentage Error) | Throughput, latency point estimates | Primary accuracy metric |
| KS Test (Kolmogorov-Smirnov) | Comparing latency distributions | Validate full CDF shape |
| R² (Coefficient of Determination) | Scaling/trend validation | Validate parameter sensitivity |
| Relative Error per Percentile | TTFT P50/P95/P99, TPOT P50/P95/P99 | SLO-relevant accuracy |

### Reference Simulator Accuracy Benchmarks

| Simulator | E2E Error | Approach | Year |
|-----------|-----------|----------|------|
| TokenSim | <1% throughput, <0.6% P50 latency | DES + transformer model | 2025 |
| Vidur | <5% at 85% load, <12% at 95% | Profiling + RF regression | 2024 |
| LLMCompass | 4.1% E2E (0.69% prefill, 7.5% decode) | Analytical tile simulation | 2024 |
| LLMServingSim | 14.7% average | HW/SW co-simulation | 2024 |
| Frontier | 19-23% E2E | Stage-centric DES | 2025 |
| **GenZ (our foundation)** | **2.73% prefill, 1.85% decode** | Analytical roofline | 2025 |
| RAPID-LLM | 10.4% E2E | Tile-based multi-level memory | 2025 |
| LUMOS | 3.3% execution time | Trace-driven replay | 2025 |
| SimAI (Alibaba) | 1.9% (98.1% alignment) | Full-stack analytical+simulation | 2025 |
| LIMINAL (NVIDIA) | 7.6% MAE | Analytical limit study | 2025 |

### GenZ Analyzer (Our Foundation) — Published Accuracy

**Critical finding**: GenZ itself has published validation data (HPCA 2025):
- Prefill: **2.73% geomean error**
- Decode: **1.85% geomean error**
- Chunked prefill: **1.43% geomean error**
- Cross-architecture: **5.82% max geomean error**
- AllReduce collectives: 2.7-3.89% error
- Hardware tested: H100 (1/2/4/8 GPU), A100, Intel Gaudi2, AMD MI300X, SambaNova SN40L
- Models tested: LLaMA2-7B, LLaMA2-13B, OPT-175B, LLaMA3-8B

**Hardware efficiency factors (empirically calibrated)**:
V100=0.45, A100=0.4, 1xH100=0.55, 2xH100=0.64, 4xH100=0.66, 8xH100=0.75, SN40L=0.9, MI300X=0.25, Gaudi2=0.6

**Our target: <10% error for throughput, <15% for tail latency** — our foundation (GenZ) already
achieves <3% for per-operator latency. The serving simulation layer adds scheduling/queuing dynamics
on top, which may introduce additional 5-10% error. LLMCompass proves 4.1% E2E is possible analytically.

---

## 2. KV Cache Memory Model

### Formula Validation

Our formula: `bytes_per_token_kv = 2 * num_kv_heads * head_dim * num_layers * precision_bytes / TP`

**Status: CONFIRMED across all major sources** (vLLM, SGLang, TensorRT-LLM, NVIDIA, LMCache).

### Ground Truth Test Cases

| Model | num_kv_heads | head_dim | layers | Precision | TP | Expected bytes/token | Source |
|-------|-------------|----------|--------|-----------|----|--------------------|--------|
| Llama-2-7B (MHA) | 32 | 128 | 32 | bf16 (2B) | 1 | 524,288 (512 KB) | NVIDIA blog, Omri Mallis |
| Llama-2-13B (MHA) | 40 | 128 | 40 | bf16 (2B) | 1 | 819,200 (800 KB) | Omri Mallis |
| Llama-2-70B (GQA) | 8 | 128 | 80 | bf16 (2B) | 1 | 327,680 (320 KB) | Multiple sources |
| Llama-3.1-8B (GQA) | 8 | 128 | 32 | bf16 (2B) | 1 | 131,072 (128 KB) | SGLang docs (exact) |
| Llama-3-70B (GQA) | 8 | 128 | 80 | bf16 (2B) | 1 | 327,680 (320 KB) | ~40GB for 128K ctx confirmed |
| Llama-3.1-405B (GQA) | 8 | 128 | 126 | bf16 (2B) | 1 | 516,096 (~504 KB) | Calculated |
| Mistral-7B (GQA) | 8 | 128 | 32 | bf16 (2B) | 1 | 131,072 (128 KB) | HuggingFace config |
| Mixtral-8x7B (GQA) | 8 | 128 | 32 | bf16 (2B) | 1 | 131,072 (128 KB) | HuggingFace config |
| Llama-3.1-8B with TP=4 | 8 | 128 | 32 | bf16 (2B) | 4 | 32,768 (32 KB) | Formula |
| Llama-3.1-8B FP8 | 8 | 128 | 32 | fp8 (1B) | 1 | 65,536 (64 KB) | Formula |

### Aggregate Validation

| Model | Context | Expected Total KV | Source |
|-------|---------|-------------------|--------|
| Llama-2-7B | 4,096 | ~2 GB | NVIDIA blog |
| Llama-3-70B | 128,000 | ~40-42 GB | Published profiling |
| Llama-2-70B MHA→GQA | any | 8x reduction | Multiple |

### Key Notes
- MoE models (Mixtral): KV cache unaffected by expert count — only shared attention matters
- head_dim = 128 for all modern Llama/Mistral/Mixtral (strong convention)
- TP must evenly divide num_kv_heads; when TP > num_kv_heads, heads are replicated
- PagedAttention achieves <4% memory fragmentation waste (vLLM)

---

## 3. Power Model

### Current Model vs Measured Reality

| Parameter | Our Model | Real Measured | Error | Source |
|-----------|-----------|---------------|-------|--------|
| Idle power | 30% TDP | 12-15% TDP | **2x overestimate** | nvidia-smi, datacenter measurements |
| Active power (inference) | 100% TDP | 60-80% TDP | **1.25-1.67x overestimate** | BLOOM-176B profiling, TokenPowerBench |
| Standby power | 50% TDP | 15-25% TDP | **2-3x overestimate** | Estimated from power state data |

### Recommended Calibration

```
P_idle = 0.125 * TDP          (was 0.30, measured A100 = 50W/400W)
P_active = 0.70 * TDP         (was 1.00, inference is memory-bound)
P_standby = 0.20 * TDP        (was 0.50)
P_utilization = P_idle + (P_active - P_idle) * utilization_fraction
```

### Energy Per Token Reference Data

| Model | Hardware | Config | Energy/Token | Source |
|-------|----------|--------|-------------|--------|
| Llama-3.1-8B | 1x H100 | batched | 0.0075 Wh/prompt | "From Prompts to Power" (2025) |
| Llama-3.3-70B | H100 | FP8, batch=128 | ~0.39 J/token | TokenPowerBench (2024) |
| Llama-3-405B | 16x H100 | TP16, FP16 | ~40 J/token | TokenPowerBench (2024) |
| Llama-3-405B | 16x H100 | TP16, FP8 | ~28 J/token (30% less) | TokenPowerBench (2024) |
| LLaMA-65B | V100/A100 | baseline | ~3-4 J/token | "From Words to Watts" (2023) |

### Server Power Breakdown (DGX H100, 10.2kW total)

| Component | % of Server Power | Our Model Component |
|-----------|-------------------|---------------------|
| GPUs (8x H100) | 55% | ACCELERATOR |
| Host CPUs | 10-12% | HOST_CPU |
| DRAM (2TB DDR5) | 5-8% | DRAM |
| NVLink/NVSwitch | 8-10% | INTERCONNECT |
| NICs (8x CX-7) | 3-5% | NIC |
| Fans/Cooling | 10-15% | COOLING |
| SSD + PSU loss | ~5% | SSD |

### Key Factors Affecting Power (TokenPowerBench)
- Batch size: 25% reduction in per-token energy from batch 32→256
- Context length: 3x increase in energy/token from 2K→10K context
- FP8 vs FP16: ~30% energy reduction
- Output tokens consume ~3.6x more energy than input tokens
- MoE (Mixtral) uses 2-3x less energy than comparable dense models

### Measurement Caveat
nvidia-smi only samples power for 25% of runtime (25ms/101ms period). Can introduce up to 39%
error in energy measurements. Corrected methodology reduces to ~5%.

---

## 4. Workload Generator

### Production Trace Statistics

| Use Case | Source | Avg Input | Avg Output | Ratio | Distribution |
|----------|--------|-----------|------------|-------|-------------|
| General Chat | LMSYS-Chat-1M | 215 | 215 | 1:1 | Lognormal |
| Multi-turn Chat | ShareGPT | ~1024 | ~415 | 2.5:1 | Lognormal, high var |
| Chat (Azure) | Splitwise Conv | ~1020 | ~129 | 8:1 | Bimodal output |
| API Chat | BurstGPT | ~575 | ~340 | 1.7:1 | Zipf in, bimodal out |
| Coding Completion | Splitwise Code | ~1500 | ~13 | 115:1 | Very skewed |
| RAG/Summarization | ArXiv/Mooncake | ~8088 | ~229 | 35:1 | Heavy-tailed input |
| Long Context | Kimi production | ~7955 | ~194 | 41:1 | Long-tailed |
| Classification | Standard | ~128 | ~16 | 8:1 | Short, fixed-ish |

### Current Presets vs Recommended

| Preset | Current (in/out) | Recommended (in/out) | Basis |
|--------|------------------|---------------------|-------|
| chat | 512/256 | 1024/128 | Azure Conversation trace |
| rag | 1024/128 | 8000/200 | Mooncake/ArXiv traces |
| coding | 768/384 | 1500/16 | Splitwise Coding trace |
| batch | 2048/512 | 2048/512 | Reasonable, keep |
| classification | 128/16 | 128/16 | Reasonable, keep |

### Missing Presets to Add
- `general_chat`: 215/215 (LMSYS-Chat-1M)
- `multi_turn_chat`: 1024/415 (ShareGPT)
- `inline_completion`: 1500/16 (Splitwise Coding)
- `long_context`: 8000/200 (Kimi production)
- `summarization`: 4096/256 (LongBench patterns)

### Arrival Pattern Validation

| Pattern | Best Fit | Parameters | Source |
|---------|----------|-----------|--------|
| General API traffic | **Gamma** | alpha=0.5, beta=2 | BurstGPT (10.31M traces) |
| Large models | Gamma | CV > 1 (bursty) | ServeGen/Alibaba |
| Medium models | Weibull | - | ServeGen/Alibaba |
| Reasoning workloads | Exponential/Poisson | CV ≈ 1 | ServeGen/Alibaba |
| Production diurnal | Sinusoidal modulation | peak:valley = 3-35x | DynamoLLM/Azure |

### Distribution Shape Validation
- **Input lengths**: Pareto + Lognormal mixture (ServeGen). Pure lognormal is reasonable approx.
- **Output lengths**: Exponential (ServeGen), NOT lognormal. Bimodal for chat (BurstGPT).
- **Correlation**: ChatGPT shows positive linear correlation. Llama shows inverse correlation.
- **Distribution drift**: Input shifts up to 1.63x over time, output up to 1.46x.

### Missing Features Identified
1. Diurnal arrival pattern (peak-to-valley configurable, 3-35x realistic)
2. Weibull distribution for arrivals
3. Exponential distribution for output lengths
4. Zipf distribution for input lengths
5. Bimodal output distribution option
6. Multi-turn conversation modeling (~3.5 turns avg, ~100s inter-turn)

---

## 5. Batch Scheduler & Latency Estimation

### Ground Truth: TensorRT-LLM Throughput (tokens/s)

#### Llama-3.1-8B FP8, H100, TP=1

| ISL | OSL | Throughput (tok/s) |
|-----|-----|--------------------|
| 128 | 128 | 26,401 |
| 128 | 2048 | 21,413 |
| 500 | 2000 | 17,571 |
| 1000 | 1000 | 14,992 |
| 2048 | 2048 | 9,462 |
| 5000 | 500 | 3,276 |
| 20000 | 2000 | 1,341 |

#### Llama-3.3-70B FP8, H100, TP=2

| ISL | OSL | Throughput (tok/s) |
|-----|-----|--------------------|
| 128 | 128 | 6,092 |
| 128 | 2048 | 5,893 |
| 500 | 2000 | 4,655 |
| 1000 | 1000 | 4,181 |
| 2048 | 2048 | 2,786 |
| 20000 | 2000 | 412 |

### vLLM Serving Latency Reference (Llama-3.1-8B, H100)

| Workload [ISL:OSL] | req/s | TTFT (ms) | TPOT (ms) |
|---------------------|-------|-----------|-----------|
| Chat [128:128] | 35.04 | 28.84 | 22.27 |
| Summarization [2024:128] | 8.95 | 200.06 | 35.51 |
| Classification [1024:30] | 19.03 | 92.74 | 22.07 |
| Creative [200:2024] | 3.42 | 47.25 | 21.57 |

### vLLM Serving Latency Reference (Llama-3.1-70B, 4xH100, TP=4)

| Workload [ISL:OSL] | req/s | TTFT (ms) | TPOT (ms) |
|---------------------|-------|-----------|-----------|
| Chat [128:128] | 15.00 | 84.84 | 30.63 |
| Summarization [2024:128] | 3.45 | 461.72 | 48.43 |
| Classification [1024:30] | 6.95 | 331.69 | 62.41 |
| Creative [200:2024] | 2.04 | 74.20 | 27.31 |

### MLPerf Inference Constraints
- TTFT <= 2s, TPOT <= 200ms
- Llama-2-70B on 8xH200: 30,128-32,790 tok/s (server scenario)

### Scaling Behaviors to Validate
- H100 FP8 is 3-4.6x vs A100 FP16 throughput
- H200 is 1.1-1.4x vs H100 (memory bandwidth advantage)
- Throughput scales 3-39x from batch=1 to batch=64 (model/GPU dependent)
- TTFT grows linearly with input length (prefill-dominated)
- TPOT is relatively stable until batch size saturates memory bandwidth

### Chunked Prefill Impact (Sarathi-Serve)
| Model | Hardware | Capacity Improvement |
|-------|---------|---------------------|
| Mistral-7B | 1x A100 | up to 2.6x |
| Yi-34B | 2x A100 | up to 2.8x |
| Falcon-180B | 8x A100 | up to 6.9x |

Optimal chunk sizes: 512 (strict SLO), 1024 (default), 2048 (relaxed SLO).

---

## 6. Serving Simulator (End-to-End)

### Validation Configuration Matrix

| Model | Hardware | TP | Precision | Workload Trace |
|-------|---------|----|-----------|----|
| Llama-3.1-8B | 1x A100-80GB | 1 | bf16 | ShareGPT |
| Llama-3.1-8B | 1x H100-80GB | 1 | fp8 | LMSYS-Chat-1M |
| Llama-2-70B | 4x A100-80GB | 4 | bf16 | ShareGPT |
| Llama-3-70B | 4x H100-80GB | 4 | fp8 | Arxiv-Summarization |

### Validation Methodology

For each configuration:

1. **Static validation**: Run simulation with fixed batch size, compare throughput (tok/s) against
   TensorRT-LLM published numbers. Target: <15% error.

2. **Dynamic validation**: Run with Poisson arrivals at 50%, 75%, 85% of max throughput.
   Compare TTFT P50/P95/P99 and TPOT P50/P95/P99 against vLLM serving benchmarks.
   Target: <15% error for P50, <25% for P99.

3. **Saturation validation**: Run at 95%+ load. Verify queue depth grows, TTFT increases,
   throughput plateaus. Qualitative behavior match.

4. **Sweep validation**: Sweep arrival rate from 10% to 100% of capacity. Plot throughput
   vs TTFT tradeoff curve. Compare shape against published curves (should show hockey-stick
   latency near saturation).

### Workload Traces for Validation (from Vidur)

| Trace | Queries | Mean Prefill | Mean Decode | P:D Ratio |
|-------|---------|-------------|-------------|-----------|
| LMSYS-Chat-1M | 2M | 786 tokens | 215 tokens | 2.3 |
| Arxiv-Summarization | 28K | 2588 tokens | 291 tokens | 15.7 |
| ShareGPT | 70K | ~1024 tokens | ~415 tokens | 2.5 |

### Key Accuracy Targets

| Metric | Target Error | Basis |
|--------|-------------|-------|
| Throughput (tok/s) | <10% | LLMCompass (4.1%), Vidur (<5%) |
| TTFT P50 | <15% | Vidur (<5% at 85% load) |
| TTFT P99 | <25% | All simulators struggle at tail |
| TPOT P50 | <15% | LLMCompass decode (7.5%) |
| TPOT P99 | <25% | Higher error expected |
| Prefill latency | <5% | LLMCompass (0.69%) |
| Decode latency | <10% | LLMCompass (7.5%) |

### Error Source Awareness
- Prefill is easier to model than decode (0.69% vs 7.5% in LLMCompass)
- Small models have higher simulation error (CPU overhead, Vidur: 12.65% for 7B at 95%)
- Skewed batch lengths cause worst errors (up to 55% for FlashAttention in Vidur)
- Communication ops are hardest (AllReduce: 14.9% error in LLMCompass)
- At 500+ requests, all simulators converge to ~12% error regardless of approach

---

## 7. Config Optimizer

### Optimal Configurations (Ground Truth)

| Model Size | Hardware | Best TP | Best PP | Batch (throughput) | Batch (latency) | Precision |
|-----------|---------|---------|---------|-------------------|-----------------|-----------|
| 7B-8B | 1x H100 | 1 | 1 | 32-64 | 1-8 | FP8 |
| 13B | 1-2x H100 | 1-2 | 1 | 32-64 | 1-8 | FP8 |
| 70B | 4x H100 | 4 | 1 | 64-128 | 8-16 | FP8 |
| 70B | 8x A100 | 4 (2 replicas) | 1 | 128 | 16-32 | BF16 |
| 405B | 8x H100 | 8 | 1 | 32-64 | 1-8 | FP8 |
| 405B | 16x H100 | 8 | 2 | 64-128 | 8-16 | FP8 |

### TP Scaling Efficiency (AMD MI300X Data)

| Transition | Throughput Gain (BS=128) | Throughput Gain (BS=16) |
|-----------|------------------------|------------------------|
| TP1→TP2 | +79.6% | +50.5% |
| TP2→TP4 | +70% | +52.6% |
| TP4→TP8 | +55.5% | +12.3% |

**Critical: TP=4 often beats TP=8 for throughput** (QwQ-32B: TP=4 = 2350 tok/s, TP=8 = 1790 tok/s).
ConfigOptimizer should prefer lower TP + data parallelism for throughput targets.

### Parameter Sensitivity Ranking (Validated)

1. **Batch Size** — HIGHEST IMPACT (14-39x throughput range)
2. **Precision/Quantization** — FP8 ~2x, INT4 ~2.7x throughput
3. **Tensor Parallelism** — Determines per-GPU load + communication overhead
4. **Sequence Length** — KV cache memory pressure
5. **KV Cache Management** — PagedAttention, prefix caching
6. **Chunk Size** — Prefill-decode interference
7. **Pipeline Parallelism** — Mainly for multi-node

Our sensitivity analysis should show this ranking.

### FP8 vs BF16 Validation
- H100 FP8 vs FP16 at BS=64: 2.2x more tokens/sec
- FP8 is effectively lossless (Llama-3.1 family, 500K+ evaluations)
- FP8 configs should dominate Pareto frontier on H100

### Pareto Frontier Shape
1. Low batch: latency-optimal, low throughput, memory-bandwidth bound
2. Increasing batch: throughput grows sublinearly
3. Saturation: throughput plateaus, latency grows linearly
4. FP8 shifts entire frontier right+down (more throughput at same latency)

### Chunked Prefill Configuration
| SLO Type | Recommended Chunk Size |
|----------|----------------------|
| Strict | 512 |
| Default | 1024 |
| Relaxed | 2048 |

---

## 8. Prefix Cache

### Cache Hit Rate Validation Targets

| Workload | Expected Hit Rate | Source |
|----------|------------------|--------|
| Few-shot learning | 85-95% | SGLang |
| Multi-turn chat | 75-90% | SGLang |
| Code analysis | 60-80% | SGLang |
| RAG (exact prefix) | 8-18% tokens | RAGCache |
| RAG (knowledge tree) | 50-70% | RAGCache/LMCache |
| Enterprise multi-tenant | 50-87% | LMCache/llm-d |
| Agent workflows | 70%+ | KVComm |

### Throughput Improvement Targets

| Scenario | Improvement | Source |
|----------|------------|--------|
| General prefix sharing | 2-5x | SGLang |
| Enterprise B2B | 2x vs cache-blind | llm-d |
| RAG pipelines | 2-4x | RAGCache/LMCache |
| Multi-turn Q&A | 7-10x | LMCache (Llama 70B) |

### TTFT Reduction Targets

| Prefix Length | Expected TTFT Reduction |
|--------------|------------------------|
| <1K tokens | 10-30% |
| 1K-8K tokens | 30-60% |
| 8K+ tokens | 60-80% |

### Eviction Policy Comparison

| Policy | Relative Performance | Source |
|--------|---------------------|--------|
| LRU (baseline, our implementation) | 1x | vLLM/SGLang |
| LRU + cache-aware scheduling | 1.9x throughput | SGLang v0.4 |
| FLOP-aware (Marconi) | 4.5-34.4x hit rate | Marconi |
| Learned (LPC) | 5-18% higher hit ratio | NeurIPS 2025 |

Our LRU implementation is the industry-standard baseline.

### Key Behaviors to Validate
1. Cache hit eliminates prefill compute proportionally (TTFT reduction ~ cached_tokens / total_tokens)
2. TPOT/ITL is unaffected by cache hits (decode is identical)
3. Higher hit rates enable larger effective batch sizes
4. Memory tradeoff: cached KV reduces space for new requests
5. No overhead when cache misses (SGLang confirmed "no noticeable overhead")

---

## 9. Disaggregation Analyzer

### P:D Ratio Validation (Splitwise)

| Workload | Optimal P:D Ratio | Source |
|----------|-------------------|--------|
| Coding (1500 in / 13 out) | 7:1 | Splitwise |
| Conversation (1020 in / 129 out) | 5:3 | Splitwise |
| Mixed (Splitwise-AA) | ~3.7:1 | Splitwise |

### Throughput Improvement Targets

| System | Model | Improvement | Source |
|--------|-------|------------|--------|
| Splitwise | LLaMA-2-70B | 1.4x throughput, 20% lower cost | ISCA 2024 |
| DistServe | OPT-13B | 2.0-3.41x goodput | OSDI 2024 |
| DistServe | OPT-66B (code) | 3.2x goodput | OSDI 2024 |
| DistServe | OPT-66B (summarize) | 4.48x goodput | OSDI 2024 |
| Mooncake | LLaMA2-70B | up to 525% throughput | FAST 2025 |
| TetriInfer | General | 1.4-2.4x perf/$ | 2024 |

### KV Transfer Overhead

| Scenario | Overhead | Source |
|----------|---------|--------|
| NVLink per-layer (A100) | ~8ms | Splitwise |
| NVLink per-layer (H100) | ~5ms | Splitwise |
| % of prefill time | <7% | Splitwise |
| % of total E2E latency | <0.1% | DistServe (OPT-175B) |
| 25Gbps cross-node | <30ms for 95% reqs | DistServe |

### When Disaggregation Helps vs Hurts

**Helps:** Large models (66B+), long context (4K+), mixed workloads with P/D interference,
high throughput + strict SLOs, workloads with very different P vs D characteristics.

**Hurts:** Small models (<13B), short contexts, limited network bandwidth,
low batch sizes, causes higher energy consumption.

### SLO Targets Used in Literature

| Application | Model | TTFT SLO | TPOT SLO |
|------------|-------|---------|---------|
| Chatbot | OPT-13B | 0.2s | 0.1s |
| Chatbot | OPT-66B | 0.4s | 0.1s |
| Chatbot | OPT-175B | 4.0s | 0.2s |
| Code Completion | OPT-66B | 0.125s | 0.2s |
| Summarization | OPT-66B | 15s | 0.15s |

---

## 10. Cluster Analyzer

### Multi-GPU Scaling Efficiency

| TP Transition | Scaling Efficiency (BS=128) | Scaling Efficiency (BS=16) |
|--------------|---------------------------|---------------------------|
| TP1→TP2 | ~80% | ~75% |
| TP2→TP4 | ~70% | ~53% |
| TP4→TP8 | ~55% | ~12% |

### Communication Overhead

- AllReduce can be up to **30% of E2E latency** on NVLink
- On PCIe/low-bandwidth: up to **65% of total latency**
- NVLink (900 GB/s) vs PCIe (32 GB/s) = 28x bandwidth difference

### Data Parallelism (multiple replicas) vs Tensor Parallelism

- Data parallelism scales nearly linearly (up to ~4 GPUs, then sublinear)
- TP is always sublinear due to AllReduce
- PCIe-connected: expect 1.4-1.6x for 2 GPUs (vs 2x ideal)
- **Key validation**: 2x TP=4 replicas often outperform 1x TP=8 for throughput

### Multi-Node Rule (vLLM docs)
- TP = GPUs per node, PP = number of nodes
- E.g., 16 GPUs / 2 nodes = TP=8, PP=2

---

## 11. What is Hardest to Simulate (Error Source Analysis)

Based on all papers reviewed, ranked by difficulty:

1. **FlashAttention / Kernel Optimizations** — Vidur shows >55% error for FlashAttention with skewed
   batches. The gap between naive and optimized kernels is the single largest source of simulation error.

2. **AllReduce / Collective Communication** — LLMCompass (14.9%) and GenZ (2.7-3.89%) report higher
   errors for AllReduce than compute operators. Network congestion and topology effects are hard to
   model analytically.

3. **Dynamic Batching / Continuous Batching** — Real serving systems use complex scheduling that changes
   behavior dynamically. Event-driven simulators handle this better than analytical models.

4. **Memory Management / Paged Attention** — Block allocation, fragmentation, and eviction policies
   create second-order effects not captured in roofline models.

5. **Tail Latency (P95/P99)** — Most simulators report mean/median well but struggle with P95/P99
   due to OS jitter, GC pauses, and scheduling variability. At 500+ requests, all simulators converge
   to ~12% error regardless of approach.

6. **Small Models at High Load** — Vidur: 12.65% error for 7B at 95% capacity due to CPU overhead
   cascading. Larger models maintain <5% error even at 95%.

### Accuracy by Simulator Approach

| Approach | Error Range | Speed | Best For |
|----------|------------|-------|----------|
| Analytical (GenZ, LIMINAL) | 2-8% per-op, 5-10% E2E | ms-seconds | Hardware DSE |
| Profile-based (Vidur, LUMOS) | 3-9% E2E | minutes | Config search |
| Event-driven (LLMServingSim, Frontier) | 9-23% E2E | min-hours | System behavior |
| Cycle-accurate (SimAI) | 1-5% | hours-days | Hardware validation |
| Hybrid ML+Analytical | Improves R² by 12% | seconds | Cloud prediction |

---

## 12. Accuracy Targets Summary

### Per-Component Accuracy Targets

| Component | Metric | Target Error | Validation Method |
|-----------|--------|-------------|-------------------|
| Memory Model | bytes/token KV | **0% (exact)** | Formula comparison |
| Memory Model | total KV for context | <5% | Published profiling data |
| Power Model | Active power | <20% | TDP vs measured |
| Power Model | Energy/token | <30% (after calibration) | TokenPowerBench data |
| Power Model | Component breakdown | <25% | DGX H100 specs |
| Workload Generator | Arrival pattern shape | KS test p>0.05 | BurstGPT Gamma fit |
| Workload Generator | Length distributions | KS test p>0.05 | Production traces |
| Batch Scheduler | Per-batch latency | <15% | TensorRT-LLM numbers |
| Simulator | Throughput (tok/s) | <10% | TRT-LLM + vLLM benchmarks |
| Simulator | TTFT P50 | <15% | vLLM serving benchmarks |
| Simulator | TTFT P99 | <25% | Published P99 data |
| Simulator | TPOT P50 | <15% | vLLM serving benchmarks |
| Config Optimizer | Optimal TP recommendation | Exact match | Published best configs |
| Config Optimizer | Sensitivity ranking | Top-3 match | Published analysis |
| Prefix Cache | Hit rate (known workload) | <10% abs error | SGLang benchmarks |
| Disaggregation | P:D ratio recommendation | Within 2x | Splitwise data |
| Disaggregation | Throughput gain direction | Correct sign | Published gains |
| Cluster | Scaling efficiency trend | Correct monotonic decrease | AMD TP data |

---

## 13. Identified Gaps & Calibration Needs

### Critical Calibrations Required

1. **Power Model Idle/Active Fractions**
   - Change idle from 0.30 to 0.125 TDP
   - Change active from 1.0 to 0.70 TDP
   - Change standby from 0.50 to 0.20 TDP
   - Add utilization-dependent power: `P = P_idle + (P_peak - P_idle) * util`

2. **Workload Presets**
   - Fix coding preset: 768/384 → 1500/16 (Azure Splitwise data)
   - Fix chat preset: 512/256 → 1024/128 (Azure Conversation data)
   - Fix rag preset: 1024/128 → 8000/200 (Mooncake/ArXiv data)
   - Add 5 new presets (general_chat, multi_turn_chat, inline_completion, long_context, summarization)

3. **Output Length Distribution**
   - Current: uses same distribution as input (lognormal)
   - Should default to: exponential distribution (ServeGen finding)

### Missing Features for Production Accuracy

4. **Diurnal arrival pattern** — 3-35x peak-to-valley ratios observed in production
5. **Weibull arrival distribution** — best for medium-sized models (ServeGen)
6. **Utilization-dependent power model** — linear interpolation between idle and active
7. **PUE multiplier** — facility cooling overhead (1.09 Google → 1.5 avg)
8. **Multi-turn conversation modeling** — 3.5 turns avg, 100s inter-turn time

### Known Limitations (Acceptable)

9. **Roofline-based latency** — may have 10-20% error vs profiling-based (LLMCompass shows 4.1% is achievable)
10. **No FlashAttention kernel-level modeling** — affects accuracy at skewed batch lengths
11. **Static hardware specs** — doesn't model DVFS or thermal throttling
12. **No network contention modeling** — assumes dedicated interconnect bandwidth

---

## 14. Sources

### Papers
- **Splitwise** — Patel et al., ISCA 2024. Prefill-decode disaggregation on Azure.
- **DistServe** — Zhong et al., OSDI 2024. Disaggregated prefill and decoding.
- **Mooncake** — Qin et al., FAST 2025 Best Paper. KV cache centric disaggregation.
- **TetriInfer** — Hu et al., 2024. Multi-GPU disaggregated inference.
- **Vidur** — Agrawal et al., MLSys 2024. LLM inference simulation.
- **LLMCompass** — Zhang et al., ISCA 2024. Hardware evaluation framework.
- **LLMServingSim** — Kim et al., MLArchSys/ISCA 2024. HW/SW co-simulation.
- **TokenSim** — 2025. High-accuracy DES simulator.
- **Frontier** — 2025. Next-gen disaggregated system simulator.
- **Sarathi-Serve** — Agrawal et al., 2024. Chunked prefill scheduling.
- **SGLang/RadixAttention** — Zheng et al., LMSYS 2024. Prefix caching.
- **PagedAttention/vLLM** — Kwon et al., SOSP 2023. Paged KV cache.
- **Orca** — Yu et al., OSDI 2022. Continuous batching.
- **BurstGPT** — Wang et al., KDD 2025. 10.31M production traces.
- **ServeGen** — Alibaba, 2025. Billion-request workload characterization.
- **DynamoLLM** — Azure, HPCA 2025. Dynamic energy management.
- **TokenPowerBench** — arXiv:2512.03024. Power benchmarking.
- **"From Words to Watts"** — arXiv:2310.03003. Energy measurement.
- **"From Prompts to Power"** — arXiv:2511.05597. Prompt-level energy.
- **SCOOT** — Ant Group, WWW 2025. Config optimization.
- **AlpaServe** — OSDI 2023. Automatic parallelization.
- **SpotServe** — ASPLOS 2024. Preemptible instance serving.
- **ChunkAttention** — ACL 2024. Prefix tree attention.
- **CacheGen** — SIGCOMM 2024. KV cache compression.
- **LMCache** — arXiv:2510.09665. Enterprise KV cache layer.
- **Marconi** — Amazon, 2024. Hybrid LLM prefix caching.
- **RAGCache** — 2024. RAG-specific caching.
- **FlexGen** — Sheng et al., 2023. Throughput-oriented offloading.

- **GenZ-LLM Analyzer** — Bambhaniya et al., HPCA 2025. Roofline-based LLM performance modeling.
- **TokenSim** — 2025. High-accuracy DES simulator (<1% throughput error).
- **RAPID-LLM** — arXiv:2512.19606. Tile-based latency prediction.
- **LUMOS** — MLSys 2025. Training performance replay (3.3% error).
- **SimAI** — Wang et al., NSDI 2025. Full-stack AI simulation (98.1% alignment).
- **LIMINAL** — NVIDIA, arXiv:2507.14397. Analytical inference limit study (7.6% MAE).
- **SCOOT** — Ant Group, WWW 2025. Bayesian config optimization for serving.

### Benchmarks & Documentation
- **TensorRT-LLM Performance Overview** — nvidia.github.io/TensorRT-LLM/performance/
- **MLPerf Inference v4.1** — mlcommons.org/2024/03/mlperf-llama2-70b/
- **vLLM v0.6.0 Performance Update** — blog.vllm.ai/2024/09/05/perf-update.html
- **E2E Cloud GPU Benchmarks** — docs.e2enetworks.com
- **vLLM A100 Benchmark** — databasemart.com/blog/vllm-gpu-benchmark-a100-80gb
- **BentoML LLM Benchmark** — bentoml.com/blog/benchmarking-llm-inference-backends
- **SGLang vs vLLM on Llama-3** — lmsys.org/blog/2024-07-25-sglang-llama3/
- **AMD ROCm TP Analysis** — rocm.blogs.amd.com
- **NVIDIA H100 vs A100** — nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html
- **llm-d KV Cache Scheduling** — llm-d.ai/blog/kvcache-wins-you-can-see
- **Epoch AI GPU Power Analysis** — epochai.org

### Datasets for Validation
- **LMSYS-Chat-1M** — huggingface.co/datasets/lmsys/lmsys-chat-1m (1M conversations)
- **ShareGPT** — huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered (~70K)
- **BurstGPT** — github.com/HPMLL/BurstGPT (10.31M traces, 213 days)
- **Azure/Splitwise Traces** — via Splitwise paper supplementary
- **OpenORCA** — MLPerf Inference standard dataset (24,576 samples)
