# 18. Serving Simulation Validation Plan

**Document:** 18 -- Comprehensive Validation Plan for Serving Simulation
**Date:** 2026-03-01
**Status:** Final
**Scope:** Reference data, test cases, accuracy targets, and continuous validation methodology for all serving simulation components

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Workload Generator Validation](#2-workload-generator-validation)
3. [Throughput and Latency Validation](#3-throughput-and-latency-validation)
4. [KV Cache and Memory Model Validation](#4-kv-cache-and-memory-model-validation)
5. [Prefix Caching Validation](#5-prefix-caching-validation)
6. [Prefill-Decode Disaggregation Validation](#6-prefill-decode-disaggregation-validation)
7. [Power and Energy Model Validation](#7-power-and-energy-model-validation)
8. [Configuration Optimizer Validation](#8-configuration-optimizer-validation)
9. [Simulator Accuracy Methodology](#9-simulator-accuracy-methodology)
10. [Continuous Validation Framework](#10-continuous-validation-framework)

---

## 1. Executive Summary

This document synthesizes research from 8 research tracks (#31-38) and 2 additional deep-dive tasks (#45, #37) into a concrete validation plan for the BudSimulator serving simulation. For each component, we provide:

- **Reference data sources** with specific numbers from published papers
- **Validation test cases** with expected ranges
- **Accuracy targets** (% error thresholds) based on state-of-the-art simulator benchmarks
- **Methodology** for continuous validation as the system evolves

### Target Accuracy Summary

| Component | Target Error | Gold Standard | Our Approach |
|-----------|-------------|---------------|--------------|
| E2E throughput | <10% | Vidur <5%, LLMCompass 4.1% | Analytical roofline |
| E2E latency (P50) | <10% | TokenSim 0.6% | Analytical + calibration |
| E2E latency (P99) | <15% | All simulators ~12% at scale | Analytical + calibration |
| Prefill latency | <5% | LLMCompass 0.69% | Roofline (compute-bound) |
| Decode latency | <10% | LLMCompass 7.5% | Roofline (memory-bound) |
| Memory estimation | <5% | Well-understood formulas | Analytical |
| Power estimation | <10% | Published TDP curves | Polynomial model |
| Workload distributions | Qualitative match | BurstGPT, ServeGen | Statistical validation |

---

## 2. Workload Generator Validation

### 2.1 Reference Data Sources

#### Arrival Pattern Validation

| Source | Dataset Size | Arrival Model | Key Parameters |
|--------|-------------|---------------|----------------|
| BurstGPT (KDD 2025) | 10.31M traces, 213 days | **Gamma distribution** | alpha=0.5, beta=2, CV>1 |
| ServeGen/Alibaba (2025) | Billions of requests, 4 months | **Varies by model size** | Large: Gamma; Medium: Weibull; Small: Exponential |
| DynamoLLM/Azure (HPCA 2025) | 1 week Azure traces | **Diurnal + bursty** | Conv: 3.3x peak/valley; Code: 34.6x peak/valley |
| BurstGPT conversation | 0.16M traces | Gamma | 0.019 RPS average |
| BurstGPT API | 4.81M traces | Gamma | 0.21 RPS average |

#### Input Length Distribution Validation

| Source | Use Case | Mean Input | Median Input | Distribution Shape |
|--------|----------|-----------|-------------|-------------------|
| LMSYS-Chat-1M (ICLR 2024) | General chat | 214.5 | -- | Lognormal |
| ShareGPT (~70K convos) | Multi-turn chat | ~1024 | -- | Lognormal, high variance |
| Splitwise/Azure Conv (ISCA 2024) | Chat service | -- | 1,020 | Wide range |
| Splitwise/Azure Code (ISCA 2024) | Code completion | -- | 1,500 | Skewed (large code context) |
| BurstGPT balanced | API chat | 575 | -- | Zipf (theta=1.1) |
| BurstGPT Azure-Conv | Conversation | 749 | -- | Zipf |
| Mooncake/Kimi (FAST 2025) | Long context | 7,955 | -- | Heavy-tailed |
| ServeGen/Alibaba (2025) | Language models | -- | -- | **Pareto + Lognormal mixture** |
| Vidur Chat-1M trace | Chat | 786 | -- | -- |
| Vidur Arxiv-4K trace | Summarization | 2,588 | -- | -- |
| Vidur BWB-4K trace | Book/web | 1,067 | -- | -- |

#### Output Length Distribution Validation

| Source | Use Case | Mean Output | Median Output | Distribution Shape |
|--------|----------|------------|--------------|-------------------|
| LMSYS-Chat-1M | General chat | 214.5 | -- | Lognormal |
| ShareGPT | Multi-turn chat | ~415 | -- | Lognormal |
| Splitwise/Azure Conv | Chat service | -- | 129 | Bimodal |
| Splitwise/Azure Code | Code completion | -- | 13 | Very short |
| BurstGPT balanced | API chat | 340 | -- | Bimodal |
| BurstGPT Azure-Conv | Conversation | 232 | -- | Bimodal |
| Mooncake/Kimi | Long context | 194 | -- | -- |
| ServeGen/Alibaba | Language models | -- | -- | **Exponential** |
| Vidur Chat-1M | Chat | 215 | -- | -- |
| Vidur Arxiv-4K | Summarization | 291 | -- | -- |
| Vidur BWB-4K | Book/web | 1,612 | -- | -- |

### 2.2 Validation Test Cases

#### TC-WL-01: Arrival Pattern Statistical Tests
```
For each arrival pattern (poisson, gamma, bursty, constant):
  1. Generate 10,000 inter-arrival times
  2. Fit to theoretical distribution using MLE
  3. Run Kolmogorov-Smirnov test: p-value > 0.05
  4. Verify CV within expected range:
     - Poisson: CV = 1.0 (+/- 0.05)
     - Gamma(0.5, 2): CV = 1/sqrt(0.5) = 1.414 (+/- 0.1)
     - Bursty: CV > 1.5
     - Constant: CV < 0.05
```

#### TC-WL-02: Preset Token Length Validation
```
For each preset, generate 10,000 requests and verify:
  - Mean input length within 20% of reference
  - Mean output length within 20% of reference
  - Distribution shape matches expected (KS test against lognormal/exponential)
  - Min/max bounds respected
  - No negative or zero lengths generated

Expected reference values for updated presets:
  chat:           input ~1024, output ~128  (Azure Conversation)
  general_chat:   input ~215,  output ~215  (LMSYS-Chat-1M)
  multi_turn:     input ~1024, output ~415  (ShareGPT)
  rag:            input ~8000, output ~200  (Mooncake/ArXiv)
  coding:         input ~1500, output ~16   (Azure Coding)
  batch:          input ~2048, output ~512  (synthetic)
  classification: input ~128,  output ~16   (synthetic)
  long_context:   input ~8000, output ~200  (Kimi production)
```

#### TC-WL-03: Input-Output Correlation
```
Generate 10,000 chat requests and verify:
  - Pearson correlation coefficient between input and output length
  - Expected: weak positive correlation (r = 0.1-0.4) for chat
  - Expected: near-zero or negative for long-context (r < 0.1)
  Reference: BurstGPT shows linear correlation for ChatGPT, inverse for Llama
```

#### TC-WL-04: Multi-turn Conversation Properties
```
If multi-turn is implemented:
  - Average turns per conversation: ~3.5 (ServeGen DeepSeek-R1)
  - Average inter-turn time: ~100 seconds (ServeGen)
  - Input length grows with turns (context accumulation)
  - Multi-turn fraction: ~9.6% of total requests (ServeGen)
```

### 2.3 Accuracy Targets

| Metric | Target | Method |
|--------|--------|--------|
| Arrival rate CV | Within 10% of theoretical | KS goodness-of-fit |
| Mean token lengths | Within 20% of reference | Statistical comparison |
| Distribution shape | KS p-value > 0.05 | Goodness-of-fit test |
| Bounds enforcement | 100% compliance | Hard assertion |

---

## 3. Throughput and Latency Validation

### 3.1 Reference Data Sources

#### Published Benchmark Results

| Source | Model | Hardware | TP | Throughput | TTFT | TPOT | Conditions |
|--------|-------|----------|-----|-----------|------|------|-----------|
| Vidur (MLSys 2024) | LLaMA2-7B | A100-80GB | 1 | Profiled | <200ms P90 | -- | Chat-1M, 85% load |
| Vidur | LLaMA2-70B | 4xA100 | 4 | Profiled | <2s P90 | <200ms P99 | Chat-1M, 85% load |
| Vidur | Qwen-72B | 4xA100 | 4 | Profiled | <2s P90 | <200ms P99 | Multiple traces |
| DistServe (OSDI 2024) | OPT-13B | -- | -- | -- | 200ms | 100ms | ShareGPT chatbot |
| vLLM (SOSP 2023) | Various | A100 | -- | 2-4x Orca | -- | -- | ShareGPT + Alpaca |

#### Simulator Cross-Validation Targets

| Simulator | E2E Error | Prefill Error | Decode Error | Conditions |
|-----------|-----------|--------------|--------------|-----------|
| Vidur | <5% at 85% load | -- | -- | 4 models, 2 HW platforms |
| LLMCompass | 4.1% E2E | 0.69% prefill | 7.5% decode | GPT-3 175B on A100 |
| TokenSim | <1% throughput | -- | -- | LLaMA2-7B on A100 |
| LLMServingSim | 14.7% | -- | -- | GPT-3/LLaMA on RTX 3090 |
| All simulators | ~12% | -- | -- | At 500+ requests |

#### Per-Operator Error Targets (from LLMCompass, ISCA 2024)

| Operator | Target Error | LLMCompass Measured |
|----------|-------------|-------------------|
| GEMM/Matmul | <12% | 9.0% |
| Softmax | <15% | 12.0% |
| LayerNorm | <15% | 11.3% |
| GELU/Activation | <8% | 5.0% |
| All-reduce | <18% | 14.9% |

### 3.2 Validation Test Cases

#### TC-TL-01: Roofline Model Accuracy
```
For each operator type (GEMM, Attention, FFN):
  For each hardware (A100, H100):
    1. Calculate roofline-predicted latency
    2. Compare against published profiling data
    3. Verify compute-bound vs memory-bound classification matches known behavior:
       - Prefill GEMM: compute-bound for large batch
       - Decode GEMV: memory-bound
       - Attention: memory-bound for decode, compute-bound for long prefill
    Target: <12% error per operator
```

#### TC-TL-02: E2E Throughput Validation
```
Scenario: LLaMA2-70B on 4xA100-80GB, TP=4
Workload: Chat-1M trace (mean prefill 786, mean decode 215, P:D ratio 2.3)
Load levels: 50%, 75%, 85%, 95% of capacity

Validate:
  - Throughput within 10% of Vidur's published results
  - Throughput degrades gracefully as load increases
  - Throughput at 85% load > 90% of peak capacity
```

#### TC-TL-03: Latency Distribution Validation
```
At 85% load:
  - TTFT P50 error < 10% vs reference
  - TTFT P99 error < 15% vs reference
  - TBT P50 error < 10% vs reference
  - TBT P99 error < 15% vs reference
  - TTFT P90 < 2s (standard SLO from literature)
  - TBT P99 < 200ms (standard SLO from literature)
```

#### TC-TL-04: Scaling Behavior
```
For LLaMA2-70B on A100:
  - TP=1 to TP=4: throughput should scale ~3.5-3.8x (sublinear)
  - Increasing batch size: throughput increases, TTFT increases
  - Adding PP: throughput increases but adds pipeline bubble overhead
```

#### TC-TL-05: Model Size Scaling
```
Compare across model sizes on same hardware:
  - 7B vs 70B: ~10x more memory, ~8-10x less throughput
  - Prefill latency scales with model size (compute-bound)
  - Decode latency scales with model size (memory-bound)
```

### 3.3 Accuracy Targets

| Metric | Target Error | Measurement |
|--------|-------------|------------|
| Throughput (tokens/s) | <10% | vs Vidur published or vLLM benchmarks |
| TTFT P50 | <10% | vs Vidur published |
| TTFT P99 | <15% | vs Vidur published |
| TBT P50 | <10% | vs Vidur published |
| TBT P99 | <15% | vs Vidur published |
| Per-operator latency | <12% | vs LLMCompass published |
| Prefill stage | <5% | vs LLMCompass (0.69% achieved) |
| Decode stage | <10% | vs LLMCompass (7.5% achieved) |

---

## 4. KV Cache and Memory Model Validation

### 4.1 Reference Data Sources

| Source | Finding | Specific Data |
|--------|---------|--------------|
| Mooncake (FAST 2025) | KV cache reusability in production | ~50% prefix sharing in general, >80% for tool/agent workloads |
| vLLM (SOSP 2023) | PagedAttention memory efficiency | Near-zero waste vs 60-80% waste in naive allocation |
| LLMCompass (ISCA 2024) | Memory access modeling | Tile-by-tile simulation with area-based cost model |
| Standard formulas | KV cache size per token | 2 * n_layers * n_heads * head_dim * 2 (K+V) * bytes_per_element |
| FlashAttention papers | Memory savings | Eliminates O(s^2) attention score materialization |

#### KV Cache Size Reference Values

| Model | Per-Token KV (FP16) | Per-Token KV (INT8) | Formula Verification |
|-------|---------------------|--------------------|-----------------------|
| LLaMA2-7B (32L, 32H, 128d) | 0.5 MB/token | 0.25 MB/token | 2*32*32*128*2 = 524,288 bytes |
| LLaMA2-70B (80L, 64H, 128d) | 2.5 MB/token | 1.25 MB/token | 2*80*64*128*2 = 2,621,440 bytes |
| GPT-3 175B (96L, 96H, 128d) | 4.5 MB/token | 2.25 MB/token | 2*96*96*128*2 = 4,718,592 bytes |

### 4.2 Validation Test Cases

#### TC-MEM-01: KV Cache Size Accuracy
```
For each model (7B, 13B, 70B, 175B):
  For each precision (FP16, INT8, FP4):
    1. Calculate KV cache per token using our formula
    2. Compare against known reference values
    3. Verify total KV for batch: per_token * seq_len * batch_size
    Target: <1% error (analytical formula, should be exact)
```

#### TC-MEM-02: Total Memory Budget
```
For LLaMA2-70B on A100-80GB with TP=4:
  - Model weights: ~35GB FP16 / 4 GPUs = ~8.75GB per GPU
  - Remaining for KV cache: 80 - 8.75 - overhead = ~68GB
  - Max concurrent tokens: 68GB / 2.5MB = ~27,200 tokens
  - At seq_len=2048: max batch ~13
  Verify our memory model produces consistent results.
```

#### TC-MEM-03: Memory Utilization Under Load
```
At increasing batch sizes:
  - Memory utilization should increase linearly with batch*seq_len
  - At capacity: utilization should be >90%
  - Beyond capacity: OOM or eviction should trigger
```

#### TC-MEM-04: GQA/MQA Memory Savings
```
For models with GQA (e.g., LLaMA2 uses GQA with n_kv_heads < n_heads):
  - KV cache should scale with n_kv_heads, not n_heads
  - LLaMA2-70B with GQA (8 KV heads vs 64 query heads):
    KV = 2 * 80 * 8 * 128 * 2 = 327,680 bytes/token (not 2.5MB)
  Verify GQA reduction is correctly modeled.
```

### 4.3 Accuracy Targets

| Metric | Target Error | Notes |
|--------|-------------|-------|
| KV cache per token | <1% | Analytical, should be exact |
| Total memory budget | <3% | Overhead estimation adds small error |
| Max batch size | <5% | Depends on overhead accounting |
| GQA/MQA savings | Exact | Simple formula change |

---

## 5. Prefix Caching Validation

### 5.1 Reference Data Sources

| Source | Metric | Value |
|--------|--------|-------|
| Mooncake/Kimi (FAST 2025) | General production cache hit rate | ~50% |
| Mooncake L-Eval | Tool/agent cache hit rate | >80% |
| Mooncake ArXiv | Summarization cache hit rate | ~0% (unique documents) |
| SGLang (2024) | RadixAttention cache hit rate | Up to 5x speedup for multi-turn |
| vLLM prefix caching | Prefix sharing benefit | 2-5x TTFT reduction for shared prefixes |

### 5.2 Validation Test Cases

#### TC-PC-01: Cache Hit Rate for Repeated Prefixes
```
Scenario: 100 requests with 80% shared system prompt (512 tokens)
Expected: Cache hit rate >= 75% after warmup
Verify: TTFT reduction of 2-4x for cached requests vs uncached
```

#### TC-PC-02: Cache Hit Rate by Workload Type
```
Chat workload (multi-turn, shared system prompt): 40-60% hit rate
RAG workload (unique documents): <10% hit rate
Tool/agent workload (common tool descriptions): >70% hit rate
Verify our simulation produces rates in these ranges.
```

#### TC-PC-03: Cache Eviction Under Memory Pressure
```
When KV cache is full:
  - LRU eviction should evict least-recently-used blocks
  - Cache hit rate should degrade gracefully
  - No OOM errors from cache growth
```

#### TC-PC-04: Prefix Tree Correctness
```
For a sequence of multi-turn requests:
  1. First request: full prefill, no cache hit
  2. Second request (same conversation): prefix cached, only new tokens prefilled
  3. Third request (different conversation, same system prompt): system prompt cached
  Verify token savings match expected prefix lengths.
```

### 5.3 Accuracy Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Cache hit rate (chat) | Within 15% of Mooncake's ~50% | Depends on workload mix |
| TTFT savings from cache | Within 20% of expected | Proportional to cached prefix length |
| Cache memory accounting | <1% | Block-level tracking |

---

## 6. Prefill-Decode Disaggregation Validation

### 6.1 Reference Data Sources

| Source | Finding | Specific Data |
|--------|---------|--------------|
| Splitwise (ISCA 2024) | Optimal PD ratio varies by workload | Code: prefill-heavy; Chat: balanced |
| DistServe (OSDI 2024) | Disaggregation benefit | 7.4x more requests OR 12.6x tighter SLO |
| Mooncake (FAST 2025) | 75% more requests with disaggregation | Production Kimi deployment |
| DynamoLLM (HPCA 2025) | Request type classification | Short(<256/<100), Medium(<1024/<350), Long(<=8192/>=350) |
| Frontier simulator (2025) | PD system error | 19-23.2% relative error |

### 6.2 Validation Test Cases

#### TC-DG-01: Disaggregation Throughput Improvement
```
Compare colocated vs disaggregated for LLaMA2-70B:
  Colocated: prefill and decode share GPUs
  Disaggregated: separate prefill and decode pools

Expected improvement: 1.5-3x throughput depending on workload
Reference: DistServe reports 7.4x at high load, Mooncake 1.75x
```

#### TC-DG-02: Optimal PD Ratio
```
For chat workload (P:D ratio ~2.3, Vidur Chat-1M):
  - Test PD ratios: 1:3, 1:2, 1:1, 2:1, 3:1
  - Optimal should be near the P:D compute ratio
  - Too many prefill GPUs: decode becomes bottleneck
  - Too many decode GPUs: prefill becomes bottleneck
```

#### TC-DG-03: KV Cache Transfer Overhead
```
When transferring KV cache from prefill to decode node:
  - Transfer time = KV_size / interconnect_bandwidth
  - For LLaMA2-70B, seq_len=2048: ~640MB KV cache
  - Over NVLink (600GB/s): ~1ms
  - Over PCIe (64GB/s): ~10ms
  - Over network (100Gbps): ~51ms
  Verify our disaggregation model accounts for this overhead.
```

#### TC-DG-04: Request Type Routing
```
Using DynamoLLM classification:
  Short requests (<256 in, <100 out): fast-track, minimal KV
  Medium requests (<1024 in, <350 out): standard path
  Long requests (>1024 in or >350 out): may need more resources
  Verify routing decisions are consistent.
```

### 6.3 Accuracy Targets

| Metric | Target Error | Notes |
|--------|-------------|-------|
| Disaggregation throughput gain | Within 25% of published | Complex system, harder to model |
| KV transfer latency | <10% | Bandwidth-based calculation |
| Optimal PD ratio | Within 1 ratio step | E.g., if optimal is 1:2, accept 1:1 or 1:3 |

---

## 7. Power and Energy Model Validation

### 7.1 Reference Data Sources

| Source | Hardware | Power (Idle) | Power (Peak) | Power (Inference) |
|--------|----------|-------------|-------------|-------------------|
| NVIDIA specs | A100-SXM | ~50W | 400W TDP | 250-350W |
| NVIDIA specs | H100-SXM | ~70W | 700W TDP | 350-550W |
| MLPerf inference | Various | -- | -- | Measured per-query energy |
| DynamoLLM (HPCA 2025) | A100 cluster | -- | -- | 52% energy savings with dynamic management |

### 7.2 Validation Test Cases

#### TC-PW-01: Idle vs Active Power
```
GPU at 0% utilization: power ~= idle_power (e.g., 50W for A100)
GPU at 100% utilization: power ~= TDP (e.g., 400W for A100)
GPU at 50% utilization: power between idle and TDP
Verify polynomial interpolation produces reasonable values.
```

#### TC-PW-02: Power Scaling with Batch Size
```
As batch size increases:
  - GPU utilization increases
  - Power should increase accordingly
  - Energy per token should DECREASE (amortized fixed costs)
  Reference: DynamoLLM shows up to 52% energy savings through optimal batching
```

#### TC-PW-03: Cluster Power Aggregation
```
For a 4-GPU node:
  Total power = sum(GPU power) + CPU power + memory power + network power + cooling overhead
  Verify total is within reasonable bounds (e.g., 1.5-3kW for 4xA100 node under load)
  PUE factor: 1.1-1.4 for modern data centers
```

### 7.3 Accuracy Targets

| Metric | Target Error | Notes |
|--------|-------------|-------|
| Single GPU power | <15% | Depends on utilization model accuracy |
| Cluster power | <20% | Aggregation adds uncertainty |
| Energy per token | <20% | Compound error from power * latency |

---

## 8. Configuration Optimizer Validation

### 8.1 Reference Data Sources

| Source | Finding | Specific Data |
|--------|---------|--------------|
| Vidur-Search (MLSys 2024) | Config search | Best config in 1 hour on CPU vs 42K GPU hours |
| SCOOT (WWW 2025) | Bayesian optimization | 68.3% throughput improvement vs default |
| DynamoLLM (HPCA 2025) | Dynamic configuration | 52% energy savings, 61% cost reduction |

### 8.2 Validation Test Cases

#### TC-CO-01: Optimizer Finds Valid Configurations
```
For LLaMA2-70B on A100:
  - All recommended configs should fit in memory
  - TP * PP * DP <= available GPUs
  - Batch size should not exceed KV cache capacity
  - No config should violate SLO constraints
```

#### TC-CO-02: Optimizer Ranking Quality
```
Top-ranked config should have:
  - Higher throughput than >80% of explored configs
  - Lower cost than >70% of explored configs
  - SLO compliance (TTFT P90 < 2s, TBT P99 < 200ms)
```

#### TC-CO-03: Pareto Frontier Correctness
```
For multi-objective optimization (throughput vs cost):
  - All Pareto-optimal configs should be non-dominated
  - No config in the Pareto set should be strictly worse than another on all objectives
  - Pareto set should contain 3-10 configs for a typical search space
```

### 8.3 Accuracy Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Config validity | 100% | All recommended configs must be feasible |
| Ranking quality | Top config within 15% of true optimal | Hard to verify without ground truth |
| Search completeness | Cover 90% of Pareto frontier | vs exhaustive search on small space |

---

## 9. Simulator Accuracy Methodology

### 9.1 How Other Simulators Validate (State of the Art)

Based on research into 7 major simulators:

#### Vidur Methodology (Gold Standard)
1. **Operator profiling**: Profile each operator type on real hardware across input dimensions
2. **ML interpolation**: Random forest regression to predict unseen dimensions
3. **DES simulation**: Discrete event simulation of request lifecycle
4. **Validation**: Compare E2E latency CDFs at multiple load levels
5. **Result**: <5% error at 85% load, <3.33% P95 error for static workloads

#### LLMCompass Methodology (Best for Hardware Exploration)
1. **Analytical tile modeling**: Simulate operator execution tile-by-tile
2. **Mapper search**: 26,400 configurations searched for optimal mapping
3. **Multi-hardware**: Validate on A100, MI210, TPUv3
4. **Result**: 4.1% E2E error, 0.69% prefill, 7.5% decode

#### TokenSim Methodology (Highest Accuracy)
1. **DES with SimPy**: Discrete event simulation framework
2. **Block-granularity memory**: Track memory at block level
3. **Transformer-oriented**: Model transformer-specific execution patterns
4. **Result**: 0.109% throughput error, 0.6% P50 latency error

### 9.2 Our Validation Approach

Given that we use an analytical roofline model (like GenZ/LLMCompass) rather than profiling-based (like Vidur):

#### Phase 1: Operator-Level Validation
- Compare our roofline predictions against LLMCompass published per-operator errors
- Target: GEMM <12%, Attention <15%, Communication <18%
- Method: Compute arithmetic intensity, compare against hardware roofline

#### Phase 2: Cross-Validation with Vidur
- Use Vidur's 3 published workload traces with exact same parameters
- Compare our throughput and latency predictions against their published results
- Target: Within 2x of Vidur's error (i.e., <10% since Vidur achieves <5%)

#### Phase 3: Workload Trace Replay
- Replay BurstGPT or Azure traces through our simulator
- Compare request-level metrics (TTFT, TBT) against known distributions
- Verify that our simulator produces realistic SLO compliance curves

#### Phase 4: Absolute Accuracy (Optional, Requires Hardware)
- Run vLLM benchmarks on A100/H100 and compare directly
- This provides ground truth but requires GPU access

### 9.3 Key Insights from Research

1. **Prefill is easier to model** (LLMCompass: 0.69% error) because it's compute-bound and predictable
2. **Decode is harder** (7.5% error) because it's memory-bound with more variability
3. **Small models have higher simulation error** (Vidur: 12.65% for 7B at 95% load) due to CPU overhead
4. **All simulators converge to ~12% error at 500+ requests** regardless of approach
5. **Skewed batch lengths cause worst errors** (Vidur: up to 55% for FlashAttention with skewed batches)
6. **Communication ops are hardest** (LLMCompass: 14.9% for all-reduce)
7. **Calibration with minimal profiling data dramatically improves accuracy** (IBM: 80% MSE reduction)

---

## 10. Continuous Validation Framework

### 10.1 Automated Validation Pipeline

```
On every PR that modifies serving simulation code:
  1. Run unit tests (existing test suite)
  2. Run validation test cases TC-WL-01 through TC-WL-04 (workload)
  3. Run validation test cases TC-MEM-01 through TC-MEM-04 (memory)
  4. Run validation test cases TC-TL-01 (operator accuracy spot check)
  5. Compare against baseline results (stored in validation/baselines/)
  6. Flag if any metric degrades by >5% from baseline
```

### 10.2 Periodic Full Validation

```
Monthly or on major releases:
  1. Full cross-validation against Vidur traces (TC-TL-02, TC-TL-03)
  2. Full workload distribution validation (TC-WL-01 through TC-WL-04)
  3. Power model validation (TC-PW-01 through TC-PW-03)
  4. Config optimizer validation (TC-CO-01 through TC-CO-03)
  5. Generate accuracy report using existing AccuracyReporter
  6. Compare against production-readiness thresholds:
     - MRE < 15%
     - Correlation > 0.8
     - Bias < 5%
     - Within-tolerance > 85%
```

### 10.3 Drift Detection

```
When new benchmark papers or traces are published:
  1. Add to reference data sources
  2. Re-run validation against new data
  3. If accuracy degrades, trigger recalibration
  4. Update this document with new reference values
```

### 10.4 Validation Data Files

Store reference data in structured format for automated testing:

```
llm-memory-calculator/
  tests/serving/
    validation_data/
      workload_references.json    # Token length statistics from papers
      throughput_references.json   # Published throughput numbers
      latency_references.json     # Published latency percentiles
      memory_references.json      # KV cache size calculations
      power_references.json       # Published power measurements
```

---

## Appendix A: Key Paper References

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 1 | Splitwise | ISCA | 2024 | Azure traces (code/conv), PD disaggregation |
| 2 | DynamoLLM | HPCA | 2025 | Azure traces (diurnal), energy optimization |
| 3 | BurstGPT | KDD | 2025 | 10.31M traces, Gamma arrivals, burstiness |
| 4 | ServeGen | Alibaba | 2025 | Pareto+Lognormal input, Exponential output |
| 5 | LMSYS-Chat-1M | ICLR | 2024 | 1M conversations, avg prompt/response 215 tokens |
| 6 | Mooncake | FAST | 2025 | 23K traces, 50% prefix sharing, long context |
| 7 | Vidur | MLSys | 2024 | <5% error, profiling+RF, 3 workload traces |
| 8 | LLMCompass | ISCA | 2024 | 4.1% E2E error, analytical tile simulation |
| 9 | TokenSim | 2025 | 2025 | <1% throughput error, DES+SimPy |
| 10 | LLMServingSim | MLArchSys | 2024 | 14.7% error, HW/SW co-simulation |
| 11 | Frontier | 2025 | 2025 | 19-23% error, disaggregation+MoE support |
| 12 | vLLM | SOSP | 2023 | PagedAttention, ShareGPT benchmarks |
| 13 | DistServe | OSDI | 2024 | 7.4x throughput with PD disaggregation |
| 14 | SCOOT | WWW | 2025 | Bayesian optimization for LLM serving config |
| 15 | ShareGPT | -- | 2023 | 70K conversations, standard benchmark dataset |

## Appendix B: Serving Simulation Module Mapping

| Module | Validation Section | Key Test Cases |
|--------|-------------------|---------------|
| `workload.py` | Section 2 | TC-WL-01 to TC-WL-04 |
| `simulator.py` | Section 3 | TC-TL-01 to TC-TL-05 |
| `memory_model.py` | Section 4 | TC-MEM-01 to TC-MEM-04 |
| `prefix_cache.py` | Section 5 | TC-PC-01 to TC-PC-04 |
| `disaggregation.py` | Section 6 | TC-DG-01 to TC-DG-04 |
| `power_model.py` | Section 7 | TC-PW-01 to TC-PW-03 |
| `config_optimizer.py` | Section 8 | TC-CO-01 to TC-CO-03 |
| `request.py` | Sections 2, 3 | Request lifecycle |
| `slo_tracker.py` | Section 3 | SLO compliance |
| `batch_scheduler.py` | Section 3 | Scheduling decisions |
| `cluster.py` | Section 6 | Multi-node behavior |
| `constants.py` | All sections | Reference values |
