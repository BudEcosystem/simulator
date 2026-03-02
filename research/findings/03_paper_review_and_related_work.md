# Paper Review and Related Work Research

## Table of Contents

1. [Primary Paper: SCOOT](#1-primary-paper-scoot)
2. [Core LLM Serving Optimization Techniques](#2-core-llm-serving-optimization-techniques)
3. [Bayesian Optimization for System Configuration Tuning](#3-bayesian-optimization-for-system-configuration-tuning)
4. [Performance Modeling for LLM Inference](#4-performance-modeling-for-llm-inference)
5. [Optimization Frameworks and Alternatives to Optuna](#5-optimization-frameworks-and-alternatives-to-optuna)
6. [Synthesis: Implications for vllm-tuner](#6-synthesis-implications-for-vllm-tuner)

---

## 1. Primary Paper: SCOOT

### Full Citation

**SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines**
Ke Cheng, Zhi Wang, Wen Hu, Tiannuo Yang, Jianguo Li, and Sheng Zhang.
In *Proceedings of the ACM Web Conference 2025 (WWW '25)*, April 28-May 2, 2025, Sydney, NSW, Australia.
ACM, New York, NY, USA, 11 pages.
https://doi.org/10.1145/3696410.3714930

**arXiv**: 2408.04323v2 (20 Feb 2025)

### Authors and Affiliations

- Ke Cheng (Nanjing University) -- co-first author
- Zhi Wang (Ant Group) -- co-first author
- Wen Hu (Ant Group)
- Tiannuo Yang (Nankai University)
- Jianguo Li (Ant Group) -- corresponding author
- Sheng Zhang (Nanjing University) -- corresponding author

### Key Contributions

1. **First study** to introduce performance tuning into the field of LLM serving, uncovering the significance of tuning LLM inference engines with real-world request traces.
2. **General formulation** of the inference engine tuning problem that accommodates various optimization objectives and constraints, solved via Bayesian Optimization (BO).
3. **Random forest regression** to learn hidden constraints (infeasible parameter combinations that cause engine crashes) during tuning, avoiding invalid explorations.
4. **Parallel suggestion technique** to accelerate tuning by evaluating multiple configurations concurrently.
5. **Extensive experiments** confirming superiority in both optimality and efficiency across various LLMs, computing resources, and request traces.

### Problem Formulation

SCOOT formalizes the tuning problem as:

```
maximize  lambda_T * T(x), lambda_l * L(x), lambda_phi * Phi(x), lambda_theta * Theta(x)
subject to:  c_i, for all c_i in C (known constraints)
             POF(x) >= Delta (hidden constraint feasibility)
```

Where:
- **T(x)**: Request throughput
- **L(x)**: Tail latency (95th percentile)
- **Phi(x)**: Average TTFT (Time-to-First-Token)
- **Theta(x)**: Average TPOT (Time-per-Output-Token)
- **C**: Known constraints between parameters
- **POF(x)**: Probability of Feasibility (hidden constraints)

Lambda weights control single vs. multi-objective optimization:
- `(1, 0, 0, 0)` = maximize throughput only (offline/recommendation apps)
- `(0, -1, 0, 0)` = minimize tail latency only (classification apps)
- `(0, 0, -1, -1)` = minimize TTFT and TPOT simultaneously (interactive/chatbot apps)

### Parameters Tuned (vLLM)

| Parameter | Type | Range |
|-----------|------|-------|
| `tensor-parallel` | Integer | [1, #GPUs] |
| `max-num-seqs` | Integer | [64, 8192] |
| `max-num-batched-tokens` | Integer | [64, 8192] |
| `block-size` | Enumeration | {8, 16, 32} |
| `scheduler-delay-factor` | Float | [0, 2] |
| `enable-chunked-prefill` | Boolean | {True, False} |
| `enable-prefix-caching` | Boolean | {True, False} |
| `disable-custom-all-reduce` | Boolean | {True, False} |
| `use-v2-block-manager` | Boolean | {True, False} |

Search space: over 100 billion configuration points.

### Algorithm Design (9-Step Workflow)

1. Customer defines optimization objectives and provides request traces.
2. Sobol sequence-based Quasi-Monte Carlo uniformly samples configurations.
3. Stress test inference engine with each sampled configuration to collect observations.
4. Build a **random forest** to learn POF(x) -- feasibility of configurations.
5. Build a **Gaussian Process surrogate model** to predict performance distribution for each objective.
6. Use **acquisition functions** to assess configurations (exploration vs. exploitation).
7. A **solver** suggests multiple configurations in parallel (MACE ensemble for SOBO, EHVI for MOBO).
8. Run inference engine with suggested configurations and stress test.
9. Collect new observations, refine random forest and surrogate model. Repeat steps 5-9.

### Surrogate Model Details

- **Gaussian Process (GP)** with Matern kernel (5/2) and input wrapping.
- For SOBO: single-output GP.
- For MOBO: multi-output GP treating each output as independent.
- White noise parameter tau^2 learned via maximum likelihood estimation.

### Acquisition Functions

**SOBO (Single-Objective):**
- **UCB** (Upper Confidence Bound): mu(f(x)) + beta * sigma(f(x)), with dynamic beta.
- **PI** (Probability of Improvement): P(f(x) + xi > f(x+)).
- **EI** (Expected Improvement): E(max(0, f(x) + xi - f(x+))).
- Uses **MACE** (Multi-objective Acquisition function ensemble) -- runs all three, finds Pareto frontier of trade-offs, randomly selects from it.

**MOBO (Multi-Objective):**
- **EHVI** (Expected Hypervolume Improvement): measures potential improvement to Pareto front hypervolume.
- Reference point is the default configuration of the inference engine.

### Hidden Constraint Handling

- Random forest learns POF(x) from observations of crashes vs. successes.
- Dynamic threshold Delta starts at 0.5 and adjusts:
  - Infeasible suggestion: Delta = min(0.75, max(0.5, Delta + v))
  - 5+ consecutive feasible: Delta = max(0.25, Delta - v) where v = 0.05
- This avoids both over-conservative (missing good configs) and over-aggressive (wasting evaluations on crashes) behavior.

### Known Constraints (vLLM-0.4.2)

- `max-num-batched-tokens` must be >= `max-num-seqs`
- `enable-chunked-prefill` and `enable-prefix-caching` cannot both be True simultaneously
- When `enable-chunked-prefill` is False, `max-num-batched-tokens` >= max model length

### Parallel Suggestion

- Parallelism degree (PD) k: suggest k configurations simultaneously.
- For SOBO: randomly select k points from MACE's Pareto frontier.
- For MOBO: top-k points with largest EHVI.
- PD=2 nearly halves tuning time with same optimization quality.
- PD > 4 can compromise quality (increasing observation budget mitigates this).

### SLO Robustness

- After finding best config, run stress tests multiple times with varying request arrival orders.
- Select worst-case objectives as the SLOs to guarantee robustness under different traffic patterns.

### Experimental Setup

- **Implementation**: Built on top of HEBO (NeurIPS 2020 champion BO library).
- **Random forest**: sklearn library.
- **LLMs**: LLAMA2-7B, LLAMA2-13B.
- **Inference engine**: vLLM-0.4.2.
- **Applications**: text-to-SQL (SQL), chatbot (BOT), classification (CLS), recommendation (REC).
- **Hardware**: NVIDIA A10 24GB, NVIDIA A100 80GB; 2 and 4 GPU configs.
- **Evaluation protocol**: 100-second stress tests per configuration, Poisson arrival, 30 total observations, PD=1.
- **Baselines**: Random sampling (RD), Genetic Algorithm (GA), Vanilla BO (VBO).

### Key Results

| Metric | Improvement vs. Default | Improvement vs. Baselines |
|--------|------------------------|--------------------------|
| Request throughput | Up to 68.3% increase | Consistent across all |
| Tail latency (95th) | Up to 40.6% reduction | Outperforms all baselines avg |
| TTFT | Up to 99.8% reduction | Consistent Pareto front improvements |
| TPOT | Up to 61.0% reduction | Better diversity + optimality |

**Real-world deployment**: On Ant Group's production cluster with 8 NVIDIA L20 48GB GPUs:
- vLLM-0.5.5: SCOOT achieved 0.695 RPS (story-teller-long) vs. 0.527 default, 0.595 VBO.
- TensorRT-LLM-0.15.0: SCOOT achieved 0.763 RPS vs. 0.557 default, 0.670 VBO.

### Limitations and Future Work

1. **Evaluation overhead**: Each stress test takes 5-10 minutes; 30 evaluations = 2.5-5 hours per service. With many LLM services, cumulative tuning time is prohibitive.
2. **PD scaling**: Performance degrades when PD > 4, suggesting need for better observation budgeting.
3. **vLLM version**: Tested on vLLM-0.4.2 (older); newer versions have significantly different parameter sets.
4. **Static workload assumption**: Tuning is per-request-trace; changing workload distributions may require re-tuning.
5. **No speculative decoding or quantization tuning**: These are excluded despite being significant for production.

### Applicability to vllm-tuner

SCOOT is the most directly relevant paper to vllm-tuner. Key techniques that should be incorporated:

1. **MACE acquisition function ensemble** -- instead of relying on a single acquisition function, use all three (UCB, PI, EI) and select from their Pareto frontier. This is more robust than Optuna's TPE.
2. **Random forest for hidden constraint learning** -- critical for vLLM where many parameter combinations crash the engine. vllm-tuner currently has no such mechanism.
3. **Dynamic feasibility threshold** -- the adaptive Delta mechanism prevents wasting evaluations on infeasible configurations while still exploring near constraint boundaries.
4. **Parallel suggestion** -- evaluate multiple configs simultaneously to utilize idle GPU resources during off-peak hours.
5. **Known constraint pruning** -- encode vLLM's documented parameter constraints to prune the search space upfront.
6. **Multi-objective optimization** -- support TTFT+TPOT joint optimization for interactive applications via EHVI, not just single-metric optimization.
7. **SLO robustness protocol** -- stress test with varying arrival patterns to guarantee worst-case performance.

**Implementation complexity**: Medium-high. SCOOT is built on HEBO, which is an established BO library. The random forest + GP + MACE ensemble requires careful engineering but has well-understood components. vllm-tuner could adopt HEBO as a backend or implement the key ideas on top of Optuna.

---

## 2. Core LLM Serving Optimization Techniques

### 2.1 PagedAttention

**Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
**Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.
**Venue**: SOSP 2023
**Source**: https://arxiv.org/abs/2309.06180

**Key technique**: Inspired by OS virtual memory paging, PagedAttention stores KV cache in non-contiguous memory blocks with a block table for lookup. This eliminates the 60-80% memory waste from fragmentation and over-reservation in traditional systems, reducing waste to under 4%.

**Performance**: vLLM with PagedAttention delivers up to 24x higher throughput than HuggingFace Transformers. Memory sharing across requests (parallel sampling, beam search) reduces memory overhead by up to 55%, translating to 2.2x throughput improvement.

**Relevance to vllm-tuner**: The `block-size` parameter (8, 16, 32) directly controls PagedAttention granularity. Larger blocks reduce lookup overhead but increase fragmentation. vllm-tuner should understand this tradeoff when selecting block sizes based on workload characteristics (short vs. long sequences).

**Implementation complexity**: Low -- already a vLLM parameter to tune.

---

### 2.2 Continuous Batching (Orca)

**Paper**: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
**Authors**: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, et al.
**Venue**: OSDI 2022

**Key technique**: Iteration-level scheduling where batch composition changes dynamically at each decoding step. As soon as a sequence finishes, a new request takes its place, maximizing GPU occupancy.

**Performance**: Achieves 23x LLM inference throughput improvement with reduced p50 latency compared to static batching.

**Relevance to vllm-tuner**: The `max-num-seqs` parameter controls maximum batch size for continuous batching. The `scheduler-delay-factor` controls how long the scheduler waits to accumulate requests before starting a batch. Both are critical tuning knobs.

**Implementation complexity**: Low -- these are existing vLLM parameters.

---

### 2.3 Chunked Prefill (Sarathi / Sarathi-Serve)

**Paper**: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
**Authors**: Amey Agrawal, Nitin Kedia, Ashish Panwar, et al.
**Venue**: OSDI 2024
**Source**: https://arxiv.org/abs/2403.02310

**Key technique**: Splits large prefill computations into smaller chunks and interleaves them with decode operations (decode-maximal batching). Uses "stall-free scheduling" where new requests join running batches without pausing ongoing decodes.

**Performance**:
- LLaMA-13B on A6000: up to 10x decode throughput improvement.
- LLaMA-33B on A100: 1.25x end-to-end throughput, up to 4.25x decode throughput.
- Mistral-7B on A100: 2.6x higher serving capacity.
- Falcon-180B with pipeline parallelism: up to 5.6x serving capacity gain.
- Standard vLLM: chunked prefill increases total token throughput by +50%.

**Relevance to vllm-tuner**: The `enable-chunked-prefill` boolean is a key parameter. When enabled, `max-num-batched-tokens` controls the chunk size. The interaction between chunked prefill and prefix caching (they conflict in older vLLM versions) is a critical known constraint. vllm-tuner must model the chunk-size / TTFT tradeoff: larger chunks = faster prefill but more decode stalling.

**Recent advance -- Layered Prefill**: Treats transformer layer groups as the scheduling unit, reducing TTFT by up to 70% and end-to-end latency by 41%. Not yet a vLLM parameter but worth monitoring.

**Implementation complexity**: Low -- existing vLLM parameter. Medium for modeling chunk size interactions.

---

### 2.4 Disaggregated Prefill/Decode (DistServe)

**Paper**: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
**Authors**: Yinmin Zhong et al.
**Venue**: OSDI 2024
**Source**: https://arxiv.org/abs/2401.09670

**Key technique**: Assigns prefill and decoding computation to different GPUs, eliminating prefill-decoding interference. Co-optimizes resource allocation and parallelism strategy for each phase independently.

**Performance**: Can serve 7.4x more requests or achieve 12.6x tighter SLO compared to colocated systems while maintaining >90% SLO compliance.

**Industry adoption**: Almost every production-grade framework now supports disaggregation: NVIDIA Dynamo, llm-d, Ray Serve, SGLang, vLLM, LMCache, MoonCake.

**Relevance to vllm-tuner**: Disaggregated serving introduces new tuning dimensions -- separate parallelism configs for prefill and decode GPUs, communication bandwidth between them, prefill/decode resource ratios. vllm-tuner should eventually support disaggregated deployment tuning when vLLM fully supports it in production.

**Implementation complexity**: High -- requires multi-node orchestration and new parameter dimensions.

---

### 2.5 Speculative Decoding

**Papers**:
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., ICML 2023)
- "SpecInfer: Accelerating Large Language Model Serving with Tree-Based Speculative Inference" (Miao et al., ASPLOS 2024)
- "Mirror Speculative Decoding" (Apple Research, 2025)

**Key technique**: A smaller "draft" model proposes multiple tokens; the larger "target" model verifies them in parallel. Output is provably identical to the target model alone.

**Performance**:
- Standard: up to 3x faster LLM inference with the right draft model.
- Mirror Speculative Decoding: 2.8x-5.8x wall-time speedups (30% improvement over EAGLE3).
- Online Speculative Decoding (OSD): continuously adapts draft model to evolving query distribution.
- TurboSpec: closed-loop control system with online feedback for dynamic parameter adjustment.

**Key insight**: Speculative decoding helps most at batch size 1. With larger batches, compute is already better utilized and draft/verify overhead becomes less attractive.

**Relevance to vllm-tuner**: SCOOT explicitly excludes speculative decoding from tuning. However, vllm-tuner could gain significant value by tuning speculative decoding parameters (draft model selection, speculation length, acceptance threshold) especially for latency-sensitive single-user deployments.

**Implementation complexity**: Medium -- vLLM supports speculative decoding with configurable parameters.

---

### 2.6 Prefix Caching

**Key systems**:
- **vLLM Automatic Prefix Caching (APC)**: Hash-based KV cache reuse for shared prefixes.
- **Marconi** (arxiv 2411.19379): First system for prefix caching with hybrid LLMs; achieves 34.4x higher token hit rates.
- **LMCache**: Enterprise-scale multi-tier KV caching (GPU/CPU/disk); 3x-10x latency reduction with vLLM.
- **KVFlow**: Workflow-aware KV cache management for agentic workflows.

**Production impact**:
- Anthropic: 90% cost reduction, 85% latency reduction for long prompts.
- OpenAI: 50% cost savings (auto-enabled).
- LMCache + vLLM: 15x throughput, 2x latency reduction.

**Relevance to vllm-tuner**: The `enable-prefix-caching` parameter is already tuned. However, vllm-tuner should consider workload-dependent recommendations -- prefix caching is most beneficial when requests share common prefixes (system prompts, RAG contexts). For workloads with unique prefixes, it adds overhead.

**Implementation complexity**: Low -- existing parameter. Medium for workload-aware recommendation.

---

### 2.7 KV Cache Management Optimizations

**Key techniques**:
- **Token-level**: KV cache selection (evict unimportant tokens), budget allocation, merging, quantization, low-rank decomposition.
- **Model-level**: MiniCache (cross-layer merging), SwiftKV (skip half the layers), MorphKV (fixed-size cache with iterative refinement).
- **System-level**: LMCache (multi-tier caching), KV cache offloading to CPU/disk.

**Performance**: Advanced strategies reduce memory by 50-80% while maintaining quality. MorphKV achieves >50% memory savings with improved long-form accuracy.

**Relevance to vllm-tuner**: As vLLM introduces new KV cache management parameters (e.g., FP8 KV cache quantization, eviction policies), vllm-tuner's parameter space should expand. The `block-size` and memory management parameters interact with KV cache efficiency.

**Implementation complexity**: Medium -- some techniques require vLLM version updates.

---

### 2.8 Multi-Step Scheduling

**Source**: vLLM v0.6.0+ (https://github.com/vllm-project/vllm/issues/6854)

**Key technique**: Execute multiple decode forward passes before syncing with the CPU scheduler. Reduces CPU scheduling overhead and GPU idle time by amortizing the scheduling cost over multiple steps.

**Performance**: Recommended `--num-scheduler-steps` between 10-15 for optimal GPU utilization. vLLM v0.6.0 achieved 1.8-2.7x throughput improvement.

**Tradeoffs**:
- Higher `num-scheduler-steps` increases TTFT at low request rates (new requests wait for current multi-step to finish).
- Bumpy inter-token latency (tokens arrive in bursts).

**Relevance to vllm-tuner**: `--num-scheduler-steps` is a high-impact tuning parameter not included in SCOOT (which used vLLM-0.4.2). vllm-tuner should add this parameter and model its interaction with TTFT requirements.

**Implementation complexity**: Low -- straightforward parameter addition.

---

### 2.9 Flash Attention

**Papers**:
- FlashAttention (Dao et al., NeurIPS 2022): Tiling-based attention reducing HBM I/O.
- FlashAttention-2 (Dao, 2023): Better parallelism and work partitioning.

**Key technique**: Computes attention in smaller blocks ("tiles") to reduce transfers between GPU VRAM and system RAM, achieving 2-3x speedups.

**Relevance to vllm-tuner**: Flash attention is generally always enabled and not a tuning parameter. However, its interaction with chunked prefill and block sizes affects optimal configuration.

**Implementation complexity**: N/A -- not a tunable parameter.

---

### 2.10 GPU Memory Management

**Key developments**:
- **CPU Offloading (NEO)**: Just-in-time model loading based on request patterns.
- **Q-Infer**: GPU-CPU collaborative inference with sparsity-aware dynamic scheduling.
- **Quantization**: NVFP4 and FP8 formats enable 40-60% memory savings.

**Key insight**: DRAM bandwidth saturation is the primary bottleneck in large-batch inference, leaving significant compute underutilized.

**Relevance to vllm-tuner**: Memory-related parameters (`gpu-memory-utilization`, quantization settings) should be part of the tuning space. Understanding the memory-compute tradeoff is essential for setting `max-num-seqs` and `max-num-batched-tokens`.

**Implementation complexity**: Medium -- requires understanding of memory hierarchy.

---

## 3. Bayesian Optimization for System Configuration Tuning

### 3.1 HEBO (Heteroscedastic Evolutionary Bayesian Optimisation)

**Paper**: "HEBO: Pushing The Limits of Sample-Efficient Hyperparameter Optimisation"
**Authors**: Cowen-Rivers et al. (Huawei Noah's Ark Lab)
**Venue**: JAIR, Vol 74; NeurIPS 2020 Black-Box Optimization Challenge Winner
**Source**: https://arxiv.org/abs/2012.03826

**Key techniques**:
- Non-linear input and output warping to handle heteroscedasticity and non-stationarity.
- Multi-objective acquisition ensembles with Pareto front solutions.
- Robust acquisition maximization.
- Exact marginal log-likelihood optimization.

**Performance**: Winner of NeurIPS 2020 Black-Box Optimization Challenge. Significantly outperforms existing optimizers on 108 ML hyperparameter tuning tasks (Bayesmark benchmark).

**Relevance to vllm-tuner**: SCOOT is built on top of HEBO. vllm-tuner could either:
(a) Switch from Optuna to HEBO as the optimization backend for better sample efficiency, or
(b) Use Optuna with HEBO-inspired enhancements (output warping, multi-objective ensembles).

**Implementation complexity**: Medium -- HEBO is a well-maintained library with clean API.

---

### 3.2 GPTuner (LLM-Guided Database Tuning)

**Paper**: "GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization"
**Authors**: Jiale Lao, Yibo Wang, Yufei Li, et al.
**Venue**: PVLDB 2024
**Source**: https://arxiv.org/abs/2311.03157

**Key techniques**:
- LLM-based pipeline to collect and refine heterogeneous knowledge from documentation.
- Workload-aware, training-free knob selection strategy.
- Search space optimization via domain knowledge extraction.
- Coarse-to-Fine Bayesian Optimization framework.

**Performance**: Identifies better configurations in 16x less time compared to SOTA. Up to 30% performance improvement on PostgreSQL and MySQL across TPC-C and TPC-H benchmarks.

**Relevance to vllm-tuner**: The LLM-guided search space pruning is directly applicable. An LLM could read vLLM documentation and release notes to automatically identify parameter interactions, valid ranges, and best practices -- reducing the search space before BO begins. This would complement SCOOT's manual constraint specification.

**Implementation complexity**: Medium -- requires LLM API calls during initialization but straightforward to implement.

---

### 3.3 LATuner (LLM-Enhanced Database Tuning)

**Paper**: "LATuner: An LLM-enhanced Database Tuning System based on Adaptive Surrogate Model"
**Venue**: ECML-PKDD 2024

**Key technique**: Uses LLM to enhance surrogate model construction for database configuration tuning.

**Relevance to vllm-tuner**: Similar concept to GPTuner but focused on surrogate model enhancement. Could inform how to build better performance models for vLLM configurations.

**Implementation complexity**: Medium.

---

### 3.4 BOAT (Building Auto-Tuners with Structured BO)

**Paper**: "Boat: Building Auto-Tuners with Structured Bayesian Optimization"
**Authors**: Dalibard, Schaarschmidt, Yoneki
**Venue**: WWW 2017

**Key technique**: Structured Bayesian optimization that exploits system-specific knowledge to improve tuning efficiency.

**Relevance to vllm-tuner**: Demonstrates that incorporating system structure (e.g., parameter dependencies, monotonic relationships) into BO improves both sample efficiency and final performance.

**Implementation complexity**: Medium.

---

### 3.5 Cotumer (Hierarchical Learning for Resource Partitioning)

**Paper**: "Cotumer: A Hierarchical Learning Framework for Coordinating Optimizing Resource Partitioning and Parameter Tuning"
**Venue**: ICPP 2023

**Key technique**: Hierarchical learning framework that jointly optimizes resource allocation and parameter tuning.

**Relevance to vllm-tuner**: The hierarchical approach (first optimize high-level resource allocation like GPU count, then fine-tune engine parameters) could be applied to vllm-tuner's workflow.

**Implementation complexity**: High.

---

### 3.6 Restune (Resource-Oriented Tuning)

**Paper**: "Restune: Resource Oriented Tuning Boosted by Meta-Learning for Cloud Databases"
**Authors**: Xinyi Zhang et al.
**Venue**: SIGMOD 2021

**Key technique**: Meta-learning to warm-start BO for new workloads based on experience from previous tuning sessions.

**Relevance to vllm-tuner**: Meta-learning across different model/hardware combinations could dramatically reduce tuning time. If vllm-tuner has tuned LLaMA-7B on A100, that experience could warm-start tuning LLaMA-13B on A100.

**Implementation complexity**: Medium-high -- requires building a knowledge base of past tuning sessions.

---

### 3.7 Clite and Satori (Resource-Aware Scheduling)

**Papers**:
- "Clite: Efficient and QoS-aware Co-location of Multiple Latency-Critical Jobs" (HPCA 2020)
- "Satori: Efficient and Fair Resource Partitioning" (ISCA 2021)

**Key technique**: Co-locate multiple latency-critical workloads on shared hardware while meeting individual QoS constraints through intelligent resource partitioning.

**Relevance to vllm-tuner**: When multiple LLM services share a GPU cluster, tuning must account for co-location interference. vllm-tuner could incorporate resource contention models.

**Implementation complexity**: High.

---

### 3.8 LLAMBO (LLM-Enhanced Bayesian Optimization)

**Paper**: "Large Language Models to Enhance Bayesian Optimization"
**Authors**: Tennison Liu et al.
**Venue**: ICLR 2024
**Source**: https://arxiv.org/abs/2402.03921

**Key technique**: Frames the BO problem in natural language, enabling LLMs to propose and evaluate configurations conditioned on historical evaluations. Excels at zero-shot warmstarting and enhances surrogate modeling when observations are sparse.

**Performance**: Competitive with or superior to traditional BO methods in early stages of search. Available as an OptunaHub sampler.

**Relevance to vllm-tuner**: LLAMBO could be used for the initial exploration phase when few observations exist. By describing vLLM parameters in natural language and asking an LLM to suggest promising configurations, vllm-tuner could achieve better initial samples than random or Sobol sampling. Available as an Optuna integration.

**Implementation complexity**: Low-Medium -- available as OptunaHub sampler, but requires LLM API access.

---

## 4. Performance Modeling for LLM Inference

### 4.1 Hardware-Agnostic Analytical Modeling

**Paper**: "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling"
**Source**: https://arxiv.org/pdf/2508.00904

**Key technique**: Develops hardware-agnostic analytical models to predict LLM inference performance across heterogeneous devices (CPUs, NPUs, integrated GPUs) without requiring actual hardware access.

**Relevance to vllm-tuner**: An analytical performance model could serve as a cheap surrogate, reducing the number of expensive stress tests needed. Instead of running 30 evaluations at 5-10 minutes each, an analytical model could pre-filter configurations and only stress-test the most promising ones.

**Implementation complexity**: High -- requires accurate modeling of vLLM's internal scheduling and memory management.

---

### 4.2 vLLMSim (Runtime Latency Simulation)

**Paper**: "Simulating LLM Runtime Latency" (MIT Master's Thesis, 2025)

**Key technique**: Provides highly accurate runtime predictions and precomputed performance profiles sufficient to simulate workloads. Enables hardware exploration without GPU access.

**Relevance to vllm-tuner**: Could be used as a cheap surrogate model to replace or supplement expensive stress tests during tuning. If vLLMSim can predict throughput/latency for a given configuration with reasonable accuracy, it could be used for initial screening before real evaluation.

**Implementation complexity**: Medium -- requires integration with vLLMSim.

---

### 4.3 LLM Inference Scheduling Survey

**Paper**: "LLM Inference Scheduling: A Survey of Techniques, Frameworks, and Trade-offs"
**Source**: TechRxiv, October 2025

**Key coverage**: Comprehensive survey of scheduling techniques covering both control plane (admission control, queue management, load balancing) and data plane (batching, memory management, KV cache, speculative execution).

**Relevance to vllm-tuner**: Provides a taxonomy of all tunable dimensions in LLM inference, helping to ensure vllm-tuner covers the complete parameter space.

**Implementation complexity**: N/A -- reference material.

---

### 4.4 Roofline Analysis for LLM Inference

**Context**: Multiple papers apply roofline modeling to LLM inference, characterizing whether operations are compute-bound or memory-bandwidth-bound.

**Key insight**: Prefill is typically compute-bound (benefits from more FLOPs), while decode is memory-bandwidth-bound (benefits from more memory bandwidth). This fundamental difference drives the need for different optimization strategies per phase.

**Relevance to vllm-tuner**: Understanding the compute/memory regime for a given workload helps predict which parameters will be most impactful. For memory-bound decode workloads, `max-num-seqs` (batching more requests) is more impactful. For compute-bound prefill, `enable-chunked-prefill` and chunk size matter more.

**Implementation complexity**: Medium -- requires workload characterization.

---

### 4.5 MLArchSys LLM Serving Simulator

**Paper**: "LLMServingSim" (ISCA 2024 Workshop)
**Source**: https://jongse-park.github.io/files/paper/2024-mlarchsys-llmservingsim.pdf

**Key technique**: Simulation framework for LLM serving system design space exploration.

**Relevance to vllm-tuner**: Could provide cheap evaluations to augment real stress tests, dramatically reducing tuning time.

**Implementation complexity**: Medium-high.

---

## 5. Optimization Frameworks and Alternatives to Optuna

### 5.1 Optuna (Current vllm-tuner Backend)

**Strengths**:
- User-friendly API with define-by-run syntax.
- TPE (Tree-structured Parzen Estimator) sampler -- efficient for moderate-dimensional spaces.
- Built-in pruning (early stopping of unpromising trials).
- Multi-objective optimization support (NSGA-II, MOTPE).
- Extensive integration ecosystem (MLflow, W&B, etc.).
- OptunaHub for community-contributed samplers (including LLAMBO).

**Limitations**:
- TPE is less sample-efficient than GP-based BO for small budgets.
- No native support for hidden constraint learning.
- No parallel suggestion with acquisition function diversity (MACE-style).

---

### 5.2 HEBO (Potential Replacement)

**Advantages over Optuna for vllm-tuner**:
- More sample-efficient (GP-based, not TPE).
- Input/output warping handles heteroscedasticity.
- Multi-objective acquisition ensembles.
- SCOOT is built on HEBO, so adoption is validated.

**Disadvantages**:
- Less ecosystem integration.
- Smaller community.
- No built-in pruning.

---

### 5.3 Ray Tune

**Strengths**: Distributed execution, many search algorithms (including Optuna and Bayesian backends), PBT (Population-Based Training), ASHA scheduler.

**Relevance**: Good for distributed tuning across multiple GPU nodes, but adds significant infrastructure complexity.

**Implementation complexity**: Medium-high.

---

### 5.4 Hyperopt

**Strengths**: TPE algorithm, MongoDB-based distributed execution.
**Weaknesses**: Less actively maintained, inferior API compared to Optuna.

**Recommendation**: Not recommended over Optuna.

---

### 5.5 LLM-Based Optimization (Emerging)

**Paper**: "Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?" (ICCV 2025 Workshop)

**Finding**: LLMs can achieve competitive or superior results compared to Optuna while reducing computational overhead, especially for early-stage exploration.

**Relevance to vllm-tuner**: Could be used for initialization or warmstarting.

**Implementation complexity**: Low -- available as OptunaHub integration.

---

## 6. Synthesis: Implications for vllm-tuner

### 6.1 High-Priority Improvements (from SCOOT and related work)

| Improvement | Source | Impact | Complexity |
|-------------|--------|--------|------------|
| Hidden constraint learning (random forest) | SCOOT | Critical -- prevents wasted evaluations on crashing configs | Medium |
| Known constraint pruning | SCOOT | High -- reduces search space significantly | Low |
| MACE acquisition ensemble | SCOOT/HEBO | High -- more robust than single acquisition function | Medium |
| Multi-objective optimization (TTFT+TPOT) | SCOOT | High -- essential for interactive applications | Medium |
| Parallel suggestion | SCOOT | High -- cuts tuning time proportionally | Low |
| Dynamic feasibility threshold | SCOOT | Medium -- fine-tunes exploration near constraint boundaries | Low |
| SLO robustness testing | SCOOT | Medium -- ensures worst-case guarantees | Low |

### 6.2 Extended Parameter Space (from vLLM evolution)

Modern vLLM (v0.6+) has additional critical parameters not in SCOOT:

| Parameter | Impact | Source |
|-----------|--------|--------|
| `--num-scheduler-steps` | Multi-step scheduling -- 1.8-2.7x throughput | vLLM v0.6.0 |
| `--gpu-memory-utilization` | Controls KV cache memory allocation | vLLM core |
| `--max-model-len` | Limits context length, affects memory | vLLM core |
| `--enforce-eager` | Disables CUDA graphs for debugging | vLLM core |
| Speculative decoding params | Draft model, speculation length | vLLM speculative |
| KV cache quantization | FP8 KV cache | vLLM recent |
| Disaggregated serving params | Prefill/decode separation | vLLM/DistServe |

### 6.3 Recommended Architecture for vllm-tuner

Based on the literature review, the ideal vllm-tuner should:

1. **Optimization Backend**: Use Optuna with HEBO-inspired enhancements, or switch to HEBO directly. The LLAMBO sampler from OptunaHub provides an easy LLM-warmstarting capability.

2. **Constraint Handling**: Implement a two-tier constraint system:
   - Tier 1: Known constraints (from vLLM docs) to prune search space.
   - Tier 2: Random forest classifier to learn hidden constraints from crash observations.

3. **Acquisition Strategy**: If using Optuna, implement MACE-style ensemble by running multiple samplers and selecting from their combined suggestions. If using HEBO, this is built-in.

4. **Multi-Objective Support**: Support joint optimization of TTFT+TPOT via Pareto front discovery (EHVI or NSGA-II). Allow users to specify which metrics to optimize.

5. **Parallel Evaluation**: Support configurable parallelism degree (PD) for evaluating multiple configurations simultaneously on different GPU sets.

6. **Workload Characterization**: Before tuning, profile the workload to determine:
   - Whether it's prefill-heavy or decode-heavy (affects chunked prefill recommendation).
   - Whether requests share common prefixes (affects prefix caching recommendation).
   - Request arrival pattern (bursty vs. uniform affects scheduler-delay-factor).

7. **Surrogate Pre-Screening**: Use an analytical performance model (e.g., roofline analysis) to pre-filter obviously bad configurations before expensive stress tests.

8. **Transfer Learning / Warmstarting**: Store results from previous tuning sessions and use them to warmstart new sessions on similar hardware/model combinations (inspired by Restune).

### 6.4 Key Research Gaps Identified

1. **No analytical cost model for vLLM parameter interactions**: All existing work relies on black-box evaluation. An analytical model mapping (parameters, workload) -> (throughput, latency) would be transformative.

2. **No dynamic workload adaptation**: All tuning approaches assume static workload. Real production traffic is non-stationary. Online re-tuning or adaptive parameter adjustment is unexplored.

3. **No co-tuning of model compression and serving parameters**: Quantization, pruning, and serving parameters are tuned independently, but they interact significantly.

4. **Limited multi-engine comparison**: SCOOT tunes one engine at a time. A meta-tuner comparing vLLM, TensorRT-LLM, SGLang, etc. for a given workload would be valuable.

5. **No hardware-aware warm-starting**: Transferring tuning knowledge from one GPU type to another (e.g., A100 -> H100) is unexplored.

---

## References

### Primary Paper
1. Ke Cheng et al. "SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines." WWW 2025. https://doi.org/10.1145/3696410.3714930

### LLM Serving Systems
2. Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. https://arxiv.org/abs/2309.06180
3. Gyeong-In Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022.
4. Amey Agrawal et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI 2024. https://arxiv.org/abs/2403.02310
5. Yinmin Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024. https://arxiv.org/abs/2401.09670
6. Yaniv Leviathan et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
7. Xupeng Miao et al. "SpecInfer: Accelerating Large Language Model Serving with Tree-Based Speculative Inference." ASPLOS 2024.
8. Tri Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
9. Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2023.

### Bayesian Optimization and Tuning
10. Alexander I Cowen-Rivers et al. "HEBO: Pushing the Limits of Sample-Efficient Hyperparameter Optimisation." JAIR 2022. https://arxiv.org/abs/2012.03826
11. Jiale Lao et al. "GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization." PVLDB 2024. https://arxiv.org/abs/2311.03157
12. Tennison Liu et al. "Large Language Models to Enhance Bayesian Optimization." ICLR 2024. https://arxiv.org/abs/2402.03921
13. Xinyi Zhang et al. "Restune: Resource Oriented Tuning Boosted by Meta-Learning for Cloud Databases." SIGMOD 2021.
14. Valentin Dalibard et al. "Boat: Building Auto-Tuners with Structured Bayesian Optimization." WWW 2017.
15. Tiannuo Yang et al. "Cotumer: A Hierarchical Learning Framework for Coordinating Optimizing Resource Partitioning and Parameter Tuning." ICPP 2023.

### Performance Modeling
16. "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling." 2025. https://arxiv.org/pdf/2508.00904
17. Sarah Y. Wang. "Simulating LLM Runtime Latency." MIT Master's Thesis, 2025.
18. "LLM Inference Scheduling: A Survey of Techniques, Frameworks, and Trade-offs." TechRxiv, 2025.

### KV Cache Optimization
19. "A Survey on Large Language Model Acceleration based on KV Cache Management." OpenReview 2024.
20. "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference." 2024. https://arxiv.org/pdf/2510.09665
21. "Marconi: Prefix Caching for the Era of Hybrid LLMs." 2024. https://arxiv.org/abs/2411.19379

### vLLM Resources
22. vLLM Optimization and Tuning Documentation: https://docs.vllm.ai/en/latest/configuration/optimization/
23. vLLM v0.6.0 Performance Update: https://blog.vllm.ai/2024/09/05/perf-update.html
24. vLLM 2024 Retrospective and 2025 Vision: https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html
25. vLLM Multi-Step Scheduling RFC: https://github.com/vllm-project/vllm/issues/6854
