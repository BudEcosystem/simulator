# Web Research: vLLM Tuning, Optimization & Best Practices

Comprehensive findings from web research on vLLM tuning, auto-tuning techniques, GPU memory management, production deployment patterns, and advanced features.

---

## Topic 1: vLLM Official Documentation & Tuning

### 1.1 Core Configuration Parameters

Source: [vLLM Optimization and Tuning Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)

| Parameter | Default | Description | Tuning Impact |
|-----------|---------|-------------|---------------|
| `gpu_memory_utilization` | 0.9 | Fraction of GPU VRAM pre-allocated for KV cache | Higher = more KV cache space = more concurrent requests. Set as high as possible without OOM (0.90-0.95 recommended). |
| `max_num_batched_tokens` | 2048 (varies) | Maximum tokens processed per scheduler iteration | Lower (2048) = better ITL; Higher (>8192) = better TTFT and throughput. Key latency-throughput tradeoff lever. |
| `max_num_seqs` | 256 | Maximum concurrent sequences in active batch | 128-256 for typical workloads. Too low (<2) underutilizes GPU; too high (2000+) causes OOM. Acts as ceiling; actual concurrency determined dynamically by KV cache. |
| `max_model_len` | Model default | Maximum context length in tokens | Reduce for short-form tasks to save memory. Most effective memory-saving lever for high concurrency. |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism | Set to number of GPUs in node. Shards model weights, freeing per-GPU memory for KV cache. |
| `pipeline_parallel_size` | 1 | Number of pipeline stages across nodes | Use for multi-node; set to number of nodes. Combines with TP for very large models. |
| `num_scheduler_steps` | 1 | Steps per scheduler invocation (multi-step) | Set 10-15 for optimal GPU utilization. Diminishing returns above 15. |
| `dtype` | auto | Model weight data type | FP16 for FP32/FP16 models; BF16 for BF16 models. Use `fp8` for dynamic quantization. |
| `kv_cache_dtype` | auto | KV cache data type | `fp8` reduces KV cache memory by 50%. |
| `enforce_eager` | False | Disable CUDA graphs | Set True for debugging; False for production (CUDA graphs improve performance). |
| `compilation_level` | - | torch.compile optimization level | Level 3 recommended for production. |

Source: [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/), [Anyscale Parameter Tuning](https://docs.anyscale.com/llm/serving/parameter-tuning)

### 1.2 Memory Budget Breakdown

The GPU memory is divided into:
1. **Model weights** - Fixed cost, determined by model size and dtype
2. **Activations** - Temporary, scales with batch size
3. **KV cache** - Dynamic, managed by `gpu_memory_utilization`

Formula for KV cache memory per token:
```
kv_cache_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_size
```

Total KV cache memory:
```
kv_cache_memory = batch_size * seq_len * kv_cache_per_token
```

Source: [GPU Memory Calculation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/)

### 1.3 Preemption Handling

When KV cache space is insufficient for all batched requests, vLLM preempts lower-priority requests to free KV cache. Preempted requests are recomputed when space becomes available. To reduce preemption:
- Increase `gpu_memory_utilization`
- Reduce `max_model_len`
- Add more GPUs (tensor/pipeline parallelism)

### 1.4 CPU Resource Requirements

Minimum CPU cores: `2 + N` physical cores where N = number of GPU workers:
- 1 core for API server
- 1 core for engine core
- 1 core per GPU worker

In practice, allocate more cores for OS, PyTorch background threads, and system processes.

Source: [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)

---

## Topic 2: Blog Posts & Articles on vLLM Optimization

### 2.1 vLLM v0.6.0 Performance Improvements

Source: [vLLM v0.6.0 Blog Post](https://blog.vllm.ai/2024/09/05/perf-update.html)

**Key Results:**
- Llama 8B (1xH100): **2.7x throughput**, **5x faster TPOT** vs v0.5.3
- Llama 70B (4xH100): **1.8x throughput**, **2x faster TPOT** vs v0.5.3

**Root Cause of Prior Bottleneck:**
CPU execution breakdown before optimization:
- HTTP API server: 33%
- Scheduling + request preparation: 29%
- Actual GPU execution: only 38%

**Architectural Changes:**
1. **API Server Separation**: Decoupled HTTP server from inference engine via ZMQ sockets, eliminating Python GIL contention
2. **Multi-Step Scheduling**: Batches multiple decode steps per scheduler call (28% throughput improvement on Llama 70B)
3. **Asynchronous Output Processing**: Output handling overlaps with model execution (8.7% TPOT improvement)
4. **Object Pooling**: Reduced allocation overhead (24% throughput gain)
5. **Non-blocking CPU-to-GPU transfers**: Eliminated synchronous data movement
6. **Fast sampling paths**: Optimized code path for simple sampling requests

**Actionable Insight for vllm-tuner:** The CPU overhead breakdown reveals that tuning is not just about GPU parameters -- the interplay between scheduling, batching, and API overhead matters significantly. Multi-step scheduling (`num_scheduler_steps=10-15`) is a critical tuning parameter.

### 2.2 vLLM V1 Architecture

Source: [vLLM V1 Alpha Release](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

**Performance:**
- Up to **1.7x higher throughput** vs V0 (without multi-step scheduling)
- Consistently lower latency at high QPS
- Vision-language models see even larger speedups

**Key Architectural Changes:**
1. **Unified Scheduler**: Eliminates prefill/decode distinction; treats all tokens uniformly. Scheduling = dict mapping request IDs to token counts.
2. **Near-Zero Prefix Caching Overhead**: Less than 1% throughput decrease even at 0% cache hit rate. Safe to enable by default.
3. **Persistent Batch**: Caches tensors with diff-based updates for input preparation
4. **Symmetric TP Architecture**: Worker-side state caching with differential updates
5. **FlashAttention 3**: Flexible high-performance attention for dynamic batching

**Enable V1:** `export VLLM_USE_V1=1`

**Actionable Insight for vllm-tuner:** V1's near-zero prefix caching overhead means `enable_prefix_caching` should always be True in the tuner's search space (or defaulted on). The unified scheduler simplifies the tuning landscape.

### 2.3 vLLM 2024 Retrospective & 2025 Vision

Source: [vLLM Blog 2025 Vision](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html)

**2025 Focus Areas:**
- KV cache and attention optimization (sliding windows, cross-layer attention, native quantization)
- MoE optimizations (shared experts, expert parallelism)
- Disaggregated prefill/decode serving
- Production stack improvements

### 2.4 Anyscale Production Serving Guide

Source: [Anyscale Parameter Tuning](https://docs.anyscale.com/llm/serving/parameter-tuning), [Performance Optimization](https://docs.anyscale.com/llm/serving/performance-optimization)

**Key Recommendations:**
1. Find maximum KV cache concurrency from deployment logs
2. Calculate effective batch size from average request token length
3. Set `max_num_seqs` to calculated concurrency value
4. Use `max_num_batched_tokens` as primary latency/throughput lever
5. Run `vllm bench serve` with various `--request-rate` and `--max-concurrency` values

**Actionable Insight for vllm-tuner:** The tuner should include a "workload profiling" phase that estimates average prompt/output lengths and request rates, then uses these to set initial parameter ranges before Bayesian optimization begins.

### 2.5 Google Cloud xPU Tuning Guide

Source: [Google Cloud vLLM Tuning Blog](https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration)

**Key Insights:**
- `gpu_memory_utilization` should be set as high as possible (0.90-0.95)
- Prefix caching is essential for workloads with shared system prompts
- TP adds communication overhead -- balance performance gain against inter-accelerator latency
- Monitor `num_requests_waiting` and `gpu_cache_usage_perc` for scaling decisions

### 2.6 vLLM vs Competitors Comparison

Source: [vLLM vs TGI vs TensorRT-LLM](https://compute.hivenet.com/post/vllm-vs-tgi-vs-tensorrt-llm-vs-ollama), [MarkTechPost Comparison](https://www.marktechpost.com/2025/11/19/vllm-vs-tensorrt-llm-vs-hf-tgi-vs-lmdeploy-a-deep-technical-comparison-for-production-llm-inference/)

| Feature | vLLM | TensorRT-LLM | TGI v3 |
|---------|------|---------------|--------|
| Throughput (req/sec) | 120-160 | 180-220 | 100-140 |
| TTFT (100 concurrent) | 50-80ms | 35-50ms (low conc.) | Varies |
| Long prompt performance | Baseline | Similar | 3x-13x faster |
| Setup complexity | Low | High (compilation) | Low |
| Model support | Broad (HF) | NVIDIA-specific | HF ecosystem |
| Key technology | PagedAttention | CUDA graph fusion | Chunking + prefix cache |

**Actionable Insight for vllm-tuner:** For interactive chat with high concurrency, vLLM is optimal. The tuner should optimize for the specific workload pattern (chat vs. batch vs. long-context) since competitor advantages vary by use case.

---

## Topic 3: Auto-Tuning Techniques for Inference Systems

### 3.1 SCOOT: SLO-Oriented Performance Tuning

Source: [SCOOT Paper (arXiv 2408.04323)](https://arxiv.org/abs/2408.04323), [SCOOT GitHub](https://github.com/antgroup/SCOOT-SLO-Oriented-Performance-Tuning), [ACM WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714930)

**Architecture:**
- Uses joint single-objective and multi-objective Bayesian Optimization (BO)
- Prunes search space with known constraints
- Uses Random Forest to learn hidden constraints during tuning
- Handles multiple SLO objectives simultaneously (throughput, tail latency, TTFT, TPOT)

**Performance Results:**
- Throughput improvement: up to **68.3%** vs default configs
- Tail latency reduction: up to **40.6%**
- TTFT reduction: up to **99.8%**
- TPOT reduction: up to **61.0%**

**Universally applicable** to vLLM and TensorRT-LLM. Deployed in production at Ant Group.

**Actionable Insights for vllm-tuner:**
1. Search space pruning with known constraints is critical -- eliminates invalid configurations early
2. Random Forest for learning hidden constraints during optimization is a powerful technique
3. Multi-objective BO is essential when optimizing for SLAs (not just throughput)
4. The tuner should support both single-objective (maximize throughput) and multi-objective (throughput + latency SLA) modes

### 3.2 Optimization Framework Comparison

Source: [Analytics Insight](https://www.analyticsinsight.net/machine-learning/best-tools-for-hyperparameter-optimization-in-machine-learning), [Neptune.ai](https://neptune.ai/blog/best-tools-for-model-tuning-and-hyperparameter-optimization), [Ray Tune Docs](https://docs.ray.io/en/latest/tune/api/suggestion.html)

| Framework | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| **Optuna** | Clean API, pruning, async parallel, fast (35% faster than Hyperopt), define-by-run | Less distributed than Ray Tune | Most tuning tasks; standalone Python projects |
| **SMAC3** | Handles conditional/structured spaces, robust BO | Steep learning curve, scalability issues | Complex config spaces with conditionals |
| **Ray Tune** | Multi-GPU/node, integrates BO backends (Ax, Hyperopt, Optuna), fault-tolerant | Heavier dependency; needs Ray cluster | Distributed training/tuning at scale |
| **BoTorch** | Flexible GP models, cost-aware BO, multi-objective (qNEHVI) | Lower-level API, requires Ax for user-friendliness | Custom acquisition functions, research |
| **Hyperopt** | TPE sampler, mature | Slower than Optuna, less maintained | Legacy projects |
| **Vizier** | Google-backed, multi-objective, production-grade | Less community adoption | Google Cloud integration |

**Recommendation for vllm-tuner:** Optuna is the strongest choice for the primary optimization backend due to:
1. Best speed/quality tradeoff for typical tuning tasks
2. Native multi-objective support (via `optuna.create_study(directions=...)`)
3. Built-in pruning with `MedianPruner` or `HyperbandPruner`
4. Easy integration with existing Python code
5. Supports custom samplers (TPE, CMA-ES, GP-based)

Consider BoTorch integration (via Ax) for cost-aware optimization where evaluation budgets matter.

### 3.3 Cost-Aware Bayesian Optimization

Source: [Amazon Science](https://www.amazon.science/publications/cost-aware-bayesian-optimization), [BoTorch Tutorial](https://botorch.org/docs/v0.14.0/tutorials/cost_aware_bayesian_optimization/), [Intel Developer](https://www.intel.com/content/www/us/en/developer/articles/technical/cost-matters-importance-cost-aware-hyperparameter-optimization.html)

**Key Concepts:**
- Standard BO measures convergence in iterations (each iteration assumed equal cost)
- In practice, evaluation costs vary dramatically (e.g., large batch size = longer benchmark)
- **Cost-aware BO** weights acquisition functions by expected evaluation cost

**Techniques:**
1. **EI per unit cost (EIpu)**: Divides Expected Improvement by predicted cost
2. **Cost-cooling**: Starts with EIpu, transitions to standard EI as optimization progresses
3. **CArBO (Cost Apportioned BO)**: Combines cost-effective initial design with cost-cooling
4. **Pandora's Box Gittins Index (PBGI)**: Principled stopping rule that adapts to varying costs

**Actionable Insight for vllm-tuner:** Benchmark evaluation costs vary significantly based on configuration (e.g., large batch size + long sequences = minutes vs. small batch + short sequences = seconds). The tuner should:
1. Track evaluation cost per trial
2. Use cost-aware acquisition functions (EIpu) to explore cheap configurations first
3. Implement early stopping for clearly poor configurations
4. Consider total wall-clock budget, not just iteration count

### 3.4 SIGMOD 2025 Autotuning Tutorial

Source: [Microsoft Research - SIGMOD 2025](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/06/SIGMOD_2025_Autotuning_Tutorial.pdf)

**Key Patterns for System Autotuning:**
- Offline optimization: Use trial history to guide tuning (surrogate models)
- Online optimization: Adapt parameters during serving
- Transfer learning: Reuse tuning knowledge across similar workloads/hardware
- Multi-fidelity: Use cheap approximations (shorter benchmarks) for initial screening

---

## Topic 4: GPU Memory Management & Scheduling

### 4.1 GPU Memory Utilization Deep Dive

Source: [vLLM GPU Memory Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/), [DigitalOcean GPU Sizing Guide](https://www.digitalocean.com/community/conceptual-articles/vllm-gpu-sizing-configuration-guide)

**Memory Allocation Order:**
1. Model weights loaded first (fixed)
2. Activation memory reserved (proportional to batch size)
3. Remaining memory allocated to KV cache (controlled by `gpu_memory_utilization`)

**KV Cache Sizing Formula:**
```
available_kv_memory = total_gpu_memory * gpu_memory_utilization - model_weight_memory - activation_memory
num_kv_blocks = available_kv_memory / (block_size * kv_cache_per_token)
max_concurrent_tokens = num_kv_blocks * block_size
```

**Tuning Strategy:**
- Start with `gpu_memory_utilization=0.90` (default)
- If frequently preempting: increase to 0.92-0.95
- If OOM: decrease to 0.85-0.88
- Monitor `gpu_cache_usage_perc` metric -- if consistently <50%, memory is wasted

**Multi-GPU Memory:**
When stages share a GPU, sum of `gpu_memory_utilization` must not exceed 1.0. When on separate GPUs, each can use up to 1.0.

### 4.2 max_num_seqs and Batch Size Optimization

Source: [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [Medium - vLLM Parameters](https://medium.com/@kaige.yang0110/vllm-throughput-optimization-1-basic-of-vllm-parameters-c39ace00a519)

**Behavior:**
- `max_num_seqs` is a ceiling, not actual concurrency
- vLLM dynamically determines actual batch size based on available KV cache
- Setting too high causes OOM; setting too low wastes GPU compute

**Traffic Pattern Considerations:**
- **Bursty traffic**: High `max_num_seqs` (256-512) to absorb spikes
- **Constant-rate traffic**: Lower value reduces individual request latency
- **Mixed workloads**: Use default (256) with monitoring

**Interaction with max_num_batched_tokens:**
- `max_num_batched_tokens` controls total tokens per batch (prefill + decode)
- `max_num_seqs` controls number of individual sequences
- Effective batch size = min(max_num_seqs, tokens_that_fit_in_max_num_batched_tokens)

### 4.3 Tensor Parallel vs Pipeline Parallel Trade-offs

Source: [BentoML LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism), [vLLM Parallelism Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/), [ROCm TP Analysis](https://rocm.blogs.amd.com/artificial-intelligence/tensor-parallelism/README.html)

**Tensor Parallelism (TP):**
- Slices individual layers across GPUs
- Requires AllReduce after each layer (high communication)
- Best within a node (NVLink/NVSwitch)
- Each GPU holds 1/TP_size of model weights, leaving more room for KV cache
- Super-linear KV cache scaling: more GPUs = disproportionately more cache = larger batch sizes

**Pipeline Parallelism (PP):**
- Assigns contiguous layer groups to different GPUs
- Communication only at pipeline stage boundaries (lower overhead)
- Causes pipeline bubbles (GPU idle time)
- Best across nodes (slower interconnect)
- May add latency penalties

**Practical Guidelines:**
- Single node: Use TP = number of GPUs
- Multi-node: TP within node, PP across nodes
- For MoE models: Consider Expert Parallelism (EP) as alternative to TP

**KV Cache Impact:**
- Higher TP = each GPU has less model weight memory = more KV cache per GPU
- Memory benefit is super-linear because larger caches enable larger batch sizes
- Lower TP = less room for KV cache = may degrade prefix caching effectiveness

**Actionable Insight for vllm-tuner:** The tuner should model the TP/PP configuration space carefully. For single-node multi-GPU, TP is almost always better. The tuner should include TP and PP as discrete parameters in the search space, with constraints like `TP * PP = total_gpus`.

### 4.4 KV Cache Block Size

Source: [vLLM Parallelism Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/), [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)

- vLLM uses block-based KV cache management (PagedAttention)
- Default block size: 16 tokens
- Smaller blocks = less internal fragmentation, more overhead
- Larger blocks = less overhead, more waste
- Block size is generally not a high-priority tuning parameter for most workloads
- For very short sequences, smaller blocks reduce waste; for long sequences, larger blocks reduce metadata overhead

---

## Topic 5: Production Deployment Patterns

### 5.1 vLLM Production Stack

Source: [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack), [IBM Research](https://research.ibm.com/publications/scalable-and-efficient-llm-serving-with-the-vllm-production-stack)

**Architecture:**
- Kubernetes-native cluster deployment
- Helm chart-based installation
- Includes prefix-aware routing, KV cache sharing, observability

**Key Features:**
1. **Prefix-aware routing**: Directs requests to vLLM instances holding relevant KV cache
2. **KV cache sharing**: Shares computed KV blocks across instances
3. **Observability**: Prometheus/Grafana metrics for engine status and autoscaling
4. **Auto-scaling**: Based on request queue depth and cache utilization

### 5.2 Autoscaling Strategies

Source: [Red Hat Autoscaling vLLM](https://developers.redhat.com/articles/2025/11/26/autoscaling-vllm-openshift-ai-model-serving), [Kubernetes HPA Guide](https://medium.com/@shivank1128/deploying-a-production-ready-vllm-stack-on-kubernetes-with-hpa-autoscaling-107501b8b687)

**KEDA-Based SLO-Driven Scaling:**
- Uses actual SLI metrics (ITL, end-to-end latency) rather than simple CPU/memory
- For heterogeneous workloads: KServe + KEDA achieved 86.9% request success rate
- Scales based on `num_requests_waiting` and `gpu_cache_usage_perc`

**Kubernetes HPA:**
- Use ReadWriteMany (RWX) storage for model weights (critical for multi-pod scaling)
- Dramatically reduces scaling time vs. downloading model per pod

**Actionable Insight for vllm-tuner:** The tuner should output not just engine parameters but also scaling recommendations (min/max replicas, target metrics for autoscaler). This makes tuner output directly deployable.

### 5.3 Disaggregated Prefill/Decode

Source: [vLLM Blog Large Scale Serving](https://blog.vllm.ai/2025/12/17/large-scale-serving.html), [Anyscale Performance Optimization](https://docs.anyscale.com/llm/serving/performance-optimization)

**Pattern:**
- Separate prefill and decode into different GPU pools
- Prefill is compute-bound; decode is memory-bound
- Each pool can be independently scaled and optimized
- Ray Serve LLM provides first-class support for this pattern

**Benefits:**
- Independent autoscaling per phase
- Optimized GPU utilization (prefill GPUs run at high compute, decode GPUs maximize batch size)
- Better tail latency control

### 5.4 Multi-Model Serving

Source: [Kubernetes Multi-Model Deployment](https://medium.com/@shivank1128/deploying-a-production-ready-vllm-stack-on-kubernetes-with-hpa-autoscaling-107501b8b687)

**Patterns:**
- Separate vLLM instances per model with individual HPA
- Shared GPU with `gpu_memory_utilization` split (sum < 1.0)
- Router-based request dispatch based on model ID

### 5.5 Latency vs Throughput Optimization

Source: [Databricks LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), [Sarathi-Serve Paper](https://arxiv.org/html/2403.02310v1), [BentoML Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)

**Key Metrics:**
| Metric | Definition | Target For |
|--------|-----------|------------|
| TTFT | Time to first token | Interactive responsiveness |
| TPOT | Time per output token | Streaming fluency |
| ITL | Inter-token latency | Consistent streaming |
| Throughput | Tokens/sec or req/sec | Batch processing, cost efficiency |
| Goodput | Throughput meeting SLA | Production SLO compliance |
| P95/P99 latency | Tail latency percentiles | SLA definition |

**Trade-off Dynamics:**
- Batch size 1: Best latency, worst throughput
- Batch size 64: ~14x throughput but ~4x latency increase (A100 benchmark)
- Continuous batching mitigates this by dynamically adding/removing requests

**Optimization Strategies:**
1. **For latency-sensitive (chat)**: Lower `max_num_batched_tokens` (2048), lower `max_num_seqs`
2. **For throughput (batch)**: Higher `max_num_batched_tokens` (>8192), higher `max_num_seqs`
3. **Goodput-focused**: Maximize throughput subject to P95 latency constraints
4. **SLA-aware scheduling**: Dynamically adjust batch composition based on queue length and arrival rate

**Actionable Insight for vllm-tuner:** The tuner needs a "mode" or "objective" parameter that selects between:
- `latency` mode: Optimize for TTFT + TPOT, constrain throughput
- `throughput` mode: Optimize for tokens/sec, constrain latency
- `balanced` mode: Multi-objective optimization (Pareto frontier)
- `sla` mode: Maximize goodput under specific SLA constraints (P95 TTFT < X, P95 TPOT < Y)

---

## Topic 6: Advanced vLLM Features

### 6.1 Chunked Prefill

Source: [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [vLLM Engine Args](https://docs.vllm.ai/en/v0.8.3/serving/engine_args.html)

**How it works:**
- Large prefills are chunked into smaller pieces
- Chunks are batched together with decode requests
- Scheduling policy: decode requests prioritized, then pending prefills fill remaining token budget

**Configuration:**
```
--enable-chunked-prefill  # or enable_chunked_prefill=True
```

**Tuning:**
- `max_num_batched_tokens` controls chunk size when chunked prefill is enabled
- Lower value = smaller prefill chunks = less interference with decode = better ITL
- Higher value = faster prefill processing = better TTFT
- Recommended: Start with 2048, increase for throughput-focused workloads

**When to use:**
- Mixed prefill/decode workloads (most production scenarios)
- When ITL consistency matters (streaming applications)
- When serving models with varying prompt lengths

### 6.2 Prefix Caching (Automatic Prefix Caching / APC)

Source: [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [Prompt Caching Blog](https://sankalp.bearblog.dev/how-prompt-caching-works/)

**How it works:**
- Caches KV blocks for shared prefixes across requests
- Uses block hashing and string-matching heuristics (not just exact token-ID matches)
- In V1: near-zero overhead (less than 1% throughput decrease at 0% hit rate)

**Configuration:**
```
--enable-prefix-caching  # Enable APC
--no-enable-prefix-caching  # Disable APC
--prefix-caching-hash-algo builtin  # or sha256 (collision-resistant but slower)
```

**When beneficial:**
- Many requests share system prompts (chatbots, assistants)
- Multi-turn conversations with shared history
- Batch processing with common document prefixes

**V1 vs V0:**
- V0: Disabled by default due to overhead
- V1: Near-zero overhead, safe to enable always

**Actionable Insight for vllm-tuner:** In V1, `enable_prefix_caching` should default to True. The tuner should detect workload patterns (shared prefix ratio) and adjust cache-related parameters accordingly.

### 6.3 Speculative Decoding

Source: [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [Anyscale Optimization](https://docs.anyscale.com/llm/serving/performance-optimization)

**How it works:**
1. Small "draft" model proposes N candidate tokens
2. Main model verifies proposals in single forward pass
3. Correct predictions accepted; incorrect ones rejected and regenerated
4. Net effect: multiple tokens generated per forward pass of the main model

**Supported Methods:**
- Draft model (separate smaller model)
- N-gram matching (no additional model needed)
- Suffix decoding
- MLP speculators
- EAGLE-based models

**Configuration:**
```
--speculative-model <draft_model>
--num-speculative-tokens <N>  # Number of tokens to speculate
--speculative-disable-mqa-scorer  # Disable MQA scorer
```

**When beneficial:**
- Low-concurrency, latency-sensitive workloads
- When draft model acceptance rate is high (>70%)
- NOT beneficial for high-throughput batch workloads (adds overhead)

**Actionable Insight for vllm-tuner:** Speculative decoding parameters should be in the tuner's search space for latency-focused optimization. The tuner should measure acceptance rate and disable speculative decoding if it's too low.

### 6.4 Quantization Performance

Source: [GPUStack Quantization Impact](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/), [JarvisLabs Quantization Guide](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks), [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/features/quantization/)

**Performance Comparison:**

| Method | Type | Throughput vs BF16 | Quality | Memory Savings |
|--------|------|--------------------|---------|----------------|
| FP8 (W8A8) | Weight-Activation | +30-50% | Lossless | ~50% |
| AWQ (W4) | Weight-only | ~3x req/sec | Good | ~75% |
| GPTQ (W4) | Weight-only | ~3x req/sec | Better on coding | ~75% |
| Marlin-AWQ | Optimized kernel | 10.9x speedup | Good | ~75% |
| Marlin-GPTQ | Optimized kernel | 2.6x speedup | Good | ~75% |

**Key Findings:**
- **FP8 is the sweet spot**: Lossless quality with significant speedup (recommended for most scenarios)
- **Weight-only quantization** (AWQ/GPTQ) has dequantization overhead when VRAM is not constrained
- **Marlin kernels** provide massive speedups for weight-only methods
- **Marlin-AWQ** is the overall sweet spot for quality + speed
- Combining quantization with chunked prefill and higher `max_num_seqs` yields 2-3x additional throughput

**Actionable Insight for vllm-tuner:** The tuner should include quantization method as a categorical parameter. FP8 should be the default recommendation. When GPU memory is tight, AWQ with Marlin kernels is the best alternative.

### 6.5 Multi-Step Scheduling

Source: [vLLM GitHub RFC](https://github.com/vllm-project/vllm/issues/6854), [vLLM GitHub Issue](https://github.com/vllm-project/vllm/issues/9158), [vLLM v0.6.0 Blog](https://blog.vllm.ai/2024/09/05/perf-update.html)

**How it works:**
- Multiple decode passes performed before GPU-CPU sync
- Scheduling and input preparation done once, model runs for N consecutive steps
- GPU-CPU memory transfer for sampled tokens happens in separate CUDA stream

**Configuration:**
```
--num-scheduler-steps <N>  # Default: 1, Recommended: 10-15
```

**Performance Impact:**
- 28% throughput improvement on Llama 70B (vLLM v0.6.0)
- Diminishing returns above 15 steps
- Slightly bumpy inter-token latency (tokens returned in batches)
- Higher TTFT at low request rates

**Note:** In V1, multi-step scheduling is architecturally integrated, making the `num_scheduler_steps` parameter less critical (V1 achieves similar benefits through its unified scheduler).

**Actionable Insight for vllm-tuner:** For V0, `num_scheduler_steps` is a high-impact tuning parameter (10-15 range). For V1, it's less important but should still be in the search space.

---

## Synthesis: Actionable Recommendations for vllm-tuner

### Priority 1: Core Tuning Parameters (Must-Have in Search Space)

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `gpu_memory_utilization` | [0.80, 0.95] | High | Primary memory lever |
| `max_num_batched_tokens` | [512, 32768] | High | Key latency/throughput tradeoff |
| `max_num_seqs` | [32, 512] | Medium-High | Concurrency control |
| `tensor_parallel_size` | [1, num_gpus] | High | Must divide num_gpus evenly |
| `max_model_len` | [256, model_max] | Medium-High | Memory savings lever |
| `num_scheduler_steps` | [1, 15] | Medium (V0 High) | Multi-step scheduling |
| `enable_chunked_prefill` | [True, False] | Medium | Enable for mixed workloads |
| `enable_prefix_caching` | [True, False] | Medium | Default True in V1 |

### Priority 2: Advanced Parameters (Should-Have)

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `dtype` | [auto, float16, bfloat16] | Low-Medium | Usually auto is fine |
| `kv_cache_dtype` | [auto, fp8] | Medium | 50% KV cache reduction |
| `quantization` | [None, fp8, awq, gptq] | High | Model-dependent |
| `speculative_model` | [None, draft_model] | Varies | Only for latency mode |
| `num_speculative_tokens` | [1, 7] | Low-Medium | Only with spec decode |
| `pipeline_parallel_size` | [1, num_nodes] | Medium | Multi-node only |

### Priority 3: Optimization Framework Recommendations

1. **Use Optuna** as primary optimizer with TPE sampler
2. **Implement cost-aware acquisition** (track trial duration, prefer cheap evaluations early)
3. **Support multi-objective optimization** for SLA-aware tuning
4. **Include constraint handling**: invalid configs (OOM, too-high TP) should be pruned
5. **Add early stopping**: use Optuna's `MedianPruner` or `HyperbandPruner`
6. **Consider transfer learning**: reuse results across similar model/hardware combos

### Priority 4: Benchmarking Methodology

1. **Standardize workload profiles**: Define standard workloads (chat, batch, long-context)
2. **Warm-up phase**: Run N requests before measurement (prefix cache warm-up)
3. **Measure multiple metrics**: TTFT, TPOT, ITL, throughput, goodput simultaneously
4. **Statistical rigor**: Multiple runs per config, report P50/P95/P99
5. **Capture hardware context**: GPU type, count, interconnect, driver version

### Priority 5: Production Integration

1. **Output deployment configs**: Generate vLLM serve command, Docker compose, K8s Helm values
2. **Scaling recommendations**: Min/max replicas, autoscaler metrics/thresholds
3. **Monitoring integration**: Expose tuner results as Prometheus metrics
4. **CI/CD integration**: Run tuner as part of deployment pipeline

---

## Key Sources Referenced

### Official Documentation
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [vLLM GPU Memory Calculation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/)
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)

### Blog Posts & Articles
- [vLLM v0.6.0 Performance Update](https://blog.vllm.ai/2024/09/05/perf-update.html)
- [vLLM V1 Alpha Release](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [vLLM 2024 Retrospective and 2025 Vision](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html)
- [Google Cloud vLLM Performance Tuning Guide](https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration)
- [Anyscale Parameter Tuning](https://docs.anyscale.com/llm/serving/parameter-tuning)
- [Anyscale Performance Optimization](https://docs.anyscale.com/llm/serving/performance-optimization)
- [Databricks LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [vLLM Large Scale Serving (DeepSeek)](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)
- [AI21 Scaling vLLM](https://www.ai21.com/blog/scaling-vllm-without-oom/)

### Comparisons & Benchmarks
- [vLLM vs TGI vs TensorRT-LLM (Hivenet)](https://compute.hivenet.com/post/vllm-vs-tgi-vs-tensorrt-llm-vs-ollama)
- [Deep Technical Comparison (MarkTechPost)](https://www.marktechpost.com/2025/11/19/vllm-vs-tensorrt-llm-vs-hf-tgi-vs-lmdeploy-a-deep-technical-comparison-for-production-llm-inference/)
- [GPUStack Quantization Impact](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/)
- [JarvisLabs Quantization Guide](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
- [ROCm Tensor Parallelism Analysis](https://rocm.blogs.amd.com/artificial-intelligence/tensor-parallelism/README.html)
- [ROCm vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [Comparative Analysis arXiv (vLLM vs TGI)](https://arxiv.org/html/2511.17593v1)

### Research Papers
- [SCOOT: SLO-Oriented Performance Tuning (arXiv 2408.04323)](https://arxiv.org/abs/2408.04323)
- [SCOOT at ACM WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714930)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/html/2403.02310v1)
- [Cost-Aware Bayesian Optimization (Amazon Science)](https://www.amazon.science/publications/cost-aware-bayesian-optimization)
- [Efficient Generative LLM Serving Survey (ACM)](https://dl.acm.org/doi/10.1145/3754448)
- [Black-box Auto-Tuning Comparative Analysis (USENIX ATC'18)](https://www.usenix.org/conference/atc18/presentation/cao)
- [SIGMOD 2025 Autotuning Tutorial](https://dl.acm.org/doi/10.1145/3722212.3725638)

### Optimization Frameworks
- [Optuna](https://optuna.org/)
- [BoTorch Cost-Aware BO](https://botorch.org/docs/v0.14.0/tutorials/cost_aware_bayesian_optimization/)
- [Ray Tune](https://docs.ray.io/en/latest/tune/api/suggestion.html)
- [SCOOT GitHub](https://github.com/antgroup/SCOOT-SLO-Oriented-Performance-Tuning)

### Production Deployment
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack)
- [Introl vLLM Production Deployment](https://introl.com/blog/vllm-production-deployment-inference-serving-architecture)
- [Red Hat Autoscaling vLLM](https://developers.redhat.com/articles/2025/11/26/autoscaling-vllm-openshift-ai-model-serving)
- [Red Hat vLLM Server Arguments](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/all-server-arguments-server-arguments)
- [Kubernetes vLLM Deployment Guide](https://dasroot.net/posts/2026/02/deploying-vllm-scale-kubernetes/)
