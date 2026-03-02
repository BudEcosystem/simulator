# LLMServingSim 2.0: Deep Analysis

**Paper:** "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure"
**Authors:** Jaehong Cho*, Hyunmin Choi*, Guseul Heo, Jongse Park (KAIST)
**Date:** February 2026, arXiv 2602.23036
**Repository:** https://github.com/casys-kaist/LLMServingSim

---

## 1. Architecture Deep-Dive

### 1.1 High-Level Overview

LLMServingSim 2.0 takes three input specifications:
1. **Workload configuration** -- LLM description, request patterns (arrival rates, per-request execution traces)
2. **Cluster configuration** -- node type/count, CPU settings, memory capacity/bandwidth, device placement, plus serving policies (request routing, parallelism strategies, KV cache eviction, compute/memory offloading decisions)
3. **Hardware performance profiles** -- operator-level latency and power measurements per model-device pair

It produces two categories of output:
- **System-level metrics:** memory usage over time, power/energy consumption, throughput
- **Per-request metrics:** TTFT, TPOT, queueing delay, end-to-end latency

### 1.2 Execution Planner

The Execution Planner is a one-time initialization phase that constructs the entire Serving Engine:

1. **Request Router configuration** -- Sets up routing logic to dispatch incoming requests to appropriate MSGs. For P/D disaggregation, the router directs requests to prefill MSGs and associates them with target decode MSGs.
2. **MSG instantiation** -- Creates one Model Serving Group per model. For P/D disaggregation, it creates M prefill MSGs + N decode MSGs for the same model. Each MSG gets:
   - A customizable **device pool** (mix of GPUs, NPUs, PIM devices, CXL-attached memory)
   - **Serving policies**: parallelism strategies (TP, PP, DP, EP), compute/memory offloading rules, KV cache management policies, memory sharing settings
   - **Operator-level performance profiles** populated from the profiler output
3. **System Simulator initialization** -- Configures cluster-level timing, network topology, memory hierarchy, and power model parameters.

After initialization, the Execution Planner hands control to the Serving Engine, which runs the iterative simulation loop.

### 1.3 Model Serving Group (MSG)

The MSG is the central abstraction -- a logical execution unit for serving one LLM instance. Its internal components form a pipeline:

#### Device Pool
- Contains one or more accelerators (heterogeneous allowed: GPU + PIM, GPU + NPU, etc.)
- Serves as the execution substrate for operator mapping
- Supports CXL-attached memory channels alongside compute devices

#### Request Queue
- Receives requests from the Request Router
- Tracks each request through prefill and decode phases
- Accumulates per-request statistics: queueing delay, TTFT, TPOT, end-to-end latency

#### Batch Scheduler
The batch scheduler is the decision-making core. For each scheduling cycle, it:
1. Selects pending requests from the queue to form a candidate batch
2. **Evaluates system and device memory capacity** -- checks if the batch fits given current memory state
3. **Computes KV cache footprint** -- calculates memory needed for new KV cache blocks
4. **Checks prefix-cache residency** -- determines which KV blocks already exist in the multi-tier cache, reducing both memory allocation and compute
5. **Enforces configured maximum batch size**
6. Consults the **memory model** to check if the candidate batch satisfies constraints including available resources, eviction costs, and tier-specific capacity
7. If feasible, forwards the batch to the Graph Generator for execution graph construction

This is fundamentally different from static schedulers: the batch composition depends on the evolving runtime state of memory, cache, and queues.

#### Graph Generator
Produces the initial operator sequence for the batch. This is model-specific: for dense models it generates the standard transformer layer sequence; for MoE models it generates the base attention + FFN structure with expert routing hooks.

#### Operation Mapper
Assigns each operator to a device within the MSG's device pool based on configured policies:
- **Standard case (homogeneous):** All operators go to the same device type with configured parallelism
- **Heterogeneous offloading:** Fine-grained, operator-level decisions -- e.g., attention operators to PIM, FFN to GPU (as in Fig. 3a)
- **MoE expert routing:** Per-token routing to experts via a configurable Expert Router that supports random selection, round-robin, proportional-load balancing, and custom policies
- For each assignment, the mapper attaches **latency and power estimates** from the operator profile

#### Operation Scheduler
Transforms the mapped operators into a complete execution DAG:
1. Encodes data dependencies and ordering constraints
2. Inserts **communication operations** for parallelism (AllReduce, AllGather, All2All for EP)
3. Inserts **data movement operations** for heterogeneous offloading (GPU-to-PIM transfers for activations/KV caches, return transfers)
4. Inserts **memory operations** for KV cache loads/stores across tiers
5. Produces the final execution graph passed to the System Simulator

### 1.4 System Simulator

The System Simulator is the runtime evaluation engine. It builds on modified versions of **ASTRA-sim** and **Chakra** -- originally designed for training, now extended for inference:

**Key extensions to ASTRA-sim/Chakra:**
- Support for heterogeneous compute fabrics (not just homogeneous GPU clusters)
- Operator-driven execution graphs (vs. repetitive training patterns)
- Inference-specific parallelism (expert parallelism, P/D disaggregation)
- Multi-tier memory hierarchy modeling
- PIM operations beyond simple load/store

**Execution graph evaluation:**
Each node in the execution graph is an operator annotated with:
- Device type assignment
- Latency estimate (from profile)
- Memory footprint
- Communication statistics
- Power information

Edges encode data dependencies and parallelism. The simulator evaluates:
- **Synchronization overhead** from parallelism
- **Network contention** and inter-device communication latency based on cluster topology
- **Memory access latency** based on tier (device HBM, host DRAM, CXL, storage)
- **Bandwidth contention** across the memory hierarchy

**Cluster-scale timing:**
The simulator models the full cluster: multiple MSGs executing concurrently, sharing network/memory resources. This captures contention effects that single-instance simulators miss.

**Memory tracking:**
Refined memory model with explicit multi-tier hierarchy:
- Device memory (HBM): fastest, limited capacity
- Host memory (DRAM): shared second tier
- CXL-attached memory: third tier, higher latency
- Storage: last resort

KV cache blocks are placed, migrated, and evicted across these tiers. The memory model tracks:
- Per-device memory occupancy over time
- Cache residency per tier
- Load/store overhead injected into execution graphs
- Bandwidth contention from concurrent memory accesses

**Power tracking:**
Instantaneous power per MSG + total system energy. Details in Section 1.5.

### 1.5 Request Router

The Request Router dispatches incoming requests to MSGs:

- **Standard routing:** Maps requests to the appropriate MSG for the requested model
- **P/D disaggregation routing:** Dispatches to a prefill MSG + associates a decode MSG. The prefill MSG's operation scheduler automatically inserts layer-wise KV cache transfer operations to the decode MSG
- **KV transfer logic:** Transfer operations are handled by the memory model and evaluated by the System Simulator, which computes inter-node communication latency based on network topology and bandwidth
- **Multi-instance routing:** When multiple MSGs serve the same model, the router can distribute load across them

### 1.6 Runtime Loop

The simulation proceeds iteratively:
1. Request Router dispatches incoming requests (based on arrival pattern) to MSGs
2. Each MSG's batch scheduler forms a batch from its queue
3. Graph generator + operation mapper + operation scheduler produce an execution graph
4. System Simulator evaluates the graph, computing timing and resource usage
5. Loop repeats until all requests complete
6. Online statistics (throughput over time, memory usage, power) are collected throughout
7. Final aggregated metrics are reported

This runtime-driven loop is the paper's central innovation: serving decisions and hardware behavior are interleaved in time, not computed separately.

---

## 2. Key Technical Contributions

### 2.1 Runtime-Driven Simulation Loop

**What makes it better than static models:**

Static models (like GenZ/roofline analysis) compute performance for a single operator or a single pass independently. They cannot capture:

1. **Temporal effects:** Queue buildup/drainage, batch size fluctuation over time, phase transitions between bursty and idle periods
2. **Contention amplification:** As queues grow, batch sizes increase, memory pressure rises, eviction costs grow, which further delays requests, creating a feedback loop
3. **Phase-dependent behavior:** Prefix cache hit rate improves as requests accumulate (more shared prefixes resident), but memory pressure from growing KV caches counteracts this
4. **Dynamic batch composition:** Real schedulers see different batch sizes each iteration depending on arrival rate, ongoing decodes, and memory state. Static models assume fixed batch sizes.
5. **P/D interaction dynamics:** In disaggregated setups, KV transfer timing depends on when prefill finishes and network contention at that moment -- not a fixed transfer cost
6. **MoE load imbalance:** Expert routing creates variable latency per token depending on which experts are hot, whether experts need loading from host memory, and how tokens cluster on experts

The runtime loop captures all these by making each scheduling decision depend on the current system state, which itself evolves from prior decisions. This is fundamentally a discrete-event simulation of the serving system, not an analytical model.

**Practical impact:** The throughput-over-time graphs (Fig. 5) show LLMServingSim 2.0 tracking real system dynamics -- ramp-up, steady state, drain-down -- with temporal fidelity impossible for static models.

### 2.2 Operator-Level Profiler

The profiler is a key enabler for extensibility:

- **Implementation:** Built on PyTorch/HuggingFace profiling API
- **What it measures:** Per-operator latency and power, using a single device and a single decode block
- **Profiling cost:** ~2.1 hours per model-device pair (e.g., Llama 3.1-70B on H100)
- **Reusability:** Profiles are collected once, reused across all experiments with that model-device combo
- **External profiles:** Can ingest profiles from external hardware simulators, enabling evaluation of future/hypothetical hardware without physical devices

This design choice is pragmatic: instead of building an analytical model for every operator on every hardware (which requires deep hardware knowledge), they measure empirically once and replay. The tradeoff is that you need access to the hardware (or a simulator of it) for profiling, but not for the serving simulation itself.

**Key advantage over analytical approaches:** Automatically captures hardware-specific optimizations (kernel fusion, memory layout, compiler effects) that analytical models must manually account for.

### 2.3 Memory Model

The memory model is arguably the most sophisticated component:

**Multi-tier hierarchy:**
- **Device memory (HBM):** Primary tier. KV cache blocks allocated here first. Fastest access.
- **Host memory (CPU DRAM):** Shared across devices on the same node. Used for overflow and as a shared prefix cache tier.
- **CXL-attached memory:** Emerging third tier. Higher capacity than HBM, lower latency than storage. Shared globally across MSGs when configured.
- **Storage (NVMe/SSD):** Last tier for cold KV cache blocks.

**KV cache lifecycle:**
1. **Allocation:** New KV blocks allocated on device memory during batch scheduling
2. **Eviction:** When device memory is full, blocks are evicted to lower tiers based on configured policy
3. **Migration:** Blocks can be promoted back to device memory when needed (prefix cache hit on a lower tier)
4. **Prefix caching:** Radix-tree-based prefix caches instantiated at each configured tier. During operation mapping, prefix hits reduce effective execution latency. Misses trigger KV transfer operations from the appropriate tier.

**Eviction and promotion decisions** are derived from per-device memory capacity and cache residency. The corresponding load/store overheads are injected as explicit operations in the execution graph, meaning their latency and bandwidth impact are modeled by the System Simulator.

**Multi-MSG sharing:**
- Device-only prefix caching: Each MSG maintains its own cache
- CPU-based shared cache: MSGs on the same node share a common CPU-resident prefix cache (Fig. 4b)
- CXL-based shared cache: All MSGs access a single globally shared prefix cache

This captures real-world deployment patterns (e.g., LMCache [39] for centralized CPU-based KV cache sharing).

### 2.4 Power Model

**Seven-component breakdown:**
1. **Accelerators (GPUs/TPUs/PIM):** Three-state model -- idle, active, standby. Utilization-dependent.
2. **CPUs:** Constant power when active
3. **DRAM:** Energy proportional to data volume transferred
4. **Interconnect links (including switches):** Energy proportional to data volume
5. **NICs:** Constant power
6. **Storage devices:** Constant power
7. **Other (motherboard, cooling, etc.):** Constant power

**Accelerator three-state model:**
- **Idle:** Base power draw when no work is scheduled
- **Active:** Full power during operator execution
- **Standby:** Intermediate power between batches or during communication waits

This captures the temporal power dynamics visible in Fig. 6: three distinct request pulses create alternating active/standby/idle phases, and LLMServingSim 2.0 tracks the transitions accurately.

**Validation:** 1.34% average error in total energy consumption compared to real RTX A6000 system (Fig. 6c). The per-component energy breakdown matches real measurements, with accelerators dominating, followed by CPUs and memory.

### 2.5 Batch Scheduler Intelligence

The batch scheduler integrates several models:

1. **Memory capacity check:** Ensures candidate batch fits in available device memory considering current occupancy
2. **KV footprint estimation:** Computes memory needed for new KV cache blocks based on sequence lengths in the candidate batch
3. **Prefix-cache residency lookup:** Checks which KV blocks are already cached across tiers, reducing both memory allocation needs and effective compute (prefix hits skip recomputation)
4. **Eviction cost evaluation:** If eviction is needed, accounts for the cost of moving blocks to lower tiers
5. **Maximum batch size enforcement:** Respects configured limits

This is a runtime-aware scheduler, not a static policy. Its decisions depend on the current state of memory, cache, and queues, which evolve with each simulation step.

---

## 3. Serving Techniques Modeled

### 3.1 Heterogeneous Devices + Operator-Granular Offloading

**Mechanism:** A single MSG can contain a mixed device pool (e.g., 1x H100 + 1x PIM). The Execution Planner installs offloading rules specifying which operators run on which devices at operator granularity.

**GPU+PIM example (Fig. 3a):**
- FFN and embedding operators run on GPU
- Attention operators offloaded to PIM (memory-centric, benefits from in-memory compute for attention's memory-bound nature)
- Operation scheduler automatically inserts data movement: GPU-to-PIM for activations and KV caches before attention, PIM-to-GPU for results after

**Significance:** This is not just "run the whole model on device X vs Y" -- it's per-layer, per-operator placement, which is critical for emerging heterogeneous hardware where different operators have different compute/memory characteristics.

### 3.2 MoE Expert Routing, Parallelism, and Offloading

**Full MoE modeling pipeline:**
1. **Expert placement:** Experts distributed across devices via expert parallelism (EP). Configuration specifies which experts reside on which devices.
2. **Expert Router:** Per-token routing decisions at each MoE layer. Supports:
   - Random selection
   - Round-robin
   - Proportional-load balancing
   - Custom user-defined policies
3. **Expert offloading:** Inactive experts can be evicted to host memory. When a token routes to an evicted expert, the operation scheduler inserts an expert load operation.
4. **Communication:** All-to-all communication for expert parallelism is modeled explicitly, accounting for token routing across devices.

**Key modeling detail:** The interaction between expert routing and memory is dynamic -- which experts are hot changes with the input distribution, affecting both compute latency (expert load from host) and communication (all-to-all volume depends on routing distribution).

### 3.3 Prefill-Decode Disaggregation

**Configuration:** The Execution Planner creates M prefill MSGs + N decode MSGs for the same model, with arbitrary M:N ratios.

**Runtime flow (Fig. 4a):**
1. Request Router dispatches request to a prefill MSG, along with the associated decode MSG identity
2. Prefill MSG processes the input sequence
3. Prefill MSG's operation scheduler automatically inserts **layer-wise KV cache transfer operations** to the decode MSG
4. KV transfers are handled by the memory model and evaluated by the System Simulator, computing inter-node communication latency based on network topology and bandwidth
5. Decode MSG receives KV caches and generates tokens autoregressively

**Heterogeneous P/D:** Prefill and decode MSGs can have different hardware types, memory hierarchies, and interconnects. This enables exploring heterogeneous P/D configurations (e.g., high-compute GPUs for prefill, high-memory-bandwidth devices for decode).

### 3.4 Prefix Caching Across Tiers

**Three-tier prefix caching:**
1. **Device-only:** Each MSG maintains its own on-device radix-tree prefix cache (like vLLM native prefix caching with block size 16)
2. **Device + shared CPU:** MSGs retain device-level caches + share a common CPU-resident prefix cache on the same node (like LMCache [39] with block size 256)
3. **Device + CPU + CXL:** All MSGs access a globally shared CXL-attached prefix cache

**Runtime behavior:**
- Prefix hits at the device tier: zero transfer cost, reduced compute
- Prefix hits at CPU tier: KV transfer from CPU to device injected into execution graph
- Prefix hits at CXL tier: KV transfer from CXL pool, potentially higher latency but globally shared
- Prefix misses: full computation, new KV blocks allocated

**Validation (Fig. 7):** 0.93% average error in memory usage and prefix cache hit rate for single-instance setup. Multi-instance setup with CPU-based shared cache shows 0.41% error, capturing synchronized increases in hit rate from cross-instance reuse.

### 3.5 Multi-MSG Sharing

Multiple MSGs serving the same model can share prefix cache state:
- CPU-resident shared cache visible to all MSGs on the same node
- CXL-resident globally shared cache visible to all MSGs across nodes
- This captures the benefit of serving-system-level KV cache sharing, not just per-instance caching

---

## 4. Validation Results

### 4.1 Performance Metrics (0.97% Average Error)

**Test setup:**
- Three platforms: RTX A6000 (4x), H100 (8x), TPU-v6e-1
- Models: Llama 3.1-8B, Phi-mini MoE (A6000/TPU), Llama 3.1-70B, Mixtral 8x7B (H100 with TP4)
- Workloads: 300 requests from ShareGPT, Poisson arrival at 10 req/s
- Serving framework: vLLM

**Configurations tested:**
- Multi-instance dense serving
- Single instance with prefix caching
- Prefill-decode disaggregation
- Single MoE model serving

**Results (Fig. 5):**
- Time-series throughput tracking: 5.66% error on A6000, 2.98% on H100 (per-timestep)
- Aggregated metrics (throughput + latency): 0.85% error (A6000), 1.59% error (H100)
- Captures temporal patterns: ramp-up, peak, drain-down phases

### 4.2 Power Consumption (1.34% Average Error)

**Test setup:** RTX A6000, TP1 and TP2 configurations, pulsed workload (3 bursts with idle gaps)

**Results (Fig. 6):**
- Three-state power transitions (idle/active/standby) closely tracked
- Total energy: 1.34% average error across TP1 and TP2
- Higher TP activates more GPUs -> higher peak power, narrower pulses (faster execution)
- Seven-component energy breakdown matches real measurements

### 4.3 Memory Usage + Prefix Cache Hit Rate (0.93% Average Error)

**Results (Fig. 7):**
- Single instance: Step-wise memory growth + increasing prefix hit rate accurately reproduced (0.93% error)
- Multi-instance with shared CPU prefix cache: Per-instance memory + shared CPU cache behavior captured (0.41% error)
- Captures synchronized hit-rate increases from cross-instance prefix reuse

### 4.4 Comparison with Other Simulators (Fig. 8)

| Simulator | Single Dense Error | Multi Dense Error | P/D Disagg Error | MoE Error | Avg Error |
|---|---|---|---|---|---|
| LLMServingSim v1 [25] | 5.2% | N/A | N/A | N/A | N/A |
| Vidur [26] | 7.2% | 6.7% | 4.5% | N/A | N/A |
| APEX [27] | N/A | 3.3% | 2.4% | N/A | N/A |
| TokenSim [28] | N/A | N/A | N/A | N/A | N/A |
| **LLMServingSim 2.0** | **2.35%** | **Low** | **1.42%** | **Low** | **~1-2%** |

**Key findings:**
- Prior simulators tend to be accurate for specific metrics but show larger deviations on others
- Under complex configurations (P/D disaggregation, MoE), prior simulators either fail to execute (marked unavailable) or show large errors from simplified abstractions
- LLMServingSim 2.0 maintains consistent accuracy across all configurations

**Simulation time (Fig. 8b):**
- LLMServingSim 2.0: 302-633s depending on configuration (minutes, not hours)
- Higher than lightweight simulators (Vidur 2.5-8.4s, TokenSim 2.8-7.2s)
- Much lower than LLMServingSim v1 (182,164s = ~50 hours for single dense)
- Practical tradeoff: minutes of simulation vs. hours of real deployment

### 4.5 TPU Case Study

- Platform: TPU-v6e-1 on Google Cloud running vLLM-TPU
- Model: Llama 3.1-8B, TP1
- Per-timestep throughput error: 4.24%
- **Aggregated error: 0.2%** (geomean across TPS, TPOT, ITL)
- Demonstrates extensibility to non-GPU accelerators through profile-based modeling

### 4.6 PIM Case Study

- Setup: RTX A6000 GPU vs GPU+PIM (256-channel PIM, 1GB HBM2 per channel at 2000 MT/s)
- Model: Llama 3.1-8B, 256 requests, input/output 128/512 tokens
- **Decode throughput: GPU+PIM achieves 1.43x higher throughput** than GPU-only after prefill completes
- GPU+PIM with sub-batch interleaving (SBI): comparable to GPU-only (SBI only helps with very large batches >= 256)
- **Energy: GPU+PIM reduces watts/token by 32.3%** while achieving comparable or better performance
- The energy breakdown shows modest increase in memory energy with PIM, but overall reduction from faster completion

---

## 5. Limitations and Gaps

### 5.1 What LLMServingSim 2.0 Does NOT Model

1. **Speculative decoding:** Not mentioned anywhere in the paper. This is a major serving optimization (draft model generates candidate tokens, target model verifies in parallel) that changes the execution pattern fundamentally.

2. **Quantization effects on accuracy:** The simulator models quantized operators via profiles but does not model accuracy degradation from quantization -- it's purely a performance/power simulator.

3. **Continuous batching internals:** While batching is modeled dynamically, the specific policies of iteration-level scheduling (as in Orca/vLLM's continuous batching) may be simplified compared to real implementations.

4. **Request scheduling policies beyond basic strategies:** Advanced scheduling like priority queues, fairness constraints, SLO-aware scheduling are not explicitly described.

5. **Model architecture innovations:** Techniques like sliding window attention (Mistral), grouped-query attention variations, multi-query attention are captured implicitly through profiling but not as explicit architectural features.

6. **Compiler optimizations and kernel fusion:** These are captured by profiling the compiled model, but the simulator cannot predict the effect of future compiler optimizations on a new hardware target.

7. **Network contention from non-LLM traffic:** The simulator models inference traffic only; shared datacenter network effects are not captured.

8. **Thermal throttling:** No thermal modeling beyond the power model. In real systems, sustained high power can trigger throttling.

9. **Reliability/fault tolerance:** No modeling of device failures, request retries, or redundancy.

10. **Multi-model serving on shared infrastructure:** Each MSG serves one model. Cross-model resource contention beyond memory sharing is not modeled.

### 5.2 Where Simulation Fidelity Breaks Down

1. **Time-series granularity:** Per-timestep errors are higher (5.66% on A6000) than aggregated errors (0.85%). This is expected -- the simulator approximates the exact timing of individual scheduling decisions but converges when aggregated.

2. **MoE with dynamic expert popularity:** The expert routing model uses configured policies, but real expert popularity distributions can shift with input data in ways that are hard to predict without running the actual model.

3. **Profiling representativeness:** The profile captures operator latency for a single decode block. If operator behavior varies significantly with batch size, sequence length, or memory pressure in ways not captured by the single-point profile, accuracy may degrade.

4. **CXL and PIM:** These are modeled through specifications and simulated profiles, not validated against real CXL/PIM hardware at scale. The PIM case study uses NeuPIMs simulation [45], not a physical PIM deployment.

5. **vLLM-specific behaviors:** The validation is against vLLM. Other serving frameworks (TensorRT-LLM, TGI, SGLang) may have different scheduling, batching, and memory management behaviors that would require re-profiling and potentially different policy modeling.

### 5.3 Missing Hardware/Software Features

- **RDMA / GPUDirect:** Not explicitly modeled as distinct from general interconnect
- **NVSwitch vs PCIe heterogeneity within a node:** Modeled as uniform interconnect bandwidth
- **Disaggregated memory pools (beyond CXL):** Emerging architectures like CXL memory pools with switches are not explicitly modeled
- **Dynamic voltage/frequency scaling:** No DVFS modeling
- **Multi-tenant GPU sharing (MIG, MPS):** Not modeled
- **Flash attention / memory-efficient attention kernels:** Captured via profiling but not as a configurable feature
- **Chunked prefill:** Not explicitly mentioned as a modeling target (though BudSimulator's GenZ framework does model this)

---

## 6. Cross-Pollination Analysis

### 6.1 For vllm-tuner

#### Simulation as Surrogate for Empirical Benchmarking

vllm-tuner currently relies on empirical benchmarking to evaluate vLLM configuration parameters. LLMServingSim 2.0 could serve as a **fast surrogate model** in the optimization loop:

**Current vllm-tuner flow:**
```
Optuna proposes config -> Deploy vLLM with config -> Run benchmark -> Measure metrics -> Feed back to Optuna
```

**Potential simulation-augmented flow:**
```
Optuna proposes config -> Simulate with LLMServingSim 2.0 -> Get estimated metrics -> Feed to Optuna
  |
  (Only run real benchmarks for top-K promising configs)
```

**Specific benefits:**
1. **Parameter space pruning:** Use simulation to quickly eliminate clearly suboptimal configurations before expensive real benchmarking. E.g., if simulation shows a TP=8 config is 2x slower than TP=4 for a given model/hardware, skip it entirely.
2. **Warm-starting Optuna:** Run simulation for a grid of configurations to build an initial surrogate model, then warm-start Optuna's Bayesian optimization with these points. This reduces the number of expensive real trials needed.
3. **Batch size exploration:** LLMServingSim 2.0's dynamic batching model can predict the effect of `max-num-seqs`, `max-num-batched-tokens` parameters much faster than real experiments.
4. **Parallelism strategy exploration:** TP/PP/EP combinations are expensive to test empirically (require different GPU allocations). Simulation can quickly evaluate dozens of strategies.

#### vllm-tuner Parameters Explorable via Simulation

| Parameter | Simulation Feasibility | Notes |
|---|---|---|
| `tensor-parallel-size` | High | Directly modeled in MSG parallelism |
| `pipeline-parallel-size` | High | Directly modeled |
| `max-num-seqs` | High | Affects batch scheduler behavior |
| `max-num-batched-tokens` | High | Affects batch size formation |
| `gpu-memory-utilization` | High | Affects KV cache budget |
| `block-size` | Medium | Affects prefix cache granularity |
| `enable-prefix-caching` | High | Explicitly modeled |
| `enable-chunked-prefill` | Low | Not explicitly modeled in the paper |
| `swap-space` | Medium | Relates to host memory tier |
| `enforce-eager` | Low | Compilation effect, not modeled |
| `quantization` | Medium | Captured via different profiles |
| `kv-cache-dtype` | Medium | Affects memory calculations |

#### Specific Improvements to vllm-tuner

1. **Two-phase optimization:** Phase 1 uses simulation to identify promising regions of the parameter space (fast, minutes per config). Phase 2 runs real benchmarks only on the top candidates (slow, minutes per config on real hardware).
2. **Simulation-informed priors:** Use simulation results to set informed priors for Optuna's TPE sampler, concentrating search in regions likely to perform well.
3. **Transfer learning across hardware:** Profile model on hardware A, use simulation to predict performance on hardware B. Useful when vllm-tuner users want to evaluate new hardware before purchasing.

### 6.2 For BudSimulator

#### Comparison: LLMServingSim 2.0 vs GenZ Framework

| Aspect | GenZ (BudSimulator) | LLMServingSim 2.0 |
|---|---|---|
| **Approach** | Analytical roofline model | Runtime-driven simulation with profiled operators |
| **Operator modeling** | Mathematical: compute = ops/FLOPS, memory = bytes/BW | Empirical: profiled latency per operator per device |
| **Temporal dynamics** | None -- single-pass analysis | Full dynamic: queues, batching evolution, memory state |
| **Power model** | None | 7-component, 3-state accelerator model |
| **Memory model** | Static: compute total weight + KV cache memory | Dynamic: multi-tier KV placement, eviction, migration, prefix caching |
| **P/D disaggregation** | Not modeled | First-class citizen via M:N MSG mapping |
| **Prefix caching** | Not modeled | Multi-tier radix-tree based with hit-rate tracking |
| **MoE** | Basic parallelism | Full expert routing, offloading, dynamic load |
| **Heterogeneous HW** | Single device type per analysis | Mixed device pools with operator-granular offloading |
| **Parallelism** | TP, PP, DP, EP with communication modeling | Same + runtime contention and dynamic scheduling effects |
| **Accuracy** | Roofline-bounded (optimistic or pessimistic) | 0.97% average error vs real systems |
| **Speed** | Milliseconds | Minutes |
| **Hardware required** | None (analytical) | One device for profiling (or external simulator) |
| **Extensibility** | Need to add analytical formulas | Just profile the new hardware |

#### Features BudSimulator Should Adopt

**Priority 1 -- High Impact, Feasible:**

1. **Power model:** BudSimulator has no power/energy modeling. The 7-component breakdown from LLMServingSim 2.0 is well-defined and could be added to BudSimulator's analysis. Even a simplified version (accelerator + DRAM + other) would be valuable for TCO analysis.

2. **P/D disaggregation modeling:** BudSimulator currently analyzes prefill and decode as separate phases on the same hardware. Adding support for M:N prefill:decode configurations with KV transfer costs would enable analysis of disaggregated architectures (DistServe, Splitwise style).

3. **Prefix caching effect on performance:** BudSimulator could add a prefix cache hit-rate parameter that reduces effective compute and memory access for cached prefixes. Even without runtime dynamics, a steady-state prefix hit-rate model would improve analysis accuracy.

4. **Multi-tier memory hierarchy:** BudSimulator's System class models on-chip and off-chip memory. Extending to device/host/CXL/storage tiers with associated bandwidth parameters would enable memory offloading analysis.

**Priority 2 -- Medium Impact:**

5. **MSG abstraction pattern:** BudSimulator's analysis is per-model, per-hardware. Adopting an MSG-like abstraction that bundles model + hardware + parallelism + serving policies into a single analyzable unit would make the API cleaner and support multi-instance analysis.

6. **Execution graph approach:** Instead of BudSimulator's current approach of summing operator latencies with roofline analysis, constructing an explicit execution DAG with dependencies would enable more accurate parallelism and communication overlap modeling.

7. **MoE expert offloading analysis:** BudSimulator's MoE support could be extended with expert placement and offloading cost modeling, including the latency of loading evicted experts from host memory.

**Priority 3 -- Nice to Have:**

8. **Profile-based operator modeling as an option:** BudSimulator could support both analytical (current GenZ) and profile-based (LLMServingSim 2.0 style) operator modeling. Profiles would provide higher accuracy for supported hardware, while analytical remains the default for unsupported hardware.

9. **Runtime dynamics (future):** A lightweight runtime simulation mode that models queue evolution and dynamic batching would be a major enhancement, but also a significant engineering effort. This could be a long-term goal.

#### Architecture Patterns Worth Borrowing

1. **Operator-level profiler integration:** LLMServingSim 2.0's profiler collects per-operator latency with minimal code changes. BudSimulator could offer a profiling mode that calibrates its analytical models against real hardware, using measured data to adjust roofline predictions.

2. **Three-input design (workload, cluster, profiles):** BudSimulator already separates model and hardware. Adding workload characterization (arrival patterns, sequence length distributions) as a first-class input would enable serving-level analysis.

3. **MSG as a composable unit:** The MSG pattern of {device pool + policies + model} enables systematic exploration of configurations. BudSimulator's API could adopt this: create an MSG-like object, assign hardware, set parallelism, run analysis.

4. **Memory model with explicit tier tracking:** BudSimulator's current memory calculation is "does the model fit?" LLMServingSim 2.0's approach of tracking memory occupancy across tiers, with explicit eviction/migration costs, is more informative.

---

## 7. Referenced Papers Worth Investigating

### 7.1 Papers Introducing Key Techniques

| Ref | Paper | Key Contribution | Relevance |
|---|---|---|---|
| [13] | **DistServe** (OSDI '24) | Disaggregating prefill and decode for goodput-optimized serving | High -- P/D disaggregation is a first-class feature |
| [14] | **Splitwise** (ISCA '24) | Phase splitting for efficient LLM inference | High -- complementary to DistServe for P/D |
| [15] | **DeepSpeed-MoE** (2022) | Advancing MoE inference and training | Medium -- MoE foundations |
| [16] | **SiDA** (MLSys '24) | Sparsity-inspired data-aware MoE serving | Medium -- expert routing optimization |
| [17] | **MoE-Lightning** (ASPLOS '25) | High-throughput MoE on memory-constrained GPUs | Medium -- MoE offloading |
| [18] | **CachedAttention** (USENIX ATC '24) | Cost-efficient multi-turn conversations with cached attention | High -- prefix caching foundations |
| [19] | **Prompt Cache** (MLSys '24) | Modular attention reuse for low-latency inference | Medium -- prefix caching |
| [20] | **CacheBlend** (EuroSys '25) | Fast LLM serving for RAG with cached knowledge fusion | Medium -- prefix caching for RAG |
| [25] | **LLMServingSim v1** (IISWC '24) | HW/SW co-simulation for LLM inference | High -- predecessor, baseline |
| [26] | **Vidur** (MLSys '24) | Large-scale LLM serving simulation | High -- primary competitor, good baseline comparison |
| [27] | **APEX** (arXiv '25) | Extensible dynamics-aware simulator for LLM serving | High -- primary competitor, automation focus |
| [28] | **TokenSim** (APPT '25) | Hardware and software exploration for LLM inference | Medium -- lightweight competitor |
| [35] | **Pre-gated MoE** (ISCA '24) | Algorithm-system co-design for fast MoE inference | Medium -- MoE optimization |
| [36] | **IMPRESS** (FAST '25) | Importance-informed multi-tier prefix KV storage | High -- multi-tier prefix caching is directly modeled |
| [37] | **KVCache in the Wild** (USENIX ATC '25) | Characterizing and optimizing KVCache at scale | High -- real-world KV cache behavior |
| [38] | **SGLang** (NeurIPS '24) | Efficient execution of structured generation programs | Medium -- alternative serving framework |
| [39] | **LMCache** (arXiv '25) | Efficient KV cache layer for enterprise-scale inference | High -- directly used in validation (CPU-based shared prefix cache) |
| [40] | **Mooncake** (FAST '25) | Trading memory for less computation in LLM serving | Medium -- KV cache architecture |
| [41] | **P/D-Serve** (arXiv '24) | Serving disaggregated LLM at scale | Medium -- P/D disaggregation |
| [45] | **NeuPIMs** (ASPLOS '24) | NPU-PIM heterogeneous acceleration for batched LLM | High -- PIM case study basis |
| [46] | **AttAcc** (ASPLOS '24) | PIM for batched transformer-based inference | Medium -- PIM attention offloading |
| [54] | **LLMCompass** (ISCA '24) | Accelerator and microarchitectural analysis for LLM workloads | Medium -- hardware-centric simulation |
| [55] | **ADOR** (ISPASS '25) | Design exploration for LLM serving | Medium -- hardware design space exploration |
| [56] | **ASTRA-sim 2.0** (ISPASS '23) | Modeling hierarchical networks and disaggregated systems | High -- LLMServingSim 2.0's backbone |
| [57] | **Chakra** (arXiv '23) | Performance benchmarking with standardized execution traces | High -- execution graph format used |
| [58] | **vLLM** (SOSP '23) | PagedAttention for efficient KV cache management | High -- serving framework used in all validation |
| [21] | **LIA** (ISCA '25) | Single-GPU LLM acceleration with AMX-enabled CPU-GPU offloading | Medium -- heterogeneous offloading |
| [22] | **InstAttention** (HPCA '25) | In-storage attention offloading for long-context LLM | Medium -- storage-tier attention |
| [23] | **OASIS** (CAL '25) | Outlier-aware KV cache clustering for CXL scaling | High -- CXL for KV cache |
| [24] | **AiDE** (CAL '25) | Disaggregated FFN execution on CXL-PNM | Medium -- CXL-based disaggregation |

### 7.2 Most Relevant for Improving vllm-tuner

1. **APEX [27]** -- Its "dynamics-aware" simulation and automated parallel execution search could inform vllm-tuner's parameter space exploration. APEX specifically targets automated configuration optimization.
2. **Vidur [26]** -- As a lightweight simulator (~seconds), it could serve as a fast surrogate in Optuna's loop, even if less accurate than LLMServingSim 2.0.
3. **SGLang [38]** -- If vllm-tuner extends beyond vLLM, SGLang's structured generation paradigm presents different optimization opportunities.
4. **DistServe [13] / Splitwise [14]** -- P/D disaggregation parameters are a natural extension to vllm-tuner's parameter space.

### 7.3 Most Relevant for Improving BudSimulator

1. **ASTRA-sim 2.0 [56] + Chakra [57]** -- BudSimulator already uses ASTRA-sim for collective communication. Upgrading to ASTRA-sim 2.0's hierarchical network model would improve communication modeling fidelity.
2. **LMCache [39]** -- Multi-tier KV cache sharing is a key feature BudSimulator should model.
3. **IMPRESS [36]** -- Multi-tier prefix caching with importance-based eviction is directly relevant to extending BudSimulator's memory model.
4. **NeuPIMs [45] / AttAcc [46]** -- PIM modeling would differentiate BudSimulator as a tool for emerging hardware exploration.
5. **LLMCompass [54]** -- Hardware-centric simulation could complement BudSimulator's analytical approach for deeper hardware exploration.
6. **Vidur [26]** -- As a comparative baseline, understanding Vidur's approach helps position BudSimulator's value proposition.
7. **OASIS [23] / AiDE [24]** -- CXL-based memory disaggregation is an emerging area BudSimulator should prepare for.

---

## 8. Summary of Key Takeaways

### For the Field
LLMServingSim 2.0 represents a significant advance in LLM serving simulation by unifying heterogeneous hardware, disaggregated serving, and runtime dynamics into a single framework. Its 0.97% accuracy with ~10-minute simulation times makes it practical for architectural exploration. The open-source release (GitHub) enables community adoption.

### For vllm-tuner
The most immediately actionable insight is using simulation (either LLMServingSim 2.0 or the lighter Vidur/APEX) as a fast surrogate model in the Bayesian optimization loop. This could reduce the number of expensive real benchmarks by 5-10x while maintaining optimization quality.

### For BudSimulator
BudSimulator's analytical GenZ approach is complementary to LLMServingSim 2.0's profile-based simulation. The key features to adopt are:
1. Power/energy modeling (7-component model)
2. P/D disaggregation analysis
3. Multi-tier memory hierarchy with prefix caching
4. MSG-like composable analysis units

These can be added incrementally to BudSimulator's existing GenZ framework without abandoning its analytical strengths (speed, no hardware needed for profiling).

### Critical Gap
Neither LLMServingSim 2.0 nor BudSimulator currently models **speculative decoding**, which is becoming a standard serving optimization. This is a high-priority gap for both projects.
