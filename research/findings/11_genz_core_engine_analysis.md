# GenZ Core Engine: Comprehensive Analysis

**Analyst:** Deep Code Analysis Agent
**Date:** 2026-02-28
**Scope:** Full analysis of `llm-memory-calculator/src/llm_memory_calculator/genz/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Engine Architecture](#2-core-engine-architecture)
3. [System Abstraction (system.py)](#3-system-abstraction)
4. [Operator Framework (operator_base.py + operators.py)](#4-operator-framework)
5. [Parallelism Management (parallelism.py)](#5-parallelism-management)
6. [Collective Communication (collective_times.py)](#6-collective-communication)
7. [Model Definition System (Models/)](#7-model-definition-system)
8. [LLM Inference Modeling (LLM_inference/)](#8-llm-inference-modeling)
9. [Simulation Engine (simulation/)](#9-simulation-engine)
10. [Features System (features/)](#10-features-system)
11. [Power Modeling (power.py)](#11-power-modeling)
12. [Unit System (unit.py)](#12-unit-system)
13. [Gap Analysis: GenZ vs LLMServingSim 2.0](#13-gap-analysis-vs-llmservingsim)
14. [Gap Analysis: GenZ vs SCOOT](#14-gap-analysis-vs-scoot)
15. [Extension Points and Recommendations](#15-extension-points)

---

## 1. Executive Summary

GenZ is an analytical performance modeling engine that uses **roofline analysis** to estimate LLM inference latency, throughput, and memory requirements. It operates as a **static, single-pass analytical model** -- given a model architecture and hardware configuration, it computes per-operator performance metrics and aggregates them into system-level results.

### Core Strengths
- Comprehensive operator-level roofline model with tensor core efficiency
- Extensive precision/quantization support (int2 through fp32, mixed precision, QLoRA)
- Flexible parallelism (TP, PP, DP, EP, SP, CP, ZeRO)
- Support for diverse model architectures: MHA, GQA, MQA, MLA (DeepSeek), Mamba/SSM, MoE, heterogeneous layers (DeciLM/Nemotron)
- ASTRA-SIM integration for realistic collective timing
- Speculative decoding, chunked prefill, LoRA/QLoRA modeling
- HuggingFace dynamic model loading
- Training simulation with RLHF stages

### Core Limitations
- **Static analytical model** -- no temporal simulation, no request scheduling, no queue dynamics
- **No batch scheduling** -- no continuous batching, no dynamic request management
- **No KV cache management** -- no eviction, no multi-tier caching, no prefix caching
- **No power model sophistication** -- single utilization-proportional formula vs multi-component temporal model
- **No contention modeling** -- memory bandwidth contention, network contention between operations are not modeled
- **No heterogeneous device support** -- single accelerator type per simulation

---

## 2. Core Engine Architecture

### Data Flow

```
ModelConfig (from MODEL_DICT or HuggingFace)
    │
    ├── get_configs(name) → ModelConfig
    │
    ├── create_full_prefill_model() / create_full_decode_model()
    │   │
    │   ├── attention.py: mha_flash_attention_prefill/decode/chunked
    │   ├── ffn.py: ffn_prefill/decode, deepseek_ffn_prefill
    │   ├── embedding.py: input_embedding, output_embedding
    │   └── mamba.py: mamba_prefill/decode
    │   │
    │   └── save_layers() → CSV file on disk
    │
    ├── get_model_df(csv_path, system, unit, batch_size)
    │   │
    │   ├── Reads CSV, inserts batch_size column
    │   ├── Instantiates operators (FC, GEMM, Logit, Attend, Sync, etc.)
    │   └── analysis_model() → per-operator roofline → DataFrame
    │
    └── get_summary_table(df) → aggregated metrics
        │
        └── prefill_moddeling() / decode_moddeling() → ModdelingOutput
            ├── Latency (ms)
            ├── Throughput (tokens/s)
            ├── RuntimeBreakdown (MHA, FFN, Embedding, Collective)
            └── is_offload flag
```

### Key Design Pattern
Models are defined as **lists of layer descriptors** (arrays), serialized to CSV, then read back and analyzed. Each layer descriptor is:
`[Name, dim1, dim2, ..., dimN, ResidencyInfo, OpType]`

This CSV-based pipeline is the central data flow mechanism. It enables decoupling model definition from analysis but introduces I/O overhead and limits composability.

---

## 3. System Abstraction

**File:** `system.py`
**Key Class:** `System`

### Purpose
Represents a hardware accelerator with compute, memory, and interconnect capabilities.

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `flops` | 123 TOPS | Peak compute (TeraOps) |
| `onchip_mem_bw` | 18000 GB/s | On-chip (SRAM) bandwidth |
| `offchip_mem_bw` | 900 GB/s | Off-chip (HBM) bandwidth |
| `off_chip_mem_size` | Inf MB | HBM capacity |
| `interchip_link_bw` | 25 GB/s | Inter-chip link bandwidth (NVLink/IB) |
| `interchip_link_latency` | 1.9 us | Inter-chip link latency |
| `bits` | 'bf16' | Precision format |
| `compute_efficiency` | 1.0 | Achieved/peak compute ratio |
| `memory_efficiency` | 1.0 | Achieved/peak memory BW ratio |
| `compute_engine` | 'GenZ' | 'GenZ' or 'Scale-sim' |
| `collective_strategy` | 'GenZ' | 'GenZ' or 'ASTRA-SIM' |
| `topology` | 'FullyConnected' | Network topology |

### Precision Support
36 precision formats supported across two dictionaries:
- `compute_multiplier`: Ratio of compute throughput relative to fp16 (e.g., int8=0.5x, fp32=2x)
- `mem_multiplier`: Bytes per element (e.g., int4=0.5B, bf16=2B, fp32=4B)

Includes specialized types: `nf4` (QLoRA), `fp8_e4m3`/`fp8_e5m2`, mixed precision variants (`mixed_bf16`, `amp_fp16`), and TF32.

### GEAR Quantization
Special KV cache quantization (GEAR) that combines:
- Base quantization type (`gear_b`)
- Sparsity ratio (`gear_s`)
- Low-rank decomposition (`gear_r`)
Formula: `mem_mult = mem[gear_b] + (gear_s/100)*mem[bits] + low_rank_factor`

### Extension Points
- `from_dict()` / `from_json()` class methods for config-driven creation
- `claim_onchip_mem()` / `release_onchip_mem()` for on-chip memory tracking
- `get_bit_multiplier()` with data-type awareness (distinguishes K, V, activations, weights)

### What's Missing
- No multi-tier memory hierarchy (HBM only, no CXL, no host DRAM tiers)
- No per-device power model (handled separately in power.py)
- No heterogeneous multi-device support within a single system
- No memory bandwidth contention model

---

## 4. Operator Framework

**Files:** `operator_base.py`, `operators.py`

### Base Class: `Operator`

The `Operator` base class defines the interface for all computational operations:

**Core Methods:**
| Method | Purpose |
|--------|---------|
| `get_tensors()` | Returns (input_a, input_w, output) shapes |
| `get_num_ops()` | Returns FLOPs for the operation |
| `get_sz_list()` | Returns [input_size, weight_size, output_size] |
| `get_memory_time(system)` | Computes memory access time based on tensor sizes and locations |
| `get_compute_time(system)` | Computes compute time: `ops * bit_mult / op_per_sec` |
| `get_communication_time(system)` | Returns 0 for compute ops; delegates to collective functions for Sync |
| `get_roofline(system, unit)` | Full roofline analysis returning detailed metrics dict |

### Roofline Analysis Algorithm

```
exec_time = max(compute_time, memory_time, comm_time)

compute_time = (num_ops * 2 * bit_multiplier) / (op_per_sec * compute_efficiency * tensor_core_efficiency)
memory_time = sum(tensor_size / bandwidth for each tensor) / memory_efficiency
comm_time = collective_time / comm_efficiency

boundedness = 'Compute' if compute_time > memory_time else 'Memory'
             (forced to 'Memory' for Norm, Special_Func)
             (set to 'Collective' if compute_time == 0)
```

### Phase 2 Enhancements

The operator framework includes sophisticated improvements:

1. **Tensor Core Efficiency Modeling** (`get_tensor_core_efficiency(m, n, k)`):
   - Calculates padding waste for non-tile-aligned dimensions (16x16x16 tiles)
   - Models wave quantization penalty for under-utilized SMs
   - Returns efficiency factor 0.4 to 0.95

2. **Shape-Dependent Operational Intensity** (`get_shape_dependent_op_intensity()`):
   - Computes realistic FLOPs/byte based on actual data reuse patterns
   - Different formulas for GEMM, Logit, Attend, Norm operations
   - Accounts for input reuse in different cache levels

3. **Operation Characteristics Table**:
   - Pre-defined typical intensity ranges for each operation type
   - Threshold dimensions for compute-vs-memory boundedness determination

### Concrete Operators

| Operator | Dimensions | Num Ops | Tensor Shapes |
|----------|-----------|---------|---------------|
| `FC` | B, O, I | B*O*I | in=(B,I), w=(O,I), out=(B,O) |
| `GEMM` | B, M, N, K | B*M*N*K | in=(B,K,N), w=(M,K), out=(B,M,N) |
| `Logit` | B, H, M, N, D, Hkv | B*H*M*N*D | Q=(B,H,M,D), K=(B,Hkv,N,D), out=(B,H,M,N) |
| `Attend` | B, H, M, N, D, Hkv | B*H*M*N*D | attn=(B,H,M,N), V=(B,Hkv,N,D), out=(B,H,M,D) |
| `CONV2D` | B, K, C, Y, X, R, S | B*K*C*Y*X*R*S | Standard convolution |
| `CONV1D` | B, OF, IF, N, C | B*OF*N*C | 1D convolution (Mamba) |
| `Einsum` | equation-based | Product of all dims | Arbitrary tensor contractions |
| `Sync` | B, M, N + collective_type | 0 (communication only) | Communication-only operator |
| `Repeat`/`EndRepeat` | repeat_count | 0 | Layer repetition markers |
| `LoraA`/`LoraB`/`LoraMerge` | Inherits GEMM | Same as GEMM | LoRA adapter operators |
| `Add` | M, N | M*N | Element-wise addition |
| `Norm`/`Avg`/`Special_Func` | VXU operations | Element-wise | Norm, activation functions |

### Memory Location Control

Operators support pinning tensors to on-chip or off-chip memory via `ResidencyInfo`:
- `All_offchip`: All tensors in HBM
- `AC_onchip`: Input activation and output on-chip (SRAM) -- used for decode
- `C_onchip`: Output on-chip (used for Logit output = softmax input)
- `A_onchip`: Input A on-chip (used for Attend input = softmax output)

This models the software-managed data movement typical of GPU SRAM usage.

### Extension Points
- Add new operator types by subclassing `Operator`
- Override `get_effective_num_ops()` for sparsity-aware compute
- Override `get_default_mem_loc()` for different residency defaults

---

## 5. Parallelism Management

**File:** `parallelism.py`
**Key Class:** `ParallelismConfig`

### Supported Parallelism Dimensions

| Dimension | Description | Effect on Computation |
|-----------|-------------|----------------------|
| `tensor_parallel` (TP) | Splits weights within a layer | Divides weight dims, adds AllReduce |
| `pipeline_parallel` (PP) | Splits layers across devices | Divides num_layers, adds MessagePass |
| `data_parallel` (DP) | Replicates model across devices | Multiplies batch |
| `expert_parallel` (EP) | Distributes MoE experts | Divides experts, adds All2All |
| `sequence_parallel` (SP) | Splits sequence dimension | Divides sequence length |
| `context_parallel` (CP) | Ring/Ulysses attention | Splits KV context |
| `zero_stage` | ZeRO 0/1/2/3 | Shards optimizer/gradient/weights |

### Communication Overhead Model

`get_communication_overhead()` uses physics-based models:
- **TP:** `0.03 * log2(tp)` + congestion for tp>4
- **PP:** Bubble fraction `(pp-1)/(pp + micro_batches - 1)` + scale factor for >2048 GPUs
- **DP:** `0.04 * log2(dp)` + ZeRO multipliers (1.25x for stage 2, 1.4x for stage 3)
- **EP:** `0.06 * log2(ep)` + superlinear congestion for ep>4
- **CP:** `0.03 * log2(cp)`

Overhead capped at 60%.

### Training-Specific Factory

`for_training()` class method auto-selects parallelism for training:
- Estimates memory from model size and training method (full/LoRA/QLoRA)
- Increases TP until model fits in memory
- Falls back to ZeRO-2/3 if TP insufficient
- Targets ~32 effective batch size via gradient accumulation

### How Parallelism Affects Model Definition

Parallelism is applied at model definition time (in `attention.py`, `ffn.py`):
- TP divides attention heads and FFN intermediate dimension
- PP divides decoder layers into stages with MessagePass between them
- EP divides MoE experts and inserts All2All communication
- SP divides sequence length for QKV projections

The parallelism hierarchy string `"TP{4}_EP{2}_PP{1}"` is passed to the System for collective routing.

---

## 6. Collective Communication

**File:** `collective_times.py` (~900 lines)

### Collective Operations

| Function | Algorithm | Return Unit |
|----------|-----------|-------------|
| `get_AR_time()` | Ring/Tree AllReduce, hierarchical | milliseconds |
| `get_AG_time()` | AllGather, hierarchical | milliseconds |
| `get_A2A_time()` | All-to-All with MoE extensions | milliseconds |
| `get_message_pass_time()` | Point-to-point | milliseconds |
| `get_reduce_scatter_time()` | ReduceScatter, hierarchical | milliseconds |

### AllReduce Model (Most Detailed)

Multi-level algorithm selection based on message size and node count:

1. **Protocol Selection:**
   - <64KB: Low-latency protocol (bw_eff=0.3, latency=20us)
   - 64KB-1MB: Transitional (bw_eff=0.6, latency=10us)
   - >1MB: Simple protocol (bw_eff=0.85, latency=5us)

2. **Algorithm Selection:**
   - Ring AllReduce for bandwidth-optimal large messages
   - Double binary tree for latency-optimal small messages at scale (>16 nodes)

3. **Hierarchical Decomposition** (multi-node):
   - Step 1: Intra-node ReduceScatter (NVLink, 90% efficiency)
   - Step 2: Inter-node AllReduce (InfiniBand, with ring or tree)
   - Step 3: Intra-node AllGather (NVLink, 90% efficiency)

4. **Scale-Aware Overhead** (Phase 14):
   - Base NCCL overhead by scale tier (8/64/256/1024/4096+ GPUs)
   - Network congestion factor (topology-dependent: fat_tree=0.10, torus=0.15)
   - Straggler overhead: `1 + epsilon * sqrt(N/1000)`
   - Message-size adjustment (small messages get 15% extra overhead)

### MoE All2All Enhancements

Two implementations:
1. **Basic `get_A2A_time()`:** Standard A2A with optional MoE overhead
2. **Advanced `get_moe_a2a_time_with_alpha()`:** Local activation rate (alpha) modeling
   - Based on Speculative MoE paper (arxiv 2503.04398)
   - Models non-local fraction: `1 - local_activation_rate`
   - Splits intra-node (NVLink) vs inter-node (IB) communication
   - Two-phase: dispatch + combine (2x)
   - Load imbalance factor: 1.15x
   - EP superlinear scaling: `1 + 0.05 * EP^0.7`

### Hardware-Specific Efficiency Tables

`COMM_EFFICIENCY_TABLES` provides calibrated efficiencies for:
- H100 NVLink 4.0
- A100 NVLink 3.0
- TPU v4/v5
- AMD MI300X
- GB200 (projected)

Efficiencies vary by message size bucket (<64KB, 64KB-1MB, 1MB-64MB, >64MB) and scope (intra-node vs inter-node).

---

## 7. Model Definition System

**Directory:** `Models/`

### ModelConfig Class (`default_models.py`)

Comprehensive configuration supporting:

| Feature | Parameters |
|---------|-----------|
| **Base Architecture** | vocab_size, hidden_size, intermediate_size, num_decoder_layers, num_attention_heads |
| **GQA/MQA** | num_key_value_heads, head_dim |
| **MoE** | num_experts, expert_top_k, moe_intermediate_size, n_shared_experts, shared_expert_intermediate_size, moe_layer_freq, first_k_dense_replace |
| **Mamba/SSM** | mamba_d_state, mamba_d_conv, mamba_expand, mamba_dt_rank |
| **MLA (DeepSeek V2/V3)** | kv_lora_rank, q_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim |
| **Sliding Window** | sliding_window, global_attn_every_n_layers |
| **Heterogeneous Layers** | layer_configs (list[LayerConfig]) -- per-layer overrides for DeciLM/Nemotron |
| **LoRA** | lora_config (LoraConfig) |
| **Quality** | model_quality (QualityMetricsCollection) |

### Layer Type System

Models define layer types via `layer_type` list:
- `["MHA-global", "Dense"]` -- standard transformer
- `["MHA-local", "Dense"]` -- sliding window attention
- `["Mamba", "Dense"]` -- SSM layer
- `["MHA-global", "MoE"]` -- MoE transformer
- `["no_op", "no_op"]` -- skipped layer (heterogeneous)
- `["linear", "linear"]` -- linear replacement (DeciLM)

### Pre-Configured Models

`MODEL_DICT` contains models from:
- **Meta:** LLaMA 2/3/3.1 (7B-405B)
- **Google:** Gemma 2/3 families
- **Microsoft:** Phi models
- **Mistral:** Mistral/Mixtral (including MoE)
- **Alibaba:** Qwen/QwQ models
- **NVIDIA:** Nemotron/DeciLM
- **Misc:** DeepSeek V2/V3, Jamba, Command-R, etc.

### Dynamic Model Loading (`dynamic_loader.py`)

- `import_model_from_hf(model_id)`: Fetches config.json from HuggingFace
- `huggingface_config_to_model_config()`: Converts HF config format to ModelConfig
- Handles different naming conventions across model families

### Attention Implementations (`attention.py`)

| Function | Description |
|----------|-------------|
| `mha_flash_attention_prefill()` | Standard MHA for prefill phase |
| `mha_flash_attention_decode()` | Standard MHA for decode (single token query) |
| `mha_flash_attention_prefill_local()` | Sliding window attention for prefill |
| `mha_flash_attention_decode_local()` | Sliding window attention for decode |
| `mha_flash_attention_chunked()` | Mixed prefill+decode in single batch |
| `mla_attention_prefill()` | DeepSeek V2/V3 Multi-head Latent Attention |
| `mla_attention_decode()` | MLA decode with compressed KV cache |
| `linear_attention_prefill/decode()` | Linear replacement (DeciLM-style) |

Key modeling patterns:
- **Prefill attention:** Full QKV projection + Logit (Q@K^T) + Attend (softmax*V) + Output projection + AllReduce
- **Decode attention:** Same projections but Q has seq_len=1; splits into "Pre" (over input context) and "Suf" (over generated tokens) logit/attend
- **MLA:** Additional Q down-projection, Q up-projection, KV compression, KV up-projection

### FFN Implementations (`ffn.py`)

| Function | Description |
|----------|-------------|
| `ffn_prefill()` | Standard dense or MoE FFN for prefill |
| `ffn_decode()` | Dense or MoE FFN for decode |
| `deepseek_ffn_prefill()` | DeepSeek-style with shared experts |
| `linear_ffn_prefill/decode()` | Linear replacement |

MoE FFN modeling includes:
- Router (Gate) projection
- Dispatch All2All for EP>1
- Per-expert up+gate and down projections
- Unused expert weight tracking (for memory calculation)
- Collect All2All for EP>1
- `calculate_activated_experts()` for decode-phase load estimation

---

## 8. LLM Inference Modeling

**Directory:** `LLM_inference/`

### Prefill Modeling (`llm_prefill.py`)

`prefill_moddeling()` function flow:

1. **System Creation:** `get_inference_system()` maps system_name to System object
2. **Memory Check:** Creates model, computes weights + KV cache size, checks against per-chip memory
3. **Offloading:** If memory insufficient and `model_offload=True`, computes effective bandwidth:
   `new_BW = total_mem / max(min(total, hbm)/BW_hbm, overflow/BW_offload)`
4. **Latency Computation:** Creates full model CSV, runs `get_model_df()` + `get_summary_table()`
5. **Output:** `ModdelingOutput(Latency, Throughput, RuntimeBreakdown, is_offload, model_df, summary_table)`

Throughput = `1000 * batch_size / prefill_latency` (requests/s)
Tokens/s = `throughput * input_tokens`

### Decode Modeling (`llm_decode.py`)

`decode_moddeling()` adds KV cache growth modeling:
- For `output_tokens > 1`: Two-point sampling (initial + final KV cache size)
- Weighted average: `0.8 * initial_latency + 0.2 * final_latency`
- For short outputs (<10): Simple growth factor `1 + 0.1 * output_tokens / 10`

TPOT = `decode_latency` (ms per token per request)
Throughput = `1000 * batch_size / decode_latency` (tokens/s)

### Chunked Prefill (`llm_chunked.py` -- imported via `create_full_chunked_model`)

Models mixed prefill+decode batches:
- Prefill KV sizes: list of `(processed_tokens, new_tokens)` tuples
- Decode KV sizes: list of past context lengths
- Creates a single "chunk" with both types of attention

### Speculative Decoding (`llm_spec_decode.py`)

Models draft+target speculative decoding:
- Prefill: Both draft and target models run prefill
- Decode: Draft model generates `num_parallel_tokens` tokens, target verifies in parallel
- Expected tokens: `sum(k * x^k * (1-x) for k in 1..N-1) + N * x^N`
- Total latency: `target_decode + N * draft_decode`

### Best Parallelization (`best_parallelization.py`)

`get_best_parallization_strategy()`:
- Enumerates valid (TP, PP) combinations based on head count and layer count
- Runs simulation for each combination
- Returns configuration with highest throughput

`get_pareto_optimal_performance()`:
- Varies both batch size and parallelism
- Uses paretoset for multi-objective optimization (min latency, max throughput)

### Platform Sizing (`platform_size.py`)

`get_minimum_system_size()`:
- Computes minimum number of chips needed for a given model
- Based on memory requirements: `model_weights + kv_cache <= chips * chip_memory`

### RuntimeBreakdown (`utils.py`)

Categorizes operator latencies into:
- **Embedding:** embeddings, classifier
- **MHA:** QKV, Out Proj, Logit, Attend (all variants)
- **FFN:** Gate, up+gate, down, shared variants
- **Collective:** MHA AR, FFN AR, Gate AR, Dispatch/Collect A2A, Message Pass
- **Mamba:** Inproj, Conv, BC proj, xt proj, deltaA/B/Bu, x/y calc, etc.

---

## 9. Simulation Engine

**Directory:** `simulation/`

### SimulationEngine (`engine.py`)

Provides a **unified interface** wrapping the existing prefill/decode/chunked modeling:

```python
engine = SimulationEngine()
config = SimulationConfig(
    model="llama3_8b",
    features=["decode", "tensor_parallel", "flash_attention"],
    simulation_params={"batch_size": 32, "input_tokens": 2048}
)
result = engine.simulate(config)
```

Flow:
1. Validate configuration and feature combination
2. Initialize features (pre/post hooks)
3. Apply pre-simulation features (model/hardware modifications)
4. Run main simulation (delegates to prefill/decode/chunked modeling)
5. Convert to unified SimulationResult
6. Apply post-simulation features
7. Return result

### SimulationConfig (`config.py`)

Configuration dataclass with:
- `SimulationType`: PREFILL, DECODE, CHUNKED, CONTINUOUS
- Feature validation: incompatibility checks, dependency checking
- Default parameter population per simulation type

### SimulationResult (`results.py`)

Unified result with:
- `latency`, `throughput`, `runtime_breakdown`
- `feature_metrics`: per-feature metrics dict
- `memory_usage`, `hardware_utilization`
- Serialization: `to_dict()`, `to_json()`, `from_dict()`, `from_json()`

### Current State
The simulation engine is a **facade layer** over existing functions. It does NOT add:
- Temporal simulation
- Request scheduling
- Multi-iteration dynamics
- Event-driven execution

---

## 10. Features System

**Directory:** `features/`

### FeatureRegistry (`registry.py`)

Auto-discovery system for simulation features:
- Scans `features/` package for `BaseFeature` subclasses
- Registers built-in pseudo-features (prefill, decode, chunked, lora, flash_attention, etc.)
- Validates feature compatibility (incompatible pairs, dependency checking)

### Built-in Features
- `prefill`, `decode`, `chunked` -- simulation type selectors
- `tensor_parallel`, `pipeline_parallel` -- parallelism
- `lora` -- LoRA adapter simulation
- `flash_attention` -- flash attention optimization
- `memory_offload` -- memory offloading
- `speculative_decode` -- speculative decoding
- `cpu_optimization` -- CPU-specific optimizations

### BaseFeature (`base.py`)

Abstract interface:
- `FeatureCategory`: MODEL, HARDWARE, OPTIMIZATION, INFERENCE, PARALLELISM
- `FeatureMetadata`: name, category, version, description, dependencies, incompatible_with
- Methods: `initialize()`, `apply()`, `cleanup()`

### Current State
The features system is **infrastructure without implementations**. The built-in features are markers that map to existing functionality. No custom feature implementations exist yet. The apply/pre/post hooks are wired but unused.

---

## 11. Power Modeling

**File:** `power.py`

### Current Implementation

Single function `get_energy()`:
```python
energy = sum over operators of:
    latency * (static_power + compute_power*compute_util + memory_power*memory_util + network_power*comm_util)
```

Default power breakdown: Static 30%, Compute 40%, Memory 20%, Network 10%.
Returns energy in kWh.

### Limitations vs LLMServingSim 2.0
- No multi-component model (7 components in LLMServingSim: accelerators, CPUs, DRAM, interconnect, NICs, storage, other)
- No three-state accelerator model (idle/active/standby)
- No temporal power tracking (instantaneous power over time)
- No per-device power modeling for heterogeneous setups
- 1.34% error in LLMServingSim vs unknown accuracy in GenZ

---

## 12. Unit System

**File:** `unit.py`
**Key Class:** `Unit`

Provides consistent unit conversion throughout the codebase:

| Type Code | Default Unit | Conversion |
|-----------|-------------|------------|
| C | Tflops | SI (1e12) |
| M | MB | Binary (2^20) |
| T | msec | SI (1e-3) |
| BW | GB/s | Binary (2^30) |
| F | MHz | SI (1e6) |
| E | pJ | SI (1e-12) |
| O | MFLOP | SI (1e6) |

Methods:
- `raw_to_unit(data, type)`: Convert raw (SI base) to display unit
- `unit_to_raw(data, type)`: Convert display unit to raw

---

## 13. Gap Analysis: GenZ vs LLMServingSim 2.0

### Critical Gaps

| Feature | GenZ | LLMServingSim 2.0 | Gap Severity |
|---------|------|-------------------|--------------|
| **Runtime-driven simulation** | Static single-pass | Discrete-event iterative loop | CRITICAL |
| **Batch scheduling** | Fixed batch size | Dynamic continuous batching with queue management | CRITICAL |
| **KV cache management** | Size calculation only | Multi-tier allocation, eviction, prefix caching, migration | CRITICAL |
| **Request routing** | N/A | Request Router with P/D disaggregation | HIGH |
| **Heterogeneous devices** | Single device type | Mixed device pools (GPU+PIM, GPU+NPU) | HIGH |
| **Operator-granular offloading** | N/A | Per-operator device assignment | HIGH |
| **Per-request metrics** | Aggregate only | TTFT, TPOT, queueing delay, end-to-end per request | HIGH |
| **Temporal dynamics** | N/A | Throughput/power/memory over time | HIGH |
| **Memory contention** | N/A | Bandwidth contention from concurrent accesses | MEDIUM |
| **Power model** | Simple utilization-proportional | 7-component, 3-state, temporal | MEDIUM |
| **Execution graph** | Sequential operators | DAG with data dependencies | MEDIUM |
| **Profiler integration** | Analytical only | Can ingest real profiler data | MEDIUM |

### What GenZ Has That LLMServingSim 2.0 Doesn't

| Feature | GenZ | LLMServingSim 2.0 |
|---------|------|-------------------|
| **Extensive model registry** | 50+ pre-configured models | Manual profiling needed |
| **HuggingFace auto-loading** | Dynamic from HF Hub | N/A |
| **MLA attention (DeepSeek)** | Full implementation | Not mentioned |
| **Mamba/SSM modeling** | Full implementation | Not mentioned |
| **Training simulation** | Full training pipeline with RLHF | Inference only |
| **Speculative decoding** | Draft+target modeling | No |
| **LoRA/QLoRA modeling** | Full integration | Not mentioned |
| **Best parallelization search** | Brute-force + Pareto | Manual configuration |
| **GEAR quantization** | KV cache quantization | Not mentioned |
| **Heterogeneous layer configs** | DeciLM/Nemotron support | Not mentioned |

### Architectural Differences

**GenZ's approach:** Compute everything analytically from model architecture + hardware specs. Fast (milliseconds), deterministic, no hardware access needed. Good for quick exploration and comparison.

**LLMServingSim's approach:** Build a discrete-event simulation of the serving system. Slower but captures temporal dynamics, contention, scheduling effects that analytical models miss. Requires profiler data for accuracy.

**Complementary, not competitive:** GenZ excels at quick "what-if" analysis and architectural exploration. LLMServingSim excels at system-level behavior prediction under realistic workloads. BudSimulator should offer both.

---

## 14. Gap Analysis: GenZ vs SCOOT

### SCOOT's Contributions Not Present in GenZ

| Feature | SCOOT | GenZ Status |
|---------|-------|-------------|
| **SLO-oriented tuning** | Multi-objective optimization of TTFT, TPOT, throughput, tail latency | Not applicable (no serving system) |
| **Bayesian optimization** | GP surrogate + acquisition function for config search | `get_best_parallization_strategy` uses brute force |
| **Hidden constraint learning** | Random forest for infeasible configurations | No infeasibility detection |
| **Request trace-driven evaluation** | Real workload traces | Fixed batch_size/input_tokens |
| **Multi-objective Pareto optimization** | EHVI for MOBO | Simple Pareto via paretoset library |
| **Parameter space exploration** | 9 parameters, 100B+ search space | TP/PP only (2 parameters) |

### How SCOOT Patterns Could Enhance GenZ

1. **Bayesian parallelism search:** Replace brute-force enumeration with GP-based search, especially when adding batch_size, bits, and other dimensions
2. **Request trace integration:** Add workload characterization (arrival rate, sequence length distribution) to inform batch sizing and throughput estimation
3. **SLO awareness:** Add latency constraints to `get_best_parallization_strategy()` (e.g., "maximize throughput subject to TTFT < 500ms")
4. **Parallel configuration evaluation:** Run multiple configurations concurrently for faster search

---

## 15. Extension Points and Recommendations

### Near-Term Extensions (Low Risk)

1. **Multi-tier memory model:** Extend `System` with `host_mem_size`, `host_mem_bw`, `cxl_mem_size`, `cxl_mem_bw`. Modify `get_offload_system()` to use tiered bandwidth calculation.

2. **Enhanced power model:** Add per-component power modeling with idle/active/standby states. Requires extending `System` with power parameters.

3. **Prefix caching estimation:** Add a `prefix_cache_hit_rate` parameter to prefill modeling. Multiply KV cache read by `(1 - hit_rate)` and reduce compute for matched prefixes.

4. **Continuous batching throughput:** Add a `continuous_batching_modeling()` that estimates steady-state throughput from prefill+decode latencies and arrival rate.

5. **GPU utilization curves:** Add MFU (Model FLOPs Utilization) calculation from roofline analysis results.

### Medium-Term Extensions (Moderate Risk)

6. **Execution DAG:** Replace sequential operator list with a directed acyclic graph. Enable overlap of communication and computation (pipeline and tensor parallelism overlap).

7. **Memory bandwidth contention:** Model concurrent memory accesses when multiple operations share HBM bandwidth.

8. **Request-level modeling:** Add request scheduling simulation on top of per-operator analysis. Model queue buildup and TTFT/TPOT distributions.

9. **P/D disaggregation:** Model prefill and decode on separate device pools with KV transfer overhead.

### Long-Term Extensions (High Risk)

10. **Discrete-event simulation layer:** Build a lightweight event-driven scheduler that uses GenZ's analytical models as operator-level cost functions. This would bridge the gap to LLMServingSim without requiring full profiler infrastructure.

11. **Heterogeneous device support:** Model mixed device pools with per-operator device assignment and data movement costs.

12. **Real profiler calibration:** Add ability to calibrate analytical models with real profiler data for improved accuracy.

### Code Quality Notes

1. **CSV-based pipeline:** The model definition -> CSV -> read-back pattern is functional but fragile. Consider in-memory operator lists for performance-critical paths.

2. **Typo in API:** `prefill_moddeling` / `decode_moddeling` (double 'd') -- preserve for backward compatibility but consider deprecation wrappers.

3. **Global state:** `unit = Unit()` at module level in multiple files. Consider dependency injection.

4. **Test coverage:** No tests visible in the genz directory itself. Tests are in `BudSimulator/tests/`.

---

## Appendix A: File Index

| File | Lines | Purpose |
|------|-------|---------|
| `system.py` | ~196 | Hardware abstraction |
| `operator_base.py` | ~537 | Base operator with roofline |
| `operators.py` | ~436 | Concrete operator implementations |
| `parallelism.py` | ~245 | Parallelism configuration |
| `collective_times.py` | ~900+ | Collective communication timing |
| `power.py` | ~38 | Basic power/energy estimation |
| `unit.py` | ~45 | Unit conversion |
| `analyse_model.py` | ~291 | Model analysis and summary |
| `Models/__init__.py` | ~39 | Model exports |
| `Models/default_models.py` | ~389 | ModelConfig + MODEL_DICT |
| `Models/get_language_model.py` | ~747 | Model creation (prefill/decode/chunked) |
| `Models/attention.py` | ~316 | Attention layer definitions |
| `Models/ffn.py` | ~271 | FFN layer definitions |
| `Models/embedding.py` | ~30 (est) | Embedding layer definitions |
| `Models/dynamic_loader.py` | ~100 (est) | HuggingFace dynamic loading |
| `LLM_inference/__init__.py` | ~20 | Inference exports |
| `LLM_inference/llm_prefill.py` | ~128 | Prefill modeling |
| `LLM_inference/llm_decode.py` | ~194 | Decode modeling |
| `LLM_inference/llm_spec_decode.py` | ~364 | Speculative decoding |
| `LLM_inference/best_parallelization.py` | ~132 | Parallelism search |
| `LLM_inference/utils.py` | ~155 | Modeling output, system creation |
| `simulation/engine.py` | ~320 | Unified simulation facade |
| `simulation/config.py` | ~212 | Simulation configuration |
| `simulation/results.py` | ~218 | Unified result format |
| `features/registry.py` | ~258 | Feature discovery/management |
| `features/base.py` | ~100 (est) | Base feature interface |

## Appendix B: Key Algorithms

### Roofline Model (per operator)

```
Input: operator dimensions, system specs
1. Compute FLOPs = 2 * num_ops (MAC = multiply + add)
2. Compute data bytes = sum(tensor_sizes * precision_bytes)
3. Operational intensity = FLOPs / bytes
4. Compute time = FLOPs * precision_multiplier / (peak_ops * compute_eff * tensor_core_eff)
5. Memory time = sum(tensor_bytes / appropriate_bandwidth) / memory_eff
6. Comm time = collective_time / comm_eff
7. Execution time = max(compute_time, memory_time, comm_time)
8. Throughput = FLOPs / execution_time
```

### Memory Offloading

```
Input: total_memory_req, system
1. device_memory = system.off_chip_mem_size
2. overflow = total_memory_req - device_memory
3. If overflow > 0:
   effective_BW = total_memory_req / max(device_memory/BW_hbm, overflow/BW_offload)
   system.offchip_mem_bw = effective_BW
```

### Speculative Decoding Expected Tokens

```
Input: N (parallel tokens), x (acceptance rate)
E[tokens] = 1 + sum(k * x^k * (1-x) for k=1..N-1) + N * x^N
```

### Hierarchical AllReduce (multi-node)

```
Input: data_size, num_gpus, gpus_per_node
1. num_nodes = num_gpus / gpus_per_node
2. Intra-RS: (gpus-1) * data/gpus / (NVLink_BW * 0.9)
3. Inter-AR: 2*(nodes-1) * (data/gpus) / nodes / (IB_BW * bw_eff)  [ring]
   OR: 2 * log2(nodes) * inter_latency + data / (IB_BW * 0.8)  [tree]
4. Intra-AG: same as intra-RS
5. Total = (RS + AR + AG) * overhead_factor
```
