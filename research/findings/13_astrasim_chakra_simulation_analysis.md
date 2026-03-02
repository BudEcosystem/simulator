# ASTRA-SIM, Chakra, Scale-Sim & Simulation Engine: BudSimulator vs LLMServingSim 2.0

**Analyst**: Systems Architect
**Date**: 2026-02-28
**Scope**: Complete analysis of simulation subsystem integration in both projects, with actionable integration plan

---

## Table of Contents

1. [BudSimulator File-by-File Analysis](#1-budsimulator-file-by-file-analysis)
2. [LLMServingSim File-by-File Analysis](#2-llmservingsim-file-by-file-analysis)
3. [Side-by-Side Comparison](#3-side-by-side-comparison)
4. [LLMServingSim Extensions BudSim Should Adopt](#4-llmservingsim-extensions-budsim-should-adopt)
5. [Specific Reusable Code from LLMServingSim](#5-specific-reusable-code-from-llmservingsim)
6. [Integration Architecture](#6-integration-architecture)
7. [Proposed Hybrid Approach](#7-proposed-hybrid-approach)

---

## 1. BudSimulator File-by-File Analysis

### 1.1 `get_astra_sim_time.py` (ASTRA-SIM Integration)

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/Astra_sim/get_astra_sim_time.py`

#### A. Current Functionality

This file provides a **single-operation collective timing estimator**. Its primary function `get_astrasim_collective_time()` takes a collective operation type and size, then:

1. **Writes a Chakra text trace** to `/tmp/genz/chakra/txt_file.txt` containing a single collective operation (ALLREDUCE, ALLTOALL, ALLGATHER, or REDUCESCATTER)
2. **Converts text to Chakra ET format** using `chakra.src.converter.converter` subprocess
3. **Cleans ET traces** via `fix_chakra_traces.convert_chakra_file()` to fix protobuf attribute types
4. **Generates network.yml** from either a `System` object or a `network_config` dict
5. **Runs ASTRA-SIM binary** (`AnalyticalAstra`) via `run.sh` shell script
6. **Parses output** for per-system cycle counts

Supporting functions:
- `merge_parallelism_heirarchy()`: Merges parallelism dimensions (e.g., folds EP into TP). Lines 23-52.
- `get_network_config()`: Extracts network config for a specific parallelism dimension from a multi-dimensional hierarchy. Lines 80-115.
- `divide_npus_count()`: Maps parallelism dimensions to physical network dimensions. Lines 55-78.
- `replace_collective_implementation()`: Updates `system.json` with topology-appropriate algorithms. Lines 119-129.

#### B. Data Flows

**Input:**
- `collective_size: int` -- Size in bytes of the collective operation
- `collective_type: str` -- One of ALLREDUCE/ALLTOALL/ALLGATHER/REDUCESCATTER
- `system: System` (optional) -- GenZ System object with topology, bandwidth, latency, num_nodes
- `network_config: dict` (optional) -- Direct network config with keys: topology, npus_count, bandwidth, latency

**Intermediate formats:**
- Chakra text format: `"MICRO\n1\nDUMMYNAME -1 5 NONE 0 5 NONE 0 5 {TYPE} {SIZE} 5\n"`
- ET protobuf files: Binary Chakra execution traces (.et files)
- Cleaned ET files: Post-processed protobuf with corrected attribute types
- `network.yml`: YAML with topology, npus_count, bandwidth, latency arrays
- `system.json`: JSON with collective algorithm implementations and system parameters

**Output:**
- `dict` mapping system ID (string) to cycle count (integer in nanoseconds)

#### C. Issues and Limitations

1. **Hardcoded path** at line 210: `replace_collective_implementation('/home/abhimanyu/synergy3/work/GenZ-LLM-Analyzer/GenZ/Astra_sim/system.json', ...)` -- absolute path to developer's machine, will break on any other system
2. **Single-operation scope**: Can only simulate one collective at a time; no support for overlapping compute+communication or multi-operation execution graphs
3. **No runtime loop**: Fires ASTRA-SIM once per collective call; no iterative scheduling
4. **Temporary file pollution**: Uses `/tmp/genz/chakra/` without cleanup
5. **No memory model integration**: Only times collectives; does not track memory state

---

### 1.2 `fix_chakra_traces.py` (Chakra Trace Post-Processing)

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/Astra_sim/fix_chakra_traces.py`

#### A. Current Functionality

Fixes type mismatches in Chakra protobuf attributes that the converter generates incorrectly. The key operations:

1. `process_attr()` (lines 38-48): Converts protobuf attributes to required types per `required_type` mapping (e.g., `num_ops` -> `int64_val`, `tensor_size` -> `uint64_val`, `comm_size` -> `int64_val`)
2. `process_chakra_node()` (lines 51-63): Iterates node attributes, applies type corrections, and adds `is_cpu_op=False` for communication nodes
3. `convert_chakra_file()` (lines 66-77): Stream-processes a single .et file, reading GlobalMetadata + Node messages and writing corrected versions

#### B. Data Flows

**Input:** Binary protobuf .et file (Chakra execution trace)
**Output:** Corrected binary protobuf .et file with proper attribute types

Uses dual import strategy (lines 6-24) to support both pre- and post-refactored Chakra package layouts:
- `chakra.et_def.et_def_pb2` (old)
- `chakra.schema.protobuf.et_def_pb2` (new)

#### C. Comparison Note

LLMServingSim does **not** need this fix because it generates traces through its own `trace_generator.py` which writes the text format correctly for its version of the Chakra converter. BudSim needs this because it uses a generic text format that the converter does not type-cast correctly.

---

### 1.3 `path_utils.py` (Security Utilities)

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/Astra_sim/path_utils.py`

#### A. Current Functionality

Security-hardening module for subprocess calls:
- `validate_path()`: Checks paths for shell metacharacters (`;`, `&`, `|`, `>`, `<`, `` ` ``, `$`, etc.) and command injection patterns
- `validate_script_path()`: Additionally resolves symlinks and optionally constrains paths to an allowed base directory

#### B. Data Flows

**Input:** String or Path object
**Output:** Validated `Path` object or `PathValidationError` exception

#### C. Note

This module was added as part of security hardening (commit `6099419`). It is **not currently used** by `get_astra_sim_time.py` -- the subprocess calls in that file directly invoke `['bash', run_file]` without path validation. This is a gap.

---

### 1.4 `get_scale_sim_time.py` (Scale-Sim Integration)

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/Scale_Sim/get_scale_sim_time.py`

#### A. Current Functionality

Provides compute timing for individual operators via a lookup table approach:
- Takes an `Operator` and `System` object
- Determines operator type (GEMM, Logit, Attend) and dimensions
- Looks up cycle counts from a **pre-computed CSV file** at a hardcoded path (`/Users/abambhaniya3/GenZ/GenZ_paper_charts/...`)
- Maps cycles to time using system frequency and MXU shape

#### B. Data Flows

**Input:** `Operator` object (with dimensions), `System` object (with frequency, MXU shape)
**Output:** Float time value (seconds)

**Lookup table format:** CSV with columns B, M, N, K and columns for MXU sizes (256, 128)

#### C. Issues

1. **Hardcoded absolute path** (line 23): Points to a specific developer's machine
2. **Limited operator coverage**: Only GEMM, Logit, Attend; falls back to analytical formula for unknown shapes
3. **No integration with ASTRA-SIM**: Scale-Sim timing is completely separate from collective timing
4. **Read-only**: Only reads pre-computed data; does not invoke Scale-Sim dynamically

---

### 1.5 Simulation Engine (`engine.py`, `config.py`, `results.py`)

**Files:**
- `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/simulation/engine.py`
- `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/simulation/config.py`
- `/home/bud/Desktop/bud_model_factory/bud_simulator/llm-memory-calculator/src/llm_memory_calculator/genz/simulation/results.py`

#### A. Current Functionality

The simulation engine provides a **unified interface** wrapping the GenZ analytical modeling functions:

**`SimulationEngine` (engine.py):**
- Orchestrates simulation with pre/post feature hooks via `FeatureRegistry`
- Delegates to `prefill_moddeling()`, `decode_moddeling()`, or `chunked_moddeling()` from the GenZ LLM inference module
- Converts raw outputs to `SimulationResult` objects
- Supports feature-based extensions (tensor_parallel, pipeline_parallel, lora, flash_attention)

**`SimulationConfig` (config.py):**
- Dataclass with model, features list, simulation_params, and system_config
- Auto-determines simulation type from features (PREFILL, DECODE, CHUNKED, CONTINUOUS)
- Validates feature compatibility (e.g., prefill/decode mutually exclusive)
- Sets default parameters per simulation type

**`SimulationResult` (results.py):**
- Unified result object with latency, throughput, runtime_breakdown, memory_usage, hardware_utilization
- Factory methods: `from_prefill_output()`, `from_decode_output()`, `from_chunked_output()`
- Serialization to/from dict/JSON

#### B. Data Flows

**Input:** `SimulationConfig` with model name, features, and parameters
**Processing:** Analytical roofline modeling via GenZ framework
**Output:** `SimulationResult` with all metrics

#### C. Critical Observation

The simulation engine is **entirely analytical** -- it calls GenZ modeling functions that compute performance from first principles (operator FLOPs, roofline analysis, memory bandwidth). There is **no integration** with ASTRA-SIM or Scale-Sim in this engine. The ASTRA-SIM integration (`get_astra_sim_time.py`) is called separately at a lower level by the `parallelism.py` module for collective timing, not orchestrated by this engine.

---

### 1.6 Chakra Stubs

**Files:**
- `/home/bud/Desktop/bud_model_factory/bud_simulator/BudSimulator/chakra/__init__.py`
- `/home/bud/Desktop/bud_model_factory/bud_simulator/chakra/__init__.py`
- `/home/bud/Desktop/bud_model_factory/bud_simulator/BudSimulator/chakra/et_def/et_def_pb2.py`

#### A. Current Functionality

These are **stub packages** providing minimal Chakra types for testing:
- `GlobalMetadata`, `Node`, `NodeType`, `AttributeProto` classes
- Stub protolib functions (`encodeMessage`, `decodeMessage`, `openFileRd`) that return None/False
- Module registration in `sys.modules` to satisfy import chains

The BudSimulator chakra stubs are test/development shims. The real Chakra package must be separately installed for actual ASTRA-SIM execution.

---

## 2. LLMServingSim File-by-File Analysis

### 2.1 `main.py` -- Runtime Loop & Orchestration

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/main.py`

Monolithic ~546-line entry point that:
1. Parses 20+ CLI arguments (lines 34-58)
2. Builds cluster config via `build_cluster_config()` (line 97)
3. Creates per-instance `Scheduler` objects with `MemoryModel` (lines 188-204)
4. Creates `Controller` for ASTRA-SIM IPC, `Router` for request routing (lines 207-209)
5. Generates initial event handler trace + Chakra graph (lines 256-260)
6. Starts ASTRA-SIM as a **long-running subprocess** via `Popen` with stdin/stdout pipes (line 269)
7. **Runs iterative simulation loop** (lines 273-451): read ASTRA-SIM output -> schedule next batch -> generate trace + graph -> write workload back
8. Collects throughput, TTFT, TPOT, ITL, power metrics (lines 453-538)

**Key difference from BudSim**: ASTRA-SIM runs as a **persistent subprocess** with bidirectional IPC, not a fire-and-forget invocation.

### 2.2 `controller.py` -- ASTRA-SIM IPC Protocol

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/controller.py`

Manages stdin/stdout communication with the ASTRA-SIM C++ process:
- `read_wait()` (line 13): Blocks until ASTRA-SIM outputs "Waiting" or system check message
- `write_flush()` (line 32): Writes workload path or control command ("exit", "done", "pass") to ASTRA-SIM's stdin
- `parse_output()` (line 39): Regex-parses `sys[X] iteration Y finished, Z cycles, exposed communication W cycles` to extract system ID, iteration, cycle count, and exposed communication cycles
- `check_end()` (line 23): Waits for "All Request Has Been Exited" confirmation

### 2.3 `scheduler.py` -- Batch Scheduler

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/scheduler.py`

A 715-line batch scheduler that implements vLLM/Orca-style continuous batching:

- **`schedule_base()`** (lines 59-239): Selects ready requests, checks memory via `MemoryModel`, applies KV cache eviction under memory pressure, enforces `max_batch` and `max_num_batched_tokens` constraints, creates `Batch` objects with prefill/decode token lists
- **`schedule_with_prefix()`** (lines 241-477): Extended scheduler with prefix cache matching, multi-tier eviction (NPU -> CPU/CXL), and prefix-aware batch formation
- **`add_done()`** (lines 480-578): Handles batch completion: transitions requests from prefill to decode phase, updates KV cache state, collects TTFT/TPOT/ITL metrics
- **`print_result()`** (lines 632-669): Computes mean/median/P99 for TTFT, TPOT, ITL
- **`save_output()`** (lines 687-715): Writes per-request CSV with full lifecycle metrics

### 2.4 `memory_model.py` -- Multi-Tier Memory Management

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/memory_model.py`

A 558-line memory model implementing:

- **Weight calculation** (`get_weight()`, lines 92-153): Per-layer weight computation accounting for TP sharding, embedding, QKV projections, FFN, MoE experts, layernorms, lm_head
- **KV cache management** (lines 156-196): Block-based KV allocation with `get_kv()`, `get_block_kv()`, `get_evict_kv()`, `get_total_kv()`
- **Multi-tier allocation** (lines 217-284): NPU/CPU/CXL memory allocation with bounds checking, error reporting with node/instance context
- **Prefix cache integration** (lines 337-541): RadixCache-based prefix matching, lock/unlock reference counting, event-driven allocation tracking (`apply_kv_cache_events()`)
- **`calculate_sizes()`** (lines 544-752): Per-layer input/weight/output size calculation with full TP-aware sharding -- 200+ lines covering every layer type (embedding, QKV, attention, FFN, MoE, lm_head)

### 2.5 `graph_generator.py` -- Execution Graph Construction

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/graph_generator.py`

Converts text traces to Chakra execution graphs:
- Calls `chakra.src.converter.converter` with mode `LLM` (vs BudSim's `Text` mode)
- Supports `--npu-offset` for multi-instance NPU mapping
- Supports `--local-offloading` flag for weight placement
- Per-batch graph generation: each scheduling decision creates a new execution graph

### 2.6 `config_builder.py` -- Cluster Configuration

**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/config_builder.py`

A 558-line configuration builder that:
- Parses multi-node, multi-instance cluster JSON configs
- Generates `network.yml`, `system.json`, `memory_expansion.json` for ASTRA-SIM
- Supports heterogeneous instances (different models, hardware, parallelism per instance)
- Implements memory placement policies (NPU/CPU/CXL per-layer weight/KV placement)
- Supports PIM (Processing-in-Memory) configuration
- Supports power modeling configuration with 7 component types
- Validates placement against available memory configuration

---

## 3. Side-by-Side Comparison

### 3.1 ASTRA-SIM Usage

| Aspect | BudSimulator | LLMServingSim 2.0 |
|--------|-------------|-------------------|
| **Invocation model** | Fire-and-forget: runs ASTRA-SIM binary to completion per collective | Persistent subprocess: ASTRA-SIM runs continuously, fed workloads via stdin/stdout IPC |
| **Scope of simulation** | Single collective operation at a time | Full multi-iteration serving workload with dynamic scheduling |
| **Lifecycle** | `subprocess.run()` -- blocks until done | `subprocess.Popen()` with `Controller` managing bidirectional pipes |
| **Purpose** | Measure collective latency (AllReduce, AllGather, etc.) for roofline analysis | Simulate end-to-end inference serving with network, compute, memory timing |
| **Network config** | Generated per-call from `System` object or raw dict | Generated once from cluster config JSON at initialization |
| **System config** | Hardcoded path to `system.json`, modified in-place | Generated from cluster config, stored in astra-sim directory |
| **Memory config** | `"NO_MEMORY_EXPANSION"` only | Multi-tier: `PER_NPU_MEMORY_EXPANSION`, `PER_NODE_MEMORY_EXPANSION`, `MEMORY_POOL` (CXL) |
| **Parallelism** | Single parallelism dimension per call | Multi-dimensional: TP within instance, PP across instances, EP for MoE |
| **Integration depth** | Isolated utility function | Core simulation backbone; every scheduling cycle produces a new ASTRA-SIM workload |

### 3.2 Chakra Usage

| Aspect | BudSimulator | LLMServingSim 2.0 |
|--------|-------------|-------------------|
| **Converter mode** | `Text` mode -- generic text-to-ET conversion | `LLM` mode -- LLM-specific conversion with NPU offset support |
| **Trace content** | Single collective operation per trace | Full transformer layer sequence per batch (embedding, QKV, attention, FFN, collectives, memory ops) |
| **Post-processing** | Requires `fix_chakra_traces.py` to correct attribute types | No post-processing needed (converter handles types correctly in LLM mode) |
| **Execution graphs** | Static: one collective per graph | Dynamic: per-batch graphs incorporating current batch composition, memory state, placement decisions |
| **NPU mapping** | All NPUs start at 0 | Per-instance NPU offset via `--npu-offset` flag |
| **Package** | Stub packages for testing; real Chakra needed for execution | Full Chakra included as git submodule at `extern/graph_frontend/chakra` |

### 3.3 Simulation Engine

| Aspect | BudSimulator SimulationEngine | LLMServingSim main.py Loop |
|--------|-------------------------------|---------------------------|
| **Paradigm** | Analytical (roofline-based) | Discrete-event simulation (tick-based via ASTRA-SIM) |
| **Scheduling** | None -- single batch analyzed in isolation | Continuous batching with dynamic request arrival, memory-aware scheduling |
| **Memory model** | Static: calculates memory requirements at analysis time | Dynamic: tracks NPU/CPU/CXL usage across iterations, supports eviction/loading |
| **Request lifecycle** | N/A -- analyzes one-shot prefill or decode | Full lifecycle: arrival -> queue -> schedule -> prefill -> decode -> completion |
| **Multi-instance** | N/A | Multiple instances per node, with P/D disaggregation |
| **Metrics** | Latency, throughput, runtime breakdown (analytical) | TTFT, TPOT, ITL (mean/median/P99), throughput over time, power consumption |
| **Feature system** | Plugin-based via FeatureRegistry | Hardcoded feature flags (prefix caching, PIM, sub-batch interleaving) |

### 3.4 Memory Model

| Aspect | BudSimulator | LLMServingSim 2.0 |
|--------|-------------|-------------------|
| **Weight tracking** | Analytical formula in GenZ operator framework | Per-layer calculation with TP sharding in `calculate_sizes()` |
| **KV cache** | Static capacity check in prefill/decode modeling | Dynamic block-based allocation with eviction, loading, multi-tier (NPU/CPU/CXL) |
| **Prefix caching** | Not implemented | Full RadixCache implementation with per-node sharing and multi-tier storage |
| **Memory pressure** | Reports OOM as error; no recovery | Graceful eviction: evicts decode KV caches under pressure, spills to CPU/CXL |
| **Memory operations** | Not modeled in execution | Modeled as explicit data movement in execution traces (load/evict/transfer) |

---

## 4. LLMServingSim Extensions BudSim Should Adopt

### 4.1 Persistent ASTRA-SIM Subprocess with IPC (HIGH PRIORITY)

**What:** Replace fire-and-forget `subprocess.run()` with persistent `subprocess.Popen()` and stdin/stdout IPC protocol.

**Why:** Enables multi-iteration simulation, dynamic workload injection, and amortizes ASTRA-SIM startup cost. Currently BudSim pays full process creation + initialization cost per collective call.

**Reference:** `controller.py` lines 4-57, `main.py` lines 262-269.

### 4.2 Execution Graph Approach (HIGH PRIORITY)

**What:** Generate per-batch execution graphs that include compute operators, communication, and memory operations as a connected DAG, rather than individual collective timing queries.

**Why:** The current BudSim approach times each collective independently, losing:
- Compute-communication overlap (critical for TP efficiency)
- Memory access contention
- Pipeline stage dependencies
- Network congestion effects across concurrent collectives

**Reference:** `trace_generator.py` lines 40-100, `graph_generator.py` lines 9-36.

### 4.3 Dynamic Memory Model with Eviction (HIGH PRIORITY)

**What:** Implement runtime-aware memory tracking with multi-tier allocation (NPU/CPU/CXL), block-based KV cache management, and graceful eviction under pressure.

**Why:** BudSim currently only checks if a model fits in memory. It cannot model:
- KV cache growth across decode iterations
- Memory pressure causing eviction to CPU/host
- Prefix caching benefits
- Memory-constrained scheduling decisions

**Reference:** `memory_model.py` full file, especially `allocate()`/`free()`/`is_avail()` pattern (lines 217-284) and `get_block_kv()` (lines 172-186).

### 4.4 Continuous Batching Scheduler (MEDIUM PRIORITY)

**What:** Add an iteration-level scheduler that selects which requests to batch based on current memory state, arrival time, and configurable policies.

**Why:** Real serving systems use continuous batching (Orca/vLLM style). BudSim's analytical engine models a single static batch, which cannot capture:
- Queuing delays from concurrent requests
- TTFT/TPOT distributions across request streams
- Memory-constrained scheduling effects
- P/D disaggregation dynamics

**Reference:** `scheduler.py` lines 59-239 (`schedule_base()`).

### 4.5 Multi-Tier Memory Hierarchy Configuration (MEDIUM PRIORITY)

**What:** Support CXL-attached memory, per-layer weight/KV placement policies, and PIM-capable memory tiers in the memory configuration.

**Why:** Emerging hardware architectures use heterogeneous memory (NPU HBM + CPU DRAM + CXL pooled memory + PIM). BudSim's flat memory model cannot evaluate these architectures.

**Reference:** `config_builder.py` lines 20-404, placement resolution via `get_device()` (lines 488-512).

### 4.6 Profile-Driven Operator Timing (LOW PRIORITY)

**What:** Support pre-profiled operator latency databases indexed by (hardware, model, TP, input_len, kv_cache_len, layer_name).

**Why:** Supplements analytical roofline with empirical measurements. Useful for operators where analytical models are inaccurate (attention with FlashAttention, MoE routing overhead).

**Reference:** `trace_generator.py` lines 22-35 (performance database cache).

### 4.7 Power Modeling (LOW PRIORITY)

**What:** Add 7-component power model (NPU active/idle/standby, CPU, DRAM, link, NIC, storage) with energy-per-bit and per-operation tracking.

**Why:** Power/energy is increasingly important for TCO analysis, especially for large-scale deployments.

**Reference:** `power_model.py` (not analyzed in detail but referenced throughout `main.py`).

---

## 5. Specific Reusable Code from LLMServingSim

### 5.1 Controller IPC Protocol

**Source:** `controller.py:4-57`
```python
# Directly reusable with minor adaptation:
# - read_wait(): Blocks on ASTRA-SIM stdout until "Waiting" signal
# - write_flush(): Sends workload path or control command via stdin
# - parse_output(): Regex parsing of cycle/communication results
# - check_end(): Waits for clean termination
```

**Adaptation needed:** Generalize the parse patterns; add timeout handling; integrate with BudSim's existing System/Operator abstractions.

### 5.2 Memory Model Core

**Source:** `memory_model.py:16-303`

Key reusable patterns:
- **Multi-tier allocator** (`allocate()`/`free()`/`is_avail()`, lines 217-303): Generic device-dispatched allocation with bounds checking and logging
- **KV cache sizing** (`get_kv()`, line 156-162): Formula: `2 * kv_dim * seq * n_layer * fp // npu_num`
- **Block-based KV management** (`get_block_kv()`, lines 172-186): Calculates incremental KV allocation per batch considering evicted and initial requests

**Adaptation needed:** Replace `get_config(model)` with BudSim's model registry lookup; integrate Device enum with GenZ System class.

### 5.3 Per-Layer Size Calculation

**Source:** `memory_model.py:544-752` (`calculate_sizes()`)

This 200-line function computes per-rank input/weight/output sizes for every transformer layer type with TP-aware sharding. Covers:
- Embedding (vocab-parallel), layernorms (replicated)
- Q/K/V projections (column-parallel with GQA)
- O projection (row-parallel)
- Dense FFN: gate_proj, up_proj, down_proj (SwiGLU)
- MoE: gate + expert.w1/w2/w3 (EP = TP)
- LM head (vocab-parallel)

**Adaptation needed:** BudSim already has its own weight calculation in GenZ operators; this could be used for validation or as an alternative that's aligned with vLLM conventions.

### 5.4 Batch Scheduler Logic

**Source:** `scheduler.py:59-215` (`schedule_base()`)

Reusable scheduling algorithm:
1. Filter requests by arrival time (line 71)
2. Apply max batch constraint (line 72)
3. Memory-aware batch trimming with eviction (lines 95-131)
4. Max tokens constraint (lines 136-150)
5. KV recomputation and allocation (lines 153-170)
6. Batch object creation with prefill/decode token lists (lines 176-205)

**Adaptation needed:** Extract scheduling policy from instance-specific code; parameterize eviction strategy; integrate with BudSim's analytical engine for per-batch performance estimation.

### 5.5 Cluster Configuration Builder

**Source:** `config_builder.py:20-404` (`build_cluster_config()`)

Reusable configuration system:
- Multi-node, multi-instance cluster definition
- NPU-to-instance and instance-to-node mapping
- Network topology generation (`_create_network_config()`, lines 407-426)
- Memory configuration generation with placement validation
- P/D disaggregation support

**Adaptation needed:** BudSim already has hardware configuration in its database. The cluster config builder can extend this with multi-instance serving topologies.

---

## 6. Integration Architecture

### 6.1 Current BudSim Architecture

```
User API Request
      |
      v
SimulationEngine.simulate(config)
      |
      v
GenZ Analytical Functions
  (prefill_moddeling / decode_moddeling / chunked_moddeling)
      |
      |-- per-operator roofline analysis
      |-- collective timing (ASTRA-SIM one-shot per collective)
      |-- memory capacity check
      |
      v
SimulationResult
  (latency, throughput, runtime_breakdown)
```

### 6.2 Proposed Integrated Architecture

```
User API Request
      |
      v
SimulationEngine.simulate(config)
      |
      +-- mode: "analytical" (existing, fast)
      |     |
      |     v
      |   GenZ Analytical Functions (unchanged)
      |     |
      |     v
      |   SimulationResult
      |
      +-- mode: "simulation" (new, detailed)
      |     |
      |     v
      |   RuntimeSimulator (NEW)
      |     |
      |     +-- ClusterConfig (from config_builder)
      |     +-- Scheduler[] (one per instance)
      |     |     +-- MemoryModel (multi-tier, dynamic)
      |     +-- AstraSimController (persistent subprocess IPC)
      |     +-- TraceGenerator (per-batch execution traces)
      |     +-- GraphGenerator (Chakra ET conversion)
      |     |
      |     v
      |   Simulation Loop:
      |     1. Read ASTRA-SIM output (cycle, NPU status)
      |     2. Schedule next batch (memory-aware, continuous batching)
      |     3. Generate trace + Chakra graph for batch
      |     4. Feed workload to ASTRA-SIM
      |     5. Collect per-request metrics (TTFT, TPOT, ITL)
      |     |
      |     v
      |   SimulationResult (extended with runtime metrics)
      |
      +-- mode: "hybrid" (new, balanced)
            |
            v
          HybridSimulator (NEW)
            |
            +-- Use analytical engine for compute timing
            +-- Use ASTRA-SIM for network timing
            +-- Use dynamic MemoryModel for memory state
            +-- Scheduler for batch-level orchestration
            |
            v
          SimulationResult
```

### 6.3 New Module Hierarchy

```
llm_memory_calculator/genz/
  simulation/
    engine.py              # Extended with mode selection
    config.py              # Extended with simulation mode params
    results.py             # Extended with runtime metrics
    runtime/               # NEW: runtime simulation subsystem
      __init__.py
      simulator.py         # RuntimeSimulator class
      controller.py        # ASTRA-SIM IPC (adapted from LLMServingSim)
      scheduler.py         # Continuous batching scheduler
      memory_model.py      # Multi-tier dynamic memory model
      trace_generator.py   # Per-batch trace synthesis
      cluster_config.py    # Multi-node cluster configuration
      request.py           # Request lifecycle management
```

### 6.4 Integration Points with Existing Code

1. **SimulationConfig** (config.py): Add `simulation_mode: "analytical" | "simulation" | "hybrid"` field and `cluster_config` for runtime mode
2. **SimulationEngine** (engine.py): Add `_run_runtime_simulation()` method that delegates to `RuntimeSimulator`
3. **SimulationResult** (results.py): Add `serving_metrics` field with TTFT/TPOT/ITL distributions
4. **System class**: Use existing GenZ System for hardware specs; extend with multi-tier memory configuration
5. **Operator framework**: Reuse GenZ operators for analytical compute timing in hybrid mode
6. **ASTRA-SIM integration**: Replace fire-and-forget `get_astrasim_collective_time()` with persistent controller in runtime mode; keep existing function for analytical mode

---

## 7. Proposed Hybrid Approach

### 7.1 Design Rationale

The analytical (GenZ roofline) and simulation (ASTRA-SIM runtime) approaches serve different needs:

| Criterion | Analytical | Simulation | Hybrid |
|-----------|-----------|------------|--------|
| Speed | Fast (ms per analysis) | Slow (minutes per run) | Moderate |
| Accuracy (compute) | Good (validated roofline) | Good (profile-driven) | Good (use analytical) |
| Accuracy (network) | Approximate (formula-based) | Accurate (ASTRA-SIM) | Accurate (ASTRA-SIM) |
| Memory dynamics | Static (capacity check) | Dynamic (runtime tracking) | Dynamic |
| Multi-request | No | Yes (continuous batching) | Yes |
| Setup complexity | Low | High (ASTRA-SIM binary required) | Medium |

### 7.2 Hybrid Architecture

The hybrid approach uses each tool for what it does best:

1. **Compute timing**: GenZ analytical engine (fast, validated, no external dependency)
2. **Communication timing**: ASTRA-SIM persistent subprocess (accurate network modeling)
3. **Memory state**: Dynamic MemoryModel tracking (captures runtime behavior)
4. **Scheduling**: Continuous batching scheduler with analytical compute estimates

```
Hybrid Simulation Loop:
  for each scheduling cycle:
    1. Scheduler selects batch (memory-aware)
    2. For each operator in batch:
       a. Compute time = GenZ roofline analysis (analytical)
       b. Communication time = ASTRA-SIM (if collective operation)
    3. Total batch time = max(compute_pipeline, communication_pipeline)
       (accounts for compute-communication overlap)
    4. Update MemoryModel state (KV cache growth/eviction)
    5. Advance simulation clock by batch time
    6. Collect per-request metrics
```

### 7.3 Advantages of Hybrid

1. **No profiling database required**: Uses GenZ roofline instead of hardware-specific CSV databases
2. **Accurate network timing**: ASTRA-SIM models congestion, topology effects, collective algorithms
3. **Runtime dynamics**: Dynamic memory tracking + continuous batching capture real serving behavior
4. **Graceful degradation**: If ASTRA-SIM is not installed, falls back to analytical collective timing
5. **Existing investment**: Leverages all GenZ analytical work (operator framework, parallelism strategies, memory estimation)

### 7.4 Implementation Priority

**Phase 1 -- Dynamic Memory Model (1-2 weeks)**
- Port `MemoryModel` from LLMServingSim
- Integrate with GenZ System class
- Add multi-tier support (NPU HBM + CPU DRAM)
- Add block-based KV cache management

**Phase 2 -- Persistent ASTRA-SIM Controller (1 week)**
- Implement `AstraSimController` adapted from LLMServingSim's `Controller`
- Replace `get_astrasim_collective_time()` fire-and-forget with persistent process
- Support fallback to analytical timing when ASTRA-SIM binary not available

**Phase 3 -- Batch Scheduler + Request Lifecycle (1-2 weeks)**
- Implement continuous batching scheduler adapted from LLMServingSim
- Add Request class with lifecycle tracking (arrival -> queue -> schedule -> prefill -> decode -> done)
- Add serving metrics collection (TTFT, TPOT, ITL distributions)

**Phase 4 -- Hybrid Integration (1 week)**
- Wire hybrid loop into SimulationEngine
- Add `simulation_mode` to SimulationConfig
- Extend SimulationResult with serving metrics
- Add ClusterConfig for multi-instance scenarios

**Phase 5 -- Execution Graph Generation (2 weeks)**
- Port trace_generator concept (per-batch operator sequence with memory operations)
- Generate Chakra-compatible execution graphs using LLM converter mode
- Enable full end-to-end ASTRA-SIM driven simulation as optional mode

---

## Appendix: File Reference Index

| File | Location | Lines | Role |
|------|----------|-------|------|
| `get_astra_sim_time.py` | `llm-memory-calculator/.../Astra_sim/` | 238 | BudSim ASTRA-SIM collective timing |
| `fix_chakra_traces.py` | `llm-memory-calculator/.../Astra_sim/` | 132 | BudSim Chakra protobuf type fixer |
| `path_utils.py` | `llm-memory-calculator/.../Astra_sim/` | 103 | BudSim path security validation |
| `get_scale_sim_time.py` | `llm-memory-calculator/.../Scale_Sim/` | 77 | BudSim Scale-Sim lookup timing |
| `engine.py` | `llm-memory-calculator/.../simulation/` | 320 | BudSim unified simulation engine |
| `config.py` | `llm-memory-calculator/.../simulation/` | 212 | BudSim simulation configuration |
| `results.py` | `llm-memory-calculator/.../simulation/` | 218 | BudSim simulation results |
| `main.py` | `LLMServingSim/` | 546 | LLMServingSim entry point + sim loop |
| `controller.py` | `LLMServingSim/inference_serving/` | 57 | LLMServingSim ASTRA-SIM IPC |
| `scheduler.py` | `LLMServingSim/inference_serving/` | 715 | LLMServingSim batch scheduler |
| `memory_model.py` | `LLMServingSim/inference_serving/` | 752 | LLMServingSim multi-tier memory |
| `graph_generator.py` | `LLMServingSim/inference_serving/` | 36 | LLMServingSim Chakra graph builder |
| `config_builder.py` | `LLMServingSim/inference_serving/` | 558 | LLMServingSim cluster config |
| `trace_generator.py` | `LLMServingSim/inference_serving/` | ~600 | LLMServingSim per-batch trace synthesis |
