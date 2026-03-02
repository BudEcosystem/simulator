# LLMServingSim 2.0 Codebase Analysis

**Analyst**: Core Module Specialist
**Date**: 2026-02-28
**Scope**: Full source analysis of the LLMServingSim 2.0 simulator codebase

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [main.py - Entry Point & Simulation Loop](#2-mainpy)
3. [inference_serving/controller.py - ASTRA-SIM IPC](#3-controllerpy)
4. [inference_serving/scheduler.py - Batch Scheduler](#4-schedulerpy)
5. [inference_serving/memory_model.py - Multi-Tier Memory](#5-memory_modelpy)
6. [inference_serving/radix_tree.py - Prefix Caching](#6-radix_treepy)
7. [inference_serving/power_model.py - 7-Component Power Model](#7-power_modelpy)
8. [inference_serving/graph_generator.py - Execution Graph Construction](#8-graph_generatorpy)
9. [inference_serving/router.py - Request Routing](#9-routerpy)
10. [inference_serving/request.py - Request Lifecycle](#10-requestpy)
11. [inference_serving/trace_generator.py - Trace Synthesis](#11-trace_generatorpy)
12. [inference_serving/config_builder.py - Configuration Management](#12-config_builderpy)
13. [inference_serving/utils.py & logger.py - Utilities](#13-utilspy--loggerpy)
14. [inference_serving/attn_utils.py - FlashAttention Heuristics](#14-attn_utilspy)
15. [inference_serving/gate_function.py - MoE Gating](#15-gate_functionpy)
16. [inference_serving/pim_model.py - Processing-in-Memory](#16-pim_modelpy)
17. [llm_profile/ - Profiler & Model Definitions](#17-llm_profile)
18. [Configuration Files (cluster_config, model_config)](#18-configuration-files)
19. [Comparison with GenZ/BudSimulator Framework](#19-comparison-with-genzbudsimulator)
20. [Design Patterns Worth Borrowing](#20-design-patterns-worth-borrowing)
21. [Summary of All Findings](#21-summary-of-all-findings)

---

## 1. Architecture Overview

LLMServingSim 2.0 is a discrete-event simulator for LLM serving that uses ASTRA-SIM as the underlying network/compute simulation backend. The architecture is:

```
main.py (Orchestrator)
  |
  +-- config_builder.py  ->  Reads cluster JSON, builds ASTRA-SIM configs
  +-- Router             ->  Routes requests to instances
  +-- Scheduler[]        ->  One per instance, manages batching/scheduling
  |     +-- MemoryModel  ->  Per-instance multi-tier memory (NPU, CPU, CXL)
  |     +-- RadixCache   ->  Prefix caching via radix tree
  +-- Controller         ->  IPC with ASTRA-SIM subprocess (stdin/stdout)
  +-- trace_generator    ->  Generates ASTRA-SIM execution traces per batch
  +-- graph_generator    ->  Converts traces to Chakra computation graphs
  +-- PowerModel         ->  7-component power modeling
  +-- PIMModel           ->  Processing-in-Memory attention offloading

ASTRA-SIM (C++ subprocess)
  |
  +-- Simulates compute, memory, and network timing
  +-- Returns cycle-accurate results via stdout
```

**Key Design Decisions:**
- Tick-based simulation: time advances in nanosecond cycles reported by ASTRA-SIM
- Per-instance scheduling: each GPU instance has its own scheduler and memory model
- Profile-driven: uses pre-profiled performance databases (CSV) for layer-level latency
- Subprocess-based: ASTRA-SIM runs as a separate C++ process, communicating via pipes

---

## 2. main.py

**Lines**: ~546
**Purpose**: Entry point, argument parsing, simulation main loop

### Architecture

The main function is a monolithic ~500-line function that:
1. Parses CLI arguments (20+ parameters)
2. Builds cluster configuration
3. Initializes schedulers, router, controller
4. Starts ASTRA-SIM subprocess
5. Runs the simulation loop (read output -> schedule -> write workload)
6. Collects and prints metrics

### Bugs & Issues

1. **BUG (Severity: HIGH) - CXL metrics enumeration error (line 401)**:
   ```python
   for i, cxl_id, cxl_pool in enumerate(prefix_pools):
   ```
   `enumerate` returns `(index, value)` tuples, not `(index, id, pool)` triples. This will raise a `ValueError: not enough values to unpack`. The CXL metrics display is completely broken.

2. **BUG (Severity: MEDIUM) - Division by zero in throughput calculation (line 244)**:
   ```python
   RATIO = FREQ // INTERVAL
   ```
   If `log_interval` is 0, `INTERVAL` is 0 and this is a ZeroDivisionError. No validation on `log_interval > 0`.

3. **BUG (Severity: MEDIUM) - Division by zero in final metrics (lines 494-497)**:
   If `total_latency == 0` (simulation completes instantly or `current == 0`), the throughput calculations divide by zero.

4. **BUG (Severity: MEDIUM) - Prefix caching division by zero (line 505)**:
   `total_requested_tokens` could be 0 if no requests were processed, causing division by zero in hit ratio calculation.

5. **BUG (Severity: MEDIUM) - `add_reqeust` typo (line 223)**:
   `sched.add_reqeust(...)` - this is a typo. The actual method in Scheduler is `add_request`. This line would fail at runtime when no dataset is provided.

6. **BUG (Severity: LOW) - `import sys as flush` (line 20)**:
   Importing `sys` as `flush` is confusing. Used only for `flush.stdout.flush()` which could simply be `sys.stdout.flush()`.

7. **BUG (Severity: LOW) - `os.chdir(astra_sim)` (line 31)**:
   Changes the working directory globally. All relative paths in the codebase depend on this. If any error occurs before this line, subsequent path resolution fails. This is fragile and prevents running from arbitrary directories.

8. **DESIGN - Monolithic main function**: ~500 lines in a single function with no separation of concerns. Should be decomposed into initialization, simulation loop, and reporting phases.

### Edge Cases

- No validation that ASTRA-SIM binary exists before launching subprocess
- No timeout on subprocess communication (`read_wait` can hang forever)
- No signal handling for graceful shutdown (Ctrl+C leaves subprocess running)
- No validation that `num_req > 0`

---

## 3. controller.py

**Lines**: 58
**Purpose**: IPC bridge between Python simulator and ASTRA-SIM C++ subprocess

### Architecture

Manages stdin/stdout communication with ASTRA-SIM:
- `read_wait`: Blocks reading stdout until "Waiting" keyword appears
- `write_flush`: Writes commands to stdin
- `parse_output`: Extracts sys/id/cycle from structured output

### Bugs & Issues

1. **BUG (Severity: HIGH) - Potential infinite loop in `read_wait` (lines 14-21)**:
   If the subprocess crashes or produces unexpected output, the `while "Waiting" not in out[-1]` loop runs forever. No timeout mechanism. No EOF detection (`readline()` returns empty string on EOF).

2. **BUG (Severity: HIGH) - Potential infinite loop in `check_end` (lines 23-30)**:
   Same issue: no EOF detection, no timeout. If ASTRA-SIM crashes, `check_end` hangs forever.

3. **BUG (Severity: MEDIUM) - `parse_output` returns `None` implicitly (line 58)**:
   When the regex does not match, the function falls through without a return statement, returning `None`. Callers check `if out_dict != None` which works but is fragile.

4. **BUG (Severity: LOW) - `id` shadows built-in (line 44)**: `id` is used as a variable name.

### Design Patterns

- **Simple pipe-based IPC**: Effective for sequential simulation but cannot parallelize. Reasonable for the use case.
- **Regex parsing**: Appropriate for structured text output from ASTRA-SIM.

---

## 4. scheduler.py

**Lines**: 715
**Purpose**: Per-instance batch scheduler implementing vLLM/Orca-style continuous batching

### Architecture

Implements two scheduling modes:
- `schedule_base`: Standard scheduling without prefix caching
- `schedule_with_prefix`: Prefix-aware scheduling with radix tree integration

Key operations:
- Batch formation: selects requests by arrival time, respects max_batch and max_num_batched_tokens
- Memory pressure handling: evicts decode requests to CPU when NPU memory is insufficient
- Prefill prioritization: optional mode to prioritize prefill over decode
- KV cache management: block-based allocation with eviction/loading

### Algorithms & Data Structures

- **Batch scheduling**: First-come-first-served with memory-aware admission control
- **Eviction policy**: Evicts decode requests from the back of the batch (LRU-like by arrival order)
- **Request pool**: Sorted list by arrival time, merge-sorted when returning requests to pool
- **Merge sort**: Custom `_merge_by_arrival_id` with fast-path for non-overlapping ranges

### Bugs & Issues

1. **BUG (Severity: MEDIUM) - `schedule_base` inner loop variable reuse (lines 95-99, 125-129)**:
   The outer `for i in range(batch_len, -1, -1)` reuses variable `i` for iteration. On the first iteration where `kv_size` fits, `temp_len = i` is set. But this iterates from `batch_len` down to 0, meaning it finds the **largest** fitting batch, not the smallest. The logic is correct (binary search from top down) but the naming and lack of early `break` after finding temp_len (there IS a break) could be misleading.

2. **BUG (Severity: MEDIUM) - Request deletion is O(n^2) (lines 156-160)**:
   ```python
   for req in batch_req:
       for i, req_ in enumerate(self.request):
           if req_.id == req.id:
               del self.request[i]
               break
   ```
   For each batch request, it scans the entire request list. With large request pools, this is O(n*m) where n=pool size, m=batch size. Should use a set or dict for O(1) lookup.

3. **BUG (Severity: MEDIUM) - `add_done` batch_id mismatch (line 488)**:
   ```python
   id -= 1
   ```
   The iteration ID from ASTRA-SIM is decremented by 1 to match the batch_id. This implicit convention between ASTRA-SIM and the scheduler is fragile and undocumented.

4. **BUG (Severity: LOW) - `pool + self.request` does not maintain sort order (line 574)**:
   When `prioritize_prefill` is False, returning requests are prepended to the pool without sorting. This could violate the arrival-time ordering assumption.

5. **BUG (Severity: LOW) - `npu_num * 2` for prefill instances in `add_done` (line 508)**:
   The done check for prefill instances uses `self.start_npu + self.npu_num * 2 - 1`, which assumes prefill instances always use double the NPUs. This is set in `config_builder.py` but is a fragile implicit contract.

### Comparison with GenZ

GenZ models prefill and decode as separate phases with analytical formulas. LLMServingSim models them as continuous batched iterations with profile-driven latency lookup. The LLMServingSim approach is more faithful to real vLLM behavior but requires pre-profiled databases.

---

## 5. memory_model.py

**Lines**: ~752
**Purpose**: Per-instance multi-tier memory management (NPU, CPU, CXL) with block-based KV cache

### Architecture

Key components:
- **Weight calculation**: `get_weight()` computes per-rank model weight using layer-by-layer size calculation with tensor parallelism
- **KV cache management**: Block-based allocation with configurable block size
- **Multi-tier memory**: NPU (accelerator), CPU (host DRAM), CXL (pooled memory)
- **Prefix cache integration**: RadixCache per NPU with optional shared second-tier cache
- **Event-based accounting**: Tracks BlockStored/BlockRemoved events from RadixCache

### Algorithms

- **Weight calculation**: Sums per-layer weight sizes considering tensor parallelism sharding rules (ColumnParallel, RowParallel, VocabParallel)
- **KV cache sizing**: `get_kv(seq)` computes `2 * kv_dim * seq * n_layer * fp / npu_num` per-rank KV cache size
- **Block allocation**: `get_block_kv()` computes the incremental block allocation needed for a batch

### Bugs & Issues

1. **BUG (Severity: MEDIUM) - `lock_prefix` and `unlock_prefix` raise on valid states (lines 363-381)**:
   When `device == Device.NPU` but `req.npu_last_node` is None, the code falls through to the `else` clause and raises a RuntimeError. This can happen legitimately when a request has no prefix cache hit. The condition should check both device AND node existence separately.

2. **BUG (Severity: MEDIUM) - `evict_prefix_cache` logic error (line 430)**:
   ```python
   if not self.enable_prefix_caching and bytes <= 0:
       return
   ```
   This should use `or` not `and`. As written, it only returns early if BOTH conditions are true. If prefix caching is disabled but bytes > 0, it proceeds to call `self.npu_prefix_cache.evict()` which does not exist (never created when prefix caching is disabled).

3. **BUG (Severity: LOW) - `calculate_sizes` loads config on every call (line 545)**:
   `get_config(model)` reads and parses the JSON config file on every invocation. For trace generation, this function is called hundreds of times per batch. Should be cached (the trace_generator has its own caching via `_perf_db_cache` but `calculate_sizes` does not).

4. **BUG (Severity: LOW) - Warning string missing f-string (line 510)**:
   ```python
   self.logger.warning("NPU prefix cache remove unknown block hash {h}")
   ```
   Missing `f` prefix, so `{h}` is printed literally instead of the hash value.

### Design Patterns Worth Borrowing

- **Block-based KV cache management**: The block allocation strategy (with configurable block size) accurately models vLLM's PagedAttention memory management. BudSimulator could adopt this for more accurate memory estimation.
- **Multi-tier memory hierarchy**: NPU -> CPU -> CXL memory cascade with eviction/loading is a useful pattern for modeling heterogeneous memory systems.
- **Per-rank weight calculation with TP sharding**: The `calculate_sizes` function provides a comprehensive reference for how each layer type is sharded under tensor parallelism.

---

## 6. radix_tree.py

**Lines**: ~597
**Purpose**: Radix tree data structure for prefix caching (derived from SGLang)

### Architecture

Implements a radix tree where:
- Keys are token ID sequences
- Each node stores a prefix segment and reference count
- LRU eviction via `last_access_time` on nodes
- Thread-safe operations via `RLock`
- Event system: BlockStored/BlockRemoved events for memory accounting

### Algorithms

- **Prefix matching**: Walks the tree following common prefixes, splitting nodes when partial matches occur
- **Insertion**: Follows existing paths, splits on divergence, creates new leaf for remainder
- **Eviction**: Collects all leaves, uses heap-based LRU to evict unlocked leaves
- **Page alignment**: Key matching respects configurable page sizes

### Bugs & Issues

1. **BUG (Severity: MEDIUM) - `_record_store_event` parent hash logic is inverted (lines 523-525)**:
   ```python
   if node.parent is None or node != self.root_node:
       parent_block_hash = None
   ```
   This sets `parent_block_hash = None` when parent is None OR when node is not root. Since most inserted nodes are NOT the root, this condition is almost always True. The intent was likely `if node.parent is None or node.parent == self.root_node`.

2. **BUG (Severity: MEDIUM) - `_split_node` does not update hash values (lines 420-433)**:
   When a node is split, `new_node.hash_value` is never set. If the hash values were populated on the original node, they become incorrect after splitting.

3. **BUG (Severity: LOW) - `evicted` property references `self.value` (line 97)**:
   `TreeNode` does not have a `value` attribute. This property would raise `AttributeError` if ever called.

4. **BUG (Severity: LOW) - `hash()` for block events is non-deterministic (line 537)**:
   Python's `hash()` function is randomized across processes (PYTHONHASHSEED). Block hashes will differ between simulation runs, making debugging difficult.

5. **BUG (Severity: LOW) - Thread lock but no multi-threaded usage**:
   The RadixCache has an `RLock` but the simulator is single-threaded. The lock adds unnecessary overhead.

### Design Patterns Worth Borrowing

- **Radix tree for prefix caching**: This is the state-of-the-art data structure for KV cache prefix sharing (from SGLang). BudSimulator could use this pattern for modeling prefix caching effects.
- **Event-based memory accounting**: BlockStored/BlockRemoved events decouple the tree operations from memory tracking. Clean separation of concerns.
- **LRU eviction with lock_ref protection**: In-use prefixes are protected from eviction via reference counting.

---

## 7. power_model.py

**Lines**: ~214
**Purpose**: 7-component power model for energy estimation

### Architecture

Models power consumption across 7 components per node:
1. Base node power (constant)
2. NPU (idle, standby, active states)
3. CPU (idle + utilization-based active)
4. DRAM (idle + energy-per-bit data access)
5. Link (energy-per-bit for collective communication)
6. NIC (idle power)
7. Storage (idle power)

Energy is tracked as net energy (above idle baseline) accumulated over simulation time.

### Algorithms

- **Energy = Power * Time**: Standard physics-based model
- **NPU states**: idle -> standby (bounded duration) -> active
- **DRAM energy**: Based on data movement in bytes, converted via energy-per-bit
- **Total power**: `(base_energy + net_energy) / time_interval`

### Bugs & Issues

1. **BUG (Severity: MEDIUM) - `get_current_power` accumulates base_powers incorrectly (line 116)**:
   ```python
   total_energy = sum(net_node_energy.values()) + sum(self.base_powers[node_id].values()) * current_time_s
   ```
   Base powers are multiplied by absolute time from simulation start. This means each call to `get_current_power` computes the total energy from t=0, but `last_energies` stores the previous total. The delta computation `(total_energy - self.last_energies[node_id]) / dt` then gives the correct interval power. However, floating-point precision degrades as simulation time grows (catastrophic cancellation).

2. **BUG (Severity: LOW) - CPU utilization model is simplistic (line 80)**:
   ```python
   cpu_active_util = max(0.7 - self.power_configs[node_id]["cpu"]["util"], 0)
   ```
   Hardcodes 0.7 as max CPU utilization during NPU active time. This is a rough approximation.

3. **DESIGN ISSUE - No NIC or storage active power**: Only idle power is modeled for NIC and storage. Active power during data transfers is not captured.

### Comparison with GenZ

GenZ does not model power consumption. This is a significant differentiator for LLMServingSim 2.0. BudSimulator could adopt this approach for power-aware optimization.

---

## 8. graph_generator.py

**Lines**: 37
**Purpose**: Converts trace files to Chakra computation graphs for ASTRA-SIM

### Architecture

Invokes the Chakra graph converter as a subprocess to transform trace text files into ASTRA-SIM workload format.

### Bugs & Issues

1. **BUG (Severity: HIGH) - `subprocess.run` without error checking (line 35)**:
   ```python
   subprocess.run(cmd, text=True)
   ```
   No `check=True`, no return code inspection. If graph generation fails, the simulation continues with stale/missing workload files.

2. **BUG (Severity: MEDIUM) - `cmd.split()` on string with spaces (line 34)**:
   If any path contains spaces, `cmd.split()` would break arguments incorrectly. Should use a list from the start.

3. **BUG (Severity: MEDIUM) - `os.chdir` is not thread-safe (lines 13, 36)**:
   Changes global CWD, then restores. If any other code runs concurrently, CWD is corrupted.

4. **DESIGN - External subprocess dependency**: Requires Chakra converter to be installed and available. No fallback if missing.

---

## 9. router.py

**Lines**: 93
**Purpose**: Routes incoming requests to instances using configurable policies

### Architecture

- **Round-robin (RR)**: Deterministic sequential distribution
- **Random (RAND)**: Seeded random routing
- **Custom**: Extension point (raises NotImplementedError)
- **Prefill/Decode separation**: Maintains separate counters for prefill and decode routing

### Bugs & Issues

1. **BUG (Severity: LOW) - `output_length` includes `input_toks` (line 62)**:
   ```python
   output_length = int(row['input_toks'] + row['output_toks'])
   ```
   This sets `output_length` to the sum of input + output tokens, which represents the total sequence length, not the output length. This is then passed to `Request.__init__` as the `output` parameter, which is used as the termination condition (`req.output <= req.input`). This appears intentional (total sequence position) but naming is confusing.

2. **DESIGN - No load-aware routing**: The router does not consider instance load or memory pressure. A shortest-queue or memory-aware policy would improve scheduling quality.

---

## 10. request.py

**Lines**: 85
**Purpose**: Request and Batch data classes

### Architecture

- **Request**: Tracks lifecycle from arrival through prefill, decode iterations, to completion
- **Batch**: Groups requests for a single iteration of the model

### Bugs & Issues

1. **BUG (Severity: LOW) - `del self.original_input` in `add_latency` (lines 46-48)**:
   Deletes attributes from the instance after completion. While this saves memory, it makes the object invalid for any subsequent access. Could cause `AttributeError` in debugging or reporting.

2. **DESIGN - No dataclass or slots**: Plain class with many attributes. Using `__slots__` would reduce memory overhead for large numbers of requests.

---

## 11. trace_generator.py

**Lines**: ~1200+ (very large)
**Purpose**: Generates ASTRA-SIM execution traces for each batch

### Architecture

The core of the performance modeling. For each batch iteration:
1. Looks up layer-level latency from pre-profiled performance database (CSV)
2. Computes input/weight/output tensor sizes using `calculate_sizes`
3. Generates a trace file with per-layer compute time, memory locations, and communication
4. Supports attention prediction via sklearn models
5. Handles MoE expert routing and PIM offloading

### Key Design Patterns

- **Performance database caching**: `_perf_db_cache` dict keyed by `(hardware, model, tp)` avoids re-reading CSV files
- **Attention prediction**: Uses sklearn RandomForest models trained on profiling data for attention latency prediction
- **Sub-batch interleaving**: Splits prefill and decode into separate sub-batches for better resource utilization

### Bugs & Issues

1. **BUG (Severity: HIGH) - Hardcoded KV prepare overhead (lines 350-352)**:
   ```python
   if pd_type == "prefill":
       o_proj_tp_pd_kv_prepare = 700_000
   if pd_type == "decode":
       o_proj_tp_pd_kv_prepare = 700 * decode_key[0]
   ```
   Magic numbers 700,000 and 700 without any documentation or derivation. These appear to be empirically-tuned constants that would not generalize across hardware.

2. **BUG (Severity: MEDIUM) - `_get_perf_row` lookup may fail silently**: If the performance database does not have an entry for the exact (input_len, kv_len) combination, the nearest-neighbor lookup may return an inappropriate row.

3. **BUG (Severity: MEDIUM) - sklearn/joblib/pickle imports at top level**: These are heavyweight dependencies imported unconditionally, even when attention prediction is not used.

4. **DESIGN - File I/O on every batch**: Trace files are written and immediately re-read (lines 84-88). This unnecessary I/O could be replaced with in-memory data passing.

### Comparison with GenZ

GenZ uses analytical roofline models for compute time. LLMServingSim uses profiled latency databases, which are more accurate for specific hardware but require profiling effort for each new hardware platform. The GenZ approach is more portable; the LLMServingSim approach is more accurate.

---

## 12. config_builder.py

**Lines**: 558
**Purpose**: Parses cluster JSON, builds ASTRA-SIM config files

### Architecture

Comprehensive configuration builder that:
- Validates cluster config (nodes, instances, NPU/CPU/CXL memory, power)
- Generates ASTRA-SIM network topology YAML
- Generates memory expansion JSON
- Validates memory placement against config
- Supports flexible placement rules (per-block, per-layer overrides)

### Bugs & Issues

1. **BUG (Severity: HIGH) - `total_npu` and `total_npu_group` calculated inside per-node loop (lines 367-368)**:
   These calculations are inside the `for node_config in nodes:` loop but use `total_instances` which grows each iteration. On multi-node configurations, intermediate values would be wrong. Only the final iteration produces correct values, which happens to work because `cluster` is returned after the loop.

2. **BUG (Severity: MEDIUM) - `exit(1)` on JSON decode error (line 31)**:
   Uses `exit(1)` instead of raising an exception. This cannot be caught by callers.

3. **BUG (Severity: MEDIUM) - Network config and memory config are overwritten per node (lines 231-240)**:
   `cpu_mem_enabled` flag means only the first node's CPU memory config is used. If nodes have different CPU memory configurations, only the first is applied.

4. **DESIGN - Writes config files to disk**: The builder writes YAML/JSON files that ASTRA-SIM reads. This disk-based config passing is fragile (race conditions, stale files).

### Design Patterns Worth Borrowing

- **Hierarchical placement rules**: The default -> block -> layer priority system for memory placement is elegant and flexible. BudSimulator could adopt this for heterogeneous memory modeling.
- **Comprehensive validation**: Validates all required keys at multiple levels with clear error messages.
- **Block expression parsing**: `_parse_blocks_expr("0-3,5,7-9")` is a useful utility for layer-range specifications.

---

## 13. utils.py & logger.py

### utils.py (181 lines)

Provides utility functions: workload path generation, trace formatting, model config loading, ANSI color helpers, logo printing.

**Issue**: `get_config()` reads JSON from disk on every call without caching. Used extensively throughout the codebase.

### logger.py (255 lines)

Well-designed logging infrastructure:
- Custom formatter with timestamp, component, node/instance context
- ANSI color-coded log levels
- `ComponentLoggerAdapter` injects component metadata
- Lazy configuration pattern (configure once, skip on subsequent calls)

**Design Pattern Worth Borrowing**: The `ComponentLoggerAdapter` pattern for injecting node/instance context into log messages is excellent for multi-instance simulators.

---

## 14. attn_utils.py

**Lines**: 290
**Purpose**: FlashAttention-2 split heuristics and attention metadata computation

### Architecture

Ports the FA2 `num_splits_heuristic` from C++ to Python, computing:
- Tiling parameters (block_m, block_n, num_m_blocks, num_n_blocks)
- SM occupancy and wave efficiency
- Padding ratios and waste analysis
- Per-request QK statistics

### Design Patterns Worth Borrowing

- **FA2 split heuristic**: The `_num_splits_heuristic` function is a faithful port of FlashAttention's internal scheduling logic. This is valuable for predicting attention kernel performance analytically.
- **Attention metadata**: The comprehensive metadata collection (statistics, tiling, efficiency) could be useful for BudSimulator's attention modeling.

---

## 15. gate_function.py

**Lines**: 59
**Purpose**: MoE expert routing simulation

### Architecture

Simulates the gating function for Mixture-of-Experts models:
- Round-robin routing: Deterministic expert assignment
- Random routing: Seeded random expert selection
- Custom: Extension point

Returns per-expert token counts for load balancing analysis.

**Clean and simple implementation.** No significant bugs.

---

## 16. pim_model.py

**Lines**: 149
**Purpose**: Processing-in-Memory latency model for attention offloading

### Architecture

Models PIM-accelerated DRAM for attention computation:
- Reads hardware specifications from `.ini` config files
- Computes memory channel bandwidth and capacity
- Estimates attention latency using linear regression models (hardware-specific)

### Bugs & Issues

1. **BUG (Severity: HIGH) - PIM latency model only supports Llama-3.1-8B (line 126)**:
   ```python
   if n_head != 32 or kv_head != 8 or head_dim != 128:
       raise NotImplementedError("Only Llama3.1-8B ...")
   ```
   The linear regression coefficients are hardcoded for one specific model architecture. Using any other model with PIM raises an error.

2. **BUG (Severity: MEDIUM) - Limited hardware support (line 128-148)**:
   Only 4 DRAM technologies have regression coefficients. Others raise a KeyError.

---

## 17. llm_profile/

### 17.1 models/llama.py (541 lines)

Timer-instrumented fork of HuggingFace `LlamaForCausalLM`. Each linear layer is wrapped in a `Timer` context manager for per-operation latency measurement.

**Architecture**:
- `LlamaAttention`: Timer-wrapped q_proj, k_proj, v_proj, rope, o_proj
- `LlamaDecoderLayer`: Timer-wrapped input_layernorm, post_layernorm
- `LlamaModel`: Timer-wrapped embedding, final_layernorm
- `LlamaForCausalLM`: Timer-wrapped lm_head

**TP-Aware Modifications**:
- All projection dimensions divided by `config.tp_size` (e.g. `nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim // config.tp_size)`)
- Embedding and lm_head vocab dimension divided by `config.tp_size`
- This models the per-rank compute cost for tensor-parallel sharding

**Key observation**: These are NOT simulation code. They are profiling tools that run on actual hardware to generate the CSV databases that `trace_generator.py` looks up at simulation time.

### 17.2 models/mixtral.py (743 lines)

Timer-instrumented `MixtralForCausalLM` with MoE-specific instrumentation:

**MoE-specific Architecture**:
- `MixtralBlockSparseTop2MLP`: Timer-wrapped expert.w1, expert.w2, expert.w3, act_fn
- `MixtralSparseMoeBlock`: Timer-wrapped gate operation
  - **IMPORTANT simplification**: `self.experts` is a SINGLE `MixtralBlockSparseTop2MLP` instance, not a ModuleList. The profiler routes ALL tokens through one expert to measure per-expert compute cost. The commented-out code shows the original per-expert routing loop.
  - `collect_router_stats`: Optional routing statistics logging via `torch.bincount` on selected experts

**Bugs Found**:
- **BUG (Line 694)**: `self.lm_head_timer` should be `self._lm_head_timer` (missing underscore prefix). The `_lm_head_timer` attribute is defined at line 633 with leading underscore but referenced without it. **This will raise AttributeError during profiling's forward pass**.

**Attention Module** (MixtralAttention):
- Uses `eager_attention_forward` with `repeat_kv` for GQA (Group Query Attention)
- Sliding window support via `getattr(self.config, "sliding_window", None)`
- RoPE via `MixtralRotaryEmbedding` with dynamic rope update support

**Load Balancing Loss**: Standard Switch Transformer auxiliary loss implementation with attention mask handling.

### 17.3 models/phimoe.py (~1417 lines)

Timer-instrumented `PhimoeForCausalLM` -- the largest model file. Supports Microsoft Phi-MoE architecture.

**Key Differences from Mixtral**:
- Uses `sparsemixer()` routing instead of standard top-k: Implements Heun's third-order gradient approximation method with Gumbel sampling for expert selection (from paper 2409.12136)
- `MultiplierProcessor`: Custom autograd Function for backward pass through expert routing
- Uses `LayerNorm` (with bias) instead of `RMSNorm` (matches Phi architecture)
- `PhimoeRotaryEmbedding`: Supports `short_mscale` and `long_mscale` for extended context
- Three attention implementations: Eager, FlashAttention2, SDPA

**MoE Block** (PhimoeSparseMoeBlock):
- Same single-expert profiling simplification as Mixtral (commented-out per-expert loop)
- `PhimoeBlockSparseTop2MLP`: Same w1/w2/w3 structure with Timer wrappers
- Both `input_jitter_noise` and `router_jitter_noise` parameters (Mixtral only has router jitter)

**Bugs Found**:
- `_update_causal_mask` handles FlashAttention2, FlexAttention, and SDPA separately but the FlexAttention path imports `BlockMask` only conditionally -- **if `is_torch_flex_attn_available()` returns False but config specifies flex_attention, it will crash with NameError**

### 17.4 profiler/utils/__init__.py

Defines `ProfileMethod` enum with 4 supported methods:
- `CUDA_EVENT`: torch CUDA events for timing
- `KINETO`: Torch profiler (Kineto backend)
- `PERF_COUNTER`: Python `time.perf_counter()` with explicit sync
- `RECORD_FUNCTION`: torch `record_function` with trace export

**Bug Found**:
- **`validate_tp_size` returns None for valid cases** (Line 14-23): The function checks invalid conditions and returns `False`, but never returns `True` for valid TP sizes. Missing `return True` at end. **All callers check `if validate_tp_size(tp_size, num_heads):` -- this means valid TP sizes evaluate as falsy (None is falsy) so the logic accidentally works, but semantics are inverted**: callers skip when the function returns truthy, so they skip on `False` (invalid) and proceed on `None` (valid). Fragile and confusing.

### 17.5 profiler/common/timer.py (98 lines)

Multi-method profiling timer implementing the context manager protocol.

**Architecture**:
- `__enter__`: Dispatches to appropriate profiling method (record_function, cuda_event, kineto, perf_counter)
- `__exit__`: Records timing data into the singleton `TimerStatsStore`
- `disabled` flag: Skips timing if name is None or store is disabled

**Profiling Methods**:
| Method | Precision | Overhead | Notes |
|--------|-----------|----------|-------|
| CUDA_EVENT | GPU clock | Low | Records start/end events, computes elapsed on `get_stats()` |
| RECORD_FUNCTION | GPU trace | Medium | Uses torch profiler trace export + offline parsing |
| KINETO | GPU trace | High | Full Kineto profiler with `on_trace_ready` callback |
| PERF_COUNTER | Wall clock | Low | `torch.cuda.synchronize()` before/after |

**Design Pattern Worth Borrowing**: Clean abstraction over multiple profiling backends behind a single Timer interface.

### 17.6 profiler/common/timer_stats_store.py (39 lines)

**Singleton** stats collection store (uses `Singleton` metaclass).

- `record_time(name, time)`: Appends raw timing data (float for perf_counter, event-pair for CUDA_EVENT)
- `get_stats()`: Computes min/max/mean/median/std for each named operation
- `clear_stats()`: Resets all accumulated data

**Bug Found**:
- **Singleton with constructor args is broken for re-initialization** (Line 8-11): The `Singleton` metaclass ignores subsequent constructor arguments after first instantiation. If first call is `TimerStatsStore(profile_method="cuda_event")` and second is `TimerStatsStore(profile_method="record_function")`, the second silently returns the instance configured for cuda_event. **This is exploited intentionally** -- the first call in `run_profile()` or `profile_flash_attention()` sets the mode, and subsequent `Timer.__init__` calls get the same instance. But it means you cannot change profiling method mid-session without manual `Singleton._instances.clear()`.

### 17.7 profiler/utils/singleton.py (14 lines)

Standard Singleton metaclass (from StackOverflow). Stores instances in class-level `_instances` dict keyed by class. Thread-unsafe (no locking).

### 17.8 profiler/utils/logger.py (22 lines)

Simple ANSI-colored logger with 4 functions: `log_info` (blue), `log_success` (green), `log_warning` (yellow), `log_error` (red). Uses `print()` directly -- no stdlib logging integration.

### 17.9 profiler/utils/record_function_tracer.py (104 lines)

Chrome Trace format parser for extracting CUDA kernel timings from torch profiler output.

**Algorithm**:
1. `__enter__`: Starts `torch.profiler.profile` with CPU+CUDA activities
2. `__exit__`: Exports Chrome trace JSON to disk
3. `get_operation_time_stats()`: Parses JSON, finds `user_annotation` events (Timer-marked), walks children to find `cuda_runtime` events, correlates to actual CUDA kernels via `correlation` ID, accumulates kernel durations
4. `clean_up()`: Deletes the trace file

**Complexity Concern**: `find_children()` is O(n) per event, and `find_correlated_event()` is O(n) per child. For large traces, `get_operation_time_stats()` is O(n^3) worst case. A hash-map-based approach would be O(n).

**Bug Found**:
- **Line 66**: `json.load(open(self.trace_path, "r"))` -- file handle leak. Should use `with open(...)`.
- **Line 83**: `event["name"].replace("vidur_", "")` -- leftover from Vidur/Microsoft codebase. Timer names in this project don't have "vidur_" prefix, so this is a dead code path that works by accident.

### 17.10 profiler/attention/attention_profiler.py (236 lines)

Profiles FlashAttention varlen kernel (`flash_attn_varlen_func`) for building attention latency databases.

**Algorithm**:
1. Compute per-TP-shard head counts: `num_heads_per_shard = num_attention_heads // tp_size`
2. Build dummy Q/K/V tensors with `_build_varlen_qkv()` -- uniform token distribution across batch
3. Warmup (10 iterations by default)
4. Profile (30 iterations) with either `RecordFunctionTracer` or `TimerStatsStore`
5. Return timing stats + model metadata dict

**Bug Found**:
- **`_build_varlen_qkv` integer division loses tokens** (Line 51): `Lq_list = [q_len // batch_size] * batch_size`. If `q_len` is not divisible by `batch_size`, the remainder tokens are silently dropped. For example, q_len=10, batch_size=3 gives `[3, 3, 3]` = 9 tokens but Q tensor has 10 tokens. The `cu_seqlens_q[-1]` will be 9, not 10, leading to `flash_attn_varlen_func` ignoring the last token. Should distribute remainder: `Lq_list[i] += 1 for i < q_len % batch_size`.
- **OOM handling silently returns None** (Line 229-235): The outer profiling loop in `main.py` appends None results and filters later, but this means partial profiling data when GPU memory is tight.

### 17.11 profiler/attention/attention_input.py (43 lines)

Dataclass-like (plain class) for attention profiling input parameters. Derived from Microsoft Vidur.

**Validation rules**:
- Prefill: `batch_size == 1`, `prefill_chunk_size > 0`, `kv_cache_size > 0`, total fits within `max_seq_len`
- Decode: `prefill_chunk_size == 0`, `kv_cache_size > 0`, fits within `max_model_len`

**Bug Found**:
- **Prefill validation requires `kv_cache_size > 0`** (Line 26): This rejects the first prefill chunk where there is no KV cache yet (kv_cache_size=0). The `batch_sampling.py` generates combinations with `kv_cache_size=0` for "full prefills" (Line 104: `product(prefill_lengths_to_profile, [0], [1], [True])`), but these will be filtered out by `is_valid()`. This means **full first-prefill latency is never profiled**, only chunked prefill with existing KV cache.

### 17.12 profiler/attention/batch_sampling.py (165 lines)

Generates the combinatorial space of attention configurations to profile. Derived from Microsoft Vidur.

**Token Space Generation** (logarithmically increasing density):
- 1-7: individual values {1, 2, 4}
- 8-1023: step 8
- 1024-2048: step 16
- 2048-4096: step 32
- ... up to 64K-128K: step 1024

**get_attention_input_combinations()**: Generates three categories:
1. Chunked prefills: All (chunk_size, kv_cache_partition) combinations
2. Full prefills: (seq_length, kv_cache=0)
3. Decodes: (batch_size, kv_cache_size) Cartesian product

**get_max_num_blocks()**: Calculates maximum KV cache blocks that fit in GPU memory (after model weights). Does NOT subtract activation memory -- only considers raw GPU free memory * utilization / block_size. This overestimates available blocks for large models.

### 17.13 profiler/attention/main.py (139 lines)

CLI entry point for attention profiling.

**Workflow**:
1. Parse args (model, hardware, tp-size, max-len, batch range)
2. For each TP size: generate input combinations, filter by memory limit, profile each, save to CSV

**Bug Found**:
- **validate_tp_size inversion** (Line 69-70): `if validate_tp_size(tp_size, ...): log_warning("Skipping invalid"); continue`. Since `validate_tp_size` returns `None` for valid sizes and `False` for invalid sizes, both are falsy, so the skip is never triggered. Valid and invalid TP sizes both proceed. **Same bug in layers/main.py (Line 298) and predictor/main.py (Line 51)**.

### 17.14 profiler/layers/main.py (345 lines)

CLI entry point for non-attention layer profiling.

**Workflow**:
1. Load model config, instantiate Timer-instrumented model (Llama/Mixtral/Phimoe)
2. For each TP size and input length: run warmup + profiling iterations
3. Collect per-layer timing stats, compute per-block and full-model latency estimates
4. Save to CSV

**Bugs Found**:
- **`parse_args()` missing `return` statement** (Line 57): The function defines the parser and arguments but never calls `return parser.parse_args()`. **This will crash at runtime** since `main()` calls `args = parse_args()` and expects a result.
- **`raise log_warning(...)` on Line 177**: In decode mode, executes `raise log_warning(f"This deprecated...")`. `log_warning` returns `None`, so this raises `TypeError: exceptions must derive from BaseException`. **Decode profiling is completely broken**.
- **MoE expert computation scaling** (Lines 264-267): Computes `n_tok = max(input_len // config.num_local_experts // tp_size, 1)` then multiplies by `config.num_local_experts // tp_size`. This assumes equal token distribution across experts AND TP-sharded expert count, but top-k routing with k=2 means each token activates 2 experts, not 1. The per-expert token count should be `input_len * top_k / num_experts` (approximately).

### 17.15 profiler/predictor/main.py (100 lines)

CLI entry point for building sklearn-based attention latency predictors.

**Workflow**:
1. Load profiled attention CSV data
2. Split into prefill/decode subsets
3. Train RandomForest regressors with GridSearchCV (MAPE scoring)
4. Generate prediction grids and save to CSV

**TPU Support**: Special handling for TPU-profiled data (uses "p50_ns" column instead of "time_stats.attn_*.median").

### 17.16 profiler/predictor/build_sklearn_predictor_and_pred.py (71 lines)

Core ML model training and prediction.

**Algorithm**:
1. `train_model()`: GridSearchCV over RandomForest hyperparameters (n_estimators, max_depth, min_samples_split) with MAPE scoring. **Caches trained model to pickle file**.
2. `predict_and_save()`: Generates predictions on full grid, scales by overhead factor, saves CSV
3. `load_and_split_attention_csv()`: Loads profiled data, fills NaN kv_cache_save with 0, splits by is_decode flag
4. `build_grids()`: Creates Cartesian product grids for prefill (kv_cache_size x chunk_size) and decode (batch_size x kv_cache_size)

**Design Notes**:
- Model features: Prefill uses (kv_cache_size, prefill_chunk_size), Decode uses (batch_size, kv_cache_size)
- Scaling: `preds * 1e6 * overhead` converts from ms to ns with overhead multiplier
- No train/test split validation: Only cross-validation during GridSearch, no holdout evaluation

**Bug Found**:
- **Line 51**: `df["prefill_chunk_size"] = df["prefill_chunk_size"]` -- no-op assignment, likely leftover from debugging.

**Design Pattern Worth Borrowing**: The full profiling pipeline (hardware profiling -> ML model training -> prediction lookup table) is a powerful approach for building hardware-specific performance models. BudSimulator could complement its analytical roofline model with this empirical calibration approach.

---

## 18. Configuration Files

### 18.1 cluster_config/ (12 JSON files)

Each config defines a cluster topology: `{num_nodes, link_bw, link_latency, nodes[]}`. Nodes contain `{num_instances, cpu_mem, instances[]}`. Instances specify `{model_name, hardware, npu_mem, npu_num, npu_group, pd_type}`.

| Config File | Topology | Hardware | Model | Special Features |
|-------------|----------|----------|-------|------------------|
| single_node_single_instance.json | 1N/1I | A6000 | Llama-3.1-8B | Baseline config |
| single_node_multi_instance.json | 1N/2I | A6000 | Llama-3.1-8B | Multi-instance routing |
| dual_node_multi_instance.json | 2N/4I | A6000 | Llama-3.1-8B | Multi-node cluster |
| single_node_pd_instance.json | 1N/2I | A6000 | Llama-3.1-8B | P/D disaggregation |
| single_node_cxl_instance.json | 1N/1I | A6000 | Llama-3.1-8B | CXL memory expansion |
| single_node_memory_instance.json | 1N/1I | A6000 | Llama-3.1-8B | Per-layer memory placement |
| single_node_pim_instance.json | 1N/1I | A6000 | Llama-3.1-8B | PIM + power model |
| single_node_power_instance.json | 1N/1I | A6000 | Llama-3.1-8B | Power model only |
| single_node_single_instance_H100.json | 1N/1I | H100x4 | Llama-3.1-70B | High-end GPU cluster |
| single_node_moe_single_instance.json | 1N/1I | A6000 | Phi-mini-MoE | MoE single instance |
| single_node_moe_multi_instance.json | 1N/2I | A6000 | Phi-mini-MoE | MoE multi-instance |
| single_node_moe_pd_instance.json | 1N/2I | A6000 | Phi-mini-MoE | MoE + P/D disaggregation |

**Memory Placement System** (in cxl_instance.json and memory_instance.json):
```json
"placement": {
    "default": {"weights": "cxl:0", "kv_loc": "npu", "kv_evict_loc": "cpu"},
    "blocks": [
        {"blocks": "0-3", "weights": "cxl:0", ...},
        {"blocks": "4-7", "weights": "cxl:1", ...}
    ],
    "layers": {
        "embedding": {"weights": "cxl:1", ...},
        "lm_head": {"weights": "cxl:3", ...}
    }
}
```

**CXL Memory Config**:
```json
"cxl_mem": {"mem_size": 1024, "mem_latency": 250, "mem_bw": 60, "num_devices": 4}
```

**Power Config** (7-component, in pim_instance.json and power_instance.json):
- `base_node_power`: 60W constant
- `npu.A6000`: idle=25W, standby=115W, active=300W, standby_duration=18ms
- `cpu`: idle=10W, active=200W, util=0.15
- `dram`: dimm_size=32GB, idle=2.0W, energy_per_bit=6.0pJ
- `link`: num_links=1, idle=5W, energy_per_bit=4.0pJ
- `nic`: num_nics=1, idle=20W
- `storage`: num_devices=2, idle=5W

**PIM Config**: `"pim_config": "DDR4_8GB_3200_pim"` (references pim_model.py configurations)

**H100 Config Differences**:
- link_bw: 900 GB/s (NVLink, vs 112 for A6000 PCIe)
- npu_mem: 80GB, 3350 GB/s (vs 40GB, 768 GB/s for A6000)
- cpu_mem: 1024GB (vs 128GB)
- npu_num: 4 (tensor parallel)

**Observations**:
- All A6000 configs use link_bw=112 (PCIe 4.0 x16)
- H100 config uses link_bw=900 (NVLink)
- All configs set link_latency=0 and mem_latency=0 (except CXL: 250ns)
- No configs demonstrate pipeline parallelism (all `npu_group: 1`)

### 18.2 model_config/ (4 JSON files)

| Model | Type | Hidden | Intermediate | Heads | KV Heads | Layers | Vocab | Experts |
|-------|------|--------|-------------|-------|----------|--------|-------|---------|
| Llama-3.1-8B | llama | 4096 | 14336 | 32 | 8 | 32 | 128256 | - |
| Llama-3.1-70B | llama | 8192 | 28672 | 64 | 8 | 80 | 128256 | - |
| Mixtral-8x7B-v0.1 | mixtral | 4096 | 14336 | 32 | 8 | 32 | 32000 | 8 (top-2) |
| Phi-mini-MoE | phimoe | 4096 | 960 | 32 | 8 | 32 | 32064 | 16 (top-2) |

**Note**: These are minimal subsets of HuggingFace configs containing only fields needed by the simulator. The profiler code loads full configs via `AutoConfig.from_pretrained()` while the simulator uses these stripped-down versions via `utils.get_config()`.

---

## 19. Comparison with GenZ/BudSimulator Framework

| Aspect | LLMServingSim 2.0 | GenZ / BudSimulator |
|--------|-------------------|-------------------|
| **Performance modeling** | Profile-driven (CSV lookup) | Analytical (roofline model) |
| **Accuracy** | High (real profiled data) | Medium (theoretical bounds) |
| **Portability** | Low (needs profiling per HW) | High (hardware params only) |
| **Scheduling** | Full vLLM-style continuous batching | Simplified batch modeling |
| **Memory model** | Block-based PagedAttention with prefix caching | Aggregate KV cache estimation |
| **Network simulation** | ASTRA-SIM (cycle-accurate) | Analytical collective models |
| **Power modeling** | 7-component detailed model | Not supported |
| **PIM support** | Yes (attention offloading) | Not supported |
| **P/D disaggregation** | Yes (prefill/decode split) | Not supported |
| **MoE support** | Yes (expert routing, gating) | Limited |
| **Prefix caching** | Full radix tree implementation | Not supported |
| **Simulation speed** | Slow (subprocess + file I/O) | Fast (in-process analytical) |
| **Setup complexity** | High (ASTRA-SIM build, profiling) | Low (pip install) |

---

## 20. Design Patterns Worth Borrowing

### High Priority

1. **Block-based KV cache with PagedAttention semantics**: The `MemoryModel.get_block_kv()` and `get_evict_kv()` methods accurately model vLLM's memory management. BudSimulator should adopt block-based KV allocation.

2. **Radix tree prefix caching**: The RadixCache implementation with LRU eviction and lock_ref protection provides an accurate model of prefix caching behavior. Critical for modern LLM serving estimation.

3. **7-component power model**: Novel feature not found in GenZ. The component-based energy tracking with idle/active/standby states could be integrated into BudSimulator for power-aware analysis.

4. **Prefill/Decode disaggregation**: The P/D split architecture with inter-instance request transfer models a growing deployment pattern. BudSimulator should support this topology.

5. **Per-layer weight calculation with TP sharding**: The `calculate_sizes()` function provides a comprehensive reference for tensor parallelism weight partitioning. More detailed than GenZ's aggregate approach.

### Medium Priority

6. **Continuous batching simulation**: The Scheduler's iteration-by-iteration approach models real vLLM behavior more faithfully than GenZ's aggregate estimates.

7. **FlashAttention split heuristics**: The FA2 num_splits_heuristic port enables analytical attention performance prediction.

8. **MoE expert routing policies**: The GateRouter with configurable policies (RR, RAND, FAST) models load balancing effects.

9. **Multi-tier memory cascade**: NPU -> CPU -> CXL memory hierarchy with eviction/loading models heterogeneous memory architectures.

### Low Priority

10. **Component logging with node/instance context**: The logger infrastructure provides excellent observability for multi-instance simulations.

11. **Hierarchical placement rules**: Default -> block -> layer memory placement overrides provide fine-grained control.

---

## 21. Summary of All Findings

### Critical/High-Severity Bugs (12)

| # | File | Description |
|---|------|-------------|
| 1 | main.py | CXL metrics enumerate unpacking error (3 values from 2-tuple) |
| 2 | controller.py | `read_wait` and `check_end` can hang forever (no EOF/timeout) |
| 3 | graph_generator.py | `subprocess.run` without error checking - silent failures |
| 4 | config_builder.py | `total_npu` calculated inside per-node loop with accumulating list |
| 5 | pim_model.py | PIM latency model hardcoded for Llama-3.1-8B only |
| 6 | trace_generator.py | Hardcoded magic numbers for KV prepare overhead (700,000 ns) |
| 7 | main.py | `add_reqeust` method name typo (won't work without dataset) |
| 8 | models/mixtral.py:694 | `self.lm_head_timer` should be `self._lm_head_timer` -- AttributeError on MixtralForCausalLM forward |
| 9 | profiler/layers/main.py:57 | `parse_args()` missing `return` statement -- crashes at runtime |
| 10 | profiler/layers/main.py:177 | `raise log_warning(...)` raises TypeError instead of warning -- decode profiling completely broken |
| 11 | profiler/utils/__init__.py:14-23 | `validate_tp_size()` never returns True -- accidentally works because None is falsy, but semantics inverted |
| 12 | profiler/attention/attention_input.py:26 | Prefill validation requires kv_cache_size > 0, rejecting first-prefill profiling |

### Medium-Severity Bugs (18)

| # | File | Description |
|---|------|-------------|
| 1 | main.py | Division by zero in throughput/prefix metrics when values are 0 |
| 2 | scheduler.py | O(n^2) request deletion from pool |
| 3 | scheduler.py | Implicit batch_id-1 convention with ASTRA-SIM |
| 4 | scheduler.py | Unsorted request return when prioritize_prefill is off |
| 5 | memory_model.py | `lock_prefix`/`unlock_prefix` raise on valid None node states |
| 6 | memory_model.py | `evict_prefix_cache` logic error: `and` should be `or` |
| 7 | radix_tree.py | `_record_store_event` parent hash logic inverted |
| 8 | radix_tree.py | `_split_node` does not update hash values |
| 9 | graph_generator.py | `cmd.split()` breaks on paths with spaces |
| 10 | config_builder.py | `exit(1)` instead of exception on JSON error |
| 11 | config_builder.py | Only first node's CPU memory config used |
| 12 | pim_model.py | Limited hardware support (4 DRAM technologies) |
| 13 | profiler/attention/attention_profiler.py:51 | Integer division in `_build_varlen_qkv` drops remainder tokens |
| 14 | profiler/utils/record_function_tracer.py:66 | File handle leak: `json.load(open(...))` without `with` |
| 15 | profiler/utils/record_function_tracer.py:83 | Dead `"vidur_"` prefix strip from Vidur codebase |
| 16 | profiler/utils/record_function_tracer.py | O(n^3) trace parsing in `get_operation_time_stats()` |
| 17 | profiler/predictor/build_sklearn_predictor_and_pred.py:51 | No-op assignment `df["prefill_chunk_size"] = df["prefill_chunk_size"]` |
| 18 | profiler/layers/main.py:264-267 | MoE expert computation scaling assumes uniform distribution, ignores top-k=2 routing |

### Design/Architecture Issues (10)

| # | Description |
|---|-------------|
| 1 | Monolithic 500-line main function with no separation of concerns |
| 2 | Global `os.chdir()` makes path resolution fragile |
| 3 | File I/O on every batch (write trace, read back) |
| 4 | Disk-based config passing between Python and ASTRA-SIM |
| 5 | Wildcard imports (`from .module import *`) throughout |
| 6 | `get_config()` reads JSON from disk without caching |
| 7 | No signal handling for graceful subprocess cleanup |
| 8 | No validation of ASTRA-SIM binary existence before launch |
| 9 | Singleton pattern for TimerStatsStore prevents changing profiling method mid-session |
| 10 | Profiler batch_sampling.py `get_max_num_blocks()` does not subtract activation memory from available GPU memory |

### Code Quality Issues (8)

| # | Description |
|---|-------------|
| 1 | `import sys as flush` - confusing alias |
| 2 | `id` used as variable name shadowing built-in (multiple files) |
| 3 | Missing f-string prefix in warning log (memory_model.py:510) |
| 4 | Unused thread lock in single-threaded radix tree |
| 5 | `TreeNode.evicted` property references non-existent `self.value` |
| 6 | Comment-only debugging code left in production (`# print(...)`) |
| 7 | MoE models use single expert instance instead of ModuleList for profiling (intentional simplification but commented-out code left in) |
| 8 | Phimoe model is ~1400 lines with legacy attention implementations (Eager, Flash, SDPA) that could be unified |

### Key Strengths

1. **Comprehensive LLM serving simulation**: Models all major vLLM features (continuous batching, PagedAttention, prefix caching, P/D disaggregation, MoE)
2. **Multi-tier memory hierarchy**: NPU/CPU/CXL with block-based management
3. **7-component power modeling**: Unique differentiator vs. other simulators
4. **PIM support**: Processing-in-Memory for attention offloading
5. **Profile-driven accuracy**: Real hardware profiling data for latency estimation
6. **ASTRA-SIM integration**: Cycle-accurate network simulation
7. **Well-structured configuration**: JSON-based cluster/model configs with validation
8. **Excellent logging infrastructure**: Component-aware logging with node/instance context
9. **Complete profiling pipeline**: Hardware profiling -> ML model training -> prediction lookup table for attention latency
10. **Multi-model support**: Llama, Mixtral, and Phi-MoE with proper TP-aware sharding in profiler
11. **4-method Timer abstraction**: Clean profiling backend selection (CUDA_EVENT, KINETO, PERF_COUNTER, RECORD_FUNCTION)
12. **Comprehensive cluster config matrix**: 12 configurations covering all deployment patterns (single/multi-node, P/D disaggregation, CXL, PIM, power, MoE)

---
---

# Part 2: Supplemental Deep-Dive (sim-code-analyst)

**Date**: 2026-02-28
**Analyst**: sim-code-analyst
**Scope**: Additional algorithmic deep-dive covering profiler internals, ML prediction pipeline, helper functions in trace_generator.py (1700-2355), dataset generation, evaluation scripts, and an extended GenZ/vllm-tuner comparison matrix. Supplements Part 1 with additional bug discoveries and integration analysis.

---

## S1. Additional Bug: `_make_sub_batch()` List Reset

**File**: `inference_serving/trace_generator.py:1930-1953`
**Severity**: HIGH

When building batch2, the accumulator lists `prefill_q_list`, `prefill_k_list`, and `decode_k_list` are NOT reset between batch1 and batch2 construction:

```python
# After building batch1 (lines 1896-1918):
total_len = 0; kv_len = 0; hit_len = 0
num_prefill = 0; num_decode = 0
q_list = []; k_list = []
# MISSING RESETS:
# prefill_q_list = []; prefill_k_list = []; decode_k_list = []

for req in req2:
    if req.is_init:
        prefill_q_list.append(...)  # Appends to batch1's list!
```

Batch2 will contain all of batch1's prefill and decode metadata plus its own. This corrupts attention metadata for the second sub-batch in every interleaved simulation run with mixed prefill/decode batches.

---

## S2. Trace Generator Helper Functions (Lines 1700-2355)

### S2.1 Performance DB Loading

**`_load_perf_db_dict(hardware, model, tp)`** (lines 1971-2018):
- Module-level cache: `_perf_db_cache = {}` keyed by `(hardware, model, tp)`
- Reads CSV with pandas, converts to dict: `(layer_name, input_len, kv_cache, tp_size) -> row_dict`
- On duplicate keys, keeps first occurrence (logs debug message)
- CSV path: `../llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv`

**`_load_attn_perf_db_dict(hardware, model, tp)`** (lines 2020-2121):
- Two-stage caching: checks in-memory dict first, then pickled file, then CSV
- Separate prefill and decode prediction DBs
- Prefill key: `(kv_cache_size, prefill_chunk_size)` -> `{latency(ns)}`
- Decode key: `(batch_size, kv_cache_size)` -> `{latency(ns)}`
- Pickle serialization for fast reload: first load from CSV builds dict and pickles it; subsequent loads deserialize pickle directly

**`_get_perf_row(perf_db, hardware, layer_name, input_len, kv_cache_len, tp_size)`** (lines 2123-2165):
- Exact key lookup: `(str(layer_name), int(input_len), int(kv_cache_len), int(tp_size))`
- TPU fallback: nearest-neighbor search across all rows matching layer_name and tp_size, preferring matching kv_cache and closest input_len
- Non-TPU hardware: raises KeyError on miss (no fallback)
- If TPU and no match at all, returns dummy row with `latency(ns)=1`

### S2.2 Attention Key Construction

**`_make_attn_db_key(hardware, model, batch)`** (lines 2175-2193):
- Prefill: Aggregates kv_cache_sizes from `batch.prefill_k_list`, rounds to 64-granularity. Chunk size = RMS of `batch.q_list` values, rounded to 32-granularity
- Decode: Average of `batch.decode_k_list`, rounded to 64-granularity. Batch size = `batch.num_decode`
- Uses `ceil()` for chunk size rounding (always rounds up)
- Uses floor-to-nearest for kv_cache rounding: `((val + gran - 1) // gran) * gran`

**RMS aggregation for chunk size is notable**: `round(sum([e**2 for e in q_list])**0.5)`. This gives higher weight to large prefill chunks, which makes sense because attention latency scales superlinearly with sequence length.

### S2.3 ML Prediction Integration

**`_load_attn_predictor(hardware, model, tp)`** (lines 2197-2242):
- Loads `rf_model.joblib` (RandomForest) and `attn_metadata.json`
- Sets `n_jobs=1` on loaded model to prevent multiprocessing overhead at inference time
- Comment shows they previously used XGBoost (`xgb_model.json`), migrated to sklearn RandomForest

**`_build_attn_feature_row(feature_cols, *, hardware, model, config, batch, npus_per_group)`** (lines 2247-2354):
- Reconstructs the full feature vector from batch state
- Reuses `make_attn_metadata()` from `attn_utils.py` for feature consistency with training
- Hardcoded SM counts: A6000=84, RTX3090=82, H100=132 (should be in config)
- Returns numpy array in exact column order from training metadata

### S2.4 PIM Load Balancing

**`_attn_load_balancer(requests, npus_per_group, pim_channels, channel_split)`** (lines 1824-1845):
- Greedy Min-Load Bin Packing
- Sorts requests by input length descending (longest first)
- Prefill tokens accumulated in single counter
- Decode tokens distributed across PIM channels using min-load-first assignment
- `channel_split` parameter allows one channel to handle multiple attention heads

### S2.5 Event Trace Generation

**`generate_event(alarm)`** (lines 1794-1819):
- Creates a single-event trace file for timer-based request arrival
- Writes to `inputs/trace/event_handler.txt`
- Event has compute time = alarm (ns), no data movement
- Used by main.py to inject arrival events into ASTRA-SIM

---

## S3. Profiler Infrastructure Internals

### S3.1 Timer Context Manager (4 Methods)

| Method | Enter | Exit | Measurement |
|--------|-------|------|-------------|
| `RECORD_FUNCTION` | `record_function(name).__enter__()` | `__exit__()` | Post-hoc trace parsing |
| `CUDA_EVENT` | `Event().record()` | `Event().record()` + store pair | `start.elapsed_time(end)` |
| `KINETO` | `profile().__enter__()` | `__exit__()` + callback | Callback aggregates CUDA time |
| `PERF_COUNTER` | `cuda.sync()` + `perf_counter()` | `cuda.sync()` + `perf_counter()` | Wall-clock delta |

**Key insight**: `RECORD_FUNCTION` is the default and preferred method. It creates trace annotations that are later correlated with CUDA kernels by `RecordFunctionTracer.get_operation_time_stats()`. This two-phase approach (annotate then analyze) avoids per-operation measurement overhead during execution.

### S3.2 RecordFunctionTracer Trace Correlation Algorithm

The correlation algorithm in `get_operation_time_stats()` is O(n^3) in the worst case:
1. For each `user_annotation` event: O(n)
2. For each annotation, scan all events for children: O(n)
3. For each child `cuda_runtime`, scan all events for correlated kernel: O(n)

This is acceptable for profiling (run once per hardware/model), but the O(n^3) cost means traces with many events (>10K) can be slow. The algorithm handles nested annotations correctly due to strict containment check.

### S3.3 Singleton Pattern Limitation

The `Singleton` metaclass stores instances in a class-level dict. Once `TimerStatsStore(profile_method="record_function")` is created, ALL subsequent calls to `TimerStatsStore(...)` return the same instance regardless of arguments. This prevents:
- Switching profiling methods mid-session
- Running multiple profiling configurations in parallel
- Testing different timer backends

---

## S4. Dataset Generation Analysis

### S4.1 ShareGPT Parser (`dataset/sharegpt_parser.py`, 129 lines)

**Arrival model**: Poisson process with `np.random.exponential(scale=1e9/rate)`. Seed fixed at 42 for reproducibility.

**Request generation modes**:
1. **Variable-length** (default): Parses multi-turn ShareGPT conversations, tokenizes with model-specific tokenizer, filters by max_input/output/kv length
2. **Fixed-length**: Generates random token IDs of configurable length (useful for controlled experiments)
3. **Pulse mode**: Burst of N requests followed by configurable delay (models traffic spikes)

**Output format** (JSONL):
```json
{"input_toks": 128, "output_toks": 512, "arrival_time_ns": 100000000, "input_tok_ids": [...], "output_tok_ids": [...]}
```

**Notable**: The `output_tok_ids` are included but never used by the simulator (router.py only reads `input_toks`, `output_toks`, `arrival_time_ns`). The token IDs enable future prefix sharing analysis (matching common prefixes across requests).

### S4.2 Configuration Space

| Parameter | Default | Range |
|-----------|---------|-------|
| `request_per_sec` | 10 | Any positive float |
| `max_input_length` | 2048 | Depends on model |
| `max_output_length` | 2048 | Depends on model |
| `max_kv_length` | 2048 | Depends on model |
| `max_requests` | 512 | Any positive int |
| `fix_input_length` | 128 | When `fix_len=True` |
| `fix_output_length` | 512 | When `fix_len=True` |
| `num_req_pulse` | 10 | When `pulse=True` |
| `delay_seconds` | 60 | When `pulse=True` |

---

## S5. Evaluation Scripts (`evaluation/memory/parser/`)

### S5.1 `parse.py` (51 lines) -- Single-Instance Log Parser

Extracts time-series data from simulation output:
- NPU memory usage (MB and %)
- Prefix cache hit ratio (%)
- Outputs TSV for plotting

### S5.2 `parse_md.py` (100 lines) -- Multi-Instance Log Parser

Handles two-instance (P/D disaggregation) logs:
- Per-instance NPU memory and hit ratio
- CPU memory usage per node
- Korean comments indicate KAIST team authorship
- Outputs TSV with columns: `time_s, inst0_mem_mb, inst0_hit_pct, inst1_mem_mb, inst1_hit_pct, cpu_mem_mb, cpu_hit_pct`

---

## S6. Extended GenZ/vllm-tuner/LLMServingSim Comparison

### S6.1 Three-Tier Analysis Capability Matrix

| Capability | GenZ (BudSimulator) | LLMServingSim 2.0 | vllm-tuner |
|-----------|--------------------|--------------------|-----------|
| **Latency source** | Analytical roofline | Profiled CSV DBs | Real hardware measurement |
| **Network model** | Analytical formulas | ASTRA-SIM (cycle-accurate) | Real network |
| **Speed** | <1 second | Minutes-hours | Hours (real execution) |
| **GPU required** | No | For profiling only | Yes (full deployment) |
| **Hypothetical HW** | Yes | No (needs profiling) | No (needs hardware) |
| **Continuous batching** | No | Full simulation | Real system |
| **KV cache lifecycle** | Aggregate estimate | Block-based + radix tree | Real PagedAttention |
| **Prefix caching** | No | Radix tree simulation | Real prefix caching |
| **P/D disaggregation** | No | Full simulation | Real P/D split |
| **Power modeling** | No | 7-component model | No |
| **PIM/CXL** | No | Yes | No |
| **CUDA graphs** | No | No | Yes |
| **Speculative decoding** | No | No | Yes |
| **Quantization** | Precision multipliers | fp16 only | INT8/INT4/GPTQ/AWQ |
| **Multi-step scheduling** | No | No | Yes |

### S6.2 Complementary Integration Strategy

```
Tier 1: GenZ Analytical (BudSimulator)
  Purpose: Rapid exploration
  Input: Hardware specs + model config
  Output: Latency/throughput estimates, memory requirements, optimal parallelism
  Speed: <1 second per configuration
  Use: Sweep 1000+ hardware/parallelism combinations

         |
         | Top 10 candidates
         v

Tier 2: LLMServingSim Simulation
  Purpose: Detailed serving-level analysis
  Input: Profiled performance DBs + cluster config + workload trace
  Output: TTFT/TPOT/ITL distributions, memory dynamics, power consumption
  Speed: Minutes per configuration
  Use: Validate top candidates with serving-level dynamics

         |
         | Top 2-3 configurations
         v

Tier 3: vllm-tuner Runtime Optimization
  Purpose: Final production tuning
  Input: Running vLLM deployment + tuning parameters
  Output: Optimal runtime configuration + real benchmarks
  Speed: Hours per configuration sweep
  Use: Optimize for production deployment
```

### S6.3 Key GenZ Gaps Revealed by This Analysis

1. **No attention tile efficiency**: GenZ models attention as `2*B*S*S*H*D` FLOPs, missing FA2 tiling effects (wave efficiency, tile padding, split heuristics) that LLMServingSim captures via profiling and ML prediction
2. **No continuous batching**: GenZ treats inference as isolated batches. Real serving interleaves prefill and decode, affecting memory and latency
3. **No prefix caching memory effects**: Shared prefixes reduce KV memory by up to 50%+ in production workloads
4. **No P/D disaggregation**: Separate prefill/decode instances with different batch sizes is a key production pattern
5. **No power/energy model**: Power consumption is increasingly important for TCO analysis
6. **No sub-batch interleaving**: GEMM/GEMV overlap in mixed batches improves GPU utilization

---

## S7. Performance Database Schema Reference

### S7.1 Layer Performance DB (`layers.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `layer_name` | string | e.g., "embedding", "q_proj", "k_proj", "attn", "gate_proj", etc. |
| `input` | int | Input sequence length (tokens) |
| `kv_cache` | int | KV cache length (0 for prefill) |
| `tp_size` | int | Tensor parallel degree |
| `latency(ns)` | int | Median kernel latency in nanoseconds |

**Lookup**: `(layer_name, input, kv_cache, tp_size) -> latency(ns)`

### S7.2 Attention Prediction CSVs

**Prefill** (`attn_prefill_predictions.csv`):
| Column | Type | Description |
|--------|------|-------------|
| `kv_cache_size` | int | KV cache length (0 for first chunk) |
| `prefill_chunk_size` | int | Prefill chunk token count |
| `prediction` | int | Predicted attention latency (ns) |

**Decode** (`attn_decode_predictions.csv`):
| Column | Type | Description |
|--------|------|-------------|
| `batch_size` | int | Number of decode sequences |
| `kv_cache_size` | int | Average KV cache length |
| `prediction` | int | Predicted attention latency (ns) |

### S7.3 Profiled Attention Raw Data (`attention.csv`)

Full profiling output with 25+ columns including all FA2 tiling metadata, time_stats per method, batch composition, and model parameters.

---

## S8. Cluster Config Complete Schema

```json
{
  "num_nodes": int,
  "link_bw": float,           // GB/s, inter-node bandwidth
  "link_latency": float,      // ns, inter-node latency
  "nodes": [{
    "num_instances": int,
    "cpu_mem": {
      "mem_size": float,       // GB
      "mem_bw": float,         // GB/s
      "mem_latency": float,    // ns
      "pim_config": "str"      // optional: DRAM .ini filename
    },
    "instances": [{
      "model_name": "str",     // HuggingFace model ID
      "hardware": "str",       // Must match perf DB directory name
      "npu_mem": {
        "mem_size": float,     // GB per GPU
        "mem_bw": float,       // GB/s
        "mem_latency": float   // ns
      },
      "npu_num": int,          // GPUs per instance (TP degree)
      "npu_group": int,        // Pipeline parallel groups
      "pd_type": "str"|null,   // "prefill", "decode", or null (unified)
      "placement": {           // optional
        "default": {"weights": "npu"|"cpu"|"cxl:N", "kv_loc": "...", "kv_evict_loc": "..."},
        "blocks": [{"blocks": "0-3,5", "weights": "...", ...}],
        "layers": {"embedding": {...}, "final_layernorm": {...}, "lm_head": {...}}
      }
    }],
    "power": {                 // optional
      "base_node_power": float,
      "npu": {"<hw>": {"idle_power": float, "standby_power": float, "active_power": float, "standby_duration": float}},
      "cpu": {"idle_power": float, "active_power": float, "util": float},
      "dram": {"dimm_size": float, "idle_power": float, "energy_per_bit": float},
      "link": {"num_links": int, "idle_power": float, "energy_per_bit": float},
      "nic": {"num_nics": int, "idle_power": float},
      "storage": {"num_devices": int, "idle_power": float}
    }
  }]
}
```

---

*End of supplemental analysis. All Python source files, JSON configs, and evaluation scripts in the LLMServingSim 2.0 repository have been exhaustively reviewed and cross-referenced.*
