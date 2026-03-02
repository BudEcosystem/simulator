# Reusable Code Mapping: LLMServingSim & vllm-tuner to BudSimulator

**Analyst**: Integration Architect
**Date**: 2026-02-28
**Scope**: Detailed module-by-module mapping of reusable code from LLMServingSim 2.0 and vllm-tuner with exact file paths, line numbers, adaptation requirements, and integration order

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [LLMServingSim to BudSimulator Code Map](#2-llmservingsim-to-budsimulator-code-map)
3. [vllm-tuner to BudSimulator Code Map](#3-vllm-tuner-to-budsimulator-code-map)
4. [Code That Should NOT Be Reused](#4-code-that-should-not-be-reused)
5. [Adaptation Patterns](#5-adaptation-patterns)
6. [Dependency Graph](#6-dependency-graph)
7. [License and Attribution](#7-license-and-attribution)

---

## 1. Executive Summary

### Quantitative Overview

| Metric | LLMServingSim | vllm-tuner | Total |
|--------|--------------|------------|-------|
| Total source LOC | 6,929 | 4,384 | 11,313 |
| Reusable modules identified | 9 | 5 | 14 |
| Reusable LOC (original) | ~4,200 | ~650 | ~4,850 |
| Estimated adapted LOC | ~3,100 | ~500 | ~3,600 |
| Effort saved vs. from-scratch | ~60% | ~40% | ~55% |

### Quality Assessment

**LLMServingSim code quality: HIGH (with caveats)**
- Well-structured class hierarchy with clear responsibilities
- Memory model and scheduler are production-quality with extensive edge case handling
- Uses logging framework consistently (custom `LLMServingSimFormatter`)
- Radix tree implementation is adapted from SGLang (well-tested upstream)
- Primary concern: tight coupling to ASTRA-SIM process lifecycle in `main.py`
- The `trace_generator.py` (2,354 lines) is monolithic and requires significant refactoring

**vllm-tuner code quality: MODERATE (significant bugs)**
- Clean abstract interfaces (`Workload`, `VLLMSearchSpace`)
- Pattern-level reuse is more valuable than direct code reuse
- Multiple critical bugs documented in findings 01 and 02 (study_manager resource leaks, HTML report XSS, telemetry parser overwrites)
- GPU collector is functional but needs NVML error handling improvements

### Integration Priority Summary

| Priority | Module | Source | Effort |
|----------|--------|--------|--------|
| P0 | MemoryModel (multi-tier) | LLMServingSim | 2 weeks |
| P0 | Controller (ASTRA-SIM IPC) | LLMServingSim | 1 week |
| P0 | Request/Batch data structures | LLMServingSim | 3 days |
| P1 | BatchScheduler (continuous batching) | LLMServingSim | 2 weeks |
| P1 | RadixTree (prefix caching) | LLMServingSim | 1 week |
| P1 | Statistics collector pattern | LLMServingSim | 1 week |
| P1 | Search space definition pattern | vllm-tuner | 3 days |
| P2 | PowerModel (7-component) | LLMServingSim | 1 week |
| P2 | ExecutionPlanner (trace generation) | LLMServingSim | 2 weeks |
| P2 | Workload abstraction pattern | vllm-tuner | 2 days |
| P2 | GPU telemetry collection | vllm-tuner | 3 days |
| P2 | HTML report generation pattern | vllm-tuner | 1 week |
| P2 | Router (request routing) | LLMServingSim | 3 days |
| P2 | SLO tracking pattern | LLMServingSim | 3 days |

---

## 2. LLMServingSim to BudSimulator Code Map

### 2a. MemoryModel -- Multi-Tier Memory Hierarchy Simulation

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/memory_model.py`
**Total lines**: 752

#### What it does
Implements a complete multi-tier memory management system for LLM inference serving. Tracks weight memory, KV cache allocations, and prefix cache state across NPU (HBM), CPU (DRAM), and CXL memory tiers. Supports block-based KV cache management with eviction policies, prefix cache integration via RadixTree, and event-driven memory accounting.

#### Key Classes and Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `Device` enum | 11-14 | NPU/CPU/CXL device enumeration |
| `MemoryModel.__init__()` | 17-90 | Initializes memory model with model config, memory capacities, prefix cache |
| `MemoryModel.get_weight()` | 92-153 | Per-layer weight calculation with TP sharding |
| `MemoryModel.get_kv()` | 156-162 | KV cache size formula: `2 * kv_dim * seq * n_layer * fp // npu_num` |
| `MemoryModel.get_block_kv()` | 172-186 | Incremental KV allocation per batch with eviction awareness |
| `MemoryModel.get_evict_kv()` | 189-196 | Calculate eviction size for a request |
| `MemoryModel.allocate()` | 217-249 | Device-dispatched allocation with bounds checking |
| `MemoryModel.free()` | 251-284 | Device-dispatched deallocation with safety checks |
| `MemoryModel.is_avail()` | 286-303 | Availability query across device tiers |
| `MemoryModel.need_size()` | 305-324 | Deficit calculation for allocation planning |
| `MemoryModel.prefix_match()` | 446-468 | Match request tokens against NPU and storage prefix caches |
| `MemoryModel.apply_kv_cache_events()` | 493-533 | Event-driven memory accounting from RadixCache events |
| `calculate_sizes()` (standalone) | 544-752 | Per-rank input/weight/output size calculation for every layer type |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/memory_model.py
```

This would be a new file in the proposed `runtime/` subdirectory under the existing simulation engine.

#### Adaptation Needed

1. **Replace `get_config(model)`**: LLMServingSim uses a JSON-based config lookup (`utils.py:67-79`) that reads from `model_config/<name>.json`. BudSimulator has its own model registry in `Models/` directory. Adapter needed:
   ```python
   # LLMServingSim pattern (current):
   self.config = get_config(model)  # reads JSON file
   self.n_embd = self.config['hidden_size']

   # BudSimulator adaptation:
   # Use GenZ model definition or accept config dict directly
   def __init__(self, model_config: dict, ...):
       self.n_embd = model_config['hidden_size']
   ```

2. **Integrate Device enum with GenZ System class**: The `Device` enum (NPU/CPU/CXL) needs to map to GenZ's `System` class which uses `off_chip_mem_size` and `external_mem_bw` for memory tiers. Add a translation layer:
   ```python
   def device_from_system(system: System, tier: str) -> DeviceConfig:
       if tier == 'NPU':
           return DeviceConfig(capacity=system.off_chip_mem_size, bw=system.offchip_mem_bw)
       elif tier == 'CPU':
           return DeviceConfig(capacity=..., bw=system.external_mem_bw)
   ```

3. **Decouple logger**: Replace `from .logger import get_logger` with Python standard `logging.getLogger(__name__)` to match BudSimulator's logging pattern.

4. **Make prefix caching optional at import time**: The current code imports `RadixTree` unconditionally. Gate the import behind a flag to avoid requiring `msgspec` when prefix caching is not used.

5. **Unit conversion alignment**: LLMServingSim uses `GB_TO_BYTE = 1024*1024*1024` constants. BudSimulator's `Unit` class uses its own conversion system. Standardize to use `Unit` class.

#### Dependencies
- `radix_tree.py` (required if prefix caching enabled)
- `utils.py:get_config()` (must be replaced with BudSimulator model registry)
- `logger.py:get_logger()` (replace with standard logging)
- `msgspec` package (optional, for RadixCache KV events)

#### Priority: P0
This is the foundational module. BudSimulator currently has only static memory capacity checks in its analytical engine. This module enables dynamic memory tracking which is prerequisite for the scheduler and hybrid simulation mode.

#### LOC Estimate
- Original: 752 lines
- Adapted: ~550 lines (remove `calculate_sizes()` which overlaps with GenZ operators; simplify logger; remove redundant comments)

---

### 2b. BatchScheduler -- Continuous Batching with Iteration-Level Scheduling

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/scheduler.py`
**Total lines**: 715

#### What it does
Implements vLLM/Orca-style continuous batching where requests are dynamically selected for each iteration based on memory availability, arrival time, and batch size constraints. Supports two scheduling modes: `schedule_base()` for standard scheduling and `schedule_with_prefix()` for prefix-cache-aware scheduling. Handles the full request lifecycle from arrival through completion, collecting TTFT/TPOT/ITL metrics.

#### Key Classes and Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `Scheduler.__init__()` | 16-49 | Initialize with model, memory model, batch constraints, P/D type |
| `Scheduler.schedule_base()` | 59-239 | Standard continuous batching: filter by arrival, enforce batch/token limits, memory-aware eviction |
| `Scheduler.schedule_with_prefix()` | 241-477 | Prefix-aware scheduling with multi-tier eviction and cache loading |
| `Scheduler.add_done()` | 480-578 | Handle batch completion: phase transitions, KV cache management, metric collection |
| `Scheduler.print_result()` | 632-669 | Compute mean/median/P99 for TTFT, TPOT, ITL |
| `Scheduler.save_output()` | 687-715 | Write per-request CSV with lifecycle metrics |
| `Scheduler._merge_by_arrival_id()` | 604-629 | Merge-sort requests by (arrival_time, id) for priority queue maintenance |

#### Core scheduling algorithm (`schedule_base()`, lines 59-239):

```python
# Simplified pseudocode of the scheduling algorithm:
def schedule_base(self, current_time, sys_id, batch_id):
    # 1. Filter by arrival time
    batch_req = [req for req in self.request if req.arrival <= current_time]
    # 2. Apply max_batch constraint
    batch_len = min(len(batch_req), self.max_batch)
    # 3. Memory-aware batch trimming with eviction
    for i in range(batch_len, -1, -1):
        kv_size = self.memory.get_block_kv(batch_req, i)
        if self.memory.is_avail(kv_size, Device.NPU):
            break
    # 4. If no memory: evict decode requests one-by-one
    while temp_len == 0 and gen_req:
        evict_size = self.memory.get_evict_kv(gen_req[-1])
        gen_req[-1].evict = True
        self.memory.free(evict_size, Device.NPU)
        self.memory.allocate(evict_size, Device.CPU)  # spill to CPU
    # 5. Apply max_num_batched_tokens constraint
    # 6. Create Batch object with prefill/decode token lists
    return Batch(...)
```

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/scheduler.py
```

#### Adaptation Needed

1. **Decouple from ASTRA-SIM NPU indexing**: The current scheduler uses `sys` (NPU system ID) for tracking which NPUs have fired a batch and `start_npu` for the first NPU in an instance. This maps to ASTRA-SIM's per-NPU output parsing. For BudSimulator's hybrid mode, replace with a simpler instance-based scheduling where the scheduler does not need to track individual NPU firing order.

2. **Parameterize scheduling policy**: The current code hardcodes the eviction order (last decode request first) and priority (optional `prioritize_prefill`). Extract the eviction strategy as a pluggable policy:
   ```python
   class EvictionPolicy(ABC):
       @abstractmethod
       def select_victim(self, gen_requests: List[Request]) -> Request: ...

   class LRUEviction(EvictionPolicy):
       def select_victim(self, gen_requests):
           return gen_requests[-1]  # current behavior
   ```

3. **Integration with GenZ analytical engine**: For hybrid mode, after creating a Batch, compute analytical performance estimate using GenZ's prefill/decode modeling functions before feeding to ASTRA-SIM. Add a hook:
   ```python
   batch = self.schedule(current_time, ...)
   if batch:
       analytical_time = self.compute_engine.estimate_batch_time(batch)
   ```

4. **Replace direct imports**: Change `from .request import *` to explicit imports for clarity. Replace `from .utils import *` with specific function imports.

5. **Thread safety**: The scheduler is currently single-threaded (one scheduler per instance). If BudSimulator runs multiple instances in parallel threads, add locking around `self.request`, `self.inflight`, and `self.done` lists.

#### Dependencies
- `memory_model.py` (MemoryModel class -- must be ported first)
- `request.py` (Request, Batch classes -- must be ported first)
- `utils.py:get_config()` (replace with BudSimulator model registry)
- `trace_generator.py` (for generating execution traces after scheduling -- port separately)
- `numpy` (for metric statistics computation)

#### Priority: P1
Depends on MemoryModel (P0) and Request data structures (P0). Enables continuous batching simulation which is critical for realistic serving metrics (TTFT/TPOT/ITL distributions).

#### LOC Estimate
- Original: 715 lines
- Adapted: ~500 lines (merge `schedule_base` and `schedule_with_prefix` into a single configurable method; extract eviction policy; remove ASTRA-SIM NPU-specific firing logic)

---

### 2c. PowerModel -- 7-Component Power Modeling

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/power_model.py`
**Total lines**: 214

#### What it does
Models power consumption across 7 hardware components: NPU (active/idle/standby), CPU (active/idle with utilization), DRAM (idle + energy-per-bit for data movement), link (energy-per-bit for collectives), NIC (idle), and storage (idle). Tracks energy over time with base power (always-on) and net energy (activity-dependent), providing time-series power data and final energy summaries.

#### Key Classes and Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `PowerModel.__init__()` | 5-51 | Initialize with power configs per node, compute base powers |
| `add_npu_standby_energy_consumption()` | 54-70 | Standby energy between batches with duration-aware calculation |
| `add_npu_active_energy_consumption()` | 73-83 | Active NPU + CPU energy during layer execution |
| `add_dram_energy_consumption()` | 87-93 | Energy-per-bit model for DRAM data movement |
| `add_link_energy_consumption()` | 102-108 | Energy-per-bit model for network link data movement |
| `get_current_power()` | 110-121 | Instantaneous power from energy delta over time delta |
| `get_final_energy()` | 123-130 | Total system energy at simulation end |
| `print_power_summary()` | 132-142 | Per-node per-component energy breakdown |
| `total_ring_data()` (standalone) | 186-196 | Data movement calculation for ring-based collectives |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/power_model.py
```

**Note**: BudSimulator already has a basic power module at `llm-memory-calculator/src/llm_memory_calculator/genz/power.py` (39 lines) that computes energy from a DataFrame of operator utilizations with a 4-component breakdown (Static, Compute, Memory, Network). The LLMServingSim model is significantly more detailed with 7 components, per-NPU standby tracking, and energy-per-bit models.

#### Adaptation Needed

1. **Align with existing BudSimulator power.py**: The existing `get_energy()` function in `genz/power.py` takes a DataFrame and power breakdown dict. Either:
   - (Recommended) Create the new `PowerModel` as a separate class that extends the existing pattern for runtime simulation mode, keeping `get_energy()` for analytical mode.
   - Or, make `PowerModel` a drop-in replacement that can also produce the same DataFrame-based output.

2. **Power config format**: LLMServingSim reads power config from cluster JSON. BudSimulator has hardware configs in its database. Create adapter:
   ```python
   def power_config_from_hardware(hardware_config: dict) -> dict:
       """Convert BudSimulator hardware config to PowerModel config format."""
       return {
           "base_node_power": hardware_config.get("tdp", 0) * 0.1,  # estimate
           "npu": { hardware_config["name"]: {
               "idle_power": hardware_config.get("idle_power", 0),
               "active_power": hardware_config.get("tdp", 300),
               "standby_power": hardware_config.get("tdp", 300) * 0.5,
               "standby_duration": 0.001,  # 1ms default
               "num_npus": hardware_config.get("num_nodes", 1),
           }},
           # ... CPU, DRAM, link, NIC, storage configs
       }
   ```

3. **Remove formatting dependencies**: Replace `SINGLE_BAR`, `DEVICE_STR`, `DEVICE_SPACE` from `utils.py` with standard Python formatting.

#### Dependencies
- `utils.py` (only for formatting constants -- trivially replaceable)
- `logger.py:get_logger()` (replace with standard logging)

#### Priority: P2
Power modeling is valuable for TCO analysis but is not blocking other features. Can be integrated after the core simulation loop is functional.

#### LOC Estimate
- Original: 214 lines
- Adapted: ~180 lines (minor simplifications to formatting, add hardware config adapter)

---

### 2d. RadixTree -- Prefix Caching Data Structure

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/radix_tree.py`
**Total lines**: 597

#### What it does
Implements a radix tree (Patricia trie) for KV cache prefix matching, adapted from SGLang's implementation (Apache 2.0 licensed). Supports page-aligned matching, LRU-based eviction of unlocked leaves, reference counting for concurrent access protection, and event-driven cache accounting (BlockStored/BlockRemoved events). This is the data structure that enables prefix caching -- when multiple requests share common prompt prefixes, the cached KV values can be reused.

#### Key Classes and Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `KVCacheEvent` / `BlockStored` / `BlockRemoved` | 38-61 | Event types for cache accounting (requires `msgspec`) |
| `MatchResult` | 63-71 | Named tuple for prefix match results |
| `TreeNode` | 74-107 | Tree node with children, key, lock_ref, last_access_time, hash_value |
| `_key_match_page_size1()` | 109-115 | Token-by-token key matching for page_size=1 |
| `_key_match_paged()` | 118-131 | Page-aligned key matching with remainder handling |
| `RadixCache.__init__()` | 134-171 | Initialize with page_size, capacity, kv_size, event queue |
| `RadixCache.match_prefix()` | 237-263 | Public API: find matching prefix, return (last_node, hit_length) |
| `RadixCache.insert()` | 265-266 | Insert key into tree |
| `RadixCache.cache_finished_req()` | 268-282 | Cache completed request tokens |
| `RadixCache.cache_unfinished_req()` | 283-311 | Cache in-progress request tokens |
| `RadixCache.evict()` | 325-344 | LRU eviction: collect leaves, heapify by access time, pop unlocked |
| `RadixCache.inc_lock_ref()` / `dec_lock_ref()` | 346-368 | Reference counting for eviction protection |
| `RadixCache._match_prefix_helper()` | 393-418 | Recursive prefix matching with node splitting |
| `RadixCache._insert_helper()` | 435-464 | Recursive insertion with node splitting and event recording |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/radix_cache.py
```

#### Adaptation Needed

1. **Make `msgspec` optional**: The `KVCacheEvent` classes use `msgspec.Struct` which is not a standard dependency. Add fallback:
   ```python
   try:
       import msgspec
       class KVCacheEvent(msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True):
           """Base class for KV cache events"""
   except ImportError:
       from dataclasses import dataclass
       @dataclass
       class KVCacheEvent:
           """Base class for KV cache events (dataclass fallback)"""
   ```

2. **Static counter issue**: `TreeNode.counter` (line 76) is a class-level counter that persists across instances. If multiple RadixCache instances are created (e.g., NPU cache + CPU cache), node IDs may not be unique per cache. Add per-cache counter or use `uuid`.

3. **Thread safety**: The `_lock = threading.RLock()` is used in `match_prefix()` and `cache_*` methods but NOT in `evict()`, `insert()`, or `inc_lock_ref()`/`dec_lock_ref()`. If accessed from multiple threads, extend locking to all mutating operations.

4. **Replace logger dependency**: Change `from .logger import get_logger` to standard logging.

#### Dependencies
- `msgspec` (optional, for efficient event serialization)
- `logger.py:get_logger()` (replace with standard logging)
- No other LLMServingSim dependencies

#### Priority: P1
Required by MemoryModel when prefix caching is enabled. Can be deferred if initial integration does not include prefix caching.

#### LOC Estimate
- Original: 597 lines
- Adapted: ~550 lines (make msgspec optional, fix thread safety, minor cleanups)

---

### 2e. ExecutionPlanner -- Trace Generation & DAG Construction

**Source files**:
- `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/trace_generator.py` (2,354 lines)
- `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/graph_generator.py` (37 lines)

#### What it does
`trace_generator.py` is the largest file in LLMServingSim. It synthesizes per-batch execution traces as text files describing the full transformer layer sequence (embedding, QKV, attention, FFN, collectives, memory operations) with hardware-specific timing from a performance database. `graph_generator.py` converts these text traces to Chakra execution graphs using the LLM converter mode.

#### Key Functions (trace_generator.py)

| Component | Lines (approx) | Description |
|-----------|----------------|-------------|
| `_perf_db_cache` / `_attn_perf_db_cache` | 26-27 | Global caches for performance database lookups |
| `generate_trace()` | 40-82 | Entry point: delegates to `_synthesize_trace()` or `_synthesize_interleaved_trace()` |
| `_synthesize_trace()` | ~100-600 | Main trace synthesis: iterates over transformer layers, writes operator trace lines |
| `_synthesize_interleaved_trace()` | ~600-1000 | Sub-batch interleaving variant |
| `_make_sub_batch()` | ~1000-1050 | Split batch into prefill and decode sub-batches |
| Layer-specific synthesis | ~1050-2354 | Per-layer trace generation for each layer type with TP/EP awareness |

#### Key Functions (graph_generator.py)

| Component | Lines | Description |
|-----------|-------|-------------|
| `generate_graph()` | 9-37 | Run Chakra converter in LLM mode with NPU offset support |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/trace_generator.py
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/graph_generator.py
```

#### Adaptation Needed

1. **Major refactoring required**: The 2,354-line `trace_generator.py` is monolithic. It must be decomposed into:
   - A `TraceBuilder` class that holds state (batch, config, placement)
   - Per-layer trace generation methods (one per layer type)
   - A `PerformanceDatabase` class wrapping the global caches
   - A `TraceWriter` class for file I/O

2. **Replace performance database with GenZ roofline**: For the hybrid simulation mode, operator timing should come from GenZ's analytical engine rather than pre-profiled CSV databases. The performance database pattern is still useful for when profile data is available, but should not be required.

3. **Replace `calculate_sizes()` calls**: The trace generator imports `calculate_sizes()` from `memory_model.py`. In BudSimulator, equivalent size calculations exist in the GenZ operator framework. Create an adapter that maps LLMServingSim layer names to GenZ operators.

4. **Graph generator path handling**: The current `graph_generator.py` uses `os.chdir()` to change working directory, which is not thread-safe. Replace with absolute paths.

5. **Remove sklearn/xgboost dependency**: Lines 17-19 import `sklearn` and `joblib` for attention prediction models. Make these optional imports.

#### Dependencies
- `memory_model.py:calculate_sizes()` (or replace with GenZ operators)
- `config_builder.py:get_device()` (for placement resolution)
- `power_model.py:PowerModel` (for energy tracking during trace generation)
- `pim_model.py:PIMModel` (for PIM-accelerated attention)
- `attn_utils.py` (FlashAttention heuristics)
- `gate_function.py:GateRouter` (MoE expert routing)
- `utils.py:get_config()`, `formatter()`, `header()`
- Chakra package (for graph conversion)

#### Priority: P2
This is the most complex module to adapt. It enables full end-to-end ASTRA-SIM-driven simulation but is not needed for the hybrid mode that uses GenZ analytical timing. Recommend deferring until after Phase 4 of integration.

#### LOC Estimate
- Original: 2,354 + 37 = 2,391 lines
- Adapted: ~1,200 lines (after refactoring into classes, removing redundant code, making performance database optional)

---

### 2f. ASTRA-SIM Persistent Integration -- Bidirectional IPC

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/controller.py`
**Total lines**: 58

#### What it does
Manages bidirectional stdin/stdout IPC with a persistent ASTRA-SIM subprocess. This replaces BudSimulator's current fire-and-forget approach (`subprocess.run()` per collective) with a long-running process that accepts workloads iteratively.

#### Key Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `Controller.__init__()` | 5-10 | Initialize with total NPU count, end_dict tracking |
| `Controller.read_wait()` | 13-21 | Block on stdout until "Waiting" signal from ASTRA-SIM |
| `Controller.check_end()` | 23-30 | Wait for "All Request Has Been Exited" confirmation |
| `Controller.write_flush()` | 32-37 | Write workload path or control command to stdin |
| `Controller.parse_output()` | 39-57 | Regex parse: `sys[X] iteration Y finished, Z cycles, exposed communication W cycles` |

#### Exact code (full file, 58 lines):

```python
import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1

    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))
            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys, id, cycle, com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return
```

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/Astra_sim/controller.py
```

This sits alongside the existing `get_astra_sim_time.py` in the Astra_sim directory.

#### Adaptation Needed

1. **Add timeout handling**: The current `read_wait()` blocks indefinitely. Add a timeout:
   ```python
   import select

   def read_wait(self, p, timeout_seconds=300):
       out = [""]
       while "Waiting" not in out[-1]:
           ready, _, _ = select.select([p.stdout], [], [], timeout_seconds)
           if not ready:
               raise TimeoutError(f"ASTRA-SIM did not respond within {timeout_seconds}s")
           line = p.stdout.readline()
           out.append(line)
       return out
   ```

2. **Add process health checking**: Check if the subprocess is still alive before read/write operations:
   ```python
   def _check_alive(self, p):
       if p.poll() is not None:
           stderr = p.stderr.read() if p.stderr else ""
           raise RuntimeError(f"ASTRA-SIM process died (exit code {p.returncode}): {stderr}")
   ```

3. **Integration with existing path_utils.py**: Use BudSimulator's existing `path_utils.py:validate_path()` for workload paths passed to `write_flush()`.

4. **Fix missing `com_cycle` in return dict**: The original code at line 57 returns `{'sys': sys, 'id': id, 'cycle': cycle}` omitting the extracted `com_cycle` (exposed communication cycles). This is a bug in LLMServingSim -- the exposed communication time is critical for compute-communication overlap analysis. Fix in adaptation:
   ```python
   return {'sys': sys, 'id': id, 'cycle': cycle, 'com_cycle': com_cycle}
   ```

5. **Integrate with existing ASTRA-SIM startup**: BudSimulator's `get_astra_sim_time.py` already handles network config generation and system.json creation. The Controller should accept an already-started subprocess or have a factory method that starts ASTRA-SIM using the existing config generation code.

#### Dependencies
- `re` module (standard library)
- `logger.py:get_logger()` (replace with standard logging)
- `path_utils.py` (for workload path validation)

#### Priority: P0
This is a small, self-contained module that unlocks the persistent ASTRA-SIM subprocess pattern. It amortizes ASTRA-SIM startup cost and enables multi-iteration simulation.

#### LOC Estimate
- Original: 58 lines
- Adapted: ~100 lines (add timeout, health check, path validation, exposed comm cycle fix)

---

### 2g. Request/Sequence Data Structures -- Request Lifecycle Management

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/request.py`
**Total lines**: 85

#### What it does
Defines `Request` and `Batch` classes that manage the complete lifecycle of inference requests from arrival through completion. `Request` tracks timing metrics (TTFT, TPOT, ITL), prefix cache state, eviction status, and token counts. `Batch` groups requests for a single iteration with token lists, memory sizes, and NPU firing state.

#### Key Classes

| Component | Lines | Description |
|-----------|-------|-------------|
| `Request.__init__()` | 3-30 | Full lifecycle state: id, model, input/output lengths, arrival, timing, prefix cache state |
| `Request.add_latency()` | 36-48 | Compute end-to-end latency and TPOT on completion |
| `Request.add_itl()` | 50-52 | Record inter-token latency |
| `Request.set_que_delay()` | 54-55 | Record queuing delay |
| `Request.set_ttft()` | 57-59 | Record time to first token |
| `Batch.__init__()` | 64-85 | Batch state: token lists, memory sizes, prefill/decode counts, NPU firing state |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/request.py
```

#### Adaptation Needed

1. **Add dataclass decorators**: Convert to `@dataclass` for cleaner code and auto-generated `__repr__`:
   ```python
   from dataclasses import dataclass, field
   from typing import List, Optional

   @dataclass
   class Request:
       id: int
       model: str
       input_len: int  # renamed from 'input' to avoid shadowing builtin
       output_len: int  # renamed from 'output'
       arrival: int
       instance_id: int
       is_init: bool = True
       # ... etc
   ```

2. **Rename `input`/`output` fields**: The current code uses `self.input` and `self.output` which shadow Python builtins. Rename to `input_len` and `output_len`.

3. **Add type hints**: The current code has no type annotations. Add comprehensive type hints for all fields.

4. **Add serialization**: Add `to_dict()` / `from_dict()` methods for JSON serialization, aligning with BudSimulator's API response patterns.

5. **Remove ASTRA-SIM NPU tracking from Batch**: The `fired` and `end` lists (lines 71, 73) track which ASTRA-SIM NPU systems have processed a batch. In hybrid mode, this is not needed. Make it optional.

#### Dependencies
- None (self-contained data structures)

#### Priority: P0
These are pure data structures with no external dependencies. They must be ported before the scheduler and memory model can function.

#### LOC Estimate
- Original: 85 lines
- Adapted: ~120 lines (add dataclass decorators, type hints, serialization, field renaming)

---

### 2h. SLO Tracking and Statistics Collection

**Source locations**:
- Scheduler metrics: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/scheduler.py`, lines 632-715
- Main loop metrics: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/main.py`, lines 453-538
- Throughput time series: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/main.py`, lines 328-345

#### What it does
Collects and reports per-request serving metrics (TTFT, TPOT, ITL) with mean/median/P99 statistics. The main loop additionally computes throughput time series (prompt tokens/s, generation tokens/s at configurable intervals), total token counts, prefix cache hit ratios, and power consumption summaries.

Note: LLMServingSim does NOT currently implement explicit SLO violation counting (e.g., "X% of requests exceeded 200ms TTFT"). This is a feature BudSimulator should ADD on top of the ported metrics collection.

#### Key Code Snippets

**Per-instance metrics (scheduler.py:632-669)**:
```python
def print_result(self):
    ttft_values = [req.ttft for req in self.done]
    tpot_values = [req.tpot for req in self.done]
    itl_values = [itl for req in self.done for itl in req.itl]
    # Compute mean, median, P99 for each
    mean = np.mean(ttft_values) / 1000_000  # ns to ms
    median = np.median(ttft_values) / 1000_000
    p99 = np.percentile(ttft_values, 99) / 1000_000
```

**Per-request CSV output (scheduler.py:687-715)**:
```python
writer.writerow(['instance id', 'request id', 'model', 'input', 'output',
                'arrival', 'end_time', 'latency',
                'queuing_delay', 'TTFT', 'TPOT', 'ITL'])
```

**System-wide throughput (main.py:481-498)**:
```python
total_latency = current / FREQ
print(f"Request throughput (req/s):  {req_cnt/total_latency:.2f}")
print(f"Average prompt throughput (tok/s):  {total_prompt/total_latency:.2f}")
print(f"Average generation throughput (tok/s):  {total_gen/total_latency:.2f}")
```

**Throughput time series (main.py:328-345)**:
```python
if current > last_log + INTERVAL:
    throughput.append((prompt_th*RATIO, gen_th*RATIO))
    last_log += INTERVAL
    prompt_th = 0
    gen_th = 0
```

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/statistics.py
```

#### Adaptation Needed

1. **Extract into a dedicated `StatisticsCollector` class** instead of mixing into Scheduler and main loop:
   ```python
   @dataclass
   class DistributionStats:
       mean: float
       median: float
       p50: float
       p90: float
       p99: float
       min_val: float
       max_val: float

   @dataclass
   class SLOConfig:
       ttft_target_ms: float = 200.0
       tpot_target_ms: float = 50.0
       itl_target_ms: float = 100.0

   class StatisticsCollector:
       def __init__(self, slo_config: Optional[SLOConfig] = None):
           self.requests: List[CompletedRequest] = []
           self.throughput_series: List[ThroughputSample] = []
           self.slo_config = slo_config

       def record_completion(self, request: Request, finish_time: int): ...
       def record_throughput_sample(self, time_ns: int, prompt_toks: int, gen_toks: int): ...
       def get_ttft_stats(self) -> DistributionStats: ...
       def get_tpot_stats(self) -> DistributionStats: ...
       def get_itl_stats(self) -> DistributionStats: ...
       def get_slo_violations(self) -> Dict[str, float]: ...  # NEW
       def to_dict(self) -> Dict[str, Any]: ...
   ```

2. **Add SLO violation tracking** (not present in LLMServingSim):
   ```python
   def get_slo_violations(self) -> Dict[str, float]:
       if not self.slo_config:
           return {}
       ttft_violations = sum(1 for r in self.requests
                           if r.ttft / 1_000_000 > self.slo_config.ttft_target_ms)
       return {
           "ttft_violation_rate": ttft_violations / max(len(self.requests), 1),
           "tpot_violation_rate": ...,
           "itl_violation_rate": ...,
       }
   ```

3. **Integrate with SimulationResult**: The existing `SimulationResult` class in BudSimulator (`genz/simulation/results.py`) has fields for latency and throughput but not for serving metrics distributions. Extend with a `serving_metrics` field:
   ```python
   @dataclass
   class ServingMetrics:
       ttft: DistributionStats
       tpot: DistributionStats
       itl: DistributionStats
       throughput_series: List[ThroughputSample]
       slo_violations: Optional[Dict[str, float]]
   ```

#### Dependencies
- `numpy` (for percentile calculations)
- `request.py` (Request class with timing fields)

#### Priority: P1
Essential for evaluating serving quality. The metrics collection is relatively simple but must be properly integrated with the SimulationResult schema.

#### LOC Estimate
- Original: ~120 lines (scattered across scheduler.py and main.py)
- Adapted: ~200 lines (as a standalone StatisticsCollector class with SLO tracking)

---

### 2i. Cluster Configuration Builder

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/config_builder.py`
**Total lines**: 557

#### What it does
Parses multi-node, multi-instance cluster JSON configurations and generates ASTRA-SIM input files (network.yml, system.json, memory_expansion.json). Supports heterogeneous instances with different models, hardware, and parallelism configurations per instance. Validates memory placement policies and generates NPU-to-instance mappings.

#### Key Functions

| Component | Lines (approx) | Description |
|-----------|----------------|-------------|
| `build_cluster_config()` | 20-404 | Main entry: parse cluster JSON, generate all ASTRA-SIM configs |
| `_create_network_config()` | ~407-426 | Generate network.yml from cluster topology |
| `get_device()` | ~488-512 | Resolve memory placement (NPU/CPU/CXL) per layer per instance |
| Power config validation | ~79-180 | Validate 7-component power config in cluster JSON |
| Memory placement validation | ~200-350 | Validate weight/KV placement against available memory |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/cluster_config.py
```

#### Adaptation Needed

1. **Integrate with BudSimulator hardware database**: BudSimulator stores hardware configs in SQLite. The cluster config builder should accept hardware IDs and look up specs from the database rather than requiring a standalone JSON file.

2. **Reuse GenZ network config generation**: BudSimulator's `get_astra_sim_time.py` already generates `network.yml`. Unify the network config generation to avoid duplication.

3. **The per-instance model assignment**: BudSimulator currently assumes a single model per analysis. The cluster config builder's multi-model support is valuable for P/D disaggregation scenarios. Preserve this capability.

#### Dependencies
- `json`, `yaml`, `math`, `os` (standard library)
- `utils.py:get_config()` (replace with BudSimulator model registry)
- `pim_model.py:PIMModel` (optional, for PIM configuration)

#### Priority: P2
Needed for multi-node/multi-instance simulation but not for single-instance hybrid mode.

#### LOC Estimate
- Original: 557 lines
- Adapted: ~400 lines (remove redundant validation that overlaps with BudSimulator's existing config validation)

---

### 2j. Router -- Request Routing

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/inference_serving/router.py`
**Total lines**: 93

#### What it does
Routes incoming requests to instances using configurable policies (Round-Robin, Random, Custom). Handles Prefill/Decode disaggregation by maintaining separate prefill and decode scheduler lists, with `transfer_prefill_request()` moving completed prefill requests to decode instances.

#### Key Classes and Functions

| Component | Lines | Description |
|-----------|-------|-------------|
| `Router.__init__()` | 7-36 | Initialize with schedulers, routing policy selection |
| `Router._rr_routing()` | 38-39 | Round-robin routing |
| `Router._rand_routing()` | 41-42 | Random routing |
| `Router.transfer_prefill_request()` | 47-51 | Move completed prefill requests to decode instances |
| `Router.generate()` | 54-93 | Load request dataset and distribute to instances |

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/router.py
```

#### Adaptation Needed

1. **Decouple from pandas**: The `generate()` method reads request data using `pd.read_json()`. Accept a list of request dicts instead.
2. **Add load-balancing policies**: The current "CUSTOM" policy raises `NotImplementedError`. Add least-loaded and shortest-queue policies.
3. **Make P/D disaggregation optional**: The prefill/decode separation is hardcoded. Make it configurable.

#### Dependencies
- `scheduler.py` (Scheduler instances)
- `pandas` (for request generation -- remove this dependency)
- `logger.py` (replace with standard logging)

#### Priority: P2

#### LOC Estimate
- Original: 93 lines
- Adapted: ~100 lines

---

## 3. vllm-tuner to BudSimulator Code Map

### 3a. Search Space Definition Pattern

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/src/optimization/search_space.py`
**Total lines**: 188

#### What it does
Defines a clean, extensible pattern for parameter search spaces with range parameters (continuous/integer), categorical parameters, conditional suggestion logic (e.g., TP only when num_gpus > 1), and validation. This pattern is reusable for BudSimulator's hardware recommendation engine which needs to search over hardware configurations.

#### Key Pattern

```python
class VLLMSearchSpace:
    DEFAULT_RANGES = {
        "batch_size": (1, 256, int),
        "max_num_batched_tokens": (2048, 32768, int),
        "gpu_memory_utilization": (0.6, 0.99, float),
    }
    DEFAULT_CATEGORICAL = {
        "tensor_parallel_size": [1, 2, 4, 8],
    }

    def should_suggest(self, param_name: str) -> bool:
        """Conditional suggestion based on hardware constraints."""

    def apply_params(self, trial, params: Dict) -> Dict:
        """Apply Optuna trial suggestions."""

    def validate_params(self, params: Dict) -> bool:
        """Validate parameter values against bounds."""
```

#### Target location in BudSimulator

```
BudSimulator/src/services/optimization/search_space.py
```

#### Adaptation Needed

1. **Replace vLLM parameters with BudSimulator parameters**: Change the search space to cover hardware configuration parameters:
   ```python
   DEFAULT_RANGES = {
       "num_nodes": (1, 64, int),
       "tensor_parallel": (1, 16, int),
       "pipeline_parallel": (1, 8, int),
       "batch_size": (1, 512, int),
       "sequence_length": (128, 32768, int),
   }
   DEFAULT_CATEGORICAL = {
       "precision": ["fp16", "bf16", "fp8", "int8"],
       "hardware": ["A100", "H100", "MI300X"],
   }
   ```

2. **Remove Optuna-specific `trial.suggest_*` calls**: The `apply_params()` method directly calls Optuna's trial API. Abstract this behind a generic optimizer interface so it works with BudSimulator's existing recommendation logic.

3. **Add hardware-specific constraints**: E.g., `tensor_parallel` cannot exceed `num_nodes * gpus_per_node`.

#### Dependencies
- `config.models.TuningConfig` (replace with BudSimulator config)
- No external packages required (Optuna is only needed in the optimizer, not the search space definition)

#### Priority: P1
Useful pattern for BudSimulator's hardware recommendation engine and parallelism strategy search.

#### LOC Estimate
- Original: 188 lines
- Adapted: ~150 lines (replace parameter definitions, simplify by removing Optuna-specific code)

---

### 3b. Study Management Pattern

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/src/tuner/study_manager.py`
**Total lines**: ~160

#### What it does
Orchestrates a tuning study lifecycle: load workload, create optimizer, run benchmark trials, collect results. The pattern of "create study -> run trials -> collect best" is applicable to BudSimulator's parallelism strategy optimization.

#### Reusable Pattern (not direct code)

The `StudyManager` demonstrates the orchestration pattern:
1. Initialize resources (workload, optimizer, GPU collector)
2. Define a `benchmark_func(params) -> metrics` callback
3. Call `optimizer.optimize(benchmark_func, n_trials, timeout)`
4. Collect and return results

#### WARNING: Critical bugs must be fixed first

From finding 01 (`01_core_source_analysis.md`):
- **Resource leak**: `StudyManager.run_study()` (line 54-75) has a try/finally that only stops the launcher but does not clean up the GPU collector, optimizer storage, or workload
- **Import path bug** (line 8): Uses `from src.config.models import TuningConfig` (absolute import) while all other files use relative imports, causing import failures when the package is installed
- **No cancellation support**: The `optimize()` call blocks without cancellation mechanism

#### Target location in BudSimulator

This is a **pattern reference**, not direct code reuse. BudSimulator's `get_best_parallelization_strategy()` in the performance API already implements a search loop. The pattern from vllm-tuner can inform improving that function's structure.

#### Priority: P1 (pattern only, no direct code port)

#### LOC Estimate
- Original: ~160 lines
- Adapted: 0 lines (pattern reference only; rewrite using BudSimulator's existing search infrastructure)

---

### 3c. GPU Telemetry Collection

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/src/profiling/gpu_collector.py`
**Total lines**: ~200

#### What it does
Collects real-time GPU metrics via NVIDIA NVML: memory usage, GPU/memory utilization, temperature, power usage/limit, SM/memory clocks. Uses an async polling pattern with configurable intervals and maintains a history for time-series analysis.

#### Key Classes

| Component | Lines | Description |
|-----------|-------|-------------|
| `GPUStats` | 16-45 | Data container for GPU metrics with `to_dict()` serialization |
| `GPUCollector.__init__()` | 51-54 | Initialize with optional device ID list |
| `GPUCollector.initialize()` | 56-80 | NVML initialization with device enumeration |
| `GPUCollector.collect_stats()` | (below line 80) | Collect current metrics for all monitored GPUs |

#### Target location in BudSimulator

```
BudSimulator/src/services/profiling/gpu_collector.py
```

#### Adaptation Needed

1. **Add graceful degradation**: The current code fails hard if NVML is not available. For BudSimulator (which may run on CPU-only machines for analysis), add:
   ```python
   NVML_AVAILABLE = False
   try:
       import pynvml
       NVML_AVAILABLE = True
   except ImportError:
       pass

   class GPUCollector:
       def initialize(self):
           if not NVML_AVAILABLE:
               logger.warning("NVML not available; GPU metrics disabled")
               return
   ```

2. **Add aggregation methods**: Add methods for computing average/peak utilization over a time window, which is useful for BudSimulator's hardware utilization analysis.

3. **Remove async pattern**: BudSimulator's backend is synchronous FastAPI (not async). Remove `async` from the collection loop and use threading instead.

#### Dependencies
- `pynvml` (NVIDIA NVML Python bindings)
- `asyncio` (remove in adaptation)

#### Priority: P2
Useful for real-time monitoring when BudSimulator is deployed alongside actual GPU hardware, but not needed for analytical simulation.

#### LOC Estimate
- Original: ~200 lines
- Adapted: ~160 lines (remove async, add graceful degradation)

---

### 3d. HTML Report Generation Pattern

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/src/reporting/html.py`
**Total lines**: ~300

#### What it does
Generates interactive HTML reports with Plotly charts showing optimization results, trial comparisons, and performance trade-offs. Uses Jinja2 templates for HTML structure.

#### WARNING: XSS vulnerability must be fixed

From finding 02 (`02_benchmarks_profiling_reporting_analysis.md`):
- The `_render_html()` method directly interpolates user-controlled data (study names, parameter values) into HTML without escaping. This enables Cross-Site Scripting (XSS) attacks when reports are served via a web interface.

#### Reusable Pattern

The pattern of generating self-contained HTML reports with embedded Plotly charts is valuable for BudSimulator's analysis output. Key reusable elements:
- Chart generation with Plotly's `graph_objects` and `subplots`
- Jinja2 template rendering with data injection
- Baseline comparison overlays

#### Target location in BudSimulator

```
BudSimulator/src/services/reporting/analysis_report.py
```

#### Adaptation Needed

1. **Fix XSS**: Use Jinja2's autoescape:
   ```python
   env = jinja2.Environment(
       loader=jinja2.FileSystemLoader(template_dir),
       autoescape=jinja2.select_autoescape(['html', 'xml'])  # FIX
   )
   ```

2. **Replace vLLM-specific charts**: Create charts for BudSimulator's output: roofline plots, memory breakdown, latency vs. throughput trade-offs, hardware comparison matrices.

3. **Add PDF export option**: BudSimulator's API serves reports; add PDF generation using `plotly.io.write_image()`.

#### Dependencies
- `plotly` (chart generation)
- `jinja2` (HTML templating)

#### Priority: P2

#### LOC Estimate
- Original: ~300 lines
- Adapted: ~250 lines (fix XSS, replace chart content, keep structure)

---

### 3e. Workload Abstraction

**Source file**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/src/benchmarks/workload.py`
**Total lines**: 35

#### What it does
Defines an abstract base class for benchmark workloads with lazy-loading pattern. Clean interface with `load()`, `get_prompts()`, `get_metadata()`, and `unload()` methods.

#### Full code:

```python
class Workload(ABC):
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self._prompts: Optional[List[str]] = None

    @abstractmethod
    async def load(self) -> List[str]:
        """Load prompts for the workload."""

    async def get_prompts(self) -> List[str]:
        if self._prompts is None:
            self._prompts = await self.load()
        return self._prompts

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""

    def unload(self) -> None:
        self._prompts = None
```

#### Target location in BudSimulator

```
llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/workload.py
```

#### Adaptation Needed

1. **Make synchronous**: Remove `async` since BudSimulator's simulation engine is synchronous.
2. **Extend interface**: Add methods for request arrival patterns (Poisson, trace-driven, burst):
   ```python
   class SimulationWorkload(ABC):
       @abstractmethod
       def generate_requests(self) -> List[RequestConfig]:
           """Generate request stream with arrival times."""

       @abstractmethod
       def get_arrival_pattern(self) -> str:
           """Return arrival pattern type: 'poisson', 'trace', 'burst'."""
   ```
3. **Add request generation**: LLMServingSim uses `router.py:generate()` to create requests from a JSON lines file. Combine the workload pattern with request generation.

#### Dependencies
- None (pure abstract class)

#### Priority: P2

#### LOC Estimate
- Original: 35 lines
- Adapted: ~80 lines (make sync, extend interface, add request generation)

---

## 4. Code That Should NOT Be Reused

### 4.1 LLMServingSim `main.py` Simulation Loop (lines 273-451)

**Why not reuse**: The 180-line while loop in `main.py` is a monolithic control flow that tightly couples scheduling, trace generation, graph generation, ASTRA-SIM IPC, throughput tracking, and power modeling. It uses global state (`current`, `total_prompt`, `gen_th`, etc.) and hardcoded control flow that cannot be decomposed without a full rewrite. The individual components (scheduler, controller, memory model) are reusable, but the orchestration loop must be written fresh for BudSimulator's architecture.

### 4.2 LLMServingSim `trace_generator.py` Core Synthesis Functions (lines 100-2354)

**Why not reuse directly**: At 2,254 lines, the synthesis functions are monolithic with deep nesting, hardware-specific branching, and tight coupling to the performance database format. The _pattern_ of per-layer trace generation is valuable, but the code itself needs complete refactoring. Specific issues:
- Functions like `_synthesize_trace()` are 500+ lines with no separation of concerns
- Global caches (`_perf_db_cache`, `_attn_perf_db_cache`, `_attn_predictor_cache`) are not thread-safe
- Hardcoded CSV file path patterns for performance database lookups
- Mixed concerns: trace writing, power accounting, attention prediction, memory operation insertion

### 4.3 vllm-tuner `study_manager.py` (full file)

**Why not reuse**: Critical bugs documented in finding 01:
- **Import path bug** (line 8): `from src.config.models import TuningConfig` uses absolute import that breaks when installed as a package
- **Resource leak**: GPU collector, optimizer storage, and workload are not cleaned up in the finally block
- **No cancellation support**: Blocking `optimize()` call with no way to cancel
- **Hardcoded Alpaca workload** (line 50): `self.workload = create_alpaca_workload(config.workload)` -- not pluggable
- The orchestration pattern is useful but the code should be rewritten from scratch.

### 4.4 vllm-tuner `VLLMTelemetryParser.parse_log_content()` (telemetry.py, lines 64-79)

**Why not reuse this method**: Bug documented in finding 01:
```python
# BUG: Overwrites hit_rate/miss_rate on each matching line instead of accumulating
match = self.PATTERNS["block_manager_stats"].search(line)
if match:
    hit_rate = float(match.group(1))
    miss_rate = float(match.group(2))
    self.metrics["cpu_cache_hit_rate"] = hit_rate    # OVERWRITES previous value
    self.metrics["cpu_cache_miss_rate"] = miss_rate   # OVERWRITES previous value
```
The attempted normalization at lines 71-78 divides by the last overwritten values, producing incorrect results. The regex patterns themselves (lines 16-33) are useful reference material for log parsing, but the accumulation logic must be rewritten.

### 4.5 vllm-tuner `VLLMOptimizer._run_objective()` Bridge Pattern

**Why not reuse**: Finding 01 documents a fragile async-to-sync bridge:
```python
# This pattern is error-prone:
loop = asyncio.new_event_loop()
result = loop.run_until_complete(benchmark_func(params))
loop.close()
```
Each trial creates a new event loop, which is expensive and can interfere with existing event loops. Use `asyncio.run()` or keep everything synchronous.

### 4.6 LLMServingSim `pim_model.py` (entire file, 140 lines)

**Why not reuse**: Processing-in-Memory (PIM) is a niche technology not currently relevant to BudSimulator's target hardware platforms (NVIDIA GPUs, AMD MI series). The PIM model adds complexity without benefit for the primary use cases. If PIM support is needed in the future, it can be added as an optional feature module.

### 4.7 LLMServingSim `attn_utils.py` (entire file, ~250 lines)

**Why not reuse**: Contains FlashAttention heuristics (attention kernel selection, block size determination) that are highly hardware-specific and tied to LLMServingSim's profiling database format. BudSimulator uses GenZ's analytical attention modeling which accounts for FlashAttention through its `flash_attention` feature flag. The heuristics would need complete revalidation for different hardware targets.

---

## 5. Adaptation Patterns

### 5.1 Common Pattern: Model Config Lookup Replacement

LLMServingSim uses a `get_config(model_name)` function that reads a JSON file from `model_config/<name>.json`. This is called in virtually every module. BudSimulator should provide an equivalent adapter:

```python
# Adapter pattern for BudSimulator
# Place in: llm_memory_calculator/genz/simulation/runtime/compat.py

from typing import Dict, Any

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Adapter: Convert BudSimulator model definition to LLMServingSim-compatible config dict.

    LLMServingSim expects:
    - hidden_size, num_hidden_layers, num_attention_heads
    - num_key_value_heads, vocab_size, intermediate_size
    - max_position_embeddings
    - num_local_experts (optional, for MoE)
    - num_experts_per_tok (optional, for MoE)
    """
    # Option 1: Load from GenZ model CSV definitions
    from ..Models import get_configs
    configs = get_configs(model_name)
    # Map GenZ model fields to LLMServingSim expected fields
    return {
        'hidden_size': configs.get('d_model', configs.get('hidden_size')),
        'num_hidden_layers': configs.get('num_layers', configs.get('num_hidden_layers')),
        'num_attention_heads': configs.get('n_head', configs.get('num_attention_heads')),
        'num_key_value_heads': configs.get('n_kv_head', configs.get('num_key_value_heads')),
        'vocab_size': configs.get('vocab_size', 32000),
        'intermediate_size': configs.get('d_ff', configs.get('intermediate_size')),
        'max_position_embeddings': configs.get('max_seq_len', 4096),
    }
```

### 5.2 Common Pattern: Logger Replacement

Every LLMServingSim module uses a custom logger from `logger.py` with node/instance context. Replace with standard Python logging using adapter pattern:

```python
# Replace this LLMServingSim pattern:
from .logger import get_logger
self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

# With this BudSimulator pattern:
import logging
logger = logging.getLogger(__name__)

# For node/instance context, use LoggerAdapter:
class NodeInstanceAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[node={self.extra['node_id']},inst={self.extra['instance_id']}] {msg}", kwargs

self.logger = NodeInstanceAdapter(logger, {'node_id': node_id, 'instance_id': instance_id})
```

### 5.3 Bridging Event-Driven Simulation with Analytical Modeling

The fundamental architectural difference between LLMServingSim (event-driven) and BudSimulator (analytical) requires a bridge pattern for hybrid mode:

```python
class HybridSimulator:
    """Bridge between GenZ analytical engine and runtime simulation components."""

    def __init__(self, system: System, model_config: dict):
        self.system = system
        self.analytical_engine = SimulationEngine()  # existing GenZ engine
        self.memory_model = MemoryModel(model_config, ...)  # ported from LLMServingSim
        self.scheduler = Scheduler(model_config, ...)  # ported from LLMServingSim
        self.statistics = StatisticsCollector()
        # Optional: persistent ASTRA-SIM for network timing
        self.controller = None  # lazily initialized

    def estimate_batch_time(self, batch: Batch) -> float:
        """Use GenZ analytical engine for compute timing."""
        config = SimulationConfig(
            model=batch.model,
            features=['prefill' if batch.num_prefill > 0 else 'decode'],
            simulation_params={
                'batch_size': len(batch.requests),
                'input_sequence_length': batch.total_len,
                'output_sequence_length': 1,
            }
        )
        result = self.analytical_engine.simulate(config)
        return result.latency

    def estimate_collective_time(self, collective_type: str, size_bytes: int) -> float:
        """Use ASTRA-SIM for network timing (or fallback to analytical)."""
        if self.controller and self.controller.is_alive():
            # Persistent ASTRA-SIM path
            return self._run_astra_sim_collective(collective_type, size_bytes)
        else:
            # Fallback to GenZ analytical collective timing
            from ..collective_times import get_collective_time
            return get_collective_time(collective_type, size_bytes, self.system)

    def run_simulation(self, workload: SimulationWorkload) -> SimulationResult:
        """Main simulation loop using hybrid timing."""
        requests = workload.generate_requests()
        for req in requests:
            self.scheduler.add_request(req)

        current_time = 0
        while not self.scheduler.is_request_empty():
            batch = self.scheduler.schedule(current_time, 0)
            if batch:
                batch_time = self.estimate_batch_time(batch)
                current_time += batch_time
                self.scheduler.add_done(batch.batch_id, 0, current_time)
                self.statistics.record_throughput_sample(current_time, ...)
            else:
                # Advance to next request arrival
                current_time = self._next_event_time()

        return self._build_result()
```

### 5.4 Type System Alignment

| LLMServingSim Type | GenZ/BudSimulator Equivalent | Mapping |
|-------------------|------------------------------|---------|
| `Device` enum (NPU/CPU/CXL) | No direct equivalent | Create new; NPU maps to `off_chip_mem`, CPU maps to host memory |
| `MemoryModel` | Static checks in `llm_prefill.py` | Port MemoryModel as new runtime class |
| `Request` | No equivalent | Port as new data class |
| `Batch` | Implicit in `batch_size` parameter | Port as new data class |
| `Controller` | `get_astra_sim_time.py` (fire-and-forget) | Replace with persistent controller |
| `Scheduler` | No equivalent | Port as new class |
| `PowerModel` | `power.py:get_energy()` (basic) | Extend with 7-component model |
| `RadixCache` | No equivalent | Port as new class |
| Model config dict | GenZ model definitions (CSV/Python) | Create adapter (see 5.1) |
| `calculate_sizes()` function | GenZ `operators.py` + `operator_base.py` | Use GenZ operators; keep `calculate_sizes()` for validation |

### 5.5 Memory Capacity Units

LLMServingSim uses bytes internally with GB_TO_BYTE conversion constants. BudSimulator's GenZ `Unit` class uses a different unit system. Standardize at the boundary:

```python
# At the boundary between ported code and GenZ code:
from llm_memory_calculator.genz.unit import Unit

def bytes_to_genz_mem(bytes_val: int) -> float:
    """Convert bytes to GenZ memory unit (MB by default)."""
    return bytes_val / (1024 * 1024)  # bytes to MB

def genz_mem_to_bytes(mem_val: float) -> int:
    """Convert GenZ memory unit to bytes."""
    return int(mem_val * 1024 * 1024)
```

---

## 6. Dependency Graph

### 6.1 Module Dependency Diagram

```
                    +-----------------+
                    |  Request/Batch   |  <-- P0: No dependencies
                    |    (request.py)  |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +------------+  +--------------+  +---------------+
     | RadixCache  |  | MemoryModel  |  |  Statistics   |
     |(radix_tree) |  |(memory_model)|  |  Collector    |
     |   P1        |<-|    P0        |  |    P1         |
     +------+-----+  +------+-------+  +---------------+
            |                |
            |                |
            v                v
     +----------------------------+    +------------------+
     |    Scheduler               |    |   Controller     |
     |  (scheduler.py)            |    | (controller.py)  |
     |    P1                      |    |    P0            |
     +--------------+-------------+    +--------+---------+
                    |                            |
                    |                            |
                    v                            v
     +------------------------------------------------+
     |           HybridSimulator / RuntimeSimulator     |
     |    (combines scheduler + controller + memory)    |
     |                   P1                             |
     +-----------------------+------------------------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +--------------+  +----------+  +--------------+
     |  PowerModel   |  |  Router  |  |TraceGenerator|
     |    P2         |  |   P2     |  |     P2       |
     +--------------+  +----------+  +------+-------+
                                            |
                                            v
                                   +------------------+
                                   |  GraphGenerator   |
                                   |      P2           |
                                   +------------------+
```

### 6.2 Recommended Integration Order

**Phase 1: Foundation (Week 1-2)**
1. Port `Request` and `Batch` data structures (P0, no dependencies, 3 days)
2. Port `Controller` for persistent ASTRA-SIM IPC (P0, no dependencies, 1 week)
3. Port `MemoryModel` core (P0, depends on Request, 2 weeks overlapping with #1-2)

**Phase 2: Core Features (Week 3-4)**
4. Port `RadixCache` for prefix caching (P1, depends on MemoryModel)
5. Port `Scheduler` for continuous batching (P1, depends on MemoryModel + Request)
6. Create `StatisticsCollector` (P1, depends on Request)

**Phase 3: Integration (Week 5-6)**
7. Wire `HybridSimulator` connecting GenZ analytical engine + ported runtime components
8. Extend `SimulationConfig` with `simulation_mode` field
9. Extend `SimulationResult` with serving metrics

**Phase 4: Extended Features (Week 7-8)**
10. Port `PowerModel` (P2)
11. Port `Router` for multi-instance scenarios (P2)
12. Port/refactor `TraceGenerator` for full ASTRA-SIM simulation mode (P2)
13. Port `ClusterConfigBuilder` for multi-node configs (P2)

### 6.3 Minimal Viable Integration

For a minimal viable integration that provides continuous batching simulation with hybrid timing, only the following are needed:

| Module | Source | Adapted LOC |
|--------|--------|-------------|
| Request/Batch | LLMServingSim | ~120 |
| MemoryModel (core, no prefix cache) | LLMServingSim | ~300 |
| Scheduler (schedule_base only) | LLMServingSim | ~250 |
| StatisticsCollector | New (pattern from LLMServingSim) | ~200 |
| HybridSimulator | New | ~300 |
| **Total** | | **~1,170** |

This minimal set enables BudSimulator to simulate continuous batching workloads with analytical compute timing and dynamic memory tracking, producing TTFT/TPOT/ITL distributions -- without requiring ASTRA-SIM, prefix caching, or multi-instance support.

---

## 7. License and Attribution

### 7.1 LLMServingSim

**License**: MIT License
**Copyright**: (c) 2024, CASYS Lab at KAIST
**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/LLMServingSim/LICENSE`

**Compatibility with BudSimulator**: MIT is permissive. Code can be freely adapted, modified, and distributed. Only requirement is to include the MIT copyright notice and permission notice in derived works.

**Required Attribution**:
```
Portions of this software are adapted from LLMServingSim 2.0
Copyright (c) 2024, CASYS Lab at KAIST
Licensed under the MIT License
```

**Special note on radix_tree.py**: The RadixCache implementation carries its own Apache 2.0 license header (SGLang Team, lines 3-16 of radix_tree.py). This requires:
1. Include Apache 2.0 license copy
2. State changes made to the original code
3. Retain all copyright notices

### 7.2 vllm-tuner

**License**: Apache License, Version 2.0
**File**: `/home/bud/Desktop/bud_model_factory/bud_simulator/research/repo/vllm-tuner/LICENSE`

**Compatibility with BudSimulator**: Apache 2.0 is permissive but with additional requirements compared to MIT:
1. Must include a copy of the Apache 2.0 license
2. Must state changes made to original files
3. Must retain all copyright, patent, trademark notices
4. Must include NOTICE file contents if one exists

**Required Attribution**:
```
Portions of this software are adapted from vllm-tuner
Licensed under the Apache License, Version 2.0
Original source: [vllm-tuner repository URL]
```

### 7.3 License Compatibility Matrix

| BudSimulator License | + MIT (LLMServingSim) | + Apache 2.0 (vllm-tuner) | + Apache 2.0 (SGLang RadixTree) |
|---------------------|----------------------|--------------------------|-------------------------------|
| MIT | Compatible | Compatible (add Apache notice) | Compatible (add Apache notice) |
| Apache 2.0 | Compatible | Compatible | Compatible |
| Proprietary | Compatible (include MIT notice) | Compatible (include Apache notice + state changes) | Compatible (include Apache notice + state changes) |

### 7.4 Recommended Action

Create a `THIRD_PARTY_NOTICES.md` file in the BudSimulator repository root listing all adapted code with:
- Source project name and URL
- License type
- Copyright holder
- Description of what was adapted
- Location of adapted code in BudSimulator

---

## Appendix: File Size Reference

### LLMServingSim Source Files (6,929 total LOC)

| File | LOC | Reusable | Adapted LOC |
|------|-----|----------|-------------|
| `trace_generator.py` | 2,354 | Partially (pattern) | ~1,200 |
| `memory_model.py` | 752 | Yes | ~550 |
| `scheduler.py` | 715 | Yes | ~500 |
| `radix_tree.py` | 597 | Yes | ~550 |
| `config_builder.py` | 557 | Partially | ~400 |
| `main.py` | 545 | No (monolithic) | 0 |
| `power_model.py` | 214 | Yes | ~180 |
| `attn_utils.py` | ~250 | No (hardware-specific) | 0 |
| `logger.py` | ~200 | No (replace with stdlib) | 0 |
| `utils.py` | 181 | Partially (get_config only) | ~30 |
| `router.py` | 93 | Yes | ~100 |
| `request.py` | 85 | Yes | ~120 |
| `controller.py` | 58 | Yes | ~100 |
| `graph_generator.py` | 37 | Yes | ~40 |
| `gate_function.py` | ~50 | No (MoE-specific niche) | 0 |
| `pim_model.py` | ~140 | No (PIM-specific niche) | 0 |

### vllm-tuner Source Files (4,384 total LOC, src/ only)

| File | LOC | Reusable | Adapted LOC |
|------|-----|----------|-------------|
| `optimization/search_space.py` | 188 | Yes (pattern) | ~150 |
| `profiling/gpu_collector.py` | ~200 | Yes | ~160 |
| `reporting/html.py` | ~300 | Partially (fix XSS) | ~250 |
| `tuner/study_manager.py` | ~160 | No (bugs, pattern only) | 0 |
| `tuner/optimizer.py` | ~403 | No (bugs in bridge) | 0 |
| `benchmarks/workload.py` | 35 | Yes (pattern) | ~80 |
| `vllm/telemetry.py` | 210 | Partially (regex useful) | ~100 |
| `config/models.py` | ~350 | No (vllm-specific) | 0 |
| `config/validation.py` | ~200 | No (vllm-specific) | 0 |
| `vllm/launcher.py` | ~350 | No (vllm-specific) | 0 |
| `baseline/runner.py` | ~200 | No (vllm-specific) | 0 |
| `benchmarks/alpaca.py` | ~150 | No (dataset-specific) | 0 |
| `benchmarks/request_generator.py` | ~200 | No (vllm-specific) | 0 |
| Others | ~938 | No | 0 |
