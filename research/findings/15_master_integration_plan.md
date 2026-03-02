# Master Integration Plan: BudSimulator Enhancement from Research Findings

**Document:** 15 of 16 -- Definitive Integration Plan
**Date:** 2026-02-28
**Status:** Final
**Scope:** Comprehensive, phased plan for incorporating all research findings (01-14, 16) into BudSimulator
**Audience:** BudSimulator development team, engineering leadership

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Integration Architecture](#3-integration-architecture)
4. [Phased Implementation Roadmap](#4-phased-implementation-roadmap)
5. [Backward Compatibility Strategy](#5-backward-compatibility-strategy)
6. [New API Endpoints Design](#6-new-api-endpoints-design)
7. [SDK Extension Design](#7-sdk-extension-design)
8. [Data Flow Architecture](#8-data-flow-architecture)
9. [Risk Assessment and Mitigations](#9-risk-assessment-and-mitigations)
10. [Success Metrics](#10-success-metrics)

---

## 1. Executive Summary

### What BudSimulator Will Gain

This integration plan transforms BudSimulator from a **static analytical roofline calculator** into a **full-spectrum LLM serving simulation platform**. Upon completion, BudSimulator will offer:

1. **Dynamic serving simulation** -- continuous batching, KV cache management with eviction, request queue modeling, and per-request metrics (TTFT, TPOT, ITL distributions)
2. **Multi-tier memory hierarchy** -- device HBM, host DRAM, CXL pooled memory, and storage, with block-based KV cache allocation and prefix caching via radix trees
3. **7-component power model** -- accelerators (3-state: idle/active/standby), CPUs, DRAM, interconnect, NICs, storage, and other, targeting 1.34% energy accuracy
4. **Bayesian optimization engine** -- HEBO/GP+MACE-based configuration optimization replacing brute-force parallelism search, with hidden constraint learning and multi-objective Pareto optimization
5. **Prefill-decode disaggregation** -- M:N mapping of prefill and decode device pools with KV transfer cost modeling
6. **Persistent ASTRA-SIM integration** -- bidirectional IPC with execution DAGs replacing fire-and-forget single-collective subprocess calls
7. **Cluster topology optimization** -- multi-node, multi-instance cluster configuration with TCO-aware hardware selection
8. **Speculative decoding serving model** -- draft+target serving throughput under realistic workloads (extending the existing analytical model)

### Integration Scope and Effort

| Metric | Value |
|--------|-------|
| Total estimated effort | 18-26 weeks (4 phases) |
| New Python modules | ~15 modules (~8,000-12,000 lines) |
| Modified existing modules | ~12 modules (backward-compatible changes) |
| New API endpoints | 12-15 endpoints |
| New SDK classes | 7 classes |
| Existing API endpoints preserved | All 35+ endpoints, zero breaking changes |
| Existing SDK functions preserved | All exported functions, signatures unchanged |

### Source Research Mapped to Outcomes

| Research Finding | Key Contribution to Integration |
|---|---|
| 01: Core Source Analysis | GenZ operator framework understanding, extension point identification |
| 06: LLMServingSim 2.0 Deep Analysis | Runtime simulation loop design, power model spec, P/D disaggregation architecture |
| 07: LLMServingSim 2.0 Code Analysis | Reusable code identification (12 modules), bug awareness for adaptation |
| 08: Advanced Optimization Techniques | HEBO/MACE optimization framework, multi-fidelity methods, simulation-in-loop design |
| 09: Microbenchmarking Analysis | Metric definitions (TTFT/TPOT/ITL), statistical rigor requirements |
| 10: Cross-Reference Gaps vs Solutions | Gap-to-solution mapping for 25 identified gaps |
| 11: GenZ Core Engine Analysis | Extension points, gap analysis vs LLMServingSim and SCOOT |
| 12: API/SDK Analysis | Complete API inventory, backward compatibility constraints, new endpoint designs |
| 13: ASTRA-SIM/Chakra Analysis | Persistent IPC design, hybrid simulation architecture |
| 14: Hardware/Training/Validation Analysis | Hardware layer extensions, training subsystem integration, validation framework |
| 16: Reusable Code Mapping | 12 modules with adaptation plans, deduplication analysis |

---

## 2. Current State Assessment

### 2.1 GenZ Core Engine -- Analytical Roofline Model

**Location:** `llm-memory-calculator/src/llm_memory_calculator/genz/`

**What it does well:**
- Comprehensive operator-level roofline analysis with tensor core efficiency modeling (padding waste, wave quantization) in `operator_base.py`
- 36 precision formats (int2 through fp32, mixed precision, QLoRA/GEAR) in `system.py`
- Flexible parallelism: TP, PP, DP, EP, SP, CP, ZeRO stages 0-3 in `parallelism.py`
- 50+ pre-configured models plus HuggingFace dynamic loading via `Models/` directory
- Extensive architecture support: MHA, GQA, MQA, MLA (DeepSeek V2/V3), Mamba/SSM, MoE, heterogeneous layers
- Scale-aware collective communication with hierarchical AllReduce in `collective_times.py` (~900 lines)
- Speculative decoding analytical model in `llm_spec_decode.py`
- Training simulation with 13 RLHF stage types via `LLM_training/`
- CPU-specific modeling with cache hierarchy, NUMA, ISA selection via `cpu/`

**What it cannot do:**
- No temporal simulation -- computes a single-pass analytical result, no queue dynamics
- No batch scheduling -- fixed batch size, no continuous batching, no dynamic request management
- No KV cache management -- static capacity check only, no eviction, no multi-tier caching, no prefix caching
- No power model sophistication -- single utilization-proportional formula (4 components, no temporal tracking) in `power.py` (38 lines)
- No contention modeling -- memory bandwidth and network contention between concurrent operations are not modeled
- No per-request metrics -- aggregate latency/throughput only, no TTFT/TPOT/ITL distributions

### 2.2 BudSimulator Web Application -- API Layer

**Location:** `BudSimulator/apis/` and `BudSimulator/src/`

**Current API surface: 35+ endpoints across 5 routers:**

| Router | Prefix | Endpoints | Status |
|--------|--------|-----------|--------|
| Models | `/api/models` | 11 endpoints | Fully functional |
| Hardware | `/api/hardware` | 9 endpoints | Fully functional |
| Usecases | `/api/usecases` | 12 endpoints | Fully functional |
| Usecase Optimization | `/api/usecases` | 2 endpoints | Fully functional |
| Training/Simulator | `/api/simulator` | 5 endpoints | Fully functional |
| Health | `/api/health` | 3 endpoints | Fully functional |

**Critical gap: BudSimulator core class has ZERO API endpoints.**

The `BudSimulator` class in `BudSimulator/src/bud_sim.py` is fully implemented with 8 simulation types (`SimType` enum) and `SimulationEngine` integration, but no router exposes it. The class wraps GenZ's `SimulationEngine.simulate()`, converts results to dict, and returns `{latency, throughput, runtime_breakdown, memory_usage, hardware_utilization, feature_metrics}`. This is the single largest gap -- the core simulation capability is unreachable from the frontend.

**Secondary gap: Heuristic vs GenZ performance estimation.**

The `usecases.py` router uses a heuristic-based `_estimate_performance()` function with hardcoded multipliers instead of calling GenZ's actual `estimate_prefill_performance()` / `estimate_decode_performance()`. Only the `usecases_optimization.py` router calls the real GenZ-backed `find_best_hardware_for_usecase()`. This means basic recommendations give rough estimates while optimization gives accurate results.

### 2.3 ASTRA-SIM Integration -- Fire-and-Forget

**Location:** `llm-memory-calculator/src/llm_memory_calculator/genz/Astra_sim/get_astra_sim_time.py`

**Current implementation:**
- Single function `get_astrasim_collective_time()` that writes a Chakra text trace for ONE collective operation, converts to ET format, runs ASTRA-SIM binary via `subprocess.run()` (blocking), and parses cycle counts from stdout
- Hardcoded path to `system.json` at line 210 pointing to a developer's machine (`/home/abhimanyu/...`)
- Uses `/tmp/genz/chakra/` for temporary files without cleanup
- No support for overlapping compute+communication or multi-operation execution graphs
- `path_utils.py` provides security validation but is NOT currently used by `get_astra_sim_time.py`

**Contrast with LLMServingSim 2.0:**
- Persistent `subprocess.Popen()` with stdin/stdout IPC via `controller.py`
- Per-batch execution graphs (not single-collective)
- Bidirectional communication: Python writes workloads, ASTRA-SIM returns cycle counts
- Multi-iteration simulation loop orchestrated around ASTRA-SIM's tick-based timing

### 2.4 Existing Subsystems That Are Ready for Extension

| Subsystem | Location | Extension Readiness |
|---|---|---|
| Simulation Engine | `genz/simulation/engine.py` | HIGH -- facade pattern with feature hooks, just needs new simulation modes |
| Feature Registry | `genz/features/registry.py` | HIGH -- auto-discovery, decorator-based registration, compatibility checking |
| Training Calculator | `training/calculator.py` | HIGH -- clean architecture, well-tested, easily extended |
| Cluster Optimizer | `training/cluster_optimizer.py` | MEDIUM -- has Pareto optimization, needs serving workload integration |
| TCO Calculator | `training/tco_calculator.py` | HIGH -- 7 cloud providers, easily extended to inference TCO |
| Validation Framework | `validation/` | MEDIUM -- calibration engine exists, needs serving metric support |
| Hardware Configs | `hardware/configs.py` | HIGH -- 50+ entries, just needs new fields (power, interconnect topology) |

---

## 3. Integration Architecture

### 3.1 Layered Architecture

The enhanced BudSimulator follows a five-layer architecture. Each layer builds on the one below it. The foundational principle is: **GenZ analytical modeling remains the fast path; new dynamic simulation is an optional detailed path. Existing behavior is never changed, only extended.**

```
+================================================================+
|                    LAYER 5: FRONTEND                            |
|  (React UI -- enhanced with new visualizations)                |
|                                                                |
|  New panels: Serving simulation dashboard, power breakdown,    |
|  KV cache visualization, cluster topology view, optimization   |
|  progress tracker                                              |
+================================================================+
        |                                                  |
        v                                                  v
+================================================================+
|                    LAYER 4: API LAYER                           |
|  (FastAPI -- enhanced with new routers)                        |
|                                                                |
|  EXISTING (preserved):                                         |
|    /api/models/*     (11 endpoints)                            |
|    /api/hardware/*   (9 endpoints)                             |
|    /api/usecases/*   (14 endpoints)                            |
|    /api/simulator/*  (5 endpoints)                             |
|    /api/health/*     (3 endpoints)                             |
|                                                                |
|  NEW:                                                          |
|    /api/v2/simulate/*     (serving simulation)                 |
|    /api/v2/optimize/*     (Bayesian optimization)              |
|    /api/v2/power/*        (power/energy modeling)              |
|    /api/v2/cluster/*      (cluster topology)                   |
+================================================================+
        |                                                  |
        v                                                  v
+================================================================+
|                    LAYER 3: SERVICE LAYER                       |
|  (BudSimulator core -- enhanced)                               |
|                                                                |
|  EXISTING (preserved):                                         |
|    BudSimulator class (bud_sim.py)                             |
|    BudModels, BudHardware, BudUsecases                         |
|    HardwareOptimizer, HardwareRecommendation                   |
|                                                                |
|  NEW:                                                          |
|    ServingSimulator (orchestrates runtime simulation)           |
|    ConfigOptimizer (Bayesian optimization wrapper)             |
|    PowerAnalyzer (power/energy analysis service)               |
|    ClusterDesigner (cluster topology service)                  |
+================================================================+
        |                         |                        |
        v                         v                        v
+================================================================+
|               LAYER 2: SIMULATION ENGINE                       |
|  (GenZ core + new dynamic simulation layer)                    |
|                                                                |
|  EXISTING (preserved as-is):                                   |
|    SimulationEngine (analytical mode)                          |
|    Operator framework (roofline analysis)                      |
|    Parallelism management + collective timing                  |
|    Model definition system + HuggingFace loader                |
|    prefill_moddeling / decode_moddeling / chunked_moddeling   |
|                                                                |
|  NEW (simulation/runtime/ package):                            |
|    RuntimeSimulator (discrete-event loop)                      |
|    BatchScheduler (continuous batching)                        |
|    MemoryModel (multi-tier: NPU/CPU/CXL)                      |
|    RadixCache (prefix caching with LRU eviction)               |
|    RequestManager (request lifecycle tracking)                 |
|    PowerModel (7-component temporal power)                     |
|                                                                |
|  NEW (optimization/ package):                                  |
|    BayesianOptimizer (HEBO/GP wrapper)                         |
|    ConstraintLearner (random forest feasibility)               |
|    SearchSpace (parameter space with known constraints)        |
+================================================================+
        |                         |                        |
        v                         v                        v
+================================================================+
|               LAYER 1: INFRASTRUCTURE                          |
|                                                                |
|  EXISTING (enhanced):                                          |
|    System class (+ host_mem_size, cxl_mem_size, power params)  |
|    ASTRA-SIM integration (+ persistent subprocess option)      |
|    Hardware configs (+ power, topology fields)                 |
|    Database (+ new tables for simulation results, cache)       |
|                                                                |
|  NEW:                                                          |
|    AstraSimController (persistent IPC)                         |
|    TraceGenerator (per-batch execution traces)                 |
|    ClusterConfig (multi-node, multi-instance)                  |
+================================================================+
```

### 3.2 How the Layers Interact

**Analytical path (existing, fast -- milliseconds):**
```
API request -> BudSimulator.run() -> SimulationEngine.simulate()
    -> GenZ prefill/decode_moddeling() -> Operator roofline analysis
    -> SimulationResult {latency, throughput, runtime_breakdown}
```

**Dynamic simulation path (new, detailed -- seconds to minutes):**
```
API request -> ServingSimulator.simulate_serving()
    -> ClusterConfig.build() -> BatchScheduler[].initialize()
    -> MemoryModel[].initialize() -> RequestManager.generate_arrivals()
    -> SIMULATION LOOP:
        for each scheduling cycle:
            1. BatchScheduler selects batch (memory-aware)
            2. For compute operators: GenZ roofline analysis (analytical)
            3. For communication: ASTRA-SIM persistent subprocess OR GenZ analytical
            4. PowerModel.track_power(cycle_time, utilization)
            5. MemoryModel.update_state(kv_allocations, evictions)
            6. RequestManager.update_metrics(TTFT, TPOT, ITL)
    -> ServingSimulationResult {
        per_request_metrics, throughput_over_time, power_over_time,
        memory_over_time, cache_hit_rates, SLO_compliance
    }
```

**Hybrid path (new, balanced -- seconds):**
```
API request -> ServingSimulator.simulate_serving(mode="hybrid")
    -> Uses GenZ analytical for compute timing (no ASTRA-SIM)
    -> Uses BatchScheduler + MemoryModel for runtime dynamics
    -> Uses PowerModel for energy estimation
    -> Faster than full simulation, more accurate than pure analytical
```

### 3.3 New Module Hierarchy

```
llm-memory-calculator/src/llm_memory_calculator/
  genz/
    simulation/
      engine.py                    # MODIFIED: add simulation_mode parameter
      config.py                    # MODIFIED: add simulation_mode, cluster_config
      results.py                   # MODIFIED: add serving_metrics field
      runtime/                     # NEW PACKAGE
        __init__.py
        simulator.py               # RuntimeSimulator class (main orchestrator)
        batch_scheduler.py         # Continuous batching scheduler
        memory_model.py            # Multi-tier dynamic memory model
        radix_cache.py             # Radix tree prefix cache
        request_manager.py         # Request lifecycle + arrival modeling
        power_model.py             # 7-component temporal power model
        astra_controller.py        # Persistent ASTRA-SIM IPC
        trace_generator.py         # Per-batch execution trace synthesis
        cluster_config.py          # Multi-node cluster configuration
        types.py                   # Shared types (Device enum, Request, Batch)

    optimization/                  # NEW PACKAGE
      __init__.py
      bayesian_optimizer.py        # HEBO/GP-based config optimization
      constraint_learner.py        # Random forest feasibility predictor
      search_space.py              # Parameter space with known constraints
      multi_objective.py           # EHVI / NSGA-II for Pareto optimization

  # Existing packages -- extended, not replaced:
  hardware/
    configs.py                     # MODIFIED: add power_tdp, power_idle, topology fields
  training/
    tco_calculator.py              # MODIFIED: extend to inference TCO
  validation/
    benchmark_validator.py         # MODIFIED: add serving metric validation rules

BudSimulator/
  apis/
    routers/
      simulation.py                # NEW: expose BudSimulator + serving simulation
      optimization.py              # NEW: Bayesian optimization endpoints
      power.py                     # NEW: power/energy analysis endpoints
      cluster.py                   # NEW: cluster topology endpoints
    schemas/
      simulation_schemas.py        # NEW: Pydantic models for simulation APIs
      optimization_schemas.py      # NEW: Pydantic models for optimization APIs
  src/
    serving_simulator.py           # NEW: ServingSimulator service class
    config_optimizer.py            # NEW: ConfigOptimizer service class
    power_analyzer.py              # NEW: PowerAnalyzer service class
    cluster_designer.py            # NEW: ClusterDesigner service class
```

---

## 4. Phased Implementation Roadmap

### Phase 1: Foundation (Weeks 1-6)

**Goal:** Establish the infrastructure that all subsequent phases depend on. Expose existing hidden capabilities via API. Build the memory and power models.

#### 1.1 Expose BudSimulator via API (Week 1)

**What:** Create `BudSimulator/apis/routers/simulation.py` to expose the existing `BudSimulator` class that already has full `SimulationEngine` integration but zero API endpoints.

**Files to create:**
- `BudSimulator/apis/routers/simulation.py` -- new router
- `BudSimulator/apis/schemas/simulation_schemas.py` -- Pydantic schemas

**Files to modify:**
- `BudSimulator/apis/main.py` -- add `app.include_router(simulation.router)`

**Endpoints:**

```python
# POST /api/v2/simulate/run
# Exposes BudSimulator.run() which already delegates to GenZ SimulationEngine
class SimulationRunRequest(BaseModel):
    model: str                                    # Model name or HuggingFace ID
    simulation_type: str = "prefill"              # Maps to SimType enum
    batch_size: int = 1
    input_tokens: int = 2048
    output_tokens: int = 128
    system_name: str = "A100_80GB_SXM"
    bits: str = "bf16"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    features: list[str] = []                      # GenZ features to enable

class SimulationRunResponse(BaseModel):
    latency_ms: float
    throughput_tokens_per_sec: float
    runtime_breakdown: dict
    memory_usage: Optional[dict] = None
    hardware_utilization: Optional[dict] = None
    feature_metrics: Optional[dict] = None

# GET /api/v2/simulate/features
# Exposes BudSimulator.get_supported_features()
class SimulationFeaturesResponse(BaseModel):
    features: list[dict]

# POST /api/v2/simulate/batch
# Run multiple simulations in parallel
class BatchSimulationRequest(BaseModel):
    simulations: list[SimulationRunRequest]

# POST /api/v2/simulate/compare
# Compare configurations side-by-side
class CompareRequest(BaseModel):
    model: str
    configurations: list[dict]  # Each dict has system_name, tp, pp, bits, etc.
    batch_size: int = 1
    input_tokens: int = 2048
    output_tokens: int = 128
```

**Implementation detail:** The `BudSimulator` class already handles all the GenZ delegation. The router only needs to:
1. Validate the request via Pydantic
2. Map `simulation_type` string to `SimType` enum
3. Call `BudSimulator(sim_type=sim_type).run(**params)`
4. Format the response

**Effort:** 2-3 days. This is the highest ROI task in the entire plan.

#### 1.2 Multi-Tier Memory Model (Weeks 1-2)

**What:** Implement a dynamic memory model that tracks device/host/CXL memory across simulation iterations. This replaces GenZ's static "does it fit?" check with runtime-aware memory state tracking.

**New file:** `llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/memory_model.py`

**Source:** Adapted from LLMServingSim `inference_serving/memory_model.py` (lines 1-541) per Finding 16, Module 1.

**Key classes:**

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class MemoryTier(Enum):
    """Memory hierarchy tiers."""
    DEVICE = "device"     # GPU HBM / accelerator memory
    HOST = "host"         # CPU DRAM
    CXL = "cxl"           # CXL-attached pooled memory
    STORAGE = "storage"   # NVMe/SSD (last resort)

@dataclass
class MemoryTierConfig:
    """Configuration for a single memory tier."""
    tier: MemoryTier
    capacity_bytes: int
    bandwidth_bytes_per_sec: float
    latency_ns: float = 0.0
    shared: bool = False          # True for CXL shared pools

@dataclass
class MemoryState:
    """Current state of a memory tier."""
    tier: MemoryTier
    capacity_bytes: int
    used_bytes: int = 0
    weight_bytes: int = 0         # Model weights (static after init)
    kv_cache_bytes: int = 0       # KV cache (dynamic)
    prefix_cache_bytes: int = 0   # Prefix cache (dynamic)

    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    @property
    def utilization(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

class MemoryModel:
    """Multi-tier dynamic memory model for LLM inference serving.

    Tracks memory state across device/host/CXL tiers. Supports:
    - Block-based KV cache allocation and eviction
    - Weight loading and placement
    - Prefix cache integration with RadixCache
    - Memory pressure detection and graceful degradation

    Integration with GenZ: Uses GenZ System.off_chip_mem_size for device
    capacity and existing memory estimation formulas for weight calculation.
    """

    def __init__(
        self,
        device_memory_bytes: int,
        host_memory_bytes: int = 0,
        cxl_memory_bytes: int = 0,
        device_bandwidth_bytes_per_sec: float = 0,
        host_bandwidth_bytes_per_sec: float = 0,
        cxl_bandwidth_bytes_per_sec: float = 0,
        block_size: int = 16,   # KV cache block size in tokens
    ):
        # ... initialization of per-tier MemoryState objects
        pass

    @classmethod
    def from_genz_system(cls, system: 'System', host_mem_gb: float = 0,
                         cxl_mem_gb: float = 0) -> 'MemoryModel':
        """Create MemoryModel from an existing GenZ System object.

        This is the primary integration point with the existing codebase.
        Uses System.off_chip_mem_size for device memory and
        System.offchip_mem_bw for device bandwidth.
        """
        pass

    def load_weights(self, weight_bytes: int, tier: MemoryTier = MemoryTier.DEVICE) -> bool:
        """Load model weights into the specified tier. Returns success."""
        pass

    def allocate_kv(self, num_tokens: int, num_layers: int, kv_dim: int,
                    bytes_per_element: float, tp: int = 1) -> bool:
        """Allocate KV cache for a request. Returns success."""
        pass

    def free_kv(self, num_tokens: int, num_layers: int, kv_dim: int,
                bytes_per_element: float, tp: int = 1) -> int:
        """Free KV cache. Returns bytes freed."""
        pass

    def evict_to_lower_tier(self, bytes_to_free: int,
                            from_tier: MemoryTier = MemoryTier.DEVICE) -> int:
        """Evict KV blocks to a lower tier. Returns bytes freed on from_tier."""
        pass

    def is_available(self, bytes_needed: int,
                     tier: MemoryTier = MemoryTier.DEVICE) -> bool:
        """Check if memory is available on the specified tier."""
        pass

    def get_state(self) -> dict[MemoryTier, MemoryState]:
        """Get current memory state across all tiers."""
        pass

    def get_transfer_latency(self, bytes_to_transfer: int,
                             from_tier: MemoryTier,
                             to_tier: MemoryTier) -> float:
        """Compute latency (seconds) to transfer data between tiers."""
        pass
```

**Integration with existing code:**
- `from_genz_system()` factory method bridges GenZ `System` class to the new memory model
- Does NOT modify `System` class -- wraps it
- Existing `prefill_moddeling()` / `decode_moddeling()` continue to use their own memory checks unchanged

**Effort:** 5-7 days (including radix cache from Module 2).

#### 1.3 Enhanced Power Model (Weeks 2-3)

**What:** Replace the 38-line `power.py` with a 7-component temporal power model matching LLMServingSim 2.0's validated approach (1.34% energy accuracy).

**New file:** `llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/power_model.py`

**Source:** Adapted from LLMServingSim concepts per Finding 06, Section 2.4 and Finding 16, Module 3.

**Key class:**

```python
@dataclass
class PowerConfig:
    """Power configuration for a compute node."""
    # Accelerator power (3-state model)
    accelerator_active_watts: float = 400.0     # TDP
    accelerator_idle_watts: float = 25.0        # Idle draw
    accelerator_standby_watts: float = 100.0    # Between batches
    num_accelerators: int = 1

    # Other components (constant when active)
    cpu_watts: float = 200.0
    dram_energy_per_bit: float = 3.7e-12        # Joules per bit transferred
    interconnect_energy_per_bit: float = 6.5e-12
    nic_watts: float = 25.0
    storage_watts: float = 10.0
    other_watts: float = 50.0                   # Motherboard, fans, etc.

    # Power Usage Effectiveness (datacenter overhead)
    pue: float = 1.2

    @classmethod
    def from_hardware_config(cls, hw_config: dict) -> 'PowerConfig':
        """Create from BudSimulator hardware config dict."""
        pass

class AcceleratorState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STANDBY = "standby"

class PowerModel:
    """7-component temporal power model for inference serving.

    Components:
    1. Accelerators (GPUs/TPUs) -- 3-state: idle/active/standby
    2. CPUs -- constant when node is active
    3. DRAM -- energy proportional to data volume
    4. Interconnect -- energy proportional to communication volume
    5. NICs -- constant
    6. Storage -- constant
    7. Other (cooling, motherboard) -- constant

    Maintains a time-series of power samples for visualization.
    """

    def __init__(self, config: PowerConfig):
        pass

    def record_cycle(
        self,
        timestamp_ms: float,
        duration_ms: float,
        accelerator_state: AcceleratorState,
        data_transferred_bytes: int = 0,
        comm_volume_bytes: int = 0,
    ) -> float:
        """Record power for one simulation cycle. Returns instantaneous watts."""
        pass

    def get_total_energy_kwh(self) -> float:
        """Get total energy consumption in kWh."""
        pass

    def get_power_breakdown(self) -> dict[str, float]:
        """Get per-component energy breakdown in kWh."""
        pass

    def get_power_timeseries(self) -> list[tuple[float, float]]:
        """Get (timestamp_ms, watts) time series for visualization."""
        pass

    def get_watts_per_token(self, total_tokens: int) -> float:
        """Get power efficiency metric: watts per token generated."""
        pass
```

**Backward compatibility:** The existing `power.py` `get_energy()` function is preserved unchanged. The new `PowerModel` is used only by the new runtime simulation path. Over time, a deprecation wrapper can route `get_energy()` to use `PowerModel` internally.

**Effort:** 3-4 days.

#### 1.4 Extend System Class for Multi-Tier Memory (Week 3)

**What:** Add optional fields to the GenZ `System` class for host memory, CXL memory, and power parameters. All new fields have defaults that preserve existing behavior.

**File to modify:** `llm-memory-calculator/src/llm_memory_calculator/genz/system.py`

**Changes (backward-compatible additions only):**

```python
class System(object):
    def __init__(self, unit=None,
                 flops=123, mxu_shape=None,
                 onchip_mem_bw=18000, on_chip_mem_size=float('Inf'),
                 offchip_mem_bw=900, off_chip_mem_size=float('Inf'),
                 external_mem_bw=0,
                 frequency=940, bits='bf16',
                 compute_efficiency=1, memory_efficiency=1, comm_efficiency=1,
                 interchip_link_bw=25, num_nodes=1, interchip_link_latency=1.9,
                 compute_engine='GenZ',
                 collective_strategy='GenZ',
                 topology='FullyConnected',
                 parallelism_heirarchy="TP{1}_EP{1}_PP{1}",
                 network_config=None,
                 gear_params=None,
                 # NEW OPTIONAL PARAMETERS (all with backward-compatible defaults):
                 host_mem_size=0,           # Host DRAM in MB (0 = not modeled)
                 host_mem_bw=0,             # Host DRAM bandwidth in GB/s
                 cxl_mem_size=0,            # CXL pooled memory in MB (0 = not modeled)
                 cxl_mem_bw=0,              # CXL bandwidth in GB/s
                 cxl_latency=0,             # CXL latency in us
                 power_tdp=0,               # TDP in watts (0 = not modeled)
                 power_idle=0,              # Idle power in watts
                 power_pue=1.2,             # Power Usage Effectiveness
                 interconnect_type=None,    # 'NVLink', 'InfiniBand', 'PCIe', etc.
                 ):
        # ... existing initialization unchanged ...
        # NEW: store additional parameters
        self.host_mem_size = self.unit.unit_to_raw(host_mem_size, type='M') if host_mem_size else 0
        self.host_mem_bw = self.unit.unit_to_raw(host_mem_bw, type='BW') if host_mem_bw else 0
        self.cxl_mem_size = self.unit.unit_to_raw(cxl_mem_size, type='M') if cxl_mem_size else 0
        self.cxl_mem_bw = self.unit.unit_to_raw(cxl_mem_bw, type='BW') if cxl_mem_bw else 0
        self.cxl_latency = cxl_latency * 1e-6 if cxl_latency else 0
        self.power_tdp = power_tdp
        self.power_idle = power_idle
        self.power_pue = power_pue
        self.interconnect_type = interconnect_type
```

**Effort:** 1 day. All defaults are zero/None, so no existing callers are affected.

#### 1.5 Persistent ASTRA-SIM Controller (Weeks 3-4)

**What:** Implement a persistent ASTRA-SIM subprocess manager that replaces the fire-and-forget `subprocess.run()` pattern with `subprocess.Popen()` and stdin/stdout IPC.

**New file:** `llm-memory-calculator/src/llm_memory_calculator/genz/simulation/runtime/astra_controller.py`

**Source:** Adapted from LLMServingSim `inference_serving/controller.py` per Finding 16, Module 11.

**Key class:**

```python
class AstraSimController:
    """Persistent ASTRA-SIM subprocess with bidirectional IPC.

    Replaces the fire-and-forget get_astrasim_collective_time() with a
    long-running subprocess. Amortizes startup cost and enables multi-iteration
    simulation.

    Falls back to analytical collective timing (GenZ collective_times.py)
    when the ASTRA-SIM binary is not available.
    """

    def __init__(self, astra_sim_path: Optional[str] = None,
                 timeout_seconds: float = 60.0):
        self._process: Optional[subprocess.Popen] = None
        self._timeout = timeout_seconds
        self._astra_sim_path = astra_sim_path
        self._fallback_to_analytical = astra_sim_path is None

    def start(self, network_config_path: str, system_config_path: str) -> None:
        """Start the ASTRA-SIM subprocess."""
        pass

    def submit_workload(self, workload_path: str) -> dict:
        """Submit a workload and get timing results.

        Returns dict mapping system_id -> {cycles: int, exposed_comm_cycles: int}
        """
        pass

    def shutdown(self) -> None:
        """Gracefully terminate the subprocess."""
        pass

    def is_available(self) -> bool:
        """Check if ASTRA-SIM binary is available."""
        pass

    @staticmethod
    def _parse_output(line: str) -> Optional[dict]:
        """Parse ASTRA-SIM stdout line for cycle counts."""
        pass
```

**Integration:** The existing `get_astrasim_collective_time()` in `get_astra_sim_time.py` is NOT modified. The new controller is used only by the runtime simulation path. The existing function continues to work for analytical mode.

**Effort:** 3-4 days.

#### 1.6 Extend Hardware Configs with Power Data (Week 4)

**What:** Add power and topology fields to the `HARDWARE_CONFIGS` dictionary.

**File to modify:** `llm-memory-calculator/src/llm_memory_calculator/hardware/configs.py`

**Changes:** Add optional fields to existing hardware entries:

```python
# Example extension for H100:
"H100_80GB_SXM": {
    "Flops": 989,
    "Memory_size": 80,
    "Memory_BW": 3350,
    "ICN": 900,
    # ... existing fields ...
    # NEW optional fields:
    "power_tdp": 700,              # TDP in watts
    "power_idle": 25,              # Idle power in watts
    "interconnect_type": "NVLink",
    "interconnect_version": "NVLink 4.0",
    "host_mem_bw": 204.8,          # DDR5 bandwidth in GB/s
    "fp8_flops": 1979,             # FP8 peak TFLOPS
}
```

**Database extension:** Add corresponding columns to the hardware table via Alembic migration.

```sql
ALTER TABLE hardware ADD COLUMN power_tdp REAL;
ALTER TABLE hardware ADD COLUMN power_idle REAL;
ALTER TABLE hardware ADD COLUMN pue_factor REAL DEFAULT 1.2;
ALTER TABLE hardware ADD COLUMN interconnect_type TEXT;
ALTER TABLE hardware ADD COLUMN host_mem_bw REAL;
ALTER TABLE hardware ADD COLUMN fp8_flops REAL;
```

**Effort:** 2-3 days (including database migration and populating data for all 50+ hardware entries).

#### 1.7 Expose BudSimulator Core via Existing Endpoints (Weeks 5-6)

**What:** Replace the heuristic `_estimate_performance()` in `usecases.py` with actual GenZ-backed estimation, matching what `usecases_optimization.py` already does.

**File to modify:** `BudSimulator/apis/routers/usecases.py`

**Change:** Replace the inline heuristic function with a call to GenZ's `estimate_prefill_performance()` / `estimate_decode_performance()`, with graceful fallback to the existing heuristic if GenZ import fails.

**Effort:** 2-3 days.

---

### Phase 2: Dynamic Simulation Engine (Weeks 7-14)

**Goal:** Build the runtime simulation layer that enables continuous batching, KV cache management, and per-request metrics. This is the core differentiator from the existing analytical model.

#### 2.1 Request Manager and Arrival Modeling (Week 7)

**New file:** `genz/simulation/runtime/request_manager.py`

**Key classes:**

```python
@dataclass
class Request:
    """A single inference request with lifecycle tracking."""
    id: int
    arrival_time_ms: float
    input_tokens: int
    output_tokens: int                  # Expected output length
    generated_tokens: int = 0           # Tokens generated so far
    phase: str = "queued"               # queued -> prefill -> decode -> done
    ttft_ms: Optional[float] = None     # Time to first token
    tpot_list: list[float] = field(default_factory=list)  # Per-token output times
    queue_start_ms: Optional[float] = None
    prefill_start_ms: Optional[float] = None
    decode_start_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        if self.completion_time_ms and self.arrival_time_ms:
            return self.completion_time_ms - self.arrival_time_ms
        return None

    @property
    def mean_tpot_ms(self) -> Optional[float]:
        if self.tpot_list:
            return sum(self.tpot_list) / len(self.tpot_list)
        return None

@dataclass
class Batch:
    """A batch of requests for a single scheduling cycle."""
    id: int
    requests: list[Request]
    prefill_tokens: list[int]     # Token counts for prefill requests
    decode_tokens: list[int]      # Token counts for decode requests
    total_tokens: int

class ArrivalPattern(Enum):
    POISSON = "poisson"
    GAMMA = "gamma"
    BURSTY = "bursty"
    CONSTANT = "constant"
    TRACE = "trace"               # From a real trace file

class RequestManager:
    """Manages request lifecycle and arrival pattern generation.

    Generates synthetic request streams based on configurable arrival
    patterns and sequence length distributions. Tracks per-request
    metrics throughout the simulation lifecycle.
    """

    def __init__(
        self,
        num_requests: int = 300,
        arrival_rate: float = 10.0,       # Requests per second
        arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
        input_length_mean: int = 512,
        input_length_std: int = 256,
        output_length_mean: int = 128,
        output_length_std: int = 64,
        shared_prefix_ratio: float = 0.0, # Fraction sharing a common prefix
        shared_prefix_length: int = 0,
        seed: int = 42,
    ):
        pass

    def generate_requests(self) -> list[Request]:
        """Generate the full request stream based on configuration."""
        pass

    def get_pending_requests(self, current_time_ms: float) -> list[Request]:
        """Get requests that have arrived by current_time_ms but not yet scheduled."""
        pass

    def compute_metrics(self) -> dict:
        """Compute aggregate metrics: mean/median/P50/P90/P95/P99 for TTFT, TPOT, ITL."""
        pass
```

**Effort:** 3-4 days.

#### 2.2 Continuous Batching Scheduler (Weeks 7-8)

**New file:** `genz/simulation/runtime/batch_scheduler.py`

**Source:** Adapted from LLMServingSim `inference_serving/scheduler.py` per Finding 16, Module 4.

**Key class:**

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"                       # First-come-first-served
    PREFILL_PRIORITY = "prefill_priority"  # Prioritize prefill over decode
    SJF = "sjf"                         # Shortest job first

class BatchScheduler:
    """Memory-aware continuous batching scheduler for LLM inference.

    Implements vLLM/Orca-style iteration-level scheduling:
    1. Select ready requests from the queue
    2. Check memory via MemoryModel for KV cache capacity
    3. Apply eviction under memory pressure
    4. Enforce max_batch_size and max_num_batched_tokens constraints
    5. Create Batch objects with prefill/decode token lists
    6. Track request phase transitions (queue -> prefill -> decode -> done)

    Uses GenZ analytical engine for per-batch compute time estimation.
    """

    def __init__(
        self,
        memory_model: MemoryModel,
        max_batch_size: int = 256,
        max_num_batched_tokens: int = 8192,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
        enable_prefix_caching: bool = False,
        radix_cache: Optional['RadixCache'] = None,
    ):
        pass

    def schedule(self, current_time_ms: float,
                 pending_requests: list[Request]) -> Optional[Batch]:
        """Form the next batch from pending requests.

        Returns None if no requests can be scheduled (all blocked by memory).
        """
        pass

    def complete_batch(self, batch: Batch, compute_time_ms: float,
                       current_time_ms: float) -> list[Request]:
        """Process batch completion. Returns newly completed requests."""
        pass
```

**Effort:** 5-7 days.

#### 2.3 Radix Tree Prefix Cache (Week 9)

**New file:** `genz/simulation/runtime/radix_cache.py`

**Source:** Adapted from LLMServingSim `inference_serving/radix_tree.py` per Finding 16, Module 2. Apache 2.0 licensed (derived from SGLang).

**Key class:**

```python
class RadixCache:
    """Radix tree for KV cache prefix matching with LRU eviction.

    Maintains a tree of token sequences representing cached KV blocks.
    Supports:
    - Prefix matching: find longest cached prefix for a new request
    - LRU eviction: evict least-recently-used leaf nodes under memory pressure
    - Lock reference counting: protect active prefixes from eviction
    - Page-aligned matching: configurable block_size for cache granularity
    - Thread safety via RLock

    Integration: Used by BatchScheduler to reduce compute for requests
    with shared prefixes and by MemoryModel to track cache memory usage.
    """

    def __init__(self, block_size: int = 16,
                 max_capacity_tokens: int = 0):
        pass

    def match_prefix(self, token_ids: list[int]) -> tuple[int, bool]:
        """Find longest matching prefix. Returns (matched_length, is_device_resident)."""
        pass

    def insert(self, token_ids: list[int]) -> None:
        """Insert a token sequence into the cache."""
        pass

    def evict(self, num_tokens_to_free: int) -> int:
        """Evict LRU leaves until num_tokens freed. Returns actual tokens freed."""
        pass

    def get_hit_rate(self) -> float:
        """Get cumulative cache hit rate."""
        pass
```

**Effort:** 3-4 days.

#### 2.4 Runtime Simulator Orchestrator (Weeks 10-12)

**New file:** `genz/simulation/runtime/simulator.py`

**Key class:**

```python
class SimulationMode(Enum):
    ANALYTICAL = "analytical"   # Existing GenZ path (fast, milliseconds)
    HYBRID = "hybrid"           # GenZ compute + dynamic memory/scheduling (seconds)
    FULL = "full"               # GenZ compute + ASTRA-SIM network + full dynamics (minutes)

@dataclass
class ServingSimulationConfig:
    """Configuration for runtime serving simulation."""
    model: str
    system_name: str
    bits: str = "bf16"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    mode: SimulationMode = SimulationMode.HYBRID

    # Scheduling parameters
    max_batch_size: int = 256
    max_num_batched_tokens: int = 8192
    scheduling_policy: str = "fcfs"
    enable_prefix_caching: bool = False
    prefix_cache_block_size: int = 16

    # Memory hierarchy
    host_memory_gb: float = 0
    cxl_memory_gb: float = 0

    # Workload parameters
    num_requests: int = 300
    arrival_rate: float = 10.0
    arrival_pattern: str = "poisson"
    input_length_mean: int = 512
    output_length_mean: int = 128

    # Power modeling
    enable_power_model: bool = True

    # SLO constraints (for reporting)
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None

@dataclass
class ServingSimulationResult:
    """Extended result with per-request and temporal metrics."""
    # Aggregate metrics
    total_throughput_tps: float
    total_requests_completed: int
    total_time_ms: float

    # Per-request metrics (distributions)
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float

    # Memory metrics
    peak_device_memory_gb: float
    peak_host_memory_gb: float
    prefix_cache_hit_rate: float
    eviction_count: int

    # Power metrics (if enabled)
    total_energy_kwh: Optional[float] = None
    power_breakdown: Optional[dict] = None
    watts_per_token: Optional[float] = None

    # SLO compliance
    slo_ttft_compliance: Optional[float] = None  # Fraction meeting TTFT SLO
    slo_tpot_compliance: Optional[float] = None

    # Time series (for visualization)
    throughput_over_time: Optional[list[tuple[float, float]]] = None
    power_over_time: Optional[list[tuple[float, float]]] = None
    memory_over_time: Optional[list[tuple[float, float]]] = None

class RuntimeSimulator:
    """Orchestrates runtime serving simulation using GenZ analytical engine
    for compute timing combined with dynamic scheduling, memory management,
    and power modeling.

    The simulation loop:
    1. Generate request arrivals based on workload config
    2. For each scheduling cycle:
       a. BatchScheduler selects requests, checks memory
       b. Compute time estimated via GenZ analytical roofline
       c. Communication time via ASTRA-SIM (full mode) or GenZ (hybrid)
       d. PowerModel records cycle power
       e. MemoryModel updates KV cache state
       f. RequestManager updates per-request metrics
    3. Collect and aggregate metrics
    """

    def __init__(self):
        pass

    def simulate(self, config: ServingSimulationConfig) -> ServingSimulationResult:
        """Run the serving simulation."""
        pass
```

**Integration with SimulationEngine:** Modify `genz/simulation/engine.py` to add a `simulation_mode` parameter to `SimulationConfig`. When `mode == "analytical"` (default), behavior is unchanged. When `mode == "hybrid"` or `mode == "full"`, delegate to `RuntimeSimulator`.

**Effort:** 8-10 days.

#### 2.5 SLO Tracking and Reporting (Weeks 13-14)

**What:** Add SLO-aware metrics to the simulation results and expose via API.

**Files to modify:**
- `genz/simulation/results.py` -- add `serving_metrics` field
- `BudSimulator/apis/routers/simulation.py` -- add serving simulation endpoint

**New endpoint:**

```python
# POST /api/v2/simulate/serving
class ServingSimulationRequest(BaseModel):
    model: str
    system_name: str
    bits: str = "bf16"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    mode: str = "hybrid"    # "analytical" | "hybrid" | "full"

    # Scheduling
    max_batch_size: int = 256
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = False

    # Workload
    num_requests: int = 300
    arrival_rate: float = 10.0
    arrival_pattern: str = "poisson"
    input_length_mean: int = 512
    output_length_mean: int = 128

    # SLO constraints
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None

class ServingSimulationResponse(BaseModel):
    throughput_tps: float
    ttft: MetricDistribution        # mean, p50, p95, p99
    tpot: MetricDistribution
    memory_peak_gb: float
    prefix_cache_hit_rate: float
    slo_compliance: Optional[dict] = None
    power: Optional[PowerBreakdown] = None
    timeseries: Optional[dict] = None
```

**Effort:** 3-4 days.

---

### Phase 3: Optimization Engine (Weeks 15-20)

**Goal:** Add intelligent configuration search that replaces brute-force parallelism enumeration with Bayesian optimization, enabling simulation-in-the-loop optimization.

#### 3.1 Bayesian Optimizer (Weeks 15-16)

**New file:** `genz/optimization/bayesian_optimizer.py`

**Source:** Based on HEBO framework (Finding 08, Section 1) and SCOOT approach (Findings 03, 08).

**Key class:**

```python
class ObjectiveMetric(Enum):
    THROUGHPUT = "throughput"       # Maximize
    TTFT = "ttft"                  # Minimize
    TPOT = "tpot"                  # Minimize
    COST_PER_REQUEST = "cost"      # Minimize
    ENERGY_PER_TOKEN = "energy"    # Minimize

class BayesianOptimizer:
    """Bayesian optimization for LLM serving configuration.

    Uses simulation as the objective function to search over:
    - Parallelism strategy (TP, PP, EP)
    - Batch size and scheduling parameters
    - Precision format
    - Memory hierarchy configuration
    - Prefix caching settings

    Supports multi-objective optimization via EHVI (Expected Hypervolume
    Improvement) for Pareto-optimal configuration discovery.

    Architecture:
    - Surrogate model: Gaussian Process with Matern 5/2 kernel (via HEBO)
      or Tree-Parzen Estimator (via Optuna, fallback)
    - Acquisition function: MACE ensemble (UCB + PI + EI) per SCOOT
    - Constraint handling: Random forest feasibility predictor per SCOOT
    """

    def __init__(
        self,
        objectives: list[ObjectiveMetric],
        search_space: 'SearchSpace',
        constraint_learner: Optional['ConstraintLearner'] = None,
        max_iterations: int = 50,
        parallel_suggestions: int = 1,     # PD=k from SCOOT
        seed: int = 42,
    ):
        pass

    def suggest(self) -> dict:
        """Suggest the next configuration to evaluate."""
        pass

    def observe(self, config: dict, metrics: dict,
                feasible: bool = True) -> None:
        """Record an observation from simulation."""
        pass

    def get_best(self) -> tuple[dict, dict]:
        """Get the best configuration and its metrics."""
        pass

    def get_pareto_frontier(self) -> list[tuple[dict, dict]]:
        """Get Pareto-optimal configurations for multi-objective."""
        pass
```

**Effort:** 5-7 days.

#### 3.2 Constraint Learner (Week 17)

**New file:** `genz/optimization/constraint_learner.py`

**Key class:**

```python
class ConstraintLearner:
    """Random forest classifier for predicting configuration feasibility.

    Learns from simulation outcomes which configurations are infeasible
    (OOM, excessive latency, invalid parallelism). Uses:
    1. Known constraints (TP * PP <= num_devices, memory bounds)
    2. Learned constraints (random forest trained on observations)
    3. Adaptive threshold (Delta from SCOOT)
    """

    def __init__(self, known_constraints: Optional[list] = None):
        pass

    def predict_feasibility(self, config: dict) -> float:
        """Predict probability of feasibility. Returns 0.0 to 1.0."""
        pass

    def add_observation(self, config: dict, feasible: bool) -> None:
        """Record a feasibility observation for training."""
        pass
```

**Effort:** 2-3 days.

#### 3.3 Simulation-in-the-Loop Optimization (Weeks 18-20)

**What:** Wire the Bayesian optimizer to use the RuntimeSimulator as the objective function.

**New files:**
- `BudSimulator/src/config_optimizer.py` -- service class
- `BudSimulator/apis/routers/optimization.py` -- API router

**Endpoint:**

```python
# POST /api/v2/optimize/config
class OptimizeConfigRequest(BaseModel):
    model: str
    system_name: str
    objectives: list[str] = ["throughput"]   # Metrics to optimize
    constraints: Optional[dict] = None       # {max_ttft_ms, max_tpot_ms, max_cost}
    search_space: Optional[dict] = None      # Override default parameter ranges
    num_iterations: int = 30
    workload: Optional[dict] = None          # Arrival rate, sequence lengths

class OptimizeConfigResponse(BaseModel):
    best_config: dict
    best_metrics: dict
    pareto_frontier: Optional[list[dict]] = None
    search_history: list[dict]               # All evaluated configs
    total_time_seconds: float
```

**Implementation:** The optimizer loop:
1. Initialize `BayesianOptimizer` with search space
2. For each iteration:
   a. `optimizer.suggest()` proposes a config
   b. `constraint_learner.predict_feasibility()` pre-screens
   c. `RuntimeSimulator.simulate()` evaluates (hybrid mode for speed)
   d. `optimizer.observe()` records results
3. Return `optimizer.get_pareto_frontier()`

**Effort:** 5-7 days.

---

### Phase 4: Advanced Features (Weeks 21-26)

**Goal:** Add advanced serving techniques that represent the state of the art in LLM inference.

#### 4.1 Prefill-Decode Disaggregation (Weeks 21-22)

**What:** Model M:N prefill-decode separation where prefill and decode phases run on separate device pools with KV cache transfer.

**Extension to `RuntimeSimulator`:**

```python
@dataclass
class DisaggregatedConfig:
    """Configuration for prefill-decode disaggregation."""
    prefill_system_name: str
    decode_system_name: str
    prefill_tp: int = 1
    decode_tp: int = 1
    prefill_instances: int = 1     # M prefill instances
    decode_instances: int = 1      # N decode instances
    kv_transfer_bandwidth_gbps: float = 100.0  # Inter-instance bandwidth

class RuntimeSimulator:
    def simulate_disaggregated(
        self,
        config: ServingSimulationConfig,
        disagg_config: DisaggregatedConfig,
    ) -> ServingSimulationResult:
        """Simulate P/D disaggregated serving.

        Flow:
        1. Request Router dispatches to prefill instance
        2. Prefill instance computes, generates KV cache
        3. KV cache transferred to decode instance (modeled as latency)
        4. Decode instance generates tokens
        5. Per-request metrics track both phases
        """
        pass
```

**Endpoint:**

```python
# POST /api/v2/simulate/disaggregated
class DisaggregatedSimRequest(BaseModel):
    model: str
    prefill_system_name: str
    decode_system_name: str
    prefill_tp: int = 1
    decode_tp: int = 1
    prefill_instances: int = 1
    decode_instances: int = 1
    kv_transfer_bandwidth_gbps: float = 100.0
    # ... workload parameters ...
```

**Effort:** 5-7 days.

#### 4.2 Speculative Decoding Serving Model (Weeks 22-23)

**What:** Extend the existing analytical speculative decoding model in `llm_spec_decode.py` with serving-level dynamics (draft model resource contention, acceptance rate distribution, variable token burst sizes).

**Extension to `RuntimeSimulator`:**

```python
@dataclass
class SpecDecodeConfig:
    draft_model: str
    target_model: str
    num_speculative_tokens: int = 5
    acceptance_rate: float = 0.7
```

The simulation loop treats speculative decoding as a modified decode phase where:
1. Draft model generates `num_speculative_tokens` tokens (fast, modeled via GenZ)
2. Target model verifies in parallel (modeled as a single prefill-like batch)
3. Expected accepted tokens: `sum(k * x^k * (1-x) for k in 1..N-1) + N * x^N` (already in GenZ)
4. Per-iteration latency = `target_verify_latency + draft_generate_latency`

**Effort:** 3-4 days (builds on existing `llm_spec_decode.py`).

#### 4.3 Cluster Topology Optimization (Weeks 23-24)

**What:** Multi-node, multi-instance cluster configuration with cost-optimized hardware selection.

**New files:**
- `genz/simulation/runtime/cluster_config.py`
- `BudSimulator/src/cluster_designer.py`
- `BudSimulator/apis/routers/cluster.py`

**Source:** Adapted from LLMServingSim `config_builder.py` per Finding 16, Module 10.

**Key class:**

```python
@dataclass
class ClusterNode:
    hardware_name: str
    num_devices: int
    instances_per_node: int = 1   # Multiple model instances per node

@dataclass
class ClusterTopology:
    nodes: list[ClusterNode]
    inter_node_bandwidth_gbps: float = 100.0
    inter_node_latency_us: float = 1.0
    topology_type: str = "fat_tree"  # fat_tree, ring, mesh

class ClusterDesigner:
    """Design optimal cluster topology for a given model and workload.

    Searches over:
    - Hardware type selection
    - Number of nodes
    - Devices per node
    - Instances per node
    - Parallelism configuration per instance

    Objective: Minimize cost while meeting SLO constraints.
    """

    def design_cluster(
        self,
        model: str,
        workload: dict,
        slo_constraints: dict,
        available_hardware: list[str],
        max_budget_per_hour: float,
    ) -> list[dict]:
        """Return ranked cluster designs."""
        pass
```

**Endpoint:**

```python
# POST /api/v2/cluster/design
class ClusterDesignRequest(BaseModel):
    model: str
    target_throughput_tps: float
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None
    available_hardware: list[str] = []
    max_budget_per_hour: Optional[float] = None
    max_nodes: int = 16
```

**Effort:** 5-7 days.

#### 4.4 Power API Endpoints (Week 25)

**New file:** `BudSimulator/apis/routers/power.py`

**Endpoints:**

```python
# POST /api/v2/power/estimate
class PowerEstimateRequest(BaseModel):
    model: str
    system_name: str
    batch_size: int = 1
    input_tokens: int = 2048
    output_tokens: int = 128
    bits: str = "bf16"
    tensor_parallel: int = 1
    num_devices: int = 1

class PowerEstimateResponse(BaseModel):
    total_power_watts: float
    power_breakdown: dict        # {accelerator, cpu, dram, interconnect, nic, storage, other}
    energy_per_request_joules: float
    watts_per_token: float
    pue_adjusted_power_watts: float
    estimated_cost_per_kwh: Optional[float] = None

# POST /api/v2/power/profile
class PowerProfileRequest(BaseModel):
    system_name: str
    model: str
    batch_sizes: list[int] = [1, 4, 8, 16, 32, 64]
    bits: str = "bf16"

class PowerProfileResponse(BaseModel):
    hardware_name: str
    idle_power_watts: float
    tdp_watts: float
    profiles: list[dict]   # [{batch_size, power_watts, tflops_per_watt, tokens_per_watt}]
```

**Effort:** 2-3 days.

#### 4.5 KV Cache Analysis Endpoint (Week 25)

**Endpoint:**

```python
# POST /api/v2/cache/analyze
class CacheAnalysisRequest(BaseModel):
    model: str
    system_name: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    bits: str = "bf16"
    tensor_parallel: int = 1
    enable_prefix_caching: bool = False
    prefix_cache_hit_rate: float = 0.0   # For prefix caching analysis
    prefix_length: int = 0

class CacheAnalysisResponse(BaseModel):
    kv_cache_per_token_bytes: float
    max_kv_cache_tokens: int             # How many tokens fit in device memory
    max_concurrent_requests: int         # Given average sequence length
    device_memory_breakdown: dict        # {weights, kv_cache, activations, overhead}
    with_prefix_cache: Optional[dict]    # {memory_saved_gb, compute_saved_pct}
```

**Effort:** 2 days.

#### 4.6 Integration Testing and Documentation (Week 26)

**What:** End-to-end integration tests for all new capabilities.

**Test files:**
- `BudSimulator/tests/test_serving_simulation.py`
- `BudSimulator/tests/test_optimization.py`
- `BudSimulator/tests/test_power_model.py`
- `BudSimulator/tests/test_memory_model.py`
- `BudSimulator/tests/test_backward_compatibility.py`

**Effort:** 5-7 days.

---

## 5. Backward Compatibility Strategy

### 5.1 Ten Hard Constraints

These constraints are inviolable. Any change that breaks any of these is rejected.

| # | Constraint | Verification Method |
|---|-----------|-------------------|
| 1 | All existing API endpoints preserved at their current URL paths and HTTP methods | Automated test: hit every existing endpoint, verify 2xx response |
| 2 | All existing response schema fields preserved (no field removal or rename) | Schema comparison test: validate response against frozen schema snapshots |
| 3 | All existing request schema required fields remain required | Pydantic model import test: existing request schemas parse correctly |
| 4 | `estimate_memory()` signature and behavior unchanged | Unit test: call with existing parameter combinations, verify identical output |
| 5 | `estimate_prefill_performance()` / `estimate_decode_performance()` signatures unchanged | Unit test: call from HardwareOptimizer with existing parameters |
| 6 | `MemoryReport` properties unchanged | Property enumeration test: all existing properties return correct types |
| 7 | `MODEL_DICT` interface preserved (`.models`, `.get_model()`, `.list_models()`) | Import test: exercise all MODEL_DICT methods |
| 8 | Database schema backward-compatible (no column removal, no rename) | Alembic migration test: upgrade from current schema succeeds |
| 9 | Usecase `unique_id` (string) continues to be the primary identifier | Integration test: CRUD operations on usecases use unique_id |
| 10 | Hardware key normalization (`_KEY_MAP`) unchanged | Unit test: PascalCase to lowercase mapping produces same results |

### 5.2 Versioned API Approach

**Existing endpoints:** No version prefix. Continue to serve at their current paths indefinitely.

```
/api/models/*          # Existing, unchanged
/api/hardware/*        # Existing, unchanged
/api/usecases/*        # Existing, unchanged
/api/simulator/*       # Existing, unchanged
/api/health/*          # Existing, unchanged
```

**New endpoints:** All new endpoints use `/api/v2/` prefix to clearly indicate enhanced capabilities.

```
/api/v2/simulate/*     # New serving simulation endpoints
/api/v2/optimize/*     # New Bayesian optimization endpoints
/api/v2/power/*        # New power modeling endpoints
/api/v2/cluster/*      # New cluster topology endpoints
/api/v2/cache/*        # New KV cache analysis endpoints
```

**Rationale:** The `/api/v2/` prefix signals to consumers that these are new capabilities with new response schemas. Existing integrations continue to work without any changes. When the frontend is updated, it can adopt `/api/v2/` endpoints incrementally.

### 5.3 Feature Flags

New capabilities that modify existing behavior are gated behind feature flags:

| Feature Flag | Default | Effect When Enabled |
|---|---|---|
| `BUDSIM_ENABLE_POWER_MODEL` | `false` | Adds power breakdown to hardware responses |
| `BUDSIM_ENABLE_SERVING_SIM` | `true` | Enables `/api/v2/simulate/serving` endpoint |
| `BUDSIM_ENABLE_OPTIMIZATION` | `true` | Enables `/api/v2/optimize/*` endpoints |
| `BUDSIM_USE_GENZ_FOR_USECASES` | `false` | Replaces heuristic estimation with GenZ in usecase recommendations |
| `BUDSIM_ASTRA_SIM_PATH` | `None` | Path to ASTRA-SIM binary; None = use analytical fallback |

Feature flags are read from environment variables at startup and stored in the application state.

### 5.4 SDK Extension Strategy

**Principle:** All new functionality lives in new modules. Existing module signatures are not modified.

```python
# Existing (unchanged):
from llm_memory_calculator import (
    calculate_memory,          # Same signature
    estimate_memory,           # Same signature
    estimate_prefill_performance,  # Same signature
    estimate_decode_performance,   # Same signature
    get_hardware_config,       # Same signature
    MemoryReport,              # Same class
)

# New (additive):
from llm_memory_calculator.genz.simulation.runtime import (
    RuntimeSimulator,
    ServingSimulationConfig,
    ServingSimulationResult,
    MemoryModel,
    BatchScheduler,
    PowerModel,
)

from llm_memory_calculator.genz.optimization import (
    BayesianOptimizer,
    ConstraintLearner,
    SearchSpace,
)
```

---

## 6. New API Endpoints Design

### 6.1 Complete Endpoint Inventory

| Endpoint | Method | Router | Phase | Description |
|---|---|---|---|---|
| `/api/v2/simulate/run` | POST | simulation.py | 1 | Single analytical simulation (exposes BudSimulator) |
| `/api/v2/simulate/features` | GET | simulation.py | 1 | List available simulation features |
| `/api/v2/simulate/batch` | POST | simulation.py | 1 | Run multiple simulations in parallel |
| `/api/v2/simulate/compare` | POST | simulation.py | 1 | Compare configurations side-by-side |
| `/api/v2/simulate/serving` | POST | simulation.py | 2 | Dynamic serving simulation with scheduling |
| `/api/v2/simulate/disaggregated` | POST | simulation.py | 4 | P/D disaggregated serving simulation |
| `/api/v2/optimize/config` | POST | optimization.py | 3 | Bayesian optimization of serving config |
| `/api/v2/optimize/sweep` | POST | optimization.py | 3 | Parameter sweep with ranking |
| `/api/v2/power/estimate` | POST | power.py | 4 | Detailed power breakdown for a config |
| `/api/v2/power/profile` | POST | power.py | 4 | Power profile across batch sizes |
| `/api/v2/cluster/design` | POST | cluster.py | 4 | Design optimal cluster topology |
| `/api/v2/cluster/compare` | POST | cluster.py | 4 | Compare cluster configurations |
| `/api/v2/cache/analyze` | POST | simulation.py | 4 | KV cache analysis with prefix caching |

### 6.2 Router Registration

**File:** `BudSimulator/apis/main.py`

```python
# Add to existing router registrations:
from .routers import simulation, optimization, power, cluster

app.include_router(simulation.router, prefix="/api/v2/simulate", tags=["simulation-v2"])
app.include_router(optimization.router, prefix="/api/v2/optimize", tags=["optimization-v2"])
app.include_router(power.router, prefix="/api/v2/power", tags=["power-v2"])
app.include_router(cluster.router, prefix="/api/v2/cluster", tags=["cluster-v2"])
```

---

## 7. SDK Extension Design

### 7.1 New Classes

| Class | Module | Purpose | Phase |
|---|---|---|---|
| `RuntimeSimulator` | `genz.simulation.runtime.simulator` | Orchestrates runtime serving simulation | 2 |
| `BatchScheduler` | `genz.simulation.runtime.batch_scheduler` | Continuous batching with memory awareness | 2 |
| `MemoryModel` | `genz.simulation.runtime.memory_model` | Multi-tier dynamic memory tracking | 1 |
| `RadixCache` | `genz.simulation.runtime.radix_cache` | Prefix caching with LRU eviction | 2 |
| `PowerModel` | `genz.simulation.runtime.power_model` | 7-component temporal power model | 1 |
| `BayesianOptimizer` | `genz.optimization.bayesian_optimizer` | HEBO/GP-based config optimization | 3 |
| `ConstraintLearner` | `genz.optimization.constraint_learner` | Random forest feasibility prediction | 3 |

### 7.2 Extension of Existing Classes

| Class | File | Changes | Phase |
|---|---|---|---|
| `System` | `genz/system.py` | Add optional `host_mem_size`, `cxl_mem_size`, `power_tdp`, `power_idle`, `interconnect_type` parameters with zero/None defaults | 1 |
| `SimulationConfig` | `genz/simulation/config.py` | Add optional `simulation_mode` field (default: "analytical") | 2 |
| `SimulationResult` | `genz/simulation/results.py` | Add optional `serving_metrics` field (default: None) | 2 |
| `SimulationEngine` | `genz/simulation/engine.py` | Add `_run_runtime_simulation()` method that delegates to `RuntimeSimulator` when mode != "analytical" | 2 |

### 7.3 SDK Export Strategy

**File:** `llm-memory-calculator/src/llm_memory_calculator/__init__.py`

```python
# Existing exports (unchanged):
from .calculator import ModelMemoryCalculator
from .types import MemoryReport
# ... all existing exports ...

# New conditional exports (Phase 2+):
try:
    from .genz.simulation.runtime import (
        RuntimeSimulator,
        ServingSimulationConfig,
        ServingSimulationResult,
        MemoryModel,
        PowerModel,
    )
    _RUNTIME_AVAILABLE = True
except ImportError:
    _RUNTIME_AVAILABLE = False

try:
    from .genz.optimization import (
        BayesianOptimizer,
        ConstraintLearner,
    )
    _OPTIMIZATION_AVAILABLE = True
except ImportError:
    _OPTIMIZATION_AVAILABLE = False
```

---

## 8. Data Flow Architecture

### 8.1 Analytical Path (Existing -- Preserved)

```
Client
  |
  | POST /api/models/calculate  (or any existing endpoint)
  |
  v
FastAPI Router (models.py)
  |
  | Call estimate_memory() or estimate_prefill_performance()
  |
  v
llm_memory_calculator SDK
  |
  | ModelMemoryCalculator.calculate_memory()
  |  OR
  | GenZ prefill_moddeling() / decode_moddeling()
  |
  v
GenZ Operator Framework
  |
  | Per-operator roofline analysis
  | Collective timing (GenZ analytical or ASTRA-SIM one-shot)
  |
  v
Result: {latency_ms, throughput_tps, runtime_breakdown, memory_gb}
```

### 8.2 Serving Simulation Path (New)

```
Client
  |
  | POST /api/v2/simulate/serving
  |
  v
FastAPI Router (simulation.py)
  |
  | Create ServingSimulationConfig from request
  |
  v
ServingSimulator Service (serving_simulator.py)
  |
  | Initialize RuntimeSimulator
  |
  v
RuntimeSimulator.simulate()
  |
  |  1. RequestManager.generate_requests()
  |     -> List[Request] with arrival times, token lengths
  |
  |  2. Initialize subsystems:
  |     -> MemoryModel.from_genz_system(system)
  |     -> BatchScheduler(memory_model, ...)
  |     -> PowerModel(PowerConfig.from_hardware_config(...))
  |     -> Optional: AstraSimController.start() (full mode only)
  |     -> Optional: RadixCache (if prefix caching enabled)
  |
  |  3. SIMULATION LOOP (while requests remain):
  |     |
  |     | a. pending = RequestManager.get_pending(current_time)
  |     | b. batch = BatchScheduler.schedule(current_time, pending)
  |     |    -> Memory check via MemoryModel.is_available()
  |     |    -> Eviction via MemoryModel.evict_to_lower_tier() if needed
  |     |    -> Prefix check via RadixCache.match_prefix() if enabled
  |     |
  |     | c. For each operator in batch:
  |     |    -> compute_time = GenZ roofline analysis (FAST, analytical)
  |     |    -> comm_time = GenZ collective_times (hybrid)
  |     |       OR AstraSimController.submit_workload() (full)
  |     |
  |     | d. batch_time = max(compute_pipeline, comm_pipeline)
  |     |
  |     | e. PowerModel.record_cycle(timestamp, duration, state, ...)
  |     | f. MemoryModel.allocate_kv(...) / free_kv(...)
  |     | g. RequestManager.update_metrics(batch, current_time)
  |     |
  |     | h. current_time += batch_time
  |     |
  |  4. Aggregate metrics:
  |     -> RequestManager.compute_metrics()  (TTFT/TPOT/ITL distributions)
  |     -> PowerModel.get_total_energy_kwh()
  |     -> MemoryModel.get_state()
  |
  v
ServingSimulationResult
  |
  v
JSON Response to Client
```

### 8.3 Optimization Path (New)

```
Client
  |
  | POST /api/v2/optimize/config
  |
  v
FastAPI Router (optimization.py)
  |
  v
ConfigOptimizer Service (config_optimizer.py)
  |
  |  1. Build SearchSpace from request constraints
  |  2. Initialize BayesianOptimizer(objectives, search_space)
  |  3. Initialize ConstraintLearner(known_constraints)
  |
  |  4. OPTIMIZATION LOOP (for N iterations):
  |     |
  |     | a. config = optimizer.suggest()
  |     | b. feasibility = constraint_learner.predict_feasibility(config)
  |     |    -> If < threshold: optimizer.observe(config, None, feasible=False); continue
  |     |
  |     | c. result = RuntimeSimulator.simulate(config)
  |     |    -> Uses hybrid mode for speed (~seconds per evaluation)
  |     |
  |     | d. metrics = extract_metrics(result)
  |     | e. optimizer.observe(config, metrics, feasible=True)
  |     | f. constraint_learner.add_observation(config, feasible=True)
  |
  |  5. Return optimizer.get_pareto_frontier()
  |
  v
OptimizeConfigResponse
  |
  v
JSON Response to Client
```

### 8.4 ASTRA-SIM Bidirectional IPC (Enhanced)

```
Python (RuntimeSimulator)                   C++ (ASTRA-SIM subprocess)
  |                                           |
  | Popen(stdin=PIPE, stdout=PIPE)            |
  |------------------------------------------>|  Start process
  |                                           |
  | write_flush(workload_path)                |
  |------------------------------------------>|  Load workload
  |                                           |
  |       stdout: "Waiting"                   |
  |<------------------------------------------|  Ready for work
  |                                           |
  | write_flush(next_workload_path)           |
  |------------------------------------------>|  Process workload
  |                                           |
  |  stdout: "sys[0] iteration 1 finished,    |
  |           12345 cycles, exposed comm      |
  |           2345 cycles"                    |
  |<------------------------------------------|  Return results
  |                                           |
  | parse_output(line) -> {cycles, comm}      |
  |                                           |
  | ... repeat for each scheduling cycle ...  |
  |                                           |
  | write_flush("exit")                       |
  |------------------------------------------>|  Shutdown
```

---

## 9. Risk Assessment and Mitigations

### 9.1 Performance Regression Risks

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| New imports slow down existing API response times | Medium | Low | Lazy imports: all new modules use `try/except ImportError` guards. New packages are only imported when `/api/v2/` endpoints are called. |
| Runtime simulation takes too long for interactive use | Medium | Medium | Hybrid mode uses GenZ analytical for compute (milliseconds) + dynamic scheduling (seconds). Full ASTRA-SIM mode is opt-in only. Default mode is hybrid with 300 requests completing in ~5-30 seconds. |
| Memory overhead from new subsystems | Low | Low | MemoryModel, RadixCache, and PowerModel are instantiated per-simulation, not globally. Garbage collected after each request. |
| Database migrations cause downtime | Low | Low | All migrations are ADD COLUMN with nullable defaults. No table drops, no column renames. |

### 9.2 Backward Compatibility Risks

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| System class extension breaks existing callers | High | Very Low | All new parameters have zero/None defaults. Existing `System()` calls with positional args are unaffected because new params are keyword-only and appended after all existing params. |
| SimulationConfig changes break existing usage | Medium | Low | New `simulation_mode` field defaults to "analytical", preserving exact existing behavior. |
| New router registration conflicts with existing routes | Medium | Very Low | All new routes use `/api/v2/` prefix, which has zero overlap with existing `/api/*` routes. |
| SDK export changes break existing imports | High | Very Low | Existing exports are unchanged. New exports use conditional `try/except` blocks. |

### 9.3 ASTRA-SIM Binary Dependency Risks

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| ASTRA-SIM binary not available on deployment target | Medium | High | Graceful fallback: `AstraSimController.is_available()` returns False, RuntimeSimulator automatically uses hybrid mode (GenZ analytical for all timing). ASTRA-SIM is optional, not required. |
| ASTRA-SIM subprocess hangs or crashes | Medium | Medium | `AstraSimController` implements timeout handling (default 60s). If subprocess does not respond within timeout, it is killed and simulation falls back to analytical timing for remaining cycles. |
| Chakra package version incompatibility | Low | Medium | `fix_chakra_traces.py` already handles dual import paths. The new `astra_controller.py` checks Chakra version at initialization and adapts accordingly. |

### 9.4 Testing Strategy

| Test Category | Coverage Target | Implementation |
|---|---|---|
| Backward compatibility | 100% of existing endpoints | `test_backward_compatibility.py`: frozen request/response snapshots compared against live API |
| Unit tests for new modules | 85%+ line coverage | Per-module test files: `test_memory_model.py`, `test_batch_scheduler.py`, `test_power_model.py`, `test_bayesian_optimizer.py` |
| Integration tests | All new endpoints | `test_serving_simulation.py`: end-to-end simulation request/response validation |
| Differential testing | GenZ analytical vs hybrid mode | Compare analytical results to hybrid simulation results for simple workloads (single batch, no queuing). Should agree within 5%. |
| Performance regression | No regression on existing paths | Benchmark suite: measure response times for all existing endpoints before and after integration |
| Memory leak detection | No leaks in simulation loop | Run 1000-iteration simulation with memory profiling; verify stable memory footprint |

---

## 10. Success Metrics

### 10.1 Accuracy Targets

| Metric | Target | Measurement Method | Reference |
|---|---|---|---|
| Analytical mode accuracy vs real hardware | <=15% MRE (existing) | Compare GenZ predictions to published benchmarks | Existing validation framework |
| Hybrid simulation TTFT accuracy | <=10% MRE vs vLLM real measurements | Compare against vLLM serving benchmarks on H100/A100 | LLMServingSim 2.0 achieves 0.97% |
| Hybrid simulation throughput accuracy | <=10% MRE vs vLLM real measurements | Same as above | LLMServingSim 2.0 achieves 0.85-1.59% |
| Power model accuracy | <=5% energy error | Compare against nvidia-smi power measurements during inference | LLMServingSim 2.0 achieves 1.34% |
| Memory model accuracy | <=5% peak memory error | Compare against vLLM reported GPU memory | LLMServingSim 2.0 achieves 0.93% |
| Prefix cache hit rate accuracy | <=5% absolute error | Compare simulated vs measured hit rate on ShareGPT workload | LLMServingSim 2.0 achieves 0.41% |

### 10.2 Performance Targets

| Metric | Target | Rationale |
|---|---|---|
| Analytical simulation (existing) | <500ms per evaluation | Maintain current speed; no regression |
| Hybrid simulation (300 requests) | <30 seconds | Fast enough for interactive use and optimization loops |
| Full simulation (300 requests) | <10 minutes | Acceptable for detailed analysis; comparable to LLMServingSim (5-10 min) |
| Bayesian optimization (30 iterations, hybrid) | <15 minutes | 30 * 30s = 15 min; practical for config exploration |
| API endpoint response time (existing) | No regression (p99 <200ms) | Existing endpoints must not slow down |

### 10.3 Coverage Targets

| Capability | Status Before | Status After | Notes |
|---|---|---|---|
| Static analytical modeling | Yes | Yes (preserved) | GenZ roofline, unchanged |
| Dynamic serving simulation | No | Yes (hybrid + full) | Continuous batching, scheduling dynamics |
| Per-request metrics (TTFT/TPOT/ITL) | No | Yes | P50/P90/P95/P99 distributions |
| Multi-tier memory (device/host/CXL) | No | Yes | Block-based KV cache management |
| Prefix caching | No | Yes | Radix tree with LRU eviction |
| KV cache eviction modeling | No | Yes | Graceful degradation under pressure |
| Power/energy modeling | Basic (4-component) | Enhanced (7-component, temporal) | 3-state accelerator model |
| P/D disaggregation | No | Yes | M:N prefill-decode separation |
| Bayesian optimization | No | Yes | HEBO/GP + MACE + constraint learning |
| Speculative decoding serving | Analytical only | Analytical + serving dynamics | Draft+target resource contention |
| Cluster topology optimization | No | Yes | Multi-node cost-optimized design |
| SLO-aware recommendations | Partial (usecase SLO fields) | Full (SLO compliance tracking) | TTFT/TPOT SLO validation |
| BudSimulator API exposure | 0 endpoints | 4+ endpoints | Core simulation accessible from frontend |

### 10.4 Backward Compatibility Targets

| Metric | Target |
|---|---|
| Existing API endpoints broken | 0 |
| Existing SDK functions with changed signatures | 0 |
| Existing response schema fields removed | 0 |
| Existing database columns removed | 0 |
| Existing tests that fail after integration | 0 |

---

## Appendix A: Dependency Tree

```
Phase 1 (Foundation)
  ├── 1.1 Expose BudSimulator via API         (no dependencies)
  ├── 1.2 Multi-Tier Memory Model             (no dependencies)
  ├── 1.3 Enhanced Power Model                (no dependencies)
  ├── 1.4 Extend System Class                 (no dependencies)
  ├── 1.5 Persistent ASTRA-SIM Controller     (no dependencies)
  ├── 1.6 Extend Hardware Configs             (no dependencies)
  └── 1.7 Replace Heuristic Estimation        (no dependencies)

Phase 2 (Dynamic Simulation)
  ├── 2.1 Request Manager                     (no dependencies)
  ├── 2.2 Batch Scheduler                     (depends on: 1.2 MemoryModel)
  ├── 2.3 Radix Cache                         (no dependencies)
  ├── 2.4 Runtime Simulator                   (depends on: 1.2, 1.3, 1.5, 2.1, 2.2, 2.3)
  └── 2.5 SLO Tracking                        (depends on: 2.4)

Phase 3 (Optimization)
  ├── 3.1 Bayesian Optimizer                  (no dependencies)
  ├── 3.2 Constraint Learner                  (no dependencies)
  └── 3.3 Simulation-in-Loop Optimization     (depends on: 2.4, 3.1, 3.2)

Phase 4 (Advanced)
  ├── 4.1 P/D Disaggregation                  (depends on: 2.4)
  ├── 4.2 Speculative Decoding Serving        (depends on: 2.4)
  ├── 4.3 Cluster Topology Optimization       (depends on: 2.4, 3.3)
  ├── 4.4 Power API Endpoints                 (depends on: 1.3)
  ├── 4.5 KV Cache Analysis Endpoint          (depends on: 1.2)
  └── 4.6 Integration Testing                 (depends on: all above)
```

## Appendix B: File Change Summary

### New Files (by phase)

**Phase 1 (7 files):**
```
BudSimulator/apis/routers/simulation.py
BudSimulator/apis/schemas/simulation_schemas.py
llm-memory-calculator/.../genz/simulation/runtime/__init__.py
llm-memory-calculator/.../genz/simulation/runtime/memory_model.py
llm-memory-calculator/.../genz/simulation/runtime/power_model.py
llm-memory-calculator/.../genz/simulation/runtime/astra_controller.py
llm-memory-calculator/.../genz/simulation/runtime/types.py
```

**Phase 2 (5 files):**
```
llm-memory-calculator/.../genz/simulation/runtime/request_manager.py
llm-memory-calculator/.../genz/simulation/runtime/batch_scheduler.py
llm-memory-calculator/.../genz/simulation/runtime/radix_cache.py
llm-memory-calculator/.../genz/simulation/runtime/simulator.py
BudSimulator/src/serving_simulator.py
```

**Phase 3 (5 files):**
```
llm-memory-calculator/.../genz/optimization/__init__.py
llm-memory-calculator/.../genz/optimization/bayesian_optimizer.py
llm-memory-calculator/.../genz/optimization/constraint_learner.py
llm-memory-calculator/.../genz/optimization/search_space.py
BudSimulator/apis/routers/optimization.py
BudSimulator/src/config_optimizer.py
```

**Phase 4 (6 files):**
```
llm-memory-calculator/.../genz/simulation/runtime/cluster_config.py
BudSimulator/apis/routers/power.py
BudSimulator/apis/routers/cluster.py
BudSimulator/src/power_analyzer.py
BudSimulator/src/cluster_designer.py
BudSimulator/apis/schemas/optimization_schemas.py
```

### Modified Files (backward-compatible changes only)

```
llm-memory-calculator/.../genz/system.py                     # Add optional parameters
llm-memory-calculator/.../genz/simulation/config.py           # Add simulation_mode field
llm-memory-calculator/.../genz/simulation/engine.py           # Add runtime delegation
llm-memory-calculator/.../genz/simulation/results.py          # Add serving_metrics field
llm-memory-calculator/.../hardware/configs.py                 # Add power/topology fields
llm-memory-calculator/src/llm_memory_calculator/__init__.py   # Add conditional exports
BudSimulator/apis/main.py                                     # Register new routers
BudSimulator/apis/routers/usecases.py                         # Replace heuristic estimation
```

## Appendix C: External Dependencies

| Package | Purpose | Phase | Required | Fallback |
|---|---|---|---|---|
| ASTRA-SIM binary | Cycle-accurate network simulation | 1 | No | GenZ analytical collective timing |
| Chakra | Execution trace format | 1 | No | Only needed for full simulation mode |
| HEBO | Bayesian optimization | 3 | No | Falls back to Optuna TPE sampler |
| scikit-learn | Random forest constraint learning | 3 | No | Skip constraint learning; use known constraints only |
| scipy | Statistical metrics (P50/P90/P95/P99) | 2 | Yes | Already in requirements |
| numpy | Numerical operations | All | Yes | Already in requirements |
| msgspec | Zero-copy event serialization (RadixCache) | 2 | No | Falls back to dataclass |

## Appendix D: Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time To First Token -- wall-clock time from request submission to first output token |
| **TPOT** | Time Per Output Token -- mean time between consecutive output tokens |
| **ITL** | Inter-Token Latency -- exact time between two consecutive tokens (per-pair) |
| **SLO** | Service Level Objective -- performance target (e.g., P99 TTFT < 500ms) |
| **MSG** | Model Serving Group -- LLMServingSim's abstraction for a model instance |
| **P/D Disaggregation** | Prefill-Decode Disaggregation -- running prefill and decode on separate device pools |
| **KV Cache** | Key-Value cache storing attention intermediate results for decode reuse |
| **Radix Tree** | Trie-like data structure for efficient prefix matching in KV caches |
| **HEBO** | Heteroscedastic Evolutionary Bayesian Optimisation -- NeurIPS 2020 winner |
| **MACE** | Multi-objective Acquisition Ensemble -- runs UCB+PI+EI and selects from Pareto front |
| **EHVI** | Expected Hypervolume Improvement -- multi-objective acquisition function |
| **GenZ** | The analytical roofline-based performance modeling engine at the core of BudSimulator |
| **Hybrid mode** | GenZ analytical compute + dynamic scheduling/memory (no ASTRA-SIM) |
| **Full mode** | GenZ analytical compute + ASTRA-SIM persistent IPC + dynamic scheduling/memory |
| **MRE** | Mean Relative Error -- average of |predicted - actual| / actual |
| **TCO** | Total Cost of Ownership -- full deployment cost including compute, power, network, operations |
