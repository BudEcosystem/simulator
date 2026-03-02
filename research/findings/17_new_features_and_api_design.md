# New Features and API Design for BudSimulator

**Author:** Research Integration Agent
**Date:** 2026-02-28
**Scope:** Comprehensive feature design and API specification for BudSimulator v2, integrating findings from LLMServingSim 2.0, SCOOT, vllm-tuner analysis, and GenZ engine gap analysis.
**Source Findings:** 03, 06, 08, 10, 11, 12

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature Category: Dynamic Serving Simulation](#2-feature-category-dynamic-serving-simulation)
3. [Feature Category: Advanced Memory Modeling](#3-feature-category-advanced-memory-modeling)
4. [Feature Category: Power and Energy Modeling](#4-feature-category-power-and-energy-modeling)
5. [Feature Category: Serving Configuration Optimization](#5-feature-category-serving-configuration-optimization)
6. [Feature Category: Cluster and Topology](#6-feature-category-cluster-and-topology)
7. [Feature Category: Benchmarking and Validation](#7-feature-category-benchmarking-and-validation)
8. [Complete API Specification](#8-complete-api-specification)
9. [SDK Class Design](#9-sdk-class-design)
10. [Backward Compatibility Matrix](#10-backward-compatibility-matrix)

---

## 1. Executive Summary

### Total New Capabilities

This design document specifies **34 new capabilities** organized into 6 feature categories:

| Category | New Capabilities | Complexity | Research Source |
|---|---|---|---|
| Dynamic Serving Simulation | 5 | High | LLMServingSim 2.0, SCOOT |
| Advanced Memory Modeling | 4 | Medium-High | LLMServingSim 2.0 |
| Power and Energy Modeling | 4 | Medium | LLMServingSim 2.0 |
| Serving Configuration Optimization | 5 | High | SCOOT, HEBO, vllm-tuner |
| Cluster and Topology | 4 | Medium-High | LLMServingSim 2.0, DistServe |
| Benchmarking and Validation | 4 | Medium | LLMServingSim 2.0, SCOOT |

These capabilities transform BudSimulator from a static analytical tool into a comprehensive LLM deployment planning platform.

### New API Surface

- **12 new REST endpoints** under `/api/v2/` prefix
- **7 new SDK classes** in `llm_memory_calculator`
- **4 new database tables** for results persistence
- **Full backward compatibility** with all existing v1 endpoints

### Impact on Competitive Positioning

**Current state:** BudSimulator provides static, single-pass analytical performance estimates via GenZ roofline analysis. It competes with basic roofline calculators and spreadsheet-based tools.

**After integration:** BudSimulator becomes a multi-modal analysis platform offering:
- Fast analytical estimates (milliseconds, existing GenZ)
- Dynamic serving simulation (seconds, new capability)
- Configuration optimization (minutes, new capability)
- Power/cost/TCO analysis (new capability)

This positions BudSimulator uniquely between lightweight calculators (Vidur ~seconds, less accurate) and heavyweight simulators (LLMServingSim 2.0 ~minutes, requires profiling hardware). BudSimulator would be the only tool offering the full spectrum without requiring physical hardware access.

---

## 2. Feature Category: Dynamic Serving Simulation

**Current state in GenZ:** Static single-pass analytical model. Fixed batch size, no request scheduling, no queue dynamics, no temporal simulation.

**Gap reference:** Finding 11 Section 13 (CRITICAL gaps: runtime-driven simulation, batch scheduling, KV cache management, per-request metrics), Finding 06 Sections 1.3-1.6 (LLMServingSim runtime loop).

### 2a. Continuous Batching Simulation

**What it does:** Models iteration-level batch dynamics where requests enter and exit the batch at each decoding step, replicating the behavior of vLLM/Orca-style continuous batching.

**How it works:**
1. Maintain a virtual request queue with arriving requests
2. At each iteration, the scheduler selects requests to form a batch based on memory constraints
3. Requests completing decode are removed; new requests from the queue fill their slots
4. Track per-iteration batch size, memory usage, and throughput over time
5. Use GenZ's existing per-operator roofline analysis as the cost model for each iteration

**API Endpoint:** `POST /api/v2/simulate/batch`

**SDK Class:** `BatchScheduler` (see Section 9)

**Input Schema:**
```python
class BatchSimulationRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier (e.g., 'llama3_8b' or HuggingFace ID)")
    hardware_name: str = Field(..., description="Hardware configuration name")
    arrival_rate: float = Field(10.0, gt=0, description="Request arrival rate (requests/second)")
    arrival_pattern: Literal["poisson", "gamma", "bursty", "constant"] = Field("poisson", description="Arrival distribution")
    num_requests: int = Field(300, ge=10, le=10000, description="Total requests to simulate")
    input_length_distribution: LengthDistribution = Field(default_factory=lambda: LengthDistribution())
    output_length_distribution: LengthDistribution = Field(default_factory=lambda: LengthDistribution())
    max_batch_size: int = Field(256, ge=1, description="Maximum concurrent sequences")
    max_num_batched_tokens: int = Field(8192, ge=64, description="Maximum tokens per iteration")
    precision: str = Field("bf16", description="Weight precision")
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    enable_chunked_prefill: bool = Field(False, description="Enable chunked prefill")
    chunk_size: Optional[int] = Field(None, ge=64, description="Chunk size when chunked prefill enabled")

class LengthDistribution(BaseModel):
    distribution: Literal["uniform", "normal", "lognormal", "dataset"] = Field("uniform")
    min_length: int = Field(128, ge=1)
    max_length: int = Field(2048, ge=1)
    mean: Optional[float] = Field(None, description="Mean for normal/lognormal")
    std: Optional[float] = Field(None, description="Std for normal/lognormal")
    dataset_path: Optional[str] = Field(None, description="Path to request trace file for 'dataset' distribution")
```

**Output Schema:**
```python
class BatchSimulationResponse(BaseModel):
    total_duration_ms: float
    total_requests_completed: int
    overall_throughput_rps: float
    overall_token_throughput_tps: float
    avg_batch_size: float
    max_batch_size_observed: int
    batch_size_over_time: List[TimeSeriesPoint]  # [{time_ms, value}]
    throughput_over_time: List[TimeSeriesPoint]
    memory_usage_over_time: List[TimeSeriesPoint]
    queue_depth_over_time: List[TimeSeriesPoint]
    per_request_metrics: RequestMetricsSummary
    slo_analysis: Optional[SLOAnalysis] = None

class TimeSeriesPoint(BaseModel):
    time_ms: float
    value: float

class RequestMetricsSummary(BaseModel):
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    queue_wait_p50_ms: float
    queue_wait_p95_ms: float
    queue_wait_p99_ms: float
```

**Example Usage:**
```python
from llm_memory_calculator import BatchScheduler

scheduler = BatchScheduler(
    model="llama3_8b",
    hardware="H100_80GB",
    precision="bf16",
    tensor_parallel=4
)

result = scheduler.simulate(
    arrival_rate=10.0,
    arrival_pattern="poisson",
    num_requests=300,
    input_length_distribution={"distribution": "lognormal", "mean": 512, "std": 200, "min_length": 64, "max_length": 4096},
    output_length_distribution={"distribution": "uniform", "min_length": 64, "max_length": 512},
    max_batch_size=128,
    max_num_batched_tokens=4096,
)

print(f"Throughput: {result.overall_throughput_rps:.1f} req/s")
print(f"TTFT P95: {result.per_request_metrics.ttft_p95_ms:.1f} ms")
print(f"TPOT P95: {result.per_request_metrics.tpot_p95_ms:.1f} ms")
```

### 2b. Request Queue Modeling

**What it does:** Models the request queue that feeds into the batch scheduler, supporting realistic arrival patterns and tracking queue dynamics.

**How it works:**
1. Generate request arrival times from configured distribution (Poisson for steady-state, Gamma for bursty, constant for batch processing)
2. Assign sequence lengths from configured distributions
3. Track queue depth, wait times, and admission decisions over time
4. Support request priorities and fair scheduling

This is a subcomponent of the BatchScheduler, not a standalone endpoint. It is exposed through the batch simulation configuration.

**Arrival Patterns:**
- **Poisson:** `inter_arrival = -log(uniform()) / lambda` -- standard for web traffic
- **Gamma:** `inter_arrival = gamma(shape, scale)` -- models bursty traffic with `shape < 1`
- **Bursty:** Alternating periods of high rate (`5x lambda`) and low rate (`0.2x lambda`) with configurable duty cycle
- **Constant:** Fixed inter-arrival time `1/lambda` -- for throughput benchmarking

### 2c. KV Cache Dynamics

**What it does:** Models the lifecycle of KV cache blocks -- allocation, eviction, migration across memory tiers, and fragmentation -- during serving simulation.

**How it works:**
1. Track per-request KV cache allocation as tokens are generated
2. When device memory is full, trigger eviction based on configured policy (LRU, LFU, or importance-based)
3. Model cache migration costs between device/host/CXL tiers
4. Track fragmentation from variable-length allocations
5. Model prefix cache hits that reduce both memory allocation and compute

This is integrated into the BatchScheduler simulation. Detailed cache analysis is exposed through `POST /api/v2/cache/analyze`.

**Cache Analysis Endpoint Input:**
```python
class CacheAnalysisRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    batch_size: int = Field(32, ge=1)
    input_tokens: int = Field(2048, ge=1)
    output_tokens: int = Field(256, ge=1)
    device_memory_budget_gb: Optional[float] = Field(None, description="Override device memory budget")
    eviction_policy: Literal["lru", "lfu", "fifo"] = Field("lru")
    block_size: int = Field(16, description="KV cache block size in tokens")
    enable_prefix_caching: bool = Field(False)
    prefix_length: Optional[int] = Field(None, description="Common prefix length in tokens")
    prefix_cache_hit_rate: Optional[float] = Field(None, ge=0, le=1, description="Override estimated hit rate")
    memory_tiers: Optional[List[MemoryTierConfig]] = Field(None, description="Multi-tier memory configuration")

class MemoryTierConfig(BaseModel):
    tier: Literal["device", "host", "cxl", "storage"]
    capacity_gb: float = Field(gt=0)
    bandwidth_gbs: float = Field(gt=0)
    latency_us: float = Field(ge=0)
```

**Cache Analysis Endpoint Output:**
```python
class CacheAnalysisResponse(BaseModel):
    model_weights_gb: float
    kv_cache_per_token_bytes: float
    kv_cache_per_request_gb: float
    total_kv_cache_gb: float
    device_memory_available_gb: float
    max_concurrent_requests: int
    eviction_required: bool
    eviction_rate_per_second: Optional[float]
    prefix_cache_analysis: Optional[PrefixCacheAnalysis]
    fragmentation_analysis: FragmentationAnalysis
    tier_breakdown: Optional[List[TierUsage]]

class PrefixCacheAnalysis(BaseModel):
    prefix_length_tokens: int
    prefix_memory_gb: float
    estimated_hit_rate: float
    memory_savings_gb: float
    compute_savings_percent: float
    effective_ttft_speedup: float

class FragmentationAnalysis(BaseModel):
    block_size_tokens: int
    total_blocks_allocated: int
    total_blocks_capacity: int
    internal_fragmentation_percent: float
    wasted_memory_gb: float
    paged_vs_contiguous_comparison: Dict[str, float]

class TierUsage(BaseModel):
    tier: str
    capacity_gb: float
    used_gb: float
    utilization_percent: float
    avg_access_latency_us: float
    bandwidth_utilization_percent: float
```

### 2d. SLO Analysis

**What it does:** Tracks Service Level Objective compliance for TTFT, TPOT, and end-to-end latency against configurable deadlines, computing violation rates and tail latency distributions.

**How it works:**
1. Accept SLO targets as part of simulation configuration
2. During simulation, tag each request with SLO compliance status
3. Compute violation rates and classify violations by severity
4. Report percentile distributions and goodput (requests meeting SLO)

This is an output component of the batch simulation. SLO targets are specified in the `BatchSimulationRequest`:

```python
class SLOTargets(BaseModel):
    ttft_target_ms: Optional[float] = Field(None, gt=0, description="Max acceptable TTFT")
    tpot_target_ms: Optional[float] = Field(None, gt=0, description="Max acceptable TPOT")
    e2e_target_ms: Optional[float] = Field(None, gt=0, description="Max acceptable end-to-end latency")

class SLOAnalysis(BaseModel):
    targets: SLOTargets
    ttft_violation_rate: float  # 0.0-1.0
    tpot_violation_rate: float
    e2e_violation_rate: float
    goodput_rps: float  # Requests meeting ALL SLOs per second
    goodput_ratio: float  # goodput / total throughput
    worst_case_ttft_ms: float
    worst_case_tpot_ms: float
    worst_case_e2e_ms: float
```

### 2e. Throughput-Latency Tradeoff Curves

**What it does:** Sweeps batch size and/or concurrency to produce throughput vs. latency tradeoff curves, identifying the optimal operating point for given SLO constraints.

**How it works:**
1. Run a series of simulations with varying batch sizes (or arrival rates)
2. For each point, record throughput and latency percentiles
3. Identify the Pareto-optimal operating points
4. Mark the "knee" of the curve where latency starts to degrade sharply
5. Optionally mark SLO-compliant region

This is exposed through `POST /api/v2/simulate/serving` with sweep configuration.

**Input Schema:**
```python
class ServingSimulationRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    sweep_config: Optional[SweepConfig] = Field(None, description="Parameter sweep for tradeoff curves")
    slo_targets: Optional[SLOTargets] = Field(None)
    # Single-point config (used when sweep_config is None)
    arrival_rate: Optional[float] = Field(None, gt=0)
    num_requests: int = Field(300, ge=10)
    max_batch_size: int = Field(256, ge=1)
    input_length_distribution: Optional[LengthDistribution] = None
    output_length_distribution: Optional[LengthDistribution] = None

class SweepConfig(BaseModel):
    parameter: Literal["arrival_rate", "max_batch_size", "max_num_batched_tokens"] = Field("arrival_rate")
    values: Optional[List[float]] = Field(None, description="Explicit values to sweep")
    min_value: Optional[float] = Field(None, description="Min value for auto-range")
    max_value: Optional[float] = Field(None, description="Max value for auto-range")
    num_points: int = Field(10, ge=3, le=50, description="Number of sweep points")
```

**Output Schema:**
```python
class ServingSimulationResponse(BaseModel):
    model_id: str
    hardware_name: str
    configuration: Dict[str, Any]
    # Single-point result (when no sweep)
    single_result: Optional[BatchSimulationResponse] = None
    # Sweep results (when sweep_config provided)
    sweep_results: Optional[List[SweepPoint]] = None
    optimal_operating_point: Optional[SweepPoint] = None
    slo_compliant_range: Optional[Dict[str, float]] = None  # {min_rate, max_rate}

class SweepPoint(BaseModel):
    parameter_value: float
    throughput_rps: float
    token_throughput_tps: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    e2e_p95_ms: float
    avg_batch_size: float
    meets_slo: bool
    goodput_rps: Optional[float] = None
```

---

## 3. Feature Category: Advanced Memory Modeling

**Current state in GenZ:** Single-tier memory model (device HBM only). Memory check is binary: model fits or requires offloading. Offloading uses a single effective bandwidth formula. No KV cache management, no eviction, no prefix caching.

**Gap reference:** Finding 11 Section 3 (System class has no multi-tier hierarchy), Finding 06 Section 2.3 (LLMServingSim multi-tier model with 0.93% error).

### 3a. Multi-Tier Memory Model

**What it does:** Extends GenZ's `System` class to model a 4-tier memory hierarchy: device HBM, host DDR, CXL-attached memory, and NVMe storage. Each tier has independent capacity, bandwidth, and latency parameters.

**How it works:**
1. Extend `System.__init__` with `host_mem_size`, `host_mem_bw`, `cxl_mem_size`, `cxl_mem_bw`, `storage_mem_size`, `storage_mem_bw` parameters (all defaulting to 0, meaning tier not present)
2. When computing memory requirements, assign data to tiers based on access pattern: weights to device, overflow KV cache to host, cold KV to CXL/storage
3. Compute effective bandwidth as a weighted harmonic mean across tiers based on data placement
4. Inject tier-crossing data movement costs into the operator execution timeline

**API Endpoint:** `POST /api/v2/memory/tiers`

**Input Schema:**
```python
class MultiTierMemoryRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    batch_size: int = Field(32, ge=1)
    input_tokens: int = Field(2048, ge=1)
    output_tokens: int = Field(256, ge=1)
    memory_tiers: List[MemoryTierConfig] = Field(
        ...,
        description="Memory tier specifications. Must include at least 'device' tier.",
        min_length=1
    )
    placement_policy: Literal["greedy", "balanced", "latency_optimal"] = Field(
        "greedy",
        description="How to place data across tiers: greedy fills fastest first, balanced distributes evenly, latency_optimal minimizes average access time"
    )
```

**Output Schema:**
```python
class MultiTierMemoryResponse(BaseModel):
    model_id: str
    total_memory_required_gb: float
    fits_in_device_only: bool
    tier_placement: List[TierPlacement]
    effective_memory_bandwidth_gbs: float
    effective_memory_latency_us: float
    performance_impact: PerformanceImpact
    recommendations: List[str]

class TierPlacement(BaseModel):
    tier: str
    data_type: str  # "weights", "kv_cache", "activations", "overflow"
    size_gb: float
    bandwidth_gbs: float
    access_latency_us: float
    access_frequency: Literal["high", "medium", "low"]

class PerformanceImpact(BaseModel):
    device_only_latency_ms: float
    multi_tier_latency_ms: float
    slowdown_factor: float
    bottleneck_tier: str
    bottleneck_reason: str  # "capacity" or "bandwidth"
```

### 3b. Memory Offloading Simulation

**What it does:** When KV cache exceeds device memory during serving, models the cost of offloading to host/CXL memory including data movement overhead, eviction decisions, and re-fetch latency.

**How it works:**
1. During batch simulation, track per-request KV cache accumulation
2. When device memory exceeds threshold (configurable, default 90%), trigger offloading
3. Select victim requests using configured eviction policy
4. Compute data movement cost: `transfer_time = data_size / tier_bandwidth + tier_latency`
5. When evicted data is needed (request resumes decode), compute re-fetch cost
6. Report offloading statistics: frequency, data volume, latency overhead

This is integrated into the BatchScheduler and CacheAnalysis. The offloading behavior is controlled via `memory_tiers` and `eviction_policy` parameters in those endpoints.

### 3c. Prefix Caching

**What it does:** Models radix-tree-based KV cache reuse across requests that share common prefixes (system prompts, RAG context). Estimates hit rates, memory savings, and performance improvements.

**How it works:**
1. Accept prefix configuration: common prefix length, estimated hit rate, or workload trace for automatic detection
2. For cached prefixes: skip KV computation for prefix tokens, only allocate new KV for unique suffixes
3. Compute savings: `compute_savings = prefix_length / total_length * hit_rate`
4. Memory model: prefix cache blocks are shared across requests, reducing total allocation
5. Support multi-tier prefix caching: device-local cache (fast, per-instance) + shared CPU cache (larger, cross-instance)

**Exposed through:** `POST /api/v2/cache/analyze` (see Section 2c) and as options in `BatchSimulationRequest`.

**Example:**
```python
# Analyze prefix caching impact for a RAG application
response = client.post("/api/v2/cache/analyze", json={
    "model_id": "llama3_8b",
    "hardware_name": "H100_80GB",
    "batch_size": 64,
    "input_tokens": 4096,
    "output_tokens": 256,
    "enable_prefix_caching": True,
    "prefix_length": 2048,  # 2K shared system prompt + RAG context
    "prefix_cache_hit_rate": 0.7,  # 70% of requests share prefix
    "block_size": 16
})

# Expected output:
# prefix_cache_analysis:
#   prefix_length_tokens: 2048
#   prefix_memory_gb: 0.25
#   estimated_hit_rate: 0.7
#   memory_savings_gb: 2.8
#   compute_savings_percent: 35.0
#   effective_ttft_speedup: 1.54
```

### 3d. Memory Fragmentation Analysis

**What it does:** Compares paged allocation (vLLM-style with configurable block sizes) vs. contiguous allocation for KV cache, quantifying internal fragmentation and its impact on effective memory capacity.

**How it works:**
1. Given batch size, sequence lengths, and block size, compute total blocks needed
2. Internal fragmentation = blocks_allocated * block_size - actual_tokens_used
3. Compare across block sizes (8, 16, 32, 64)
4. Report wasted memory and effective max batch size for each configuration

This is part of the `CacheAnalysisResponse.fragmentation_analysis` (see Section 2c).

---

## 4. Feature Category: Power and Energy Modeling

**Current state in GenZ:** Single function `get_energy()` in `power.py` with a 4-component utilization-proportional model (Static 30%, Compute 40%, Memory 20%, Network 10%). Returns total energy in kWh. No temporal power tracking, no per-component breakdown, no idle/active/standby states.

**Gap reference:** Finding 11 Section 11 (GenZ power limitations), Finding 06 Section 2.4 (LLMServingSim 7-component model with 1.34% error).

### 4a. 7-Component Power Model

**What it does:** Models power consumption across 7 system components with a 3-state accelerator model (idle/active/standby), matching LLMServingSim 2.0's validated approach.

**Components:**
1. **Accelerators (GPUs/TPUs):** 3-state model
   - Idle: base power when no work scheduled (e.g., H100: ~100W)
   - Active: full power during operator execution (e.g., H100: ~700W)
   - Standby: intermediate power between batches (e.g., H100: ~250W)
2. **CPUs:** Constant power when system active (e.g., ~200W per socket)
3. **DRAM:** Energy proportional to data volume transferred (e.g., ~3 pJ/bit)
4. **Interconnect links + switches:** Energy proportional to data volume (e.g., NVLink: ~5 pJ/bit)
5. **NICs:** Constant power per NIC (e.g., ~15W per 200G NIC)
6. **Storage devices:** Constant power per device (e.g., ~7W per NVMe SSD)
7. **Other (motherboard, cooling fans, VRMs):** Constant overhead (e.g., ~50W per node)

**How it works:**
1. For analytical mode: compute power from operator utilization breakdowns (existing GenZ data)
2. For simulation mode: track power state transitions over time during batch simulation
3. Apply PUE (Power Usage Effectiveness) multiplier for datacenter overhead (default 1.2)
4. Sum component contributions for total system power

**API Endpoint:** `GET /api/v2/power/breakdown` and `POST /api/v2/simulate/power`

### 4b. Energy-per-Token Metrics

**What it does:** Computes energy cost in Joules per token for both prefill and decode phases, enabling efficiency comparison across hardware and configurations.

**Formulas:**
```
prefill_energy_per_token = (prefill_power_w * prefill_latency_s) / (batch_size * input_tokens)
decode_energy_per_token  = (decode_power_w * tpot_s) / batch_size
total_energy_per_request = prefill_energy + decode_energy * output_tokens
```

### 4c. Power-Constrained Optimization

**What it does:** Finds the best configuration (batch size, parallelism, precision) that maximizes throughput while staying within a power budget.

**How it works:**
1. Accept power budget constraint (e.g., "max 3000W per node")
2. For each candidate configuration, estimate power consumption
3. Reject configurations exceeding budget
4. Return throughput-optimal configuration within budget

### 4d. TCO Calculator

**What it does:** Computes total cost of ownership over a deployment lifetime, including hardware acquisition, power consumption, cooling, networking, and operational costs.

**Input Schema:**
```python
class TCORequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    batch_size: int = Field(32, ge=1)
    target_throughput_rps: float = Field(10.0, gt=0, description="Target requests per second")
    input_tokens: int = Field(2048, ge=1)
    output_tokens: int = Field(256, ge=1)
    deployment_months: int = Field(36, ge=1, le=120, description="Deployment duration in months")
    electricity_cost_per_kwh: float = Field(0.10, gt=0, description="Electricity cost in USD/kWh")
    pue: float = Field(1.2, ge=1.0, le=3.0, description="Power Usage Effectiveness")
    hardware_cost_per_unit: Optional[float] = Field(None, description="Override per-unit hardware cost in USD")
    cloud_hourly_rate: Optional[float] = Field(None, description="Cloud instance hourly rate in USD")
    deployment_mode: Literal["on_prem", "cloud"] = Field("cloud")
    utilization_percent: float = Field(70.0, ge=0, le=100, description="Average utilization percentage")
```

**Output Schema:**
```python
class TCOResponse(BaseModel):
    deployment_mode: str
    hardware_name: str
    num_devices_required: int
    num_nodes_required: int

    # Cost breakdown (USD)
    hardware_cost: Optional[float]  # on_prem only
    monthly_cloud_cost: Optional[float]  # cloud only
    monthly_power_cost: float
    monthly_operational_cost: float

    total_cost_over_period: float
    cost_per_request: float
    cost_per_million_tokens: float

    # Power
    peak_power_w: float
    average_power_w: float
    monthly_energy_kwh: float

    # Performance
    achieved_throughput_rps: float
    meets_throughput_target: bool
    ttft_p95_ms: float
    tpot_p95_ms: float

    # Comparison
    alternatives: Optional[List[TCOAlternative]] = None

class TCOAlternative(BaseModel):
    hardware_name: str
    num_devices: int
    total_cost_over_period: float
    cost_per_million_tokens: float
    savings_vs_primary_percent: float
```

**API Endpoint for Power Breakdown:**

Input:
```python
class PowerBreakdownRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    batch_size: int = Field(32, ge=1)
    input_tokens: int = Field(2048, ge=1)
    output_tokens: int = Field(256, ge=1)
    num_devices: int = Field(1, ge=1)
    pue: float = Field(1.2, ge=1.0)
```

Output:
```python
class PowerBreakdownResponse(BaseModel):
    total_power_w: float
    pue_adjusted_power_w: float
    per_device_power_w: float
    components: PowerComponents
    energy_per_token_prefill_j: float
    energy_per_token_decode_j: float
    energy_per_request_j: float
    annual_energy_kwh: float

class PowerComponents(BaseModel):
    accelerator_idle_w: float
    accelerator_active_w: float
    accelerator_avg_w: float
    cpu_w: float
    dram_w: float
    interconnect_w: float
    nic_w: float
    storage_w: float
    other_w: float
```

---

## 5. Feature Category: Serving Configuration Optimization

**Current state in GenZ:** `get_best_parallization_strategy()` uses brute-force enumeration over TP/PP combinations. `get_pareto_optimal_performance()` varies batch size and parallelism with basic Pareto selection. No Bayesian optimization, no hidden constraint learning, no multi-objective optimization.

**Gap reference:** Finding 11 Section 14 (GenZ vs SCOOT), Finding 03 Section 1 (SCOOT algorithm), Finding 08 Section 1 (HEBO deep dive).

### 5a. Bayesian Optimization Engine

**What it does:** Uses HEBO-style Bayesian optimization with GP surrogate + MACE acquisition function ensemble to efficiently search the vLLM configuration space, replacing brute-force enumeration.

**How it works:**
1. Define search space: TP, PP, batch_size, max_num_seqs, precision, enable_chunked_prefill, enable_prefix_caching, block_size, gpu_memory_utilization
2. Initial sampling via Sobol quasi-random sequence (5-10 points)
3. Build Gaussian Process surrogate model with Matern 5/2 kernel
4. Use MACE ensemble (UCB + PI + EI) for acquisition
5. Random forest for hidden constraint (OOM) prediction
6. Each evaluation uses GenZ analytical model (milliseconds) instead of real benchmark (minutes)
7. Return best configuration after budget exhaustion

**API Endpoint:** `POST /api/v2/optimize/config`

**Input Schema:**
```python
class ConfigOptimizationRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    num_devices: int = Field(1, ge=1, description="Total devices available")
    optimization_target: Literal["throughput", "latency", "cost", "energy"] = Field("throughput")
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    search_space: Optional[SearchSpaceConfig] = Field(None, description="Override default search space")
    budget: int = Field(50, ge=10, le=500, description="Maximum number of evaluations")
    use_simulation: bool = Field(True, description="Use GenZ analytical model for evaluation")
    workload: Optional[WorkloadConfig] = Field(None, description="Workload characterization for context-aware optimization")

class OptimizationConstraints(BaseModel):
    max_ttft_ms: Optional[float] = Field(None, gt=0)
    max_tpot_ms: Optional[float] = Field(None, gt=0)
    max_e2e_ms: Optional[float] = Field(None, gt=0)
    max_memory_gb: Optional[float] = Field(None, gt=0)
    max_power_w: Optional[float] = Field(None, gt=0)
    max_cost_per_hour: Optional[float] = Field(None, gt=0)
    min_throughput_rps: Optional[float] = Field(None, gt=0)

class SearchSpaceConfig(BaseModel):
    tensor_parallel: List[int] = Field([1, 2, 4, 8])
    pipeline_parallel: List[int] = Field([1, 2, 4])
    batch_sizes: List[int] = Field([1, 4, 8, 16, 32, 64, 128])
    precisions: List[str] = Field(["bf16", "fp8", "int8", "int4"])
    max_num_seqs: Optional[List[int]] = Field(None)
    enable_chunked_prefill: List[bool] = Field([True, False])
    enable_prefix_caching: List[bool] = Field([True, False])
    block_sizes: List[int] = Field([8, 16, 32])
    gpu_memory_utilization: Optional[List[float]] = Field(None)

class WorkloadConfig(BaseModel):
    avg_input_tokens: int = Field(2048, ge=1)
    avg_output_tokens: int = Field(256, ge=1)
    arrival_rate_rps: float = Field(10.0, gt=0)
    prefix_sharing_ratio: float = Field(0.0, ge=0, le=1)
    workload_type: Literal["chat", "batch", "rag", "coding", "classification"] = Field("chat")
```

**Output Schema:**
```python
class ConfigOptimizationResponse(BaseModel):
    best_config: OptimalConfig
    performance: PerformanceMetrics
    search_summary: SearchSummary
    top_configs: List[RankedConfig]
    convergence_curve: List[ConvergencePoint]

class OptimalConfig(BaseModel):
    tensor_parallel: int
    pipeline_parallel: int
    batch_size: int
    precision: str
    max_num_seqs: int
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    block_size: int
    gpu_memory_utilization: float

class PerformanceMetrics(BaseModel):
    throughput_rps: float
    token_throughput_tps: float
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    memory_used_gb: float
    power_w: Optional[float]
    cost_per_request: Optional[float]
    meets_all_constraints: bool
    constraint_violations: List[str]

class SearchSummary(BaseModel):
    total_evaluations: int
    feasible_evaluations: int
    infeasible_evaluations: int
    infeasible_reasons: Dict[str, int]  # {"oom": 5, "constraint_violation": 3}
    wall_clock_seconds: float
    improvement_over_default_percent: float

class RankedConfig(BaseModel):
    rank: int
    config: OptimalConfig
    performance: PerformanceMetrics
    score: float

class ConvergencePoint(BaseModel):
    evaluation_number: int
    best_score_so_far: float
    current_score: float
```

### 5b. Multi-Objective Optimization

**What it does:** Produces Pareto-optimal configurations across throughput, latency, and cost using Expected Hypervolume Improvement (EHVI).

**API Endpoint:** `POST /api/v2/optimize/pareto`

**Input Schema:**
```python
class ParetoOptimizationRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    num_devices: int = Field(1, ge=1)
    objectives: List[OptimizationObjective] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="Objectives to optimize (2-4)"
    )
    constraints: Optional[OptimizationConstraints] = None
    search_space: Optional[SearchSpaceConfig] = None
    budget: int = Field(100, ge=20, le=1000)
    workload: Optional[WorkloadConfig] = None

class OptimizationObjective(BaseModel):
    metric: Literal["throughput_rps", "ttft_ms", "tpot_ms", "e2e_ms", "cost_per_request", "power_w", "memory_gb"]
    direction: Literal["maximize", "minimize"]
    weight: float = Field(1.0, gt=0, description="Relative importance weight")
    reference_point: Optional[float] = Field(None, description="EHVI reference point (worst acceptable value)")
```

**Output Schema:**
```python
class ParetoOptimizationResponse(BaseModel):
    pareto_frontier: List[ParetoPoint]
    num_pareto_points: int
    dominated_points: int
    total_evaluations: int
    hypervolume: float
    recommended_operating_point: ParetoPoint
    recommendation_reason: str

class ParetoPoint(BaseModel):
    config: OptimalConfig
    objectives: Dict[str, float]  # {metric_name: value}
    is_pareto_optimal: bool
    dominated_by: Optional[int] = None  # index of dominating point
```

**Example Request:**
```json
{
    "model_id": "llama3_8b",
    "hardware_name": "H100_80GB",
    "num_devices": 4,
    "objectives": [
        {"metric": "throughput_rps", "direction": "maximize", "weight": 1.0},
        {"metric": "ttft_ms", "direction": "minimize", "weight": 1.0}
    ],
    "constraints": {
        "max_tpot_ms": 50,
        "max_memory_gb": 320
    },
    "budget": 100,
    "workload": {
        "avg_input_tokens": 1024,
        "avg_output_tokens": 256,
        "arrival_rate_rps": 20.0,
        "workload_type": "chat"
    }
}
```

### 5c. Simulation-in-the-Loop

**What it does:** Uses BudSimulator's GenZ analytical model as a fast surrogate function inside the Bayesian optimization loop, replacing expensive real-hardware benchmarks.

**How it works:**
1. For each candidate configuration proposed by the optimizer, invoke GenZ's `estimate_prefill_performance()` and `estimate_decode_performance()` (millisecond cost)
2. Combine into end-to-end metrics
3. Feed back to the GP surrogate model
4. Optionally calibrate GenZ predictions against a small number of real measurements for improved accuracy

This is the default mode when `use_simulation=True` in `ConfigOptimizationRequest`. When `use_simulation=False`, the optimizer expects an external evaluation callback (for real benchmark integration).

### 5d. Parameter Sensitivity Analysis

**What it does:** Quantifies how sensitive performance metrics are to each configuration parameter, identifying which parameters matter most for a given model/hardware combination.

**API Endpoint:** `GET /api/v2/analyze/sensitivity`

**Input (query parameters):**
```python
class SensitivityAnalysisParams(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    num_devices: int = Field(1, ge=1)
    base_batch_size: int = Field(32, ge=1)
    input_tokens: int = Field(2048, ge=1)
    output_tokens: int = Field(256, ge=1)
    precision: str = Field("bf16")
    target_metric: Literal["throughput", "ttft", "tpot", "memory", "power"] = Field("throughput")
    num_samples: int = Field(50, ge=20, le=200, description="Number of random samples for sensitivity analysis")
```

**Output Schema:**
```python
class SensitivityAnalysisResponse(BaseModel):
    model_id: str
    hardware_name: str
    target_metric: str
    parameter_sensitivities: List[ParameterSensitivity]
    interaction_effects: List[InteractionEffect]
    recommendations: List[str]

class ParameterSensitivity(BaseModel):
    parameter: str
    importance_score: float  # 0.0-1.0, from variance-based sensitivity (Sobol indices)
    direction: Literal["positive", "negative", "non_monotonic"]
    range_tested: Dict[str, Any]  # {"min": val, "max": val}
    metric_range: Dict[str, float]  # {"min_metric": val, "max_metric": val}
    elasticity: float  # % change in metric per % change in parameter

class InteractionEffect(BaseModel):
    parameters: List[str]  # pair of interacting parameters
    interaction_strength: float  # 0.0-1.0
    description: str  # e.g., "TP and batch_size interact strongly: larger TP enables larger effective batch"
```

### 5e. Configuration Transfer

**What it does:** Transfers learned optimal configurations across similar models or hardware platforms by mapping parameter sensitivities and scaling relationships.

**How it works:**
1. Maintain a database of optimization results: (model, hardware, workload) -> optimal_config
2. When a new optimization request arrives, search for similar past results
3. Similarity metric: cosine similarity on feature vector [model_params, hidden_size, num_layers, hardware_flops, hardware_memory, hardware_bw]
4. If similarity > threshold (0.8), use past result as warm-start for new optimization
5. Apply scaling rules: e.g., if model is 2x larger, start with 2x TP

This is implemented internally in the `ConfigOptimizer` class and used automatically when `budget > 0`. Past results are stored in the `model_optimization_results` database table (see Finding 12 Section 7.4).

---

## 6. Feature Category: Cluster and Topology

**Current state in GenZ:** Basic TP/PP/DP/EP parallelism with analytical communication overhead. Single device type per simulation. No prefill-decode disaggregation, no heterogeneous device support, no network contention.

**Gap reference:** Finding 11 Section 13 (heterogeneous devices, P/D disaggregation), Finding 06 Section 3.3 (LLMServingSim P/D disaggregation with M:N mapping).

### 6a. Prefill-Decode Disaggregation

**What it does:** Models disaggregated serving where prefill and decode phases run on separate device pools (M prefill instances, N decode instances) with KV cache transfer between them.

**API Endpoint:** `POST /api/v2/cluster/disaggregate`

**Input Schema:**
```python
class DisaggregationRequest(BaseModel):
    model_id: str = Field(...)
    prefill_config: PhaseConfig = Field(...)
    decode_config: PhaseConfig = Field(...)
    kv_transfer: KVTransferConfig = Field(default_factory=KVTransferConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    num_requests: int = Field(300, ge=10)

class PhaseConfig(BaseModel):
    hardware_name: str = Field(...)
    num_instances: int = Field(1, ge=1, description="Number of MSG instances for this phase")
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    precision: str = Field("bf16")
    max_batch_size: int = Field(64, ge=1)

class KVTransferConfig(BaseModel):
    transfer_bandwidth_gbs: float = Field(25.0, gt=0, description="Network bandwidth between prefill and decode nodes (GB/s)")
    transfer_latency_us: float = Field(5.0, ge=0, description="Network latency for transfer initiation")
    transfer_mode: Literal["layer_wise", "bulk"] = Field(
        "layer_wise",
        description="Transfer KV layer-by-layer (overlaps with compute) or all at once after prefill completes"
    )
```

**Output Schema:**
```python
class DisaggregationResponse(BaseModel):
    # Per-phase results
    prefill_analysis: PhaseAnalysis
    decode_analysis: PhaseAnalysis
    kv_transfer_analysis: KVTransferAnalysis

    # System-level
    total_throughput_rps: float
    effective_ttft_ms: float
    effective_tpot_ms: float
    effective_e2e_ms: float

    # Comparison with unified serving
    unified_comparison: UnifiedComparison

    # Resource utilization
    prefill_utilization: float  # 0.0-1.0
    decode_utilization: float
    bottleneck: Literal["prefill", "decode", "kv_transfer", "balanced"]
    optimal_ratio_suggestion: str  # e.g., "Consider 2:3 prefill:decode ratio"

class PhaseAnalysis(BaseModel):
    hardware_name: str
    num_instances: int
    per_request_latency_ms: float
    throughput_rps: float
    memory_used_gb: float
    compute_utilization: float

class KVTransferAnalysis(BaseModel):
    kv_size_per_request_gb: float
    transfer_time_ms: float
    transfer_mode: str
    overlap_with_compute_ms: float  # Only for layer_wise mode
    effective_overhead_ms: float  # transfer_time - overlap

class UnifiedComparison(BaseModel):
    unified_throughput_rps: float
    unified_ttft_ms: float
    unified_tpot_ms: float
    disagg_speedup_throughput: float  # ratio
    disagg_speedup_ttft: float
    total_devices_unified: int
    total_devices_disaggregated: int
    cost_comparison: Optional[Dict[str, float]] = None
```

### 6b. Cluster Topology Optimization

**What it does:** Finds the optimal parallelism configuration and device count for a given model, hardware type, and SLO targets.

**API Endpoint:** `POST /api/v2/cluster/topology`

**Input Schema:**
```python
class TopologyOptimizationRequest(BaseModel):
    model_id: str = Field(...)
    hardware_name: str = Field(...)
    max_devices: int = Field(64, ge=1, le=4096, description="Maximum number of devices available")
    devices_per_node: int = Field(8, ge=1, description="Devices per physical node")
    optimization_target: Literal["throughput", "latency", "cost", "efficiency"] = Field("throughput")
    constraints: Optional[OptimizationConstraints] = None
    workload: Optional[WorkloadConfig] = None
    consider_disaggregation: bool = Field(False, description="Also evaluate P/D disaggregated configs")
```

**Output Schema:**
```python
class TopologyOptimizationResponse(BaseModel):
    recommended_config: TopologyConfig
    all_configs_evaluated: List[TopologyConfig]
    min_devices_required: int
    scaling_curve: List[ScalingPoint]

class TopologyConfig(BaseModel):
    num_devices: int
    num_nodes: int
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    expert_parallel: Optional[int] = None
    throughput_rps: float
    ttft_ms: float
    tpot_ms: float
    memory_per_device_gb: float
    communication_overhead_percent: float
    mfu: float  # Model FLOPs Utilization
    cost_per_hour: Optional[float] = None
    meets_constraints: bool

class ScalingPoint(BaseModel):
    num_devices: int
    throughput_rps: float
    efficiency: float  # throughput per device, normalized to 1-device baseline
    cost_per_request: Optional[float] = None
```

### 6c. Heterogeneous Device Support

**What it does:** Models clusters with mixed device types (e.g., H100 for prefill, L40 for decode, or GPU + PIM for attention offloading).

**How it works:**
1. Accept a device pool with mixed hardware types
2. Assign operators to devices based on their compute/memory characteristics
3. Model data movement between different device types
4. Compute per-device utilization and identify bottlenecks

This is primarily enabled through the `DisaggregationRequest` where `prefill_config.hardware_name` and `decode_config.hardware_name` can differ. Future extensions will support per-operator device assignment within a single phase.

### 6d. Network Contention Modeling

**What it does:** Models realistic interconnect contention when multiple communication operations share network bandwidth.

**How it works:**
1. Track concurrent communication operations across all parallelism dimensions
2. Apply bandwidth sharing when multiple collectives overlap: `effective_bw = link_bw / max(1, concurrent_operations)`
3. Add congestion factor based on topology (fat-tree: 10% overhead at scale, torus: 15%)
4. Add straggler overhead for large clusters: `1 + epsilon * sqrt(N/1000)`

This is integrated into GenZ's existing collective timing functions (`collective_times.py`) as an extension. The Phase 14 scale-aware overhead already provides a foundation; this adds explicit per-operation contention tracking during serving simulation.

---

## 7. Feature Category: Benchmarking and Validation

### 7a. Workload Characterization

**What it does:** Analyzes real request traces to extract statistical properties that inform simulation configuration.

**How it works:**
1. Accept a request trace file (JSON lines or CSV) with timestamps, prompt lengths, and output lengths
2. Compute statistics: arrival rate distribution, sequence length distributions, prefix sharing ratio, burstiness index
3. Fit parametric distributions to observed data
4. Generate a `WorkloadConfig` that can be fed into simulation endpoints

**Example usage via SDK:**
```python
from llm_memory_calculator import WorkloadGenerator

generator = WorkloadGenerator()
profile = generator.characterize_trace("/path/to/request_log.jsonl")

print(f"Arrival rate: {profile.arrival_rate_rps:.1f} req/s (pattern: {profile.arrival_pattern})")
print(f"Input tokens: {profile.input_length_distribution}")
print(f"Output tokens: {profile.output_length_distribution}")
print(f"Prefix sharing: {profile.prefix_sharing_ratio:.2f}")

# Use profile directly in simulation
result = scheduler.simulate(workload=profile.to_workload_config(), num_requests=1000)
```

### 7b. A/B Comparison

**What it does:** Runs two configurations side-by-side and produces a detailed comparison report with statistical significance testing.

**API Endpoint:** `POST /api/v2/benchmark/compare`

**Input Schema:**
```python
class ComparisonRequest(BaseModel):
    model_id: str = Field(...)
    config_a: SimulationConfigSpec = Field(..., description="First configuration")
    config_b: SimulationConfigSpec = Field(..., description="Second configuration")
    workload: Optional[WorkloadConfig] = None
    num_requests: int = Field(300, ge=10)
    comparison_metrics: List[str] = Field(
        ["throughput_rps", "ttft_p95_ms", "tpot_p95_ms", "memory_gb", "power_w"],
        description="Metrics to compare"
    )

class SimulationConfigSpec(BaseModel):
    name: str = Field(..., description="Configuration name for labeling")
    hardware_name: str = Field(...)
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    precision: str = Field("bf16")
    max_batch_size: int = Field(256, ge=1)
    max_num_batched_tokens: Optional[int] = None
    enable_chunked_prefill: bool = Field(False)
    enable_prefix_caching: bool = Field(False)
    block_size: int = Field(16)
```

**Output Schema:**
```python
class ComparisonResponse(BaseModel):
    config_a_name: str
    config_b_name: str
    metrics_comparison: List[MetricComparison]
    winner: str  # "config_a", "config_b", or "tie"
    summary: str  # Human-readable summary

class MetricComparison(BaseModel):
    metric: str
    config_a_value: float
    config_b_value: float
    delta: float
    delta_percent: float
    better: Literal["config_a", "config_b", "tie"]
    significant: bool  # Whether difference exceeds noise threshold (5%)
```

### 7c. Accuracy Validation

**What it does:** Compares BudSimulator predictions against real hardware measurements or reference simulator outputs, computing error metrics and calibration factors.

**How it works:**
1. Accept a validation dataset: [(config, measured_metrics), ...]
2. Run BudSimulator prediction for each config
3. Compute per-metric error: MAPE, RMSE, max error
4. Identify systematic biases (e.g., "TTFT consistently underestimated by 15%")
5. Optionally compute calibration factors that minimize prediction error

**SDK method:**
```python
from llm_memory_calculator import ServingSimulator

simulator = ServingSimulator()

# Load validation data
validation_data = [
    {
        "config": {"model": "llama3_8b", "hardware": "H100_80GB", "tp": 4, "batch_size": 32},
        "measured": {"ttft_ms": 45.2, "tpot_ms": 12.8, "throughput_rps": 28.5}
    },
    # ... more data points
]

report = simulator.validate_accuracy(validation_data)

print(f"TTFT MAPE: {report.ttft_mape_percent:.1f}%")
print(f"TPOT MAPE: {report.tpot_mape_percent:.1f}%")
print(f"Throughput MAPE: {report.throughput_mape_percent:.1f}%")

# Apply calibration
simulator.apply_calibration(report.calibration_factors)
```

### 7d. Regression Detection

**What it does:** Tracks performance predictions across BudSimulator versions or configuration changes, alerting when predictions change beyond expected thresholds.

**How it works:**
1. Store prediction results in database with version tag
2. When re-running with updated model/code, compare against historical baseline
3. Flag regressions: predictions that changed by >5% without configuration change
4. Generate a regression report with per-metric deltas

This is primarily a development-time tool implemented as a pytest fixture and CLI command:

```bash
# Run regression suite
python -m budsimulator.regression_test --baseline v1.0.0 --current HEAD

# Output:
# llama3_8b / H100 / TP4:   TTFT +2.1% (OK)  TPOT -0.3% (OK)  Throughput +1.8% (OK)
# llama3_70b / A100 / TP8:  TTFT +12.4% (REGRESSION!)  TPOT +0.5% (OK)
```

---

## 8. Complete API Specification

### Endpoint Summary

| Method | Path | Description | Section |
|---|---|---|---|
| `POST` | `/api/v2/simulate/serving` | Full serving simulation with optional sweep | 2e |
| `POST` | `/api/v2/simulate/batch` | Batch dynamics simulation | 2a |
| `POST` | `/api/v2/simulate/power` | Power/energy analysis (alias for power breakdown with simulation) | 4a |
| `POST` | `/api/v2/optimize/config` | Configuration optimization | 5a |
| `POST` | `/api/v2/optimize/pareto` | Multi-objective Pareto analysis | 5b |
| `GET` | `/api/v2/analyze/sensitivity` | Parameter sensitivity analysis | 5d |
| `POST` | `/api/v2/cache/analyze` | KV cache analysis | 2c |
| `POST` | `/api/v2/cluster/topology` | Topology optimization | 6b |
| `POST` | `/api/v2/cluster/disaggregate` | PD disaggregation analysis | 6a |
| `POST` | `/api/v2/benchmark/compare` | A/B comparison | 7b |
| `GET` | `/api/v2/power/breakdown` | Power component breakdown | 4a |
| `POST` | `/api/v2/memory/tiers` | Multi-tier memory analysis | 3a |

### Router Registration

New file: `BudSimulator/apis/routers/simulation_v2.py`

```python
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/v2", tags=["simulation-v2"])
```

Registration in `BudSimulator/apis/main.py`:
```python
from .routers import simulation_v2
app.include_router(simulation_v2.router)
```

### Detailed Endpoint Specifications

#### `POST /api/v2/simulate/serving`

Full serving simulation with optional parameter sweep for throughput-latency tradeoff curves.

- **Request:** `ServingSimulationRequest` (see Section 2e)
- **Response:** `ServingSimulationResponse` (see Section 2e)
- **Error Codes:**
  - `400` -- Invalid configuration (model not found, invalid parallelism)
  - `422` -- Validation error (field constraints)
  - `503` -- GenZ engine not available
- **Caching:** Results cached by hash of request body for 5 minutes. Sweep results cached for 15 minutes.
- **Rate Limiting:** 10 requests/minute (sweep), 30 requests/minute (single point)

**Example Request:**
```json
{
    "model_id": "llama3_8b",
    "hardware_name": "H100_80GB",
    "precision": "bf16",
    "tensor_parallel": 4,
    "pipeline_parallel": 1,
    "sweep_config": {
        "parameter": "arrival_rate",
        "min_value": 1.0,
        "max_value": 100.0,
        "num_points": 15
    },
    "slo_targets": {
        "ttft_target_ms": 200,
        "tpot_target_ms": 50
    },
    "num_requests": 300,
    "max_batch_size": 128,
    "input_length_distribution": {
        "distribution": "lognormal",
        "mean": 512,
        "std": 200,
        "min_length": 64,
        "max_length": 4096
    },
    "output_length_distribution": {
        "distribution": "uniform",
        "min_length": 32,
        "max_length": 512
    }
}
```

**Example Response (abbreviated):**
```json
{
    "model_id": "llama3_8b",
    "hardware_name": "H100_80GB",
    "configuration": {
        "tensor_parallel": 4,
        "pipeline_parallel": 1,
        "precision": "bf16"
    },
    "sweep_results": [
        {
            "parameter_value": 1.0,
            "throughput_rps": 1.0,
            "token_throughput_tps": 768,
            "ttft_p50_ms": 28.5,
            "ttft_p95_ms": 31.2,
            "tpot_p50_ms": 8.1,
            "tpot_p95_ms": 9.3,
            "e2e_p95_ms": 2410,
            "avg_batch_size": 1.0,
            "meets_slo": true,
            "goodput_rps": 1.0
        },
        {
            "parameter_value": 50.0,
            "throughput_rps": 42.3,
            "token_throughput_tps": 32487,
            "ttft_p50_ms": 85.2,
            "ttft_p95_ms": 187.4,
            "tpot_p50_ms": 12.5,
            "tpot_p95_ms": 28.7,
            "e2e_p95_ms": 7580,
            "avg_batch_size": 48.7,
            "meets_slo": true,
            "goodput_rps": 42.3
        },
        {
            "parameter_value": 100.0,
            "throughput_rps": 51.8,
            "token_throughput_tps": 39782,
            "ttft_p50_ms": 450.3,
            "ttft_p95_ms": 1250.7,
            "tpot_p50_ms": 35.2,
            "tpot_p95_ms": 85.3,
            "e2e_p95_ms": 22350,
            "avg_batch_size": 112.4,
            "meets_slo": false,
            "goodput_rps": 28.1
        }
    ],
    "optimal_operating_point": {
        "parameter_value": 50.0,
        "throughput_rps": 42.3,
        "ttft_p95_ms": 187.4,
        "tpot_p95_ms": 28.7,
        "meets_slo": true,
        "goodput_rps": 42.3
    },
    "slo_compliant_range": {
        "min_rate": 1.0,
        "max_rate": 65.0
    }
}
```

#### `POST /api/v2/simulate/batch`

Detailed batch dynamics simulation with per-iteration tracking.

- **Request:** `BatchSimulationRequest` (see Section 2a)
- **Response:** `BatchSimulationResponse` (see Section 2a)
- **Error Codes:** `400`, `422`, `503`
- **Rate Limiting:** 10 requests/minute

#### `POST /api/v2/simulate/power`

Power and energy simulation for a specific configuration.

- **Request:** `PowerBreakdownRequest` (see Section 4a)
- **Response:** `PowerBreakdownResponse` (see Section 4a)
- **Error Codes:** `400`, `422`, `503`
- **Caching:** Results cached by request hash for 30 minutes
- **Rate Limiting:** 30 requests/minute

#### `POST /api/v2/optimize/config`

Bayesian configuration optimization using GenZ as surrogate.

- **Request:** `ConfigOptimizationRequest` (see Section 5a)
- **Response:** `ConfigOptimizationResponse` (see Section 5a)
- **Error Codes:** `400`, `422`, `503`, `504` (timeout -- optimization exceeded time budget)
- **Rate Limiting:** 5 requests/minute (compute-intensive)
- **Timeout:** 120 seconds default, configurable via `X-Timeout` header

#### `POST /api/v2/optimize/pareto`

Multi-objective Pareto optimization.

- **Request:** `ParetoOptimizationRequest` (see Section 5b)
- **Response:** `ParetoOptimizationResponse` (see Section 5b)
- **Error Codes:** `400`, `422`, `503`, `504`
- **Rate Limiting:** 3 requests/minute

#### `GET /api/v2/analyze/sensitivity`

Parameter sensitivity analysis.

- **Request:** Query parameters per `SensitivityAnalysisParams` (see Section 5d)
- **Response:** `SensitivityAnalysisResponse` (see Section 5d)
- **Error Codes:** `400`, `422`, `503`
- **Caching:** Results cached for 1 hour (analysis is deterministic for same inputs)
- **Rate Limiting:** 10 requests/minute

#### `POST /api/v2/cache/analyze`

KV cache sizing, fragmentation, and prefix caching analysis.

- **Request:** `CacheAnalysisRequest` (see Section 2c)
- **Response:** `CacheAnalysisResponse` (see Section 2c)
- **Error Codes:** `400`, `422`, `503`
- **Caching:** 30 minutes
- **Rate Limiting:** 30 requests/minute

#### `POST /api/v2/cluster/topology`

Cluster topology and parallelism optimization.

- **Request:** `TopologyOptimizationRequest` (see Section 6b)
- **Response:** `TopologyOptimizationResponse` (see Section 6b)
- **Error Codes:** `400`, `422`, `503`
- **Rate Limiting:** 10 requests/minute

#### `POST /api/v2/cluster/disaggregate`

Prefill-decode disaggregation analysis.

- **Request:** `DisaggregationRequest` (see Section 6a)
- **Response:** `DisaggregationResponse` (see Section 6a)
- **Error Codes:** `400`, `422`, `503`
- **Rate Limiting:** 10 requests/minute

#### `POST /api/v2/benchmark/compare`

A/B configuration comparison.

- **Request:** `ComparisonRequest` (see Section 7b)
- **Response:** `ComparisonResponse` (see Section 7b)
- **Error Codes:** `400`, `422`, `503`
- **Rate Limiting:** 10 requests/minute

#### `GET /api/v2/power/breakdown`

Power component breakdown for a configuration.

- **Request:** Query parameters: `model_id`, `hardware_name`, `precision`, `tensor_parallel`, `batch_size`, `input_tokens`, `output_tokens`, `num_devices`, `pue`
- **Response:** `PowerBreakdownResponse` (see Section 4a)
- **Error Codes:** `400`, `422`, `503`
- **Caching:** 30 minutes
- **Rate Limiting:** 30 requests/minute

#### `POST /api/v2/memory/tiers`

Multi-tier memory analysis.

- **Request:** `MultiTierMemoryRequest` (see Section 3a)
- **Response:** `MultiTierMemoryResponse` (see Section 3a)
- **Error Codes:** `400`, `422`, `503`
- **Caching:** 30 minutes
- **Rate Limiting:** 30 requests/minute

---

## 9. SDK Class Design

### Module Location

All new classes are added under `llm_memory_calculator/serving/`:

```
llm-memory-calculator/
  src/llm_memory_calculator/
    serving/
      __init__.py
      simulator.py        # ServingSimulator
      batch_scheduler.py  # BatchScheduler
      memory_model.py     # MemoryModel
      power_model.py      # PowerModel
      config_optimizer.py # ConfigOptimizer
      slo_tracker.py      # SLOTracker
      workload.py         # WorkloadGenerator
```

Exports added to `llm_memory_calculator/__init__.py`:
```python
from .serving import (
    ServingSimulator,
    BatchScheduler,
    MemoryModel,
    PowerModel,
    ConfigOptimizer,
    SLOTracker,
    WorkloadGenerator,
)
```

### 9.1 ServingSimulator

**Purpose:** Top-level orchestrator that coordinates batch scheduling, memory management, power modeling, and SLO tracking for a complete serving simulation.

```python
class ServingSimulator:
    """Orchestrates dynamic LLM serving simulation.

    Combines GenZ analytical modeling with runtime simulation to predict
    serving system behavior under realistic workloads. Uses GenZ's
    per-operator roofline analysis as the cost model for each iteration.

    This is the primary entry point for all serving simulation features.
    It composes BatchScheduler, MemoryModel, PowerModel, and SLOTracker.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        expert_parallel: int = 1,
        system_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the serving simulator.

        Args:
            model: Model identifier (GenZ name or HuggingFace ID).
            hardware: Hardware configuration name (must exist in hardware registry).
            precision: Weight precision format.
            tensor_parallel: Tensor parallelism degree.
            pipeline_parallel: Pipeline parallelism degree.
            expert_parallel: Expert parallelism degree (for MoE models).
            system_config: Optional override for hardware config dict. If None,
                looks up hardware by name from the hardware registry/database.
        """
        ...

    def simulate(
        self,
        arrival_rate: float = 10.0,
        arrival_pattern: str = "poisson",
        num_requests: int = 300,
        input_length_distribution: Optional[Dict] = None,
        output_length_distribution: Optional[Dict] = None,
        max_batch_size: int = 256,
        max_num_batched_tokens: int = 8192,
        slo_targets: Optional[Dict] = None,
        enable_prefix_caching: bool = False,
        prefix_length: Optional[int] = None,
        memory_tiers: Optional[List[Dict]] = None,
        enable_power_tracking: bool = False,
    ) -> "ServingSimulationResult":
        """Run a complete serving simulation.

        Generates a request stream, schedules batches, tracks memory and
        optionally power, and collects per-request and aggregate metrics.

        Args:
            arrival_rate: Request arrival rate (requests/second).
            arrival_pattern: "poisson", "gamma", "bursty", or "constant".
            num_requests: Total number of requests to simulate.
            input_length_distribution: Dict with keys: distribution, min_length,
                max_length, mean, std. Default: uniform 128-2048.
            output_length_distribution: Dict with keys: distribution, min_length,
                max_length, mean, std. Default: uniform 64-512.
            max_batch_size: Maximum concurrent sequences.
            max_num_batched_tokens: Maximum tokens per scheduling iteration.
            slo_targets: Dict with keys: ttft_target_ms, tpot_target_ms, e2e_target_ms.
            enable_prefix_caching: Enable prefix cache modeling.
            prefix_length: Common prefix length in tokens (when prefix caching enabled).
            memory_tiers: List of tier configs [{tier, capacity_gb, bandwidth_gbs, latency_us}].
            enable_power_tracking: Track power consumption during simulation.

        Returns:
            ServingSimulationResult with aggregate metrics, per-request metrics,
            time series data, SLO analysis, and optional power profile.
        """
        ...

    def sweep(
        self,
        parameter: str,
        values: List[float],
        **simulation_kwargs,
    ) -> "SweepResult":
        """Sweep a single parameter to produce throughput-latency tradeoff curves.

        Args:
            parameter: Parameter to sweep ("arrival_rate", "max_batch_size",
                "max_num_batched_tokens").
            values: Values to test for the parameter.
            **simulation_kwargs: Other parameters passed to simulate().

        Returns:
            SweepResult with per-value results and optimal operating point.
        """
        ...

    def validate_accuracy(
        self,
        validation_data: List[Dict],
    ) -> "ValidationReport":
        """Compare predictions against real measurements.

        Args:
            validation_data: List of dicts, each containing:
                - config: Configuration dict.
                - measured: Dict of measured metrics.

        Returns:
            ValidationReport with per-metric error statistics and calibration factors.
        """
        ...

    def apply_calibration(self, calibration_factors: Dict[str, float]) -> None:
        """Apply calibration factors from validate_accuracy() to improve predictions.

        Args:
            calibration_factors: Dict mapping metric names to multiplicative factors.
        """
        ...
```

**Integration with existing classes:**
- Uses `estimate_prefill_performance()` and `estimate_decode_performance()` from `performance_estimator.py` as the per-iteration cost model
- Uses `get_hardware_config()` from `hardware.py` for hardware lookup
- Composes `BatchScheduler`, `MemoryModel`, `PowerModel`, `SLOTracker` internally

### 9.2 BatchScheduler

**Purpose:** Models iteration-level continuous batching, request queuing, and admission control.

```python
class BatchScheduler:
    """Continuous batching simulation engine.

    Models vLLM/Orca-style iteration-level scheduling where the batch
    composition changes at each decoding step. Tracks queue dynamics,
    batch size evolution, and per-request lifecycle.

    Uses GenZ analytical models to estimate per-iteration latency based
    on the current batch composition (number of prefill and decode tokens).
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        expert_parallel: int = 1,
        system_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the batch scheduler.

        Args:
            model: Model identifier.
            hardware: Hardware configuration name.
            precision: Weight precision.
            tensor_parallel: TP degree.
            pipeline_parallel: PP degree.
            expert_parallel: EP degree.
            system_config: Optional hardware config override dict.
        """
        ...

    def simulate(
        self,
        arrival_rate: float = 10.0,
        arrival_pattern: str = "poisson",
        num_requests: int = 300,
        input_length_distribution: Optional[Dict] = None,
        output_length_distribution: Optional[Dict] = None,
        max_batch_size: int = 256,
        max_num_batched_tokens: int = 8192,
        enable_chunked_prefill: bool = False,
        chunk_size: Optional[int] = None,
        memory_model: Optional["MemoryModel"] = None,
        slo_tracker: Optional["SLOTracker"] = None,
    ) -> "BatchSimulationResult":
        """Run batch scheduling simulation.

        Args:
            arrival_rate: Requests per second.
            arrival_pattern: Arrival distribution type.
            num_requests: Total requests.
            input_length_distribution: Input length config dict.
            output_length_distribution: Output length config dict.
            max_batch_size: Max concurrent sequences.
            max_num_batched_tokens: Max tokens per iteration.
            enable_chunked_prefill: Split large prefills into chunks.
            chunk_size: Chunk size for chunked prefill (required if enabled).
            memory_model: Optional MemoryModel for KV cache tracking.
            slo_tracker: Optional SLOTracker for SLO compliance.

        Returns:
            BatchSimulationResult with per-iteration and aggregate metrics.
        """
        ...

    def estimate_iteration_latency(
        self,
        num_prefill_tokens: int,
        num_decode_requests: int,
        context_lengths: List[int],
    ) -> float:
        """Estimate latency for a single scheduling iteration.

        Uses GenZ's prefill and decode analytical models to compute
        the cost of processing the current batch composition.

        Args:
            num_prefill_tokens: Total prefill tokens in this iteration.
            num_decode_requests: Number of requests in decode phase.
            context_lengths: List of current context lengths for decode requests.

        Returns:
            Iteration latency in milliseconds.
        """
        ...
```

### 9.3 MemoryModel

**Purpose:** Multi-tier memory management with KV cache lifecycle tracking.

```python
class MemoryModel:
    """Multi-tier memory model for LLM serving.

    Tracks memory allocation, KV cache placement, eviction, and migration
    across a configurable memory hierarchy (device HBM, host DDR, CXL,
    storage). Supports prefix caching with configurable sharing scope.

    Extends GenZ's static memory calculation with dynamic tracking suitable
    for serving simulation. Uses GenZ's model config to compute per-token
    KV cache sizes and weight memory requirements.
    """

    def __init__(
        self,
        model: str,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        tiers: Optional[List[Dict]] = None,
        eviction_policy: str = "lru",
        block_size: int = 16,
        enable_prefix_caching: bool = False,
    ) -> None:
        """Initialize the memory model.

        Args:
            model: Model identifier for computing KV cache sizes.
            precision: Weight/KV cache precision.
            tensor_parallel: TP degree (divides per-device memory).
            tiers: List of tier config dicts [{tier, capacity_gb, bandwidth_gbs, latency_us}].
                Default: single device tier with capacity from hardware config.
            eviction_policy: "lru", "lfu", or "fifo".
            block_size: KV cache block size in tokens.
            enable_prefix_caching: Enable prefix caching with radix tree.
        """
        ...

    def allocate_kv_cache(self, request_id: str, num_tokens: int) -> "AllocationResult":
        """Allocate KV cache blocks for a request.

        Args:
            request_id: Unique request identifier.
            num_tokens: Number of tokens to allocate cache for.

        Returns:
            AllocationResult with tier placement, latency overhead, and eviction info.
        """
        ...

    def free_kv_cache(self, request_id: str) -> None:
        """Free KV cache for a completed request.

        Args:
            request_id: Request whose cache to free.
        """
        ...

    def check_prefix_hit(self, prefix_tokens: int) -> "PrefixHitResult":
        """Check if a prefix is cached and on which tier.

        Args:
            prefix_tokens: Number of prefix tokens to check.

        Returns:
            PrefixHitResult with hit status, tier, and transfer cost if needed.
        """
        ...

    def get_memory_snapshot(self) -> "MemorySnapshot":
        """Get current memory usage across all tiers.

        Returns:
            MemorySnapshot with per-tier usage, fragmentation, and cache statistics.
        """
        ...

    def analyze_static(
        self,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
    ) -> "StaticMemoryAnalysis":
        """Static memory analysis without simulation (fast path).

        Computes memory requirements using GenZ's existing analytical model
        extended with multi-tier placement and fragmentation analysis.

        Args:
            batch_size: Number of concurrent requests.
            input_tokens: Input sequence length.
            output_tokens: Output sequence length.

        Returns:
            StaticMemoryAnalysis with tier breakdown, fragmentation, and prefix analysis.
        """
        ...
```

### 9.4 PowerModel

**Purpose:** 7-component power modeling with 3-state accelerator tracking.

```python
class PowerModel:
    """Component-based power model for LLM serving infrastructure.

    Models 7 power components (accelerator, CPU, DRAM, interconnect, NIC,
    storage, other) with a 3-state accelerator model (idle, active, standby).
    Achieves accuracy comparable to LLMServingSim 2.0 (1.34% error) through
    calibrated component power parameters.

    Can operate in two modes:
    - Static: Estimate power from operator utilization (existing GenZ data)
    - Temporal: Track power state transitions during simulation
    """

    # Default power parameters per hardware type (watts)
    HARDWARE_POWER_PROFILES = {
        "H100_80GB": {
            "accelerator_idle": 100, "accelerator_active": 700, "accelerator_standby": 250,
            "cpu_per_socket": 200, "dram_pj_per_bit": 3.0,
            "interconnect_pj_per_bit": 5.0, "nic_per_port": 15,
            "storage_per_device": 7, "other_per_node": 50
        },
        "A100_80GB": {
            "accelerator_idle": 60, "accelerator_active": 400, "accelerator_standby": 150,
            "cpu_per_socket": 180, "dram_pj_per_bit": 3.5,
            "interconnect_pj_per_bit": 6.0, "nic_per_port": 12,
            "storage_per_device": 7, "other_per_node": 45
        },
        # ... profiles for other hardware
    }

    def __init__(
        self,
        hardware: str,
        num_devices: int = 1,
        pue: float = 1.2,
        power_profile: Optional[Dict] = None,
    ) -> None:
        """Initialize the power model.

        Args:
            hardware: Hardware name for looking up default power profile.
            num_devices: Number of accelerator devices.
            pue: Power Usage Effectiveness (datacenter overhead multiplier).
            power_profile: Optional override for power parameters.
        """
        ...

    def estimate_power(
        self,
        compute_utilization: float,
        memory_utilization: float,
        comm_utilization: float,
    ) -> "PowerEstimate":
        """Estimate instantaneous power from utilization levels.

        Args:
            compute_utilization: Fraction of peak compute (0-1).
            memory_utilization: Fraction of peak memory bandwidth (0-1).
            comm_utilization: Fraction of peak interconnect bandwidth (0-1).

        Returns:
            PowerEstimate with per-component breakdown and total.
        """
        ...

    def estimate_energy_from_df(self, model_df: "DataFrame") -> "EnergyEstimate":
        """Estimate energy from a GenZ operator DataFrame.

        Enhanced replacement for GenZ's get_energy() function. Uses the
        7-component model with per-operator utilization data already
        computed by GenZ's roofline analysis.

        Args:
            model_df: GenZ operator-level DataFrame with utilization columns.

        Returns:
            EnergyEstimate with per-component energy and energy-per-token metrics.
        """
        ...

    def estimate_serving_power(
        self,
        prefill_latency_ms: float,
        decode_latency_ms: float,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
        prefill_compute_util: float = 0.8,
        decode_memory_util: float = 0.7,
    ) -> "ServingPowerEstimate":
        """Estimate power for a serving scenario.

        Combines prefill (compute-bound, high utilization) and decode
        (memory-bound, lower compute utilization) phases with idle time
        between batches to compute average and peak power.

        Args:
            prefill_latency_ms: Per-request prefill latency.
            decode_latency_ms: Per-token decode latency.
            batch_size: Concurrent batch size.
            input_tokens: Average input length.
            output_tokens: Average output length.
            prefill_compute_util: Compute utilization during prefill.
            decode_memory_util: Memory bandwidth utilization during decode.

        Returns:
            ServingPowerEstimate with phase breakdown and energy-per-token.
        """
        ...
```

### 9.5 ConfigOptimizer

**Purpose:** Bayesian optimization engine for serving configuration tuning.

```python
class ConfigOptimizer:
    """Bayesian optimization engine for LLM serving configuration.

    Uses a Gaussian Process surrogate model with MACE acquisition function
    ensemble (UCB + PI + EI) and random forest hidden constraint learning,
    following the SCOOT/HEBO methodology.

    Evaluations use GenZ analytical models as a fast surrogate (milliseconds
    per evaluation), enabling exploration of thousands of configurations
    in seconds rather than the hours required for real benchmarking.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        num_devices: int = 1,
        system_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the configuration optimizer.

        Args:
            model: Model identifier.
            hardware: Hardware configuration name.
            num_devices: Total available devices.
            system_config: Optional hardware config override.
        """
        ...

    def optimize(
        self,
        target: str = "throughput",
        constraints: Optional[Dict] = None,
        search_space: Optional[Dict] = None,
        budget: int = 50,
        workload: Optional[Dict] = None,
        warm_start: Optional[List[Dict]] = None,
    ) -> "OptimizationResult":
        """Run single-objective configuration optimization.

        Args:
            target: Metric to optimize ("throughput", "latency", "cost", "energy").
            constraints: Dict with constraint keys (max_ttft_ms, max_tpot_ms, etc.).
            search_space: Override default search space.
            budget: Maximum number of evaluations.
            workload: Workload characterization for context-aware optimization.
            warm_start: Previous optimization results for transfer learning.

        Returns:
            OptimizationResult with best config, performance, search summary.
        """
        ...

    def optimize_pareto(
        self,
        objectives: List[Dict],
        constraints: Optional[Dict] = None,
        search_space: Optional[Dict] = None,
        budget: int = 100,
        workload: Optional[Dict] = None,
    ) -> "ParetoResult":
        """Run multi-objective Pareto optimization using EHVI.

        Args:
            objectives: List of {"metric", "direction", "weight", "reference_point"}.
            constraints: Optional constraints dict.
            search_space: Override default search space.
            budget: Maximum evaluations.
            workload: Workload characterization.

        Returns:
            ParetoResult with Pareto frontier and recommended operating point.
        """
        ...

    def sensitivity_analysis(
        self,
        target_metric: str = "throughput",
        num_samples: int = 50,
        base_config: Optional[Dict] = None,
    ) -> "SensitivityResult":
        """Analyze parameter sensitivity using Sobol indices.

        Args:
            target_metric: Metric to analyze sensitivity for.
            num_samples: Number of random samples.
            base_config: Base configuration for relative analysis.

        Returns:
            SensitivityResult with per-parameter importance scores.
        """
        ...
```

### 9.6 SLOTracker

**Purpose:** Tracks SLO compliance during serving simulation.

```python
class SLOTracker:
    """Service Level Objective tracker for serving simulation.

    Monitors TTFT, TPOT, and end-to-end latency against configurable
    targets. Computes violation rates, goodput, and tail latency
    distributions. Reports real-time SLO compliance during simulation.
    """

    def __init__(
        self,
        ttft_target_ms: Optional[float] = None,
        tpot_target_ms: Optional[float] = None,
        e2e_target_ms: Optional[float] = None,
    ) -> None:
        """Initialize SLO tracker with target thresholds.

        Args:
            ttft_target_ms: Maximum acceptable TTFT in milliseconds.
            tpot_target_ms: Maximum acceptable TPOT in milliseconds.
            e2e_target_ms: Maximum acceptable end-to-end latency in milliseconds.
        """
        ...

    def record_request(
        self,
        request_id: str,
        ttft_ms: float,
        tpot_ms: float,
        e2e_ms: float,
        queue_wait_ms: float = 0.0,
    ) -> "SLOStatus":
        """Record metrics for a completed request.

        Args:
            request_id: Unique request identifier.
            ttft_ms: Time to first token.
            tpot_ms: Average time per output token.
            e2e_ms: End-to-end latency.
            queue_wait_ms: Time spent waiting in queue.

        Returns:
            SLOStatus for this request (met/violated per target).
        """
        ...

    def get_summary(self) -> "SLOSummary":
        """Get cumulative SLO summary.

        Returns:
            SLOSummary with violation rates, percentiles, and goodput.
        """
        ...

    def get_percentiles(self, metric: str, percentiles: List[float] = [50, 95, 99]) -> Dict[str, float]:
        """Get percentile values for a metric.

        Args:
            metric: "ttft", "tpot", "e2e", or "queue_wait".
            percentiles: Percentile levels to compute.

        Returns:
            Dict mapping percentile labels to values (e.g., {"p50": 45.2, "p95": 120.3}).
        """
        ...
```

### 9.7 WorkloadGenerator

**Purpose:** Generates synthetic request streams and characterizes real request traces.

```python
class WorkloadGenerator:
    """Synthetic workload generation and real trace characterization.

    Generates request streams with configurable arrival patterns and
    sequence length distributions. Also provides tools to analyze
    real request traces and extract statistical properties.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the workload generator.

        Args:
            seed: Random seed for reproducibility. Uses instance-local RNG
                (does not affect global random state).
        """
        ...

    def generate(
        self,
        num_requests: int = 300,
        arrival_rate: float = 10.0,
        arrival_pattern: str = "poisson",
        input_length_distribution: Optional[Dict] = None,
        output_length_distribution: Optional[Dict] = None,
    ) -> "RequestStream":
        """Generate a synthetic request stream.

        Args:
            num_requests: Number of requests to generate.
            arrival_rate: Requests per second.
            arrival_pattern: "poisson", "gamma", "bursty", or "constant".
            input_length_distribution: Dict with keys: distribution, min_length,
                max_length, mean, std.
            output_length_distribution: Same format for output lengths.

        Returns:
            RequestStream containing list of Request objects with arrival times
            and sequence lengths.
        """
        ...

    def characterize_trace(self, trace_path: str) -> "TraceProfile":
        """Analyze a real request trace file.

        Accepts JSON lines format where each line contains:
        {"timestamp": float, "prompt_tokens": int, "completion_tokens": int}

        Or CSV with columns: timestamp, prompt_tokens, completion_tokens.

        Args:
            trace_path: Path to trace file.

        Returns:
            TraceProfile with fitted distributions, arrival statistics,
            and prefix sharing analysis.
        """
        ...

    def from_shareGPT(self, path: str, arrival_rate: float = 10.0) -> "RequestStream":
        """Load a ShareGPT-format conversation dataset as a request stream.

        Args:
            path: Path to ShareGPT JSON file.
            arrival_rate: Desired arrival rate.

        Returns:
            RequestStream with sequence lengths derived from conversations.
        """
        ...

    @staticmethod
    def preset(name: str) -> Dict:
        """Get a preset workload configuration.

        Args:
            name: Preset name. Available presets:
                - "chat": Interactive chatbot (short prompts, medium outputs)
                - "rag": RAG application (long prompts with shared prefix, short outputs)
                - "batch": Batch processing (variable lengths, high throughput)
                - "coding": Code generation (medium prompts, long outputs)
                - "classification": Classification (long prompts, very short outputs)

        Returns:
            Dict suitable for passing as simulation kwargs.
        """
        presets = {
            "chat": {
                "input_length_distribution": {"distribution": "lognormal", "mean": 256, "std": 150, "min_length": 32, "max_length": 2048},
                "output_length_distribution": {"distribution": "lognormal", "mean": 128, "std": 80, "min_length": 16, "max_length": 1024},
                "arrival_pattern": "poisson",
            },
            "rag": {
                "input_length_distribution": {"distribution": "normal", "mean": 4096, "std": 1024, "min_length": 2048, "max_length": 8192},
                "output_length_distribution": {"distribution": "uniform", "min_length": 64, "max_length": 512},
                "arrival_pattern": "poisson",
                "enable_prefix_caching": True,
                "prefix_length": 2048,
            },
            "batch": {
                "input_length_distribution": {"distribution": "uniform", "min_length": 128, "max_length": 4096},
                "output_length_distribution": {"distribution": "uniform", "min_length": 128, "max_length": 2048},
                "arrival_pattern": "constant",
            },
            "coding": {
                "input_length_distribution": {"distribution": "lognormal", "mean": 512, "std": 300, "min_length": 64, "max_length": 4096},
                "output_length_distribution": {"distribution": "lognormal", "mean": 1024, "std": 500, "min_length": 128, "max_length": 8192},
                "arrival_pattern": "bursty",
            },
            "classification": {
                "input_length_distribution": {"distribution": "normal", "mean": 2048, "std": 512, "min_length": 512, "max_length": 4096},
                "output_length_distribution": {"distribution": "uniform", "min_length": 1, "max_length": 10},
                "arrival_pattern": "poisson",
            },
        }
        return presets[name]
```

---

## 10. Backward Compatibility Matrix

### 10.1 Existing Endpoint to v2 Equivalent Mapping

Every existing v1 endpoint continues to function unchanged. The v2 endpoints provide enhanced versions with additional capabilities.

| v1 Endpoint | v1 Function | v2 Equivalent | Migration Notes |
|---|---|---|---|
| `POST /api/models/calculate` | `estimate_memory()` | `POST /api/v2/memory/tiers` | v1 returns `MemoryReport`. v2 adds multi-tier placement and fragmentation. v1 remains unchanged. |
| `POST /api/models/compare` | `estimate_memory()` per model | `POST /api/v2/benchmark/compare` | v1 compares memory only. v2 compares full performance. v1 remains unchanged. |
| `POST /api/models/analyze` | `estimate_memory()` per seq | `GET /api/v2/analyze/sensitivity` | v1 sweeps sequence length. v2 sweeps all parameters with sensitivity analysis. v1 remains unchanged. |
| `POST /api/usecases/{id}/optimize-hardware` | `find_best_hardware_for_usecase()` | `POST /api/v2/optimize/config` | v1 uses brute-force search. v2 uses Bayesian optimization. v1 remains unchanged. |
| `POST /api/usecases/{id}/recommendations` | `_estimate_performance()` heuristic | `POST /api/v2/simulate/serving` | v1 uses rough heuristics. v2 uses full serving simulation. v1 remains unchanged. |
| `POST /api/simulator/estimate-training` | `TrainingMemoryCalculator` | No v2 equivalent | Training estimation is already well-structured. No changes needed. |
| `POST /api/simulator/recommend-cluster` | `TrainingClusterSelector` | `POST /api/v2/cluster/topology` | v1 is training-focused. v2 adds inference topology optimization. Both coexist. |
| `BudSimulator.run()` (not exposed) | `SimulationEngine.simulate()` | `POST /api/v2/simulate/serving` | v2 finally exposes the BudSimulator class via API endpoints. |
| `estimate_prefill_performance()` | `prefill_moddeling()` | `ServingSimulator.simulate()` | v1 SDK function remains unchanged. v2 SDK wraps it in the ServingSimulator orchestrator. |
| `estimate_decode_performance()` | `decode_moddeling()` | `ServingSimulator.simulate()` | Same as above. |
| `estimate_end_to_end_performance()` | Combines prefill+decode | `ServingSimulator.simulate()` | v1 combines phases statically. v2 simulates dynamically. |
| `estimate_chunked_performance()` | `chunked_moddeling()` | `BatchScheduler` with `enable_chunked_prefill` | v1 models single chunked pass. v2 models chunked prefill within continuous batching. |
| `get_best_parallelization_strategy()` | Brute-force TP/PP search | `ConfigOptimizer.optimize()` | v1 searches TP/PP only. v2 searches full parameter space with BO. |
| `get_pareto_optimal_performance()` | Brute-force Pareto | `ConfigOptimizer.optimize_pareto()` | v1 uses paretoset library on brute-force results. v2 uses EHVI with GP surrogate. |
| `get_energy()` | Single utilization formula | `PowerModel.estimate_energy_from_df()` | v1 has 4 components. v2 has 7 components with 3-state accelerator model. v1 function preserved. |
| `get_hardware_config()` | Hardware lookup | No change | Unchanged, used by v2 internals. |
| `HardwareOptimizer.find_best_hardware_for_usecase()` | GenZ perf evaluation | `ConfigOptimizer.optimize()` | v1 evaluates fixed model/hardware combos. v2 adds BO-driven search. |

### 10.2 SDK Function Preservation

All existing SDK functions in `llm_memory_calculator` are preserved with identical signatures:

```python
# These functions remain unchanged and functional
from llm_memory_calculator import (
    # Memory
    calculate_memory,
    estimate_memory,
    analyze_hf_model,
    compare_models,
    estimate_max_batch_size,
    analyze_attention_efficiency,

    # Performance
    estimate_prefill_performance,
    estimate_decode_performance,
    estimate_end_to_end_performance,
    estimate_chunked_performance,
    compare_performance_configurations,

    # Parallelism
    get_various_parallelization,
    get_best_parallelization_strategy,
    get_pareto_optimal_performance,
    get_minimum_system_size,

    # Hardware
    get_hardware_config,
    get_all_hardware,
    get_hardware_by_type,
    search_hardware,

    # Types
    MemoryReport,
    ModelMemoryCalculator,
    HuggingFaceConfigLoader,
    UniversalParameterCounter,
)

# NEW v2 classes added alongside existing functions
from llm_memory_calculator import (
    ServingSimulator,
    BatchScheduler,
    MemoryModel,
    PowerModel,
    ConfigOptimizer,
    SLOTracker,
    WorkloadGenerator,
)
```

### 10.3 Database Schema Migration

New tables are additive and do not modify existing tables. Migration uses Alembic:

```python
# alembic/versions/xxx_add_v2_tables.py

def upgrade():
    # New table for simulation results
    op.create_table(
        'simulation_results',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('simulation_type', sa.String(), nullable=False),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('hardware_name', sa.String(), nullable=False),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('result_json', sa.Text(), nullable=False),
        sa.Column('latency_ms', sa.Float()),
        sa.Column('throughput_tps', sa.Float()),
        sa.Column('memory_used_gb', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # New table for optimization results (transfer learning cache)
    op.create_table(
        'optimization_results',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('hardware_name', sa.String(), nullable=False),
        sa.Column('optimization_type', sa.String(), nullable=False),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('result_json', sa.Text(), nullable=False),
        sa.Column('score', sa.Float()),
        sa.Column('workload_hash', sa.String()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # New table for power profiles
    op.create_table(
        'power_profiles',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('hardware_name', sa.String(), nullable=False, unique=True),
        sa.Column('accelerator_idle_w', sa.Float()),
        sa.Column('accelerator_active_w', sa.Float()),
        sa.Column('accelerator_standby_w', sa.Float()),
        sa.Column('cpu_per_socket_w', sa.Float()),
        sa.Column('dram_pj_per_bit', sa.Float()),
        sa.Column('interconnect_pj_per_bit', sa.Float()),
        sa.Column('nic_per_port_w', sa.Float()),
        sa.Column('storage_per_device_w', sa.Float()),
        sa.Column('other_per_node_w', sa.Float()),
    )

    # New table for validation data
    op.create_table(
        'validation_data',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('hardware_name', sa.String(), nullable=False),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('predicted_json', sa.Text(), nullable=False),
        sa.Column('measured_json', sa.Text(), nullable=False),
        sa.Column('error_percent', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Extend hardware table with power fields (additive, nullable)
    op.add_column('hardware', sa.Column('power_tdp', sa.Float()))
    op.add_column('hardware', sa.Column('power_idle', sa.Float()))
    op.add_column('hardware', sa.Column('pue_factor', sa.Float(), server_default='1.2'))

def downgrade():
    op.drop_table('validation_data')
    op.drop_table('power_profiles')
    op.drop_table('optimization_results')
    op.drop_table('simulation_results')
    op.drop_column('hardware', 'power_tdp')
    op.drop_column('hardware', 'power_idle')
    op.drop_column('hardware', 'pue_factor')
```

### 10.4 Deprecation Timeline

No existing features are deprecated in v2. The following functions are candidates for future deprecation once v2 features are mature:

| Function/Endpoint | Status in v2 | Deprecation Target |
|---|---|---|
| `get_energy()` in `power.py` | Superseded by `PowerModel.estimate_energy_from_df()` | v3 (6+ months) |
| `_estimate_performance()` heuristic in `usecases.py` | Superseded by GenZ-backed simulation | v3 (6+ months) |
| `BudSimulator._SIM_TYPE_TO_FEATURES` static mapping | Replaced by dynamic feature composition | v3 |
| `prefill_moddeling` / `decode_moddeling` (misspelled) | Wrappers added: `prefill_modeling` / `decode_modeling` | Never remove originals; add aliases |

### 10.5 Migration Guide for Existing Users

**API Users (frontend, external consumers):**
- No changes required. All `/api/` endpoints continue to work identically.
- To access new features, call `/api/v2/` endpoints.
- Response schemas for v1 endpoints do not change.

**SDK Users (Python package consumers):**
- No changes required. All existing imports work identically.
- To access new features, import from `llm_memory_calculator.serving`.
- The new `ServingSimulator` class is the recommended entry point for all new development.

**Database:**
- Run `alembic upgrade head` to apply new table creation.
- No existing tables or columns are modified.
- New nullable columns added to `hardware` table do not affect existing queries.

**Example: Migrating from v1 brute-force optimization to v2 Bayesian optimization:**

```python
# v1 (still works, unchanged)
from llm_memory_calculator import get_best_parallelization_strategy
result = get_best_parallelization_strategy(
    stage='decode',
    model='llama3_8b',
    total_nodes=8,
    batch_size=32,
    input_tokens=2048,
    output_tokens=256,
    system_name=hw_config,
    bits='bf16'
)

# v2 (enhanced, recommended for new code)
from llm_memory_calculator import ConfigOptimizer
optimizer = ConfigOptimizer(
    model="llama3_8b",
    hardware="H100_80GB",
    num_devices=8
)
result = optimizer.optimize(
    target="throughput",
    constraints={"max_ttft_ms": 200, "max_tpot_ms": 50},
    workload={"avg_input_tokens": 2048, "avg_output_tokens": 256, "arrival_rate_rps": 20},
    budget=100
)
print(f"Best config: TP={result.best_config.tensor_parallel}, "
      f"batch={result.best_config.batch_size}, "
      f"throughput={result.performance.throughput_rps:.1f} req/s")
print(f"Improvement over brute-force default: {result.search_summary.improvement_over_default_percent:.1f}%")
```

---

## Appendix A: Pydantic Model File Layout

All Pydantic schemas for v2 endpoints are placed in a new file to avoid modifying the existing `apis/schemas.py`:

**File:** `BudSimulator/apis/schemas_v2.py`

This file contains all request/response models defined in Sections 2-7 of this document. The existing `schemas.py` is not modified.

## Appendix B: Dependencies

New Python package dependencies required for v2 features:

| Package | Purpose | Required By |
|---|---|---|
| `scikit-learn` | Random forest for hidden constraint learning | ConfigOptimizer |
| `scipy` | Statistical distributions (gamma, lognormal) | WorkloadGenerator |
| `paretoset` | Pareto frontier computation | ConfigOptimizer (already a dependency) |

Optional dependencies (enhance but not required):

| Package | Purpose | Required By |
|---|---|---|
| `hebo` | HEBO Bayesian optimization backend | ConfigOptimizer (advanced mode) |
| `botorch` | GP surrogate model with EHVI | ConfigOptimizer (advanced mode) |
| `gpytorch` | Gaussian Process implementation | ConfigOptimizer (advanced mode) |

When optional dependencies are not installed, ConfigOptimizer falls back to a simpler grid search + random forest approach that still outperforms brute-force enumeration.

## Appendix C: File Index for New Code

| File | Lines (est.) | Purpose |
|---|---|---|
| `llm_memory_calculator/serving/__init__.py` | 20 | Package exports |
| `llm_memory_calculator/serving/simulator.py` | 400 | ServingSimulator class |
| `llm_memory_calculator/serving/batch_scheduler.py` | 500 | BatchScheduler class |
| `llm_memory_calculator/serving/memory_model.py` | 450 | MemoryModel class |
| `llm_memory_calculator/serving/power_model.py` | 300 | PowerModel class |
| `llm_memory_calculator/serving/config_optimizer.py` | 600 | ConfigOptimizer class |
| `llm_memory_calculator/serving/slo_tracker.py` | 200 | SLOTracker class |
| `llm_memory_calculator/serving/workload.py` | 350 | WorkloadGenerator class |
| `BudSimulator/apis/routers/simulation_v2.py` | 500 | v2 API router |
| `BudSimulator/apis/schemas_v2.py` | 600 | v2 Pydantic schemas |
| `BudSimulator/src/db/simulation_schema.py` | 150 | v2 database tables |
| **Total** | **~4,070** | |
