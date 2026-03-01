# BudEvolve: Reverse-Optimization System for LLM Serving

**Date**: 2026-03-02
**Status**: Design Approved
**Scope**: LLM Serving (inference) — training optimization deferred to Phase 2

## Problem Statement

BudSimulator currently operates in "forward mode": given a model + hardware + config, it predicts performance. BudEvolve inverts this — given performance targets, it discovers:

1. **Optimal configurations** (TP, PP, batch, precision) via multi-objective numeric search
2. **Novel algorithms** (scheduling, caching, memory management) via LLM-driven evolution
3. **Ideal hardware specs** (FLOPS, memory BW, capacity) via hardware design space exploration

## Architecture

```
                    BudEvolve CLI
   budsim-evolve optimize | evolve | explore | analyze
                        |
         +--------------+--------------+
         |              |              |
   NumericOptimizer  AlgorithmEvolver  HardwareExplorer
   (pymoo NSGA-II)  (OpenEvolve+LLM)  (pymoo NSGA-II)
   No LLM needed    LLM required      No LLM needed
         |              |              |
         +--------------+--------------+
                        |
              BudSim Evaluation API
              +-------------------+
              | RooflineAnalyzer  |  Per-op bottleneck insights
              | GenZ Engine       |  1-10ms analytical eval
              | ServingSimulator  |  50-200ms event-driven sim
              | Memory+Power      |  Memory tier + power models
              +-------------------+
```

### Design Principles

- **Right tool for each job**: pymoo for numeric optimization, OpenEvolve for code evolution
- **BudSim as the fitness function**: All modes evaluate candidates through BudSim's analytical engine
- **Roofline feedback loop**: Per-operator bottleneck analysis feeds into OpenEvolve prompts so the LLM understands *why* solutions are slow
- **No LLM waste**: Numeric optimization (configs, hardware) uses pymoo directly — no LLM API calls for parameter sweeps

## Components

### 1. BudSim Evaluation API (`evaluator.py`)

Unified interface wrapping BudSim as a fitness function for all optimization modes.

```python
class BudSimEvaluator:
    def evaluate_config(self, config: ServingConfig) -> EvalResult:
        """Fast analytical evaluation (~1-10ms). Used by NumericOptimizer."""

    def evaluate_with_simulation(self, config, workload) -> EvalResult:
        """Full event-driven simulation (~50-200ms). Higher fidelity."""

    def evaluate_algorithm(self, algorithm_path: str, workload) -> EvalResult:
        """Evaluate an evolved algorithm by injecting it into the simulator."""

    def evaluate_hardware(self, hw_spec: HardwareSpec) -> EvalResult:
        """Evaluate a hypothetical hardware design."""

    def get_roofline_insights(self, config) -> RooflineReport:
        """Extract per-operator compute vs memory bottleneck analysis."""
```

**EvalResult** contains:
- `throughput_rps`, `token_throughput_tps`
- `ttft_ms`, `tpot_ms`, `e2e_latency_ms`
- `memory_gb` (peak), `power_w` (average)
- `slo_compliance_rate` (0.0 - 1.0)
- `roofline`: per-operator bottleneck data
- `cost_estimate`: $/1M tokens (using cloud pricing from hardware configs)

### 2. Roofline Analyzer (`roofline_analyzer.py`)

Extracts per-operator performance insights from GenZ's analytical model.

```python
class RooflineAnalyzer:
    def analyze(self, model, hardware, config) -> RooflineReport:
        """Run per-operator roofline analysis."""

@dataclass
class RooflineReport:
    overall_bottleneck: str          # "compute" | "memory_bandwidth" | "interconnect"
    compute_utilization: float       # 0.0 - 1.0
    memory_bw_utilization: float     # 0.0 - 1.0
    interconnect_utilization: float  # 0.0 - 1.0
    per_operator: List[OperatorInsight]  # Per-op breakdown
    recommendations: List[str]       # Human-readable optimization suggestions
```

**OperatorInsight** per operator:
- `name`: "attention_prefill", "ffn_decode", "kv_cache_load", etc.
- `bottleneck`: "compute" or "memory"
- `arithmetic_intensity`: ops/byte
- `compute_time_ms`, `memory_time_ms`
- `pct_of_total`: what fraction of runtime this op consumes

This data is injected into OpenEvolve prompts so the LLM can make informed mutations.

### 3. Numeric Optimizer (`numeric/config_optimizer.py`)

Multi-objective serving config search using pymoo NSGA-II/III.

**Search space** (extends existing ConfigOptimizer):
- `tensor_parallel`: {1, 2, 4, 8, 16}
- `pipeline_parallel`: {1, 2, 4, 8}
- `batch_size`: [1, 1024] continuous
- `precision`: {bf16, fp8, int8}
- `max_num_batched_tokens`: [100, 100000]
- `enable_chunked_prefill`: {True, False}
- `chunk_size`: [64, 2048]
- `enable_prefix_caching`: {True, False}
- `gpu_memory_utilization`: [0.5, 0.99]

**Objectives**: throughput, latency (TTFT/TPOT), cost ($/1M tokens), power (W), SLO compliance

**Constraints**: max memory, max TTFT, max TPOT, max devices, max power

**Output**: `ParetoResult` with non-dominated front, best per-objective, sensitivity ranking

### 4. Hardware Explorer (`numeric/hardware_explorer.py`)

Multi-objective hardware design space exploration using pymoo.

**Search space**:
- `flops_tflops`: [50, 10000] — peak BF16 TFLOPS
- `offchip_mem_bw_gbps`: [200, 16000] — HBM bandwidth
- `off_chip_mem_size_gb`: [16, 384] — HBM capacity
- `on_chip_mem_size_mb`: [10, 512] — SRAM cache
- `onchip_mem_bw_gbps`: [5000, 100000] — SRAM bandwidth
- `interchip_link_bw_gbps`: [25, 1800] — NVLink/equivalent
- `frequency_ghz`: [0.5, 3.0]
- `num_nodes`: {1, 2, 4, 8, 16}

**Constraints** (physics/economics):
- `max_tdp_watts`: power envelope
- `max_cost_usd`: estimated unit cost (derived from parameter values)
- `min_memory_for_model`: must fit model weights
- `compute_to_bw_ratio`: realistic architectural balance

**Modes**:
- `explore()`: Full Pareto search over hardware space
- `what_if()`: Sweep a single parameter (e.g., "what if A100 had 2x bandwidth?")
- `design_for_model()`: Find cheapest hardware meeting a throughput target
- `compare_vs_real()`: How does a hypothetical design compare to existing GPUs?

### 5. Algorithm Evolver (`evolve/algorithm_evolver.py`)

Uses OpenEvolve to evolve Python code, evaluated by BudSim.

**What it evolves**:
- Batch scheduling algorithms (how to form batches from queued requests)
- KV cache eviction policies (which cache entries to evict under memory pressure)
- Memory tier assignment heuristics (when to spill KV cache to DDR/CXL/NVMe)
- Prefill-decode disaggregation policies (how to split work across instances)

**How it works**:
1. Seed algorithm loaded from `base_algorithms/`
2. BudSim evaluates seed → baseline metrics + roofline analysis
3. OpenEvolve prompt includes: current code, performance metrics, roofline bottleneck data
4. LLM generates mutated algorithm code
5. BudSim evaluates mutant via simulation (injects evolved code into simulator)
6. MAP-Elites database tracks best candidates per feature bin
7. Repeat for N iterations

**Roofline-enriched prompts** (key differentiator):

```
You are evolving a batch scheduling algorithm for LLM serving.

Current algorithm: [code]
Performance: 45.2 req/s, TTFT=320ms, TPOT=28ms

Roofline Analysis:
- Overall bottleneck: MEMORY BANDWIDTH (91% utilized, compute only 42%)
- Attention decode: memory-bound, 38% of total runtime
- KV cache loads: memory-bound, 22% of total runtime
- FFN layers: compute-bound, 31% of total runtime
- Interconnect (AllReduce): 9% of total runtime

The system is heavily memory-bandwidth-bound. The scheduler should
minimize redundant memory transfers and maximize compute overlap.

Generate an improved scheduling algorithm that addresses these bottlenecks.
```

**OpenEvolve configuration**:
- Islands: 3-5 for diversity
- LLM: Together AI (default: `openai/gpt-oss-120b` via `https://api.together.xyz/v1`)
  - Also supports any OpenAI-compatible API (OpenAI, Anthropic proxy, local Ollama/vLLM)
  - API key via `TOGETHER_API_KEY` env var (never committed to repo)
- Cascade evaluation: Stage 1 (syntax check), Stage 2 (quick sim), Stage 3 (full sim)
- Migration: ring topology, every 20 generations

### 6. CLI (`cli.py`)

```bash
# Numeric config optimization
budsim-evolve optimize \
    --model meta-llama/Meta-Llama-3.1-70B \
    --hardware H100_GPU \
    --objectives throughput,latency \
    --constraints "ttft<500,tpot<50,memory<80" \
    --generations 200 \
    --output results/config_pareto.json

# Hardware design exploration
budsim-evolve explore-hardware \
    --model meta-llama/Meta-Llama-3.1-70B \
    --objectives throughput,cost \
    --constraints "power<700,throughput>100" \
    --generations 300 \
    --output results/hardware_pareto.json

# What-if hardware analysis
budsim-evolve what-if \
    --base-hardware A100_GPU \
    --param offchip_mem_bw \
    --range 1600,8000 \
    --steps 20 \
    --model meta-llama/Meta-Llama-3.1-70B

# Algorithm evolution
budsim-evolve evolve-scheduler \
    --model meta-llama/Meta-Llama-3.1-70B \
    --hardware H100_GPU \
    --workload bursty \
    --iterations 100 \
    --llm-endpoint https://api.together.xyz/v1 \
    --llm-model openai/gpt-oss-120b \
    --output results/evolved_scheduler/

# Sensitivity analysis
budsim-evolve sensitivity \
    --model meta-llama/Meta-Llama-3.1-70B \
    --hardware H100_GPU \
    --target throughput \
    --method morris
```

## Package Structure

```
llm-memory-calculator/src/llm_memory_calculator/
├── genz/                          # Existing (unchanged)
├── budevolve/                     # NEW
│   ├── __init__.py                # Public API
│   ├── evaluator.py               # BudSimEvaluator
│   ├── roofline_analyzer.py       # Per-op bottleneck extraction
│   ├── numeric/
│   │   ├── __init__.py
│   │   ├── config_optimizer.py    # pymoo NSGA-II config search
│   │   ├── hardware_explorer.py   # pymoo NSGA-II hardware DSE
│   │   ├── sensitivity.py         # Morris / Sobol analysis
│   │   └── search_spaces.py       # Search space definitions
│   ├── evolve/
│   │   ├── __init__.py
│   │   ├── algorithm_evolver.py   # OpenEvolve + BudSim orchestrator
│   │   ├── evaluator_bridge.py    # OpenEvolve evaluator wrapper
│   │   ├── prompt_templates.py    # Roofline-enriched prompts
│   │   └── base_algorithms/
│   │       ├── scheduler_v0.py
│   │       ├── cache_policy_v0.py
│   │       └── memory_mgr_v0.py
│   ├── results/
│   │   ├── __init__.py
│   │   ├── pareto.py              # Pareto front types
│   │   ├── evolution_log.py       # Evolution history
│   │   └── report.py              # Reports
│   └── cli.py
```

## Dependencies

```toml
[project.optional-dependencies]
evolve = [
    "pymoo>=0.6.0",
    "openevolve>=0.1.0",
]
```

Core BudSim functionality remains dependency-free. Evolution features are opt-in via `pip install llm-memory-calculator[evolve]`.

## Key Data Types

```python
@dataclass
class ServingConfig:
    model: str
    hardware: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    batch_size: int = 32
    precision: str = "bf16"
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = False
    chunk_size: int = 512
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.90

@dataclass
class HardwareSpec:
    flops_tflops: float
    offchip_mem_bw_gbps: float
    off_chip_mem_size_gb: float
    on_chip_mem_size_mb: float
    onchip_mem_bw_gbps: float
    interchip_link_bw_gbps: float
    frequency_ghz: float
    num_nodes: int = 1
    tdp_watts: float = 700.0
    estimated_cost_usd: float = 25000.0

@dataclass
class EvalResult:
    throughput_rps: float
    token_throughput_tps: float
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    memory_gb: float
    power_w: float
    slo_compliance: float
    cost_per_million_tokens: float
    roofline: Optional[RooflineReport] = None
    feasible: bool = True

@dataclass
class ParetoResult:
    pareto_front: List[Dict]       # Non-dominated solutions
    all_evaluated: List[Dict]      # All evaluated candidates
    best_per_objective: Dict       # Best solution per objective
    sensitivity: Dict[str, float]  # Parameter importance ranking
    num_generations: int
    total_evaluations: int
```

## Implementation Phases

### Phase 1: Foundation (evaluator + roofline analyzer)
- BudSimEvaluator wrapping GenZ engine
- RooflineAnalyzer extracting per-operator bottlenecks
- EvalResult and data types
- Unit tests

### Phase 2: Numeric Optimizer
- pymoo integration
- Config search with NSGA-II
- Sensitivity analysis (Morris method)
- CLI: `optimize` and `sensitivity` commands

### Phase 3: Hardware Explorer
- Hardware search space definition
- Hardware constraint model
- What-if analysis
- Compare-vs-real hardware
- CLI: `explore-hardware` and `what-if` commands

### Phase 4: Algorithm Evolver
- OpenEvolve integration
- Evaluator bridge (BudSim as OpenEvolve fitness function)
- Roofline-enriched prompt templates
- Base algorithms (seed schedulers, cache policies)
- CLI: `evolve-scheduler`, `evolve-cache-policy` commands

### Phase 5: Polish
- End-to-end integration tests
- JSON/CSV output formats
- Result visualization helpers
- Documentation

## Research Sources

### Evolutionary/LLM Optimization
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — Open-source AlphaEvolve
- [AlphaEvolve (DeepMind)](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [FunSearch (Nature 2023)](https://www.nature.com/articles/s41586-023-06924-6)
- [Darwinian Evolver (Imbue)](https://github.com/imbue-ai/darwinian_evolver)
- [pymoo](https://pymoo.org/) — Multi-objective optimization
- [ReEvo (NeurIPS 2024)](https://github.com/ai4co/reevo)

### Hardware Design Space Exploration
- [NAAS (MIT)](https://hanlab.mit.edu/projects/naas) — Neural Accelerator Architecture Search
- [DOSA (Berkeley)](https://github.com/ucb-bar/dosa) — Differentiable accelerator DSE
- [ConfuciuX (Georgia Tech)](https://arxiv.org/abs/2009.02010) — RL for hardware resource assignment
- [Timeloop/Accelergy (MIT)](https://timeloop.csail.mit.edu/) — DNN accelerator modeling
- [AlphaChip (DeepMind)](https://deepmind.google/blog/how-alphachip-transformed-computer-chip-design/)
- [GPT4AIGChip](https://arxiv.org/abs/2309.10730) — LLM-driven accelerator generation
- [Polaris/Starlight](https://arxiv.org/abs/2412.15548) — Multi-fidelity DSE
