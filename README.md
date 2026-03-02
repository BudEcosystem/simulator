# LLM Performance Simulator

A comprehensive ecosystem for simulating and analyzing Large Language Model (LLM) performance across diverse hardware platforms. This repository provides accurate memory estimation, **inference simulation**, **serving simulation**, performance modeling, **training simulation**, and **reverse-optimization** for LLM deployment and fine-tuning.

## Features

### Inference Simulation
- **Prefill phase modeling** — first token latency (TTFT) with operator-level roofline analysis
- **Decode phase modeling** — per-token latency (TPOT) with KV cache growth tracking
- **Speculative decoding** — draft-verify pipeline with acceptance rate estimation
- **Memory estimation** — weights + KV cache + activations with precision-aware sizing
- **Optimal parallelization search** — exhaustive TP/PP sweep with Pareto frontier analysis
- **Minimum platform sizing** — calculate minimum nodes required for a given model

### Serving Simulation
- **Event-driven simulator** — discrete-event simulation with arrival-to-completion tracking
- **Multi-tier memory model** — HBM, DRAM, DDR, CXL, NVMe with spill/fill latency modeling
- **Batch scheduling** — continuous batching with preemption, priority queuing, and padding-aware packing
- **SLO tracking** — TTFT, TPOT, ITL, E2E latency with P50/P95/P99 percentile tracking
- **Physics-based power model** — 7-component breakdown (compute, memory, interconnect, cooling, PSU, idle, leakage)
- **Prefix caching** — LRU/LFU/ARC policies with shared prefix detection and hit-rate analysis
- **Prefill-decode disaggregation** — separate prefill/decode pools with M/M/1 queuing analysis
- **Cluster topology optimization** — sub-linear scaling with communication overhead modeling
- **Workload generation** — Poisson/bursty/trace-driven arrival patterns with configurable distributions
- **Configuration optimizer** — single-objective and multi-objective (Pareto) config search
- **Parameter sensitivity analysis** — Morris method for identifying dominant configuration parameters

### Training Simulation
- **14 training stages** — SFT, DPO, PPO, GRPO, KTO, ORPO, SimPO, IPO, RM, RLOO, REINFORCE, CPO, PT, PPO_DETAILED
- **6 fine-tuning methods** — Full, LoRA, QLoRA, DoRA, PiSSA, Freeze with accurate parameter counting
- **30+ optimizers** — AdamW, Lion, Adafactor, GaLore, LOMO, Muon, Schedule-Free, 8-bit variants, and more
- **Peak-of-phases memory model** — forward/backward/optimizer phases tracked separately; total = max of phase peaks
- **Distributed training** — TP, PP, DP, EP, ZeRO-0/1/2/3 with pipeline 1F1B bubble modeling
- **Sequence parallelism** — activation memory split across TP group with correct SP formula
- **Gradient checkpointing** — selective/full recompute with activation memory reduction
- **Multi-model stages** — DPO (policy + reference), PPO (actor + critic + reference + reward), KTO (policy + reference)
- **Training time estimation** — end-to-end time, cost, MFU calculation with dataset token counts
- **Cluster ranking** — rank GPU clusters by throughput, ETA, cost, or composite score
- **Config generation** — LlamaFactory YAML, DeepSpeed JSON, Accelerate configs, and launch commands

### BudEvolve (Reverse-Optimization)
- **NSGA-II config optimization** — multi-objective search over TP, PP, BS, precision with Pareto frontier
- **Hardware design space exploration** — sweep FLOPS, bandwidth, memory, interconnect to find optimal hardware
- **What-if parameter sweeps** — single-parameter sensitivity analysis with performance curves
- **LLM-driven algorithm evolution** — evolve batch scheduling and cache eviction algorithms using LLM mutations
- **CPU-optimized scheduling** — TTFT-aware batch sizing, L3 cache budget, NUMA-aware ordering, priority scheduling
- **Roofline analysis** — compute vs memory bottleneck characterization per operator
- **Sensitivity analysis** — rank config and hardware parameters by throughput/latency impact

### CPU Inference Modeling
- **ISA-aware performance** — AMX, AVX-512, AVX2, SVE, NEON instruction set modeling with throughput multipliers
- **Cache hierarchy** — L1/L2/L3 modeling with bandwidth estimation and KV cache placement
- **NUMA topology** — multi-socket, multi-CCD topology modeling with cross-node traffic estimation
- **Threading model** — physical cores, SMT, OpenMP/TBB scheduling with NUMA-pinned execution
- **Frequency scaling** — boost/base frequency modeling with thermal and power constraints
- **CPU operator framework** — GEMM, attention, and reduction operators with CPU-specific roofline

### Hardware Support (72 Profiles)

| Category | Platforms |
|----------|-----------|
| **NVIDIA GPUs** | A100 (40GB, 80GB), H100, H100_PCIe, H200, GH200, B100, B200, GB200, V100, RTX 4090/4080/3090, L40S, A10G |
| **AMD GPUs** | MI300X, MI325X, MI210, MI100 |
| **Google TPUs** | TPUv4, TPUv5e, TPUv5p, TPUv6 |
| **Intel Accelerators** | Gaudi3, MAX 1550, MAX 1100 |
| **AWS Silicon** | Trainium, Inferentia |
| **Specialty ASICs** | Cerebras WSE-2/3, Groq LPU, SambaNova SN40L |
| **Intel Xeon CPUs** | Sapphire Rapids, Emerald Rapids, Granite Rapids |
| **AMD EPYC CPUs** | Milan, Genoa, Bergamo, Turin |
| **ARM CPUs** | NVIDIA Grace, AWS Graviton3/4 |

### Web Interfaces
- **React Frontend** (Port 3000) — hardware browser, usecase management, AI optimization, model details
- **Streamlit Dashboard** (Port 8501) — usecase comparison, model comparison, platform comparison
- **FastAPI Backend** (Port 8000) — 40+ REST API endpoints with OpenAPI documentation

### Testing & Validation
- **879+ tests** across 59 test files — inference, training, serving, CPU, BudEvolve, API
- **Validated against benchmarks** — MLPerf Training, DeepSpeed ZeRO, Megatron-LM, vendor specs
- **Accuracy** — memory ±10%, throughput ±15%, training time ±20%

---

## Repository Structure

```
simulator/
├── BudSimulator/              # Full-stack web application for LLM analysis
│   ├── frontend/              # React TypeScript UI
│   │   └── src/
│   │       ├── components/    # Reusable UI components
│   │       ├── services/      # API service layer
│   │       └── types/         # TypeScript interfaces
│   ├── apis/                  # FastAPI backend
│   │   └── routers/           # API route handlers
│   │       ├── models.py      # Model validation, memory calculation
│   │       ├── hardware.py    # Hardware management, recommendations
│   │       ├── usecases.py    # Usecase management, SLO validation
│   │       ├── training.py    # Training simulation APIs
│   │       └── serving.py     # Serving simulation v2 APIs
│   └── Website/               # Streamlit dashboard
│       └── pages/             # Comparison tools
│
├── llm-memory-calculator/     # Core LLM performance modeling engine
│   └── src/llm_memory_calculator/
│       ├── genz/              # Roofline-based performance modeling
│       │   ├── LLM_inference/ # Prefill/decode simulation
│       │   ├── LLM_training/  # Training simulation (14 stages)
│       │   ├── Models/        # 100+ model architectures
│       │   ├── serving/       # Event-driven serving simulation
│       │   └── cpu/           # CPU-specific modeling
│       ├── training/          # Training memory & cluster optimization
│       ├── hardware/          # 72 hardware profiles
│       └── budevolve/         # Reverse-optimization framework
│           ├── evolve/        # LLM-driven algorithm evolution
│           └── numeric/       # NSGA-II config & hardware optimization
│
└── research/                  # Analysis reports and findings
    ├── findings/              # 18 detailed source code analyses
    └── papers/                # Reference papers
```

---

## Inference Simulation

### Core Inference Functions

```python
from llm_memory_calculator.genz.LLM_inference import (
    prefill_moddeling,      # First token latency simulation
    decode_moddeling,       # Token generation simulation
    spec_prefill_modeling,  # Speculative decoding
)
from llm_memory_calculator.genz.LLM_inference.best_parallelization import (
    get_best_parallization_strategy,    # Find optimal TP/PP
    get_pareto_optimal_performance,     # Pareto frontier analysis
)
from llm_memory_calculator.genz.LLM_inference.platform_size import (
    get_minimum_system_size,            # Minimum nodes required
)
```

### Prefill Phase Simulation
```python
from llm_memory_calculator.genz.LLM_inference import prefill_moddeling

result = prefill_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=4,
    input_tokens=2048,
    system_name='H100_GPU',
    bits='bf16',
    tensor_parallel=1,
    pipeline_parallel=1,
)

print(f"TTFT: {result['Latency(ms)']:.1f} ms")
print(f"Throughput: {result['Throughput_tokens_per_sec']:.0f} tokens/s")
```

### Decode Phase Simulation
```python
from llm_memory_calculator.genz.LLM_inference import decode_moddeling

result = decode_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=4,
    input_tokens=2048,
    output_tokens=256,
    Bb=4,                    # Beam size
    system_name='H100_GPU',
    bits='bf16',
    tensor_parallel=1,
)

print(f"Decode Latency: {result['Latency(ms)']:.1f} ms")
print(f"Output Throughput: {result['Throughput_tokens_per_sec']:.0f} tokens/s")
```

### Find Optimal Parallelization for Inference
```python
from llm_memory_calculator.genz.LLM_inference.best_parallelization import (
    get_best_parallization_strategy
)

df = get_best_parallization_strategy(
    stage='decode',
    model='meta-llama/Llama-3.1-70B',
    total_nodes=8,
    batch_size=16,
    beam_size=4,
    input_tokens=2048,
    output_tokens=256,
    system_name='H100_GPU',
    bits='bf16',
)
print(df)  # DataFrame with TP, PP, Latency, Throughput
```

### Memory Calculation
```python
from llm_memory_calculator import calculate_memory

memory = calculate_memory(
    model="meta-llama/Llama-3.1-8B",  # HuggingFace ID or config dict
    batch_size=4,
    sequence_length=2048,
    precision="bf16",
)

print(f"Model Weights: {memory.weights_memory_gb:.2f} GB")
print(f"KV Cache: {memory.kv_cache_gb:.2f} GB")
print(f"Activations: {memory.activations_gb:.2f} GB")
print(f"Total: {memory.total_memory_gb:.2f} GB")
```

---

## Serving Simulation

The serving simulation subsystem provides production-grade serving analysis with event-driven simulation, multi-tier memory, SLO tracking, and power modeling.

### Full Serving Simulation
```python
from llm_memory_calculator.genz.serving import ServingSimulator

sim = ServingSimulator(
    model='meta-llama/Llama-3.1-8B',
    hardware='H100_GPU',
    num_instances=4,
    tensor_parallel=1,
)

results = sim.run(
    num_requests=1000,
    arrival_rate=50.0,          # requests/sec
    input_tokens_mean=512,
    output_tokens_mean=128,
    duration_seconds=60,
)

print(f"Throughput: {results['throughput_rps']:.1f} req/s")
print(f"TTFT P95: {results['ttft_p95_ms']:.1f} ms")
print(f"TPOT P95: {results['tpot_p95_ms']:.1f} ms")
print(f"SLO Attainment: {results['slo_attainment']:.1%}")
print(f"Power: {results['total_power_w']:.0f} W")
```

### Power Model (7-Component Breakdown)
```python
from llm_memory_calculator.genz.serving.power_model import PowerModel

power = PowerModel(hardware='H100_GPU')
breakdown = power.estimate(
    compute_util=0.7,
    memory_util=0.5,
    batch_size=32,
)
# Returns: compute, memory, interconnect, cooling, psu_loss, idle, leakage
```

### Configuration Optimizer
```python
from llm_memory_calculator.genz.serving.config_optimizer import ConfigOptimizer

optimizer = ConfigOptimizer(model='meta-llama/Llama-3.1-8B', hardware='H100_GPU')

# Single-objective optimization
best = optimizer.optimize(objective='throughput', constraints={'max_ttft_ms': 500})

# Multi-objective Pareto frontier
pareto = optimizer.pareto_optimize(
    objectives=['throughput', 'latency'],
    n_generations=50,
)
```

---

## Training Simulation

### Supported Training Stages

| Stage | Description | Models Required |
|-------|-------------|-----------------|
| **SFT** | Supervised Fine-Tuning | 1 (policy) |
| **PT** | Pre-Training | 1 (policy) |
| **DPO** | Direct Preference Optimization | 2 (policy + reference) |
| **PPO** | Proximal Policy Optimization | 4 (actor + critic + reference + reward) |
| **GRPO** | Group Relative Policy Optimization | 1 (with group sampling) |
| **KTO** | Kahneman-Tversky Optimization | 2 (policy + reference) |
| **ORPO** | Odds Ratio Preference Optimization | 1 (combined loss) |
| **SimPO** | Simple Preference Optimization | 1 (reference-free) |
| **IPO** | Identity Preference Optimization | 1 (reference-free) |
| **RM** | Reward Modeling | 1 (reward model) |
| **RLOO** | REINFORCE Leave-One-Out | 1 (variance reduction) |
| **REINFORCE** | Standard REINFORCE | 1 (policy gradient) |
| **CPO** | Contrastive Preference Optimization | 1 (contrastive-based) |
| **PPO_DETAILED** | PPO with detailed phase tracking | 4 (enhanced tracking) |

### Fine-Tuning Methods
| Method | Trainable % | Memory Savings |
|--------|-------------|----------------|
| **Full** | 100% | None |
| **LoRA** | ~0.5% | ~70% |
| **QLoRA** | ~0.5% | ~85% |
| **DoRA** | ~0.5% | ~70% |
| **PiSSA** | ~0.5% | ~70% |
| **Freeze** | Variable | Variable |

### Training Memory Estimation
```python
from llm_memory_calculator.training import TrainingMemoryCalculator

calculator = TrainingMemoryCalculator()
estimate = calculator.calculate_training_memory(
    config="meta-llama/Llama-3.1-8B",
    batch_size=4,
    seq_length=2048,
    precision="bf16",
    method="lora",           # full, lora, qlora, freeze, dora, pissa
    optimizer="adamw",
    gradient_checkpointing=True,
    lora_rank=16,
)

print(f"Weight Memory: {estimate.weight_memory_gb:.2f} GB")
print(f"Gradient Memory: {estimate.gradient_memory_gb:.2f} GB")
print(f"Optimizer Memory: {estimate.optimizer_memory_gb:.2f} GB")
print(f"Activation Memory: {estimate.activation_memory_gb:.2f} GB")
print(f"Total Memory: {estimate.total_memory_gb:.2f} GB")
```

### Training Simulation with GenZ
```python
from llm_memory_calculator.genz.LLM_training import training_modeling

result = training_modeling(
    model='meta-llama/Llama-3.1-8B',
    training_stage='sft',      # sft, dpo, ppo, grpo, kto, orpo, simpo, rm, ...
    method='lora',
    batch_size=4,
    seq_length=2048,
    system_name='H100_GPU',
    num_gpus=8,
    tensor_parallel=1,
    data_parallel=8,
    zero_stage=2,
    optimizer='adamw',
    lora_rank=16,
)

print(f"Step Time: {result.step_time_ms:.1f} ms")
print(f"Throughput: {result.tokens_per_second:.0f} tokens/s")
print(f"Memory/GPU: {result.memory_per_gpu_gb:.1f} GB")
print(f"MFU: {result.model_flops_utilization:.1%}")
```

### Find Optimal Training Parallelization
```python
from llm_memory_calculator.genz.LLM_training import get_best_training_parallelization

config, result = get_best_training_parallelization(
    model='meta-llama/Llama-3.1-70B',
    total_gpus=64,
    batch_size=4,
    seq_length=4096,
    system_name='H100_GPU',
)

print(f"Optimal: TP={config.tensor_parallel}, PP={config.pipeline_parallel}, DP={config.data_parallel}")
print(f"Throughput: {result.tokens_per_second:.0f} tokens/s")
```

---

## Cluster Ranking & Requirements Prediction

### Rank Clusters by Training Performance
```python
from llm_memory_calculator.training import rank_clusters

rankings = rank_clusters(
    model="meta-llama/Llama-3.1-8B",
    method="lora",
    seq_length=2048,
    dataset_tokens=1e9,
    rank_by="composite",  # throughput, eta, cost, composite
)

for r in rankings[:3]:
    print(f"{r.cluster_name}: {r.throughput:.0f} tok/s, ETA={r.eta_hours:.1f}h, ${r.cost:.0f}")
```

### Predict Minimum Cluster Requirements
```python
from llm_memory_calculator.training import predict_minimum_requirements

reqs = predict_minimum_requirements(
    model="meta-llama/Llama-3.1-70B",
    method="full",
    seq_length=4096,
    batch_size=4,
)

print(f"Min GPUs: {reqs.min_gpus}")
print(f"Min GPU Memory: {reqs.min_gpu_memory_gb:.0f} GB")
print(f"Recommended ZeRO: {reqs.recommended_zero_stage}")
```

### Generate Comprehensive Training Configs
```python
from llm_memory_calculator.training import build_comprehensive_config

config = build_comprehensive_config(
    model="meta-llama/Llama-3.1-8B",
    method="lora",
    num_gpus=8,
    optimization_focus="stable",  # stable, convergence, speed, tco
)

# Produces LlamaFactory YAML, DeepSpeed JSON, launch commands, etc.
print(config.llamafactory_yaml)
print(config.deepspeed_json)
print(config.launch_command)
```

---

## BudEvolve (Reverse-Optimization)

BudEvolve combines fast analytical simulation (~1-10ms per evaluation) with multi-objective optimization and LLM-driven algorithm evolution.

### Multi-Objective Config Optimization
```bash
budsim-evolve optimize \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware H100_GPU \
    --objectives throughput,latency \
    --constraints "ttft<500,tpot<50" \
    --generations 50
```

### Hardware Design Space Exploration
```bash
budsim-evolve explore-hardware \
    --model meta-llama/Meta-Llama-3.1-8B \
    --objectives throughput,cost \
    --generations 100
```

### Parameter Sensitivity Analysis
```bash
budsim-evolve sensitivity \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware H100_GPU \
    --mode both  # config, hardware, or both
```

### LLM-Driven Algorithm Evolution
```bash
# Evolve batch scheduling algorithm
budsim-evolve evolve-scheduler \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware H100_GPU \
    --iterations 100

# Evolve KV cache eviction policy
budsim-evolve evolve-cache-policy \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware H100_GPU \
    --iterations 100

# Evolve CPU-optimized scheduler (NUMA, L3, TTFT-aware)
budsim-evolve evolve-cpu-scheduler \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware GraniteRapids_CPU \
    --iterations 100
```

---

## REST API Endpoints

### Inference APIs (`/api/models`, `/api/hardware`, `/api/usecases`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/validate` | POST | Validate model URL/ID from HuggingFace |
| `/api/models/{model_id}/config` | GET | Get model architecture details |
| `/api/models/calculate` | POST | Calculate inference memory requirements |
| `/api/models/compare` | POST | Compare multiple models' memory |
| `/api/models/analyze` | POST | Analyze efficiency across sequence lengths |
| `/api/models/list` | GET | List all available models |
| `/api/models/popular` | GET | Get popular models with logos |
| `/api/models/filter` | GET | Advanced filtering (author, type, params) |
| `/api/hardware` | GET | List hardware with filters |
| `/api/hardware/filter` | GET | Advanced hardware filtering |
| `/api/hardware/recommend` | POST | Get hardware recommendations |
| `/api/usecases` | GET/POST | Usecase CRUD operations |
| `/api/usecases/{id}` | GET/PUT/DELETE | Single usecase operations |
| `/api/usecases/{id}/recommendations` | POST | Model/hardware recommendations for usecase |
| `/api/usecases/{id}/optimize-hardware` | POST | GenZ-based optimization sweep |

### Training APIs (`/api/simulator`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulator/hardware` | GET | List all 72 hardware profiles |
| `/api/simulator/estimate-training` | POST | Estimate training memory |
| `/api/simulator/recommend-cluster` | POST | Cluster recommendations (cost/speed) |
| `/api/simulator/check-fit` | POST | Check if training fits on hardware |
| `/api/simulator/estimate-time` | POST | Estimate training time and cost |

### Serving Simulation APIs (`/api/v2`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/memory/tiers` | POST | Multi-tier memory analysis (HBM/DRAM/CXL/NVMe) |
| `/api/v2/power/estimate` | POST | 7-component power breakdown estimation |
| `/api/v2/simulate/serving` | POST | Full serving simulation with SLO tracking |
| `/api/v2/simulate/batch` | POST | Single batch analysis with GenZ engine |
| `/api/v2/optimize/config` | POST | Find optimal serving configuration |
| `/api/v2/optimize/pareto` | POST | Multi-objective Pareto frontier optimization |
| `/api/v2/analyze/sensitivity` | POST | Parameter sensitivity analysis (Morris method) |
| `/api/v2/cache/analyze` | POST | Prefix cache effectiveness analysis |
| `/api/v2/cluster/disaggregate` | POST | Prefill-decode disaggregation analysis |
| `/api/v2/cluster/topology` | POST | Cluster topology and parallelism optimization |

### API Examples

#### Calculate Inference Memory
```bash
curl -X POST http://localhost:8000/api/models/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B",
    "batch_size": 8,
    "seq_length": 4096,
    "precision": "bf16"
  }'
```

#### Estimate Training Memory
```bash
curl -X POST http://localhost:8000/api/simulator/estimate-training \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "method": "lora",
    "batch_size": 4,
    "seq_length": 2048,
    "optimizer": "adamw",
    "lora_rank": 16
  }'
```

#### Run Serving Simulation
```bash
curl -X POST http://localhost:8000/api/v2/simulate/serving \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "hardware": "H100_GPU",
    "num_instances": 4,
    "arrival_rate": 50.0,
    "duration_seconds": 60,
    "slo_ttft_ms": 500,
    "slo_tpot_ms": 50
  }'
```

#### Power Estimation
```bash
curl -X POST http://localhost:8000/api/v2/power/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "hardware": "H100_GPU",
    "model": "meta-llama/Llama-3.1-8B",
    "batch_size": 32,
    "tensor_parallel": 1
  }'
```

---

## Web Interfaces

### React Frontend (Port 3000)

A modern TypeScript React application for interactive LLM analysis.

#### Features
- **Hardware Browser**: Searchable catalog with advanced filtering
  - Filter by type, manufacturer, memory, FLOPS, bandwidth
  - Sort by performance, cost, efficiency
  - Detailed specs with tooltips
  - Vendor and cloud pricing information
  - Model compatibility matrix

- **Usecase Management**: Configure inference workloads
  - Industry and tag-based filtering
  - Latency profiles: real-time, interactive, responsive, batch
  - SLO configuration (TTFT, E2E, inter-token latency)
  - Token range configuration

- **AI Optimization**: GenZ-powered recommendations
  - Batch size and model size selection
  - Optimization modes: Cost, Speed, Balanced
  - SLO compliance indicators
  - Deployment guidance

- **Model Details**: Architecture analysis
  - Parameters, attention type, model type
  - Memory requirements at various sequence lengths
  - Links to HuggingFace

#### Running the Frontend
```bash
cd BudSimulator/frontend
npm install
npm start  # Opens at http://localhost:3000
```

### Streamlit Dashboard (Port 8501)

An interactive analytical dashboard for performance visualization.

#### Pages

1. **Home**: GenZ framework overview and documentation
2. **Usecase Comparison**: Compare performance across use cases with scatter/bar charts
3. **Model Comparison**: Compare models at varying batch sizes with prefill/decode analysis
4. **Platform Comparison**: Compare hardware accelerators with performance quadrant analysis

#### Running the Streamlit App
```bash
cd BudSimulator/Website
streamlit run Home.py  # Opens at http://localhost:8501
```

---

## Key Functions Reference

### Inference Module (`llm_memory_calculator.genz.LLM_inference`)

| Function | Description |
|----------|-------------|
| `prefill_moddeling()` | Simulate first token latency (TTFT) |
| `decode_moddeling()` | Simulate token generation with KV cache growth |
| `spec_prefill_modeling()` | Speculative decoding simulation |
| `get_best_parallization_strategy()` | Find optimal TP/PP for inference |
| `get_pareto_optimal_performance()` | Pareto frontier analysis |
| `get_minimum_system_size()` | Calculate minimum nodes required |

### Serving Module (`llm_memory_calculator.genz.serving`)

| Function | Description |
|----------|-------------|
| `ServingSimulator` | Event-driven serving simulation |
| `BatchScheduler` | Continuous batching with preemption |
| `MemoryModel` | Multi-tier memory analysis (HBM/DRAM/CXL/NVMe) |
| `PowerModel` | 7-component power breakdown |
| `SLOTracker` | Latency percentile tracking (TTFT/TPOT/ITL) |
| `PrefixCache` | LRU/LFU/ARC cache with shared prefix detection |
| `ConfigOptimizer` | Single and multi-objective config search |
| `ClusterModel` | Cluster topology and scaling analysis |

### Training Module (`llm_memory_calculator.training`)

| Function | Description |
|----------|-------------|
| `TrainingMemoryCalculator` | Calculate training memory requirements |
| `TrainingClusterSelector` | Recommend optimal cluster configurations |
| `estimate_training_time()` | Estimate training time and cost |
| `auto_configure_training()` | Auto-configure optimal training setup |
| `build_llamafactory_config()` | Generate LlamaFactory YAML config |
| `build_deepspeed_config()` | Generate DeepSpeed JSON config |
| `rank_clusters()` | Rank clusters by throughput, ETA, cost, or composite score |
| `predict_minimum_requirements()` | Predict minimum cluster requirements |
| `build_comprehensive_config()` | Generate full training config with launch commands |

### GenZ Training Module (`llm_memory_calculator.genz.LLM_training`)

| Function | Description |
|----------|-------------|
| `training_modeling()` | Full training step simulation |
| `training_modeling_for_stage()` | Stage-aware training simulation |
| `get_best_training_parallelization()` | Find optimal parallelism strategy |
| `estimate_dpo_training()` | DPO-specific estimation |
| `estimate_ppo_training()` | PPO-specific estimation |
| `validate_against_benchmark()` | Validate against published benchmarks |

### BudEvolve Module (`llm_memory_calculator.budevolve`)

| Function | Description |
|----------|-------------|
| `NumericOptimizer` | NSGA-II multi-objective config optimization |
| `HardwareExplorer` | Hardware design space exploration |
| `SensitivityAnalyzer` | Parameter sensitivity analysis |
| `AlgorithmEvolver` | LLM-driven algorithm evolution |
| `BudSimEvaluator` | GenZ-backed evaluation bridge |
| `RooflineAnalyzer` | Compute vs memory bottleneck analysis |

### Memory Module (`llm_memory_calculator`)

| Function | Description |
|----------|-------------|
| `calculate_memory()` | Calculate inference memory requirements |
| `estimate_max_batch_size()` | Max batch for given GPU memory |
| `estimate_max_sequence_length()` | Max sequence for given constraints |
| `analyze_attention_efficiency()` | Analyze attention type efficiency |

---

## Sample Results

### LLaMA 3.1-8B Inference (H100, batch=4, seq=2048)

| Phase | Latency | Throughput | Memory |
|-------|---------|------------|--------|
| Prefill | 45 ms | 181,689 tok/s | 17.2 GB |
| Decode (256 tokens) | 312 ms | 3,282 tok/s | 18.1 GB |

### LLaMA 3.1-8B Training Memory (batch=4, seq=2048)

| Stage | Method | Weight | Gradient | Optimizer | Activation | Total/GPU |
|-------|--------|--------|----------|-----------|------------|-----------|
| SFT | Full | 17.7 GB | 35.3 GB | 70.7 GB | 10.9 GB | 148.0 GB |
| SFT | LoRA | 17.7 GB | 0.1 GB | 0.3 GB | 10.9 GB | 31.9 GB |
| SFT | QLoRA | 4.4 GB | 0.1 GB | 0.3 GB | 10.9 GB | 17.3 GB |
| PPO | Full | 17.7 GB | 35.3 GB | 70.7 GB | 10.9 GB | 323.0 GB* |
| DPO | LoRA | 17.7 GB | 0.1 GB | 0.3 GB | 10.9 GB | 51.3 GB* |

*Includes reference/reward models

### CPU Inference (LLaMA 3.1-8B, BF16, from BudEvolve Analysis)

| CPU | BS=1 Throughput | BS=1 TTFT | BS=32 Throughput | BS=32 TTFT |
|-----|----------------|-----------|-----------------|------------|
| Turin (AMD Zen5) | 42.7/s | 100ms | 1,178/s | 3,212ms |
| GraniteRapids (Intel) | 35.6/s | 114ms | 982/s | 3,660ms |
| Grace (NVIDIA ARM) | 35.6/s | 133ms | 912/s | 4,253ms |
| Graviton4 (AWS ARM) | 32.0/s | 246ms | 520/s | 7,868ms |

---

## Quick Start

### Option 1: Full Stack Application
```bash
cd BudSimulator
python setup.py  # Automated setup

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
# Frontend at http://localhost:3000 (after npm start)
```

### Option 2: Python Package Only
```bash
cd llm-memory-calculator
pip install -e .
```

### Option 3: Streamlit Dashboard
```bash
cd BudSimulator/Website
pip install -r requirements.txt
streamlit run Home.py
```

---

## Running Tests

```bash
# Full test suite (879+ tests)
cd llm-memory-calculator
pytest tests/ -v

# Serving simulation tests
pytest tests/serving/ -v

# Training tests
pytest tests/training/ -v

# BudEvolve tests
pytest tests/budevolve/ -v

# CPU-specific tests
pytest tests/test_cpu_roofline.py tests/serving/test_cpu_benchmark_validation.py -v

# API tests
cd BudSimulator
pytest tests/ -v

# Quick validation
python -c "
from llm_memory_calculator.genz.LLM_inference import prefill_moddeling
result = prefill_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=4,
    input_tokens=2048,
    system_name='H100_GPU',
    bits='bf16',
)
print(f'TTFT: {result[\"Latency(ms)\"]:.1f}ms')
"
```

---

## Accuracy & Validation

The simulator has been validated against published benchmarks:
- **MLPerf Training** results for LLaMA-2 70B
- **DeepSpeed** ZeRO efficiency measurements
- **Megatron-LM** throughput benchmarks
- Hardware vendor specifications (NVIDIA, AMD, Google, Intel, ARM)

Typical accuracy:
- Memory estimation: ±10%
- Throughput estimation: ±15%
- Training time: ±20%

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Interfaces                                │
├─────────────────┬─────────────────────────┬─────────────────────────────┤
│  React Web UI   │   Streamlit Dashboard   │      REST API Clients       │
│  (Port 3000)    │     (Port 8501)         │     (curl, Python, etc)     │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Port 8000)                        │
├──────────┬──────────┬───────────┬───────────┬───────────────────────────┤
│ /api/    │ /api/    │ /api/     │ /api/     │ /api/v2/                  │
│ models   │ hardware │ usecases  │ simulator │ serving simulation        │
│          │          │           │           │ (10 endpoints)            │
└────┬─────┴────┬─────┴────┬──────┴────┬──────┴──────────┬────────────────┘
     │          │          │           │                 │
     ▼          ▼          ▼           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     llm-memory-calculator Package                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      GenZ Engine                                 │   │
│  ├──────────────┬────────────────┬────────────────┬────────────────┤   │
│  │ LLM_inference│  LLM_training  │    serving/     │     cpu/       │   │
│  │ - prefill    │  - 14 stages   │  - simulator    │  - ISA model   │   │
│  │ - decode     │  - parallelism │  - power model  │  - cache model │   │
│  │ - spec_decode│  - validation  │  - SLO tracker  │  - NUMA model  │   │
│  │ - best_par   │               │  - prefix cache │  - threading   │   │
│  └──────────────┴────────────────┴────────────────┴────────────────┘   │
│                                                                         │
│  ┌──────────────────────────┬──────────────────────────────────────┐   │
│  │    Training Module        │         BudEvolve                    │   │
│  │  - MemoryCalculator       │  - NSGA-II optimizer                │   │
│  │  - ClusterSelector        │  - Hardware explorer                │   │
│  │  - ConfigBuilder          │  - Algorithm evolver                │   │
│  │  - 30+ optimizers         │  - Sensitivity analyzer             │   │
│  └──────────────────────────┴──────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Hardware Configs (72 Profiles)                      │   │
│  │      GPUs | TPUs | ASICs | Accelerators | CPUs (16 platforms)   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

```bash
# Clone and setup
git clone https://github.com/BudEcosystem/simulator.git
cd simulator
pip install -e llm-memory-calculator/
cd BudSimulator && pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the GenZ-LLM Analyzer framework
- Validated against MLPerf Training benchmarks
- Hardware specs from official vendor documentation
- Model configs from HuggingFace Hub

## Support

- **Issues**: [GitHub Issues](https://github.com/BudEcosystem/simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BudEcosystem/simulator/discussions)

---

**Built with care by the Bud Ecosystem team**
