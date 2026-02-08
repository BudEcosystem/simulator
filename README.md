# LLM Performance Simulator

A comprehensive ecosystem for simulating and analyzing Large Language Model (LLM) performance across diverse hardware platforms. This repository provides accurate memory estimation, **inference simulation**, performance modeling, and **training simulation** for LLM deployment and fine-tuning.

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
│   │       └── training.py    # Training simulation APIs
│   └── Website/               # Streamlit dashboard
│       └── pages/             # Comparison tools
│
└── llm-memory-calculator/     # Core LLM performance modeling engine
    └── src/                   # Python package with GenZ framework
        ├── genz/              # Roofline-based performance modeling
        │   ├── LLM_inference/ # Prefill/decode simulation
        │   └── LLM_training/  # Training simulation
        └── training/          # Training memory & cluster optimization
```

## Features Overview

| Feature | Inference | Training |
|---------|-----------|----------|
| Memory Estimation | ✅ Weights + KV Cache + Activations | ✅ + Gradients + Optimizer States |
| Performance Modeling | ✅ Prefill, Decode, Speculative | ✅ Forward, Backward, Communication |
| Parallelism | ✅ TP, PP, EP | ✅ TP, PP, DP, EP, ZeRO 0-3 |
| Hardware Support | ✅ 57 profiles (GPU/TPU/ASIC/CPU) | ✅ Same |
| Cost Estimation | ✅ Per-request | ✅ Per-training-run |
| SLO Validation | ✅ TTFT, E2E, Throughput | ✅ Time to completion |

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

## Training Simulation

### Supported Training Stages
| Stage | Description | Models Required |
|-------|-------------|-----------------|
| **SFT** | Supervised Fine-Tuning | 1 (policy) |
| **DPO** | Direct Preference Optimization | 2 (policy + reference) |
| **PPO** | Proximal Policy Optimization | 4 (actor + critic + reference + reward) |
| **GRPO** | Group Relative Policy Optimization | 1 (with group sampling) |
| **KTO** | Kahneman-Tversky Optimization | 2 (policy + reference) |
| **ORPO** | Odds Ratio Preference Optimization | 1 (combined loss) |
| **SimPO** | Simple Preference Optimization | 1 (reference-free) |
| **IPO** | Identity Preference Optimization | 1 (reference-free) |
| **RM** | Reward Modeling | 1 (reward model) |

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
    training_stage='sft',      # sft, dpo, ppo, grpo, kto, orpo, simpo, rm
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

## Hardware Support (57 Profiles)

### GPUs
- **NVIDIA**: A100 (40GB, 80GB), H100, H200, GH200, B100, GB200, V100, RTX 4090/4080/3090, L40S, A10G
- **AMD**: MI300X, MI325X, MI210, MI100

### TPUs & ASICs
- **Google TPUs**: TPUv4, TPUv5e, TPUv5p, TPUv6
- **Intel**: Gaudi3, MAX 1550, MAX 1100
- **AWS**: Trainium, Inferentia
- **Specialty**: Cerebras WSE-2/3, Groq LPU, SambaNova SN40L

### CPUs
- **Intel Xeon**: Sapphire Rapids, Emerald Rapids, Granite Rapids
- **AMD EPYC**: Milan, Genoa, Bergamo
- **ARM**: NVIDIA Grace, AWS Graviton3/4

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
| `/api/simulator/hardware` | GET | List all 57 hardware profiles |
| `/api/simulator/estimate-training` | POST | Estimate training memory |
| `/api/simulator/recommend-cluster` | POST | Cluster recommendations (cost/speed) |
| `/api/simulator/check-fit` | POST | Check if training fits on hardware |
| `/api/simulator/estimate-time` | POST | Estimate training time and cost |

### API Examples

#### List Hardware with Filters
```bash
curl "http://localhost:8000/api/hardware?type=gpu&min_memory=40"
```

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

#### Get Usecase Recommendations
```bash
curl -X POST "http://localhost:8000/api/usecases/chatbot-1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_sizes": [1, 4, 8],
    "model_categories": ["8B", "70B"]
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

#### Get Cluster Recommendations
```bash
curl -X POST http://localhost:8000/api/simulator/recommend-cluster \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "method": "full",
    "prefer_cost": true,
    "max_gpus": 32
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
   - Supported models and hardware
   - Quantization options (FP32, BF16, INT8, INT4, INT2)
   - Parallelism strategies visualization

2. **Usecase Comparison**: Compare performance across use cases
   - Pre-configured use cases: Q&A, Summarization, Chatbots, Code Gen
   - Scatter plots: TTFT vs Throughput with performance zones
   - Bar charts: Latency, Throughput, Total Response Time

3. **Model Comparison**: Compare models at varying batch sizes
   - Multi-model selection
   - Batch sweep visualization (1-256)
   - Prefill/Decode phase analysis
   - Demand curve generation

4. **Platform Comparison**: Compare hardware accelerators
   - Multi-platform selection with custom specs
   - Hardware datasheet links
   - Performance quadrant analysis
   - Memory requirement checks

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
| `predict_minimum_requirements()` | Predict minimum cluster requirements for a training config |
| `build_comprehensive_config()` | Generate full training config (LlamaFactory + DeepSpeed + launch commands) |

### GenZ Training Module (`llm_memory_calculator.genz.LLM_training`)

| Function | Description |
|----------|-------------|
| `training_modeling()` | Full training step simulation |
| `training_modeling_for_stage()` | Stage-aware training simulation |
| `get_best_training_parallelization()` | Find optimal parallelism strategy |
| `estimate_dpo_training()` | DPO-specific estimation |
| `estimate_ppo_training()` | PPO-specific estimation |
| `validate_against_benchmark()` | Validate against published benchmarks |

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

### Training Time Estimates (1B tokens, 8x H100)

| Method | Time | Cost | Throughput | MFU |
|--------|------|------|------------|-----|
| Full | 6.8h | $259 | 40,824 tok/s | 24.9% |
| LoRA | 6.8h | $259 | 40,824 tok/s | 24.9% |

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
# Test inference
cd llm-memory-calculator
pytest tests/ -v -k "inference"

# Test training module
pytest tests/training/ -v

# Test full API
cd BudSimulator
python comprehensive_api_test.py

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
- Hardware vendor specifications (NVIDIA, AMD, Google)

Typical accuracy:
- Memory estimation: ±10%
- Throughput estimation: ±15%
- Training time: ±20%

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Interfaces                                  │
├─────────────────┬─────────────────────────┬─────────────────────────────┤
│  React Web UI   │   Streamlit Dashboard   │      REST API Clients       │
│  (Port 3000)    │     (Port 8501)         │     (curl, Python, etc)     │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Port 8000)                        │
├─────────────────┬─────────────────┬─────────────────┬───────────────────┤
│  /api/models    │  /api/hardware  │  /api/usecases  │  /api/simulator   │
│  - validate     │  - list         │  - CRUD         │  - estimate       │
│  - config       │  - filter       │  - recommend    │  - recommend      │
│  - calculate    │  - recommend    │  - optimize     │  - check-fit      │
│  - compare      │                 │                 │  - estimate-time  │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬──────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     llm-memory-calculator Package                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      GenZ Engine                                 │   │
│  ├────────────────────────┬────────────────────────────────────────┤   │
│  │    LLM_inference/      │           LLM_training/                │   │
│  │    - prefill_moddeling │           - training_modeling          │   │
│  │    - decode_moddeling  │           - get_best_parallelization   │   │
│  │    - spec_decode       │           - training_stages            │   │
│  │    - best_parallelism  │           - validation                 │   │
│  └────────────────────────┴────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Training Module                               │   │
│  │    - TrainingMemoryCalculator    - auto_configure_training      │   │
│  │    - TrainingClusterSelector     - build_llamafactory_config    │   │
│  │    - estimate_training_time      - build_deepspeed_config       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Hardware Configs (57 Profiles)                      │   │
│  │         GPUs | TPUs | ASICs | Accelerators | CPUs               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recent Changes

- **ASTRA-SIM subprocess security hardening**: Replaced `shell=True` subprocess calls with `shell=False` and explicit argument lists to prevent shell injection; output redirection now uses file handles instead of shell redirects. Added path validation utilities (`path_utils.py`) and comprehensive security tests for shell injection prevention.
- **CORS configuration hardening**: Tightened CORS policy on the FastAPI backend to restrict allowed origins
- **Cluster ranking & requirements prediction**: New modules for ranking GPU clusters by throughput/cost/ETA and predicting minimum hardware requirements for training workloads
- **Comprehensive config builder**: Generate full LlamaFactory YAML, DeepSpeed JSON, Accelerate configs, and launch commands with best-practice defaults per optimization focus (stable, convergence, speed, tco)
- **Training best practices module**: Codified best practices from LlamaFactory analysis (200+ parameters) covering learning rates, optimizers, LoRA configs, PPO multi-model setups, and hardware-specific tuning
- **Versioned database migrations**: Replaced one-off migration script with a decorator-based migration framework tracked in a `migration_history` table
- **Rate limiting**: Optional `slowapi`-based rate limiting on the FastAPI backend (60 req/min default)
- **Deferred MODEL_DICT patching**: Moved model dictionary patching from import-time to application startup for reliability
- **Improved hardware detection**: Expanded keyword-based hardware type classification covering NVIDIA, AMD, Intel, ARM, and specialty accelerators
- **BudSimulator core rework**: Fully implemented `BudSimulator.run()` with GenZ `SimulationEngine` integration, proper Pydantic v2 models, and feature mapping per simulation type
- **SQL injection hardening**: Table name whitelist validation in `DatabaseConnection` for insert/update/delete operations
- **Logging cleanup**: Replaced `print()` statements across routers and training modeling with `logging` module calls
- **Expanded NVIDIA model set**: Added Nemotron, Cosmos, and additional NVIDIA model configurations
- **Enhanced attention & FFN modeling**: Improved sliding window attention, cross-attention, and GLU/bilinear FFN parameter calculations
- **Training optimizer enhancements**: Added Adan, Prodigy, ADOPT, Schedule-Free optimizers with memory/convergence profiles

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
