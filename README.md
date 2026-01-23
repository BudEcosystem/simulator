# LLM Performance Simulator

A comprehensive ecosystem for simulating and analyzing Large Language Model (LLM) performance across diverse hardware platforms. This repository provides accurate memory estimation, performance modeling, and **training simulation** for LLM deployment and fine-tuning.

## Repository Structure

```
simulator/
├── BudSimulator/              # Full-stack web application for LLM analysis
│   ├── frontend/              # React TypeScript UI
│   ├── apis/                  # FastAPI backend with training APIs
│   └── Website/               # Streamlit dashboard
│
└── llm-memory-calculator/     # Core LLM performance modeling engine
    └── src/                   # Python package with GenZ framework
        ├── genz/              # Roofline-based performance modeling
        │   ├── LLM_inference/ # Prefill/decode simulation
        │   └── LLM_training/  # Training simulation (NEW)
        └── training/          # Training memory & cluster optimization (NEW)
```

## What's New

### Training Simulation (Major Feature)
Complete training performance simulation using GenZ roofline analysis:
- **Forward/backward pass timing** with hardware-calibrated models
- **Communication overhead** (AllReduce, ReduceScatter, AllGather)
- **Memory estimation** with ZeRO stages 0-3
- **Optimal parallelization** strategy discovery (TP, PP, DP, EP)
- **Training time & cost estimation** for any dataset size

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

### Hardware Support (57 Profiles)
- **NVIDIA GPUs**: A100, H100, GH200, B100, GB200, V100, RTX 4090/4080/3090, L40S
- **AMD GPUs**: MI300X, MI325X
- **Google TPUs**: TPUv4, TPUv5e, TPUv5p, TPUv6
- **Intel**: Gaudi3, MAX 1550, MAX 1100, Arc A770
- **Specialty**: Cerebras WSE-2/3, Groq LPU, SambaNova SN40L, AWS Trainium/Inferentia
- **CPUs**: Intel Xeon (Sapphire/Emerald Rapids), AMD EPYC (Milan/Genoa/Bergamo), ARM (Grace, Graviton)

## Quick Start

### Option 1: Full Stack Application
```bash
cd BudSimulator
python setup.py  # Automated setup
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: Python Package Only
```bash
cd llm-memory-calculator
pip install -e .
```

## Training Simulation API

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulator/hardware` | GET | List all 57 hardware profiles |
| `/api/simulator/estimate-training` | POST | Estimate training memory requirements |
| `/api/simulator/recommend-cluster` | POST | Get cluster recommendations (cost/speed optimized) |
| `/api/simulator/check-fit` | POST | Check if training fits on specific hardware |
| `/api/simulator/estimate-time` | POST | Estimate training time and cost |

### API Examples

#### List Available Hardware
```bash
curl http://localhost:8000/api/simulator/hardware
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
    "precision": "bf16",
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

#### Estimate Training Time
```bash
curl -X POST http://localhost:8000/api/simulator/estimate-time \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "hardware": "H100_GPU",
    "num_gpus": 8,
    "dataset_tokens": 1000000000,
    "batch_size": 4,
    "seq_length": 2048
  }'
```

## Python SDK Usage

### Training Memory Estimation
```python
from llm_memory_calculator.training import TrainingMemoryCalculator

calculator = TrainingMemoryCalculator()
estimate = calculator.calculate_training_memory(
    config="meta-llama/Llama-3.1-8B",  # HuggingFace model ID or config dict
    batch_size=4,
    seq_length=2048,
    precision="bf16",
    method="lora",           # full, lora, qlora, freeze, dora, pissa
    optimizer="adamw",       # adamw, adamw_8bit, sgd, galore, apollo, adafactor
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
    method='lora',             # full, lora, qlora
    batch_size=4,
    seq_length=2048,
    system_name='H100_GPU',
    num_gpus=8,
    tensor_parallel=1,
    data_parallel=8,
    zero_stage=0,
    optimizer='adamw',
    lora_rank=16,
)

print(f"Step Time: {result.step_time_ms:.1f} ms")
print(f"Throughput: {result.tokens_per_second:.0f} tokens/s")
print(f"Memory/GPU: {result.memory_per_gpu_gb:.1f} GB")
print(f"MFU: {result.model_flops_utilization:.1%}")
print(f"Forward: {result.forward_pct:.1%}, Backward: {result.backward_pct:.1%}")
```

### Find Optimal Parallelization
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

### Cluster Recommendations
```python
from llm_memory_calculator.training import TrainingClusterSelector

selector = TrainingClusterSelector()
recommendations = selector.recommend_clusters(
    training_estimate=estimate.to_dict(),
    prefer_cost=True,          # Sort by cost (True) or speed (False)
    max_budget_per_hour=50.0,  # Optional budget constraint
    max_gpus=32,
)

for rec in recommendations[:5]:
    print(f"{rec.hardware_name}: {rec.total_gpus} GPUs, ${rec.estimated_cost_per_hour:.2f}/hr")
```

### Training Time Estimation
```python
from llm_memory_calculator.training import estimate_training_time

estimate = estimate_training_time(
    model='meta-llama/Llama-3.1-8B',
    dataset_tokens=1_000_000_000,
    num_epochs=1.0,
    batch_size=4,
    seq_length=2048,
    system_name='H100_GPU',
    num_gpus=8,
)

print(f"Total Steps: {estimate.total_steps:,}")
print(f"Training Time: {estimate.total_hours:.1f} hours")
print(f"Estimated Cost: ${estimate.total_cost:.2f}")
```

### Auto-Configuration
```python
from llm_memory_calculator.training import auto_configure_training

plan = auto_configure_training(
    model='meta-llama/Llama-3.1-8B',
    dataset_tokens=1_000_000_000,
    training_stage='sft',
    optimization_goal='minimize_cost',  # or 'minimize_time', 'balanced'
)

print(plan.summary())
config = plan.to_llamafactory_config()  # Export to LlamaFactory format
```

### LlamaFactory Config Generation
```python
from llm_memory_calculator.training import (
    build_llamafactory_config,
    build_deepspeed_config,
    generate_launch_command,
)

# Generate LlamaFactory YAML config
llamafactory_config = build_llamafactory_config(
    model_name='meta-llama/Llama-3.1-8B',
    training_stage='sft',
    method='lora',
    dataset_name='your_dataset',
    output_dir='./output',
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    lora_rank=16,
)

# Generate DeepSpeed config
deepspeed_config = build_deepspeed_config(
    zero_stage=2,
    batch_size=4,
    gradient_accumulation=4,
)

# Generate launch command
cmd = generate_launch_command(
    num_gpus=8,
    config_file='train_config.yaml',
)
print(cmd)  # llamafactory-cli train train_config.yaml
```

## Key Functions Reference

### Training Module (`llm_memory_calculator.training`)

| Function | Description |
|----------|-------------|
| `TrainingMemoryCalculator` | Calculate training memory requirements |
| `TrainingClusterSelector` | Recommend optimal cluster configurations |
| `estimate_training_time()` | Estimate training time and cost |
| `auto_configure_training()` | Auto-configure optimal training setup |
| `build_llamafactory_config()` | Generate LlamaFactory YAML config |
| `build_deepspeed_config()` | Generate DeepSpeed JSON config |
| `get_gpu_spec()` | Get detailed GPU specifications |
| `calculate_tco()` | Calculate total cost of ownership |

### GenZ Training Module (`llm_memory_calculator.genz.LLM_training`)

| Function | Description |
|----------|-------------|
| `training_modeling()` | Full training step simulation |
| `training_modeling_for_stage()` | Stage-aware training simulation |
| `get_best_training_parallelization()` | Find optimal parallelism strategy |
| `estimate_dpo_training()` | DPO-specific estimation |
| `estimate_ppo_training()` | PPO-specific estimation |
| `estimate_grpo_training()` | GRPO-specific estimation |
| `validate_against_benchmark()` | Validate against published benchmarks |

### Inference Module (`llm_memory_calculator.genz`)

| Function | Description |
|----------|-------------|
| `prefill_moddeling()` | Prefill phase simulation |
| `decode_moddeling()` | Decode phase simulation |
| `calculate_memory()` | Memory requirements calculation |
| `get_best_parallelization_strategy()` | Optimal parallelism for inference |

## Sample Results

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
| QLoRA | 6.8h | $259 | 40,824 tok/s | 24.9% |

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

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐
│   React Web UI  │────▶│         FastAPI Backend              │
└─────────────────┘     │  /api/simulator/* (Training APIs)    │
                        │  /api/models/*    (Model APIs)       │
                        │  /api/hardware/*  (Hardware APIs)    │
                        └──────────────────────────────────────┘
                                         │
                                         ▼
                        ┌──────────────────────────────────────┐
                        │      llm-memory-calculator           │
                        │                                      │
                        │  ┌────────────┐  ┌────────────────┐ │
                        │  │   GenZ     │  │   Training     │ │
                        │  │  Engine    │  │   Module       │ │
                        │  │            │  │                │ │
                        │  │ - Roofline │  │ - Memory Calc  │ │
                        │  │ - Prefill  │  │ - Cluster Sel  │ │
                        │  │ - Decode   │  │ - Time Est     │ │
                        │  │ - Training │  │ - Config Gen   │ │
                        │  └────────────┘  └────────────────┘ │
                        │                                      │
                        │  ┌──────────────────────────────────┐│
                        │  │     Hardware Configs (57)        ││
                        │  │  GPUs, TPUs, ASICs, CPUs         ││
                        │  └──────────────────────────────────┘│
                        └──────────────────────────────────────┘
```

## Running Tests

```bash
# Test training module
cd llm-memory-calculator
pytest tests/training/ -v

# Test full API
cd BudSimulator
python comprehensive_api_test.py

# Quick validation
python -c "
from llm_memory_calculator.genz.LLM_training import training_modeling
result = training_modeling(
    model='meta-llama/Llama-3.1-8B',
    training_stage='sft',
    batch_size=4,
    seq_length=2048,
    system_name='H100_GPU',
    num_gpus=8,
)
print(f'Step: {result.step_time_ms:.1f}ms, MFU: {result.model_flops_utilization:.1%}')
"
```

## Docker Deployment

```bash
# Build and run
docker build -t budsimulator .
docker run -p 8000:8000 -p 3000:3000 budsimulator

# Or use docker-compose
docker-compose up --build
```

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
