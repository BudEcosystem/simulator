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
- **900+ tests** across 60+ test files — inference, training, serving, CPU, BudEvolve, API
- **Validated against benchmarks** — MLPerf Training, DeepSpeed ZeRO, Megatron-LM, vendor specs
- **Accuracy** — memory ±10%, throughput ±15%, training time ±20%
- **Calibrated, no magic numbers** — every efficiency constant (MFU/MBU bands, CPU STREAM efficiency,
  optimizer-state bytes, per-device TDP) is sourced from a datasheet, ISA spec, published benchmark, or
  measured microbenchmark; physical-limit invariants (MFU/MBU ≤ 1) hold across all simulatable devices

---

## Quick Start

One command sets everything up (Python venv + core engine + backend deps + frontend + database), another
starts both servers. Portable and idempotent — no hardcoded paths.

```bash
./setup.sh          # create .venv, install the engine (editable) + backend reqs + frontend, init DB,
                    #   and verify all core imports. Options: --no-frontend, --python python3.12
./start.sh          # start FastAPI backend (:8000) + React frontend (:3000), health-check based,
                    #   clean Ctrl+C shutdown. Options: --backend-only, --backend-port, --frontend-port
```

Then open the API docs at `http://localhost:8000/docs` and the UI at `http://localhost:3000`.

<details>
<summary>Manual setup</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e llm-memory-calculator/          # core performance-modeling engine
pip install -r BudSimulator/requirements.txt   # backend + all dependencies
cd BudSimulator && python run_api.py           # backend on :8000
cd BudSimulator/frontend && npm install && npm start   # frontend on :3000
```
</details>

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

**Which layers hold a KV cache** is read from the model's own config, through one table every
part of the simulator uses (the memory calculator, the GGUF reader in `bud_component.py`, the
GenZ performance graph -- prefill, decode and chunked -- the serving memory model, the prefix
cache, the engine estimate and the parameter counter). Only attention layers keep a growing cache -- full attention
every token, sliding / chunked layers at most their window, cross-attention the encoder's tokens,
MLA one latent per layer. Whether a windowed layer's saving is taken depends on the reader: the
**memory calculator** (`calculate_kv_cache`, `kv_cache_breakdown`, the engine estimate) charges
the windowed layers of a stack that also has full layers (Gemma 2/3, gpt-oss, Cohere2) the full
sequence unless the engine is declared able to hold a per-layer KV layout
(`engine_capabilities={"heterogeneous_kv_layout": True}`) -- the measured vLLM gpt-oss behaviour,
so by default it is an over-estimate for engines that do window them (Gemma-2-9B at 32k tokens:
11.27 GB, against about 6.3 GB windowed); a stack whose every attention layer shares one window
(Mistral v0.1) keeps the window everywhere. The GenZ graph and the FRD-027 facts describe the
model and apply the window on every windowed layer; the vLLM planner lays them out as sliding
KV-cache groups and charges every running sequence its blocks in every group, full and sliding. Gated DeltaNet (Qwen3-Next, Qwen3.5), Mamba, RWKV, xLSTM, short-convolution and
FFN-only layers keep a fixed per-sequence state instead (fp32, as the engines hold it), reported
as state memory and counted in the engine total; the serving simulator charges each request its
KV blocks plus that state (`MemoryModel.state_bytes_per_sequence`, also in the cluster and
disaggregation concurrency, the prefill-to-decode transfer and `/api/v2/memory/tiers`). Each KV
layer is as wide as the model's own attention head (`head_dim` or its aliases, e.g. Zamba2's
`attention_head_dim`), read by one function for every reader. An encoder-decoder (T5, BART,
Whisper, Moonshine, Dia, ProphetNet, and one nested under a multimodal `text_config`: BLIP-2 /
InstructBLIP with flan-T5, Florence-2) is charged its **decoder's** self-attention, on the
decoder's own depth, heads and width under each family's spellings (`decoder_layers`, T5's
`num_decoder_layers` and `num_heads`, ProphetNet's `num_decoder_attention_heads`, a nested
`decoder_config`, ...), **and** each decoder layer's cross-attention cache over the encoder's
output, at the cross-attention's own width: a fixed cost a request where the config states the
encoder's length (Whisper's 1,500 frames, an encoder block's `max_position_embeddings`, an image
encoder's patches -- a square or [height, width] `image_size`, and a staged tower's (`depths`:
Swin, Donut, Nougat) grid halved per side between stages -- or Pix2Struct's `seq_len`), else
charged over the sequence. The composition is read as the model builds it, from the configs as
the hub ships them: a `text_config` that says `is_decoder` beside a vision or audio tower is the
decoder of that tower (Pix2Struct, BLIP) even where an older file keeps a stale
`is_encoder_decoder: false` in it; a nested decoder block's own `is_encoder_decoder: false` never
hides the outer config's `true` (MusicGen); and a BART-family decoder block's depth is its
`decoder_layers`, not the `num_hidden_layers` its class aliases to the encoder's (Donut, Nougat).
A decoder-only stack's own cross-attention layers (Mllama's `cross_attention_layers`) are read by
the same function: those layers hold K / V over the vision tower's tokens and no self-attention
cache. The tower's tokens are one image's: no config states how many images a request carries,
so a request with several holds that cache once per image (the plan says so in a note).
Every reader charges both -- the calculator, the engine estimate, the FRD-027 facts and the vLLM
planner (its concurrency, profile scheduler and slice sizing), the KV planner's geometry, GenZ's
graphs and runtime modelling (`prefill_moddeling` / `decode_moddeling` / `chunked_moddeling` and
the `estimate_*_performance` above them; the prefill graph also runs the encoder, and the decode
and chunked graphs hold its weights, so their fit checks weigh the resident model), the serving
model's admission and transfer, and BudSimulator's database converter. A layer kind the table does not
know, a per-layer list shorter than the depth, a window whose layers the config does not place,
layers named sliding whose window the config states under no key the table reads, and a config with an SSM state and attention heads but no layout key (charged KV and state on
every layer) are over-estimated with a warning, never treated as free or clamped. Categories, the config and GGUF keys read, and the
conservative rules: [llm-memory-calculator/docs/LAYER_KINDS.md](llm-memory-calculator/docs/LAYER_KINDS.md).

The weights are read the same way, and from the same readers: an MoE layer's shared expert(s)
(`layer_plan.shared_expert_width`) and its routed experts under every spelling, MLA's latent as
its cache, an attention output gate, a gated FFN from the config's activation or its class's
default, an encoder at its own geometry (a hierarchical tower stage by stage, a conformer's
convolution module, a tower's patch and position tables) and an adapter stack (BLIP-2's
Q-Former). The serving simulator holds the whole model on the device: its weight load and the
cluster's and disaggregation's concurrency (`B_eff`) take the model's own parameter count -- the
FRD-027 facts' count of the config the GenZ model was read from, encoder, cross-attention, vision
and audio towers, adapters and every expert -- not the decoder's fields alone, which had held
Whisper large-v3 at 1.81 of its 3.09 GB and admitted more requests than a device holds. A nested
or flat decoder spelled the GPT-2 way (`n_layer` / `n_embd` / `n_head`) is read under the names
its class aliases them to. A block that leaves a role of its geometry to its config class (a hub BLIP-2's
`vision_config` and `qformer_config` that state only their model type, Qwen2-Audio's and LLaVA-1.5's
`text_config` with no depth, width or heads) is read at the class's defaults, as transformers builds it
(`layer_plan.with_class_geometry`, from the shipped snapshot), with a perceiver resampler, a text-reading
Q-Former and a staged DaViT tower counted at their own geometry. The training readers count the whole model
too: the parameter counter behind the training API counts every encoder-decoder with its encoder and
cross-attention and a language model with its towers, and GenZ's `training_modeling` holds its decoder-field
count to at least the model's own -- as do the training parallelization (what fits on a GPU), the cluster
requirements, the config builder, the cluster optimizer's quick checks and the training optimizer
(`optimize_training`, `compare_training_strategies`, `get_minimum_hardware_for_model`, through
`analyze_model_config`, which now reads a nested decoder and routed experts under every spelling:
DeepSeek-V3 had been 38.6B of its 671B) (`memory_model.held_to_the_model`).
GenZ's `training_modeling` charges an encoder-decoder's encoder layers their LoRA adapters and activations
as the parallelization estimate does (`memory_model.trained_layers`), each layer at its own width (an encoder's,
and a decoder layer's widest attention or mixer maps). LoRA adapter buffers are counted on
every layer's linear maps, a hybrid's recurrent and linear-attention layers among them (whether the plan or the
config class places them), with a gated FFN read by family as the weight counters read it, as vLLM wraps every
linear layer -- in the calculator and in the FRD-027 vLLM planner's reservation, which is A and B over the
(in + out) of every linear map the model's layers hold (`ModelFacts.lora_linear_features`). A gate holds every figure that moves below origin/main's -- or a capacity above it, or
a figure only the branch reports, or one it no longer reports -- to a ground truth (transformers'
own model for the config, a checkpoint's tensors, or the cited arithmetic):
`llm-memory-calculator/tests/test_kv_gate.py` over every config the repository ships (the GGUF
and ONNX-GenAI readers included, on metadata-only GGUF fixtures and genai_config.json folders) --
and fails on any public function of the calculator, BudSimulator or `bud_component.py` that reads a
model-shape key (found in the source with `ast`) which the harness neither runs nor names, with its
reason, in `tests/fixtures/kv_gate/unmeasured.json` -- no reader needs transformers or torch (the
config-class defaults ship with the package, with the layout a class gives a windowed config that
states none -- applied only where the config leaves the keys the class builds that layout from
out, or states them at values the rule was checked at -- and a head width a class derives; only a
type newer than transformers 5.12, a class layout no shipped rule reproduces, or a layout key stated
at another value asks the installed release, and without it is charged full KV on every windowed
layer), and a reader that cannot import either fails; the
GGUF reader needs `gguf` and the converter and the API routes BudSimulator's `requests`, `lxml`,
`json_repair` and `fastapi`, and
where those are absent their figures are reported as not measured -- and `scripts/kv_corpus.py --gate` over a machine's
HF cache and model folders; the figures that
stay below their truth without having moved, and the serving weights of the two checkpoints that
hold modules no config key describes (each such module named, sized and sourced in its gate
entry), are listed in LAYER_KINDS.md ("What errs low, measured").

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
