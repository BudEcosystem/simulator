# LLM Memory Calculator

🧮 Calculate memory requirements for Large Language Models across diverse architectures with accurate, production-tested algorithms. Now with integrated performance estimation powered by GenZ-LLM-Analyzer!

## Features

- 🎯 **Accurate Parameter Counting**: Handles all model architectures (Transformers, Mamba, Hybrid, MoE)
- 💾 **Comprehensive Memory Calculation**: Weights, KV Cache, Activations, State Memory
- ⚡ **Performance Estimation**: Prefill/decode latency, throughput analysis, hardware comparison
- 🚀 **Parallelism Optimization**: Built-in GenZ-LLM-Analyzer for optimal tensor/pipeline parallelism
- 🏗️ **Architecture Support**: 
  - Transformers (MHA, MQA, GQA, MLA)
  - State-Space Models (Mamba, S4)
  - Mixture of Experts (MoE)
  - Hybrid Models (Jamba)
  - Diffusion Models
  - Multimodal Models
- 🤗 **HuggingFace Integration**: Analyze any model from HuggingFace Hub
- 🔧 **Framework Overhead**: Realistic memory estimates with configurable overhead
- 📊 **GPU Recommendations**: Get optimal GPU sizing for your models
- 🔧 **Hardware Configs**: Pre-defined configs for A100, H100, MI300X, TPU, and more

## Installation

```bash
pip install llm-memory-calculator
```

This installs llm-memory-calculator with built-in parallelism optimization and performance estimation powered by GenZ-LLM-Analyzer. No additional setup required!

**Optional: Pareto Optimization**
For advanced Pareto-optimal configuration analysis:
```bash
pip install "llm-memory-calculator[pareto]"
```

**Requirements:**
- Python 3.9+ (Python 3.12 fully supported)
- Works on Linux, macOS, and Windows

## Quick Start

### Basic Memory Calculation

```python
from llm_memory_calculator import calculate_memory

# From HuggingFace model ID
result = calculate_memory("meta-llama/Llama-2-7b-hf")
print(result)

# With custom parameters
result = calculate_memory(
    "mistralai/Mixtral-8x7B-v0.1",
    batch_size=4,
    seq_length=8192,
    precision="int8"
)
print(f"Total Memory: {result.total_memory_gb:.2f} GB")
print(f"Recommended GPU: {result.recommended_gpu_memory_gb} GB")
```

### Performance Estimation

```python
from llm_memory_calculator import (
    estimate_end_to_end_performance,
    get_hardware_config
)

# Get hardware configuration
hardware = get_hardware_config('A100_80GB')

# Estimate performance
result = estimate_end_to_end_performance(
    model='llama2_7b',
    batch_size=4,
    input_tokens=2048,
    output_tokens=256,
    system_name=hardware,
    bits='bf16',
    tensor_parallel=2
)

print(f"Throughput: {result['total_throughput']:.1f} tokens/s")
print(f"TTFT: {result['ttft']:.1f} ms")
print(f"TPOT: {result['average_tpot']:.1f} ms")
```

### Parallelism Optimization

```python
from llm_memory_calculator import (
    get_best_parallelization_strategy,
    get_hardware_config,
    HARDWARE_CONFIGS
)

# Use predefined hardware config
hardware = get_hardware_config('A100_80GB')

# Find optimal parallelism for 8 GPUs
best_strategy = get_best_parallelization_strategy(
    stage='decode',
    model='llama2_7b',
    total_nodes=8,
    batch_size=32,
    input_tokens=2048,
    output_tokens=256,
    system_name=hardware,
    bits='bf16'
)

print(f"Best configuration:")
print(f"  Tensor Parallel: {best_strategy['TP'].iloc[0]}")
print(f"  Pipeline Parallel: {best_strategy['PP'].iloc[0]}")
print(f"  Throughput: {best_strategy['Tokens/s'].iloc[0]:.1f} tokens/s")
```

## Core Features

### Memory Calculation

Calculate detailed memory requirements for any LLM:

```python
from llm_memory_calculator import calculate_memory

# Custom configuration
config = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # GQA
    "vocab_size": 32000,
    "intermediate_size": 11008,
}

result = calculate_memory(config, batch_size=1, seq_length=4096)
print(result)
```

### Performance Estimation API

#### Prefill Performance
```python
from llm_memory_calculator import estimate_prefill_performance

result = estimate_prefill_performance(
    model='llama2_7b',
    batch_size=4,
    input_tokens=2048,
    system_name=hardware,
    bits='bf16',
    tensor_parallel=2
)
print(f"Prefill latency: {result['Latency']:.1f} ms")
```

#### Decode Performance
```python
from llm_memory_calculator import estimate_decode_performance

result = estimate_decode_performance(
    model='llama2_7b',
    batch_size=4,
    input_tokens=2048,
    output_tokens=256,
    system_name=hardware,
    bits='bf16',
    tensor_parallel=2
)
print(f"Per-token latency: {result['Latency']:.1f} ms")
```

#### Hardware Comparison
```python
from llm_memory_calculator import compare_performance_configurations

configs = [
    {
        'name': 'A100_80GB_TP4',
        'system_name': get_hardware_config('A100_80GB'),
        'tensor_parallel': 4,
        'bits': 'bf16'
    },
    {
        'name': 'H100_80GB_TP4',
        'system_name': get_hardware_config('H100_80GB'),
        'tensor_parallel': 4,
        'bits': 'bf16'
    }
]

results = compare_performance_configurations(
    model='llama2_7b',
    configurations=configs,
    batch_size=4,
    input_tokens=2048,
    output_tokens=256
)

for result in results:
    print(f"{result['config_name']}: {result['total_throughput']:.1f} tokens/s")
```

### Advanced Analysis

#### Attention Efficiency Analysis
```python
from llm_memory_calculator import analyze_attention_efficiency

efficiency = analyze_attention_efficiency(
    "mistralai/Mistral-7B-v0.1",
    seq_lengths=[1024, 4096, 16384, 32768]
)

for seq_len, data in efficiency['results'].items():
    print(f"Seq {seq_len}: {data['kv_cache_gb']:.2f} GB ({data['kv_cache_percent']:.1f}%)")
```

#### Maximum Batch Size Estimation
```python
from llm_memory_calculator import estimate_max_batch_size

max_batch = estimate_max_batch_size(
    "meta-llama/Llama-2-70b-hf",
    gpu_memory_gb=80,
    seq_length=4096,
    precision="fp16"
)
print(f"Maximum batch size for 80GB GPU: {max_batch}")
```

## Attention Mechanisms

The calculator supports all modern attention variants:

| Type | Description | Memory Efficiency |
|------|-------------|------------------|
| **MHA** | Multi-Head Attention | Baseline (1x) |
| **MQA** | Multi-Query Attention | ~8-32x reduction |
| **GQA** | Grouped-Query Attention | ~2-8x reduction |
| **MLA** | Multi-Latent Attention | ~10-100x reduction |

## Memory Components

The calculator provides detailed breakdown:

- **Weight Memory**: Model parameters (divided by tensor parallelism)
- **KV Cache**: Key-Value cache for attention (varies by attention type)
- **Activation Memory**: Intermediate activations during forward pass
- **State Memory**: SSM/Mamba state (constant regardless of sequence length)
- **Image Memory**: Vision transformer image embeddings
- **Extra/Overhead**: Framework overhead and temporary buffers

## Supported Hardware

Pre-configured hardware available:
- `A100_40GB` - NVIDIA A100 40GB (312 TFLOPS, 1.5 TB/s)
- `A100_80GB` - NVIDIA A100 80GB (312 TFLOPS, 2.0 TB/s)
- `H100_80GB` - NVIDIA H100 80GB (989 TFLOPS, 3.35 TB/s)
- `MI300X` - AMD MI300X (1307 TFLOPS, 5.3 TB/s)
- `TPUv4` - Google TPU v4 (275 TFLOPS)
- `TPUv5e` - Google TPU v5e (197 TFLOPS)

## Examples

### Complete Example: Memory + Performance + Parallelism

```python
from llm_memory_calculator import (
    calculate_memory,
    get_best_parallelization_strategy,
    estimate_end_to_end_performance,
    get_minimum_system_size,
    HARDWARE_CONFIGS
)

# 1. Calculate memory requirements
memory = calculate_memory("meta-llama/Llama-2-70b-hf", batch_size=32)
print(f"Total memory needed: {memory.total_memory_gb:.2f} GB")

# 2. Find minimum devices needed
min_devices = get_minimum_system_size(
    model='llama2_70b',
    batch_size=32,
    system_name=HARDWARE_CONFIGS['A100_80GB']
)
print(f"Minimum devices needed: {min_devices}")

# 3. Get optimal parallelism
best = get_best_parallelization_strategy(
    model='llama2_70b',
    total_nodes=min_devices,
    batch_size=32,
    system_name=HARDWARE_CONFIGS['A100_80GB']
)
print(f"Optimal: TP={best['TP'].iloc[0]}, PP={best['PP'].iloc[0]}")

# 4. Estimate performance
perf = estimate_end_to_end_performance(
    model='llama2_70b',
    batch_size=32,
    input_tokens=2048,
    output_tokens=256,
    system_name=HARDWARE_CONFIGS['A100_80GB'],
    tensor_parallel=best['TP'].iloc[0],
    pipeline_parallel=best['PP'].iloc[0]
)
print(f"Expected throughput: {perf['total_throughput']:.1f} tokens/s")
```

### Hardware Selection for Production

```python
from llm_memory_calculator import (
    estimate_end_to_end_performance,
    get_hardware_config
)

# Compare hardware for production deployment
hardware_options = ['A100_80GB', 'H100_80GB', 'MI300X']
target_throughput = 10000  # tokens/s

print("Hardware comparison for Llama2-7B:")
for hw_name in hardware_options:
    hardware = get_hardware_config(hw_name)
    result = estimate_end_to_end_performance(
        model='llama2_7b',
        batch_size=8,
        input_tokens=2048,
        output_tokens=256,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4
    )
    
    throughput = result['total_throughput']
    meets_target = throughput >= target_throughput
    print(f"  {hw_name}: {throughput:.0f} tok/s {'✅' if meets_target else '❌'}")
```

## Supported Model Architectures

- **Decoder-only Transformers**: Llama, Mistral, GPT, Falcon, MPT
- **Encoder-Decoder**: T5, BART, Marian
- **Mixture of Experts**: Mixtral, DeepSeek-MoE, Grok
- **State-Space Models**: Mamba, S4, RWKV
- **Hybrid Models**: Jamba (Attention + Mamba)
- **Multimodal**: LLaVA, CLIP, Flamingo
- **Diffusion**: Stable Diffusion, DALL-E
- **Speech**: Whisper, Bark

## API Reference

### Main Functions

- `calculate_memory()`: Calculate memory from model ID or config
- `analyze_hf_model()`: Analyze a HuggingFace model
- `compare_models()`: Compare multiple models
- `estimate_max_batch_size()`: Find optimal batch size
- `analyze_attention_efficiency()`: Analyze KV cache scaling

### Performance Functions

- `estimate_prefill_performance()`: Prefill phase analysis
- `estimate_decode_performance()`: Decode phase analysis
- `estimate_end_to_end_performance()`: Complete inference analysis
- `compare_performance_configurations()`: Multi-config comparison

### Parallelism Functions

- `get_best_parallelization_strategy()`: Find optimal TP/PP
- `get_various_parallelization()`: List valid configurations
- `get_minimum_system_size()`: Calculate minimum GPUs needed
- `get_hardware_config()`: Get hardware specifications

### Classes

- `ModelMemoryCalculator`: Core calculator class
- `UniversalParameterCounter`: Parameter counting for all architectures
- `HuggingFaceConfigLoader`: HuggingFace Hub integration
- `MemoryReport`: Detailed memory breakdown report

## Advanced Topics

For advanced usage including LoRA support, CPU modeling, and custom hardware configurations, see the [documentation](docs/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This calculator is based on production experience from BudSimulator and includes the integrated GenZ-LLM-Analyzer for comprehensive performance analysis.