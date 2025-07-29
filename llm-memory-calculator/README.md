# LLM Memory Calculator

🧮 Calculate memory requirements for Large Language Models across diverse architectures with accurate, production-tested algorithms.

## Features

- 🎯 **Accurate Parameter Counting**: Handles all model architectures (Transformers, Mamba, Hybrid, MoE)
- 💾 **Comprehensive Memory Calculation**: Weights, KV Cache, Activations, State Memory
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

## Installation

```bash
pip install llm-memory-calculator
```

Or install from source:
```bash
git clone https://github.com/yourusername/llm-memory-calculator.git
cd llm-memory-calculator
pip install -e .
```

## Quick Start

### Basic Usage

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

### Custom Configuration

```python
# From config dictionary
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

### Advanced Analysis

```python
from llm_memory_calculator import analyze_hf_model, compare_models

# Detailed model analysis
analysis = analyze_hf_model(
    "deepseek-ai/DeepSeek-V3",
    seq_length=32768,
    batch_size=1,
    precision="fp16"
)

# Compare multiple models
models = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b"
]
comparison = compare_models(models, seq_length=4096)
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

## Examples

### Estimate Maximum Batch Size

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

### Analyze Attention Efficiency

```python
from llm_memory_calculator import analyze_attention_efficiency

efficiency = analyze_attention_efficiency(
    "mistralai/Mistral-7B-v0.1",
    seq_lengths=[1024, 4096, 16384, 32768]
)

for seq_len, data in efficiency['results'].items():
    print(f"Seq {seq_len}: {data['kv_cache_gb']:.2f} GB ({data['kv_cache_percent']:.1f}%)")
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

### Classes

- `ModelMemoryCalculator`: Core calculator class
- `UniversalParameterCounter`: Parameter counting for all architectures
- `HuggingFaceConfigLoader`: HuggingFace Hub integration
- `MemoryReport`: Detailed memory breakdown report

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This calculator is based on production experience from BudSimulator and includes optimizations for real-world deployment scenarios.