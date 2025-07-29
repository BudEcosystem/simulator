# Performance Estimation API Guide

🚀 **llm-memory-calculator** now includes comprehensive LLM performance estimation capabilities powered by the integrated GenZ-LLM-Analyzer. This guide shows how to use these features in external repositories.

## Quick Start

### Installation
```bash
pip install llm-memory-calculator
```

### Basic Usage
```python
from llm_memory_calculator import (
    estimate_end_to_end_performance,
    get_hardware_config,
    compare_performance_configurations
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
```

## Available Functions

### Core Performance Functions

#### `estimate_prefill_performance()`
Estimates performance for the prefill phase (processing input prompt).

```python
result = estimate_prefill_performance(
    model='llama2_7b',           # Model identifier
    batch_size=4,                # Number of sequences
    input_tokens=2048,           # Input sequence length
    system_name=hardware,        # Hardware configuration
    bits='bf16',                 # Precision (fp32, bf16, int8, int4)
    tensor_parallel=2,           # Tensor parallelism degree
    pipeline_parallel=1,         # Pipeline parallelism degree
    expert_parallel=1,           # Expert parallelism (MoE models)
    debug=False                  # Enable debug output
)

# Returns: {'Latency': ms, 'Throughput': tokens/s, 'TTFT': ms, ...}
```

#### `estimate_decode_performance()`
Estimates performance for the decode phase (generating output tokens).

```python
result = estimate_decode_performance(
    model='llama2_7b',
    batch_size=4,
    beam_size=1,                 # Beam search width
    input_tokens=2048,           # Context length
    output_tokens=256,           # Tokens to generate
    system_name=hardware,
    bits='bf16',
    tensor_parallel=2,
    pipeline_parallel=1,
    expert_parallel=1
)

# Returns: {'Latency': ms_per_token, 'Total_latency': ms, 'TPOT': ms, ...}
```

#### `estimate_end_to_end_performance()`
Combines prefill and decode for complete inference analysis.

```python
result = estimate_end_to_end_performance(
    model='llama2_7b',
    batch_size=4,
    input_tokens=2048,
    output_tokens=256,
    system_name=hardware,
    bits='bf16',
    tensor_parallel=2
)

# Returns comprehensive metrics:
# {
#   'total_latency': 1250.5,           # Total inference time (ms)
#   'ttft': 198.3,                     # Time to first token (ms)
#   'average_tpot': 4.2,               # Avg time per output token (ms)
#   'total_throughput': 8750.2,        # Overall throughput (tokens/s)
#   'prefill': {...},                  # Prefill phase details
#   'decode': {...}                    # Decode phase details
# }
```

#### `compare_performance_configurations()`
Compare multiple hardware/parallelism configurations.

```python
configurations = [
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
    configurations=configurations,
    batch_size=4,
    input_tokens=2048,
    output_tokens=256
)

# Results sorted by throughput (best first)
for result in results:
    name = result['config_name']
    throughput = result['total_throughput']
    print(f"{name}: {throughput:.1f} tokens/s")
```

### Hardware Configuration

#### `get_hardware_config(hardware_name)`
Get predefined hardware configurations.

```python
# Available hardware
hardware_options = [
    'A100_40GB',    # NVIDIA A100 40GB
    'A100_80GB',    # NVIDIA A100 80GB  
    'H100_80GB',    # NVIDIA H100 80GB
    'MI300X',       # AMD MI300X
    'TPUv4',        # Google TPU v4
    'TPUv5e'        # Google TPU v5e
]

hardware = get_hardware_config('A100_80GB')
# Returns: {
#   'Flops': 312,        # TFLOPS
#   'Memory_size': 80,   # GB
#   'Memory_BW': 2039,   # GB/s
#   'ICN': 600,          # Interconnect GB/s
#   'real_values': True
# }
```

#### `HARDWARE_CONFIGS`
Access all hardware configurations directly.

```python
from llm_memory_calculator import HARDWARE_CONFIGS

for name, config in HARDWARE_CONFIGS.items():
    print(f"{name}: {config['Memory_size']} GB, {config['Flops']} TFLOPS")
```

### Parallelism Optimization

#### `get_best_parallelization_strategy()`
Find optimal tensor/pipeline parallelism configuration.

```python
best_strategy = get_best_parallelization_strategy(
    model='llama2_7b',
    total_nodes=8,               # Total GPUs available
    batch_size=16,
    input_tokens=2048,
    output_tokens=512,
    system_name=hardware,
    bits='bf16',
    stage='decode'               # 'prefill' or 'decode'
)

# Returns DataFrame with best configurations
tp = best_strategy['TP'].iloc[0]           # Best tensor parallel
pp = best_strategy['PP'].iloc[0]           # Best pipeline parallel
throughput = best_strategy['Tokens/s'].iloc[0]
print(f"Optimal: TP={tp}, PP={pp} → {throughput:.1f} tokens/s")
```

#### `get_various_parallelization(model, total_nodes)`
Get all valid parallelization options.

```python
options = get_various_parallelization('llama2_7b', total_nodes=8)
# Returns: [(1, 4), (2, 2), (4, 1)] for TP/PP combinations
```

## Supported Models

Currently supported model identifiers:
- `llama2_7b`, `llama2_13b`, `llama2_70b`
- `mistral_7b`
- `mixtral_8x7b`
- `falcon_7b`, `falcon_40b`
- Custom models via configuration

## Practical Examples

### Use Case 1: Hardware Selection
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

### Use Case 2: Batch Size Optimization
```python
# Find optimal batch size for given hardware
hardware = get_hardware_config('A100_80GB')
batch_sizes = [1, 2, 4, 8, 16, 32]

print("Batch size scaling analysis:")
for batch_size in batch_sizes:
    result = estimate_decode_performance(
        model='llama2_7b',
        batch_size=batch_size,
        input_tokens=2048,
        output_tokens=1,  # Single token for per-token metrics
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4
    )
    
    latency = result['Latency']
    throughput = result.get('Effective_throughput', 0)
    print(f"  Batch {batch_size}: {latency:.1f} ms/token, {throughput:.0f} tok/s")
```

### Use Case 3: Precision Impact Analysis
```python
# Compare different precisions
precisions = ['bf16', 'int8', 'int4']
hardware = get_hardware_config('A100_80GB')

print("Precision comparison:")
for precision in precisions:
    try:
        result = estimate_end_to_end_performance(
            model='llama2_7b',
            batch_size=4,
            input_tokens=2048,
            output_tokens=128,
            system_name=hardware,
            bits=precision,
            tensor_parallel=4
        )
        
        throughput = result['total_throughput']
        ttft = result['ttft']
        print(f"  {precision}: {throughput:.0f} tok/s, TTFT: {ttft:.1f} ms")
        
    except Exception as e:
        print(f"  {precision}: Not supported - {e}")
```

### Use Case 4: Multi-Model Comparison
```python
# Compare different model sizes
models = ['llama2_7b', 'llama2_13b', 'llama2_70b']
hardware = get_hardware_config('H100_80GB')

print("Model size comparison:")
for model in models:
    # Adjust TP based on model size
    tp = {'llama2_7b': 2, 'llama2_13b': 4, 'llama2_70b': 8}[model]
    
    try:
        result = estimate_end_to_end_performance(
            model=model,
            batch_size=4,
            input_tokens=2048,
            output_tokens=256,
            system_name=hardware,
            bits='bf16',
            tensor_parallel=tp
        )
        
        throughput = result['total_throughput']
        ttft = result['ttft']
        print(f"  {model} (TP={tp}): {throughput:.0f} tok/s, TTFT: {ttft:.1f} ms")
        
    except Exception as e:
        print(f"  {model}: Error - {e}")
```

## Integration Examples

### Flask API Integration
```python
from flask import Flask, jsonify, request
from llm_memory_calculator import estimate_end_to_end_performance, get_hardware_config

app = Flask(__name__)

@app.route('/estimate_performance', methods=['POST'])
def estimate_performance():
    data = request.json
    
    try:
        hardware = get_hardware_config(data['hardware'])
        result = estimate_end_to_end_performance(
            model=data['model'],
            batch_size=data.get('batch_size', 1),
            input_tokens=data.get('input_tokens', 2048),
            output_tokens=data.get('output_tokens', 256),
            system_name=hardware,
            bits=data.get('bits', 'bf16'),
            tensor_parallel=data.get('tensor_parallel', 1)
        )
        
        return jsonify({
            'success': True,
            'throughput': result['total_throughput'],
            'ttft': result['ttft'],
            'total_latency': result['total_latency']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

### MLOps Pipeline Integration
```python
import yaml
from llm_memory_calculator import (
    get_best_parallelization_strategy,
    estimate_end_to_end_performance,
    get_hardware_config
)

def optimize_deployment_config(model_name, target_throughput, available_gpus):
    """Optimize deployment configuration for MLOps pipeline."""
    
    # Find best parallelism strategy
    hardware = get_hardware_config('A100_80GB')
    strategy = get_best_parallelization_strategy(
        model=model_name,
        total_nodes=available_gpus,
        batch_size=8,
        system_name=hardware,
        bits='bf16'
    )
    
    best_tp = strategy['TP'].iloc[0]
    best_pp = strategy['PP'].iloc[0]
    expected_throughput = strategy['Tokens/s'].iloc[0]
    
    # Validate performance meets requirements
    meets_target = expected_throughput >= target_throughput
    
    config = {
        'model': model_name,
        'parallelism': {
            'tensor_parallel': int(best_tp),
            'pipeline_parallel': int(best_pp)
        },
        'expected_performance': {
            'throughput': float(expected_throughput),
            'meets_target': meets_target
        },
        'deployment': {
            'gpus_required': int(best_tp * best_pp),
            'gpu_type': 'A100_80GB'
        }
    }
    
    return config

# Example usage
config = optimize_deployment_config('llama2_7b', 5000, 8)
print(yaml.dump(config, default_flow_style=False))
```

## Error Handling

```python
from llm_memory_calculator import estimate_end_to_end_performance, get_hardware_config

try:
    hardware = get_hardware_config('A100_80GB')
    result = estimate_end_to_end_performance(
        model='llama2_7b',
        batch_size=4,
        input_tokens=2048,
        output_tokens=256,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=2
    )
    
    print(f"Performance: {result['total_throughput']:.1f} tokens/s")
    
except ImportError:
    print("Install llm-memory-calculator: pip install llm-memory-calculator")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Estimation failed: {e}")
```

## Performance Tips

1. **Use appropriate tensor parallelism**: Start with TP=2-4 for 7B models
2. **Cache hardware configs**: Reuse `get_hardware_config()` results
3. **Batch requests**: Use `compare_performance_configurations()` for multiple configs
4. **Monitor precision tradeoffs**: INT8 can provide 1.5-2x speedup vs BF16
5. **Consider memory constraints**: Larger batch sizes need more GPU memory

## Advanced Usage

For advanced scenarios, you can also access the underlying GenZ functions directly:

```python
from llm_memory_calculator.genz.LLM_inference import (
    prefill_moddeling,
    decode_moddeling
)

# Direct access to GenZ modeling functions
# (Use wrapper functions above for easier API)
```

## Support

- 📦 **Installation**: `pip install llm-memory-calculator`
- 📚 **Documentation**: Check README.md and examples/
- 🐛 **Issues**: Report at project repository
- 💡 **Examples**: See `examples/` directory for comprehensive demos

The performance estimation API is production-ready and actively used in BudSimulator for LLM deployment optimization.