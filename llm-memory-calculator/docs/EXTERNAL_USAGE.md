# Using llm-memory-calculator in External Repositories

This guide shows how to use llm-memory-calculator with parallelism optimization in your own projects.

## Quick Setup

### Option 1: Use PyPI version (Python < 3.10)

```bash
pip install "llm-memory-calculator[genz]"
```

### Option 2: Use local GenZ (Python 3.10+, recommended)

1. **Clone or access the BudSimulator repository**:
   ```bash
   git clone https://github.com/abhibambhaniya/GenZ-LLM-Analyzer.git
   # or use existing BudSimulator directory
   ```

2. **Install GenZ locally**:
   ```bash
   pip install -e path/to/BudSimulator
   ```

3. **Install llm-memory-calculator**:
   ```bash
   pip install llm-memory-calculator
   ```

## Usage Examples

### Basic Memory Calculation

```python
from llm_memory_calculator import calculate_memory

# Calculate memory for any HuggingFace model
memory = calculate_memory("meta-llama/Llama-2-7b-hf", batch_size=32)
print(f"Total memory: {memory['total_memory_gb']:.2f} GB")
```

### Parallelism Optimization

```python
from llm_memory_calculator import (
    get_best_parallelization_strategy,
    get_various_parallelization,
    get_hardware_config,
    HARDWARE_CONFIGS
)

# Get available parallelization options
options = get_various_parallelization('llama2_7b', total_nodes=8)
print(f"Available configurations: {options}")

# Find optimal parallelism strategy
hardware = get_hardware_config('A100_80GB')
best = get_best_parallelization_strategy(
    stage='decode',
    model='llama2_7b',
    total_nodes=8,
    batch_size=32,
    system_name=hardware,
    bits='bf16'
)
print("Best configuration:")
print(best)

# List all supported hardware
print("Supported hardware:", list(HARDWARE_CONFIGS.keys()))
```

### Combining Memory and Parallelism

```python
from llm_memory_calculator import (
    calculate_memory,
    get_minimum_system_size,
    get_best_parallelization_strategy,
    HARDWARE_CONFIGS
)

model_name = "meta-llama/Llama-2-70b-hf"
batch_size = 32

# Calculate memory requirements
memory = calculate_memory(model_name, batch_size=batch_size)
print(f"Memory needed: {memory['total_memory_gb']:.2f} GB")

# Find minimum devices needed
hardware = HARDWARE_CONFIGS['A100_80GB']
min_devices = get_minimum_system_size(
    model='llama2_70b',
    batch_size=batch_size,
    system_name=hardware
)
print(f"Minimum devices: {min_devices}")

# Get optimal parallelism for minimum devices
best = get_best_parallelization_strategy(
    model='llama2_70b',
    total_nodes=min_devices,
    batch_size=batch_size,
    system_name=hardware
)
print(f"Optimal parallelism: TP={best['TP'].iloc[0]}, PP={best['PP'].iloc[0]}")
```

## Error Handling

```python
try:
    from llm_memory_calculator import get_best_parallelization_strategy
    print("Parallelism features available!")
except ImportError as e:
    print("Parallelism features not available:", e)
    print("Make sure GenZ is installed: pip install -e path/to/BudSimulator")
```

## Supported Models

The parallelism optimizer supports these model identifiers:
- `llama2_7b` - Llama 2 7B
- `llama2_13b` - Llama 2 13B  
- `llama2_70b` - Llama 2 70B
- `mixtral_8x7b` - Mixtral 8x7B
- And many more from the GenZ model registry

## Supported Hardware

Predefined configurations available in `HARDWARE_CONFIGS`:
- `A100_40GB` - NVIDIA A100 40GB
- `A100_80GB` - NVIDIA A100 80GB
- `H100_80GB` - NVIDIA H100 80GB
- `MI300X` - AMD MI300X
- `TPUv4` - Google TPU v4
- `TPUv5e` - Google TPU v5e

## Custom Hardware Configuration

```python
custom_hardware = {
    'Flops': 500,        # TFLOPS
    'Memory_size': 96,   # GB
    'Memory_BW': 3000,   # GB/s
    'ICN': 1000,         # GB/s interconnect bandwidth
    'real_values': True
}

best = get_best_parallelization_strategy(
    model='llama2_7b',
    total_nodes=4,
    system_name=custom_hardware
)
```

## Troubleshooting

### ImportError: GenZ not available
- Make sure GenZ is installed: `pip install -e path/to/BudSimulator`
- Check Python version compatibility (3.8-3.12 supported)

### ImportError: paretoset not available
- Only affects `get_pareto_optimal_performance` function
- Install with: `pip install paretoset`
- Other parallelism functions work without it

### Model not found
- Check supported models in GenZ model registry
- Use standard model identifiers like `llama2_7b`, not HuggingFace paths