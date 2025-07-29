# CPU Support for Performance Estimation

This document describes the CPU modeling capabilities integrated into llm-memory-calculator for LLM inference performance analysis on modern CPUs.

## Features

- **Multi-level Cache Hierarchy**: Models L1/L2/L3 caches with realistic hit rates and bandwidth constraints
- **ISA-aware Compute**: Supports x86 (SSE, AVX2, AVX-512, AMX) and ARM (NEON, SVE, SVE2) instruction sets
- **NUMA Modeling**: Multi-socket systems with NUMA distance matrices and memory placement optimization
- **Dynamic Frequency Scaling**: Turbo boost, AVX frequency reduction, thermal throttling
- **Threading Optimization**: SMT efficiency, parallel scaling with Amdahl's law
- **Pre-configured CPU Models**: Intel Xeon 8380, AMD EPYC 7763, AWS Graviton3

## Usage

### Method 1: Direct CPU-Aware Functions

Use the CPU-aware wrapper functions that automatically detect and handle CPU systems:

```python
from llm_memory_calculator.genz.cpu import cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling

# Prefill analysis with CPU
prefill_result = cpu_aware_prefill_moddeling(
    model='llama',
    batch_size=1,
    input_tokens=512,
    system_name='intel_xeon_8380',  # CPU preset name
    debug=False
)

print(f"Prefill Latency: {prefill_result['Latency']:.2f} ms")
print(f"Prefill Throughput: {prefill_result['Throughput']:.2f} tokens/s")

# Decode analysis with CPU
decode_result = cpu_aware_decode_moddeling(
    model='llama',
    batch_size=1,
    input_tokens=512,
    output_tokens=128,
    system_name='intel_xeon_8380',
    debug=False
)

print(f"Decode Latency: {decode_result['Latency']:.2f} ms")
print(f"Decode Throughput: {decode_result['Throughput']:.2f} tokens/s")
```

### Method 2: Using CPU System Class

For more control, use the CPUSystem class directly:

```python
from llm_memory_calculator.genz.cpu import CPUSystem
from llm_memory_calculator.genz.cpu.cpu_configs import CPU_PRESETS

# Create CPU system from preset
cpu_system = CPUSystem.from_preset('intel_xeon_8380')

# Or create custom CPU system
custom_cpu = CPUSystem(
    num_cores=64,
    frequency_ghz=2.3,
    flops_per_cycle=64,  # For AVX-512
    memory_bandwidth_gb_s=205,
    cache_sizes_kb={'L1': 48, 'L2': 1280, 'L3': 107520},
    numa_nodes=2
)

# Use in performance modeling
from llm_memory_calculator.genz.LLM_inference import prefill_moddeling

result = prefill_moddeling(
    model='llama',
    batch_size=1,
    input_tokens=512,
    system_name=cpu_system,
    debug=False
)
```

### Method 3: CPU Configuration in Hardware Dict

You can also pass CPU configurations as standard hardware dictionaries:

```python
cpu_config = {
    'Flops': 2.36,  # TFLOPS (much lower than GPUs)
    'Memory_size': 512,  # GB RAM
    'Memory_BW': 205,  # GB/s
    'ICN': 0,  # No interconnect for single CPU
    'real_values': True,
    'is_cpu': True,  # Important flag
    'cpu_config': {
        'num_cores': 40,
        'frequency_ghz': 2.3,
        'cache_sizes_kb': {'L1': 48, 'L2': 1280, 'L3': 107520}
    }
}
```

## Pre-configured CPU Models

### Intel Xeon Platinum 8380 (Ice Lake)
- 40 cores @ 2.3 GHz
- AVX-512 + AMX support
- 205 GB/s memory bandwidth
- 60 MB L3 cache

### AMD EPYC 7763 (Milan)
- 64 cores @ 2.45 GHz
- AVX2 support
- 204 GB/s memory bandwidth
- 256 MB L3 cache

### AWS Graviton3
- 64 cores @ 2.6 GHz
- ARM Neon + SVE support
- 307 GB/s memory bandwidth
- 32 MB L3 cache

## CPU vs GPU Performance Comparison

```python
from llm_memory_calculator import compare_performance_configurations

configs = [
    {
        'name': 'Intel Xeon 8380',
        'system_name': 'intel_xeon_8380',
        'tensor_parallel': 1,  # CPUs don't use tensor parallelism
        'bits': 'int8'  # CPUs perform better with INT8
    },
    {
        'name': 'NVIDIA A100',
        'system_name': get_hardware_config('A100_80GB'),
        'tensor_parallel': 4,
        'bits': 'bf16'
    }
]

results = compare_performance_configurations(
    model='llama2_7b',
    configurations=configs,
    batch_size=1,  # CPUs typically use smaller batches
    input_tokens=512,
    output_tokens=128
)

for result in results:
    print(f"{result['config_name']}: {result['total_throughput']:.1f} tokens/s")
```

## CPU Optimization Tips

1. **Use INT8 quantization**: CPUs have better INT8 support than GPUs
2. **Keep batch size small**: CPUs have limited memory bandwidth
3. **Enable all cores**: Ensure proper threading configuration
4. **Consider NUMA**: Pin memory to local NUMA nodes
5. **Monitor frequency**: Watch for thermal throttling

## Advanced CPU Features

### NUMA Optimization
```python
cpu_system = CPUSystem(
    num_cores=128,
    numa_nodes=4,
    numa_distance_matrix=[
        [10, 20, 20, 20],
        [20, 10, 20, 20],
        [20, 20, 10, 20],
        [20, 20, 20, 10]
    ]
)
```

### ISA-Specific Optimization
```python
# AMX optimization for Intel
cpu_system.isa_features = ['avx512', 'amx']
cpu_system.amx_tile_size = 16  # For INT8 operations

# SVE optimization for ARM
cpu_system.isa_features = ['neon', 'sve']
cpu_system.sve_vector_length = 512  # bits
```

## Limitations

- CPU performance is typically 10-100x slower than GPUs for LLM inference
- Best suited for edge deployment or when GPUs are unavailable
- Memory bandwidth is the primary bottleneck
- Limited parallelism options compared to multi-GPU setups