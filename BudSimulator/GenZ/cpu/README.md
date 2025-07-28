# CPU Support for GenZ

This module adds comprehensive CPU modeling capabilities to GenZ, enabling accurate performance analysis of LLM inference on modern CPUs.

## Features

- **Multi-level Cache Hierarchy**: Models L1/L2/L3 caches with realistic hit rates and bandwidth constraints
- **ISA-aware Compute**: Supports x86 (SSE, AVX2, AVX-512, AMX) and ARM (NEON, SVE, SVE2) instruction sets
- **NUMA Modeling**: Multi-socket systems with NUMA distance matrices and memory placement optimization
- **Dynamic Frequency Scaling**: Turbo boost, AVX frequency reduction, thermal throttling
- **Threading Optimization**: SMT efficiency, parallel scaling with Amdahl's law
- **Pre-configured CPU Models**: Intel Xeon 8380, AMD EPYC 7763, AWS Graviton3

## Installation

The CPU support is integrated into GenZ. No additional installation required.

## Usage

### Method 1: Direct CPU-Aware Functions

Use the CPU-aware wrapper functions that automatically detect and handle CPU systems:

```python
from GenZ.cpu import cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling

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

### Method 2: Create Custom CPU System

Create a CPU system object with custom configuration:

```python
from GenZ.cpu import create_cpu_system, cpu_aware_prefill_moddeling

# Create CPU system
cpu_system = create_cpu_system('amd_epyc_7763')

# Customize configuration
cpu_system.config.smt_enabled = False  # Disable SMT
cpu_system.config.numa_nodes = 4       # Adjust NUMA configuration

# Use with analysis
result = cpu_aware_prefill_moddeling(
    model='gpt2',
    batch_size=1,
    input_tokens=256,
    system_name=cpu_system,  # Pass CPU system object
    debug=False
)
```

### Method 3: Automatic CPU Detection

Enable automatic CPU detection for all GenZ functions:

```python
from GenZ.cpu import enable_cpu_aware_inference
from GenZ.LLM_inference import prefill_moddeling, decode_moddeling

# Enable automatic CPU detection
enable_cpu_aware_inference()

# Now regular GenZ functions automatically detect CPU systems
result = prefill_moddeling(
    model='llama',
    batch_size=1,
    input_tokens=512,
    system_name='aws_graviton3',  # Automatically detected as CPU
    debug=False
)

# Disable when done
from GenZ.cpu import disable_cpu_aware_inference
disable_cpu_aware_inference()
```

## CPU-Specific Metrics

When using CPU systems, the analysis results include additional metrics:

- **ISA_used**: The instruction set used (e.g., 'avx512', 'amx', 'sve2')
- **Thread_count**: Number of threads used for the operation
- **Parallel_efficiency**: Efficiency of parallel execution (0-1)
- **Frequency_GHz**: Operating frequency considering turbo and throttling
- **L1_hit_rate**: Fraction of memory accesses served by L1 cache
- **L2_hit_rate**: Fraction of memory accesses served by L2 cache
- **L3_hit_rate**: Fraction of memory accesses served by L3 cache
- **DRAM_access_rate**: Fraction of memory accesses going to main memory

## Pre-configured CPU Models

### Intel Xeon 8380 (Ice Lake)
- 40 cores per socket, 2 sockets, 2 threads per core
- AVX-512 and AMX support
- 60MB L3 cache per socket
- 8 memory channels per socket

### AMD EPYC 7763 (Milan)
- 64 cores per socket, 2 sockets, 2 threads per core
- AVX2 support (no AVX-512)
- 256MB L3 cache per socket
- 8 NUMA nodes (CCDs)

### AWS Graviton3 (Neoverse V1)
- 64 cores, single socket, no SMT
- SVE2 and NEON support
- 32MB L3 cache
- Fixed frequency operation

## Analyzing Prefill vs Decode

```python
from GenZ.cpu import cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling

# Analyze prefill phase
prefill = cpu_aware_prefill_moddeling(
    model='llama',
    batch_size=1,
    input_tokens=512,
    system_name='intel_xeon_8380'
)

# Analyze decode phase with different KV cache sizes
for kv_size in [128, 256, 512, 1024]:
    decode = cpu_aware_decode_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=kv_size,  # KV cache size
        output_tokens=1,       # Generate 1 token
        system_name='intel_xeon_8380'
    )
    print(f"KV={kv_size}: {decode['Latency']:.2f}ms, {decode['Throughput']:.0f} tok/s")
```

## Limitations

1. **Model Profiling Mode**: When using `model_profilling=True`, CPU-specific metrics are not included in the results to maintain compatibility with the base GenZ format.

2. **Collective Communication**: CPU systems don't model inter-node communication for distributed inference.

3. **Operator Coverage**: CPU enhancements are currently implemented for GEMM, FC, Logit, Attend, and CONV2D operators.

## Advanced Usage

### Custom Cache Configuration

```python
from GenZ.cpu import create_cpu_system

cpu = create_cpu_system('intel_xeon_8380')

# Modify cache sizes
cpu.config.l3_config.size = 90 * 1024 * 1024  # 90MB L3
cpu.config.l2_config.size = 2 * 1024 * 1024   # 2MB L2
```

### NUMA Optimization

```python
# Create CPU with custom NUMA configuration
cpu = create_cpu_system('amd_epyc_7763')

# Modify NUMA distance matrix for different topology
import numpy as np
cpu.config.numa_distance_matrix = np.array([
    [10, 20, 30, 40],
    [20, 10, 20, 30],
    [30, 20, 10, 20],
    [40, 30, 20, 10]
])
```

### ISA Selection Control

```python
# Disable certain ISAs
cpu = create_cpu_system('intel_xeon_8380')
cpu.config.isa_support = ['avx2', 'sse']  # Disable AVX-512 and AMX
```

## Integration with Existing Code

The CPU support is designed to be fully backward compatible. Existing GenZ code continues to work unchanged, and CPU support is only activated when a CPU system is explicitly used.

```python
# This still works exactly as before
from GenZ.LLM_inference import prefill_moddeling

# GPU analysis - unchanged
gpu_result = prefill_moddeling(
    model='llama',
    system_name='A100_40GB_GPU'  # GPU system
)

# CPU analysis - automatic detection with cpu_aware functions
from GenZ.cpu import cpu_aware_prefill_moddeling
cpu_result = cpu_aware_prefill_moddeling(
    model='llama',
    system_name='intel_xeon_8380'  # CPU system
)
```

## Architecture

### Module Structure

```