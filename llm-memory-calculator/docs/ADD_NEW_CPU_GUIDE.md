# Guide: Adding a New CPU to GenZ

This guide explains how to add support for a new CPU model to the GenZ CPU support system.

## Overview

To add a new CPU, you need to:
1. Add the CPU configuration to `cpu_configs.py`
2. Test the configuration
3. (Optional) Add CPU-specific optimizations

## Step 1: Add CPU Configuration

Edit `GenZ/cpu/cpu_configs.py` and add your CPU to the `CPU_PRESETS` dictionary.

### Required Parameters

Each CPU configuration consists of two parts:

#### 1. Base Parameters (`base_params`)
```python
'base_params': {
    'unit': None,                    # Leave as None (set automatically)
    'frequency': 2.1e9,              # Base frequency in Hz
    'bits': 'bf16',                  # Default precision ('fp32', 'bf16', 'fp16', 'int8')
    'compute_efficiency': 0.85,      # Compute efficiency (0-1)
    'memory_efficiency': 0.75,       # Memory efficiency (0-1)
}
```

#### 2. CPU-Specific Configuration (`cpu_specific`)
```python
'cpu_specific': CPUConfig(
    # Core configuration
    cores_per_socket=32,             # Number of cores per socket
    sockets=2,                       # Number of sockets
    threads_per_core=2,              # SMT threads (1 for no SMT, 2 for HT/SMT)
    
    # Cache hierarchy
    l1i_config=CacheConfig(...),     # L1 instruction cache
    l1d_config=CacheConfig(...),     # L1 data cache
    l2_config=CacheConfig(...),      # L2 cache
    l3_config=CacheConfig(...),      # L3 cache
    
    # NUMA configuration
    numa_nodes=2,                    # Number of NUMA nodes
    cores_per_numa=32,               # Cores per NUMA node
    numa_distance_matrix=...,        # NUMA distances
    
    # ISA support
    isa_support=['avx512', 'avx2'],  # Supported instruction sets
    
    # Frequency scaling
    base_frequency=2.1e9,            # Base frequency
    turbo_frequency_curve={...},     # Turbo frequencies
    avx_frequency_offset={...},      # ISA frequency penalties
    
    # Memory
    dram_bandwidth_per_channel=25.6, # GB/s per channel
    memory_channels_per_socket=8,    # Memory channels
    
    # Vendor info
    vendor='intel',                  # 'intel', 'amd', 'arm'
    microarchitecture='skylake'      # Architecture name
)
```

### Cache Configuration

Each cache level needs a `CacheConfig`:

```python
CacheConfig(
    size=32*1024,        # Size in bytes
    latency=4,           # Access latency in cycles
    bandwidth=3200,      # Bandwidth in GB/s
    associativity=8,     # Cache associativity
    line_size=64,        # Cache line size (default: 64)
    is_shared=False,     # Shared among cores?
    is_inclusive=True    # Inclusive cache?
)
```

### Example: Adding AMD EPYC 9654

```python
'amd_epyc_9654': {
    'base_params': {
        'unit': None,
        'frequency': 2.4e9,
        'bits': 'bf16',
        'compute_efficiency': 0.85,
        'memory_efficiency': 0.75,
    },
    'cpu_specific': CPUConfig(
        # Genoa has 96 cores per socket
        cores_per_socket=96,
        sockets=2,
        threads_per_core=2,
        
        # Zen 4 cache hierarchy
        l1i_config=CacheConfig(
            size=32*1024, latency=4, bandwidth=3200,
            associativity=8, is_shared=False
        ),
        l1d_config=CacheConfig(
            size=32*1024, latency=4, bandwidth=3200,
            associativity=8, is_shared=False
        ),
        l2_config=CacheConfig(
            size=1024*1024, latency=14, bandwidth=1600,
            associativity=8, is_shared=False
        ),
        l3_config=CacheConfig(
            size=384*1024*1024, latency=50, bandwidth=900,
            associativity=16, is_shared=True
        ),
        
        # 12 CCDs, each is a NUMA node
        numa_nodes=12,
        cores_per_numa=16,
        numa_distance_matrix=np.ones((12, 12)) * 32 - np.eye(12) * 22,
        
        # Zen 4 supports AVX-512
        isa_support=['avx512', 'avx2', 'sse'],
        
        base_frequency=2.4e9,
        turbo_frequency_curve={
            1: 3.7e9, 16: 3.5e9, 32: 3.3e9, 
            64: 3.1e9, 96: 2.9e9, 192: 2.7e9
        },
        avx_frequency_offset={'avx2': 0, 'avx512': -100e6},
        
        # DDR5 support
        dram_bandwidth_per_channel=38.4,  # DDR5-4800
        memory_channels_per_socket=12,
        
        vendor='amd',
        microarchitecture='zen4'
    )
}
```

## Step 2: Test Your Configuration

Create a test script to verify your CPU configuration:

```python
from GenZ.cpu import create_cpu_system, cpu_aware_prefill_moddeling

# Test the new CPU
cpu = create_cpu_system('amd_epyc_9654')
print(f"Created CPU: {cpu.config.vendor} {cpu.config.microarchitecture}")
print(f"Total cores: {cpu.config.cores_per_socket * cpu.config.sockets}")
print(f"Peak memory bandwidth: {cpu.peak_memory_bandwidth} GB/s")

# Run a simple benchmark
result = cpu_aware_prefill_moddeling(
    model='gpt2',
    batch_size=1,
    input_tokens=512,
    system_name='amd_epyc_9654'
)
print(f"Prefill latency: {result['Latency']:.2f} ms")
```

## Step 3: Validate Parameters

### Where to Find CPU Specifications

1. **Official Documentation**
   - Intel: [Intel ARK](https://ark.intel.com)
   - AMD: [AMD Product Specifications](https://www.amd.com/en/products/specifications)
   - ARM: Vendor-specific (e.g., AWS for Graviton)

2. **Key Specifications to Look For**
   - Core count and socket configuration
   - Cache sizes and hierarchy
   - Memory channels and supported speeds
   - Base and turbo frequencies
   - Supported instruction sets

3. **Performance Characteristics**
   - Cache latencies: Can be found in architecture manuals
   - Memory bandwidth: Calculate from channels × speed × 8 bytes
   - NUMA topology: Check `lscpu` output on actual systems

### Common ISA Values

**Intel x86:**
- `'sse'` - SSE4.2 (baseline)
- `'avx2'` - Advanced Vector Extensions 2
- `'avx512'` - AVX-512 (various subsets)
- `'amx'` - Advanced Matrix Extensions

**AMD x86:**
- `'sse'` - SSE4.2
- `'avx2'` - AVX2
- `'avx512'` - AVX-512 (Zen 4+)

**ARM:**
- `'neon'` - ARM NEON
- `'sve'` - Scalable Vector Extension
- `'sve2'` - SVE2

## Step 4: Advanced Configuration

### Custom NUMA Topology

For complex NUMA systems:

```python
# 4-socket system with mesh interconnect
numa_distance_matrix=np.array([
    [10, 20, 25, 30],  # Node 0 distances
    [20, 10, 30, 25],  # Node 1 distances
    [25, 30, 10, 20],  # Node 2 distances
    [30, 25, 20, 10],  # Node 3 distances
])
```

### Heterogeneous Cores

For CPUs with P-cores and E-cores (not yet fully supported):

```python
# Future enhancement example
'intel_core_i9_13900k': {
    'cpu_specific': CPUConfig(
        cores_per_socket=24,  # 8 P-cores + 16 E-cores
        # Note: Current implementation assumes homogeneous cores
        # P/E core distinction would require additional fields
    )
}
```

## Step 5: Contributing

If adding a commonly used CPU:

1. Ensure specifications are accurate
2. Add comprehensive comments
3. Test with multiple workloads
4. Submit with benchmark results

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to import numpy as np at the top of cpu_configs.py
2. **Invalid ISA**: Check that ISA names match those in isa_model.py
3. **Performance Issues**: Verify cache sizes and latencies are in correct units

### Debugging

Enable debug output:
```python
result = cpu_aware_prefill_moddeling(
    model='gpt2',
    system_name='your_new_cpu',
    debug=True  # Shows detailed information
)
```

## Examples of Different CPU Types

### High-Performance Server CPU
See: `intel_xeon_8380`, `amd_epyc_7763`

### ARM Server CPU
See: `aws_graviton3`

### Desktop/Workstation CPU
Add with fewer cores but higher frequencies

### Embedded/Edge CPU
Add with smaller caches and lower power

## Future Enhancements

The CPU module is designed to be extensible. Future additions might include:
- Heterogeneous core support (P/E cores)
- Power consumption modeling
- Thermal throttling dynamics
- Custom cache replacement policies
- Prefetcher modeling 