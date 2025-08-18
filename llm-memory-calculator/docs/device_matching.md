# Device Matching Documentation

## Overview

The device matching functionality enables automatic matching of cluster-reported device information to known hardware configurations. This is essential for integrating with cluster management systems that report device information via tools like NFD (Node Feature Discovery) or similar node-info systems.

## Problem Statement

Cluster management systems typically report:
- Device names (e.g., "NVIDIA A100-SXM4-40GB")
- PCI vendor/device IDs (e.g., "10de:20f1")
- Memory sizes in MB
- CUDA versions and driver info

However, they don't provide the performance specifications needed for LLM calculations:
- FLOPS (floating-point operations per second)
- Memory bandwidth
- Interconnect bandwidth
- Power consumption

The device matching system bridges this gap by matching cluster-reported devices to our hardware configuration database.

## Architecture

### Components

1. **DeviceParser**: Parses raw device information into structured identity
2. **DeviceMatcher**: Implements matching strategies (PCI ID, CPU ID, name, fuzzy)
3. **HardwareManager**: Integrates matching with hardware configuration management
4. **CPU Specs Module**: Specific SKU configurations with accurate FLOPS calculations
5. **Caching Layer**: Improves performance for repeated lookups

### Matching Strategies

The system uses a hierarchical matching approach in order of reliability:

#### For GPUs:
1. **PCI ID Matching** (Most Reliable)
   - Uses PCI vendor and device IDs
   - Combined with memory size for variant selection
   - Example: `10de:20f1` + 40GB → A100_40GB_GPU

2. **Model Name Matching** (Reliable)
   - Parses model from device name
   - Uses memory size for variant selection
   - Example: "Tesla A100 40GB" → A100_40GB_GPU

#### For CPUs:
1. **CPU Family/Model ID Matching** (Most Reliable)
   - Uses CPU family and model identifiers from CPUID
   - Example: Family 6, Model 143 → Sapphire Rapids generation
   - Combined with SKU number for specific model

2. **CPU SKU Model Number Matching** (Reliable)
   - Extracts specific model number from device name
   - Example: "Intel Xeon Platinum 8480+" → XEON_PLATINUM_8480PLUS

3. **Generation Name Matching** (Backward Compatible)
   - Matches generation names for compatibility
   - Example: "Sapphire Rapids" → SAPPHIRERAPIDS

#### For All Devices:
4. **Fuzzy Matching** (Fallback)
   - Token-based similarity scoring
   - Useful for non-standard naming
   - Example: "Hopper H100 GPU" → H100_GPU

## API Usage

### Basic Usage

```python
from llm_memory_calculator.hardware import HardwareManager

# Initialize manager
manager = HardwareManager()

# GPU device information from cluster
gpu_device_info = {
    "raw_name": "NVIDIA A100-SXM4-40GB",
    "pci_vendor": "10de",
    "pci_device": "20f1",
    "memory_mb": 40960,
    "gpu_count": 8
}

# CPU device information from cluster
cpu_device_info = {
    "raw_name": "Intel(R) Xeon(R) Platinum 8480+",
    "memory_mb": 512000,  # 500GB system RAM
    "cpu_family": "6",    # Optional: from CPUID
    "cpu_model": "143"     # Optional: from CPUID
}

# Get hardware specifications for GPU
specs = manager.get_cluster_hardware_specs(gpu_device_info)

if specs['matched']:
    print(f"GPU Device: {specs['device_name']}")
    print(f"FP16 Performance: {specs['flops_fp16']} TFLOPS")
    print(f"Memory: {specs['memory_size_gb']} GB")
    print(f"Memory Bandwidth: {specs['memory_bandwidth_gbs']} GB/s")

# Get hardware specifications for CPU
specs = manager.get_cluster_hardware_specs(cpu_device_info)

if specs['matched']:
    print(f"CPU Device: {specs['device_name']}")
    print(f"FP16 Performance: {specs['flops_fp16']} TFLOPS")
    print(f"Memory: {specs['memory_size_gb']} GB (max supported)")
    print(f"Memory Bandwidth: {specs['memory_bandwidth_gbs']} GB/s")
```

### Batch Processing

```python
# Process multiple devices (mixed GPUs and CPUs)
devices = [
    {"raw_name": "NVIDIA A100-SXM4-40GB", "memory_mb": 40960},
    {"raw_name": "NVIDIA H100 80GB", "memory_mb": 81920},
    {"raw_name": "Intel Xeon Platinum 8480+", "memory_mb": 512000},
    {"raw_name": "AMD EPYC 9654", "memory_mb": 768000},
    {"raw_name": "TPU v4", "memory_mb": 32768}
]

configs = manager.match_cluster_devices_batch(devices)

for device, config in zip(devices, configs):
    if config:
        print(f"{device['raw_name']} → {config['name']}")
```

### Direct Matching

```python
# Match a single device
config = manager.match_cluster_device(device_info)

if config:
    print(f"Matched: {config['name']}")
    print(f"FLOPS: {config['Flops']} TFLOPS")
```

## Input Format

The device matching system accepts device information in the following format:

```python
# For GPU devices
gpu_device_info = {
    # Device name as reported by cluster (recommended)
    "raw_name": "NVIDIA A100-SXM4-40GB",
    
    # PCI vendor ID in hex (recommended for GPUs)
    "pci_vendor": "10de",  # or "0x10de"
    
    # PCI device ID in hex (recommended for GPUs)
    "pci_device": "20f1",  # or "0x20f1"
    
    # Memory size in MB (recommended)
    "memory_mb": 40960,
    
    # Number of devices (optional, for context)
    "gpu_count": 8,
    
    # Additional fields are ignored but preserved
    "cuda_version": "11.7",
    "driver_version": "515.65.01"
}

# For CPU devices
cpu_device_info = {
    # Device name as reported by cluster (required)
    "raw_name": "Intel(R) Xeon(R) Platinum 8480+",
    
    # System RAM in MB (recommended)
    "memory_mb": 512000,  # 500GB
    
    # CPU identification from CPUID (optional but more reliable)
    "cpu_family": "6",    # Intel family 6
    "cpu_model": "143",   # Sapphire Rapids model
    "cpu_stepping": "8",  # Optional stepping
    
    # Additional CPU context (optional)
    "cores_per_socket": 56,
    "sockets": 2,
    "threads_per_core": 2
}
```

## Output Format

The `get_cluster_hardware_specs()` method returns:

```python
# For GPU devices
{
    # Matched configuration name
    'device_name': 'A100_40GB_GPU',
    
    # Performance specifications
    'flops_fp16': 312,  # TFLOPS
    'memory_size_gb': 40,  # Device memory
    'memory_bandwidth_gbs': 1600,
    'interconnect_bandwidth_gbs': 150,
    
    # Metadata
    'real_values': True,  # Whether specs are real or estimated
    'matched': True  # Whether matching was successful
}

# For CPU devices
{
    # Matched configuration name
    'device_name': 'Intel Xeon Platinum 8480+',
    
    # Performance specifications
    'flops_fp16': 57.3,  # TFLOPS (specific to this SKU)
    'memory_size_gb': 4096,  # Max supported system RAM
    'memory_bandwidth_gbs': 307.2,  # DDR5 bandwidth
    'interconnect_bandwidth_gbs': 128,  # UPI bandwidth
    
    # Metadata
    'real_values': True,
    'matched': True
}
```

## Supported Devices

The system supports both GPUs and CPUs for comprehensive hardware matching:

### GPUs

#### NVIDIA GPUs
- A100 (40GB, 80GB variants)
- H100 (80GB)
- H200 (141GB)
- GH200 (144GB)
- V100 (16GB, 32GB variants)
- T4, L4, L40
- RTX 3090, RTX 4090, RTX A6000
- P40, P100

#### AMD GPUs
- MI100, MI210, MI250, MI250X
- MI300A, MI300X

#### Intel GPUs
- Max 1100, Max 1350, Max 1550 (Ponte Vecchio)
- Arc A770

### CPUs

The system now supports specific CPU SKUs rather than just generation-level matching. This provides much more accurate performance modeling since different SKUs within the same generation have vastly different specifications.

#### Intel Xeon CPUs (Specific SKUs)

**Sapphire Rapids (4th Gen)**
- Xeon Platinum 8490H (60 cores @ 1.9 GHz)
- Xeon Platinum 8480+ (56 cores @ 2.0 GHz)
- Xeon Platinum 8470 (52 cores @ 2.0 GHz)
- Xeon Platinum 8460Y+ (40 cores @ 2.2 GHz)
- Xeon Gold 6430 (32 cores @ 2.1 GHz)

**Emerald Rapids (5th Gen)**
- Xeon Platinum 8592+ (64 cores @ 1.9 GHz)
- Xeon Platinum 8580 (56 cores @ 2.0 GHz)

#### AMD EPYC CPUs (Specific SKUs)

**Milan/Milan-X (3rd Gen)**
- EPYC 7763 (64 cores @ 2.45 GHz)
- EPYC 7773X (64 cores @ 2.2 GHz, 3D V-Cache)
- EPYC 7713 (64 cores @ 2.0 GHz)

**Genoa/Genoa-X (4th Gen)**
- EPYC 9654 (96 cores @ 2.4 GHz)
- EPYC 9554 (64 cores @ 3.1 GHz)
- EPYC 9454 (48 cores @ 2.75 GHz)
- EPYC 9684X (96 cores @ 2.55 GHz, 3D V-Cache)

**Bergamo (Cloud-optimized)**
- EPYC 9754 (128 cores @ 2.25 GHz)
- EPYC 9734 (112 cores @ 2.2 GHz)

#### Generation Aliases (Backward Compatibility)

For backward compatibility, generation-level names are still supported:
- Sapphire Rapids, Emerald Rapids, Granite Rapids, Sierra Forest
- Milan, Milan-X, Genoa, Genoa-X, Bergamo, Turin

#### ARM-based CPUs
- NVIDIA Grace (72-core ARM)
- AWS Graviton3 / Graviton4
- Ampere Altra

### Other Accelerators

#### Google TPUs
- TPU v4, v5e, v5p, v6

#### Specialized Chips
- Apple M1, M2, M3 (Pro/Max/Ultra variants)
- Graphcore IPUs
- Cerebras WSE
- Habana Gaudi / Gaudi2

## Why SKU-Specific CPU Matching Matters\n\nThe transition from generation-level to SKU-specific CPU matching provides significantly more accurate performance modeling:\n\n### Performance Variations Within Same Generation\n\n**Example: AMD EPYC Genoa Family**\n- EPYC 9654: 96 cores @ 2.4 GHz = **73.7 TFLOPS**\n- EPYC 9454: 48 cores @ 2.75 GHz = **42.2 TFLOPS**\n- **Difference**: 2x cores, 1.75x FLOPS!\n\n**Example: Intel Sapphire Rapids Family**\n- Xeon Platinum 8490H: 60 cores @ 1.9 GHz = **58.9 TFLOPS**\n- Xeon Gold 6430: 32 cores @ 2.1 GHz = **32.8 TFLOPS**\n- **Difference**: 1.87x cores, 1.8x FLOPS\n\n### Benefits of SKU-Specific Matching\n\n1. **Accurate Performance Estimation**: Each SKU has precise core counts, frequencies, and cache configurations\n2. **Better Hardware Recommendations**: Recommend optimal hardware based on actual specifications\n3. **Cost/Performance Analysis**: Compare different SKUs within same generation for best value\n4. **Capacity Planning**: Accurate planning for LLM deployments based on real hardware capabilities\n\n### FLOPS Calculation Methodology\n\nFor CPUs, FLOPS are calculated as:\n```\nFLOPS = cores × base_frequency_ghz × 16 (ops/cycle) × sockets / 1000\n```\n\nWhere:\n- **16 ops/cycle**: Assumes AVX-512 FP32 instructions (32 FP16 ops/cycle)\n- **sockets**: Typically 2 for server configurations\n- **cores**: Physical cores per socket\n\nThis provides realistic peak theoretical performance for comparison.\n\n## Adding New Devices

### 1. Add to Hardware Configs

Edit `llm_memory_calculator/hardware/configs.py`:

```python
'NEW_DEVICE': {
    'name': 'NEW_DEVICE',
    'Flops': 500,  # TFLOPS
    'Memory_size': 48,  # GB
    'Memory_BW': 2000,  # GB/s
    'ICN': 300,  # GB/s interconnect
    'real_values': True,
    'type': 'gpu',
    'manufacturer': 'VENDOR',
    'pci_ids': ['abcd', 'ef01'],  # PCI device IDs
    'aliases': ['Device Name', 'Alternative Name']
}
```

### 2. Add PCI Mappings

Edit `device_matcher.py`:

```python
PCI_ID_MAP = {
    # ...existing mappings...
    'vendor_id:device_id': 'NEW_DEVICE',
}
```

### 3. Add Aliases

```python
MODEL_ALIASES = {
    # ...existing aliases...
    'NEW_DEVICE': ['NEW_DEVICE', 'DEVICE NAME', 'ALT NAME'],
}
```

## Performance Considerations

### Caching

The system implements two levels of caching:

1. **Hardware Cache**: Caches merged hardware configurations
2. **Match Cache**: Caches device matching results

```python
# Clear caches if needed
manager.clear_match_cache()
```

### Best Practices

1. **Provide Complete Information**: Include PCI IDs, device name, and memory size when available
2. **Batch Processing**: Use `match_cluster_devices_batch()` for multiple devices
3. **Cache Management**: Clear cache only when hardware configs change

## Integration Examples

### With Kubernetes/OpenShift

```python
# Example: Process node labels from Kubernetes
def process_k8s_node(node_labels):
    # For GPUs
    if "nvidia.com/gpu.product-id" in node_labels:
        device_info = {
            "raw_name": node_labels.get("feature.node.kubernetes.io/pci-10de.present"),
            "pci_vendor": "10de",
            "pci_device": node_labels.get("nvidia.com/gpu.product-id"),
            "memory_mb": int(node_labels.get("nvidia.com/gpu.memory", 0))
        }
    # For CPUs
    elif "feature.node.kubernetes.io/cpu-model.family" in node_labels:
        cpu_model = node_labels.get("feature.node.kubernetes.io/cpu-model.name", "")
        memory_kb = int(node_labels.get("feature.node.kubernetes.io/memory-numa.total", 0))
        device_info = {
            "raw_name": cpu_model,
            "memory_mb": memory_kb // 1024
        }
    
    return manager.get_cluster_hardware_specs(device_info)
```

### With SLURM

```python
# Example: Parse SLURM gres information
def parse_slurm_gres(gres_string):
    # Parse "gpu:a100:8" format
    parts = gres_string.split(':')
    if len(parts) >= 2 and parts[0] == 'gpu':
        device_info = {"raw_name": f"NVIDIA {parts[1].upper()}"}
        return manager.match_cluster_device(device_info)
```

## Troubleshooting

### Device Not Matching

1. **Check Input Format**: Ensure device_info keys are correct
2. **Verify PCI IDs**: Use lowercase hex without 0x prefix
3. **Check Memory Units**: Memory should be in MB, not GB
4. **Enable Debug Logging**: 
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Adding Custom Devices

For devices not in the default configuration:

```python
# Option 1: Add to database
hardware_data = {
    'name': 'CUSTOM_GPU',
    'type': 'gpu',
    'flops': 250,
    'memory_size': 32,
    'memory_bw': 900,
    # ... other specs
}
# Use BudHardware.add_hardware() if using database

# Option 2: Extend configs programmatically
from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
HARDWARE_CONFIGS['CUSTOM_GPU'] = hardware_data
```

## Testing

Run the test suite:

```bash
pytest tests/test_device_matcher.py -v
```

Run the example:

```bash
python examples/device_matching_example.py
```

## Future Enhancements

Planned improvements:

1. **Automatic PCI ID Discovery**: Query PCI database for new devices
2. **ML-Based Matching**: Use machine learning for better fuzzy matching
3. **Performance Estimation**: Estimate specs for unknown devices
4. **Cloud Instance Mapping**: Map cloud instance types to hardware
5. **Dynamic Updates**: Fetch latest hardware specs from online database