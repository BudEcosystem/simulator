#!/usr/bin/env python3
"""
Example of using the device matching functionality.

This demonstrates how to match cluster-reported device information
to hardware configurations and retrieve performance specifications.
"""

from llm_memory_calculator.hardware import HardwareManager


def main():
    """Demonstrate device matching functionality."""
    
    # Initialize hardware manager
    manager = HardwareManager()
    
    print("=" * 60)
    print("Device Matching Example")
    print("=" * 60)
    
    # Example 1: Complete cluster information (best case)
    print("\n1. Complete cluster information (PCI ID + name + memory):")
    device_info = {
        "raw_name": "NVIDIA A100-SXM4-40GB",
        "pci_vendor": "10de",
        "pci_device": "20f1",
        "memory_mb": 40960,
        "gpu_count": 8
    }
    
    specs = manager.get_cluster_hardware_specs(device_info)
    if specs['matched']:
        print(f"   ✓ Matched: {specs['device_name']}")
        print(f"   - FP16 Performance: {specs['flops_fp16']} TFLOPS")
        print(f"   - Memory: {specs['memory_size_gb']} GB")
        print(f"   - Memory Bandwidth: {specs['memory_bandwidth_gbs']} GB/s")
        print(f"   - Interconnect: {specs['interconnect_bandwidth_gbs']} GB/s")
    else:
        print(f"   ✗ No match found for: {device_info['raw_name']}")
    
    # Example 2: Only device name and memory
    print("\n2. Device name and memory only:")
    device_info = {
        "raw_name": "Tesla A100 80GB",
        "memory_mb": 81920
    }
    
    specs = manager.get_cluster_hardware_specs(device_info)
    if specs['matched']:
        print(f"   ✓ Matched: {specs['device_name']}")
        print(f"   - FP16 Performance: {specs['flops_fp16']} TFLOPS")
        print(f"   - Memory: {specs['memory_size_gb']} GB")
    else:
        print(f"   ✗ No match found for: {device_info['raw_name']}")
    
    # Example 3: Only PCI IDs
    print("\n3. PCI IDs only:")
    device_info = {
        "pci_vendor": "10de",
        "pci_device": "2330",  # H100
    }
    
    specs = manager.get_cluster_hardware_specs(device_info)
    if specs['matched']:
        print(f"   ✓ Matched: {specs['device_name']}")
        print(f"   - FP16 Performance: {specs['flops_fp16']} TFLOPS")
        print(f"   - Memory: {specs['memory_size_gb']} GB")
    else:
        print(f"   ✗ No match found")
    
    # Example 4: Batch matching (GPUs and CPUs with specific SKUs)
    print("\n4. Batch device matching (Mixed GPUs and specific CPU SKUs):")
    devices = [
        {"raw_name": "NVIDIA H100 80GB HBM3", "memory_mb": 81920},
        {"raw_name": "AMD EPYC 9654", "memory_mb": 768000},  # 96-core Genoa
        {"raw_name": "Intel(R) Xeon(R) Platinum 8480+", "memory_mb": 512000},  # 56-core Sapphire Rapids
        {"raw_name": "AMD EPYC 7773X", "memory_mb": 524288},  # Milan-X with V-Cache
        {"raw_name": "Intel Xeon Platinum 8580", "memory_mb": 1024000},  # 56-core Emerald Rapids
        {"raw_name": "NVIDIA Grace CPU", "memory_mb": 524288},  # ARM CPU
        {"raw_name": "TPU v4", "memory_mb": 32768},
        {"raw_name": "AWS Graviton3", "memory_mb": 307200},  # ARM CPU
        {"raw_name": "Unknown Device XYZ", "memory_mb": 16384}
    ]
    
    print("   Processing batch of {} devices:".format(len(devices)))
    for device in devices:
        specs = manager.get_cluster_hardware_specs(device)
        status = "✓" if specs['matched'] else "✗"
        name = specs['device_name']
        if specs['matched']:
            # Get device type (GPU/CPU)
            config = manager.match_cluster_device(device)
            device_type = config.get('type', 'unknown').upper() if config else 'unknown'
            print(f"   {status} {device['raw_name']:35s} -> {name:40s} [{device_type}]")
        else:
            print(f"   {status} {device['raw_name']:35s} -> No match")
    
    # Example 5: Caching demonstration
    print("\n5. Caching demonstration:")
    device_info = {
        "raw_name": "NVIDIA A100-SXM4-40GB",
        "pci_vendor": "10de",
        "pci_device": "20f1",
        "memory_mb": 40960
    }
    
    # First call (populates cache)
    import time
    start = time.time()
    config1 = manager.match_cluster_device(device_info)
    time1 = time.time() - start
    
    # Second call (uses cache)
    start = time.time()
    config2 = manager.match_cluster_device(device_info)
    time2 = time.time() - start
    
    print(f"   First match: {time1*1000:.3f}ms")
    print(f"   Cached match: {time2*1000:.3f}ms")
    print(f"   Cache speedup: {time1/time2:.1f}x")
    print(f"   Same object: {config1 is config2}")
    
    # Clear cache
    manager.clear_match_cache()
    print("   Cache cleared")
    
    # Example 6: CPU SKU-specific vs generation matching
    print("\n6. CPU SKU-specific matching demonstration:")
    print("   Showing the difference between specific SKU vs generation-level matching:")
    
    cpu_examples = [
        {"raw_name": "Intel Xeon Platinum 8490H", "memory_mb": 307200, "description": "60-core Sapphire Rapids"},
        {"raw_name": "Intel Xeon Gold 6430", "memory_mb": 307200, "description": "32-core Sapphire Rapids"},
        {"raw_name": "AMD EPYC 9654", "memory_mb": 768000, "description": "96-core Genoa"},
        {"raw_name": "AMD EPYC 9454", "memory_mb": 512000, "description": "48-core Genoa"},
        {"raw_name": "AMD EPYC 7773X", "memory_mb": 524288, "description": "64-core Milan-X with V-Cache"},
    ]
    
    for cpu in cpu_examples:
        specs = manager.get_cluster_hardware_specs(cpu)
        if specs['matched']:
            config = manager.match_cluster_device(cpu)
            cores = config.get('cores', 'N/A')
            freq = config.get('base_freq_ghz', 'N/A')
            flops = specs['flops_fp16']
            print(f"   ✓ {cpu['raw_name']:30s} -> {cores:3d} cores @ {freq:3.1f} GHz = {flops:5.1f} TFLOPS")
        else:
            print(f"   ✗ {cpu['raw_name']:30s} -> No match")
    
    print("\n   Notice how different SKUs in the same generation have vastly different performance!")
    print("   This is why SKU-specific matching is crucial for accurate modeling.")
    
    print("\n" + "=" * 60)
    print("Integration with LLM calculations:")
    print("=" * 60)
    
    # Show how this integrates with LLM memory calculations
    device_info = {
        "raw_name": "NVIDIA A100-SXM4-40GB",
        "pci_vendor": "10de",
        "pci_device": "20f1",
        "memory_mb": 40960,
        "gpu_count": 8
    }
    
    specs = manager.get_cluster_hardware_specs(device_info)
    
    if specs['matched']:
        print(f"\nCluster reported: {device_info['raw_name']}")
        print(f"Matched config: {specs['device_name']}")
        print("\nPerformance specifications retrieved:")
        print(f"  - FP16 TFLOPS: {specs['flops_fp16']}")
        print(f"  - Memory: {specs['memory_size_gb']} GB")
        print(f"  - Memory BW: {specs['memory_bandwidth_gbs']} GB/s")
        print(f"  - Interconnect: {specs['interconnect_bandwidth_gbs']} GB/s")
        print(f"  - Real values: {specs['real_values']}")
        print("\nThese specs can now be used for:")
        print("  - LLM memory requirement calculations")
        print("  - Performance estimation (prefill/decode)")
        print("  - Hardware recommendation scoring")
        print("  - Distributed training planning")


if __name__ == "__main__":
    main()