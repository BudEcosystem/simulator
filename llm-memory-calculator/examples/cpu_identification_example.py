#!/usr/bin/env python3
"""
Example showing proper CPU identification methods.

CPUs should be identified using CPU-specific identifiers rather than
just name and memory, as memory is system RAM, not part of the CPU.
"""

from llm_memory_calculator.hardware import HardwareManager


def main():
    """Demonstrate proper CPU identification."""
    
    manager = HardwareManager()
    
    print("=" * 70)
    print("CPU Identification Best Practices")
    print("=" * 70)
    
    # Method 1: Using CPU Family and Model (BEST for CPUs)
    print("\n1. CPU Family/Model Identification (Most Reliable):")
    print("   This is how CPUs are actually identified in systems")
    
    cpu_info = {
        "cpu_family": "6",      # Intel Family 6
        "cpu_model": "143",     # Model 143 = Sapphire Rapids
        "cpu_stepping": "4",    # Optional stepping
        "raw_name": "Intel(R) Xeon(R) Platinum 8480+",
        # Note: memory_mb is system RAM, not CPU property
        "memory_mb": 1048576    # 1TB system RAM
    }
    
    specs = manager.get_cluster_hardware_specs(cpu_info)
    if specs['matched']:
        print(f"   ✓ Matched by CPU ID: {specs['device_name']}")
        print(f"     - CPU Family: {cpu_info['cpu_family']}")
        print(f"     - CPU Model: {cpu_info['cpu_model']}")
        print(f"     - Performance: {specs['flops_fp16']} TFLOPS")
    
    # Method 2: Using just the name (FALLBACK)
    print("\n2. Name-based Identification (Fallback Method):")
    print("   Less reliable, but works when CPU IDs aren't available")
    
    cpu_info_name_only = {
        "raw_name": "AMD EPYC 9654",
        "memory_mb": 768000  # System RAM
    }
    
    specs = manager.get_cluster_hardware_specs(cpu_info_name_only)
    if specs['matched']:
        print(f"   ✓ Matched by name: {specs['device_name']}")
        print(f"     - Input: {cpu_info_name_only['raw_name']}")
        print(f"     - Performance: {specs['flops_fp16']} TFLOPS")
    
    print("\n" + "=" * 70)
    print("How CPUs are Identified in Different Systems")
    print("=" * 70)
    
    # Linux /proc/cpuinfo example
    print("\n3. Linux /proc/cpuinfo Format:")
    linux_cpu = {
        "raw_name": "Intel(R) Xeon(R) Platinum 8480+",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "4",
        "cpu_cores": "56",      # Number of cores
        "cache_size": "105MB",  # L3 cache
        # System memory, not CPU property
        "memory_mb": 1048576
    }
    print(f"   CPU: {linux_cpu['raw_name']}")
    print(f"   Family: {linux_cpu['cpu_family']}, Model: {linux_cpu['cpu_model']}")
    print(f"   Cores: {linux_cpu['cpu_cores']}, Cache: {linux_cpu['cache_size']}")
    
    # Kubernetes node info example
    print("\n4. Kubernetes Node Labels:")
    k8s_labels = {
        "feature.node.kubernetes.io/cpu-cpuid.AESNI": "true",
        "feature.node.kubernetes.io/cpu-cpuid.AVX512": "true",
        "feature.node.kubernetes.io/cpu-model.family": "6",
        "feature.node.kubernetes.io/cpu-model.id": "143",
        "feature.node.kubernetes.io/cpu-model.vendor_id": "GenuineIntel",
        "feature.node.kubernetes.io/cpu-hardware_multithreading": "true",
        "feature.node.kubernetes.io/memory-numa.count": "2",
        "feature.node.kubernetes.io/system-os_release.ID": "ubuntu"
    }
    
    # Extract CPU info from K8s labels
    cpu_from_k8s = {
        "cpu_family": k8s_labels.get("feature.node.kubernetes.io/cpu-model.family"),
        "cpu_model": k8s_labels.get("feature.node.kubernetes.io/cpu-model.id"),
        "vendor": k8s_labels.get("feature.node.kubernetes.io/cpu-model.vendor_id")
    }
    print(f"   Extracted: Family={cpu_from_k8s['cpu_family']}, Model={cpu_from_k8s['cpu_model']}")
    
    # SLURM example
    print("\n5. SLURM/HPC Cluster Info:")
    slurm_info = {
        "CPUSpecList": "Intel(R) Xeon(R) Platinum 8480+",
        "CPUTot": "112",  # Total CPUs (2 sockets x 56 cores)
        "RealMemory": "1031842",  # MB of RAM
        "Sockets": "2",
        "CoresPerSocket": "56",
        "ThreadsPerCore": "2"
    }
    print(f"   CPU: {slurm_info['CPUSpecList']}")
    print(f"   Configuration: {slurm_info['Sockets']} sockets × {slurm_info['CoresPerSocket']} cores")
    
    print("\n" + "=" * 70)
    print("GPU vs CPU Identification Comparison")
    print("=" * 70)
    
    # GPU identification
    print("\n6. GPU Identification (uses PCI IDs):")
    gpu_info = {
        "raw_name": "NVIDIA A100-SXM4-40GB",
        "pci_vendor": "10de",  # NVIDIA vendor ID
        "pci_device": "20f1",   # A100 device ID
        "memory_mb": 40960      # GPU memory (part of device)
    }
    
    gpu_specs = manager.get_cluster_hardware_specs(gpu_info)
    if gpu_specs['matched']:
        print(f"   ✓ GPU matched by PCI ID: {gpu_specs['device_name']}")
        print(f"     - PCI: {gpu_info['pci_vendor']}:{gpu_info['pci_device']}")
        print(f"     - Memory: {gpu_specs['memory_size_gb']} GB (GPU memory)")
    
    # CPU identification
    print("\n7. CPU Identification (uses CPU family/model):")
    cpu_info = {
        "raw_name": "Intel(R) Xeon(R) Platinum 8480+",
        "cpu_family": "6",
        "cpu_model": "143",
        "memory_mb": 1048576  # System RAM, not CPU property
    }
    
    cpu_specs = manager.get_cluster_hardware_specs(cpu_info)
    if cpu_specs['matched']:
        print(f"   ✓ CPU matched by family/model: {cpu_specs['device_name']}")
        print(f"     - CPU ID: Family {cpu_info['cpu_family']}, Model {cpu_info['cpu_model']}")
        print(f"     - Memory: {cpu_info['memory_mb']//1024} GB (System RAM, not CPU memory)")
    
    print("\n" + "=" * 70)
    print("Key Differences:")
    print("=" * 70)
    print("\nGPUs:")
    print("  - Identified by PCI vendor:device IDs")
    print("  - Memory is part of the GPU device")
    print("  - Clear memory variants (40GB vs 80GB)")
    
    print("\nCPUs:")
    print("  - Identified by CPU family and model numbers")
    print("  - Memory is system RAM, not part of CPU")
    print("  - Same CPU can be used with different RAM amounts")
    print("  - CPU cache is the actual CPU memory (L1/L2/L3)")


if __name__ == "__main__":
    main()