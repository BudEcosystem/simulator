#!/usr/bin/env python3
"""
Example showing how to use cluster device information to estimate LLM performance.

This demonstrates the complete flow:
1. Get device info from cluster (simulated here)
2. Use HardwareManager to match and get hardware specs
3. Convert specs to the format needed for estimate_end_to_end_performance
4. Run performance estimation
"""

from llm_memory_calculator.hardware import HardwareManager
from llm_memory_calculator.performance_estimator import estimate_end_to_end_performance

def example_cluster_device_to_performance():
    """
    Example of using cluster device info to estimate LLM performance.
    """
    
    # Initialize the hardware manager
    manager = HardwareManager()
    
    # Example 1: NVIDIA GPU device info from cluster
    print("=" * 70)
    print("Example 1: NVIDIA A100 GPU from cluster")
    print("=" * 70)
    
    # Simulated device info that might come from a cluster
    gpu_device_info = {
        'raw_name': 'NVIDIA A100-SXM4-80GB',
        'vendor': 'NVIDIA Corporation',
        'device_id': '20b2',  # A100 80GB PCI ID
        'memory_mb': 81920,    # 80GB in MB
        'compute_capability': '8.0',
        'cuda_cores': 6912,
        'driver_version': '535.129.03'
    }
    
    # Get hardware specs using the manager
    gpu_specs = manager.get_cluster_hardware_specs(gpu_device_info)
    
    if gpu_specs['matched']:
        print(f"✓ Matched device: {gpu_specs['device_name']}")
        print(f"  - FP16 Performance: {gpu_specs['flops_fp16']} TFLOPS")
        print(f"  - Memory: {gpu_specs['memory_size_gb']} GB")
        print(f"  - Memory BW: {gpu_specs['memory_bandwidth_gbs']} GB/s")
        
        # Convert to format needed for estimate_end_to_end_performance
        gpu_config = {
            'Flops': gpu_specs['flops_fp16'],
            'Memory_size': gpu_specs['memory_size_gb'],
            'Memory_BW': gpu_specs['memory_bandwidth_gbs'],
            'ICN': gpu_specs['interconnect_bandwidth_gbs'],
            'real_values': gpu_specs.get('real_values', True),
            # Note: The matched config should have 'type' field if it's from our database
        }
        
        # Get the full config to check if it has type field
        full_config = manager.get_hardware_config(gpu_specs['device_name'])
        if full_config and 'type' in full_config:
            gpu_config['type'] = full_config['type']
        print(f"\nRunning performance estimation for Llama2-7B...")
        
        # Run performance estimation
        result = estimate_end_to_end_performance(
            model='llama2_7b',
            batch_size=1,
            input_tokens=512,
            output_tokens=128,
            system_name=gpu_config,
            bits='bf16',
            debug=False
        )
        
        print(f"✓ Performance Estimation Results:")
        print(f"  - Time to First Token: {result['ttft']:.2f} ms")
        print(f"  - Time per Output Token: {result['average_tpot']:.2f} ms")
        print(f"  - Total Latency: {result['total_latency']:.2f} ms")
        print(f"  - Throughput: {result['total_throughput']:.2f} tokens/s")
    else:
        print(f"✗ Could not match device: {gpu_device_info['raw_name']}")
    
    # Example 2: Intel CPU device info from cluster
    print("\n" + "=" * 70)
    print("Example 2: Intel Xeon CPU from cluster")
    print("=" * 70)
    
    # Simulated CPU device info from cluster
    cpu_device_info = {
        'raw_name': 'Intel(R) Xeon(R) Platinum 8490H CPU @ 1.90GHz',
        'vendor': 'GenuineIntel',
        'cpu_family': '6',
        'model': '143',
        'stepping': '8',
        'cores': 60,
        'threads': 120,
        'base_freq_mhz': 1900,
        'max_freq_mhz': 3500,
        'cache_l3_kb': 112500
    }
    
    # Get hardware specs using the manager
    cpu_specs = manager.get_cluster_hardware_specs(cpu_device_info)
    
    if cpu_specs['matched']:
        print(f"✓ Matched device: {cpu_specs['device_name']}")
        print(f"  - FP16 Performance: {cpu_specs['flops_fp16']} TFLOPS")
        print(f"  - Memory: {cpu_specs['memory_size_gb']} GB")
        print(f"  - Memory BW: {cpu_specs['memory_bandwidth_gbs']} GB/s")
        
        # Convert to format for estimate_end_to_end_performance
        cpu_config = {
            'Flops': cpu_specs['flops_fp16'],
            'Memory_size': cpu_specs['memory_size_gb'],
            'Memory_BW': cpu_specs['memory_bandwidth_gbs'],
            'ICN': cpu_specs['interconnect_bandwidth_gbs'],
            'real_values': cpu_specs.get('real_values', True),
            'type': 'cpu'  # Important: mark as CPU
        }
        
        print(f"\nRunning performance estimation for GPT-2...")
        
        # Run performance estimation for CPU
        result = estimate_end_to_end_performance(
            model='gpt2',  # Smaller model for CPU
            batch_size=1,
            input_tokens=128,
            output_tokens=32,
            system_name=cpu_config,
            bits='bf16',
            debug=False
        )
        
        print(f"✓ Performance Estimation Results:")
        print(f"  - Time to First Token: {result['ttft']:.2f} ms")
        print(f"  - Time per Output Token: {result['average_tpot']:.2f} ms")
        print(f"  - Total Latency: {result['total_latency']:.2f} ms")
        print(f"  - Throughput: {result['total_throughput']:.2f} tokens/s")
    else:
        print(f"✗ Could not match device: {cpu_device_info['raw_name']}")
        print(f"  Using unmatched device specs (may be less accurate):")
        print(f"  - Memory: {cpu_specs['memory_size_gb']} GB")
    
    # Example 3: Direct approach - get config by name and use it
    print("\n" + "=" * 70)
    print("Example 3: Direct approach using device name")
    print("=" * 70)
    
    # If you know the device name directly
    device_name = 'H100_GPU'
    config = manager.get_hardware_config(device_name)
    
    if config:
        print(f"✓ Retrieved config for: {config.get('name', device_name)}")
        print(f"  - Type: {config.get('type', 'gpu')}")
        print(f"  - Performance: {config.get('Flops')} TFLOPS")
        
        # Use directly with estimate_end_to_end_performance
        try:
            result = estimate_end_to_end_performance(
                model='llama2_13b',  # Use 13B model for H100
                batch_size=1,
                input_tokens=2048,
                output_tokens=256,
                system_name=config,  # Use config directly
                bits='bf16',
                debug=False
            )
            
            print(f"\n✓ Performance Estimation Results for Llama2-13B on H100:")
            print(f"  - Time to First Token: {result['ttft']:.2f} ms")
            print(f"  - Time per Output Token: {result['average_tpot']:.2f} ms")
            print(f"  - Total Latency: {result['total_latency']:.2f} ms")
            print(f"  - Throughput: {result['total_throughput']:.2f} tokens/s")
        except ValueError as e:
            if "would not fit" in str(e):
                print(f"\n⚠ Model too large for device memory. Try with smaller model or more GPUs.")
            else:
                print(f"\n✗ Error: {str(e)[:100]}...")
    else:
        print(f"✗ Could not find config for: {device_name}")

def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("CLUSTER DEVICE TO PERFORMANCE ESTIMATION EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("1. Getting device info from cluster (simulated)")
    print("2. Using HardwareManager to match devices and get specs")
    print("3. Converting specs to config format")
    print("4. Running performance estimation with the config")
    print()
    
    example_cluster_device_to_performance()
    
    print("\n" + "=" * 70)
    print("KEY POINTS:")
    print("=" * 70)
    print("• HardwareManager.get_cluster_hardware_specs() matches cluster devices")
    print("• The returned specs need to be converted to the right format")
    print("• For CPU devices, set 'type': 'cpu' in the config")
    print("• For GPU devices, 'type' is optional (defaults to 'gpu')")
    print("• estimate_end_to_end_performance accepts the config dict directly")

if __name__ == "__main__":
    main()