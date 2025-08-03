"""
Example: Using CPU Systems with GenZ Prefill and Decode Methods
This example demonstrates three ways to use CPU systems with existing GenZ methods.
"""
import pandas as pd
from llm_memory_calculator.genz.cpu import (
    create_cpu_system, 
    cpu_aware_prefill_moddeling, 
    cpu_aware_decode_moddeling,
    enable_cpu_aware_inference,
    disable_cpu_aware_inference
)
from llm_memory_calculator.genz.LLM_inference import prefill_moddeling, decode_moddeling


def method1_direct_cpu_aware_functions():
    """
    Method 1: Use CPU-aware wrapper functions directly
    These functions automatically detect CPU systems and apply enhancements
    """
    print("=== Method 1: Direct CPU-Aware Functions ===")
    
    # Use string preset name - will be automatically converted to CPU system
    prefill_result = cpu_aware_prefill_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        system_name='intel_xeon_8380',  # CPU preset name
        debug=True
    )
    
    print(f"Prefill Latency: {prefill_result['Latency']:.2f} ms")
    print(f"Prefill Throughput: {prefill_result['Throughput']:.2f} tokens/s")
    
    # Decode with CPU system
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
    
    return prefill_result, decode_result


def method2_pass_cpu_system_object():
    """
    Method 2: Create CPU system object and pass it directly
    This gives you full control over CPU configuration
    """
    print("\n=== Method 2: Pass CPU System Object ===")
    
    # Create custom CPU system
    cpu_system = create_cpu_system('amd_epyc_7763')
    
    # Customize if needed
    cpu_system.config.smt_enabled = False  # Disable SMT for this run
    
    # Use with CPU-aware functions
    prefill_result = cpu_aware_prefill_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        system_name=cpu_system,  # Pass CPU system object
        debug=False
    )
    
    print(f"Prefill Latency: {prefill_result['Latency']:.2f} ms")
    print(f"Prefill Throughput: {prefill_result['Throughput']:.2f} tokens/s")
    
    # Check CPU-specific metrics in the model_df
    if 'ISA_used' in prefill_result['model_df'].columns:
        print(f"ISA Used: {prefill_result['model_df']['ISA_used'].mode()[0]}")
        print(f"Avg Thread Count: {prefill_result['model_df']['Thread_count'].mean():.0f}")
        print(f"Avg Parallel Efficiency: {prefill_result['model_df']['Parallel_efficiency'].mean():.1%}")
    
    return prefill_result


def method3_automatic_detection():
    """
    Method 3: Enable automatic CPU detection for all GenZ functions
    This patches the original functions to automatically use CPU enhancements
    """
    print("\n=== Method 3: Automatic CPU Detection ===")
    
    # Enable automatic CPU detection
    enable_cpu_aware_inference()
    
    # Now use original GenZ functions - they will automatically detect CPU systems
    prefill_result = prefill_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        system_name='aws_graviton3',  # CPU preset - will be auto-detected
        debug=False
    )
    
    print(f"Prefill Latency: {prefill_result['Latency']:.2f} ms")
    print(f"Prefill Throughput: {prefill_result['Throughput']:.2f} tokens/s")
    
    # Decode also works automatically
    decode_result = decode_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        output_tokens=128,
        system_name='aws_graviton3',
        debug=False
    )
    
    print(f"Decode Latency: {decode_result['Latency']:.2f} ms")
    print(f"Decode Throughput: {decode_result['Throughput']:.2f} tokens/s")
    
    # Disable automatic detection when done
    disable_cpu_aware_inference()
    
    return prefill_result, decode_result


def compare_cpu_vs_gpu():
    """
    Compare CPU vs GPU performance for the same model
    """
    print("\n=== CPU vs GPU Comparison ===")
    
    model_config = {
        'model': 'llama',
        'batch_size': 1,
        'input_tokens': 512,
        'output_tokens': 128,
        'debug': False
    }
    
    # CPU Performance
    cpu_prefill = cpu_aware_prefill_moddeling(
        **model_config,
        system_name='intel_xeon_8380'
    )
    
    cpu_decode = cpu_aware_decode_moddeling(
        **model_config,
        system_name='intel_xeon_8380'
    )
    
    # GPU Performance (using default A100)
    gpu_prefill = prefill_moddeling(
        **model_config,
        system_name='A100_40GB_GPU'
    )
    
    gpu_decode = decode_moddeling(
        **model_config,
        system_name='A100_40GB_GPU'
    )
    
    # Create comparison table
    comparison_data = [
        {
            'System': 'Intel Xeon 8380 (CPU)',
            'Prefill Latency (ms)': f"{cpu_prefill['Latency']:.2f}",
            'Prefill Throughput (tokens/s)': f"{cpu_prefill['Throughput']:.2f}",
            'Decode Latency (ms)': f"{cpu_decode['Latency']:.2f}",
            'Decode Throughput (tokens/s)': f"{cpu_decode['Throughput']:.2f}"
        },
        {
            'System': 'NVIDIA A100 (GPU)',
            'Prefill Latency (ms)': f"{gpu_prefill['Latency']:.2f}",
            'Prefill Throughput (tokens/s)': f"{gpu_prefill['Throughput']:.2f}",
            'Decode Latency (ms)': f"{gpu_decode['Latency']:.2f}",
            'Decode Throughput (tokens/s)': f"{gpu_decode['Throughput']:.2f}"
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Calculate speedup
    cpu_prefill_time = cpu_prefill['Latency']
    gpu_prefill_time = gpu_prefill['Latency']
    cpu_decode_time = cpu_decode['Latency']
    gpu_decode_time = gpu_decode['Latency']
    
    print(f"\nGPU Speedup:")
    print(f"Prefill: {cpu_prefill_time/gpu_prefill_time:.1f}x faster")
    print(f"Decode: {cpu_decode_time/gpu_decode_time:.1f}x faster")


def analyze_decode_scaling_with_kv_cache():
    """
    Analyze how decode performance scales with KV cache size on CPU
    """
    print("\n=== Decode Scaling with KV Cache Size ===")
    
    kv_sizes = [128, 256, 512, 1024, 2048]
    results = []
    
    for kv_size in kv_sizes:
        decode_result = cpu_aware_decode_moddeling(
            model='llama',
            batch_size=1,
            input_tokens=kv_size,  # KV cache size
            output_tokens=1,       # Generate 1 token
            system_name='intel_xeon_8380',
            debug=False
        )
        
        results.append({
            'KV Cache Size': kv_size,
            'Latency (ms)': f"{decode_result['Latency']:.2f}",
            'Throughput (tokens/s)': f"{decode_result['Throughput']:.2f}"
        })
    
    scaling_df = pd.DataFrame(results)
    print(scaling_df.to_string(index=False))


def main():
    print("CPU Integration with GenZ Prefill/Decode Methods")
    print("=" * 60)
    
    # Demonstrate different methods
    method1_direct_cpu_aware_functions()
    method2_pass_cpu_system_object()
    method3_automatic_detection()
    
    # Compare CPU vs GPU
    compare_cpu_vs_gpu()
    
    # Analyze decode scaling
    analyze_decode_scaling_with_kv_cache()


if __name__ == '__main__':
    main() 