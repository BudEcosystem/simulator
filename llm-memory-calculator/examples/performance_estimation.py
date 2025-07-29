#!/usr/bin/env python3
"""
Example demonstrating LLM performance estimation capabilities.

This example shows how to estimate detailed performance metrics for LLM inference
including prefill, decode, and end-to-end performance across different configurations.
"""

from llm_memory_calculator import (
    estimate_prefill_performance,
    estimate_decode_performance,
    estimate_end_to_end_performance,
    compare_performance_configurations,
    get_hardware_config,
    HARDWARE_CONFIGS
)


def print_performance_summary(result, title):
    """Helper function to print performance results in a readable format."""
    print(f"\n📊 {title}")
    print("-" * 50)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print key metrics
    if 'Latency' in result:
        print(f"⏱️  Latency: {result['Latency']:.2f} ms")
    if 'Throughput' in result:
        print(f"🚀 Throughput: {result['Throughput']:.1f} tokens/s")
    if 'total_latency' in result:
        print(f"⏱️  Total Latency: {result['total_latency']:.2f} ms")
    if 'ttft' in result:
        print(f"🎯 Time to First Token: {result['ttft']:.2f} ms")
    if 'average_tpot' in result:
        print(f"⚡ Time per Output Token: {result['average_tpot']:.2f} ms")
    if 'total_throughput' in result:
        print(f"🚀 Total Throughput: {result['total_throughput']:.1f} tokens/s")
    
    # Print memory info if available
    if 'Memory_used' in result:
        print(f"💾 Memory Used: {result.get('Memory_used', 'N/A')}")


def main():
    print("🧮 LLM Performance Estimation Examples")
    print("=" * 60)
    
    # Hardware configuration
    hardware = get_hardware_config('A100_80GB')
    print(f"\n🖥️  Using Hardware: A100 80GB")
    print(f"   • Memory: {hardware['Memory_size']} GB")
    print(f"   • Compute: {hardware['Flops']} TFLOPS") 
    print(f"   • Memory BW: {hardware['Memory_BW']} GB/s")
    
    # Example 1: Prefill Performance
    print("\n" + "="*60)
    print("Example 1: Prefill Phase Performance")
    print("="*60)
    
    prefill_result = estimate_prefill_performance(
        model='llama2_7b',
        batch_size=8,
        input_tokens=2048,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        pipeline_parallel=1
    )
    
    print_performance_summary(prefill_result, "Prefill Performance (Llama2-7B, batch=8, TP=4)")
    
    # Example 2: Decode Performance  
    print("\n" + "="*60)
    print("Example 2: Decode Phase Performance")
    print("="*60)
    
    decode_result = estimate_decode_performance(
        model='llama2_7b',
        batch_size=8,
        beam_size=1,
        input_tokens=2048,
        output_tokens=256,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        pipeline_parallel=1
    )
    
    print_performance_summary(decode_result, "Decode Performance (Llama2-7B, batch=8, TP=4)")
    
    # Example 3: End-to-End Performance
    print("\n" + "="*60)
    print("Example 3: End-to-End Performance")
    print("="*60)
    
    e2e_result = estimate_end_to_end_performance(
        model='llama2_7b',
        batch_size=8,
        beam_size=1,
        input_tokens=2048,
        output_tokens=256,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        pipeline_parallel=1
    )
    
    print_performance_summary(e2e_result, "End-to-End Performance (Llama2-7B, batch=8, TP=4)")
    
    # Show detailed breakdown
    if 'prefill' in e2e_result and 'decode' in e2e_result:
        prefill = e2e_result['prefill']
        decode = e2e_result['decode']
        
        print(f"\n📋 Detailed Breakdown:")
        print(f"   • Prefill: {prefill.get('Latency', 0):.2f} ms")
        print(f"   • Decode: {decode.get('Total_latency', 0):.2f} ms")
        print(f"   • Total: {e2e_result.get('total_latency', 0):.2f} ms")
    
    # Example 4: Compare Hardware Configurations
    print("\n" + "="*60)
    print("Example 4: Hardware Performance Comparison")
    print("="*60)
    
    configurations = [
        {
            'name': 'A100_80GB_TP4',
            'system_name': get_hardware_config('A100_80GB'),
            'tensor_parallel': 4,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        },
        {
            'name': 'H100_80GB_TP4',
            'system_name': get_hardware_config('H100_80GB'),
            'tensor_parallel': 4,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        },
        {
            'name': 'MI300X_TP8',
            'system_name': get_hardware_config('MI300X'),
            'tensor_parallel': 8,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        }
    ]
    
    print("⚖️  Comparing hardware configurations for Llama2-7B:")
    print("   • Input: 2048 tokens, Output: 256 tokens, Batch: 4")
    
    comparison_results = compare_performance_configurations(
        model='llama2_7b',
        configurations=configurations,
        batch_size=4,
        input_tokens=2048,
        output_tokens=256
    )
    
    print(f"\n🏆 Performance Ranking:")
    for i, result in enumerate(comparison_results, 1):
        config_name = result['config_name']
        throughput = result.get('total_throughput', 0)
        ttft = result.get('ttft', 0)
        tpot = result.get('tpot', 0)
        
        if 'error' in result:
            print(f"   {i}. {config_name}: ❌ {result['error']}")
        else:
            print(f"   {i}. {config_name}")
            print(f"      • Throughput: {throughput:.1f} tokens/s")
            print(f"      • TTFT: {ttft:.1f} ms")
            print(f"      • TPOT: {tpot:.1f} ms")
    
    # Example 5: Precision Impact on Performance
    print("\n" + "="*60)
    print("Example 5: Precision Impact on Performance")
    print("="*60)
    
    print("🎚️  Comparing different precisions for Llama2-7B:")
    
    for precision in ['bf16', 'int8']:
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
            
            throughput = result.get('total_throughput', 0)
            ttft = result.get('ttft', 0)
            tpot = result.get('average_tpot', 0)
            
            print(f"\n   {precision.upper()}:")
            print(f"   • Throughput: {throughput:.1f} tokens/s")
            print(f"   • TTFT: {ttft:.1f} ms")
            print(f"   • TPOT: {tpot:.1f} ms")
            
        except Exception as e:
            print(f"\n   {precision.upper()}: ❌ {str(e)[:50]}...")
    
    # Example 6: Scaling Analysis
    print("\n" + "="*60)
    print("Example 6: Batch Size Scaling Analysis")
    print("="*60)
    
    print("📈 Analyzing performance scaling with batch size:")
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        try:
            result = estimate_decode_performance(
                model='llama2_7b',
                batch_size=batch_size,
                input_tokens=2048,
                output_tokens=1,  # Single token for TPOT measurement
                system_name=hardware,
                bits='bf16',
                tensor_parallel=4
            )
            
            throughput = result.get('Throughput', 0)
            latency = result.get('Latency', 0)
            
            print(f"   Batch {batch_size:2d}: {throughput:6.1f} tokens/s, {latency:5.1f} ms/token")
            
        except Exception as e:
            print(f"   Batch {batch_size:2d}: ❌ Error")
    
    print("\n✅ Performance estimation examples completed!")
    print("🎯 Use these functions to optimize your LLM deployment configuration.")


if __name__ == "__main__":
    main()