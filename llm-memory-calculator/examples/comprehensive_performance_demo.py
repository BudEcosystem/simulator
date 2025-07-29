#!/usr/bin/env python3
"""
Comprehensive Performance Estimation Demo

This example demonstrates how external repositories can use llm-memory-calculator
to get detailed performance analysis for LLM inference across different hardware
configurations and parallelism strategies.

Key features demonstrated:
- Prefill vs Decode performance analysis
- Hardware comparison across A100, H100, MI300X
- Precision impact (bf16 vs int8)
- Batch size scaling analysis
- Tensor parallelism optimization
- End-to-end performance estimation
"""

import time
from llm_memory_calculator import (
    # Performance estimation functions
    estimate_prefill_performance,
    estimate_decode_performance,
    estimate_end_to_end_performance,
    estimate_chunked_performance,
    compare_performance_configurations,
    
    # Hardware and parallelism functions
    get_hardware_config,
    get_best_parallelization_strategy,
    get_various_parallelization,
    HARDWARE_CONFIGS,
    
    # Memory calculation
    calculate_memory
)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print(f"{'='*60}")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'─'*40}")
    print(f"📊 {title}")
    print(f"{'─'*40}")


def format_performance_result(result, title):
    """Format performance results for display."""
    print(f"\n🎯 {title}")
    print("   " + "─" * 45)
    
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
        return
    
    # Key metrics
    if 'Latency' in result:
        print(f"   ⏱️  Latency: {result['Latency']:.2f} ms")
    if 'total_latency' in result:
        print(f"   ⏱️  Total Latency: {result['total_latency']:.2f} ms")
    if 'ttft' in result:
        print(f"   🎯 TTFT: {result['ttft']:.2f} ms")
    if 'average_tpot' in result:
        print(f"   ⚡ TPOT: {result['average_tpot']:.2f} ms")
    if 'total_throughput' in result:
        print(f"   🚀 Throughput: {result['total_throughput']:.1f} tokens/s")
    if 'Throughput' in result:
        print(f"   🚀 Throughput: {result['Throughput']:.1f} tokens/s")


def demo_basic_performance_analysis():
    """Demonstrate basic performance analysis capabilities."""
    print_section("Basic Performance Analysis")
    
    # Standard configuration
    hardware = get_hardware_config('A100_80GB')
    model = 'llama2_7b'
    batch_size = 8
    input_tokens = 2048
    output_tokens = 512
    
    print(f"\n🔧 Configuration:")
    print(f"   • Model: {model}")
    print(f"   • Hardware: A100 80GB ({hardware['Flops']} TFLOPS, {hardware['Memory_size']} GB)")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Input Tokens: {input_tokens}")
    print(f"   • Output Tokens: {output_tokens}")
    print(f"   • Tensor Parallel: 4")
    
    # Prefill analysis
    print_subsection("Prefill Phase Analysis")
    prefill_result = estimate_prefill_performance(
        model=model,
        batch_size=batch_size,
        input_tokens=input_tokens,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        debug=False
    )
    format_performance_result(prefill_result, "Processing Input Prompt")
    
    # Decode analysis
    print_subsection("Decode Phase Analysis")
    decode_result = estimate_decode_performance(
        model=model,
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        debug=False
    )
    format_performance_result(decode_result, "Generating Output Tokens")
    
    # End-to-end analysis
    print_subsection("End-to-End Analysis")
    e2e_result = estimate_end_to_end_performance(
        model=model,
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4,
        debug=False
    )
    format_performance_result(e2e_result, "Complete Inference Pipeline")
    
    # Show breakdown
    if 'prefill' in e2e_result and 'decode' in e2e_result:
        prefill_lat = e2e_result['prefill'].get('Latency', 0)
        decode_lat = e2e_result['decode'].get('Total_latency', 0)
        total_lat = e2e_result.get('total_latency', 0)
        
        print(f"\n📋 Detailed Time Breakdown:")
        print(f"   • Prefill Phase: {prefill_lat:.1f} ms ({prefill_lat/total_lat*100:.1f}%)")
        print(f"   • Decode Phase: {decode_lat:.1f} ms ({decode_lat/total_lat*100:.1f}%)")
        print(f"   • Total Time: {total_lat:.1f} ms")
        
        # Efficiency metrics
        effective_tokens = batch_size * (input_tokens + output_tokens)
        tokens_per_ms = effective_tokens / total_lat
        print(f"   • Effective Throughput: {tokens_per_ms * 1000:.1f} tokens/s")


def demo_hardware_comparison():
    """Compare performance across different hardware platforms."""
    print_section("Hardware Performance Comparison")
    
    # Test configurations
    configurations = [
        {
            'name': 'NVIDIA A100 80GB (TP=4)',
            'system_name': get_hardware_config('A100_80GB'),
            'tensor_parallel': 4,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        },
        {
            'name': 'NVIDIA H100 80GB (TP=4)', 
            'system_name': get_hardware_config('H100_80GB'),
            'tensor_parallel': 4,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        },
        {
            'name': 'AMD MI300X (TP=8)',
            'system_name': get_hardware_config('MI300X'),
            'tensor_parallel': 8,
            'pipeline_parallel': 1,
            'bits': 'bf16'
        }
    ]
    
    print(f"\n🏆 Comparing Hardware for Llama2-7B:")
    print(f"   • Workload: Batch=4, Input=2048, Output=256 tokens")
    
    results = compare_performance_configurations(
        model='llama2_7b',
        configurations=configurations,
        batch_size=4,
        input_tokens=2048,
        output_tokens=256,
        debug=False
    )
    
    print(f"\n📈 Performance Ranking:")
    for i, result in enumerate(results, 1):
        name = result['config_name']
        if 'error' in result:
            print(f"   {i}. {name}: ❌ {result['error']}")
            continue
            
        throughput = result.get('total_throughput', 0)
        ttft = result.get('ttft', 0)
        tpot = result.get('tpot', 0)
        
        print(f"   {i}. {name}")
        print(f"      💰 Throughput: {throughput:.1f} tokens/s")
        print(f"      ⚡ TTFT: {ttft:.1f} ms")
        print(f"      🔄 TPOT: {tpot:.1f} ms")
        
        # Calculate relative performance
        if i == 1:
            baseline_throughput = throughput
        else:
            speedup = throughput / baseline_throughput
            print(f"      📊 Relative Performance: {speedup:.2f}x")


def demo_precision_impact():
    """Analyze the impact of different precisions on performance."""
    print_section("Precision Impact Analysis")
    
    hardware = get_hardware_config('A100_80GB')
    precisions = ['bf16', 'int8', 'int4']
    
    print(f"\n🎚️ Precision Comparison for Llama2-7B (TP=4):")
    print(f"   • Hardware: A100 80GB")
    print(f"   • Workload: Batch=4, Input=2048, Output=128 tokens")
    
    precision_results = {}
    
    for precision in precisions:
        try:
            result = estimate_end_to_end_performance(
                model='llama2_7b',
                batch_size=4,
                input_tokens=2048,
                output_tokens=128,
                system_name=hardware,
                bits=precision,
                tensor_parallel=4,
                debug=False
            )
            
            precision_results[precision] = {
                'throughput': result.get('total_throughput', 0),
                'ttft': result.get('ttft', 0),
                'tpot': result.get('average_tpot', 0),
                'total_latency': result.get('total_latency', 0)
            }
            
        except Exception as e:
            precision_results[precision] = {'error': str(e)}
    
    # Display results
    baseline_throughput = None
    for precision in precisions:
        result = precision_results[precision]
        
        if 'error' in result:
            print(f"\n   {precision.upper()}: ❌ {result['error'][:50]}...")
            continue
            
        throughput = result['throughput']
        ttft = result['ttft']
        tpot = result['tpot']
        
        print(f"\n   {precision.upper()}:")
        print(f"      🚀 Throughput: {throughput:.1f} tokens/s")
        print(f"      ⚡ TTFT: {ttft:.1f} ms")
        print(f"      🔄 TPOT: {tpot:.1f} ms")
        
        if baseline_throughput is None:
            baseline_throughput = throughput
            print(f"      📊 Baseline")
        else:
            speedup = throughput / baseline_throughput
            print(f"      📊 Speedup: {speedup:.2f}x")


def demo_batch_scaling():
    """Analyze how performance scales with batch size."""
    print_section("Batch Size Scaling Analysis")
    
    hardware = get_hardware_config('H100_80GB')
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"\n📈 Batch Size Scaling for Llama2-7B:")
    print(f"   • Hardware: H100 80GB (TP=4)")
    print(f"   • Single token decode latency measurement")
    
    print(f"\n   {'Batch':<8} {'Latency':<12} {'Throughput':<15} {'Efficiency'}")
    print(f"   {'Size':<8} {'(ms/tok)':<12} {'(tokens/s)':<15} {'(vs batch=1)'}")
    print(f"   {'-'*8} {'-'*12} {'-'*15} {'-'*15}")
    
    baseline_efficiency = None
    
    for batch_size in batch_sizes:
        try:
            result = estimate_decode_performance(
                model='llama2_7b',
                batch_size=batch_size,
                input_tokens=2048,
                output_tokens=1,  # Single token for per-token measurement
                system_name=hardware,
                bits='bf16',
                tensor_parallel=4,
                debug=False
            )
            
            latency = result.get('Latency', 0)
            throughput = result.get('Effective_throughput', 0)
            
            if baseline_efficiency is None:
                baseline_efficiency = throughput
                efficiency_ratio = 1.0
            else:
                efficiency_ratio = throughput / baseline_efficiency
            
            print(f"   {batch_size:<8} {latency:<12.1f} {throughput:<15.1f} {efficiency_ratio:<15.2f}x")
            
        except Exception as e:
            print(f"   {batch_size:<8} {'ERROR':<12} {'N/A':<15} {'N/A'}")


def demo_parallelism_optimization():
    """Demonstrate parallelism strategy optimization."""
    print_section("Parallelism Strategy Optimization")
    
    model = 'llama2_7b'
    total_nodes = 8
    hardware = get_hardware_config('A100_80GB')
    
    print(f"\n⚡ Optimizing Parallelism for {model.upper()}:")
    print(f"   • Available GPUs: {total_nodes}")
    print(f"   • Hardware: A100 80GB")
    print(f"   • Target: Maximum throughput")
    
    # Get available parallelization options
    print_subsection("Available Parallelization Options")
    options = get_various_parallelization(model, total_nodes=total_nodes)
    
    print(f"   Available TP/PP combinations for {total_nodes} GPUs:")
    for tp, pp in sorted(options):
        nodes_used = tp * pp
        print(f"   • TP={tp}, PP={pp} (uses {nodes_used}/{total_nodes} GPUs)")
    
    # Find optimal strategy
    print_subsection("Optimal Strategy Search")
    try:
        best_strategy = get_best_parallelization_strategy(
            model=model,
            total_nodes=total_nodes,
            batch_size=16,
            input_tokens=2048,
            output_tokens=512,
            system_name=hardware,
            bits='bf16',
            debug=False
        )
        
        # Extract best configuration
        best_tp = best_strategy['TP'].iloc[0]
        best_pp = best_strategy['PP'].iloc[0]
        best_throughput = best_strategy['Tokens/s'].iloc[0]
        best_latency = best_strategy['Latency(ms)'].iloc[0]
        
        print(f"\n🏆 Optimal Configuration:")
        print(f"   • Tensor Parallel: {best_tp}")
        print(f"   • Pipeline Parallel: {best_pp}")
        print(f"   • Expected Throughput: {best_throughput:.1f} tokens/s")
        print(f"   • Expected Latency: {best_latency:.1f} ms")
        print(f"   • GPU Utilization: {best_tp * best_pp}/{total_nodes} ({best_tp * best_pp / total_nodes * 100:.1f}%)")
        
        # Show top 3 options if available
        if len(best_strategy) > 1:
            print(f"\n📊 Top Configurations:")
            for i in range(min(3, len(best_strategy))):
                tp = best_strategy['TP'].iloc[i]
                pp = best_strategy['PP'].iloc[i]
                throughput = best_strategy['Tokens/s'].iloc[i]
                latency = best_strategy['Latency(ms)'].iloc[i]
                
                print(f"   {i+1}. TP={tp}, PP={pp}: {throughput:.1f} tok/s, {latency:.1f} ms")
        
    except Exception as e:
        print(f"   ❌ Error finding optimal strategy: {str(e)}")


def demo_memory_vs_performance():
    """Demonstrate relationship between memory usage and performance."""
    print_section("Memory vs Performance Analysis")
    
    model_id = "meta-llama/Llama-2-7b-hf"
    batch_size = 8
    seq_length = 4096
    
    print(f"\n💾 Memory and Performance Analysis:")
    print(f"   • Model: {model_id}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Sequence Length: {seq_length}")
    
    # Calculate memory requirements
    print_subsection("Memory Requirements")
    try:
        memory_report = calculate_memory(
            model_id,
            batch_size=batch_size,
            seq_length=seq_length,
            precision="bf16"
        )
        
        print(f"   📊 Memory Breakdown (BF16):")
        print(f"   • Model Weights: {memory_report.weight_memory_gb:.2f} GB")
        print(f"   • KV Cache: {memory_report.kv_cache_memory_gb:.2f} GB")
        print(f"   • Activations: {memory_report.activation_memory_gb:.2f} GB")
        print(f"   • Total Memory: {memory_report.total_memory_gb:.2f} GB")
        print(f"   • Recommended GPU: {memory_report.recommended_gpu_memory_gb} GB")
        
    except Exception as e:
        print(f"   ❌ Memory calculation error: {str(e)}")
    
    # Performance with different memory configurations
    print_subsection("Performance with Memory Constraints")
    hardware_configs = [
        ('A100_40GB', 'Limited Memory'),
        ('A100_80GB', 'Standard Memory'), 
        ('H100_80GB', 'High Performance')
    ]
    
    for hw_name, description in hardware_configs:
        try:
            hw_config = get_hardware_config(hw_name)
            result = estimate_end_to_end_performance(
                model='llama2_7b',
                batch_size=batch_size,
                input_tokens=seq_length,
                output_tokens=256,
                system_name=hw_config,
                bits='bf16',
                tensor_parallel=2,
                debug=False
            )
            
            throughput = result.get('total_throughput', 0)
            memory_size = hw_config['Memory_size']
            
            print(f"   • {hw_name} ({description}): {throughput:.1f} tok/s ({memory_size} GB)")
            
        except Exception as e:
            print(f"   • {hw_name}: ❌ Error - {str(e)[:30]}...")


def demo_chunked_processing():
    """Demonstrate chunked processing for long sequences."""
    print_section("Chunked Processing for Long Sequences")
    
    hardware = get_hardware_config('A100_80GB')
    long_input = 16384  # Long context
    chunk_sizes = [512, 1024, 2048, 4096]
    
    print(f"\n📜 Long Context Processing Analysis:")
    print(f"   • Input Length: {long_input} tokens")
    print(f"   • Output Length: 512 tokens")
    print(f"   • Hardware: A100 80GB")
    
    print(f"\n   {'Chunk Size':<12} {'Processing':<12} {'Memory':<12} {'Notes'}")
    print(f"   {'':<12} {'Time (ms)':<12} {'Efficient':<12} {''}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
    
    for chunk_size in chunk_sizes:
        try:
            result = estimate_chunked_performance(
                model='llama2_7b',
                batch_size=2,
                input_tokens=long_input,
                output_tokens=512,
                chunk_size=chunk_size,
                system_name=hardware,
                bits='bf16',
                tensor_parallel=4,
                debug=False
            )
            
            # Extract timing if available
            if 'Latency' in result:
                latency = result['Latency']
                num_chunks = result.get('num_chunks', 1)
                
                memory_efficient = "Yes" if chunk_size <= 2048 else "Moderate"
                notes = f"{num_chunks} chunks"
                
                print(f"   {chunk_size:<12} {latency:<12.1f} {memory_efficient:<12} {notes}")
            else:
                print(f"   {chunk_size:<12} {'N/A':<12} {'N/A':<12} {'No data'}")
                
        except Exception as e:
            print(f"   {chunk_size:<12} {'ERROR':<12} {'N/A':<12} {str(e)[:15]}...")


def main():
    """Run comprehensive performance estimation demo."""
    start_time = time.time()
    
    print("🧮 LLM Performance Estimation - Comprehensive Demo")
    print("=" * 65)
    print("This demo shows how external repos can use llm-memory-calculator")
    print("for detailed LLM inference performance analysis.")
    print()
    print("📋 Demo Sections:")
    print("   1. Basic Performance Analysis")
    print("   2. Hardware Performance Comparison") 
    print("   3. Precision Impact Analysis")
    print("   4. Batch Size Scaling Analysis")
    print("   5. Parallelism Strategy Optimization")
    print("   6. Memory vs Performance Analysis")
    print("   7. Chunked Processing for Long Sequences")
    
    try:
        # Run all demo sections
        demo_basic_performance_analysis()
        demo_hardware_comparison()
        demo_precision_impact()
        demo_batch_scaling()
        demo_parallelism_optimization()
        demo_memory_vs_performance()
        demo_chunked_processing()
        
        # Summary
        elapsed = time.time() - start_time
        print_section("Demo Complete")
        print(f"\n✅ All performance estimation demos completed successfully!")
        print(f"⏱️  Total demo time: {elapsed:.1f} seconds")
        print(f"\n🎯 Key Takeaways:")
        print(f"   • llm-memory-calculator provides comprehensive LLM performance analysis")
        print(f"   • Supports all major hardware platforms (A100, H100, MI300X)")
        print(f"   • Enables optimization of parallelism strategies")
        print(f"   • Analyzes memory-performance tradeoffs")
        print(f"   • Ready for integration in external repositories")
        print(f"\n📦 Installation: pip install llm-memory-calculator")
        print(f"📚 Import: from llm_memory_calculator import estimate_end_to_end_performance")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {str(e)}")
        print(f"   This may indicate a configuration or dependency issue.")


if __name__ == "__main__":
    main()