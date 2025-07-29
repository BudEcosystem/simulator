#!/usr/bin/env python3
"""
Simple example showing the basic usage of llm-memory-calculator
with built-in parallelism optimization.
"""

from llm_memory_calculator import (
    get_best_parallelization_strategy,
    get_various_parallelization,
    get_hardware_config,
    HARDWARE_CONFIGS
)


def main():
    print("🧮 LLM Memory Calculator with Parallelism Optimization")
    print("=" * 60)
    
    # Show available hardware
    print("\n📊 Available Hardware Configurations:")
    for hw_name in HARDWARE_CONFIGS.keys():
        config = HARDWARE_CONFIGS[hw_name]
        print(f"  • {hw_name}: {config['Memory_size']}GB, {config['Flops']} TFLOPS")
    
    # Example 1: Find parallelization options
    print("\n⚡ Parallelization Options for Llama2-7B on 8 nodes:")
    options = get_various_parallelization('llama2_7b', total_nodes=8)
    for tp, pp in sorted(options):
        print(f"  • TP={tp}, PP={pp} (uses {tp*pp} nodes)")
    
    # Example 2: Find best strategy
    print("\n🎯 Best Parallelization Strategy:")
    hardware = get_hardware_config('A100_80GB')
    
    best = get_best_parallelization_strategy(
        model='llama2_7b',
        total_nodes=8,
        batch_size=32,
        system_name=hardware,
        bits='bf16'
    )
    
    tp = best['TP'].iloc[0]
    pp = best['PP'].iloc[0] 
    throughput = best['Tokens/s'].iloc[0]
    latency = best['Latency(ms)'].iloc[0]
    
    print(f"  • Best configuration: TP={tp}, PP={pp}")
    print(f"  • Performance: {throughput:.1f} tokens/s, {latency:.1f}ms latency")
    
    # Example 3: Compare hardware
    print("\n🏆 Hardware Performance Comparison:")
    hardware_list = ['A100_80GB', 'H100_80GB', 'MI300X']
    
    for hw_name in hardware_list:
        hw_config = get_hardware_config(hw_name)
        result = get_best_parallelization_strategy(
            model='llama2_7b',
            total_nodes=8,
            batch_size=32,
            system_name=hw_config,
            debug=False
        )
        
        throughput = result['Tokens/s'].iloc[0]
        print(f"  • {hw_name}: {throughput:.1f} tokens/s")
    
    print("\n✅ All examples completed successfully!")
    print("📦 Ready for use in external repositories with: pip install llm-memory-calculator")


if __name__ == "__main__":
    main()