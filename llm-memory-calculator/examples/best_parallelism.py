#!/usr/bin/env python3
"""
Example script demonstrating how to find the best parallelism strategy
for LLM inference using the integrated GenZ functionality in llm-memory-calculator.
"""

from llm_memory_calculator import (
    get_best_parallelization_strategy,
    get_various_parallelization,
    get_pareto_optimal_performance,
    get_minimum_system_size,
    get_hardware_config,
    HARDWARE_CONFIGS,
    ModelMemoryCalculator
)


def main():
    # Show available hardware configurations
    print("=== Available Hardware Configurations ===")
    print("Pre-defined hardware configs:", list(HARDWARE_CONFIGS.keys()))
    
    # Example 1: Get best parallelization for Llama2-7B on 8 GPUs
    print("\n=== Example 1: Best Parallelization for Llama2-7B ===")
    
    # Use predefined hardware config
    hardware_config = get_hardware_config('A100_80GB')
    print(f"Using hardware: {hardware_config}")
    
    best_strategy = get_best_parallelization_strategy(
        stage='decode',
        model='llama2_7b',
        total_nodes=8,
        batch_size=32,
        beam_size=1,
        input_tokens=2048,
        output_tokens=256,
        system_name=hardware_config,
        bits='bf16',
        debug=True
    )
    
    print("\nBest parallelization strategy:")
    print(best_strategy)
    
    # Example 2: Get various parallelization options
    print("\n=== Example 2: Various Parallelization Options ===")
    
    parallelism_options = get_various_parallelization(
        model='llama2_7b',
        total_nodes=16
    )
    
    print(f"Available parallelization combinations for 16 nodes: {parallelism_options}")
    for tp, pp in sorted(parallelism_options):
        print(f"  TP={tp}, PP={pp} (uses {tp*pp} nodes)")
    
    # Example 3: Get Pareto-optimal configurations (requires paretoset)
    print("\n=== Example 3: Pareto-Optimal Configurations ===")
    
    try:
        pareto_configs = get_pareto_optimal_performance(
            stage='prefill',
            model='llama2_7b',
            total_nodes=8,
            batch_list=[1, 8, 16, 32],
            beam_size=1,
            input_tokens=2048,
            output_tokens=256,
            system_name=hardware_config,
            bits='int8',
            debug=False
        )
        
        print("Pareto-optimal configurations (balancing latency vs throughput):")
        print(pareto_configs)
    except ImportError as e:
        print(f"Pareto optimization not available: {e}")
        print("Install with: pip install llm-memory-calculator[pareto]")
    
    # Example 4: Find minimum system size for different configurations
    print("\n=== Example 4: Minimum System Size Analysis ===")
    
    configs_to_test = [
        {'model': 'llama2_7b', 'batch': 1, 'name': 'Llama2-7B (batch=1)'},
        {'model': 'llama2_7b', 'batch': 32, 'name': 'Llama2-7B (batch=32)'},
        {'model': 'llama2_13b', 'batch': 16, 'name': 'Llama2-13B (batch=16)'},
    ]
    
    for config in configs_to_test:
        try:
            min_size = get_minimum_system_size(
                model=config['model'],
                batch_size=config['batch'],
                input_tokens=2048,
                output_tokens=256,
                system_name=hardware_config,
                bits='bf16'
            )
            print(f"{config['name']}: {min_size} nodes minimum")
        except Exception as e:
            print(f"{config['name']}: Could not determine ({str(e)[:50]}...)")
    
    # Example 5: Compare different precision levels
    print("\n=== Example 5: Precision Impact on Parallelism ===")
    
    for precision in ['fp32', 'bf16', 'int8']:
        print(f"\nPrecision: {precision}")
        try:
            best = get_best_parallelization_strategy(
                stage='decode',
                model='llama2_7b',
                total_nodes=8,
                batch_size=64,
                input_tokens=2048,
                output_tokens=256,
                system_name=hardware_config,
                bits=precision,
                debug=False
            )
            print(best[['TP', 'PP', 'Latency(ms)', 'Tokens/s']].to_string(index=False))
        except Exception as e:
            print(f"  Error with {precision}: {e}")
    
    # Example 6: Compare different hardware
    print("\n=== Example 6: Hardware Comparison ===")
    
    hardware_types = ['A100_80GB', 'H100_80GB', 'MI300X']
    
    for hw_name in hardware_types:
        print(f"\nHardware: {hw_name}")
        try:
            hw_config = get_hardware_config(hw_name)
            best = get_best_parallelization_strategy(
                stage='decode',
                model='llama2_7b',
                total_nodes=8,
                batch_size=32,
                system_name=hw_config,
                debug=False
            )
            tp, pp = best['TP'].iloc[0], best['PP'].iloc[0]
            throughput = best['Tokens/s'].iloc[0]
            latency = best['Latency(ms)'].iloc[0]
            print(f"  Best: TP={tp}, PP={pp} → {throughput:.1f} tokens/s, {latency:.1f}ms")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()