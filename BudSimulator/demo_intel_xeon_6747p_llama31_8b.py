#!/usr/bin/env python3
"""
Demo: Intel Xeon 6747P CPU Simulation with Llama 3.1 8B Model
Simple performance analysis for LLM inference.
"""

from llm_memory_calculator.genz.cpu import (
    cpu_aware_prefill_moddeling, 
    cpu_aware_decode_moddeling,
    create_cpu_system
)

def main():
    """Main demo execution."""
    # Configuration
    cpu_name = 'intel_xeon_6430'  # Sapphire Rapids CPU
    model_name = 'meta-llama/Llama-3.1-8B'
    batch_size = 100
    input_tokens = 128
    output_tokens = 120
    
    print("Intel Xeon 6430 (Sapphire Rapids) + Llama 3.1 8B Performance Analysis")
    print("=" * 60)
    
    # 1. Create CPU system
    cpu_system = create_cpu_system(cpu_name)
    print(f"✅ Created CPU system: {cpu_system.config.vendor} {cpu_system.config.microarchitecture}")
    
    # 2. Run prefill simulation
    prefill_result = cpu_aware_prefill_moddeling(
        model=model_name,
        batch_size=batch_size,
        input_tokens=input_tokens,
        system_name=cpu_name,
        debug=False  # Disable debug for clean output
    )
    
    # 3. Run decode simulation
    decode_result = cpu_aware_decode_moddeling(
        model=model_name,
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=1,  # Single token for TPOT
        system_name=cpu_name,
        debug=False  # Disable debug for clean output
    )
    
    # 4. Calculate metrics
    ttft = prefill_result['Latency']  # Time to First Token
    tpot = decode_result['Latency']   # Time Per Output Token
    e2e_latency = ttft + (tpot * output_tokens)  # End-to-End latency
    prefill_throughput = prefill_result['Throughput']
    decode_throughput = decode_result['Throughput']
    
    # 5. Print results
    print(f"\nPerformance Results:")
    print(f"TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"TPOT (Time Per Output Token): {tpot:.3f} ms")
    print(f"E2E Latency: {e2e_latency:.2f} ms")
    print(f"Prefill Throughput: {prefill_throughput:.1f} tokens/s")
    print(f"Decode Throughput: {decode_throughput:.1f} tokens/s")


if __name__ == "__main__":
    main() 