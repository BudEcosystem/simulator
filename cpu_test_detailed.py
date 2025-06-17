from GenZ.cpu import cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling
import pandas as pd

# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=== CPU Performance Analysis for LLM ===")
print("Model: Qwen-7B")
print("System: Intel Xeon 8380 (Ice Lake)")
print()

# Analyze prefill on CPU
print("--- PREFILL PHASE ---")
prefill_result = cpu_aware_prefill_moddeling(
    model='llama',  # Use a known model name
    batch_size=1,
    input_tokens=512,
    system_name='intel_xeon_8380',
    debug=False
)

print(f"Prefill Latency: {prefill_result['Latency']:.2f} ms")
print(f"Prefill Throughput: {prefill_result['Throughput']:.2f} tokens/s")

# Check if CPU metrics are available
if 'model_df' in prefill_result and 'ISA_used' in prefill_result['model_df'].columns:
    df = prefill_result['model_df']
    print(f"\nCPU Metrics (Prefill):")
    print(f"  Primary ISA: {df['ISA_used'].mode()[0] if len(df['ISA_used'].mode()) > 0 else 'N/A'}")
    print(f"  Avg Thread Count: {df['Thread_count'].mean():.0f}")
    print(f"  Avg Parallel Efficiency: {df['Parallel_efficiency'].mean():.1%}")
    print(f"  Avg Frequency: {df['Frequency_GHz'].mean():.2f} GHz")
    print(f"  Cache Hit Rates:")
    print(f"    L1: {df['L1_hit_rate'].mean():.1%}")
    print(f"    L2: {df['L2_hit_rate'].mean():.1%}")
    print(f"    L3: {df['L3_hit_rate'].mean():.1%}")
    print(f"    DRAM Access: {df['DRAM_access_rate'].mean():.1%}")

# Analyze decode with different KV cache sizes
print("\n--- DECODE PHASE ---")
kv_sizes = [128, 256, 512, 1024]
decode_results = []

for kv_size in kv_sizes:
    decode_result = cpu_aware_decode_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=kv_size,  # KV cache size
        output_tokens=1,       # Generate 1 token
        system_name='intel_xeon_8380',
        debug=False
    )
    
    decode_results.append({
        'KV Cache Size': kv_size,
        'Latency (ms)': decode_result['Latency'],
        'Throughput (tokens/s)': decode_result['Throughput'],
        'Compute Bound': 'Yes' if decode_result['model_df']['Bound'].mode()[0] == 'Compute' else 'No'
    })

# Display decode results
decode_df = pd.DataFrame(decode_results)
print("\nDecode Performance vs KV Cache Size:")
print(decode_df.to_string(index=False))

# Compare different CPU systems
print("\n--- CPU COMPARISON ---")
cpu_systems = ['intel_xeon_8380', 'amd_epyc_7763', 'aws_graviton3']
comparison_results = []

for cpu_name in cpu_systems:
    prefill = cpu_aware_prefill_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        system_name=cpu_name,
        debug=False
    )
    
    decode = cpu_aware_decode_moddeling(
        model='llama',
        batch_size=1,
        input_tokens=512,
        output_tokens=1,
        system_name=cpu_name,
        debug=False
    )
    
    comparison_results.append({
        'CPU': cpu_name.replace('_', ' ').title(),
        'Prefill (ms)': f"{prefill['Latency']:.2f}",
        'Decode (ms)': f"{decode['Latency']:.2f}",
        'Prefill tok/s': f"{prefill['Throughput']:.0f}",
        'Decode tok/s': f"{decode['Throughput']:.0f}"
    })

comparison_df = pd.DataFrame(comparison_results)
print("\nCPU System Comparison:")
print(comparison_df.to_string(index=False)) 