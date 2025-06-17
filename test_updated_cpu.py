from GenZ.cpu import create_cpu_system, cpu_aware_prefill_moddeling
from GenZ.cpu.isa_model import ISAType

# Create a fresh CPU system
cpu = create_cpu_system('intel_xeon_8592_plus')

# Check AMX configuration
amx_config = cpu.isa_selector.isa_configs.get(ISAType.AMX)
print("=== AMX Configuration ===")
print(f"Vector width BF16: {amx_config.vector_width['bf16']:,}")
print(f"Throughput: {amx_config.throughput}")
print(f"Latency: {amx_config.latency}")
print(f"AMX units per core: {amx_config.special_constraints.get('amx_units_per_core', 1)}")

# Calculate peak FLOPS
cores = cpu.config.cores_per_socket * cpu.config.sockets
ops_per_cycle_per_core = amx_config.vector_width['bf16'] * amx_config.throughput['tilemmul']
frequency = 2.5e9  # Turbo
peak_flops = cores * ops_per_cycle_per_core * frequency

print(f"\nPeak FLOPS calculation:")
print(f"  Cores: {cores}")
print(f"  Ops/cycle/core: {ops_per_cycle_per_core:,.0f}")
print(f"  Frequency: {frequency/1e9} GHz")
print(f"  Peak: {peak_flops/1e12:.1f} TFLOPS")

# Run a quick test
print("\n=== Running Prefill Test ===")
result = cpu_aware_prefill_moddeling(
    model='meta-llama/llama-2-7b',
    batch_size=4,
    input_tokens=400,
    system_name='intel_xeon_8592_plus',
    debug=False
)

print(f"Latency: {result['Latency']:.2f} ms")
print(f"Throughput: {result['Throughput']:.2f} tokens/sec")

# Compare to real-world
real_throughput = 900  # From your data
print(f"\nReal-world throughput: {real_throughput} tokens/sec")
print(f"Simulation is {real_throughput/result['Throughput']:.1f}x slower than reality")

# Check what ISA was used
df = result['model_df']
gemm_ops = df[df['Op Type'] == 'GEMM']
if len(gemm_ops) > 0:
    print(f"\nGEMM operations:")
    print(f"  ISA used: {gemm_ops['ISA_used'].iloc[0]}")
    print(f"  Total latency: {gemm_ops['Latency (msec)'].sum():.2f} ms") 