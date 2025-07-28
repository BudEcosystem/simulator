from GenZ.cpu import cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling

# Analyze prefill on CPU

prefill_result = cpu_aware_prefill_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=50,
    input_tokens=1200,
    system_name='intel_xeon_8592_plus'  # CPU preset
)

# Analyze decode with growing KV cache

decode_result = cpu_aware_decode_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=50,
    input_tokens=340,  # KV cache size
    output_tokens=128,   # Generate 1 token
    system_name='intel_xeon_8592_plus'
    
)

print("Prefill on CPU")
print(prefill_result)
print("Decode on CPU")
print(decode_result)