"""
Simple example demonstrating the simplified LoRA memory calculation API.

No need to import or create LoraConfig - just pass max_loras and max_lora_rank directly!
"""

from llm_memory_calculator import calculate_memory

# Model configuration
LLAMA2_7B = {
    'hidden_size': 4096,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,
    'intermediate_size': 11008,
    'num_key_value_heads': 32,
    'hidden_act': 'silu',
    'num_parameters': 7_000_000_000,
    'vocab_size': 32000,
}


print("=" * 60)
print("Simplified LoRA Memory Calculation")
print("=" * 60)

# 1. Basic memory calculation (no LoRA)
print("\n1. Without LoRA:")
report = calculate_memory(LLAMA2_7B, batch_size=1, seq_length=2048, precision='bf16')
print(f"   Total Memory: {report.total_memory_gb:.2f} GB")

# 2. With LoRA - SIMPLE API (most common use case)
print("\n2. With LoRA (max_loras=5, max_lora_rank=256):")
report = calculate_memory(
    LLAMA2_7B,
    batch_size=1,
    seq_length=2048,
    precision='bf16',
    max_loras=5,
    max_lora_rank=256
)
print(f"   Total Memory: {report.total_memory_gb:.2f} GB")
print(f"   LoRA Memory: {report.lora_adapter_memory_gb:.2f} GB")

# 3. With custom target modules
print("\n3. LoRA on attention only (more efficient):")
report = calculate_memory(
    LLAMA2_7B,
    batch_size=1,
    seq_length=2048,
    precision='bf16',
    max_loras=5,
    max_lora_rank=256,
    target_modules=['attn']  # Only attention, not FFN
)
print(f"   Total Memory: {report.total_memory_gb:.2f} GB")
print(f"   LoRA Memory: {report.lora_adapter_memory_gb:.2f} GB")

# 4. With tensor parallelism
print("\n4. LoRA with tensor parallelism (TP=4):")
report = calculate_memory(
    LLAMA2_7B,
    batch_size=1,
    seq_length=2048,
    precision='bf16',
    tensor_parallel=4,
    max_loras=5,
    max_lora_rank=256
)
print(f"   Per-GPU Memory: {report.total_memory_gb:.2f} GB")
print(f"   LoRA Memory (per GPU): {report.lora_adapter_memory_gb:.2f} GB")

# 5. Fully sharded LoRAs (both A and B matrices sharded)
print("\n5. With fully_sharded_loras=True:")
report = calculate_memory(
    LLAMA2_7B,
    batch_size=1,
    seq_length=2048,
    precision='bf16',
    tensor_parallel=4,
    max_loras=5,
    max_lora_rank=256,
    fully_sharded_loras=True
)
print(f"   Per-GPU Memory: {report.total_memory_gb:.2f} GB")
print(f"   LoRA Memory (per GPU): {report.lora_adapter_memory_gb:.2f} GB")

print("\n" + "=" * 60)
print("That's it! No need to import LoraConfig separately.")
print("=" * 60)
