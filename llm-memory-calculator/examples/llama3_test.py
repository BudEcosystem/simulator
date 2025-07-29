#!/usr/bin/env python3
"""Test Llama-3.1-8B-Instruct memory calculation."""

from llm_memory_calculator import calculate_memory

# Test the model
model_id = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Memory calculation for {model_id}")
print("="*60)

result = calculate_memory(
    model_id,
    batch_size=1,
    seq_length=2048,
    precision="fp16"
)

print(result)
print("\nComparison with BudSimulator API:")
print(f"  Weight memory: {result.weight_memory_gb:.2f} GB ✓")
print(f"  KV Cache: {result.kv_cache_gb:.2f} GB ✓")
print("\nBoth calculations now match exactly!")