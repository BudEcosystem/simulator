"""
Example demonstrating vLLM-style LoRA memory calculation.

This example shows how to calculate memory requirements for serving
multiple LoRA adapters simultaneously, following vLLM's allocation strategy.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.lora.config import LoraConfig

# Model configurations (to avoid HuggingFace API calls)
LLAMA2_7B_CONFIG = {
    'hidden_size': 4096,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,
    'intermediate_size': 11008,
    'num_key_value_heads': 32,
    'hidden_act': 'silu',
    'num_parameters': 7_000_000_000,
    'vocab_size': 32000,
}

LLAMA2_13B_CONFIG = {
    'hidden_size': 5120,
    'num_hidden_layers': 40,
    'num_attention_heads': 40,
    'intermediate_size': 13824,
    'num_key_value_heads': 40,
    'hidden_act': 'silu',
    'num_parameters': 13_000_000_000,
    'vocab_size': 32000,
}

LLAMA2_70B_CONFIG = {
    'hidden_size': 8192,
    'num_hidden_layers': 80,
    'num_attention_heads': 64,
    'intermediate_size': 28672,
    'num_key_value_heads': 8,  # GQA
    'hidden_act': 'silu',
    'num_parameters': 70_000_000_000,
    'vocab_size': 32000,
}


def example_basic_lora_memory():
    """Calculate memory for a model with LoRA adapters."""
    print("=" * 60)
    print("Example 1: Basic LoRA Memory Calculation")
    print("=" * 60)

    # Model: Llama-2-7B
    # Scenario: Serving 5 LoRA adapters with max rank 256
    print("\nModel: Llama-2-7B")

    # Without LoRA
    print("\n--- Without LoRA ---")
    report_baseline = calculate_memory(
        LLAMA2_7B_CONFIG,
        batch_size=1,
        seq_length=2048,
        precision='bf16'
    )
    print(report_baseline)

    # With LoRA (vLLM-style: max_loras=5, max_lora_rank=256)
    print("\n--- With LoRA (max_loras=5, max_lora_rank=256) ---")
    lora_config = LoraConfig(
        enabled=True,
        max_loras=5,
        max_lora_rank=256,
        target_modules=['attn', 'ffn'],  # Apply to both attention and FFN layers
        lora_dtype='bf16'
    )

    report_lora = calculate_memory(
        LLAMA2_7B_CONFIG,
        batch_size=1,
        seq_length=2048,
        precision='bf16',
        lora_config=lora_config
    )
    print(report_lora)

    # Show the difference
    print("\n--- Memory Overhead from LoRA ---")
    lora_overhead_gb = report_lora.lora_adapter_memory_gb
    print(f"LoRA Adapter Memory: {lora_overhead_gb:.3f} GB")
    print(f"Percentage of base model weights: {(lora_overhead_gb / report_baseline.weight_memory_gb * 100):.2f}%")
    print(f"Total memory increase: {(report_lora.total_memory_gb - report_baseline.total_memory_gb):.3f} GB")


def example_scaling_with_lora_count():
    """Show how memory scales with number of LoRA adapters."""
    print("\n" + "=" * 60)
    print("Example 2: Scaling Memory with Number of LoRA Adapters")
    print("=" * 60)

    max_lora_rank = 256

    print(f"\nModel: Llama-2-7B")
    print(f"Max LoRA Rank: {max_lora_rank}")
    print(f"Target Modules: attention + FFN")
    print("\n{:<12} {:<20} {:<15}".format("max_loras", "LoRA Memory (GB)", "Total Memory (GB)"))
    print("-" * 50)

    for max_loras in [1, 2, 5, 10, 20]:
        lora_config = LoraConfig(
            enabled=True,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            target_modules=['attn', 'ffn'],
            lora_dtype='bf16'
        )

        report = calculate_memory(
            LLAMA2_7B_CONFIG,
            batch_size=1,
            seq_length=2048,
            precision='bf16',
            lora_config=lora_config
        )

        print(f"{max_loras:<12} {report.lora_adapter_memory_gb:<20.3f} {report.total_memory_gb:<15.2f}")


def example_different_ranks():
    """Show how memory scales with different LoRA ranks."""
    print("\n" + "=" * 60)
    print("Example 3: Scaling Memory with LoRA Rank")
    print("=" * 60)

    max_loras = 5

    print(f"\nModel: Llama-2-7B")
    print(f"Max LoRAs: {max_loras}")
    print(f"Target Modules: attention + FFN")
    print("\n{:<15} {:<20} {:<15}".format("max_lora_rank", "LoRA Memory (GB)", "Total Memory (GB)"))
    print("-" * 52)

    for rank in [8, 16, 32, 64, 128, 256, 512]:
        lora_config = LoraConfig(
            enabled=True,
            max_loras=max_loras,
            max_lora_rank=rank,
            target_modules=['attn', 'ffn'],
            lora_dtype='bf16'
        )

        report = calculate_memory(
            LLAMA2_7B_CONFIG,
            batch_size=1,
            seq_length=2048,
            precision='bf16',
            lora_config=lora_config
        )

        print(f"{rank:<15} {report.lora_adapter_memory_gb:<20.3f} {report.total_memory_gb:<15.2f}")


def example_tensor_parallelism():
    """Show memory distribution with tensor parallelism."""
    print("\n" + "=" * 60)
    print("Example 4: LoRA Memory with Tensor Parallelism")
    print("=" * 60)

    print(f"\nModel: Llama-2-70B")
    print(f"LoRA Config: max_loras=5, max_lora_rank=256")
    print("\nTensor Parallelism Comparison:")
    print("{:<10} {:<20} {:<20} {:<15}".format("TP Size", "LoRA Memory (GB)", "Total Memory (GB)", "Per-GPU (GB)"))
    print("-" * 70)

    lora_config = LoraConfig(
        enabled=True,
        max_loras=5,
        max_lora_rank=256,
        target_modules=['attn', 'ffn'],
        lora_dtype='bf16',
        fully_sharded_loras=False  # Default: only B matrix is sharded
    )

    for tp_size in [1, 2, 4, 8]:
        report = calculate_memory(
            LLAMA2_70B_CONFIG,
            batch_size=1,
            seq_length=2048,
            precision='bf16',
            tensor_parallel=tp_size,
            lora_config=lora_config
        )

        per_gpu_memory = report.total_memory_gb
        print(f"{tp_size:<10} {report.lora_adapter_memory_gb:<20.3f} {report.total_memory_gb * tp_size:<20.2f} {per_gpu_memory:<15.2f}")

    # Compare with fully sharded LoRAs
    print("\nWith fully_sharded_loras=True:")
    print("{:<10} {:<20} {:<20} {:<15}".format("TP Size", "LoRA Memory (GB)", "Total Memory (GB)", "Per-GPU (GB)"))
    print("-" * 70)

    lora_config_sharded = LoraConfig(
        enabled=True,
        max_loras=5,
        max_lora_rank=256,
        target_modules=['attn', 'ffn'],
        lora_dtype='bf16',
        fully_sharded_loras=True  # Both A and B matrices are sharded
    )

    for tp_size in [1, 2, 4, 8]:
        report = calculate_memory(
            LLAMA2_70B_CONFIG,
            batch_size=1,
            seq_length=2048,
            precision='bf16',
            tensor_parallel=tp_size,
            lora_config=lora_config_sharded
        )

        per_gpu_memory = report.total_memory_gb
        print(f"{tp_size:<10} {report.lora_adapter_memory_gb:<20.3f} {report.total_memory_gb * tp_size:<20.2f} {per_gpu_memory:<15.2f}")


def example_target_modules():
    """Show memory with different target modules."""
    print("\n" + "=" * 60)
    print("Example 5: LoRA Memory with Different Target Modules")
    print("=" * 60)

    print(f"\nModel: Llama-2-7B")
    print(f"LoRA Config: max_loras=5, max_lora_rank=256")
    print("\n{:<30} {:<20}".format("Target Modules", "LoRA Memory (GB)"))
    print("-" * 52)

    target_configs = [
        (['attn'], "Attention only"),
        (['ffn'], "FFN only"),
        (['attn', 'ffn'], "Attention + FFN"),
    ]

    for targets, description in target_configs:
        lora_config = LoraConfig(
            enabled=True,
            max_loras=5,
            max_lora_rank=256,
            target_modules=targets,
            lora_dtype='bf16'
        )

        report = calculate_memory(
            LLAMA2_7B_CONFIG,
            batch_size=1,
            seq_length=2048,
            precision='bf16',
            lora_config=lora_config
        )

        print(f"{description:<30} {report.lora_adapter_memory_gb:<20.3f}")


def example_real_world_scenario():
    """Real-world scenario: vLLM deployment with multiple LoRA adapters."""
    print("\n" + "=" * 60)
    print("Example 6: Real-World vLLM Deployment Scenario")
    print("=" * 60)

    print("\nScenario: Multi-tenant LLM serving with LoRA adapters")
    print("- Base Model: Llama-2-13B")
    print("- Hardware: 8x A100 80GB GPUs (Tensor Parallelism = 8)")
    print("- LoRA Config: max_loras=5, max_lora_rank=256")
    print("- Workload: Mixed batch sizes and sequence lengths")

    lora_config = LoraConfig(
        enabled=True,
        max_loras=5,
        max_lora_rank=256,
        target_modules=['attn', 'ffn'],
        lora_dtype='bf16',
        fully_sharded_loras=False
    )

    print("\n{:<15} {:<15} {:<20} {:<20}".format(
        "Batch Size", "Seq Length", "Per-GPU Memory (GB)", "GPU Utilization"
    ))
    print("-" * 75)

    for batch_size in [1, 4, 8]:
        for seq_length in [2048, 4096, 8192]:
            report = calculate_memory(
                LLAMA2_13B_CONFIG,
                batch_size=batch_size,
                seq_length=seq_length,
                precision='bf16',
                tensor_parallel=8,
                lora_config=lora_config
            )

            per_gpu_memory = report.total_memory_gb
            gpu_utilization = (per_gpu_memory / 80) * 100  # A100 80GB

            print(f"{batch_size:<15} {seq_length:<15} {per_gpu_memory:<20.2f} {gpu_utilization:<20.1f}%")

    print("\nBreakdown for batch_size=4, seq_length=4096:")
    report = calculate_memory(
        LLAMA2_13B_CONFIG,
        batch_size=4,
        seq_length=4096,
        precision='bf16',
        tensor_parallel=8,
        lora_config=lora_config
    )
    print(f"  Weights:         {report.weight_memory_gb:.2f} GB")
    print(f"  KV Cache:        {report.kv_cache_gb:.2f} GB")
    print(f"  Activations:     {report.activation_memory_gb:.2f} GB")
    print(f"  LoRA Adapters:   {report.lora_adapter_memory_gb:.2f} GB")
    print(f"  Extra/Overhead:  {report.extra_work_gb:.2f} GB")
    print(f"  Total Per GPU:   {report.total_memory_gb:.2f} GB")
    print(f"  Total Cluster:   {report.total_memory_gb * 8:.2f} GB")


if __name__ == "__main__":
    print("\nvLLM-Style LoRA Memory Calculation Examples")
    print("=" * 60)

    example_basic_lora_memory()
    example_scaling_with_lora_count()
    example_different_ranks()
    example_tensor_parallelism()
    example_target_modules()
    example_real_world_scenario()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
