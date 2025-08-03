"""
Example demonstrating LoRA simulation in GenZ/BudSimulator.

This example shows how to configure and use LoRA (Low-Rank Adaptation) 
in the GenZ transformer performance simulator.
"""

import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_memory_calculator.lora.config import LoraConfig
from llm_memory_calculator.genz.Models import create_full_prefill_model, create_full_decode_model, get_configs
from llm_memory_calculator.genz.Models.utils import OpType


def demonstrate_lora_merge_strategy():
    """Demonstrate LoRA with merge strategy (one-time adapter merging cost)."""
    print("=== LoRA Merge Strategy Example ===")
    
    # Get base model configuration
    model_config = get_configs('llama-7b')
    
    # Configure LoRA with merge strategy
    model_config.lora_config = LoraConfig(
        enabled=True,
        rank=16,  # Low rank for adapter matrices
        target_modules=['attn', 'ffn'],  # Apply to attention and FFN layers
        strategy='merge'  # One-time merge cost before inference
    )
    
    # Create prefill model
    csv_file = create_full_prefill_model(
        name=model_config,
        input_sequence_length=512,
        data_path='/tmp/lora_example',
        tensor_parallel=1
    )
    
    # Load and analyze the generated operations
    csv_path = os.path.join('/tmp/lora_example', 'model', csv_file)
    df = pd.read_csv(csv_path)
    
    # Count LoRA merge operations
    lora_merge_ops = df[df['T'] == OpType.LORA_MERGE.value]
    print(f"Number of LORA_MERGE operations: {len(lora_merge_ops)}")
    print(f"Total operations: {len(df)}")
    
    return df


def demonstrate_lora_dynamic_strategy():
    """Demonstrate LoRA with dynamic strategy (per-token overhead)."""
    print("\n=== LoRA Dynamic Strategy Example ===")
    
    # Get base model configuration
    model_config = get_configs('llama-7b')
    
    # Configure LoRA with dynamic strategy
    model_config.lora_config = LoraConfig(
        enabled=True,
        rank=32,  # Higher rank for better quality
        target_modules=['ffn'],  # Apply only to FFN layers
        strategy='dynamic'  # Per-token computation overhead
    )
    
    # Create decode model
    csv_file = create_full_decode_model(
        name=model_config,
        input_sequence_length=2048,
        output_gen_tokens=128,
        data_path='/tmp/lora_example',
        tensor_parallel=4
    )
    
    # Load and analyze the generated operations
    csv_path = os.path.join('/tmp/lora_example', 'model', csv_file)
    df = pd.read_csv(csv_path)
    
    # Count LoRA operations
    lora_a_ops = df[df['T'] == OpType.GEMM_LORA_A.value]
    lora_b_ops = df[df['T'] == OpType.GEMM_LORA_B.value]
    add_ops = df[df['T'] == OpType.ADD.value]
    
    print(f"Number of GEMM_LORA_A operations: {len(lora_a_ops)}")
    print(f"Number of GEMM_LORA_B operations: {len(lora_b_ops)}")
    print(f"Number of ADD operations: {len(add_ops)}")
    print(f"Total operations: {len(df)}")
    
    return df


def compare_with_baseline():
    """Compare LoRA-enabled model with baseline (no LoRA)."""
    print("\n=== Baseline vs LoRA Comparison ===")
    
    model_config = get_configs('gpt-2')
    
    # Baseline (no LoRA)
    model_config.lora_config = LoraConfig(enabled=False)
    baseline_csv = create_full_prefill_model(
        name=model_config,
        input_sequence_length=128,
        data_path='/tmp/lora_example'
    )
    baseline_df = pd.read_csv(os.path.join('/tmp/lora_example', 'model', baseline_csv))
    
    # With LoRA
    model_config.lora_config = LoraConfig(
        enabled=True,
        rank=8,
        target_modules=['attn', 'ffn'],
        strategy='dynamic'
    )
    lora_csv = create_full_prefill_model(
        name=model_config,
        input_sequence_length=128,
        data_path='/tmp/lora_example'
    )
    lora_df = pd.read_csv(os.path.join('/tmp/lora_example', 'model', lora_csv))
    
    print(f"Baseline operations: {len(baseline_df)}")
    print(f"LoRA operations: {len(lora_df)}")
    print(f"Additional operations from LoRA: {len(lora_df) - len(baseline_df)}")
    
    # Show operation type distribution
    print("\nOperation type distribution with LoRA:")
    op_counts = lora_df['T'].value_counts()
    for op_type, count in op_counts.items():
        print(f"  OpType {op_type}: {count} operations")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_lora_merge_strategy()
    demonstrate_lora_dynamic_strategy()
    compare_with_baseline()
    
    print("\n✅ LoRA simulation examples completed!")
    print("Check /tmp/lora_example/model/ for generated CSV files.") 