#!/usr/bin/env python3
"""Basic usage examples for LLM Memory Calculator."""

from llm_memory_calculator import calculate_memory, MemoryReport


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def example_huggingface_model():
    """Example: Calculate memory for a HuggingFace model."""
    print_separator("Example 1: HuggingFace Model")
    
    # Calculate memory for Llama-2-7B
    result = calculate_memory("meta-llama/Llama-2-7b-hf")
    print(result)
    
    # Access specific values
    print(f"\nDetailed breakdown:")
    print(f"- Model Type: {result.model_type}")
    print(f"- Attention Type: {result.attention_type}")
    print(f"- Parameters: {result.parameter_count:,}")
    print(f"- Total Memory: {result.total_memory_gb:.2f} GB")
    print(f"- Can fit in 24GB GPU: {'Yes' if result.can_fit_24gb_gpu else 'No'}")


def example_custom_parameters():
    """Example: Calculate with custom inference parameters."""
    print_separator("Example 2: Custom Parameters")
    
    # Different batch sizes and sequence lengths
    configs = [
        {"batch_size": 1, "seq_length": 2048, "precision": "fp16"},
        {"batch_size": 4, "seq_length": 4096, "precision": "fp16"},
        {"batch_size": 1, "seq_length": 8192, "precision": "int8"},
    ]
    
    model_id = "mistralai/Mistral-7B-v0.1"
    
    for config in configs:
        result = calculate_memory(model_id, **config)
        print(f"\nBatch={config['batch_size']}, Seq={config['seq_length']}, Precision={config['precision']}:")
        print(f"  Total Memory: {result.total_memory_gb:.2f} GB")
        print(f"  KV Cache: {result.kv_cache_gb:.2f} GB ({result.kv_cache_gb/result.total_memory_gb*100:.1f}%)")


def example_custom_config():
    """Example: Calculate from custom model configuration."""
    print_separator("Example 3: Custom Configuration")
    
    # Define a custom model config (similar to Llama-3-8B)
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA with 4:1 compression
        "vocab_size": 128256,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "rope_scaling": {"type": "linear", "factor": 2.0},
    }
    
    result = calculate_memory(config, batch_size=1, seq_length=4096)
    
    print(f"Custom Model Analysis:")
    print(f"- Attention Type: {result.attention_type} (detected from config)")
    print(f"- Estimated Parameters: {result.parameter_count:,}")
    print(f"- Memory Breakdown:")
    print(f"  - Weights: {result.weight_memory_gb:.2f} GB")
    print(f"  - KV Cache: {result.kv_cache_gb:.2f} GB")
    print(f"  - Activations: {result.activation_memory_gb:.2f} GB")
    print(f"  - Total: {result.total_memory_gb:.2f} GB")


def example_tensor_parallelism():
    """Example: Calculate with tensor parallelism."""
    print_separator("Example 4: Tensor Parallelism")
    
    model_id = "meta-llama/Llama-2-70b-hf"
    
    # Compare different tensor parallel configurations
    for tp in [1, 2, 4, 8]:
        result = calculate_memory(
            model_id,
            batch_size=1,
            seq_length=4096,
            tensor_parallel=tp
        )
        print(f"\nTensor Parallel = {tp}:")
        print(f"  Per-GPU Memory: {result.total_memory_gb:.2f} GB")
        print(f"  Recommended GPU: {result.recommended_gpu_memory_gb} GB")


def example_moe_model():
    """Example: Mixture of Experts model."""
    print_separator("Example 5: Mixture of Experts")
    
    # Mixtral 8x7B - a sparse MoE model
    result = calculate_memory(
        "mistralai/Mixtral-8x7B-v0.1",
        batch_size=1,
        seq_length=32768,  # Mixtral supports 32K context
        precision="fp16"
    )
    
    print(f"Mixtral-8x7B Analysis:")
    print(f"- Total Parameters: {result.parameter_count/1e9:.1f}B")
    print(f"- Total Memory at 32K context: {result.total_memory_gb:.2f} GB")
    print(f"- KV Cache alone: {result.kv_cache_gb:.2f} GB")
    print(f"- Recommended GPU: {result.recommended_gpu_memory_gb} GB")


def example_quantization_comparison():
    """Example: Compare different quantization levels."""
    print_separator("Example 6: Quantization Impact")
    
    model_id = "meta-llama/Llama-2-13b-hf"
    precisions = ["fp32", "fp16", "int8", "int4"]
    
    print(f"\nModel: {model_id}")
    print(f"Batch=1, Seq=2048")
    print(f"\n{'Precision':<10} {'Total GB':<12} {'Weights GB':<12} {'Reduction':<10}")
    print("-" * 45)
    
    base_memory = None
    for precision in precisions:
        result = calculate_memory(
            model_id,
            batch_size=1,
            seq_length=2048,
            precision=precision
        )
        
        if base_memory is None:
            base_memory = result.total_memory_gb
            reduction = "baseline"
        else:
            reduction = f"{base_memory/result.total_memory_gb:.2f}x"
        
        print(f"{precision:<10} {result.total_memory_gb:<12.2f} {result.weight_memory_gb:<12.2f} {reduction:<10}")


def main():
    """Run all examples."""
    examples = [
        example_huggingface_model,
        example_custom_parameters,
        example_custom_config,
        example_tensor_parallelism,
        example_moe_model,
        example_quantization_comparison,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("(This might be due to missing model access or network issues)")
    
    print("\n" + "="*60)
    print(" Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()