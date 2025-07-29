#!/usr/bin/env python3
"""Advanced examples for comparing and analyzing models."""

from llm_memory_calculator import (
    compare_models,
    analyze_attention_efficiency,
    estimate_max_batch_size,
    calculate_memory
)


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def example_compare_models():
    """Compare memory requirements across different models."""
    print_separator("Model Comparison")
    
    # Compare popular 7B models
    models = [
        # "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "google/gemma-7b",
    ]
    
    print("\nComparing 7B-class models at 4K context:")
    
    # Header
    print(f"\n{'Model':<30} {'Params':<10} {'Total GB':<10} {'Weights':<10} {'KV Cache':<10}")
    print("-" * 70)
    
    for model_id in models:
        try:
            result = calculate_memory(model_id, seq_length=4096)
            model_name = model_id.split('/')[-1]
            params = f"{result.parameter_count/1e9:.1f}B"
            
            print(f"{model_name:<30} {params:<10} "
                  f"{result.total_memory_gb:<10.2f} "
                  f"{result.weight_memory_gb:<10.2f} "
                  f"{result.kv_cache_gb:<10.2f}")
        except Exception as e:
            print(f"{model_id}: Error - {e}")


def example_attention_efficiency():
    """Analyze how different attention mechanisms scale with context."""
    print_separator("Attention Efficiency Analysis")
    
    # Models with different attention types
    models = {
        "Standard MHA": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,  # MHA
            "vocab_size": 32000,
        },
        "GQA (8 groups)": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,   # GQA with 8:1 compression
            "vocab_size": 32000,
        },
        "MQA": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,   # MQA
            "vocab_size": 32000,
        },
        "MLA (DeepSeek-style)": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "q_lora_rank": 1536,        # MLA parameters
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "vocab_size": 32000,
        }
    }
    
    seq_lengths = [2048, 8192, 32768, 131072]
    
    print("\nKV Cache Memory (GB) at different sequence lengths:")
    print(f"\n{'Attention Type':<20}", end="")
    for seq_len in seq_lengths:
        print(f"{seq_len:>12}", end="")
    print("\n" + "-" * 70)
    
    for name, config in models.items():
        print(f"{name:<20}", end="")
        for seq_len in seq_lengths:
            result = calculate_memory(config, seq_length=seq_len)
            print(f"{result.kv_cache_gb:>12.2f}", end="")
        print()
    
    # Show compression ratios
    print("\n\nCompression vs Standard MHA:")
    mha_baseline = {}
    for seq_len in seq_lengths:
        result = calculate_memory(models["Standard MHA"], seq_length=seq_len)
        mha_baseline[seq_len] = result.kv_cache_gb
    
    for name, config in models.items():
        if name == "Standard MHA":
            continue
        print(f"\n{name}:")
        for seq_len in seq_lengths:
            result = calculate_memory(config, seq_length=seq_len)
            compression = mha_baseline[seq_len] / result.kv_cache_gb
            print(f"  {seq_len:,} tokens: {compression:.1f}x compression")


def example_gpu_sizing():
    """Find optimal GPU configurations for different models."""
    print_separator("GPU Sizing Recommendations")
    
    models = [
        ("meta-llama/Llama-2-7b-hf", 4096),
        ("meta-llama/Llama-2-13b-hf", 4096),
        ("meta-llama/Llama-2-70b-hf", 4096),
        ("mistralai/Mixtral-8x7B-v0.1", 32768),
    ]
    
    gpu_sizes = [24, 48, 80, 160]  # Common GPU memory sizes
    
    print("\nMaximum batch size by GPU and model:")
    print(f"\n{'Model':<30} {'Context':<10}", end="")
    for gpu in gpu_sizes:
        print(f"{gpu}GB:>8", end="")
    print("\n" + "-" * 75)
    
    for model_id, context in models:
        model_name = model_id.split('/')[-1]
        print(f"{model_name:<30} {context:<10}", end="")
        
        for gpu_memory in gpu_sizes:
            try:
                max_batch = estimate_max_batch_size(
                    model_id,
                    gpu_memory_gb=gpu_memory,
                    seq_length=context,
                    precision="fp16"
                )
                print(f"{max_batch:>8}", end="")
            except:
                print(f"{'N/A':>8}", end="")
        print()


def example_long_context_analysis():
    """Analyze memory scaling for long-context scenarios."""
    print_separator("Long Context Analysis")
    
    model_id = "mistralai/Mistral-7B-v0.1"
    
    print(f"\nModel: {model_id}")
    print("Analyzing memory requirements for different context lengths...")
    
    # Analyze increasing context lengths
    contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    print(f"\n{'Context':<12} {'Total GB':<10} {'KV Cache':<10} {'KV %':<8} {'GPU Needed':<12}")
    print("-" * 60)
    
    for context in contexts:
        try:
            result = calculate_memory(model_id, seq_length=context)
            kv_percent = (result.kv_cache_gb / result.total_memory_gb) * 100
            
            print(f"{context:<12,} {result.total_memory_gb:<10.2f} "
                  f"{result.kv_cache_gb:<10.2f} {kv_percent:<8.1f} "
                  f"{result.recommended_gpu_memory_gb}GB")
        except Exception as e:
            print(f"{context:<12,} Error: {e}")
    
    # Show efficiency analysis
    print("\n" + "="*60)
    efficiency = analyze_attention_efficiency(
        model_id,
        seq_lengths=[2048, 8192, 32768, 131072]
    )
    
    print(f"\nMemory per token: {efficiency['memory_per_token_bytes']:,} bytes")
    print(f"Efficiency rating: {efficiency['efficiency_rating']}")


def example_training_memory():
    """Calculate memory requirements for training vs inference."""
    print_separator("Training vs Inference Memory")
    
    model_configs = [
        ("7B Model", {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_parameters": 7_000_000_000,
        }),
        ("13B Model", {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "vocab_size": 32000,
            "num_parameters": 13_000_000_000,
        }),
    ]
    
    print("\nComparing inference vs training memory requirements:")
    print("(Batch size=1, Sequence length=2048, FP16)")
    
    print(f"\n{'Model':<15} {'Inference':<15} {'Training':<15} {'Increase':<10}")
    print("-" * 55)
    
    for name, config in model_configs:
        # Inference memory
        inference = calculate_memory(
            config,
            batch_size=1,
            seq_length=2048,
            include_gradients=False
        )
        
        # Training memory (with gradients)
        training = calculate_memory(
            config,
            batch_size=1,
            seq_length=2048,
            include_gradients=True
        )
        
        increase = training.total_memory_gb / inference.total_memory_gb
        
        print(f"{name:<15} {inference.total_memory_gb:<15.2f} "
              f"{training.total_memory_gb:<15.2f} {increase:<10.2f}x")


def main():
    """Run all examples."""
    examples = [
        example_compare_models,
        example_attention_efficiency,
        example_gpu_sizing,
        example_long_context_analysis,
        example_training_memory,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("(This might be due to missing model access or network issues)")
    
    print("\n" + "="*60)
    print(" Advanced examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()