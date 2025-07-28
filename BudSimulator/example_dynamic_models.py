#!/usr/bin/env python3
"""
Example script demonstrating dynamic model loading in GenZ Simulator.
"""

import sys
from pathlib import Path

# Add GenZ to path
sys.path.insert(0, str(Path(__file__).parent / 'GenZ'))

# Import GenZ models - this automatically enables dynamic loading
from Models import MODEL_DICT, import_model_from_hf

def main():
    print("GenZ Dynamic Model Loading Example")
    print("=" * 50)
    
    # 1. Access a static model
    print("\n1. Accessing a static model:")
    model = MODEL_DICT.get_model("mistral_7b")
    if model:
        print(f"   Model: {model.model}")
        print(f"   Hidden size: {model.hidden_size}")
        print(f"   Layers: {model.num_decoder_layers}")
        print(f"   Attention heads: {model.num_attention_heads}")
        print(f"   KV heads: {model.num_key_value_heads} (GQA)")
    
    # 2. Import a new model from HuggingFace
    print("\n2. Importing a new model from HuggingFace:")
    model_id = "microsoft/phi-2"
    print(f"   Importing {model_id}...")
    
    success = import_model_from_hf(model_id)
    if success:
        print(f"   ✓ Successfully imported {model_id}")
        
        # Now access it
        model = MODEL_DICT.get_model(model_id)
        if model:
            print(f"   Model: {model.model}")
            print(f"   Hidden size: {model.hidden_size}")
            print(f"   Layers: {model.num_decoder_layers}")
    else:
        print(f"   ✗ Failed to import {model_id}")
    
    # 3. List available models
    print("\n3. Available models:")
    all_models = MODEL_DICT.list_models()
    print(f"   Total models: {len(all_models)}")
    
    # Show a few examples
    print("   Examples:")
    for model_name in ["llama_7b", "gpt2", "microsoft/phi-2", "mixtral_8x7b"]:
        if model_name in all_models:
            print(f"   - {model_name} ✓")
    
    # 4. Use the model manager directly for advanced queries
    print("\n4. Advanced model queries:")
    try:
        from src.db import ModelManager
        manager = ModelManager()
        
        # Find all GQA models
        gqa_models = manager.list_models(attention_type="gqa")
        print(f"   GQA models in database: {len(gqa_models)}")
        
        # Find large models
        large_models = manager.list_models(min_params=10_000_000_000)
        print(f"   Large models (>10B) in database: {len(large_models)}")
        
    except Exception as e:
        print(f"   Database queries not available: {e}")
    
    # 5. Memory analysis example
    print("\n5. Memory analysis for a model:")
    model = MODEL_DICT.get_model("mistral_7b")
    if model:
        # Estimate memory for different sequence lengths
        from src.bud_models import estimate_memory
        
        # Convert GenZ config to dict for memory estimation
        config_dict = {
            'vocab_size': model.vocab_size,
            'hidden_size': model.hidden_size,
            'intermediate_size': model.intermediate_size,
            'num_hidden_layers': model.num_decoder_layers,
            'num_attention_heads': model.num_attention_heads,
            'num_key_value_heads': model.num_key_value_heads,
            'model_type': 'mistral'
        }
        
        for seq_len in [2048, 8192, 32768]:
            result = estimate_memory(config_dict, seq_length=seq_len)
            print(f"   {seq_len:5d} tokens: {result.total_memory_gb:.1f} GB")

if __name__ == "__main__":
    main() 