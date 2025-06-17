"""
Example flow for handling gated models in BudSimulator

This demonstrates the complete flow from detecting a gated model
to successfully calculating memory after user provides config.
"""

import asyncio
import json
from typing import Dict, Any

# Simulated API calls

async def validate_model(model_uri: str) -> Dict[str, Any]:
    """Simulate model validation API call"""
    # For gated models like Llama 3
    if "llama-3" in model_uri.lower() or "llama-2" in model_uri.lower():
        return {
            "valid": False,
            "error": "Model is gated and requires access approval",
            "error_code": "MODEL_GATED",
            "model_id": model_uri,
            "requires_config": True,
            "config_submission_url": "/api/models/config/submit"
        }
    
    # For non-gated models
    return {
        "valid": True,
        "error": None
    }

async def submit_config(model_uri: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate config submission API call"""
    # Validate required fields
    required_fields = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']
    missing_fields = [f for f in required_fields if f not in config]
    
    if missing_fields:
        return {
            "success": False,
            "message": "Invalid configuration: missing required fields",
            "error_code": "INVALID_CONFIG",
            "missing_fields": missing_fields
        }
    
    # Success response
    return {
        "success": True,
        "message": "Configuration saved and validated successfully",
        "model_id": model_uri,
        "validation": {
            "valid": True,
            "model_type": "decoder-only",
            "attention_type": "gqa",
            "parameter_count": 70000000000
        }
    }

async def calculate_memory(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate memory calculation API call"""
    return {
        "model_type": "decoder-only",
        "attention_type": "gqa",
        "precision": params.get("precision", "fp16"),
        "parameter_count": 70000000000,
        "memory_breakdown": {
            "weight_memory_gb": 130.0,
            "kv_cache_gb": 8.0,
            "activation_memory_gb": 2.5,
            "state_memory_gb": 0.0,
            "image_memory_gb": 0.0,
            "extra_work_gb": 1.5
        },
        "total_memory_gb": 142.0,
        "recommendations": {
            "recommended_gpu_memory_gb": 160,
            "can_fit_24gb_gpu": False,
            "can_fit_80gb_gpu": False,
            "min_gpu_memory_gb": 142.0
        }
    }

async def user_flow_example():
    """Example of complete user flow"""
    
    print("=== BudSimulator Gated Model Flow Example ===\n")
    
    # Step 1: User enters a gated model
    model_uri = "meta-llama/Llama-3-70b"
    calculation_params = {
        "model_id": model_uri,
        "precision": "fp16",
        "batch_size": 1,
        "seq_length": 4096
    }
    
    print(f"1. User wants to calculate memory for: {model_uri}")
    print(f"   Parameters: batch_size={calculation_params['batch_size']}, seq_length={calculation_params['seq_length']}")
    
    # Step 2: Validate the model
    print("\n2. Validating model...")
    validation_result = await validate_model(model_uri)
    
    if not validation_result["valid"]:
        if validation_result.get("error_code") == "MODEL_GATED":
            print(f"   ❌ {validation_result['error']}")
            print(f"   ⚠️  Model requires configuration input")
            
            # Step 3: User provides config
            print("\n3. User obtains and pastes config.json...")
            user_config = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8192,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "intermediate_size": 28672,
                "vocab_size": 128256,
                "max_position_embeddings": 8192,
                "rope_theta": 500000.0,
                "hidden_act": "silu"
            }
            print(f"   📋 Config provided with {len(user_config)} fields")
            
            # Step 4: Submit config
            print("\n4. Submitting configuration...")
            submit_result = await submit_config(model_uri, user_config)
            
            if submit_result["success"]:
                print(f"   ✅ {submit_result['message']}")
                print(f"   Model type: {submit_result['validation']['model_type']}")
                print(f"   Attention type: {submit_result['validation']['attention_type']}")
                print(f"   Parameters: {submit_result['validation']['parameter_count']:,}")
                
                # Step 5: Continue with original calculation
                print("\n5. Automatically continuing with memory calculation...")
                calc_result = await calculate_memory(model_uri, calculation_params)
                
                print(f"   ✅ Calculation complete!")
                print(f"\n   Memory Breakdown:")
                print(f"   - Model weights: {calc_result['memory_breakdown']['weight_memory_gb']:.1f} GB")
                print(f"   - KV cache: {calc_result['memory_breakdown']['kv_cache_gb']:.1f} GB")
                print(f"   - Activations: {calc_result['memory_breakdown']['activation_memory_gb']:.1f} GB")
                print(f"   - Total: {calc_result['total_memory_gb']:.1f} GB")
                print(f"\n   Recommendations:")
                print(f"   - Minimum GPU memory: {calc_result['recommendations']['min_gpu_memory_gb']:.0f} GB")
                print(f"   - Recommended GPU: {calc_result['recommendations']['recommended_gpu_memory_gb']} GB")
                
                print("\n6. Future requests for this model will use the saved config automatically! 🎉")
            else:
                print(f"   ❌ Config submission failed: {submit_result['message']}")
    else:
        # Non-gated model, proceed directly
        print("   ✅ Model validated successfully")
        calc_result = await calculate_memory(model_uri, calculation_params)
        print("   ✅ Calculation complete!")

if __name__ == "__main__":
    asyncio.run(user_flow_example()) 