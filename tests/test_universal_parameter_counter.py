"""
Test suite for UniversalParameterCounter integration.
Ensures accuracy improvements and no regressions.
"""

import pytest
import json
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bud_models import ModelMemoryCalculator

class TestUniversalParameterCounter:
    """Test the UniversalParameterCounter integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ModelMemoryCalculator()
        
        # Test configurations with known parameter counts
        self.test_configs = {
            "qwen_8b": {
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": False,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "vocab_size": 151936,
                "tie_word_embeddings": False,
                "rope_theta": 1000000,
                "expected_params": 8_190_726_144  # ~8.2B
            },
            "llama_70b": {
                "architectures": ["LlamaForCausalLM"],
                "hidden_act": "silu", 
                "hidden_size": 8192,
                "intermediate_size": 28672,
                "num_attention_heads": 64,
                "num_hidden_layers": 80,
                "num_key_value_heads": 8,
                "vocab_size": 128256,
                "tie_word_embeddings": False,
                "rope_theta": 500000.0,
                "expected_params": 70_600_000_000  # ~70.6B
            },
            "deepseek_v3": {
                "architectures": ["DeepseekV3ForCausalLM"],
                "attention_bias": False,
                "hidden_act": "silu",
                "hidden_size": 7168,
                "intermediate_size": 18432,
                "kv_lora_rank": 512,
                "moe_intermediate_size": 2048,
                "moe_layer_freq": 1,
                "n_routed_experts": 256,
                "n_shared_experts": 1,
                "num_attention_heads": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 61,
                "num_key_value_heads": 128,
                "num_nextn_predict_layers": 1,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "vocab_size": 129280,
                "tie_word_embeddings": False,
                "first_k_dense_replace": 3,
                "expected_params": 685_000_000_000  # ~685B
            },
            "simple_transformer": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 30000,
                "tie_word_embeddings": True,
                "expected_params": 108_000_000  # ~108M (more accurate estimate)
            }
        }
    
    def test_parameter_calculation_accuracy(self):
        """Test parameter calculation accuracy against known values."""
        for model_name, config in self.test_configs.items():
            expected = config.pop('expected_params')
            calculated = self.calculator._calculate_transformer_params(config)
            
            # Allow up to 5% error for complex models
            error_rate = abs(calculated - expected) / expected
            print(f"{model_name}: Expected {expected:,}, Got {calculated:,}, Error: {error_rate:.2%}")
            
            # Different error tolerances for different model types
            if 'moe' in model_name or 'deepseek' in model_name:
                assert error_rate < 0.10, f"{model_name}: Error rate {error_rate:.2%} too high"
            else:
                assert error_rate < 0.05, f"{model_name}: Error rate {error_rate:.2%} too high"
    
    def test_moe_parameter_calculation(self):
        """Test MoE-specific parameter calculations."""
        moe_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "moe_intermediate_size": 1408,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "vocab_size": 102400,
            "hidden_act": "silu",
            "tie_word_embeddings": False
        }
        
        params = self.calculator._calculate_transformer_params(moe_config)
        assert params > 0, "MoE parameter calculation should return positive value"
        
        # Should be significantly larger than a dense model with same hidden_size
        dense_config = moe_config.copy()
        dense_config.pop('n_routed_experts', None)
        dense_config.pop('n_shared_experts', None) 
        dense_config.pop('moe_intermediate_size', None)
        dense_config.pop('num_experts_per_tok', None)
        
        dense_params = self.calculator._calculate_transformer_params(dense_config)
        assert params > dense_params * 5, "MoE should have significantly more parameters"
    
    def test_attention_type_variants(self):
        """Test different attention mechanisms."""
        base_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 50000,
            "tie_word_embeddings": True
        }
        
        # MHA (Multi-Head Attention)
        mha_config = base_config.copy()
        mha_params = self.calculator._calculate_transformer_params(mha_config)
        
        # GQA (Grouped-Query Attention)
        gqa_config = base_config.copy()
        gqa_config["num_key_value_heads"] = 4
        gqa_params = self.calculator._calculate_transformer_params(gqa_config)
        
        # MQA (Multi-Query Attention)
        mqa_config = base_config.copy()
        mqa_config["num_key_value_heads"] = 1
        mqa_params = self.calculator._calculate_transformer_params(mqa_config)
        
        # MLA (Multi-head Latent Attention)
        mla_config = base_config.copy()
        mla_config.update({
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128
        })
        mla_params = self.calculator._calculate_transformer_params(mla_config)
        
        # Verify relationships
        assert mha_params > gqa_params > mqa_params, "Parameter count should decrease: MHA > GQA > MQA"
        assert all(p > 0 for p in [mha_params, gqa_params, mqa_params, mla_params]), "All should be positive"
    
    def test_activation_function_handling(self):
        """Test different activation function parameter calculations."""
        base_config = {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "vocab_size": 30000,
            "tie_word_embeddings": True
        }
        
        # Test gated activations (should use 3 matrices)
        gated_activations = ["silu", "swiglu", "geglu", "swish"]
        ungated_activations = ["gelu", "relu", "tanh"]
        
        gated_params = []
        for act in gated_activations:
            config = base_config.copy()
            config["hidden_act"] = act
            params = self.calculator._calculate_transformer_params(config)
            gated_params.append(params)
        
        ungated_params = []
        for act in ungated_activations:
            config = base_config.copy()
            config["hidden_act"] = act
            params = self.calculator._calculate_transformer_params(config)
            ungated_params.append(params)
        
        # All gated should be equal and larger than ungated
        assert len(set(gated_params)) == 1, "All gated activations should have same param count"
        assert len(set(ungated_params)) == 1, "All ungated activations should have same param count"
        assert gated_params[0] > ungated_params[0], "Gated activations should use more parameters"
    
    def test_embedding_configuration(self):
        """Test different embedding configurations."""
        base_config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 30000
        }
        
        # Tied embeddings
        tied_config = base_config.copy()
        tied_config["tie_word_embeddings"] = True
        tied_params = self.calculator._calculate_transformer_params(tied_config)
        
        # Untied embeddings
        untied_config = base_config.copy()
        untied_config["tie_word_embeddings"] = False
        untied_params = self.calculator._calculate_transformer_params(untied_config)
        
        # Untied should have more parameters (output embedding)
        vocab_size = base_config["vocab_size"]
        hidden_size = base_config["hidden_size"]
        expected_diff = vocab_size * hidden_size
        
        actual_diff = untied_params - tied_params
        assert abs(actual_diff - expected_diff) < expected_diff * 0.1, "Embedding difference should match vocab*hidden"
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Empty config
        empty_config = {}
        params = self.calculator._calculate_transformer_params(empty_config)
        assert params > 0, "Should handle empty config gracefully"
        
        # Missing required fields
        minimal_config = {"vocab_size": 1000}
        params = self.calculator._calculate_transformer_params(minimal_config)
        assert params > 0, "Should handle minimal config"
        
        # Invalid values - should still return reasonable defaults
        invalid_config = {
            "hidden_size": -100,
            "num_hidden_layers": 0,
            "vocab_size": 1000
        }
        params = self.calculator._calculate_transformer_params(invalid_config)
        # For truly invalid configs, we expect the fallback to handle it reasonably
        # The fallback should return at least the default 1M params
        assert params >= 1000000, "Should handle invalid values by returning fallback minimum"
    
    def test_backward_compatibility(self):
        """Test that existing model calculations still work."""
        # Test with a typical config that should work with old system
        typical_config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "hidden_act": "silu"
        }
        
        params = self.calculator._calculate_transformer_params(typical_config)
        expected_range = (6_000_000_000, 8_000_000_000)  # 6-8B range
        assert expected_range[0] <= params <= expected_range[1], f"Params {params:,} not in expected range"
    
    def test_memory_calculation_integration(self):
        """Test that memory calculations work with new parameter counter."""
        config = self.test_configs["simple_transformer"].copy()
        config.pop('expected_params', None)  # Remove safely if it exists
        
        # Test full memory calculation
        report = self.calculator.calculate_total_memory(
            config=config,
            batch_size=1,
            seq_length=2048,
            precision='fp16'
        )
        
        assert report.parameter_count > 0, "Should calculate parameter count"
        assert report.weight_memory_gb > 0, "Should calculate weight memory"
        assert report.total_memory_gb > 0, "Should calculate total memory"
        assert report.model_type is not None, "Should detect model type"

if __name__ == "__main__":
    # Run basic tests
    test_instance = TestUniversalParameterCounter()
    test_instance.setup_method()
    
    print("Running parameter calculation accuracy tests...")
    test_instance.test_parameter_calculation_accuracy()
    
    print("Running MoE parameter tests...")
    test_instance.test_moe_parameter_calculation()
    
    print("Running attention variant tests...")
    test_instance.test_attention_type_variants()
    
    print("Running activation function tests...")
    test_instance.test_activation_function_handling()
    
    print("Running embedding configuration tests...")
    test_instance.test_embedding_configuration()
    
    print("Running error handling tests...")
    test_instance.test_error_handling()
    
    print("Running backward compatibility tests...")
    test_instance.test_backward_compatibility()
    
    print("Running memory calculation integration tests...")
    test_instance.test_memory_calculation_integration()
    
    print("✅ All tests passed!") 