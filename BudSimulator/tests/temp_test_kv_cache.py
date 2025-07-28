import unittest
import json
from src.bud_models import ModelMemoryCalculator
from src.db.model_manager import ModelManager

class TempTestKVCache(unittest.TestCase):

    def setUp(self):
        self.model_manager = ModelManager()

    def test_llama_3_1_deep_dive(self):
        """
        Deep dive diagnostics for KV cache calculation.
        """
        model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        config = self.model_manager.get_model_config(model_id)
        self.assertIsNotNone(config, f"Config for model {model_id} not found.")

        batch_size = 1
        seq_length = 15000 + 10
        precision = 'fp16'
        expected_kv_cache_gb = 1.8311

        # --- Enhanced Standalone Calculation ---
        print("\n--- Deep Dive Diagnostics ---")
        bytes_per_value = ModelMemoryCalculator.PRECISION_BYTES.get(precision.lower(), 2)
        print(f"Bytes per Value: {bytes_per_value}")
        
        num_layers = config.get('num_hidden_layers', 12)
        print(f"Num Layers: {num_layers}")
        
        hidden_size = config.get('hidden_size', 768)
        print(f"Hidden Size: {hidden_size}")
        
        num_attention_heads = config.get('num_attention_heads', 12)
        print(f"Num Attention Heads: {num_attention_heads}")
        
        num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
        print(f"Num KV Heads: {num_key_value_heads}")
        
        head_dim = config.get('head_dim', hidden_size // num_attention_heads)
        print(f"Head Dim: {head_dim}")
        
        # RoPE Scaling
        rope_scaling_factor = 1.0
        if 'rope_scaling' in config and config['rope_scaling'] is not None:
            rope_scaling_factor = config['rope_scaling'].get('factor', 1.0)
        print(f"RoPE Scaling Factor: {rope_scaling_factor}")

        # Final Calculation
        kv_elements = batch_size * seq_length * num_layers * num_key_value_heads * head_dim * 2
        print(f"KV Elements (before RoPE): {kv_elements}")
        
        kv_elements /= rope_scaling_factor
        print(f"KV Elements (after RoPE): {kv_elements}")
        
        calculated_kv_cache_gb = (kv_elements * bytes_per_value) / 1e9
        print(f"Final Calculated GB: {calculated_kv_cache_gb}")
        print("---------------------------\n")
        # --- End of Calculation ---

        self.assertAlmostEqual(expected_kv_cache_gb, calculated_kv_cache_gb, places=2, 
                             msg=f"KV cache for {model_id} is {calculated_kv_cache_gb:.4f} GB, but expected {expected_kv_cache_gb:.4f} GB.")

if __name__ == '__main__':
    unittest.main() 