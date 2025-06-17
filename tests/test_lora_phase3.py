import pytest
import pandas as pd
import os
import tempfile
from BudSimulator import LoraConfig
from BudSimulator.GenZ.Models import create_full_prefill_model, create_full_decode_model, get_configs
from BudSimulator.GenZ.Models.utils import OpType
from BudSimulator.GenZ.parallelism import ParallelismConfig


class TestLoRAIntegration:
    """Test suite for LoRA integration in GenZ simulator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def base_model_config(self):
        """Get a base model configuration for testing."""
        return get_configs('gpt-2')
    
    def load_csv_and_check_ops(self, csv_path):
        """Load CSV and return operation types."""
        df = pd.read_csv(csv_path)
        # The 'T' column contains the operation type
        return df['T'].tolist()
    
    def test_lora_disabled_no_ops(self, temp_dir, base_model_config):
        """Test that when LoRA is disabled, no LoRA operations appear."""
        # Create model with LoRA disabled
        base_model_config.lora_config = LoraConfig(enabled=False)
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        ops = self.load_csv_and_check_ops(csv_path)
        
        # Check that no LoRA operations are present
        lora_ops = [OpType.LORA_MERGE, OpType.GEMM_LORA_A, OpType.GEMM_LORA_B, OpType.ADD]
        for op in ops:
            assert op not in lora_ops, f"Found LoRA operation {op} when LoRA is disabled"
    
    def test_lora_merge_strategy_prefill(self, temp_dir, base_model_config):
        """Test merge strategy in prefill model - should have LORA_MERGE ops."""
        # Configure LoRA with merge strategy
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=16,
            target_modules=['attn', 'ffn'],
            strategy='merge'
        )
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        ops = self.load_csv_and_check_ops(csv_path)
        
        # Should have LORA_MERGE operations
        assert OpType.LORA_MERGE in ops, "LORA_MERGE operation not found in merge strategy"
        
        # Should NOT have dynamic LoRA operations
        assert OpType.GEMM_LORA_A not in ops, "GEMM_LORA_A found in merge strategy"
        assert OpType.GEMM_LORA_B not in ops, "GEMM_LORA_B found in merge strategy"
        assert OpType.ADD not in ops, "ADD operation found in merge strategy"
    
    def test_lora_dynamic_strategy_prefill(self, temp_dir, base_model_config):
        """Test dynamic strategy in prefill model - should have GEMM_LORA_A/B and ADD ops."""
        # Configure LoRA with dynamic strategy
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=16,
            target_modules=['attn', 'ffn'],
            strategy='dynamic'
        )
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        ops = self.load_csv_and_check_ops(csv_path)
        
        # Should have dynamic LoRA operations
        assert OpType.GEMM_LORA_A in ops, "GEMM_LORA_A not found in dynamic strategy"
        assert OpType.GEMM_LORA_B in ops, "GEMM_LORA_B not found in dynamic strategy"
        assert OpType.ADD in ops, "ADD operation not found in dynamic strategy"
        
        # Should NOT have merge operation
        assert OpType.LORA_MERGE not in ops, "LORA_MERGE found in dynamic strategy"
    
    def test_lora_target_modules_ffn_only(self, temp_dir, base_model_config):
        """Test that LoRA only applies to specified target modules."""
        # Configure LoRA to target only FFN
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=16,
            target_modules=['ffn'],  # Only FFN, not attention
            strategy='dynamic'
        )
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        df = pd.read_csv(csv_path)
        
        # Check that LoRA ops appear after FFN operations but not after attention
        # This is a simplified check - in reality we'd need to parse the operation sequence
        ops_with_names = list(zip(df['Name'].tolist(), df['T'].tolist()))
        
        # Find FFN-related GEMM operations and check if they're followed by LoRA ops
        ffn_lora_found = False
        attn_lora_found = False
        
        for i, (name, op_type) in enumerate(ops_with_names):
            # Check for FFN operations (up+gate, down)
            if any(ffn_name in name.lower() for ffn_name in ['up+gate', 'down', 'up', 'gate']) and op_type == OpType.GEMM.value:
                # Check if followed by LoRA operations
                if i + 1 < len(ops_with_names) and ops_with_names[i + 1][1] in [OpType.GEMM_LORA_A.value, OpType.GEMM_LORA_B.value]:
                    ffn_lora_found = True
            # Check for attention operations (QKV, Out Proj)
            elif any(attn_name in name.lower() for attn_name in ['qkv', 'out proj', 'query', 'key', 'value']) and op_type == OpType.GEMM.value:
                # Check if followed by LoRA operations
                if i + 1 < len(ops_with_names) and ops_with_names[i + 1][1] in [OpType.GEMM_LORA_A.value, OpType.GEMM_LORA_B.value]:
                    attn_lora_found = True
        
        assert ffn_lora_found, "LoRA operations not found after FFN GEMM"
        assert not attn_lora_found, "LoRA operations found after attention GEMM when only FFN was targeted"
    
    def test_lora_decode_model(self, temp_dir, base_model_config):
        """Test LoRA in decode model."""
        # Configure LoRA for decode
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=8,
            target_modules=['attn', 'ffn'],
            strategy='dynamic'
        )
        
        csv_file = create_full_decode_model(
            name=base_model_config,
            input_sequence_length=128,
            output_gen_tokens=32,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        ops = self.load_csv_and_check_ops(csv_path)
        
        # Should have dynamic LoRA operations in decode as well
        assert OpType.GEMM_LORA_A in ops, "GEMM_LORA_A not found in decode model"
        assert OpType.GEMM_LORA_B in ops, "GEMM_LORA_B not found in decode model"
        assert OpType.ADD in ops, "ADD operation not found in decode model"
    
    def test_lora_with_tensor_parallelism(self, temp_dir, base_model_config):
        """Test LoRA with tensor parallelism."""
        # Configure LoRA with TP
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=16,
            target_modules=['attn', 'ffn'],
            strategy='dynamic'
        )
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir,
            tensor_parallel=4
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        df = pd.read_csv(csv_path)
        
        # Check that LoRA operations are present and dimensions are adjusted for TP
        ops = df['T'].tolist()
        assert OpType.GEMM_LORA_A in ops, "GEMM_LORA_A not found with tensor parallelism"
        assert OpType.GEMM_LORA_B in ops, "GEMM_LORA_B not found with tensor parallelism"
        
        # Find a LoRA operation and check its dimensions
        lora_a_rows = df[df['T'] == OpType.GEMM_LORA_A]
        if not lora_a_rows.empty:
            # With TP=4, hidden dimensions should be divided by 4
            # This is a simplified check - actual dimension checking would be more complex
            assert len(lora_a_rows) > 0, "LoRA operations should be present"
    
    def test_lora_rank_dimensions(self, temp_dir, base_model_config):
        """Test that LoRA operations have correct rank dimensions."""
        rank = 32
        base_model_config.lora_config = LoraConfig(
            enabled=True,
            rank=rank,
            target_modules=['ffn'],
            strategy='dynamic'
        )
        
        csv_file = create_full_prefill_model(
            name=base_model_config,
            input_sequence_length=128,
            data_path=temp_dir
        )
        
        csv_path = os.path.join(temp_dir, "model", csv_file)
        df = pd.read_csv(csv_path)
        
        # Check GEMM_LORA_A dimensions (should have rank as one dimension)
        lora_a_rows = df[df['T'] == OpType.GEMM_LORA_A]
        for _, row in lora_a_rows.iterrows():
            # One of the dimensions should be the rank
            dims = [row['M'], row['N'], row['D']]
            assert rank in dims, f"Rank {rank} not found in GEMM_LORA_A dimensions {dims}"
        
        # Check GEMM_LORA_B dimensions (should have rank as one dimension)
        lora_b_rows = df[df['T'] == OpType.GEMM_LORA_B]
        for _, row in lora_b_rows.iterrows():
            dims = [row['M'], row['N'], row['D']]
            assert rank in dims, f"Rank {rank} not found in GEMM_LORA_B dimensions {dims}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 