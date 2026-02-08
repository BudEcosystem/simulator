"""
Test suite for TrainingMemoryCalculator.

TDD approach: Write tests first, then implement the calculator.
These tests validate training memory calculations for:
- Weight memory
- Gradient memory (full, lora, freeze methods)
- Optimizer memory (adamw, adamw_8bit, sgd, galore, apollo)
- Activation memory (with/without gradient checkpointing)
- DeepSpeed ZeRO sharding effects
"""

import pytest
from typing import Dict, Any

# Import will fail until we implement the module
try:
    from llm_memory_calculator.training import (
        TrainingMemoryCalculator,
        TrainingMemoryEstimate,
        TrainingMethod,
        OptimizerType,
        DeepSpeedStage,
    )
    TRAINING_MODULE_AVAILABLE = True
except ImportError:
    TRAINING_MODULE_AVAILABLE = False


# Test fixtures
@pytest.fixture
def llama_8b_config() -> Dict[str, Any]:
    """Llama 3.1 8B model config."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "head_dim": 128,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "num_parameters": 8_030_261_248,  # ~8B
    }


@pytest.fixture
def llama_70b_config() -> Dict[str, Any]:
    """Llama 3.1 70B model config."""
    return {
        "model_type": "llama",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "head_dim": 128,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "num_parameters": 70_553_706_496,  # ~70B
    }


@pytest.fixture
def qwen2_7b_config() -> Dict[str, Any]:
    """Qwen2 7B model config."""
    return {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "num_parameters": 7_615_616_000,  # ~7.6B
    }


@pytest.fixture
def calculator() -> "TrainingMemoryCalculator":
    """Create a TrainingMemoryCalculator instance."""
    if not TRAINING_MODULE_AVAILABLE:
        pytest.skip("Training module not yet implemented")
    return TrainingMemoryCalculator()


class TestTrainingMemoryCalculatorBasics:
    """Test basic calculator initialization and functionality."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_calculator_instantiation(self):
        """Test that calculator can be instantiated."""
        calc = TrainingMemoryCalculator()
        assert calc is not None

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_training_methods_enum(self):
        """Test TrainingMethod enum values."""
        assert TrainingMethod.FULL.value == "full"
        assert TrainingMethod.LORA.value == "lora"
        assert TrainingMethod.FREEZE.value == "freeze"
        assert TrainingMethod.QLORA.value == "qlora"

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimizer_types_enum(self):
        """Test OptimizerType enum values."""
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.ADAMW_8BIT.value == "adamw_8bit"
        assert OptimizerType.SGD.value == "sgd"
        assert OptimizerType.GALORE.value == "galore"
        assert OptimizerType.APOLLO.value == "apollo"

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_deepspeed_stages_enum(self):
        """Test DeepSpeedStage enum values."""
        assert DeepSpeedStage.NONE.value == "none"
        assert DeepSpeedStage.ZERO2.value == "zero2"
        assert DeepSpeedStage.ZERO3.value == "zero3"


class TestWeightMemory:
    """Test weight memory calculations."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_weight_memory_bf16_8b(self, calculator, llama_8b_config):
        """Test weight memory for 8B model in bf16."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
        )
        # 8B params × 2 bytes (bf16) + 8B params × 4 bytes (fp32 master copy) = 48 GB
        assert 46.0 < estimate.weight_memory_gb < 50.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_weight_memory_fp32_8b(self, calculator, llama_8b_config):
        """Test weight memory for 8B model in fp32."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="fp32",
            method="full",
        )
        # 8B params × 4 bytes = 32 GB
        assert 30.0 < estimate.weight_memory_gb < 34.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_weight_memory_int4_8b(self, calculator, llama_8b_config):
        """Test weight memory for 8B model quantized to int4."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="int4",
            method="qlora",
        )
        # QLoRA: 8B params × 0.516 bytes (NF4+DQ) + dequant buffer ≈ 7.9 GB
        assert 6.0 < estimate.weight_memory_gb < 10.0


class TestGradientMemory:
    """Test gradient memory calculations for different training methods."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_gradient_memory_full_finetuning(self, calculator, llama_8b_config):
        """Test gradient memory for full finetuning - same as weight memory."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
        )
        # Full finetuning: gradients for all params (fp32)
        # 8B params × 4 bytes = 32 GB
        assert 30.0 < estimate.gradient_memory_gb < 34.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_gradient_memory_lora(self, calculator, llama_8b_config):
        """Test gradient memory for LoRA - only adapter parameters."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
        )
        # LoRA: ~1-2% of full params -> ~160-320M params
        # Gradients: ~0.6-1.2 GB (much smaller than full)
        assert 0.1 < estimate.gradient_memory_gb < 2.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_gradient_memory_lora_rank_64(self, calculator, llama_8b_config):
        """Test gradient memory for LoRA with higher rank."""
        estimate_16 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
        )
        estimate_64 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=64,
        )
        # Higher rank should have ~4x more gradient memory
        assert estimate_64.gradient_memory_gb > estimate_16.gradient_memory_gb * 3

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_gradient_memory_freeze(self, calculator, llama_8b_config):
        """Test gradient memory for freeze method - only unfrozen layers."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="freeze",
            freeze_layers=28,  # Freeze all but last 4 layers
        )
        # Freeze: ~12.5% of params unfrozen (4/32 layers)
        # Gradients: ~4 GB
        assert 2.0 < estimate.gradient_memory_gb < 6.0


class TestOptimizerMemory:
    """Test optimizer state memory calculations."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimizer_adamw_full(self, calculator, llama_8b_config):
        """Test AdamW optimizer states - 2x parameter memory (momentum + variance)."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
        )
        # AdamW: 2 states × 8B params × 4 bytes = 64 GB
        assert 60.0 < estimate.optimizer_memory_gb < 68.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimizer_adamw_8bit(self, calculator, llama_8b_config):
        """Test AdamW 8-bit - ~50% reduction in optimizer memory."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw_8bit",
        )
        # AdamW 8-bit: 2 states × 8B params × 1 byte = 16 GB
        assert 14.0 < estimate.optimizer_memory_gb < 18.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimizer_sgd(self, calculator, llama_8b_config):
        """Test SGD - only momentum state."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="sgd",
        )
        # SGD: 1 state × 8B params × 4 bytes = 32 GB
        assert 30.0 < estimate.optimizer_memory_gb < 34.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimizer_lora_adamw(self, calculator, llama_8b_config):
        """Test AdamW with LoRA - optimizer only for adapter params."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
            optimizer="adamw",
        )
        # LoRA + AdamW: much smaller (only ~1-2% of params)
        assert 0.2 < estimate.optimizer_memory_gb < 4.0


class TestActivationMemory:
    """Test activation memory calculations."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_activation_no_checkpointing(self, calculator, llama_8b_config):
        """Test activation memory without gradient checkpointing."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            gradient_checkpointing=False,
        )
        # Without checkpointing: store all activations including attention scores
        # Per layer: hidden states + attention scores (seq×seq) + FFN intermediates
        # For Llama-8B with batch=4, seq=2048: ~40-70 GB
        assert 30.0 < estimate.activation_memory_gb < 80.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_activation_with_checkpointing(self, calculator, llama_8b_config):
        """Test activation memory with gradient checkpointing."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            gradient_checkpointing=True,
        )
        # With checkpointing: sqrt(num_layers) effective layers
        # ~58 GB / sqrt(32) = ~10 GB
        assert 5.0 < estimate.activation_memory_gb < 15.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_activation_scales_with_batch(self, calculator, llama_8b_config):
        """Test that activation memory scales linearly with batch size."""
        estimate_4 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            gradient_checkpointing=True,
        )
        estimate_8 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=8,
            seq_length=2048,
            precision="bf16",
            method="full",
            gradient_checkpointing=True,
        )
        # Should be ~2x
        ratio = estimate_8.activation_memory_gb / estimate_4.activation_memory_gb
        assert 1.8 < ratio < 2.2

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_activation_scales_with_seq_length(self, calculator, llama_8b_config):
        """Test that activation memory scales super-linearly with sequence length.

        Attention scores scale O(seq²) while hidden states scale O(seq).
        For 2x seq length, expect ~3-4x increase due to quadratic attention.
        """
        estimate_2k = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            gradient_checkpointing=True,
        )
        estimate_4k = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=4096,
            precision="bf16",
            method="full",
            gradient_checkpointing=True,
        )
        # Attention is O(seq²), so 2x seq → ~3-4x memory increase
        ratio = estimate_4k.activation_memory_gb / estimate_2k.activation_memory_gb
        assert 2.5 < ratio < 4.5


class TestDeepSpeedSharding:
    """Test DeepSpeed ZeRO memory sharding."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_zero2_optimizer_sharding(self, calculator, llama_8b_config):
        """Test ZeRO-2: optimizer states sharded across data parallel ranks."""
        estimate_no_ds = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
            deepspeed_stage=None,
            data_parallel=1,
        )
        estimate_zero2 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
            deepspeed_stage="zero2",
            data_parallel=4,
        )
        # ZeRO-2: optimizer memory divided by DP
        ratio = estimate_no_ds.optimizer_memory_gb / estimate_zero2.optimizer_memory_gb
        assert 3.5 < ratio < 4.5  # Should be ~4x reduction

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_zero3_full_sharding(self, calculator, llama_8b_config):
        """Test ZeRO-3: weights, gradients, and optimizer states all sharded."""
        estimate_no_ds = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
            deepspeed_stage=None,
            data_parallel=1,
        )
        estimate_zero3 = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
            deepspeed_stage="zero3",
            data_parallel=8,
        )
        # ZeRO-3: all memory divided by DP
        total_no_ds = (
            estimate_no_ds.weight_memory_gb +
            estimate_no_ds.gradient_memory_gb +
            estimate_no_ds.optimizer_memory_gb
        )
        total_zero3 = (
            estimate_zero3.weight_memory_gb +
            estimate_zero3.gradient_memory_gb +
            estimate_zero3.optimizer_memory_gb
        )
        ratio = total_no_ds / total_zero3
        assert 7.0 < ratio < 9.0  # Should be ~8x reduction


class TestTotalMemory:
    """Test total training memory estimation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_total_memory_full_8b(self, calculator, llama_8b_config):
        """Test total memory for full finetuning 8B model."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
            optimizer="adamw",
            gradient_checkpointing=True,
        )
        # Full 8B: ~48 (weights+master) + 32 (grads) + 64 (opt) + 10 (act) = ~154 GB
        # With 10% framework overhead: ~170 GB
        assert estimate.total_memory_gb > 140.0
        assert estimate.total_memory_gb < 200.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_total_memory_lora_8b(self, calculator, llama_8b_config):
        """Test total memory for LoRA finetuning 8B model."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
            optimizer="adamw",
            gradient_checkpointing=True,
        )
        # LoRA 8B: ~48 (weights+master) + 0.3 (grads) + 0.7 (opt) + 10 (act) = ~59 GB
        # With 10% framework overhead: ~65 GB
        assert estimate.total_memory_gb > 50.0
        assert estimate.total_memory_gb < 80.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_total_memory_qlora_8b(self, calculator, llama_8b_config):
        """Test total memory for QLoRA finetuning 8B model."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="int4",  # 4-bit weights
            method="qlora",
            lora_rank=16,
            optimizer="adamw",
            gradient_checkpointing=True,
        )
        # QLoRA 8B: ~7.9 (NF4+dequant) + 0.3 (grads) + 0.7 (opt) + 10 (act) = ~19 GB
        # With 10% framework overhead: ~21 GB. Should fit on RTX 4090 24GB
        assert estimate.total_memory_gb > 15.0
        assert estimate.total_memory_gb < 30.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_memory_breakdown_consistency(self, calculator, llama_8b_config):
        """Test that memory breakdown sums to total (minus overhead)."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
            optimizer="adamw",
            gradient_checkpointing=True,
        )
        components_sum = (
            estimate.weight_memory_gb +
            estimate.gradient_memory_gb +
            estimate.optimizer_memory_gb +
            estimate.activation_memory_gb
        )
        # Total should be components + overhead (typically 10-20%)
        assert estimate.total_memory_gb >= components_sum
        assert estimate.total_memory_gb <= components_sum * 1.3


class TestRealWorldScenarios:
    """Test real-world training scenarios."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_llama_70b_qlora_multi_gpu(self, calculator, llama_70b_config):
        """Test 70B QLoRA training across multiple GPUs."""
        estimate = calculator.calculate_training_memory(
            config=llama_70b_config,
            batch_size=1,
            seq_length=4096,
            precision="int4",
            method="qlora",
            lora_rank=32,
            optimizer="adamw_8bit",
            gradient_checkpointing=True,
            tensor_parallel=2,
            data_parallel=2,
        )
        # QLoRA 70B with TP=2: ~35 GB / 2 = ~17.5 GB per GPU (weights)
        # Plus activations and optimizer states
        assert estimate.total_memory_gb < 80.0  # Should fit on A100 80GB

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_efficient_training_config(self, calculator, llama_8b_config):
        """Test most memory-efficient training configuration."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=1,
            seq_length=512,
            precision="int4",
            method="qlora",
            lora_rank=8,
            optimizer="adamw_8bit",
            gradient_checkpointing=True,
        )
        # Minimal config: should fit on 8GB GPU
        assert estimate.total_memory_gb < 10.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_invalid_method(self, calculator, llama_8b_config):
        """Test that invalid training method raises error."""
        with pytest.raises(ValueError, match="Invalid training method"):
            calculator.calculate_training_memory(
                config=llama_8b_config,
                batch_size=4,
                seq_length=2048,
                precision="bf16",
                method="invalid_method",
            )

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_invalid_optimizer(self, calculator, llama_8b_config):
        """Test that invalid optimizer warns and falls back to AdamW."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimate = calculator.calculate_training_memory(
                config=llama_8b_config,
                batch_size=4,
                seq_length=2048,
                precision="bf16",
                method="full",
                optimizer="invalid_optimizer",
            )
            assert len(w) >= 1
            assert "Unknown optimizer" in str(w[0].message)
            # Should fall back to AdamW memory estimate
            assert estimate.total_memory_gb > 0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_lora_without_rank(self, calculator, llama_8b_config):
        """Test that LoRA without rank uses default."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            # No lora_rank specified - should use default (16)
        )
        assert estimate.gradient_memory_gb > 0
        assert estimate.lora_rank == 16  # Default

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_zero_batch_size(self, calculator, llama_8b_config):
        """Test that zero batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculator.calculate_training_memory(
                config=llama_8b_config,
                batch_size=0,
                seq_length=2048,
                precision="bf16",
                method="full",
            )


class TestTrainingEstimateDataclass:
    """Test TrainingMemoryEstimate dataclass."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimate_has_all_fields(self, calculator, llama_8b_config):
        """Test that estimate contains all required fields."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="full",
        )
        assert hasattr(estimate, 'weight_memory_gb')
        assert hasattr(estimate, 'gradient_memory_gb')
        assert hasattr(estimate, 'optimizer_memory_gb')
        assert hasattr(estimate, 'activation_memory_gb')
        assert hasattr(estimate, 'total_memory_gb')
        assert hasattr(estimate, 'trainable_params')
        assert hasattr(estimate, 'method')
        assert hasattr(estimate, 'optimizer')
        assert hasattr(estimate, 'precision')
        assert hasattr(estimate, 'batch_size')
        assert hasattr(estimate, 'seq_length')

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimate_to_dict(self, calculator, llama_8b_config):
        """Test that estimate can be converted to dict."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
        )
        estimate_dict = estimate.to_dict()
        assert isinstance(estimate_dict, dict)
        assert 'weight_memory_gb' in estimate_dict
        assert 'total_memory_gb' in estimate_dict

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimate_fits_in_memory(self, calculator, llama_8b_config):
        """Test fits_in_memory helper method."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4,
            seq_length=2048,
            precision="bf16",
            method="lora",
            lora_rank=16,
        )
        # Should fit in 80GB
        assert estimate.fits_in_memory(80.0)
        # Might not fit in 16GB
        if estimate.total_memory_gb > 16.0:
            assert not estimate.fits_in_memory(16.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
