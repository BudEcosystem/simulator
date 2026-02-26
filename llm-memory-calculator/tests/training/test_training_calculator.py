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
        # QLoRA: 8B × 0.516 (NF4+DQ) + ONE layer dequant buffer + adapter master copy
        # ≈ 4.14 + 0.12 + 0.34 = ~4.6 GB
        assert 4.0 < estimate.weight_memory_gb < 6.0


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
        # Without checkpointing, FA=True (default): 32 layers × ~1.21 GB ≈ 38.7 GB
        # Config-aware formula: sbh(10 + (5+21)/T) = sbh*36 per layer
        assert 35.0 < estimate.activation_memory_gb < 45.0

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
        # With checkpointing: ceil(sqrt(32))=6 layers × ~1.21 GB ≈ 7.25 GB
        assert 5.0 < estimate.activation_memory_gb < 10.0

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
        """Test that activation memory scales linearly with flash attention (default).

        With flash attention (default), the O(s^2) attention score term is
        eliminated, so scaling should be ~2x for 2x sequence length.
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
        # With flash attention: linear scaling → exactly 2x
        ratio = estimate_4k.activation_memory_gb / estimate_2k.activation_memory_gb
        assert 1.8 < ratio < 2.3


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
        # Peak-of-phases: optimizer_peak = 48(w) + 32(g) + 64(o) = 144 dominates
        # With 10% overhead: ~159 GB
        assert estimate.total_memory_gb > 140.0
        assert estimate.total_memory_gb < 175.0

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
        # LoRA 8B: ~16.4(w+adapter master) + 7.25(act) + 0.34(g) → backward_peak ~24
        # With 10% overhead: ~26 GB (no fp32 master for frozen base params)
        assert estimate.total_memory_gb > 22.0
        assert estimate.total_memory_gb < 35.0

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
        # QLoRA 8B: ~4.6(NF4+1-layer dequant+adapter master) + 7.25(act) + 0.34(g)
        # backward_peak ~12.2, total ~13.4 GB. Should fit on RTX 4090 24GB.
        assert estimate.total_memory_gb > 10.0
        assert estimate.total_memory_gb < 18.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_memory_breakdown_consistency(self, calculator, llama_8b_config):
        """Test that total equals peak-of-phases with overhead."""
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
        # Peak-of-phases model
        backward_peak = (
            estimate.weight_memory_gb +
            estimate.activation_memory_gb +
            estimate.gradient_memory_gb
        )
        optimizer_peak = (
            estimate.weight_memory_gb +
            estimate.gradient_memory_gb +
            estimate.optimizer_memory_gb
        )
        peak = max(backward_peak, optimizer_peak)
        # Total should be peak * (1 + overhead%)
        assert estimate.total_memory_gb >= peak
        assert estimate.total_memory_gb <= peak * 1.15


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


class TestActivationFormulaImprovements:
    """Test config-aware activation memory formula improvements."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_flash_attention_eliminates_quadratic(self, calculator, llama_8b_config):
        """With flash attention, seq_length scaling is linear (no O(s^2) term)."""
        estimate_fa = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=False, flash_attention=True,
        )
        estimate_no_fa = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=False, flash_attention=False,
        )
        # Without FA, O(s^2) term adds substantial memory
        assert estimate_no_fa.activation_memory_gb > estimate_fa.activation_memory_gb * 2.5

        # With FA, 2x seq → 2x memory (linear)
        estimate_fa_4k = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=4096, precision="bf16",
            method="full", gradient_checkpointing=False, flash_attention=True,
        )
        ratio_fa = estimate_fa_4k.activation_memory_gb / estimate_fa.activation_memory_gb
        assert 1.95 < ratio_fa < 2.05

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_gqa_reduces_activation(self, calculator, llama_8b_config):
        """GQA models (kv_heads < heads) should have lower activation memory."""
        # Llama-8B has GQA: 32 heads, 8 kv_heads (kv_ratio=0.25)
        mha_config = {**llama_8b_config, 'num_key_value_heads': 32}  # MHA: ratio=1.0

        estimate_gqa = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=True,
        )
        estimate_mha = calculator.calculate_training_memory(
            config=mha_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=True,
        )
        # GQA should use less activation memory than MHA
        assert estimate_gqa.activation_memory_gb < estimate_mha.activation_memory_gb

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_lora_no_master_copy_for_frozen_params(self, calculator, llama_8b_config):
        """LoRA weight memory should be much less than full FT (no master copy for base)."""
        estimate_lora = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="lora", lora_rank=16,
        )
        estimate_full = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full",
        )
        # Full FT: 8B*2 + 8B*4 = 48 GB weights
        # LoRA: 8B*2 + 84M*4 = 16.3 GB weights (no master copy for frozen 8B)
        assert estimate_lora.weight_memory_gb < estimate_full.weight_memory_gb * 0.5

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_swiglu_vs_gelu_activation(self, calculator, llama_8b_config):
        """SwiGLU models should have higher activation than equivalent GeLU model."""
        gelu_config = {**llama_8b_config, 'hidden_act': 'gelu'}  # Standard GeLU

        estimate_swiglu = calculator.calculate_training_memory(
            config=llama_8b_config,  # default: silu (SwiGLU)
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=True,
        )
        estimate_gelu = calculator.calculate_training_memory(
            config=gelu_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", gradient_checkpointing=True,
        )
        # SwiGLU stores 3 projections vs 2 for GeLU → higher activation
        assert estimate_swiglu.activation_memory_gb > estimate_gelu.activation_memory_gb

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_qwen_0_5b_lora_reasonable(self, calculator):
        """Validate Qwen2.5-0.5B LoRA produces a reasonable estimate."""
        qwen_05b_config = {
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "hidden_act": "silu",
            "num_parameters": 494_032_896,
        }
        estimate = calculator.calculate_training_memory(
            config=qwen_05b_config,
            batch_size=2, seq_length=2048, precision="bf16",
            method="lora", lora_rank=16, optimizer="adamw",
            gradient_checkpointing=True,
        )
        # Measured on RTX 3080: 5.14 GB (includes ~1.5 GB CUDA context).
        # Formula estimates computational memory only: ~2-4 GB.
        assert 1.5 < estimate.total_memory_gb < 5.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_peak_phases_backward_dominates_lora(self, calculator, llama_8b_config):
        """For LoRA (small optimizer), backward_peak should dominate."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="lora", lora_rank=16, optimizer="adamw",
            gradient_checkpointing=True,
        )
        backward_peak = (
            estimate.weight_memory_gb +
            estimate.activation_memory_gb +
            estimate.gradient_memory_gb
        )
        optimizer_peak = (
            estimate.weight_memory_gb +
            estimate.gradient_memory_gb +
            estimate.optimizer_memory_gb
        )
        # LoRA: activations >> optimizer states → backward dominates
        assert backward_peak > optimizer_peak

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_peak_phases_optimizer_dominates_full_ft(self, calculator, llama_8b_config):
        """For full FT + AdamW + checkpointing, optimizer_peak should dominate."""
        estimate = calculator.calculate_training_memory(
            config=llama_8b_config,
            batch_size=4, seq_length=2048, precision="bf16",
            method="full", optimizer="adamw",
            gradient_checkpointing=True,
        )
        backward_peak = (
            estimate.weight_memory_gb +
            estimate.activation_memory_gb +
            estimate.gradient_memory_gb
        )
        optimizer_peak = (
            estimate.weight_memory_gb +
            estimate.gradient_memory_gb +
            estimate.optimizer_memory_gb
        )
        # Full FT + AdamW: optimizer states (64 GB) >> activations (~7 GB)
        assert optimizer_peak > backward_peak


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
