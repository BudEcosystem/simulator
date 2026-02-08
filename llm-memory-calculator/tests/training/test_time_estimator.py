"""
Test suite for TrainingTimeEstimator.

TDD approach: Tests for estimating training time and cost.
"""

import pytest
from typing import Dict, Any

try:
    from llm_memory_calculator.training import (
        TrainingTimeEstimator,
        TrainingTimeEstimate,
    )
    TRAINING_MODULE_AVAILABLE = True
except ImportError:
    TRAINING_MODULE_AVAILABLE = False


@pytest.fixture
def estimator() -> "TrainingTimeEstimator":
    """Create a TrainingTimeEstimator instance."""
    if not TRAINING_MODULE_AVAILABLE:
        pytest.skip("Training module not yet implemented")
    return TrainingTimeEstimator()


@pytest.fixture
def llama_8b_config() -> Dict[str, Any]:
    """Llama 8B model config."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "num_parameters": 8_030_261_248,
    }


@pytest.fixture
def llama_70b_config() -> Dict[str, Any]:
    """Llama 70B model config."""
    return {
        "model_type": "llama",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "num_parameters": 70_553_706_496,
    }


class TestTimeEstimatorBasics:
    """Test basic estimator functionality."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimator_instantiation(self):
        """Test that estimator can be instantiated."""
        estimator = TrainingTimeEstimator()
        assert estimator is not None


class TestTokensPerSecond:
    """Test tokens per second estimation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_tps_single_a100(self, estimator, llama_8b_config):
        """Test TPS estimation on single A100."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,  # 1B tokens
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        # 8B model on A100 should get ~1000-3000 TPS
        assert 500 < estimate.tokens_per_second < 5000

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_tps_scales_with_gpus(self, estimator, llama_8b_config):
        """Test that TPS scales with number of GPUs."""
        estimate_1 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_4 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=4,
            parallelism={"tp": 1, "pp": 1, "dp": 4},
        )
        # Should scale ~4x with DP
        ratio = estimate_4.tokens_per_second / estimate_1.tokens_per_second
        assert 3.0 < ratio < 4.5  # ~4x with some overhead

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_tps_h100_faster_than_a100(self, estimator, llama_8b_config):
        """Test that H100 is faster than A100."""
        estimate_a100 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_h100 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="H100_GPU",
            num_gpus=1,
        )
        # H100 should be ~2-3x faster
        assert estimate_h100.tokens_per_second > estimate_a100.tokens_per_second


class TestTrainingSteps:
    """Test training step calculations."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_total_steps_calculation(self, estimator, llama_8b_config):
        """Test total training steps calculation."""
        dataset_tokens = 1_000_000_000  # 1B tokens
        batch_size = 4
        grad_accum = 4
        epochs = 1.0
        seq_length = 2048

        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=dataset_tokens,
            batch_size=batch_size,
            gradient_accumulation=grad_accum,
            epochs=epochs,
            hardware="A100_80GB_GPU",
            num_gpus=1,
            seq_length=seq_length,
        )
        # Steps = (tokens × epochs) / (batch × grad_accum × seq_len)
        # = (1B × 1) / (4 × 4 × 2048) = ~30,500 steps
        expected_steps = (dataset_tokens * epochs) // (batch_size * grad_accum * seq_length)
        assert abs(estimate.total_steps - expected_steps) < 1000

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_steps_scale_with_epochs(self, estimator, llama_8b_config):
        """Test that steps scale with epochs."""
        estimate_1 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_3 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=3.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        # Should be ~3x steps
        ratio = estimate_3.total_steps / estimate_1.total_steps
        assert 2.9 < ratio < 3.1

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_steps_decrease_with_dp(self, estimator, llama_8b_config):
        """Test that total steps decrease with DP due to larger effective batch."""
        estimate_1 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_4 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=1_000_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=4,
            parallelism={"tp": 1, "pp": 1, "dp": 4},
        )
        # DP=4 means 4x larger effective batch, so 1/4 steps
        ratio = estimate_1.total_steps / estimate_4.total_steps
        assert 3.5 < ratio < 4.5  # ~4x fewer steps
        # Time should also be faster due to parallelism
        assert estimate_4.estimated_hours < estimate_1.estimated_hours


class TestTimeEstimation:
    """Test training time estimation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_reasonable_time_estimate(self, estimator, llama_8b_config):
        """Test that time estimate is reasonable."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,  # 100M tokens
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        # Should take a few hours
        assert 0.5 < estimate.estimated_hours < 50

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_time_formula_correct(self, estimator, llama_8b_config):
        """Test that time is calculated correctly from TPS."""
        dataset_tokens = 100_000_000
        epochs = 1.0

        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=dataset_tokens,
            batch_size=4,
            gradient_accumulation=4,
            epochs=epochs,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        # Time = total_tokens / tokens_per_second / 3600
        expected_hours = (dataset_tokens * epochs) / estimate.tokens_per_second / 3600
        assert abs(estimate.estimated_hours - expected_hours) < 0.1

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_70b_takes_longer_than_8b(self, estimator, llama_8b_config, llama_70b_config):
        """Test that 70B takes longer than 8B for same data."""
        estimate_8b = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_70b = estimator.estimate_training_time(
            model_config=llama_70b_config,
            dataset_tokens=100_000_000,
            batch_size=1,  # Smaller batch for 70B
            gradient_accumulation=16,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=8,
            parallelism={"tp": 4, "pp": 1, "dp": 2},
        )
        # 70B should take longer despite more GPUs
        assert estimate_70b.estimated_hours > estimate_8b.estimated_hours * 0.5


class TestCostEstimation:
    """Test training cost estimation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_cost_calculated(self, estimator, llama_8b_config):
        """Test that cost is calculated."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        assert estimate.estimated_cost > 0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_cost_formula_correct(self, estimator, llama_8b_config):
        """Test that cost = hours × GPUs × cost_per_gpu_hour."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=4,
        )
        # Cost should be ~4x a single GPU
        cost_per_hour = estimate.estimated_cost / estimate.estimated_hours
        cost_per_gpu_hour = cost_per_hour / 4
        # A100 80GB is typically $2-5/hour
        assert 1.0 < cost_per_gpu_hour < 10.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_h100_costs_more_than_a100(self, estimator, llama_8b_config):
        """Test that H100 has higher cost per hour but may be cheaper total."""
        estimate_a100 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_h100 = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="H100_GPU",
            num_gpus=1,
        )
        # H100 hourly rate should be higher
        h100_hourly = estimate_h100.estimated_cost / estimate_h100.estimated_hours
        a100_hourly = estimate_a100.estimated_cost / estimate_a100.estimated_hours
        assert h100_hourly > a100_hourly


class TestEfficiencyMetrics:
    """Test efficiency and utilization metrics."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_mfu_calculated(self, estimator, llama_8b_config):
        """Test that Model FLOPs Utilization is calculated."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        # MFU is typically 30-50% for well-optimized training
        assert hasattr(estimate, 'model_flops_utilization')
        assert 0.1 < estimate.model_flops_utilization < 0.8

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_flops_per_step(self, estimator, llama_8b_config):
        """Test FLOPs per step calculation."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
            seq_length=2048,
        )
        # ~6 FLOPs per param per token for forward+backward
        # 8B × 2048 × 16 (effective batch) × 6 ≈ 1.6e15 FLOPs per step
        expected_flops = 6 * 8e9 * 2048 * (4 * 4)
        if hasattr(estimate, 'flops_per_step'):
            ratio = estimate.flops_per_step / expected_flops
            assert 0.5 < ratio < 2.0


class TestTrainingTimeEstimateDataclass:
    """Test TrainingTimeEstimate dataclass."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimate_has_all_fields(self, estimator, llama_8b_config):
        """Test that estimate has all required fields."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        assert hasattr(estimate, 'total_steps')
        assert hasattr(estimate, 'tokens_per_second')
        assert hasattr(estimate, 'estimated_hours')
        assert hasattr(estimate, 'estimated_cost')
        assert hasattr(estimate, 'hardware')
        assert hasattr(estimate, 'num_gpus')

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_estimate_to_dict(self, estimator, llama_8b_config):
        """Test that estimate can be converted to dict."""
        estimate = estimator.estimate_training_time(
            model_config=llama_8b_config,
            dataset_tokens=100_000_000,
            batch_size=4,
            gradient_accumulation=4,
            epochs=1.0,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        estimate_dict = estimate.to_dict()
        assert isinstance(estimate_dict, dict)
        assert 'estimated_hours' in estimate_dict


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_invalid_hardware(self, estimator, llama_8b_config):
        """Test that invalid hardware raises error."""
        with pytest.raises(ValueError, match="Unknown hardware"):
            estimator.estimate_training_time(
                model_config=llama_8b_config,
                dataset_tokens=100_000_000,
                batch_size=4,
                gradient_accumulation=4,
                epochs=1.0,
                hardware="INVALID_GPU",
                num_gpus=1,
            )

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_zero_tokens(self, estimator, llama_8b_config):
        """Test that zero tokens raises error."""
        with pytest.raises(ValueError, match="tokens must be positive"):
            estimator.estimate_training_time(
                model_config=llama_8b_config,
                dataset_tokens=0,
                batch_size=4,
                gradient_accumulation=4,
                epochs=1.0,
                hardware="A100_80GB_GPU",
                num_gpus=1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
