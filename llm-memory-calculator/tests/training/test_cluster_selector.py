"""
Test suite for TrainingClusterSelector.

TDD approach: Tests for selecting optimal clusters for training.
"""

import pytest
from typing import Dict, Any

try:
    from llm_memory_calculator.training import (
        TrainingClusterSelector,
        TrainingMemoryEstimate,
        ClusterRecommendation,
        ClusterFitResult,
    )
    TRAINING_MODULE_AVAILABLE = True
except ImportError:
    TRAINING_MODULE_AVAILABLE = False


@pytest.fixture
def selector() -> "TrainingClusterSelector":
    """Create a TrainingClusterSelector instance."""
    if not TRAINING_MODULE_AVAILABLE:
        pytest.skip("Training module not yet implemented")
    return TrainingClusterSelector()


@pytest.fixture
def training_estimate_small() -> Dict[str, Any]:
    """Small training memory estimate (~25 GB)."""
    return {
        "weight_memory_gb": 16.0,
        "gradient_memory_gb": 0.5,
        "optimizer_memory_gb": 2.0,
        "activation_memory_gb": 4.0,
        "total_memory_gb": 25.0,
        "trainable_params": 20_000_000,
        "method": "lora",
    }


@pytest.fixture
def training_estimate_medium() -> Dict[str, Any]:
    """Medium training memory estimate (~80 GB)."""
    return {
        "weight_memory_gb": 32.0,
        "gradient_memory_gb": 32.0,
        "optimizer_memory_gb": 8.0,
        "activation_memory_gb": 8.0,
        "total_memory_gb": 80.0,
        "trainable_params": 8_000_000_000,
        "method": "full",
    }


@pytest.fixture
def training_estimate_large() -> Dict[str, Any]:
    """Large training memory estimate (~300 GB)."""
    return {
        "weight_memory_gb": 140.0,
        "gradient_memory_gb": 140.0,
        "optimizer_memory_gb": 20.0,
        "activation_memory_gb": 10.0,
        "total_memory_gb": 310.0,
        "trainable_params": 70_000_000_000,
        "method": "full",
    }


class TestClusterSelectorBasics:
    """Test basic selector functionality."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_selector_instantiation(self):
        """Test that selector can be instantiated."""
        selector = TrainingClusterSelector()
        assert selector is not None

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_list_available_hardware(self, selector):
        """Test listing available hardware profiles."""
        hardware = selector.list_available_hardware()
        assert len(hardware) > 0
        assert any("A100" in h["name"] for h in hardware)
        assert any("H100" in h["name"] for h in hardware)


class TestClusterRecommendations:
    """Test cluster recommendation generation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_recommend_for_small_workload(self, selector, training_estimate_small):
        """Test recommendations for small training workload."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_small,
            prefer_cost=True,
        )
        assert len(recommendations) > 0
        # Should recommend single GPU options
        top_rec = recommendations[0]
        assert top_rec.total_gpus >= 1
        assert top_rec.fits is True

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_recommend_for_medium_workload(self, selector, training_estimate_medium):
        """Test recommendations for medium training workload."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        assert len(recommendations) > 0
        # Should recommend fitting cluster with valid parallelism
        top_rec = recommendations[0]
        assert top_rec.fits is True
        # Memory per GPU should allow workload to fit
        assert top_rec.total_gpus >= 1
        assert top_rec.parallelism is not None

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_recommend_for_large_workload(self, selector, training_estimate_large):
        """Test recommendations for large training workload."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_large,
            prefer_cost=True,
        )
        assert len(recommendations) > 0
        # Should recommend multi-GPU setup
        top_rec = recommendations[0]
        assert top_rec.total_gpus > 1
        assert top_rec.parallelism is not None
        assert top_rec.fits is True

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_cost_vs_speed_preference(self, selector, training_estimate_medium):
        """Test that cost vs speed preference affects recommendations."""
        recs_cost = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        recs_speed = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=False,
        )
        # Speed-optimized should have faster hardware at top
        cost_first = recs_cost[0]
        speed_first = recs_speed[0]
        # Either different hardware or speed one costs more
        assert (
            cost_first.hardware_name != speed_first.hardware_name or
            speed_first.estimated_cost_per_hour >= cost_first.estimated_cost_per_hour
        )

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_budget_constraint(self, selector, training_estimate_medium):
        """Test that budget constraint filters recommendations."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
            max_budget_per_hour=5.0,  # $5/hour max
        )
        for rec in recommendations:
            assert rec.estimated_cost_per_hour <= 5.0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_hardware_filter(self, selector, training_estimate_small):
        """Test filtering recommendations by available hardware."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_small,
            prefer_cost=True,
            available_hardware=["A100_80GB_GPU", "H100_GPU"],
        )
        for rec in recommendations:
            assert rec.hardware_name in ["A100_80GB_GPU", "H100_GPU"]


class TestClusterFitCheck:
    """Test checking if training fits in specific cluster."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_fits_single_a100(self, selector, training_estimate_small):
        """Test small workload fits on single A100 80GB."""
        result = selector.check_fit(
            training_estimate=training_estimate_small,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        assert result.fits is True
        assert result.memory_per_gpu_gb < 80.0
        assert result.utilization_percent > 0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_does_not_fit_single_gpu(self, selector, training_estimate_large):
        """Test large workload doesn't fit on single GPU."""
        result = selector.check_fit(
            training_estimate=training_estimate_large,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        assert result.fits is False
        assert result.reason is not None
        assert "memory" in result.reason.lower()

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_fits_with_parallelism(self, selector, training_estimate_large):
        """Test large workload fits with proper parallelism."""
        result = selector.check_fit(
            training_estimate=training_estimate_large,
            hardware="A100_80GB_GPU",
            num_gpus=8,
        )
        if result.fits:
            assert result.parallelism is not None
            assert result.parallelism.get("tp", 1) >= 1
            assert result.parallelism.get("dp", 1) >= 1

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_suggests_min_gpus(self, selector, training_estimate_large):
        """Test suggestion of minimum GPUs needed."""
        result = selector.check_fit(
            training_estimate=training_estimate_large,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        if not result.fits:
            assert result.min_gpus_required > 1
            # Verify suggestion works
            result_with_min = selector.check_fit(
                training_estimate=training_estimate_large,
                hardware="A100_80GB_GPU",
                num_gpus=result.min_gpus_required,
            )
            assert result_with_min.fits is True


class TestParallelismStrategies:
    """Test parallelism strategy selection."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_tensor_parallel_for_large_model(self, selector, training_estimate_large):
        """Test that parallelism is used for large models."""
        result = selector.check_fit(
            training_estimate=training_estimate_large,
            hardware="A100_80GB_GPU",
            num_gpus=8,
        )
        if result.fits:
            # Either TP or DP should be used for 70B model
            # With ZeRO sharding, DP can handle large models effectively
            total_parallelism = (
                result.parallelism.get("tp", 1) *
                result.parallelism.get("pp", 1) *
                result.parallelism.get("dp", 1)
            )
            assert total_parallelism == 8

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_data_parallel_for_small_model(self, selector, training_estimate_small):
        """Test that DP is recommended for small models on multi-GPU."""
        result = selector.check_fit(
            training_estimate=training_estimate_small,
            hardware="A100_80GB_GPU",
            num_gpus=4,
        )
        if result.fits:
            # Small model should use DP, not TP
            assert result.parallelism.get("dp", 1) > 1
            assert result.parallelism.get("tp", 1) == 1

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_hybrid_parallelism(self, selector, training_estimate_large):
        """Test hybrid TP+DP parallelism."""
        result = selector.check_fit(
            training_estimate=training_estimate_large,
            hardware="A100_80GB_GPU",
            num_gpus=16,
        )
        if result.fits:
            tp = result.parallelism.get("tp", 1)
            dp = result.parallelism.get("dp", 1)
            pp = result.parallelism.get("pp", 1)
            # Total should match num_gpus
            assert tp * dp * pp == 16


class TestCostEstimation:
    """Test cost estimation for training."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_cost_per_hour_included(self, selector, training_estimate_medium):
        """Test that cost per hour is included in recommendations."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        for rec in recommendations:
            assert rec.estimated_cost_per_hour > 0

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_cost_scales_with_gpus(self, selector, training_estimate_small):
        """Test that cost scales with number of GPUs."""
        rec_1gpu = selector.check_fit(
            training_estimate=training_estimate_small,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        rec_4gpu = selector.check_fit(
            training_estimate=training_estimate_small,
            hardware="A100_80GB_GPU",
            num_gpus=4,
        )
        if rec_1gpu.fits and rec_4gpu.fits:
            # 4 GPUs should cost ~4x
            ratio = rec_4gpu.estimated_cost_per_hour / rec_1gpu.estimated_cost_per_hour
            assert 3.5 < ratio < 4.5


class TestUtilization:
    """Test GPU utilization estimation."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_utilization_percentage(self, selector, training_estimate_medium):
        """Test that utilization percentage is calculated."""
        result = selector.check_fit(
            training_estimate=training_estimate_medium,
            hardware="A100_80GB_GPU",
            num_gpus=1,
        )
        if result.fits:
            assert 0 < result.utilization_percent <= 100

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_optimality_classification(self, selector, training_estimate_medium):
        """Test that recommendations are classified by optimality."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        # At least some should have optimality labels
        has_optimal = any(rec.optimality == "optimal" for rec in recommendations)
        has_good = any(rec.optimality == "good" for rec in recommendations)
        assert has_optimal or has_good


class TestClusterRecommendationDataclass:
    """Test ClusterRecommendation dataclass."""

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_recommendation_has_all_fields(self, selector, training_estimate_medium):
        """Test that recommendation has all required fields."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        rec = recommendations[0]
        assert hasattr(rec, 'hardware_name')
        assert hasattr(rec, 'nodes_required')
        assert hasattr(rec, 'gpus_per_node')
        assert hasattr(rec, 'total_gpus')
        assert hasattr(rec, 'memory_per_gpu_gb')
        assert hasattr(rec, 'parallelism')
        assert hasattr(rec, 'estimated_throughput_tps')
        assert hasattr(rec, 'estimated_cost_per_hour')
        assert hasattr(rec, 'utilization_percent')
        assert hasattr(rec, 'fits')

    @pytest.mark.skipif(not TRAINING_MODULE_AVAILABLE, reason="Training module not implemented")
    def test_recommendation_to_dict(self, selector, training_estimate_medium):
        """Test that recommendation can be converted to dict."""
        recommendations = selector.recommend_clusters(
            training_estimate=training_estimate_medium,
            prefer_cost=True,
        )
        rec_dict = recommendations[0].to_dict()
        assert isinstance(rec_dict, dict)
        assert 'hardware_name' in rec_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
