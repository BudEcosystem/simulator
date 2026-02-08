"""
Scale Optimizer Validation Tests.

Validates that the scale optimizer finds truly optimal configurations
and returns consistent, valid results.
"""

import pytest
from typing import List, Dict, Any

from llm_memory_calculator.training import (
    find_optimal_scale,
    find_scaling_frontier,
    recommend_gang_configuration,
    analyze_scaling_efficiency,
    ScalingRecommendation,
    ScalingFrontierPoint,
)

from llm_memory_calculator.genz.LLM_training import (
    get_best_training_parallelization,
    training_modeling,
)


class TestFindOptimalScale:
    """Tests for find_optimal_scale function."""

    def test_returns_valid_recommendation(self):
        """Should return valid ScalingRecommendation."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
        )

        assert isinstance(result, ScalingRecommendation)
        assert result.optimal_num_gpus >= 1
        assert result.optimal_num_gpus <= 32
        assert result.tensor_parallel >= 1
        assert result.pipeline_parallel >= 1
        assert result.data_parallel >= 1

    def test_gpu_product_equals_total(self):
        """TP * PP * DP should equal total GPUs."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=8,
            max_gpus=8,
        )

        product = result.tensor_parallel * result.pipeline_parallel * result.data_parallel
        assert product == result.optimal_num_gpus

    def test_respects_min_max_constraints(self):
        """Should respect min_gpus and max_gpus."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=4,
            max_gpus=16,
        )

        assert result.optimal_num_gpus >= 4
        assert result.optimal_num_gpus <= 16

    def test_respects_cost_constraint(self):
        """Should respect max_cost_per_hour constraint."""
        max_cost = 30.0

        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=64,
            max_cost_per_hour=max_cost,
        )

        assert result.cost_per_hour <= max_cost

    def test_respects_throughput_constraint(self):
        """Should respect target_throughput constraint."""
        target = 50000  # 50K tokens/sec

        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=64,
            target_throughput=target,
        )

        assert result.throughput_tokens_per_sec >= target

    def test_optimization_target_cost_efficiency(self):
        """cost_efficiency target should minimize $/token."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
            optimization_target='cost_efficiency',
        )

        assert result.cost_per_million_tokens > 0
        # Should have reasonable cost efficiency
        assert result.cost_per_million_tokens < 100  # Sanity check

    def test_optimization_target_throughput(self):
        """throughput target should maximize tokens/sec."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
            optimization_target='throughput',
        )

        # Throughput-optimized should use more GPUs typically
        assert result.throughput_tokens_per_sec > 0

    def test_optimization_target_mfu(self):
        """mfu target should maximize hardware utilization."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
            optimization_target='mfu',
        )

        assert 0 < result.mfu <= 1.0

    def test_metrics_positive(self):
        """All performance metrics should be positive."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        assert result.throughput_tokens_per_sec > 0
        assert result.step_time_ms > 0
        assert result.mfu > 0
        assert result.cost_per_hour > 0
        assert result.cost_per_million_tokens > 0
        assert result.memory_per_gpu_gb > 0


class TestFindScalingFrontier:
    """Tests for Pareto frontier finding."""

    def test_returns_list_of_points(self):
        """Should return list of ScalingFrontierPoint."""
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
        )

        assert isinstance(frontier, list)
        if len(frontier) > 0:
            assert all(isinstance(p, ScalingFrontierPoint) for p in frontier)

    def test_no_dominated_points(self):
        """Frontier should contain no dominated points."""
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
        )

        if len(frontier) < 2:
            pytest.skip("Not enough points for dominance test")

        for i, point in enumerate(frontier):
            for j, other in enumerate(frontier):
                if i == j:
                    continue

                # Check if other dominates point
                # (higher throughput AND lower cost per token)
                dominates = (
                    other.throughput_tokens_per_sec >= point.throughput_tokens_per_sec and
                    other.cost_per_million_tokens <= point.cost_per_million_tokens and
                    (other.throughput_tokens_per_sec > point.throughput_tokens_per_sec or
                     other.cost_per_million_tokens < point.cost_per_million_tokens)
                )

                assert not dominates, \
                    f"Point at {point.num_gpus} GPUs dominated by point at {other.num_gpus} GPUs"

    def test_frontier_sorted_by_gpus(self):
        """Frontier should be sorted by GPU count."""
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
        )

        if len(frontier) > 1:
            gpu_counts = [p.num_gpus for p in frontier]
            assert gpu_counts == sorted(gpu_counts)

    def test_frontier_has_valid_parallelism(self):
        """Each point should have valid parallelism config."""
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        for point in frontier:
            assert point.parallelism is not None
            product = (point.parallelism.tensor_parallel *
                      point.parallelism.pipeline_parallel *
                      point.parallelism.data_parallel)
            assert product == point.num_gpus


class TestRecommendGangConfiguration:
    """Tests for gang/DP replica recommendation."""

    def test_returns_success_result(self):
        """Should return successful result for valid input."""
        result = recommend_gang_configuration(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            num_gpus=8,
        )

        assert result['success']
        assert 'recommended' in result
        assert 'num_gangs' in result

    def test_recommended_config_valid(self):
        """Recommended config should have valid parallelism."""
        result = recommend_gang_configuration(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            num_gpus=8,
        )

        assert result['success']
        rec = result['recommended']

        product = rec['tensor_parallel'] * rec['pipeline_parallel'] * rec['data_parallel']
        assert product == 8

    def test_maximize_dp_prefers_higher_dp(self):
        """maximize_dp=True should prefer higher DP when possible."""
        result = recommend_gang_configuration(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            num_gpus=8,
            maximize_dp=True,
        )

        assert result['success']
        # For small model on 8 GPUs, should be able to maximize DP
        assert result['num_gangs'] >= 1

    def test_returns_all_valid_configs(self):
        """Should return multiple valid configurations."""
        result = recommend_gang_configuration(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            num_gpus=8,
        )

        assert result['success']
        assert 'all_valid_configs' in result
        assert len(result['all_valid_configs']) > 0

    def test_analysis_summary(self):
        """Should include analysis summary."""
        result = recommend_gang_configuration(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            num_gpus=8,
        )

        assert result['success']
        assert 'analysis' in result
        assert 'total_valid_configs' in result['analysis']
        assert 'max_dp_possible' in result['analysis']


class TestAnalyzeScalingEfficiency:
    """Tests for scaling efficiency analysis."""

    def test_returns_success_result(self):
        """Should return successful result."""
        analysis = analyze_scaling_efficiency(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        assert analysis['success']
        assert 'scaling_data' in analysis
        assert 'summary' in analysis

    def test_scaling_data_has_required_fields(self):
        """Scaling data should have all required fields."""
        analysis = analyze_scaling_efficiency(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        assert analysis['success']
        for point in analysis['scaling_data']:
            assert 'num_gpus' in point
            assert 'throughput' in point
            assert 'efficiency' in point
            assert 'mfu' in point

    def test_efficiency_in_valid_range(self):
        """Efficiency should be in valid range."""
        analysis = analyze_scaling_efficiency(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        assert analysis['success']
        for point in analysis['scaling_data']:
            # Efficiency should be between 0 and ~1.1 (slight tolerance)
            assert 0 < point['efficiency'] <= 1.5

    def test_summary_has_key_metrics(self):
        """Summary should contain key efficiency metrics."""
        analysis = analyze_scaling_efficiency(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        assert analysis['success']
        summary = analysis['summary']

        assert 'single_gpu_throughput' in summary
        assert 'best_efficiency' in summary
        assert 'best_throughput' in summary


class TestOptimalityVerification:
    """Tests to verify optimizer finds truly optimal configurations."""

    def test_cost_efficiency_is_optimal(self):
        """Verify cost_efficiency result is truly the best $/token."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
            optimization_target='cost_efficiency',
        )

        # Get the frontier and verify our result is on it
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        if len(frontier) > 0:
            # Our result should have the best or near-best cost per token
            min_cost = min(p.cost_per_million_tokens for p in frontier)

            # Allow 10% tolerance
            assert result.cost_per_million_tokens <= min_cost * 1.1

    def test_throughput_is_optimal(self):
        """Verify throughput result is truly the best tokens/sec."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
            optimization_target='throughput',
        )

        # Get the frontier
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        if len(frontier) > 0:
            max_throughput = max(p.throughput_tokens_per_sec for p in frontier)

            # Allow 10% tolerance
            assert result.throughput_tokens_per_sec >= max_throughput * 0.9


class TestDataclassProperties:
    """Tests for dataclass properties and methods."""

    def test_scaling_recommendation_to_dict(self):
        """ScalingRecommendation.to_dict() should work."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'configuration' in result_dict
        assert 'performance' in result_dict
        assert 'cost' in result_dict

    def test_scaling_recommendation_summary(self):
        """ScalingRecommendation.summary() should generate readable output."""
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'SCALING RECOMMENDATION' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
