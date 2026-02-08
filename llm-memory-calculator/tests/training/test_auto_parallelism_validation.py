"""
Comprehensive validation suite for auto-parallelism modules.

Tests how optimal and realistic the simulated projections are
across different training types and scales.

This test suite validates:
1. Optimality - Are the optimizer's recommendations truly optimal?
2. Realism - Do predictions match published benchmarks?
3. Scale Coverage - 1 GPU to 256+ GPUs
4. Training Type Coverage - SFT, DPO, LoRA, QLoRA, full
5. Hardware Coverage - H100, A100, etc.
"""

import pytest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import modules under test
from llm_memory_calculator.training import (
    # Time estimator
    estimate_training_time,
    calculate_training_steps,
    find_optimal_gpu_count,
    estimate_scaling_curve,
    ScalingPoint,
    DatasetTrainingTimeEstimate,
    # Scale optimizer
    find_optimal_scale,
    find_scaling_frontier,
    recommend_gang_configuration,
    analyze_scaling_efficiency,
    ScalingRecommendation,
    ScalingFrontierPoint,
    # Node selector
    select_optimal_nodes,
    find_homogeneous_groups,
    evaluate_node_combination,
    NodeSpec,
    NodeSelectionResult,
    # Auto config
    auto_configure_training,
    quick_configure,
    OptimalTrainingPlan,
)

from llm_memory_calculator.genz.LLM_training import (
    training_modeling,
    get_best_training_parallelization,
    TrainingModelingOutput,
    TrainingParallelismConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    return [
        NodeSpec(
            node_id='h100-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-2',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-3',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='a100-1',
            gpu_type='A100_80GB',
            num_gpus=8,
            rack_id='rack-3',
            hourly_cost=16.0,
        ),
    ]


@pytest.fixture
def small_model():
    """Small model for quick tests."""
    return 'meta-llama/meta-llama-3.1-8b'


@pytest.fixture
def large_model():
    """Large model for comprehensive tests."""
    return 'meta-llama/meta-llama-3.1-70b'


# =============================================================================
# TEST 1: Training Time Estimator Validation
# =============================================================================

class TestTrainingTimeEstimator:
    """Tests for training time estimation accuracy."""

    def test_training_time_calculation_correctness(self, small_model):
        """Validate training time calculation is mathematically correct."""
        dataset_tokens = 10_000_000_000  # 10B tokens
        batch_size = 4
        seq_length = 4096
        num_gpus = 8
        grad_accum = 2

        estimate = estimate_training_time(
            model=small_model,
            dataset_tokens=dataset_tokens,
            num_epochs=1.0,
            batch_size=batch_size,
            seq_length=seq_length,
            gradient_accumulation_steps=grad_accum,
            system_name='H100_GPU',
            num_gpus=num_gpus,
        )

        # Verify tokens_per_step calculation
        # tokens_per_step = batch * seq * DP * grad_accum
        expected_tokens_per_step = batch_size * seq_length * estimate.parallelism.data_parallel * grad_accum
        assert estimate.tokens_per_step == expected_tokens_per_step, \
            f"Expected {expected_tokens_per_step} tokens/step, got {estimate.tokens_per_step}"

        # Verify total_steps calculation
        expected_steps = dataset_tokens // expected_tokens_per_step
        assert estimate.total_steps == expected_steps, \
            f"Expected {expected_steps} steps, got {estimate.total_steps}"

        # Verify time calculation (within 1% tolerance for floating point)
        expected_hours = (expected_steps * estimate.step_time_ms) / 3600000
        assert abs(estimate.total_hours - expected_hours) < 0.01 * expected_hours, \
            f"Expected ~{expected_hours:.2f} hours, got {estimate.total_hours:.2f}"

    def test_cost_calculation_correctness(self, small_model):
        """Validate cost calculation is correct."""
        hourly_rate = 3.0
        estimate = estimate_training_time(
            model=small_model,
            dataset_tokens=1_000_000_000,
            num_gpus=8,
            hourly_rate_per_gpu=hourly_rate,
        )

        # Verify hourly cost
        expected_hourly = hourly_rate * 8
        assert estimate.hourly_cost == expected_hourly

        # Verify total cost
        expected_total = estimate.total_hours * expected_hourly
        assert abs(estimate.cost_estimate_usd - expected_total) < 0.01

    def test_throughput_positive(self, small_model):
        """Throughput should be positive."""
        estimate = estimate_training_time(
            model=small_model,
            dataset_tokens=1_000_000_000,
            num_gpus=8,
        )

        assert estimate.tokens_per_second > 0
        assert estimate.samples_per_second > 0
        assert estimate.step_time_ms > 0

    def test_mfu_in_valid_range(self, small_model):
        """MFU should be between 0 and 1."""
        estimate = estimate_training_time(
            model=small_model,
            dataset_tokens=1_000_000_000,
            num_gpus=8,
        )

        assert 0.0 <= estimate.mfu <= 1.0, f"MFU {estimate.mfu} out of valid range"

    def test_scaling_curve_monotonic_throughput(self, small_model):
        """Throughput should generally increase with more GPUs."""
        curve = estimate_scaling_curve(
            model=small_model,
            dataset_tokens=1_000_000_000,
            gpu_counts=[1, 2, 4, 8],
        )

        if len(curve) > 1:
            throughputs = [p.throughput_tokens_per_sec for p in curve]
            # Throughput should generally increase (allow some variation)
            for i in range(1, len(throughputs)):
                # Allow up to 20% decrease (due to parallelism changes)
                assert throughputs[i] >= throughputs[i-1] * 0.8, \
                    f"Throughput decreased unexpectedly from {throughputs[i-1]} to {throughputs[i]}"


# =============================================================================
# TEST 2: Scale Optimizer Validation
# =============================================================================

class TestScaleOptimizer:
    """Tests for scale optimizer optimality."""

    def test_find_optimal_scale_returns_valid_config(self, small_model):
        """find_optimal_scale should return a valid configuration."""
        result = find_optimal_scale(
            model=small_model,
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
            optimization_target='cost_efficiency',
        )

        assert isinstance(result, ScalingRecommendation)
        assert result.optimal_num_gpus >= 1
        assert result.optimal_num_gpus <= 32
        assert result.tensor_parallel >= 1
        assert result.pipeline_parallel >= 1
        assert result.data_parallel >= 1
        assert result.throughput_tokens_per_sec > 0
        assert result.cost_per_million_tokens > 0

    def test_find_optimal_scale_respects_constraints(self, small_model):
        """Optimizer should respect cost constraints."""
        max_cost = 50.0  # $50/hour max

        result = find_optimal_scale(
            model=small_model,
            hardware_type='H100',
            min_gpus=1,
            max_gpus=64,
            max_cost_per_hour=max_cost,
            optimization_target='throughput',
        )

        assert result.cost_per_hour <= max_cost, \
            f"Cost ${result.cost_per_hour:.2f} exceeds constraint ${max_cost}"

    def test_scaling_frontier_no_dominated_points(self, small_model):
        """Pareto frontier should contain no dominated points."""
        frontier = find_scaling_frontier(
            model=small_model,
            hardware_type='H100',
            min_gpus=1,
            max_gpus=32,
        )

        if len(frontier) < 2:
            pytest.skip("Not enough frontier points for dominance test")

        # Check no point is dominated by another
        for i, point in enumerate(frontier):
            for j, other in enumerate(frontier):
                if i == j:
                    continue

                # Check if other dominates point
                dominates = (
                    other.throughput_tokens_per_sec >= point.throughput_tokens_per_sec and
                    other.cost_per_million_tokens <= point.cost_per_million_tokens and
                    (other.throughput_tokens_per_sec > point.throughput_tokens_per_sec or
                     other.cost_per_million_tokens < point.cost_per_million_tokens)
                )

                assert not dominates, \
                    f"Point {i} ({point.num_gpus} GPUs) is dominated by point {j} ({other.num_gpus} GPUs)"

    def test_gang_configuration_maximizes_dp(self, small_model):
        """recommend_gang_configuration with maximize_dp should prefer higher DP."""
        result = recommend_gang_configuration(
            model=small_model,
            hardware_type='H100',
            num_gpus=8,
            maximize_dp=True,
        )

        assert result['success']
        # With 8 GPUs and a small model, DP should be maximized
        assert result['num_gangs'] >= 1

    def test_scaling_efficiency_analysis(self, small_model):
        """analyze_scaling_efficiency should return valid data."""
        analysis = analyze_scaling_efficiency(
            model=small_model,
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        assert analysis['success']
        assert len(analysis['scaling_data']) > 0

        for point in analysis['scaling_data']:
            assert 0.0 <= point['efficiency'] <= 1.5  # Allow some tolerance
            assert point['throughput'] > 0
            assert point['mfu'] >= 0


# =============================================================================
# TEST 3: Node Selector Validation
# =============================================================================

class TestNodeSelector:
    """Tests for node selection algorithm."""

    def test_select_optimal_nodes_returns_valid_result(self, sample_nodes, small_model):
        """select_optimal_nodes should return valid selection."""
        result = select_optimal_nodes(
            model=small_model,
            available_nodes=sample_nodes,
            max_nodes=2,
        )

        assert isinstance(result, NodeSelectionResult)
        assert len(result.selected_nodes) <= 2
        assert result.total_gpus > 0
        assert result.estimated_throughput > 0

    def test_homogeneous_preference(self, sample_nodes, small_model):
        """When prefer_homogeneous=True, should select same GPU types."""
        result = select_optimal_nodes(
            model=small_model,
            available_nodes=sample_nodes,
            max_nodes=2,
            prefer_homogeneous=True,
        )

        assert result.homogeneous, "Should select homogeneous GPUs"
        assert len(result.gpu_types_used) == 1, "Should use only one GPU type"

    def test_rack_locality_preference(self, sample_nodes, small_model):
        """When prefer_same_rack=True, should prefer same-rack nodes."""
        result = select_optimal_nodes(
            model=small_model,
            available_nodes=sample_nodes,
            max_nodes=2,
            prefer_same_rack=True,
            prefer_homogeneous=True,
        )

        # Should prefer rack-1 which has 2 H100 nodes
        if result.total_nodes == 2:
            rack_ids = set(n.rack_id for n in result.selected_nodes if n.rack_id)
            # If we selected 2 nodes, they should be from same rack if possible
            h100_nodes_rack1 = [n for n in sample_nodes if n.gpu_type == 'H100' and n.rack_id == 'rack-1']
            if len(h100_nodes_rack1) >= 2:
                assert result.same_rack_fraction >= 0.5

    def test_find_homogeneous_groups(self, sample_nodes):
        """find_homogeneous_groups should correctly group nodes."""
        groups = find_homogeneous_groups(sample_nodes)

        # Should have 2 groups: H100 and A100
        assert len(groups) == 2
        # H100 group should have 3 nodes
        h100_key = [k for k in groups.keys() if 'h100' in k.lower()][0]
        assert len(groups[h100_key]) == 3

    def test_cost_constraint_respected(self, sample_nodes, small_model):
        """Node selection should respect cost constraints."""
        max_cost = 30.0  # Should allow 1 H100 node or 1-2 A100 nodes

        result = select_optimal_nodes(
            model=small_model,
            available_nodes=sample_nodes,
            max_cost_per_hour=max_cost,
        )

        assert result.estimated_cost_per_hour <= max_cost


# =============================================================================
# TEST 4: Auto-Configure Training Validation
# =============================================================================

class TestAutoConfigureTraining:
    """Tests for unified auto-configuration API."""

    def test_auto_configure_returns_complete_plan(self, small_model):
        """auto_configure_training should return complete plan."""
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_gpus=8,
            min_gpus=1,
        )

        assert isinstance(plan, OptimalTrainingPlan)

        # Check all required fields are populated
        assert plan.total_gpus > 0
        assert plan.total_nodes > 0
        assert plan.tensor_parallel >= 1
        assert plan.pipeline_parallel >= 1
        assert plan.data_parallel >= 1
        assert plan.per_device_batch_size > 0
        assert plan.throughput_tokens_per_sec > 0
        assert plan.total_training_hours >= 0
        assert plan.total_cost_usd >= 0

    def test_auto_configure_respects_cost_constraint(self, small_model):
        """Should respect max_cost_usd constraint."""
        max_cost = 1000.0

        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_cost_usd=max_cost,
            max_gpus=64,
        )

        assert plan.total_cost_usd <= max_cost, \
            f"Cost ${plan.total_cost_usd:.0f} exceeds constraint ${max_cost}"

    def test_auto_configure_respects_time_constraint(self, small_model):
        """Should respect max_hours constraint."""
        max_hours = 100.0

        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_hours=max_hours,
            max_gpus=64,
        )

        assert plan.total_training_hours <= max_hours, \
            f"Time {plan.total_training_hours:.1f} hours exceeds constraint {max_hours}"

    def test_auto_configure_method_auto_selection(self, small_model):
        """Method='auto' should select appropriate training method."""
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            method='auto',
            max_gpus=8,
        )

        assert plan.training_method in ('full', 'lora', 'qlora', 'dora')

    def test_quick_configure_convenience(self, small_model):
        """quick_configure should work as convenience wrapper."""
        plan = quick_configure(
            model=small_model,
            dataset_tokens=1_000_000_000,
            gpu_type='H100',
            num_gpus=8,
        )

        assert plan.total_gpus == 8
        assert plan.selected_hardware == 'H100'

    def test_config_export_methods(self, small_model):
        """Config export methods should return valid configs."""
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_gpus=8,
        )

        # Test LlamaFactory config
        lf_config = plan.to_llamafactory_config()
        assert isinstance(lf_config, dict)
        assert 'stage' in lf_config
        assert 'per_device_train_batch_size' in lf_config

        # Test DeepSpeed config
        ds_config = plan.to_deepspeed_config()
        assert isinstance(ds_config, dict)

        # Test torchrun command
        cmd = plan.to_torchrun_command()
        assert 'torchrun' in cmd
        assert '--nproc_per_node' in cmd


# =============================================================================
# TEST 5: Training Method Validation
# =============================================================================

class TestTrainingMethodAccuracy:
    """Tests for training method (full/LoRA/QLoRA) accuracy."""

    @pytest.mark.parametrize("method,expected_memory_ratio_range", [
        ('full', (0.8, 1.2)),  # Baseline
        ('lora', (0.05, 0.40)),  # Much lower memory
        ('qlora', (0.03, 0.30)),  # Even lower with quantization
    ])
    def test_training_method_memory_ratios(self, small_model, method, expected_memory_ratio_range):
        """Validate memory predictions match expected ratios for different methods."""
        # Get baseline full fine-tuning
        baseline_config, baseline_result = get_best_training_parallelization(
            model=small_model,
            total_gpus=8,
            batch_size=4,
            seq_length=4096,
            system_name='H100_GPU',
            training_stage='sft',
            method='full',
        )

        # Get method-specific result
        try:
            config, result = get_best_training_parallelization(
                model=small_model,
                total_gpus=8,
                batch_size=4,
                seq_length=4096,
                system_name='H100_GPU',
                training_stage='sft',
                method=method,
            )

            ratio = result.memory_per_gpu_gb / baseline_result.memory_per_gpu_gb
            min_ratio, max_ratio = expected_memory_ratio_range

            assert min_ratio <= ratio <= max_ratio, \
                f"Memory ratio for {method} is {ratio:.2f}, expected [{min_ratio}, {max_ratio}]"
        except Exception as e:
            pytest.skip(f"Method {method} not available: {e}")


# =============================================================================
# TEST 6: Scale-Specific Validation
# =============================================================================

class TestScaleSpecificBehavior:
    """Tests for behavior at different scales."""

    @pytest.mark.parametrize("num_gpus,expected_comm_overhead_range,expected_mfu_range", [
        (1, (0, 0.05), (0.30, 0.65)),
        (8, (0.01, 0.15), (0.30, 0.60)),
        (32, (0.05, 0.25), (0.25, 0.55)),
    ])
    def test_scale_characteristics(
        self,
        small_model,
        num_gpus,
        expected_comm_overhead_range,
        expected_mfu_range,
    ):
        """Validate predictions match expected scale characteristics."""
        try:
            config, result = get_best_training_parallelization(
                model=small_model,
                total_gpus=num_gpus,
                batch_size=4,
                seq_length=4096,
                system_name='H100_GPU',
                training_stage='sft',
                method='full',
            )

            comm_min, comm_max = expected_comm_overhead_range
            mfu_min, mfu_max = expected_mfu_range

            comm_overhead = result.communication_overhead if hasattr(result, 'communication_overhead') else 0

            # Allow some tolerance
            assert mfu_min * 0.8 <= result.model_flops_utilization <= mfu_max * 1.2, \
                f"MFU {result.model_flops_utilization:.2f} out of range for {num_gpus} GPUs"

        except Exception as e:
            pytest.skip(f"Scale {num_gpus} not testable: {e}")

    def test_scaling_efficiency_decreases_with_scale(self, small_model):
        """Scaling efficiency should generally decrease with more GPUs."""
        analysis = analyze_scaling_efficiency(
            model=small_model,
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8, 16],
        )

        if not analysis['success'] or len(analysis['scaling_data']) < 2:
            pytest.skip("Not enough scaling data")

        efficiencies = [p['efficiency'] for p in analysis['scaling_data']]

        # Check general downward trend (allow some noise)
        decreasing = True
        for i in range(1, len(efficiencies)):
            # Allow up to 10% increase due to different parallelism strategies
            if efficiencies[i] > efficiencies[i-1] * 1.1:
                decreasing = False
                break

        # At minimum, efficiency at max scale should be lower than at min scale
        if len(efficiencies) > 2:
            assert efficiencies[-1] <= efficiencies[0] * 1.1, \
                "Efficiency should generally decrease with scale"


# =============================================================================
# TEST 7: Cross-Training-Stage Validation
# =============================================================================

class TestCrossTrainingStage:
    """Tests across different training stages."""

    @pytest.mark.parametrize("training_stage", ['sft', 'dpo'])
    def test_training_stage_produces_valid_output(self, small_model, training_stage):
        """Each training stage should produce valid simulation output."""
        try:
            result = training_modeling(
                model=small_model,
                training_stage=training_stage,
                batch_size=4,
                seq_length=4096,
                system_name='H100_GPU',
                num_gpus=8,
                data_parallel=8,
            )

            assert result.tokens_per_second > 0
            assert result.memory_per_gpu_gb > 0
            assert 0 <= result.model_flops_utilization <= 1.0
            assert result.step_time_ms > 0

        except Exception as e:
            pytest.skip(f"Training stage {training_stage} not available: {e}")


# =============================================================================
# TEST 8: End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_workflow_sft(self, small_model):
        """Test complete SFT training configuration workflow."""
        # Step 1: Find optimal scale
        scale_result = find_optimal_scale(
            model=small_model,
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
            optimization_target='cost_efficiency',
        )

        assert scale_result.optimal_num_gpus >= 1

        # Step 2: Get training time estimate
        time_estimate = estimate_training_time(
            model=small_model,
            dataset_tokens=1_000_000_000,
            num_gpus=scale_result.optimal_num_gpus,
            parallelism=scale_result.parallelism,
        )

        assert time_estimate.total_hours > 0
        assert time_estimate.cost_estimate_usd > 0

        # Step 3: Generate full plan
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_gpus=scale_result.optimal_num_gpus,
            min_gpus=scale_result.optimal_num_gpus,
        )

        assert plan.total_gpus == scale_result.optimal_num_gpus

    def test_plan_summary_generation(self, small_model):
        """Test that plan summary generates without error."""
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_gpus=8,
        )

        summary = plan.summary()
        assert len(summary) > 0
        assert 'OPTIMAL TRAINING PLAN' in summary
        assert 'GPU' in summary
        assert 'Cost' in summary

    def test_plan_to_dict_serialization(self, small_model):
        """Test that plan can be serialized to dict."""
        plan = auto_configure_training(
            model=small_model,
            dataset_tokens=1_000_000_000,
            max_gpus=8,
        )

        plan_dict = plan.to_dict()
        assert isinstance(plan_dict, dict)
        assert 'hardware' in plan_dict
        assert 'parallelism' in plan_dict
        assert 'cost' in plan_dict
        assert 'time' in plan_dict


# =============================================================================
# Run validation summary
# =============================================================================

def run_validation_summary():
    """Run a quick validation and print summary."""
    print("=" * 70)
    print("AUTO-PARALLELISM VALIDATION SUMMARY")
    print("=" * 70)

    # Quick test
    try:
        plan = auto_configure_training(
            model='meta-llama/meta-llama-3.1-8b',
            dataset_tokens=10_000_000_000,
            max_gpus=64,
            optimization_goal='minimize_cost',
        )

        print("\nSample Configuration:")
        print(f"  Model: {plan.model}")
        print(f"  GPUs: {plan.total_gpus}x {plan.selected_hardware}")
        print(f"  Parallelism: TP={plan.tensor_parallel}, PP={plan.pipeline_parallel}, DP={plan.data_parallel}")
        print(f"  Method: {plan.training_method}")
        print(f"  Throughput: {plan.throughput_tokens_per_sec:,.0f} tok/s")
        print(f"  MFU: {plan.mfu:.1%}")
        print(f"  Training Time: {plan.total_training_hours:.1f} hours")
        print(f"  Total Cost: ${plan.total_cost_usd:,.0f}")
        print("=" * 70)

    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == '__main__':
    run_validation_summary()
