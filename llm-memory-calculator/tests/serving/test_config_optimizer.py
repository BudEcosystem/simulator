"""Tests for the configuration optimizer."""
import pytest

from llm_memory_calculator.genz.serving.config_optimizer import (
    SearchSpace,
    OptimizationConstraints,
    ConfigResult,
    ConfigOptimizer,
)

MODEL = "meta-llama/Meta-Llama-3.1-8B"
HARDWARE = "A100_80GB_GPU"


@pytest.fixture
def optimizer():
    """Optimizer with a known model/hardware combo and 8 devices."""
    return ConfigOptimizer(model=MODEL, hardware=HARDWARE, num_devices=8)


@pytest.fixture
def small_search_space():
    """Restricted search space for fast tests."""
    return SearchSpace(
        tensor_parallel=[1, 2],
        pipeline_parallel=[1],
        batch_sizes=[1, 32],
        precisions=["bf16"],
        block_sizes=[16],
        enable_chunked_prefill=[False],
        enable_prefix_caching=[False],
        gpu_memory_utilization=[0.90],
    )


class TestSearchSpace:
    def test_search_space_defaults(self):
        space = SearchSpace()
        assert space.tensor_parallel == [1, 2, 4, 8]
        assert space.pipeline_parallel == [1, 2, 4]
        assert space.batch_sizes == [1, 8, 32, 64, 128, 256]
        assert space.precisions == ["bf16", "fp8", "int8"]
        assert space.block_sizes == [8, 16, 32]
        assert space.enable_chunked_prefill == [True, False]
        assert space.enable_prefix_caching == [True, False]
        assert space.gpu_memory_utilization == [0.85, 0.90, 0.95]


class TestOptimizationConstraints:
    def test_optimization_constraints_defaults(self):
        c = OptimizationConstraints()
        assert c.max_ttft_ms == float('inf')
        assert c.max_tpot_ms == float('inf')
        assert c.max_memory_gb == float('inf')
        assert c.max_power_w == float('inf')
        assert c.max_total_devices == 64


class TestOptimizeThroughput:
    def test_optimize_throughput(self, optimizer, small_search_space):
        result = optimizer.optimize(
            target="throughput",
            search_space=small_search_space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        assert result["best_config"] is not None
        assert "tensor_parallel" in result["best_config"]
        assert "pipeline_parallel" in result["best_config"]
        assert "batch_size" in result["best_config"]
        assert "precision" in result["best_config"]
        assert result["best_score"] > 0
        assert result["feasible_count"] > 0
        assert result["total_evaluated"] > 0
        assert "best_metrics" in result
        assert result["best_metrics"]["throughput_rps"] > 0


class TestOptimizeWithConstraints:
    def test_optimize_with_constraints(self, optimizer, small_search_space):
        constraints = OptimizationConstraints(max_ttft_ms=100)
        result = optimizer.optimize(
            target="throughput",
            constraints=constraints,
            search_space=small_search_space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        # All feasible results must respect the TTFT constraint
        for r in result["all_results"]:
            if r["feasible"]:
                assert r["ttft_ms"] <= 100


class TestOptimizeLatency:
    def test_optimize_latency(self, optimizer):
        space = SearchSpace(
            tensor_parallel=[1],
            pipeline_parallel=[1],
            batch_sizes=[1, 8, 32, 64],
            precisions=["bf16"],
        )
        result = optimizer.optimize(
            target="latency",
            search_space=space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        assert result["best_config"] is not None
        # For latency optimization, smaller batch sizes tend to win
        assert result["best_config"]["batch_size"] <= 32


class TestOptimizeNoFeasible:
    def test_optimize_no_feasible(self, optimizer, small_search_space):
        # Extremely tight TTFT constraint that no config can meet
        constraints = OptimizationConstraints(max_ttft_ms=0.001)
        result = optimizer.optimize(
            target="throughput",
            constraints=constraints,
            search_space=small_search_space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        assert result["best_config"] is None
        assert result["feasible_count"] == 0
        assert result["best_score"] == 0.0


class TestParetoOptimize:
    def test_pareto_optimize(self, optimizer, small_search_space):
        result = optimizer.pareto_optimize(
            objectives=["throughput", "latency"],
            search_space=small_search_space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        assert len(result["pareto_front"]) > 0
        assert result["feasible_count"] > 0
        assert result["objectives"] == ["throughput", "latency"]


class TestParetoDominance:
    def test_pareto_dominance(self, optimizer, small_search_space):
        result = optimizer.pareto_optimize(
            objectives=["throughput", "latency"],
            search_space=small_search_space,
            budget=10,
            input_tokens=256,
            output_tokens=64,
        )
        front = result["pareto_front"]
        # No point on the Pareto front should dominate another
        # For throughput: higher is better. For latency: lower is better (negated internally).
        for i, pi in enumerate(front):
            for j, pj in enumerate(front):
                if i == j:
                    continue
                # pi dominates pj if pi >= pj in all and > in at least one
                # Throughput: higher better; Latency (total): lower better
                ti, tj = pi["throughput_rps"], pj["throughput_rps"]
                li = pi["prefill_latency_ms"] + pi["decode_latency_ms"]
                lj = pj["prefill_latency_ms"] + pj["decode_latency_ms"]
                # Check that pi does NOT dominate pj
                throughput_geq = ti >= tj
                latency_leq = li <= lj  # lower latency is better
                throughput_gt = ti > tj
                latency_lt = li < lj
                dominates = (throughput_geq and latency_leq) and (throughput_gt or latency_lt)
                assert not dominates, (
                    f"Front point {i} dominates point {j}: "
                    f"throughput ({ti} vs {tj}), latency ({li} vs {lj})"
                )


class TestSensitivityAnalysis:
    def test_sensitivity_analysis(self, optimizer):
        result = optimizer.analyze_sensitivity(
            target="throughput",
            input_tokens=256,
            output_tokens=64,
        )
        assert "sensitivity" in result
        assert "ranking" in result
        assert "baseline_config" in result
        assert "baseline_score" in result
        assert result["target"] == "throughput"

        # Should have entries for each tested parameter
        assert len(result["ranking"]) > 0
        assert result.get("method") == "morris_elementary_effects"

        # Each parameter should have Morris statistics
        for param, data in result["sensitivity"].items():
            assert "mu_star" in data
            assert "mu" in data
            assert "sigma" in data
            assert data["mu_star"] >= 0


class TestCustomSearchSpace:
    def test_custom_search_space(self, optimizer):
        custom_space = SearchSpace(
            tensor_parallel=[1, 2],
            pipeline_parallel=[1],
            batch_sizes=[8, 16],
            precisions=["bf16"],
        )
        result = optimizer.optimize(
            target="throughput",
            search_space=custom_space,
            budget=10,
            input_tokens=256,
            output_tokens=64,
        )
        # All evaluated configs should be within the custom search space bounds
        for r in result["all_results"]:
            cfg = r["config"]
            assert cfg["tensor_parallel"] in [1, 2]
            assert cfg["pipeline_parallel"] in [1]
            assert cfg["batch_size"] in [8, 16]
            assert cfg["precision"] in ["bf16"]


class TestBudgetRespected:
    def test_budget_respected(self, optimizer, small_search_space):
        result = optimizer.optimize(
            target="throughput",
            search_space=small_search_space,
            budget=5,
            input_tokens=256,
            output_tokens=64,
        )
        assert result["total_evaluated"] <= 5
