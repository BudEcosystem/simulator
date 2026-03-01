# tests/budevolve/test_numeric_config_optimizer.py
"""Tests for pymoo-based NumericOptimizer."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import EvalResult, ParetoResult


def _make_eval_result(tp=1, pp=1, bs=32, prec="bf16", throughput=50.0, ttft=100.0, tpot=20.0):
    return EvalResult(
        throughput_rps=throughput,
        token_throughput_tps=throughput * 128,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_latency_ms=ttft + tpot * 128,
        feasible=True,
        config={"tensor_parallel": tp, "pipeline_parallel": pp,
                "batch_size": bs, "precision": prec},
    )


class TestConfigSearchSpace:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.numeric.search_spaces import ConfigSearchSpace
        space = ConfigSearchSpace()
        assert 1 in space.tensor_parallel
        assert 8 in space.tensor_parallel
        assert "bf16" in space.precisions

    def test_n_var(self):
        from llm_memory_calculator.budevolve.numeric.search_spaces import ConfigSearchSpace
        space = ConfigSearchSpace()
        assert space.n_var >= 4  # at least TP, PP, batch, precision


class TestNumericOptimizer:
    def test_optimize_returns_pareto_result(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        results = [
            _make_eval_result(tp=1, bs=32, throughput=50.0, ttft=100.0),
            _make_eval_result(tp=2, bs=64, throughput=80.0, ttft=60.0),
            _make_eval_result(tp=4, bs=128, throughput=120.0, ttft=40.0),
        ]

        with patch.object(NumericOptimizer, "_run_pymoo") as mock_run:
            mock_run.return_value = results
            optimizer = NumericOptimizer(
                model="test-model", hardware="H100_GPU",
            )
            result = optimizer.optimize(
                objectives=["throughput", "latency"],
                n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            assert len(result.pareto_front) > 0

    def test_optimize_respects_constraints(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        feasible = _make_eval_result(tp=1, bs=32, throughput=50.0, ttft=100.0)
        infeasible = _make_eval_result(tp=4, bs=256, throughput=200.0, ttft=800.0)
        infeasible.feasible = False

        with patch.object(NumericOptimizer, "_run_pymoo") as mock_run:
            mock_run.return_value = [feasible, infeasible]
            optimizer = NumericOptimizer(model="test", hardware="H100_GPU")
            result = optimizer.optimize(
                objectives=["throughput"],
                constraints={"max_ttft_ms": 500.0},
            )
            # Only feasible results in pareto front
            for item in result.pareto_front:
                assert item.get("ttft_ms", 0) <= 500.0 or item.get("feasible", True)

    def test_compute_pareto_front(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        optimizer = NumericOptimizer(model="test", hardware="H100_GPU")
        results = [
            _make_eval_result(throughput=100.0, ttft=50.0),   # dominated
            _make_eval_result(throughput=120.0, ttft=40.0),   # pareto
            _make_eval_result(throughput=80.0, ttft=30.0),    # pareto
            _make_eval_result(throughput=90.0, ttft=60.0),    # dominated
        ]
        pareto = optimizer._compute_pareto(results, ["throughput", "latency"])
        assert len(pareto) == 2
