"""Tests for pymoo-based HardwareExplorer."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import (
    EvalResult, ParetoResult, HardwareSpec,
)
from llm_memory_calculator.budevolve.numeric.search_spaces import HardwareSearchSpace


def _make_hw_eval(flops, bw, mem, throughput, ttft, cost):
    return EvalResult(
        throughput_rps=throughput, ttft_ms=ttft,
        tpot_ms=10.0, feasible=True,
        config={"flops_tflops": flops, "offchip_mem_bw_gbps": bw,
                "off_chip_mem_size_gb": mem, "estimated_cost_usd": cost},
    )


class TestHardwareSearchSpace:
    def test_bounds(self):
        space = HardwareSearchSpace()
        assert len(space.xl) == 7
        assert len(space.xu) == 7
        assert all(lo < hi for lo, hi in zip(space.xl, space.xu))


class TestHardwareExplorer:
    def test_explore_returns_pareto_result(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        results = [
            _make_hw_eval(312, 1600, 40, 50.0, 100.0, 10000),
            _make_hw_eval(990, 3350, 80, 120.0, 40.0, 25000),
            _make_hw_eval(1979, 8000, 192, 200.0, 20.0, 40000),
        ]

        with patch.object(HardwareExplorer, "_run_search") as mock_run:
            mock_run.return_value = results
            explorer = HardwareExplorer(model="test-model")
            result = explorer.explore(
                objectives=["throughput", "cost"],
                n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            assert len(result.pareto_front) > 0

    def test_what_if_sweeps_parameter(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            result = explorer.what_if(
                base_hardware="A100_40GB_GPU",
                param="offchip_mem_bw_gbps",
                param_range=(1600, 8000),
                steps=5,
            )
            assert len(result) == 5
            assert all("offchip_mem_bw_gbps" in r for r in result)

    def test_design_for_model(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        results = [
            _make_hw_eval(312, 1600, 40, 50.0, 100.0, 10000),
            _make_hw_eval(990, 3350, 80, 120.0, 40.0, 25000),
        ]

        with patch.object(HardwareExplorer, "_run_search") as mock_run:
            mock_run.return_value = results
            explorer = HardwareExplorer(model="test-model")
            result = explorer.design_for_model(
                target_throughput_rps=100.0, n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            # Only the second result meets 100 rps target
            for item in result.pareto_front:
                assert item["throughput_rps"] >= 100.0

    def test_compare_vs_real(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=150.0, ttft_ms=30.0, tpot_ms=8.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0,
                              off_chip_mem_size_gb=80.0)
            result = explorer.compare_vs_real(hw, real_hardware=["H100_GPU"])
            assert "hypothetical" in result
            assert "H100_GPU" in result
