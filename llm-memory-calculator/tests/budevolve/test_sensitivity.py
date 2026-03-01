"""Tests for sensitivity analysis."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import EvalResult


class TestSensitivityAnalyzer:
    def test_analyze_returns_ranking(self):
        from llm_memory_calculator.budevolve.numeric.sensitivity import SensitivityAnalyzer

        call_count = 0
        def mock_evaluate(config, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            tp = config.tensor_parallel
            bs = config.batch_size
            return EvalResult(
                throughput_rps=bs * 2.0 / tp,
                ttft_ms=100.0 / tp,
                tpot_ms=10.0, feasible=True,
            )

        with patch("llm_memory_calculator.budevolve.numeric.sensitivity.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.side_effect = mock_evaluate
            mock_cls.return_value = mock_eval

            analyzer = SensitivityAnalyzer(model="test", hardware="H100_GPU")
            result = analyzer.analyze(target="throughput")

            assert "ranking" in result
            assert "scores" in result
            assert len(result["ranking"]) > 0

    def test_hardware_sensitivity(self):
        from llm_memory_calculator.budevolve.numeric.sensitivity import SensitivityAnalyzer

        def mock_hw_evaluate(hw, **kwargs):
            return EvalResult(
                throughput_rps=hw.flops_tflops * 0.1,
                ttft_ms=1000.0 / hw.offchip_mem_bw_gbps,
                tpot_ms=10.0, feasible=True,
            )

        with patch("llm_memory_calculator.budevolve.numeric.sensitivity.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_hardware.side_effect = mock_hw_evaluate
            mock_cls.return_value = mock_eval

            analyzer = SensitivityAnalyzer(model="test", hardware="H100_GPU")
            result = analyzer.analyze_hardware(target="throughput")

            assert "ranking" in result
            assert len(result["ranking"]) > 0
