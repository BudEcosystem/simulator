"""Tests for OpenEvolve-based AlgorithmEvolver."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight


class TestPromptTemplates:
    def test_build_scheduler_prompt(self):
        from llm_memory_calculator.budevolve.evolve.prompt_templates import build_scheduler_prompt

        report = RooflineReport(
            overall_bottleneck="memory_bandwidth",
            compute_utilization=0.42,
            memory_bw_utilization=0.91,
            interconnect_utilization=0.05,
            per_operator=[
                OperatorInsight("attn", "memory", 8.0, 0.3, 0.8, 0.4),
                OperatorInsight("ffn", "compute", 95.0, 1.2, 0.5, 0.3),
            ],
            recommendations=["Increase batch size"],
        )

        prompt = build_scheduler_prompt(
            current_code="def schedule(queue): return queue[:32]",
            metrics={"throughput_rps": 45.0, "ttft_ms": 320.0},
            roofline=report,
        )
        assert "memory_bandwidth" in prompt.lower() or "memory bandwidth" in prompt.lower()
        assert "schedule" in prompt
        assert "45.0" in prompt or "45" in prompt


class TestEvaluatorBridge:
    def test_evaluate_returns_score(self):
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            from llm_memory_calculator.budevolve.types import EvalResult
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0,
                slo_compliance=0.95, feasible=True,
            )
            mock_cls.return_value = mock_eval

            bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
            score = bridge.evaluate("dummy_path.py")

            assert isinstance(score, dict)
            assert "combined_score" in score
            assert score["combined_score"] > 0


class TestAlgorithmEvolver:
    def test_init(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model="test-model", hardware="H100_GPU",
            llm_endpoint="https://api.openai.com/v1",
            llm_model="gpt-4o-mini",
        )
        assert evolver._model == "test-model"

    def test_get_base_algorithm(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model="test", hardware="H100_GPU",
            llm_endpoint="http://localhost", llm_model="test",
        )
        code = evolver.get_base_algorithm("scheduler")
        assert "def schedule_batch" in code or "def schedule" in code
