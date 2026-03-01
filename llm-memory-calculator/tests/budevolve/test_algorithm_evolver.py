"""Tests for OpenEvolve-based AlgorithmEvolver."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight, EvalResult


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
    def test_evaluate_valid_code(self):
        """Bridge should load the file, validate code, and return a combined score."""
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        # Write a valid evolved algorithm to a temp file
        code = (
            "def schedule_batch(queue, max_batch_size=256):\n"
            "    if not queue:\n"
            "        return []\n"
            "    batch = []\n"
            "    for r in queue:\n"
            "        if len(batch) >= max_batch_size:\n"
            "            break\n"
            "        batch.append(r)\n"
            "    return batch\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        try:
            with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator") as mock_cls:
                mock_eval = MagicMock()
                mock_eval.evaluate_config.return_value = EvalResult(
                    throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0,
                    slo_compliance=0.95, feasible=True,
                )
                mock_cls.return_value = mock_eval

                bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
                score = bridge.evaluate(path)

                assert isinstance(score, dict)
                assert "combined_score" in score
                assert score["combined_score"] > 0
                assert score["code_quality"] > 0
                assert "error" not in score
        finally:
            os.unlink(path)

    def test_evaluate_syntax_error_returns_zero(self):
        """Bridge should reject files with syntax errors."""
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken(:\n    pass\n")
            path = f.name

        try:
            with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator"):
                bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
                score = bridge.evaluate(path)
                assert score["combined_score"] == 0.0
                assert "error" in score
        finally:
            os.unlink(path)

    def test_evaluate_missing_file_returns_zero(self):
        """Bridge should handle missing files gracefully."""
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator"):
            bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
            score = bridge.evaluate("/nonexistent/path.py")
            assert score["combined_score"] == 0.0
            assert "error" in score

    def test_evaluate_unsafe_import_rejected(self):
        """Bridge should reject code with unsafe imports."""
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import os\nos.system('echo pwned')\n")
            path = f.name

        try:
            with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator"):
                bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
                score = bridge.evaluate(path)
                assert score["combined_score"] == 0.0
                assert "error" in score
                assert "Unsafe import" in score["error"]
        finally:
            os.unlink(path)


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

    def test_get_base_algorithm_unknown_type_raises(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model="test", hardware="H100_GPU",
            llm_endpoint="http://localhost", llm_model="test",
        )
        with pytest.raises(ValueError, match="Unknown algorithm type"):
            evolver.get_base_algorithm("nonexistent")

    def test_evolve_scheduler_fallback_without_openevolve(self):
        """evolve_scheduler should return base algo when openevolve is missing."""
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver

        with patch("llm_memory_calculator.budevolve.evolve.algorithm_evolver.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=50.0, ttft_ms=100.0, tpot_ms=20.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            with patch("llm_memory_calculator.budevolve.evolve.algorithm_evolver.RooflineAnalyzer") as mock_ra:
                mock_ra.return_value.analyze_config.side_effect = Exception("no GenZ")

                evolver = AlgorithmEvolver(
                    model="test", hardware="H100_GPU",
                    llm_endpoint="http://localhost", llm_model="test",
                )
                result = evolver.evolve_scheduler(iterations=5)
                assert "best_code" in result
                assert result["error"] == "openevolve not installed"
                assert result["improvement_pct"] == 0.0
