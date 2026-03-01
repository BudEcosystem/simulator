"""Integration tests -- run real GenZ evaluations (slow, ~30s)."""
import pytest

# Mark all tests as integration (skip with: pytest -m "not integration")
pytestmark = pytest.mark.integration


class TestEvaluatorIntegration:
    def test_evaluate_real_model(self):
        from llm_memory_calculator.budevolve import BudSimEvaluator, ServingConfig
        evaluator = BudSimEvaluator()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        result = evaluator.evaluate_config(cfg, input_tokens=512, output_tokens=128)

        assert result.feasible is True
        assert result.throughput_rps > 0
        assert result.ttft_ms > 0
        assert result.tpot_ms > 0

    def test_evaluate_hardware_spec(self):
        from llm_memory_calculator.budevolve import BudSimEvaluator, HardwareSpec
        evaluator = BudSimEvaluator()
        hw = HardwareSpec(
            flops_tflops=990.0, offchip_mem_bw_gbps=3350.0,
            off_chip_mem_size_gb=80.0,
        )
        result = evaluator.evaluate_hardware(
            hw, model="meta-llama/Meta-Llama-3.1-8B",
            input_tokens=512, output_tokens=128,
        )
        assert result.throughput_rps > 0


class TestRooflineIntegration:
    def test_analyze_real_model(self):
        from llm_memory_calculator.budevolve import RooflineAnalyzer, ServingConfig
        analyzer = RooflineAnalyzer()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        report = analyzer.analyze_config(cfg, input_tokens=512)

        assert report.overall_bottleneck in ("compute", "memory_bandwidth", "interconnect", "unknown")
        assert len(report.per_operator) > 0

    def test_format_for_prompt(self):
        from llm_memory_calculator.budevolve import RooflineAnalyzer, ServingConfig
        analyzer = RooflineAnalyzer()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        report = analyzer.analyze_config(cfg, input_tokens=512)
        text = analyzer.format_for_prompt(report)

        assert len(text) > 100
        assert "bottleneck" in text.lower() or "utilization" in text.lower()


class TestSensitivityIntegration:
    @pytest.mark.slow
    def test_config_sensitivity(self):
        from llm_memory_calculator.budevolve import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(
            model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU",
        )
        result = analyzer.analyze(target="throughput")

        assert "ranking" in result
        assert "batch_size" in result["ranking"] or "tensor_parallel" in result["ranking"]
