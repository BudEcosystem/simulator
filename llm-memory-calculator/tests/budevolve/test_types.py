"""Tests for BudEvolve core data types."""
import pytest
from dataclasses import asdict


class TestServingConfig:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import ServingConfig
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        assert cfg.model == "meta-llama/Meta-Llama-3.1-8B"
        assert cfg.hardware == "H100_GPU"
        assert cfg.tensor_parallel == 1
        assert cfg.pipeline_parallel == 1
        assert cfg.batch_size == 32
        assert cfg.precision == "bf16"
        assert cfg.gpu_memory_utilization == 0.90

    def test_custom_values(self):
        from llm_memory_calculator.budevolve.types import ServingConfig
        cfg = ServingConfig(
            model="meta-llama/Meta-Llama-3.1-70B", hardware="A100_40GB_GPU",
            tensor_parallel=4, batch_size=128, precision="fp8",
        )
        assert cfg.tensor_parallel == 4
        assert cfg.batch_size == 128
        assert cfg.precision == "fp8"


class TestHardwareSpec:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import HardwareSpec
        hw = HardwareSpec(flops_tflops=312.0, offchip_mem_bw_gbps=1600.0, off_chip_mem_size_gb=40.0)
        assert hw.flops_tflops == 312.0
        assert hw.num_nodes == 1
        assert hw.tdp_watts == 700.0

    def test_to_system_kwargs(self):
        from llm_memory_calculator.budevolve.types import HardwareSpec
        hw = HardwareSpec(
            flops_tflops=312.0, offchip_mem_bw_gbps=1600.0,
            off_chip_mem_size_gb=40.0, on_chip_mem_size_mb=40.0,
            onchip_mem_bw_gbps=18000.0, interchip_link_bw_gbps=150.0,
            frequency_ghz=1.41,
        )
        kwargs = hw.to_system_kwargs()
        assert kwargs["flops"] == 312.0
        assert kwargs["offchip_mem_bw"] == 1600.0
        assert kwargs["off_chip_mem_size"] == 40.0 * 1024  # GB -> MB
        assert kwargs["frequency"] == 1410  # GHz -> MHz


class TestEvalResult:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import EvalResult
        r = EvalResult(throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0)
        assert r.feasible is True
        assert r.memory_gb == 0.0
        assert r.power_w == 0.0

    def test_is_serializable(self):
        from llm_memory_calculator.budevolve.types import EvalResult
        r = EvalResult(throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0)
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["throughput_rps"] == 100.0


class TestOperatorInsight:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import OperatorInsight
        op = OperatorInsight(
            name="attention_prefill", bottleneck="compute",
            arithmetic_intensity=45.2,
            compute_time_ms=1.5, memory_time_ms=0.8, pct_of_total=0.35,
        )
        assert op.bottleneck == "compute"
        assert op.pct_of_total == 0.35


class TestRooflineReport:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight
        ops = [OperatorInsight("attn", "compute", 45.0, 1.5, 0.8, 0.35)]
        report = RooflineReport(
            overall_bottleneck="compute",
            compute_utilization=0.85, memory_bw_utilization=0.42,
            interconnect_utilization=0.1, per_operator=ops,
            recommendations=["Increase batch size"],
        )
        assert report.overall_bottleneck == "compute"
        assert len(report.per_operator) == 1


class TestParetoResult:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import ParetoResult
        r = ParetoResult(
            pareto_front=[{"throughput": 100}],
            all_evaluated=[{"throughput": 100}, {"throughput": 50}],
            best_per_objective={"throughput": {"throughput": 100}},
            num_generations=50, total_evaluations=500,
        )
        assert len(r.pareto_front) == 1
        assert r.total_evaluations == 500
