"""Tests for the serving simulator."""
import pytest
import warnings

from llm_memory_calculator.genz.serving.constants import NS_PER_MS, NS_PER_S
from llm_memory_calculator.genz.serving.workload import WorkloadConfig
from llm_memory_calculator.genz.serving.batch_scheduler import SchedulerConfig
from llm_memory_calculator.genz.serving.slo_tracker import SLOTargets
from llm_memory_calculator.genz.serving.simulator import (
    ServingSimulator, ServingSimulationResult,
)


@pytest.fixture
def simulator():
    """Simulator with a known model/hardware combo."""
    return ServingSimulator(
        model="meta-llama/Meta-Llama-3.1-8B",
        hardware="A100_80GB_GPU",
        precision="bf16",
        tensor_parallel=1,
    )


@pytest.fixture
def small_workload():
    """Small workload for fast testing."""
    return WorkloadConfig(
        arrival_rate_rps=5.0,
        arrival_pattern="constant",
        num_requests=10,
        input_length_distribution={"dist": "fixed", "mean": 256, "min": 64, "max": 1024},
        output_length_distribution={"dist": "fixed", "mean": 32, "min": 8, "max": 128},
        random_seed=42,
    )


@pytest.fixture
def small_scheduler():
    return SchedulerConfig(max_batch_size=8, max_num_batched_tokens=4096)


class TestServingSimulationResult:
    def test_default_values(self):
        result = ServingSimulationResult()
        assert result.total_duration_ms == 0.0
        assert result.total_requests_completed == 0
        assert result.overall_throughput_rps == 0.0
        assert isinstance(result.time_series, dict)
        assert isinstance(result.raw_requests, list)


class TestServingSimulatorInit:
    def test_create_simulator(self):
        sim = ServingSimulator("test-model", "A100_80GB_GPU")
        assert sim._model == "test-model"
        assert sim._hardware == "A100_80GB_GPU"

    def test_create_with_parallelism(self):
        sim = ServingSimulator(
            "test", "A100_80GB_GPU",
            tensor_parallel=4, pipeline_parallel=2,
        )
        assert sim._tensor_parallel == 4
        assert sim._pipeline_parallel == 2


class TestServingSimulatorSimulate:
    def test_basic_simulation(self, simulator, small_workload, small_scheduler):
        """Run a basic simulation with known model and verify results."""
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
        )
        assert isinstance(result, ServingSimulationResult)
        assert result.total_requests_completed > 0
        assert result.total_duration_ms > 0
        assert result.overall_throughput_rps > 0

    def test_all_requests_complete(self, simulator, small_workload, small_scheduler):
        """All requests should eventually complete."""
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
        )
        assert result.total_requests_completed == small_workload.num_requests

    def test_slo_summary_populated(self, simulator, small_workload, small_scheduler):
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
            slo_targets=SLOTargets.from_ms(ttft_ms=5000, tpot_ms=500, e2e_ms=60000),
        )
        assert "total_completed" in result.slo_summary
        assert "ttft" in result.slo_summary
        assert "goodput" in result.slo_summary

    def test_time_series_populated(self, simulator, small_workload, small_scheduler):
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
        )
        assert len(result.time_series["time_ms"]) > 0

    def test_raw_requests_populated(self, simulator, small_workload, small_scheduler):
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
        )
        assert len(result.raw_requests) == result.total_requests_completed
        for rr in result.raw_requests:
            assert "request_id" in rr
            assert "ttft_ns" in rr
            assert "e2e_ns" in rr

    def test_per_request_metrics(self, simulator, small_workload, small_scheduler):
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
        )
        assert "ttft_avg_ms" in result.per_request_metrics
        assert "tpot_avg_ms" in result.per_request_metrics
        assert "e2e_avg_ms" in result.per_request_metrics
        assert result.per_request_metrics["ttft_avg_ms"] > 0

    def test_with_power_tracking(self, simulator, small_workload, small_scheduler):
        result = simulator.simulate(
            workload_config=small_workload,
            scheduler_config=small_scheduler,
            enable_power_tracking=True,
        )
        assert len(result.power_summary) > 0
        assert result.power_summary.get("total_energy_j", 0) > 0

    def test_default_configs(self, simulator):
        """Simulation with all defaults should work."""
        wl = WorkloadConfig(
            num_requests=5,
            arrival_rate_rps=2.0,
            arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 128, "min": 64, "max": 256},
            output_length_distribution={"dist": "fixed", "mean": 16, "min": 8, "max": 32},
        )
        result = simulator.simulate(workload_config=wl)
        assert result.total_requests_completed > 0


class TestServingSimulatorSweep:
    def test_sweep_arrival_rate(self, simulator):
        wl = WorkloadConfig(
            num_requests=5,
            arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 128, "min": 64, "max": 256},
            output_length_distribution={"dist": "fixed", "mean": 8, "min": 4, "max": 16},
        )
        sc = SchedulerConfig(max_batch_size=8, max_num_batched_tokens=2048)
        results = simulator.sweep(
            parameter="arrival_rate_rps",
            values=[1.0, 5.0],
            workload_config=wl,
            scheduler_config=sc,
        )
        assert len(results["parameter_values"]) == 2
        assert len(results["throughput_rps"]) == 2
        assert all(t > 0 for t in results["throughput_rps"])

    def test_sweep_batch_size(self, simulator):
        wl = WorkloadConfig(
            num_requests=5,
            arrival_pattern="constant",
            arrival_rate_rps=2.0,
            input_length_distribution={"dist": "fixed", "mean": 128, "min": 64, "max": 256},
            output_length_distribution={"dist": "fixed", "mean": 8, "min": 4, "max": 16},
        )
        results = simulator.sweep(
            parameter="max_batch_size",
            values=[4, 16],
            workload_config=wl,
        )
        assert len(results["parameter_values"]) == 2


class TestServingSimulatorWithUnknownModel:
    def test_unknown_model_uses_fallback(self):
        """Unknown models should use fallback config, not crash."""
        sim = ServingSimulator("nonexistent/model", "A100_80GB_GPU")
        wl = WorkloadConfig(
            num_requests=3,
            arrival_rate_rps=1.0,
            arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 64, "min": 32, "max": 128},
            output_length_distribution={"dist": "fixed", "mean": 8, "min": 4, "max": 16},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.simulate(workload_config=wl)
        assert result.total_requests_completed > 0
