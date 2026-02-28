"""Tests for cluster-level analysis."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.genz.serving.cluster import ClusterAnalyzer


MODEL = "meta-llama/Meta-Llama-3.1-8B"
HARDWARE = "A100_80GB_GPU"


@pytest.fixture
def analyzer():
    return ClusterAnalyzer(MODEL, HARDWARE)


@pytest.fixture
def patched_analyzer():
    """Analyzer with mocked GenZ calls for deterministic tests."""
    with patch(
        "llm_memory_calculator.genz.serving.cluster.ClusterAnalyzer._get_latencies",
        return_value=(5.12, 0.5),
    ):
        yield ClusterAnalyzer(MODEL, HARDWARE)


class TestAnalyzeScaling:
    def test_analyze_scaling_linear(self, patched_analyzer):
        """[1, 2, 4, 8] instances, verify linear scaling."""
        result = patched_analyzer.analyze_scaling(
            instance_counts=[1, 2, 4, 8],
            input_tokens=512,
            output_tokens=128,
        )
        scaling = result["scaling_results"]
        assert len(scaling) == 4

        # Throughput at N instances should be N * single-instance throughput
        base_tp = scaling[0]["throughput_rps"]
        for entry in scaling:
            expected = base_tp * entry["instances"]
            assert abs(entry["throughput_rps"] - expected) < 1e-6
            assert entry["scaling_efficiency"] == 1.0

    def test_scaling_result_structure(self, patched_analyzer):
        """Verify all expected keys in results."""
        result = patched_analyzer.analyze_scaling(
            instance_counts=[1, 2, 4, 8],
        )
        # Top-level keys
        assert "model" in result
        assert "hardware" in result
        assert "single_instance_throughput_rps" in result
        assert "single_instance_latency_ms" in result
        assert "scaling_results" in result

        # Per-instance keys
        entry = result["scaling_results"][0]
        expected_keys = {
            "instances",
            "throughput_rps",
            "tokens_per_second",
            "scaling_efficiency",
            "time_per_request_ms",
        }
        assert expected_keys.issubset(set(entry.keys()))

    def test_tokens_per_second(self, patched_analyzer):
        """tokens_per_second = throughput_rps * (input + output tokens)."""
        result = patched_analyzer.analyze_scaling(
            instance_counts=[1],
            input_tokens=512,
            output_tokens=128,
        )
        entry = result["scaling_results"][0]
        expected_tps = entry["throughput_rps"] * (512 + 128)
        assert abs(entry["tokens_per_second"] - expected_tps) < 1e-6

    def test_single_instance_throughput(self, patched_analyzer):
        """single_instance_throughput_rps matches scaling_results for n=1."""
        result = patched_analyzer.analyze_scaling(instance_counts=[1])
        assert abs(
            result["single_instance_throughput_rps"]
            - result["scaling_results"][0]["throughput_rps"]
        ) < 1e-9


class TestOptimizeParallelism:
    def test_optimize_parallelism_8_devices(self, patched_analyzer):
        """8 devices should produce multiple valid TP/PP combinations."""
        mock_prefill = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                Latency=512 * 0.01 / kw.get("tensor_parallel", 1),
                Throughput=100.0,
            ),
        )
        mock_decode = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                Latency=0.5 / kw.get("tensor_parallel", 1),
                Throughput=200.0,
            ),
        )
        with patch(
            "llm_memory_calculator.genz.prefill_moddeling", mock_prefill,
        ), patch(
            "llm_memory_calculator.genz.decode_moddeling", mock_decode,
        ):
            result = patched_analyzer.optimize_parallelism(
                num_devices=8,
                input_tokens=512,
                output_tokens=128,
            )
            assert result["num_devices"] == 8
            assert result["optimal"] is not None
            assert len(result["all_configs"]) > 0

            # All configs must use at most 8 devices
            for cfg in result["all_configs"]:
                total_used = cfg["devices_per_instance"] * cfg["num_instances"]
                assert total_used <= 8
                assert cfg["tensor_parallel"] * cfg["pipeline_parallel"] == cfg["devices_per_instance"]

            # Optimal should have highest throughput
            best_tp = result["optimal"]["total_throughput_rps"]
            for cfg in result["all_configs"]:
                assert cfg["total_throughput_rps"] <= best_tp + 1e-9

    def test_optimize_parallelism_1_device(self, patched_analyzer):
        """1 device should only allow TP=1, PP=1."""
        result = patched_analyzer.optimize_parallelism(
            num_devices=1,
            input_tokens=512,
            output_tokens=128,
        )
        assert result["optimal"] is not None
        assert result["optimal"]["tensor_parallel"] == 1
        assert result["optimal"]["pipeline_parallel"] == 1
        assert result["optimal"]["num_instances"] == 1
        assert len(result["all_configs"]) == 1

    def test_optimize_parallelism_structure(self, patched_analyzer):
        """Verify all expected keys in parallelism config entries."""
        result = patched_analyzer.optimize_parallelism(num_devices=4)
        assert result["optimal"] is not None
        cfg = result["optimal"]
        expected_keys = {
            "tensor_parallel",
            "pipeline_parallel",
            "num_instances",
            "devices_per_instance",
            "prefill_latency_ms",
            "decode_latency_ms",
            "time_per_request_ms",
            "instance_throughput_rps",
            "total_throughput_rps",
        }
        assert expected_keys.issubset(set(cfg.keys()))


class TestRealModel:
    """Integration test using real GenZ modeling (may be slow)."""

    def test_scaling_real_model(self):
        """Smoke test with real model."""
        try:
            from llm_memory_calculator.genz import prefill_moddeling
            prefill_moddeling(
                model=MODEL, batch_size=1, input_tokens=128,
                system_name=HARDWARE, bits="bf16",
            )
        except Exception:
            pytest.skip("GenZ engine not available for real model test")

        analyzer = ClusterAnalyzer(MODEL, HARDWARE)
        result = analyzer.analyze_scaling(
            instance_counts=[1, 2],
            input_tokens=512,
            output_tokens=128,
        )
        assert result["single_instance_throughput_rps"] > 0
        assert len(result["scaling_results"]) == 2
