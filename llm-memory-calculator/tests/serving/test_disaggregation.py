"""Tests for prefill-decode disaggregation analysis."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.genz.serving.disaggregation import (
    DisaggregationAnalyzer,
)


MODEL = "meta-llama/Meta-Llama-3.1-8B"
HARDWARE = "A100_80GB_GPU"


def _mock_prefill(model, batch_size, input_tokens, system_name, bits, **kw):
    """Return a mock prefill result with latency proportional to tokens."""
    return SimpleNamespace(Latency=input_tokens * 0.01, Throughput=100.0)


def _mock_decode(model, batch_size, input_tokens, output_tokens, system_name, bits, **kw):
    """Return a mock per-token decode latency."""
    return SimpleNamespace(Latency=0.5, Throughput=200.0)


@pytest.fixture
def analyzer():
    return DisaggregationAnalyzer(MODEL, HARDWARE)


@pytest.fixture
def patched_analyzer():
    """Analyzer with mocked GenZ calls for deterministic tests."""
    with patch(
        "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
        return_value=(5.12, 0.5),
    ), patch(
        "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._estimate_kv_bytes",
        return_value=512 * 131072,
    ):
        yield DisaggregationAnalyzer(MODEL, HARDWARE)


class TestAnalyzeBasic:
    def test_analyze_basic(self, patched_analyzer):
        """Analyze with 2 prefill + 6 decode instances, verify result structure."""
        result = patched_analyzer.analyze(
            prefill_instances=2,
            decode_instances=6,
            input_tokens=512,
            output_tokens=128,
        )
        expected_keys = {
            "prefill_instances",
            "decode_instances",
            "total_instances",
            "prefill_latency_ms",
            "decode_latency_ms",
            "kv_transfer_time_ms",
            "kv_transfer_bytes",
            "effective_ttft_ms",
            "prefill_throughput_rps",
            "decode_throughput_rps",
            "system_throughput_rps",
            "bottleneck",
            "prefill_utilization",
            "decode_utilization",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["prefill_instances"] == 2
        assert result["decode_instances"] == 6
        assert result["total_instances"] == 8
        assert result["prefill_latency_ms"] > 0
        assert result["decode_latency_ms"] > 0
        assert result["system_throughput_rps"] > 0
        assert result["bottleneck"] in ("prefill", "decode")


class TestAnalyzeBottleneck:
    def test_analyze_bottleneck_prefill(self):
        """With 1 prefill and 7 decode, prefill should be bottleneck."""
        with patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
            return_value=(10.0, 0.5),
        ), patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._estimate_kv_bytes",
            return_value=1024,
        ):
            analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
            # 1 prefill: throughput = 1 * 1000/10 = 100 rps
            # 7 decode: throughput = 7 * 1000/(0.5*128) = 7*15.625 = 109.375 rps
            result = analyzer.analyze(
                prefill_instances=1,
                decode_instances=7,
                input_tokens=512,
                output_tokens=128,
            )
            assert result["bottleneck"] == "prefill"
            assert result["prefill_throughput_rps"] < result["decode_throughput_rps"]

    def test_analyze_bottleneck_decode(self):
        """With 7 prefill and 1 decode, decode should be bottleneck."""
        with patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
            return_value=(10.0, 0.5),
        ), patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._estimate_kv_bytes",
            return_value=1024,
        ):
            analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
            # 7 prefill: throughput = 7 * 1000/10 = 700 rps
            # 1 decode: throughput = 1 * 1000/(0.5*128) = 15.625 rps
            result = analyzer.analyze(
                prefill_instances=7,
                decode_instances=1,
                input_tokens=512,
                output_tokens=128,
            )
            assert result["bottleneck"] == "decode"
            assert result["decode_throughput_rps"] < result["prefill_throughput_rps"]


class TestFindOptimalRatio:
    def test_find_optimal_ratio(self, patched_analyzer):
        """8 total instances, verify optimal is found with all ratios enumerated."""
        result = patched_analyzer.find_optimal_ratio(
            total_instances=8,
            input_tokens=512,
            output_tokens=128,
        )
        assert result["total_instances"] == 8
        assert result["optimal"] is not None
        assert len(result["all_ratios"]) == 7  # 1:7 through 7:1
        # Optimal should have highest throughput
        optimal_tp = result["optimal"]["system_throughput_rps"]
        for r in result["all_ratios"]:
            assert r["throughput_rps"] <= optimal_tp + 1e-9

    def test_optimal_ratio_valid_split(self, patched_analyzer):
        """Optimal split must sum to total instances."""
        result = patched_analyzer.find_optimal_ratio(total_instances=4)
        opt = result["optimal"]
        assert opt["prefill_instances"] + opt["decode_instances"] == 4


class TestCompareColocatedVsDisaggregated:
    def test_compare_has_both_modes(self, patched_analyzer):
        """Verify both modes are compared with speedup metric."""
        result = patched_analyzer.compare_colocated_vs_disaggregated(
            total_instances=8,
            input_tokens=512,
            output_tokens=128,
        )
        assert "colocated" in result
        assert "disaggregated" in result
        assert "speedup" in result
        assert "recommendation" in result
        assert result["colocated"]["throughput_rps"] > 0
        assert result["disaggregated"]["throughput_rps"] > 0
        assert result["speedup"] > 0
        assert result["recommendation"] in ("colocated", "disaggregated")

    def test_compare_colocated_structure(self, patched_analyzer):
        """Verify colocated result has expected keys."""
        result = patched_analyzer.compare_colocated_vs_disaggregated(
            total_instances=4,
        )
        co = result["colocated"]
        assert "instances" in co
        assert "throughput_rps" in co
        assert "ttft_ms" in co
        assert "time_per_request_ms" in co


class TestKVTransferCost:
    def test_higher_bandwidth_lower_transfer_time(self):
        """Higher KV transfer bandwidth should yield lower transfer time."""
        with patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
            return_value=(5.0, 0.5),
        ), patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._estimate_kv_bytes",
            return_value=67108864,  # 64 MB
        ):
            analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
            result_low_bw = analyzer.analyze(
                prefill_instances=2,
                decode_instances=6,
                kv_transfer_bw_gbps=50.0,
            )
            result_high_bw = analyzer.analyze(
                prefill_instances=2,
                decode_instances=6,
                kv_transfer_bw_gbps=200.0,
            )
            assert result_low_bw["kv_transfer_time_ms"] > result_high_bw["kv_transfer_time_ms"]
            assert result_low_bw["effective_ttft_ms"] > result_high_bw["effective_ttft_ms"]

    def test_kv_transfer_time_positive(self, patched_analyzer):
        """KV transfer time must be positive."""
        result = patched_analyzer.analyze(
            prefill_instances=4,
            decode_instances=4,
        )
        assert result["kv_transfer_time_ms"] > 0
        assert result["kv_transfer_bytes"] > 0


class TestKVBytesIncludesOutput:
    def test_kv_bytes_includes_output_tokens(self):
        """KV transfer bytes should account for both input and output tokens."""
        with patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
            return_value=(5.0, 0.5),
        ):
            analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
            # _estimate_kv_bytes should include output_tokens
            kv_input_only = analyzer._estimate_kv_bytes(512, 0)
            kv_with_output = analyzer._estimate_kv_bytes(512, 128)
            assert kv_with_output > kv_input_only


class TestNetworkContention:
    def test_queuing_factor_in_result(self):
        """Result should include link_utilization and queuing_factor."""
        with patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._get_latencies",
            return_value=(5.0, 0.5),
        ), patch(
            "llm_memory_calculator.genz.serving.disaggregation.DisaggregationAnalyzer._estimate_kv_bytes",
            return_value=67108864,
        ):
            analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
            result = analyzer.analyze(
                prefill_instances=2, decode_instances=6,
                kv_transfer_bw_gbps=100.0,
            )
            assert "link_utilization" in result
            assert "queuing_factor" in result
            assert result["queuing_factor"] >= 1.0
            assert 0 <= result["link_utilization"] <= 0.95


class TestRealModel:
    """Integration test using real GenZ modeling (may be slow)."""

    def test_analyze_real_model(self):
        """Smoke test with real model - should not crash."""
        try:
            from llm_memory_calculator.genz import prefill_moddeling
            # Quick check if GenZ works
            prefill_moddeling(
                model=MODEL, batch_size=1, input_tokens=128,
                system_name=HARDWARE, bits="bf16",
            )
        except Exception:
            pytest.skip("GenZ engine not available for real model test")

        analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
        result = analyzer.analyze(
            prefill_instances=2,
            decode_instances=6,
            input_tokens=512,
            output_tokens=128,
        )
        assert result["system_throughput_rps"] > 0
        assert result["prefill_latency_ms"] > 0
