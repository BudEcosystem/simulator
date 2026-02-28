"""Tests for workload generation in the serving simulation."""
import pytest
import numpy as np

from llm_memory_calculator.genz.serving.constants import NS_PER_S
from llm_memory_calculator.genz.serving.request import Request
from llm_memory_calculator.genz.serving.workload import WorkloadConfig, WorkloadGenerator


# ---------------------------------------------------------------------------
# WorkloadConfig defaults
# ---------------------------------------------------------------------------

class TestWorkloadConfig:
    """Verify default configuration values."""

    def test_default_config(self):
        cfg = WorkloadConfig()
        assert cfg.arrival_rate_rps == 10.0
        assert cfg.arrival_pattern == "poisson"
        assert cfg.num_requests == 300
        assert cfg.model == "default"
        assert cfg.random_seed == 42
        assert cfg.gamma_shape == 2.0
        assert cfg.burst_period_s == 10.0
        assert cfg.burst_amplitude == 3.0
        assert cfg.input_length_distribution["dist"] == "lognormal"
        assert cfg.input_length_distribution["mean"] == 512
        assert cfg.output_length_distribution["dist"] == "lognormal"
        assert cfg.output_length_distribution["mean"] == 128


# ---------------------------------------------------------------------------
# Generation basics
# ---------------------------------------------------------------------------

class TestWorkloadGeneration:
    """Test core workload generation behavior."""

    def test_generate_returns_sorted_requests(self):
        gen = WorkloadGenerator(WorkloadConfig(num_requests=100))
        requests = gen.generate()
        arrival_times = [r.arrival_time_ns for r in requests]
        assert arrival_times == sorted(arrival_times)

    def test_generate_correct_count(self):
        for n in [1, 10, 50, 200]:
            gen = WorkloadGenerator(WorkloadConfig(num_requests=n))
            requests = gen.generate()
            assert len(requests) == n

    def test_generate_returns_request_objects(self):
        gen = WorkloadGenerator(WorkloadConfig(num_requests=5))
        requests = gen.generate()
        for r in requests:
            assert isinstance(r, Request)
            assert isinstance(r.request_id, int)
            assert isinstance(r.arrival_time_ns, int)
            assert isinstance(r.input_tokens, int)
            assert isinstance(r.max_output_tokens, int)


# ---------------------------------------------------------------------------
# Arrival patterns
# ---------------------------------------------------------------------------

class TestArrivalPatterns:
    """Test different arrival pattern generators."""

    def test_poisson_arrival_pattern(self):
        cfg = WorkloadConfig(arrival_pattern="poisson", num_requests=200)
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        arrival_times = [r.arrival_time_ns for r in requests]
        # All non-negative
        assert all(t >= 0 for t in arrival_times)
        # Monotonically increasing (sorted)
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] >= arrival_times[i - 1]

    def test_constant_arrival_pattern(self):
        rate = 10.0
        cfg = WorkloadConfig(
            arrival_pattern="constant", arrival_rate_rps=rate, num_requests=100
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        arrival_times = [r.arrival_time_ns for r in requests]
        # Compute inter-arrival times
        inter_arrivals = [
            arrival_times[i] - arrival_times[i - 1]
            for i in range(1, len(arrival_times))
        ]
        expected_gap_ns = int(NS_PER_S / rate)
        # All inter-arrival times should be roughly equal for constant pattern
        for gap in inter_arrivals:
            assert abs(gap - expected_gap_ns) < NS_PER_S * 0.01  # within 1%

    def test_gamma_arrival_pattern(self):
        cfg = WorkloadConfig(
            arrival_pattern="gamma", num_requests=200, gamma_shape=2.0
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        arrival_times = [r.arrival_time_ns for r in requests]
        assert len(requests) == 200
        assert all(t >= 0 for t in arrival_times)
        # Monotonically increasing
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] >= arrival_times[i - 1]

    def test_bursty_arrival_pattern(self):
        cfg = WorkloadConfig(
            arrival_pattern="bursty", num_requests=200,
            burst_period_s=5.0, burst_amplitude=2.0,
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        assert len(requests) == 200
        arrival_times = [r.arrival_time_ns for r in requests]
        assert all(t >= 0 for t in arrival_times)
        # Sorted
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] >= arrival_times[i - 1]

    def test_unknown_arrival_pattern_raises(self):
        cfg = WorkloadConfig(arrival_pattern="unknown_pattern")
        gen = WorkloadGenerator(cfg)
        with pytest.raises(ValueError, match="Unknown arrival pattern"):
            gen.generate()


# ---------------------------------------------------------------------------
# Determinism / seed behavior
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify reproducibility via random seeds."""

    def test_deterministic_seed(self):
        cfg = WorkloadConfig(num_requests=50, random_seed=123)
        gen1 = WorkloadGenerator(cfg)
        gen2 = WorkloadGenerator(cfg)
        r1 = gen1.generate()
        r2 = gen2.generate()
        for a, b in zip(r1, r2):
            assert a.arrival_time_ns == b.arrival_time_ns
            assert a.input_tokens == b.input_tokens
            assert a.max_output_tokens == b.max_output_tokens

    def test_different_seeds(self):
        cfg1 = WorkloadConfig(num_requests=50, random_seed=1)
        cfg2 = WorkloadConfig(num_requests=50, random_seed=999)
        r1 = WorkloadGenerator(cfg1).generate()
        r2 = WorkloadGenerator(cfg2).generate()
        # Very unlikely all arrival times match with different seeds
        arrivals1 = [r.arrival_time_ns for r in r1]
        arrivals2 = [r.arrival_time_ns for r in r2]
        assert arrivals1 != arrivals2


# ---------------------------------------------------------------------------
# Length distributions
# ---------------------------------------------------------------------------

class TestLengthDistributions:
    """Test sequence length sampling and clamping."""

    def test_input_length_range(self):
        cfg = WorkloadConfig(
            num_requests=500,
            input_length_distribution={
                "dist": "lognormal", "mean": 512, "std": 200, "min": 64, "max": 4096,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        for r in requests:
            assert 64 <= r.input_tokens <= 4096

    def test_output_length_range(self):
        cfg = WorkloadConfig(
            num_requests=500,
            output_length_distribution={
                "dist": "lognormal", "mean": 128, "std": 64, "min": 16, "max": 2048,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        for r in requests:
            assert 16 <= r.max_output_tokens <= 2048

    def test_lognormal_distribution(self):
        target_mean = 512
        cfg = WorkloadConfig(
            num_requests=2000,
            input_length_distribution={
                "dist": "lognormal", "mean": target_mean, "std": 200,
                "min": 1, "max": 100000,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        actual_mean = np.mean([r.input_tokens for r in requests])
        # Within 50% of target mean
        assert abs(actual_mean - target_mean) / target_mean < 0.5

    def test_fixed_distribution(self):
        fixed_val = 256
        cfg = WorkloadConfig(
            num_requests=100,
            input_length_distribution={
                "dist": "fixed", "mean": fixed_val, "min": 1, "max": 10000,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        for r in requests:
            assert r.input_tokens == fixed_val

    def test_uniform_distribution(self):
        cfg = WorkloadConfig(
            num_requests=500,
            input_length_distribution={
                "dist": "uniform", "mean": 500, "min": 100, "max": 900,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        for r in requests:
            assert 100 <= r.input_tokens <= 900

    def test_normal_distribution(self):
        cfg = WorkloadConfig(
            num_requests=500,
            input_length_distribution={
                "dist": "normal", "mean": 512, "std": 100, "min": 1, "max": 10000,
            },
        )
        gen = WorkloadGenerator(cfg)
        requests = gen.generate()
        actual_mean = np.mean([r.input_tokens for r in requests])
        assert abs(actual_mean - 512) / 512 < 0.2

    def test_unknown_distribution_raises(self):
        cfg = WorkloadConfig(
            num_requests=10,
            input_length_distribution={"dist": "beta", "mean": 100, "min": 1, "max": 1000},
        )
        gen = WorkloadGenerator(cfg)
        with pytest.raises(ValueError, match="Unknown distribution"):
            gen.generate()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    """Test named workload presets."""

    def test_preset_chat(self):
        gen = WorkloadGenerator.preset("chat", num_requests=50)
        requests = gen.generate()
        assert len(requests) == 50
        assert gen.config.arrival_pattern == "poisson"
        assert gen.config.input_length_distribution["mean"] == 256

    def test_preset_rag(self):
        gen = WorkloadGenerator.preset("rag", num_requests=50)
        requests = gen.generate()
        assert len(requests) == 50
        # RAG has longer inputs
        assert gen.config.input_length_distribution["mean"] == 2048
        avg_input = np.mean([r.input_tokens for r in requests])
        avg_output = np.mean([r.max_output_tokens for r in requests])
        # RAG: long inputs, shorter outputs
        assert avg_input > avg_output

    def test_preset_batch(self):
        gen = WorkloadGenerator.preset("batch", num_requests=50)
        assert gen.config.arrival_pattern == "constant"
        requests = gen.generate()
        assert len(requests) == 50

    def test_preset_coding(self):
        gen = WorkloadGenerator.preset("coding", num_requests=50)
        assert gen.config.arrival_pattern == "bursty"
        requests = gen.generate()
        assert len(requests) == 50

    def test_preset_classification(self):
        gen = WorkloadGenerator.preset("classification", num_requests=100)
        requests = gen.generate()
        assert len(requests) == 100
        # Classification: very short outputs (fixed at 5)
        assert gen.config.output_length_distribution["dist"] == "fixed"
        assert gen.config.output_length_distribution["mean"] == 5

    def test_preset_unknown(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            WorkloadGenerator.preset("nonexistent")


# ---------------------------------------------------------------------------
# Arrival rate scaling
# ---------------------------------------------------------------------------

class TestArrivalRateScaling:
    """Higher rate should produce shorter average inter-arrival time."""

    def test_arrival_rate_scales(self):
        low_rate = WorkloadGenerator(
            WorkloadConfig(arrival_rate_rps=5.0, num_requests=200, random_seed=42)
        ).generate()
        high_rate = WorkloadGenerator(
            WorkloadConfig(arrival_rate_rps=50.0, num_requests=200, random_seed=42)
        ).generate()

        def avg_inter_arrival(reqs):
            times = [r.arrival_time_ns for r in reqs]
            gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
            return np.mean(gaps)

        assert avg_inter_arrival(high_rate) < avg_inter_arrival(low_rate)
