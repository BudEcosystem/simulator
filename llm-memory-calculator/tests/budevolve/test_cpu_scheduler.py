"""Tests for CPU-optimized batch scheduling algorithm."""
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import EvalResult


def _make_request(input_tokens=512, output_tokens=128, priority=0,
                  arrival_time=0, numa_node=None):
    """Create a mock request object."""
    req = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        priority=priority,
        arrival_time=arrival_time,
    )
    if numa_node is not None:
        req.numa_node = numa_node
    return req


class TestCPUBatchScheduler:
    """Tests for the standalone CPU batch scheduling algorithm."""

    def test_empty_queue(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        assert schedule_batch_cpu([]) == []

    def test_single_request(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        req = _make_request(input_tokens=256, output_tokens=64)
        result = schedule_batch_cpu([req])
        assert len(result) == 1
        assert result[0] is req

    def test_max_batch_size_respected(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        queue = [_make_request() for _ in range(100)]
        result = schedule_batch_cpu(queue, max_batch_size=8)
        assert len(result) <= 8

    def test_token_budget_constraint(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # Each request = 512 + 128 = 640 tokens
        queue = [_make_request() for _ in range(10)]
        result = schedule_batch_cpu(queue, max_tokens=2000)
        # 2000 / 640 ≈ 3.12, so max 3 requests
        assert len(result) <= 3

    def test_kv_cache_budget_constraint(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # Very small L3 cache → strict KV cache budget
        queue = [_make_request(input_tokens=2048, output_tokens=512) for _ in range(10)]
        result = schedule_batch_cpu(
            queue, max_batch_size=10, l3_cache_mb=1.0,
            model_hidden_dim=4096, bytes_per_param=2,
        )
        # With 1MB L3, 60% for KV = 614KB
        # KV per token = 2 * 4096 * 2 = 16384 bytes
        # Budget: 614400 / 16384 ≈ 37 tokens
        # Each request: 2560 tokens → 0 can fit
        # Actually 37 tokens max, each needs 2560, so 0
        assert len(result) == 0

    def test_ttft_aware_limit(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # target_ttft_ms=100, avg_input=512 → per_request=114ms
        # ttft_limit = 100/114 ≈ 0 → clamped to 1
        queue = [_make_request() for _ in range(10)]
        result = schedule_batch_cpu(
            queue, max_batch_size=64, target_ttft_ms=100.0,
        )
        # Should be limited by TTFT
        assert len(result) <= 1

    def test_ttft_generous_limit(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # target_ttft_ms=5000 → ttft_limit = 5000/114 ≈ 43
        queue = [_make_request() for _ in range(50)]
        result = schedule_batch_cpu(
            queue, max_batch_size=64, target_ttft_ms=5000.0,
        )
        assert len(result) > 1

    def test_priority_ordering(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        low = _make_request(priority=0, arrival_time=0)
        high = _make_request(priority=1, arrival_time=1)
        # High priority should be selected first even if it arrived later
        result = schedule_batch_cpu(
            [low, high], max_batch_size=1, target_ttft_ms=5000.0,
        )
        assert len(result) == 1
        assert result[0] is high

    def test_numa_aware_ordering(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # Create requests with different input lengths
        queue = [
            _make_request(input_tokens=100),
            _make_request(input_tokens=500),
            _make_request(input_tokens=200),
            _make_request(input_tokens=400),
        ]
        result = schedule_batch_cpu(
            queue, max_batch_size=4, numa_nodes=2, target_ttft_ms=5000.0,
        )
        assert len(result) == 4

    def test_numa_node_preference(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            schedule_batch_cpu,
        )
        # Requests with explicit NUMA node preferences
        queue = [
            _make_request(input_tokens=512, numa_node=0),
            _make_request(input_tokens=512, numa_node=1),
            _make_request(input_tokens=512, numa_node=0),
            _make_request(input_tokens=512, numa_node=1),
        ]
        result = schedule_batch_cpu(
            queue, max_batch_size=4, numa_nodes=2, target_ttft_ms=5000.0,
        )
        assert len(result) == 4


class TestCPUEvictionPolicy:
    """Tests for the CPU-optimized cache eviction policy."""

    def _make_entry(self, last_access=0, frequency=1, size=1024, prefix_len=0):
        return SimpleNamespace(
            last_access_time=last_access,
            frequency=frequency,
            size_bytes=size,
            prefix_length=prefix_len,
        )

    def test_evict_none(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            evict_entries_cpu,
        )
        entries = [self._make_entry()]
        assert evict_entries_cpu(entries, 0) == []

    def test_evict_all(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            evict_entries_cpu,
        )
        entries = [self._make_entry() for _ in range(3)]
        result = evict_entries_cpu(entries, 5)
        assert len(result) == 3

    def test_evict_least_valuable(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            evict_entries_cpu,
        )
        # Low frequency, large size = low value → evicted first
        low_value = self._make_entry(last_access=0, frequency=1, size=10000)
        high_value = self._make_entry(last_access=0, frequency=100, size=100)
        result = evict_entries_cpu([low_value, high_value], 1)
        assert len(result) == 1
        assert result[0] is low_value

    def test_prefix_sharing_bonus(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            evict_entries_cpu,
        )
        # Same frequency and size, but one has long prefix
        no_prefix = self._make_entry(frequency=10, size=1024, prefix_len=0)
        long_prefix = self._make_entry(frequency=10, size=1024, prefix_len=1000)
        result = evict_entries_cpu([no_prefix, long_prefix], 1)
        assert result[0] is no_prefix  # No prefix → less valuable → evicted first

    def test_empty_cache(self):
        from llm_memory_calculator.budevolve.evolve.base_algorithms.cpu_batch_scheduler import (
            evict_entries_cpu,
        )
        assert evict_entries_cpu([], 5) == []


class TestCPUSchedulerEvolution:
    """Tests for the AlgorithmEvolver.evolve_cpu_scheduler method."""

    def test_evolve_cpu_scheduler_returns_base_without_api_key(self):
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
                    model="test", hardware="GraniteRapids_CPU",
                    llm_endpoint="http://localhost", llm_model="test",
                )
                result = evolver.evolve_cpu_scheduler(iterations=5)

                assert "best_code" in result
                assert "schedule_batch_cpu" in result["best_code"]
                assert "numa" in result["best_code"].lower()
                assert result["error"] == "no LLM API key"

    def test_get_base_algorithm_cpu_scheduler(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver

        evolver = AlgorithmEvolver(
            model="test", hardware="GraniteRapids_CPU",
            llm_endpoint="http://localhost", llm_model="test",
        )
        code = evolver.get_base_algorithm("cpu_scheduler")
        assert "schedule_batch_cpu" in code
        assert "numa_nodes" in code
        assert "l3_cache_mb" in code

    def test_cpu_prompt_template(self):
        from llm_memory_calculator.budevolve.evolve.prompt_templates import (
            build_cpu_scheduler_prompt,
        )
        prompt = build_cpu_scheduler_prompt(
            current_code="def schedule_batch_cpu(queue): return queue",
            metrics={"throughput_rps": 50.0, "ttft_ms": 200.0},
        )
        assert "CPU" in prompt
        assert "compute-bound" in prompt.lower() or "COMPUTE-BOUND" in prompt
        assert "NUMA" in prompt
        assert "L3" in prompt
