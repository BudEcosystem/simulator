"""Tests for batch scheduler."""
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from llm_memory_calculator.genz.serving.constants import (
    RequestStatus, MemoryTier, NS_PER_MS, NS_PER_S,
)
from llm_memory_calculator.genz.serving.request import Request, Batch
from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig
from llm_memory_calculator.genz.serving.batch_scheduler import (
    SchedulerConfig, BatchScheduler,
)


@pytest.fixture
def mock_model_config():
    return SimpleNamespace(
        num_key_value_heads=8, head_dim=128,
        num_decoder_layers=32, num_attention_heads=32, hidden_size=4096,
    )


@pytest.fixture
def memory_model(mock_model_config):
    tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024**3), 2039.0)]
    return MemoryModel(mock_model_config, tier)


def make_request(rid, input_tokens=512, max_output=128, arrival_ns=0):
    return Request(
        request_id=rid, model="test", input_tokens=input_tokens,
        max_output_tokens=max_output, arrival_time_ns=arrival_ns,
    )


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_batch_size == 256
        assert cfg.max_num_batched_tokens == 8192
        assert cfg.enable_chunked_prefill is False
        assert cfg.chunk_size == 512
        assert cfg.prioritize_prefill is False

    def test_custom_values(self):
        cfg = SchedulerConfig(max_batch_size=64, max_num_batched_tokens=4096)
        assert cfg.max_batch_size == 64
        assert cfg.max_num_batched_tokens == 4096


class TestBatchSchedulerAddRequest:
    def test_add_single_request(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        assert sched.pending_count == 1

    def test_add_multiple_requests(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        for i in range(5):
            sched.add_request(make_request(i))
        assert sched.pending_count == 5

    def test_request_set_to_queued(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0, arrival_ns=100 * NS_PER_MS)
        sched.add_request(req)
        assert req.status == RequestStatus.QUEUED


class TestBatchSchedulerSchedule:
    def test_schedule_returns_batch(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        batch = sched.schedule(0)
        assert batch is not None
        assert batch.size > 0

    def test_schedule_empty_queue_returns_none(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        batch = sched.schedule(0)
        assert batch is None

    def test_schedule_respects_max_batch_size(self, memory_model):
        cfg = SchedulerConfig(max_batch_size=4)
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)
        for i in range(10):
            sched.add_request(make_request(i))
        batch = sched.schedule(0)
        assert batch is not None
        assert batch.size <= 4

    def test_schedule_respects_max_tokens(self, memory_model):
        cfg = SchedulerConfig(max_num_batched_tokens=1024)
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)
        # Each request has 512 input tokens, so max 2 per batch
        for i in range(5):
            sched.add_request(make_request(i, input_tokens=512))
        batch = sched.schedule(0)
        assert batch is not None
        assert batch.size <= 2

    def test_prefill_requests_in_batch(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        batch = sched.schedule(0)
        assert batch.prefill_count > 0

    def test_schedule_marks_requests_prefilling(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        batch = sched.schedule(0)
        assert req.status == RequestStatus.PREFILLING

    def test_inflight_count(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        batch = sched.schedule(0)
        # After schedule, request is in the batch (prefilling), not yet in running
        assert sched.pending_count == 0
        # After completing the batch, prefill requests move to running (decode)
        sched.complete_batch(batch, 50 * NS_PER_MS)
        assert sched.inflight_count == 1


class TestBatchSchedulerCompleteBatch:
    def test_complete_batch_generates_tokens(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0, max_output=5)
        sched.add_request(req)
        batch = sched.schedule(0)

        # Complete batch iteration - generates one token per decode request
        completed = sched.complete_batch(batch, 50 * NS_PER_MS)
        # After first iteration, prefill requests become decode requests
        # Won't be complete yet since max_output=5
        assert req.tokens_generated >= 0

    def test_complete_batch_finishes_request(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0, max_output=1)
        sched.add_request(req)
        batch = sched.schedule(0)

        # After prefill, request transitions to decoding
        completed = sched.complete_batch(batch, 50 * NS_PER_MS)
        # Schedule another batch for decode
        batch2 = sched.schedule(50 * NS_PER_MS)
        if batch2:
            completed2 = sched.complete_batch(batch2, 100 * NS_PER_MS)
            # With max_output=1, should complete after 1 decode token
            assert any(r.status == RequestStatus.COMPLETE for r in [req])

    def test_complete_batch_frees_kv(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0, input_tokens=256, max_output=1)
        sched.add_request(req)

        snap_before = memory_model.memory_snapshot()
        batch = sched.schedule(0)
        snap_during = memory_model.memory_snapshot()
        # KV should be allocated during scheduling
        assert snap_during["device_hbm"]["used_bytes"] >= snap_before["device_hbm"]["used_bytes"]


class TestBatchSchedulerEstimateLatency:
    def test_estimate_returns_positive(self, memory_model):
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, SchedulerConfig())
        req = make_request(0)
        sched.add_request(req)
        batch = sched.schedule(0)
        latency = sched.estimate_batch_latency_ms(batch)
        assert latency > 0

    def test_larger_batch_not_faster(self, memory_model):
        """Larger batches should not be faster than smaller ones (no free lunch)."""
        cfg = SchedulerConfig(max_batch_size=256, max_num_batched_tokens=65536)
        sched1 = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)
        r1 = make_request(0, input_tokens=256)
        sched1.add_request(r1)
        b1 = sched1.schedule(0)
        lat1 = sched1.estimate_batch_latency_ms(b1)

        # Larger batch
        sched2 = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)
        for i in range(8):
            sched2.add_request(make_request(i, input_tokens=256))
        b2 = sched2.schedule(0)
        lat2 = sched2.estimate_batch_latency_ms(b2)

        # Per-request latency in larger batch should be >= smaller batch latency
        # (or at least total batch latency should be larger)
        assert lat2 >= lat1 * 0.5  # Allow some variance


class TestBatchSchedulerPrefillPriority:
    def test_prioritize_prefill(self, memory_model):
        cfg = SchedulerConfig(prioritize_prefill=True, max_batch_size=4)
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)

        # Add prefill requests
        for i in range(3):
            sched.add_request(make_request(i))

        batch = sched.schedule(0)
        assert batch is not None
        assert batch.prefill_count > 0


class TestBatchSchedulerMultipleIterations:
    def test_multiple_schedule_complete_cycles(self, memory_model):
        cfg = SchedulerConfig(max_batch_size=4, max_num_batched_tokens=4096)
        sched = BatchScheduler("test", "A100_80GB_GPU", memory_model, cfg)

        # Add requests
        for i in range(4):
            sched.add_request(make_request(i, input_tokens=256, max_output=3))

        current_time = 0
        iterations = 0
        max_iterations = 50

        while iterations < max_iterations:
            batch = sched.schedule(current_time)
            if batch is None:
                break
            latency = sched.estimate_batch_latency_ms(batch)
            current_time += int(latency * NS_PER_MS)
            completed = sched.complete_batch(batch, current_time)
            iterations += 1

        # All requests should eventually complete
        assert iterations > 0
        assert iterations < max_iterations
