"""Tests for Request and Batch dataclasses in the serving simulation."""
import pytest

from llm_memory_calculator.genz.serving.constants import RequestStatus
from llm_memory_calculator.genz.serving.request import Batch, Request


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> Request:
    defaults = dict(
        request_id=1,
        model="llama-70b",
        input_tokens=128,
        max_output_tokens=256,
        arrival_time_ns=1_000_000_000,  # 1 s
    )
    defaults.update(overrides)
    return Request(**defaults)


# ---------------------------------------------------------------------------
# Request lifecycle tests
# ---------------------------------------------------------------------------

class TestRequestLifecycle:
    """Test the full lifecycle: ARRIVED -> QUEUED -> PREFILLING -> DECODING -> COMPLETE."""

    def test_initial_status_is_arrived(self):
        req = _make_request()
        assert req.status == RequestStatus.ARRIVED

    def test_set_queued(self):
        req = _make_request(arrival_time_ns=1_000_000_000)
        req.set_queued(time_ns=1_050_000_000)
        assert req.status == RequestStatus.QUEUED
        assert req.queuing_delay_ns == 50_000_000  # 50 ms

    def test_set_prefilling(self):
        req = _make_request(arrival_time_ns=1_000_000_000)
        req.set_prefilling(time_ns=1_100_000_000)
        assert req.status == RequestStatus.PREFILLING
        assert req.prefill_start_ns == 1_100_000_000
        assert req.queuing_delay_ns == 100_000_000  # 100 ms

    def test_set_decoding(self):
        req = _make_request(arrival_time_ns=1_000_000_000)
        req.set_decoding(time_ns=1_200_000_000)
        assert req.status == RequestStatus.DECODING
        assert req.decode_start_ns == 1_200_000_000

    def test_full_lifecycle(self):
        arrival = 1_000_000_000
        req = _make_request(arrival_time_ns=arrival)

        req.set_queued(time_ns=arrival + 10_000_000)       # +10 ms
        assert req.status == RequestStatus.QUEUED

        req.set_prefilling(time_ns=arrival + 50_000_000)    # +50 ms
        assert req.status == RequestStatus.PREFILLING

        req.set_decoding(time_ns=arrival + 200_000_000)     # +200 ms
        assert req.status == RequestStatus.DECODING

        # Generate 3 tokens at 100 ms intervals
        for i in range(3):
            token_time = arrival + 300_000_000 + i * 100_000_000
            req.record_token(time_ns=token_time)

        req.set_complete(time_ns=arrival + 500_000_000)     # +500 ms
        assert req.status == RequestStatus.COMPLETE
        assert req.tokens_generated == 3
        assert req.e2e_ns == 500_000_000


# ---------------------------------------------------------------------------
# TTFT tests
# ---------------------------------------------------------------------------

class TestTTFT:
    """Test Time-To-First-Token calculation."""

    def test_ttft_is_time_from_arrival_to_decode_start(self):
        arrival = 1_000_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_decoding(time_ns=arrival + 250_000_000)
        assert req.ttft_ns == 250_000_000

    def test_ttft_zero_when_decode_starts_immediately(self):
        arrival = 1_000_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_decoding(time_ns=arrival)
        assert req.ttft_ns == 0


# ---------------------------------------------------------------------------
# TPOT tests
# ---------------------------------------------------------------------------

class TestTPOT:
    """Test Time-Per-Output-Token (average) calculation."""

    def test_tpot_average_across_tokens(self):
        arrival = 0
        req = _make_request(arrival_time_ns=arrival)
        req.set_decoding(time_ns=100_000_000)  # decode starts at 100 ms

        # 4 tokens, decode phase lasts 400 ms total
        for i in range(4):
            req.record_token(time_ns=200_000_000 + i * 100_000_000)

        req.set_complete(time_ns=500_000_000)  # complete at 500 ms
        # decode_duration = 500ms - 100ms = 400ms, tokens = 4
        assert req.tpot_ns == 100_000_000  # 100 ms per token

    def test_tpot_single_token(self):
        arrival = 0
        req = _make_request(arrival_time_ns=arrival)
        req.set_decoding(time_ns=100_000_000)
        req.record_token(time_ns=200_000_000)
        req.set_complete(time_ns=200_000_000)
        # decode_duration = 200ms - 100ms = 100ms, tokens = 1
        assert req.tpot_ns == 100_000_000

    def test_tpot_zero_tokens(self):
        arrival = 0
        req = _make_request(arrival_time_ns=arrival)
        req.set_decoding(time_ns=100_000_000)
        req.set_complete(time_ns=200_000_000)
        # No tokens generated, tpot should remain 0
        assert req.tpot_ns == 0


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestE2E:
    """Test End-to-End latency calculation."""

    def test_e2e_from_arrival_to_completion(self):
        arrival = 500_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_complete(time_ns=arrival + 2_000_000_000)
        assert req.e2e_ns == 2_000_000_000

    def test_e2e_on_failure(self):
        arrival = 500_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_failed(time_ns=arrival + 1_000_000_000)
        assert req.e2e_ns == 1_000_000_000


# ---------------------------------------------------------------------------
# Queuing delay tests
# ---------------------------------------------------------------------------

class TestQueuingDelay:
    """Test queuing delay tracking."""

    def test_queuing_delay_set_by_set_queued(self):
        arrival = 1_000_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_queued(time_ns=arrival + 75_000_000)
        assert req.queuing_delay_ns == 75_000_000

    def test_queuing_delay_updated_by_set_prefilling(self):
        arrival = 1_000_000_000
        req = _make_request(arrival_time_ns=arrival)
        req.set_queued(time_ns=arrival + 10_000_000)
        req.set_prefilling(time_ns=arrival + 30_000_000)
        # set_prefilling overwrites queuing_delay
        assert req.queuing_delay_ns == 30_000_000


# ---------------------------------------------------------------------------
# ITL (Inter-Token Latency) tests
# ---------------------------------------------------------------------------

class TestITL:
    """Test inter-token latency tracking."""

    def test_itl_first_token_relative_to_decode_start(self):
        req = _make_request(arrival_time_ns=0)
        req.set_decoding(time_ns=100_000_000)
        req.record_token(time_ns=150_000_000)
        assert req.itl_ns == [50_000_000]

    def test_itl_subsequent_tokens(self):
        req = _make_request(arrival_time_ns=0)
        req.set_decoding(time_ns=100_000_000)
        req.record_token(time_ns=150_000_000)   # ITL: 50ms
        req.record_token(time_ns=230_000_000)   # ITL: 80ms
        req.record_token(time_ns=350_000_000)   # ITL: 120ms
        assert req.itl_ns == [50_000_000, 80_000_000, 120_000_000]

    def test_itl_empty_when_no_tokens(self):
        req = _make_request(arrival_time_ns=0)
        req.set_decoding(time_ns=100_000_000)
        assert req.itl_ns == []


# ---------------------------------------------------------------------------
# set_failed tests
# ---------------------------------------------------------------------------

class TestSetFailed:
    """Test failure handling."""

    def test_set_failed_status(self):
        req = _make_request(arrival_time_ns=0)
        req.set_failed(time_ns=500_000_000)
        assert req.status == RequestStatus.FAILED

    def test_set_failed_records_completion_time(self):
        req = _make_request(arrival_time_ns=0)
        req.set_failed(time_ns=500_000_000)
        assert req.completion_time_ns == 500_000_000

    def test_set_failed_calculates_e2e(self):
        req = _make_request(arrival_time_ns=100_000_000)
        req.set_failed(time_ns=600_000_000)
        assert req.e2e_ns == 500_000_000


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

class TestBatch:
    """Test Batch dataclass properties."""

    def test_batch_empty(self):
        batch = Batch(batch_id=0, model="llama-70b")
        assert batch.size == 0
        assert batch.prefill_count == 0
        assert batch.decode_count == 0

    def test_batch_size(self):
        reqs = [_make_request(request_id=i) for i in range(5)]
        batch = Batch(batch_id=1, model="llama-70b", requests=reqs)
        assert batch.size == 5

    def test_batch_prefill_count(self):
        prefill_reqs = [_make_request(request_id=i) for i in range(3)]
        batch = Batch(batch_id=1, model="llama-70b", prefill_requests=prefill_reqs)
        assert batch.prefill_count == 3

    def test_batch_decode_count(self):
        decode_reqs = [_make_request(request_id=i) for i in range(7)]
        batch = Batch(batch_id=1, model="llama-70b", decode_requests=decode_reqs)
        assert batch.decode_count == 7

    def test_batch_mixed(self):
        all_reqs = [_make_request(request_id=i) for i in range(10)]
        prefill = all_reqs[:3]
        decode = all_reqs[3:]
        batch = Batch(
            batch_id=2,
            model="llama-70b",
            requests=all_reqs,
            prefill_requests=prefill,
            decode_requests=decode,
        )
        assert batch.size == 10
        assert batch.prefill_count == 3
        assert batch.decode_count == 7


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for Request."""

    def test_default_field_values(self):
        req = _make_request()
        assert req.instance_id == 0
        assert req.tokens_generated == 0
        assert req.ttft_ns == 0
        assert req.tpot_ns == 0
        assert req.e2e_ns == 0
        assert req.queuing_delay_ns == 0
        assert req.itl_ns == []
        assert req.prefill_start_ns == 0
        assert req.decode_start_ns == 0
        assert req.completion_time_ns == 0

    def test_complete_with_zero_tokens_does_not_divide_by_zero(self):
        req = _make_request(arrival_time_ns=0)
        req.set_decoding(time_ns=100_000_000)
        req.set_complete(time_ns=200_000_000)
        # Should not raise; tpot stays 0
        assert req.tpot_ns == 0
        assert req.e2e_ns == 200_000_000

    def test_itl_list_independence_across_requests(self):
        """Ensure field(default_factory=list) creates separate lists."""
        r1 = _make_request(request_id=1)
        r2 = _make_request(request_id=2)
        r1.set_decoding(time_ns=100_000_000)
        r1.record_token(time_ns=200_000_000)
        assert r1.itl_ns == [100_000_000]
        assert r2.itl_ns == []
