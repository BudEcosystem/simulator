"""Tests for SLOTargets and SLOTracker in the serving simulation."""
import pytest

from llm_memory_calculator.genz.serving.constants import NS_PER_MS, RequestStatus
from llm_memory_calculator.genz.serving.request import Request
from llm_memory_calculator.genz.serving.slo_tracker import SLOTargets, SLOTracker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _completed_request(
    request_id: int = 1,
    arrival_ns: int = 0,
    ttft_ns: int = 200_000_000,
    tpot_ns: int = 50_000_000,
    e2e_ns: int = 5_000_000_000,
    tokens_generated: int = 10,
    itl_ns: list | None = None,
) -> Request:
    """Build a Request already in COMPLETE state with the given metrics."""
    req = Request(
        request_id=request_id,
        model="llama-70b",
        input_tokens=128,
        max_output_tokens=256,
        arrival_time_ns=arrival_ns,
        status=RequestStatus.COMPLETE,
        ttft_ns=ttft_ns,
        tpot_ns=tpot_ns,
        e2e_ns=e2e_ns,
        tokens_generated=tokens_generated,
        itl_ns=itl_ns if itl_ns is not None else [],
    )
    return req


# ---------------------------------------------------------------------------
# SLOTargets tests
# ---------------------------------------------------------------------------

class TestSLOTargets:
    """Test SLOTargets defaults and from_ms classmethod."""

    def test_defaults(self):
        t = SLOTargets()
        assert t.ttft_target_ns == 500_000_000     # 500 ms
        assert t.tpot_target_ns == 100_000_000     # 100 ms
        assert t.e2e_target_ns == 30_000_000_000   # 30 s

    def test_from_ms(self):
        t = SLOTargets.from_ms(ttft_ms=250.0, tpot_ms=75.0, e2e_ms=10_000.0)
        assert t.ttft_target_ns == 250_000_000
        assert t.tpot_target_ns == 75_000_000
        assert t.e2e_target_ns == 10_000_000_000

    def test_from_ms_fractional(self):
        t = SLOTargets.from_ms(ttft_ms=0.5, tpot_ms=0.1, e2e_ms=1.0)
        assert t.ttft_target_ns == 500_000
        assert t.tpot_target_ns == 100_000
        assert t.e2e_target_ns == 1_000_000


# ---------------------------------------------------------------------------
# Empty tracker tests
# ---------------------------------------------------------------------------

class TestEmptyTracker:
    """Test that an empty tracker returns sensible defaults."""

    def test_empty_violation_rates(self):
        tracker = SLOTracker()
        assert tracker.ttft_violation_rate() == 0.0
        assert tracker.tpot_violation_rate() == 0.0
        assert tracker.e2e_violation_rate() == 0.0

    def test_empty_percentiles(self):
        tracker = SLOTracker()
        for pct in [tracker.ttft_percentiles(), tracker.tpot_percentiles(),
                     tracker.e2e_percentiles(), tracker.itl_percentiles()]:
            assert pct == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_empty_goodput(self):
        tracker = SLOTracker()
        assert tracker.goodput() == 0.0

    def test_empty_summary_total_completed(self):
        tracker = SLOTracker()
        assert tracker.summary()["total_completed"] == 0


# ---------------------------------------------------------------------------
# Recording completed requests
# ---------------------------------------------------------------------------

class TestRecordCompletedRequest:
    """Test that completed requests are properly recorded."""

    def test_total_completed_increments(self):
        tracker = SLOTracker()
        for i in range(5):
            tracker.record_completed_request(_completed_request(request_id=i))
        assert tracker.summary()["total_completed"] == 5

    def test_non_complete_requests_ignored(self):
        tracker = SLOTracker()
        # Create a FAILED request
        req = Request(
            request_id=1,
            model="llama-70b",
            input_tokens=128,
            max_output_tokens=256,
            arrival_time_ns=0,
            status=RequestStatus.FAILED,
            ttft_ns=100_000_000,
            tpot_ns=50_000_000,
            e2e_ns=1_000_000_000,
        )
        tracker.record_completed_request(req)
        assert tracker.summary()["total_completed"] == 0

    def test_queued_request_ignored(self):
        tracker = SLOTracker()
        req = Request(
            request_id=1,
            model="llama-70b",
            input_tokens=128,
            max_output_tokens=256,
            arrival_time_ns=0,
            status=RequestStatus.QUEUED,
        )
        tracker.record_completed_request(req)
        assert tracker.summary()["total_completed"] == 0


# ---------------------------------------------------------------------------
# Violation rate tests
# ---------------------------------------------------------------------------

class TestViolationRates:
    """Test violation rate calculations with mixed compliant/violating requests."""

    def test_no_violations(self):
        targets = SLOTargets.from_ms(ttft_ms=500, tpot_ms=100, e2e_ms=30_000)
        tracker = SLOTracker(targets=targets)
        # All within limits
        for i in range(4):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=200_000_000,   # 200 ms < 500 ms
                tpot_ns=50_000_000,    # 50 ms  < 100 ms
                e2e_ns=5_000_000_000,  # 5 s    < 30 s
            ))
        assert tracker.ttft_violation_rate() == 0.0
        assert tracker.tpot_violation_rate() == 0.0
        assert tracker.e2e_violation_rate() == 0.0

    def test_all_violations(self):
        targets = SLOTargets.from_ms(ttft_ms=100, tpot_ms=20, e2e_ms=1_000)
        tracker = SLOTracker(targets=targets)
        for i in range(4):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=200_000_000,    # 200 ms > 100 ms
                tpot_ns=50_000_000,     # 50 ms  > 20 ms
                e2e_ns=5_000_000_000,   # 5 s    > 1 s
            ))
        assert tracker.ttft_violation_rate() == 1.0
        assert tracker.tpot_violation_rate() == 1.0
        assert tracker.e2e_violation_rate() == 1.0

    def test_partial_ttft_violations(self):
        targets = SLOTargets.from_ms(ttft_ms=300, tpot_ms=1000, e2e_ms=60_000)
        tracker = SLOTracker(targets=targets)
        # 2 within, 2 violating TTFT
        for i, ttft in enumerate([100_000_000, 200_000_000, 400_000_000, 500_000_000]):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=ttft,
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        assert tracker.ttft_violation_rate() == pytest.approx(0.5)

    def test_exact_target_is_not_violation(self):
        targets = SLOTargets.from_ms(ttft_ms=200, tpot_ms=50, e2e_ms=5_000)
        tracker = SLOTracker(targets=targets)
        tracker.record_completed_request(_completed_request(
            ttft_ns=200_000_000,
            tpot_ns=50_000_000,
            e2e_ns=5_000_000_000,
        ))
        assert tracker.ttft_violation_rate() == 0.0
        assert tracker.tpot_violation_rate() == 0.0
        assert tracker.e2e_violation_rate() == 0.0


# ---------------------------------------------------------------------------
# Percentile tests
# ---------------------------------------------------------------------------

class TestPercentiles:
    """Test percentile calculations with known data."""

    def test_single_value(self):
        tracker = SLOTracker()
        tracker.record_completed_request(_completed_request(
            ttft_ns=100_000_000,
            tpot_ns=50_000_000,
            e2e_ns=1_000_000_000,
        ))
        pct = tracker.ttft_percentiles()
        assert pct["p50"] == 100_000_000.0
        assert pct["p95"] == 100_000_000.0
        assert pct["p99"] == 100_000_000.0

    def test_known_distribution(self):
        tracker = SLOTracker()
        # 100 requests with TTFT = 1ms, 2ms, ..., 100ms
        for i in range(1, 101):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=i * NS_PER_MS,
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        pct = tracker.ttft_percentiles()
        assert pct["p50"] == 50.0 * NS_PER_MS
        assert pct["p95"] == 95.0 * NS_PER_MS
        assert pct["p99"] == 99.0 * NS_PER_MS

    def test_itl_percentiles(self):
        tracker = SLOTracker()
        # A request with known ITL values
        itl = [10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
        tracker.record_completed_request(_completed_request(
            itl_ns=itl,
            tokens_generated=5,
        ))
        pct = tracker.itl_percentiles()
        # 5 values sorted: [10, 20, 30, 40, 50] (in millions)
        assert pct["p50"] == 30_000_000.0
        assert pct["p99"] == 50_000_000.0


# ---------------------------------------------------------------------------
# Goodput tests
# ---------------------------------------------------------------------------

class TestGoodput:
    """Test goodput (fraction of requests meeting ALL SLOs)."""

    def test_all_meet_slo(self):
        targets = SLOTargets.from_ms(ttft_ms=500, tpot_ms=100, e2e_ms=30_000)
        tracker = SLOTracker(targets=targets)
        for i in range(10):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=200_000_000,
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        assert tracker.goodput() == 1.0

    def test_none_meet_slo(self):
        targets = SLOTargets.from_ms(ttft_ms=10, tpot_ms=5, e2e_ms=100)
        tracker = SLOTracker(targets=targets)
        for i in range(10):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=200_000_000,
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        assert tracker.goodput() == 0.0

    def test_partial_goodput(self):
        targets = SLOTargets.from_ms(ttft_ms=250, tpot_ms=100, e2e_ms=30_000)
        tracker = SLOTracker(targets=targets)
        # 3 good requests
        for i in range(3):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=200_000_000,
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        # 2 bad requests (TTFT violates)
        for i in range(3, 5):
            tracker.record_completed_request(_completed_request(
                request_id=i,
                ttft_ns=300_000_000,  # > 250 ms target
                tpot_ns=50_000_000,
                e2e_ns=5_000_000_000,
            ))
        assert tracker.goodput() == pytest.approx(0.6)

    def test_goodput_requires_all_slos(self):
        """One SLO violated means the request does not count toward goodput."""
        targets = SLOTargets.from_ms(ttft_ms=500, tpot_ms=100, e2e_ms=30_000)
        tracker = SLOTracker(targets=targets)
        # TTFT and E2E ok, but TPOT violates
        tracker.record_completed_request(_completed_request(
            ttft_ns=200_000_000,
            tpot_ns=150_000_000,  # > 100 ms
            e2e_ns=5_000_000_000,
        ))
        assert tracker.goodput() == 0.0


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    """Test the summary dict structure."""

    def test_summary_keys(self):
        tracker = SLOTracker()
        tracker.record_completed_request(_completed_request())
        s = tracker.summary()
        assert "total_completed" in s
        assert "goodput" in s
        assert "ttft" in s
        assert "tpot" in s
        assert "e2e" in s
        assert "itl" in s

    def test_summary_nested_keys(self):
        tracker = SLOTracker()
        tracker.record_completed_request(_completed_request())
        s = tracker.summary()
        for metric in ["ttft", "tpot", "e2e"]:
            assert "violation_rate" in s[metric]
            assert "percentiles_ns" in s[metric]
        assert "percentiles_ns" in s["itl"]

    def test_summary_values_consistent(self):
        targets = SLOTargets.from_ms(ttft_ms=500, tpot_ms=100, e2e_ms=30_000)
        tracker = SLOTracker(targets=targets)
        tracker.record_completed_request(_completed_request(
            ttft_ns=200_000_000,
            tpot_ns=50_000_000,
            e2e_ns=5_000_000_000,
        ))
        s = tracker.summary()
        assert s["total_completed"] == 1
        assert s["goodput"] == 1.0
        assert s["ttft"]["violation_rate"] == 0.0
        assert s["tpot"]["violation_rate"] == 0.0
        assert s["e2e"]["violation_rate"] == 0.0
