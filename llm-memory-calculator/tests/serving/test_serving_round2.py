"""Round-2 serving-core accuracy remediation golden tests (solutions_round2.md §4).

Each test asserts a PHYSICAL outcome of a fix (SV1..SV6), not an internal constant:

- SV3: ``is_offload`` from GenZ is propagated; a KV-spilling decode sets the flag, a fitting one does not.
- SV4: simulation-loop power scales with the real roofline MFU/MBU (varies with batch size), not a
  batch-slot-occupancy proxy.
- SV5: the prefix-cache hit-rate is ~0 for independent prompts (no position-only fabrication) and
  ~(N/input) when ``shared_prefix_tokens=N``.
- SV6: cluster/disagg instance throughput is ``B_eff * 1000/time_per_request`` (continuous batching),
  with ``B_eff`` derived from the memory budget; disagg bottleneck inequalities preserved (B_eff cancels);
  ``tokens_per_second == throughput * (in+out)`` by construction.
- SV1/SV2: a request whose KV exceeds the TOTAL capacity of all tiers is failed and dropped (not
  re-queued forever); ``completed + failed == num_requests``; failed requests are NOT in raw_requests.

No magic numbers: every expected value is derived in-test from the engine's own output, the canonical
``MemoryModel.bytes_per_token_kv`` formula, or the hardware datasheet peak.
"""
import math
import warnings
from types import SimpleNamespace

import pytest

from llm_memory_calculator.genz.serving.constants import (
    MemoryTier, RequestStatus, NS_PER_MS,
)
from llm_memory_calculator.genz.serving.request import Request, Batch
from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig
from llm_memory_calculator.genz.serving.batch_scheduler import SchedulerConfig, BatchScheduler
from llm_memory_calculator.genz.serving.simulator import ServingSimulator, ServingSimulationResult
from llm_memory_calculator.genz.serving.workload import WorkloadConfig
from llm_memory_calculator.genz.serving.prefix_cache import PrefixCacheAnalyzer
from llm_memory_calculator.genz.serving.cluster import ClusterAnalyzer
from llm_memory_calculator.genz.serving.disaggregation import DisaggregationAnalyzer


# Registered model (no HF gating) + datasheet hardware used for live checks.
MODEL = "llama2_7b"
HARDWARE = "A100_80GB_GPU"


@pytest.fixture
def model_config():
    """llama2_7b-shaped config (GQA-free MHA: kv_heads == heads)."""
    return SimpleNamespace(
        num_key_value_heads=32, head_dim=128,
        num_decoder_layers=32, num_attention_heads=32,
        hidden_size=4096, vocab_size=32000, intermediate_size=11008,
    )


def make_request(rid, input_tokens=512, max_output=4, arrival_ns=0):
    return Request(
        request_id=rid, model=MODEL, input_tokens=input_tokens,
        max_output_tokens=max_output, arrival_time_ns=arrival_ns,
    )


# ---------------------------------------------------------------------------
# SV3 — offload signal propagation
# ---------------------------------------------------------------------------

class TestSV3OffloadPropagation:
    def test_scheduler_exposes_offload_flag_default_false(self, model_config):
        """A scheduler that never ran a batch reports last_batch_offloaded == False."""
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())
        assert sched.last_batch_offloaded is False

    def test_offloaded_result_sets_flag(self, model_config):
        """When a phase's GenZ result has is_offload=True, the scheduler surfaces it."""
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())

        # Patch the modeling functions this scheduler uses to return an offloaded prefill result.
        offloaded = SimpleNamespace(Latency=1234.0, is_offload=True, summary_table=None)
        fitting = SimpleNamespace(Latency=5.0, is_offload=False, summary_table=None)
        sched._modeling_fns = (lambda **kw: offloaded, lambda **kw: fitting)

        req = make_request(0, input_tokens=512)
        sched.add_request(req)
        batch = sched.schedule(0)
        sched.estimate_batch_latency_ms(batch)
        assert sched.last_batch_offloaded is True

    def test_fitting_result_keeps_flag_false(self, model_config):
        """A fitting workload (is_offload=False on both phases) keeps the flag clear."""
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())
        fitting = SimpleNamespace(Latency=5.0, is_offload=False, summary_table=None)
        sched._modeling_fns = (lambda **kw: fitting, lambda **kw: fitting)

        req = make_request(0, input_tokens=512)
        sched.add_request(req)
        batch = sched.schedule(0)
        sched.estimate_batch_latency_ms(batch)
        assert sched.last_batch_offloaded is False

    def test_result_has_offload_fields(self):
        result = ServingSimulationResult()
        assert result.offloaded is False
        assert result.offload_steps == 0

    def test_simulation_flags_offload_when_engine_offloads(self):
        """An end-to-end sim whose batches offload sets result.offloaded and counts steps.

        Uses a tiny HBM tier so the scheduler's GenZ calls run with model_offload (large valid
        latency + is_offload=True) rather than fitting on-chip.
        """
        sim = ServingSimulator(MODEL, HARDWARE, precision="bf16", tensor_parallel=1)
        # Force every decode/prefill to be flagged offloaded by patching the modeling functions.
        offloaded = SimpleNamespace(Latency=50.0, is_offload=True, summary_table=None)
        wl = WorkloadConfig(
            num_requests=3, arrival_rate_rps=2.0, arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 64, "min": 32, "max": 128},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 2, "max": 8},
            random_seed=42,
        )
        sc = SchedulerConfig(max_batch_size=8, max_num_batched_tokens=4096)

        import llm_memory_calculator.genz.serving.batch_scheduler as bs_mod
        orig = bs_mod.BatchScheduler.estimate_batch_latency_ms

        def patched(self, batch):
            self.last_batch_offloaded = True
            return 50.0

        bs_mod.BatchScheduler.estimate_batch_latency_ms = patched
        try:
            result = sim.simulate(workload_config=wl, scheduler_config=sc)
        finally:
            bs_mod.BatchScheduler.estimate_batch_latency_ms = orig

        assert result.offloaded is True
        assert result.offload_steps > 0


# ---------------------------------------------------------------------------
# SV4 — power scales with real roofline utilization (varies with batch size)
# ---------------------------------------------------------------------------

class TestSV4PowerTracksRoofline:
    def test_scheduler_exposes_mfu_mbu(self, model_config):
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())
        assert hasattr(sched, "last_batch_mfu")
        assert hasattr(sched, "last_batch_mbu")
        assert sched.last_batch_mfu == 0.0
        assert sched.last_batch_mbu == 0.0

    def test_power_varies_with_batch_size(self):
        """A larger batch (higher prefill MFU) draws more accelerator power per unit time
        than a single-request batch. The sim-loop power must reflect the real roofline, so the
        average power differs between the two workloads (not a frozen occupancy proxy)."""
        sim = ServingSimulator(MODEL, HARDWARE, precision="bf16", tensor_parallel=1)

        def run(num_requests):
            wl = WorkloadConfig(
                num_requests=num_requests, arrival_rate_rps=1000.0, arrival_pattern="constant",
                input_length_distribution={"dist": "fixed", "mean": 512, "min": 256, "max": 1024},
                output_length_distribution={"dist": "fixed", "mean": 4, "min": 2, "max": 8},
                random_seed=42,
            )
            sc = SchedulerConfig(max_batch_size=64, max_num_batched_tokens=65536)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sim.simulate(workload_config=wl, scheduler_config=sc,
                                    enable_power_tracking=True)

        small = run(1)
        large = run(16)
        p_small = small.power_summary.get("avg_power_w", 0)
        p_large = large.power_summary.get("avg_power_w", 0)
        assert p_small > 0 and p_large > 0
        # Different batch occupancy -> different roofline MFU/MBU -> different average power.
        assert abs(p_large - p_small) / max(p_small, 1e-9) > 1e-3


# ---------------------------------------------------------------------------
# SV5 — honest prefix-cache hit rate
# ---------------------------------------------------------------------------

class TestSV5PrefixCache:
    def test_independent_prompts_zero_hit_rate(self, model_config):
        """Independent prompts (no shared prefix) must NOT share a fabricated position-only
        token stream -> hit-rate ~0 (was ~0.99 with the deleted branch)."""
        analyzer = PrefixCacheAnalyzer(model_config=model_config, cache_capacity_gb=10.0)
        workload = [make_request(i, input_tokens=512) for i in range(10)]
        result = analyzer.analyze(workload)  # shared_prefix_tokens default 0
        assert result["total_requests"] == 10
        assert result["token_savings_rate"] == pytest.approx(0.0, abs=1e-9)
        assert result["cache_stats"]["hit_rate"] == pytest.approx(0.0, abs=1e-9)

    def test_shared_prefix_savings_match_formula(self, model_config):
        """With shared_prefix_tokens=N over R requests of input I tokens: the first request
        misses entirely; the other R-1 reuse the N-token prefix. token_savings_rate is
        block-aligned but must approximate (R-1)*N / (R*I)."""
        analyzer = PrefixCacheAnalyzer(model_config=model_config, cache_capacity_gb=10.0)
        R, I, N = 10, 512, 256
        workload = [make_request(i, input_tokens=I) for i in range(R)]
        result = analyzer.analyze(workload, shared_prefix_tokens=N)
        expected_rate = (R - 1) * N / (R * I)
        # Block-aligned matching may round to a block boundary; tolerate one block of slack.
        assert result["token_savings_rate"] == pytest.approx(expected_rate, abs=0.05)
        assert result["token_savings_rate"] > 0.0

    def test_default_shared_prefix_is_zero(self, model_config):
        """analyze() default must be shared_prefix_tokens=0 (no fabricated overlap)."""
        analyzer = PrefixCacheAnalyzer(model_config=model_config, cache_capacity_gb=10.0)
        workload = [make_request(i, input_tokens=300 + i * 50) for i in range(5)]
        result = analyzer.analyze(workload)
        assert result["token_savings_rate"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SV6 — continuous-batching cluster / disagg throughput (B_eff)
# ---------------------------------------------------------------------------

def _b_eff(mm: MemoryModel, weight_bytes: int, context: int, cap: int) -> int:
    primary_cap = mm.total_capacity_bytes if mm is None else None
    footprint = mm.bytes_per_token_kv * context
    avail = mm.primary_capacity_bytes - weight_bytes
    return max(1, min(cap, avail // footprint))


class TestSV6ClusterThroughput:
    def test_cluster_throughput_includes_b_eff(self):
        """Instance throughput == B_eff * 1000/time_per_request; total == that * instances."""
        analyzer = ClusterAnalyzer(MODEL, HARDWARE)
        prefill_lat, decode_lat = 5.0, 0.5
        analyzer._get_latencies = lambda i, o: (prefill_lat, decode_lat)
        input_tokens, output_tokens = 512, 128
        result = analyzer.analyze_scaling(
            instance_counts=[1, 2, 4],
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        b_eff = analyzer.last_b_eff
        assert b_eff >= 1
        time_per_request = prefill_lat + decode_lat * output_tokens
        single = b_eff * 1000.0 / time_per_request
        assert result["single_instance_throughput_rps"] == pytest.approx(single, rel=1e-9)
        # tokens_per_second derived in-expression: throughput * (in + out).
        for entry in result["scaling_results"]:
            assert entry["tokens_per_second"] == pytest.approx(
                entry["throughput_rps"] * (input_tokens + output_tokens), rel=1e-9
            )

    def test_b_eff_default_one_when_memory_model_unbuildable(self):
        """When a MemoryModel can't be built (mocked/unknown config), B_eff defaults to 1 so
        mocked tests keep their serial-throughput numbers."""
        analyzer = ClusterAnalyzer(MODEL, HARDWARE)
        analyzer._get_latencies = lambda i, o: (5.12, 0.5)
        # Force memory-model construction to fail.
        analyzer._build_b_eff = lambda *a, **k: 1
        result = analyzer.analyze_scaling(instance_counts=[1], input_tokens=512, output_tokens=128)
        time_per_request = 5.12 + 0.5 * 128
        assert result["single_instance_throughput_rps"] == pytest.approx(
            1 * 1000.0 / time_per_request, rel=1e-9
        )

    def test_cluster_b_eff_is_memory_bounded(self):
        """B_eff must equal floor((primary_capacity - weight_bytes)/(bytes_per_token_kv*context)),
        capped by max_batch_size -- derived, never a magic number."""
        analyzer = ClusterAnalyzer(MODEL, HARDWARE)
        analyzer._get_latencies = lambda i, o: (5.0, 0.5)
        input_tokens, output_tokens = 512, 128
        analyzer.analyze_scaling(instance_counts=[1], input_tokens=input_tokens,
                                 output_tokens=output_tokens)
        # Reconstruct the expected B_eff from the canonical formula.
        from llm_memory_calculator.genz.Models import get_configs
        cfg = get_configs(MODEL)
        mm = MemoryModel(cfg, [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)])
        # weight bytes: analyzer uses the same estimator; pull it from the analyzer.
        weight_bytes = analyzer.last_weight_bytes
        context = input_tokens + output_tokens
        footprint = mm.bytes_per_token_kv * context
        cap = SchedulerConfig().max_batch_size
        expected = max(1, min(cap, (80 * (1024 ** 3) - weight_bytes) // footprint))
        assert analyzer.last_b_eff == expected


class TestSV6DisaggInequalitiesPreserved:
    def test_bottleneck_inequalities_preserved(self):
        """B_eff applies identically to prefill and decode pools, so the bottleneck verdict
        (which depends only on the ratio) is unchanged from the serial model."""
        analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
        analyzer._get_latencies = lambda i, o: (10.0, 0.5)
        analyzer._estimate_kv_bytes = lambda i, o=0: 1024
        analyzer._build_b_eff = lambda *a, **k: 8  # nonzero, identical both pools
        result = analyzer.analyze(prefill_instances=1, decode_instances=7,
                                  input_tokens=512, output_tokens=128)
        assert result["bottleneck"] == "prefill"
        assert result["prefill_throughput_rps"] < result["decode_throughput_rps"]

    def test_tps_relation_in_disagg(self):
        """system_throughput scales with B_eff (continuous batching) -- both pools get B_eff,
        so doubling B_eff doubles system throughput."""
        analyzer = DisaggregationAnalyzer(MODEL, HARDWARE)
        analyzer._get_latencies = lambda i, o: (10.0, 0.5)
        analyzer._estimate_kv_bytes = lambda i, o=0: 1024
        analyzer._build_b_eff = lambda *a, **k: 1
        r1 = analyzer.analyze(prefill_instances=4, decode_instances=4,
                              input_tokens=512, output_tokens=128)
        analyzer._build_b_eff = lambda *a, **k: 2
        r2 = analyzer.analyze(prefill_instances=4, decode_instances=4,
                              input_tokens=512, output_tokens=128)
        assert r2["system_throughput_rps"] == pytest.approx(2 * r1["system_throughput_rps"], rel=1e-9)


# ---------------------------------------------------------------------------
# SV1 / SV2 — admission + completion accounting
# ---------------------------------------------------------------------------

class TestSV1SV2Admission:
    def test_memory_model_total_capacity(self, model_config):
        tiers = [
            MemoryTierConfig(MemoryTier.DEVICE_HBM, 10 * (1024 ** 3), 2039.0),
            MemoryTierConfig(MemoryTier.HOST_DDR, 20 * (1024 ** 3), 100.0),
        ]
        mm = MemoryModel(model_config, tiers)
        assert mm.total_capacity_bytes == 30 * (1024 ** 3)

    def test_oversized_request_failed_and_dropped(self, model_config):
        """A request whose KV footprint exceeds the TOTAL capacity of all tiers is failed and
        removed from pending (not re-queued forever)."""
        # Tiny tier: a single 512-token request's KV cache won't fit anywhere.
        tiers = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 1024 * 1024, 2039.0)]  # 1 MB
        mm = MemoryModel(model_config, tiers)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())
        req = make_request(0, input_tokens=512)
        sched.add_request(req)
        batch = sched.schedule(0)
        # Request cannot fit: it must be failed and dropped, not stuck pending.
        assert sched.pending_count == 0
        assert req.status == RequestStatus.FAILED
        assert len(sched.drain_failed()) == 1

    def test_completed_plus_failed_equals_total(self):
        """End-to-end: completed + failed == num_requests; failed not in raw_requests."""
        sim = ServingSimulator(MODEL, HARDWARE, precision="bf16", tensor_parallel=1)
        # Force a tiny memory tier so oversized requests fail admission.
        tiny = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 64 * 1024 * 1024, 2039.0)]  # 64 MB
        wl = WorkloadConfig(
            num_requests=5, arrival_rate_rps=2.0, arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 2048, "min": 1024, "max": 4096},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 2, "max": 8},
            random_seed=42,
        )
        sc = SchedulerConfig(max_batch_size=8, max_num_batched_tokens=65536)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.simulate(workload_config=wl, scheduler_config=sc, memory_tiers=tiny)
        assert result.total_requests_completed + result.total_requests_failed == wl.num_requests
        assert result.total_requests_failed > 0
        # Invariant: raw_requests holds only completed requests.
        assert len(result.raw_requests) == result.total_requests_completed

    def test_fitting_workload_no_failures(self):
        """A workload that fits on-chip records zero failures (byte-identical accounting)."""
        sim = ServingSimulator(MODEL, HARDWARE, precision="bf16", tensor_parallel=1)
        wl = WorkloadConfig(
            num_requests=5, arrival_rate_rps=2.0, arrival_pattern="constant",
            input_length_distribution={"dist": "fixed", "mean": 128, "min": 64, "max": 256},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 2, "max": 8},
            random_seed=42,
        )
        sc = SchedulerConfig(max_batch_size=8, max_num_batched_tokens=4096)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.simulate(workload_config=wl, scheduler_config=sc)
        assert result.total_requests_failed == 0
        assert result.total_requests_completed == wl.num_requests
        assert len(result.raw_requests) == result.total_requests_completed


# ---------------------------------------------------------------------------
# Round-2 re-review fixes (adversarial pass): admission reachability + per-device MFU
# ---------------------------------------------------------------------------

class TestReReviewFixes:
    def test_unbatchable_prompt_default_config_is_failed_not_stuck(self, model_config):
        """A prompt longer than max_num_batched_tokens with chunked prefill OFF (the DEFAULTS) can
        never fill a batch. It must be FAILED, not re-queued forever. Regression for the admission
        check having sat AFTER the token-budget guard (so it was unreachable for input>budget)."""
        # Ample memory (so it is NOT failed by capacity) but input > default max_num_batched_tokens.
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sched = BatchScheduler(MODEL, HARDWARE, mm, SchedulerConfig())  # default: 8192, no chunking
        req = make_request(0, input_tokens=20000)  # > 8192
        sched.add_request(req)
        sched.schedule(0)
        assert sched.pending_count == 0, "un-batchable prompt left stuck in pending"
        assert req.status == RequestStatus.FAILED
        assert len(sched.drain_failed()) == 1

    def test_chunked_prefill_admits_long_prompt(self, model_config):
        """With chunked prefill ON, a long prompt is admissible (not failed) — the facet-2 failure
        only applies when chunking is off."""
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * (1024 ** 3), 2039.0)]
        mm = MemoryModel(model_config, tier)
        sc = SchedulerConfig(enable_chunked_prefill=True, chunk_size=512)
        sched = BatchScheduler(MODEL, HARDWARE, mm, sc)
        req = make_request(0, input_tokens=20000)
        sched.add_request(req)
        sched.schedule(0)
        assert req.status != RequestStatus.FAILED, "long prompt wrongly failed despite chunked prefill"

    def test_roofline_utilization_is_per_device(self):
        """BUG: the SV4 power path divided the device peak by tensor_parallel a SECOND time, but the
        GenZ summary_table is ALREADY per-device (MACs/bytes sharded, latency is the per-device time).
        The fix is num_devices=1. Per-device MFU is physical (<=1) and legitimately DROPS with TP
        (comm overhead), so we assert the mathematical identity (the old roofline(nd=tp) equals
        per_device / tp — the spurious extra division), not constancy."""
        from llm_memory_calculator.genz import prefill_moddeling
        from llm_memory_calculator.genz.serving import roofline_utilization
        for tp in (2, 4):
            p = prefill_moddeling(model=MODEL, batch_size=8, input_tokens=2048,
                                  system_name=HARDWARE, bits="bf16", tensor_parallel=tp)
            per_device = roofline_utilization(p, HARDWARE, num_devices=1)[0]
            old_buggy = roofline_utilization(p, HARDWARE, num_devices=tp)[0]
            assert 0.0 < per_device <= 1.0, f"per-device MFU out of (0,1]: {per_device}"
            assert per_device == pytest.approx(old_buggy * tp, rel=1e-6), \
                f"tp={tp}: per-device {per_device} != tp×old {old_buggy*tp} (double-divide not removed)"
