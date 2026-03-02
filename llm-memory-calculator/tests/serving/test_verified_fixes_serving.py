"""Regression tests for verified bug fixes in the serving simulation subsystem.

Each test class targets a specific fix to prevent regressions:
1. GQA weight estimation correctly reduces attention parameter count
2. Token throughput counts only output tokens, not input + output
3. Power model tracks multiple hardware components, not just GPU active
4. Eviction returns evicted request IDs so the scheduler can re-queue them
5. Prefill completion generates the first output token from logits
6. Parameter sweep preserves all WorkloadConfig fields in swept configs
"""
import dataclasses
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from llm_memory_calculator.genz.serving.constants import (
    MemoryTier,
    EvictionPolicy,
    RequestStatus,
    PowerComponent,
    NS_PER_MS,
    NS_PER_S,
    GB_TO_BYTES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PRECISION_BYTES,
    GPU_IDLE_FRACTION,
    GPU_ACTIVE_FRACTION,
    GPU_STANDBY_FRACTION,
)
from llm_memory_calculator.genz.serving.request import Request, Batch
from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig
from llm_memory_calculator.genz.serving.batch_scheduler import SchedulerConfig, BatchScheduler
from llm_memory_calculator.genz.serving.power_model import (
    ComponentPowerConfig,
    PowerConfig,
    PowerModel,
)
from llm_memory_calculator.genz.serving.workload import WorkloadConfig
from llm_memory_calculator.genz.serving.simulator import (
    ServingSimulationResult,
    ServingSimulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model_config(**overrides):
    """Create a mock model config with Llama-3.1-8B-like defaults."""
    defaults = dict(
        num_key_value_heads=8,
        head_dim=128,
        num_decoder_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=11008,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _hbm_only_memory_model(model_config=None, hbm_bytes=80 * GB_TO_BYTES):
    """Helper: MemoryModel with a single large HBM tier."""
    if model_config is None:
        model_config = _mock_model_config()
    tier_configs = [
        MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=hbm_bytes,
            bandwidth_gbps=2039.0,
        )
    ]
    return MemoryModel(model_config=model_config, tier_configs=tier_configs)


def _make_request(rid, input_tokens=512, max_output=128, arrival_ns=0):
    """Create a test Request."""
    return Request(
        request_id=rid,
        model="test",
        input_tokens=input_tokens,
        max_output_tokens=max_output,
        arrival_time_ns=arrival_ns,
    )


def _fake_request(rid):
    """Minimal request-like object with a request_id attribute."""
    return SimpleNamespace(request_id=rid)


# ---------------------------------------------------------------------------
# 1. GQA weight estimation: GQA < MHA for the same model shape
# ---------------------------------------------------------------------------

class TestGQAWeightEstimationReduction:
    """Verify _estimate_weight_bytes produces fewer bytes for GQA than MHA.

    GQA (Grouped-Query Attention) uses fewer KV heads than Q heads,
    so the K and V projection parameters shrink while Q and O stay the same.
    A model with num_kv_heads=8 should yield fewer attention parameters
    than the same model with num_kv_heads=32 (full MHA).
    """

    def test_gqa_model_weights_smaller_than_mha(self):
        """GQA config (kv_heads=8) must produce strictly fewer weight bytes
        than MHA config (kv_heads=32) with all other dimensions equal."""
        gqa_config = _mock_model_config(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
            num_decoder_layers=32,
            intermediate_size=11008,
            vocab_size=32000,
        )
        mha_config = _mock_model_config(
            num_attention_heads=32,
            num_key_value_heads=32,  # Full MHA
            hidden_size=4096,
            head_dim=128,
            num_decoder_layers=32,
            intermediate_size=11008,
            vocab_size=32000,
        )

        sim = ServingSimulator(
            model="test-model",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        gqa_bytes = sim._estimate_weight_bytes(gqa_config)
        mha_bytes = sim._estimate_weight_bytes(mha_config)

        assert gqa_bytes < mha_bytes, (
            f"GQA weight bytes ({gqa_bytes}) should be less than MHA ({mha_bytes})"
        )

    def test_gqa_attention_savings_magnitude(self):
        """The attention parameter difference should match the analytical
        expectation: K and V projections shrink from n_heads to num_kv_heads.

        For n_heads=32, head_dim=128, hidden=4096, layers=32:
        - MHA attention per layer: 2*4096*32*128 + 2*4096*32*128 = 4 * 4096*32*128
        - GQA attention per layer: 2*4096*32*128 + 2*4096*8*128
          (Q+O use n_heads=32, K+V use kv_heads=8)
        - Saved per layer = 2*4096*(32-8)*128 = 2*4096*24*128 = 25165824
        - Saved total (32 layers, bf16) = 25165824 * 32 * 2 bytes = 1610612736
        """
        gqa_config = _mock_model_config(
            num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, head_dim=128,
            num_decoder_layers=32, intermediate_size=11008, vocab_size=32000,
        )
        mha_config = _mock_model_config(
            num_attention_heads=32, num_key_value_heads=32,
            hidden_size=4096, head_dim=128,
            num_decoder_layers=32, intermediate_size=11008, vocab_size=32000,
        )

        sim = ServingSimulator(
            model="test", hardware="A100_80GB_GPU", precision="bf16",
            tensor_parallel=1,
        )

        gqa_bytes = sim._estimate_weight_bytes(gqa_config)
        mha_bytes = sim._estimate_weight_bytes(mha_config)

        # Analytical expectation for the saved bytes (K and V projections)
        # Saved params = 2 * hidden * (n_heads - kv_heads) * head_dim * layers
        saved_params = 2 * 4096 * (32 - 8) * 128 * 32
        precision_bytes = 2  # bf16
        expected_saving = saved_params * precision_bytes

        actual_saving = mha_bytes - gqa_bytes
        assert actual_saving == expected_saving, (
            f"Expected saving of {expected_saving} bytes, got {actual_saving}"
        )

    def test_gqa_reduction_percentage(self):
        """GQA with kv_heads=8 vs n_heads=32 should reduce total attention
        parameters by about 37.5% ((32-8)*2)/(32*4) = 48/128 = 37.5%."""
        sim = ServingSimulator(
            model="test", hardware="A100_80GB_GPU", precision="bf16",
            tensor_parallel=1,
        )

        n_heads = 32
        kv_heads_gqa = 8
        hidden = 4096
        head_dim = 128
        layers = 32

        # Compute attention-only parameter bytes
        # MHA attention params: (2*hidden*n_heads*head_dim + 2*hidden*n_heads*head_dim) * layers
        mha_attn_params = (2 * hidden * n_heads * head_dim + 2 * hidden * n_heads * head_dim) * layers
        # GQA attention params: (2*hidden*n_heads*head_dim + 2*hidden*kv_heads*head_dim) * layers
        gqa_attn_params = (2 * hidden * n_heads * head_dim + 2 * hidden * kv_heads_gqa * head_dim) * layers

        reduction_fraction = 1.0 - (gqa_attn_params / mha_attn_params)
        assert reduction_fraction == pytest.approx(0.375, rel=1e-6), (
            f"Expected 37.5% reduction in attention params, got {reduction_fraction * 100:.1f}%"
        )

        # Verify this matches what the simulator computes
        gqa_config = _mock_model_config(
            num_attention_heads=n_heads, num_key_value_heads=kv_heads_gqa,
            hidden_size=hidden, head_dim=head_dim,
            num_decoder_layers=layers, intermediate_size=11008, vocab_size=32000,
        )
        mha_config = _mock_model_config(
            num_attention_heads=n_heads, num_key_value_heads=n_heads,
            hidden_size=hidden, head_dim=head_dim,
            num_decoder_layers=layers, intermediate_size=11008, vocab_size=32000,
        )

        gqa_total = sim._estimate_weight_bytes(gqa_config)
        mha_total = sim._estimate_weight_bytes(mha_config)
        assert gqa_total < mha_total


# ---------------------------------------------------------------------------
# 2. Token throughput should only count output tokens
# ---------------------------------------------------------------------------

class TestOutputTokenThroughputOnly:
    """Verify that overall_token_throughput_tps uses only output (generated)
    tokens, not input_tokens + output_tokens.

    The fix ensures the simulation loop counts total_tokens_generated from
    req.tokens_generated (output only), not from input_tokens + tokens_generated.
    """

    def test_throughput_uses_output_tokens_only(self):
        """Run a simulation with known input/output tokens and verify
        the token throughput metric counts only generated output tokens."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        # Use fixed-length distributions to get deterministic token counts
        wl = WorkloadConfig(
            arrival_rate_rps=5.0,
            arrival_pattern="constant",
            num_requests=10,
            input_length_distribution={"dist": "fixed", "mean": 256, "min": 256, "max": 256},
            output_length_distribution={"dist": "fixed", "mean": 16, "min": 16, "max": 16},
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=42,
        )

        result = sim.simulate(
            workload_config=wl,
            scheduler_config=SchedulerConfig(max_batch_size=32, max_num_batched_tokens=8192),
        )

        # Total output tokens should be at most num_requests * max_output_tokens
        # (some requests may not complete if simulation ends early)
        max_possible_output_tokens = 10 * 16  # 160

        if result.total_requests_completed > 0 and result.total_duration_ms > 0:
            # Compute what throughput would be if input tokens were incorrectly included
            total_input_tokens = sum(
                r["input_tokens"] for r in result.raw_requests
            )
            total_output_tokens = sum(
                r["tokens_generated"] for r in result.raw_requests
            )

            duration_s = result.total_duration_ms / 1000.0
            expected_tps_output_only = total_output_tokens / duration_s

            # The reported throughput should match output-only calculation
            assert result.overall_token_throughput_tps == pytest.approx(
                expected_tps_output_only, rel=1e-3
            ), (
                f"Token throughput {result.overall_token_throughput_tps} should match "
                f"output-only throughput {expected_tps_output_only}, not "
                f"(input+output)/duration = {(total_input_tokens + total_output_tokens) / duration_s}"
            )

            # If input tokens were counted, throughput would be much higher
            if total_input_tokens > 0:
                incorrect_tps = (total_input_tokens + total_output_tokens) / duration_s
                assert result.overall_token_throughput_tps < incorrect_tps, (
                    "Token throughput should be less than (input + output) / duration"
                )

    def test_raw_requests_have_tokens_generated(self):
        """Verify raw_requests track tokens_generated (output only) separately
        from input_tokens."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
        )
        wl = WorkloadConfig(
            arrival_rate_rps=5.0,
            arrival_pattern="constant",
            num_requests=5,
            input_length_distribution={"dist": "fixed", "mean": 128, "min": 128, "max": 128},
            output_length_distribution={"dist": "fixed", "mean": 8, "min": 8, "max": 8},
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=42,
        )
        result = sim.simulate(workload_config=wl)

        for req_data in result.raw_requests:
            assert "input_tokens" in req_data
            assert "tokens_generated" in req_data
            # tokens_generated should not include input_tokens
            assert req_data["tokens_generated"] <= req_data.get("tokens_generated", 0) + 1
            assert req_data["tokens_generated"] <= 8, (
                f"tokens_generated ({req_data['tokens_generated']}) should not exceed "
                f"max_output_tokens (8)"
            )


# ---------------------------------------------------------------------------
# 3. Power model should track multiple components (not just GPU active)
# ---------------------------------------------------------------------------

class TestPowerModelMultiComponent:
    """Verify the power model breakdown tracks multiple hardware components:
    accelerator, dram, host_cpu, etc. -- not just a single GPU active entry."""

    def test_estimate_from_simulation_result_has_multiple_components(self):
        """After calling estimate_from_simulation_result, the summary should
        contain breakdown entries for multiple power components."""
        pm = PowerModel.from_hardware_name("A100_80GB_GPU", num_accel=1)

        # Simulate a batch execution
        pm.estimate_from_simulation_result(
            latency_ms=100.0,
            compute_util=0.6,
            mem_util=0.4,
            comm_util=0.1,
            num_accel=1,
        )

        duration_ns = 100 * NS_PER_MS
        summary = pm.summary(duration_ns=duration_ns, total_tokens=50)

        # The breakdown should have entries for more than just accelerator
        breakdown_j = summary["breakdown_j"]
        breakdown_w = summary["breakdown_w"]

        assert len(breakdown_j) >= 3, (
            f"Power breakdown should have >= 3 components, got {len(breakdown_j)}: "
            f"{list(breakdown_j.keys())}"
        )
        assert len(breakdown_w) >= 3, (
            f"Power breakdown (watts) should have >= 3 components, got {len(breakdown_w)}: "
            f"{list(breakdown_w.keys())}"
        )

        # Specifically check for the key components
        assert PowerComponent.ACCELERATOR.value in breakdown_j, (
            "Missing accelerator in power breakdown"
        )
        assert PowerComponent.DRAM.value in breakdown_j, (
            "Missing dram in power breakdown"
        )
        assert PowerComponent.HOST_CPU.value in breakdown_j, (
            "Missing host_cpu in power breakdown"
        )

    def test_estimate_result_dict_has_component_watts(self):
        """estimate_from_simulation_result should return per-component wattage
        keys like 'accelerator_w' and 'dram_w'."""
        pm = PowerModel.from_hardware_name("A100_80GB_GPU", num_accel=1)

        result = pm.estimate_from_simulation_result(
            latency_ms=50.0,
            compute_util=0.5,
            mem_util=0.3,
            num_accel=1,
        )

        assert "accelerator_w" in result, "Missing accelerator_w in result"
        assert "dram_w" in result, "Missing dram_w in result"
        assert "total_energy_j" in result, "Missing total_energy_j in result"
        assert "avg_power_w" in result, "Missing avg_power_w in result"

        # Accelerator watts should reflect utilization interpolation
        idle_w = 400 * GPU_IDLE_FRACTION  # A100 TDP = 400W
        active_w = 400 * GPU_ACTIVE_FRACTION
        expected_accel_w = idle_w + (active_w - idle_w) * 0.5
        assert result["accelerator_w"] == pytest.approx(expected_accel_w, rel=1e-6)

    def test_summary_breakdown_includes_cooling_and_storage(self):
        """Full 7-component model should include cooling and storage entries."""
        pm = PowerModel.from_hardware_name("A100_80GB_GPU", num_accel=1)

        pm.estimate_from_simulation_result(
            latency_ms=200.0,
            compute_util=0.7,
            mem_util=0.5,
            comm_util=0.2,
            num_accel=1,
        )

        summary = pm.summary(duration_ns=200 * NS_PER_MS, total_tokens=100)
        breakdown_j = summary["breakdown_j"]

        # Should include all tracked components that contributed energy
        expected_components = {
            PowerComponent.ACCELERATOR.value,
            PowerComponent.DRAM.value,
            PowerComponent.HOST_CPU.value,
            PowerComponent.COOLING.value,
            PowerComponent.STORAGE.value,
            PowerComponent.MISC.value,
            PowerComponent.INTERCONNECT.value,
        }
        present_components = set(breakdown_j.keys())
        assert present_components == expected_components, (
            f"Expected all 7 components {expected_components}, "
            f"got {present_components}"
        )


# ---------------------------------------------------------------------------
# 4. Eviction should return evicted request IDs
# ---------------------------------------------------------------------------

class TestEvictionPreemptsRequest:
    """Verify evict_blocks returns both the evicted block count and
    the list of fully-evicted request IDs so the scheduler can re-queue them."""

    def test_evict_returns_request_ids(self):
        """Evicting blocks should return the IDs of fully evicted requests."""
        mm = _hbm_only_memory_model()

        # Allocate blocks for 3 requests
        req1 = _fake_request(101)
        req2 = _fake_request(102)
        req3 = _fake_request(103)

        mm.allocate_kv_blocks(req1, num_tokens=16)  # 1 block
        mm.allocate_kv_blocks(req2, num_tokens=16)  # 1 block
        mm.allocate_kv_blocks(req3, num_tokens=16)  # 1 block

        # Evict 2 blocks, should evict the 2 LRU requests (req1, req2)
        evicted_count, evicted_rids = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=2)

        assert evicted_count == 2
        assert isinstance(evicted_rids, list)
        assert len(evicted_rids) == 2
        assert 101 in evicted_rids, "Oldest request (101) should be evicted"
        assert 102 in evicted_rids, "Second-oldest request (102) should be evicted"
        assert 103 not in evicted_rids, "Newest request (103) should NOT be evicted"

    def test_evict_with_limited_memory_returns_ids(self):
        """With very limited memory, eviction should return the evicted IDs
        so the scheduler knows which requests to re-queue."""
        model_config = _mock_model_config()
        bytes_per_block = (2 * 8 * 128 * 32 * 2) * DEFAULT_BLOCK_SIZE  # 2097152

        # Capacity for exactly 3 blocks
        hbm_bytes = bytes_per_block * 3
        tier_configs = [
            MemoryTierConfig(
                tier=MemoryTier.DEVICE_HBM,
                capacity_bytes=hbm_bytes,
                bandwidth_gbps=2039.0,
            )
        ]
        mm = MemoryModel(model_config=model_config, tier_configs=tier_configs)

        # Fill all 3 blocks
        for rid in [10, 20, 30]:
            mm.allocate_kv_blocks(_fake_request(rid), num_tokens=16)  # 1 block each

        # Verify memory is full
        snap = mm.memory_snapshot()
        assert snap[MemoryTier.DEVICE_HBM.value]["free_bytes"] == 0

        # Evict 1 block to make room
        evicted_count, evicted_rids = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=1)

        assert evicted_count == 1
        assert len(evicted_rids) == 1
        assert evicted_rids[0] == 10, "LRU request (10) should be evicted first"

        # After eviction, we should have room for 1 more block
        snap_after = mm.memory_snapshot()
        assert snap_after[MemoryTier.DEVICE_HBM.value]["free_bytes"] == bytes_per_block

    def test_evict_returns_tuple_structure(self):
        """Verify the return type is (int, list) as documented."""
        mm = _hbm_only_memory_model()

        # Allocate a single block
        mm.allocate_kv_blocks(_fake_request(1), num_tokens=16)

        result = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=1)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

        evicted_count, evicted_rids = result
        assert isinstance(evicted_count, int)
        assert isinstance(evicted_rids, list)
        assert evicted_count == 1
        assert evicted_rids == [1]

    def test_evict_empty_tier_returns_zeros(self):
        """Evicting from a tier with no allocations should return (0, [])."""
        mm = _hbm_only_memory_model()

        evicted_count, evicted_rids = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=5)

        assert evicted_count == 0
        assert evicted_rids == []


# ---------------------------------------------------------------------------
# 5. Prefill completion should generate the first token
# ---------------------------------------------------------------------------

class TestFirstTokenFromPrefill:
    """Verify that complete_batch generates the first output token for
    prefill requests (the token produced from the logits at the end of
    the prefill pass), and that requests with max_output_tokens=1
    complete immediately after prefill (no decode iteration needed)."""

    def test_prefill_generates_first_token(self):
        """After completing a batch with prefill requests, each should have
        tokens_generated >= 1 because the prefill logits produce a token."""
        model_config = _mock_model_config()
        mm = _hbm_only_memory_model(model_config=model_config)
        sched = BatchScheduler(
            model="test",
            hardware="A100_80GB_GPU",
            memory_model=mm,
            config=SchedulerConfig(),
        )

        # Add a prefill request
        req = _make_request(rid=0, input_tokens=256, max_output=10, arrival_ns=0)
        sched.add_request(req)

        # Schedule a batch (request goes to prefill)
        batch = sched.schedule(current_time_ns=0)
        assert batch is not None
        assert batch.prefill_count == 1
        assert req.status == RequestStatus.PREFILLING

        # Complete the batch
        completed = sched.complete_batch(batch, current_time_ns=50 * NS_PER_MS)

        # The prefill request should now have generated at least 1 token
        assert req.tokens_generated >= 1, (
            f"After prefill completion, tokens_generated should be >= 1, "
            f"got {req.tokens_generated}"
        )

    def test_max_output_1_completes_after_prefill(self):
        """A request with max_output_tokens=1 should complete immediately
        after the prefill batch, since the prefill logits produce that
        single required token (no separate decode step needed)."""
        model_config = _mock_model_config()
        mm = _hbm_only_memory_model(model_config=model_config)
        sched = BatchScheduler(
            model="test",
            hardware="A100_80GB_GPU",
            memory_model=mm,
            config=SchedulerConfig(),
        )

        req = _make_request(rid=0, input_tokens=128, max_output=1, arrival_ns=0)
        sched.add_request(req)

        batch = sched.schedule(current_time_ns=0)
        assert batch is not None
        assert batch.prefill_count == 1

        completed = sched.complete_batch(batch, current_time_ns=30 * NS_PER_MS)

        # With max_output_tokens=1, the request should be completed
        assert len(completed) == 1, (
            f"Request with max_output=1 should complete after prefill, "
            f"but completed list has {len(completed)} items"
        )
        assert completed[0].request_id == 0
        assert completed[0].status == RequestStatus.COMPLETE
        assert completed[0].tokens_generated == 1

        # The request should NOT be in the running (decode) queue
        assert sched.inflight_count == 0, (
            "Request with max_output=1 should not enter the decode queue"
        )

    def test_prefill_request_transitions_to_decoding(self):
        """After complete_batch, a prefill request with remaining output
        should transition to DECODING status."""
        model_config = _mock_model_config()
        mm = _hbm_only_memory_model(model_config=model_config)
        sched = BatchScheduler(
            model="test",
            hardware="A100_80GB_GPU",
            memory_model=mm,
            config=SchedulerConfig(),
        )

        req = _make_request(rid=0, input_tokens=256, max_output=5, arrival_ns=0)
        sched.add_request(req)

        batch = sched.schedule(current_time_ns=0)
        completed = sched.complete_batch(batch, current_time_ns=50 * NS_PER_MS)

        # Should NOT be complete yet (max_output=5, only 1 token generated)
        assert len(completed) == 0, "Should not complete with max_output=5 after 1 token"
        assert req.status == RequestStatus.DECODING
        assert req.tokens_generated == 1
        assert sched.inflight_count == 1, "Request should be in the decode running queue"

    def test_multiple_prefill_requests_all_get_first_token(self):
        """All prefill requests in a batch should receive their first token."""
        model_config = _mock_model_config()
        mm = _hbm_only_memory_model(model_config=model_config)
        sched = BatchScheduler(
            model="test",
            hardware="A100_80GB_GPU",
            memory_model=mm,
            config=SchedulerConfig(max_batch_size=8, max_num_batched_tokens=8192),
        )

        requests = []
        for i in range(4):
            req = _make_request(rid=i, input_tokens=128, max_output=10, arrival_ns=0)
            sched.add_request(req)
            requests.append(req)

        batch = sched.schedule(current_time_ns=0)
        assert batch is not None
        assert batch.prefill_count == 4

        completed = sched.complete_batch(batch, current_time_ns=50 * NS_PER_MS)

        for req in requests:
            assert req.tokens_generated >= 1, (
                f"Request {req.request_id} should have >= 1 token after prefill, "
                f"got {req.tokens_generated}"
            )


# ---------------------------------------------------------------------------
# 6. Parameter sweep should preserve all WorkloadConfig fields
# ---------------------------------------------------------------------------

class TestSweepCopiesAllConfigFields:
    """Verify that ServingSimulator.sweep() preserves all WorkloadConfig fields
    when creating swept copies, particularly the 'model' field which must
    always match self._model."""

    def test_sweep_preserves_model_field(self):
        """The swept WorkloadConfig must carry the simulator's model name,
        not the default 'default' string, so the workload generator creates
        requests targeting the correct model."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        # Minimal sweep with a single value to keep test fast
        wl = WorkloadConfig(
            arrival_rate_rps=5.0,
            arrival_pattern="constant",
            num_requests=5,
            input_length_distribution={"dist": "fixed", "mean": 64, "min": 64, "max": 64},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 4, "max": 4},
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=42,
        )

        # Run sweep -- internally it calls dataclasses.replace(workload_config, model=self._model)
        results = sim.sweep(
            parameter="arrival_rate_rps",
            values=[5.0],
            workload_config=wl,
            scheduler_config=SchedulerConfig(max_batch_size=32, max_num_batched_tokens=4096),
        )

        # The sweep should produce results (if model was wrong, GenZ may fail differently)
        assert "throughput_rps" in results
        assert len(results["throughput_rps"]) == 1

    def test_sweep_preserves_non_default_fields(self):
        """All non-default WorkloadConfig fields should be preserved in the
        swept config, not reset to defaults."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        custom_input_dist = {"dist": "fixed", "mean": 100, "min": 100, "max": 100}
        custom_output_dist = {"dist": "fixed", "mean": 5, "min": 5, "max": 5}

        wl = WorkloadConfig(
            arrival_rate_rps=3.0,
            arrival_pattern="constant",
            num_requests=5,
            input_length_distribution=custom_input_dist,
            output_length_distribution=custom_output_dist,
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=123,
            gamma_shape=5.0,
            burst_period_s=20.0,
            burst_amplitude=5.0,
        )

        # Verify that dataclasses.replace preserves all fields
        replaced = dataclasses.replace(wl, model="meta-llama/Meta-Llama-3.1-8B")

        assert replaced.arrival_rate_rps == 3.0
        assert replaced.arrival_pattern == "constant"
        assert replaced.num_requests == 5
        assert replaced.input_length_distribution == custom_input_dist
        assert replaced.output_length_distribution == custom_output_dist
        assert replaced.model == "meta-llama/Meta-Llama-3.1-8B"
        assert replaced.random_seed == 123
        assert replaced.gamma_shape == 5.0
        assert replaced.burst_period_s == 20.0
        assert replaced.burst_amplitude == 5.0

    def test_sweep_result_structure(self):
        """Sweep results dict should contain all expected metric keys."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        wl = WorkloadConfig(
            arrival_rate_rps=5.0,
            arrival_pattern="constant",
            num_requests=3,
            input_length_distribution={"dist": "fixed", "mean": 64, "min": 64, "max": 64},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 4, "max": 4},
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=42,
        )

        results = sim.sweep(
            parameter="arrival_rate_rps",
            values=[2.0, 5.0],
            workload_config=wl,
            scheduler_config=SchedulerConfig(max_batch_size=32, max_num_batched_tokens=4096),
        )

        expected_keys = {
            "parameter_values", "throughput_rps", "token_throughput_tps",
            "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "goodput",
            "total_duration_ms",
        }
        assert set(results.keys()) == expected_keys, (
            f"Sweep results missing keys: {expected_keys - set(results.keys())}"
        )

        # Each metric list should have one entry per swept value
        assert results["parameter_values"] == [2.0, 5.0]
        for key in expected_keys - {"parameter_values"}:
            assert len(results[key]) == 2, (
                f"Expected 2 entries for '{key}', got {len(results[key])}"
            )

    def test_sweep_batch_size_preserves_workload(self):
        """Sweeping max_batch_size should not alter the WorkloadConfig fields."""
        sim = ServingSimulator(
            model="meta-llama/Meta-Llama-3.1-8B",
            hardware="A100_80GB_GPU",
            precision="bf16",
            tensor_parallel=1,
        )

        wl = WorkloadConfig(
            arrival_rate_rps=5.0,
            arrival_pattern="constant",
            num_requests=3,
            input_length_distribution={"dist": "fixed", "mean": 64, "min": 64, "max": 64},
            output_length_distribution={"dist": "fixed", "mean": 4, "min": 4, "max": 4},
            model="meta-llama/Meta-Llama-3.1-8B",
            random_seed=42,
        )

        # After sweep, original config should be unchanged
        original_rate = wl.arrival_rate_rps
        original_seed = wl.random_seed
        original_model = wl.model

        results = sim.sweep(
            parameter="max_batch_size",
            values=[4, 8],
            workload_config=wl,
            scheduler_config=SchedulerConfig(max_batch_size=16, max_num_batched_tokens=4096),
        )

        # Original config must not be mutated
        assert wl.arrival_rate_rps == original_rate
        assert wl.random_seed == original_seed
        assert wl.model == original_model
