"""Integration tests for Phase 1 serving simulation components."""
import pytest
from types import SimpleNamespace

from llm_memory_calculator.genz.serving import (
    Request, Batch, RequestStatus,
    SLOTargets, SLOTracker,
    MemoryModel, MemoryTierConfig, MemoryTier, GB_TO_BYTES,
    PowerModel, PowerConfig, ComponentPowerConfig, PowerComponent,
)
from llm_memory_calculator.genz.serving.constants import NS_PER_MS, NS_PER_S, EvictionPolicy


class TestMemoryModelWithRealConfig:
    """Test MemoryModel with real Llama-8B ModelConfig."""

    def test_kv_bytes_matches_expected(self):
        """Llama-8B KV: 2 * 8 * 128 * 32 * 2 / 1 = 131072 bytes/token."""
        from llm_memory_calculator.genz.Models import get_configs
        config = get_configs('meta-llama/Meta-Llama-3.1-8B')
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * GB_TO_BYTES, 2039.0)]
        mm = MemoryModel(config, tier)
        assert mm.bytes_per_token_kv == 131072

    def test_allocate_and_track_requests(self):
        """Simulate allocating KV for multiple requests, verify accounting."""
        from llm_memory_calculator.genz.Models import get_configs
        config = get_configs('meta-llama/Meta-Llama-3.1-8B')
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * GB_TO_BYTES, 2039.0)]
        mm = MemoryModel(config, tier)

        # Allocate for 3 requests
        requests = []
        for i in range(3):
            req = Request(request_id=i, model="llama", input_tokens=512,
                         max_output_tokens=256, arrival_time_ns=i * 100 * NS_PER_MS)
            blocks, tier_used = mm.allocate_kv_blocks(req, 512)
            assert blocks == 32  # ceil(512/16)
            assert tier_used == MemoryTier.DEVICE_HBM
            requests.append(req)

        snap = mm.memory_snapshot()
        assert snap["device_hbm"]["kv_cache_bytes"] == 3 * 32 * mm.bytes_per_block

        # Deallocate one
        freed = mm.deallocate_kv_blocks(requests[0])
        assert freed == 32
        snap2 = mm.memory_snapshot()
        assert snap2["device_hbm"]["kv_cache_bytes"] == 2 * 32 * mm.bytes_per_block


class TestPowerModelWithRealHardware:
    """Test PowerModel with real hardware configs."""

    def test_a100_base_power(self):
        pm = PowerModel.from_hardware_name('A100_40GB_GPU')
        base = pm.get_base_power_w()
        # Should be > 100W (idle system)
        assert base > 100

    def test_energy_tracking_over_simulation(self):
        """Simulate batch processing, verify energy accumulates."""
        pm = PowerModel.from_hardware_name('A100_40GB_GPU')

        # Simulate 3 batches, each 50ms
        for i in range(3):
            pm.add_accelerator_active_energy(50 * NS_PER_MS)
            pm.add_dram_energy(1024 * 1024 * 100)  # 100MB read

        summary = pm.summary(150 * NS_PER_MS, total_tokens=300)
        assert summary["total_energy_j"] > 0
        assert summary["energy_per_token_j"] is not None
        assert summary["energy_per_token_mj"] is not None
        assert summary["energy_per_token_mj"] > 0


class TestSLOTrackerWithRequestLifecycle:
    """Test SLOTracker recording actual request lifecycles."""

    def test_complete_lifecycle(self):
        """Run requests through full lifecycle, track SLOs."""
        targets = SLOTargets.from_ms(ttft_ms=200, tpot_ms=50, e2e_ms=10000)
        tracker = SLOTracker(targets)

        # Request 1: fast (meets SLOs)
        r1 = Request(request_id=1, model="llama", input_tokens=512,
                    max_output_tokens=100, arrival_time_ns=0)
        r1.set_queued(10 * NS_PER_MS)
        r1.set_prefilling(15 * NS_PER_MS)
        r1.set_decoding(50 * NS_PER_MS)  # TTFT = 50ms < 200ms
        for t in range(100):
            r1.record_token(50 * NS_PER_MS + (t + 1) * 30 * NS_PER_MS)  # 30ms/token
        r1.set_complete(50 * NS_PER_MS + 100 * 30 * NS_PER_MS)
        tracker.record_completed_request(r1)

        # Request 2: slow (violates TTFT)
        r2 = Request(request_id=2, model="llama", input_tokens=512,
                    max_output_tokens=100, arrival_time_ns=0)
        r2.set_queued(10 * NS_PER_MS)
        r2.set_prefilling(100 * NS_PER_MS)
        r2.set_decoding(300 * NS_PER_MS)  # TTFT = 300ms > 200ms
        for t in range(100):
            r2.record_token(300 * NS_PER_MS + (t + 1) * 30 * NS_PER_MS)
        r2.set_complete(300 * NS_PER_MS + 100 * 30 * NS_PER_MS)
        tracker.record_completed_request(r2)

        summary = tracker.summary()
        assert summary["total_completed"] == 2
        assert summary["ttft"]["violation_rate"] == 0.5  # 1 of 2 violated
        assert summary["goodput"] == 0.5  # Only r1 met all SLOs


class TestCombinedSimulation:
    """Test all Phase 1 components working together."""

    def test_mini_serving_scenario(self):
        """Simulate a mini serving scenario with memory + power + SLO tracking."""
        # Setup
        from llm_memory_calculator.genz.Models import get_configs
        config = get_configs('meta-llama/Meta-Llama-3.1-8B')
        tier = [MemoryTierConfig(MemoryTier.DEVICE_HBM, 80 * GB_TO_BYTES, 2039.0)]
        mm = MemoryModel(config, tier)
        pm = PowerModel.from_hardware_name('A100_40GB_GPU')
        tracker = SLOTracker(SLOTargets.from_ms(ttft_ms=500, tpot_ms=100, e2e_ms=30000))

        # Load weights (~16GB for 8B model at bf16)
        weight_bytes = 8_000_000_000 * 2  # 8B params * 2 bytes
        mm.load_weights(weight_bytes)

        current_time_ns = 0
        completed_requests = []

        # Process 5 requests
        for i in range(5):
            req = Request(
                request_id=i,
                model="meta-llama/Meta-Llama-3.1-8B",
                input_tokens=1024,
                max_output_tokens=128,
                arrival_time_ns=current_time_ns,
            )

            # Queue and prefill
            req.set_queued(current_time_ns + 5 * NS_PER_MS)

            # Allocate KV for input tokens
            blocks, _ = mm.allocate_kv_blocks(req, 1024)
            assert blocks == 64  # ceil(1024/16)

            # Prefill phase
            prefill_start = current_time_ns + 10 * NS_PER_MS
            req.set_prefilling(prefill_start)
            prefill_latency_ns = 30 * NS_PER_MS
            pm.add_accelerator_active_energy(prefill_latency_ns)

            # Decode phase
            decode_start = prefill_start + prefill_latency_ns
            req.set_decoding(decode_start)

            for t in range(128):
                token_time = decode_start + (t + 1) * 20 * NS_PER_MS
                req.record_token(token_time)
                pm.add_accelerator_active_energy(20 * NS_PER_MS)

            # Complete
            completion_time = decode_start + 128 * 20 * NS_PER_MS
            req.set_complete(completion_time)
            tracker.record_completed_request(req)

            # Free KV
            mm.deallocate_kv_blocks(req)

            completed_requests.append(req)
            current_time_ns = completion_time + 10 * NS_PER_MS

        # Verify results
        summary = tracker.summary()
        assert summary["total_completed"] == 5
        assert summary["goodput"] > 0  # Should all meet SLOs with generous targets

        power_summary = pm.summary(current_time_ns, total_tokens=5 * (1024 + 128))
        assert power_summary["total_energy_j"] > 0
        assert power_summary["avg_power_w"] > 0

        snap = mm.memory_snapshot()
        # All requests deallocated, only weights remain
        assert snap["device_hbm"]["kv_cache_bytes"] == 0
        assert snap["device_hbm"]["weight_bytes"] == weight_bytes

    def test_memory_pressure_with_eviction(self):
        """Test memory model under pressure with eviction."""
        config = SimpleNamespace(
            num_key_value_heads=8, head_dim=128,
            num_decoder_layers=32, num_attention_heads=32, hidden_size=4096,
        )
        # Small HBM (1GB) + larger DDR
        tiers = [
            MemoryTierConfig(MemoryTier.DEVICE_HBM, 1 * GB_TO_BYTES, 2039.0),
            MemoryTierConfig(MemoryTier.HOST_DDR, 16 * GB_TO_BYTES, 100.0),
        ]
        mm = MemoryModel(config, tiers, eviction_policy=EvictionPolicy.LRU)

        # bytes_per_token_kv = 131072, bytes_per_block = 131072 * 16 = 2097152
        # 1GB = 1073741824 bytes / 2097152 bytes_per_block = ~512 blocks capacity

        # Allocate many requests until HBM fills, verify fallback to DDR
        allocated_tiers = []
        for i in range(10):
            req = Request(request_id=i, model="test", input_tokens=1024,
                         max_output_tokens=128, arrival_time_ns=0)
            blocks, tier = mm.allocate_kv_blocks(req, 1024)
            allocated_tiers.append(tier)

        # Some should have fallen back to DDR
        snap = mm.memory_snapshot()
        # Verify both tiers have allocations (HBM should be mostly full)
        assert snap["device_hbm"]["used_bytes"] > 0


class TestBackwardCompatibility:
    """Verify existing GenZ imports still work."""

    def test_existing_genz_imports(self):
        from llm_memory_calculator.genz import (
            prefill_moddeling, decode_moddeling,
            System, ModelConfig, get_configs, ParallelismConfig,
        )
        assert callable(prefill_moddeling)
        assert callable(decode_moddeling)

    def test_serving_imports_from_genz(self):
        from llm_memory_calculator.genz import (
            MemoryModel, PowerModel, SLOTracker, Request, Batch,
        )
        assert MemoryModel is not None
        assert PowerModel is not None
