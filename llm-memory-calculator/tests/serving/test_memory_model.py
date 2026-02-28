"""Tests for the multi-tier memory model for KV cache management."""
import pytest
from types import SimpleNamespace

from llm_memory_calculator.genz.serving.constants import (
    MemoryTier,
    EvictionPolicy,
    GB_TO_BYTES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PRECISION_BYTES,
)
from llm_memory_calculator.genz.serving.memory_model import (
    MemoryModel,
    MemoryTierConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_config(**overrides):
    """Create a mock model config with Llama-3.1-8B-like defaults."""
    defaults = dict(
        num_key_value_heads=8,
        head_dim=128,
        num_decoder_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _hbm_only_model(model_config=None, hbm_bytes=80 * GB_TO_BYTES, **kwargs):
    """Helper: MemoryModel with a single large HBM tier."""
    if model_config is None:
        model_config = _mock_config()
    tier_configs = [
        MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=hbm_bytes,
            bandwidth_gbps=3350.0,
        )
    ]
    return MemoryModel(
        model_config=model_config,
        tier_configs=tier_configs,
        **kwargs,
    )


def _two_tier_model(hbm_bytes, ddr_bytes, model_config=None, **kwargs):
    """Helper: MemoryModel with HBM + DDR tiers."""
    if model_config is None:
        model_config = _mock_config()
    tier_configs = [
        MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=hbm_bytes,
            bandwidth_gbps=3350.0,
        ),
        MemoryTierConfig(
            tier=MemoryTier.HOST_DDR,
            capacity_bytes=ddr_bytes,
            bandwidth_gbps=204.8,
        ),
    ]
    return MemoryModel(
        model_config=model_config,
        tier_configs=tier_configs,
        **kwargs,
    )


def _fake_request(rid):
    """Minimal request-like object with a request_id attribute."""
    return SimpleNamespace(request_id=rid)


# ---------------------------------------------------------------------------
# 1. bytes_per_token_kv - Llama 3.1 8B
# ---------------------------------------------------------------------------

class TestBytesPerTokenKV:

    def test_llama_8b_mock(self):
        """2 * kv_heads(8) * head_dim(128) * layers(32) * precision(2) / TP(1) = 131072."""
        mm = _hbm_only_model()
        assert mm.bytes_per_token_kv == 2 * 8 * 128 * 32 * 2 // 1
        assert mm.bytes_per_token_kv == 131072

    def test_with_real_model_config(self):
        """Same calculation using the real ModelConfig from the model registry."""
        from llm_memory_calculator.genz.Models import get_configs
        config = get_configs("meta-llama/Meta-Llama-3.1-8B")
        mm = _hbm_only_model(model_config=config)
        # Llama-3.1-8B: kv_heads=8, head_dim=128, layers=32, precision=2
        assert mm.bytes_per_token_kv == 131072


# ---------------------------------------------------------------------------
# 2. bytes_per_token_kv with tensor parallelism
# ---------------------------------------------------------------------------

class TestBytesPerTokenKVTensorParallel:

    def test_tp2(self):
        """With tensor_parallel=2, bytes halve: 131072 / 2 = 65536."""
        mm = _hbm_only_model(tensor_parallel=2)
        assert mm.bytes_per_token_kv == 65536

    def test_tp4(self):
        mm = _hbm_only_model(tensor_parallel=4)
        assert mm.bytes_per_token_kv == 131072 // 4


# ---------------------------------------------------------------------------
# 3. bytes_per_block
# ---------------------------------------------------------------------------

class TestBytesPerBlock:

    def test_default_block_size(self):
        """block_size=16 => bytes_per_block = 131072 * 16 = 2097152."""
        mm = _hbm_only_model(block_size=16)
        assert mm.bytes_per_block == 131072 * 16
        assert mm.bytes_per_block == 2097152

    def test_custom_block_size(self):
        mm = _hbm_only_model(block_size=32)
        assert mm.bytes_per_block == 131072 * 32


# ---------------------------------------------------------------------------
# 4. load_weights
# ---------------------------------------------------------------------------

class TestLoadWeights:

    def test_load_weights_updates_snapshot(self):
        mm = _hbm_only_model()
        weight_bytes = 16 * GB_TO_BYTES  # 16 GB
        mm.load_weights(weight_bytes, tier=MemoryTier.DEVICE_HBM)

        snap = mm.memory_snapshot()
        hbm = snap[MemoryTier.DEVICE_HBM.value]
        assert hbm["used_bytes"] == weight_bytes
        assert hbm["weight_bytes"] == weight_bytes
        assert hbm["kv_cache_bytes"] == 0
        assert hbm["free_bytes"] == 80 * GB_TO_BYTES - weight_bytes

    def test_load_weights_invalid_tier(self):
        mm = _hbm_only_model()
        with pytest.raises(ValueError, match="not configured"):
            mm.load_weights(1024, tier=MemoryTier.CXL)


# ---------------------------------------------------------------------------
# 5. allocate_kv_blocks
# ---------------------------------------------------------------------------

class TestAllocateKVBlocks:

    def test_basic_allocation(self):
        mm = _hbm_only_model()
        req = _fake_request(1)
        blocks, tier = mm.allocate_kv_blocks(req, num_tokens=1024)

        # ceil(1024 / 16) = 64 blocks
        assert blocks == 64
        assert tier == MemoryTier.DEVICE_HBM

    def test_partial_block(self):
        """17 tokens => ceil(17/16) = 2 blocks."""
        mm = _hbm_only_model()
        req = _fake_request(2)
        blocks, _ = mm.allocate_kv_blocks(req, num_tokens=17)
        assert blocks == 2

    def test_allocation_updates_used(self):
        mm = _hbm_only_model()
        req = _fake_request(3)
        blocks, _ = mm.allocate_kv_blocks(req, num_tokens=1024)
        snap = mm.memory_snapshot()
        expected_used = blocks * mm.bytes_per_block
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == expected_used


# ---------------------------------------------------------------------------
# 6. deallocate_kv_blocks
# ---------------------------------------------------------------------------

class TestDeallocateKVBlocks:

    def test_deallocate_frees_memory(self):
        mm = _hbm_only_model()
        req = _fake_request(10)
        mm.allocate_kv_blocks(req, num_tokens=512)
        freed = mm.deallocate_kv_blocks(req)
        # ceil(512/16) = 32 blocks
        assert freed == 32
        snap = mm.memory_snapshot()
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == 0

    def test_deallocate_unknown_request(self):
        mm = _hbm_only_model()
        freed = mm.deallocate_kv_blocks(_fake_request(999))
        assert freed == 0

    def test_deallocate_only_target_request(self):
        mm = _hbm_only_model()
        req_a = _fake_request(100)
        req_b = _fake_request(101)
        mm.allocate_kv_blocks(req_a, num_tokens=256)
        mm.allocate_kv_blocks(req_b, num_tokens=256)

        freed = mm.deallocate_kv_blocks(req_a)
        # ceil(256/16) = 16 blocks
        assert freed == 16
        # req_b should still have its blocks
        snap = mm.memory_snapshot()
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == 16 * mm.bytes_per_block


# ---------------------------------------------------------------------------
# 7. eviction - LRU policy
# ---------------------------------------------------------------------------

class TestEvictionLRU:

    def test_lru_evicts_oldest_first(self):
        mm = _hbm_only_model()
        req_old = _fake_request(1)
        req_mid = _fake_request(2)
        req_new = _fake_request(3)

        mm.allocate_kv_blocks(req_old, num_tokens=16)   # 1 block
        mm.allocate_kv_blocks(req_mid, num_tokens=16)   # 1 block
        mm.allocate_kv_blocks(req_new, num_tokens=16)   # 1 block

        evicted = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=1)
        assert evicted == 1

        # req_old (rid=1) should have been evicted
        snap = mm.memory_snapshot()
        expected_used = 2 * mm.bytes_per_block  # mid + new remain
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == expected_used

    def test_lru_evicts_multiple(self):
        mm = _hbm_only_model()
        for i in range(5):
            mm.allocate_kv_blocks(_fake_request(i), num_tokens=16)

        evicted = mm.evict_blocks(MemoryTier.DEVICE_HBM, num_blocks=3)
        assert evicted == 3
        snap = mm.memory_snapshot()
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == 2 * mm.bytes_per_block

    def test_evict_unconfigured_tier(self):
        mm = _hbm_only_model()
        evicted = mm.evict_blocks(MemoryTier.CXL, num_blocks=1)
        assert evicted == 0


# ---------------------------------------------------------------------------
# 8. multi-tier fallback
# ---------------------------------------------------------------------------

class TestMultiTierFallback:

    def test_fallback_to_ddr_when_hbm_full(self):
        # Very small HBM, large DDR
        block_bytes = 2097152  # bytes_per_block for default config
        hbm_bytes = block_bytes * 2  # room for 2 blocks only
        ddr_bytes = 100 * GB_TO_BYTES
        mm = _two_tier_model(hbm_bytes=hbm_bytes, ddr_bytes=ddr_bytes)

        req1 = _fake_request(1)
        blocks1, tier1 = mm.allocate_kv_blocks(req1, num_tokens=32)  # 2 blocks - fills HBM
        assert tier1 == MemoryTier.DEVICE_HBM
        assert blocks1 == 2

        req2 = _fake_request(2)
        blocks2, tier2 = mm.allocate_kv_blocks(req2, num_tokens=16)  # 1 block - should spill to DDR
        assert tier2 == MemoryTier.HOST_DDR
        assert blocks2 == 1

    def test_memory_error_when_all_full(self):
        block_bytes = 2097152
        mm = _two_tier_model(
            hbm_bytes=block_bytes,  # 1 block capacity
            ddr_bytes=block_bytes,  # 1 block capacity
        )
        req1 = _fake_request(1)
        mm.allocate_kv_blocks(req1, num_tokens=16)  # fills HBM
        req2 = _fake_request(2)
        mm.allocate_kv_blocks(req2, num_tokens=16)  # fills DDR

        with pytest.raises(MemoryError):
            mm.allocate_kv_blocks(_fake_request(3), num_tokens=16)


# ---------------------------------------------------------------------------
# 9. spill_to_next_tier
# ---------------------------------------------------------------------------

class TestSpillToNextTier:

    def test_spill_hbm_to_ddr(self):
        mm = _two_tier_model(
            hbm_bytes=80 * GB_TO_BYTES,
            ddr_bytes=100 * GB_TO_BYTES,
        )
        # Manually mark some HBM as used
        req = _fake_request(1)
        mm.allocate_kv_blocks(req, num_tokens=64)  # 4 blocks

        moved, target = mm.spill_to_next_tier(MemoryTier.DEVICE_HBM, num_blocks=2)
        assert moved == 2
        assert target == MemoryTier.HOST_DDR

        snap = mm.memory_snapshot()
        # HBM: 4 blocks allocated - 2 spilled = 2 blocks worth
        expected_hbm = 2 * mm.bytes_per_block
        assert snap[MemoryTier.DEVICE_HBM.value]["used_bytes"] == expected_hbm
        # DDR: 2 blocks spilled in
        expected_ddr = 2 * mm.bytes_per_block
        assert snap[MemoryTier.HOST_DDR.value]["used_bytes"] == expected_ddr

    def test_spill_no_lower_tier(self):
        mm = _hbm_only_model()
        req = _fake_request(1)
        mm.allocate_kv_blocks(req, num_tokens=16)
        moved, target = mm.spill_to_next_tier(MemoryTier.DEVICE_HBM, num_blocks=1)
        assert moved == 0
        assert target == MemoryTier.DEVICE_HBM

    def test_spill_nvme_has_no_lower_tier(self):
        tier_configs = [
            MemoryTierConfig(tier=MemoryTier.NVME, capacity_bytes=1 * GB_TO_BYTES),
        ]
        mm = MemoryModel(
            model_config=_mock_config(),
            tier_configs=tier_configs,
        )
        moved, target = mm.spill_to_next_tier(MemoryTier.NVME, num_blocks=1)
        assert moved == 0
        assert target == MemoryTier.NVME


# ---------------------------------------------------------------------------
# 10. memory_snapshot
# ---------------------------------------------------------------------------

class TestMemorySnapshot:

    def test_snapshot_has_all_fields(self):
        mm = _hbm_only_model()
        snap = mm.memory_snapshot()
        hbm = snap[MemoryTier.DEVICE_HBM.value]
        expected_fields = {
            "capacity_bytes",
            "used_bytes",
            "free_bytes",
            "utilization",
            "weight_bytes",
            "kv_cache_bytes",
            "bandwidth_gbps",
            "latency_ns",
        }
        assert set(hbm.keys()) == expected_fields

    def test_snapshot_utilization(self):
        mm = _hbm_only_model(hbm_bytes=100 * GB_TO_BYTES)
        mm.load_weights(50 * GB_TO_BYTES)
        snap = mm.memory_snapshot()
        assert snap[MemoryTier.DEVICE_HBM.value]["utilization"] == pytest.approx(0.5)

    def test_snapshot_multi_tier(self):
        mm = _two_tier_model(
            hbm_bytes=80 * GB_TO_BYTES,
            ddr_bytes=64 * GB_TO_BYTES,
        )
        snap = mm.memory_snapshot()
        assert MemoryTier.DEVICE_HBM.value in snap
        assert MemoryTier.HOST_DDR.value in snap
        assert len(snap) == 2

    def test_snapshot_kv_cache_bytes(self):
        mm = _hbm_only_model()
        weight_bytes = 16 * GB_TO_BYTES
        mm.load_weights(weight_bytes)
        mm.allocate_kv_blocks(_fake_request(1), num_tokens=1024)

        snap = mm.memory_snapshot()
        hbm = snap[MemoryTier.DEVICE_HBM.value]
        expected_kv = 64 * mm.bytes_per_block  # ceil(1024/16)=64 blocks
        assert hbm["weight_bytes"] == weight_bytes
        assert hbm["kv_cache_bytes"] == expected_kv
        assert hbm["used_bytes"] == weight_bytes + expected_kv


# ---------------------------------------------------------------------------
# 11. from_system factory
# ---------------------------------------------------------------------------

class TestFromSystem:

    def test_from_system_hbm_only(self):
        """Construct from a real-ish System object and verify tier setup."""
        from llm_memory_calculator.genz.system import System
        system = System(
            off_chip_mem_size=81920,  # 80 GB in MB (default unit)
            offchip_mem_bw=3350,      # GB/s (default unit)
        )
        config = _mock_config()
        mm = MemoryModel.from_system(model_config=config, system=system)

        snap = mm.memory_snapshot()
        assert MemoryTier.DEVICE_HBM.value in snap
        hbm = snap[MemoryTier.DEVICE_HBM.value]
        assert hbm["capacity_bytes"] == 81920 * 1024 * 1024  # 80 GB in bytes
        assert hbm["bandwidth_gbps"] == pytest.approx(3350.0)

    def test_from_system_with_extra_tiers(self):
        from llm_memory_calculator.genz.system import System
        system = System(off_chip_mem_size=81920, offchip_mem_bw=3350)
        config = _mock_config()
        mm = MemoryModel.from_system(
            model_config=config,
            system=system,
            host_ddr_gb=128,
            cxl_gb=256,
            nvme_gb=1024,
        )
        snap = mm.memory_snapshot()
        assert len(snap) == 4
        assert snap[MemoryTier.HOST_DDR.value]["capacity_bytes"] == 128 * GB_TO_BYTES
        assert snap[MemoryTier.CXL.value]["capacity_bytes"] == 256 * GB_TO_BYTES
        assert snap[MemoryTier.NVME.value]["capacity_bytes"] == 1024 * GB_TO_BYTES

    def test_from_system_tp(self):
        from llm_memory_calculator.genz.system import System
        system = System(off_chip_mem_size=81920, offchip_mem_bw=3350)
        config = _mock_config()
        mm = MemoryModel.from_system(
            model_config=config, system=system, tensor_parallel=4,
        )
        assert mm.bytes_per_token_kv == 131072 // 4
