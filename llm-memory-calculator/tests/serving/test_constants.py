"""Tests for serving constants and enums."""
import pytest
from llm_memory_calculator.genz.serving.constants import (
    MemoryTier,
    EvictionPolicy,
    RequestStatus,
    PowerState,
    PowerComponent,
    GB_TO_BYTES,
    MB_TO_BYTES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PRECISION_BYTES,
    NS_PER_MS,
    NS_PER_S,
)


class TestMemoryTier:
    def test_has_expected_members(self):
        assert MemoryTier.DEVICE_HBM.value == "device_hbm"
        assert MemoryTier.HOST_DDR.value == "host_ddr"
        assert MemoryTier.CXL.value == "cxl"
        assert MemoryTier.NVME.value == "nvme"

    def test_member_count(self):
        assert len(MemoryTier) == 4

    def test_ordering_by_value(self):
        tiers = sorted(MemoryTier, key=lambda t: t.value)
        assert tiers[0] == MemoryTier.CXL


class TestEvictionPolicy:
    def test_has_expected_members(self):
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"

    def test_member_count(self):
        assert len(EvictionPolicy) == 2


class TestRequestStatus:
    def test_has_expected_members(self):
        expected = ["arrived", "queued", "prefilling", "decoding", "complete", "failed"]
        for val in expected:
            assert RequestStatus(val) is not None

    def test_member_count(self):
        assert len(RequestStatus) == 6

    def test_lifecycle_order(self):
        statuses = [
            RequestStatus.ARRIVED,
            RequestStatus.QUEUED,
            RequestStatus.PREFILLING,
            RequestStatus.DECODING,
            RequestStatus.COMPLETE,
        ]
        values = [s.value for s in statuses]
        assert values == ["arrived", "queued", "prefilling", "decoding", "complete"]


class TestPowerState:
    def test_has_expected_members(self):
        assert PowerState.IDLE.value == "idle"
        assert PowerState.ACTIVE.value == "active"
        assert PowerState.STANDBY.value == "standby"

    def test_member_count(self):
        assert len(PowerState) == 3


class TestPowerComponent:
    def test_has_expected_members(self):
        expected = [
            "accelerator", "dram", "interconnect",
            "host_cpu", "cooling", "storage", "misc",
        ]
        for val in expected:
            assert PowerComponent(val) is not None

    def test_member_count(self):
        assert len(PowerComponent) == 7


class TestConstants:
    def test_gb_to_bytes(self):
        assert GB_TO_BYTES == 1024 ** 3
        assert GB_TO_BYTES == 1_073_741_824

    def test_mb_to_bytes(self):
        assert MB_TO_BYTES == 1024 ** 2
        assert MB_TO_BYTES == 1_048_576

    def test_default_block_size(self):
        assert DEFAULT_BLOCK_SIZE == 16

    def test_default_precision_bytes(self):
        assert DEFAULT_PRECISION_BYTES == 2

    def test_ns_per_ms(self):
        assert NS_PER_MS == 1_000_000

    def test_ns_per_s(self):
        assert NS_PER_S == 1_000_000_000
