"""Shared test fixtures for serving simulation tests."""
import pytest
from types import SimpleNamespace

from llm_memory_calculator.genz.serving.constants import (
    MemoryTier, EvictionPolicy,
    GB_TO_BYTES, DEFAULT_BLOCK_SIZE, DEFAULT_PRECISION_BYTES,
)


@pytest.fixture
def llama_8b_config():
    """Mock ModelConfig matching Llama-3.1-8B architecture.

    KV bytes/token = 2 * 8 * 128 * 32 * 2 / 1 = 131072
    """
    return SimpleNamespace(
        model="meta-llama/Meta-Llama-3.1-8B",
        num_key_value_heads=8,
        head_dim=128,
        num_decoder_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=128256,
        intermediate_size=14336,
    )


@pytest.fixture
def llama_70b_config():
    """Mock ModelConfig matching Llama-3.1-70B architecture."""
    return SimpleNamespace(
        model="meta-llama/Meta-Llama-3.1-70B",
        num_key_value_heads=8,
        head_dim=128,
        num_decoder_layers=80,
        num_attention_heads=64,
        hidden_size=8192,
        vocab_size=128256,
        intermediate_size=28672,
    )


@pytest.fixture
def a100_tier_config():
    """Single-tier A100 80GB HBM config."""
    from llm_memory_calculator.genz.serving.memory_model import MemoryTierConfig
    return [
        MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=80 * GB_TO_BYTES,
            bandwidth_gbps=2039.0,
            latency_ns=100.0,
        )
    ]


@pytest.fixture
def multi_tier_config():
    """Multi-tier config: 80GB HBM + 256GB DDR + 1TB NVME."""
    from llm_memory_calculator.genz.serving.memory_model import MemoryTierConfig
    return [
        MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=80 * GB_TO_BYTES,
            bandwidth_gbps=2039.0,
            latency_ns=100.0,
        ),
        MemoryTierConfig(
            tier=MemoryTier.HOST_DDR,
            capacity_bytes=256 * GB_TO_BYTES,
            bandwidth_gbps=100.0,
            latency_ns=500.0,
        ),
        MemoryTierConfig(
            tier=MemoryTier.NVME,
            capacity_bytes=1024 * GB_TO_BYTES,
            bandwidth_gbps=7.0,
            latency_ns=10000.0,
        ),
    ]
