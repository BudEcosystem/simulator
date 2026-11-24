"""vLLM v0.34.0 moved its default KV block from 16 to 64 tokens (vllm-project/vllm#63231).

Neither budsim nor budcluster passes ``--block-size``, so the planner models the block the engine picks by itself,
which depends on the build: the upstream base when BudConnect lists one, else the Bud build number.
"""

import pytest

from llm_memory_calculator.kv.geometry import KVGeometry, vllm_default_block_size
from llm_memory_calculator.runtime.modules.legacy import engine_block_size, plan_t1

DENSE = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}


@pytest.mark.parametrize(
    "version, block",
    [
        ("0.34.0", 64),
        ("v0.34.1", 64),
        ("0.35.0|vllm/vllm-openai:v0.35.0", 64),
        ("1.0.0", 64),
        ("0.32.0", 16),
        ("0.30.0|vllm/vllm-openai:v0.30.0", 16),
        # Bud build numbers: Bud 0.12.0 is upstream v0.34.0, Bud 0.11.0 is v0.32.0
        ("0.12.0|registry.bud.studio/runtime/vllm-cuda:0.12.0", 64),
        ("0.11.0", 16),
        ("0.10.0", 16),
        # nothing to go on: the measured default every calibration here ran on
        (None, 16),
        ("", 16),
        ("latest", 16),
    ],
)
def test_default_block_follows_the_engine_build(version, block):
    assert vllm_default_block_size(version) == block


def test_dense_geometry_takes_the_engine_default():
    old = KVGeometry.from_model(DENSE, seq_length=4096)
    new = KVGeometry.from_model(DENSE, seq_length=4096, default_block_size=64)
    assert old.block_size == 16
    assert new.block_size == 64
    # whole blocks only: a 100-token prefix caches 96 tokens at 16 and 64 at 64
    assert new.prefix_bytes(100)[0] == 64
    assert old.prefix_bytes(100)[0] == 96
    assert new.offload_block(256) == 256
    assert new.offload_block(80) == 128


def _t1_group(**extra):
    return {
        "type": "cuda",
        "engine_type": "vllm",
        "kv_features": ["cpu_offload_shm"],
        "args": {},
        **extra,
    }


T1_PLAN = {"tiers": [{"tier": "T1", "backend": "cpu", "size_gib": 2, "params": {"block_size": 80}}]}


@pytest.mark.parametrize(
    "extra, block",
    [
        ({"upstream_vllm": "0.34.0"}, 128),
        ({"engine_version": "0.12.0"}, 128),
        ({"upstream_vllm": "0.32.0"}, 80),
        ({}, 80),
        # an explicit --block-size from budsim's evolution search wins
        ({"upstream_vllm": "0.34.0", "args": {"block-size": 32}}, 96),
    ],
)
def test_t1_offload_block_is_a_whole_number_of_engine_blocks(extra, block):
    group = _t1_group(**extra)
    out = plan_t1(group, T1_PLAN, engine_block_size(group))
    assert out["block"] == block


def test_a_hybrids_block_aligns_to_the_engine_default():
    """vLLM v0.34.0 aligns a hybrid's attention block to max(16, the default block): 784 tokens at 16 is 832 at 64."""
    from tests.test_kv_planner import QWEN3_8_27B

    old = KVGeometry.from_model(QWEN3_8_27B, seq_length=4096)
    new = KVGeometry.from_model(QWEN3_8_27B, seq_length=4096, default_block_size=64)
    assert old.block_size == 784
    assert new.block_size == 832
    assert new.state_bytes_stored() == 48 * 832 * 4096
