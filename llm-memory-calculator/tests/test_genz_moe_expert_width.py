"""GenZ's default FFN path sizes routed experts by moe_intermediate_size (Qwen2/Qwen3-MoE).

Qwen3-30B-A3B declares a dense intermediate_size of 6144 and 768 per expert. Sizing its 128 experts
at 6144 made GenZ count 435 GB of weights, refuse an 80 GB H100 ("All params would not fit") and
fall back to heuristics, so prefill and KV break-even estimates for these models were wrong.
"""

import pytest

from llm_memory_calculator.genz.Models.ffn import ffn_decode, ffn_prefill
from llm_memory_calculator.genz.Models.get_language_model import get_configs
from llm_memory_calculator.genz.parallelism import ParallelismConfig

QWEN3_30B_A3B = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
MIXTRAL = {
    "architectures": ["MixtralForCausalLM"],
    "model_type": "mixtral",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
}


def _expert_rows(layers):
    """Output rows of every expert up+gate GEMM, used or not (the weights GenZ counts)."""
    return sum(layer[1] for layer in layers if layer[0] == "up+gate")


@pytest.mark.parametrize(
    "builder", [lambda c, p: ffn_prefill(c, p, 4096), lambda c, p: ffn_decode(c, p)]
)
def test_routed_experts_use_their_own_width(builder):
    config = get_configs(QWEN3_30B_A3B)
    rows = _expert_rows(builder(config, ParallelismConfig(tensor_parallel=1)))
    assert rows == 128 * 768 * config.num_ffi


@pytest.mark.parametrize(
    "builder", [lambda c, p: ffn_prefill(c, p, 4096), lambda c, p: ffn_decode(c, p)]
)
def test_a_model_without_an_expert_width_keeps_intermediate_size(builder):
    config = get_configs(MIXTRAL)
    rows = _expert_rows(builder(config, ParallelismConfig(tensor_parallel=1)))
    assert rows == 8 * 14336 * config.num_ffi


def test_tensor_parallel_splits_the_expert_width():
    config = get_configs(QWEN3_30B_A3B)
    rows = _expert_rows(ffn_prefill(config, ParallelismConfig(tensor_parallel=2), 4096))
    assert rows == 128 * 384 * config.num_ffi
