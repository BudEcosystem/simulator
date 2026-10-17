"""The parameter counter keeps DeepSeek's first ``first_k_dense_replace`` layers out of the MoE
count."""

import json
import pathlib

import pytest

from llm_memory_calculator.parameter_counter import UniversalParameterCounter

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

DEEPSEEK_V3 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "first_k_dense_replace": 3,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_layer_freq": 1,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 129280,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}
DEEPSEEK_V2_LITE = json.loads((FIXTURES / "deepseek_v2_lite_chat_config.json").read_text())


@pytest.mark.parametrize(
    "config, published_b",
    [(DEEPSEEK_V3, 671.0), (DEEPSEEK_V2_LITE, 15.7)],
    ids=["deepseek-v3", "deepseek-v2-lite"],
)
def test_deepseek_leading_dense_layers_hold_no_experts(config, published_b):
    total = UniversalParameterCounter().count_parameters(config) / 1e9
    assert total == pytest.approx(published_b, rel=0.01)
