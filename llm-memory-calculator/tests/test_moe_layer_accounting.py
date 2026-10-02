"""MoE layer accounting for the KV planner's active parameters: which layers hold routed experts.

DeepSeek keeps its first ``first_k_dense_replace`` layers dense, and ERNIE-4.5 names its experts
``moe_num_experts`` / ``moe_k`` and routes only between ``moe_layer_start_index`` and
``moe_layer_end_index``.
"""

import json
import pathlib

from llm_memory_calculator.kv.cost import active_parameters, is_moe

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

DEEPSEEK_V2_LITE = json.loads((FIXTURES / "deepseek_v2_lite_chat_config.json").read_text())
ERNIE_45_21B_A3B = {
    "architectures": ["Ernie4_5_MoeForCausalLM"],
    "model_type": "ernie4_5_moe",
    "hidden_size": 2560,
    "intermediate_size": 12288,
    "moe_intermediate_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 20,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "moe_num_experts": 64,
    "moe_num_shared_experts": 2,
    "moe_k": 6,
    "moe_layer_start_index": 1,
    "moe_layer_end_index": 27,
    "moe_layer_interval": 1,
    "vocab_size": 103424,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
}


def test_ernie_routes_through_its_own_expert_keys():
    assert is_moe(ERNIE_45_21B_A3B)
    # 21.8B in all; ~3B per token plus the embeddings
    assert 3.0 < active_parameters(ERNIE_45_21B_A3B) / 1e9 < 4.0


def test_deepseek_active_parameters_count_only_the_routed_layers():
    # V2-Lite: 15.7B in all, 2.4B per token without the 0.4B of embeddings and LM head
    assert 2.5 < active_parameters(DEEPSEEK_V2_LITE) / 1e9 < 3.0
