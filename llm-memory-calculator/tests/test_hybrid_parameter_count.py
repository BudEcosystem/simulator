"""Parameter counts for hybrid/recurrent architectures, against real checkpoints.

Ground truth is `metadata.total_size` from each model's shipped
`model.safetensors.index.json`, divided by its dtype width. These are not
self-consistency checks -- every expected value comes from the actual published
weights.

The bug: every layer was charged a full Q/K/V/O projection set. Recurrent layers
have no QKVO at all, and what they do have (a fused in_proj, a depthwise conv,
low-rank gates, per-head decay vectors) is shaped nothing like attention.
"""

import json
from pathlib import Path

import pytest

from llm_memory_calculator.layer_plan import resolve_layer_plan
from llm_memory_calculator.mixer_params import (
    gdn_params,
    kda_params,
    mamba2_params,
    recurrent_mixer_params,
    shortconv_params,
)
from llm_memory_calculator.parameter_counter import UniversalParameterCounter

FIXTURES = Path(__file__).parent / "fixtures" / "hybrid_configs"


def cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def count(name):
    return UniversalParameterCounter().count_parameters(
        cfg(name), respect_weight_tying=False
    )


# (fixture, real params in billions, tolerance %, source note)
GROUND_TRUTH = [
    ("Qwen_Qwen3.6-27B", 27.781, 3),
    ("Qwen_Qwen3-Next-80B-A3B-Instruct", 81.325, 4),
    ("moonshotai_Kimi-Linear-48B-A3B-Instruct", 49.123, 5),
    ("ibm-granite_granite-4.0-h-small", 32.207, 4),
    ("nvidia_Nemotron-H-8B-Base-8K", 8.101, 4),
    ("nvidia_NVIDIA-Nemotron-Nano-9B-v2", 8.888, 8),
    ("tiiuae_Falcon-H1-34B-Instruct", 33.643, 3),
    ("ibm-ai-platform_Bamba-9B-v2", 9.780, 3),
    ("Zyphra_Zamba2-2.7B", 2.689, 8),
    ("ai21labs_Jamba-v0.1", 51.574, 5),
    ("MiniMaxAI_MiniMax-Text-01", 456.089, 3),
    ("LiquidAI_LFM2-2.6B", 2.569, 6),
    ("mistralai_Mamba-Codestral-7B-v0.1", 7.285, 3),
    ("state-spaces_mamba-2.8b-hf", 2.768, 6),
    ("PowerInfer_SmallThinker-21BA3B-Instruct", 21.507, 3),
    ("tencent_Hunyuan-A13B-Instruct", 80.0, 5),
]


@pytest.mark.parametrize("name,real_b,tol_pct", GROUND_TRUTH)
def test_parameter_count_matches_shipped_checkpoint(name, real_b, tol_pct):
    got_b = count(name) / 1e9
    err = abs(got_b - real_b) / real_b * 100
    assert err < tol_pct, f"{name}: {got_b:.2f}B vs real {real_b:.2f}B ({err:+.1f}%)"


# ------------------------------------------------- per-mechanism derivations


def test_gdn_mixer_matches_qwen36_tensor_shapes():
    """Every term pinned to a tensor in Qwen3.6-27B's safetensors header."""
    c = cfg("Qwen_Qwen3.6-27B")["text_config"]
    hidden = 5120
    expected = (
        10240 * hidden        # in_proj_qkv [10240, 5120]
        + 6144 * hidden       # in_proj_z   [6144, 5120]
        + 48 * hidden * 2     # in_proj_a, in_proj_b [48, 5120]
        + 10240 * 4           # conv1d      [10240, 1, 4]
        + 48 * 2              # A_log, dt_bias [48]
        + 128                 # norm        [128]
        + 5120 * 6144         # out_proj    [5120, 6144]
    )
    assert gdn_params(c, hidden) == expected
    # 10240 is 2*16*128 + 48*128: q and k are key-width, v is value-width.
    assert 2 * 16 * 128 + 48 * 128 == 10240


def test_kda_gates_are_low_rank_not_dense():
    """Kimi's f/g gates factor through head_dim: [128, 2304] then [4096, 128].

    Counting them dense would add ~2 x hidden x n_heads*head_dim per layer.
    """
    c = cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct")
    hidden, nh, hd, k = 2304, 32, 128, 4
    inner = nh * hd
    expected = (
        4 * hidden * inner            # q, k, v, o [4096, 2304] x3 + [2304, 4096]
        + 3 * inner * k               # q/k/v_conv1d [4096, 1, 4]
        + 2 * (hidden * hd + hd * inner)  # f_a/f_b and g_a/g_b
        + nh + inner + hd             # A_log, dt_bias [4096], o_norm [128]
    )
    assert kda_params(c, hidden) == expected
    dense_gates = 2 * hidden * inner
    assert dense_gates > 2 * (hidden * hd + hd * inner)


def test_mamba2_in_proj_and_conv_widths_match_two_vendors():
    """granite-4.0-h and Nemotron-Nano ship the same shape under different keys.

    granite:  in_proj [16768, 4096] = 2*8192 + 2*1*128 + 128
    nemotron: in_proj [22656, 4480] = 2*10240 + 2*8*128 + 128

    Nemotron's d_inner is mamba_num_heads*mamba_head_dim = 10240, which has no
    relation to expand*hidden_size -- inferring it that way is silently wrong.
    """
    g = cfg("ibm-granite_granite-4.0-h-small")
    assert 2 * 8192 + 2 * 1 * 128 + 128 == 16768
    assert 8192 + 2 * 1 * 128 == 8448
    got = mamba2_params(g, 4096)
    expected = (
        4096 * 16768 + 8448 * 4 + 8448 + 3 * 128 + 8192 + 8192 * 4096
    )
    assert got == expected

    n = cfg("nvidia_NVIDIA-Nemotron-Nano-9B-v2")
    assert n["mamba_num_heads"] * n["mamba_head_dim"] == 10240
    assert 2 * 10240 + 2 * 8 * 128 + 128 == 22656
    assert mamba2_params(n, 4480) == (
        4480 * 22656 + 12288 * 4 + 12288 + 3 * 128 + 10240 + 10240 * 4480
    )


def test_shortconv_matches_lfm2_shapes():
    """LFM2: in_proj [6144, 2048], conv [2048, 1, 3], out_proj [2048, 2048]."""
    c = cfg("LiquidAI_LFM2-2.6B")
    assert shortconv_params(c, 2048) == 3 * 2048 * 2048 + 2048 * 3 + 2048 * 2048


# --------------------------------------------------------- structural rules


def test_pure_ssm_stacks_are_charged_no_ffn():
    """A pure Mamba stack has no FFN -- the gated mixer is the whole block.

    Charging one per layer nearly doubled Mamba-Codestral (13.7B vs 7.29B).
    """
    for name in ("mistralai_Mamba-Codestral-7B-v0.1", "state-spaces_mamba-2.8b-hf"):
        plan = resolve_layer_plan(cfg(name))
        assert plan["num_ffn_layers"] == 0
        assert plan["num_attention_layers"] == 0


def test_nemotron_gives_the_ffn_its_own_layer_slot():
    """M/*/- : 27 mamba, 4 attention, 25 MLP-only -- they partition the depth."""
    c = cfg("nvidia_NVIDIA-Nemotron-Nano-9B-v2")
    plan = resolve_layer_plan(c)
    assert plan["num_ffn_layers"] == c["hybrid_override_pattern"].count("-")
    assert (
        plan["num_ffn_layers"]
        + plan["num_recurrent_layers"]
        + plan["num_attention_layers"]
        == c["num_hidden_layers"]
    )


def test_zamba2_shared_blocks_are_counted_once_not_per_layer():
    """Zamba2 re-invokes num_mem_blocks=2 shared attention+MLP blocks.

    Each invocation allocates its own KV, so the runtime attention-layer count
    stays at 9 -- but the weights exist twice, not nine times.
    """
    c = cfg("Zyphra_Zamba2-2.7B")
    plan = resolve_layer_plan(c)
    assert plan["num_attention_layers"] == 9        # runtime: KV on all of them
    assert plan["num_attention_param_layers"] == 2  # weights: shared blocks
    assert plan["num_ffn_param_layers"] == c["num_mem_blocks"]


def test_jamba_expert_layer_period_is_honored():
    """Jamba spells MoE frequency as expert_layer_period, not moe_layer_freq.

    Missing the alias charged a 16-expert FFN on all 32 layers instead of 16.
    """
    from llm_memory_calculator.config_normalizer import ConfigNormalizer

    c = cfg("ai21labs_Jamba-v0.1")
    assert c["expert_layer_period"] == 2 and "moe_layer_freq" not in c
    assert ConfigNormalizer.normalize_config(c)["moe_layer_freq"] == 2


def test_gated_ffn_falls_back_to_the_architecture_name():
    """SmallThinker ships neither `model_type` nor any activation key.

    Only `architectures: ["SmallThinkerForCausalLM"]`. With both primary signals
    absent the activation default ('gelu') won and charged 2 FFN matrices where
    the checkpoint has three -- experts are gate/up/down -- a flat 1/3
    under-count on the model's largest weight block.
    """
    c = cfg("PowerInfer_SmallThinker-21BA3B-Instruct")
    assert "model_type" not in c and "hidden_act" not in c
    assert UniversalParameterCounter()._is_gated_ffn(c) is True


def test_smallthinker_moe_key_aliases_resolve():
    """`moe_num_primary_experts` / `moe_ffn_hidden_size`, not the usual names."""
    from llm_memory_calculator.config_normalizer import ConfigNormalizer

    c = cfg("PowerInfer_SmallThinker-21BA3B-Instruct")
    n = ConfigNormalizer.normalize_config(c)
    assert n["n_routed_experts"] == 64
    assert n["moe_intermediate_size"] == 768
    assert n["expert_top_k"] == 6


def test_recurrent_mixer_params_is_zero_without_recurrent_layers():
    c = cfg("openai_gpt-oss-120b")
    assert recurrent_mixer_params(c, resolve_layer_plan(c), c["hidden_size"]) == 0


# ------------------------------------------------------ null-vs-absent keys


def test_null_valued_mla_ranks_do_not_collapse_the_count():
    """`q_lora_rank: null` means "no Q low-rank factorization", not "missing".

    `.get(key, 0)` returns the stored None, the arithmetic raised NoneType+int,
    and the whole count silently fell through to the crude fallback. Kimi-Linear
    reported 2.10B against 49.12B; DeepSeek-V2-Lite 1.57B against 15.7B.
    """
    c = cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct")
    assert c["q_lora_rank"] is None and c["kv_lora_rank"] == 512
    got_b = count("moonshotai_Kimi-Linear-48B-A3B-Instruct") / 1e9
    assert got_b > 40, f"still collapsing to the fallback: {got_b:.2f}B"


def test_approximate_models_are_the_only_ones_left_outside_tolerance():
    """Phi-4-flash and Hymba stay loose, and both say why.

    Phi-4-flash's Mamba/attention split is not expressible in the plan; Hymba is
    a parallel hybrid whose attention is narrower than its config implies. Both
    over-count, which is the safe direction, and neither is silent about it.
    """
    plan = resolve_layer_plan(cfg("microsoft_Phi-4-mini-flash-reasoning"))
    assert plan["approximate"] is True

    hymba = resolve_layer_plan(cfg("nvidia_Hymba-1.5B-Base"))
    assert hymba["parallel"] is True
    got_b = count("nvidia_Hymba-1.5B-Base") / 1e9
    assert 1.5 < got_b < 1.8  # real 1.52B; over-counts but stays in the family
