"""Config-shape hazards that make a model fail to size at all.

Two recurring shapes, both of which produced a hard TypeError rather than a
wrong number -- so the model returned nothing instead of something checkable:

* **per-layer schedules** -- a key the counters read as an integer, shipped as a
  list with one entry per layer (Hunyuan-A13B's `moe_intermediate_size`,
  `moe_topk`, `num_shared_expert`).
* **null vs absent** -- `cfg.get(k, default)` returns a stored ``None``, so a key
  present with an explicit JSON null bypasses its default. Models that borrowed
  DeepSeek's config schema without using MLA ship `kv_lora_rank: null`.
"""

import json
from pathlib import Path

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.config_normalizer import ConfigNormalizer

FIXTURES = Path(__file__).parent / "fixtures" / "hybrid_configs"


def cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ------------------------------------------------- per-layer schedule lists


def test_hunyuan_per_layer_lists_collapse_to_scalars():
    c = cfg("tencent_Hunyuan-A13B-Instruct")
    depth = c["num_hidden_layers"]
    assert isinstance(c["moe_intermediate_size"], list) and len(c["moe_intermediate_size"]) == depth
    assert isinstance(c["moe_topk"], list) and len(c["moe_topk"]) == depth

    n = ConfigNormalizer.normalize_config(c)
    assert n["moe_intermediate_size"] == 3072
    assert n["moe_topk"] == 8
    assert n["num_shared_expert"] == 1
    # The collapse is recorded rather than done silently.
    assert set(n["_collapsed_per_layer"]) == {
        "moe_intermediate_size", "moe_topk", "num_shared_expert"
    }


def test_only_lists_matching_model_depth_are_collapsed():
    """Length is the discriminator between a schedule and something else.

    `attn_layer_indices` (3 entries naming which layers hold attention) and
    `time_step_limit` (a 2-element min/max pair) are numeric lists that would be
    destroyed by averaging.
    """
    c = cfg("ibm-ai-platform_Bamba-9B-v2")
    n = ConfigNormalizer.normalize_config(c)
    assert n["attn_layer_indices"] == [9, 18, 27]
    assert "_collapsed_per_layer" not in n

    c2 = cfg("mistralai_Mamba-Codestral-7B-v0.1")
    assert ConfigNormalizer.normalize_config(c2)["time_step_limit"] == c2["time_step_limit"]


def test_width_uses_mean_and_topk_uses_max():
    """Aggregation is chosen per key, not uniformly.

    A width schedule is summed over layers downstream, so mean x depth
    reproduces the true total exactly. Top-k drives an activation peak, which is
    set by the worst layer.
    """
    base = dict(num_hidden_layers=4, hidden_size=512)
    n = ConfigNormalizer.normalize_config(
        dict(base, moe_intermediate_size=[100, 200, 300, 400], moe_topk=[1, 2, 8, 2])
    )
    assert n["moe_intermediate_size"] == 250  # mean; 250*4 == 100+200+300+400
    assert n["moe_topk"] == 8                 # max, not mean


def test_collapsed_value_does_not_reappear_through_its_alias():
    """The alias pass must read the normalized dict, not the original.

    Hunyuan's `moe_topk` list was collapsed, then copied back verbatim from the
    raw config under the canonical name `expert_top_k`, and crashed downstream.
    """
    n = ConfigNormalizer.normalize_config(cfg("tencent_Hunyuan-A13B-Instruct"))
    assert n["expert_top_k"] == 8
    assert not isinstance(n["expert_top_k"], list)


def test_hunyuan_sizes_end_to_end():
    """Real ~80B, GQA with 32 q heads and 8 kv heads at head_dim 128."""
    c = cfg("tencent_Hunyuan-A13B-Instruct")
    r = calculate_memory(c, batch_size=1, seq_length=4096, precision="bf16")
    assert abs(r.parameter_count / 1e9 - 80) / 80 < 0.05
    expected_kv = 2 * 8 * 128 * 2 * 32 * 4096 / 1e9
    assert r.kv_cache_gb == pytest.approx(expected_kv, rel=1e-9)


# ------------------------------------------------------- null vs absent MLA


def test_null_mla_keys_do_not_make_a_gqa_model_look_like_mla():
    """Hunyuan declares kv_lora_rank: null and use_mla: false, and is GQA.

    Testing `key in config` labelled it MLA; the compressed-dim arithmetic then
    raised NoneType + NoneType and the model could not be sized at all.
    """
    c = cfg("tencent_Hunyuan-A13B-Instruct")
    assert c.get("use_mla") is False
    assert c.get("kv_lora_rank") is None and "kv_lora_rank" in c
    assert ModelMemoryCalculator().detect_attention_type(c) == "gqa"


def test_genuine_mla_models_are_still_detected():
    """The guard must not cost real MLA models their classification.

    Both of these ship `q_lora_rank: null` -- DeepSeek-V2-Lite and Kimi-Linear
    use MLA without the Q low-rank factorization -- so a naive "all keys must be
    non-null" rule would misfire. kv_lora_rank is the load-bearing one.
    """
    calc = ModelMemoryCalculator()
    kimi = cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct")
    assert kimi["q_lora_rank"] is None and kimi["kv_lora_rank"] == 512
    assert calc.detect_attention_type(kimi) == "mla"


def test_mla_kv_math_survives_null_ranks():
    """Defense in depth: even if something reaches the MLA path with nulls."""
    calc = ModelMemoryCalculator()
    c = {
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "kv_lora_rank": None,
        "qk_rope_head_dim": None,
    }
    assert calc._calculate_kv_cache_mla(c, 1, 128, 2) > 0
