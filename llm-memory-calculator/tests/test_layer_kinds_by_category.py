"""Which layers hold a KV cache, how big it is, and how it grows -- one rule per category.

The over-count this guards against: a layer was charged a full, context-growing KV cache
whenever its name contained "attention", or whenever the reader in front of the calculator
dropped the key that said otherwise. Gated DeltaNet's "linear_attention" layers, a GGUF's
`full_attention_interval`, and every hybrid built into the GenZ performance graph all paid
for a cache they never allocate -- Qwen3.5 four times over.

Every category is read from the model's own config keys through ONE table
(`layer_plan.LAYER_KINDS` / `layer_plan.layer_kinds`), and every caller -- the memory
calculator, the GGUF reader, the GenZ graph, the serving memory model, the parameter
counter, the FRD-027 recogniser -- answers from it. Real shipped configs are used where
they exist (tests/fixtures); the rest are minimal configs carrying exactly the keys the
transformers config class of that family writes (transformers 5.12).
"""

import json
import os
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.layer_plan import resolve_layer_plan

# Mechanism names, by state geometry (layer_plan.MECH_*). Spelled out so this module also
# runs -- and fails, case by case -- against the code it was written to fix.
MECH_GDN, MECH_LIGHTNING, MECH_MAMBA2 = "gdn", "lightning", "mamba2"
MECH_MLSTM, MECH_RGLRU, MECH_RWKV = "mlstm", "rglru", "rwkv"


def hf_layer_keys_from_gguf(get, n):
    from llm_memory_calculator.gguf_layers import hf_layer_keys_from_gguf as f

    return f(get, n)


def layer_kinds(config):
    from llm_memory_calculator.layer_plan import layer_kinds as f

    return f(config)


def _layer_kinds_table():
    from llm_memory_calculator.layer_plan import LAYER_KINDS

    return LAYER_KINDS

FIXTURES = Path(__file__).parent / "fixtures"
HYBRID = FIXTURES / "hybrid_configs"
MiB = 1024 * 1024


def cfg(name):
    return json.loads((HYBRID / f"{name}.json").read_text())


def kv_bytes(config, seq, batch=1, engine=None):
    calc = ModelMemoryCalculator()
    calc.model_type = calc.detect_model_type(config)
    return calc.calculate_kv_cache(config, batch, seq, "bf16", engine_capabilities=engine) * 1e9


def state_bytes(config, batch=1):
    calc = ModelMemoryCalculator()
    calc.model_type = calc.detect_model_type(config)
    return calc.calculate_state_memory(config, batch, "bf16") * 1e9


def per_layer_token(heads_kv, head_dim, nbytes=2):
    return 2 * heads_kv * head_dim * nbytes


# ------------------------------------------------- the measured case: Qwen3.5 on the GB10


def test_qwen35_kv_is_its_eight_full_attention_layers_as_measured():
    """Qwen3.5-4B (the shipped config, layer kinds nested under text_config).

    Measured on the GB10, 27 Oct 2026: llama.cpp and mistral.rs both hold a growing paged
    pool only on the 8 `full_attention` layers -- 129 MiB at 4,096 tokens (mistral.rs:
    "Allocating 129 MB for PagedAttention"). The 24 `linear_attention` (Gated DeltaNet)
    layers hold a fixed state instead. The simulator charged all 32 layers before.
    """
    c = cfg("Qwen_Qwen3.5-4B")
    expected = 8 * per_layer_token(4, 256) * 4096
    assert expected == 128 * MiB
    assert kv_bytes(c, 4096) == expected
    # within the page rounding of the server's own line
    assert abs(kv_bytes(c, 4096) - 129 * MiB) / (129 * MiB) < 0.01
    bd = ModelMemoryCalculator().kv_cache_breakdown(c, 1, 4096, "bf16")
    assert (bd["num_full_layers"], bd["num_sliding_layers"], bd["num_no_kv_layers"]) == (8, 0, 24)


def test_qwen35_recurrent_state_is_fixed_and_sized_from_the_config():
    c = cfg("Qwen_Qwen3.5-4B")
    tc = c["text_config"]
    nv, kd, vd = tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    nk, k = tc["linear_num_key_heads"], tc["linear_conv_kernel_dim"]
    per_layer = nv * kd * vd * 4 + (2 * nk * kd + nv * vd) * (k - 1) * 2
    assert state_bytes(c) == pytest.approx(24 * per_layer)
    # constant in the context: there is no sequence term at all
    assert kv_bytes(c, 32768) == 8 * kv_bytes(c, 4096)


def test_the_genz_graph_charges_kv_only_on_the_attention_layers():
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    mc = get_configs(cfg("Qwen_Qwen3.5-4B"))
    kinds = [lc.attention_type for lc in mc.layer_configs]
    assert kinds.count("MHA-global") == 8 and kinds.count("Recurrent") == 24
    assert mc.get_kv_size() == 8 * 2 * 4 * 256  # elements per token


def test_the_genz_decode_graph_holds_128_mib_of_kv_plus_the_fixed_state(tmp_path):
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        get_configs,
    )
    from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

    (tmp_path / "model").mkdir()
    (tmp_path / "sparsity").mkdir()

    def kv_mb(seq):
        mc = get_configs(cfg("Qwen_Qwen3.5-4B"))
        path = create_full_decode_model(
            name=mc, input_sequence_length=seq, output_gen_tokens=0, data_path=str(tmp_path)
        )
        df = get_model_df(path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
        return float(get_summary_table(df, model_characterstics=True)["KV Cache (MB)"].values[0])

    # GenZ's "KV Cache" column carries the recurrent state too (as it always has for Mamba);
    # the growth from 4,096 to 8,192 tokens is the attention KV alone: 8 layers, not 32.
    growth = kv_mb(8192) - kv_mb(4096)
    assert growth == pytest.approx(128.0, rel=1e-6)


def test_bud_component_reads_the_same_layers_from_the_gguf():
    """The GGUF reader used to drop `qwen35.full_attention_interval` and every `ssm.*` key,
    so the calculator saw a dense 24-layer stack: 4x the KV llama.cpp allocates."""
    import sys

    path = "/home/bud/models/qwen3.5-2b-gguf/Qwen3.5-2B-Q4_K_M.gguf"
    if not os.path.exists(path):
        pytest.skip("Qwen3.5-2B GGUF not on this machine")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        import bud_component
    finally:
        sys.path.pop(0)
    c, _, _ = bud_component.gguf_to_hf_config(path)
    # 24 layers, one full every 4: 6 full, 18 recurrent; 2 KV heads of 256
    assert kv_bytes(c, 4096) == 6 * per_layer_token(2, 256) * 4096 == 48 * MiB
    # llama.cpp's own recurrent sizing: (d_conv-1)(d_inner + 2 n_group d_state) conv
    # elements plus d_state * d_inner state elements per layer.
    conv = 3 * (2048 + 2 * 16 * 128)
    state = 128 * 2048
    assert state_bytes(c) == pytest.approx(18 * (state * 4 + conv * 2))


# ------------------------------------------------- GGUF keys -> the same plan as the checkpoint


def _gguf(meta):
    return lambda suffix: meta.get(suffix)


def qwen35_4b_gguf_meta():
    """What llama.cpp's converter writes for Qwen3.5-4B (convert_hf_to_gguf, Qwen3Next):
    ssm.state_size = linear_key_head_dim, ssm.group_count = linear_num_key_heads,
    ssm.time_step_rank = linear_num_value_heads, ssm.inner_size = value_dim * value_heads."""
    return {
        "attention.head_count": 16,
        "attention.head_count_kv": 4,
        "full_attention_interval": 4,
        "ssm.conv_kernel": 4,
        "ssm.state_size": 128,
        "ssm.group_count": 16,
        "ssm.time_step_rank": 32,
        "ssm.inner_size": 128 * 32,
    }


def test_a_gguf_and_its_checkpoint_resolve_to_the_same_kv_and_state():
    hf = cfg("Qwen_Qwen3.5-4B")
    tc = hf["text_config"]
    g = dict(
        model_type="qwen35", num_hidden_layers=32, hidden_size=tc["hidden_size"],
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
    )
    g.update(hf_layer_keys_from_gguf(_gguf(qwen35_4b_gguf_meta()), 32))
    assert kv_bytes(g, 4096) == kv_bytes(hf, 4096)
    assert state_bytes(g) == pytest.approx(state_bytes(hf))


@pytest.mark.parametrize(
    "meta,n,expect",
    [
        # Jamba-style: per-layer KV heads, 0 on the Mamba layers; every layer has an FFN
        (
            {"attention.head_count": 32, "attention.head_count_kv": [0, 0, 0, 0, 8, 0, 0, 0],
             "feed_forward_length": [14336] * 8, "ssm.state_size": 16, "ssm.conv_kernel": 4,
             "ssm.inner_size": 8192},
            8, {"full": 1, "recurrent": 7, "ffn_only": 0},
        ),
        # Nemotron-H-style: FFN gets its own slot (KV 0, FFN > 0); mixers have FFN 0
        (
            {"attention.head_count": 32, "attention.head_count_kv": [0, 0, 8, 0, 0, 0],
             "feed_forward_length": [0, 21504, 0, 21504, 0, 21504], "ssm.state_size": 128,
             "ssm.conv_kernel": 4, "ssm.inner_size": 8192, "ssm.group_count": 8},
            6, {"full": 1, "recurrent": 2, "ffn_only": 3},
        ),
        # LFM2-style: short-conv layers state their cache length
        (
            {"attention.head_count": 32, "attention.head_count_kv": [0, 0, 8, 0],
             "shortconv.l_cache": 3},
            4, {"full": 1, "recurrent": 3, "ffn_only": 0},
        ),
        # Falcon-H1-style: attention AND a Mamba mixer in every layer
        (
            {"attention.head_count": 32, "attention.head_count_kv": 8, "ssm.state_size": 256,
             "ssm.conv_kernel": 4, "ssm.inner_size": 8192, "ssm.group_count": 1},
            4, {"full": 4, "recurrent": 4, "ffn_only": 0},
        ),
    ],
)
def test_gguf_per_layer_keys_resolve_each_hybrid_layout(meta, n, expect):
    keys = hf_layer_keys_from_gguf(_gguf(meta), n)
    kinds = layer_kinds(dict(num_hidden_layers=n, hidden_size=4096, **keys))
    full = sum(1 for a in kinds["attn"] if a is not None)
    rec = sum(1 for r in kinds["recurrent"] if r)
    ffn_only = sum(1 for a, r in zip(kinds["attn"], kinds["recurrent"]) if a is None and not r)
    assert {"full": full, "recurrent": rec, "ffn_only": ffn_only} == expect


def test_a_gguf_window_without_a_pattern_is_not_applied_to_every_layer():
    """gpt-oss GGUFs state `attention.sliding_window: 128` but which layers slide is set in
    llama.cpp's code. Applying the window to every layer would under-count the full layers."""
    keys = hf_layer_keys_from_gguf(
        _gguf({"attention.head_count": 64, "attention.head_count_kv": 8,
               "attention.sliding_window": 128}), 24)
    assert keys == {}


def test_a_gguf_window_pattern_marks_the_windowed_layers():
    keys = hf_layer_keys_from_gguf(
        _gguf({"attention.head_count": 8, "attention.head_count_kv": 4,
               "attention.sliding_window": 1024, "attention.sliding_window_pattern": 6}), 12)
    assert keys["layer_types"].count("full_attention") == 2
    assert keys["layer_types"][5] == keys["layer_types"][11] == "full_attention"
    assert keys["sliding_window"] == 1024


def test_a_dense_gguf_adds_nothing():
    assert hf_layer_keys_from_gguf(
        _gguf({"attention.head_count": 32, "attention.head_count_kv": 8}), 32) == {}


# ------------------------------------------------- one rule per category, from real configs


@pytest.mark.parametrize(
    "fixture,kv_layers,state",
    [
        ("Qwen_Qwen3-Next-80B-A3B-Instruct", 12, True),  # full_attention_interval
        ("ai21labs_Jamba-v0.1", 4, True),  # attn_layer_period / offset
        ("ibm-granite_granite-4.0-h-small", 4, True),  # layer_types mamba/attention
        ("nvidia_Nemotron-H-8B-Base-8K", 4, True),  # hybrid_override_pattern
        ("Zyphra_Zamba2-2.7B", 9, True),  # layers_block_type with 'hybrid'
        ("tiiuae_Falcon-H1-34B-Instruct", 72, True),  # parallel: every layer both
        ("mistralai_Mamba-Codestral-7B-v0.1", 0, True),  # pure SSM
    ],
)
def test_recurrent_categories_hold_kv_only_on_attention_layers(fixture, kv_layers, state):
    c = cfg(fixture)
    bd = ModelMemoryCalculator().kv_cache_breakdown(c, 1, 4096, "bf16")
    assert bd["num_full_layers"] + bd["num_sliding_layers"] == kv_layers
    assert (state_bytes(c) > 0) == state
    if kv_layers:
        assert kv_bytes(c, 8192) == pytest.approx(2 * kv_bytes(c, 4096))


def test_gemma3_interleave_is_read_from_its_window_pattern():
    """Gemma 3 states `sliding_window_pattern: 6`, no list: one full layer in six."""
    c = cfg("unsloth_gemma-3-27b-it")
    kinds = layer_kinds(c)
    assert kinds["attn"].count("full") == 10 and kinds["attn"].count("sliding") == 52
    tc = c["text_config"]
    per = per_layer_token(tc["num_key_value_heads"], tc["head_dim"])
    window = tc["sliding_window"]
    # an engine with a per-layer KV layout takes the window on the 52 local layers
    assert kv_bytes(c, 32768, engine={"heterogeneous_kv_layout": True}) == per * (
        10 * 32768 + 52 * window
    )


def test_a_uniformly_windowed_stack_is_bounded_by_its_window():
    """Mistral-7B v0.1: `sliding_window: 4096` and no per-layer list -- every layer slides."""
    c = dict(model_type="mistral", hidden_size=4096, num_hidden_layers=32,
             num_attention_heads=32, num_key_value_heads=8, sliding_window=4096)
    assert kv_bytes(c, 32768) == 32 * per_layer_token(8, 128) * 4096


def test_mla_holds_one_latent_per_layer():
    c = json.loads((FIXTURES / "deepseek_v2_lite_chat_config.json").read_text())
    latent = c["kv_lora_rank"] + c["qk_rope_head_dim"]
    assert kv_bytes(c, 4096) == c["num_hidden_layers"] * latent * 2 * 4096


def test_a_dense_transformer_holds_kv_on_every_layer():
    c = json.loads((FIXTURES / "qwen2_5_0_5b_config.json").read_text())
    assert resolve_layer_plan(c) is None
    # `use_sliding_window: false` -- its stated window is not applied
    assert kv_bytes(c, 32768) == 24 * per_layer_token(2, 64) * 32768


def test_kv_shared_layers_hold_no_cache_of_their_own():
    """Gemma 3n: the last `num_kv_shared_layers` layers reuse an earlier layer's KV."""
    pattern = ["sliding_attention"] * 4 + ["full_attention"]
    c = dict(model_type="gemma3n_text", hidden_size=2048, num_hidden_layers=35,
             num_attention_heads=8, num_key_value_heads=2, head_dim=256,
             sliding_window=512, layer_types=pattern * 7, num_kv_shared_layers=15)
    plan = resolve_layer_plan(c)
    assert plan["num_kv_shared_layers"] == 15
    assert (plan["num_full_layers"], plan["num_sliding_layers"]) == (4, 16)
    assert plan["num_attention_layers"] == 35  # they still attend
    assert kv_bytes(c, 4096) == 20 * per_layer_token(2, 256) * 4096


def test_chunked_attention_is_bounded_by_its_chunk():
    """Llama 4: `chunked_attention` layers keep at most `attention_chunk_size` tokens."""
    names = ["chunked_attention"] * 3 + ["full_attention"]
    c = dict(model_type="llama4_text", hidden_size=5120, num_hidden_layers=48,
             num_attention_heads=40, num_key_value_heads=8, head_dim=128,
             attention_chunk_size=8192, layer_types=names * 12)
    per = per_layer_token(8, 128)
    hybrid = {"heterogeneous_kv_layout": True}
    assert kv_bytes(c, 32768, engine=hybrid) == per * (12 * 32768 + 36 * 8192)
    # an engine that cannot hold a per-layer layout pays full on every layer
    assert kv_bytes(c, 32768) == per * 48 * 32768


def test_cross_attention_layers_hold_the_encoder_tokens_not_the_text():
    """Mllama: `cross_attention_layers` attend to the vision encoder's output, a fixed
    count per request: max_num_tiles x ((image_size / patch_size)^2 + 1) = 4 x 1601."""
    c = {
        "model_type": "mllama",
        "vision_config": {"image_size": 560, "patch_size": 14, "max_num_tiles": 4},
        "text_config": dict(model_type="mllama_text_model", hidden_size=4096,
                            num_hidden_layers=40, num_attention_heads=32,
                            num_key_value_heads=8, cross_attention_layers=[3, 8, 13, 18, 23,
                                                                           28, 33, 38]),
    }
    per = per_layer_token(8, 128)
    assert kv_bytes(c, 32768) == per * (32 * 32768 + 8 * 6404)


@pytest.mark.parametrize(
    "config,mech",
    [
        # transformers RwkvConfig: no attention heads, an attention_hidden_size
        (dict(model_type="rwkv", hidden_size=2048, num_hidden_layers=24,
              attention_hidden_size=2048, intermediate_size=8192, vocab_size=50277), MECH_RWKV),
        # transformers xLSTMConfig: matrix memory sized by num_heads and the dim factors
        (dict(model_type="xlstm", hidden_size=4096, num_hidden_layers=32, num_heads=8,
              qk_dim_factor=0.5, v_dim_factor=1.0, vocab_size=50304), MECH_MLSTM),
    ],
)
def test_attention_free_recurrent_stacks_hold_no_kv(config, mech):
    plan = resolve_layer_plan(config)
    assert plan["mechanisms"] == [mech]
    assert kv_bytes(config, 4096) == 0
    assert state_bytes(config) > 0


def test_recurrentgemma_block_types_are_read():
    c = dict(model_type="recurrent_gemma", hidden_size=2560, num_hidden_layers=26,
             num_attention_heads=10, num_key_value_heads=1, head_dim=256, lru_width=2560,
             attention_window_size=2048, conv1d_width=4,
             block_types=["recurrent", "recurrent", "attention"])
    plan = resolve_layer_plan(c)
    assert plan["num_recurrent_layers"] == 18 and plan["num_sliding_layers"] == 8
    assert plan["mechanisms"] == [MECH_RGLRU]
    assert kv_bytes(c, 32768) == 8 * per_layer_token(1, 256) * 2048


def test_linear_attention_is_gdn_or_plain_linear_by_the_config_keys():
    base = dict(hidden_size=4096, num_hidden_layers=4, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128,
                layer_types=["linear_attention"] * 3 + ["full_attention"])
    assert resolve_layer_plan(base)["mechanisms"] == [MECH_LIGHTNING]
    gdn = dict(base, linear_num_value_heads=32, linear_num_key_heads=16,
               linear_key_head_dim=128, linear_value_head_dim=128)
    assert resolve_layer_plan(gdn)["mechanisms"] == [MECH_GDN]
    mamba = dict(base, layer_types=["mamba"] * 3 + ["attention"], mamba_d_state=128)
    assert resolve_layer_plan(mamba)["mechanisms"] == [MECH_MAMBA2]


# ------------------------------------------------- unknown kinds: loud and conservative


def test_an_unknown_layer_kind_is_charged_full_kv_and_said_so():
    c = dict(model_type="futurenet", hidden_size=4096, num_hidden_layers=4,
             num_attention_heads=32, num_key_value_heads=8,
             layer_types=["warp_attention", "full_attention"] * 2)
    with pytest.warns(UserWarning, match="warp_attention"):
        plan = resolve_layer_plan(c)
    assert plan["approximate"] is True
    assert plan["num_full_layers"] == 4
    assert any("warp_attention" in n for n in plan["notes"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kv_bytes(c, 4096) == 4 * per_layer_token(8, 128) * 4096


def test_the_table_covers_every_layer_kind_transformers_allows():
    allowed = {
        "full_attention", "sliding_attention", "chunked_attention",
        "compressed_sparse_attention", "heavily_compressed_attention", "minimax_m3_sparse",
        "linear_attention", "conv", "mamba", "attention", "sparse", "dense", "hybrid", "moe",
        "deepseek_sparse_attention",
    }
    try:  # the installed transformers, when there is one, is the reference
        from transformers.configuration_utils import ALLOWED_LAYER_TYPES

        allowed |= set(ALLOWED_LAYER_TYPES)
    except Exception:
        pass
    assert allowed <= set(_layer_kinds_table())


def test_no_name_containing_attention_is_counted_by_substring():
    """The substring test is gone from every caller: a hybrid without a resolvable plan is
    read through the table (and an unstated layout is charged every layer, not half)."""
    calc = ModelMemoryCalculator()
    c = dict(model_type="jamba", hidden_size=4096, num_hidden_layers=8,
             num_attention_heads=32, num_key_value_heads=8)
    calc.model_type = "hybrid"
    assert calc._count_attention_layers(c, 8) == 8
    c2 = dict(c, attention_ratio=0.25)
    assert calc._count_attention_layers(c2, 8) == 2


def test_the_recogniser_and_the_table_are_one_vocabulary():
    from llm_memory_calculator.model.recognise import KNOWN_LAYER_TYPES

    assert KNOWN_LAYER_TYPES == frozenset(_layer_kinds_table())


# ------------------------------------------------- every caller answers from the table


def test_the_serving_memory_model_charges_only_attention_layers():
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig
    from llm_memory_calculator.genz.serving.constants import MemoryTier

    mc = get_configs(cfg("Qwen_Qwen3.5-4B"))
    mm = MemoryModel(mc, [MemoryTierConfig(tier=MemoryTier.DEVICE_HBM, capacity_bytes=10**10,
                                           bandwidth_gbps=100.0)], precision_bytes=2)
    assert mm.bytes_per_token_kv == 8 * per_layer_token(4, 256)


def test_engine_memory_counts_the_recurrent_state_in_the_total():
    from llm_memory_calculator.inference_engines import estimate_memory_for_engine

    c = cfg("Qwen_Qwen3.5-4B")
    em = estimate_memory_for_engine(c, "llamacpp", batch_size=2, seq_length=4096)
    assert em.kv_cache_bytes == 2 * 128 * MiB
    assert em.state_bytes == pytest.approx(state_bytes(c, batch=2))
    assert em.total_bytes == pytest.approx(
        em.weights_bytes + em.kv_cache_bytes + em.state_bytes + em.activation_bytes
    )


def test_a_hybrid_parameter_count_does_not_treat_linear_attention_as_attention():
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    pc = UniversalParameterCounter()
    c = dict(model_type="hybrid", hidden_size=1024, num_hidden_layers=4,
             num_attention_heads=8, num_key_value_heads=8, vocab_size=1000,
             intermediate_size=4096)
    linear = dict(c, layer_types=["linear_attention", "full_attention"] * 2)
    attention = dict(c, layer_types=["full_attention"] * 4)
    assert pc.count_parameters(linear) != pc.count_parameters(attention)
