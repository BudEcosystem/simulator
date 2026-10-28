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


def needs_transformers_layouts():
    """A case that reads a layout from a transformers config class. transformers is optional
    for the calculator, and only 4.53+ derives `layer_types` (without it a window is charged on
    no layer, which is what the case does not test)."""
    transformers = pytest.importorskip("transformers")
    from llm_memory_calculator.layer_plan import transformers_states_layer_layouts

    if not transformers_states_layer_layouts():
        pytest.skip(f"transformers {transformers.__version__} derives no layer_types")


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
    pytest.importorskip("gguf")
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
    needs_transformers_layouts()
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


# ------------------------------------------------- review round 1: the paths and shapes still wrong


def _genz_kv_mb(builder, tmp_path, **kwargs):
    from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir(exist_ok=True)
    path = builder(data_path=str(tmp_path), **kwargs)
    df = get_model_df(path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
    return float(get_summary_table(df, model_characterstics=True)["KV Cache (MB)"].values[0])


def _chunked_kv_mb(config, tmp_path, decode_kv):
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_chunked_model,
        get_configs,
    )

    return _genz_kv_mb(
        create_full_chunked_model, tmp_path, name=get_configs(config),
        prefill_kv_sizes=[(0, 512)], decode_kv_sizes=[decode_kv],
    )


def test_the_chunked_graph_charges_kv_only_on_the_attention_layers(tmp_path):
    """`create_full_chunked_model` (chunked serving, speculative decoding, the simulation
    engine's chunked feature) built every layer as attention: Qwen3.5-4B paid 32 layers of KV.
    The growth from 4,096 to 8,192 cached tokens is 8 full-attention layers, 128 MiB."""
    c = cfg("Qwen_Qwen3.5-4B")
    growth = _chunked_kv_mb(c, tmp_path, 8192) - _chunked_kv_mb(c, tmp_path, 4096)
    assert growth == pytest.approx(128.0, rel=1e-6)


def test_the_chunked_graph_bounds_windowed_layers_by_their_window(tmp_path):
    """gpt-oss-120b: 18 full + 18 sliding (window 128). Past the window only the full layers grow."""
    c = cfg("openai_gpt-oss-120b")
    per_token_mib = per_layer_token(c["num_key_value_heads"], c["head_dim"]) / MiB
    growth = _chunked_kv_mb(c, tmp_path, 8192) - _chunked_kv_mb(c, tmp_path, 4096)
    assert growth == pytest.approx(18 * per_token_mib * 4096, rel=1e-6)


def test_the_genz_recurrent_state_is_priced_like_the_calculator(tmp_path):
    """The recurrent matrix is held in fp32 (as vLLM and llama.cpp hold it); the graph priced
    it at the activation precision, half the calculator's figure for the same config."""
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        get_configs,
    )

    c = cfg("Qwen_Qwen3.5-4B")
    kv = _genz_kv_mb(create_full_decode_model, tmp_path, name=get_configs(c),
                     input_sequence_length=4096, output_gen_tokens=0)
    assert kv == pytest.approx(128.0 + state_bytes(c) / MiB, rel=1e-6)


def test_the_genz_graph_places_moe_only_on_the_layers_the_config_says():
    """Jamba: `expert_layer_period: 2, expert_layer_offset: 1` -- half its FFNs are dense.
    Kimi Linear / DeepSeek: the first `first_k_dense_replace` layers are dense."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    jamba = get_configs(cfg("ai21labs_Jamba-v0.1"))
    ffn = [lc.ffn_type for lc in jamba.layer_configs]
    assert ffn.count("MoE") == 16 and ffn.count("Dense") == 16
    assert all(f == ("MoE" if i % 2 == 1 else "Dense") for i, f in enumerate(ffn))
    # a dense layer of an MoE model is graphed as one FFN, not as the experts
    assert jamba.get_layer_view(0).num_experts == 1
    assert jamba.get_layer_view(1).num_experts == 16

    kimi = get_configs(cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct"))
    ffn = [lc.ffn_type for lc in kimi.layer_configs]
    assert ffn[0] == "Dense" and set(ffn[1:]) == {"MoE"}


def test_the_genz_jamba_weights_match_the_parameter_count(tmp_path):
    from llm_memory_calculator import calculate_memory
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        get_configs,
    )
    from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

    c = cfg("ai21labs_Jamba-v0.1")
    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir(exist_ok=True)
    path = create_full_decode_model(name=get_configs(c), input_sequence_length=1024,
                                    output_gen_tokens=0, data_path=str(tmp_path))
    df = get_model_df(path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
    weights_mb = float(get_summary_table(df, model_characterstics=True)["Total Weights (MB)"].values[0])
    params = calculate_memory(c, batch_size=1, seq_length=1024).parameter_count
    # bf16 weights; the graph omits norms and small vectors, so allow a few percent
    assert weights_mb * MiB / 2 == pytest.approx(params, rel=0.05)


def test_a_window_without_a_stated_layout_follows_the_config_class():
    """Gemma 2's hub config.json carries `sliding_window` and no `layer_types`; its config
    class (transformers) alternates sliding and full. Clamping every layer to the window
    under-counted KV 8x at 32k. Mistral's class states no per-layer list: every layer slides."""
    needs_transformers_layouts()
    gemma2 = dict(model_type="gemma2", hidden_size=2304, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=256,
                  sliding_window=4096, vocab_size=1000, intermediate_size=9216)
    kinds = layer_kinds(gemma2)
    assert kinds["attn"].count("full") == 4 and kinds["attn"].count("sliding") == 4
    per = per_layer_token(4, 256)
    assert kv_bytes(gemma2, 32768, engine={"heterogeneous_kv_layout": True}) == pytest.approx(per * (
        4 * 32768 + 4 * 4096
    ))
    mistral = dict(model_type="mistral", hidden_size=4096, num_hidden_layers=32,
                   num_attention_heads=32, num_key_value_heads=8, sliding_window=4096)
    assert layer_kinds(mistral)["attn"] == ["sliding"] * 32


def test_a_window_on_an_unknown_model_type_is_charged_in_full_and_said_so():
    c = dict(model_type="futurenet", hidden_size=4096, num_hidden_layers=4,
             num_attention_heads=32, num_key_value_heads=8, sliding_window=512)
    with pytest.warns(UserWarning, match="window"):
        kinds = layer_kinds(c)
    assert kinds["attn"] == ["full"] * 4
    assert any("window" in n for n in kinds["notes"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kv_bytes(c, 4096) == 4 * per_layer_token(8, 128) * 4096


def test_depth_comes_from_the_per_layer_list_when_the_config_omits_it():
    """transformers 5.12's NemotronHConfig deprecates `num_hidden_layers` for
    `layers_block_type`: mamba, moe, attention, mlp -- one attention layer, one Mamba-2."""
    c = dict(model_type="nemotron_h", hidden_size=4096, num_attention_heads=32,
             num_key_value_heads=8, head_dim=128, mamba_num_heads=128, mamba_head_dim=64,
             ssm_state_size=128, n_groups=8, conv_kernel=4, vocab_size=1000,
             layers_block_type=["mamba", "moe", "attention", "mlp"])
    kinds = layer_kinds(c)
    assert kinds["attn"] == [None, None, "full", None]
    assert kinds["recurrent"] == [MECH_MAMBA2, None, None, None]
    assert kinds["ffn"] == [False, True, False, True]
    assert kv_bytes(c, 32768) == per_layer_token(8, 128) * 32768


def test_a_nemotron_pattern_reads_its_moe_slots():
    """Nemotron 3's pattern adds `E` (an MoE FFN slot) to M / * / -."""
    c = dict(model_type="nemotron_h", hidden_size=2688, num_hidden_layers=6,
             num_attention_heads=32, num_key_value_heads=2, head_dim=128,
             ssm_state_size=128, n_groups=8, hybrid_override_pattern="MEM*EM")
    kinds = layer_kinds(c)
    assert kinds["ffn"] == [False, True, False, False, True, False]
    assert sum(1 for a in kinds["attn"] if a) == 1


def test_a_mamba1_gguf_is_sized_as_mamba1():
    """A Mamba-1 GGUF states no `ssm.group_count`: llama.cpp's Mamba-1 has no B/C groups."""
    meta = {"ssm.state_size": 16, "ssm.conv_kernel": 4, "ssm.inner_size": 8192,
            "ssm.time_step_rank": 256}
    keys = hf_layer_keys_from_gguf(_gguf(meta), 64)
    kinds = layer_kinds(dict(model_type="falcon_mamba", num_hidden_layers=64,
                             hidden_size=4096, **keys))
    assert kinds["mechanisms"] == ["mamba1"]


def test_the_qwen35_gguf_parameter_count_matches_its_tensors():
    """The GGUF reader stated no RoPE (so the counter added 537M learned position
    embeddings) and no gated FFN; the file's own tensors are the truth."""
    import sys

    path = "/home/bud/models/qwen3.5-2b-gguf/Qwen3.5-2B-Q4_K_M.gguf"
    if not os.path.exists(path):
        pytest.skip("Qwen3.5-2B GGUF not on this machine")
    gguf = pytest.importorskip("gguf")
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        import bud_component
    finally:
        sys.path.pop(0)
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    c, _, _ = bud_component.gguf_to_hf_config(path)
    truth = sum(int(np.prod(t.shape)) for t in gguf.GGUFReader(path).tensors)
    assert UniversalParameterCounter().count_parameters(c) == pytest.approx(truth, rel=0.03)


def test_sambay_kv_is_never_below_its_one_shared_full_cache():
    """Phi-4-mini-flash (SambaY): its YOCO cross-decoder reuses ONE full-attention cache that
    grows with the context. Charging every layer a 512-token window under-counted it at long
    context; the bound charges that cache in full and never less than the published design
    (8 windowed layers + 1 full) needs."""
    c = cfg("microsoft_Phi-4-mini-flash-reasoning")
    per = per_layer_token(c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"])
    for seq in (4096, 32768, 131072):
        kv = kv_bytes(c, seq, engine={"heterogeneous_kv_layout": True})
        assert kv >= per * (8 * 512 + seq)
        assert kv < per * 32 * seq


def test_a_dense_layer_of_an_moe_model_graphs_under_expert_parallelism(tmp_path):
    """Jamba with expert_parallel=2: its dense layers have no experts to shard."""
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_chunked_model,
        create_full_decode_model,
        create_full_prefill_model,
        get_configs,
    )

    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir(exist_ok=True)
    mc = get_configs(cfg("ai21labs_Jamba-v0.1"))
    create_full_prefill_model(name=mc, input_sequence_length=256, data_path=str(tmp_path),
                              expert_parallel=2)
    create_full_decode_model(name=mc, input_sequence_length=256, data_path=str(tmp_path),
                             expert_parallel=2)
    create_full_chunked_model(name=mc, prefill_kv_sizes=[(0, 128)], decode_kv_sizes=[256],
                              data_path=str(tmp_path), expert_parallel=2)


# ------------------------------------------------- review round 2: what still erred low or by name


def test_zamba2_runs_its_shared_mlp_on_every_hybrid_layer():
    """Zamba2's shared attention+MLP block runs beside the Mamba mixer on each `hybrid` layer.
    The plan's FFN axis is what runs; the shared weights are counted once per block
    (`num_mem_blocks`) by the parameter count alone. Graphing no FFN at all under-counted
    ~0.7B params of weight traffic per decoded token."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    c = cfg("Zyphra_Zamba2-2.7B")
    hybrid = [t == "hybrid" for t in c["layers_block_type"]]
    assert layer_kinds(c)["ffn"] == hybrid
    ffn = [lc.ffn_type for lc in get_configs(c).layer_configs]
    assert ffn == ["Dense" if h else "no_op" for h in hybrid]
    plan = resolve_layer_plan(c)
    assert plan["num_ffn_layers"] == 9
    assert plan["num_ffn_param_layers"] == c["num_mem_blocks"]


def test_zamba2_facts_count_the_shared_mlp_once_per_block():
    from llm_memory_calculator.model.facts import ModelFacts

    c = cfg("Zyphra_Zamba2-2.7B")
    roles = ModelFacts.from_config(c).params_by_role
    assert roles.get("dense_ffn") == c["num_mem_blocks"] * 3 * c["hidden_size"] * c["intermediate_size"]


def test_an_unknown_name_in_a_zamba_layer_list_is_charged_full_kv_and_said_so():
    c = dict(model_type="zamba2", hidden_size=2560, num_attention_heads=32,
             num_key_value_heads=32, head_dim=80, num_hidden_layers=4, mamba_d_state=64,
             layers_block_type=["mamba", "hybrid", "newkind", "mamba"])
    with pytest.warns(UserWarning, match="newkind"):
        kinds = layer_kinds(c)
    assert kinds["attn"] == [None, "full", "full", None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = resolve_layer_plan(c)
        assert plan["approximate"]
        assert kv_bytes(c, 4096) == 2 * per_layer_token(32, 80) * 4096


def test_a_transformers_release_without_layer_layouts_places_no_window(monkeypatch):
    """transformers before 4.53 derives no `layer_types` for any family, so a config class
    with none says nothing about the layout: Gemma 2 alternates, yet 4.50's Gemma2Config
    states no list. Reading that silence as "every layer takes the window" under-counted
    Gemma 2 8x at 32k; the window is placed on no layer instead, and said so."""
    transformers = pytest.importorskip("transformers")
    from transformers import CONFIG_MAPPING

    import llm_memory_calculator.layer_plan as lp

    class Gemma2Config450:  # what transformers 4.50's Gemma2Config exposes: no layer_types
        def __init__(self, **kw):
            self.__dict__.update({k: v for k, v in kw.items() if k != "layer_types"})

    monkeypatch.setattr(transformers, "__version__", "4.50.3")
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "gemma2", Gemma2Config450)
    monkeypatch.setattr(lp, "_CLASS_LAYOUT_CACHE", {})
    gemma2 = dict(model_type="gemma2", hidden_size=2304, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=256,
                  sliding_window=4096, vocab_size=1000, intermediate_size=9216)
    with pytest.warns(UserWarning, match="window"):
        kinds = layer_kinds(gemma2)
    assert kinds["attn"] == ["full"] * 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kv_bytes(gemma2, 32768) == 8 * per_layer_token(4, 256) * 32768


def test_model_facts_read_layer_kinds_for_a_model_type_the_recogniser_lacks():
    """FRD-027's facts consulted the layer-kind table only for an allow-listed model_type, so
    a Qwen3.5 GGUF (`qwen35`) and MiniMax-Text-01 were charged KV on every layer. The type
    stays flagged unrecognised; which layers hold KV is read from the config's own keys."""
    from llm_memory_calculator.model.facts import ModelFacts

    hf = cfg("Qwen_Qwen3.5-4B")
    g = dict(model_type="qwen35", num_hidden_layers=32, hidden_size=hf["text_config"]["hidden_size"],
             num_attention_heads=16, num_key_value_heads=4, head_dim=256, vocab_size=1000)
    g.update(hf_layer_keys_from_gguf(_gguf(qwen35_4b_gguf_meta()), 32))
    f = ModelFacts.from_config(g)
    assert not f.recognised and "architecture: unrecognised" in f.flags
    assert (f.layers.full, f.layers.recurrent) == (8, 24)
    known = ModelFacts.from_config(hf)
    assert f.kv_bytes_per_token() == known.kv_bytes_per_token() == 8 * 2 * 4 * 256 * 2
    # the 24 recurrent layers are charged their state and their mixers' weights, not dropped
    assert f.state_bytes_per_sequence() == pytest.approx(known.state_bytes_per_sequence())
    assert f.params_by_role["attention"] >= known.params_by_role["attention"] * 0.9

    mm = cfg("MiniMaxAI_MiniMax-Text-01")
    f = ModelFacts.from_config(mm)
    assert f.layers.full == sum(mm["attn_type_list"]) == 10
    assert f.layers.recurrent == 70


def test_model_facts_slide_a_uniform_window_by_the_table_not_by_family_name():
    """facts slid every layer only for model_type `mistral`; the table slides every layer of any
    uniform stack whose config class states no per-layer list (Phi-3's 2,047-token window)."""
    needs_transformers_layouts()
    from llm_memory_calculator.model.facts import ModelFacts

    phi3 = dict(model_type="phi3", hidden_size=3072, num_hidden_layers=32,
                num_attention_heads=32, num_key_value_heads=32, sliding_window=2047,
                vocab_size=32064, intermediate_size=8192)
    assert layer_kinds(phi3)["attn"] == ["sliding"] * 32
    f = ModelFacts.from_config(phi3)
    assert (f.layers.full, f.layers.sliding, f.layers.sliding_window) == (0, 32, 2047)


@pytest.mark.parametrize(
    "config",
    [
        dict(model_type="rwkv", hidden_size=2048, num_hidden_layers=24, attention_hidden_size=2048,
             intermediate_size=8192, vocab_size=50277, context_length=4096),
        dict(model_type="xlstm", hidden_size=4096, num_hidden_layers=32, num_heads=8,
             qk_dim_factor=0.5, v_dim_factor=1.0, vocab_size=50304),
    ],
    ids=["rwkv", "xlstm"],
)
def test_the_genz_state_of_every_mechanism_is_priced_like_the_calculator(config, tmp_path):
    """RWKV's token-shift vectors and xLSTM's normaliser are state, held at the state dtype;
    the graph priced them at the activation precision."""
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        get_configs,
    )

    kv = _genz_kv_mb(create_full_decode_model, tmp_path, name=get_configs(config),
                     input_sequence_length=4096, output_gen_tokens=0)
    assert kv == pytest.approx(state_bytes(config) / MiB, rel=1e-6)


def test_the_genz_graph_bounds_chunked_layers_by_their_chunk(tmp_path):
    """Llama 4: 36 `chunked_attention` layers keep at most `attention_chunk_size` (8,192) tokens.
    The graph read only `sliding_window`, so they were graphed over the whole context."""
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        get_configs,
    )

    n = 48
    c = dict(model_type="llama4_text", hidden_size=5120, num_hidden_layers=n,
             num_attention_heads=40, num_key_value_heads=8, head_dim=128, intermediate_size=8192,
             intermediate_size_mlp=16384, num_local_experts=16, num_experts_per_tok=1,
             interleave_moe_layer_step=1, attention_chunk_size=8192, vocab_size=202048,
             layer_types=["full_attention" if (i + 1) % 4 == 0 else "chunked_attention"
                          for i in range(n)])
    per_token_mib = per_layer_token(8, 128) / MiB

    def kv(seq):
        return _genz_kv_mb(create_full_decode_model, tmp_path, name=get_configs(c),
                           input_sequence_length=seq, output_gen_tokens=0)

    assert kv(32768) == pytest.approx(per_token_mib * (12 * 32768 + 36 * 8192), rel=1e-6)
    assert _chunked_kv_mb(c, tmp_path, 32768) - _chunked_kv_mb(c, tmp_path, 16384) == (
        pytest.approx(per_token_mib * 12 * 16384, rel=1e-6)
    )


# ------------------------------------------- serving: the KV that grows and the state that does not
#
# A request in the serving memory model costs its KV blocks AND its recurrent state. Once the
# recurrent layers stopped being charged KV, a serving path with no state term admitted too
# many hybrid requests and charged a pure SSM nothing at all.

SERVING_RECURRENT = [
    "Qwen_Qwen3.5-4B",  # Gated DeltaNet + full attention, nested under text_config
    "ibm-granite_granite-4.0-h-small",  # Mamba-2 + attention, by layer_types
    "nvidia_Nemotron-H-8B-Base-8K",  # Mamba-2 + attention + FFN-only slots, by pattern
    "tiiuae_Falcon-H1-34B-Instruct",  # parallel hybrid: KV and state on every layer
    "LiquidAI_LFM2-2.6B",  # short convolution
    "state-spaces_mamba-2.8b-hf",  # pure Mamba-1: no KV at all
    "mistralai_Mamba-Codestral-7B-v0.1",  # pure Mamba-2
]
GiB = 1024**3


def _memory_model(config, capacity=8 * GiB, tp=1, tiers=None):
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.constants import MemoryTier
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig

    tiers = tiers or [MemoryTierConfig(tier=MemoryTier.DEVICE_HBM, capacity_bytes=capacity)]
    return MemoryModel(get_configs(config), tiers, tensor_parallel=tp, precision_bytes=2)


@pytest.mark.parametrize("name", SERVING_RECURRENT)
def test_the_serving_memory_model_charges_each_request_its_fixed_state(name):
    """Measured by the reviewer on the branch: Qwen3.5-4B admitted 256 1,024-token requests into
    8 GiB (truth ~101), granite-4.0-h-small 512 (truth ~50), mamba-2.8b without bound."""
    c = cfg(name)
    mm = _memory_model(c)
    assert mm.state_bytes_per_sequence == pytest.approx(state_bytes(c), abs=1)
    assert mm.state_bytes_per_sequence > 0
    per_request = 1024 * mm.bytes_per_token_kv + mm.state_bytes_per_sequence
    admitted = 0
    with pytest.raises(MemoryError):
        while admitted < 10**5:
            mm.allocate_kv_blocks(admitted, 1024)
            admitted += 1
    assert admitted == 8 * GiB // per_request


def test_the_serving_state_is_split_by_tensor_parallelism():
    c = cfg("Qwen_Qwen3.5-4B")
    assert _memory_model(c, tp=2).state_bytes_per_sequence == pytest.approx(
        state_bytes(c) / 2, abs=1
    )


def test_a_dense_model_carries_no_serving_state():
    llama = dict(model_type="llama", hidden_size=4096, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336,
                 vocab_size=128256)
    mm = _memory_model(llama)
    assert mm.state_bytes_per_sequence == 0
    assert mm.bytes_per_token_kv == 32 * per_layer_token(8, 128)


def test_the_serving_state_is_freed_with_its_request():
    from llm_memory_calculator.genz.serving.constants import MemoryTier

    mm = _memory_model(cfg("Qwen_Qwen3.5-4B"))
    per_request = 1024 * mm.bytes_per_token_kv + mm.state_bytes_per_sequence

    def used():
        return mm.memory_snapshot()[MemoryTier.DEVICE_HBM.value]["used_bytes"]

    mm.allocate_kv_blocks(1, 1024)
    mm.allocate_kv_blocks(2, 1024)
    assert used() == 2 * per_request
    mm.deallocate_kv_blocks(1)
    assert used() == per_request
    evicted, gone = mm.evict_blocks(MemoryTier.DEVICE_HBM, 64)
    assert (evicted, gone) == (64, [2])
    assert used() == 0


def test_a_pure_ssm_spills_without_dividing_by_its_zero_kv():
    from llm_memory_calculator.genz.serving.constants import MemoryTier
    from llm_memory_calculator.genz.serving.memory_model import MemoryTierConfig

    mm = _memory_model(cfg("state-spaces_mamba-2.8b-hf"), tiers=[
        MemoryTierConfig(tier=MemoryTier.DEVICE_HBM, capacity_bytes=GiB),
        MemoryTierConfig(tier=MemoryTier.HOST_DDR, capacity_bytes=GiB),
    ])
    mm.allocate_kv_blocks(1, 1024)
    assert mm.spill_to_next_tier(MemoryTier.DEVICE_HBM, 4) == (0, MemoryTier.DEVICE_HBM)


def test_the_scheduler_fails_a_request_whose_state_alone_exceeds_every_tier():
    """The permanent-failure check priced a request by its KV blocks only, so a pure SSM
    request (0 KV) was never failed however small the memory."""
    mm = _memory_model(cfg("state-spaces_mamba-2.8b-hf"), capacity=1024 * 1024)
    assert mm.request_bytes(1024) == mm.state_bytes_per_sequence
    assert mm.request_bytes(1024) > mm.total_capacity_bytes


@pytest.mark.parametrize("name", SERVING_RECURRENT)
@pytest.mark.parametrize("analyzer", ["cluster", "disaggregation"])
def test_serving_concurrency_counts_the_fixed_state(name, analyzer):
    """B_eff = (capacity - weights) // (KV per token x context + state). A pure SSM's footprint
    was 0, which the analyzers read as 'cannot size' and answered 1 (serial)."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.cluster import ClusterAnalyzer
    from llm_memory_calculator.genz.serving.constants import GB_TO_BYTES
    from llm_memory_calculator.genz.serving.disaggregation import DisaggregationAnalyzer
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes
    from llm_memory_calculator.hardware import get_hardware_config

    c = cfg(name)
    hw = "A100_80GB_GPU"
    cls = ClusterAnalyzer if analyzer == "cluster" else DisaggregationAnalyzer
    a = cls(model=c, hardware=hw)
    mm = _memory_model(c)
    cap = int(get_hardware_config(hw)["Memory_size"] * GB_TO_BYTES)
    weights = estimate_weight_bytes(get_configs(c), "bf16", 1)
    context = 4096
    footprint = mm.bytes_per_token_kv * context + mm.state_bytes_per_sequence
    expected = max(1, (cap - weights) // footprint) if cap > weights else 1
    assert a._build_b_eff(context, max_batch_size=10**7) == min(expected, 10**7)


@pytest.mark.parametrize("name", SERVING_RECURRENT)
def test_the_disaggregated_transfer_carries_the_state(name):
    """Prefill hands decode the KV and the recurrent state; the transfer counted the KV only, so a
    pure SSM moved 0 bytes."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.disaggregation import DisaggregationAnalyzer

    c = cfg(name)
    a = DisaggregationAnalyzer(model=c, hardware="A100_80GB_GPU")
    kv = get_configs(c).get_kv_size() * 2 * (512 + 128)
    assert a._estimate_kv_bytes(512, 128) == pytest.approx(kv + state_bytes(c), abs=1)


# ------------------------------------------- a layout stated under another key, and a short list


def test_gpt_neo_alternates_its_window_by_attention_types():
    """GPT-Neo states its global/local layout in `attention_types` (its config class expands it
    into `attention_layers`), not in `layer_types`; its config class states no `layer_types`, so
    the class-silence rule read every layer as windowed: 4.5x low at 2,048 tokens."""
    from llm_memory_calculator.model.facts import ModelFacts

    neo = cfg("EleutherAI_gpt-neo-1.3B")
    k = layer_kinds(neo)
    assert k["attn"] == ["full", "sliding"] * 12
    assert k["sliding_window"] == 256
    f = ModelFacts.from_config(neo)
    assert (f.layers.full, f.layers.sliding, f.layers.sliding_window) == (12, 12, 256)
    per = per_layer_token(16, 128)
    assert f.kv_bytes_for_context(2048) == 12 * 2048 * per + 12 * 256 * per


def test_gpt_neo_attention_types_alone_state_the_layout():
    neo = {k: v for k, v in cfg("EleutherAI_gpt-neo-1.3B").items() if k != "attention_layers"}
    neo["attention_types"] = [[["global"], 2], [["global", "local"], 11]]
    assert layer_kinds(neo)["attn"] == ["full"] * 2 + ["full", "sliding"] * 11


def test_the_calculator_charges_gpt_neo_its_global_and_local_kv():
    """GPT-Neo spells its head count `num_heads`; the calculator found no attention heads and
    charged no KV at all."""
    neo = cfg("EleutherAI_gpt-neo-1.3B")
    per = per_layer_token(16, 128)
    caps = {"heterogeneous_kv_layout": True}
    assert kv_bytes(neo, 2048, engine=caps) == pytest.approx(12 * 2048 * per + 12 * 256 * per)
    assert kv_bytes(neo, 2048) == pytest.approx(24 * 2048 * per)


def test_gpt_neo_is_not_taken_for_a_hybrid_by_its_attention_layers_key():
    """`attention_layers` is GPT-Neo's per-layer list of global / local attention, not a hybrid
    marker: reading it as one guessed a Mamba state on every layer (1.7 MB on origin/main)."""
    neo = cfg("EleutherAI_gpt-neo-1.3B")
    assert ModelMemoryCalculator().detect_model_type(neo) == "decoder-only"
    assert state_bytes(neo) == 0


def test_a_per_layer_list_shorter_than_the_depth_charges_the_rest_full_and_says_so():
    """A `layer_types` list that names 16 of 32 layers left the other 16 free."""
    from llm_memory_calculator.model.facts import ModelFacts

    short = dict(model_type="llama", hidden_size=4096, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336,
                 vocab_size=32000, layer_types=["full_attention"] * 16)
    with pytest.warns(UserWarning, match="16 of 32 layers"):
        k = layer_kinds(short)
    assert k["attn"] == ["full"] * 32
    assert any("16 of 32 layers" in n for n in k["notes"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kv_bytes(short, 4096) == pytest.approx(32 * 4096 * per_layer_token(8, 128))
        f = ModelFacts.from_config(short)
    assert (f.layers.full, f.layers.sliding) == (32, 0)


def test_a_short_hybrid_list_keeps_its_stated_kinds_and_charges_the_rest_full():
    hybrid = dict(cfg("Qwen_Qwen3.5-4B")["text_config"])
    stated = hybrid["layer_types"][:16]
    hybrid["layer_types"] = stated
    with pytest.warns(UserWarning, match="16 of 32 layers"):
        plan = resolve_layer_plan(hybrid)
    assert plan["attn"][16:] == ["full"] * 16
    assert plan["recurrent"][:16] == ["gdn" if t == "linear_attention" else None for t in stated]
    assert plan["approximate"]


def test_kimi_layer_lists_are_one_indexed_even_without_kda_layers():
    """Kimi's modelling code tests `layer_idx + 1` against the lists. Without `kda_layers` the
    one-indexing guess needed min == 1 and dropped the last full layer (index 27 of 27)."""
    kimi = cfg("moonshotai_Kimi-Linear-48B-A3B-Instruct")
    lac = dict(kimi["linear_attn_config"])
    lac.pop("kda_layers", None)
    kimi = dict(kimi, linear_attn_config=lac)
    plan = resolve_layer_plan(kimi)
    n = kimi["num_hidden_layers"]
    full = {i for i, a in enumerate(plan["attn"]) if a == "full"}
    assert full == {i - 1 for i in lac["full_attn_layers"] if 1 <= i <= n}
