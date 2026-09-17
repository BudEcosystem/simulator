"""Pre-quantized checkpoints must be timed at the precision they are stored in.

The performance model used to size every weight at the caller's `bits`. budsim passes `bf16` for
any checkpoint it is not quantizing itself, so a checkpoint that ships quantized was timed as if it
were bf16. gpt-oss-20b stores its MoE experts in MXFP4 and keeps attention/router/embeddings/lm_head
in bf16; timed at bf16 its decode came out 1.3x slow at batch 1 and 3.4x slow at batch 32, because the
experts a batch touches grow with batch size and each was charged ~3.8x its real bytes.

These tests pin the contract generically — by format and by exclusion pattern, not by model — plus
one checkpoint-backed guard against the measured gpt-oss numbers.
"""
import io
import contextlib
import json
from pathlib import Path

import pytest

from llm_memory_calculator.genz.weight_precision import (
    WeightRoleTracker,
    resolve_weight_precision,
)

FIXTURES = Path(__file__).parent / "fixtures" / "measured_checkpoints"
GPT_OSS = FIXTURES / "openai_gpt-oss-20b"
QWEN_DENSE = FIXTURES / "Qwen_Qwen3-0.6B"

LARGE_LINEAR_ROLES = {"attention", "dense_ffn", "expert", "shared_expert"}


def _cfg(**quantization_config):
    return {"model_type": "x", "hidden_size": 64, "quantization_config": quantization_config}


# ------------------------------------------------------------------ nothing declared


def test_a_checkpoint_without_quantization_config_has_no_plan():
    """The common case must resolve to None, which keeps the model byte-identical."""
    assert resolve_weight_precision({"model_type": "llama", "hidden_size": 4096}) is None
    assert resolve_weight_precision({"quantization_config": {}}) is None
    assert resolve_weight_precision("not a dict") is None


def test_an_unknown_format_warns_and_falls_back_rather_than_guessing():
    with pytest.warns(UserWarning, match="not understood"):
        assert resolve_weight_precision(_cfg(quant_method="some_future_format")) is None


# ------------------------------------------------------------------ formats


@pytest.mark.parametrize(
    "qc, expected",
    [
        # 4-bit elements + one 8-bit scale per 32
        ({"quant_method": "mxfp4"}, 4.25 / 8),
        # 4-bit elements + one 8-bit scale per 16
        ({"quant_method": "nvfp4"}, 4.5 / 8),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, 4.5 / 8),
        ({"quant_method": "fp8"}, 1.0),
        ({"quant_method": "modelopt", "quant_algo": "FP8"}, 1.0),
        # bits + a 16-bit scale per group
        ({"quant_method": "awq", "bits": 4, "group_size": 128}, (4 + 16 / 128) / 8),
        ({"quant_method": "gptq", "bits": 8, "group_size": 32}, (8 + 16 / 32) / 8),
        ({"quant_method": "gptq", "bits": 4, "group_size": -1}, 0.5),
        ({"quant_method": "bitsandbytes", "load_in_8bit": True}, 1.0),
        ({"quant_method": "bitsandbytes", "load_in_4bit": True}, (4 + 32 / 64) / 8),
        ({"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_use_double_quant": True},
         (4 + 8 / 64) / 8),
        ({"quant_method": "compressed-tensors",
          "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 128}}}}, (4 + 16 / 128) / 8),
    ],
)
def test_each_format_is_sized_from_its_own_layout(qc, expected):
    plan = resolve_weight_precision(_cfg(**qc))
    assert plan is not None
    assert set(plan) == LARGE_LINEAR_ROLES, "with no exclusions, the large linear layers are converted"
    assert all(v == pytest.approx(expected) for v in plan.values())


def test_embeddings_lm_head_and_router_stay_at_checkpoint_precision_by_default():
    plan = resolve_weight_precision(_cfg(quant_method="awq", bits=4, group_size=128))
    assert not {"embedding", "lm_head", "router"} & set(plan)


# ------------------------------------------------------------------ exclusions


@pytest.mark.parametrize(
    "key, patterns, excluded",
    [
        ("modules_to_not_convert", ["model.layers.*.self_attn"], {"attention"}),
        ("ignore", ["re:.*self_attn.*"], {"attention"}),
        ("exclude_modules", ["*mlp.experts*"], {"expert"}),
        ("modules_to_not_convert", ["model.layers.*.mlp.shared_expert"], {"shared_expert"}),
        # a bare `mlp` names the whole FFN block, experts included
        ("modules_to_not_convert", ["model.layers.*.mlp"], {"dense_ffn", "expert", "shared_expert"}),
        # but `gate_proj` is a dense FFN weight, never the MoE router
        ("modules_to_not_convert", ["model.layers.*.mlp.gate_proj"], {"dense_ffn"}),
        ("llm_int8_skip_modules", ["lm_head"], set()),
    ],
)
def test_exclusion_patterns_remove_exactly_the_roles_they_name(key, patterns, excluded):
    plan = resolve_weight_precision(_cfg(quant_method="fp8", **{key: patterns}))
    assert set(plan) == LARGE_LINEAR_ROLES - excluded


def test_a_router_pattern_is_not_mistaken_for_an_ffn_projection():
    for pattern in ("model.layers.*.mlp.gate", "re:.*mlp.gate$", "model.layers.*.mlp.router"):
        plan = resolve_weight_precision(_cfg(quant_method="fp8", modules_to_not_convert=[pattern]))
        assert set(plan) == LARGE_LINEAR_ROLES, pattern


def test_a_pattern_pinned_to_one_layer_does_not_unquantize_the_whole_model():
    """A pinned pattern is not model-wide. (With the layer count known it is blended by layer fraction, see
    test_layer_pinned_exclusions_blend_by_the_fraction_of_layers_they_cover.)"""
    plan = resolve_weight_precision(_cfg(quant_method="fp8", ignore=["model.layers.0.mlp", "model.layers.61.mlp"]))
    assert set(plan) == LARGE_LINEAR_ROLES


def test_everything_excluded_resolves_to_no_plan():
    plan = resolve_weight_precision(_cfg(quant_method="fp8", ignore=["self_attn", "mlp"]))
    assert plan is None


def test_real_gpt_oss_config_converts_only_its_experts():
    config = json.loads((GPT_OSS / "config.json").read_text())
    plan = resolve_weight_precision(config)
    assert set(plan) == {"dense_ffn", "expert", "shared_expert"}, "attention/router/embed/lm_head are bf16"
    assert plan["expert"] == pytest.approx(4.25 / 8)


# ------------------------------------------------------------------ operator graph -> role


def test_ffn_rows_after_a_router_are_experts_and_without_one_are_dense():
    tracker = WeightRoleTracker()
    rows = ["Repeat", "QKV", "Logit Pre", "Out Proj", "Gate", "up+gate", "down", "up+gate", "down", "End Repeat",
            "Repeat", "QKV", "Out Proj", "up+gate", "down", "shared up+gate", "End Repeat", "classifier"]
    roles = [tracker.role_for(r) for r in rows]
    assert roles == [None, "attention", None, "attention", "router", "expert", "expert", "expert", "expert", None,
                     None, "attention", "attention", "dense_ffn", "dense_ffn", "shared_expert", None, "lm_head"]


def test_mla_and_ssm_mixer_projections_are_attention_role():
    tracker = WeightRoleTracker()
    assert {tracker.role_for(n) for n in ("Q Down", "Q Up", "KV Compress", "KV Up", "Inproj", "Out proj")} \
        == {"attention"}


# ------------------------------------------------------------------ end to end through GenZ


def _decode_df(model, bs, ctx, system="H100_GPU"):
    from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling

    with contextlib.redirect_stdout(io.StringIO()):
        return decode_moddeling(model=str(model), batch_size=bs, input_tokens=ctx, output_tokens=1, Bb=1,
                                system_name=system, bits="bf16", model_profilling=True)


def _first(df, name):
    return df[df["Layer Name"] == name].iloc[0]


def test_only_the_converted_weight_tensor_shrinks():
    """Expert weights shrink by 2 / 0.53125; attention weights and every activation are untouched."""
    from llm_memory_calculator.genz.LLM_inference import utils

    quantized, _ = _decode_df(GPT_OSS, 8, 2048)
    original = utils.apply_checkpoint_weight_precision
    try:
        utils.apply_checkpoint_weight_precision = lambda system, model: system
        import llm_memory_calculator.genz.LLM_inference.llm_decode as decode_module
        decode_module.apply_checkpoint_weight_precision = utils.apply_checkpoint_weight_precision
        uniform, _ = _decode_df(GPT_OSS, 8, 2048)
    finally:
        utils.apply_checkpoint_weight_precision = original
        decode_module.apply_checkpoint_weight_precision = original

    up_q, up_u = _first(quantized, "up+gate"), _first(uniform, "up+gate")
    assert up_u["Input_w (MB)"] / up_q["Input_w (MB)"] == pytest.approx(2 / (4.25 / 8), rel=1e-6)
    assert up_q["Input_a (MB)"] == up_u["Input_a (MB)"], "activations must stay in the activation dtype"
    assert up_q["Output (MB)"] == up_u["Output (MB)"]
    assert _first(quantized, "QKV")["Input_w (MB)"] == _first(uniform, "QKV")["Input_w (MB)"]
    assert _first(quantized, "classifier")["Input_w (MB)"] == _first(uniform, "classifier")["Input_w (MB)"]


def test_a_dense_bf16_checkpoint_is_unaffected():
    from llm_memory_calculator.genz.Models import get_configs

    assert get_configs(str(QWEN_DENSE)).weight_precision is None


def test_a_prequantized_model_is_no_longer_refused_for_bf16_weights_it_does_not_have():
    """Decode refuses (ValueError) a device whose memory cannot hold the modelled weights. Timed at
    bf16, gpt-oss-20b modelled ~38 GB of weights and was refused on a 32 GB accelerator that vLLM
    loads it onto in 13.0 GiB. This is the planner's path: no profiling flag, so no downgrade to a
    warning."""
    from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling

    small = {"Flops": 989.5, "Memory_size": 32, "Memory_BW": 3350, "ICN": 450, "real_values": True,
             "type": "gpu", "memory_type": "hbm3", "tensor_cores": "gen4"}
    with contextlib.redirect_stdout(io.StringIO()):
        result = decode_moddeling(model=str(GPT_OSS), batch_size=1, input_tokens=2048, output_tokens=1, Bb=1,
                                  system_name=small, bits="bf16")
    assert result["Latency"] > 0

    _, summary = _decode_df(GPT_OSS, 1, 2048, system=small)
    weights_gb = summary["Total Weights (MB)"].values[0] / 1024
    assert weights_gb < 16, f"modelled {weights_gb:.1f} GB of weights; the checkpoint loads in 13.0 GiB"


# ------------------------------------------------------------------ sidecar quantization files


def _checkpoint_dir(tmp_path, sidecars, quantization_config=None):
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "num_hidden_layers": 4,
              "hidden_size": 256, "intermediate_size": 688, "num_attention_heads": 4, "num_key_value_heads": 2,
              "vocab_size": 1000, "torch_dtype": "bfloat16"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name, body in sidecars.items():
        (tmp_path / name).write_text(body if isinstance(body, str) else json.dumps(body))
    return str(tmp_path)


@pytest.mark.parametrize(
    "sidecar, body, expected",
    [
        ("hf_quant_config.json",
         {"producer": {"name": "modelopt", "version": "0.19.0"},
          "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8", "exclude_modules": ["lm_head"]}},
         1.0),
        ("hf_quant_config.json",
         {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "group_size": 16}},
         4.5 / 8),
        ("hf_quant_config.json",
         {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "W4A8_AWQ", "group_size": 128}},
         (4 + 16 / 128) / 8),
        ("quantize_config.json", {"bits": 4, "group_size": 128, "desc_act": False, "sym": True}, (4 + 16 / 128) / 8),
        ("quant_config.json", {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
         (4 + 16 / 128) / 8),
    ],
)
def test_a_sidecar_only_checkpoint_is_sized_from_its_sidecar(tmp_path, sidecar, body, expected):
    """ModelOpt, AutoGPTQ and legacy AutoAWQ exports carry no quantization_config in config.json."""
    from llm_memory_calculator.genz.Models import get_configs
    from llm_memory_calculator.huggingface_loader import HuggingFaceConfigLoader

    path = _checkpoint_dir(tmp_path, {sidecar: body})
    loaded = HuggingFaceConfigLoader().fetch_model_config(path)
    assert loaded["_quantization_config_source"] == sidecar
    plan = get_configs(path).weight_precision
    assert plan is not None, f"{sidecar} was ignored: the checkpoint would be timed at bf16"
    assert set(plan) == LARGE_LINEAR_ROLES
    assert all(v == pytest.approx(expected) for v in plan.values())


def test_config_json_quantization_wins_over_a_sidecar(tmp_path):
    path = _checkpoint_dir(tmp_path, {"quantize_config.json": {"bits": 8, "group_size": -1}},
                           quantization_config={"quant_method": "fp8"})
    from llm_memory_calculator.huggingface_loader import HuggingFaceConfigLoader

    loaded = HuggingFaceConfigLoader().fetch_model_config(path)
    assert loaded["quantization_config"] == {"quant_method": "fp8"}
    assert "_quantization_config_source" not in loaded


def test_an_unreadable_sidecar_is_ignored_rather_than_fatal(tmp_path):
    from llm_memory_calculator.huggingface_loader import HuggingFaceConfigLoader

    path = _checkpoint_dir(tmp_path, {"hf_quant_config.json": "{not json"})
    loaded = HuggingFaceConfigLoader().fetch_model_config(path)
    assert "quantization_config" not in loaded


# ------------------------------------------------------------------ compressed-tensors with several schemes


def _ct(groups, ignore=None):
    qc = {"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups": groups}
    if ignore is not None:
        qc["ignore"] = ignore
    return _cfg(**qc)


def test_compressed_tensors_groups_give_each_role_its_own_scheme():
    plan = resolve_weight_precision(_ct({
        "group_0": {"targets": ["re:.*self_attn.*"], "weights": {"num_bits": 8, "type": "int", "strategy": "channel"}},
        "group_1": {"targets": ["re:.*mlp.*"], "weights": {"num_bits": 4, "type": "int", "group_size": 128}},
    }, ignore=["lm_head"]))
    assert plan["attention"] == pytest.approx(1.0)
    for role in ("dense_ffn", "expert", "shared_expert"):
        assert plan[role] == pytest.approx((4 + 16 / 128) / 8)


def test_a_named_target_overrides_a_class_target():
    """W8A8 on every Linear, NVFP4 on the experts: the regex group wins for the roles it names."""
    plan = resolve_weight_precision(_ct({
        "group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
        "group_1": {"targets": ["re:.*mlp.experts.*"], "weights": {"num_bits": 4, "type": "float", "group_size": 16}},
    }))
    assert plan["expert"] == pytest.approx(4.5 / 8), "NVFP4 scales are FP8, not 16-bit"
    assert plan["attention"] == pytest.approx(1.0)
    assert plan["dense_ffn"] == pytest.approx(1.0)


def test_a_group_without_a_weight_scheme_leaves_its_roles_unquantized():
    plan = resolve_weight_precision(_ct({
        "group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "int", "group_size": 128}},
        "group_1": {"targets": ["re:.*self_attn.*"], "weights": None, "input_activations": {"num_bits": 8}},
    }))
    assert "attention" not in plan
    assert plan["dense_ffn"] == pytest.approx((4 + 16 / 128) / 8)


# ------------------------------------------------------------------ exclusions pinned to layers


@pytest.mark.parametrize(
    "patterns, role, excluded_layers",
    [
        (["model.layers.0.mlp", "model.layers.3.mlp"], "dense_ffn", 2),
        (["re:model.layers.(0|1|2).self_attn.*"], "attention", 3),
        (["model.layers.[0-1].mlp"], "expert", 2),
        (["model.layers.0.mlp", "model.layers.0.mlp.down_proj"], "dense_ffn", 1),  # same layer counted once
    ],
)
def test_layer_pinned_exclusions_blend_by_the_fraction_of_layers_they_cover(patterns, role, excluded_layers):
    """Decode reads every layer, so a role left at 16-bit on n of L layers streams the layer-weighted mean."""
    config = _cfg(quant_method="fp8", ignore=patterns)
    config["num_hidden_layers"] = 4
    plan = resolve_weight_precision(config)
    fraction = excluded_layers / 4
    assert plan[role] == pytest.approx(1.0 * (1 - fraction) + 2.0 * fraction)


def test_a_pinned_exclusion_never_unquantizes_the_layers_it_does_not_name():
    config = _cfg(quant_method="fp8", ignore=["model.layers.0.mlp"])
    config["num_hidden_layers"] = 48
    plan = resolve_weight_precision(config)
    assert plan["attention"] == pytest.approx(1.0)
    assert plan["dense_ffn"] == pytest.approx(1.0 + 1.0 / 48)


# ------------------------------------------------------------------ GPTQ's explicit module list


def test_gptq_modules_in_block_to_quantize_limits_the_converted_roles():
    plan = resolve_weight_precision(_cfg(
        quant_method="gptq", bits=4, group_size=128,
        modules_in_block_to_quantize=[["self_attn.q_proj", "self_attn.k_proj"], ["mlp.down_proj"]],
    ))
    assert set(plan) == {"attention", "dense_ffn"}


@pytest.mark.parametrize("target, role", [
    ("re:.*(q|k|v|o)_proj$", "attention"),
    ("re:.*(gate|up|down)_proj$", "dense_ffn"),
    ("self_attn.qkv_proj", "attention"),
    ("mlp.gate_up_proj", "dense_ffn"),
])
def test_bare_projection_names_are_classified(target, role):
    from llm_memory_calculator.genz.weight_precision import _roles_named_by

    assert _roles_named_by(target) == {role}
