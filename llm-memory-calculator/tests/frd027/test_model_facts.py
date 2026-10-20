"""FRD-027 §7, T-MDL-01…06: one engine-agnostic model description, for memory and performance."""

import pytest

from llm_memory_calculator.kv.geometry import KVGeometry
from llm_memory_calculator.model import ModelFacts, size_bucket
from llm_memory_calculator.parameter_counter import UniversalParameterCounter

from .conftest import load_model

GIB = 1024**3


def facts(name: str, **kw) -> ModelFacts:
    return ModelFacts.from_config(load_model(name), **kw)


# --------------------------------------------------------------------------- T-MDL-02 weight bytes


@pytest.mark.parametrize(
    "name, fmt, measured_gib",
    [
        # study/results startup.json model_load_gib (vLLM v0.30 upstream, H100XM-80C)
        ("qwen3.8-27b", "native", 51.10),
        ("qwen3.8-27b-fp8", "native", 28.51),
        ("qwen3.8-27b", "fp8", 28.51),  # online --quantization fp8 measured 28.06-28.63
        ("qwen3-8b", "native", 15.27),
        ("qwen3-8b-fp8", "native", 8.80),
        ("qwen3-8b", "fp8", 9.06),
        ("gpt-oss-20b", "native", 13.03),
        ("gpt-oss-20b-bf16", "native", 39.15),
    ],
)
def test_weight_bytes_match_measured_load(name, fmt, measured_gib):
    f = facts(name)
    assert f.weight_bytes(fmt) / GIB == pytest.approx(measured_gib, rel=0.03)


def test_mtp_and_vision_are_separate_components():
    f = facts("qwen3.8-27b")
    with_mtp = f.weight_bytes("native", include_mtp=True)
    # measured: MTP spec decode loads 51.97 GiB vs 51.10 without
    assert (with_mtp - f.weight_bytes("native")) / GIB == pytest.approx(0.87, abs=0.15)
    # --language-model-only measured 50.22 GiB
    assert f.weight_bytes("native", include_vision=False) / GIB == pytest.approx(50.22, rel=0.02)
    assert f.mtp_layers == 1
    assert f.component("vision_encoder").params > 0
    assert [c.name for c in facts("qwen3-8b").components] == ["decoder"]


def test_text_params_agree_with_universal_counter():
    for name in ("qwen3-8b", "llama-3.1-8b", "gpt-oss-20b-bf16"):
        f = facts(name)
        universal = UniversalParameterCounter().count_parameters(load_model(name))
        assert f.params_total == pytest.approx(universal, rel=0.03), name


# --------------------------------------------------------------------------- T-MDL-01 hybrid bytes


def test_gdn_layers_are_recurrent_not_attention():
    f = facts("qwen3.8-27b")
    assert f.layers.full == 16 and f.layers.recurrent == 48 and f.layers.sliding == 0
    geometry = KVGeometry.from_model(load_model("qwen3.8-27b"), seq_length=4096)
    # KV per token: 16 full layers only, the value the memory path implies
    assert f.kv_bytes_per_token("bf16") == pytest.approx(geometry.bytes_per_token_total, rel=0.01)
    # state per sequence: 48 layers of GDN state (fp32 recurrent + bf16 conv) = ~3.06 MiB per layer
    assert f.state_bytes_per_sequence("float32") == pytest.approx(
        48 * geometry.state_bytes_per_layer, rel=0.01
    )
    assert f.state_bytes_per_sequence("bfloat16") < 0.55 * f.state_bytes_per_sequence("float32")
    # decode bytes at batch 1, 4K context: weights + 16 layers of KV + 48 layers of state read+write
    step = f.decode_bytes(
        batch=1, context=4096, weight_format="native", kv_dtype="bf16", state_dtype="float32"
    )
    expected = (
        f.weight_bytes_streamed("native")
        + 4096 * f.kv_bytes_per_token("bf16")
        + 2 * f.state_bytes_per_sequence("float32")
    )
    assert step == pytest.approx(expected, rel=1e-6)
    full_attention_kv = 4096 * 64 * f.kv_bytes_per_token("bf16") / 16
    assert step < f.weight_bytes_streamed("native") + full_attention_kv


def test_dense_kv_per_token():
    f = facts("qwen3-8b")
    assert f.layers.full == 36 and f.layers.recurrent == 0
    # 36 layers x 2 (K,V) x 8 heads x 128 x 2 bytes
    assert f.kv_bytes_per_token("bf16") == 36 * 2 * 8 * 128 * 2
    assert f.kv_bytes_per_token("fp8") == 36 * 2 * 8 * 128


def test_sliding_layers_cost_the_window():
    f = facts("gpt-oss-20b")
    assert f.layers.sliding == 12 and f.layers.full == 12 and f.layers.sliding_window == 128
    short = f.kv_bytes_for_context(100, "bf16")
    long = f.kv_bytes_for_context(10000, "bf16")
    per_layer_token = 2 * 8 * 64 * 2
    assert short == 100 * 24 * per_layer_token
    assert long == (10000 * 12 + 128 * 12) * per_layer_token


def test_moe_union_grows_with_batch():
    f = facts("gpt-oss-20b")
    one = f.weight_bytes_streamed("native", batch=1)
    many = f.weight_bytes_streamed("native", batch=64)
    assert one < many <= f.weight_bytes("native") * 1.0001
    assert f.experts == (32, 4)
    assert f.params_active < f.params_total / 4


# --------------------------------------------------------------------------- T-MDL-03 facts


def test_spec_vision_sinks_facts():
    q27 = facts("qwen3.8-27b")
    assert q27.mtp_layers == 1 and "multimodal" in q27.attributes and "mtp" in q27.attributes
    oss = facts("gpt-oss-20b")
    assert oss.attention_sinks and "sinks" in oss.attributes and "sliding" in oss.attributes
    q8 = facts("qwen3-8b")
    assert q8.mtp_layers == 0 and not q8.attention_sinks and "multimodal" not in q8.attributes
    assert q8.max_context == 40960 and q27.max_context == 262144


# --------------------------------------------------------------------------- T-MDL-05 unknown


def test_unknown_architecture_degrades_to_dense():
    f = facts("unknown-arch")
    assert f.recognised is False
    assert "architecture: unrecognised" in f.flags
    assert f.arch_class == "dense-GQA"
    assert f.layers.full == 28 and f.layers.recurrent == 0
    assert f.weight_bytes("native") > 0 and f.kv_bytes_per_token("bf16") > 0


def test_empty_config_never_raises():
    f = ModelFacts.from_config({})
    assert f.recognised is False and f.weight_bytes("native") >= 0


# --------------------------------------------------------------------------- T-MDL-06 class and family


@pytest.mark.parametrize(
    "name, klass, attrs",
    [
        ("qwen3.8-27b", "hybrid-linear", {"mtp", "multimodal"}),
        ("qwen3-8b", "dense-GQA", set()),
        ("llama-3.1-8b", "dense-GQA", set()),
        ("gpt-oss-20b", "MoE", {"sliding", "sinks"}),
        ("deepseek-v3", "MLA", {"moe", "mtp"}),
        ("whisper-large-v3", "encoder-decoder", set()),
    ],
)
def test_class_derivation(name, klass, attrs):
    f = facts(name)
    assert f.arch_class == klass
    assert attrs <= set(f.attributes)


def test_family_inherits_lineage_and_size_buckets():
    assert facts("qwen3-8b-fp8", family="Qwen3-8B").family == "Qwen3-8B"
    assert facts("qwen3-8b").family == "qwen3"  # no lineage given: model type
    assert facts("qwen3-8b").size_bucket == "4-10B"
    assert facts("qwen3.8-27b").size_bucket == "10-40B"
    oss = facts("gpt-oss-20b")
    assert oss.size_bucket == "<=4B"  # by active parameters
    assert oss.params_total > 20e9
    assert size_bucket(150e9) == ">120B" and size_bucket(50e9) == "40-120B"


def test_native_format():
    assert facts("qwen3.8-27b").native_format == "bf16"
    assert facts("qwen3.8-27b-fp8").native_format == "fp8"
    assert facts("gpt-oss-20b").native_format == "mxfp4"


def test_digest_is_stable():
    assert facts("qwen3-8b").digest() == facts("qwen3-8b").digest()
    assert facts("qwen3-8b").digest() != facts("qwen3-8b-fp8").digest()


# ---------------------------------------------------------------- W4A16 checkpoints (gtech 2026-10-20)

import json as _json
import pytest as _pytest

from .conftest import FIXTURES as _FIXTURES

_GIB = 1024**3


def _w4a16():
    return ModelFacts.from_config(load_model("qwen3.8-27b-w4a16"))


def test_a_w4a16_checkpoint_is_sized_like_its_files():
    """cyankiwi's W4A16 Qwen3.8-27B (compressed-tensors, int4 groups of 32 with zero points) is 19.57 GiB on disk,
    MTP head and vision tower included. It was sized at 27.7 GiB: its ignore list names each ``layers.N.linear_attn``
    module, which marked the whole attention role 16-bit, though the projections under it are packed int4."""
    index = _json.loads((_FIXTURES / "models" / "qwen3.8-27b-w4a16.index.json").read_text())
    on_disk = index["metadata"]["total_size"] / _GIB
    m = _w4a16()
    assert m.native_format == "int4"
    assert m.weight_bytes("native", include_mtp=True) / _GIB == _pytest.approx(on_disk, rel=0.05)


def test_int4_bytes_follow_the_group_size_and_zero_points():
    """4 bits + an fp16 scale and an int8 zero point per group of 32 = 0.594 bytes per parameter."""
    rb = _w4a16().native_role_bytes
    assert rb["dense_ffn"] == _pytest.approx((4 + (16 + 8) / 32) / 8, rel=1e-6)
    assert rb["attention"] == _pytest.approx(rb["dense_ffn"], rel=0.02)
    assert rb["embedding"] == rb["lm_head"] == 2.0


def test_a_role_listed_without_children_stays_16_bit():
    """A checkpoint that leaves every layer's MLP in 16-bit lists the containers themselves: those stay 16-bit."""
    config = load_model("qwen3.8-27b-w4a16")
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    qc = dict(config["quantization_config"])
    qc["ignore"] = [f"model.language_model.layers.{i}.mlp" for i in range(layers)]
    rb = ModelFacts.from_config({**config, "quantization_config": qc}).native_role_bytes
    assert rb["dense_ffn"] == 2.0 and rb["attention"] < 1.0


@_pytest.mark.parametrize(
    "name, gib",
    [("qwen3.8-27b-fp8", 28.31), ("qwen3-8b-fp8", 8.79), ("qwen3-30b-a3b-fp8", 29.03), ("gpt-oss-20b", 12.81)],
)
def test_other_quantized_checkpoints_keep_their_size(name, gib):
    """The FP8 and MXFP4 fixtures are sized as before."""
    assert ModelFacts.from_config(load_model(name)).weight_bytes("native") / _GIB == _pytest.approx(gib, abs=0.01)
