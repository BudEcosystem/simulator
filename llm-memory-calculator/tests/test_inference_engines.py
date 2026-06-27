"""Tests for the inference-engine layer (runtime-correct memory on top of the base GenZ model).

The base ModelMemoryCalculator is calibrated for a layer-FREEING runtime (llama.cpp/vLLM): the prefill
activation peak is depth-independent (`seq*hidden*{10|15}`). ONNX Runtime serves the whole decoder graph
through ONE non-freeing BFC arena, so its prefill peak holds ~ALL layers' working sets at once — MEASURED
on a real llama-3.1-8B GB10 prefill to grow LINEARLY ~1.55 GB/1000 tok (3G@4K, 17G@12K, 31G@20K). This
layer adjusts the base per runtime WITHOUT modifying the calculator, so one simulator serves every engine.
"""
import pytest

from llm_memory_calculator.inference_engines import (
    InferenceEngine,
    LlamaCppEngine,
    OnnxRuntimeEngine,
    get_engine,
    estimate_memory_for_engine,
)

# The real llama-3.1-8B decoder shape (GQA, hidden 4096, 32 layers, vocab 128256).
LLAMA3_8B = {
    "model_type": "llama",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}

GB = 1e9


def _ort():
    return OnnxRuntimeEngine()


class TestEngineRegistry:
    def test_known_names_map_to_engines(self):
        assert isinstance(get_engine("onnxruntime"), OnnxRuntimeEngine)
        assert isinstance(get_engine("ort"), OnnxRuntimeEngine)
        assert isinstance(get_engine("onnx"), OnnxRuntimeEngine)
        assert isinstance(get_engine("llamacpp"), LlamaCppEngine)
        assert isinstance(get_engine("llama.cpp"), LlamaCppEngine)

    def test_unknown_or_none_falls_back_to_base_generic(self):
        # Unknown / None must NOT crash and must give the base (layer-freeing) model — never the ORT one.
        assert type(get_engine(None)) is InferenceEngine
        assert type(get_engine("vllm")) is InferenceEngine
        assert type(get_engine("totally-unknown-runtime")) is InferenceEngine


class TestOnnxRuntimeActivation:
    def test_matches_measured_8b_prefill_curve_safely(self):
        """Must be >= the MEASURED ORT prefill at each point (never under-count → never OOM admission),
        and within a sane multiple (not absurd over-reserve). Measured: 3G@4K, 17G@12K, 31G@20K."""
        eng = _ort()
        for seq, measured_g in [(4000, 3.0), (12000, 17.0), (20000, 31.0)]:
            act = eng.activation_bytes(LLAMA3_8B, batch_size=1, seq_length=seq,
                                       dtype_bytes=2, base_activation_bytes=0.0)
            act_g = act / GB
            assert act_g >= measured_g, f"seq={seq}: {act_g:.1f}G < measured {measured_g}G (would OOM)"
            assert act_g <= 3 * measured_g, f"seq={seq}: {act_g:.1f}G absurdly over {measured_g}G"

    def test_scales_linearly_with_seq(self):
        eng = _ort()
        a10 = eng.activation_bytes(LLAMA3_8B, 1, 10000, 2, 0.0)
        a20 = eng.activation_bytes(LLAMA3_8B, 1, 20000, 2, 0.0)
        # linear ⇒ doubling seq ~doubles activation (within 1% — the logits term is also linear in seq).
        assert abs(a20 / a10 - 2.0) < 0.01

    def test_scales_with_depth_unlike_base(self):
        """The whole point: ORT activation MUST grow with num_hidden_layers (the base does not)."""
        eng = _ort()
        shallow = {**LLAMA3_8B, "num_hidden_layers": 8}
        deep = {**LLAMA3_8B, "num_hidden_layers": 80}
        a_shallow = eng.activation_bytes(shallow, 1, 8000, 2, 0.0)
        a_deep = eng.activation_bytes(deep, 1, 8000, 2, 0.0)
        assert a_deep > a_shallow * 5, "ORT activation must scale with depth"

    def test_never_below_base(self):
        """The non-freeing arena can never need LESS than the layer-freeing peak."""
        eng = _ort()
        huge_base = 999 * GB
        act = eng.activation_bytes(LLAMA3_8B, 1, 100, 2, huge_base)
        assert act >= huge_base

    def test_batch_scales(self):
        eng = _ort()
        b1 = eng.activation_bytes(LLAMA3_8B, 1, 4000, 2, 0.0)
        b4 = eng.activation_bytes(LLAMA3_8B, 4, 4000, 2, 0.0)
        assert abs(b4 / b1 - 4.0) < 0.01

    def test_dtype_bytes_scales_activation(self):
        """Activation scales linearly with the runtime dtype size (fp16=2 ⇒ 4× an int4=0.5 cache)."""
        eng = _ort()
        fp16 = eng.activation_bytes(LLAMA3_8B, 1, 4000, 2, 0.0)
        int4 = eng.activation_bytes(LLAMA3_8B, 1, 4000, 0.5, 0.0)
        assert abs(fp16 / int4 - 4.0) < 0.01


class TestLlamaCppAndGenericAreBaseline:
    def test_llamacpp_is_identity_on_base(self):
        """llama.cpp (ggml graph reuse) IS the base calibration ⇒ the layer must not change it."""
        base = 12.3 * GB
        assert LlamaCppEngine().activation_bytes(LLAMA3_8B, 1, 20000, 2, base) == base

    def test_generic_is_identity_on_base(self):
        base = 7.7 * GB
        assert InferenceEngine().activation_bytes(LLAMA3_8B, 1, 20000, 2, base) == base


class TestEstimateMemoryForEngine:
    def test_onnx_total_is_weights_plus_kv_plus_ort_activation(self):
        """End-to-end through the base calculate_memory + the ORT activation override."""
        em = estimate_memory_for_engine(
            LLAMA3_8B, engine_name="onnxruntime", batch_size=1, seq_length=20000,
            weight_precision="fp16", runtime_precision="fp16")
        # weights ~16G, kv ~2.6G, ORT activation ~36G → total ~55G; and total == sum of parts.
        assert abs(em.total_bytes - (em.weights_bytes + em.kv_cache_bytes + em.activation_bytes)) < 1.0
        assert em.activation_bytes / GB >= 31.0  # ORT model, not the base 2.46G
        assert em.attention_type == "gqa"
        assert em.parameter_count > 7_000_000_000

    def test_weight_override_is_exact(self):
        em = estimate_memory_for_engine(
            LLAMA3_8B, engine_name="onnxruntime", batch_size=1, seq_length=4096,
            weight_precision="fp16", weight_bytes_override=16_094_134_272)
        assert em.weights_bytes == 16_094_134_272

    def test_llamacpp_uses_base_activation(self):
        """A llama.cpp estimate must equal the base model's activation (no ORT inflation)."""
        em_llama = estimate_memory_for_engine(
            LLAMA3_8B, engine_name="llamacpp", batch_size=1, seq_length=20000,
            weight_precision="int4", runtime_precision="fp16")
        em_ort = estimate_memory_for_engine(
            LLAMA3_8B, engine_name="onnxruntime", batch_size=1, seq_length=20000,
            weight_precision="int4", runtime_precision="fp16")
        # ORT activation must be much larger than the llama.cpp (base) activation at long context.
        assert em_ort.activation_bytes > em_llama.activation_bytes * 5
