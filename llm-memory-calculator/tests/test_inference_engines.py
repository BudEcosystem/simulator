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


# Mixtral-8x7B: a real sparse-MoE (8 experts, top-2 routing, MoE intermediate 14336).
MIXTRAL_8X7B = {
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 32000,
}


class TestMoEActivation:
    def test_moe_adds_active_experts_above_a_dense_baseline(self):
        """ORT's non-freeing arena holds the k ACTIVE experts' MLP intermediates per layer — so a sparse
        MoE's activation must EXCEED a dense model of the same hidden/layers (which has no experts)."""
        eng = _ort()
        dense = {k: v for k, v in MIXTRAL_8X7B.items()
                 if k not in ("num_local_experts", "num_experts_per_tok")}
        a_moe = eng.activation_bytes(MIXTRAL_8X7B, 1, 4000, 2, 0.0)
        a_dense = eng.activation_bytes(dense, 1, 4000, 2, 0.0)
        assert a_moe > a_dense, "MoE must add the active-expert activation"

    def test_moe_scales_with_k_not_n(self):
        """Activation depends on k ACTIVE experts, not N total (weights hold N, the arena holds k)."""
        eng = _ort()
        a_k2 = eng.activation_bytes(MIXTRAL_8X7B, 1, 4000, 2, 0.0)
        a_k4 = eng.activation_bytes({**MIXTRAL_8X7B, "num_experts_per_tok": 4}, 1, 4000, 2, 0.0)
        a_n16 = eng.activation_bytes({**MIXTRAL_8X7B, "num_local_experts": 16}, 1, 4000, 2, 0.0)
        assert a_k4 > a_k2, "more active experts ⇒ more activation"
        assert a_n16 == a_k2, "more TOTAL experts (same k) ⇒ SAME activation (arena holds only k)"

    def test_moe_expert_key_variants_normalize(self):
        eng = _ort()
        a1 = eng.activation_bytes(MIXTRAL_8X7B, 1, 4000, 2, 0.0)
        v2 = {**MIXTRAL_8X7B}
        v2["num_experts"] = v2.pop("num_local_experts")
        v2["expert_top_k"] = v2.pop("num_experts_per_tok")
        a2 = eng.activation_bytes(v2, 1, 4000, 2, 0.0)
        assert abs(a1 - a2) < 1.0, "num_experts/expert_top_k aliases must give the same result"


class TestMultimodalActivation:
    """A vision/audio LLM nests its DECODER dims under `text_config`; the ORT arena's dominant prefill
    term is the decoder's all-layers working set, so the layer must read text_config — not mis-read the
    multimodal wrapper as a tiny default decoder (which would massively under-count the activation)."""

    def test_reads_decoder_dims_from_text_config(self):
        eng = _ort()
        mm = {
            "model_type": "llava",
            "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
            "text_config": {
                "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000,
                "intermediate_size": 11008,
            },
        }
        decoder_only = {
            "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        a_mm = eng.activation_bytes(mm, 1, 4000, 2, 0.0)
        a_dec = eng.activation_bytes(decoder_only, 1, 4000, 2, 0.0)
        assert a_mm == a_dec, "must use the text_config decoder dims, not the multimodal wrapper"
        # NOT the tiny default (12 layers / 768 hidden) it would read from the bare wrapper.
        bare = eng.activation_bytes({"model_type": "llava"}, 1, 4000, 2, 0.0)
        assert a_mm > bare * 5, "multimodal decoder activation must reflect the real (large) decoder"

    def test_multimodal_moe_reads_experts_from_text_config(self):
        """A multimodal MoE (experts nested in text_config) still counts the active experts."""
        eng = _ort()
        mm_moe = {
            "model_type": "multimodal",
            "vision_config": {"hidden_size": 1024},
            "text_config": {
                "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000,
                "num_local_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 14336,
            },
        }
        mm_dense = {
            "model_type": "multimodal",
            "vision_config": {"hidden_size": 1024},
            "text_config": {"num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000},
        }
        assert eng.activation_bytes(mm_moe, 1, 4000, 2, 0.0) > eng.activation_bytes(mm_dense, 1, 4000, 2, 0.0)


class TestChunkedPrefillDoesNotReduceMemory:
    """MEASURED on an 8B GB10 prefill: genai `search.chunk_size` off/4096/512 ALL held ~31 GB — the
    non-freeing BFC arena grows to the full-prompt high-water mark regardless of the chunk (each chunk's
    attention still spans the growing KV). So the activation estimate must NOT shrink with a chunk — a
    chunk-reduced estimate would under-count and OOM admission. This guards against re-introducing it."""

    def test_activation_is_full_seq_independent_of_any_chunk_notion(self):
        eng = _ort()
        # The signature takes NO chunk_size: the activation is always the whole-prompt peak.
        a20 = eng.activation_bytes(LLAMA3_8B, 1, 20000, 2, 0.0)
        a4 = eng.activation_bytes(LLAMA3_8B, 1, 4000, 2, 0.0)
        assert a20 / a4 > 4.9, "20K activation must be ~5× the 4K (full-seq linear, never chunk-bounded)"
