"""Inference-engine layer — RUNTIME-correct memory on top of the base GenZ / calculator model.

The base :class:`ModelMemoryCalculator` (and the GenZ perf model) are calibrated for a layer-FREEING
serving runtime — llama.cpp's ggml graph reuses one compute buffer across layers, vLLM frees per layer —
so the prefill activation peak is depth-INDEPENDENT (``seq * hidden * {10|15}``). Different runtimes
allocate differently; the load-bearing case is **ONNX Runtime**, which executes the whole decoder graph
through ONE non-freeing BFC arena, so its prefill peak holds ~ALL layers' working sets simultaneously.
MEASURED on a real llama-3.1-8B GB10 prefill, ORT's transient memory grows LINEARLY ~1.55 GB / 1000
tokens (3 GB@4K, 17 GB@12K, 31 GB@20K) — the depth-independent base under-counts that ~13x at 20K and
would OOM admission.

This module is a thin LAYER: it calls the base estimator unchanged, then overrides the runtime-specific
part (the activation) per engine. Weights and KV are runtime-independent and come straight from the base.
One simulator therefore serves every engine Bud Gaia runs (ONNX Runtime, llama.cpp, and future runtimes
added here), with NO per-call hacks in the callers and NO change to the calibrated base calculator.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .utils import calculate_memory

# Runtime dtype sizes (bytes) for the KV/activation working set — independent of the weight quant.
_DTYPE_BYTES = {
    "fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
    "int8": 1, "fp8": 1, "int4": 0.5,
}


def _dtype_bytes(precision: str) -> float:
    return _DTYPE_BYTES.get(str(precision).lower(), 2)


@dataclass
class EngineMemory:
    """Runtime-correct memory breakdown (bytes), with the engine that produced it disclosed."""
    weights_bytes: float
    kv_cache_bytes: float
    activation_bytes: float
    total_bytes: float
    attention_type: Optional[str]
    parameter_count: int
    engine: str


class InferenceEngine:
    """Base engine == the calibrated base model (a layer-FREEING runtime: llama.cpp ggml / vLLM).

    Subclass and override :meth:`activation_bytes` (and later :meth:`adjust_slo`) to model a runtime
    whose allocator differs from the base. The base itself is identity, so an unknown runtime degrades
    safely to the (well-calibrated) base rather than to a wrong guess.
    """

    name = "generic"

    def activation_bytes(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        dtype_bytes: float,
        base_activation_bytes: float,
    ) -> float:
        """Prefill activation peak (bytes). Default: the base model's depth-independent value."""
        return base_activation_bytes

    def adjust_slo(self, slo: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for runtime-specific perf. Default: identity (the base GenZ roofline is used as-is —
        validated safe + TPOT-accurate for ONNX Runtime; tighten here per runtime when measured)."""
        return slo


class LlamaCppEngine(InferenceEngine):
    """llama.cpp (ggml): the base GenZ calibration *is* llama.cpp on the GB10 ⇒ identity over the base."""

    name = "llamacpp"


class OnnxRuntimeEngine(InferenceEngine):
    """ONNX Runtime: one non-freeing BFC arena ⇒ the prefill peak holds ~ALL layers' working sets.

    Per token ≈ ``layers * hidden * k + vocab`` (k≈6 live hidden-sized buffers/layer: residual + Q,K,V +
    attn_out + MLP gate/up/down; plus the logits buffer over all positions), times the runtime dtype.
    """

    name = "onnxruntime"
    # k: measured ~5.4 net of logits on llama-3.1-8B (GB10); 6 is a safe over-estimate that also covers
    # the >20K extrapolation so admission never under-counts (S081 no-OOM).
    LAYER_WORK = 6

    def activation_bytes(self, config, batch_size, seq_length, dtype_bytes, base_activation_bytes):
        hidden = config.get("hidden_size", config.get("d_model", 768))
        layers = config.get("num_hidden_layers", config.get("n_layers", 12))
        vocab = config.get("vocab_size", 32000)
        per_token = layers * hidden * self.LAYER_WORK + vocab
        ort = batch_size * seq_length * per_token * dtype_bytes
        # The non-freeing arena can never need LESS than the layer-freeing peak.
        return max(float(ort), float(base_activation_bytes))


_ENGINES = {
    "onnxruntime": OnnxRuntimeEngine, "ort": OnnxRuntimeEngine, "onnx": OnnxRuntimeEngine,
    "llamacpp": LlamaCppEngine, "llama.cpp": LlamaCppEngine, "ggml": LlamaCppEngine, "gguf": LlamaCppEngine,
    "generic": InferenceEngine, "vllm": InferenceEngine,
}


def get_engine(name: Optional[str]) -> InferenceEngine:
    """Resolve a runtime name to an :class:`InferenceEngine`. Unknown/``None`` ⇒ the safe base engine."""
    if not name:
        return InferenceEngine()
    return _ENGINES.get(str(name).lower(), InferenceEngine)()


def _resolve_config(model: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The raw config dict (for the activation dims). A dict passes through; a HuggingFace id is
    resolved the same way the base calculator resolves it."""
    if isinstance(model, dict):
        return model
    from .huggingface_loader import HuggingFaceConfigLoader

    return HuggingFaceConfigLoader().get_model_config(model)


def estimate_memory_for_engine(
    model: Union[str, Dict[str, Any]],
    engine_name: Optional[str],
    batch_size: int = 1,
    seq_length: int = 2048,
    weight_precision: str = "fp16",
    runtime_precision: str = "fp16",
    weight_bytes_override: Optional[float] = None,
) -> EngineMemory:
    """Architecture-aware, RUNTIME-correct memory for ``model`` served on ``engine_name``.

    Calls the base :func:`calculate_memory` (weights at ``weight_precision``, KV+activation at
    ``runtime_precision``) then overrides the activation with the engine's model. Weights and KV are
    runtime-independent; only the activation is runtime-specific. ``weight_bytes_override`` (e.g. the
    exact on-disk byte sum of a GGUF/ONNX dir) wins over the analytic weight estimate.
    """
    engine = get_engine(engine_name)
    config = _resolve_config(model)
    rep_w = calculate_memory(model, batch_size=batch_size, seq_length=seq_length, precision=weight_precision)
    rep_rt = calculate_memory(model, batch_size=batch_size, seq_length=seq_length, precision=runtime_precision)
    weights = float(weight_bytes_override) if weight_bytes_override else float(rep_w.weight_memory_bytes)
    kv = float(rep_rt.kv_cache_bytes)
    activation = float(
        engine.activation_bytes(
            config, batch_size, seq_length, _dtype_bytes(runtime_precision),
            float(rep_rt.activation_memory_bytes),
        )
    )
    total = weights + kv + activation
    return EngineMemory(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        activation_bytes=activation,
        total_bytes=total,
        attention_type=rep_rt.attention_type,
        parameter_count=int(rep_rt.parameter_count),
        engine=engine.name,
    )
