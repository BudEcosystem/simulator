"""Which architectures the layer-plan recogniser knows (FRD-027 FR-MDL-5).

An unknown ``model_type`` or layer type never fails a plan: it is costed as dense full attention, flagged
``architecture: unrecognised`` and grounded at the prior level, so it gets the capacity rules and nothing
numerics-changing. A new family lands as an entry here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..layer_plan import LAYER_KINDS

KNOWN_MODEL_TYPES = frozenset(
    {
        # dense decoders
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "phi",
        "phi3",
        "granite",
        "cohere",
        "cohere2",
        "olmo",
        "olmo2",
        "internlm2",
        "exaone",
        "starcoder2",
        "gpt2",
        "gpt_bigcode",
        "falcon",
        "stablelm",
        "baichuan",
        "chatglm",
        "glm",
        "glm4",
        "yi",
        "smollm3",
        # MoE
        "mixtral",
        "qwen2_moe",
        "qwen3_moe",
        "gpt_oss",
        "olmoe",
        "granitemoe",
        "dbrx",
        "phimoe",
        "jamba",
        "minimax",
        "minimax_m2",
        "hunyuan_v1_moe",
        "bailing_moe",
        "ling",
        "glm4_moe",
        "llama4",
        "llama4_text",
        # MLA
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "kimi_k2",
        # hybrids
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_6",
        "nemotron_h",
        "bamba",
        "granitemoehybrid",
        "falcon_h1",
        "zamba2",
        "lfm2",
        "mamba",
        "mamba2",
        "falcon_mamba",
        "hymba",
        "kimi_linear",
        "phi4flash",
        # multimodal wrappers whose text config is one of the above
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "llava",
        "llava_next",
        "mllama",
        "idefics3",
        "gemma3",
        # encoder-decoder
        "whisper",
        "t5",
        "bart",
    }
)

# The per-layer kind names the simulator models: the one table in layer_plan.
KNOWN_LAYER_TYPES = frozenset(LAYER_KINDS)


def _text(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def unknown_layer_types(layer_types: Iterable[Any]) -> list:
    return sorted({str(t) for t in layer_types or () if str(t).lower() not in KNOWN_LAYER_TYPES})


def recognised(config: Mapping[str, Any]) -> tuple:
    """``(recognised, reason)`` for a model config."""
    if not config:
        return False, "empty config"
    text = _text(config)
    model_type = str(config.get("model_type") or text.get("model_type") or "").lower()
    if (
        model_type not in KNOWN_MODEL_TYPES
        and str(text.get("model_type") or "").lower() not in KNOWN_MODEL_TYPES
    ):
        return False, f"model_type {model_type or '<missing>'!r} is not in the recogniser"
    bad = unknown_layer_types(text.get("layer_types") or ())
    if bad:
        return False, f"layer types {bad} are not in the recogniser"
    return True, ""
