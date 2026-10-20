"""The engine-neutral settings vocabulary, v1 (FRD-027 §5.1; FRD-025 §5 ``settings``).

Modules decide these settings; an engine record maps each to that engine's argument, environment variable
or value transform, or marks it unsupported. This is the single source the JSON schema
``runtime/settings/v1.json`` is generated from, and that engine records are validated against.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

VERSION = 1

#: name -> {type, unit, semantics, enum?, minimum?}
VOCABULARY: Dict[str, Dict[str, Any]] = {
    "batch.max_running": {
        "type": "integer",
        "minimum": 1,
        "unit": "sequences",
        "semantics": "sequences decoded concurrently (the engine's batch limit, not the planned concurrency)",
    },
    "batch.prefill_chunk_tokens": {
        "type": "integer",
        "minimum": 1,
        "unit": "tokens",
        "semantics": "tokens one scheduler step may process (chunked prefill budget)",
    },
    "context.max_tokens": {
        "type": "integer",
        "minimum": 1,
        "unit": "tokens",
        "semantics": "longest request (prompt + output)",
    },
    "memory.device_fraction": {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 1,
        "unit": "fraction",
        "semantics": "device memory the engine may use; per-engine semantics live in the engine layer "
        "(vLLM: of the visible memory, or of the HAMi slice in shared mode)",
    },
    "weights.format": {
        "type": "string",
        "enum": ["bf16", "fp16", "fp8", "int8", "int4", "awq", "gptq", "mxfp4", "nvfp4"],
        "semantics": "weight format served: the checkpoint's own, a registry variant, or an online conversion",
    },
    "weights.online": {
        "type": "boolean",
        "semantics": "true when weights.format is an online conversion of the checkpoint",
    },
    "weights.variant": {
        "type": "string",
        "semantics": "registry id of the checkpoint variant served (changes the model path)",
    },
    "kv.dtype": {
        "type": "string",
        "enum": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        "semantics": "KV-cache dtype",
    },
    "state.dtype": {
        "type": "string",
        "enum": ["auto", "float32", "bfloat16", "float16"],
        "semantics": "recurrent (Mamba/linear-attention) state dtype",
    },
    "spec.method": {
        "type": "string",
        "enum": ["mtp", "eagle3", "ngram", "draft"],
        "semantics": "speculative decoding method",
    },
    "spec.tokens": {
        "type": "integer",
        "minimum": 1,
        "unit": "tokens",
        "semantics": "speculative tokens per step (k)",
    },
    "spec.draft": {
        "type": "string",
        "semantics": "draft model path (EAGLE3 / draft-model methods)",
    },
    "lora.max_adapters": {
        "type": "integer",
        "minimum": 0,
        "unit": "adapters",
        "semantics": "adapter slots resident at once",
    },
    "lora.max_rank": {
        "type": "integer",
        "minimum": 1,
        "semantics": "largest adapter rank a slot holds",
    },
    "prefix_cache.enabled": {"type": "boolean", "semantics": "prefix (radix) cache on"},
    "mm.limits": {"type": "object", "semantics": "multimodal items per request, by modality"},
    "mm.text_only": {
        "type": "boolean",
        "semantics": "serve a multimodal checkpoint for text only (skip the encoders)",
    },
    "mm.encoder_cache": {
        "type": "integer",
        "minimum": 0,
        "unit": "tokens",
        "semantics": "encoder cache budget",
    },
    "parallel.tensor": {"type": "integer", "minimum": 1, "semantics": "tensor-parallel degree"},
    "parallel.pipeline": {"type": "integer", "minimum": 1, "semantics": "pipeline-parallel degree"},
}

#: Keys an engine record's per-setting mapping may carry.
MAPPING_KEYS = {
    "arg",
    "env",
    "json_key",
    "implies",
    "invert",
    "negated_arg",
    "when",
    "unsupported",
    "transform",
    "note",
}


def is_setting(name: str) -> bool:
    return name in VOCABULARY


def validate_value(name: str, value: Any) -> None:
    """Raise ValueError when ``value`` is not a valid value of setting ``name``."""
    spec = VOCABULARY.get(name)
    if spec is None:
        raise ValueError(f"unknown setting {name!r}")
    kind = spec["type"]
    ok = {
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, Mapping),
    }[kind]
    if not ok:
        raise ValueError(f"setting {name!r} must be {kind}, got {value!r}")
    if "enum" in spec and value not in spec["enum"]:
        raise ValueError(f"setting {name!r} must be one of {spec['enum']}, got {value!r}")
    if "minimum" in spec and value < spec["minimum"]:
        raise ValueError(f"setting {name!r} must be >= {spec['minimum']}, got {value!r}")
    if "maximum" in spec and value > spec["maximum"]:
        raise ValueError(f"setting {name!r} must be <= {spec['maximum']}, got {value!r}")
    if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
        raise ValueError(f"setting {name!r} must be > {spec['exclusiveMinimum']}, got {value!r}")
