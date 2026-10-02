"""The KV namespace every shared tier is partitioned by (FR-ISO-3).

vLLM's block hash covers tokens, LoRA, multimodal hashes and the tenant salt, but not the model. In
a shared tier two models could produce the same hash, so the root path (fs) or key prefix (the
Mooncake pool's ``cache_prefix``; vLLM's default key holds only the model directory's basename)
carries this namespace.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def kv_namespace(
    *,
    model_id: Optional[str],
    revision: Optional[str],
    weight_quantization: Optional[str],
    kv_cache_dtype: str,
    block_size: int,
    tensor_parallel: int,
    pipeline_parallel: int,
    attention_layout: str,
    hash_algo: str,
    engine_version: Optional[str],
) -> Optional[str]:
    """``kvns-`` plus 16 hex digits of a SHA-256 over everything that changes the KV layout; None
    without a model id."""
    if not model_id:
        return None
    fields: Any = {
        "model_id": model_id,
        "revision": revision,
        "weight_quantization": weight_quantization,
        "kv_cache_dtype": kv_cache_dtype,
        "block_size": int(block_size),
        "tp": int(tensor_parallel),
        "pp": int(pipeline_parallel),
        "attention_layout": attention_layout,
        "hash_algo": hash_algo,
        "engine_version": engine_version,
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return "kvns-" + digest[:16]
