"""KV cache dtype: fp8 only where validated for the model family on this GPU generation
(FR-QUANT-2)."""

from __future__ import annotations

from typing import Any, FrozenSet, Mapping, Optional, Tuple

from . import constants


def choose_kv_dtype(
    model_config: Mapping[str, Any],
    device_generation: Optional[str],
    validated: FrozenSet[Tuple[str, str]] = frozenset(),
) -> Tuple[str, str]:
    """Return ``(kv_cache_dtype, decision)``: ``fp8`` where budeval (or the package list) validated
    the pair, else the model's default."""
    model_type = str(model_config.get("model_type") or "")
    pair = (model_type, str(device_generation or ""))
    if pair in (validated | constants.FP8_KV_VALIDATED):
        return "fp8", f"dtype: fp8 (validated for {model_type} on {device_generation})"
    return "auto", f"dtype: model default (no validated fp8 entry for {model_type or 'this model'})"
