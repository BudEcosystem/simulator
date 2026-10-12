"""KV cache dtype (FR-QUANT-2): fp8 where the deployment asks for it and the model and engine allow
it.

vLLM can store almost any model's KV cache in fp8: half the bytes per token, so every tier holds
twice the tokens and every load and store moves half the bytes. What decides it:

- **The engine build** lists ``kv_fp8`` (``KV_FP8_FEATURE``) once it has passed a long-context check
  on each GPU generation it serves: a 128K needle-in-haystack recall above 90% (ARCHITECTURE §13;
  the Hopper accumulation bug shows as recall below 50%). A kernel property, checked per build, not
  per model.
- **The model's architecture**, from its config: MLA latents need vLLM's own fp8 format, and
  recurrent (Mamba or linear-attention) state isn't KV, so neither is offered yet; sliding-window
  layers keep the model's dtype (``skip_layers_sliding_window``).
- **Calibration.** A checkpoint that ships calibrated KV scales (compressed-tensors
  ``kv_cache_scheme``, ModelOpt ``kv_cache_quant_algo``) was calibrated and checked by its
  publisher: fp8 is recommended. Without them vLLM uses a scale of 1.0, usually within 1-2 points on
  reasoning evals, with long-context recall most at risk: offered, not recommended.

The user decides at deploy time (``DeploymentSpec.kv_fp8``), from :func:`kv_fp8_support`, which
budsim shows in the deployment form.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from . import constants

RECURRENT_LAYER_TYPES = ("linear_attention", "mamba", "mamba2", "ssm", "recurrent")
RECURRENT_CONFIG_KEYS = (
    "mamba_d_state",
    "ssm_state_size",
    "linear_conv_kernel_dim",
    "mamba_expand",
)


def _text(model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = model_config.get("text_config")
    return text if isinstance(text, Mapping) else model_config


def calibrated_kv_scales(model_config: Mapping[str, Any]) -> bool:
    """Whether the checkpoint ships calibrated fp8 KV scales (compressed-tensors or ModelOpt)."""
    for config in (model_config, _text(model_config)):
        quant = config.get("quantization_config")
        if not isinstance(quant, Mapping):
            continue
        if quant.get("kv_cache_scheme"):
            return True
        if str(quant.get("kv_cache_quant_algo") or "").upper().startswith("FP8"):
            return True
    return False


def _sliding_window_layers(text: Mapping[str, Any]) -> bool:
    layer_types = text.get("layer_types")
    if isinstance(layer_types, (list, tuple)) and any("sliding" in str(t) for t in layer_types):
        return True
    return bool(text.get("sliding_window_pattern"))


def _recurrent_layers(text: Mapping[str, Any]) -> bool:
    layer_types = text.get("layer_types")
    if isinstance(layer_types, (list, tuple)) and any(
        str(t).lower() in RECURRENT_LAYER_TYPES for t in layer_types
    ):
        return True
    return any(text.get(key) is not None for key in RECURRENT_CONFIG_KEYS)


def kv_fp8_support(model_config: Mapping[str, Any], features: FrozenSet[str]) -> Dict[str, Any]:
    """Whether the deployment form may offer fp8 KV, whether to default it on, the dtype vLLM takes,
    and why.

    ``{"eligible", "recommended", "kv_cache_dtype", "skip_layers_sliding_window", "reason"}``.
    """
    text = _text(model_config)

    def out(
        eligible: bool, reason: str, recommended: bool = False, skip: bool = False
    ) -> Dict[str, Any]:
        return {
            "eligible": eligible,
            "recommended": eligible and recommended,
            "kv_cache_dtype": "fp8" if eligible else None,
            "skip_layers_sliding_window": eligible and skip,
            "reason": reason,
        }

    if constants.KV_FP8_FEATURE not in features:
        return out(
            False,
            f"the engine build doesn't list '{constants.KV_FP8_FEATURE}':"
            " fp8 KV isn't verified on it",
        )
    if text.get("kv_lora_rank"):
        return out(False, "MLA latents need vLLM's own fp8 format, which isn't offered yet")
    if _recurrent_layers(text):
        return out(
            False, "a hybrid model's recurrent state isn't KV cache; fp8 isn't offered for it yet"
        )
    skip = _sliding_window_layers(text)
    note = " Sliding-window layers keep the model's dtype." if skip else ""
    if calibrated_kv_scales(model_config):
        return out(
            True,
            "The checkpoint ships calibrated fp8 KV scales." + note,
            recommended=True,
            skip=skip,
        )
    return out(
        True,
        "No calibrated KV scales in the checkpoint: vLLM uses a scale of 1.0. Usually within 1-2"
        " points on reasoning evals; long-context recall is most at risk." + note,
        skip=skip,
    )


def choose_kv_dtype(
    model_config: Mapping[str, Any],
    device_generation: Optional[str],
    validated: FrozenSet[Tuple[str, str]] = frozenset(),
    *,
    requested: Optional[bool] = None,
    features: FrozenSet[str] = frozenset(),
) -> Tuple[str, str, bool]:
    """Return ``(kv_cache_dtype, decision, skip_layers_sliding_window)``.

    ``requested`` is the deployment's choice: True asks for fp8 (granted where
    :func:`kv_fp8_support` allows it), False keeps the model's dtype. None (not asked) keeps the
    model's dtype unless budeval or the package list validated the (model type, GPU generation)
    pair, on an engine build that lists ``kv_fp8``.
    """
    if requested is not None:
        if not requested:
            return "auto", "dtype: model default (fp8 KV off for this deployment)", False
        support = kv_fp8_support(model_config, features)
        if not support["eligible"]:
            return (
                "auto",
                f"dtype: model default (fp8 KV asked for, but {support['reason']})",
                False,
            )
        return (
            support["kv_cache_dtype"],
            f"dtype: fp8 (asked for at deploy time; {support['reason'].rstrip('.')})",
            support["skip_layers_sliding_window"],
        )
    model_type = str(model_config.get("model_type") or "")
    pair = (model_type, str(device_generation or ""))
    if pair in (validated | constants.FP8_KV_VALIDATED) and constants.KV_FP8_FEATURE in features:
        return "fp8", f"dtype: fp8 (validated for {model_type} on {device_generation})", False
    return (
        "auto",
        f"dtype: model default (no validated fp8 entry for {model_type or 'this model'})",
        False,
    )
