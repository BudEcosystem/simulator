"""Per-sequence recurrent and convolutional state for hybrid/SSM models.

The defining property of every mechanism here is that its state is **constant in
sequence length**. That is the whole reason these architectures exist: a Gated
DeltaNet layer costs the same 3.1 MB per sequence at 1k context as at 262k, where
a full-attention layer of the same width would want 16 GB of KV. A memory model
that reports zero for this term gets the trade exactly backwards -- it inflates
the part that does not grow and drops the part that does not shrink.

Two terms per layer:

* **recurrent state** -- the running matrix/vector the mechanism carries forward.
  Held in fp32: it accumulates over the whole sequence and loses too much
  precision at bf16, so vLLM pins ``mamba_ssm_cache_dtype=float32``. Pricing it
  at model precision under-counts by 2x.
* **conv state** -- the short causal convolution's ring buffer, ``d_conv - 1``
  previous inputs wide (you need the last ``d_conv-1`` to produce the next
  output). Held at model precision. Small per layer, but it was missing
  entirely, and on wide Mamba-2 stacks it is not negligible.

Shapes are keyed by mechanism rather than by vendor because several vendors share
one: Nemotron-H, Bamba, granite-4.0-h and Mamba-2 proper all carry the same
``n_heads x d_head x d_state`` matrix under four different sets of config keys.
"""

from typing import Any, Dict, Optional

from .layer_plan import (
    MECH_GDN,
    MECH_KDA,
    MECH_LIGHTNING,
    MECH_MAMBA1,
    MECH_MAMBA2,
    MECH_SHORTCONV,
)

_FP32 = 4


def _first(config: Dict[str, Any], *keys, default=None):
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def _state_dtype_bytes(config: Dict[str, Any]) -> int:
    """Bytes per recurrent-state element. fp32 unless the config overrides."""
    dt = str(
        _first(config, "mamba_ssm_cache_dtype", "mamba_ssm_dtype", default="float32")
    ).lower()
    return {"float32": 4, "fp32": 4, "float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}.get(
        dt, 4
    )


def _mamba_dims(config: Dict[str, Any], hidden_size: int) -> Dict[str, int]:
    """Resolve Mamba dims across the four key dialects in circulation.

    ``state_size`` (Mamba proper), ``mamba_d_state`` (Jamba, Falcon-H1, granite,
    Bamba, Zamba2) and ``ssm_state_size`` (Nemotron-H) all mean d_state. Missing
    aliases used to fall through to a hardcoded 16, which is off by 16x on a
    Falcon-H1 that ships 256.
    """
    d_state = int(_first(config, "state_size", "d_state", "mamba_d_state", "ssm_state_size", default=16))
    expand = int(_first(config, "expand", "expand_factor", "mamba_expand", "ssm_expand", default=2))
    d_conv = int(_first(config, "conv_kernel", "d_conv", "mamba_d_conv", "ssm_conv_kernel", default=4))
    n_groups = int(_first(config, "n_groups", "mamba_n_groups", "ssm_n_groups", default=1))

    n_heads = _first(config, "mamba_n_heads", "mamba_num_heads", "n_mamba_heads")
    d_head = _first(config, "mamba_d_head", "mamba_head_dim")

    if n_heads and d_head:
        d_inner = int(n_heads) * int(d_head)
    else:
        d_inner = expand * hidden_size
        if n_heads:
            d_head = max(1, d_inner // int(n_heads))
        else:
            # Mamba-2 proper puts head counts under the generic names; only trust
            # them when they are consistent with d_inner, since `num_heads` also
            # means attention heads on hybrid configs.
            nh, dh = config.get("num_heads"), config.get("head_dim")
            if nh and dh and int(nh) * int(dh) == d_inner:
                n_heads, d_head = int(nh), int(dh)
            else:
                n_heads, d_head = 1, d_inner

    return {
        "d_state": d_state,
        "d_conv": d_conv,
        "n_groups": n_groups,
        "d_inner": d_inner,
        "n_heads": int(n_heads),
        "d_head": int(d_head),
    }


def _per_layer_bytes(
    mech: str, config: Dict[str, Any], hidden_size: int, model_bytes: int
) -> Dict[str, float]:
    """Recurrent + conv bytes for ONE layer of the given mechanism, one sequence."""
    ssm_bytes = _state_dtype_bytes(config)

    if mech == MECH_GDN:
        # Gated DeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6). The delta-rule state is
        # per *value* head: keys are shared GQA-style across value heads, so the
        # count that sizes the matrix is linear_num_value_heads, not key heads.
        nk = int(_first(config, "linear_num_key_heads", default=16))
        nv = int(_first(config, "linear_num_value_heads", default=32))
        kd = int(_first(config, "linear_key_head_dim", default=128))
        vd = int(_first(config, "linear_value_head_dim", default=128))
        k = int(_first(config, "linear_conv_kernel_dim", default=4))
        recurrent = nv * kd * vd * ssm_bytes
        # The conv runs over the concatenated q|k|v projection; its width matches
        # the shipped conv1d weight (verified [10240,1,4] on Qwen3.6-27B).
        conv = (2 * nk * kd + nv * vd) * max(0, k - 1) * model_bytes
        return {"recurrent": recurrent, "conv": conv}

    if mech == MECH_KDA:
        lac = config.get("linear_attn_config") or {}
        nh = int(_first(lac, "num_heads", default=32))
        hd = int(_first(lac, "head_dim", default=128))
        k = int(_first(lac, "short_conv_kernel_size", default=4))
        recurrent = nh * hd * hd * ssm_bytes
        conv = 3 * nh * hd * max(0, k - 1) * model_bytes
        return {"recurrent": recurrent, "conv": conv}

    if mech == MECH_LIGHTNING:
        # MiniMax lightning attention keeps an n_heads x d x d summary matrix.
        nh = int(_first(config, "num_attention_heads", default=32))
        hd = int(_first(config, "head_dim", default=128))
        return {"recurrent": nh * hd * hd * ssm_bytes, "conv": 0.0}

    if mech == MECH_SHORTCONV:
        # LFM2: a short conv cache only, no recurrent matrix.
        width = int(_first(config, "conv_L_cache", "conv_bias", default=3) or 3)
        return {"recurrent": 0.0, "conv": hidden_size * width * model_bytes}

    dims = _mamba_dims(config, hidden_size)
    if mech == MECH_MAMBA2:
        recurrent = dims["n_heads"] * dims["d_head"] * dims["d_state"] * ssm_bytes
        # Mamba-2 convolves the B and C group projections alongside x.
        conv_width = dims["d_inner"] + 2 * dims["n_groups"] * dims["d_state"]
    else:  # MECH_MAMBA1
        recurrent = dims["d_inner"] * dims["d_state"] * ssm_bytes
        conv_width = dims["d_inner"]
    conv = conv_width * max(0, dims["d_conv"] - 1) * model_bytes
    return {"recurrent": recurrent, "conv": conv}


def calculate_recurrent_state_bytes(
    config: Dict[str, Any],
    plan: Optional[Dict[str, Any]],
    batch_size: int,
    model_bytes: int = 2,
) -> float:
    """Total recurrent + conv state in bytes for ``batch_size`` sequences.

    ``plan`` is a layer plan from :mod:`layer_plan`; ``None`` or an
    attention-only plan yields 0.0. Independent of sequence length by
    construction -- there is deliberately no seq_length parameter.
    """
    if not plan or not plan.get("num_recurrent_layers"):
        return 0.0

    tc = config.get("text_config")
    tc = tc if isinstance(tc, dict) else config
    hidden_size = int(_first(tc, "hidden_size", "d_model", default=768))

    total = 0.0
    for mech in plan["recurrent"]:
        if not mech:
            continue
        parts = _per_layer_bytes(mech, tc, hidden_size, model_bytes)
        total += parts["recurrent"] + parts["conv"]
    return total * batch_size
