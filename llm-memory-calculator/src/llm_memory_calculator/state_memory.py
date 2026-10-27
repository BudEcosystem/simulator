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

from .mechanism_dims import first as _first, mamba_dims as _mamba_dims
from .layer_plan import (
    MECH_GDN,
    MECH_KDA,
    MECH_LIGHTNING,
    MECH_MAMBA1,
    MECH_MAMBA2,
    MECH_MLSTM,
    MECH_RGLRU,
    MECH_RWKV,
    MECH_SHORTCONV,
)

_FP32 = 4


def _state_dtype_bytes(config: Dict[str, Any]) -> int:
    """Bytes per recurrent-state element. fp32 unless the config overrides."""
    dt = str(
        _first(
            config, "mamba_ssm_cache_dtype", "mamba_ssm_dtype", "inference_state_dtype",
            default="float32",
        )
    ).lower()
    return {"float32": 4, "fp32": 4, "float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}.get(
        dt, 4
    )


def recurrent_state_shape(mech: str, config: Dict[str, Any], hidden_size: int) -> Dict[str, int]:
    """The state ONE layer of ``mech`` carries for one sequence, as element counts.

    ``heads x key_dim x value_dim`` is the recurrent matrix (a vector state is a
    matrix with key_dim 1), ``extra`` the small vectors some mechanisms keep
    beside it, ``conv`` the short convolution's ring buffer. State sizing and
    the performance graph both read this, so the two can never disagree.
    """
    if mech == MECH_GDN:
        # Gated DeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6). The delta-rule state is
        # per *value* head: keys are shared GQA-style across value heads, so the
        # count that sizes the matrix is linear_num_value_heads, not key heads.
        nk = int(_first(config, "linear_num_key_heads", default=16))
        nv = int(_first(config, "linear_num_value_heads", default=32))
        kd = int(_first(config, "linear_key_head_dim", default=128))
        vd = int(_first(config, "linear_value_head_dim", default=128))
        k = int(_first(config, "linear_conv_kernel_dim", default=4))
        # The conv runs over the concatenated q|k|v projection; its width matches
        # the shipped conv1d weight (verified [10240,1,4] on Qwen3.6-27B).
        return {"heads": nv, "key_dim": kd, "value_dim": vd, "extra": 0,
                "conv": (2 * nk * kd + nv * vd) * max(0, k - 1)}

    if mech == MECH_KDA:
        lac = config.get("linear_attn_config") or {}
        nh = int(_first(lac, "num_heads", default=32))
        hd = int(_first(lac, "head_dim", default=128))
        k = int(_first(lac, "short_conv_kernel_size", default=4))
        return {"heads": nh, "key_dim": hd, "value_dim": hd, "extra": 0,
                "conv": 3 * nh * hd * max(0, k - 1)}

    if mech == MECH_LIGHTNING:
        # Plain linear attention (MiniMax lightning): an n_heads x d x d summary.
        nh = int(_first(config, "num_attention_heads", default=32))
        hd = int(_first(config, "head_dim", default=max(1, hidden_size // max(nh, 1))))
        return {"heads": nh, "key_dim": hd, "value_dim": hd, "extra": 0, "conv": 0}

    if mech == MECH_SHORTCONV:
        # LFM2: a short conv cache only, no recurrent matrix.
        width = int(_first(config, "conv_L_cache", "conv_bias", default=3) or 3)
        return {"heads": 0, "key_dim": 0, "value_dim": 0, "extra": 0, "conv": hidden_size * width}

    if mech == MECH_RWKV:
        # RWKV: per layer, the token-shift vectors of the time-mix and channel-mix
        # blocks plus the WKV state. With a head size (RWKV-5/6/7, GGUF
        # `wkv.head_size`) the WKV state is a head_size x head_size matrix per
        # head; RWKV-4 (transformers) keeps three hidden-wide vectors instead.
        att = int(_first(config, "attention_hidden_size", default=hidden_size))
        head = _first(config, "head_size", "wkv_head_size")
        if head:
            heads = max(1, att // int(head))
            return {"heads": heads, "key_dim": int(head), "value_dim": int(head),
                    "extra": 2 * hidden_size, "conv": 0}
        return {"heads": 1, "key_dim": 1, "value_dim": 3 * att, "extra": 2 * hidden_size,
                "conv": 0}

    if mech == MECH_MLSTM:
        # xLSTM mLSTM cell: a qk_head x v_head matrix memory per head, plus the
        # normaliser vector and the max-stabiliser scalar.
        nh = int(_first(config, "num_heads", default=8))
        qk = _round64(hidden_size * float(_first(config, "qk_dim_factor", default=0.5)))
        v = _round64(hidden_size * float(_first(config, "v_dim_factor", default=1.0)))
        qkh, vh = qk // max(nh, 1), v // max(nh, 1)
        return {"heads": nh, "key_dim": qkh, "value_dim": vh, "extra": nh * (qkh + 1),
                "conv": 0}

    if mech == MECH_RGLRU:
        # RecurrentGemma RG-LRU: a diagonal recurrence, one lru_width vector,
        # beside a conv1d_width - 1 deep conv buffer of the same width.
        width = int(_first(config, "lru_width", default=hidden_size) or hidden_size)
        k = int(_first(config, "conv1d_width", default=4))
        return {"heads": 1, "key_dim": 1, "value_dim": width, "extra": 0,
                "conv": width * max(0, k - 1)}

    dims = _mamba_dims(config, hidden_size)
    if mech == MECH_MAMBA2:
        # Mamba-2 convolves the B and C group projections alongside x.
        return {"heads": dims["n_heads"], "key_dim": dims["d_state"], "value_dim": dims["d_head"],
                "extra": 0,
                "conv": (dims["d_inner"] + 2 * dims["n_groups"] * dims["d_state"])
                * max(0, dims["d_conv"] - 1)}
    # MECH_MAMBA1
    return {"heads": 1, "key_dim": dims["d_state"], "value_dim": dims["d_inner"], "extra": 0,
            "conv": dims["d_inner"] * max(0, dims["d_conv"] - 1)}


def _round64(x: float) -> int:
    return int(-(-int(round(x)) // 64) * 64)


def _per_layer_bytes(
    mech: str, config: Dict[str, Any], hidden_size: int, model_bytes: int
) -> Dict[str, float]:
    """Recurrent + conv bytes for ONE layer of the given mechanism, one sequence."""
    shape = recurrent_state_shape(mech, config, hidden_size)
    ssm_bytes = _state_dtype_bytes(config)
    recurrent = (shape["heads"] * shape["key_dim"] * shape["value_dim"] + shape["extra"]) * ssm_bytes
    return {"recurrent": float(recurrent), "conv": float(shape["conv"] * model_bytes)}


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
