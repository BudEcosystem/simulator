"""Inference activation peak, sized the way serving engines actually run.

The previous model was ``batch x seq_length x hidden x 10..15``. That is a
*training* shape -- it prices keeping every layer's activations alive for a
backward pass -- and it is wrong for inference twice over:

1. **No engine ever materializes batch x seq tokens at once.** Every production
   serving stack chunks prefill (vLLM's ``--max-num-batched-tokens``, TRT-LLM's
   ``max_num_tokens``, SGLang's ``chunked_prefill_size``), so the tokens in
   flight are bounded by the chunk, not by the context. At 5 x 100k this was the
   difference between a claimed 76.8 GB and a real figure under 1 GB.
2. **Inference frees each layer's activations as it goes.** There is no
   retention across depth, so the peak is one layer's working set plus the
   residual stream -- not ``num_layers`` times anything.

What this does model: the residual stream, the widest single-layer working set
(attention projections vs the FFN intermediate, whichever is larger), and an
allocator slack factor. What it deliberately does not model: an S x S attention
matrix, because every engine worth sizing for uses FlashAttention-style tiling
and never forms one.
"""

from typing import Any, Dict, Optional

# vLLM V1's default --max-num-batched-tokens. Engines differ (TRT-LLM commonly
# 8192, SGLang 8192, older vLLM 2048/512) but they are all within a small factor,
# and every one of them is bounded -- which is the property that matters here.
DEFAULT_PREFILL_CHUNK_TOKENS = 8192

# Live hidden-width buffers at the peak: the residual stream, the block's input
# copy, and one normalization scratch.
_RESIDENT_HIDDEN_BUFFERS = 3

# Allocator slack: workspace, alignment, and the fact that a freed buffer is not
# instantly reusable. Deliberately modest -- the old model's 10-15x multiplier
# was standing in for retention that inference does not do.
_ALLOCATOR_SLACK = 1.2


def _get(config: Dict[str, Any], *keys, default=None):
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def prefill_tokens_in_flight(
    batch_size: int,
    seq_length: int,
    max_num_batched_tokens: Optional[int] = None,
) -> int:
    """Tokens resident in one forward pass.

    Bounded by the engine's prefill chunk. A request with a 100k context does not
    make a 100k-token forward pass; it makes ceil(100k/chunk) of them.
    """
    total = max(1, int(batch_size) * int(seq_length))
    chunk = int(max_num_batched_tokens or DEFAULT_PREFILL_CHUNK_TOKENS)
    return min(total, max(1, chunk))


def _attention_working_width(config: Dict[str, Any], hidden: int) -> int:
    """Widest concurrent attention-path buffer, in elements per token."""
    n_heads = int(_get(config, "num_attention_heads", "n_head", default=max(1, hidden // 128)))
    n_kv = int(_get(config, "num_key_value_heads", "num_kv_heads", default=n_heads))
    head_dim = int(_get(config, "head_dim", default=0) or max(1, hidden // max(1, n_heads)))

    q_dim = n_heads * head_dim
    # Qwen3.5/3.6 gate the attention output, so q_proj emits q and its gate.
    if config.get("attn_output_gate"):
        q_dim *= 2
    kv_dim = 2 * n_kv * head_dim
    # q/k/v live alongside the attention output before o_proj consumes them.
    return q_dim + kv_dim + n_heads * head_dim


def _ffn_working_width(config: Dict[str, Any], hidden: int) -> int:
    """Widest concurrent FFN buffer, in elements per token."""
    act = str(_get(config, "hidden_act", "activation_function", default="silu")).lower()
    gated = any(x in act for x in ("silu", "swish", "swiglu", "geglu", "glu"))
    fan = 2 if gated else 1  # gate and up are both live before down runs

    n_experts = _get(config, "n_routed_experts", "num_local_experts", "num_experts")
    top_k = _get(config, "expert_top_k", "num_experts_per_tok", "experts_per_token")
    moe_inter = _get(config, "moe_intermediate_size", "expert_intermediate_size")
    if n_experts and int(n_experts) > 1 and moe_inter and top_k:
        # Only the routed experts run per token, not all of them.
        return int(top_k) * int(moe_inter) * fan

    inter = int(_get(config, "intermediate_size", "ffn_dim", "d_ff", default=4 * hidden))
    return inter * fan


def _vision_activation_bytes(
    vision_config: Dict[str, Any], batch_size: int, dtype_bytes: float
) -> float:
    """One-shot encoder pass over a single image's patches. Small next to the LM."""
    if not vision_config:
        return 0.0
    v_hidden = int(_get(vision_config, "hidden_size", default=768))
    patch = int(_get(vision_config, "patch_size", default=16))
    image_size = int(_get(vision_config, "image_size", default=224))
    merge = int(_get(vision_config, "spatial_merge_size", default=1)) or 1
    num_patches = max(1, (image_size // max(1, patch)) ** 2 // (merge * merge))
    v_inter = int(_get(vision_config, "intermediate_size", default=4 * v_hidden))
    per_token = _RESIDENT_HIDDEN_BUFFERS * v_hidden + v_inter
    return num_patches * per_token * dtype_bytes * batch_size * _ALLOCATOR_SLACK


def calculate_activation_bytes(
    config: Dict[str, Any],
    batch_size: int,
    seq_length: int,
    dtype_bytes: float,
    max_num_batched_tokens: Optional[int] = None,
) -> float:
    """Peak inference activation bytes for one forward pass.

    Independent of ``num_hidden_layers`` by design: layer N's activations are
    freed before layer N+1 allocates, so depth does not multiply the peak.
    """
    text_config = config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else config
    vision_config = config.get("vision_config")
    vision_config = vision_config if isinstance(vision_config, dict) else {}

    hidden = int(_get(text_config, "hidden_size", "d_model", default=768))
    tokens = prefill_tokens_in_flight(batch_size, seq_length, max_num_batched_tokens)

    per_token = _RESIDENT_HIDDEN_BUFFERS * hidden + max(
        _attention_working_width(text_config, hidden),
        _ffn_working_width(text_config, hidden),
    )
    text_bytes = tokens * per_token * dtype_bytes * _ALLOCATOR_SLACK

    return text_bytes + _vision_activation_bytes(vision_config, batch_size, dtype_bytes)
