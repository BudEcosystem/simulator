"""Parameters of the stacks around a decoder: an encoder-decoder's encoder, and the adapter stacks
a multimodal model puts between its tower and its language model (BLIP-2's Q-Former).

Read from the stack's own geometry, as ``layer_plan.encoder_view`` and ``layer_plan.encoder_stages``
give it -- never from the decoder's: Donut's Swin encoder runs 20 blocks over four widths
(128 to 1,024), and was sized as 4 blocks at the decoder's 1,024.

Every count is of the matrices and the vectors beside them (biases, norms, a windowed tower's
relative-position tables) as transformers' modelling builds them (5.12), so a reader that adds
these to a decoder's weights holds at least what the checkpoint holds.
"""

from typing import Any, Dict, Optional

from .layer_plan import (
    _pos_int,
    _text_config,
    attention_head_dim,
    encoder_view,
    ffn_activation,
    gated_ffn_by_family,
    stated_gated_ffn,
)

#: Gated activations (three FFN matrices: gate, up, down) by name.
_GATED_ACTIVATIONS = ("silu", "swiglu", "geglu", "glu", "swish")


def _gated(view: Dict[str, Any]) -> bool:
    stated = stated_gated_ffn(view)
    if stated is not None:
        return stated
    if gated_ffn_by_family(view):
        return True
    act = ffn_activation(view)
    return any(a in act for a in _GATED_ACTIVATIONS)


#: Spellings a tower or stack states its FFN width under.
_FFN_WIDTH_KEYS = ("intermediate_size", "d_ff", "ffn_dim", "encoder_ffn_dim", "mlp_dim")


def tower_view(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A tower's or stack's block (a `vision_config`, an `audio_config`) under the names
    ``stack_params`` reads: its width, depth, heads, head width (`d_kv`) and FFN width under
    any of their spellings, and its stages when it is hierarchical. None for anything else."""
    from .layer_plan import ATTENTION_HEAD_KEYS, ENCODER_HEAD_KEYS, _block_depth, _first, encoder_stages

    if not isinstance(block, dict):
        return None
    view = {k: v for k, v in block.items() if not isinstance(v, dict)}
    width = _pos_int(_first(block, "hidden_size", "d_model", "embed_dim", "n_embd"))
    if width:
        view["hidden_size"] = width
    view["num_hidden_layers"] = _block_depth(block)
    # a Whisper-shaped audio tower (Qwen2-Audio's, MOSS-Transcribe's `audio_config`) states its
    # heads as an encoder's
    heads = (_pos_int(_first(block, *ATTENTION_HEAD_KEYS)) or _pos_int(block.get("num_heads"))
             or _pos_int(_first(block, *ENCODER_HEAD_KEYS)))
    if heads:
        view["num_attention_heads"] = heads
    ffn = _pos_int(_first(block, *_FFN_WIDTH_KEYS))
    if ffn:
        view["intermediate_size"] = ffn
    if _pos_int(block.get("d_kv")) and not _pos_int(block.get("head_dim")):
        view["head_dim"] = int(block["d_kv"])
    stages = encoder_stages(block)
    if stages:
        view["stages"] = stages
    return view


def _block(h: int, heads: int, kv_heads: int, head_dim: int, ffn: int, gated: bool) -> int:
    """One transformer block: Q / K / V / O with their biases, the FFN with its biases, two
    norms (weight and bias)."""
    q = h * heads * head_dim + heads * head_dim
    kv = 2 * (h * kv_heads * head_dim + kv_heads * head_dim)
    o = heads * head_dim * h + h
    mlp = (3 if gated else 2) * h * ffn + ffn + h
    return q + kv + o + mlp + 4 * h


def stack_params(view: Optional[Dict[str, Any]]) -> int:
    """The parameters of one stack, given under the decoder-only names (``encoder_view``): a
    uniform stack's blocks, or a hierarchical tower's stages (``stages``) with the 2 x 2 patch
    merging between them, its patch embedding and its windows' relative-position tables. 0 for
    None."""
    if not isinstance(view, dict):
        return 0
    gated = _gated(view)
    stages = view.get("stages")
    if stages:
        total = 0
        window = _pos_int(view.get("window_size"))
        for i, st in enumerate(stages):
            c, heads = st["hidden_size"], st["num_attention_heads"]
            hd = max(c // heads, 1)
            total += st["layers"] * _block(c, heads, heads, hd, st["intermediate_size"], gated)
            if window:
                total += st["layers"] * (2 * window - 1) ** 2 * heads
            if i + 1 < len(stages):
                total += 4 * c * 2 * c + 2 * 4 * c  # PatchMerging: Linear(4c, 2c), LayerNorm(4c)
        c0 = stages[0]["hidden_size"]
        patch = view.get("patch_size")
        patch = patch if _pos_int(patch) else 4
        channels = _pos_int(view.get("num_channels")) or 3
        total += channels * patch * patch * c0 + c0 + 2 * c0  # patch embedding and its norm
        return int(total)
    layers = _pos_int(view.get("num_hidden_layers")) or 0
    h = _pos_int(view.get("hidden_size")) or 0
    heads = _pos_int(view.get("num_attention_heads")) or 0
    if not (layers and h and heads):
        return 0
    kv_heads = _pos_int(view.get("num_key_value_heads")) or heads
    hd = attention_head_dim(view, heads, h) or max(h // heads, 1)
    ffn = _pos_int(view.get("intermediate_size")) or 4 * h
    per_layer = _block(h, heads, kv_heads, hd, ffn, gated)
    kernel = _pos_int(view.get("conv_kernel_size"))
    if kernel:
        # a conformer block (its stack states a convolution module: Parakeet / FastConformer,
        # Wav2Vec2-Conformer): a second, macaron FFN, and the convolution module -- a pointwise
        # projection to two widths (GLU), a depthwise convolution, a pointwise projection back,
        # their biases and a norm. Counted as a plain block, cohere-transcribe's 48-layer encoder
        # was half its 1.89B.
        per_layer += (3 if gated else 2) * h * ffn + ffn + h + 2 * h
        per_layer += h * 2 * h + 2 * h + h * kernel + h + h * h + h + 4 * h
    return int(layers * per_layer + 2 * h)


def tower_embedding_params(view: Optional[Dict[str, Any]]) -> int:
    """What a tower holds before its first block, from its own keys (transformers 5.12): a patch
    encoder whose length is its `seq_len` (Pix2Struct: a linear patch projection from
    `patch_embed_hidden_size`, and row and column position tables of `seq_len` each); a ViT's
    convolutional patch embedding with its class token and a position per patch; an audio
    encoder's two convolutions over `num_mel_bins` and a position per frame
    (`max_source_positions`: Whisper). 0 for a text encoder (its token table is counted as the
    embedding) and for a hierarchical tower (``stack_params`` counts its patch embedding)."""
    if not isinstance(view, dict):
        return 0
    stages = view.get("stages")
    if stages:
        # a hierarchical tower: its patch embedding and its windows' relative-position tables
        # (the blocks and the merges between stages are graphed as operators)
        c0 = stages[0]["hidden_size"]
        patch = _pos_int(view.get("patch_size")) or 4
        channels = _pos_int(view.get("num_channels")) or 3
        window = _pos_int(view.get("window_size")) or 0
        tables = sum(st["layers"] * (2 * window - 1) ** 2 * st["num_attention_heads"]
                     for st in stages) if window else 0
        return channels * patch * patch * c0 + c0 + tables
    h = _pos_int(view.get("hidden_size")) or 0
    if not h:
        return 0
    seq_len = _pos_int(view.get("seq_len"))
    if seq_len:
        patch_in = _pos_int(view.get("patch_embed_hidden_size")) or h
        return 2 * seq_len * h + patch_in * h + h
    mels = _pos_int(view.get("num_mel_bins"))
    if mels:
        frames = _pos_int(view.get("max_source_positions")) or 0
        return mels * h * 3 + h + h * h * 3 + h + frames * h
    size, patch = view.get("image_size"), view.get("patch_size")
    if _pos_int(size) and _pos_int(patch):
        channels = _pos_int(view.get("num_channels")) or 3
        patches = (int(size) // int(patch)) ** 2
        return channels * int(patch) ** 2 * h + h + (patches + 1) * h + h
    return 0


def encoder_params(config: Dict[str, Any]) -> int:
    """An encoder-decoder's encoder stack (``layer_plan.encoder_view``), 0 for any other config."""
    return stack_params(encoder_view(config)) if isinstance(config, dict) else 0


#: Adapter stacks between a tower and the language model, by the key their config sits under:
#: BLIP-2's and InstructBLIP's Q-Former (`qformer_config`), a BERT-like stack whose every
#: `cross_attention_frequency`-th layer cross-attends to the tower (`encoder_hidden_size` wide),
#: fed `num_query_tokens` learned queries and projected to the language model's width.
ADAPTER_KEYS = ("qformer_config",)


def adapter_params(config: Dict[str, Any]) -> int:
    """The adapter stacks a config states (``ADAPTER_KEYS``), 0 when it states none. BLIP-2's
    Q-Former and its projection are 106.7M parameters the FRD-027 facts left out."""
    if not isinstance(config, dict):
        return 0
    total = 0
    text_width = _pos_int(_first_width(_text_config(config)))
    for key in ADAPTER_KEYS:
        block = config.get(key)
        if not isinstance(block, dict):
            continue
        layers = _pos_int(block.get("num_hidden_layers")) or 0
        h = _pos_int(block.get("hidden_size")) or 0
        if not (layers and h):
            continue
        total += stack_params(block)
        every = _pos_int(block.get("cross_attention_frequency")) or 1
        enc_w = _pos_int(block.get("encoder_hidden_size")) or h
        cross_layers = -(-layers // every)
        # each cross-attention: Q and O at its own width (with biases), K and V from the tower's,
        # and its output norm
        total += cross_layers * (2 * (h * h + h) + 2 * (enc_w * h + h) + 2 * h)
        queries = _pos_int(config.get("num_query_tokens")) or 0
        total += queries * h + 2 * h  # the learned queries and the stack's input norm
        if text_width:
            total += h * text_width + text_width  # the projection to the language model
    return int(total)


def _first_width(tc: Dict[str, Any]) -> Optional[int]:
    for k in ("hidden_size", "d_model", "n_embd"):
        v = _pos_int(tc.get(k)) if isinstance(tc, dict) else None
        if v:
            return v
    return None
