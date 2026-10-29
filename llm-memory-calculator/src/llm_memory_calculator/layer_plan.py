"""Canonical per-layer plan for hybrid attention/recurrent architectures.

Every hybrid family invented its own config key for "which layers actually carry
a KV cache". There are eleven such dialects in the wild and they share nothing
but intent, so the calculator used to read one of them (``layer_types``, and only
its sliding/full spelling) and silently charge full KV on every layer of
everything else. That over-counts by 3.75x on LFM2 and 14x on Nemotron-H, and it
gets worse the longer the context -- exactly the regime these models exist for.

This module resolves all of them into one shape, so KV and state each have a
single consumer instead of eleven special cases.

Two axes, deliberately independent:

* ``attn[i]``      -- ``'full'``, ``'sliding'`` or ``None``. Drives KV cache.
* ``recurrent[i]`` -- a mechanism name or ``None``. Drives conv/recurrent state.

They are independent because *parallel* hybrids (Falcon-H1, Hymba) run attention
and a recurrent mixer side by side in the same layer, so a single "layer kind"
enum cannot describe them: their KV layer count is genuinely the full depth while
their state is also charged on every layer. Sequential hybrids (Qwen3-Next,
Jamba, Nemotron-H, ...) set exactly one axis per layer.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Recurrent mechanisms, by state geometry rather than by vendor name -- the
# state shape is what the memory model needs, and several vendors share one.
MECH_MAMBA1 = "mamba1"  # d_inner x d_state              (Mamba-1, Jamba, falcon_mamba)
MECH_MAMBA2 = "mamba2"  # n_heads x d_head x d_state      (Mamba-2, Nemotron-H, Bamba, granite-4)
MECH_GDN = "gdn"  # n_v_heads x k_head_dim x v_head_dim  (Gated DeltaNet: Qwen3-Next/3.5/3.6)
MECH_KDA = "kda"  # n_heads x head_dim x head_dim        (Kimi Delta Attention)
MECH_LIGHTNING = "lightning"  # n_heads x head_dim x head_dim (plain linear attention, MiniMax)
MECH_SHORTCONV = "shortconv"  # conv cache only, no recurrent matrix (LFM2)
MECH_RWKV = "rwkv"  # token-shift vectors + (WKV heads x head_size x head_size)
MECH_MLSTM = "mlstm"  # n_heads x qk_head_dim x v_head_dim matrix memory (xLSTM)
MECH_RGLRU = "rglru"  # a diagonal recurrence: one lru_width vector + conv buffer (RecurrentGemma)

ATTN_FULL = "full"
ATTN_SLIDING = "sliding"
# Cross-attention to an encoder's output (vision tokens, audio frames). Its cache
# holds the ENCODER's tokens, a fixed count per request -- it does not grow with
# the text the decoder generates.
ATTN_CROSS = "cross"

# --------------------------------------------------------------------------
# The ONE table of per-layer kind names.
#
# Keys are the strings configs put in `layer_types` / `block_types` /
# `layers_block_type`. The vocabulary is transformers' own
# (`configuration_utils.ALLOWED_LAYER_TYPES`, transformers 5.12) plus the
# spellings older or vendor configs ship. Each name maps to the two axes:
#
#   attn       -- which KV a layer holds: full, sliding (bounded by a window),
#                 "chunked" (bounded by `attention_chunk_size`), or None.
#   recurrent  -- "auto" (pick the mechanism from the config's own keys, see
#                 `_recurrent_mechanism`), a fixed mechanism, or None.
#
# A name NOT in this table is not guessed at: it is charged as full attention
# (the over-estimate -- under-sizing KV is what OOMs a deployment) and the plan
# says so in its notes and with a warning. It is never silently treated as free.
# --------------------------------------------------------------------------
_AUTO = "auto"
# A name-level kind only: a chunked layer is planned as ATTN_SLIDING whose window
# is `attention_chunk_size`.
ATTN_CHUNKED = "chunked"

LAYER_KINDS: Dict[str, Dict[str, Any]] = {
    # softmax attention over every token
    "full_attention": {"attn": ATTN_FULL, "recurrent": None},
    "attention": {"attn": ATTN_FULL, "recurrent": None},
    "full": {"attn": ATTN_FULL, "recurrent": None},
    "global": {"attn": ATTN_FULL, "recurrent": None},
    "global_attention": {"attn": ATTN_FULL, "recurrent": None},
    "attn": {"attn": ATTN_FULL, "recurrent": None},
    # Sparse / compressed attention (DeepSeek-V3.2 DSA, DeepSeek-V4 CSA/HCA,
    # MiniMax-M3). These still keep a per-token cache; what they compress is
    # read-side. Charged as full: an upper bound, never an under-count.
    "sparse": {"attn": ATTN_FULL, "recurrent": None},
    "deepseek_sparse_attention": {"attn": ATTN_FULL, "recurrent": None},
    "compressed_sparse_attention": {"attn": ATTN_FULL, "recurrent": None},
    "heavily_compressed_attention": {"attn": ATTN_FULL, "recurrent": None},
    "minimax_m3_sparse": {"attn": ATTN_FULL, "recurrent": None},
    # windowed attention: KV bounded by `sliding_window`
    "sliding_attention": {"attn": ATTN_SLIDING, "recurrent": None},
    "sliding": {"attn": ATTN_SLIDING, "recurrent": None},
    "local": {"attn": ATTN_SLIDING, "recurrent": None},
    "local_attention": {"attn": ATTN_SLIDING, "recurrent": None},
    "swa": {"attn": ATTN_SLIDING, "recurrent": None},
    # chunked local attention (Llama 4): KV bounded by `attention_chunk_size`
    "chunked_attention": {"attn": ATTN_CHUNKED, "recurrent": None},
    # recurrent / linear mixers: a fixed state, no growing cache
    "linear_attention": {"attn": None, "recurrent": _AUTO},
    "linear": {"attn": None, "recurrent": _AUTO},
    "recurrent": {"attn": None, "recurrent": _AUTO},
    "gdn": {"attn": None, "recurrent": MECH_GDN},
    "mamba": {"attn": None, "recurrent": MECH_MAMBA2},
    "mamba2": {"attn": None, "recurrent": MECH_MAMBA2},
    "mamba1": {"attn": None, "recurrent": MECH_MAMBA1},
    "ssm": {"attn": None, "recurrent": MECH_MAMBA2},
    "rwkv": {"attn": None, "recurrent": MECH_RWKV},
    "conv": {"attn": None, "recurrent": MECH_SHORTCONV},
    "shortconv": {"attn": None, "recurrent": MECH_SHORTCONV},
    # attention and a Mamba mixer in the same layer (Zamba / Zamba2)
    "hybrid": {"attn": ATTN_FULL, "recurrent": MECH_MAMBA2},
    # an FFN-only slot (Nemotron-H's MoE / MLP layers): no mixer, no cache. A stack that names
    # one gives the FFN its own layer (`ffn_slot`): only the slots carry an FFN
    "moe": {"attn": None, "recurrent": None, "ffn_slot": True},
    "dense": {"attn": None, "recurrent": None, "ffn_slot": True},
    "mlp": {"attn": None, "recurrent": None, "ffn_slot": True},
    "mlp_only": {"attn": None, "recurrent": None, "ffn_slot": True},
    "ffn": {"attn": None, "recurrent": None, "ffn_slot": True},
    # a layer whose mixer holds nothing -- no KV, no state -- and that runs its FFN like every
    # other layer of its stack (a GGUF's head_count_kv-0 layer with no recurrent mixer: Deci's
    # attention no-op or linear replacement). Not an FFN-only slot
    "no_op": {"attn": None, "recurrent": None},
}

# The config keys that let a mechanism be sized from the model itself. If none
# of them are present the sizing functions fall back to built-in defaults, which
# are one particular model's geometry -- so the plan is marked approximate
# instead of passing a fabricated number off as a measurement.
_MECHANISM_REQUIRED_KEYS = {
    MECH_GDN: {
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim",
    },
    MECH_KDA: {"linear_attn_config"},
    MECH_MAMBA1: {"state_size", "d_state", "mamba_d_state", "ssm_state_size"},
    MECH_MAMBA2: {"state_size", "d_state", "mamba_d_state", "ssm_state_size"},
    MECH_SHORTCONV: {"conv_L_cache"},
    MECH_LIGHTNING: {"num_attention_heads"},
    MECH_RWKV: {"attention_hidden_size", "hidden_size"},
    MECH_MLSTM: {"num_heads", "qk_dim_factor"},
    MECH_RGLRU: {"lru_width"},
}


#: Nemotron-H's pattern characters, as transformers translates them.
_NEMOTRON_PATTERN = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}


def _first(config: Dict[str, Any], *keys, default=None):
    """First present, non-None key among aliases. Families disagree on spelling."""
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def _text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The sub-config carrying the language model.

    A model can be multimodal *and* hybrid at once -- Qwen3.6 is both -- so this
    must not be gated on a detected "multimodal" model type the way the KV path
    used to be. Nesting is a packaging detail, orthogonal to architecture.
    """
    tc = config.get("text_config")
    # what the block leaves to its config class is the class's (``with_class_geometry``)
    return with_class_geometry(tc if isinstance(tc, dict) else config)


def _blank(n: int) -> Dict[str, Any]:
    return {"attn": [None] * n, "recurrent": [None] * n}


#: Per-layer lists whose length IS the depth. transformers 5.12's NemotronHConfig deprecates
#: `num_hidden_layers` in favour of `layers_block_type` and drops it from the saved config,
#: so a reader that requires the scalar loses the whole layout.
_PER_LAYER_LIST_KEYS = (
    "layer_types", "layers_block_type", "hybrid_override_pattern", "attn_type_list",
    "sliding_window_layout", "mlp_layer_types", "attention_layers",
)


#: The decoder-only names a config class aliases to another key (transformers' `attribute_map`):
#: GPT-2, GPT-J, CodeGen, GPT-BigCode, CTRL and OpenAI-GPT state `n_layer`, `n_embd`, `n_head` and
#: `n_positions`, BLOOM `n_layer` and `n_head` -- flat, or as an EncoderDecoderModel's or a
#: VisionEncoderDecoder's nested `decoder` (a ViT-GPT2 captioner). GPT-2's FFN width is `n_inner`
#: (unset: 4 x `n_embd`). ``with_decoder_names`` puts each under the decoder-only name where the
#: config does not state that name itself; read under the decoder-only names alone, a nested
#: GPT-2 decoder had no KV and GenZ, the serving path and the prefix cache raised.
DECODER_NAME_ALIASES = {
    "num_hidden_layers": ("n_layer",),
    "hidden_size": ("n_embd",),
    "num_attention_heads": ("n_head",),
    "max_position_embeddings": ("n_positions",),
    "intermediate_size": ("n_inner",),
}


def with_decoder_names(block: Dict[str, Any]) -> Dict[str, Any]:
    """``block`` with its geometry also under the decoder-only names (``DECODER_NAME_ALIASES``)
    where it states it only under an alias; ``block`` itself when it states none that way."""
    if not isinstance(block, dict):
        return block
    add = {}
    for name, aliases in DECODER_NAME_ALIASES.items():
        if _first(block, name) is not None:
            continue
        for alias in aliases:
            v = block.get(alias)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                add[name] = v
                break
    return {**block, **add} if add else block


def depth(tc: Dict[str, Any]) -> int:
    """Decoder depth: the stated layer count, else the length of a per-layer list, else -- a
    block that states no depth at all -- its config class's (``class_geometry``: Qwen2Config's
    32 for Qwen2-Audio's `text_config`), else 0."""
    n = _first(tc, "num_hidden_layers", "n_layers", "num_layers", "n_layer")
    if n:
        return int(n)
    for key in _PER_LAYER_LIST_KEYS:
        v = tc.get(key)
        if isinstance(v, (list, tuple, str)) and v:
            return len(v)
    if isinstance(tc, dict) and tc.get("model_type") and not any(
            tc.get(k) is not None for k in CLASS_GEOMETRY_ROLES[0]):
        n = _first(class_geometry(tc.get("model_type")), "num_hidden_layers", "n_layers",
                   "num_layers", "n_layer")
        if _pos_int(n):
            return int(n)
    return 0


#: An encoder-decoder's two stacks, as transformers' config classes (5.12) spell them. The
#: decoder holds the per-token KV cache. BART / Whisper / Marian / M2M-100 / Pegasus / ... say
#: `decoder_layers`, `decoder_attention_heads`, `decoder_ffn_dim` and their `encoder_*` twins.
#: The T5 family (T5, mT5, UMT5, LongT5, Switch) says `num_decoder_layers` beside `num_layers`,
#: which is the ENCODER's depth, and `num_heads` / `d_model` / `d_ff` / `d_kv` for both. Moonshine
#: says `decoder_num_hidden_layers` / `decoder_num_attention_heads` / `decoder_num_key_value_heads`.
#: ProphetNet says `num_encoder_layers` / `num_decoder_layers` and `num_encoder_attention_heads`
#: / `num_decoder_attention_heads`. Dia and EncoderDecoderModel nest each stack in its own block
#: (`encoder_config` / `decoder_config`, `encoder` / `decoder`; MusicGen's encoder is its
#: `text_encoder`); cohere_asr nests only the encoder and keeps the decoder at the top level. A
#: multimodal model can nest a whole encoder-decoder under `text_config` (BLIP-2 / InstructBLIP
#: with flan-T5, Florence-2): it is read there, as every other decoder is.
DECODER_DEPTH_KEYS = ("decoder_layers", "num_decoder_layers", "decoder_num_hidden_layers")
ENCODER_DEPTH_KEYS = ("encoder_layers", "num_encoder_layers", "encoder_num_hidden_layers")
DECODER_HEAD_KEYS = ("decoder_attention_heads", "decoder_num_attention_heads",
                     "num_decoder_attention_heads")
DECODER_KV_HEAD_KEYS = ("decoder_num_key_value_heads",)
DECODER_FFN_KEYS = ("decoder_ffn_dim",)
ENCODER_HEAD_KEYS = ("encoder_attention_heads", "encoder_num_attention_heads",
                     "num_encoder_attention_heads")
ENCODER_KV_HEAD_KEYS = ("encoder_num_key_value_heads",)
ENCODER_FFN_KEYS = ("encoder_ffn_dim",)
NESTED_DECODER_KEYS = ("decoder_config", "decoder")
NESTED_ENCODER_KEYS = ("encoder_config", "encoder", "text_encoder")
#: The cross-attention's own K / V geometry, where a decoder states it apart from its
#: self-attention (Dia's `decoder_config`: 16 KV heads of 128 across, 4 for self-attention).
#: Unstated, the cross-attention is built like the self-attention (Whisper, BART, T5, Moonshine).
CROSS_KV_HEAD_KEYS = ("cross_num_key_value_heads", "cross_num_attention_heads")
CROSS_HEAD_DIM_KEYS = ("cross_head_dim",)


#: A hierarchical tower's per-stage widths, where it states them (ConvNeXt / PVT `hidden_sizes`,
#: some `embed_dims`, Florence-2's DaViT `embed_dim` as a list -- `dim_embed` in its hub file);
#: a Swin-family tower states only its first stage's `embed_dim` (a number) and doubles it at
#: every stage after (SwinConfig: `hidden_size = embed_dim * 2 ** (len(depths) - 1)`).
STAGE_WIDTH_KEYS = ("hidden_sizes", "embed_dims", "embed_dim", "dim_embed")


def encoder_stages(block: Any) -> Optional[List[Dict[str, int]]]:
    """A hierarchical tower's stages, ``[{"layers", "hidden_size", "num_attention_heads",
    "intermediate_size"}, ...]``, or None for a uniform stack.

    A tower that states its stages (`depths`: Swin, Donut-Swin, Swin2, FocalNet, NAT/DiNAT,
    ConvNeXt, PVT) runs `depths[i]` blocks at stage i's own width -- its `hidden_sizes` /
    `embed_dims`, else `embed_dim` x 2^i -- with an MLP `mlp_ratio` (default 4) times that wide,
    and merges 2 x 2 neighbours between stages (``_image_tokens``). Read as one stack of
    `num_layers` (Swin's STAGE count: 4) at the last stage's width, Donut's 20-block, 74M-parameter
    encoder was sized as 4 blocks of 1,024 (50M)."""
    if not isinstance(block, dict):
        return None
    depths = block.get("depths")
    if not (isinstance(depths, (list, tuple)) and depths and all(_pos_int(d) for d in depths)):
        return None
    n = len(depths)
    widths = None
    for k in STAGE_WIDTH_KEYS:
        v = block.get(k)
        if isinstance(v, (list, tuple)) and len(v) == n and all(_pos_int(x) for x in v):
            widths = [int(x) for x in v]
            break
    if widths is None:
        base = _pos_int(block.get("embed_dim"))
        if not base:
            last = _pos_int(block.get("hidden_size"))
            if not last:
                return None
            base = max(last // 2 ** (n - 1), 1)
        widths = [base * 2 ** i for i in range(n)]
    heads = block.get("num_heads", block.get("num_attention_heads"))
    ratio = block.get("mlp_ratio") or 4
    ratios = ratio if isinstance(ratio, (list, tuple)) and len(ratio) == n else [ratio] * n
    # A stage that states channel-attention groups (`num_groups`: DaViT, Florence-2's tower) runs
    # a channel-attention block after each window block -- two blocks, each with its MLP, per
    # unit of `depths` (transformers 5.12 Florence2VisionBlock: spatial_block, channel_block).
    groups = block.get("num_groups")
    dual = isinstance(groups, (list, tuple)) and len(groups) == n and all(
        _pos_int(g) for g in groups)
    stages = []
    for i, (d, w) in enumerate(zip(depths, widths)):
        h = heads[i] if isinstance(heads, (list, tuple)) and len(heads) == n else heads
        stages.append({"layers": int(d) * (2 if dual else 1), "hidden_size": int(w),
                       "num_attention_heads": _pos_int(h) or max(w // 32, 1),
                       "intermediate_size": int(float(ratios[i]) * w)})
    return stages


def _block_depth(block: Dict[str, Any], role: Optional[str] = None) -> int:
    """A nested stack block's depth, read for the ``role`` it plays ("decoder" / "encoder").

    A block that is itself a whole encoder-decoder config (a VisionEncoderDecoder's `mbart`
    decoder block: Donut, Nougat) states both of its own stacks, and its `num_hidden_layers` is
    one of them by its class's alias (MBartConfig maps it to `encoder_layers`: 12, where the
    decoder is `decoder_layers`: 4). So the role's own spelling is read first, then the
    decoder-only one (`num_hidden_layers`, ...; MusicGen's decoder, a T5 text encoder), then the
    other stack's (TrOCR's decoder block says only `decoder_layers`). With no role, any stated
    depth (whether the block holds a stack at all)."""
    own, other = ((DECODER_DEPTH_KEYS, ENCODER_DEPTH_KEYS) if role != "encoder"
                  else (ENCODER_DEPTH_KEYS, DECODER_DEPTH_KEYS))
    stages = encoder_stages(block)
    if stages:
        # a hierarchical tower's blocks, every stage's (its `num_layers` counts the stages)
        return sum(st["layers"] for st in stages)
    if role is not None:
        v = _pos_int(_first(block, *own))
        if v:
            return v
    return depth(block) or _pos_int(_first(block, *own)) or _pos_int(_first(block, *other)) or 0


def _stack_block(config: Dict[str, Any], keys) -> Optional[Dict[str, Any]]:
    """The nested block holding one stack (`encoder_config` / `decoder_config`, ...), or None."""
    for k in keys:
        v = config.get(k)
        if isinstance(v, dict) and _block_depth(v) > 0:
            return v
    return None


def encoder_block(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An encoder-decoder's nested encoder block (Dia's `encoder_config`, cohere_asr's, ...)."""
    return _stack_block(config, NESTED_ENCODER_KEYS) if isinstance(config, dict) else None


def _own_stacks(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """The two depths a config block states for itself, or None when it is no encoder-decoder."""
    dec_key = _pos_int(_first(config, *DECODER_DEPTH_KEYS))
    enc_key = _pos_int(_first(config, *ENCODER_DEPTH_KEYS))
    dec_block = _stack_block(config, NESTED_DECODER_KEYS)
    enc_block = _stack_block(config, NESTED_ENCODER_KEYS)
    if not (config.get("is_encoder_decoder") is True or dec_key
            or (dec_block is not None and enc_block is not None)):
        return None
    stated = depth(config)
    decoder = dec_key or (_block_depth(dec_block, "decoder") if dec_block is not None else stated)
    encoder = enc_key or (_block_depth(enc_block, "encoder") if enc_block is not None else stated)
    return {"encoder": int(encoder or 0), "decoder": int(decoder or 0)}


#: Another modality's tower beside a decoder-only `text_config`: the encoder an encoder-decoder
#: whose decoder is that block cross-attends to (Pix2Struct's `vision_config`), and the tower
#: interleaved cross-attention layers read (Mllama's `vision_config`).
TOWER_KEYS = ("vision_config", "audio_config")


def decoder_only_block(block: Dict[str, Any]) -> bool:
    """A nested block that is a decoder alone: it says `is_decoder` and states no encoder of its
    own (no encoder depth key, no encoder block) -- Pix2Struct's `text_config`, which transformers
    also flags `is_encoder_decoder`. Its encoder is a tower beside it, not a stack inside it."""
    return (block.get("is_decoder") is True and not _pos_int(_first(block, *ENCODER_DEPTH_KEYS))
            and _stack_block(block, NESTED_ENCODER_KEYS) is None)


def composed_text_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The config's `text_config` as the composite model builds it, or None when it has none.

    A decoder alone (``decoder_only_block``) beside a tower (``TOWER_KEYS``) is the decoder of an
    encoder-decoder whose encoder is that tower: every layer cross-attends to the tower's output
    (Pix2StructTextBlock always builds `encoder_decoder_attention`; BlipTextLayer builds its
    `crossattention` exactly when `is_decoder`). transformers' composite config says so by
    copying its own `is_encoder_decoder` into the block (Pix2StructConfig, whose class default is
    True), but a hub file saved by an older release omits the outer flag (it is the class default)
    and keeps the block's stale `is_encoder_decoder: false`, which is what the simulator reads
    (google/deplot, pix2struct-*). So the block's own `is_decoder` beside a tower is read as the
    statement, and the block is returned flagged as transformers flags it. Erring on a decoder
    that would not attend to the tower over-counts, never under-counts. Any other `text_config`
    is returned as it is."""
    config = with_class_geometry(config)
    tc = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(tc, dict):
        return None
    if (tc.get("is_encoder_decoder") is not True and decoder_only_block(tc)
            and _tower(config) is not None and depth(tc) > 0):
        return dict(tc, is_encoder_decoder=True)
    return tc


def _tower(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The config's other-modality tower that states a depth (``TOWER_KEYS``), or None."""
    for k in TOWER_KEYS:
        v = config.get(k) if isinstance(config, dict) else None
        if isinstance(v, dict) and _block_depth(v) > 0:
            return v
    return None


def _encoder_of(config: Dict[str, Any], holder: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The block holding an encoder-decoder's encoder: its own nested block (Dia's
    `encoder_config`, a VisionEncoderDecoder's `encoder`, MusicGen's `text_encoder`); for a
    decoder-only `text_config` the tower beside it (Pix2Struct's `vision_config`); else None (the
    encoder is stated flat in the holder: T5, BART, Whisper)."""
    block = _stack_block(holder, NESTED_ENCODER_KEYS)
    if block is not None:
        return block
    if holder is not config and decoder_only_block(holder):
        return _tower(config)
    return None


def _stacks_holder(config: Dict[str, Any]):
    """``(block, stacks)``: the config block that states an encoder-decoder and its two depths.

    The config itself when it states a decoder; else its `text_config` when that is an
    encoder-decoder (BLIP-2's flan-T5, Florence-2's BART-like language model: the outer config
    may say `is_encoder_decoder` and state no stack of its own) or the decoder of one whose
    encoder is the tower beside it (Pix2Struct: the encoder's depth is the tower's); else the
    config's own reading (a flagged config with no decoder depth), or ``(None, None)``.
    """
    if not isinstance(config, dict):
        return None, None
    config = with_class_geometry(config)
    own = _own_stacks(config)
    if own and own["decoder"]:
        return config, own
    tc = composed_text_config(config)
    if isinstance(tc, dict):
        inner = _own_stacks(tc)
        if inner:
            enc = _encoder_of(config, tc)
            if enc is not None and decoder_only_block(tc):
                inner = dict(inner, encoder=_block_depth(enc, "encoder"))
            return tc, inner
    return (config, own) if own else (None, None)


def encoder_is_tower(config: Dict[str, Any]) -> bool:
    """Whether an encoder-decoder's encoder is the config's other-modality tower (Pix2Struct's
    and BLIP's `vision_config` beside a decoder-only `text_config`): the tower is then counted
    once, as the encoder, never again beside it."""
    if not isinstance(config, dict):
        return False
    config = with_class_geometry(config)
    holder, stacks = _stacks_holder(config)
    if not stacks or holder is None or holder is config:
        return False
    enc = _encoder_of(config, holder)
    return enc is not None and any(config.get(k) is enc for k in TOWER_KEYS)


def encoder_decoder_stacks(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """``{"encoder": layers, "decoder": layers}`` for an encoder-decoder config, else None.

    A config is one when it says so (`is_encoder_decoder`), states a decoder depth of its own,
    or nests both stacks -- itself or in its `text_config`. Each depth is the stack's own key,
    else its nested block's depth, else the config's stated depth: transformers reads
    `num_hidden_layers` of an encoder-decoder as one stack's depth (BART and Whisper map it to
    `encoder_layers`, T5 to `num_layers`, and T5's decoder defaults to `num_layers`), never as
    the two stacks' sum. A depth the config does not state is 0.
    """
    return _stacks_holder(config)[1]


#: What a nested decoder block states about itself, not about the model it is lifted into: its
#: own composition flags (MusicGen's `decoder` and a VisionEncoderDecoder's carry the `to_dict`
#: defaults `is_encoder_decoder: false`, `add_cross_attention: false` under an outer
#: `is_encoder_decoder: true`) and the encoder stack of a block that is a whole encoder-decoder
#: config (Donut's `mbart` decoder states `encoder_layers: 12`; the model's encoder is the Swin
#: beside it). Lifting them over the top level made the decoder's view deny its own
#: cross-attention, so a reader that viewed the decoder first charged none.
_BLOCK_OWN_KEYS = (("is_encoder_decoder", "add_cross_attention") + ENCODER_DEPTH_KEYS
                   + ENCODER_HEAD_KEYS + ENCODER_KV_HEAD_KEYS + ENCODER_FFN_KEYS)


def _lift(config: Dict[str, Any], stacks: Dict[str, int]) -> Dict[str, Any]:
    """One encoder-decoder block with its decoder's geometry under the decoder-only names."""
    view = dict(config)
    block = _stack_block(config, NESTED_DECODER_KEYS)
    if block is not None:
        view.update({k: v for k, v in with_decoder_names(block).items()
                     if v is not None and k not in _BLOCK_OWN_KEYS})
    if stacks["decoder"]:
        view["num_hidden_layers"] = stacks["decoder"]
    heads = _pos_int(_first(view, *DECODER_HEAD_KEYS))
    if heads:
        view["num_attention_heads"] = heads
        view["num_key_value_heads"] = _pos_int(_first(view, *DECODER_KV_HEAD_KEYS)) or heads
    elif not _pos_int(_first(view, *ATTENTION_HEAD_KEYS)):
        # T5's `num_heads`: in an encoder-decoder these are attention heads (the guard the
        # normalizer keeps for `num_heads` is for xLSTM's matrix-memory heads, a decoder-only).
        heads = _pos_int(view.get("num_heads"))
        if heads:
            view["num_attention_heads"] = heads
    if not _pos_int(view.get("hidden_size")) and _pos_int(view.get("d_model")):
        view["hidden_size"] = int(view["d_model"])
    ffn = _pos_int(_first(view, *DECODER_FFN_KEYS))
    if ffn:
        view["intermediate_size"] = ffn
    elif not _pos_int(view.get("intermediate_size")) and _pos_int(view.get("d_ff")):
        view["intermediate_size"] = int(view["d_ff"])
    return view


def decoder_view(config: Dict[str, Any]) -> Dict[str, Any]:
    """An encoder-decoder config with its DECODER's geometry under the decoder-only names.

    The per-token KV cache is the decoder's self-attention: its depth, heads and width, which
    such a config states under its own spellings (``DECODER_*_KEYS``, T5's ``num_heads`` /
    ``d_model`` / ``d_ff``). Read under the decoder-only names alone they were missed: the
    decoder was charged the encoder's depth, a default head count or none (Whisper and BART
    were charged no KV at all). A decoder in its own block (Dia's `decoder_config`,
    EncoderDecoderModel's `decoder`) is lifted over the top level, as a multimodal config's
    `text_config` is. An encoder-decoder nested as a `text_config` (BLIP-2's flan-T5) is viewed
    in place, so every reader that looks there finds the decoder. A decoder spelled the GPT-2 way
    (`n_layer` / `n_embd` / `n_head`: ``DECODER_NAME_ALIASES``), nested or flat, is also given
    the decoder-only names. Any other config is returned as it is.
    """
    config = with_class_geometry(config)
    holder, stacks = _stacks_holder(config)
    if stacks is None:
        # a decoder-only config: its own geometry, under the decoder-only names (GPT-2's)
        if not isinstance(config, dict):
            return config
        tc = config.get("text_config")
        if isinstance(tc, dict):
            named = with_decoder_names(tc)
            return config if named is tc else {**config, "text_config": named}
        return with_decoder_names(config)
    view = _lift(holder, stacks)
    if holder is config:
        return view
    return {**config, "text_config": view}


def encoder_view(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An encoder-decoder's ENCODER stack under the decoder-only names, or None.

    Its depth (``encoder_decoder_stacks``), width, heads and FFN width, from its own block
    (Dia's `encoder_config`, EncoderDecoderModel's `encoder`) or its own spellings
    (``ENCODER_*_KEYS``; T5's `num_heads` / `d_model` / `d_ff`, shared by both stacks). The GenZ
    prefill graph runs this stack once a request, over the encoder's tokens.
    """
    config = with_class_geometry(config)
    holder, stacks = _stacks_holder(config)
    if not stacks or not stacks["encoder"]:
        return None
    block = _encoder_of(config, holder)
    src = block if block is not None else holder
    view = {k: v for k, v in src.items() if not isinstance(v, dict)}
    view["num_hidden_layers"] = stacks["encoder"]
    # an encoder in its own block embeds its own tokens (MusicGen's T5 text encoder, Dia's,
    # EncoderDecoderModel's BERT); one stated flat shares the decoder's table (T5, BART)
    view["own_embedding"] = bool(block is not None and _pos_int(block.get("vocab_size")))
    # A generic composite (VisionEncoderDecoder / SpeechEncoderDecoder / EncoderDecoderModel's
    # `encoder` + `decoder`, MusicGen's `text_encoder` + `decoder`) projects an encoder of another
    # width to the decoder's (`enc_to_dec_proj`) unless the decoder states the width its
    # cross-attention reads (`cross_attention_hidden_size`); its cross K / V then read the
    # decoder's width. A model's own sub-configs (Dia's `encoder_config`) read the encoder's.
    dec_block = holder.get("decoder") if isinstance(holder.get("decoder"), dict) else None
    enc_key = next((k for k in ("encoder", "text_encoder") if holder.get(k) is block), None)
    if block is not None and dec_block is not None and enc_key:
        enc_w = _pos_int(_first(block, "hidden_size", "d_model", "n_embd"))
        dec_w = _pos_int(_first(dec_block, "hidden_size", "d_model", "n_embd"))
        if enc_w and dec_w and enc_w != dec_w and not dec_block.get("cross_attention_hidden_size"):
            view["projected_to"] = dec_w
    heads = None if block is not None else _pos_int(_first(src, *ENCODER_HEAD_KEYS))
    if heads:
        view["num_attention_heads"] = heads
        view["num_key_value_heads"] = _pos_int(_first(src, *ENCODER_KV_HEAD_KEYS)) or heads
    else:
        heads = _pos_int(_first(src, *ATTENTION_HEAD_KEYS)) or _pos_int(src.get("num_heads"))
        if heads:
            view["num_attention_heads"] = heads
    width = _pos_int(_first(src, "hidden_size", "d_model", "n_embd"))
    if width:
        view["hidden_size"] = width
    ffn = None if block is not None else _pos_int(_first(src, *ENCODER_FFN_KEYS))
    ffn = ffn or _pos_int(_first(src, "intermediate_size", "d_ff", "encoder_ffn_dim"))
    if ffn:
        view["intermediate_size"] = ffn
    # A hierarchical tower runs each stage at its own width (``encoder_stages``); the view's
    # width is its last stage's, the one the decoder reads.
    stages = encoder_stages(src)
    if stages:
        view["stages"] = stages
        last = stages[-1]
        view["hidden_size"] = last["hidden_size"]
        view["num_attention_heads"] = last["num_attention_heads"]
        view["intermediate_size"] = last["intermediate_size"]
    # Head widths are the encoder's own (Dia's encoder `head_dim`; T5's `d_kv` for both).
    if block is None:
        for k in ("head_dim", "kv_channels", "attention_head_dim"):
            view.pop(k, None)
        if _pos_int(src.get("d_kv")):
            view["head_dim"] = int(src["d_kv"])
    return view


def _two_sides(v: Any) -> Optional[Tuple[int, int]]:
    """``(height, width)`` from a square size (an int) or a stated [height, width] pair."""
    if _pos_int(v):
        return int(v), int(v)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_pos_int(x) for x in v):
        return int(v[0]), int(v[1])
    return None


def _image_tokens(vc: Any) -> Optional[int]:
    """Tokens a vision tower hands the decoder for one image: its patch grid, from `image_size`
    and `patch_size` (each a square int or a [height, width] pair: Donut's 2,560 x 1,920).

    A flat ViT keeps every patch plus its class token, per tile (Mllama's `max_num_tiles`). A
    hierarchical tower that states its stages (`depths`: Swin, Donut-Swin, FocalNet, NAT/DiNAT,
    ConvNeXt) merges 2 x 2 neighbours between consecutive stages and has no class token: its last
    stage, the one a decoder reads, is the grid halved per side once per stage after the first
    (Donut: 2,560 / 4 / 8 x 1,920 / 4 / 8 = 80 x 60 = 4,800; Nougat 896 x 672 -> 28 x 21 = 588).
    Each division rounds up, as those towers pad an odd side. None when the size or patch is
    not stated as such."""
    if not isinstance(vc, dict):
        return None
    size, patch = _two_sides(vc.get("image_size")), _two_sides(vc.get("patch_size"))
    if not (size and patch):
        return None
    rows, cols = -(-size[0] // patch[0]), -(-size[1] // patch[1])
    stages = vc.get("depths")
    if isinstance(stages, (list, tuple)) and len(stages) > 1 and all(_pos_int(d) for d in stages):
        for _ in range(len(stages) - 1):
            rows, cols = -(-rows // 2), -(-cols // 2)
        return int(rows * cols)
    per_tile = rows * cols + 1
    tiles = _pos_int(vc.get("max_num_tiles")) or 1
    return int(per_tile * tiles)


#: Spellings of an encoder's output length, the tokens every cross-attention layer caches.
#: Whisper and Speech2Text say `max_source_positions` (Whisper always pads to 30 s: 1,500
#: frames), on the config, its encoder block or its `audio_config`.
ENCODER_TOKEN_KEYS = ("max_source_positions",)
#: An encoder block's own bound on its sequence: Dia's 1,024-token text encoder says
#: `max_position_embeddings`, Pix2Struct's image encoder `seq_len` (4,096 patches). Read on the
#: encoder's block only -- at the top level they are the decoder's.
ENCODER_BLOCK_LENGTH_KEYS = ("max_position_embeddings", "seq_len")


def _block_output_tokens(block: Any):
    """``(tokens, where from)`` one encoder block (or tower) states it hands a decoder, or
    ``(None, None)``: its `max_source_positions`, its own length bound
    (``ENCODER_BLOCK_LENGTH_KEYS``), else an image encoder's patches and tiles."""
    if not isinstance(block, dict):
        return None, None
    v = _pos_int(_first(block, *ENCODER_TOKEN_KEYS))
    if v:
        return v, "max_source_positions"
    for k in ENCODER_BLOCK_LENGTH_KEYS:
        v = _pos_int(block.get(k))
        if v:
            return v, f"the encoder's {k}"
    v = _image_tokens(block)
    if v:
        return v, "the image encoder's patches"
    return None, None


def _encoder_output_tokens(config: Dict[str, Any], holder: Dict[str, Any]):
    """``(tokens, where from)`` the encoder hands the decoder, or ``(None, None)``.

    `max_source_positions` (the config, its encoder block or `audio_config`); the encoder
    block's own length bound (Dia's 1,024-token text encoder: `max_position_embeddings`;
    Pix2Struct's image encoder: `seq_len`); an image encoder's patches (a VisionEncoderDecoder's
    ViT). A text encoder stated flat (T5, BART, BLIP-2's flan-T5, Florence-2's BART) reads the
    request's own input, whose length the config cannot state: None, and each reader charges
    the cache over the sequence instead (docs/LAYER_KINDS.md, "What is conservative", says when
    that is a bound).
    """
    enc = _encoder_of(config, holder)
    for scope in (holder, enc, config, holder.get("audio_config"), config.get("audio_config")):
        if isinstance(scope, dict):
            v = _pos_int(_first(scope, *ENCODER_TOKEN_KEYS))
            if v:
                return v, "max_source_positions"
    for scope in (enc, holder.get("encoder")):
        tokens, source = _block_output_tokens(scope)
        if tokens:
            return tokens, source
    return None, None


def _decoder_cross_attends(holder: Dict[str, Any]) -> bool:
    """Whether the decoder attends to the encoder's output, as the config states it: the config
    says `is_encoder_decoder` (transformers builds the decoder's cross-attention exactly then),
    states a decoder stack of its own spelling (BART, T5, Whisper, Moonshine: their decoder
    layers always hold one), or its decoder block says `add_cross_attention` or states a
    cross-attention's geometry. Two nested stacks alone do not: an audio codec (Qwen3-TTS's
    speech tokenizer) nests an encoder and a decoder that never attend to each other."""
    if holder.get("is_encoder_decoder") is True or _pos_int(_first(holder, *DECODER_DEPTH_KEYS)):
        return True
    block = _stack_block(holder, NESTED_DECODER_KEYS) or {}
    return bool(block.get("add_cross_attention")) or any(
        k in block for k in CROSS_KV_HEAD_KEYS + CROSS_HEAD_DIM_KEYS)


def cross_attention_cache(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A decoder's cross-attention cache: what every reader of a request's KV charges beside the
    self-attention, or None for a config with none. The one reader of it, for both ways a config
    states cross-attention:

    * an encoder-decoder: each decoder layer keeps K and V over the encoder's output beside its
      self-attention (transformers' `EncoderDecoderCache.cross_attention_cache`, vLLM's
      `CrossAttentionSpec`): the decoder's depth, the cross-attention's own K / V heads and head
      width (``CROSS_*_KEYS``, else the decoder's self-attention's);
    * cross-attention layers of their own among a decoder-only stack's self-attention layers
      (Mllama's `cross_attention_layers`, the layer table's ``ATTN_CROSS``): those layers alone,
      at the stack's K / V width; they hold no self-attention cache (``own_layers``).

    Either way the cache is computed once a request and read at every decode step; it does not
    grow with the generated text. Returns ``{"layers", "kv_heads", "head_dim", "tokens",
    "tokens_from", "own_layers"}``; ``tokens`` is the encoder's output length
    (``_encoder_output_tokens``; for interleaved layers the tower's, ``encoder_tokens``), None
    when the config does not state it: the readers then charge the cache over the whole sequence.
    """
    config = with_class_geometry(config)
    holder, stacks = _stacks_holder(config)
    if not stacks:
        return _interleaved_cross_cache(config)
    if not stacks["decoder"] or not _decoder_cross_attends(holder):
        return None
    view = _lift(holder, stacks)
    heads = _pos_int(_first(view, *ATTENTION_HEAD_KEYS))
    kv_heads = (_pos_int(_first(view, *CROSS_KV_HEAD_KEYS))
                or _pos_int(view.get("num_key_value_heads")) or heads)
    head_dim = _pos_int(_first(view, *CROSS_HEAD_DIM_KEYS)) or attention_head_dim(view, heads)
    if not (kv_heads and head_dim):
        return None
    tokens, source = _encoder_output_tokens(config, holder)
    return {"layers": int(stacks["decoder"]), "kv_heads": int(kv_heads),
            "head_dim": int(head_dim), "tokens": tokens, "tokens_from": source,
            "own_layers": False}


def _interleaved_cross_cache(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The cross-attention cache of a decoder-only stack's own cross-attention layers (the layer
    table's ``ATTN_CROSS`` layers that hold a cache), or None."""
    if not isinstance(config, dict):
        return None
    plan = resolve_layer_plan(config)
    n = int(plan.get("num_cross_layers") or 0) if plan else 0
    if not n:
        return None
    tc = _text_config(config)
    heads = _pos_int(_first(tc, *ATTENTION_HEAD_KEYS))
    kv_heads = (_pos_int(_first(tc, *CROSS_KV_HEAD_KEYS))
                or _pos_int(tc.get("num_key_value_heads")) or heads)
    head_dim = _pos_int(_first(tc, *CROSS_HEAD_DIM_KEYS)) or attention_head_dim(tc, heads)
    if not (kv_heads and head_dim):
        return None
    tokens, source = _block_output_tokens(_tower(config) or config.get("vision_config"))
    return {"layers": n, "kv_heads": int(kv_heads), "head_dim": int(head_dim),
            "tokens": tokens, "tokens_from": source, "own_layers": True}


def cross_kv_width(cross: Optional[Dict[str, Any]]) -> int:
    """K + V elements one encoder token adds to one decoder layer's cross-attention cache."""
    return 2 * cross["kv_heads"] * cross["head_dim"] if cross else 0


def cross_attention_bytes(cross: Optional[Dict[str, Any]], bytes_per_element: float,
                          tokens: Optional[int] = None) -> float:
    """One request's cross-attention cache in bytes: every decoder layer's K and V over the
    encoder's ``tokens`` (the stated output length, else the caller's ``tokens``)."""
    if not cross:
        return 0.0
    n = cross["tokens"] or tokens or 0
    return float(cross["layers"] * cross_kv_width(cross) * n * bytes_per_element)


#: Spellings of "how many tokens a windowed layer keeps". SmallThinker ships
#: `sliding_window_size`, RecurrentGemma `attention_window_size`.
WINDOW_KEYS = ("sliding_window", "sliding_window_size", "attention_window_size", "window_size")


def _window(tc: Dict[str, Any]) -> Optional[int]:
    """The window a sliding layer keeps, or None. `use_sliding_window: false`
    (Qwen2/2.5 ship a window with the feature switched off) means no window."""
    if tc.get("use_sliding_window") is False:
        return None
    w = _first(tc, *WINDOW_KEYS)
    return int(w) if isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 else None


#: Spellings of one attention head's width, in the order they are believed. transformers aliases
#: `head_dim` to another key in some config classes (`attribute_map`, 5.12): Zamba and Zamba2 to
#: `attention_head_dim`, JetMoE to `kv_channels`, the T5 family to `d_kv`. Zamba2's hub config
#: also ships Megatron's `kv_channels: 80` beside `attention_head_dim: 160`; transformers builds
#: k_proj / v_proj at 160, so `attention_head_dim` is read first. (GLM-4-MoE-Lite's alias,
#: `qk_rope_head_dim`, is MLA's RoPE width: an MLA cache is the latent, sized by its own path.)
HEAD_DIM_KEYS = ("head_dim", "attention_head_dim", "kv_channels", "d_kv")

#: Spellings of the attention head count.
ATTENTION_HEAD_KEYS = ("num_attention_heads", "n_head", "n_heads", "num_attention_head")


def attention_heads(config: Dict[str, Any]) -> Optional[int]:
    """The attention head count under any of its spellings (``ATTENTION_HEAD_KEYS``), else T5's,
    GPT-Neo's and MPT's `num_heads` / `n_heads` when the stack's layers attend (by the layer
    table: xLSTM's `num_heads` are its matrix memory's, not attention's). The reader that goes with
    ``attention_head_dim``: T5 states `num_heads` beside `d_kv`, and reading the width without the
    count sized a T5 block at a default 12 heads of 64."""
    tc = config if isinstance(config, dict) else {}
    heads = _pos_int(_first(tc, *ATTENTION_HEAD_KEYS))
    if heads:
        return heads
    for key in ("num_heads", "n_heads"):
        heads = _pos_int(tc.get(key))
        if heads:
            return heads if any(a is not None for a in layer_kinds(tc)["attn"]) else None
    return None


def _pos_int(v) -> Optional[int]:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 or int(v) != v:
        return None
    return int(v)


_CLASS_HEAD_DIM_CACHE: Dict[str, Optional[int]] = {}


def _config_class_head_dim(tc: Dict[str, Any]) -> Optional[int]:
    """The head width the model type's own config class derives, for a class that keeps it under
    another name (`attribute_map["head_dim"]`, or a width argument of its constructor) and a
    config that does not state it. ZambaConfig derives `attention_head_dim = 2 x hidden_size /
    heads`: its attention reads the hidden state concatenated with the input embeddings. None
    when transformers is absent, does not know the type, or the class has no other name for the
    width (then hidden_size / heads is what its modelling uses too)."""
    mt = str(tc.get("model_type") or "")
    if not mt or tc.get("kv_lora_rank"):
        # MLA keeps a latent, not per-head K and V; its class's head_dim alias is the RoPE width.
        return None
    # the shipped rule first (``config_class_defaults.json``'s ``head_dims``: transformers 5.12's
    # classes), so the width does not depend on whether or which transformers is installed
    rule = (_class_default_snapshot().get("head_dims") or {}).get(mt)
    if isinstance(rule, dict):
        hidden = _pos_int(_first(tc, "hidden_size", "d_model", "n_embd"))
        heads = _pos_int(_first(tc, *ATTENTION_HEAD_KEYS))
        if hidden and heads:
            return int(float(rule["ratio"]) * hidden) // heads
    try:
        import json

        key = json.dumps(tc, sort_keys=True, default=str)
    except Exception:
        key = None
    if key is not None and key in _CLASS_HEAD_DIM_CACHE:
        return _CLASS_HEAD_DIM_CACHE[key]
    result = None
    names: List[str] = []
    try:
        from transformers import CONFIG_MAPPING

        cls = CONFIG_MAPPING[mt]
        alias = (getattr(cls, "attribute_map", None) or {}).get("head_dim")
        if alias and alias != "head_dim":
            names.append(alias)
        # A release whose class keeps the width under its own name without aliasing `head_dim`
        # (transformers 4.50's ZambaConfig) still takes it as a constructor argument.
        import inspect

        params = inspect.signature(cls.__init__).parameters
        names += [k for k in HEAD_DIM_KEYS[1:] if k in params and k not in names]
    except Exception:
        cls = None
    if cls is not None and names:
        import logging

        drop = {"model_type", "transformers_version", "architectures", "auto_map", "text_config",
                "vision_config", "audio_config"}
        whole = {k: v for k, v in tc.items() if k not in drop and not k.startswith("_")}
        reduced = {k: whole[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                                         "num_hidden_layers", "attention_hidden_size") if k in whole}
        tf_log = logging.getLogger("transformers")
        level = tf_log.level
        tf_log.setLevel(logging.ERROR)
        try:
            for kwargs in (whole, reduced):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        obj = cls(**kwargs)
                except Exception:
                    continue
                result = next((v for v in (_pos_int(getattr(obj, k, None)) for k in names) if v), None)
                break
        finally:
            tf_log.setLevel(level)
    if key is not None:
        _CLASS_HEAD_DIM_CACHE[key] = result
    return result


def attention_head_dim(
    config: Dict[str, Any], heads: Optional[int] = None, hidden: Optional[int] = None
) -> Optional[int]:
    """The width of one attention head, as the model's own attention builds its K and V.

    In order: a stated width under any of its spellings (``HEAD_DIM_KEYS``); the attention's own
    input width over the heads (`attention_hidden_size`, Zamba's 2 x hidden); the width the
    model type's config class derives under another name (transformers installed); and only then
    hidden_size / heads. Reading `head_dim` alone and deriving the rest from hidden_size sized
    Zamba2's KV at half: its heads are 160 wide, not 2560 / 32 = 80.

    ``heads`` and ``hidden`` are the caller's head count and hidden size (with its own
    defaults); the config's are read when they are None. The config given is read as it is --
    a caller sizing a decoder passes the decoder's (text) config. None when neither a width nor
    a head count is known.
    """
    tc = config if isinstance(config, dict) else {}
    for k in HEAD_DIM_KEYS:
        v = _pos_int(tc.get(k))
        if v:
            return v
    heads = _pos_int(heads) or _pos_int(_first(tc, *ATTENTION_HEAD_KEYS))
    if not heads:
        return None
    width = _pos_int(tc.get("attention_hidden_size"))
    if width:
        return width // heads
    derived = _config_class_head_dim(tc)
    if derived:
        return derived
    hidden = _pos_int(hidden) or _pos_int(_first(tc, "hidden_size", "d_model", "n_embd"))
    return hidden // heads if hidden else None


def attention_input_width(config: Dict[str, Any], hidden_size: int) -> int:
    """What an attention layer's Q / K / V projections read: `attention_hidden_size` when an
    attention-bearing config states it (Zamba feeds its shared attention the hidden state
    concatenated with the embeddings, 2 x hidden), else the hidden size."""
    tc = config if isinstance(config, dict) else {}
    if _pos_int(_first(tc, *ATTENTION_HEAD_KEYS)):
        width = _pos_int(tc.get("attention_hidden_size"))
        if width:
            return width
    return int(hidden_size)


def encoder_tokens(config: Dict[str, Any]) -> Optional[int]:
    """Tokens an interleaved cross-attention layer caches per request: the tower's output, read
    as an encoder-decoder's encoder block is (``_block_output_tokens``: a stated length, else an
    image tower's patches per side squared plus the class token, per tile). None when the config
    does not state it, and the caller then charges the full sequence instead."""
    return _block_output_tokens(_tower(config) or config.get("vision_config"))[0]


def _classify_layer_type_string(s: str) -> Dict[str, Optional[str]]:
    """Map one per-layer kind name to the two axes, by exact name.

    By the table above, not by substring. ``"linear_attention"`` contains the
    substring ``"attention"``, and a substring test is exactly how Gated DeltaNet
    layers came to be charged a full KV cache they never allocate.

    A name the table does not know comes back as full attention with
    ``unknown`` set: the conservative over-estimate, flagged by the caller.
    """
    key = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    kind = LAYER_KINDS.get(key)
    if kind is None:
        return {"attn": ATTN_FULL, "recurrent": None, "unknown": str(s)}
    return dict(kind)


def _recurrent_mechanism(name: Optional[str], tc: Dict[str, Any], layer_name: str = "") -> Optional[str]:
    """Resolve an ``"auto"`` recurrent kind from the keys the config carries.

    The same layer name means different state shapes in different families --
    ``linear_attention`` is Gated DeltaNet in Qwen3-Next/3.5 and plain linear
    attention in MiniMax -- so the config's own dimension keys decide, never the
    family name.
    """
    if name != _AUTO:
        return name
    if "linear_num_value_heads" in tc or "linear_key_head_dim" in tc:
        return MECH_GDN
    if isinstance(tc.get("linear_attn_config"), dict):
        return MECH_KDA
    if "lru_width" in tc:
        return MECH_RGLRU
    if any(k in tc for k in ("ssm_state_size", "mamba_d_state", "state_size", "d_state")):
        return MECH_MAMBA2
    if str(layer_name).strip().lower() in ("linear_attention", "linear"):
        # Plain linear attention: a heads x d x d state per layer, stated by the
        # attention geometry the config does carry.
        return MECH_LIGHTNING
    # An unnamed recurrent mixer with no state keys at all: sized by the GDN
    # fallbacks and flagged as an ESTIMATE in `_finish`.
    return MECH_GDN


def _expand_attention_types(value) -> Optional[List[str]]:
    """GPT-Neo's compact layout, ``[[["global", "local"], 12], ...]``: each entry a pattern of
    kind names and how many times it repeats, in order (``GPTNeoConfig.
    expand_attention_types_params``). None for anything else."""
    if not (isinstance(value, (list, tuple)) and value):
        return None
    out: List[str] = []
    for item in value:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return None
        pattern, repeat = item
        if not (isinstance(pattern, (list, tuple)) and pattern
                and all(isinstance(x, str) for x in pattern) and isinstance(repeat, int)):
            return None
        out.extend(list(pattern) * repeat)
    return out or None


def per_layer_names(tc: Dict[str, Any]):
    """``(names, key)``: every layer's kind name, from whichever key the config states it in.

    `layer_types` is transformers' own key. GPT-Neo states the same thing as `attention_types`
    (a compact pattern its config class expands into `attention_layers`, which the hub config
    also ships). The names resolve through LAYER_KINDS like any other list. ``(None, None)``
    when the config states no per-layer list.
    """
    names = tc.get("layer_types")
    if isinstance(names, list) and names:
        return list(names), "layer_types"
    names = _expand_attention_types(tc.get("attention_types"))
    if names:
        return names, "attention_types"
    names = tc.get("attention_layers")
    if isinstance(names, list) and names and all(isinstance(x, str) for x in names):
        return list(names), "attention_layers"
    return None, None


def unstated_layers_note(stated: int, n: int, key: str) -> str:
    """The note (and warning) for a per-layer list that names fewer layers than the depth."""
    return (
        f"`{key}` names {stated} of {n} layers; the other {n - stated} are charged "
        "FULL-ATTENTION KV -- an over-estimate, not a measurement. The list should name "
        "every layer."
    )


def _fit_names(names, n: int, key: str):
    """``(names, notes)``: a per-layer list fitted to the depth. Layers the list does not reach
    are charged full attention, with a note and a warning -- never left free."""
    names = list(names)
    if len(names) >= n:
        return names[:n], []
    msg = unstated_layers_note(len(names), n, key)
    warnings.warn(msg, stacklevel=3)
    return names + ["full_attention"] * (n - len(names)), [msg]


def _charge_unstated(plan, start: int, n: int, key: str, notes: List[str]) -> None:
    """Layers ``start..n-1`` that an index-free per-layer list did not reach: full attention,
    noted, warned, and the plan marked approximate."""
    if start >= n:
        return
    msg = unstated_layers_note(start, n, key)
    warnings.warn(msg, stacklevel=3)
    notes.append(msg)
    for i in range(start, n):
        plan["attn"][i] = ATTN_FULL
        plan["recurrent"][i] = None
    plan["_unknown"] = True


def _apply_layer_names(plan, names, tc, n, notes, key="layer_types"):
    """Fill both axes from a list of per-layer kind names; flag unknown names.

    A stack that names FFN-only slots (Nemotron-H's ``moe`` / ``mlp``: the table's ``ffn_slot``
    kinds) gives the FFN its own layer: then only those slots carry one, and the mixers (and
    ``no_op`` layers) carry none. Every other stack pairs
    an FFN with each mixer. The answer is left in ``plan["_ffn"]`` (None = every layer).
    A list shorter than the depth charges the layers it does not reach full attention.
    """
    names, short = _fit_names(names, n, key)
    if short:
        notes.extend(short)
        plan["_unknown"] = True
    unknown = []
    ffn_only = []
    for i, lt in enumerate(names):
        k = _classify_layer_type_string(lt)
        if k.get("unknown"):
            unknown.append(k["unknown"])
        ffn_only.append(bool(not k.get("unknown") and k.get("ffn_slot")))
        plan["attn"][i] = ATTN_SLIDING if k["attn"] == ATTN_CHUNKED else k["attn"]
        if k["attn"] == ATTN_CHUNKED:
            plan.setdefault("_chunked", True)
        plan["recurrent"][i] = _recurrent_mechanism(k["recurrent"], tc, lt)
    plan["_ffn"] = ffn_only if any(ffn_only) else None
    if unknown:
        msg = (
            f"layer kinds {sorted(set(unknown))} are not in the simulator's layer-kind "
            f"table, so {len(unknown)} layers are charged FULL-ATTENTION KV -- an "
            "over-estimate, not a measurement. Add the kind to layer_plan.LAYER_KINDS."
        )
        notes.append(msg)
        warnings.warn(msg, stacklevel=3)
        plan["_unknown"] = True
    return plan


def resolve_layer_plan(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a config into a canonical per-layer plan.

    Returns ``None`` when the config declares no hybrid structure at all, so
    callers keep their existing uniform-depth behavior for ordinary
    transformers. Returns a plan with a ``notes`` list when a dialect is
    recognized but cannot be modelled faithfully -- a loud approximation beats a
    silent wrong number.
    """
    config = with_class_geometry(config)
    tc = _text_config(config)
    n = depth(tc)
    if not n:
        return None

    model_type = str(_first(tc, "model_type", default="") or "").lower()
    plan = _blank(n)
    notes: List[str] = []

    # --- 1. Explicit per-layer list (HF standard: Qwen3.5/3.6, granite-4, LFM2,
    # gpt-oss, Gemma-3n, MiniMax, Llama 4). Names resolve through LAYER_KINDS;
    # a recurrent name's mechanism comes from the config's own keys. -----------
    layer_types, names_key = per_layer_names(tc)
    if layer_types:
        _apply_layer_names(plan, layer_types, tc, n, notes, names_key)
        return _finish(plan, config, names_key, notes, ffn=plan.pop("_ffn", None))

    # --- 1a. A block-type pattern that repeats over the depth (RecurrentGemma's
    # `block_types`, e.g. ["recurrent", "recurrent", "attention"]). Its one
    # attention kind is bounded by the window the config states, as a uniform
    # stack's would be. -------------------------------------------------------
    block_types = tc.get("block_types")
    if isinstance(block_types, (list, tuple)) and block_types and isinstance(block_types[0], str):
        cycle = [block_types[i % len(block_types)] for i in range(n)]
        _apply_layer_names(plan, cycle, tc, n, notes)
        plan.pop("_ffn", None)
        if _window(tc):
            plan["attn"] = [ATTN_SLIDING if a == ATTN_FULL else a for a in plan["attn"]]
        return _finish(plan, config, "block_types", notes)

    # --- 1b. SmallThinker: a 0/1 mask over layers, 1 = sliding --------------
    # Polarity is the whole game here and it is the opposite of what the name
    # suggests to a casual reader: PowerInfer's own configuration_smallthinker.py
    # documents "0 for normal attention, 1 for SWA", and modeling_smallthinker.py
    # applies the sliding mask when the entry == 1. Inverting this would mark the
    # 13 genuinely-global layers as windowed and the 39 windowed ones as global,
    # which under-counts KV instead of over-counting it.
    swl = tc.get("sliding_window_layout")
    if isinstance(swl, list) and swl and set(swl) <= {0, 1}:
        for i, is_swa in enumerate(swl[:n]):
            plan["attn"][i] = ATTN_SLIDING if is_swa else ATTN_FULL
        _charge_unstated(plan, len(swl), n, "sliding_window_layout", notes)
        return _finish(plan, config, "sliding_window_layout", notes)

    # --- 2. Kimi Linear: explicit index lists, ONE-INDEXED -------------------
    lac = tc.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("full_attn_layers"):
        full = set(lac["full_attn_layers"])
        kda = set(lac.get("kda_layers") or [])
        # ONE-indexed: Kimi's modelling code tests `layer_idx + 1` against the lists, which
        # together cover 1..n. Reading them as 0-indexed silently drops the last layer and
        # shifts every other one. Only a list naming layer 0 is read as 0-indexed. (Requiring
        # min == 1 dropped the last full layer whenever `kda_layers` was absent.)
        off = 0 if 0 in (full | kda) else 1
        for i in range(n):
            plan["attn"][i] = ATTN_FULL if (i + off) in full else None
            plan["recurrent"][i] = None if (i + off) in full else MECH_KDA
        return _finish(plan, config, "linear_attn_config.full_attn_layers", notes)

    # --- 3. MiniMax: attn_type_list, 1 = softmax attention, 0 = lightning ----
    atl = tc.get("attn_type_list")
    if isinstance(atl, list) and atl:
        for i, t in enumerate(atl[:n]):
            if t:
                plan["attn"][i] = ATTN_FULL
            else:
                plan["recurrent"][i] = MECH_LIGHTNING
        _charge_unstated(plan, len(atl), n, "attn_type_list", notes)
        return _finish(plan, config, "attn_type_list", notes)

    # --- 4. Nemotron-H: a pattern string, one character per layer, in transformers'
    # own translation (NemotronHConfig._pattern_to_list): M=mamba, E=moe, *=attention,
    # -=mlp. The names then resolve through the table like any per-layer list. ----------
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat:
        # Unlike every other family here, Nemotron-H gives the FFN its OWN layer
        # slot rather than pairing one with each mixer. Charging an FFN on all 56
        # layers instead of the 25 that have one is a ~2.2x over-count on the
        # single largest parameter block in the model.
        names = [_NEMOTRON_PATTERN.get(ch, ch) for ch in pat]
        _apply_layer_names(plan, names, tc, n, notes, "hybrid_override_pattern")
        ffn = plan.pop("_ffn", None) or [False] * n
        return _finish(plan, config, "hybrid_override_pattern", notes, ffn=ffn)

    # --- 5. layers_block_type. Resolves through the table like `layer_types`
    # (Nemotron-H as transformers 5.12 saves it: mamba / moe / attention / mlp).
    # Zamba / Zamba2 name a 'hybrid' layer: the SHARED attention+MLP block runs
    # beside that layer's Mamba mixer. -----------------------------------------
    lbt = tc.get("layers_block_type")
    if isinstance(lbt, list) and lbt and not any(str(t).lower() == "hybrid" for t in lbt):
        _apply_layer_names(plan, lbt, tc, n, notes, "layers_block_type")
        return _finish(plan, config, "layers_block_type", notes, ffn=plan.pop("_ffn", None))
    if isinstance(lbt, list) and lbt:
        _apply_layer_names(plan, lbt, tc, n, notes, "layers_block_type")
        plan.pop("_ffn", None)
        # The MLP is the shared block's: it runs on every layer that invokes the block
        # (every layer with attention, an unknown name included, as charged), and on no
        # plain Mamba layer.
        ffn = [a is not None for a in plan["attn"]]
        # Zamba2's attention+MLP block is SHARED: `num_mem_blocks` distinct
        # blocks are re-invoked across every hybrid layer. Each invocation
        # allocates its own KV and streams the block's weights, so the runtime
        # counts above stand -- but the weights exist only once per block, so
        # parameter counting must use num_mem_blocks. Counting them per-layer
        # doubles the model. Zamba (v1) states no count: it has one shared block,
        # the default both transformers config classes give.
        blocks = int(tc.get("num_mem_blocks") or 1)
        return _finish(
            plan, config, "layers_block_type", notes,
            ffn=ffn,
            attention_param_layers=blocks,
            ffn_param_layers=blocks,
        )

    # --- 6. Bamba: explicit attention layer indices, rest Mamba-2 ------------
    ali = tc.get("attn_layer_indices")
    if isinstance(ali, list) and ali:
        s = set(ali)
        for i in range(n):
            plan["attn"][i] = ATTN_FULL if i in s else None
            plan["recurrent"][i] = None if i in s else MECH_MAMBA2
        return _finish(plan, config, "attn_layer_indices", notes)

    # --- 7. Qwen3-Next: interval only, no per-layer list ---------------------
    iv = tc.get("full_attention_interval")
    if isinstance(iv, int) and iv > 1:
        # Verified against Qwen3.6, which ships both keys: the full-attention
        # layer is the LAST of each group of `iv`, i.e. i % iv == iv - 1.
        for i in range(n):
            if i % iv == iv - 1:
                plan["attn"][i] = ATTN_FULL
            else:
                plan["recurrent"][i] = MECH_GDN
        return _finish(plan, config, "full_attention_interval", notes)

    # --- 8. Jamba: attention every `period` layers at `offset` ---------------
    period = tc.get("attn_layer_period")
    if isinstance(period, int) and period > 1:
        offset = tc.get("attn_layer_offset", 0) or 0
        for i in range(n):
            if i % period == offset:
                plan["attn"][i] = ATTN_FULL
            else:
                plan["recurrent"][i] = MECH_MAMBA1
        return _finish(plan, config, "attn_layer_period", notes)

    # --- 9. Hymba: PARALLEL hybrid. Every layer runs attention and Mamba side
    # by side, so both axes are set on every layer. The listed indices are
    # global attention; the rest are windowed, not absent. ---------------------
    gai = tc.get("global_attn_idx")
    if isinstance(gai, list) and gai:
        s = set(gai)
        windowed = bool(tc.get("sliding_window"))
        for i in range(n):
            if i in s:
                plan["attn"][i] = ATTN_FULL
            else:
                plan["attn"][i] = ATTN_SLIDING if windowed else ATTN_FULL
            plan["recurrent"][i] = MECH_MAMBA1
        return _finish(plan, config, "global_attn_idx", notes, parallel=True)

    # --- 10. Cross-attention layers interleaved with self-attention (Mllama's
    # `cross_attention_layers`): their cache holds the encoder's tokens. -------
    cal = tc.get("cross_attention_layers")
    if isinstance(cal, list) and cal:
        s = set(cal)
        for i in range(n):
            plan["attn"][i] = ATTN_CROSS if i in s else ATTN_FULL
        return _finish(plan, config, "cross_attention_layers", notes, keep_uniform=True)

    # --- 11. SambaY / Phi-4-flash: an upper bound, never an under-count ----------
    # SambaY interleaves Mamba with sliding-window attention in its self-decoder,
    # then its YOCO cross-decoder reuses ONE full-attention KV cache -- written by
    # a single layer and read by the cross-attention layers after it -- which
    # grows with the context. The config states the Mamba spacing (`mb_per_layer`,
    # a Mamba mixer every that-many layers) and the window, but not which layer
    # writes the shared cache, nor any SSM dimension (they live in the modelling
    # code). Recognised by that key: nothing else ships `mb_per_layer`.
    #
    # Charged: every non-Mamba layer the window, plus ONE layer the full context
    # for the shared cache. That is never below the published design (the
    # self-decoder's windowed layers + one growing cache) and never charges the
    # cross-attention layers a cache of their own. Every layer windowed (the old
    # reading) dropped the growing cache and under-counted at long context; every
    # layer full would charge 16x that cache. The Mamba state falls back to
    # default geometry and is flagged as an estimate below.
    if "mb_per_layer" in tc:
        mb = int(tc.get("mb_per_layer") or 0)
        for i in range(n):
            if mb > 0 and i % mb == 0:
                plan["recurrent"][i] = MECH_MAMBA1
            else:
                plan["attn"][i] = ATTN_SLIDING if _window(tc) else ATTN_FULL
        shared_cache = next(
            (i for i in range(n // 2, n) if plan["attn"][i] is not None),
            next((i for i in range(n) if plan["attn"][i] is not None), None),
        )
        if shared_cache is not None:
            plan["attn"][shared_cache] = ATTN_FULL
        return _finish(
            plan, config, "phi4flash",
            notes + [
                "SambaY (Phi-4-flash): an UPPER BOUND. Every non-Mamba layer is "
                "charged the window and one layer the full context for the YOCO "
                "cache the cross-attention layers share; the config does not say "
                "which layers are windowed and which only read the shared cache."
            ],
            approximate=True,
        )

    # The structural fallbacks: no key above states the layout, so the config's own
    # dimension keys decide -- an SSM state, attention heads, or both. A family name decides
    # nothing while a structural key is present.
    no_attention_heads = not any(tc.get(k) for k in ATTENTION_HEAD_KEYS)
    has_ssm_state = any(
        tc.get(k) is not None for k in ("state_size", "d_state", "ssm_state_size", "mamba_d_state")
    )

    # --- 12. Parallel hybrid: an SSM state AND attention heads, no layout key. Falcon-H1 is
    # exactly this (every layer runs attention and Mamba-2 side by side). For any other config
    # of this shape it is the upper bound: every layer charged a growing KV and a fixed state,
    # never a layer left free. Graphing it as pure Mamba charged a serving request 0 KV. ------
    if has_ssm_state and not no_attention_heads:
        for i in range(n):
            plan["attn"][i] = ATTN_FULL
            plan["recurrent"][i] = MECH_MAMBA2
        msg = (
            "the config states an SSM state and attention heads but no layout key saying which "
            "layers are which; every layer is charged attention KV AND recurrent state (the "
            "parallel-hybrid reading, as Falcon-H1 runs) -- an upper bound if the model "
            "interleaves instead."
        )
        warnings.warn(msg, stacklevel=2)
        return _finish(plan, config, "parallel_hybrid", notes + [msg], parallel=True)

    # --- 13. Pure SSM: no attention heads of any spelling, and an SSM state -- or, with no
    # state key either, a state-space family name (configs that omit the state keys). --------
    if no_attention_heads and (has_ssm_state or model_type in (
        "mamba", "mamba2", "falcon_mamba", "s4", "ssm", "state-space"
    )):
        mech = MECH_MAMBA2 if (model_type == "mamba2" or "n_groups" in tc) else MECH_MAMBA1
        for i in range(n):
            plan["recurrent"][i] = mech
        # A pure SSM stack has no FFN at all -- the gated mixer *is* the block.
        # Charging one per layer nearly doubled Mamba-Codestral (13.7B vs 7.3B).
        return _finish(plan, config, "pure_ssm", notes, ffn=[False] * n)

    # --- 14. Attention-free recurrent stacks other than Mamba, by the keys
    # that size their state: RWKV (`attention_hidden_size` with no attention
    # heads) and xLSTM's matrix memory (`qk_dim_factor` / `v_dim_factor`). ----
    if no_attention_heads and ("qk_dim_factor" in tc or "v_dim_factor" in tc):
        plan["recurrent"] = [MECH_MLSTM] * n
        return _finish(plan, config, "mlstm_stack", notes)
    if no_attention_heads and "attention_hidden_size" in tc:
        plan["recurrent"] = [MECH_RWKV] * n
        return _finish(plan, config, "rwkv_stack", notes)

    # --- 15. The layout the config's own class derives (``class_layout``), when it places a
    # layer that is not attention: LFM2's `full_attn_idxs` with no `layer_types` -- the class
    # builds the other layers as short convolutions, each with its conv state and maps. A layout
    # of attention kinds alone stays with the stated-names path (``stated_layer_types``). -----
    names = class_layout(tc)
    if names:
        _apply_layer_names(plan, names, tc, n, notes)
        if any(plan["recurrent"]):
            return _finish(plan, config, "config_class", notes, ffn=plan.pop("_ffn", None))
    return None


def synthesized_layer_types(tc: Dict[str, Any]) -> Optional[List[str]]:
    """Per-layer names for an interleave stated as a period, not a list.

    Gemma 2/3 configs declare `sliding_window_pattern: N` -- one full-attention
    layer every N, the last of each group -- instead of shipping `layer_types`.
    Without this every layer reads as full attention and the global layers'
    window is lost. None when the config states no such period.
    """
    n = depth(tc)
    pattern = tc.get("sliding_window_pattern")
    if not (
        n and isinstance(pattern, int) and not isinstance(pattern, bool) and pattern > 1
        and _window(tc)
    ):
        return None
    return [
        "full_attention" if i % pattern == pattern - 1 else "sliding_attention"
        for i in range(int(n))
    ]


#: Keys a config class may need to derive its layout, for the reduced retry when a
#: config does not validate as a whole.
_LAYOUT_KEYS = (
    "num_hidden_layers", "sliding_window", "use_sliding_window", "max_window_layers",
    "sliding_window_pattern", "layer_types", "attention_chunk_size", "no_rope_layers",
)

_CLASS_LAYOUT_CACHE: Dict[str, Any] = {}

#: The first transformers release whose config classes derive ``layer_types`` (Gemma2Config,
#: Qwen2Config, ...). Before it no class states one, for any family. Classes gained the list in
#: different releases after it (4.53's Qwen2MoeConfig still has none), so from 4.53 on a class
#: that states none is read as "every layer takes the window" only when the config's own
#: `max_window_layers` does not say otherwise (``max_window_layers_layout``).
LAYER_LAYOUTS_SINCE = (4, 53)


def transformers_states_layer_layouts() -> bool:
    """Whether the installed transformers derives ``layer_types`` in its config classes.

    Only then does a class that states none mean "no per-layer layout" (Mistral: every layer
    takes the window). An older release states none for Gemma 2 either, which alternates.
    """
    try:
        import transformers
    except Exception:
        return False
    m = re.match(r"(\d+)\.(\d+)", str(getattr(transformers, "__version__", "")))
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= LAYER_LAYOUTS_SINCE


def class_layout_names(rule: Dict[str, Any], tc: Dict[str, Any]) -> Optional[List[str]]:
    """The per-layer names a config class's layout rule (``config_class_defaults.json``'s
    ``layouts``, written by ``scripts/config_class_defaults.py``) gives a config, or None for a
    class that derives no layout (the window is on every layer, unless the config's own
    `max_window_layers` says otherwise). A ``period`` is layer i = ``names[i % len(names)]``, the
    last layer ``last`` where the class forces it; ``max_window_layers`` windows the layers from
    the config's `max_window_layers` on, the class's default where the config states none. A class
    with a `use_sliding_window` switch (``switch``: its default) builds every layer full when the
    config leaves the switch to it and it is off -- a window the config states is then unused
    (Qwen2Config, whose default is off)."""
    n = depth(tc)
    kind = rule.get("rule")
    if not n or kind == "none":
        return None
    if kind in ("period", "max_window_layers") and "switch" in rule and \
            tc.get("use_sliding_window", rule["switch"]) is False:
        return ["full_attention"] * n
    if kind == "period":
        names = [str(x) for x in rule.get("names") or ()]
        if not names:
            return None
        out = [names[i % len(names)] for i in range(n)]
        if rule.get("last"):
            out[-1] = str(rule["last"])
        return out
    if kind == "max_window_layers":
        mwl = tc.get("max_window_layers")
        if not (isinstance(mwl, int) and not isinstance(mwl, bool)):
            mwl = rule.get("default")
        if not (isinstance(mwl, int) and not isinstance(mwl, bool)):
            return None
        return ["full_attention" if i < mwl else "sliding_attention" for i in range(n)]
    return None


def _json_same(a: Any, b: Any) -> bool:
    """Equal as JSON values (``True`` is not ``1``, ``4.0`` is not ``4``)."""
    import json

    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except Exception:
        return False


def layout_rule_applies(rule: Dict[str, Any], tc: Dict[str, Any]) -> bool:
    """Whether a shipped layout rule holds for this config: every key the class builds its layout
    from besides the depth, the window and its switch (the rule's ``depends``, found by
    ``scripts/config_class_defaults.py`` by building the class at other values) is left out or
    stated at a value the rule was checked at -- the class's default, the value it stores, or a
    per-layer list repeating the ``period`` the class derives. A config that states another value
    (AfmoeConfig's `global_attn_every_n_layers: 2`, a Llama 4 `no_rope_layer_interval`, a Qwen2-VL
    `max_window_layers` under the window, LFM2's `full_attn_idxs`) gets a layout the rule does not
    describe: the class is asked, and without it every windowed layer is charged full."""
    n = depth(tc)
    for key, entry in (rule.get("depends") or {}).items():
        if key not in tc:
            continue
        value = tc[key]
        if any(_json_same(value, v) for v in entry.get("values") or ()):
            continue
        unit = entry.get("period")
        if unit and isinstance(value, (list, tuple)) and n and len(value) == n and \
                _json_same(list(value), [unit[i % len(unit)] for i in range(n)]):
            continue
        return False
    return True


def _snapshot_layer_types(tc: Dict[str, Any]):
    """``(True, names)`` from the shipped snapshot's layout rule for the config's model type
    (``class_layout_names``), or None when the snapshot holds no rule for it -- a model type
    transformers 5.12 does not know, or a class whose layout no rule reproduces (Qwen2-MoE's,
    which mixes a period with `max_window_layers`; Zamba's) -- or the config states a key the
    class builds its layout from at a value the rule was not checked at
    (``layout_rule_applies``): the installed transformers is asked then."""
    rule = (_class_default_snapshot().get("layouts") or {}).get(str(tc.get("model_type") or ""))
    if not isinstance(rule, dict) or not layout_rule_applies(rule, tc):
        return None
    return True, class_layout_names(rule, tc)


def _config_class_layer_types(tc: Dict[str, Any]):
    """``(known, names)``: what the model type's own config class says the layers are.

    Read from the shipped snapshot's layout rule (``_snapshot_layer_types``: transformers 5.12's
    classes) for every model type it holds one for, so the layout does not depend on whether or
    which transformers is installed; the installed transformers is asked only about a type the
    snapshot holds no rule for.

    transformers derives a per-layer list for families whose hub configs predate it --
    Gemma 2 alternates sliding and full in `Gemma2Config`, Qwen2 windows the layers from
    `max_window_layers` on -- and states none for a family where every layer takes the
    window (Mistral). ``known`` is False when the installed transformers does not have this
    model type (or cannot read the config), and when it predates `layer_types` altogether
    (before 4.53 a class without one says nothing about the layout), so the caller can stay
    conservative.
    """
    mt = str(tc.get("model_type") or "")
    if not mt:
        return False, None
    shipped = _snapshot_layer_types(tc)
    if shipped is not None:
        return shipped
    try:
        import json

        key = json.dumps(tc, sort_keys=True, default=str)
    except Exception:
        key = None
    if key is not None and key in _CLASS_LAYOUT_CACHE:
        return _CLASS_LAYOUT_CACHE[key]
    result = (False, None)
    layouts = transformers_states_layer_layouts()
    try:
        from transformers import CONFIG_MAPPING

        cls = CONFIG_MAPPING[mt]
    except Exception:
        cls = None
    if cls is not None:
        import logging

        drop = {"model_type", "transformers_version", "architectures", "auto_map", "text_config",
                "vision_config", "audio_config"}
        whole = {k: v for k, v in tc.items() if k not in drop and not k.startswith("_")}
        # the reduced retry keeps every key the class builds its layout from (the shipped rule's
        # ``depends``): without them it would answer for the class's defaults
        rule = (_class_default_snapshot().get("layouts") or {}).get(mt)
        keys = set(_LAYOUT_KEYS) | set((rule or {}).get("depends") or ())
        reduced = {k: whole[k] for k in keys if k in whole}
        tf_log = logging.getLogger("transformers")
        level = tf_log.level
        tf_log.setLevel(logging.ERROR)
        try:
            for kwargs in (whole, reduced):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        obj = cls(**kwargs)
                except Exception:
                    continue
                names = getattr(obj, "layer_types", None)
                if not (isinstance(names, (list, tuple)) and names):
                    # The same list under another key (GPT-Neo's `attention_layers`).
                    try:
                        names = per_layer_names(obj.to_dict())[0]
                    except Exception:
                        names = None
                n = depth(tc)
                if isinstance(names, (list, tuple)) and names and (not n or len(names) == n):
                    result = (True, [str(x) for x in names])
                else:
                    result = (layouts, None)
                break
        finally:
            tf_log.setLevel(level)
    if key is not None:
        _CLASS_LAYOUT_CACHE[key] = result
    return result


def class_layout(tc: Dict[str, Any]) -> Optional[List[str]]:
    """The per-layer names the config's own class derives (``_config_class_layer_types``) where
    the config needs it: it states a window but no layout, or -- with no window -- it states a key
    the class builds the layout from at a value the shipped rule was not checked at
    (``layout_rule_applies``: LFM2's `full_attn_idxs`). None otherwise, and where the class
    cannot be asked (every layer then stays full attention: the over-count)."""
    if not _window(tc):
        rule = (_class_default_snapshot().get("layouts") or {}).get(str(tc.get("model_type") or ""))
        if not isinstance(rule, dict) or layout_rule_applies(rule, tc):
            return None
    return _config_class_layer_types(tc)[1]


def max_window_layers_layout(tc: Dict[str, Any]) -> Optional[List[str]]:
    """Per-layer names from `max_window_layers`, or None.

    transformers documents the key (Qwen2 / Qwen2-MoE / Qwen3 families): "the first
    `max_window_layers` layers will use full attention, while any additional layer afterwards
    will use SWA". It is the config's own statement that the window is NOT on every layer, so
    a config class that derives no `layer_types` (transformers 4.53's Qwen2MoeConfig, whose
    modelling windows only layers >= max_window_layers) does not make the window uniform. It
    only ever takes the window off layers: a release whose modelling windows every layer
    anyway (Qwen3-MoE) is over-counted, never under-counted. None when the key is absent or
    states no full layer (then the window is uniform).
    """
    n = depth(tc)
    mwl = tc.get("max_window_layers")
    if not (n and isinstance(mwl, int) and not isinstance(mwl, bool) and mwl > 0):
        return None
    return ["full_attention" if i < mwl else "sliding_attention" for i in range(n)]


def stated_layer_types(tc: Dict[str, Any]):
    """``(names, source, notes)``: the per-layer kind names a decoder config states.

    In order: an explicit per-layer list (``per_layer_names``: ``layer_types``, or GPT-Neo's
    ``attention_types`` / ``attention_layers``), fitted to the depth -- layers a short list does
    not reach are charged full attention, noted and warned; a period
    (``sliding_window_pattern``); and, for a
    config that carries a window but no layout, the layout its own config class derives
    (transformers); a class that derives none leaves the window uniform unless the config
    states `max_window_layers` (``max_window_layers_layout``). A window whose layers no source places -- the model type is unknown to
    the installed transformers, the release predates 4.53 (whose classes derive no layout
    for any type), or transformers is not installed -- is charged on NO layer: every layer
    is full attention, the over-estimate, and the notes and a warning say so. Clamping every
    layer to the window instead under-counts any model that alternates (Gemma 2's hub config
    is exactly that shape). ``names`` is None for a uniform stack the window, if any, applies
    to as a whole.
    """
    names, key = per_layer_names(tc)
    if names:
        names, notes = _fit_names(names, depth(tc), key)
        return names, "layer_types", notes
    names = synthesized_layer_types(tc)
    if names:
        return names, "sliding_window_pattern", []
    if not _window(tc):
        names = class_layout(tc)
        if names:
            return names, "config_class", []
        return None, "uniform", []
    known, names = _config_class_layer_types(tc)
    if names:
        return names, "config_class", []
    if known:
        names = max_window_layers_layout(tc)
        if names:
            return names, "max_window_layers", []
        return None, "uniform", []
    if resolve_layer_plan(tc) is not None:
        # A hybrid dialect (Hymba's global indices, SambaY's Mamba spacing) places the
        # window itself; the layer plan answers for it.
        return None, "layer_plan", []
    n = depth(tc)
    msg = (
        f"model type {tc.get('model_type')!r} states a {_window(tc)}-token window but not which "
        "layers take it, and the installed transformers does not derive the layout (no config "
        "class for the type, or a release before 4.53, or none installed); every "
        f"layer is charged FULL-ATTENTION KV ({n} layers) -- an over-estimate, not a measurement. "
        "Add `layer_types` to the config to apply the window."
    )
    warnings.warn(msg, stacklevel=3)
    return ["full_attention"] * n, "unplaced_window", [msg]


#: Spellings of the routed-expert count. Hunyuan ships one per layer.
_EXPERT_COUNT_KEYS = (
    "num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts",
    "moe_num_primary_experts",
)


def moe_layers(config: Dict[str, Any]) -> Optional[List[bool]]:
    """Which decoder layers route to experts, from the config's own placement keys.

    None for a model without routed experts. MoE changes no layer's KV; it decides which
    FFN a layer runs. Rules as each family's modelling code applies them (transformers
    5.12): `mlp_layer_types` (dense / sparse), `moe_layers` (Llama 4's index list), a
    per-layer expert count, `expert_layer_period` / `expert_layer_offset` (Jamba:
    i % period == offset), `decoder_sparse_step` + `mlp_only_layers` (Qwen MoE:
    (i + 1) % step == 0), `interleave_moe_layer_step` (Llama 4), `moe_layer_start_index` /
    `moe_layer_end_index` / `moe_layer_interval` (ERNIE 4.5), and `first_k_dense_replace` /
    `moe_layer_freq` (DeepSeek, Kimi, GLM: i >= k and i % freq == 0). A stack that names its
    FFN slots (Nemotron-H's `moe` / `mlp`) says it per layer. With none of these, every
    layer is MoE.
    """
    tc = _text_config(config)
    n = depth(tc)
    if not n:
        return None
    mlt = tc.get("mlp_layer_types")
    if isinstance(mlt, list) and len(mlt) == n:
        return [str(t).lower() == "sparse" for t in mlt]
    count = _first(tc, *_EXPERT_COUNT_KEYS)
    if isinstance(count, list) and len(count) == n:
        return [isinstance(c, (int, float)) and c > 1 for c in count]
    if not (isinstance(count, (int, float)) and not isinstance(count, bool) and count > 1):
        return None
    for key in ("hybrid_override_pattern", "layers_block_type", "layer_types"):
        names = tc.get(key)
        if isinstance(names, str):
            names = [_NEMOTRON_PATTERN.get(ch, ch) for ch in names]
        if isinstance(names, list) and len(names) == n and any(
            str(x).lower() in ("moe", "mlp", "mlp_only", "ffn", "dense") for x in names
        ):
            return [str(x).lower() == "moe" for x in names]
    idx = tc.get("moe_layers")
    if isinstance(idx, list) and idx and all(isinstance(i, int) for i in idx):
        s = set(idx)
        return [i in s for i in range(n)]
    period = tc.get("expert_layer_period")
    if isinstance(period, int) and period > 1:
        off = int(tc.get("expert_layer_offset") or 0)
        return [i % period == off for i in range(n)]
    step = tc.get("decoder_sparse_step")
    mlp_only = set(tc.get("mlp_only_layers") or ())
    if (isinstance(step, int) and step > 1) or mlp_only:
        step = step if isinstance(step, int) and step > 0 else 1
        return [(i + 1) % step == 0 and i not in mlp_only for i in range(n)]
    inter = tc.get("interleave_moe_layer_step")
    if isinstance(inter, int) and inter > 1:
        return [(i + 1) % inter == 0 for i in range(n)]
    if "moe_layer_start_index" in tc or "moe_layer_interval" in tc:
        start = int(tc.get("moe_layer_start_index") or 0)
        end = tc.get("moe_layer_end_index")
        end = n - 1 if end is None or int(end) < 0 else int(end)
        every = int(tc.get("moe_layer_interval") or 1) or 1
        return [(i + 1) % every == 0 and start <= i <= end for i in range(n)]
    k = int(tc.get("first_k_dense_replace") or 0)
    freq = tc.get("moe_layer_freq")
    freq = int(freq) if isinstance(freq, int) and freq > 0 else 1
    return [i >= k and i % freq == 0 for i in range(n)]


#: Spellings of the width of an MoE model's DENSE (non-routed) layers, for the families whose
#: `intermediate_size` is the expert width (transformers 5.12): Llama 4 builds its dense layers at
#: `intermediate_size_mlp`, MiniMax-M3 at `dense_intermediate_size`, Cohere2-MoE at
#: `prefix_dense_intermediate_size`. A family that states none of these (Qwen MoE, DeepSeek, Jamba,
#: Nemotron-H) builds them at `intermediate_size` and names the experts' width apart.
DENSE_FFN_WIDTH_KEYS = ("intermediate_size_mlp", "dense_intermediate_size",
                        "prefix_dense_intermediate_size")


#: Spellings of a shared expert's own width (the whole MLP every MoE layer runs beside its routed
#: experts), in transformers' config classes (5.12): Qwen2-MoE / Qwen3-Next / Qwen3.5-MoE /
#: Laguna `shared_expert_intermediate_size`, Granite-MoE-Shared / Granite 4 / MiniMax-M3
#: `shared_intermediate_size` (0: none), Nemotron-H and Bailing (Ling)
#: `moe_shared_expert_intermediate_size`.
SHARED_EXPERT_WIDTH_KEYS = ("shared_expert_intermediate_size", "shared_intermediate_size",
                            "moe_shared_expert_intermediate_size")
#: Spellings of a shared-expert COUNT, whose MLP is that many routed experts wide (DeepSeek V2 /
#: V3 / V4, Kimi, GLM-4-MoE, dots1, Mistral 4 `n_shared_experts`; AFMoE, EXAONE-MoE, HY-V3,
#: Cohere2-MoE `num_shared_experts`; ERNIE 4.5 / Aria `moe_num_shared_experts`; Hunyuan's
#: per-layer `num_shared_expert`).
SHARED_EXPERT_COUNT_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts",
                            "num_shared_expert")
#: The width of one routed expert: `moe_intermediate_size` (or a per-layer list of it), else
#: `expert_intermediate_size`, SmallThinker's `moe_ffn_hidden_size`, else -- Mixtral, Llama 4,
#: Aria, Cohere2-MoE -- `intermediate_size`.
EXPERT_WIDTH_KEYS = ("moe_intermediate_size", "expert_intermediate_size", "moe_ffn_hidden_size",
                     "intermediate_size")
#: Config classes whose modelling builds a shared expert that no config key states: the class
#: itself is the statement (transformers 5.12, `Llama4TextMoe.shared_expert =
#: Llama4TextMLP(config)`; `HunYuanMoEV1Moe.shared_mlp = HunYuanMoEV1MLP(config)`). One shared
#: expert, one routed expert wide.
CLASS_SHARED_EXPERTS = {"llama4_text": 1, "llama4": 1, "hunyuan_v1_moe": 1}


def _largest(v) -> Optional[int]:
    """A positive int, or the largest of a per-layer list of them (Hunyuan states its widths and
    counts per layer): the conservative reading of a list."""
    if isinstance(v, (list, tuple)):
        vals = [_pos_int(x) for x in v]
        vals = [x for x in vals if x]
        return max(vals) if vals else None
    return _pos_int(v)


def routed_expert_count(config: Dict[str, Any]) -> int:
    """The routed experts on an MoE layer (any spelling, ``_EXPERT_COUNT_KEYS``; the largest of a
    per-layer list), 0 for a model without routed experts."""
    n = _largest(_first(_text_config(config), *_EXPERT_COUNT_KEYS))
    return n if n and n > 1 else 0


def routed_expert_width(config: Dict[str, Any]) -> Optional[int]:
    """The width of one routed expert (``EXPERT_WIDTH_KEYS``), or None when nothing states it."""
    tc = _text_config(config)
    for k in EXPERT_WIDTH_KEYS:
        v = _largest(tc.get(k))
        if v:
            return v
    return None


def shared_expert_width(config: Dict[str, Any]) -> int:
    """The total width of the shared expert(s) an MoE layer runs beside its routed experts, as
    the model's own modelling builds it; 0 for none (or for a model without routed experts).

    In order: a stated width (``SHARED_EXPERT_WIDTH_KEYS``, a stated 0 meaning none); a stated
    count (``SHARED_EXPERT_COUNT_KEYS``) times one routed expert's width; the config class's own
    count (``CLASS_SHARED_EXPERTS``) times that width. A graph without it sized DeepSeek-V3 2.55B
    parameters below the checkpoint, and Llama 4 Maverick 3.0B below.
    """
    tc = _text_config(config)
    if not routed_expert_count(config):
        return 0
    for k in SHARED_EXPERT_WIDTH_KEYS:
        v = tc.get(k)
        if v is None:
            continue
        w = _largest(v)
        return w or 0
    count = None
    for k in SHARED_EXPERT_COUNT_KEYS:
        v = tc.get(k)
        if v is not None:
            count = _largest(v) or 0
            break
    if count is None:
        count = CLASS_SHARED_EXPERTS.get(str(tc.get("model_type") or "").lower(), 0)
    width = routed_expert_width(config) or 0
    return int(count * width)


#: How a config states its FFN is gated (gate, up and down: three matrices) apart from its
#: activation's name, in transformers' config classes (5.12): the T5 family (T5, mT5, UMT5,
#: LongT5, UDOP, Pop2Piano) writes `feed_forward_proj: "gated-<act>"` and, since 4.25, the derived
#: `is_gated_act`; DINOv3 says `use_gated_mlp`.
GATED_FFN_FLAG_KEYS = ("is_gated_act", "use_gated_mlp")
#: Families whose FFN is gated (gate, up, down) however their activation is spelled, or when the
#: config states none: an architectural fact of each family's modelling (Gemma's GeGLU says
#: `gelu_pytorch_tanh`; LFM2 names its matrices w1 / w2 / w3 and SmallThinker states no
#: activation). Read by model type and by architecture name, by the parameter counter and the
#: GenZ graph alike.
GATED_FFN_MODEL_TYPES = frozenset({
    'llama', 'llama4', 'mistral', 'mixtral', 'qwen2', 'qwen3', 'qwen2_moe', 'qwen3_moe',
    # Qwen3.5 MoE, including the text tower of the multimodal wrapper. Without
    # these, a 3-matrix (gate/up/down) expert is charged as 2 -- a 33%
    # under-count -- resting entirely on `hidden_act` being present, which is
    # exactly the unreliable signal this set exists to bypass.
    'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next', 'qwen3_5',
    'gemma', 'gemma2', 'gemma3', 'phi3', 'phi4', 'yi', 'deepseek', 'deepseek_v2', 'deepseek_v3',
    'stablelm', 'starcoder2', 'cohere', 'command_r', 'olmo', 'olmo2', 'granite', 'minicpm',
    'internlm2', 'baichuan', 'exaone',
    # Hybrid/recurrent families. All of these ship SwiGLU FFNs but name
    # the matrices w1/w2/w3 (LFM2) or omit `hidden_act` entirely, so the
    # activation-string fallback misses them and charges 2 matrices
    # instead of 3 -- a flat 33% under-count on the model's single
    # largest weight block. LFM2-2.6B came out at 2.04B against 2.57B.
    'lfm2', 'jamba', 'zamba', 'zamba2', 'bamba', 'falcon_h1',
    'granitemoehybrid', 'kimi_linear', 'minimax_text_01', 'minimax_m1',
    'minimax_m2', 'qwen3_5_text', 'hymba', 'plamo2', 'smallthinker',
    'llama4_text', 'deepseek_v32', 'qwen3_moe', 'qwen3_5_moe_text', 'glm4_moe', 'gpt_oss',
    # ChatGLM's dense_h_to_4h is 2 x ffn_hidden_size wide (SwiGLU), its config states no act
    'chatglm', 'glm', 'glm4',

})


def gated_ffn_by_family(config: Dict[str, Any]) -> bool:
    """Whether the config's family builds a gated FFN (``GATED_FFN_MODEL_TYPES``, by its model
    type or its architecture's name)."""
    tc = config if isinstance(config, dict) else {}
    if str(tc.get("model_type") or "").lower() in GATED_FFN_MODEL_TYPES:
        return True
    arch = " ".join(str(a) for a in (tc.get("architectures") or [])).lower()
    return bool(arch) and any(name in arch for name in GATED_FFN_MODEL_TYPES)


#: Config classes whose FFN is gated whatever its activation is called: Pix2Struct's vision and
#: text MLPs build `wi_0` and `wi_1` (transformers 5.12) and their configs say only
#: `dense_act_fn: gelu_new`.
CLASS_GATED_FFN = frozenset({"pix2struct_vision_model", "pix2struct_text_model"})
GATED_FFN_PROJ_KEYS = ("feed_forward_proj",)


def stated_gated_ffn(config: Dict[str, Any]) -> Optional[bool]:
    """Whether the config says its FFN is gated: True / False where it states it
    (``GATED_FFN_FLAG_KEYS``, or a ``feed_forward_proj`` spelled `gated-<act>` or not), None where
    it does not -- the caller then decides from the activation and the model type, as before.
    flan-T5's `gated-gelu` names its activation `gelu_new`: read from the activation alone its
    3-matrix FFN was graphed as 2 (2.35B of its 2.78B parameters)."""
    tc = config if isinstance(config, dict) else {}
    for k in GATED_FFN_FLAG_KEYS:
        v = tc.get(k)
        if isinstance(v, bool):
            return v
    for k in GATED_FFN_PROJ_KEYS:
        v = tc.get(k)
        if isinstance(v, str) and v:
            return v.lower().startswith("gated-")
    if str(tc.get("model_type") or "").lower() in CLASS_GATED_FFN:
        return True
    # a config that leaves the projection out takes its class's (MT5Config's `gated-gelu`, where
    # T5Config's is `relu`): an mT5 without the key had its encoder counted with two matrices
    v = config_class_default(tc.get("model_type"), "feed_forward_proj")
    if isinstance(v, str) and v:
        return v.lower().startswith("gated-")
    return None


#: Config classes whose attention gates its output from a second query-width projection that
#: no config key of the class states (transformers 5.12, and llama.cpp's architectures for them: Qwen3-Next's and Qwen3.5's
#: `q_proj = Linear(hidden, heads x head_dim x 2)`). A hub config may say so itself
#: (`attn_output_gate`, Qwen3.5), and is read first.
CLASS_GATED_ATTENTION = frozenset({"qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
                                   "qwen3_5_moe_text",
                                   # the same classes as llama.cpp names them in a GGUF
                                   "qwen3next", "qwen35", "qwen35moe"})
ATTENTION_OUTPUT_GATE_KEYS = ("attn_output_gate",)


def attention_output_gate(config: Dict[str, Any]) -> bool:
    """Whether the decoder's attention builds a query-width output gate beside its queries: the
    config's own statement (``ATTENTION_OUTPUT_GATE_KEYS``), else its config class's modelling
    (``CLASS_GATED_ATTENTION``). Left out, Qwen3-Next's 12 attention layers were sized 100M
    parameters below the checkpoint."""
    tc = _text_config(config) if isinstance(config, dict) else {}
    for k in ATTENTION_OUTPUT_GATE_KEYS:
        v = tc.get(k)
        if isinstance(v, bool):
            return v
    return str(tc.get("model_type") or "").lower() in CLASS_GATED_ATTENTION


#: Spellings of a decoder's FFN width and of its token vocabulary (ChatGLM's `ffn_hidden_size` and
#: `padded_vocab_size`, GPT-2's `n_vocab`), first to last.
FFN_WIDTH_KEYS = ("intermediate_size", "ffn_dim", "ffn_hidden_size", "mlp_dim", "d_ff")
VOCAB_KEYS = ("vocab_size", "padded_vocab_size", "n_vocab")


def ffn_width(config: Dict[str, Any], default: Optional[int] = None) -> Optional[int]:
    """The decoder's FFN width under any of its spellings (``FFN_WIDTH_KEYS``), or ``default``."""
    tc = config if isinstance(config, dict) else {}
    return _pos_int(_first(tc, *FFN_WIDTH_KEYS)) or default


def vocab_size(config: Dict[str, Any], default: Optional[int] = None) -> Optional[int]:
    """The token vocabulary under any of its spellings (``VOCAB_KEYS``), or ``default``. Read
    from `vocab_size` alone, ChatGLM3's 65,024 tokens were a default 32,000."""
    tc = config if isinstance(config, dict) else {}
    return _pos_int(_first(tc, *VOCAB_KEYS)) or default


#: Keys that say a stack positions its tokens by rotation (RoPE) rather than a learned table.
ROPE_KEYS = ("rope_theta", "rope_scaling", "rope_parameters", "rotary_pct", "rotary_emb_base",
             "partial_rotary_factor", "rotary_dim")
#: Spellings of a learned position table's length, first to last (Whisper's decoder says
#: `max_target_positions`; GPT-2 `n_positions`).
POSITION_LENGTH_KEYS = ("max_target_positions", "max_position_embeddings", "n_positions")


def learned_positions(config: Dict[str, Any]) -> int:
    """Rows of a learned absolute position table the decoder holds, 0 for none: a config whose
    `position_embedding_type` is absolute (BERT-style decoders), or whose config class positions
    by neither rotation (``ROPE_KEYS``, stated or the class's default) nor relative buckets nor
    ALiBi (MBart / BART, Whisper, GPT-2). A class transformers does not know is read as holding
    none, as before. Left out, Nougat's MBart decoder was 4.2M parameters short."""
    tc = config if isinstance(config, dict) else {}
    pet = str(tc.get("position_embedding_type") or "").lower()
    if pet and pet != "absolute":
        return 0
    if any(tc.get(k) is not None for k in ROPE_KEYS) or tc.get("relative_attention_num_buckets") \
            or tc.get("alibi") or tc.get("use_alibi"):
        return 0
    if pet != "absolute":
        mt = tc.get("model_type")
        if not mt or config_class_default(mt, "model_type") is None:
            return 0  # a class transformers does not know: nothing said, nothing charged
        if any(config_class_default(mt, k) is not None
               for k in ROPE_KEYS + ("relative_attention_num_buckets",)):
            return 0
    return _pos_int(_first(tc, *POSITION_LENGTH_KEYS)) or 0


def decoder_streams(config: Dict[str, Any]) -> int:
    """How many parallel token streams the decoder embeds and predicts at once: an audio decoder's
    codebooks (MusicGen's `num_codebooks`; Dia's decoder block's `num_channels`), each with its
    own embedding table and output head. 1 for a text decoder. Read as one stream, Dia's 9 x 1,028
    embeddings and logits were one table and one head."""
    if not isinstance(config, dict):
        return 1
    for scope in (config, _text_config(config)):
        v = _pos_int(scope.get("num_codebooks")) if isinstance(scope, dict) else None
        if v:
            return v
    for scope in (config, _text_config(config)):
        if not isinstance(scope, dict):
            continue
        for k in NESTED_DECODER_KEYS:
            block = scope.get(k)
            if isinstance(block, dict):
                v = _pos_int(block.get("num_codebooks")) or _pos_int(block.get("num_channels"))
                if v:
                    return v
    return 1


#: Spellings of the FFN's activation (Gemma's `hidden_activation`, GPT-2's `activation_function`).
ACTIVATION_KEYS = ("hidden_act", "activation_function", "hidden_activation")

#: Spellings of "the attention's Q / K / V (and O) projections carry biases": transformers'
#: `attention_bias`, and Qwen2-MoE's `qkv_bias` (its class default True, so a Qwen1.5-MoE config
#: that states neither builds them).
ATTENTION_BIAS_KEYS = ("attention_bias", "qkv_bias")


def attention_bias(config: Dict[str, Any]) -> bool:
    """Whether the attention projections carry biases: the config's own flag under either
    spelling (``ATTENTION_BIAS_KEYS``), else its config class's default."""
    tc = config if isinstance(config, dict) else {}
    for key in ATTENTION_BIAS_KEYS:
        if isinstance(tc.get(key), bool):
            return tc[key]
    mt = tc.get("model_type")
    return any(config_class_default(mt, key) is True for key in ATTENTION_BIAS_KEYS) if mt else False

#: Spellings of "the attention normalises its queries and keys" (a norm at least head_dim wide
#: each; the facts count head_dim): `qk_norm`, Hunyuan's `use_qk_norm`, Persimmon's / StableLM's
#: `qk_layernorm`.
QK_NORM_KEYS = ("qk_norm", "use_qk_norm", "qk_layernorm", "use_qk_layernorm")


def stated_qk_norm(config: Dict[str, Any]) -> bool:
    """Whether the config states a query / key norm under any of ``QK_NORM_KEYS``."""
    tc = config if isinstance(config, dict) else {}
    return any(bool(tc.get(k)) for k in QK_NORM_KEYS)


#: An image tower's keys whose class default stands in for a tower config that leaves them out
#: (Mllama's tower builds 8 global layers and a 7,680-wide output whether or not it says so).
TOWER_CLASS_KEYS = ("num_global_layers", "vision_output_dim", "intermediate_size",
                    "supported_aspect_ratios", "max_num_tiles", "image_size", "patch_size")

#: Every key a reader asks a config class's default for (``config_class_default``): the shipped
#: snapshot (``config_class_defaults.json``) holds each one for every model type transformers 5.12
#: knows.
CLASS_DEFAULT_KEYS = tuple(dict.fromkeys(
    ("model_type", "tie_word_embeddings", "relative_attention_num_buckets", "feed_forward_proj")
    + ATTENTION_BIAS_KEYS
    + ACTIVATION_KEYS
    + ROPE_KEYS + TOWER_CLASS_KEYS))

_CLASS_DEFAULT_CACHE: Dict[str, Any] = {}
_CLASS_DEFAULT_SNAPSHOT: Dict[str, Any] = {}


def _class_default_snapshot() -> Dict[str, Any]:
    """The shipped config-class defaults (``scripts/config_class_defaults.py`` writes them from
    transformers 5.12): ``{"keys": [...], "defaults": {model_type: {key: value}}}``, empty when
    the file is missing."""
    if not _CLASS_DEFAULT_SNAPSHOT:
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config_class_defaults.json")
        try:
            with open(path) as f:
                _CLASS_DEFAULT_SNAPSHOT.update(json.load(f))
        except Exception:
            _CLASS_DEFAULT_SNAPSHOT.update({"keys": [], "defaults": {}})
    return _CLASS_DEFAULT_SNAPSHOT


def config_class_default(model_type: str, name: str) -> Any:
    """The default the model type's own config class gives ``name``, or None. What a config that
    leaves a key out means is its class's default: Llama4TextConfig's `hidden_act` is "silu", so a
    Llama 4 config without one is gated.

    Read from the shipped snapshot (``config_class_defaults.json``: transformers 5.12's classes)
    for every model type and key it holds, so a figure does not depend on whether transformers
    is installed or which release; from the installed transformers only for a model type or key
    the snapshot does not hold (a class newer than 5.12)."""
    mt = str(model_type or "")
    key = mt + "\0" + name
    if key in _CLASS_DEFAULT_CACHE:
        return _CLASS_DEFAULT_CACHE[key]
    value = None
    snap = _class_default_snapshot()
    if mt in snap.get("defaults", {}) and name in snap.get("keys", ()):
        value = snap["defaults"][mt].get(name)
    elif mt:
        try:
            import logging

            from transformers import CONFIG_MAPPING

            tf_log = logging.getLogger("transformers")
            level = tf_log.level
            tf_log.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = getattr(CONFIG_MAPPING[mt](), name, None)
            finally:
                tf_log.setLevel(level)
        except Exception:
            value = None
    _CLASS_DEFAULT_CACHE[key] = value
    return value


#: A stack's geometry, by role, as config classes spell it (transformers 5.12's `to_dict()`
#: names). A config block that states NO spelling of a role leaves that role to its config class
#: -- BLIP-2's and InstructBLIP's hub `vision_config` and `qformer_config` state only their
#: model type, Qwen2-Audio's and LLaVA-1.5's `text_config` no depth, width or heads -- and the
#: class builds its defaults there (Blip2VisionConfig: 39 layers 1,408 wide; Qwen2Config: 32
#: layers 4,096 wide). A key the class sizes from its own width is not filled beside a stated
#: width (``width_derived``). A role is filled whole or not at all: a block that states any spelling of
#: a role keeps its own reading of the rest, because a class derives the rest from what is
#: stated (T5's `num_decoder_layers` from `num_layers`, every class's KV heads from its heads,
#: a head width from the width and heads), which a class default cannot stand in for.
CLASS_GEOMETRY_ROLES = (
    ("num_hidden_layers", "num_layers", "n_layer", "n_layers", "depth", "num_decoder_layers",
     "decoder_layers", "encoder_layers", "num_encoder_layers", "decoder_num_hidden_layers",
     "encoder_num_hidden_layers", "depths"),
    ("hidden_size", "d_model", "n_embd", "embed_dim", "dim", "hidden_sizes", "embed_dims",
     "dim_embed"),
    ("num_attention_heads", "n_head", "n_heads", "num_heads", "num_attention_head",
     "encoder_attention_heads", "decoder_attention_heads", "encoder_num_attention_heads",
     "decoder_num_attention_heads", "num_encoder_attention_heads", "num_decoder_attention_heads",
     "num_key_value_heads", "num_kv_heads", "multi_query_group_num", "decoder_num_key_value_heads",
     "encoder_num_key_value_heads", "head_dim", "d_kv", "kv_channels", "attention_head_dim"),
    ("intermediate_size", "d_ff", "ffn_dim", "encoder_ffn_dim", "decoder_ffn_dim", "mlp_dim",
     "n_inner", "ffn_hidden_size", "mlp_ratio"),
    ("encoder_hidden_size",),
    ("cross_attention_frequency",),
    ("image_size",),
    ("patch_size",),
    ("max_num_tiles",),
    ("num_mel_bins",),
    ("max_source_positions",),
    ("resampler_depth",),
    ("resampler_n_latents",),
    ("resampler_n_heads",),
    ("resampler_head_dim",),
)
CLASS_GEOMETRY_KEYS = tuple(k for role in CLASS_GEOMETRY_ROLES for k in role)

_CLASS_GEOMETRY_CACHE: Dict[str, Dict[str, Any]] = {}


def geometry_value(v: Any) -> bool:
    """Whether ``v`` is a geometry value a class default can stand in with: a positive number,
    or a list of positive numbers (a hierarchical tower's per-stage depths and widths)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v > 0
    return (isinstance(v, (list, tuple)) and bool(v)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in v))


def class_geometry(model_type: Any) -> Dict[str, Any]:
    """The geometry (``CLASS_GEOMETRY_KEYS``) the model type's config class builds when a config
    states none: from the shipped snapshot (``config_class_defaults.json``'s ``geometry``,
    transformers 5.12's classes), else the installed transformers' class for a model type the
    snapshot does not hold, else nothing."""
    mt = str(model_type or "")
    if not mt:
        return {}
    if mt in _CLASS_GEOMETRY_CACHE:
        return _CLASS_GEOMETRY_CACHE[mt]
    snap = _class_default_snapshot().get("geometry", {})
    if mt in snap:
        value = dict(snap[mt])
    else:
        value = {}
        try:
            import logging

            from transformers import CONFIG_MAPPING

            tf_log = logging.getLogger("transformers")
            level = tf_log.level
            tf_log.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stated = CONFIG_MAPPING[mt]().to_dict()
            finally:
                tf_log.setLevel(level)
            value = {k: stated[k] for k in CLASS_GEOMETRY_KEYS
                     if k in stated and geometry_value(stated[k])}
        except Exception:
            value = {}
    _CLASS_GEOMETRY_CACHE[mt] = value
    return value


def with_class_geometry(config: Any) -> Any:
    """``config`` with every role of its geometry it leaves to its config class
    (``CLASS_GEOMETRY_ROLES``) filled with the class's default, in the block itself and in every
    nested block that names its model type (a `text_config`, `vision_config`, `qformer_config`,
    `audio_config`, an encoder-decoder's `encoder` / `decoder`). ``config`` itself when nothing
    is left out; never mutated.

    Read as nothing, such a block was a stack of 0 layers: BLIP-2's 985.9M-parameter tower and
    105.1M Q-Former were left out of the weights, and Qwen2-Audio's language model had no layer
    holding KV."""
    return _with_class_geometry(config, None)


def nested_config_class(model_type: Any, key: str) -> Optional[str]:
    """The model type of the class the ``model_type``'s config class builds its nested ``key``
    block as: the class it fixes (Blip2Config's `vision_config`: blip_2_vision_model; Idefics2's
    `perceiver_config`: idefics2_perceiver, though the hub file names it `idefics2`), else, for a
    block read through AutoConfig, the class it builds when the block names none (Florence2's
    `text_config`: bart); None when the composite is unknown."""
    subs = _class_default_snapshot().get("sub_configs", {}).get(str(model_type or ""), {})
    v = subs.get(key)
    return v if isinstance(v, str) and v else None


#: Composite keys whose block AutoConfig reads by the block's own model type (a `text_config`,
#: LLaVA's `vision_config`): the composite's class is only the default for a block that names none.
def _auto_read(model_type: Any, key: str) -> bool:
    auto = _class_default_snapshot().get("auto_sub_configs", {}).get(str(model_type or ""), ())
    return key in auto


def _with_class_geometry(config: Any, class_type: Optional[str], auto: bool = False) -> Any:
    if not isinstance(config, dict):
        return config
    stated_type = config.get("model_type")
    # a fixed sub-config class is what transformers builds whatever the block names; a block read
    # through AutoConfig is its own model type's class, the composite's default when it names none
    own = (stated_type or class_type) if auto else (class_type or stated_type)
    add: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict):
            done = _with_class_geometry(v, nested_config_class(own, k), _auto_read(own, k))
            if done is not v:
                add[k] = done
    if own and not stated_type and class_type:
        # the class the composite builds for a block that names none is the block's model type
        add["model_type"] = own
    geometry = class_geometry(own)
    if geometry:
        # beside a width the block states, what the class sizes from its own width is another
        # model's (FalconConfig's `ffn_hidden_size` is 4 x its 4,544; a Llama `head_dim` is its
        # width over its heads): those keys (the snapshot's `width_derived`) are not filled
        derived = set()
        if any(config.get(k) is not None for k in CLASS_GEOMETRY_ROLES[1]):
            derived = set(_class_default_snapshot().get("width_derived", {}).get(str(own), ()))
        for role in CLASS_GEOMETRY_ROLES:
            if any(config.get(k) is not None for k in role):
                continue
            for k in role:
                if k in geometry and k not in derived:
                    add[k] = geometry[k]
    # ChatGLM states its KV heads as `multi_query_group_num` beside `multi_query_attention`; a
    # stored `num_key_value_heads: true` (prepopulated.db's glm-4-9b-chat) is a flag, not a count,
    # and read as one head it halved every reader's KV
    kv_heads, groups = config.get("num_key_value_heads"), _pos_int(config.get("multi_query_group_num"))
    if (kv_heads is None or isinstance(kv_heads, bool)) and config.get("multi_query_attention") \
            and groups:
        add["num_key_value_heads"] = groups
    # a table padded past the vocabulary (ChatGLM's `padded_vocab_size`, beside which that db row
    # also stores a default `vocab_size: 32000`) holds the padded rows
    padded = _pos_int(config.get("padded_vocab_size"))
    vocab = config.get("vocab_size")
    if padded and (not _pos_int(vocab) or padded > int(vocab)):
        add["vocab_size"] = padded
    return {**config, **add} if add else config


def ffn_activation(config: Dict[str, Any]) -> str:
    """The FFN's activation, lower-cased: the config's own (``ACTIVATION_KEYS``), else its config
    class's default (``config_class_default``), else ''. The config given is read as it is -- a
    caller sizing a decoder passes the decoder's (text) config."""
    tc = config if isinstance(config, dict) else {}
    for k in ACTIVATION_KEYS:
        v = tc.get(k)
        if isinstance(v, str) and v:
            return v.lower()
    for k in ACTIVATION_KEYS:
        v = config_class_default(tc.get("model_type"), k)
        if isinstance(v, str) and v:
            return v.lower()
    return ""


def dense_ffn_width(config: Dict[str, Any]) -> Optional[int]:
    """The width a dense layer of an MoE model is built at, when the config names it apart from
    `intermediate_size`; None when it does not (the dense layer is `intermediate_size` wide)."""
    return _pos_int(_first(_text_config(config), *DENSE_FFN_WIDTH_KEYS))


def layer_kinds(config: Dict[str, Any]) -> Dict[str, Any]:
    """Every layer's kind, for ANY decoder config -- the one answer callers use.

    The hybrid plan when there is one; otherwise the config's own per-layer
    names (explicit or synthesized from a period); otherwise a uniform stack,
    windowed if the config states a window. Same axes as a plan: ``attn``,
    ``recurrent``, ``ffn``, ``kv_shared``, plus ``sliding_window`` and
    ``encoder_tokens``.
    """
    config = with_class_geometry(config)
    plan = resolve_layer_plan(config)
    tc = _text_config(config)
    if plan is not None:
        out = {k: plan[k] for k in ("attn", "recurrent", "ffn", "kv_shared")}
        out.update(
            moe=moe_layers(config),
            sliding_window=plan.get("sliding_window"),
            encoder_tokens=plan.get("encoder_tokens"),
            source="layer_plan:" + plan["dialect"],
            notes=list(plan.get("notes") or []),
            mechanisms=plan.get("mechanisms") or [],
        )
        return out
    n = depth(tc)
    blank = _blank(n)
    names, stated_by, notes = stated_layer_types(tc)
    ffn = [True] * n
    if names:
        _apply_layer_names(blank, names, tc, n, notes)
        blank.pop("_unknown", None)
        chunked = blank.pop("_chunked", False)
        ffn = blank.pop("_ffn", None) or ffn
        source = "layer_types" if stated_by in ("layer_types", "sliding_window_pattern") else stated_by
    else:
        chunked = False
        blank["attn"] = [ATTN_SLIDING if _window(tc) else ATTN_FULL] * n
        source = "uniform"
    window, _ = _sliding_bound(blank["attn"], tc, chunked, notes)
    return {
        "attn": blank["attn"],
        "recurrent": blank["recurrent"],
        "ffn": ffn,
        "moe": moe_layers(config),
        "kv_shared": [False] * n,
        "sliding_window": window,
        "encoder_tokens": None,
        "source": source,
        "notes": notes,
        "mechanisms": [],
    }


def _sliding_bound(attn: List[Optional[str]], tc: Dict[str, Any], chunked: bool,
                   notes: List[str]) -> Tuple[Optional[int], bool]:
    """The tokens a sliding layer keeps: the window, or Llama 4's chunk for chunked layers.

    A layout that names sliding layers while the window sits under a key this table does not
    read (ModernBERT's `local_attention`) or nowhere leaves no bound: those layers are charged
    FULL attention here, in place, so no reader can charge a windowed layer with no window less
    than full (the vLLM planner charged it nothing). Returns (window, whether it charged any).
    """
    window = _window(tc)
    if chunked and not window:
        chunk = tc.get("attention_chunk_size")
        window = int(chunk) if isinstance(chunk, int) and chunk > 0 else None
    unbounded = [i for i, a in enumerate(attn) if a == ATTN_SLIDING]
    if window or not unbounded:
        return window, False
    for i in unbounded:
        attn[i] = ATTN_FULL
    msg = (
        f"{len(unbounded)} layers are named sliding/local, but the config states their window "
        f"under none of {list(WINDOW_KEYS)} (nor `attention_chunk_size`), so they are charged "
        "FULL-ATTENTION KV -- an over-estimate, not a measurement. State the window under one "
        "of those keys to apply it."
    )
    notes.append(msg)
    warnings.warn(msg, stacklevel=4)
    return None, True


def _finish(
    plan: Dict[str, Any],
    config: Dict[str, Any],
    dialect: str,
    notes: List[str],
    parallel: bool = False,
    ffn: Optional[List[bool]] = None,
    attention_param_layers: Optional[int] = None,
    ffn_param_layers: Optional[int] = None,
    approximate: bool = False,
    keep_uniform: bool = False,
) -> Optional[Dict[str, Any]]:
    """Attach summary counts; drop plans that found no hybrid structure."""
    tc = _text_config(config)
    attn, rec = plan["attn"], plan["recurrent"]
    n_layers = len(attn)
    if plan.pop("_unknown", False):
        approximate = True
    chunked = plan.pop("_chunked", False)
    notes = list(notes)
    window, unbounded = _sliding_bound(attn, tc, chunked, notes)
    approximate = approximate or unbounded

    # KV SHARING. The last `num_kv_shared_layers` layers (Gemma 3n) read the
    # cache an earlier layer of the same kind wrote and allocate none of their
    # own. They keep their attention (weights, compute); only the KV goes.
    shared = int(tc.get("num_kv_shared_layers") or 0)
    shared = max(0, min(shared, n_layers))
    kv_shared = [False] * n_layers
    for i in range(n_layers - shared, n_layers):
        if attn[i] is not None:
            kv_shared[i] = True
    plan["kv_shared"] = kv_shared

    # Almost every family pairs one FFN with each mixer layer; Nemotron-H is the
    # exception and passes an explicit list.
    plan["ffn"] = list(ffn) if ffn is not None else [True] * len(attn)
    owns_kv = [a is not None and not kv_shared[i] for i, a in enumerate(attn)]
    n_full = sum(1 for i, a in enumerate(attn) if a == ATTN_FULL and owns_kv[i])
    n_slide = sum(1 for i, a in enumerate(attn) if a == ATTN_SLIDING and owns_kv[i])
    n_cross = sum(1 for i, a in enumerate(attn) if a == ATTN_CROSS and owns_kv[i])
    n_attn = sum(1 for a in attn if a is not None)
    n_rec = sum(1 for r in rec if r)

    # A plain transformer resolved through `layer_types` (all full, no
    # recurrence) carries no information the uniform path lacks. Returning None
    # keeps those models on their existing, already-correct code path.
    if n_rec == 0 and n_slide == 0 and not shared and not keep_uniform and not approximate:
        return None

    # A mechanism whose defining dimensions are absent from the config gets
    # sized from this module's fallbacks -- which are one specific model's
    # geometry. Producing a plausible number from another model's shape is the
    # same failure the hardcoded `state_size = 16` default used to cause, so say
    # so rather than let it pass as a measurement.
    for mech, keys in _MECHANISM_REQUIRED_KEYS.items():
        if mech in {r for r in rec if r} and not any(k in tc for k in keys):
            approximate = True
            notes = notes + [
                f"{mech}: none of {sorted(keys)} present in the config, so its "
                f"state and parameter counts fall back to default geometry and "
                f"are an ESTIMATE, not derived from this model."
            ]
    if shared:
        notes = notes + [
            f"{sum(kv_shared)} layers reuse an earlier layer's KV "
            f"(num_kv_shared_layers={shared}) and are charged none of their own."
        ]

    enc_tokens = encoder_tokens(config) if n_cross else None
    if n_cross and enc_tokens is None:
        notes = notes + [
            f"{n_cross} cross-attention layers are charged the full sequence: the "
            "config does not state the encoder's token count."
        ]
    elif n_cross:
        # The tower's output is per image (transformers reshapes Mllama's cross-attention states
        # to `num_concurrent_media` x tiles x patches): the image count is the request's, as its
        # length is, and no config states it. Not a bound for a request with several images.
        notes = notes + [
            f"{n_cross} cross-attention layers are charged one image's {enc_tokens} tower "
            "tokens a request: the config states no image count, and each further image a "
            "request carries adds as much again."
        ]

    plan.update(
        {
            "dialect": dialect,
            "parallel": parallel,
            "approximate": approximate,
            # Layers that run attention at all (weights, compute).
            "num_attention_layers": n_attn,
            # Layers that hold a KV cache of their own, by how it grows.
            "num_full_layers": n_full,
            "num_sliding_layers": n_slide,
            "num_cross_layers": n_cross,
            "num_kv_shared_layers": sum(kv_shared),
            "num_recurrent_layers": n_rec,
            "num_ffn_layers": sum(1 for f in plan["ffn"] if f),
            # What a sliding layer keeps (the window, or Llama 4's chunk), and
            # what a cross-attention layer keeps (the encoder's tokens).
            "sliding_window": window,
            "encoder_tokens": enc_tokens,
            # Weight-counting layer counts. These differ from the runtime counts
            # only when blocks are SHARED across layers (Zamba2): the cache is
            # allocated per invocation, the weights exist once.
            "num_attention_param_layers": (
                n_attn if attention_param_layers is None else attention_param_layers
            ),
            "num_ffn_param_layers": (
                sum(1 for f in plan["ffn"] if f) if ffn_param_layers is None else ffn_param_layers
            ),
            "mechanisms": sorted({r for r in rec if r}),
            "notes": notes,
        }
    )
    return plan
