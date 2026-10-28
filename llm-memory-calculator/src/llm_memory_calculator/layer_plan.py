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

LAYER_KINDS: Dict[str, Dict[str, Optional[str]]] = {
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
    # an FFN-only slot (Nemotron-H's MoE / MLP layers): no mixer, no cache
    "moe": {"attn": None, "recurrent": None},
    "dense": {"attn": None, "recurrent": None},
    "mlp": {"attn": None, "recurrent": None},
    "mlp_only": {"attn": None, "recurrent": None},
    "ffn": {"attn": None, "recurrent": None},
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
    return tc if isinstance(tc, dict) else config


def _blank(n: int) -> Dict[str, Any]:
    return {"attn": [None] * n, "recurrent": [None] * n}


#: Per-layer lists whose length IS the depth. transformers 5.12's NemotronHConfig deprecates
#: `num_hidden_layers` in favour of `layers_block_type` and drops it from the saved config,
#: so a reader that requires the scalar loses the whole layout.
_PER_LAYER_LIST_KEYS = (
    "layer_types", "layers_block_type", "hybrid_override_pattern", "attn_type_list",
    "sliding_window_layout", "mlp_layer_types", "attention_layers",
)


def depth(tc: Dict[str, Any]) -> int:
    """Decoder depth: the stated layer count, else the length of a per-layer list."""
    n = _first(tc, "num_hidden_layers", "n_layers", "num_layers")
    if n:
        return int(n)
    for key in _PER_LAYER_LIST_KEYS:
        v = tc.get(key)
        if isinstance(v, (list, tuple, str)) and v:
            return len(v)
    return 0


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
    """Tokens a cross-attention layer caches per request: the encoder's output.

    Derived from the vision tower's own geometry: patches per image side
    squared, plus the class token, per tile. None when the config does not
    state it, and the caller then charges the full sequence instead.
    """
    vc = config.get("vision_config")
    if not isinstance(vc, dict):
        return None
    size, patch = vc.get("image_size"), vc.get("patch_size")
    if not (isinstance(size, int) and isinstance(patch, int) and patch > 0):
        return None
    per_tile = (size // patch) ** 2 + 1
    tiles = vc.get("max_num_tiles") or 1
    return int(per_tile * int(tiles))


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

    A stack that names FFN-only slots (Nemotron-H's ``moe`` / ``mlp``) gives the FFN its own
    layer: then only those slots carry one, and the mixers carry none. Every other stack pairs
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
        ffn_only.append(not k.get("unknown") and k["attn"] is None and k["recurrent"] is None)
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


def _config_class_layer_types(tc: Dict[str, Any]):
    """``(known, names)``: what the model type's own config class says the layers are.

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
        reduced = {k: whole[k] for k in _LAYOUT_KEYS if k in whole}
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
