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

import warnings
from typing import Any, Dict, List, Optional

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


def _apply_layer_names(plan, names, tc, n, notes):
    """Fill both axes from a list of per-layer kind names; flag unknown names."""
    unknown = []
    for i, lt in enumerate(list(names)[:n]):
        k = _classify_layer_type_string(lt)
        if k.get("unknown"):
            unknown.append(k["unknown"])
        plan["attn"][i] = ATTN_SLIDING if k["attn"] == ATTN_CHUNKED else k["attn"]
        if k["attn"] == ATTN_CHUNKED:
            plan.setdefault("_chunked", True)
        plan["recurrent"][i] = _recurrent_mechanism(k["recurrent"], tc, lt)
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
    n = _first(tc, "num_hidden_layers", "n_layers", "num_layers")
    if not n:
        return None

    model_type = str(_first(tc, "model_type", default="") or "").lower()
    plan = _blank(n)
    notes: List[str] = []

    # --- 1. Explicit per-layer list (HF standard: Qwen3.5/3.6, granite-4, LFM2,
    # gpt-oss, Gemma-3n, MiniMax, Llama 4). Names resolve through LAYER_KINDS;
    # a recurrent name's mechanism comes from the config's own keys. -----------
    layer_types = tc.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        _apply_layer_names(plan, layer_types, tc, n, notes)
        return _finish(plan, config, "layer_types", notes)

    # --- 1a. A block-type pattern that repeats over the depth (RecurrentGemma's
    # `block_types`, e.g. ["recurrent", "recurrent", "attention"]). Its one
    # attention kind is bounded by the window the config states, as a uniform
    # stack's would be. -------------------------------------------------------
    block_types = tc.get("block_types")
    if isinstance(block_types, (list, tuple)) and block_types and isinstance(block_types[0], str):
        cycle = [block_types[i % len(block_types)] for i in range(n)]
        _apply_layer_names(plan, cycle, tc, n, notes)
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
        return _finish(plan, config, "sliding_window_layout", notes)

    # --- 2. Kimi Linear: explicit index lists, ONE-INDEXED -------------------
    lac = tc.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("full_attn_layers"):
        full = set(lac["full_attn_layers"])
        kda = set(lac.get("kda_layers") or [])
        # The two lists together cover 1..n inclusive, not 0..n-1. Reading them
        # as 0-indexed silently drops the last layer and shifts every other one.
        one_indexed = (full | kda) and min(full | kda) == 1 and max(full | kda) == n
        off = 1 if one_indexed else 0
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
        return _finish(plan, config, "attn_type_list", notes)

    # --- 4. Nemotron-H: a pattern string, M=mamba * =attention - =MLP-only ---
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat:
        # Unlike every other family here, Nemotron-H gives the FFN its OWN layer
        # slot rather than pairing one with each mixer. Charging an FFN on all 56
        # layers instead of the 25 that have one is a ~2.2x over-count on the
        # single largest parameter block in the model.
        ffn = [False] * n
        for i, ch in enumerate(pat[:n]):
            if ch == "*":
                plan["attn"][i] = ATTN_FULL
            elif ch == "M":
                plan["recurrent"][i] = MECH_MAMBA2
            elif ch == "-":
                ffn[i] = True
        return _finish(plan, config, "hybrid_override_pattern", notes, ffn=ffn)

    # --- 5. Zamba2: layers_block_type, 'hybrid' carries the shared attention -
    lbt = tc.get("layers_block_type")
    if isinstance(lbt, list) and lbt:
        for i, t in enumerate(lbt[:n]):
            t = str(t).lower()
            plan["recurrent"][i] = MECH_MAMBA2
            if "hybrid" in t or "attention" in t:
                plan["attn"][i] = ATTN_FULL
        # Zamba2's attention+MLP block is SHARED: `num_mem_blocks` distinct
        # blocks are re-invoked across every hybrid layer. Each invocation
        # allocates its own KV, so the runtime counts above stand -- but the
        # weights exist only once per block, so parameter counting must use
        # num_mem_blocks. Counting them per-layer doubles the model.
        blocks = int(tc.get("num_mem_blocks") or 0)
        return _finish(
            plan, config, "layers_block_type", notes,
            ffn=[False] * n,  # mamba layers carry no FFN; the shared block owns it
            attention_param_layers=blocks or None,
            ffn_param_layers=blocks or None,
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

    # --- 10. Falcon-H1: parallel hybrid with no layer pattern at all. Mamba
    # keys present + no interleave dialect means every layer is both. ---------
    if "mamba_d_state" in tc and "num_attention_heads" in tc and not layer_types:
        for i in range(n):
            plan["attn"][i] = ATTN_FULL
            plan["recurrent"][i] = MECH_MAMBA2
        return _finish(plan, config, "parallel_hybrid", notes, parallel=True)

    # --- 11. Pure SSM: no attention anywhere. Structural first -- state keys
    # and no attention-head count of any spelling -- so a new SSM family is
    # recognised without being listed; the model_type list stays as a fallback
    # for configs that name the family but omit the state keys. ---------------
    no_attention_heads = not any(
        tc.get(k) for k in ("num_attention_heads", "n_head", "n_heads", "num_attention_head")
    )
    has_ssm_state = any(
        tc.get(k) is not None for k in ("state_size", "d_state", "ssm_state_size", "mamba_d_state")
    )
    if (has_ssm_state and no_attention_heads) or model_type in (
        "mamba", "mamba2", "falcon_mamba", "s4", "ssm", "state-space"
    ):
        mech = MECH_MAMBA2 if (model_type == "mamba2" or "n_groups" in tc) else MECH_MAMBA1
        for i in range(n):
            plan["recurrent"][i] = mech
        # A pure SSM stack has no FFN at all -- the gated mixer *is* the block.
        # Charging one per layer nearly doubled Mamba-Codestral (13.7B vs 7.3B).
        return _finish(plan, config, "pure_ssm", notes, ffn=[False] * n)

    # --- 12. Attention-free recurrent stacks other than Mamba, by the keys
    # that size their state: RWKV (`attention_hidden_size` with no attention
    # heads) and xLSTM's matrix memory (`qk_dim_factor` / `v_dim_factor`). ----
    if no_attention_heads and ("qk_dim_factor" in tc or "v_dim_factor" in tc):
        plan["recurrent"] = [MECH_MLSTM] * n
        return _finish(plan, config, "mlstm_stack", notes)
    if no_attention_heads and "attention_hidden_size" in tc:
        plan["recurrent"] = [MECH_RWKV] * n
        return _finish(plan, config, "rwkv_stack", notes)

    # --- 13. Cross-attention layers interleaved with self-attention (Mllama's
    # `cross_attention_layers`): their cache holds the encoder's tokens. -------
    cal = tc.get("cross_attention_layers")
    if isinstance(cal, list) and cal:
        s = set(cal)
        for i in range(n):
            plan["attn"][i] = ATTN_CROSS if i in s else ATTN_FULL
        return _finish(plan, config, "cross_attention_layers", notes, keep_uniform=True)

    # --- Known-but-unmodelled: SambaY / Phi-4-flash --------------------------
    # SambaY interleaves Mamba with sliding-window attention and then adds YOCO
    # cross-attention layers that *share a single* global KV cache. Neither the
    # sharing nor the Mamba/attention split is expressible here, and the config
    # carries no SSM dimensions at all (no d_state, no conv_kernel -- they live
    # in the modeling code), so the recurrent state cannot be sized from it.
    #
    # Every layer is therefore marked sliding, which preserves the window clamp
    # this model genuinely uses. Marking them `full` instead would silently
    # discard the 512-token window and inflate KV by 195x at 100k context -- a
    # far worse answer than the one being approximated.
    # Recognised by its own config key: `mb_per_layer` (Mamba blocks per layer)
    # is the SambaY decoder's, and nothing else ships it.
    if "mb_per_layer" in tc:
        plan["attn"] = [ATTN_SLIDING] * n
        return _finish(
            plan, config, "phi4flash",
            notes + [
                "SambaY (Phi-4-flash): YOCO cross-attention layers share one "
                "global KV cache and half the stack is Mamba, so charging all "
                "layers a windowed KV is an OVER-estimate. Recurrent state is "
                "reported as 0 because the config ships no SSM dimensions."
            ],
            approximate=True,
        )

    return None


def synthesized_layer_types(tc: Dict[str, Any]) -> Optional[List[str]]:
    """Per-layer names for an interleave stated as a period, not a list.

    Gemma 2/3 configs declare `sliding_window_pattern: N` -- one full-attention
    layer every N, the last of each group -- instead of shipping `layer_types`.
    Without this every layer reads as full attention and the global layers'
    window is lost. None when the config states no such period.
    """
    n = _first(tc, "num_hidden_layers", "n_layers", "num_layers")
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
            sliding_window=plan.get("sliding_window"),
            encoder_tokens=plan.get("encoder_tokens"),
            source="layer_plan:" + plan["dialect"],
            notes=list(plan.get("notes") or []),
            mechanisms=plan.get("mechanisms") or [],
        )
        return out
    n = int(_first(tc, "num_hidden_layers", "n_layers", "num_layers", default=0) or 0)
    blank = _blank(n)
    notes: List[str] = []
    names = tc.get("layer_types")
    if not (isinstance(names, list) and names):
        names = synthesized_layer_types(tc)
    if names:
        _apply_layer_names(blank, names, tc, n, notes)
        blank.pop("_unknown", None)
        chunked = blank.pop("_chunked", False)
        source = "layer_types"
    else:
        chunked = False
        blank["attn"] = [ATTN_SLIDING if _window(tc) else ATTN_FULL] * n
        source = "uniform"
    window = _window(tc)
    if chunked and not window:
        chunk = tc.get("attention_chunk_size")
        window = int(chunk) if isinstance(chunk, int) and chunk > 0 else None
    return {
        "attn": blank["attn"],
        "recurrent": blank["recurrent"],
        "ffn": [True] * n,
        "kv_shared": [False] * n,
        "sliding_window": window,
        "encoder_tokens": None,
        "source": source,
        "notes": notes,
        "mechanisms": [],
    }


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

    window = _window(tc)
    if chunked and not window:
        chunk = tc.get("attention_chunk_size")
        window = int(chunk) if isinstance(chunk, int) and chunk > 0 else None
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
