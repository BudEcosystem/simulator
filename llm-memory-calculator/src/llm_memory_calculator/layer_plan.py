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

from typing import Any, Dict, List, Optional

# Recurrent mechanisms, by state geometry rather than by vendor name -- the
# state shape is what the memory model needs, and several vendors share one.
MECH_MAMBA1 = "mamba1"  # d_inner x d_state              (Mamba-1, Jamba, falcon_mamba)
MECH_MAMBA2 = "mamba2"  # n_heads x d_head x d_state      (Mamba-2, Nemotron-H, Bamba, granite-4)
MECH_GDN = "gdn"  # n_v_heads x k_head_dim x v_head_dim  (Gated DeltaNet: Qwen3-Next/3.5/3.6)
MECH_KDA = "kda"  # n_heads x head_dim x head_dim        (Kimi Delta Attention)
MECH_LIGHTNING = "lightning"  # n_heads x head_dim x head_dim (MiniMax lightning attn)
MECH_SHORTCONV = "shortconv"  # conv cache only, no recurrent matrix (LFM2)

ATTN_FULL = "full"
ATTN_SLIDING = "sliding"

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


def _classify_layer_type_string(s: str) -> Dict[str, Optional[str]]:
    """Map one ``layer_types`` entry to the two axes.

    Order matters. ``"linear_attention"`` contains the substring ``"attention"``,
    so a naive ``'attention' in s`` test -- which is what the normalizer did --
    classifies Gated DeltaNet layers as full attention and charges them KV they
    never allocate. Linear and conv must be tested first.
    """
    s = str(s).lower()
    if "linear" in s or "mamba" in s or "ssm" in s or "recurrent" in s or "gdn" in s:
        mech = MECH_MAMBA2 if ("mamba" in s or "ssm" in s) else MECH_GDN
        return {"attn": None, "recurrent": mech}
    if "conv" in s:
        return {"attn": None, "recurrent": MECH_SHORTCONV}
    if "sliding" in s or s == "swa":
        return {"attn": ATTN_SLIDING, "recurrent": None}
    if "full" in s or "global" in s or "attention" in s or s == "attn":
        return {"attn": ATTN_FULL, "recurrent": None}
    # MLP-only / unrecognized: no cache of either kind.
    return {"attn": None, "recurrent": None}


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
    arch = " ".join(config.get("architectures") or []).lower()
    plan = _blank(n)
    notes: List[str] = []

    # --- 1. Explicit per-layer list (HF standard: Qwen3.5/3.6, granite-4, LFM2,
    # gpt-oss, Gemma-3 via the normalizer's synthesized list) -------------------
    layer_types = tc.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        for i, lt in enumerate(layer_types[:n]):
            k = _classify_layer_type_string(lt)
            plan["attn"][i] = k["attn"]
            plan["recurrent"][i] = k["recurrent"]
        # Qwen3.5/3.6 spell GDN as "linear_attention"; pin the mechanism by the
        # config keys actually present rather than by the string alone.
        if "linear_num_value_heads" in tc:
            plan["recurrent"] = [
                MECH_GDN if r else None for r in plan["recurrent"]
            ]
        return _finish(plan, config, "layer_types", notes)

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
    if model_type.startswith("falcon_h") or (
        "mamba_d_state" in tc and "num_attention_heads" in tc and not layer_types
    ):
        for i in range(n):
            plan["attn"][i] = ATTN_FULL
            plan["recurrent"][i] = MECH_MAMBA2
        return _finish(plan, config, "parallel_hybrid", notes, parallel=True)

    # --- 11. Pure SSM: no attention anywhere --------------------------------
    if model_type in ("mamba", "mamba2", "falcon_mamba", "s4", "ssm", "state-space"):
        mech = MECH_MAMBA2 if (model_type == "mamba2" or "n_groups" in tc) else MECH_MAMBA1
        for i in range(n):
            plan["recurrent"][i] = mech
        # A pure SSM stack has no FFN at all -- the gated mixer *is* the block.
        # Charging one per layer nearly doubled Mamba-Codestral (13.7B vs 7.3B).
        return _finish(plan, config, "pure_ssm", notes, ffn=[False] * n)

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
    if model_type == "phi4flash" or "phi4flash" in arch:
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
    attn, rec = plan["attn"], plan["recurrent"]
    # Almost every family pairs one FFN with each mixer layer; Nemotron-H is the
    # exception and passes an explicit list.
    plan["ffn"] = list(ffn) if ffn is not None else [True] * len(attn)
    n_full = sum(1 for a in attn if a == ATTN_FULL)
    n_slide = sum(1 for a in attn if a == ATTN_SLIDING)
    n_rec = sum(1 for r in rec if r)

    # A plain transformer resolved through `layer_types` (all full, no
    # recurrence) carries no information the uniform path lacks. Returning None
    # keeps those models on their existing, already-correct code path.
    if n_rec == 0 and n_slide == 0 and not keep_uniform:
        return None

    # A mechanism whose defining dimensions are absent from the config gets
    # sized from this module's fallbacks -- which are one specific model's
    # geometry. Producing a plausible number from another model's shape is the
    # same failure the hardcoded `state_size = 16` default used to cause, so say
    # so rather than let it pass as a measurement.
    for mech, keys in _MECHANISM_REQUIRED_KEYS.items():
        if mech in {r for r in rec if r} and not any(k in _text_config(config) for k in keys):
            approximate = True
            notes = notes + [
                f"{mech}: none of {sorted(keys)} present in the config, so its "
                f"state and parameter counts fall back to default geometry and "
                f"are an ESTIMATE, not derived from this model."
            ]

    plan.update(
        {
            "dialect": dialect,
            "parallel": parallel,
            "approximate": approximate,
            "num_attention_layers": n_full + n_slide,
            "num_full_layers": n_full,
            "num_sliding_layers": n_slide,
            "num_recurrent_layers": n_rec,
            "num_ffn_layers": sum(1 for f in plan["ffn"] if f),
            # Weight-counting layer counts. These differ from the runtime counts
            # only when blocks are SHARED across layers (Zamba2): the cache is
            # allocated per invocation, the weights exist once.
            "num_attention_param_layers": (
                n_full + n_slide if attention_param_layers is None else attention_param_layers
            ),
            "num_ffn_param_layers": (
                sum(1 for f in plan["ffn"] if f) if ffn_param_layers is None else ffn_param_layers
            ),
            "mechanisms": sorted({r for r in rec if r}),
            "notes": notes,
        }
    )
    return plan
