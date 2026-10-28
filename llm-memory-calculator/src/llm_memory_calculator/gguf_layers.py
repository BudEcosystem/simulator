"""The layer kinds a GGUF states, as the HF config keys the layer plan reads.

A GGUF does not carry `layer_types`. llama.cpp states the same facts in its own keys, and a
reader that drops them hands the calculator a uniform dense stack: every layer of a Qwen3.5
GGUF charged a full KV cache, four times what llama.cpp allocates (its "full_attention_interval"
of 4 was simply not read). This module translates them, by llama.cpp's own rules
(``llama-hparams``/``llama-model.cpp``), into the HF spelling, so a GGUF and its source
checkpoint resolve to the same plan through the one table in :mod:`layer_plan`:

* ``{arch}.full_attention_interval`` -- every N-th layer attends, the rest are recurrent
  (``(il + 1) % N != 0`` is recurrent, as llama.cpp marks it).
* ``{arch}.attention.head_count_kv`` as a per-layer ARRAY -- a layer with 0 KV heads holds no
  KV cache (Jamba, granite-hybrid, LFM2, Nemotron-H, Deci). The stack gives FFNs their own
  slots when its KV layers run no FFN (``feed_forward_length`` 0 on every layer with KV heads)
  and some 0-KV layer runs one (Nemotron-H): then a 0-KV layer WITH an FFN is an FFN-only slot
  (``mlp``). Any other 0-KV layer is recurrent when the file states a recurrent mixer, else a
  layer whose mixer holds nothing (``no_op``: Deci's attention no-op or linear replacement),
  which in a stack without FFN slots runs its FFN like every other layer.
* recurrent keys with a scalar KV-head count and no interval -- attention and the mixer in
  every layer (Falcon-H1's parallel hybrid).
* ``{arch}.attention.sliding_window`` + ``sliding_window_pattern`` (a per-layer bool array, or
  "one full layer every N") -- windowed layers. A window with NO pattern is not applied: which
  layers slide is then decided in llama.cpp's code, not the file, so every layer is charged in
  full (the over-estimate) rather than every layer clamped to the window (an under-count).
* ``{arch}.attention.shared_kv_layers`` -- trailing layers that reuse an earlier layer's KV.
* the recurrent state's size: ``ssm.{conv_kernel,inner_size,state_size,group_count,
  time_step_rank}`` (llama.cpp sizes every SSM-style mixer, Mamba and Gated DeltaNet alike,
  as ``(d_conv-1)(d_inner + 2 n_group d_state)`` conv plus ``d_state d_inner`` state, which is
  the Mamba-2 formula with these dimensions), ``wkv.head_size`` (RWKV), ``shortconv.l_cache``.

Key names are gguf-py's (``gguf.constants.Keys``).
"""

from typing import Any, Callable, Dict, List, Optional


def _as_list(v: Any) -> Optional[List[int]]:
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return None


def _positive(v: Any) -> Optional[int]:
    if isinstance(v, (list, tuple)):
        vals = [int(x) for x in v if x]
        return max(vals) if vals else 0
    return int(v) if v is not None else None


#: llama.cpp architectures whose `ssm.*` keys describe a Gated DeltaNet (llama-model.cpp's
#: qwen3next / qwen35 / qwen35moe builders): `ssm.group_count` key heads of `ssm.state_size`,
#: `ssm.time_step_rank` value heads sharing `ssm.inner_size`, the `ssm.conv_kernel` short conv.
#: Its state is the Mamba-2 formula's with these dimensions, but its weights are a GDN's (a
#: fused q|k|v, an output gate, per-head decay and beta), so the mixer is named for what it is.
GGUF_GATED_DELTANET_ARCHS = frozenset({"qwen3next", "qwen35", "qwen35moe"})


def hf_layer_keys_from_gguf(get: Callable[[str], Any], n_layers: int,
                            arch: Optional[str] = None) -> Dict[str, Any]:
    """HF-style layer-kind and recurrent-state keys from a GGUF's metadata.

    ``get(suffix)`` returns the value of ``{arch}.{suffix}`` or None; ``arch`` is the file's
    `general.architecture`. Returns only the keys the file justifies; an ordinary dense
    transformer gets ``{}``.
    """
    out: Dict[str, Any] = {}
    kv = get("attention.head_count_kv")
    ff = get("feed_forward_length")
    kv_list, ff_list = _as_list(kv), _as_list(ff)

    # The recurrent mixer, if the file states one, and the keys that size its state.
    recurrent_name = None
    l_cache = get("shortconv.l_cache")
    wkv = get("wkv.head_size")
    ssm_state = get("ssm.state_size")
    if l_cache:
        recurrent_name = "conv"
        out["conv_L_cache"] = int(l_cache)
    elif wkv:
        recurrent_name = "rwkv"
        out["head_size"] = int(wkv)
    elif ssm_state:
        # Mamba-2 (and every grouped SSM-style mixer llama.cpp sizes the same way) states
        # `ssm.group_count`; Mamba-1 (mamba, falcon-mamba, jamba) states none. The mechanism
        # follows from the file's keys, as it does from an HF config's.
        recurrent_name = "mamba" if get("ssm.group_count") else "mamba1"
        out["ssm_state_size"] = int(ssm_state)
        for suffix, key in (
            ("ssm.conv_kernel", "ssm_conv_kernel"),
            ("ssm.inner_size", "ssm_inner_size"),
            ("ssm.time_step_rank", "time_step_rank"),
        ):
            v = get(suffix)
            if v is not None:
                out[key] = int(v)
        # No group count is Mamba-1: no B/C groups in the conv.
        out["ssm_n_groups"] = int(get("ssm.group_count") or 0)
        if str(arch or "").lower() in GGUF_GATED_DELTANET_ARCHS and out["ssm_n_groups"]:
            v_heads = int(get("ssm.time_step_rank") or out["ssm_n_groups"])
            inner = int(get("ssm.inner_size") or 0)
            recurrent_name = "linear_attention"
            out.update(linear_num_key_heads=out["ssm_n_groups"],
                       linear_num_value_heads=v_heads,
                       linear_key_head_dim=int(ssm_state),
                       linear_value_head_dim=max(inner // max(v_heads, 1), 1),
                       linear_conv_kernel_dim=int(get("ssm.conv_kernel") or 4))

    window = get("attention.sliding_window")
    pattern = get("attention.sliding_window_pattern")
    swa: Optional[List[bool]] = None
    if window:
        if isinstance(pattern, (list, tuple)):
            swa = [bool(x) for x in pattern][:n_layers]
        elif isinstance(pattern, int) and pattern > 1:
            swa = [i % pattern < pattern - 1 for i in range(n_layers)]

    names: Optional[List[str]] = None
    interval = get("full_attention_interval")
    if kv_list is not None and len(kv_list) >= n_layers and any(k == 0 for k in kv_list[:n_layers]):
        # FFNs have their own slots when the layers that attend run none and a 0-KV layer runs
        # one. A stack whose attention layers carry FFNs (Deci, Jamba) pairs an FFN with each
        # layer: naming its 0-KV layers FFN-only slots took the FFN off every attention layer.
        own_ffn_slots = (
            ff_list is not None
            and len(ff_list) >= n_layers
            and all(ff_list[i] == 0 for i in range(n_layers) if kv_list[i] > 0)
            and any(kv_list[i] == 0 and ff_list[i] > 0 for i in range(n_layers))
        )
        names = []
        for i in range(n_layers):
            if kv_list[i] > 0:
                # A pattern shorter than the stack marks only the layers it reaches; the rest
                # attend in full, as `_fit_names` charges the layers a short list leaves out.
                windowed = swa is not None and i < len(swa) and swa[i]
                names.append("sliding_attention" if windowed else "full_attention")
            elif own_ffn_slots and ff_list[i] > 0:
                names.append("mlp")
            elif recurrent_name:
                names.append(recurrent_name)
            else:
                names.append("no_op")
    elif isinstance(interval, int) and interval > 1 and recurrent_name:
        names = [
            "full_attention" if (i + 1) % interval == 0 else recurrent_name for i in range(n_layers)
        ]
    elif recurrent_name:
        heads = _positive(get("attention.head_count")) or 0
        kv_heads = _positive(kv) or 0
        mixed = "hybrid" if recurrent_name in ("mamba", "mamba1") else recurrent_name
        names = [mixed if (heads and kv_heads) else recurrent_name] * n_layers
    elif swa is not None:
        names = ["sliding_attention" if s else "full_attention" for s in swa]

    if names is not None:
        out["layer_types"] = names
    if window and swa is not None:
        out["sliding_window"] = int(window)
    shared = get("attention.shared_kv_layers")
    if shared:
        out["num_kv_shared_layers"] = int(shared)
    return out
