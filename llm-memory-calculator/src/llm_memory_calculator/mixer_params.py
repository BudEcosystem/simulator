"""Per-layer token-mixer parameter counts, by mechanism.

The parameter counter charged every layer a full Q/K/V/O projection set. On a
hybrid that is wrong twice: the recurrent layers have no QKVO at all, and the
projections they *do* have (a fused in_proj, a depthwise conv, low-rank gates,
per-head decay vectors) are shaped nothing like attention. Kimi-Linear came out
at 2.10B against a real 49.12B; Qwen3.6 at 25.88B against 27.78B.

Every formula here is derived from the shipped ``model.safetensors`` tensor
shapes, not from a paper, and the docstrings record the shapes that pin them.
That matters because several of these families ship dimensions that cannot be
inferred from the obvious config keys -- Nemotron-H's d_inner is
``mamba_num_heads * mamba_head_dim`` (10240) and has no relation to
``expand * hidden_size``.
"""

from typing import Any, Dict, List, Optional, Tuple

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


#: A mechanism's linear maps, ``[(in_features, out_features), ...]`` for one layer: every
#: projection, and the depthwise convolution where an engine holds it as one (vLLM builds every
#: Mamba-1 / Mamba-2 / short-conv / KDA convolution as a ``ColumnParallelLinear`` of kernel ->
#: channels, so it is a LoRA-wrapped linear there). ``*_params`` is these maps' weights plus the
#: vectors no linear holds (biases, decays, norms), so a parameter count and a LoRA buffer count
#: read the same shapes.


def _weights(linears) -> int:
    return sum(i * o for i, o in linears)


def gdn_linears(config: Dict[str, Any], hidden: int):
    """Gated DeltaNet's maps: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, the conv (kernel ->
    q|k|v channels) and out_proj (Qwen3.5's split spelling; Qwen3-Next fuses qkv|z and a|b, the
    same shapes side by side)."""
    nk = int(_first(config, "linear_num_key_heads", default=16))
    nv = int(_first(config, "linear_num_value_heads", default=32))
    kd = int(_first(config, "linear_key_head_dim", default=128))
    vd = int(_first(config, "linear_value_head_dim", default=128))
    k = int(_first(config, "linear_conv_kernel_dim", default=4))
    qkv = 2 * nk * kd + nv * vd
    v_dim = nv * vd
    return [(hidden, qkv), (hidden, v_dim), (hidden, nv), (hidden, nv), (k, qkv), (v_dim, hidden)]


def gdn_params(config: Dict[str, Any], hidden: int) -> int:
    """Gated DeltaNet (Qwen3-Next, Qwen3.5/3.6), one layer.

    Pinned against Qwen3.6-27B (hidden 5120, 16/128 key, 48/128 value, k=4)::

        in_proj_qkv [10240, 5120]   in_proj_z [6144, 5120]
        in_proj_a   [48, 5120]      in_proj_b [48, 5120]
        conv1d      [10240, 1, 4]   out_proj  [5120, 6144]
        A_log [48]  dt_bias [48]    norm [128]

    10240 = 2*16*128 + 48*128, i.e. q and k are key-width and v is value-width.
    """
    nv = int(_first(config, "linear_num_value_heads", default=32))
    vd = int(_first(config, "linear_value_head_dim", default=128))
    return _weights(gdn_linears(config, hidden)) + 2 * nv + vd  # A_log, dt_bias; head-wise norm


def _kda_dims(config: Dict[str, Any]):
    lac = config.get("linear_attn_config") or {}
    nh = int(_first(lac, "num_heads", default=32))
    hd = int(_first(lac, "head_dim", default=128))
    k = int(_first(lac, "short_conv_kernel_size", default=4))
    return nh, hd, k


def kda_linears(config: Dict[str, Any], hidden: int):
    """Kimi Delta Attention's maps: q / k / v / o, the three short convs, the low-rank f and g
    gates (through head_dim) and the per-head beta."""
    nh, hd, k = _kda_dims(config)
    inner = nh * hd
    rank = hd  # f_a/g_a project to head_dim, per the shipped [128, 2304]
    return ([(hidden, inner)] * 3 + [(inner, hidden)] + [(k, inner)] * 3
            + [(hidden, rank), (rank, inner)] * 2 + [(hidden, nh)])


def kda_params(config: Dict[str, Any], hidden: int) -> int:
    """Kimi Delta Attention, one layer.

    Pinned against Kimi-Linear-48B (hidden 2304, 32 heads x 128)::

        q_proj/k_proj/v_proj [4096, 2304]   o_proj [2304, 4096]
        q_conv1d/k_conv1d/v_conv1d [4096, 1, 4]
        f_a_proj/g_a_proj [128, 2304]       f_b_proj/g_b_proj [4096, 128]
        b_proj [32, 2304]   (the per-head beta)
        A_log [1,1,32,1]  dt_bias [4096]    o_norm [128]

    The f/g gates are low-rank, factored through head_dim -- counting them dense
    would add ~2 x hidden x n_heads*head_dim per layer that does not exist.
    """
    nh, hd, _ = _kda_dims(config)
    return _weights(kda_linears(config, hidden)) + nh + nh * hd + hd  # A_log, dt_bias, o_norm


def mamba2_linears(config: Dict[str, Any], hidden: int):
    """Mamba-2's maps: in_proj (x, z, B, C, dt), the conv (kernel -> x|B|C channels), out_proj."""
    d = _mamba_dims(config, hidden)
    conv_dim = d["d_inner"] + 2 * d["n_groups"] * d["d_state"]
    return [(hidden, 2 * d["d_inner"] + 2 * d["n_groups"] * d["d_state"] + d["n_heads"]),
            (d["d_conv"], conv_dim), (d["d_inner"], hidden)]


def mamba2_params(config: Dict[str, Any], hidden: int) -> int:
    """Mamba-2 (Nemotron-H, granite-4.0-h, Bamba, Falcon-H1, Codestral), one layer.

    Pinned against granite-4.0-h-small (hidden 4096, 128 heads x 64, state 128,
    1 group) and NVIDIA-Nemotron-Nano-9B-v2 (hidden 4480, 128 x 80, state 128,
    8 groups)::

        in_proj [2*d_inner + 2*n_groups*d_state + n_heads, hidden]
        conv1d  [d_inner + 2*n_groups*d_state, 1, d_conv] (+ bias)
        A_log/D/dt_bias [n_heads]  norm [d_inner]  out_proj [hidden, d_inner]

    granite: 2*8192 + 2*1*128 + 128 = 16768 and 8192 + 256 = 8448. Both match.
    """
    d = _mamba_dims(config, hidden)
    conv_dim = d["d_inner"] + 2 * d["n_groups"] * d["d_state"]
    return (
        _weights(mamba2_linears(config, hidden))
        + conv_dim                           # conv bias
        + 3 * d["n_heads"]                   # A_log, D, dt_bias
        + d["d_inner"]                       # gated norm
    )


def mamba1_linears(config: Dict[str, Any], hidden: int):
    """Mamba-1's maps: in_proj (x and z), the conv, x_proj (dt, B, C), dt_proj, out_proj."""
    d = _mamba_dims(config, hidden)
    return [(hidden, 2 * d["d_inner"]), (d["d_conv"], d["d_inner"]),
            (d["d_inner"], d["dt_rank"] + 2 * d["d_state"]), (d["dt_rank"], d["d_inner"]),
            (d["d_inner"], hidden)]


def mamba1_params(config: Dict[str, Any], hidden: int) -> int:
    """Mamba-1 (Jamba, falcon_mamba, mamba-2.8b), one layer.

    Distinct from Mamba-2 in that dt is produced by a low-rank x_proj/dt_proj
    pair rather than a per-head bias, and A is a full d_inner x d_state matrix.
    """
    d = _mamba_dims(config, hidden)
    return (
        _weights(mamba1_linears(config, hidden))
        + d["d_inner"]                  # conv bias
        + d["d_inner"]                  # dt_proj bias
        + d["d_inner"] * d["d_state"]   # A_log
        + d["d_inner"]                  # D
    )


def shortconv_linears(config: Dict[str, Any], hidden: int):
    """LFM2's short-convolution maps: in_proj (B, C, x), the conv, out_proj."""
    L = int(_first(config, "conv_L_cache", default=3) or 3)
    return [(hidden, 3 * hidden), (L, hidden), (hidden, hidden)]


def shortconv_params(config: Dict[str, Any], hidden: int) -> int:
    """LFM2 short-convolution block, one layer.

    Pinned against LFM2-2.6B (hidden 2048, conv_L_cache 3)::

        conv.in_proj [6144, 2048]   conv.conv [2048, 1, 3]   conv.out_proj [2048, 2048]
    """
    return _weights(shortconv_linears(config, hidden))


def lightning_linears(config: Dict[str, Any], hidden: int):
    """MiniMax lightning attention's maps: the fused qkv_proj, the output gate, out_proj."""
    nh = int(_first(config, "num_attention_heads", default=32))
    hd = int(_first(config, "head_dim", default=max(1, hidden // nh)))
    inner = nh * hd
    return [(hidden, 3 * inner), (hidden, inner), (inner, hidden)]


def lightning_params(config: Dict[str, Any], hidden: int) -> int:
    """MiniMax lightning attention, one layer.

    Same q/k/v/o projection set as softmax attention plus an output gate; the
    difference is in how the scores are computed, not in the weight shapes.
    """
    return _weights(lightning_linears(config, hidden))  # q,k,v,o + output gate


def rwkv_linears(config: Dict[str, Any], hidden: int):
    """RWKV time-mix maps: key, value, receptance (hidden -> attention_hidden_size), output."""
    att = int(_first(config, "attention_hidden_size", default=hidden) or hidden)
    return [(hidden, att)] * 3 + [(att, hidden)]


def rwkv_params(config: Dict[str, Any], hidden: int) -> int:
    """RWKV time-mix block, one layer (transformers ``RwkvSelfAttention``).

    key / value / receptance project hidden -> attention_hidden_size, output
    projects back; time_decay and time_first are one vector each, the three
    time_mix interpolations one hidden-wide vector each.
    """
    att = int(_first(config, "attention_hidden_size", default=hidden) or hidden)
    return _weights(rwkv_linears(config, hidden)) + 2 * att + 3 * hidden


def mlstm_linears(config: Dict[str, Any], hidden: int):
    """xLSTM mLSTM maps: q, k (hidden -> qk_dim), v and the output gate (hidden -> v_dim), the
    input and forget gates (hidden -> num_heads each), the output projection."""
    nh = int(_first(config, "num_heads", default=8))
    qk = -(-int(round(hidden * float(_first(config, "qk_dim_factor", default=0.5)))) // 64) * 64
    v = -(-int(round(hidden * float(_first(config, "v_dim_factor", default=1.0)))) // 64) * 64
    return [(hidden, qk)] * 2 + [(hidden, v)] * 2 + [(hidden, nh)] * 2 + [(v, hidden)]


def mlstm_params(config: Dict[str, Any], hidden: int) -> int:
    """xLSTM mLSTM layer: q, k (hidden -> qk_dim), v and the output gate
    (hidden -> v_dim), input/forget gates (hidden -> num_heads each) and the
    output projection (v_dim -> hidden)."""
    return _weights(mlstm_linears(config, hidden))


def rglru_linears(config: Dict[str, Any], hidden: int):
    """RecurrentGemma recurrent block's maps: linear_x, linear_y, linear_out and the conv (the
    RG-LRU's block-diagonal gates are parameters, not linear modules)."""
    width = int(_first(config, "lru_width", default=hidden) or hidden)
    k = int(_first(config, "conv1d_width", default=4))
    return [(hidden, width), (hidden, width), (width, hidden), (k, width)]


def rglru_params(config: Dict[str, Any], hidden: int) -> int:
    """RecurrentGemma recurrent block: linear_x, linear_y (hidden -> lru_width)
    and linear_out back, the depthwise conv, and the RG-LRU's block-diagonal
    input and recurrence gates plus its recurrence parameter."""
    width = int(_first(config, "lru_width", default=hidden) or hidden)
    nh = int(_first(config, "num_attention_heads", default=1) or 1)
    return _weights(rglru_linears(config, hidden)) + 2 * width * width // max(nh, 1) + width


_DISPATCH = {
    MECH_GDN: gdn_params,
    MECH_KDA: kda_params,
    MECH_MAMBA2: mamba2_params,
    MECH_MAMBA1: mamba1_params,
    MECH_SHORTCONV: shortconv_params,
    MECH_LIGHTNING: lightning_params,
    MECH_RWKV: rwkv_params,
    MECH_MLSTM: mlstm_params,
    MECH_RGLRU: rglru_params,
}


_LINEARS = {
    MECH_GDN: gdn_linears,
    MECH_KDA: kda_linears,
    MECH_MAMBA2: mamba2_linears,
    MECH_MAMBA1: mamba1_linears,
    MECH_SHORTCONV: shortconv_linears,
    MECH_LIGHTNING: lightning_linears,
    MECH_RWKV: rwkv_linears,
    MECH_MLSTM: mlstm_linears,
    MECH_RGLRU: rglru_linears,
}


def recurrent_mixer_linears(
    config: Dict[str, Any], plan: Optional[Dict[str, Any]], hidden: int
) -> List[Tuple[int, int]]:
    """Every linear map of every recurrent layer in the plan, ``[(in, out), ...]``: what an engine
    that wraps every linear layer with LoRA (vLLM: "all linear layers support LoRA") allocates
    adapter buffers for on the layers that hold no attention."""
    if not plan or not plan.get("num_recurrent_layers"):
        return []
    out: List[Tuple[int, int]] = []
    for mech in plan["recurrent"]:
        if mech:
            out.extend(_LINEARS[mech](config, hidden))
    return out


def recurrent_mixer_params(
    config: Dict[str, Any], plan: Optional[Dict[str, Any]], hidden: int
) -> int:
    """Total parameters across every recurrent layer in the plan."""
    if not plan or not plan.get("num_recurrent_layers"):
        return 0
    total = 0
    for mech in plan["recurrent"]:
        if mech:
            total += _DISPATCH[mech](config, hidden)
    return int(total)
