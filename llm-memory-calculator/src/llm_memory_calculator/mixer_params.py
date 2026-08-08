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

from typing import Any, Dict, Optional

from .layer_plan import (
    MECH_GDN,
    MECH_KDA,
    MECH_LIGHTNING,
    MECH_MAMBA1,
    MECH_MAMBA2,
    MECH_SHORTCONV,
)


def _first(config: Dict[str, Any], *keys, default=None):
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def _mamba_dims(config: Dict[str, Any], hidden: int) -> Dict[str, int]:
    d_state = int(_first(config, "state_size", "d_state", "mamba_d_state", "ssm_state_size", default=16))
    expand = int(_first(config, "expand", "expand_factor", "mamba_expand", "ssm_expand", default=2))
    d_conv = int(_first(config, "conv_kernel", "d_conv", "mamba_d_conv", "ssm_conv_kernel", default=4))
    n_groups = int(_first(config, "n_groups", "mamba_n_groups", "ssm_n_groups", default=1))
    n_heads = _first(config, "mamba_n_heads", "mamba_num_heads", "n_mamba_heads")
    d_head = _first(config, "mamba_d_head", "mamba_head_dim")

    if n_heads and d_head:
        d_inner = int(n_heads) * int(d_head)
    else:
        d_inner = expand * hidden
        n_heads = int(n_heads) if n_heads else max(1, d_inner // 64)
        d_head = max(1, d_inner // n_heads)
    dt_rank = int(_first(config, "dt_rank", "time_step_rank", "mamba_dt_rank", default=max(1, hidden // 16)))
    return dict(
        d_state=d_state, d_conv=d_conv, n_groups=n_groups, d_inner=d_inner,
        n_heads=int(n_heads), d_head=int(d_head), dt_rank=dt_rank,
    )


def gdn_params(config: Dict[str, Any], hidden: int) -> int:
    """Gated DeltaNet (Qwen3-Next, Qwen3.5/3.6), one layer.

    Pinned against Qwen3.6-27B (hidden 5120, 16/128 key, 48/128 value, k=4)::

        in_proj_qkv [10240, 5120]   in_proj_z [6144, 5120]
        in_proj_a   [48, 5120]      in_proj_b [48, 5120]
        conv1d      [10240, 1, 4]   out_proj  [5120, 6144]
        A_log [48]  dt_bias [48]    norm [128]

    10240 = 2*16*128 + 48*128, i.e. q and k are key-width and v is value-width.
    """
    nk = int(_first(config, "linear_num_key_heads", default=16))
    nv = int(_first(config, "linear_num_value_heads", default=32))
    kd = int(_first(config, "linear_key_head_dim", default=128))
    vd = int(_first(config, "linear_value_head_dim", default=128))
    k = int(_first(config, "linear_conv_kernel_dim", default=4))
    qkv = 2 * nk * kd + nv * vd
    v_dim = nv * vd
    return (
        hidden * qkv        # fused q|k|v projection
        + hidden * v_dim    # in_proj_z (output gate)
        + 2 * hidden * nv   # in_proj_a, in_proj_b (per-head decay/beta)
        + qkv * k           # depthwise conv
        + 2 * nv            # A_log, dt_bias
        + vd                # head-wise output norm
        + v_dim * hidden    # out_proj
    )


def kda_params(config: Dict[str, Any], hidden: int) -> int:
    """Kimi Delta Attention, one layer.

    Pinned against Kimi-Linear-48B (hidden 2304, 32 heads x 128)::

        q_proj/k_proj/v_proj [4096, 2304]   o_proj [2304, 4096]
        q_conv1d/k_conv1d/v_conv1d [4096, 1, 4]
        f_a_proj/g_a_proj [128, 2304]       f_b_proj/g_b_proj [4096, 128]
        A_log [1,1,32,1]  dt_bias [4096]    o_norm [128]

    The f/g gates are low-rank, factored through head_dim -- counting them dense
    would add ~2 x hidden x n_heads*head_dim per layer that does not exist.
    """
    lac = config.get("linear_attn_config") or {}
    nh = int(_first(lac, "num_heads", default=32))
    hd = int(_first(lac, "head_dim", default=128))
    k = int(_first(lac, "short_conv_kernel_size", default=4))
    inner = nh * hd
    rank = hd  # f_a/g_a project to head_dim, per the shipped [128, 2304]
    return (
        4 * hidden * inner          # q, k, v, o
        + 3 * inner * k             # per-projection depthwise convs
        + 2 * (hidden * rank + rank * inner)  # f and g low-rank gates
        + nh                        # A_log
        + inner                     # dt_bias
        + hd                        # o_norm
    )


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
        hidden * (2 * d["d_inner"] + 2 * d["n_groups"] * d["d_state"] + d["n_heads"])
        + conv_dim * d["d_conv"] + conv_dim  # conv weight + bias
        + 3 * d["n_heads"]                   # A_log, D, dt_bias
        + d["d_inner"]                       # gated norm
        + d["d_inner"] * hidden              # out_proj
    )


def mamba1_params(config: Dict[str, Any], hidden: int) -> int:
    """Mamba-1 (Jamba, falcon_mamba, mamba-2.8b), one layer.

    Distinct from Mamba-2 in that dt is produced by a low-rank x_proj/dt_proj
    pair rather than a per-head bias, and A is a full d_inner x d_state matrix.
    """
    d = _mamba_dims(config, hidden)
    return (
        hidden * 2 * d["d_inner"]                       # in_proj (x and z)
        + d["d_inner"] * d["d_conv"] + d["d_inner"]     # conv weight + bias
        + d["d_inner"] * (d["dt_rank"] + 2 * d["d_state"])  # x_proj
        + d["dt_rank"] * d["d_inner"] + d["d_inner"]    # dt_proj + bias
        + d["d_inner"] * d["d_state"]                   # A_log
        + d["d_inner"]                                  # D
        + d["d_inner"] * hidden                         # out_proj
    )


def shortconv_params(config: Dict[str, Any], hidden: int) -> int:
    """LFM2 short-convolution block, one layer.

    Pinned against LFM2-2.6B (hidden 2048, conv_L_cache 3)::

        conv.in_proj [6144, 2048]   conv.conv [2048, 1, 3]   conv.out_proj [2048, 2048]
    """
    L = int(_first(config, "conv_L_cache", default=3) or 3)
    return 3 * hidden * hidden + hidden * L + hidden * hidden


def lightning_params(config: Dict[str, Any], hidden: int) -> int:
    """MiniMax lightning attention, one layer.

    Same q/k/v/o projection set as softmax attention plus an output gate; the
    difference is in how the scores are computed, not in the weight shapes.
    """
    nh = int(_first(config, "num_attention_heads", default=32))
    hd = int(_first(config, "head_dim", default=max(1, hidden // nh)))
    inner = nh * hd
    return 4 * hidden * inner + hidden * inner  # q,k,v,o + output gate


_DISPATCH = {
    MECH_GDN: gdn_params,
    MECH_KDA: kda_params,
    MECH_MAMBA2: mamba2_params,
    MECH_MAMBA1: mamba1_params,
    MECH_SHORTCONV: shortconv_params,
    MECH_LIGHTNING: lightning_params,
}


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
