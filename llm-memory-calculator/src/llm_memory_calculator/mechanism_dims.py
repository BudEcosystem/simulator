"""Shared dimension resolution for recurrent mixers.

State sizing and parameter counting both need the same numbers out of the same
configs, and they must agree. Two independent copies of this logic drifted
almost immediately: one validated ``num_heads``/``head_dim`` against d_inner
while the other assumed a head width of 64. The product ``n_heads * d_head``
came out the same either way, so the recurrent-state term hid the divergence --
but ``n_heads`` enters the parameter formula additively, so the counts would
have disagreed for any Mamba-2 model whose head width is not 64.
"""

from typing import Any, Dict

# Config key aliases, per dialect. `state_size` (Mamba proper), `mamba_d_state`
# (Jamba, Falcon-H1, granite-4.0-h, Bamba, Zamba2) and `ssm_state_size`
# (Nemotron-H) all mean d_state. A missing alias used to fall through to a
# hardcoded 16, which is 16x off on a Falcon-H1 that ships 256.
_D_STATE = ("state_size", "d_state", "mamba_d_state", "ssm_state_size")
_EXPAND = ("expand", "expand_factor", "mamba_expand", "ssm_expand")
_D_CONV = ("conv_kernel", "d_conv", "mamba_d_conv", "ssm_conv_kernel")
_N_GROUPS = ("n_groups", "mamba_n_groups", "ssm_n_groups")
_N_HEADS = ("mamba_n_heads", "mamba_num_heads", "n_mamba_heads")
_D_HEAD = ("mamba_d_head", "mamba_head_dim")
_DT_RANK = ("dt_rank", "time_step_rank", "mamba_dt_rank")


def first(config: Dict[str, Any], *keys, default=None):
    """First present, non-None key among aliases.

    `.get(key, default)` returns a stored ``None``, and these configs routinely
    ship optional keys as explicit JSON null, so presence is not enough.
    """
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def mamba_dims(config: Dict[str, Any], hidden_size: int) -> Dict[str, int]:
    """Resolve Mamba-1/Mamba-2 dimensions across every key dialect in use."""
    d_state = int(first(config, *_D_STATE, default=16))
    expand = int(first(config, *_EXPAND, default=2))
    d_conv = int(first(config, *_D_CONV, default=4))
    n_groups = int(first(config, *_N_GROUPS, default=1))

    n_heads = first(config, *_N_HEADS)
    d_head = first(config, *_D_HEAD)

    if n_heads and d_head:
        # Authoritative. Nemotron-H's d_inner is mamba_num_heads x
        # mamba_head_dim = 10240, which has no relation to expand x hidden_size.
        d_inner = int(n_heads) * int(d_head)
    else:
        d_inner = expand * hidden_size
        if n_heads:
            d_head = max(1, d_inner // int(n_heads))
        else:
            # Mamba-2 proper puts head counts under the generic names, but
            # `num_heads` also means *attention* heads on hybrid configs -- only
            # trust them when they are consistent with d_inner.
            nh, dh = config.get("num_heads"), config.get("head_dim")
            if nh and dh and int(nh) * int(dh) == d_inner:
                n_heads, d_head = int(nh), int(dh)
            else:
                n_heads, d_head = 1, d_inner

    return {
        "d_state": d_state,
        "d_conv": d_conv,
        "n_groups": n_groups,
        "d_inner": d_inner,
        "n_heads": int(n_heads),
        "d_head": int(d_head),
        "dt_rank": int(first(config, *_DT_RANK, default=max(1, hidden_size // 16))),
    }
