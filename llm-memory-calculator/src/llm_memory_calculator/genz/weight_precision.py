"""Per-role weight precision declared by a pre-quantized checkpoint.

The performance model sizes every weight tensor at ONE caller-supplied precision (`bits`). That is
right for a bf16 checkpoint and wrong for any checkpoint that ships already quantized, because those
are almost never uniform: the quantizer converts the large linear layers and leaves the rest in the
checkpoint dtype. gpt-oss is the extreme case -- its MoE experts are MXFP4 (4.25 bits) while
attention, the router, the embeddings and lm_head stay bf16 (its `modules_to_not_convert`).

Decode is memory-bound, so this is not a rounding error. Timing gpt-oss-20b's experts at bf16 streams
~3.8x the bytes the engine actually reads, and because the union of experts a batch touches grows
with batch size, the error grows with it: measured on an H100 (vLLM, tp=1) the bf16 prediction was
1.3x slow at batch 1 and 3.4x slow at batch 32. The memory calculator already sized these weights
from `quantization_config`; only the performance path ignored it.

`resolve_weight_precision` turns a HuggingFace `quantization_config` into bytes-per-parameter for
each weight ROLE the quantizer converted. Roles it did not convert are absent from the result and
keep the caller's precision, so a checkpoint without a `quantization_config` -- or with a method this
module cannot size -- produces ``None`` and the model is byte-identical to before.

What this deliberately does not change:
  * COMPUTE precision. MXFP4/AWQ/GPTQ/NF4 are weight-only formats: kernels dequantize and run the
    GEMM in the activation dtype, so compute stays at `bits`. (FP8 W8A8 does compute in 8 bits on
    supporting hardware; charging it at bf16 is the conservative side.)
  * KV-cache precision, which is governed by the engine's kv-cache dtype, not by weight quantization.
"""
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

#: Weight roles the performance model can tell apart from its operator graph.
WEIGHT_ROLES = ('attention', 'dense_ffn', 'expert', 'shared_expert', 'router', 'embedding', 'lm_head')

#: What a quantizer converts when the checkpoint names no exclusions: the decoder's large linear
#: layers. Embeddings, lm_head and MoE routers stay in the checkpoint dtype in every mainstream
#: format (HF quantizers skip them by default; routers are tiny and accuracy-critical).
_DEFAULT_QUANTIZED_ROLES: FrozenSet[str] = frozenset({'attention', 'dense_ffn', 'expert', 'shared_expert'})

#: Keys different formats use for "leave these modules unquantized".
_EXCLUSION_KEYS = ('modules_to_not_convert', 'ignore', 'ignored_layers', 'exclude_modules',
                   'llm_int8_skip_modules')

_FFN_ROLES = frozenset({'dense_ffn', 'expert', 'shared_expert'})

#: Bytes per parameter of a weight the quantizer left alone. Serving engines run unquantized layers of a
#: quantized checkpoint in 16-bit (vLLM's dtype=auto serves FP32 and FP16 checkpoints as FP16, BF16 as BF16).
_UNQUANTIZED_BYTES = 2.0


def _bits_to_bytes(bits: float, group_size: Optional[int] = None, scale_bits: int = 16) -> float:
    """Bytes per parameter for `bits`-bit weights with one `scale_bits` scale per group.

    Per-group scales are real bytes the kernel reads with the weights: 16-bit scales on groups of
    128 add 0.125 bits per weight. A non-positive group size means per-channel scales, which are
    negligible per weight.
    """
    overhead = (scale_bits / group_size) if group_size and group_size > 0 else 0.0
    return (bits + overhead) / 8.0


def _quantized_bytes_per_param(qc: Dict[str, Any]) -> Optional[float]:
    method = str(qc.get('quant_method') or qc.get('quant_algo') or '').lower()

    if method == 'mxfp4':
        # OCP Microscaling FP4: 4-bit elements, one 8-bit E8M0 scale per 32-element block.
        return _bits_to_bytes(4, 32, scale_bits=8)
    if method in ('nvfp4', 'modelopt_fp4'):
        # NVFP4: 4-bit elements, one FP8 scale per 16-element block.
        return _bits_to_bytes(4, 16, scale_bits=8)
    if method == 'modelopt':
        return _modelopt_bytes_per_param(qc)
    if method in ('fp8', 'fbgemm_fp8', 'fp8_e4m3', 'fp8_e5m2'):
        return 1.0
    if method in ('awq', 'gptq', 'auto-round', 'autoround', 'gptq_marlin', 'awq_marlin'):
        return _bits_to_bytes(int(qc.get('bits', 4)), int(qc.get('group_size', 128) or -1))
    if method == 'bitsandbytes':
        if qc.get('load_in_8bit'):
            return 1.0
        if qc.get('load_in_4bit'):
            # 4-bit blocks of 64 with a 32-bit absmax; double quantization shrinks that scale to 8 bits.
            scale_bits = 8 if qc.get('bnb_4bit_use_double_quant') else 32
            return _bits_to_bytes(4, 64, scale_bits=scale_bits)
        return None
    return None


def _modelopt_bytes_per_param(qc: Dict[str, Any]) -> Optional[float]:
    """NVIDIA ModelOpt names its format in `quant_algo` rather than `quant_method`."""
    algo = str(qc.get('quant_algo') or '').upper()
    if 'NVFP4' in algo or algo == 'FP4':
        return _bits_to_bytes(4, 16, scale_bits=8)
    if 'MXFP4' in algo:
        return _bits_to_bytes(4, 32, scale_bits=8)
    if 'AWQ' in algo or algo.startswith('INT4') or algo.startswith('W4'):
        return _bits_to_bytes(4, int(qc.get('group_size') or 128))
    if 'FP8' in algo or 'INT8' in algo or algo.startswith('W8'):
        return 1.0
    return None  # e.g. MIXED_PRECISION, which is declared per layer


def _compressed_tensors_weight_bytes(weights: Any) -> Optional[float]:
    """Bytes per parameter for one compressed-tensors `weights` scheme; None when it quantizes no weights."""
    if not isinstance(weights, dict) or not weights.get('num_bits'):
        return None
    bits = int(weights['num_bits'])
    group_size = weights.get('group_size')
    if str(weights.get('strategy') or '').lower() in ('channel', 'tensor'):
        group_size = None
    # 4-bit float is NVFP4, whose per-group scales are FP8; integer schemes store 16-bit scales.
    scale_bits = 8 if str(weights.get('type') or '').lower() == 'float' and bits == 4 else 16
    return _bits_to_bytes(bits, group_size, scale_bits=scale_bits)


def _compressed_tensors_roles(qc: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Resolve compressed-tensors `config_groups`, which may give different roles different schemes.

    A target naming a module CLASS ("Linear") applies to every large linear layer; a target naming modules by
    name or regex ("re:.*mlp.*") applies to the roles it names and takes precedence, as it does in
    compressed-tensors itself. A group with no `weights` scheme quantizes no weights, so the roles it claims
    stay unquantized.
    """
    groups = [g for g in (qc.get('config_groups') or {}).values() if isinstance(g, dict)]
    if not groups:
        return None
    base: Dict[str, Optional[float]] = {}
    override: Dict[str, Optional[float]] = {}
    for group in groups:
        scheme = _compressed_tensors_weight_bytes(group.get('weights'))
        # A group that names no targets applies at class level, like a "Linear" target.
        for target in group.get('targets') or ['Linear']:
            target = str(target)
            if target.startswith('re:') or any(ch in target for ch in '.*|('):
                for role in _roles_named_by(target):
                    override[role] = scheme
            else:
                for role in _DEFAULT_QUANTIZED_ROLES:
                    base.setdefault(role, scheme)
    merged = {**base, **override}
    return {role: value for role, value in merged.items() if value is not None}


def _roles_named_by(pattern: str) -> FrozenSet[str]:
    """Weight roles a module-name pattern refers to ('model.layers.*.self_attn' -> attention)."""
    p = pattern.lower()
    if p.startswith('re:'):
        p = p[3:]
    if 'lm_head' in p:
        return frozenset({'lm_head'})
    if 'embed' in p:
        return frozenset({'embedding'})
    if 'shared_expert' in p:
        return frozenset({'shared_expert'})
    if 'experts' in p:
        return frozenset({'expert'})
    # A MoE router: `mlp.router` or `mlp.gate` -- but never `gate_proj`, which is a dense FFN weight.
    if 'router' in p or re.search(r'(^|[.\\*])gate($|[.$\\])', p):
        return frozenset({'router'})
    if any(k in p for k in ('self_attn', 'attention', 'attn', 'mixer', 'mamba')):
        return frozenset({'attention'})
    if any(k in p for k in ('mlp', 'feed_forward', 'ffn')):
        # `mlp.down_proj` names a dense FFN projection; a bare `mlp` names the whole block,
        # experts included.
        return frozenset({'dense_ffn'}) if '_proj' in p else _FFN_ROLES
    # Bare projection names, as GPTQ's modules_in_block_to_quantize and regex targets spell them.
    if re.search(r'(^|[^a-z])(gate_up|gate|up|down|w1|w2|w3)(_proj)?\b', p) and '_proj' in p \
            or re.search(r'\((gate|up|down)(\|(gate|up|down))+\)_proj', p):
        return frozenset({'dense_ffn'})
    if re.search(r'(^|[^a-z])(q|k|v|o|qkv|qk|kv|query_key_value|out)_proj\b', p) \
            or re.search(r'\(([qkvo])(\|[qkvo])+\)_proj', p):
        return frozenset({'attention'})
    return frozenset()


def _exclusion_patterns(qc: Dict[str, Any]) -> List[str]:
    patterns: List[str] = []
    for key in _EXCLUSION_KEYS:
        value = qc.get(key)
        if isinstance(value, str):
            patterns.append(value)
        elif isinstance(value, (list, tuple)):
            patterns.extend(v for v in value if isinstance(v, str))
    return patterns


def _pinned_layer_indices(pattern: str) -> Optional[Set[int]]:
    """The layer indices a pattern is pinned to, or None when it names every layer.

    "model.layers.0.mlp" -> {0}; "re:model.layers.(0|1|61).mlp" -> {0, 1, 61}; "model.layers.[0-2].mlp" ->
    {0, 1, 2}; "model.layers.*.mlp" -> None. A pinned pattern whose indices cannot be read returns an empty
    set: it is known NOT to be model-wide, even though how many layers it covers is not.
    """
    match = re.search(r'layers?\.((?:\(|\[|\d)[^.]*)', pattern)
    if not match:
        return None
    segment = match.group(1)
    indices: Set[int] = set()
    for lo, hi in re.findall(r'(\d+)\s*-\s*(\d+)', segment):
        indices.update(range(int(lo), int(hi) + 1))
    segment = re.sub(r'\d+\s*-\s*\d+', '', segment)
    indices.update(int(n) for n in re.findall(r'\d+', segment))
    return indices


def _excluded_roles(qc: Dict[str, Any]) -> FrozenSet[str]:
    """Roles excluded on EVERY layer. Layer-pinned patterns are accounted by _pinned_exclusion_fractions."""
    excluded = set()
    for pattern in _exclusion_patterns(qc):
        if _pinned_layer_indices(pattern) is None:
            excluded |= _roles_named_by(pattern)
    return frozenset(excluded)


def _pinned_exclusion_fractions(qc: Dict[str, Any], num_layers: Optional[int]) -> Dict[str, float]:
    """Fraction of layers each role is left unquantized on by layer-pinned exclusions.

    Decode reads every layer's weights each step, so a role converted on L-n of L layers streams the
    layer-weighted mean of the two precisions -- exact for the bytes, which is what memory-bound decode
    charges.
    """
    if not num_layers or num_layers <= 0:
        return {}
    per_role: Dict[str, Set[int]] = defaultdict(set)
    for pattern in _exclusion_patterns(qc):
        indices = _pinned_layer_indices(pattern)
        if not indices:
            continue
        for role in _roles_named_by(pattern):
            per_role[role] |= {i for i in indices if 0 <= i < num_layers}
    return {role: len(layers) / num_layers for role, layers in per_role.items() if layers}


def _quantized_roles(qc: Dict[str, Any]) -> FrozenSet[str]:
    """Roles a single-scheme quantizer converted: GPTQ's explicit module list when it gives one, otherwise
    the large linear layers minus the model-wide exclusions."""
    listed = qc.get('modules_in_block_to_quantize')
    if isinstance(listed, (list, tuple)) and listed:
        names = [n for item in listed for n in (item if isinstance(item, (list, tuple)) else [item])]
        roles = set()
        for name in names:
            roles |= _roles_named_by(str(name))
        return frozenset(roles & _DEFAULT_QUANTIZED_ROLES) - _excluded_roles(qc)
    return _DEFAULT_QUANTIZED_ROLES - _excluded_roles(qc)


def resolve_weight_precision(hf_config: Any) -> Optional[Dict[str, float]]:
    """Bytes per parameter for each weight role a pre-quantized checkpoint converted.

    Returns ``None`` when the config declares no quantization, or one this module cannot size --
    the caller's uniform precision then applies, exactly as before.
    """
    if not isinstance(hf_config, dict):
        return None
    qc = hf_config.get('quantization_config') or hf_config.get('compression_config')
    if not isinstance(qc, dict) or not qc:
        return None
    method = str(qc.get('quant_method') or qc.get('quant_algo') or '').lower()
    if method == 'compressed-tensors':
        role_bytes = _compressed_tensors_roles(qc)
        if role_bytes is not None:
            for role in _excluded_roles(qc):
                role_bytes.pop(role, None)
    else:
        bytes_per_param = _quantized_bytes_per_param(qc)
        role_bytes = None if bytes_per_param is None else {role: bytes_per_param for role in _quantized_roles(qc)}
    if role_bytes is None:
        warnings.warn(
            f"quantization_config method {qc.get('quant_method') or qc.get('quant_algo')!r} is not "
            "understood by the performance model; weights are timed at the caller's precision",
            stacklevel=2,
        )
        return None
    num_layers = hf_config.get('num_hidden_layers')
    num_layers = int(num_layers) if isinstance(num_layers, (int, float)) else None
    for role, fraction in _pinned_exclusion_fractions(qc, num_layers).items():
        if role in role_bytes:
            role_bytes[role] = role_bytes[role] * (1 - fraction) + _UNQUANTIZED_BYTES * fraction
    return {role: role_bytes[role] for role in sorted(role_bytes)} or None


# ------------------------------------------------------------------ operator graph -> role


_ATTENTION_OPS = frozenset({'QKV', 'Out Proj', 'Q Down', 'Q Up', 'KV Compress', 'KV Up', 'Attn Linear',
                            'Inproj', 'Out proj', 'xt proj', 'BC proj'})
_FFN_OPS = frozenset({'up+gate', 'down'})


class WeightRoleTracker:
    """Assigns a weight role to each operator row of a GenZ layer graph, in order.

    Expert and dense FFN projections share the operator names `up+gate`/`down`. What tells them
    apart is structural: ffn.py always emits a MoE layer's router (`Gate`) before its experts, and
    a dense FFN has no router. So a `Gate` seen since the layer's attention marks the FFN rows that
    follow as experts.
    """

    def __init__(self):
        self._in_moe_block = False

    def role_for(self, layer_name: Any) -> Optional[str]:
        name = str(layer_name)
        if name in ('Repeat', 'End Repeat') or name in _ATTENTION_OPS:
            self._in_moe_block = False
            return 'attention' if name in _ATTENTION_OPS else None
        if name == 'Gate':
            self._in_moe_block = True
            return 'router'
        if name in _FFN_OPS:
            return 'expert' if self._in_moe_block else 'dense_ffn'
        if name in ('shared up+gate', 'shared down'):
            return 'shared_expert'
        if name == 'FFN Linear':
            return 'dense_ffn'
        if name == 'embeddings':
            return 'embedding'
        if name == 'classifier':
            return 'lm_head'
        return None
