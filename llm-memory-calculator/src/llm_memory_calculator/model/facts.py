"""Layer 1 of the FRD-027 predictor: one engine-agnostic description of a model (FRD-027 §7).

``ModelFacts`` is built once from a Hugging Face config and answers, for memory **and** performance
(FR-MDL-2), the same questions: parameters and bytes per weight role at a given weight format, KV bytes per
token per layer kind, recurrent state per sequence, bytes streamed and FLOPs per decode step and per prefill.
Linear-attention (Gated DeltaNet, Mamba) layers cost their recurrent state read and write, not attention
(FR-MDL-1); sliding layers cost the window; MLA costs the latent. Components (FR-MDL-4) keep the vision
tower and the MTP head apart from the decoder so they are charged only when loaded.

Parameters are counted by role from the config, so that a precision choice can change exactly the roles a
quantizer converts. The counts are checked against ``UniversalParameterCounter`` and against the weights
vLLM reported loading in the FRD-027 study (tests/planner/test_model_facts.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from ..layer_plan import (
    ATTN_CROSS,
    ATTN_FULL,
    ATTN_SLIDING,
    attention_head_dim,
    decoder_view,
    encoder_decoder_stacks,
    layer_kinds,
    resolve_layer_plan,
)
from ..state_memory import _per_layer_bytes  # noqa: PLC2701 - the one per-mechanism state formula
from .classes import arch_class, size_bucket
from .recognise import recognised as _recognised

#: Roles a quantizer converts (FP8/INT8/INT4 checkpoints and online quantization).
QUANTIZED_ROLES = frozenset({"attention", "dense_ffn", "expert", "shared_expert"})
#: Roles that stay 16-bit in every mainstream format.
ALWAYS_16 = frozenset({"embedding", "lm_head", "router", "mixer_small", "norm"})

_DTYPE_BYTES = {
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "half": 2.0,
    "auto": 2.0,
    "fp32": 4.0,
    "float32": 4.0,
    "float": 4.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
}

#: Bytes per parameter of a converted role, per weight format.
_FORMAT_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0 + 4.0 / (128 * 128),  # e4m3 + one fp32 scale per 128x128 block
    "int8": 1.0 + 2.0 / 128,
    "int4": (4 + 16 + 4) / 8 / 128 * 128 / 128
    + 0.5
    - 0.5
    + 4.25 / 8
    - 4.25 / 8
    + (4.0 + 20.0 / 128) / 8,
    "awq": (4.0 + 20.0 / 128) / 8,
    "gptq": (4.0 + 20.0 / 128) / 8,
    "mxfp4": 4.25 / 8,
    "nvfp4": 4.5 / 8,
}
_FORMAT_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "w8a8": "int8",
    "fp8_w8a8": "fp8",
    "int4_awq": "awq",
}

_QUANT_METHODS = {
    "fp8": "fp8",
    "fbgemm_fp8": "fp8",
    "modelopt": "fp8",
    "compressed-tensors": "fp8",
    "mxfp4": "mxfp4",
    "awq": "awq",
    "awq_marlin": "awq",
    "gptq": "gptq",
    "gptq_marlin": "gptq",
    "bitsandbytes": "int4",
    "nvfp4": "nvfp4",
    "int8": "int8",
    "w8a8": "int8",
}


def _text(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def _get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def dtype_bytes(dtype: str) -> float:
    return _DTYPE_BYTES.get(str(dtype or "bf16").lower(), 2.0)


def canonical_format(fmt: str) -> str:
    text = str(fmt or "bf16").lower()
    return _FORMAT_ALIASES.get(text, text)


@dataclass(frozen=True)
class LayerCounts:
    full: int
    sliding: int
    recurrent: int
    sliding_window: Optional[int] = None
    mechanisms: Tuple[str, ...] = ()
    mla: bool = False

    @property
    def total(self) -> int:
        return self.full + self.sliding + self.recurrent


@dataclass(frozen=True)
class Component:
    """A model component (FR-MDL-4): ``decoder`` (v1), ``vision_encoder`` and ``mtp`` charged only when
    loaded; ``audio_encoder``/``pooling_head`` are named for later tasks."""

    name: str
    params: int


@dataclass(frozen=True)
class ModelFacts:
    model_type: str
    architectures: Tuple[str, ...]
    family: str
    recognised: bool
    flags: FrozenSet[str]
    arch_class: str
    attributes: FrozenSet[str]
    hidden_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    layers: LayerCounts
    params_by_role: Mapping[str, int]
    mtp_params_by_role: Mapping[str, int]
    vision_params: int
    experts: Optional[Tuple[int, int]]  # (routed experts, experts per token)
    mtp_layers: int
    attention_sinks: bool
    max_context: int
    native_format: str
    native_role_bytes: Mapping[str, float]
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    gdn_dims: Optional[Tuple[int, int, int, int]] = (
        None  # (key heads, value heads, key dim, value dim)
    )
    state_per_layer: Mapping[str, Tuple[float, float]] = field(
        default_factory=dict
    )  # mech -> (recurrent elems, conv bytes@bf16)
    components: Tuple[Component, ...] = ()
    encoder_decoder: bool = False
    conv_kernel: int = 4
    moe_layers: int = 0
    draft_vocab: int = 0  # EAGLE3 drafts: the draft head's vocabulary
    shares_embedding: bool = False  # a draft that reads the target's embedding table
    config_digest: str = ""

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, family: Optional[str] = None
    ) -> "ModelFacts":
        config = dict(config or {})
        return _build(config, family)

    # ------------------------------------------------------------------ attention kinds

    @property
    def full_attention(self) -> bool:
        """Every layer is full attention: no sliding-window and no recurrent (Mamba / Gated DeltaNet) layers. vLLM
        serves such a model from one full-attention KV manager, whose prefix-cache hit path is the only one the
        speculative-decoding scans cover (FRD-027 §10.5)."""
        return self.layers.full > 0 and self.layers.sliding == 0 and self.layers.recurrent == 0

    # ------------------------------------------------------------------ parameters

    @property
    def params_total(self) -> int:
        """Decoder parameters (text model), without the MTP head and the vision tower."""
        return int(sum(self.params_by_role.values()))

    @property
    def params_active(self) -> int:
        """Parameters a token touches (MoE: k of E routed experts; embedding lookups excluded)."""
        roles = dict(self.params_by_role)
        roles.pop("embedding", None)
        if self.experts and roles.get("expert"):
            n, k = self.experts
            roles["expert"] = int(roles["expert"] * k / max(n, 1))
        return int(sum(roles.values()))

    @property
    def size_bucket(self) -> str:
        return size_bucket(self.params_active)

    def component(self, name: str) -> Component:
        for comp in self.components:
            if comp.name == name:
                return comp
        raise KeyError(name)

    # ------------------------------------------------------------------ weights

    def role_bytes(self, fmt: str = "native") -> Dict[str, float]:
        """Bytes per parameter for each role at a weight format (FR-MDL-2: memory and performance both read
        this). ``native`` is the checkpoint as stored; another format is an online conversion of the
        16-bit checkpoint, or a registry variant in that format."""
        fmt = canonical_format(fmt)
        if fmt == "native" or fmt == self.native_format:
            return dict(self.native_role_bytes)
        per = _FORMAT_BYTES.get(fmt)
        if per is None:
            return dict(self.native_role_bytes)
        out = {}
        for role in self.params_by_role:
            if role in QUANTIZED_ROLES:
                # MXFP4 converts only the experts (gpt-oss); other formats every large linear layer.
                out[role] = per if (fmt != "mxfp4" or role == "expert") else 2.0
            else:
                out[role] = 2.0
        return out

    def weight_bytes(
        self,
        fmt: str = "native",
        *,
        include_mtp: bool = False,
        include_vision: bool = True,
    ) -> float:
        """Bytes the engine loads for the weights at ``fmt``."""
        rb = self.role_bytes(fmt)
        total = sum(n * rb.get(role, 2.0) for role, n in self.params_by_role.items())
        if include_mtp and self.mtp_params_by_role:
            total += sum(n * rb.get(role, 2.0) for role, n in self.mtp_params_by_role.items())
        if include_vision:
            total += 2.0 * self.vision_params
        return float(total)

    def mtp_weight_bytes(self, fmt: str = "native") -> float:
        rb = self.role_bytes(fmt)
        return float(sum(n * rb.get(role, 2.0) for role, n in self.mtp_params_by_role.items()))

    def expert_union_fraction(self, batch: int) -> float:
        """Fraction of routed experts a decode step at ``batch`` sequences touches (uniform routing)."""
        if not self.experts:
            return 1.0
        n, k = self.experts
        if n <= 0:
            return 1.0
        b = max(int(batch), 1)
        return 1.0 - (1.0 - min(k, n) / n) ** b

    def weight_bytes_streamed(
        self, fmt: str = "native", *, batch: int = 1, tokens: Optional[int] = None
    ) -> float:
        """Weight bytes one forward step reads: every decoder weight except the embedding table (a gather),
        with only the experts the step's tokens route to."""
        rb = self.role_bytes(fmt)
        union = self.expert_union_fraction(tokens if tokens is not None else batch)
        total = 0.0
        for role, n in self.params_by_role.items():
            if role == "embedding":
                continue
            share = union if role == "expert" else 1.0
            total += n * rb.get(role, 2.0) * share
        return float(total)

    # ------------------------------------------------------------------ KV and state

    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        """KV bytes per cached token, whole model, counting every attention layer (sliding layers too: the
        figure vLLM allocates per block)."""
        per_layer = self._kv_bytes_per_token_layer(kv_dtype)
        return float(per_layer * (self.layers.full + self.layers.sliding))

    def _kv_bytes_per_token_layer(self, kv_dtype: str) -> float:
        b = dtype_bytes(kv_dtype)
        if self.layers.mla:
            return (self.kv_lora_rank + self.qk_rope_head_dim) * b
        return 2 * self.num_kv_heads * self.head_dim * b

    def kv_bytes_for_context(self, context: int, kv_dtype: str = "bf16") -> float:
        """KV bytes one sequence of ``context`` tokens holds and a decode step reads (sliding layers bounded
        by the window)."""
        per = self._kv_bytes_per_token_layer(kv_dtype)
        ctx = max(int(context), 0)
        window = self.layers.sliding_window or ctx
        return float(per * (self.layers.full * ctx + self.layers.sliding * min(ctx, window)))

    def state_bytes_per_layer(self, state_dtype: str = "float32") -> float:
        """One recurrent layer's state for one sequence (recurrent at ``state_dtype``, conv at 16-bit)."""
        if not self.state_per_layer:
            return 0.0
        b = dtype_bytes(state_dtype)
        total = 0.0
        count = 0
        for mech, (elems, conv) in self.state_per_layer.items():
            total += elems * b + conv
            count += 1
        return total / max(count, 1)

    def state_page_bytes(self, state_dtype: str = "float32", spec_tokens: int = 0) -> float:
        """One recurrent layer's state page as vLLM sizes it: the recurrent matrix at ``state_dtype`` and the conv
        ring buffer, which speculative decoding lengthens by ``spec_tokens`` positions."""
        if not self.state_per_layer:
            return 0.0
        b = dtype_bytes(state_dtype)
        k = max(int(self.conv_kernel), 2)
        pages = []
        for elems, conv in self.state_per_layer.values():
            pages.append(elems * b + conv * (k - 1 + max(int(spec_tokens), 0)) / (k - 1))
        return max(pages)

    def state_bytes_per_sequence(self, state_dtype: str = "float32") -> float:
        return float(self.layers.recurrent * self.state_bytes_per_layer(state_dtype))

    # ------------------------------------------------------------------ per-step work

    def decode_bytes(
        self,
        *,
        batch: int,
        context: int,
        weight_format: str = "native",
        kv_dtype: str = "bf16",
        state_dtype: str = "float32",
    ) -> float:
        """Bytes one decode step reads and writes: weights once, each sequence's KV, and each sequence's
        recurrent state read and written back."""
        b = max(int(batch), 1)
        return float(
            self.weight_bytes_streamed(weight_format, batch=b)
            + b * self.kv_bytes_for_context(context, kv_dtype)
            + b * 2.0 * self.state_bytes_per_sequence(state_dtype)
        )

    def linear_flops_per_token(self) -> float:
        """2 x the parameters a token multiplies through (lm_head included, embedding gather excluded)."""
        return 2.0 * self.params_active

    def attention_flops_per_token(self, context: int) -> float:
        """Score and value FLOPs for one new token attending to ``context`` cached tokens, plus the recurrent
        update of linear layers."""
        ctx = max(int(context), 1)
        window = self.layers.sliding_window or ctx
        if self.layers.mla:
            width = self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        else:
            width = self.num_heads * self.head_dim
        attn = 4.0 * width * (self.layers.full * ctx + self.layers.sliding * min(ctx, window))
        recurrent = 0.0
        if self.gdn_dims and self.layers.recurrent:
            _, nv, kd, vd = self.gdn_dims
            recurrent = 6.0 * nv * kd * vd * self.layers.recurrent
        return attn + recurrent

    def prefill_flops(self, tokens: int, prefix: int = 0) -> float:
        """FLOPs to prefill ``tokens`` new tokens after ``prefix`` cached ones (causal attention)."""
        n = max(int(tokens), 0)
        if n == 0:
            return 0.0
        linear = self.linear_flops_per_token() * n
        mean_ctx = prefix + n / 2.0
        attention = self.attention_flops_per_token(int(mean_ctx)) * n
        return float(linear + attention)

    # ------------------------------------------------------------------ identity

    def digest(self) -> str:
        return self.config_digest


# ====================================================================== construction helpers


def _native_format(config: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    for scope in (config, _text(config)):
        qc = scope.get("quantization_config") or scope.get("compression_config")
        if isinstance(qc, Mapping) and qc:
            method = str(qc.get("quant_method") or qc.get("quant_algo") or "").lower()
            fmt = _QUANT_METHODS.get(method)
            if method == "compressed-tensors":
                fmt = _compressed_tensors_format(qc)
            if method == "modelopt":
                algo = str(qc.get("quant_algo") or "").lower()
                fmt = "nvfp4" if "fp4" in algo else "fp8"
            if method in ("awq", "gptq") and int(qc.get("bits") or 4) == 8:
                fmt = "int8"
            return fmt or "unknown-quant", dict(qc)
    dtype = str(
        _get(_text(config), "torch_dtype", "dtype", default=None)
        or _get(config, "torch_dtype", "dtype", default="bf16")
    ).lower()
    return ("fp16" if dtype in ("float16", "fp16", "half") else "bf16"), {}


def _compressed_tensors_format(qc: Mapping[str, Any]) -> str:
    for group in (qc.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        bits = int(weights.get("num_bits") or 8)
        kind = str(weights.get("type") or "float")
        if bits == 4:
            return "nvfp4" if kind == "float" else "int4"
        if bits == 8:
            return "fp8" if kind == "float" else "int8"
    return "fp8"


_VISION_PREFIXES = ("visual", "vision", "model.visual", "vision_tower", "vision_model", "model.vision")


def _excluded(qc: Mapping[str, Any]) -> Tuple[str, ...]:
    """The modules a checkpoint leaves unquantized that can keep a decoder role 16-bit.

    Vision-tower entries never apply to the decoder. A module listed together with modules under it names a container
    whose own (non-linear) parameters were skipped, not its projections: llm-compressor lists
    ``layers.N.linear_attn`` beside ``layers.N.linear_attn.in_proj_a`` while ``in_proj_qkv`` under it is packed int4
    (gtech 2026-10-20: a W4A16 27B sized 27.7 GiB instead of 19.6).
    """
    items = []
    for key in ("modules_to_not_convert", "ignore", "ignored_layers", "exclude_modules"):
        for item in qc.get(key) or ():
            items.append(str(item).lower())
    items = [i for i in items if not i.startswith(_VISION_PREFIXES) and ".visual." not in i]
    listed = set(items)
    out = []
    for item in items:
        if any(other.startswith(item + ".") for other in listed):
            continue
        out.append(item)
    return tuple(out)


def _int_bytes(fmt: str, qc: Mapping[str, Any]) -> Optional[float]:
    """Bytes per parameter of a compressed-tensors integer checkpoint: its bits plus a 16-bit scale (and an 8-bit
    zero point when asymmetric) per group."""
    if fmt not in ("int4", "int8") or str(qc.get("quant_method") or "").lower() != "compressed-tensors":
        return None
    for group in (qc.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type") or "float") != "int":
            continue
        bits = int(weights.get("num_bits") or (4 if fmt == "int4" else 8))
        size = weights.get("group_size")
        per_group = 16 + (0 if weights.get("symmetric", True) else 8)
        overhead = per_group / int(size) if size else 0.0  # per-channel: one scale per row, negligible
        return (bits + overhead) / 8
    return None


def _native_role_bytes(
    fmt: str, qc: Mapping[str, Any], roles: Mapping[str, int]
) -> Dict[str, float]:
    if not qc:
        return {role: 2.0 for role in roles}
    per = _int_bytes(fmt, qc) or _FORMAT_BYTES.get(fmt, 2.0)
    excluded = _excluded(qc)

    def kept_16(role: str) -> bool:
        if role in ALWAYS_16:
            return True
        names = {
            "attention": ("self_attn", "attn", "attention", "linear_attn"),
            "dense_ffn": ("mlp", "ffn"),
            "expert": ("experts", "mlp.experts", "block_sparse_moe"),
            "shared_expert": ("shared_expert",),
        }.get(role, ())
        for pattern in excluded:
            leaf = pattern.split(".")[-1].strip("*")
            if role == "expert" and pattern.endswith("mlp") and "router" not in pattern:
                return True
            if leaf and any(leaf == n for n in names):
                return True
        return False

    out = {}
    for role in roles:
        if fmt == "mxfp4":
            out[role] = per if role == "expert" else 2.0
        else:
            out[role] = 2.0 if kept_16(role) else per
    return out


def _layer_counts(config: Mapping[str, Any], text: Mapping[str, Any], n_layers: int) -> LayerCounts:
    kinds = None
    try:
        kinds = layer_kinds(dict(config))
    except Exception:  # noqa: BLE001 - an exotic config must not fail the plan (NFR-5)
        kinds = None
    window = _get(text, "sliding_window", default=None)
    window = int(window) if isinstance(window, (int, float)) and window else None
    mla = bool(_get(text, "kv_lora_rank", default=0))
    if kinds and (kinds["source"] != "uniform" or len(kinds["attn"]) == n_layers):
        # The one layer-kind table decides: a layer holds KV only if it has
        # attention and does not reuse another layer's cache. Cross-attention
        # layers are counted as full (their encoder cache, over-estimated). A
        # uniform stack with a window slides every layer, as the table says (a
        # depth the table cannot read -- `n_layer`, `decoder_layers` -- stays full).
        owns = [
            a for a, shared in zip(kinds["attn"], kinds["kv_shared"]) if a is not None and not shared
        ]
        full = sum(1 for a in owns if a in (ATTN_FULL, ATTN_CROSS))
        sliding = sum(1 for a in owns if a == ATTN_SLIDING)
        recurrent = sum(1 for r in kinds["recurrent"] if r)
        mechs = tuple(kinds.get("mechanisms") or ())
        window = kinds.get("sliding_window") or window
        if sliding == 0:
            window = None
        return LayerCounts(
            full=full,
            sliding=sliding,
            recurrent=recurrent,
            sliding_window=window,
            mechanisms=mechs,
            mla=mla,
        )
    return LayerCounts(full=n_layers, sliding=0, recurrent=0, sliding_window=None, mla=mla)


def _attention_params(
    text: Mapping[str, Any], h: int, nh: int, nkv: int, hd: int
) -> Tuple[int, int]:
    """(large linear params, small params) of one attention layer."""
    if _get(text, "kv_lora_rank", default=0):
        q_lora = int(_get(text, "q_lora_rank", default=0) or 0)
        kv_lora = int(text["kv_lora_rank"])
        nope = int(_get(text, "qk_nope_head_dim", default=128))
        rope = int(_get(text, "qk_rope_head_dim", default=64))
        vh = int(_get(text, "v_head_dim", default=128))
        q = h * q_lora + q_lora * nh * (nope + rope) if q_lora else h * nh * (nope + rope)
        kv = h * (kv_lora + rope) + kv_lora * nh * (nope + vh)
        o = nh * vh * h
        return q + kv + o, (q_lora + kv_lora)
    gate = 2 if text.get("attn_output_gate") else 1
    q = h * nh * hd * gate
    kv = 2 * h * nkv * hd
    o = nh * hd * h
    small = 0
    if text.get("attention_bias"):
        small += (nh * hd * gate) + 2 * nkv * hd + h
    if "qwen3" in str(text.get("model_type") or "") or text.get("qk_norm"):
        small += 2 * hd
    return q + kv + o, small


def _gdn_params(text: Mapping[str, Any], h: int) -> Tuple[int, int, Tuple[int, int, int, int]]:
    nk = int(_get(text, "linear_num_key_heads", default=16))
    nv = int(_get(text, "linear_num_value_heads", default=32))
    kd = int(_get(text, "linear_key_head_dim", default=128))
    vd = int(_get(text, "linear_value_head_dim", default=128))
    k = int(_get(text, "linear_conv_kernel_dim", default=4))
    large = h * (2 * nk * kd + 2 * nv * vd) + nv * vd * h
    small = h * 2 * nv + (2 * nk * kd + nv * vd) * k + 2 * nv + vd
    return large, small, (nk, nv, kd, vd)


def _mamba_params(text: Mapping[str, Any], h: int) -> Tuple[int, int]:
    expand = int(_get(text, "mamba_expand", "expand", default=2))
    d_inner = int(_get(text, "mamba_d_inner", "intermediate_size_mamba", default=expand * h))
    d_state = int(_get(text, "mamba_d_state", "state_size", "ssm_state_size", default=128))
    groups = int(_get(text, "mamba_n_groups", "n_groups", default=1))
    heads = int(_get(text, "mamba_num_heads", "mamba_n_heads", default=max(d_inner // 64, 1)))
    in_proj = h * (2 * d_inner + 2 * groups * d_state + heads)
    out_proj = d_inner * h
    small = (d_inner + 2 * groups * d_state) * 4 + 3 * heads + d_inner
    return in_proj + out_proj, small


def _ffn_params(
    text: Mapping[str, Any], h: int, n_layers: int, ffn_layers: Optional[int]
) -> Dict[str, int]:
    """Dense FFN, routed experts, shared experts and routers, summed over layers.

    ``ffn_layers`` is the layer plan's weight-counting FFN count: Nemotron-H's FFN slots, none
    in a pure SSM, one per shared block in Zamba (whose block runs on many layers)."""
    inter = int(_get(text, "intermediate_size", "ffn_dim", default=4 * h))
    n_experts = int(
        _get(
            text,
            "num_local_experts",
            "n_routed_experts",
            "num_experts",
            "moe_num_experts",
            default=0,
        )
        or 0
    )
    roles = {"dense_ffn": 0, "expert": 0, "shared_expert": 0, "router": 0}
    gated = str(text.get("hidden_act") or "silu") not in ("gelu_new_plain",)
    width = 3 if gated else 2
    if not n_experts:
        layers = n_layers if ffn_layers is None else int(ffn_layers)
        roles["dense_ffn"] = layers * width * h * inter
        return roles
    moe_inter = int(_get(text, "moe_intermediate_size", "expert_intermediate_size", default=inter))
    first_dense = int(_get(text, "first_k_dense_replace", default=0) or 0)
    sparse_step = int(_get(text, "decoder_sparse_step", default=1) or 1)
    mlp_only = set(text.get("mlp_only_layers") or ())
    moe_layers = [
        i
        for i in range(n_layers)
        if i >= first_dense and i not in mlp_only and (i + 1) % sparse_step == 0
    ]
    dense_layers = n_layers - len(moe_layers)
    roles["dense_ffn"] = dense_layers * width * h * inter
    per_expert = width * h * moe_inter
    if str(text.get("model_type")) == "gpt_oss":
        per_expert += 2 * moe_inter + h  # gate_up and down biases
    roles["expert"] = len(moe_layers) * n_experts * per_expert
    shared_inter = int(_get(text, "shared_expert_intermediate_size", default=0) or 0)
    n_shared = int(_get(text, "n_shared_experts", "num_shared_experts", default=0) or 0)
    if shared_inter:
        roles["shared_expert"] = len(moe_layers) * width * h * shared_inter
    elif n_shared:
        roles["shared_expert"] = len(moe_layers) * n_shared * width * h * moe_inter
    roles["router"] = len(moe_layers) * (h * n_experts + n_experts)
    return roles


def _moe_layer_count(text: Mapping[str, Any], n_layers: int) -> int:
    first_dense = int(_get(text, "first_k_dense_replace", default=0) or 0)
    sparse_step = int(_get(text, "decoder_sparse_step", default=1) or 1)
    mlp_only = set(text.get("mlp_only_layers") or ())
    return sum(
        1
        for i in range(n_layers)
        if i >= first_dense and i not in mlp_only and (i + 1) % sparse_step == 0
    )


def _vision_params(config: Mapping[str, Any]) -> int:
    vision = config.get("vision_config")
    if not isinstance(vision, Mapping):
        return 0
    vh = int(_get(vision, "hidden_size", "embed_dim", default=0) or 0)
    if not vh:
        return 0
    depth = int(_get(vision, "depth", "num_hidden_layers", default=24))
    vi = int(_get(vision, "intermediate_size", "mlp_dim", default=4 * vh))
    patch = int(_get(vision, "patch_size", default=14))
    temporal = int(_get(vision, "temporal_patch_size", default=1))
    channels = int(_get(vision, "in_channels", "num_channels", default=3))
    merge = int(_get(vision, "spatial_merge_size", default=1))
    out_h = int(
        _get(vision, "out_hidden_size", default=_get(_text(config), "hidden_size", default=vh))
    )
    pos = int(_get(vision, "num_position_embeddings", default=0) or 0)
    block = 4 * vh * vh + 4 * vh + 2 * vh * vi + vi + vh + 4 * vh
    merged = vh * merge * merge
    merger = merged * merged + merged + merged * out_h + out_h + 2 * vh
    return int(depth * block + channels * temporal * patch * patch * vh + vh + pos * vh + merger)


def _state_table(
    text: Mapping[str, Any], mechanisms: Tuple[str, ...], h: int
) -> Dict[str, Tuple[float, float]]:
    """mech -> (recurrent state elements, conv bytes at 16-bit), one layer, one sequence."""
    table = {}
    for mech in mechanisms:
        try:
            fp32 = _per_layer_bytes(mech, dict(text), h, 2)
        except Exception:  # noqa: BLE001
            continue
        table[mech] = (
            float(fp32["recurrent"]) / 4.0 if fp32["recurrent"] else 0.0,
            float(fp32["conv"]),
        )
    return table


def _mtp_params(
    text: Mapping[str, Any], h: int, layer_attn: int, ffn_roles: Mapping[str, int], n_layers: int
) -> Tuple[int, Dict[str, int]]:
    n = int(_get(text, "mtp_num_hidden_layers", "num_nextn_predict_layers", default=0) or 0)
    if not n:
        return 0, {}
    per_layer_ffn = {role: count // max(n_layers, 1) for role, count in ffn_roles.items()}
    # an MTP block is one decoder layer of the model's own kind plus the embedding/hidden fusion
    if ffn_roles.get("expert"):
        moe_layers = max(1, n_layers - int(_get(text, "first_k_dense_replace", default=0) or 0))
        per_layer_ffn = {
            role: count // moe_layers for role, count in ffn_roles.items() if role != "dense_ffn"
        }
    roles = {
        "attention": n * layer_attn,
        "mixer_small": n * (2 * h * h),  # fc fusing [embedding, hidden]
        "norm": n * 4 * h,
    }
    for role, count in per_layer_ffn.items():
        roles[role] = roles.get(role, 0) + n * count
    return n, roles


def _build(config: Dict[str, Any], family: Optional[str]) -> ModelFacts:
    # An encoder-decoder's KV and decoder weights are its decoder's, under the decoder's own
    # spellings (Whisper's `decoder_layers`, T5's `num_heads`); its encoder is its own stack.
    stacks = encoder_decoder_stacks(config) if isinstance(config, dict) else None
    if stacks is not None:
        config = decoder_view(config)
    text = dict(_text(config))
    flags = set()
    ok, reason = _recognised(config)
    if not ok:
        flags.add("architecture: unrecognised")
    encoder_decoder = bool(config.get("is_encoder_decoder") or text.get("is_encoder_decoder"))
    h = int(_get(text, "hidden_size", "d_model", "n_embd", default=4096) or 4096)
    n_layers = int(
        _get(text, "num_hidden_layers", "n_layer", "num_layers", "decoder_layers", default=32) or 32
    )
    nh = int(
        _get(
            text,
            "num_attention_heads",
            "n_head",
            "decoder_attention_heads",
            default=max(h // 128, 1),
        )
        or 1
    )
    nkv = int(_get(text, "num_key_value_heads", "num_kv_heads", default=nh) or nh)
    hd = int(attention_head_dim(text, nh, h) or (h // max(nh, 1)))
    vocab = int(
        _get(text, "vocab_size", default=_get(config, "vocab_size", default=32000)) or 32000
    )

    # Which layers hold KV is read from the config's own keys through the one layer-kind
    # table, recognised family or not: an unknown layer name is charged full there, so an
    # unrecognised model is still never under-counted, and a recognised-family allowlist
    # never decides a layer's kind. A layer read as recurrent is then charged its mixer's
    # weights and its fixed state, from the same keys, so no weight or state goes unpriced.
    layers = _layer_counts(config, text, n_layers)

    attn_large, attn_small = _attention_params(text, h, nh, nkv, hd)
    roles: Dict[str, int] = {"embedding": vocab * h}
    tied = bool(
        _get(
            text, "tie_word_embeddings", default=_get(config, "tie_word_embeddings", default=False)
        )
    )
    roles["lm_head"] = 0 if tied else vocab * h
    n_attn_layers = layers.full + layers.sliding
    roles["attention"] = n_attn_layers * attn_large
    roles["mixer_small"] = n_attn_layers * attn_small
    gdn_dims = None
    if layers.recurrent:
        if "gdn" in layers.mechanisms or "linear_num_value_heads" in text:
            large, small, gdn_dims = _gdn_params(text, h)
        else:
            large, small = _mamba_params(text, h)
        roles["attention"] += layers.recurrent * large
        roles["mixer_small"] += layers.recurrent * small
    sinks = str(text.get("model_type")) == "gpt_oss" or bool(text.get("attention_sinks"))
    if sinks:
        roles["mixer_small"] += n_attn_layers * nh
    ffn_layers = None
    try:
        plan = resolve_layer_plan(dict(config))
        ffn_layers = plan.get("num_ffn_param_layers") if plan else None
    except Exception:  # noqa: BLE001
        ffn_layers = None
    ffn = _ffn_params(text, h, n_layers, ffn_layers)
    roles.update({k: v for k, v in ffn.items() if v})
    roles["norm"] = (2 * n_layers + 1) * h
    if encoder_decoder:
        enc_layers = int((stacks or {}).get("encoder") or _get(text, "encoder_layers", default=n_layers)
                         or n_layers)
        enc_ffn = int(_get(text, "encoder_ffn_dim", default=4 * h) or 4 * h)
        roles["attention"] += (
            enc_layers * 4 * h * h + n_layers * 4 * h * h
        )  # encoder self + decoder cross
        roles["dense_ffn"] = roles.get("dense_ffn", 0) + enc_layers * 2 * h * enc_ffn
    roles = {k: int(v) for k, v in roles.items() if v}

    n_experts = int(
        _get(
            text,
            "num_local_experts",
            "n_routed_experts",
            "num_experts",
            "moe_num_experts",
            default=0,
        )
        or 0
    )
    k_experts = int(
        _get(text, "num_experts_per_tok", "experts_per_token", "moe_topk", default=0) or 0
    )
    experts = (n_experts, k_experts or 1) if n_experts else None

    mtp_layers, mtp_roles = _mtp_params(text, h, attn_large, ffn, n_layers)
    vision = _vision_params(config)
    native, qc = _native_format(config)
    native_rb = _native_role_bytes(native, qc, {**roles, **mtp_roles})

    attributes = set()
    if mtp_layers:
        attributes.add("mtp")
    if vision:
        attributes.add("multimodal")
    if layers.sliding:
        attributes.add("sliding")
    if sinks:
        attributes.add("sinks")
    if experts:
        attributes.add("moe")
    klass = arch_class(
        encoder_decoder=encoder_decoder,
        recurrent_layers=layers.recurrent,
        mla=layers.mla,
        moe=bool(experts),
    )
    max_ctx = int(
        _get(text, "max_position_embeddings", "max_seq_len", "max_target_positions", default=4096)
        or 4096
    )
    model_type = str(config.get("model_type") or text.get("model_type") or "unknown")
    components = [Component("decoder", int(sum(roles.values())))]
    if vision:
        components.append(Component("vision_encoder", vision))
    if mtp_roles:
        components.append(Component("mtp", int(sum(mtp_roles.values()))))
    digest = (
        "sha256:"
        + hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    )
    return ModelFacts(
        model_type=model_type,
        architectures=tuple(config.get("architectures") or ()),
        family=str(family or model_type),
        recognised=ok,
        flags=frozenset(flags),
        arch_class=klass,
        attributes=frozenset(attributes),
        hidden_size=h,
        vocab_size=vocab,
        num_heads=nh,
        num_kv_heads=nkv,
        head_dim=hd,
        layers=layers,
        params_by_role=roles,
        mtp_params_by_role=mtp_roles,
        vision_params=vision,
        experts=experts,
        mtp_layers=mtp_layers,
        attention_sinks=sinks,
        max_context=max_ctx,
        native_format=native,
        native_role_bytes=native_rb,
        kv_lora_rank=int(_get(text, "kv_lora_rank", default=0) or 0),
        qk_rope_head_dim=int(_get(text, "qk_rope_head_dim", default=0) or 0),
        gdn_dims=gdn_dims,
        state_per_layer=_state_table(text, layers.mechanisms, h),
        components=tuple(components),
        encoder_decoder=encoder_decoder,
        conv_kernel=int(
            _get(text, "linear_conv_kernel_dim", "mamba_d_conv", "conv_kernel", "d_conv", default=4)
            or 4
        ),
        moe_layers=_moe_layer_count(text, n_layers) if experts else 0,
        draft_vocab=int(_get(text, "draft_vocab_size", default=0) or 0),
        shares_embedding=bool(
            _get(config, "share_embedding", "tie_embeddings_with_target", default=False)
        ),
        config_digest=digest,
    )
