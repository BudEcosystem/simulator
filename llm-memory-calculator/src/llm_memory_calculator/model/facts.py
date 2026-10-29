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
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from ..layer_plan import (
    ATTN_FULL,
    ATTN_SLIDING,
    attention_bias,
    attention_head_dim,
    attention_heads,
    attention_output_gate,
    cross_attention_bytes,
    cross_attention_cache,
    decoder_view,
    encoder_decoder_stacks,
    encoder_is_tower,
    encoder_view,
    routed_expert_count,
    routed_expert_width,
    shared_expert_width,
    stated_qk_norm,
    layer_kinds,
    resolve_layer_plan,
    with_class_geometry,
)
from ..stack_params import (
    adapter_params,
    vision_tower_params,
    audio_tower_params,
    encoder_linear_features,
    encoder_params,
    stack_params,
    tower_embedding_params,
    tower_view,
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
    # each recurrent layer's mechanism, in layer order (the layer table's), for its weights
    recurrent_mechanisms: Tuple[str, ...] = ()

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
    # an encoder-decoder's cross-attention cache (layer_plan.cross_attention_cache): every decoder layer's K and V
    # over the encoder's output, fixed per sequence
    cross_attention: Optional[Mapping[str, Any]] = None
    # the sum of (in + out) over every linear map the model's layers hold -- every attention,
    # cross-attention and recurrent mixer's maps, the dense FFN, the routed and shared experts and
    # the routers, an encoder-decoder's encoder --: what vLLM allocates LoRA A and B buffers over
    # ("all linear layers support LoRA"; its fused-MoE LoRA wraps every expert)
    lora_linear_features: int = 0

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
        by the window; an encoder-decoder's cross-attention cache over its encoder's output)."""
        per = self._kv_bytes_per_token_layer(kv_dtype)
        ctx = max(int(context), 0)
        window = self.layers.sliding_window or ctx
        return float(
            per * (self.layers.full * ctx + self.layers.sliding * min(ctx, window))
            + self.cross_bytes_per_sequence(kv_dtype, ctx)
        )

    def cross_bytes_per_sequence(self, kv_dtype: str = "bf16", tokens: Optional[int] = None) -> float:
        """An encoder-decoder's cross-attention cache for one sequence: every decoder layer's K and V over the
        encoder's output -- its stated length (Whisper's 1,500 frames), else ``tokens`` (the caller's context:
        a text encoder reads the prompt). 0 for any other model. It does not grow with the decoded tokens, so
        ``kv_bytes_per_token`` leaves it out."""
        return cross_attention_bytes(self.cross_attention, dtype_bytes(kv_dtype), tokens)

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
        # attention and does not reuse another layer's cache. A cross-attention
        # layer of its own (Mllama) holds no self-attention cache: its K / V over
        # the encoder's output is `cross_attention` (layer_plan.cross_attention_cache),
        # fixed per sequence. A uniform stack with a window slides every layer, as
        # the table says (a depth the table cannot read -- `n_layer`, `decoder_layers`
        # -- stays full).
        owns = [
            a for a, shared in zip(kinds["attn"], kinds["kv_shared"]) if a is not None and not shared
        ]
        full = sum(1 for a in owns if a == ATTN_FULL)
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
            recurrent_mechanisms=tuple(r for r in kinds["recurrent"] if r),
        )
    return LayerCounts(full=n_layers, sliding=0, recurrent=0, sliding_window=None, mla=mla)


def _attention_linears(
    text: Mapping[str, Any], h: int, nh: int, nkv: int, hd: int
) -> List[Tuple[int, int]]:
    """One attention layer's linear maps, ``[(in, out), ...]``: Q (with a query-width output gate),
    K, V and O; MLA's down- and up-projections of the query and the latent, and O."""
    if _get(text, "kv_lora_rank", default=0):
        q_lora = int(_get(text, "q_lora_rank", default=0) or 0)
        kv_lora = int(text["kv_lora_rank"])
        nope = int(_get(text, "qk_nope_head_dim", default=128))
        rope = int(_get(text, "qk_rope_head_dim", default=64))
        vh = int(_get(text, "v_head_dim", default=128))
        q = [(h, q_lora), (q_lora, nh * (nope + rope))] if q_lora else [(h, nh * (nope + rope))]
        return q + [(h, kv_lora + rope), (kv_lora, nh * (nope + vh)), (nh * vh, h)]
    # a query-width output gate: stated, or built by the config class (layer_plan's one reader)
    gate = 2 if attention_output_gate(dict(text)) else 1
    return [(h, nh * hd * gate), (h, nkv * hd), (h, nkv * hd), (nh * hd, h)]


def _features(linears) -> int:
    """The sum of (in + out) over linear maps: the A and B rows a LoRA slot holds for them."""
    return int(sum(i + o for i, o in linears))


def _attention_params(
    text: Mapping[str, Any], h: int, nh: int, nkv: int, hd: int
) -> Tuple[int, int]:
    """(large linear params, small params) of one attention layer."""
    large = int(sum(i * o for i, o in _attention_linears(text, h, nh, nkv, hd)))
    if _get(text, "kv_lora_rank", default=0):
        q_lora = int(_get(text, "q_lora_rank", default=0) or 0)
        return large, (q_lora + int(text["kv_lora_rank"]))
    gate = 2 if attention_output_gate(dict(text)) else 1
    small = 0
    if attention_bias(dict(text)):
        small += (nh * hd * gate) + 2 * nkv * hd + h
    if "qwen3" in str(text.get("model_type") or "") or stated_qk_norm(dict(text)):
        small += 2 * hd
    return large, small


def _gdn_params(text: Mapping[str, Any], h: int) -> Tuple[int, int, Tuple[int, int, int, int]]:
    nk = int(_get(text, "linear_num_key_heads", default=16))
    nv = int(_get(text, "linear_num_value_heads", default=32))
    kd = int(_get(text, "linear_key_head_dim", default=128))
    vd = int(_get(text, "linear_value_head_dim", default=128))
    k = int(_get(text, "linear_conv_kernel_dim", default=4))
    large = h * (2 * nk * kd + 2 * nv * vd) + nv * vd * h
    small = h * 2 * nv + (2 * nk * kd + nv * vd) * k + 2 * nv + vd
    return large, small, (nk, nv, kd, vd)


def _recurrent_params(mech: str, text: Mapping[str, Any], h: int) -> Tuple[int, int]:
    """One recurrent layer's (projections, small parameters): its mechanism's one count
    (``mixer_params``: Mamba-1, Mamba-2, Kimi Delta Attention, lightning attention, short
    convolution, RWKV, mLSTM, RG-LRU), split for the weight roles -- a Mamba layer's in / out (and
    Mamba-1's x / dt) projections are its large matrices, its convolution, decay and norm vectors
    small; another mechanism's whole count is large. Every non-GDN layer had been counted with
    one Mamba-2 formula whatever its mechanism: Kimi-Linear's 20 KDA layers left its facts
    137,675,680 parameters below its checkpoint's 49,122,681,728."""
    from ..mechanism_dims import mamba_dims
    from ..mixer_params import _DISPATCH

    if mech == "mamba2" or mech not in _DISPATCH:
        return _mamba_params(text, h)
    total = int(_DISPATCH[mech](dict(text), h))
    if mech == "mamba1":
        d = mamba_dims(dict(text), h)
        large = (h * 2 * d["d_inner"] + d["d_inner"] * (d["dt_rank"] + 2 * d["d_state"])
                 + d["dt_rank"] * d["d_inner"] + d["d_inner"] * h)
        return large, total - large
    return total, 0


def _recurrent_linears(mech: str, text: Mapping[str, Any], h: int) -> List[Tuple[int, int]]:
    """One recurrent layer's linear maps (``mixer_params``' one statement of each mechanism's),
    read by the same dispatch as its parameters (``_recurrent_params``: Mamba-2's for a mechanism
    with no count of its own)."""
    from ..mixer_params import _DISPATCH, _LINEARS, mamba2_linears

    if mech == "mamba2" or mech not in _DISPATCH or mech not in _LINEARS:
        return list(mamba2_linears(dict(text), h))
    return list(_LINEARS[mech](dict(text), h))


def _mamba_params(text: Mapping[str, Any], h: int) -> Tuple[int, int]:
    """One Mamba-2 layer's (projections, small parameters), from the one count of its mixer
    (``mixer_params.mamba2_params``: dims from ``mechanism_dims.mamba_dims``) -- its depthwise
    convolution's bias and its stated kernel among the small ones. Counted here apart, with a
    kernel of 4 and no conv bias, Mamba-Codestral's facts sat 393,216 parameters below
    transformers' (64 layers x (10,240 bias - a 4,096 norm the formula had added))."""
    from ..mechanism_dims import mamba_dims
    from ..mixer_params import mamba2_params

    d = mamba_dims(dict(text), h)
    d_inner, groups, d_state, heads = d["d_inner"], d["n_groups"], d["d_state"], d["n_heads"]
    large = h * (2 * d_inner + 2 * groups * d_state + heads) + d_inner * h
    return large, int(mamba2_params(dict(text), h)) - large


def _ffn_params(
    text: Mapping[str, Any], h: int, n_layers: int, ffn_layers: Optional[int]
) -> Dict[str, int]:
    """Dense FFN, routed experts, shared experts and routers, summed over layers.

    ``ffn_layers`` is the layer plan's weight-counting FFN count: Nemotron-H's FFN slots, none
    in a pure SSM, one per shared block in Zamba (whose block runs on many layers)."""
    shape = _ffn_shape(text, h, n_layers, ffn_layers)
    width, inter, moe = shape["width"], shape["inter"], shape["moe_layers"]
    roles = {"dense_ffn": shape["dense_layers"] * width * h * inter, "expert": 0,
             "shared_expert": 0, "router": 0}
    if not shape["experts"]:
        return roles
    n_experts, moe_inter = shape["experts"], shape["moe_inter"]
    per_expert = width * h * moe_inter
    if str(text.get("model_type")) == "gpt_oss":
        per_expert += 2 * moe_inter + h  # gate_up and down biases
    roles["expert"] = moe * n_experts * per_expert
    # the shared expert(s) beside the routed ones, under any spelling (layer_plan's one reader),
    # and the gate that scales its output (Linear(h, 1): Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE build
    # one; no config key says so, and a family that builds none is over-counted by h a layer)
    roles["shared_expert"] = moe * (width * h * shape["shared"] + (h if shape["shared"] else 0))
    roles["router"] = moe * (h * n_experts + n_experts)
    return roles


def _ffn_shape(
    text: Mapping[str, Any], h: int, n_layers: int, ffn_layers: Optional[int]
) -> Dict[str, int]:
    """The FFN's shape, as ``_ffn_params`` and ``_ffn_linear_features`` read it: the matrices a
    FFN holds (``width``: 3 gated, 2 not), the dense width and layers, the routed experts, their
    width and the layers that route, and the shared experts' width."""
    inter = int(_get(text, "intermediate_size", "ffn_dim", default=4 * h))
    # the routed experts and their width under any spelling (layer_plan's readers): read from four
    # spellings alone, SmallThinker's 64 `moe_num_primary_experts` of `moe_ffn_hidden_size` were
    # a dense model (6.0B of 21.5B)
    n_experts = routed_expert_count(dict(text))
    gated = str(text.get("hidden_act") or "silu") not in ("gelu_new_plain",)
    shape = {"width": 3 if gated else 2, "inter": inter, "experts": int(n_experts or 0),
             "moe_layers": 0, "moe_inter": 0, "shared": 0}
    if not n_experts:
        shape["dense_layers"] = n_layers if ffn_layers is None else int(ffn_layers)
        return shape
    first_dense = int(_get(text, "first_k_dense_replace", default=0) or 0)
    sparse_step = int(_get(text, "decoder_sparse_step", default=1) or 1)
    mlp_only = set(text.get("mlp_only_layers") or ())
    moe_layers = sum(
        1
        for i in range(n_layers)
        if i >= first_dense and i not in mlp_only and (i + 1) % sparse_step == 0
    )
    shape.update(dense_layers=n_layers - moe_layers, moe_layers=moe_layers,
                 moe_inter=int(routed_expert_width(dict(text)) or inter),
                 shared=int(shared_expert_width(dict(text)) or 0))
    return shape


def _ffn_linear_features(
    text: Mapping[str, Any], h: int, n_layers: int, ffn_layers: Optional[int]
) -> int:
    """(in + out) over the FFN's linear maps, every layer: each of the dense FFN's, every routed
    expert's and the shared experts' gate / up / down (h <-> width), and the router (h -> experts)."""
    shape = _ffn_shape(text, h, n_layers, ffn_layers)
    width, moe = shape["width"], shape["moe_layers"]
    total = shape["dense_layers"] * width * (h + shape["inter"])
    if shape["experts"]:
        total += moe * shape["experts"] * width * (h + shape["moe_inter"])
        if shape["shared"]:
            total += moe * (width * (h + shape["shared"]) + h + 1)  # and its gate, h -> 1
        total += moe * (h + shape["experts"])
    return int(total)


def _moe_layer_count(text: Mapping[str, Any], n_layers: int) -> int:
    first_dense = int(_get(text, "first_k_dense_replace", default=0) or 0)
    sparse_step = int(_get(text, "decoder_sparse_step", default=1) or 1)
    mlp_only = set(text.get("mlp_only_layers") or ())
    return sum(
        1
        for i in range(n_layers)
        if i >= first_dense and i not in mlp_only and (i + 1) % sparse_step == 0
    )


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
    # what a block leaves to its config class is the class's (a hub BLIP-2 `vision_config` that
    # states only its model type is Blip2VisionConfig's 39-layer tower)
    config = with_class_geometry(config) if isinstance(config, dict) else config
    # An encoder-decoder's KV and decoder weights are its decoder's, under the decoder's own
    # spellings (Whisper's `decoder_layers`, T5's `num_heads`); its encoder is its own stack.
    stacks = encoder_decoder_stacks(config) if isinstance(config, dict) else None
    # the cross-attention cache: an encoder-decoder's decoder layers, or a decoder-only stack's own
    # cross-attention layers (Mllama), from the one reader
    cross = cross_attention_cache(config)
    whole = config
    if stacks is not None:
        config = decoder_view(config)
    text = dict(_text(config))
    flags = set()
    ok, reason = _recognised(config)
    if not ok:
        flags.add("architecture: unrecognised")
    # an encoder-decoder is what the stacks reader finds (a decoder depth of its own, both stacks
    # nested), flagged or not: an mT5 config that leaves `is_encoder_decoder` to its class (True)
    # was counted without its encoder and cross-attention
    encoder_decoder = bool(stacks is not None or config.get("is_encoder_decoder")
                           or text.get("is_encoder_decoder"))
    h = int(_get(text, "hidden_size", "d_model", "n_embd", default=4096) or 4096)
    n_layers = int(
        _get(text, "num_hidden_layers", "n_layer", "num_layers", "decoder_layers", default=32) or 32
    )
    # the head count from the one reader that goes with the head width (layer_plan.attention_heads:
    # every spelling, T5's and Pix2Struct's `num_heads` beside their `d_kv` among them), so a block
    # is never sized at its stated width over a default count
    nh = int(
        attention_heads(text)
        or _get(text, "decoder_attention_heads", default=max(h // 128, 1))
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
    # a cross-attention layer of its own carries the attention's projections too (Q / K / V / O)
    own_cross = int(cross["layers"]) if cross and cross.get("own_layers") else 0
    roles["attention"] = (n_attn_layers + own_cross) * attn_large
    roles["mixer_small"] = (n_attn_layers + own_cross) * attn_small
    gdn_dims = None
    if layers.recurrent:
        if "gdn" in layers.mechanisms or "linear_num_value_heads" in text:
            large, small, gdn_dims = _gdn_params(text, h)
            roles["attention"] += layers.recurrent * large
            roles["mixer_small"] += layers.recurrent * small
        else:
            # every other recurrent layer by its own mechanism's count
            for mech in layers.recurrent_mechanisms or ("mamba2",) * layers.recurrent:
                large, small = _recurrent_params(mech, text, h)
                roles["attention"] += large
                roles["mixer_small"] += small
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
        # the decoder's cross-attention (Q / K / V / O at its width), and the encoder at its OWN
        # geometry (stack_params.encoder_params: a hierarchical tower's stages, T5's gated FFN);
        # sized at the decoder's width and its stated stage count, Donut's Swin encoder was 50M
        # of its 74M
        roles["attention"] += n_layers * 4 * h * h
        # with what the encoder holds before its first block (a patch encoder's projection and
        # positions -- Pix2Struct's tower is its encoder --, an audio encoder's convolutions)
        enc = (encoder_params(whole) + tower_embedding_params(encoder_view(whole))
               if isinstance(whole, dict) else 0)
        if not enc:
            enc_layers = int((stacks or {}).get("encoder")
                             or _get(text, "encoder_layers", default=n_layers) or n_layers)
            enc_ffn = int(_get(text, "encoder_ffn_dim", default=4 * h) or 4 * h)
            enc = enc_layers * (4 * h * h + 2 * h * enc_ffn)
        roles["encoder"] = roles.get("encoder", 0) + int(enc)
    roles = {k: int(v) for k, v in roles.items() if v}
    lora_features = _lora_linear_features(
        text, whole, h, nh, nkv, hd, layers, n_attn_layers + own_cross, n_layers, ffn_layers,
        encoder_decoder)

    n_experts = routed_expert_count(dict(text))
    k_experts = _get(text, "num_experts_per_tok", "experts_per_token", "num_experts_per_token",
                     "moe_topk", "moe_num_active_primary_experts", default=0) or 0
    k_experts = int(max(k_experts) if isinstance(k_experts, list) else k_experts)
    experts = (n_experts, k_experts or 1) if n_experts else None

    mtp_layers, mtp_roles = _mtp_params(text, h, attn_large, ffn, n_layers)
    # the tower, and an adapter stack between it and the language model (BLIP-2's Q-Former and
    # its projection: stack_params.adapter_params)
    # (a tower that is the encoder -- Pix2Struct's, BLIP's -- is counted once, as the encoder)
    tower = 0 if encoder_is_tower(whole) else vision_tower_params(config)
    vision = tower + (adapter_params(whole) if isinstance(whole, dict) else 0)
    # an audio encoder beside a decoder-only language model (an encoder-decoder's encoder is
    # counted with its stacks, above)
    if stacks is None:
        vision += audio_tower_params(config)
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
        cross_attention=cross,
        lora_linear_features=lora_features,
    )


def _lora_linear_features(
    text: Mapping[str, Any], whole: Any, h: int, nh: int, nkv: int, hd: int,
    layers: LayerCounts, attn_layers: int, n_layers: int, ffn_layers: Optional[int],
    encoder_decoder: bool,
) -> int:
    """The sum of (in + out) over every linear map the model's layers hold, from the same shapes
    its parameters are counted from: each attention layer's (``_attention_linears``; a
    cross-attention layer of its own among them), each recurrent layer's mixer
    (``_recurrent_linears``, by the same dispatch as its weights), the FFN's -- dense, every routed
    and shared expert, the router (``_ffn_linear_features``) --, an encoder-decoder's
    cross-attention (Q / K / V / O of every decoder layer) and its encoder at its own geometry
    (``stack_params.encoder_linear_features``; a stack the view cannot read at the decoder's
    width). vLLM allocates LoRA A and B buffers over every one of them."""
    total = attn_layers * _features(_attention_linears(text, h, nh, nkv, hd))
    if layers.recurrent:
        if "gdn" in layers.mechanisms or "linear_num_value_heads" in text:
            from ..mixer_params import gdn_linears

            total += layers.recurrent * _features(gdn_linears(dict(text), h))
        else:
            for mech in layers.recurrent_mechanisms or ("mamba2",) * layers.recurrent:
                total += _features(_recurrent_linears(mech, text, h))
    total += _ffn_linear_features(text, h, n_layers, ffn_layers)
    if encoder_decoder:
        # the decoder's cross-attention: Q / O at the decoder's width, K / V reading the encoder's
        # output (projected to the decoder's); at least four h x h maps, as its weights count them
        inner = nh * hd
        total += n_layers * max(8 * h, 4 * h + 4 * inner)
        enc = encoder_linear_features(whole) if isinstance(whole, dict) else 0
        if not enc:
            enc_layers = int(_get(text, "encoder_layers", default=n_layers) or n_layers)
            enc_ffn = int(_get(text, "encoder_ffn_dim", default=4 * h) or 4 * h)
            enc = enc_layers * (8 * h + 2 * (h + enc_ffn))
        total += enc
    return int(total)
