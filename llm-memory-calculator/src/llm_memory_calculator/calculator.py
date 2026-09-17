"""Core memory calculator for LLM models."""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .types import MemoryReport
from .parameter_counter import UniversalParameterCounter
from .config_normalizer import ConfigNormalizer
from .layer_plan import (
    ATTN_FULL,
    ATTN_SLIDING,
    MECH_MAMBA1,
    MECH_MAMBA2,
    resolve_layer_plan,
)
from .state_memory import calculate_recurrent_state_bytes
from .activation_memory import calculate_activation_bytes

# Pure state-space models: no attention layers at all, so no KV cache. `mamba2`
# and `falcon_mamba` were absent here, which sent them to "unknown" and from
# there into the generic-transformer fallback -- hidden_size 768, 12 layers,
# vocab 50257 -- for a 7B model.
SSM_MODEL_TYPES = {
    "mamba",
    "mamba2",
    "falcon_mamba",
    "s4",
    "ssm",
    "state-space",
    "rwkv",
    "rwkv6",
    "rwkv7",
}

# Interleaved or parallel attention + recurrence. Deliberately excludes
# `minimax_m2`, which dropped lightning attention and is genuinely full-attention
# on every layer -- listing it would under-count its KV by the interleave ratio.
HYBRID_MODEL_TYPES = {
    "jamba",
    "hybrid",
    "qwen3_next",
    "qwen3_5",
    "qwen3_5_text",
    "falcon_h1",
    "nemotron_h",
    "granitemoehybrid",
    "zamba",
    "zamba2",
    "bamba",
    "lfm2",
    "minimax_text_01",
    "minimax_m1",
    "kimi_linear",
    "hymba",
    "plamo2",
    "phi4flash",
}


@dataclass(frozen=True)
class EngineKVCapabilities:
    """What the SERVING ENGINE does with a KV layout that is not uniform.

    Architecture says which layers *could* hold a smaller cache. The engine
    decides whether it actually allocates one, and the two answers differ. This
    object carries the engine half, and it is never inferred from the model --
    the same checkpoint served by two engines has two different KV footprints.

    ``heterogeneous_kv_layout``
        The engine can give different attention layers different amounts of KV
        (vLLM calls it the hybrid KV-cache allocator/manager). Only with that
        does a windowed layer inside a stack that ALSO has full-attention layers
        cost less than a full one.

        Default False, and deliberately the pessimistic answer. Measured on an
        H100 vGPU, vLLM serving gpt-oss-20b (12 windowed layers at window 128,
        12 full) allocated 49,254 B/token -- the full 2*24*8*64*2 = 49,152, with
        no window saving at all, because vLLM rewrites every windowed layer's
        spec to a full one when it cannot manage a heterogeneous layout. Taking
        the saving by default halves the prediction, and an under-sized KV cache
        is the dangerous direction: the deployment gets a GPU slice too small
        and either starves for blocks or OOMs. Over-sizing merely costs money.

        A stack whose attention layers ALL share one window (Mistral-style) is
        not heterogeneous -- there is nothing for the engine to reconcile, every
        engine allocates the window -- so it keeps the saving regardless of this
        flag.
    """

    heterogeneous_kv_layout: bool = False

    @classmethod
    def resolve(cls, value: Any) -> "EngineKVCapabilities":
        """Coerce a caller's argument into capabilities, loudly.

        Accepts None (all capabilities off), a bool (shorthand for
        ``heterogeneous_kv_layout``), a mapping of field names, or an instance.
        An unknown key raises instead of being ignored: a caller who passes
        ``{"sliding_window_kv": True}`` and is silently given the default has
        exactly the confident-wrong-number failure this parameter exists to end.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(heterogeneous_kv_layout=value)
        if isinstance(value, Mapping):
            known = {f.name for f in fields(cls)}
            unknown = set(value) - known
            if unknown:
                raise ValueError(
                    f"Unknown engine capability {sorted(unknown)}; "
                    f"known capabilities are {sorted(known)}."
                )
            return cls(**value)
        raise TypeError(
            "engine_capabilities must be None, a bool, a mapping or an "
            f"EngineKVCapabilities, not {type(value).__name__}."
        )


class ModelMemoryCalculator:
    """
    Comprehensive memory calculator for various model architectures during inference.

    Handles all edge cases from production deployments.

    Supported attention mechanisms:
    - MHA (Multi-Head Attention): Standard attention with Q, K, V projections
    - MQA (Multi-Query Attention): Single K, V head shared across all Q heads
    - GQA (Grouped-Query Attention): Groups of Q heads share K, V heads
    - MLA (Multi-head Latent Attention): Compresses K, V into low-rank latent space

    MLA provides superior KV cache compression by projecting keys and values into a
    lower-dimensional latent space before attention computation, reducing memory by 80-90%.
    """

    # Precision to bytes mapping
    PRECISION_BYTES = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "int8": 1,
        "uint8": 1,
        "int4": 0.5,
        "uint4": 0.5,
        # Advanced quantization methods
        "mxfp4": 0.5,  # Microsoft MX-FP4
        "fp4": 0.5,  # 4-bit float
        "nf4": 0.5,  # NormalFloat4 (QLoRA)
        "fp8": 1.0,  # FP8 formats
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
        "awq": 0.5,  # Activation-aware Weight Quantization
        "gptq": 0.5,  # GPTQ 4-bit
        "squeezellm": 0.5,
    }

    def __init__(self):
        """Initialize the calculator."""
        self.param_counter = UniversalParameterCounter()
        self.model_type = None
        self.attention_type = None

    def detect_model_type(self, config: Dict[str, Any]) -> str:
        """Detect the model type from configuration."""
        # Check model_type field
        model_type = config.get("model_type", "").lower()

        # Audio-LLM models (check first)
        if "audio_config" in config:
            # Check for text_config OR top-level LLM parameters (Ultravox style)
            if "text_config" in config:
                return "audio-llm"
            # Ultravox has hidden_size at top level instead of text_config
            if "hidden_size" in config and config.get("hidden_size", 0) > 2000:
                return "audio-llm"
            return "encoder-decoder"

        # Pure audio model types
        if model_type in ["whisper", "speech_to_text", "wav2vec2", "hubert"]:
            return "encoder-decoder"

        # Audio-LLM model types
        if model_type in ["qwen2_audio", "ultravox", "audio-flamingo"]:
            return "audio-llm"

        # Check for encoder-decoder by structure
        if "encoder_layers" in config and "decoder_layers" in config:
            return "encoder-decoder"

        # Multimodal models (vision + text)
        if "vision_config" in config and "text_config" in config:
            return "multimodal"

        # Mamba/SSM models
        if model_type in SSM_MODEL_TYPES:
            return "state-space"

        # Hybrid models
        if model_type in HYBRID_MODEL_TYPES or "mamba_config" in config:
            return "hybrid"

        # Text-to-speech
        if model_type in ["bark", "vall-e", "tortoise-tts", "xtts"]:
            return "text-to-speech"

        # Diffusion models
        if model_type in ["unet", "diffusion", "stable-diffusion", "dit"]:
            return "diffusion"

        # Check for encoder-decoder
        if config.get("is_encoder_decoder", False):
            return "encoder-decoder"

        # Check for hybrid architectures
        if "mamba_config" in config or "attention_layers" in config:
            return "hybrid"

        # Structural fallback: let the config's own layer layout speak. New
        # hybrid families ship roughly quarterly and each invents its own
        # model_type string, so an allow-list alone goes stale by design -- this
        # catches the ones not yet enumerated above.
        plan = resolve_layer_plan(config)
        if plan is not None and plan.get("num_recurrent_layers"):
            return "hybrid" if plan.get("num_attention_layers") else "state-space"

        # Infer from config structure if model_type not specified
        if (
            "num_attention_heads" in config
            and "hidden_size" in config
            and "num_hidden_layers" in config
        ):
            # It's likely a transformer model
            if config.get("is_decoder", True) and not config.get(
                "is_encoder_decoder", False
            ):
                return "decoder-only"
            elif config.get("is_encoder_decoder", False):
                return "encoder-decoder"
            else:
                return "encoder-only"

        return "unknown"

    def detect_attention_type(self, config: Dict[str, Any]) -> Optional[str]:
        """Detect the attention mechanism type."""
        # For multimodal models, check text_config
        if "text_config" in config and isinstance(config["text_config"], dict):
            config = config["text_config"]

        # Check for MLA first (DeepSeek V2/V3 uses this).
        #
        # Test the VALUE, not mere key presence. Models that borrowed DeepSeek's
        # config schema without using MLA ship these keys as explicit null --
        # Hunyuan-A13B declares `kv_lora_rank: null` alongside `use_mla: false`
        # and is ordinary GQA (32 q heads, 8 kv heads). Keying off `key in
        # config` labelled it MLA, then the compressed-dim arithmetic raised
        # `NoneType + NoneType` and the model could not be sized at all.
        if config.get("use_mla") is False:
            pass  # explicit opt-out wins over any leftover schema keys
        elif any(
            config.get(key)
            for key in [
                "q_lora_rank",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "qk_nope_head_dim",
            ]
        ):
            return "mla"  # Multi-Latent Attention
        elif config.get("latent_attention_dim") or config.get("compressed_kv_dim"):
            return "mla"  # Multi-Latent Attention

        num_attention_heads = config.get("num_attention_heads", config.get("n_head", 0))
        num_key_value_heads = config.get(
            "num_key_value_heads", config.get("num_kv_heads", num_attention_heads)
        )

        if num_key_value_heads == 0 or num_attention_heads == 0:
            return None
        elif num_key_value_heads == 1:
            return "mqa"  # Multi-Query Attention
        elif num_key_value_heads < num_attention_heads:
            return "gqa"  # Grouped-Query Attention
        else:
            return "mha"  # Multi-Head Attention

    def _get_activation_function_multiplier(self, config: Dict[str, Any]) -> int:
        """Get FFN matrix count based on activation function."""
        act_fn = (
            config.get("activation_function", config.get("hidden_act", "gelu"))
        ).lower()
        # SwiGLU, SiLU use 3 matrices (gate, up, down), others use 2 (up, down)
        if any(x in act_fn for x in ["swiglu", "silu", "swish"]):
            return 3
        return 2

    def calculate_model_weights(
        self, config: Dict[str, Any], precision: str, respect_weight_tying: bool = True
    ) -> tuple[int, float]:
        """Calculate model weight memory in GB and return (param_count, memory_gb)."""
        # Normalize config for consistent key handling
        config = ConfigNormalizer.normalize_config(config)

        # Get parameter count
        param_count = config.get("num_parameters")
        if not param_count:
            # Check for encoder-decoder / audio models
            model_type = self.detect_model_type(config)
            if model_type in ["encoder-decoder", "audio-llm"]:
                param_count = self._calculate_encoder_decoder_params(config)
            else:
                param_count = self.param_counter.count_parameters(
                    config, respect_weight_tying=respect_weight_tying
                )

        # Check for mixed precision quantization
        if "_quantization" in config and config["_quantization"].get(
            "has_mixed_precision"
        ):
            weight_memory_gb = self._calculate_mixed_precision_memory(
                config, param_count, precision
            )
        else:
            # Standard calculation
            bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)
            weight_memory_gb = (
                param_count * bytes_per_param
            ) / 1e9  # Use decimal GB to match API

        return param_count, weight_memory_gb

    def _calculate_encoder_decoder_params(self, config: Dict[str, Any]) -> int:
        """Calculate total parameters for encoder-decoder models."""
        encoder_params = self.calculate_encoder_params(config)
        decoder_params = self.calculate_decoder_params(config)
        projector_params = self.calculate_projector_params(config)

        return encoder_params + decoder_params + projector_params

    def _calculate_mixed_precision_memory(
        self, config: Dict[str, Any], param_count: int, default_precision: str
    ) -> float:
        """
        Calculate weight memory with mixed precision quantization.

        Handles cases where different modules use different precisions,
        such as quantized experts but fp16 attention/embeddings.

        Args:
            config: Normalized config with _quantization metadata
            param_count: Total parameter count
            default_precision: Default precision for non-quantized modules

        Returns:
            Memory in GB
        """
        quant_info = config.get("_quantization", {})

        # Get quantization settings
        skip_patterns = quant_info.get("skip_modules", [])
        quant_bytes = quant_info.get("bytes_per_param", 0.5)
        default_bytes = self.PRECISION_BYTES.get(default_precision.lower(), 2)

        # Estimate proportion of non-quantized parameters
        skip_ratio = self._estimate_skip_ratio(config, skip_patterns, param_count)

        # Calculate mixed memory
        quantized_params = param_count * (1 - skip_ratio)
        non_quantized_params = param_count * skip_ratio

        memory_gb = (
            quantized_params * quant_bytes + non_quantized_params * default_bytes
        ) / 1e9

        return memory_gb

    def _estimate_skip_ratio(
        self, config: Dict[str, Any], skip_patterns: List[str], total_params: int
    ) -> float:
        """
        Estimate proportion of parameters that are NOT quantized.

        Uses heuristics based on module patterns:
        - "*.self_attn" -> attention layers
        - "*.mlp.router" -> MoE router
        - "*embed*" -> embeddings
        - "*lm_head*" -> output head

        Args:
            config: Model configuration
            skip_patterns: List of module patterns to skip
            total_params: Total parameter count

        Returns:
            Ratio of non-quantized parameters (0.0 to 1.0)
        """
        if not skip_patterns:
            return 0.0

        skip_param_count = 0

        # Extract config values
        vocab_size = config.get("vocab_size", 50000)
        hidden_size = config.get("hidden_size", 4096)
        num_layers = config.get("num_hidden_layers", 24)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim") or (hidden_size // num_heads)

        # Check if embeddings are tied
        tie_embeddings = config.get("tie_word_embeddings", False)

        for pattern in skip_patterns:
            pattern_lower = pattern.lower()

            if "embed" in pattern_lower:
                # Embeddings: input + output (if not tied)
                embedding_params = vocab_size * hidden_size
                if not tie_embeddings:
                    embedding_params *= 2
                skip_param_count += embedding_params

            elif "lm_head" in pattern_lower or (
                "output" in pattern_lower and "layer" not in pattern_lower
            ):
                # Output head (if not already counted in embeddings)
                if tie_embeddings:
                    head_params = vocab_size * hidden_size
                    skip_param_count += head_params

            elif "attn" in pattern_lower or "attention" in pattern_lower:
                # Attention layers: Q, K, V, O projections
                # Q projection: hidden -> num_heads * head_dim
                q_params = hidden_size * num_heads * head_dim
                # K, V projections: hidden -> num_kv_heads * head_dim
                kv_params = 2 * hidden_size * num_kv_heads * head_dim
                # O projection: num_heads * head_dim -> hidden
                o_params = num_heads * head_dim * hidden_size

                attn_params_per_layer = q_params + kv_params + o_params
                skip_param_count += num_layers * attn_params_per_layer

            elif "router" in pattern_lower:
                # MoE router
                num_experts = config.get("n_routed_experts", 1)
                if num_experts > 1:
                    moe_freq = config.get("moe_layer_freq", 1)
                    num_moe_layers = num_layers // moe_freq if moe_freq > 0 else 0
                    router_params = num_moe_layers * hidden_size * num_experts
                    skip_param_count += router_params

        # Calculate ratio, capped at 1.0
        ratio = min(skip_param_count / total_params, 1.0) if total_params > 0 else 0.0

        return ratio

    # ------------------------------------------------------------------ KV cache
    #
    # KV is priced from a per-layer plan over attention KINDS, never from a bare
    # depth. Three kinds exist and each has a different token count behind it:
    #
    #   full     -- caches every token of the sequence.
    #   sliding  -- caches at most `sliding_window` tokens, but only if the
    #               ENGINE allocates per layer; see EngineKVCapabilities.
    #   none     -- linear-attention / SSM / short-conv mixers and MLP-only
    #               layers hold no GROWING cache and contribute exactly zero.
    #               Their fixed-size recurrent state is a separate term
    #               (`calculate_state_memory`), constant in context length.
    #
    # The kinds come from `resolve_layer_plan`, which reads whichever of the
    # eleven per-layer config dialects the model happens to speak, so nothing
    # here keys off a model name or a hardcoded list of models.

    def resolve_kv_layer_plan(
        self,
        config: Dict[str, Any],
        seq_length: int,
        engine_capabilities: Any = None,
    ) -> Dict[str, Any]:
        """Which layers pay KV, over how many tokens each, and why.

        Returns the priced groups plus the layers that pay nothing, so a caller
        can audit the number rather than trust it. Pure geometry -- no batch, no
        precision -- which is what makes it comparable against an engine's own
        "GPU KV cache size" line.
        """
        caps = EngineKVCapabilities.resolve(engine_capabilities)
        normalized = ConfigNormalizer.normalize_config(config)
        # Structural, not by detected model type: a model can be multimodal AND
        # hybrid at once, and `self.model_type` holds only one label.
        if isinstance(normalized.get("text_config"), dict):
            text_config = ConfigNormalizer.normalize_config(normalized["text_config"])
        else:
            text_config = normalized

        depth = int(
            text_config.get("num_hidden_layers")
            or text_config.get("n_layers")
            or text_config.get("num_layers")
            or 0
        )
        notes: List[str] = []

        plan = resolve_layer_plan(config)
        if plan is not None:
            source = "layer_plan:" + plan["dialect"]
            num_full = plan["num_full_layers"]
            num_sliding = plan["num_sliding_layers"]
            depth = depth or len(plan["attn"])
            notes.extend(plan.get("notes") or [])
        else:
            metadata = text_config.get("_layer_metadata") or normalized.get(
                "_layer_metadata"
            )
            if metadata:
                # Whenever the config declares per-layer kinds, they decide --
                # not just when they happen to MIX windowed and full. A stack
                # that declares every layer full-attention while also carrying a
                # `sliding_window` key is the common Qwen2.5/gpt-oss shape, and
                # the uniform fallback below would clamp all of it to a window
                # the layers do not use.
                source = "_layer_metadata"
                num_full = metadata.get("num_full_layers", 0)
                num_sliding = metadata.get("num_sliding_layers", 0)
                if not num_full and not num_sliding:
                    notes.append(
                        "KV is charged as zero: no entry in `layer_types` "
                        f"({metadata.get('num_no_kv_layers', 0)} layers) was "
                        "recognized as an attention layer. If this model does "
                        "hold a cache, its dialect is unknown here."
                    )
            else:
                # Uniform stack: every layer is the same kind, so a declared
                # window applies to all of them -- homogeneous by construction.
                source = "uniform"
                num_sliding = depth if text_config.get("sliding_window") else 0
                num_full = depth - num_sliding

        # `.get(key, default)` returns a stored None, and the window is
        # legitimately None here: a config carrying layer_types can also carry
        # `use_sliding_window: false`, which the normalizer scrubs to None. "No
        # window" means those layers attend to the full sequence.
        window = text_config.get("sliding_window") or None
        windowed_seq = min(seq_length, window) if window else seq_length

        # The engine half of the answer. Windowed layers cost less than full ones
        # only where the engine can hold a different amount of KV per layer; a
        # stack whose attention layers all share one window needs no such
        # ability, so it keeps the saving unconditionally.
        heterogeneous = bool(num_sliding and num_full)
        withheld = heterogeneous and not caps.heterogeneous_kv_layout
        sliding_seq = seq_length if withheld else windowed_seq
        if withheld and windowed_seq < seq_length:
            notes.append(
                f"{num_sliding} windowed layers are charged the full {seq_length} "
                f"tokens, not their {window}-token window: the stack also has "
                f"{num_full} full-attention layers, and an engine that cannot "
                "manage a heterogeneous KV layout rewrites every windowed layer "
                "to a full one (measured: vLLM on gpt-oss-20b). Pass "
                "engine_capabilities={'heterogeneous_kv_layout': True} for an "
                "engine that does take the saving."
            )

        groups: List[Dict[str, Any]] = []
        if num_full > 0:
            groups.append(
                {"kind": ATTN_FULL, "layers": num_full, "effective_seq": seq_length}
            )
        if num_sliding > 0:
            groups.append(
                {
                    "kind": ATTN_SLIDING,
                    "layers": num_sliding,
                    "effective_seq": sliding_seq,
                }
            )

        return {
            "source": source,
            "num_layers": depth,
            "num_full_layers": num_full,
            "num_sliding_layers": num_sliding,
            # Linear-attention / SSM / conv / MLP-only layers. Zero KV, and the
            # count is reported so a 2x error shows up as a layer count rather
            # than as an unexplained factor.
            "num_no_kv_layers": max(depth - num_full - num_sliding, 0),
            "sliding_window": window,
            "groups": groups,
            "heterogeneous_attention": heterogeneous,
            "sliding_window_saving_applied": bool(
                num_sliding and sliding_seq < seq_length
            ),
            "sliding_window_saving_withheld": bool(withheld and windowed_seq < seq_length),
            "engine_capabilities": caps,
            "notes": notes,
        }

    def calculate_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str,
        engine_capabilities: Any = None,
    ) -> float:
        """KV cache in GB, per-layer attention kind and engine behavior aware.

        ``engine_capabilities`` is the serving engine's ability to hold a
        different amount of KV per layer; see EngineKVCapabilities. It defaults
        to the pessimistic answer (no saving on windowed layers inside a mixed
        stack) because under-sizing KV is what OOMs a deployment.
        """
        raw_config = config
        # Normalize config for consistent key handling
        config = ConfigNormalizer.normalize_config(config)

        # Pick the language-model sub-config structurally rather than gating on a
        # detected model type. A model can be multimodal *and* hybrid at once --
        # Qwen3.5 is both -- and `self.model_type` holds only one label, so gating
        # on "multimodal" made the hybrid layout unreachable for exactly the
        # models that need it most.
        if isinstance(config.get("text_config"), dict):
            text_config = ConfigNormalizer.normalize_config(config["text_config"])
        else:
            text_config = config

        attention_type = self.detect_attention_type(config)

        if not attention_type:
            return 0.0

        # Get bytes per element
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)

        kv_plan = self.resolve_kv_layer_plan(
            raw_config, seq_length, engine_capabilities
        )

        # A per-layer plan (canonical hybrid dialects, or the normalizer's
        # sliding/full metadata) prices each kind over its own token count.
        # Charging KV on a Gated-DeltaNet or Mamba layer over-counts by 3.75x
        # (LFM2) to 14x (Nemotron-H); charging a windowed layer only its window
        # on an engine that does not do that UNDER-counts by 2x (gpt-oss).
        if kv_plan["source"] != "uniform":
            if not kv_plan["groups"]:
                return 0.0  # pure SSM: no attention layer, so no KV cache at all
            return sum(
                self._calculate_kv_for_layers(
                    text_config,
                    group["layers"],
                    batch_size,
                    group["effective_seq"],
                    bytes_per_element,
                    attention_type,
                )
                for group in kv_plan["groups"]
            )

        # Uniform stack: one kind for every layer, so a declared window applies
        # to all of them and needs no engine capability to be honored.
        if (
            "sliding_window" in text_config
            and text_config["sliding_window"] is not None
        ):
            seq_length = min(seq_length, text_config["sliding_window"])

        # Calculate based on attention type
        if attention_type == "mla":
            return self._calculate_kv_cache_mla(
                text_config, batch_size, seq_length, bytes_per_element
            )
        elif attention_type == "mqa":
            return self._calculate_kv_cache_mqa(
                text_config, batch_size, seq_length, bytes_per_element
            )
        elif attention_type == "gqa":
            return self._calculate_kv_cache_gqa(
                text_config, batch_size, seq_length, bytes_per_element
            )
        else:  # mha
            return self._calculate_kv_cache_mha(
                text_config, batch_size, seq_length, bytes_per_element
            )

    def kv_cache_breakdown(
        self,
        config: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 2048,
        precision: str = "bf16",
        engine_capabilities: Any = None,
    ) -> Dict[str, Any]:
        """The KV number with its working shown, for auditing a deployment size.

        ``marginal_bytes_per_token`` is what ONE more token costs -- the figure
        that matters once a windowed layer has saturated its window, and the one
        directly comparable to an engine's per-token KV accounting. It differs
        from ``bytes / tokens`` exactly when some layer is clamped.
        """
        plan = self.resolve_kv_layer_plan(config, seq_length, engine_capabilities)
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        normalized = ConfigNormalizer.normalize_config(config)
        if isinstance(normalized.get("text_config"), dict):
            text_config = ConfigNormalizer.normalize_config(normalized["text_config"])
        else:
            text_config = normalized
        attention_type = self.detect_attention_type(normalized)

        total = self.calculate_kv_cache(
            config, batch_size, seq_length, precision, engine_capabilities
        )
        # Only groups still growing at this length add to the marginal cost.
        growing = sum(
            g["layers"] for g in plan["groups"] if g["effective_seq"] >= seq_length
        )
        marginal = (
            self._calculate_kv_for_layers(
                text_config, growing, 1, 1, bytes_per_element, attention_type
            )
            * 1e9
            if growing and attention_type
            else 0.0
        )
        return dict(
            plan,
            attention_type=attention_type,
            bytes=total * 1e9,
            marginal_bytes_per_token=marginal,
        )

    def _calculate_kv_cache_per_layer(
        self,
        config: Dict[str, Any],
        layer_metadata: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
        attention_type: str,
        engine_capabilities: Any = None,
    ) -> float:
        """
        Calculate KV cache for models with per-layer attention types.

        Handles a stack that mixes windowed and full-attention layers. Whether
        the windowed layers actually cost less is an ENGINE property, not a model
        one: see EngineKVCapabilities.

        Args:
            config: Model configuration (the language-model sub-config)
            layer_metadata: num_sliding_layers / num_full_layers
            batch_size: Batch size
            seq_length: Full sequence length
            bytes_per_element: Bytes per KV element
            attention_type: Attention mechanism type
            engine_capabilities: Engine's KV-layout abilities; default pessimistic

        Returns:
            Total KV cache memory in GB
        """
        caps = EngineKVCapabilities.resolve(engine_capabilities)
        num_sliding_layers = layer_metadata.get("num_sliding_layers", 0)
        num_full_layers = layer_metadata.get("num_full_layers", 0)

        # `.get(key, default)` returns a stored None, and the window is legitimately
        # None here: a config carrying layer_types can also carry
        # `use_sliding_window: false`, which the normalizer scrubs to None to stop
        # the global path clamping. "No window" means those layers attend to the
        # full sequence, so fall back to seq_length rather than comparing to None.
        sliding_window = config.get("sliding_window") or seq_length

        total_kv_cache = 0.0

        # Calculate KV cache for sliding attention layers
        if num_sliding_layers > 0:
            heterogeneous = num_full_layers > 0
            if heterogeneous and not caps.heterogeneous_kv_layout:
                effective_seq = seq_length
            else:
                effective_seq = min(seq_length, sliding_window)
            sliding_kv = self._calculate_kv_for_layers(
                config,
                num_sliding_layers,
                batch_size,
                effective_seq,
                bytes_per_element,
                attention_type,
            )
            total_kv_cache += sliding_kv

        # Calculate KV cache for full attention layers
        if num_full_layers > 0:
            full_kv = self._calculate_kv_for_layers(
                config,
                num_full_layers,
                batch_size,
                seq_length,
                bytes_per_element,
                attention_type,
            )
            total_kv_cache += full_kv

        return total_kv_cache


    def _calculate_kv_for_layers(
        self,
        config: Dict[str, Any],
        num_layers: int,
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
        attention_type: str,
    ) -> float:
        """Calculate KV cache for a specific number of layers.

        Args:
            config: Model configuration
            num_layers: Number of layers to calculate for
            batch_size: Batch size
            seq_length: Sequence length for these layers
            bytes_per_element: Bytes per KV element
            attention_type: Attention mechanism type

        Returns:
            KV cache memory in GB
        """
        hidden_size = config.get("hidden_size", config.get("d_model", 768))
        num_attention_heads = config.get(
            "num_attention_heads", config.get("n_head", 12)
        )
        num_key_value_heads = config.get(
            "num_key_value_heads", config.get("num_kv_heads", num_attention_heads)
        )
        head_dim = config.get("head_dim") or (hidden_size // num_attention_heads)

        if attention_type == "mla":
            # One compressed latent per token per layer; no factor of 2.
            kv_lora_rank = config.get("kv_lora_rank") or 512
            qk_rope_head_dim = config.get("qk_rope_head_dim") or 0
            compressed_kv_dim = config.get(
                "compressed_kv_dim", kv_lora_rank + qk_rope_head_dim
            )
            kv_elements = batch_size * num_layers * seq_length * compressed_kv_dim
        elif attention_type == "mqa":
            kv_elements = 2 * batch_size * num_layers * seq_length * head_dim
        elif attention_type == "gqa":
            kv_elements = (
                2
                * batch_size
                * num_layers
                * seq_length
                * num_key_value_heads
                * head_dim
            )
        else:  # mha
            kv_elements = (
                2
                * batch_size
                * num_layers
                * seq_length
                * num_attention_heads
                * head_dim
            )

        return (kv_elements * bytes_per_element) / 1e9

    def _calculate_kv_cache_mha(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
    ) -> float:
        """Calculate KV cache for Multi-Head Attention."""
        num_layers = config.get("num_hidden_layers", config.get("n_layers", 12))
        hidden_size = config.get("hidden_size", config.get("d_model", 768))
        num_attention_heads = config.get(
            "num_attention_heads", config.get("n_head", 12)
        )
        # Read explicit head_dim; models such as Qwen3 decouple it from
        # hidden_size // num_heads (128 vs 64), and deriving it halves the cache.
        head_dim = config.get("head_dim") or (hidden_size // num_attention_heads)

        # Count only attention layers for hybrid models
        if self.model_type == "hybrid":
            num_layers = self._count_attention_layers(config, num_layers)

        # 2 for K and V, one full set of heads each
        kv_elements = (
            2 * batch_size * num_layers * seq_length * num_attention_heads * head_dim
        )
        return (kv_elements * bytes_per_element) / 1e9

    def _calculate_kv_cache_mqa(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
    ) -> float:
        """Calculate KV cache for Multi-Query Attention."""
        num_layers = config.get("num_hidden_layers", config.get("n_layers", 12))
        hidden_size = config.get("hidden_size", config.get("d_model", 768))
        num_attention_heads = config.get(
            "num_attention_heads", config.get("n_head", 12)
        )
        head_dim = config.get("head_dim") or (hidden_size // num_attention_heads)

        # Count only attention layers for hybrid models
        if self.model_type == "hybrid":
            num_layers = self._count_attention_layers(config, num_layers)

        # MQA has single K, V head
        kv_elements = 2 * batch_size * num_layers * seq_length * head_dim
        return (kv_elements * bytes_per_element) / 1e9

    def _calculate_kv_cache_gqa(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
    ) -> float:
        """Calculate KV cache for Grouped-Query Attention."""
        num_layers = config.get("num_hidden_layers", config.get("n_layers", 12))
        hidden_size = config.get("hidden_size", config.get("d_model", 768))
        num_attention_heads = config.get(
            "num_attention_heads", config.get("n_head", 12)
        )
        num_key_value_heads = config.get(
            "num_key_value_heads", config.get("num_kv_heads", num_attention_heads)
        )
        head_dim = config.get("head_dim") or (hidden_size // num_attention_heads)

        # Count only attention layers for hybrid models
        if self.model_type == "hybrid":
            num_layers = self._count_attention_layers(config, num_layers)

        # GQA has num_key_value_heads K, V heads
        kv_elements = (
            2 * batch_size * num_layers * seq_length * num_key_value_heads * head_dim
        )
        return (kv_elements * bytes_per_element) / 1e9

    def _calculate_kv_cache_mla(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
    ) -> float:
        """Calculate KV cache for Multi-Latent Attention (DeepSeek V2/V3)."""
        num_layers = config.get("num_hidden_layers", config.get("n_layers", 12))

        # MLA caches ONE compressed latent per token per layer: the decoupled
        # RoPE key (qk_rope_head_dim) plus the compressed KV latent (kv_lora_rank).
        # No factor of 2 -- K and V are reconstructed from the single stored
        # latent (matches vLLM MLAAttentionSpec: num_kv_heads=1,
        # head_dim=kv_lora_rank+qk_rope_head_dim).
        # `or` rather than a .get default: these ship as explicit null on configs
        # that borrowed DeepSeek's schema without using MLA.
        kv_lora_rank = config.get("kv_lora_rank") or 512  # DeepSeek V3 default
        qk_rope_head_dim = config.get("qk_rope_head_dim") or 0
        compressed_kv_dim = config.get(
            "compressed_kv_dim", kv_lora_rank + qk_rope_head_dim
        )

        # Count only attention layers for hybrid models
        if self.model_type == "hybrid":
            num_layers = self._count_attention_layers(config, num_layers)

        # Latent KV cache is much smaller
        kv_elements = batch_size * num_layers * seq_length * compressed_kv_dim
        return (kv_elements * bytes_per_element) / 1e9

    def _count_attention_layers(self, config: Dict[str, Any], total_layers: int) -> int:
        """Count the number of attention layers in hybrid models."""
        # Check for explicit layer types
        layer_types = config.get("layer_types", [])
        if layer_types:
            return sum(1 for lt in layer_types if "attention" in str(lt).lower())

        # Use attention ratio if specified
        attention_ratio = config.get("attention_ratio", 0.5)
        return int(total_layers * attention_ratio)

    # =========================================================================
    # Encoder-Decoder / Audio Model Support
    # =========================================================================

    def _get_encoder_d_model(self, config: Dict[str, Any]) -> int:
        """Get encoder hidden dimension."""
        # Check for audio_config (Qwen2-Audio, Ultravox style)
        if "audio_config" in config:
            audio_config = config["audio_config"]
            return audio_config.get("d_model", audio_config.get("hidden_size", 1280))
        # Whisper-style config
        return config.get("d_model", config.get("hidden_size", 1280))

    def _get_decoder_d_model(self, config: Dict[str, Any]) -> int:
        """Get decoder hidden dimension."""
        # Check for text_config (Qwen2-Audio style)
        if "text_config" in config:
            text_config = config["text_config"]
            # text_config may be incomplete, infer from intermediate_size if needed
            if "hidden_size" in text_config:
                return text_config["hidden_size"]
            # Infer from intermediate_size (typically 4x or 2.67x hidden_size)
            if "intermediate_size" in text_config:
                # Qwen2 uses ~2.75x ratio: 4096 * 2.6875 ≈ 11008
                return int(text_config["intermediate_size"] / 2.6875)
            return text_config.get("d_model", 4096)
        # Ultravox style: audio_config + top-level LLM params
        if "audio_config" in config and "hidden_size" in config:
            return config["hidden_size"]
        # Whisper-style (shared d_model for encoder/decoder)
        return config.get("d_model", config.get("hidden_size", 1280))

    def _get_encoder_layers(self, config: Dict[str, Any]) -> int:
        """Get number of encoder layers."""
        if "audio_config" in config:
            return config["audio_config"].get("encoder_layers", 32)
        return config.get("encoder_layers", 0)

    def _get_decoder_layers(self, config: Dict[str, Any]) -> int:
        """Get number of decoder layers."""
        if "text_config" in config:
            text_cfg = config["text_config"]
            return text_cfg.get("num_hidden_layers", text_cfg.get("decoder_layers", 32))
        # For encoder-decoder models (Whisper), prefer decoder_layers over num_hidden_layers
        # num_hidden_layers in Whisper config refers to encoder, not decoder
        if "decoder_layers" in config:
            return config["decoder_layers"]
        # For audio-LLM with top-level params (Ultravox), use num_hidden_layers
        if "audio_config" in config and "hidden_size" in config:
            return config.get("num_hidden_layers", 32)
        return config.get("num_hidden_layers", 4)

    def _get_num_attention_heads(self, config: Dict[str, Any]) -> int:
        """Get number of attention heads for decoder."""
        if "text_config" in config:
            text_cfg = config["text_config"]
            if "num_attention_heads" in text_cfg:
                return text_cfg["num_attention_heads"]
            # Infer from intermediate_size / hidden_size ratio for Qwen2 family
            hidden_size = self._get_decoder_d_model(config)
            # Qwen2-7B: 4096 hidden, 28 heads -> head_dim ~146 (non-standard)
            # Common: head_dim = 128 -> num_heads = hidden_size / 128
            return hidden_size // 128  # Default to 128 head_dim
        # Ultravox style: audio_config + top-level LLM params
        if "audio_config" in config and "num_attention_heads" in config:
            return config["num_attention_heads"]
        return config.get(
            "decoder_attention_heads", config.get("num_attention_heads", 20)
        )

    def _get_num_kv_heads(self, config: Dict[str, Any]) -> int:
        """Get number of key-value heads for decoder (for GQA)."""
        if "text_config" in config:
            text_cfg = config["text_config"]
            if "num_key_value_heads" in text_cfg:
                return text_cfg["num_key_value_heads"]
            # Qwen2 family typically uses GQA with 4 KV heads
            if text_cfg.get("model_type", "").lower().startswith("qwen"):
                return 4
            num_heads = self._get_num_attention_heads(config)
            return num_heads
        # Ultravox style: audio_config + top-level LLM params
        if "audio_config" in config and "num_attention_heads" in config:
            num_heads = config["num_attention_heads"]
            return config.get("num_key_value_heads", num_heads)
        num_heads = config.get(
            "decoder_attention_heads", config.get("num_attention_heads", 20)
        )
        return config.get("num_key_value_heads", num_heads)

    def _get_bytes_per_element(self, precision: str) -> float:
        """Get bytes per element for given precision."""
        return self.PRECISION_BYTES.get(precision.lower(), 2)

    def calculate_encoder_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate encoder-only parameters for encoder-decoder models.

        For Whisper-like encoders:
        - Self-attention: 4 × d_model² per layer (Q, K, V, O)
        - FFN: 2 × d_model × ffn_dim per layer
        - LayerNorm: 2 × d_model per layer
        - Conv layers (audio): 2 conv1d layers
        """
        # Get encoder config
        if "audio_config" in config:
            enc_config = config["audio_config"]
        else:
            enc_config = config

        d_model = enc_config.get("d_model", enc_config.get("hidden_size", 1280))
        encoder_layers = enc_config.get("encoder_layers", 32)
        encoder_ffn_dim = enc_config.get("encoder_ffn_dim", d_model * 4)
        num_mel_bins = enc_config.get("num_mel_bins", 128)

        params = 0

        # Audio conv layers (Whisper-style: 2 Conv1D)
        # Conv1: num_mel_bins -> d_model, kernel=3
        # Conv2: d_model -> d_model, kernel=3
        conv1_params = num_mel_bins * d_model * 3 + d_model  # weights + bias
        conv2_params = d_model * d_model * 3 + d_model
        params += conv1_params + conv2_params

        # Position embedding
        max_source_positions = enc_config.get("max_source_positions", 1500)
        params += max_source_positions * d_model

        # Self-attention per layer (Q, K, V, O)
        params += encoder_layers * 4 * d_model * d_model

        # FFN per layer (up + down)
        params += encoder_layers * 2 * d_model * encoder_ffn_dim

        # LayerNorm per layer (2 per layer: pre-attn, pre-ffn) + final
        params += (2 * encoder_layers + 1) * d_model

        return params

    def calculate_decoder_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate decoder-only parameters for encoder-decoder models.

        Includes:
        - Self-attention
        - Cross-attention (for encoder-decoder)
        - FFN
        - LayerNorm
        - Embeddings
        """
        # Use helper methods for consistent extraction
        d_model = self._get_decoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)
        num_heads = self._get_num_attention_heads(config)
        num_kv_heads = self._get_num_kv_heads(config)
        head_dim = d_model // num_heads if num_heads > 0 else 64

        # Determine if this is an audio-LLM (uses projector) vs encoder-decoder (uses cross-attention)
        is_audio_llm = ("text_config" in config) or (
            "audio_config" in config
            and "hidden_size" in config
            and config["hidden_size"] > 2000
        )

        # Get decoder-specific config for other params
        if "text_config" in config:
            dec_config = config["text_config"]
        elif "audio_config" in config and "hidden_size" in config:
            # Ultravox style: top-level params
            dec_config = config
        else:
            dec_config = config

        vocab_size = dec_config.get("vocab_size", config.get("vocab_size", 51866))
        decoder_ffn_dim = dec_config.get(
            "intermediate_size", dec_config.get("decoder_ffn_dim", d_model * 4)
        )

        params = 0

        # Embeddings (input + output, often tied)
        tie_embeddings = dec_config.get("tie_word_embeddings", True)
        params += vocab_size * d_model
        if not tie_embeddings:
            params += vocab_size * d_model

        # Position embeddings (if not RoPE)
        max_positions = dec_config.get(
            "max_position_embeddings", dec_config.get("max_target_positions", 448)
        )
        use_rope = (
            dec_config.get("rope_theta", 0) > 0
            or dec_config.get("rope_scaling") is not None
        )
        if not use_rope and max_positions > 0:
            params += max_positions * d_model

        # Self-attention per layer
        # Q projection: d_model -> num_heads * head_dim
        params += decoder_layers * d_model * (num_heads * head_dim)
        # K, V projections: d_model -> num_kv_heads * head_dim (with GQA)
        params += decoder_layers * 2 * d_model * (num_kv_heads * head_dim)
        # O projection
        params += decoder_layers * (num_heads * head_dim) * d_model

        # Cross-attention per layer (if encoder-decoder, not audio-LLM)
        # Audio-LLM models typically fuse via projector, not cross-attention
        if not is_audio_llm and (
            config.get("is_encoder_decoder") or "encoder_layers" in config
        ):
            encoder_d_model = self._get_encoder_d_model(config)
            # Q from decoder, K/V from encoder
            params += decoder_layers * d_model * d_model  # Q
            params += (
                decoder_layers * 2 * encoder_d_model * d_model
            )  # K, V from encoder
            params += decoder_layers * d_model * d_model  # O

        # FFN per layer
        # Check for SwiGLU
        act_fn = dec_config.get(
            "hidden_act", dec_config.get("activation_function", "gelu")
        ).lower()
        if any(x in act_fn for x in ["swiglu", "silu", "swish"]):
            params += decoder_layers * 3 * d_model * decoder_ffn_dim
        else:
            params += decoder_layers * 2 * d_model * decoder_ffn_dim

        # LayerNorm (3 per layer for encoder-decoder: self-attn, cross-attn, FFN; 2 for LLM)
        norms_per_layer = 3 if not is_audio_llm else 2
        params += (norms_per_layer * decoder_layers + 1) * d_model

        return params

    def calculate_projector_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate projector parameters for audio-LLM models.

        The projector maps audio encoder output to text decoder input space.
        """
        # Check if this is an audio-LLM model
        if "audio_config" not in config:
            return 0
        # Must have text_config OR top-level hidden_size (Ultravox style)
        if "text_config" not in config and "hidden_size" not in config:
            return 0

        audio_d_model = self._get_encoder_d_model(config)
        text_d_model = self._get_decoder_d_model(config)

        # Base linear projection
        params = audio_d_model * text_d_model

        # Check for projector config (could be nested or at top level)
        proj_config = config.get("projector_config", {})

        # Stack factor - check both projector_config and top level
        stack_factor = proj_config.get("stack_factor", config.get("stack_factor", 1))
        if stack_factor > 1:
            params = (audio_d_model * stack_factor) * text_d_model

        # SwiGLU projector has 3x parameters - check both locations
        projector_act = proj_config.get(
            "projector_act", config.get("projector_act", "")
        ).lower()
        if projector_act == "swiglu":
            params *= 3

        # Layer norm in projector - check both locations
        if proj_config.get("projector_ln_mid", config.get("projector_ln_mid", False)):
            params += text_d_model

        return params

    def calculate_encoder_kv_cache(
        self, config: Dict[str, Any], batch_size: int, precision: str = "fp16"
    ) -> float:
        """
        Calculate encoder self-attention KV cache (STATIC after encoding).

        This is computed once when processing audio input and reused
        during all decoder steps.

        Size: 2 × batch × encoder_layers × encoder_seq_len × d_model
        """
        encoder_d_model = self._get_encoder_d_model(config)
        encoder_layers = self._get_encoder_layers(config)

        # Get max source positions (audio sequence length)
        if "audio_config" in config:
            max_source_positions = config["audio_config"].get(
                "max_source_positions", 1500
            )
        else:
            max_source_positions = config.get("max_source_positions", 1500)

        bytes_per_element = self._get_bytes_per_element(precision)

        # 2 (K+V) × batch × layers × seq_len × d_model
        elements = (
            2 * batch_size * encoder_layers * max_source_positions * encoder_d_model
        )
        return (elements * bytes_per_element) / 1e9

    def calculate_decoder_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str = "fp16",
    ) -> float:
        """
        Calculate decoder self-attention KV cache (GROWS during generation).

        This grows by one position per generated token.
        Supports GQA where num_kv_heads < num_attention_heads.

        Size: 2 × batch × decoder_layers × seq_len × num_kv_heads × head_dim
        """
        decoder_d_model = self._get_decoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)
        num_heads = self._get_num_attention_heads(config)
        num_kv_heads = self._get_num_kv_heads(config)
        head_dim = decoder_d_model // num_heads

        bytes_per_element = self._get_bytes_per_element(precision)

        # With GQA: 2 × batch × layers × seq_len × num_kv_heads × head_dim
        elements = (
            2 * batch_size * decoder_layers * seq_length * num_kv_heads * head_dim
        )
        return (elements * bytes_per_element) / 1e9

    def calculate_cross_attention_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        encoder_seq_length: int,
        precision: str = "fp16",
    ) -> float:
        """
        Calculate cross-attention KV cache (STATIC, from encoder output).

        This is the encoder output projected to K,V for each decoder layer.
        It's computed once and reused for all decoder steps.

        Size: 2 × batch × decoder_layers × encoder_seq_len × d_model
        """
        # Cross-attention uses encoder hidden size for K,V
        encoder_d_model = self._get_encoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)

        bytes_per_element = self._get_bytes_per_element(precision)

        # 2 × batch × decoder_layers × encoder_seq_len × encoder_d_model
        elements = (
            2 * batch_size * decoder_layers * encoder_seq_length * encoder_d_model
        )
        return (elements * bytes_per_element) / 1e9

    def calculate_audio_input_memory(
        self,
        num_mel_bins: int = 128,
        audio_frames: int = 3000,
        batch_size: int = 1,
        precision: str = "fp32",
    ) -> float:
        """
        Calculate memory for mel spectrogram input.

        Whisper uses 128 mel bins × ~3000 frames for 30s audio.
        """
        bytes_per_element = self._get_bytes_per_element(precision)
        elements = batch_size * num_mel_bins * audio_frames
        return (elements * bytes_per_element) / 1e9

    def calculate_audio_conv_memory(
        self,
        config: Dict[str, Any],
        audio_frames: int,
        batch_size: int,
        precision: str = "fp16",
    ) -> float:
        """
        Calculate memory for audio conv layer activations.

        Whisper uses 2 Conv1D layers:
        - Conv1: mel_bins -> d_model, stride=1
        - Conv2: d_model -> d_model, stride=2
        """
        encoder_d_model = self._get_encoder_d_model(config)
        bytes_per_element = self._get_bytes_per_element(precision)

        # Conv1 output: batch × d_model × audio_frames
        conv1_elements = batch_size * encoder_d_model * audio_frames
        # Conv2 output: batch × d_model × (audio_frames // 2)
        conv2_elements = batch_size * encoder_d_model * (audio_frames // 2)

        total_elements = conv1_elements + conv2_elements
        return (total_elements * bytes_per_element) / 1e9

    def calculate_activation_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str,
        max_num_batched_tokens: Optional[int] = None,
    ) -> float:
        """Peak inference activation memory in GB.

        Sized from the engine's prefill chunk and one layer's working set. The
        previous model multiplied ``batch_size * seq_length`` by a 10-15x
        retention factor, which is a training shape: it assumed both that a
        forward pass materializes the whole context at once (no engine does --
        they all chunk prefill) and that every layer's activations stay live for
        a backward pass (inference frees them as it goes). For a 5 x 100k
        workload that read 76.8 GB against a real figure under 1 GB.
        """
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        # `calculate_activation_bytes` normalizes internally, including the
        # nested text_config -- no pre-pass needed here.
        return (
            calculate_activation_bytes(
                config,
                batch_size,
                seq_length,
                bytes_per_element,
                max_num_batched_tokens,
            )
            / 1e9
        )

    def calculate_state_memory(
        self, config: Dict[str, Any], batch_size: int, precision: str
    ) -> float:
        """Recurrent + convolutional state for SSM/hybrid models, in GB.

        Constant in sequence length by construction -- that is the property these
        architectures are built around, and the reason this term must not be
        folded into the KV cache.

        No longer gated on ``self.model_type``. That gate returned 0.0 for every
        hybrid whose model_type string was not literally "jamba" or "mamba",
        which is to say all of Qwen3-Next, Qwen3.5/3.6, Nemotron-H,
        granite-4.0-h, Bamba, Zamba2, Falcon-H1, LFM2, Kimi-Linear and MiniMax.
        The two existing tests only passed because they assigned
        ``calc.model_type = "hybrid"`` by hand before calling.
        """
        plan = resolve_layer_plan(config)

        if plan is None and self.model_type in ("state-space", "hybrid"):
            # A config that declares SSM-ness only through a nested `mamba_config`
            # or a bare `attention_ratio`, with no per-layer dialect to resolve.
            # Synthesize a plan so there is still exactly one arithmetic path.
            n = config.get("num_hidden_layers", config.get("n_layers", 12))
            if self.model_type == "hybrid":
                n = int(n * (1 - config.get("attention_ratio", 0.5)))
            mech = MECH_MAMBA2 if "n_groups" in config else MECH_MAMBA1
            plan = {"recurrent": [mech] * n, "num_recurrent_layers": n}

        if plan is None:
            return 0.0

        model_bytes = self.PRECISION_BYTES.get(precision.lower(), 2)
        return (
            calculate_recurrent_state_bytes(config, plan, batch_size, model_bytes) / 1e9
        )

    def calculate_lora_adapter_memory(
        self,
        config: Dict[str, Any],
        lora_config: Any,
        precision: str,
        tensor_parallel: int = 1,
    ) -> float:
        """
        Calculate memory for LoRA adapters using vLLM-style allocation.

        Following vLLM's approach:
        - Pre-allocates memory for max_loras adapters
        - Each LoRA layer has A and B matrices
        - LoRA A: [max_loras, 1, max_lora_rank, input_size]
        - LoRA B: [max_loras, 1, output_size, max_lora_rank]

        Args:
            config: Model configuration
            lora_config: LoRA configuration with max_loras, max_lora_rank, etc.
            precision: Model precision for base model
            tensor_parallel: Tensor parallelism degree

        Returns:
            Total LoRA adapter memory in GB
        """
        if not lora_config or not lora_config.enabled:
            return 0.0

        # For multimodal models, use text_config
        if self.model_type == "multimodal" and "text_config" in config:
            text_config = config["text_config"]
        else:
            text_config = config

        # Get LoRA dtype (defaults to model precision if 'auto')
        lora_dtype = lora_config.lora_dtype
        if lora_dtype == "auto":
            lora_dtype = precision
        bytes_per_element = self.PRECISION_BYTES.get(lora_dtype.lower(), 2)

        # Extract model dimensions
        hidden_size = text_config.get("hidden_size", text_config.get("d_model", 768))
        intermediate_size = text_config.get(
            "intermediate_size", text_config.get("ffn_dim", hidden_size * 4)
        )
        num_layers = text_config.get(
            "num_hidden_layers", text_config.get("n_layers", 12)
        )

        # Count only attention layers for hybrid models
        if self.model_type == "hybrid":
            num_attention_layers = self._count_attention_layers(text_config, num_layers)
        else:
            num_attention_layers = num_layers

        max_loras = lora_config.max_loras
        max_lora_rank = lora_config.max_lora_rank
        target_modules = lora_config.target_modules

        total_elements = 0

        # Calculate memory for each target module type
        for module in target_modules:
            module_lower = module.lower()

            # Attention modules: Q, K, V, O projections
            if any(
                x in module_lower
                for x in ["attn", "attention", "qkv", "query", "key", "value", "out"]
            ):
                # Each attention layer typically has 4 projections: Q, K, V, O
                # Each projection: hidden_size -> hidden_size
                num_projections = 4

                for _ in range(num_projections):
                    # LoRA A: [max_loras, 1, max_lora_rank, input_size]
                    lora_a_elements = max_loras * 1 * max_lora_rank * hidden_size

                    # LoRA B: [max_loras, 1, output_size, max_lora_rank]
                    lora_b_elements = max_loras * 1 * hidden_size * max_lora_rank

                    # Apply tensor parallelism sharding
                    if lora_config.fully_sharded_loras:
                        # Both A and B are sharded
                        lora_a_elements = lora_a_elements / tensor_parallel
                        lora_b_elements = lora_b_elements / tensor_parallel
                    else:
                        # Only B is sharded by default
                        lora_b_elements = lora_b_elements / tensor_parallel

                    total_elements += (
                        lora_a_elements + lora_b_elements
                    ) * num_attention_layers

            # FFN modules: up, down, gate projections
            if any(
                x in module_lower
                for x in ["ffn", "mlp", "feed_forward", "up", "down", "gate"]
            ):
                # Typical FFN has 3 projections for SwiGLU: up, down, gate
                # or 2 for standard: up, down
                act_fn = text_config.get(
                    "activation_function", text_config.get("hidden_act", "gelu")
                ).lower()
                num_ffn_projections = (
                    3 if any(x in act_fn for x in ["swiglu", "silu", "swish"]) else 2
                )

                for i in range(num_ffn_projections):
                    if i < num_ffn_projections - 1:  # up/gate projections
                        input_dim = hidden_size
                        output_dim = intermediate_size
                    else:  # down projection
                        input_dim = intermediate_size
                        output_dim = hidden_size

                    # LoRA A: [max_loras, 1, max_lora_rank, input_size]
                    lora_a_elements = max_loras * 1 * max_lora_rank * input_dim

                    # LoRA B: [max_loras, 1, output_size, max_lora_rank]
                    lora_b_elements = max_loras * 1 * output_dim * max_lora_rank

                    # Apply tensor parallelism sharding
                    if lora_config.fully_sharded_loras:
                        lora_a_elements = lora_a_elements / tensor_parallel
                        lora_b_elements = lora_b_elements / tensor_parallel
                    else:
                        lora_b_elements = lora_b_elements / tensor_parallel

                    total_elements += (lora_a_elements + lora_b_elements) * num_layers

        # Convert to GB
        return (total_elements * bytes_per_element) / 1e9

    # ---------------------------------------------------------------- LoRA prefill scratch
    #
    # ROOT CAUSE (docs/lora-scratch-root-cause.md): the scratch is NOT model physics.
    # vLLM's CPU LoRA path (`lora/ops/torch_ops`, used by PunicaWrapperCPU) indexes the
    # stacked weight tensor with a PER-TOKEN index vector:
    #
    #     selected_loras = lora_b_weights[lora_indices_tensor]   # indices shape (T,)
    #
    # materializing (T, out_features, max_lora_rank) -- one copy of the LoRA matrix per
    # token -- per wrapped module, per forward. Probes on Qwen3-0.6B pinned every axis:
    #
    #   rank 32 vs 64        -> 0.573x   linear in CONFIGURED rank, with a floor
    #   max_loras 2 vs 1     -> +7.7%    NOT proportional to slot count
    #   MALLOC trim forced   -> same     live memory, not allocator retention
    #   OMP 4 vs 12 threads  -> same     no per-thread component
    #
    # It is CPU-ONLY: punica_gpu uses Triton grouped-GEMM kernels that never copy.
    #
    # TOKENS ARE LINEAR, NOT CONVEX. An earlier revision carried a "knee at 4273 with a
    # 2.75x steeper slope above it" from a cgroup-`anon` campaign. Re-measured against
    # LoRA-off controls using cgroup memory.peak, that shape is wrong -- and wrong in
    # the expensive direction:
    #
    #   Qwen3-0.6B, rank 64:  T=4273 -> 6.10 GiB     T=11183 -> 14.81 GiB
    #                         2.43x the memory for 2.62x the tokens: SUB-linear.
    #   The convex shape predicted 23.60 GiB at T=11183 -- 1.59x the measurement.
    #
    # The mechanism explains it exactly. Normalising by one live copy of the largest
    # wrapped slice (T * S_max * rank * 2 B) gives a CONSTANT live-copy count:
    #
    #   0.6B T=4273 -> C=3.44        0.6B T=11183 -> C=3.44
    #
    # Constant C across a 2.6x token range is what "one copy per token, a few live at
    # once" predicts. The convex shape was an artifact of the older methodology, and it
    # cost a real deployment: a Qwen3-4B plan was sized at 99.98 GB of scratch where the
    # linear form gives 54.6 GB, and the inflated figure made budsim report "no valid
    # configuration" for a model that fits.
    #
    # So: scale each measured anchor LINEARLY in tokens, and never re-introduce a knee
    # without measuring one.
    #
    # Anchors are control-subtracted cgroup memory.peak deltas (LoRA-on pod minus a
    # LoRA-off pod, same node, same args, generous limits, oom_kills=0), at rank 64,
    # max_loras=1, TP=1. Stored in MiB exactly as measured:
    #
    #   arch                    hidden   T      LoRA    control   scratch
    #   Qwen3ForCausalLM          1024   4273   11343    5095      6248 MiB
    #   Qwen3ForCausalLM          1024  11183   23418    8252     15166 MiB
    #   Qwen3ForCausalLM          2560   2048   28146   12319     15827 MiB
    #   Qwen3ForCausalLM          2560   4273   37128   12478     24650 MiB
    #   LlamaForCausalLM          4096   4273   54574   33872     20702 MiB
    #   Gemma4Unified             3840   2048   56959   48122      8837 MiB
    #   Qwen3_5ForCondGen         2560   4273   18362   18148       214 MiB  (WARMUP_SKIP)
    LORA_SCRATCH_ANCHORS_MIB = {
        "Qwen3ForCausalLM": {
            1024: [(4273, 6248), (11183, 15166)],
            2560: [(2048, 15827), (4273, 24650)],
        },
        "LlamaForCausalLM": {4096: [(4273, 20702)]},
        "Gemma4UnifiedForConditionalGeneration": {3840: [(2048, 8837)]},
        "Qwen3_5ForConditionalGeneration": {2560: [(4273, 214)]},
    }
    # `architectures` is absent from some configs; these `model_type` values were read
    # from the same checkpoints the anchors were measured on, so the alias is verified.
    LORA_SCRATCH_MODEL_TYPE_ALIAS = {
        "qwen3": "Qwen3ForCausalLM",
        "llama": "LlamaForCausalLM",
        "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
    }
    # Architectures whose ~zero measurement is suspected to be warmup SKIPPING the copy
    # path (GDN-hybrid execution), not the path being cheap. A real adapter request at
    # serving time may still take it -- a pod budgeted from the near-zero anchor would
    # then OOM AFTER passing warmup. The anchor is used (it is what was measured) but
    # the caller is warned every time.
    LORA_SCRATCH_WARMUP_SKIP_SUSPECTED = {"Qwen3_5ForConditionalGeneration"}
    # Live-copy count for architectures with no anchor. Measured C (scratch over one
    # live copy of the largest slice) is 3.90 / 4.86 / 2.77 / 2.30 on the four dense
    # anchors -- max 4.86. The envelope sits above all of them because an unmeasured
    # architecture must err high: over-reserving wastes memory on a machine that has it,
    # under-reserving OOMKills at warmup and destroys the evidence.
    LORA_SCRATCH_LIVE_COPY_ENVELOPE = 6.0
    # Beyond the largest measured token budget the linear form is extrapolating. The one
    # extrapolation actually checked (0.6B, 4273 -> 11183, a 2.6x reach) came in 7.8%
    # HIGH against a proportional scale, so the shape is already mildly conservative;
    # 10% covers the residual without the 20% the convex shape needed.
    LORA_SCRATCH_EXTRAPOLATION_MARGIN = 1.10
    # RANK 64 IS THE ONLY VALIDATED RANK. Linear with a floor below it (measured: rank
    # 32 -> 0.573x, not 0.5x); plain linear above, where the floor form would
    # under-predict and rank >64 is separately flagged UNVALIDATED. An earlier revision
    # claimed rank^1.29 from a cross-campaign comparison; same-series data puts rank 64
    # and 256 within 4.5% of linear, so that claim is withdrawn rather than restated.
    LORA_SCRATCH_CALIBRATED_RANK = 64
    LORA_SCRATCH_RANK_SLOPE = 0.855
    LORA_SCRATCH_RANK_FLOOR = 0.145
    # Each adapter slot past the first cost +7.7% measured at max_loras=2; budget 10%.
    LORA_SCRATCH_EXTRA_LORA_FACTOR = 0.10
    # Device strings that use the CPU torch_ops path and therefore pay this term.
    LORA_SCRATCH_CPU_DEVICES = {"cpu", "cpu_high"}
    # Device strings with dedicated non-copying kernels: the term is zero.
    LORA_SCRATCH_ZERO_DEVICES = {"cuda", "gpu", "rocm"}

    @staticmethod
    def _lora_scratch_from_anchors(anchors, tokens: int) -> float:
        """Bytes at ``tokens``, scaling the measured anchors LINEARLY.

        Two or more anchors give a straight line through them (base + slope), which
        separates the token-proportional copies from the fixed overhead. One anchor can
        only be scaled proportionally, which is the conservative reading -- it
        attributes all of the measurement to the token term.
        """
        pts = sorted(anchors)
        if len(pts) >= 2:
            (t1, m1), (t2, m2) = pts[0], pts[-1]
            slope = (m2 - m1) / (t2 - t1)
            base = m1 - slope * t1
            if base < 0:
                return m2 * tokens / t2 * 1024**2
            return max(0.0, base + slope * tokens) * 1024**2
        t0, m0 = pts[0]
        return m0 * tokens / t0 * 1024**2

    def _lora_rank_factor(self, rank: int) -> float:
        """Scratch at ``rank`` relative to the rank-64 anchors."""
        ref = self.LORA_SCRATCH_CALIBRATED_RANK
        if rank >= ref:
            return rank / ref
        return self.LORA_SCRATCH_RANK_SLOPE * rank / ref + self.LORA_SCRATCH_RANK_FLOOR

    @staticmethod
    def _lora_largest_slice(core: Dict[str, Any]) -> int:
        """out_features of the widest LoRA-wrapped slice, from the config.

        gate/up (intermediate_size) dominates on every measured model, but a model with
        unusually wide attention could flip that, so take the max over the candidates
        rather than assuming.
        """
        hidden = int(core.get("hidden_size", core.get("d_model", 0)) or 0)
        heads = int(core.get("num_attention_heads", 0) or 0)
        head_dim = int(core.get("head_dim", 0) or 0)
        if not head_dim and heads and hidden:
            head_dim = hidden // heads
        kv_heads = int(core.get("num_key_value_heads", heads) or 0)
        return max(
            int(core.get("intermediate_size", 0) or 0),
            int(core.get("moe_intermediate_size", 0) or 0),
            heads * head_dim,
            2 * kv_heads * head_dim,
            hidden,
        )

    def calculate_lora_prefill_scratch(
        self,
        config: Dict[str, Any],
        lora_config: Any,
        batched_tokens: int,
        tensor_parallel: int = 1,
        target_device: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """Peak transient of vLLM's CPU LoRA path, in GB, plus provenance notes.

        This is the term that dominates a LoRA-enabled CPU serving pod. It is separate
        from :meth:`calculate_lora_adapter_memory`, the persistent A/B storage --
        reporting them as one number is how a 16 GB term stayed invisible behind a
        0.3 GB one.

        ``batched_tokens`` is the engine's ``max_num_batched_tokens`` (tokens in one
        forward pass), NOT the context length. ``target_device`` decides whether the
        term exists at all: it is an artifact of the CPU torch_ops implementation and
        CUDA's Triton kernels do not pay it. ``None`` is treated as CPU so callers that
        do not know their device stay conservative.

        Returns ``(gb, notes)``. ``notes`` is non-empty whenever any input leaves the
        measured envelope; callers must surface them rather than drop them.
        """
        notes: List[str] = []
        if not lora_config or not getattr(lora_config, "enabled", False):
            return 0.0, notes

        device = (target_device or "").strip().lower()
        if device in self.LORA_SCRATCH_ZERO_DEVICES:
            notes.append(
                f"LoRA prefill scratch is 0 on '{device}': the term is an artifact of "
                f"vLLM's CPU torch_ops (per-token copies of the stacked LoRA weights); "
                f"accelerator kernels are grouped GEMMs and do not materialize them."
            )
            return 0.0, notes
        if device and device not in self.LORA_SCRATCH_CPU_DEVICES:
            notes.append(
                f"LoRA prefill scratch is UNVALIDATED on device '{device}'; the "
                f"CPU-derived term is applied conservatively. If this device has "
                f"dedicated (non-copying) LoRA kernels the reservation is phantom."
            )

        rank = int(getattr(lora_config, "max_lora_rank", 0) or 0)
        if rank <= 0:
            notes.append("LoRA prefill scratch not sized: max_lora_rank is unset")
            return 0.0, notes
        max_loras = int(getattr(lora_config, "max_loras", 1) or 1)
        tokens = max(int(batched_tokens or 0), 1)

        core = (
            config.get("text_config")
            if isinstance(config.get("text_config"), dict)
            else config
        )
        hidden = int(core.get("hidden_size", core.get("d_model", 0)) or 0)

        arch = (config.get("architectures") or [None])[0]
        if arch not in self.LORA_SCRATCH_ANCHORS_MIB:
            aliased = self.LORA_SCRATCH_MODEL_TYPE_ALIAS.get(config.get("model_type"))
            if aliased:
                arch = aliased

        if rank > self.LORA_SCRATCH_CALIBRATED_RANK:
            notes.append(
                f"LoRA prefill scratch is UNVALIDATED at rank {rank}: only rank "
                f"{self.LORA_SCRATCH_CALIBRATED_RANK} is characterised. A rank-256 pod at 4273 "
                f"tokens OOMKilled at a 55 GiB limit where linear-in-rank predicts 20.6 GiB, so "
                f"treat this as a LOWER BOUND, not an estimate."
            )

        anchors = (self.LORA_SCRATCH_ANCHORS_MIB.get(arch) or {}).get(hidden)
        if anchors:
            scratch = self._lora_scratch_from_anchors(anchors, tokens)
            t_max = max(t for t, _ in anchors)
            if tokens > t_max:
                scratch *= self.LORA_SCRATCH_EXTRAPOLATION_MARGIN
                notes.append(
                    f"LoRA prefill scratch is extrapolated to {tokens} batched tokens "
                    f"(measured to {t_max} for this model); the term is linear in tokens "
                    f"(measured, not assumed) and a "
                    f"{self.LORA_SCRATCH_EXTRAPOLATION_MARGIN:.0%} margin is applied."
                )
            if arch in self.LORA_SCRATCH_WARMUP_SKIP_SUSPECTED:
                notes.append(
                    f"LoRA prefill scratch for '{arch}' is budgeted from a near-zero WARMUP "
                    f"measurement, and warmup is suspected to SKIP the CPU copy path for this "
                    f"hybrid architecture rather than the path being cheap. A real adapter "
                    f"request at serving time may still take it -- if so the pod OOMs after "
                    f"passing warmup. Validate with a real adapter before relying on this."
                )
        else:
            slice_out = self._lora_largest_slice(core)
            if slice_out <= 0:
                notes.append(
                    "LoRA prefill scratch not sized: the config exposes no usable module "
                    "dimensions (no intermediate_size, attention dims, or hidden_size)."
                )
                return 0.0, notes
            scratch = (
                self.LORA_SCRATCH_LIVE_COPY_ENVELOPE
                * tokens
                * self.LORA_SCRATCH_CALIBRATED_RANK
                * slice_out
                * 2
            )
            known = sorted((self.LORA_SCRATCH_ANCHORS_MIB.get(arch) or {}))
            where = (
                f"architecture '{arch}'"
                if not known
                else f"'{arch}' at hidden_size {hidden} (measured: {known})"
            )
            notes.append(
                f"LoRA prefill scratch is UNMEASURED for {where}. Sized from the mechanism: "
                f"{self.LORA_SCRATCH_LIVE_COPY_ENVELOPE:g} live copies x {tokens} tokens x rank "
                f"{self.LORA_SCRATCH_CALIBRATED_RANK} x {slice_out} (largest wrapped slice) x 2 B. "
                f"The envelope sits above every measured model (C = 2.30-4.86) and errs high on "
                f"purpose -- measure this model and add an anchor to remove the margin."
            )

        scratch *= self._lora_rank_factor(rank)

        if max_loras > 1:
            scratch *= 1 + self.LORA_SCRATCH_EXTRA_LORA_FACTOR * (max_loras - 1)
            if max_loras > 2:
                notes.append(
                    f"LoRA prefill scratch at max_loras={max_loras} is extrapolated: only 1 and "
                    f"2 slots are measured (+7.7% for the second; 10% per extra slot budgeted)."
                )

        # The copies are per-rank working memory: column-parallel slices shard their
        # out_features across TP, so the term divides like the rest of the forward.
        scratch = scratch / max(int(tensor_parallel or 1), 1)
        return scratch / 1e9, notes

    # ------------------------------------------------------------- sampler / logits
    #
    # vLLM's sampler keeps logits + log_softmax + sorted values + an int64 index
    # (two 4-byte words) live per in-flight sequence. The width is the VOCABULARY,
    # not the hidden size, so a small-hidden/large-vocab model carries a real
    # per-sequence cost that a weights+KV formula never sees: Qwen3-0.6B has a
    # 151936 vocabulary against a hidden size of 1024.
    #
    # This lived in budcluster, which had to re-open the model's config.json to get
    # the vocabulary -- deployment code reaching into a checkpoint for model
    # geometry. It is model dims x workload concurrency, exactly the shape of KV and
    # activation, so it belongs with the physics.
    SAMPLER_LOGITS_DTYPE_BYTES = 4
    SAMPLER_LOGITS_COPIES = 5

    def calculate_sampler_logits(
        self, config: Dict[str, Any], max_num_seqs: Optional[int]
    ) -> Tuple[float, List[str]]:
        """Sampler working set in bytes for one rank, plus notes.

        ``max_num_seqs`` is the engine's concurrent-sequence ceiling. Returns 0 with
        a note when the vocabulary is unknown -- at concurrency 10 the term is only
        ~29 MiB, too small to refuse sizing over, but silently dropping it is how
        terms go missing.
        """
        notes: List[str] = []
        core = (
            config.get("text_config")
            if isinstance(config.get("text_config"), dict)
            else config
        )
        vocab = int(core.get("vocab_size") or config.get("vocab_size") or 0)
        seqs = max(int(max_num_seqs or 0), 0)
        if not vocab or not seqs:
            if not vocab:
                notes.append(
                    "Sampler logits not sized: the config declares no vocab_size. The "
                    "term is small (~29 MiB at concurrency 10) but is omitted, not covered."
                )
            return 0.0, notes
        return (
            float(seqs * vocab * self.SAMPLER_LOGITS_COPIES * self.SAMPLER_LOGITS_DTYPE_BYTES),
            notes,
        )

    def calculate_total_memory(
        self,
        config: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 2048,
        precision: str = "fp16",
        tensor_parallel: int = 1,
        framework_overhead: float = 1.2,
        include_gradients: bool = False,
        decode_length: Optional[int] = None,
        num_images: Optional[int] = None,
        image_resolution: int = 1024,
        lora_config: Optional[Any] = None,
        respect_weight_tying: bool = True,
        encoder_seq_length: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        target_device: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        engine_capabilities: Any = None,
    ) -> MemoryReport:
        """
        Calculate total memory requirements for model inference.

        Args:
            config: Model configuration dictionary
            batch_size: Batch size for inference
            seq_length: Maximum sequence length (context + generation) or decoder output length
            precision: Model precision (fp32, fp16, bf16, int8, int4)
            tensor_parallel: Tensor parallelism degree
            framework_overhead: Multiplicative overhead for framework/kernel memory (default 1.2)
            include_gradients: Include gradient memory (for training)
            decode_length: Length of tokens to generate (defaults to seq_length)
            num_images: Number of images for multimodal models
            image_resolution: Image resolution for vision models
            lora_config: Optional LoRA configuration for adapter memory calculation
            respect_weight_tying: Whether to respect tie_word_embeddings config (default True)
            encoder_seq_length: Encoder sequence length for encoder-decoder models (audio/speech)
            engine_capabilities: What the serving ENGINE does with a non-uniform KV
                layout (EngineKVCapabilities, a mapping of its fields, or a bool).
                Defaults to the pessimistic answer: windowed layers inside a stack
                that also has full-attention layers are charged the full context,
                because under-sizing the KV cache is what OOMs a deployment.

        Returns:
            MemoryReport with detailed breakdown
        """
        # Detect model and attention types
        self.model_type = self.detect_model_type(config)
        self.attention_type = self.detect_attention_type(config)

        # Calculate model weights
        param_count, weight_memory = self.calculate_model_weights(
            config, precision, respect_weight_tying
        )

        # Divide weights by tensor parallelism
        weight_memory = weight_memory / tensor_parallel

        # Calculate KV cache based on model type
        encoder_kv_cache = 0.0
        decoder_kv_cache = 0.0
        cross_attn_kv_cache = 0.0
        kv_notes: List[str] = []

        if self.model_type in ["encoder-decoder", "audio-llm"]:
            # Get encoder sequence length (default from config or parameter)
            enc_seq_len = encoder_seq_length
            if enc_seq_len is None:
                if "audio_config" in config:
                    enc_seq_len = config["audio_config"].get(
                        "max_source_positions", 1500
                    )
                else:
                    enc_seq_len = config.get("max_source_positions", 1500)

            # Encoder self-attention KV (static)
            encoder_kv_cache = self.calculate_encoder_kv_cache(
                config, batch_size, precision
            )

            # Decoder self-attention KV (grows with output)
            decoder_kv_cache = self.calculate_decoder_kv_cache(
                config, batch_size, seq_length, precision
            )

            # Cross-attention KV (static, based on encoder output)
            cross_attn_kv_cache = self.calculate_cross_attention_kv_cache(
                config, batch_size, enc_seq_len, precision
            )

            kv_cache = encoder_kv_cache + decoder_kv_cache + cross_attn_kv_cache
        else:
            # Standard decoder-only model
            kv_cache = self.calculate_kv_cache(
                config, batch_size, seq_length, precision, engine_capabilities
            )
            # Say out loud when the KV number rests on an engine assumption or on
            # an approximated layer plan. A silent 2x is how a pod gets sized
            # against a context it cannot hold.
            kv_notes = self.resolve_kv_layer_plan(
                config, seq_length, engine_capabilities
            )["notes"]

        # Shard the KV cache across ranks. Each rank holds
        # max(1, num_kv_heads // tp) heads: vLLM replicates KV heads when there
        # are fewer of them than ranks (at least one per rank), so KV stops
        # shrinking once tp exceeds num_kv_heads -- a flat divide-by-tp
        # under-counts every rank in that regime. MLA caches a single shared
        # latent that is replicated on every rank, so it does not shard at all.
        if self.attention_type == "mla":
            pass  # replicated per rank; no division
        else:
            kv_cfg = config.get("text_config", config)
            if not isinstance(kv_cfg, dict):
                kv_cfg = config
            num_kv_heads = (
                kv_cfg.get(
                    "num_key_value_heads",
                    kv_cfg.get(
                        "num_kv_heads",
                        kv_cfg.get("num_attention_heads", tensor_parallel),
                    ),
                )
                or tensor_parallel
            )
            kv_heads_per_rank = max(1, num_kv_heads // tensor_parallel)
            kv_cache = kv_cache * kv_heads_per_rank / num_kv_heads

        # Calculate activations. `max_num_batched_tokens` is the engine's prefill
        # chunk, so it bounds the tokens in flight and therefore the peak.
        activations = self.calculate_activation_memory(
            config, batch_size, seq_length, precision, max_num_batched_tokens
        )

        # Calculate state memory (for SSM models). Mamba shards its inner
        # projection across ranks, so the recurrent state shards with tp like
        # weights and KV; keep it per-rank for a consistent report.
        state_memory = self.calculate_state_memory(config, batch_size, precision)
        state_memory = state_memory / tensor_parallel

        # Calculate LoRA adapter memory (if enabled)
        lora_memory = 0.0
        lora_scratch = 0.0
        notes: List[str] = list(kv_notes)
        if lora_config:
            lora_memory = self.calculate_lora_adapter_memory(
                config, lora_config, precision, tensor_parallel
            )
            # The Punica prefill transient is sized by the tokens in ONE forward
            # pass. With chunked prefill that is max_num_batched_tokens, which can
            # be well below the context; default to seq_length so a caller that
            # does not know its engine's batch behaves exactly as before.
            batched = max_num_batched_tokens if max_num_batched_tokens else seq_length
            lora_scratch, scratch_notes = self.calculate_lora_prefill_scratch(
                config, lora_config, batched, tensor_parallel, target_device=target_device
            )
            notes.extend(scratch_notes)

        # Sampler working set: model vocabulary x concurrent sequences. Zero for
        # pooling/embedding models, which never sample.
        sampler_logits = 0.0
        if max_num_seqs:
            sampler_logits, sampler_notes = self.calculate_sampler_logits(config, max_num_seqs)
            notes.extend(sampler_notes)

        # Calculate image memory (for multimodal models)
        image_memory = 0.0
        if num_images and num_images > 0:
            # Estimate based on common vision encoder sizes
            patches_per_image = (image_resolution // 16) ** 2  # Assuming 16x16 patches
            hidden_size = config.get("vision_config", {}).get(
                "hidden_size", config.get("hidden_size", 768)
            )
            bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
            image_memory = (
                num_images * patches_per_image * hidden_size * bytes_per_value
            ) / 1e9

        # Calculate audio input memory (for encoder-decoder/audio models)
        audio_memory = 0.0
        if self.model_type in ["encoder-decoder", "audio-llm"]:
            # Get audio config
            if "audio_config" in config:
                audio_config = config["audio_config"]
            else:
                audio_config = config

            num_mel_bins = audio_config.get("num_mel_bins", 128)
            if num_mel_bins > 0:
                # Approximate audio frames from encoder sequence length
                enc_seq_len = encoder_seq_length or audio_config.get(
                    "max_source_positions", 1500
                )
                audio_frames = enc_seq_len * 2  # Approximate: ~2 frames per position
                audio_memory = self.calculate_audio_input_memory(
                    num_mel_bins, audio_frames, batch_size, precision
                )
                # Add conv layer activations
                audio_memory += self.calculate_audio_conv_memory(
                    config, audio_frames, batch_size, precision
                )

        # Extra work buffers (important for TTS, diffusion, etc.)
        extra_work_bytes = config.get("extra_work_bytes", 0)

        # Default work buffer for TTS models
        if self.model_type == "text-to-speech" and "extra_work_bytes" not in config:
            extra_work_bytes = 2 * 1024**3  # 2 GiB default for latents/overlap-add

        # For diffusion models, add latent buffer estimate if not specified
        if self.model_type == "diffusion" and "extra_work_bytes" not in config:
            # More generous estimate for SDXL (includes attention maps, etc.)
            latent_size = image_resolution // 8
            latent_channels = config.get("in_channels", 4)
            # Base latent buffer + attention maps + intermediate features
            latent_elements = (
                batch_size * latent_size * latent_size * latent_channels * 50
            )  # Increased multiplier
            extra_work_bytes = latent_elements * self.PRECISION_BYTES.get(
                precision.lower(), 2
            )

        # Convert all to bytes
        weight_bytes = weight_memory * 1e9
        kv_bytes = kv_cache * 1e9
        activation_bytes = activations * 1e9
        state_bytes = state_memory * 1e9
        lora_bytes = lora_memory * 1e9
        image_bytes = image_memory * 1e9
        audio_bytes = audio_memory * 1e9

        # Apply framework overhead to runtime components (not weights)
        runtime_bytes = (
            kv_bytes + activation_bytes + state_bytes + image_bytes + audio_bytes
        )
        runtime_bytes_with_overhead = (
            runtime_bytes * framework_overhead + extra_work_bytes
        )

        # Add gradients if training
        if include_gradients:
            weight_bytes *= 2  # Gradients are same size as weights

        # Create report
        return MemoryReport(
            model_type=self.model_type,
            attention_type=self.attention_type,
            precision=precision,
            parameter_count=param_count,
            weight_memory_bytes=weight_bytes,
            kv_cache_bytes=kv_bytes,
            activation_memory_bytes=activation_bytes,
            state_memory_bytes=state_bytes,
            image_memory_bytes=image_bytes,
            lora_adapter_memory_bytes=lora_bytes,
            lora_prefill_scratch_bytes=lora_scratch * 1e9,
            sampler_logits_bytes=sampler_logits,
            notes=notes,
            extra_work_bytes=extra_work_bytes
            + (runtime_bytes_with_overhead - runtime_bytes),
        )
