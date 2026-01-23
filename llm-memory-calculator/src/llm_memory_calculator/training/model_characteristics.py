"""
Model Characteristics Analysis for Training Optimization.

Provides detailed analysis of model architectures including:
- Attention mechanisms (MHA, MQA, GQA, MLA)
- FFN variants (Dense, MoE, GLU)
- Memory patterns and compute requirements
- Quality-cost tradeoffs for different configurations

Sources:
- DeepSeek-V3 MLA paper
- Mixtral MoE paper
- GQA paper (Google)
- Various architecture papers
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import math


class AttentionType(Enum):
    """Types of attention mechanisms."""
    MHA = "mha"       # Multi-Head Attention (standard)
    MQA = "mqa"       # Multi-Query Attention (single KV head)
    GQA = "gqa"       # Grouped Query Attention
    MLA = "mla"       # Multi-Head Latent Attention (DeepSeek)
    SLIDING = "sliding"  # Sliding window attention


class FFNType(Enum):
    """Types of Feed-Forward Networks."""
    DENSE = "dense"           # Standard dense FFN
    GATED = "gated"          # Gated linear unit (GLU)
    SWIGLU = "swiglu"        # SwiGLU activation
    MOE = "moe"              # Mixture of Experts
    MOE_SHARED = "moe_shared" # MoE with shared experts


class ModelFamily(Enum):
    """Model architecture families."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    PHI = "phi"
    GEMMA = "gemma"
    GPT = "gpt"
    MAMBA = "mamba"
    JAMBA = "jamba"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""

    attention_type: AttentionType
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int

    # MLA-specific
    kv_lora_rank: int = 0       # For MLA compression
    q_lora_rank: int = 0        # For MLA Q compression
    qk_rope_head_dim: int = 0   # RoPE dimension for MLA
    qk_nope_head_dim: int = 0   # Non-RoPE dimension for MLA
    v_head_dim: int = 0         # Value head dimension for MLA

    # Sliding window
    sliding_window: Optional[int] = None

    @property
    def kv_cache_ratio(self) -> float:
        """KV cache size relative to MHA baseline."""
        if self.attention_type == AttentionType.MHA:
            return 1.0
        elif self.attention_type == AttentionType.MQA:
            return 1.0 / self.num_attention_heads
        elif self.attention_type == AttentionType.GQA:
            return self.num_kv_heads / self.num_attention_heads
        elif self.attention_type == AttentionType.MLA:
            # MLA compresses to latent dimension
            if self.kv_lora_rank > 0:
                return self.kv_lora_rank / (self.head_dim * self.num_attention_heads)
            return 0.3  # Default ~70% reduction
        return 1.0

    @property
    def quality_factor(self) -> float:
        """Relative quality compared to MHA (1.0 = same)."""
        # Based on published benchmarks
        if self.attention_type == AttentionType.MHA:
            return 1.0
        elif self.attention_type == AttentionType.MQA:
            # MQA degrades at scale
            return 0.95
        elif self.attention_type == AttentionType.GQA:
            # GQA is close to MHA
            return 0.98
        elif self.attention_type == AttentionType.MLA:
            # MLA can exceed MHA quality
            return 1.02
        return 1.0

    def compute_kv_cache_size(
        self,
        batch_size: int,
        seq_length: int,
        precision_bytes: float = 2,
    ) -> float:
        """Compute KV cache size in GB."""
        if self.attention_type == AttentionType.MLA:
            # MLA stores compressed latent vectors
            kv_elements = batch_size * seq_length * self.kv_lora_rank
        else:
            # Standard KV: batch × seq × kv_heads × head_dim × 2 (K and V)
            kv_elements = batch_size * seq_length * self.num_kv_heads * self.head_dim * 2

        return (kv_elements * precision_bytes) / 1e9

    def compute_attention_flops(
        self,
        batch_size: int,
        seq_length: int,
    ) -> int:
        """Compute attention FLOPs per layer."""
        # Q × K^T: batch × heads × seq × seq × head_dim
        qk_flops = 2 * batch_size * self.num_attention_heads * seq_length * seq_length * self.head_dim

        # Attention × V: batch × heads × seq × seq × head_dim
        av_flops = 2 * batch_size * self.num_attention_heads * seq_length * seq_length * self.head_dim

        # Output projection: batch × seq × hidden × hidden
        out_flops = 2 * batch_size * seq_length * self.hidden_size * self.hidden_size

        return qk_flops + av_flops + out_flops


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts."""

    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1  # Top-K
    expert_intermediate_size: int = 0

    # Shared experts (DeepSeek-V3)
    num_shared_experts: int = 0
    shared_expert_intermediate_size: int = 0

    # Dense layers (some models have first N layers as dense)
    first_k_dense_replace: int = 0

    # Router configuration
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    @property
    def effective_params_ratio(self) -> float:
        """Ratio of active params to total params per token."""
        if not self.is_moe:
            return 1.0

        # Active experts + shared experts
        active = self.num_experts_per_tok + self.num_shared_experts
        total = self.num_experts + self.num_shared_experts

        return active / total if total > 0 else 1.0

    @property
    def memory_multiplier(self) -> float:
        """Memory multiplier compared to dense model of same active params."""
        if not self.is_moe:
            return 1.0

        # MoE requires all experts loaded even though only top-k active
        return self.num_experts / self.num_experts_per_tok

    @property
    def compute_efficiency(self) -> float:
        """Compute efficiency factor (1.0 = no overhead)."""
        if not self.is_moe:
            return 1.0

        # MoE has router overhead and potential load imbalance
        base_efficiency = 0.95  # Router overhead

        # Load imbalance penalty (increases with more experts)
        if self.num_experts > 32:
            base_efficiency *= 0.95
        elif self.num_experts > 8:
            base_efficiency *= 0.97

        return base_efficiency

    def estimate_communication_overhead(self, num_gpus: int) -> float:
        """Estimate all-to-all communication overhead for expert parallel."""
        if not self.is_moe or num_gpus == 1:
            return 0.0

        # Expert parallel requires all-to-all for dispatch/combine
        # Overhead increases with number of GPUs
        base_overhead = 0.10  # 10% baseline

        if num_gpus > 8:
            base_overhead += 0.05
        if num_gpus > 32:
            base_overhead += 0.05

        return base_overhead


@dataclass
class ModelCharacteristics:
    """Complete model characteristics for training optimization."""

    # Basic info
    name: str
    family: ModelFamily
    num_parameters: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    # Architecture
    attention: AttentionConfig
    ffn_type: FFNType
    moe: Optional[MoEConfig] = None

    # Context
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0

    # Activation
    hidden_act: str = "silu"

    # Normalization
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    @property
    def is_moe(self) -> bool:
        return self.moe is not None and self.moe.is_moe

    @property
    def active_parameters(self) -> int:
        """Number of parameters active per forward pass."""
        if not self.is_moe:
            return self.num_parameters

        # For MoE: embedding + attention + active experts
        embedding_params = self.vocab_size * self.hidden_size * 2  # Input + output
        attention_params_per_layer = (
            self.attention.num_attention_heads * self.attention.head_dim * self.hidden_size * 4
        )
        total_attention = attention_params_per_layer * self.num_layers

        # Active FFN params
        active_ffn_per_layer = (
            self.moe.expert_intermediate_size * self.hidden_size * 3 *
            self.moe.num_experts_per_tok
        )
        # Add shared experts if present
        if self.moe.num_shared_experts > 0:
            active_ffn_per_layer += (
                self.moe.shared_expert_intermediate_size * self.hidden_size * 3 *
                self.moe.num_shared_experts
            )

        total_ffn = active_ffn_per_layer * self.num_layers

        return embedding_params + total_attention + total_ffn

    @property
    def training_memory_efficiency(self) -> float:
        """
        Memory efficiency factor for training.
        Lower is better (less memory per active param).
        """
        if not self.is_moe:
            return 1.0

        # MoE has worse memory efficiency due to loading all experts
        return self.moe.memory_multiplier

    def estimate_training_flops_per_token(self) -> int:
        """Estimate training FLOPs per token (forward + backward)."""
        # Rule of thumb: 6 * params for training
        # For MoE, use active parameters
        return 6 * self.active_parameters

    def estimate_quality_score(
        self,
        training_method: str = "full",
        optimizer: str = "adamw",
        precision: str = "bf16",
        quantization: Optional[str] = None,
    ) -> float:
        """
        Estimate relative quality score (1.0 = baseline).

        Considers model architecture and training configuration.
        """
        score = 1.0

        # Attention type impact
        score *= self.attention.quality_factor

        # Training method impact
        method_factors = {
            "full": 1.0,
            "lora": 0.95,  # Slight quality reduction
            "qlora": 0.93,
            "freeze": 0.90,
            "dora": 0.96,
            "pissa": 0.95,
        }
        score *= method_factors.get(training_method, 1.0)

        # Optimizer impact
        optimizer_factors = {
            "adamw": 1.0,
            "adam": 1.0,
            "adamw_8bit": 0.995,  # Negligible quality loss
            "galore": 0.98,
            "apollo": 0.98,
            "sgd": 0.95,
            "adafactor": 0.97,
        }
        score *= optimizer_factors.get(optimizer.lower(), 1.0)

        # Precision impact
        if precision == "fp32":
            score *= 1.0
        elif precision in ("bf16", "fp16"):
            score *= 0.999  # Minimal impact
        elif precision == "fp8":
            score *= 0.99

        # Quantization impact
        if quantization:
            quant_factors = {
                "int8": 0.99,
                "nf4": 0.98,
                "int4": 0.97,
                "fp4": 0.96,
            }
            score *= quant_factors.get(quantization.lower(), 0.98)

        return score

    def get_optimal_parallelism(
        self,
        num_gpus: int,
        gpu_memory_gb: float,
        training_method: str = "lora",
    ) -> Dict[str, int]:
        """
        Get optimal parallelism configuration for this model.

        Returns dict with tp, pp, dp, ep values.
        """
        precision_bytes = 2  # Assume bf16

        # Estimate memory per GPU needed
        if training_method == "qlora":
            weight_mem = (self.num_parameters * 0.5) / 1e9
        elif training_method in ("lora", "dora", "pissa"):
            weight_mem = (self.num_parameters * precision_bytes) / 1e9
        else:
            # Full fine-tuning
            weight_mem = (self.num_parameters * precision_bytes) / 1e9

        # Check if model fits on single GPU
        available_mem = gpu_memory_gb * 0.85  # 85% utilization target

        if weight_mem <= available_mem:
            # Model fits, use pure data parallel
            return {"tp": 1, "pp": 1, "dp": num_gpus, "ep": 1}

        # Need model parallelism
        tp = 1
        pp = 1
        ep = 1

        # TP first (lowest overhead)
        while weight_mem / tp > available_mem and tp < 8:
            tp *= 2

        # If still doesn't fit, add PP
        if weight_mem / tp > available_mem and self.num_layers >= 40:
            pp = min(4, num_gpus // tp)

        # For MoE, consider expert parallel
        if self.is_moe and self.moe.num_experts >= 8:
            # Expert parallel can help distribute experts
            if num_gpus >= 8:
                ep = min(self.moe.num_experts, num_gpus // (tp * pp))

        # Remaining GPUs for data parallel
        dp = max(1, num_gpus // (tp * pp * ep))

        return {"tp": tp, "pp": pp, "dp": dp, "ep": ep}

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to standard config dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.attention.num_attention_heads,
            "num_key_value_heads": self.attention.num_kv_heads,
            "vocab_size": self.vocab_size,
            "num_parameters": self.num_parameters,
            "max_position_embeddings": self.max_position_embeddings,
            # MoE fields
            "num_experts": self.moe.num_experts if self.is_moe else 1,
            "num_experts_per_tok": self.moe.num_experts_per_tok if self.is_moe else 1,
        }


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def create_llama_characteristics(
    name: str,
    params_b: float,
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    vocab_size: int = 128256,
    max_seq: int = 8192,
) -> ModelCharacteristics:
    """Create LLaMA-style model characteristics."""
    head_dim = hidden // heads

    return ModelCharacteristics(
        name=name,
        family=ModelFamily.LLAMA,
        num_parameters=int(params_b * 1e9),
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            attention_type=AttentionType.GQA if kv_heads < heads else AttentionType.MHA,
            num_attention_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
        ),
        ffn_type=FFNType.SWIGLU,
        max_position_embeddings=max_seq,
    )


def create_moe_characteristics(
    name: str,
    params_b: float,
    active_params_b: float,
    layers: int,
    hidden: int,
    expert_intermediate: int,
    heads: int,
    kv_heads: int,
    num_experts: int,
    top_k: int,
    vocab_size: int = 32000,
    num_shared_experts: int = 0,
    shared_intermediate: int = 0,
) -> ModelCharacteristics:
    """Create MoE model characteristics."""
    head_dim = hidden // heads

    return ModelCharacteristics(
        name=name,
        family=ModelFamily.MISTRAL,
        num_parameters=int(params_b * 1e9),
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=expert_intermediate,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            attention_type=AttentionType.GQA if kv_heads < heads else AttentionType.MHA,
            num_attention_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
        ),
        ffn_type=FFNType.MOE_SHARED if num_shared_experts > 0 else FFNType.MOE,
        moe=MoEConfig(
            is_moe=True,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            expert_intermediate_size=expert_intermediate,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_intermediate,
        ),
    )


def create_deepseek_mla_characteristics(
    name: str,
    params_b: float,
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    vocab_size: int = 102400,
    num_experts: int = 1,
    top_k: int = 1,
) -> ModelCharacteristics:
    """Create DeepSeek-style model with MLA attention."""
    head_dim = hidden // heads

    moe_config = None
    if num_experts > 1:
        moe_config = MoEConfig(
            is_moe=True,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            expert_intermediate_size=intermediate,
        )

    return ModelCharacteristics(
        name=name,
        family=ModelFamily.DEEPSEEK,
        num_parameters=int(params_b * 1e9),
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            attention_type=AttentionType.MLA,
            num_attention_heads=heads,
            num_kv_heads=heads,
            head_dim=head_dim,
            hidden_size=hidden,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
        ),
        ffn_type=FFNType.MOE if num_experts > 1 else FFNType.SWIGLU,
        moe=moe_config,
    )


# Pre-defined model characteristics
MODEL_CHARACTERISTICS: Dict[str, ModelCharacteristics] = {
    # LLaMA family
    "llama-3-8b": create_llama_characteristics(
        "llama-3-8b", 8, 32, 4096, 14336, 32, 8, 128256, 8192
    ),
    "llama-3-70b": create_llama_characteristics(
        "llama-3-70b", 70, 80, 8192, 28672, 64, 8, 128256, 8192
    ),
    "llama-3.1-8b": create_llama_characteristics(
        "llama-3.1-8b", 8, 32, 4096, 14336, 32, 8, 128256, 131072
    ),
    "llama-3.1-70b": create_llama_characteristics(
        "llama-3.1-70b", 70, 80, 8192, 28672, 64, 8, 128256, 131072
    ),
    "llama-3.1-405b": create_llama_characteristics(
        "llama-3.1-405b", 405, 126, 16384, 53248, 128, 8, 128256, 131072
    ),

    # Qwen family
    "qwen2-7b": create_llama_characteristics(
        "qwen2-7b", 7, 32, 4096, 11008, 32, 4, 152064, 32768
    ),
    "qwen2-72b": create_llama_characteristics(
        "qwen2-72b", 72, 80, 8192, 29568, 64, 8, 152064, 32768
    ),

    # Mistral family
    "mistral-7b": create_llama_characteristics(
        "mistral-7b", 7, 32, 4096, 14336, 32, 8, 32000, 32768
    ),

    # MoE models
    "mixtral-8x7b": create_moe_characteristics(
        "mixtral-8x7b", 46.7, 12.9, 32, 4096, 14336, 32, 8, 8, 2, 32000
    ),
    "mixtral-8x22b": create_moe_characteristics(
        "mixtral-8x22b", 176, 39, 56, 6144, 16384, 48, 8, 8, 2, 32000
    ),

    # DeepSeek with MLA
    "deepseek-v2-lite": create_deepseek_mla_characteristics(
        "deepseek-v2-lite", 16, 27, 2048, 10944, 16, 512, 1536, 102400, 64, 6
    ),
    "deepseek-v2": create_deepseek_mla_characteristics(
        "deepseek-v2", 236, 60, 5120, 12288, 128, 512, 1536, 102400, 160, 6
    ),
    "deepseek-v3": create_deepseek_mla_characteristics(
        "deepseek-v3", 671, 61, 7168, 18432, 128, 512, 1536, 129280, 256, 8
    ),
}


def get_model_characteristics(name: str) -> ModelCharacteristics:
    """Get pre-defined model characteristics."""
    name_lower = name.lower().replace("_", "-")

    if name_lower in MODEL_CHARACTERISTICS:
        return MODEL_CHARACTERISTICS[name_lower]

    # Try partial match
    for key, char in MODEL_CHARACTERISTICS.items():
        if name_lower in key or key in name_lower:
            return char

    raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CHARACTERISTICS.keys())}")


def analyze_model_config(config: Dict[str, Any]) -> ModelCharacteristics:
    """
    Analyze a model config dict and return characteristics.

    Automatically detects architecture type (MoE, attention variant, etc).
    """
    # Extract basic params
    hidden = config.get("hidden_size", 4096)
    intermediate = config.get("intermediate_size", hidden * 4)
    layers = config.get("num_hidden_layers", 32)
    heads = config.get("num_attention_heads", 32)
    kv_heads = config.get("num_key_value_heads", heads)
    vocab = config.get("vocab_size", 32000)
    params = config.get("num_parameters", 0)
    max_pos = config.get("max_position_embeddings", 8192)

    # Detect attention type
    if config.get("kv_lora_rank", 0) > 0:
        attn_type = AttentionType.MLA
    elif kv_heads == 1:
        attn_type = AttentionType.MQA
    elif kv_heads < heads:
        attn_type = AttentionType.GQA
    else:
        attn_type = AttentionType.MHA

    head_dim = hidden // heads

    attention = AttentionConfig(
        attention_type=attn_type,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        kv_lora_rank=config.get("kv_lora_rank", 0),
        q_lora_rank=config.get("q_lora_rank", 0),
        sliding_window=config.get("sliding_window"),
    )

    # Detect MoE
    num_experts = config.get("num_experts", config.get("num_local_experts", 1))
    moe_config = None
    ffn_type = FFNType.SWIGLU  # Default

    if num_experts > 1:
        moe_config = MoEConfig(
            is_moe=True,
            num_experts=num_experts,
            num_experts_per_tok=config.get("num_experts_per_tok", config.get("num_selected_experts", 2)),
            expert_intermediate_size=config.get("expert_intermediate_size", intermediate),
            num_shared_experts=config.get("n_shared_experts", 0),
            shared_expert_intermediate_size=config.get("shared_expert_intermediate_size", 0),
            first_k_dense_replace=config.get("first_k_dense_replace", 0),
        )
        ffn_type = FFNType.MOE_SHARED if moe_config.num_shared_experts > 0 else FFNType.MOE

    # Estimate parameters if not provided
    if params == 0:
        # Rough estimate
        emb_params = vocab * hidden * 2
        attn_params = layers * (hidden * hidden * 4)  # QKVO
        ffn_params = layers * (hidden * intermediate * 3)
        if moe_config:
            ffn_params *= moe_config.num_experts
        params = emb_params + attn_params + ffn_params

    # Detect family
    family = ModelFamily.LLAMA  # Default
    arch = config.get("architectures", [""])[0].lower() if config.get("architectures") else ""
    model_type = config.get("model_type", "").lower()

    if "qwen" in arch or "qwen" in model_type:
        family = ModelFamily.QWEN
    elif "mistral" in arch or "mixtral" in arch:
        family = ModelFamily.MISTRAL
    elif "deepseek" in arch or "deepseek" in model_type:
        family = ModelFamily.DEEPSEEK
    elif "phi" in arch or "phi" in model_type:
        family = ModelFamily.PHI
    elif "gemma" in arch or "gemma" in model_type:
        family = ModelFamily.GEMMA
    elif "mamba" in arch or "mamba" in model_type:
        family = ModelFamily.MAMBA
    elif "jamba" in arch or "jamba" in model_type:
        family = ModelFamily.JAMBA

    return ModelCharacteristics(
        name=config.get("_name_or_path", "unknown"),
        family=family,
        num_parameters=params,
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab,
        attention=attention,
        ffn_type=ffn_type,
        moe=moe_config,
        max_position_embeddings=max_pos,
    )


def list_model_characteristics() -> List[str]:
    """List available pre-defined models."""
    return list(MODEL_CHARACTERISTICS.keys())
