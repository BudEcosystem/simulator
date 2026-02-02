"""
Optimizer Configurations for Training Memory Estimation.

Provides detailed configurations for various optimizers including:
- Standard optimizers (AdamW, SGD, Adafactor)
- Memory-efficient optimizers (8-bit Adam, GaLore, APOLLO)
- Advanced optimizers (Adam-mini, Muon, Lion)

Based on analysis of LlamaFactory implementations and published papers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class OptimizerCategory(Enum):
    """Categories of optimizers."""
    STANDARD = "standard"          # Full precision, standard memory
    MEMORY_EFFICIENT = "memory_efficient"  # Reduced memory footprint
    LOW_RANK = "low_rank"          # Low-rank gradient approximation
    QUANTIZED = "quantized"        # Quantized optimizer states
    HYBRID = "hybrid"              # Combination approaches


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""

    name: str
    display_name: str
    description: str
    category: OptimizerCategory

    # Memory characteristics
    state_count: int  # Number of states per parameter (e.g., 2 for Adam)
    state_precision_bytes: float  # Bytes per parameter per state
    total_bytes_per_param: float  # Total optimizer memory per param

    # Memory reduction factors
    memory_reduction_vs_adamw: float = 1.0  # 1.0 = same as AdamW

    # Compatibility
    supports_distributed: bool = True
    supports_gradient_accumulation: bool = True
    supports_mixed_precision: bool = True
    requires_pure_bf16: bool = False  # Some optimizers work better without AMP

    # Special requirements
    min_batch_size: Optional[int] = None
    requires_large_batch: bool = False
    requires_specific_hardware: Optional[List[str]] = None

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def calculate_memory_gb(
        self,
        trainable_params: int,
        rank: Optional[int] = None,
    ) -> float:
        """
        Calculate optimizer state memory in GB.

        Args:
            trainable_params: Number of trainable parameters
            rank: For low-rank optimizers, the rank used

        Returns:
            Memory in GB
        """
        if self.category == OptimizerCategory.LOW_RANK and rank is not None:
            # Low-rank optimizers scale with rank, not full params
            effective_params = trainable_params * (rank / self.extra_params.get('default_rank', 16))
            effective_params = min(effective_params, trainable_params * 0.5)  # Cap at 50%
            return (effective_params * self.total_bytes_per_param) / 1e9

        return (trainable_params * self.total_bytes_per_param) / 1e9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "state_count": self.state_count,
            "state_precision_bytes": self.state_precision_bytes,
            "total_bytes_per_param": self.total_bytes_per_param,
            "memory_reduction_vs_adamw": self.memory_reduction_vs_adamw,
            "supports_distributed": self.supports_distributed,
            "supports_gradient_accumulation": self.supports_gradient_accumulation,
            "supports_mixed_precision": self.supports_mixed_precision,
            "extra_params": self.extra_params,
        }


# Standard AdamW baseline: 2 states × 4 bytes = 8 bytes per param
ADAMW_BASELINE_BYTES = 8.0


OPTIMIZER_CONFIGS: Dict[str, OptimizerConfig] = {
    # Standard Optimizers
    "adamw": OptimizerConfig(
        name="adamw",
        display_name="AdamW",
        description="Standard AdamW optimizer with momentum and variance in FP32",
        category=OptimizerCategory.STANDARD,
        state_count=2,  # momentum + variance
        state_precision_bytes=4.0,  # FP32
        total_bytes_per_param=8.0,  # 2 × 4 bytes
        memory_reduction_vs_adamw=1.0,
    ),

    "adam": OptimizerConfig(
        name="adam",
        display_name="Adam",
        description="Standard Adam optimizer (without weight decay correction)",
        category=OptimizerCategory.STANDARD,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=8.0,
        memory_reduction_vs_adamw=1.0,
    ),

    "sgd": OptimizerConfig(
        name="sgd",
        display_name="SGD with Momentum",
        description="Stochastic Gradient Descent with momentum",
        category=OptimizerCategory.STANDARD,
        state_count=1,  # momentum only
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,  # 1 × 4 bytes
        memory_reduction_vs_adamw=0.5,  # 50% of AdamW
    ),

    "adafactor": OptimizerConfig(
        name="adafactor",
        display_name="Adafactor",
        description="Memory-efficient optimizer with factorized second moment",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=1,  # Factorized states
        state_precision_bytes=4.0,  # FP32
        total_bytes_per_param=4.0,  # Approximately 4 bytes effective
        memory_reduction_vs_adamw=0.5,  # 50% of AdamW
        extra_params={
            "scale_parameter": True,
            "relative_step": True,
        },
    ),

    "lion": OptimizerConfig(
        name="lion",
        display_name="Lion",
        description="Sign-based optimizer with momentum, competitive with Adam",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=1,  # momentum only
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
    ),

    # Quantized Optimizers
    "adamw_8bit": OptimizerConfig(
        name="adamw_8bit",
        display_name="8-bit AdamW",
        description="AdamW with quantized optimizer states (bitsandbytes)",
        category=OptimizerCategory.QUANTIZED,
        state_count=2,
        state_precision_bytes=1.0,  # INT8
        total_bytes_per_param=2.0,  # 2 × 1 byte
        memory_reduction_vs_adamw=0.25,  # 75% reduction
        extra_params={
            "requires_bitsandbytes": True,
            "min_version": "0.37.0",
        },
    ),

    "paged_adamw_8bit": OptimizerConfig(
        name="paged_adamw_8bit",
        display_name="Paged 8-bit AdamW",
        description="8-bit AdamW with CPU offloading for memory spikes",
        category=OptimizerCategory.QUANTIZED,
        state_count=2,
        state_precision_bytes=1.0,
        total_bytes_per_param=2.0,
        memory_reduction_vs_adamw=0.25,
        extra_params={
            "requires_bitsandbytes": True,
            "min_version": "0.39.0",
            "supports_paging": True,
        },
    ),

    # Low-Rank Optimizers
    "galore": OptimizerConfig(
        name="galore",
        display_name="GaLore",
        description="Gradient Low-Rank Projection for memory-efficient full-param training",
        category=OptimizerCategory.LOW_RANK,
        state_count=2,  # Still AdamW states, but on low-rank gradients
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,  # 2 states × 4 bytes (LOW_RANK category handles reduction)
        memory_reduction_vs_adamw=0.25,  # 75% reduction
        requires_pure_bf16=True,  # Works better without AMP
        supports_gradient_accumulation=False,  # Not in layerwise mode
        extra_params={
            "default_rank": 16,
            "update_interval": 200,
            "scale": 2.0,
            "proj_type": "std",  # std, reverse_std, right, left, full
        },
    ),

    "galore_8bit": OptimizerConfig(
        name="galore_8bit",
        display_name="8-bit GaLore",
        description="GaLore with 8-bit quantized states for maximum memory savings",
        category=OptimizerCategory.LOW_RANK,
        state_count=2,
        state_precision_bytes=1.0,
        total_bytes_per_param=2.0,  # 2 states × 1 byte (LOW_RANK category handles reduction)
        memory_reduction_vs_adamw=0.0625,  # 93.75% reduction
        requires_pure_bf16=True,
        supports_gradient_accumulation=False,
        extra_params={
            "default_rank": 16,
            "update_interval": 200,
            "enables_7b_on_24gb": True,
        },
    ),

    "apollo": OptimizerConfig(
        name="apollo",
        display_name="APOLLO",
        description="Approximate second-order optimizer with low-rank projection",
        category=OptimizerCategory.LOW_RANK,
        state_count=1,
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
        requires_pure_bf16=True,
        supports_gradient_accumulation=False,  # Not in layerwise mode
        extra_params={
            "default_rank": 16,
            "update_interval": 200,
            "scale": 32.0,
            "proj": "svd",  # svd or random
        },
    ),

    # Advanced Optimizers
    "adam_mini": OptimizerConfig(
        name="adam_mini",
        display_name="Adam-mini",
        description="Architecture-aware optimizer reducing states by 50%",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=1,  # Effectively 1 due to sharing
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
        supports_distributed=True,
        extra_params={
            "uses_model_architecture": True,
        },
    ),

    "q_adam_mini": OptimizerConfig(
        name="q_adam_mini",
        display_name="Q-Adam-mini",
        description="Quantized Adam-mini with 8× memory reduction",
        category=OptimizerCategory.QUANTIZED,
        state_count=1,
        state_precision_bytes=1.0,  # INT8 for momentum
        total_bytes_per_param=1.0,
        memory_reduction_vs_adamw=0.125,  # 87.5% reduction
        extra_params={
            "quantizes_momentum": True,
            "keeps_variance_fp32": False,
        },
    ),

    "muon": OptimizerConfig(
        name="muon",
        display_name="Muon",
        description="Momentum + Newton-Schulz orthogonalization, hybrid with AdamW",
        category=OptimizerCategory.HYBRID,
        state_count=1,  # Momentum only for Muon part
        state_precision_bytes=4.0,  # FP32
        total_bytes_per_param=4.0,  # Mixed with AdamW for some params
        memory_reduction_vs_adamw=0.5,
        requires_large_batch=True,
        supports_mixed_precision=True,
        extra_params={
            "ns_steps": 5,  # Newton-Schulz iterations
            "momentum": 0.95,
            "nesterov": True,
            "note": "May not work well with small batches or fine-tuning",
        },
    ),

    # Block-wise Optimizers
    "badam_layer": OptimizerConfig(
        name="badam_layer",
        display_name="BAdam (Layer-wise)",
        description="Block-wise AdamW updating one layer at a time",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=1.0,  # Only 1 layer active at a time
        memory_reduction_vs_adamw=0.03,  # ~97% reduction (for 32 layers)
        supports_distributed=True,  # Works with DeepSpeed ZeRO-3
        extra_params={
            "switch_mode": "ascending",  # ascending, descending, random, fixed
            "switch_interval": 50,
        },
    ),

    # Paged optimizers (CPU offload variants)
    "paged_adamw_32bit": OptimizerConfig(
        name="paged_adamw_32bit",
        display_name="Paged 32-bit AdamW",
        description="AdamW with CPU offloading for memory spikes",
        category=OptimizerCategory.STANDARD,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=8.0,
        memory_reduction_vs_adamw=1.0,
        extra_params={
            "requires_bitsandbytes": True,
            "supports_paging": True,
        },
    ),

    "lion_8bit": OptimizerConfig(
        name="lion_8bit",
        display_name="8-bit Lion",
        description="Lion optimizer with quantized momentum state",
        category=OptimizerCategory.QUANTIZED,
        state_count=1,
        state_precision_bytes=1.0,
        total_bytes_per_param=1.0,
        memory_reduction_vs_adamw=0.125,
        extra_params={
            "requires_bitsandbytes": True,
        },
    ),

    "paged_lion_8bit": OptimizerConfig(
        name="paged_lion_8bit",
        display_name="Paged 8-bit Lion",
        description="8-bit Lion with CPU offloading",
        category=OptimizerCategory.QUANTIZED,
        state_count=1,
        state_precision_bytes=1.0,
        total_bytes_per_param=1.0,
        memory_reduction_vs_adamw=0.125,
        extra_params={
            "requires_bitsandbytes": True,
            "supports_paging": True,
        },
    ),

    "paged_lion_32bit": OptimizerConfig(
        name="paged_lion_32bit",
        display_name="Paged 32-bit Lion",
        description="Lion with CPU offloading",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=1,
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
        extra_params={
            "requires_bitsandbytes": True,
            "supports_paging": True,
        },
    ),

    # ADEMAMIX variants
    "ademamix": OptimizerConfig(
        name="ademamix",
        display_name="AdemaMix",
        description="Adam with exponential moving average mixture for better convergence",
        category=OptimizerCategory.STANDARD,
        state_count=3,  # momentum1 + momentum2 + variance
        state_precision_bytes=4.0,
        total_bytes_per_param=12.0,
        memory_reduction_vs_adamw=1.5,  # 50% more than AdamW
    ),

    "ademamix_8bit": OptimizerConfig(
        name="ademamix_8bit",
        display_name="8-bit AdemaMix",
        description="AdemaMix with quantized optimizer states",
        category=OptimizerCategory.QUANTIZED,
        state_count=3,
        state_precision_bytes=1.0,
        total_bytes_per_param=3.0,
        memory_reduction_vs_adamw=0.375,
        extra_params={
            "requires_bitsandbytes": True,
        },
    ),

    # LOMO family
    "lomo": OptimizerConfig(
        name="lomo",
        display_name="LOMO",
        description="LOw-Memory Optimization - fuses gradient into weight update",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=0,  # No optimizer states
        state_precision_bytes=0.0,
        total_bytes_per_param=0.0,
        memory_reduction_vs_adamw=0.0,
        supports_gradient_accumulation=False,
        extra_params={
            "note": "Zero optimizer memory but may converge slower",
        },
    ),

    "adalomo": OptimizerConfig(
        name="adalomo",
        display_name="AdaLOMO",
        description="Adaptive LOMO with second-moment estimation for stability",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=1,  # running variance only
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
        supports_gradient_accumulation=False,
    ),

    # Schedule-free optimizers
    "schedule_free_adamw": OptimizerConfig(
        name="schedule_free_adamw",
        display_name="Schedule-Free AdamW",
        description="AdamW without learning rate schedule, uses interpolation",
        category=OptimizerCategory.STANDARD,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=8.0,
        memory_reduction_vs_adamw=1.0,
    ),

    "schedule_free_sgd": OptimizerConfig(
        name="schedule_free_sgd",
        display_name="Schedule-Free SGD",
        description="SGD without learning rate schedule",
        category=OptimizerCategory.STANDARD,
        state_count=1,
        state_precision_bytes=4.0,
        total_bytes_per_param=4.0,
        memory_reduction_vs_adamw=0.5,
    ),

    # GrokAdamW
    "grokadamw": OptimizerConfig(
        name="grokadamw",
        display_name="GrokAdamW",
        description="AdamW variant for grokking with weight norm penalty",
        category=OptimizerCategory.STANDARD,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=8.0,
        memory_reduction_vs_adamw=1.0,
    ),

    "badam_ratio": OptimizerConfig(
        name="badam_ratio",
        display_name="BAdam (Ratio-wise)",
        description="Block-wise AdamW updating fraction of parameters",
        category=OptimizerCategory.MEMORY_EFFICIENT,
        state_count=2,
        state_precision_bytes=4.0,
        total_bytes_per_param=8.0 * 0.05,  # 5% default ratio
        memory_reduction_vs_adamw=0.05,  # 95% reduction at 5% ratio
        supports_distributed=False,  # Single-GPU only
        extra_params={
            "update_ratio": 0.05,
            "mask_mode": "adjacent",  # adjacent or scatter
        },
    ),
}


OPTIMIZER_ALIASES: Dict[str, str] = {
    "adamw_torch": "adamw", "adamw_torch_fused": "adamw",
    "adamw_hf": "adamw", "adamw_bnb_8bit": "adamw_8bit",
    "adamw_apex_fused": "adamw", "adamw_anyprecision": "adamw",
    "paged_adamw": "paged_adamw_8bit",
    "adam_bnb_8bit": "adamw_8bit",
    "adamw_torch_xla": "adamw", "adamw_torch_npu_fused": "adamw",
    "adamax": "adamw", "rmsprop": "sgd", "rprop": "sgd",
    "asgd": "sgd", "adagrad": "adafactor", "lbfgs": "sgd",
    "adadelta": "sgd",
    # LlamaFactory custom
    "galore_adamw": "galore", "galore_adamw_8bit": "galore_8bit",
    "galore_adafactor": "galore", "galore_adamw_layerwise": "galore",
    "galore_adamw_8bit_layerwise": "galore_8bit",
    "badam": "badam_layer", "adam_mini": "adam_mini",
}


def get_optimizer_config(name: str) -> OptimizerConfig:
    """
    Get configuration for an optimizer.

    Args:
        name: Optimizer name (e.g., 'adamw', 'galore', 'adamw_8bit')

    Returns:
        OptimizerConfig for the specified optimizer

    Raises:
        ValueError: If optimizer is not found
    """
    name = name.lower()
    # Resolve aliases
    name = OPTIMIZER_ALIASES.get(name, name)
    if name in OPTIMIZER_CONFIGS:
        return OPTIMIZER_CONFIGS[name]

    raise ValueError(
        f"Unknown optimizer: {name}. "
        f"Valid optimizers: {list(OPTIMIZER_CONFIGS.keys())}"
    )


def list_optimizers(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available optimizers.

    Args:
        category: Filter by category (standard, memory_efficient, low_rank, quantized, hybrid)

    Returns:
        List of optimizer information dictionaries
    """
    result = []
    for name, config in OPTIMIZER_CONFIGS.items():
        if category is None or config.category.value == category:
            result.append({
                "name": name,
                "display_name": config.display_name,
                "description": config.description,
                "category": config.category.value,
                "memory_reduction": config.memory_reduction_vs_adamw,
                "bytes_per_param": config.total_bytes_per_param,
            })
    return result


def calculate_optimizer_memory(
    optimizer: str,
    trainable_params: int,
    rank: Optional[int] = None,
    deepspeed_stage: Optional[str] = None,
    data_parallel: int = 1,
) -> float:
    """
    Calculate optimizer state memory in GB.

    Args:
        optimizer: Optimizer name
        trainable_params: Number of trainable parameters
        rank: For low-rank optimizers, the rank used
        deepspeed_stage: DeepSpeed ZeRO stage (zero2, zero3)
        data_parallel: Data parallelism degree

    Returns:
        Optimizer state memory in GB per GPU
    """
    config = get_optimizer_config(optimizer)
    memory_gb = config.calculate_memory_gb(trainable_params, rank)

    # Apply DeepSpeed sharding
    if deepspeed_stage in ('zero2', 'zero3'):
        memory_gb /= data_parallel

    return memory_gb


def get_recommended_optimizer(
    model_params: int,
    available_memory_gb: float,
    num_gpus: int = 1,
    training_type: str = "sft",
    prefer_quality: bool = True,
) -> str:
    """
    Get recommended optimizer based on constraints.

    Args:
        model_params: Total model parameters
        available_memory_gb: Available GPU memory
        num_gpus: Number of GPUs
        training_type: Training type (sft, dpo, ppo, etc.)
        prefer_quality: If True, prefer quality over memory savings

    Returns:
        Recommended optimizer name
    """
    # Calculate base memory needs (weights + gradients + activations estimate)
    # Rough estimate: weights in bf16 = params × 2 bytes
    # Gradients = params × 4 bytes (fp32)
    # Activations ~= weights for training
    base_memory_gb = (model_params * (2 + 4 + 2)) / 1e9 / num_gpus

    # Available for optimizer
    optimizer_budget_gb = available_memory_gb - base_memory_gb

    if optimizer_budget_gb <= 0:
        # Need most aggressive optimization
        return "galore_8bit"

    # Calculate which optimizers fit
    candidates = []
    for name, config in OPTIMIZER_CONFIGS.items():
        opt_memory = config.calculate_memory_gb(model_params) / num_gpus
        if opt_memory <= optimizer_budget_gb:
            candidates.append((name, config, opt_memory))

    if not candidates:
        return "galore_8bit"  # Fallback to most memory-efficient

    # Sort by preference
    if prefer_quality:
        # Prefer standard optimizers if they fit
        priority = ['adamw', 'adam', 'lion', 'adafactor', 'adamw_8bit', 'galore', 'apollo']
    else:
        # Prefer memory-efficient options
        priority = ['galore_8bit', 'galore', 'adamw_8bit', 'adam_mini', 'adafactor', 'lion', 'adamw']

    for name in priority:
        if name in [c[0] for c in candidates]:
            return name

    # Return first candidate if none in priority list
    return candidates[0][0]
