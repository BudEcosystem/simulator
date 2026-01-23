"""
Distributed Training Configurations.

Provides detailed modeling for:
- DeepSpeed ZeRO stages (0, 1, 2, 3, offload)
- FSDP sharding strategies
- Communication overhead calculations
- Memory partitioning across GPUs

Based on analysis of DeepSpeed documentation and benchmarks.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math


class DeepSpeedZeROStage(Enum):
    """DeepSpeed ZeRO optimization stages."""
    STAGE_0 = "zero0"  # No sharding (baseline)
    STAGE_1 = "zero1"  # Optimizer state sharding
    STAGE_2 = "zero2"  # + Gradient sharding
    STAGE_3 = "zero3"  # + Parameter sharding


class FSDPStrategy(Enum):
    """FSDP sharding strategies."""
    NO_SHARD = "no_shard"           # Data parallel only
    SHARD_GRAD_OP = "shard_grad_op" # Shard gradients and optimizer
    FULL_SHARD = "full_shard"       # Full model sharding (like ZeRO-3)


class OffloadTarget(Enum):
    """Offload targets for ZeRO-Offload/Infinity."""
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed ZeRO stage."""

    stage: DeepSpeedZeROStage
    name: str
    description: str

    # Sharding characteristics
    optimizer_sharded: bool
    gradient_sharded: bool
    param_sharded: bool

    # Memory reduction factors (relative to baseline)
    memory_reduction_factor: float  # 1/N where N is effective reduction

    # Communication overhead (as percentage of baseline throughput)
    communication_overhead: float  # 0.0 = no overhead, 0.5 = 50% slower

    # Offload support
    supports_optimizer_offload: bool = False
    supports_param_offload: bool = False
    supports_nvme_offload: bool = False

    # Compatibility
    supports_mixed_precision: bool = True
    supports_gradient_checkpointing: bool = True
    min_gpus: int = 1

    def calculate_memory_per_gpu(
        self,
        model_params: int,
        precision_bytes: float,
        optimizer_bytes_per_param: float,
        num_gpus: int,
        offload_optimizer: bool = False,
        offload_params: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate memory per GPU for this ZeRO stage.

        Args:
            model_params: Total model parameters
            precision_bytes: Bytes per parameter for weights (e.g., 2 for bf16)
            optimizer_bytes_per_param: Optimizer state bytes per param
            num_gpus: Number of GPUs
            offload_optimizer: Whether to offload optimizer states
            offload_params: Whether to offload parameters

        Returns:
            Dictionary with memory breakdown in GB
        """
        # Base memory (no sharding)
        weights_gb = (model_params * precision_bytes) / 1e9
        gradients_gb = (model_params * 4) / 1e9  # FP32 gradients
        optimizer_gb = (model_params * optimizer_bytes_per_param) / 1e9

        # Apply sharding
        if self.optimizer_sharded and not offload_optimizer:
            optimizer_gb /= num_gpus
        elif offload_optimizer:
            optimizer_gb = 0  # Offloaded to CPU

        if self.gradient_sharded:
            gradients_gb /= num_gpus

        if self.param_sharded and not offload_params:
            weights_gb /= num_gpus
        elif offload_params:
            weights_gb = weights_gb * 0.1  # Small staging buffer

        return {
            "weights_gb": weights_gb,
            "gradients_gb": gradients_gb,
            "optimizer_gb": optimizer_gb,
            "total_gb": weights_gb + gradients_gb + optimizer_gb,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "name": self.name,
            "description": self.description,
            "optimizer_sharded": self.optimizer_sharded,
            "gradient_sharded": self.gradient_sharded,
            "param_sharded": self.param_sharded,
            "memory_reduction_factor": self.memory_reduction_factor,
            "communication_overhead": self.communication_overhead,
            "supports_optimizer_offload": self.supports_optimizer_offload,
            "supports_param_offload": self.supports_param_offload,
            "supports_nvme_offload": self.supports_nvme_offload,
        }


DEEPSPEED_CONFIGS: Dict[str, DeepSpeedConfig] = {
    "zero0": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_0,
        name="ZeRO Stage 0 (Disabled)",
        description="No ZeRO optimization, standard data parallelism",
        optimizer_sharded=False,
        gradient_sharded=False,
        param_sharded=False,
        memory_reduction_factor=1.0,  # No reduction
        communication_overhead=0.0,
    ),
    "zero1": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_1,
        name="ZeRO Stage 1",
        description="Optimizer state partitioning across data parallel ranks",
        optimizer_sharded=True,
        gradient_sharded=False,
        param_sharded=False,
        memory_reduction_factor=0.25,  # 4× reduction in optimizer states
        communication_overhead=0.05,  # ~5% overhead
    ),
    "zero2": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_2,
        name="ZeRO Stage 2",
        description="Optimizer + gradient partitioning",
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=False,
        memory_reduction_factor=0.125,  # 8× reduction
        communication_overhead=0.10,  # ~10% overhead
        supports_optimizer_offload=True,
    ),
    "zero2_offload": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_2,
        name="ZeRO Stage 2 + Offload",
        description="ZeRO-2 with optimizer offload to CPU",
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=False,
        memory_reduction_factor=0.0625,  # 16× effective reduction
        communication_overhead=0.20,  # ~20% overhead due to CPU transfer
        supports_optimizer_offload=True,
    ),
    "zero3": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_3,
        name="ZeRO Stage 3",
        description="Full model partitioning (optimizer + gradient + parameters)",
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=True,
        memory_reduction_factor=0.0,  # Linear with num_gpus
        communication_overhead=0.20,  # ~20% overhead
        supports_optimizer_offload=True,
        supports_param_offload=True,
    ),
    "zero3_offload": DeepSpeedConfig(
        stage=DeepSpeedZeROStage.STAGE_3,
        name="ZeRO Stage 3 + Offload (ZeRO-Infinity)",
        description="ZeRO-3 with CPU/NVMe offloading for very large models",
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=True,
        memory_reduction_factor=0.0,  # Minimal GPU memory
        communication_overhead=0.40,  # ~40% overhead due to offloading
        supports_optimizer_offload=True,
        supports_param_offload=True,
        supports_nvme_offload=True,
    ),
}


@dataclass
class FSDPConfig:
    """Configuration for FSDP strategy."""

    strategy: FSDPStrategy
    name: str
    description: str

    # Equivalent ZeRO stage for comparison
    equivalent_zero: str

    # Performance characteristics relative to DeepSpeed
    throughput_factor: float  # 1.0 = same as DeepSpeed, >1 = faster

    # Memory characteristics (same as equivalent ZeRO)
    optimizer_sharded: bool
    gradient_sharded: bool
    param_sharded: bool

    # FSDP-specific options
    supports_backward_prefetch: bool = True
    supports_forward_prefetch: bool = True
    supports_cpu_offload: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "name": self.name,
            "description": self.description,
            "equivalent_zero": self.equivalent_zero,
            "throughput_factor": self.throughput_factor,
            "optimizer_sharded": self.optimizer_sharded,
            "gradient_sharded": self.gradient_sharded,
            "param_sharded": self.param_sharded,
        }


FSDP_CONFIGS: Dict[str, FSDPConfig] = {
    "no_shard": FSDPConfig(
        strategy=FSDPStrategy.NO_SHARD,
        name="FSDP No Shard",
        description="Data parallel only, no sharding",
        equivalent_zero="zero0",
        throughput_factor=1.2,  # 20% faster than DeepSpeed for small models
        optimizer_sharded=False,
        gradient_sharded=False,
        param_sharded=False,
    ),
    "shard_grad_op": FSDPConfig(
        strategy=FSDPStrategy.SHARD_GRAD_OP,
        name="FSDP Shard Grad+Op",
        description="Shard gradients and optimizer states",
        equivalent_zero="zero2",
        throughput_factor=1.1,  # 10% faster than DeepSpeed
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=False,
    ),
    "full_shard": FSDPConfig(
        strategy=FSDPStrategy.FULL_SHARD,
        name="FSDP Full Shard",
        description="Full model sharding (equivalent to ZeRO-3)",
        equivalent_zero="zero3",
        throughput_factor=1.0,  # Similar to DeepSpeed at scale
        optimizer_sharded=True,
        gradient_sharded=True,
        param_sharded=True,
    ),
}


@dataclass
class ParallelismConfig:
    """Configuration for multi-dimensional parallelism."""

    tensor_parallel: int = 1  # TP - split tensors across GPUs
    pipeline_parallel: int = 1  # PP - split layers across GPUs
    data_parallel: int = 1  # DP - replicate across GPUs
    expert_parallel: int = 1  # EP - for MoE models

    # Context/Sequence parallelism (for very long sequences)
    context_parallel: int = 1

    @property
    def total_gpus(self) -> int:
        """Total GPUs required."""
        return (
            self.tensor_parallel *
            self.pipeline_parallel *
            self.data_parallel *
            self.expert_parallel *
            self.context_parallel
        )

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate parallelism configuration."""
        if any(p < 1 for p in [self.tensor_parallel, self.pipeline_parallel,
                               self.data_parallel, self.expert_parallel]):
            return False, "All parallelism values must be >= 1"

        # TP should typically be power of 2 and <= 8
        if self.tensor_parallel > 1 and not (self.tensor_parallel & (self.tensor_parallel - 1) == 0):
            return False, "tensor_parallel should be a power of 2"

        if self.tensor_parallel > 8:
            return False, "tensor_parallel > 8 is not recommended due to communication overhead"

        return True, None

    def get_communication_overhead(self) -> float:
        """
        Estimate communication overhead as fraction of compute time.

        Returns:
            Overhead factor (0.0 = no overhead, 0.5 = 50% slower)
        """
        overhead = 0.0

        # DP communication: ~10% per log2(ranks) after 1
        if self.data_parallel > 1:
            dp_ranks = int(math.log2(self.data_parallel))
            overhead += 0.10 * dp_ranks

        # TP communication: ~5% per log2(ranks)
        if self.tensor_parallel > 1:
            tp_ranks = int(math.log2(self.tensor_parallel))
            overhead += 0.05 * tp_ranks

        # PP adds pipeline bubbles: ~15% overhead
        if self.pipeline_parallel > 1:
            overhead += 0.15

        # EP has moderate overhead for all-to-all
        if self.expert_parallel > 1:
            overhead += 0.10

        return min(overhead, 0.60)  # Cap at 60% overhead

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "tp": self.tensor_parallel,
            "pp": self.pipeline_parallel,
            "dp": self.data_parallel,
            "ep": self.expert_parallel,
            "cp": self.context_parallel,
            "total_gpus": self.total_gpus,
        }


@dataclass
class DistributedMemoryEstimate:
    """Memory estimate for distributed training."""

    # Per-GPU memory breakdown
    weights_per_gpu_gb: float
    gradients_per_gpu_gb: float
    optimizer_per_gpu_gb: float
    activations_per_gpu_gb: float
    total_per_gpu_gb: float

    # Configuration
    num_gpus: int
    parallelism: ParallelismConfig
    deepspeed_stage: Optional[str] = None
    fsdp_strategy: Optional[str] = None

    # Efficiency metrics
    communication_overhead: float = 0.0
    effective_throughput_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "per_gpu_memory": {
                "weights_gb": self.weights_per_gpu_gb,
                "gradients_gb": self.gradients_per_gpu_gb,
                "optimizer_gb": self.optimizer_per_gpu_gb,
                "activations_gb": self.activations_per_gpu_gb,
                "total_gb": self.total_per_gpu_gb,
            },
            "num_gpus": self.num_gpus,
            "parallelism": self.parallelism.to_dict(),
            "deepspeed_stage": self.deepspeed_stage,
            "fsdp_strategy": self.fsdp_strategy,
            "communication_overhead": self.communication_overhead,
            "effective_throughput_factor": self.effective_throughput_factor,
        }


def get_deepspeed_config(stage: str) -> DeepSpeedConfig:
    """Get DeepSpeed configuration by stage name."""
    stage = stage.lower()
    if stage in DEEPSPEED_CONFIGS:
        return DEEPSPEED_CONFIGS[stage]
    raise ValueError(f"Unknown DeepSpeed stage: {stage}")


def get_fsdp_config(strategy: str) -> FSDPConfig:
    """Get FSDP configuration by strategy name."""
    strategy = strategy.lower()
    if strategy in FSDP_CONFIGS:
        return FSDP_CONFIGS[strategy]
    raise ValueError(f"Unknown FSDP strategy: {strategy}")


def calculate_distributed_memory(
    model_params: int,
    trainable_params: int,
    precision: str,
    optimizer_bytes_per_param: float,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    num_gpus: int,
    deepspeed_stage: Optional[str] = None,
    fsdp_strategy: Optional[str] = None,
    parallelism: Optional[ParallelismConfig] = None,
    gradient_checkpointing: bool = True,
    offload_optimizer: bool = False,
    offload_params: bool = False,
) -> DistributedMemoryEstimate:
    """
    Calculate memory requirements for distributed training.

    Args:
        model_params: Total model parameters
        trainable_params: Number of trainable parameters
        precision: Weight precision (fp32, bf16, fp16, etc.)
        optimizer_bytes_per_param: Optimizer state bytes per param
        batch_size: Per-device batch size
        seq_length: Sequence length
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        num_gpus: Total number of GPUs
        deepspeed_stage: DeepSpeed ZeRO stage
        fsdp_strategy: FSDP sharding strategy
        parallelism: Multi-dimensional parallelism config
        gradient_checkpointing: Whether gradient checkpointing is enabled
        offload_optimizer: Whether to offload optimizer to CPU
        offload_params: Whether to offload parameters to CPU

    Returns:
        DistributedMemoryEstimate with per-GPU breakdown
    """
    # Precision bytes
    precision_bytes = {
        'fp32': 4, 'float32': 4,
        'bf16': 2, 'bfloat16': 2,
        'fp16': 2, 'float16': 2,
        'int8': 1,
        'int4': 0.5, 'nf4': 0.5,
    }.get(precision.lower(), 2)

    # Set up parallelism
    if parallelism is None:
        parallelism = ParallelismConfig(data_parallel=num_gpus)

    # Base memory calculations
    weights_gb = (model_params * precision_bytes) / 1e9
    gradients_gb = (trainable_params * 4) / 1e9  # FP32 gradients
    optimizer_gb = (trainable_params * optimizer_bytes_per_param) / 1e9

    # Activation memory estimation
    # Per-layer: batch × seq × hidden × 4 (QKV + intermediate) × precision
    if gradient_checkpointing:
        effective_layers = math.sqrt(num_layers)
    else:
        effective_layers = num_layers

    activation_elements = (
        batch_size * seq_length * hidden_size * 4 * effective_layers
    )
    activations_gb = (activation_elements * precision_bytes) / 1e9

    # Apply tensor parallelism
    tp = parallelism.tensor_parallel
    weights_gb /= tp
    activations_gb /= tp

    # Apply DeepSpeed/FSDP sharding
    dp = parallelism.data_parallel

    if deepspeed_stage:
        ds_config = get_deepspeed_config(deepspeed_stage)

        if ds_config.optimizer_sharded and not offload_optimizer:
            optimizer_gb /= dp
        elif offload_optimizer:
            optimizer_gb = 0.0

        if ds_config.gradient_sharded:
            gradients_gb /= dp

        if ds_config.param_sharded and not offload_params:
            weights_gb /= dp
        elif offload_params:
            weights_gb *= 0.1  # Small staging buffer

        communication_overhead = ds_config.communication_overhead
        throughput_factor = 1.0 - communication_overhead

    elif fsdp_strategy:
        fsdp_config = get_fsdp_config(fsdp_strategy)

        if fsdp_config.optimizer_sharded:
            optimizer_gb /= dp

        if fsdp_config.gradient_sharded:
            gradients_gb /= dp

        if fsdp_config.param_sharded:
            weights_gb /= dp

        ds_equiv = get_deepspeed_config(fsdp_config.equivalent_zero)
        communication_overhead = ds_equiv.communication_overhead
        throughput_factor = fsdp_config.throughput_factor * (1.0 - communication_overhead)

    else:
        # Standard data parallel
        communication_overhead = 0.10 * int(math.log2(max(dp, 2)))
        throughput_factor = 1.0 - communication_overhead

    # Add parallelism overhead
    parallelism_overhead = parallelism.get_communication_overhead()
    throughput_factor *= (1.0 - parallelism_overhead)

    total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb

    return DistributedMemoryEstimate(
        weights_per_gpu_gb=weights_gb,
        gradients_per_gpu_gb=gradients_gb,
        optimizer_per_gpu_gb=optimizer_gb,
        activations_per_gpu_gb=activations_gb,
        total_per_gpu_gb=total_gb,
        num_gpus=num_gpus,
        parallelism=parallelism,
        deepspeed_stage=deepspeed_stage,
        fsdp_strategy=fsdp_strategy,
        communication_overhead=communication_overhead + parallelism_overhead,
        effective_throughput_factor=max(0.4, throughput_factor),  # Min 40%
    )


def recommend_distributed_strategy(
    model_params: int,
    available_memory_per_gpu_gb: float,
    num_gpus: int,
    training_type: str = "sft",
) -> Dict[str, Any]:
    """
    Recommend distributed training strategy based on constraints.

    Args:
        model_params: Total model parameters
        available_memory_per_gpu_gb: Memory per GPU
        num_gpus: Number of available GPUs
        training_type: Training type (sft, dpo, ppo, etc.)

    Returns:
        Recommendation dictionary with strategy and rationale
    """
    # Estimate base memory needs (bf16 weights)
    base_weights_gb = (model_params * 2) / 1e9

    # Training type multipliers
    type_multipliers = {
        "sft": 1.0,
        "dpo": 1.7,
        "ppo": 2.8,
        "kto": 1.7,
        "rm": 1.1,
    }
    multiplier = type_multipliers.get(training_type, 1.0)

    total_base_memory = base_weights_gb * multiplier

    recommendations = []

    # Check what fits
    if total_base_memory / num_gpus < available_memory_per_gpu_gb * 0.4:
        # Model fits easily, use simple strategy
        recommendations.append({
            "strategy": "data_parallel",
            "deepspeed_stage": "zero1",
            "fsdp_strategy": "no_shard",
            "rationale": "Model fits comfortably, minimal sharding needed",
            "priority": 1,
        })

    if total_base_memory / num_gpus < available_memory_per_gpu_gb * 0.7:
        # Model fits with moderate optimization
        recommendations.append({
            "strategy": "zero2_or_shard_grad",
            "deepspeed_stage": "zero2",
            "fsdp_strategy": "shard_grad_op",
            "rationale": "Good balance of memory savings and throughput",
            "priority": 2,
        })

    if total_base_memory / num_gpus >= available_memory_per_gpu_gb * 0.7:
        # Need aggressive sharding
        recommendations.append({
            "strategy": "full_shard",
            "deepspeed_stage": "zero3",
            "fsdp_strategy": "full_shard",
            "rationale": "Full sharding needed due to memory constraints",
            "priority": 3,
        })

    if total_base_memory / num_gpus > available_memory_per_gpu_gb:
        # Need offloading
        recommendations.append({
            "strategy": "offload",
            "deepspeed_stage": "zero3_offload",
            "fsdp_strategy": "full_shard",  # with offload
            "rationale": "Model too large, CPU offloading required",
            "priority": 4,
        })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return {
        "model_params": model_params,
        "memory_per_gpu_gb": available_memory_per_gpu_gb,
        "num_gpus": num_gpus,
        "training_type": training_type,
        "estimated_base_memory_gb": total_base_memory,
        "recommendations": recommendations,
        "best_recommendation": recommendations[0] if recommendations else None,
    }
