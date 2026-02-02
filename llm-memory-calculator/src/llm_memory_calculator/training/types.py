"""
Training memory calculation types and dataclasses.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


class TrainingMethod(Enum):
    """Training/fine-tuning method types."""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    FREEZE = "freeze"
    DORA = "dora"
    PISSA = "pissa"


class OptimizerType(Enum):
    """Optimizer types with different memory footprints."""
    ADAMW = "adamw"
    ADAM = "adam"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    SGD = "sgd"
    GALORE = "galore"
    GALORE_8BIT = "galore_8bit"
    APOLLO = "apollo"
    ADAFACTOR = "adafactor"
    LION = "lion"
    LION_8BIT = "lion_8bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    PAGED_LION_32BIT = "paged_lion_32bit"
    ADAM_MINI = "adam_mini"
    Q_ADAM_MINI = "q_adam_mini"
    MUON = "muon"
    BADAM_LAYER = "badam_layer"
    BADAM_RATIO = "badam_ratio"
    ADEMAMIX = "ademamix"
    ADEMAMIX_8BIT = "ademamix_8bit"
    LOMO = "lomo"
    ADALOMO = "adalomo"
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"
    GROKADAMW = "grokadamw"


class DeepSpeedStage(Enum):
    """DeepSpeed ZeRO stages."""
    NONE = "none"
    ZERO2 = "zero2"
    ZERO3 = "zero3"


@dataclass
class TrainingMemoryEstimate:
    """
    Complete training memory breakdown.

    Memory components:
    - weight_memory_gb: Base model weights
    - gradient_memory_gb: Gradients for trainable parameters
    - optimizer_memory_gb: Optimizer states (momentum, variance, etc.)
    - activation_memory_gb: Forward pass activations for backward pass
    - total_memory_gb: Total including framework overhead

    Training configuration:
    - trainable_params: Number of trainable parameters
    - total_params: Total model parameters
    - method: Training method (full, lora, etc.)
    - optimizer: Optimizer type
    - precision: Weight precision
    """

    # Memory breakdown (in GB)
    weight_memory_gb: float
    gradient_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float

    # Training configuration
    trainable_params: int
    total_params: int
    method: str
    optimizer: str
    precision: str
    batch_size: int
    seq_length: int

    # Optional fields
    lora_rank: Optional[int] = None
    gradient_checkpointing: bool = True
    deepspeed_stage: Optional[str] = None
    tensor_parallel: int = 1
    data_parallel: int = 1
    framework_overhead_percent: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weight_memory_gb": self.weight_memory_gb,
            "gradient_memory_gb": self.gradient_memory_gb,
            "optimizer_memory_gb": self.optimizer_memory_gb,
            "activation_memory_gb": self.activation_memory_gb,
            "total_memory_gb": self.total_memory_gb,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "method": self.method,
            "optimizer": self.optimizer,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "lora_rank": self.lora_rank,
            "gradient_checkpointing": self.gradient_checkpointing,
            "deepspeed_stage": self.deepspeed_stage,
            "tensor_parallel": self.tensor_parallel,
            "data_parallel": self.data_parallel,
        }

    def fits_in_memory(self, available_memory_gb: float) -> bool:
        """Check if training fits in given memory."""
        return self.total_memory_gb <= available_memory_gb


@dataclass
class ClusterRecommendation:
    """
    Recommendation for cluster configuration.
    """

    hardware_name: str
    nodes_required: int
    gpus_per_node: int
    total_gpus: int
    memory_per_gpu_gb: float

    # Parallelism strategy
    parallelism: Dict[str, int]  # {"tp": X, "pp": Y, "dp": Z}

    # Estimates
    estimated_throughput_tps: float  # tokens per second
    estimated_cost_per_hour: float
    utilization_percent: float

    # Fit status
    fits: bool
    optimality: str = "good"  # "optimal", "good", "suboptimal"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hardware_name": self.hardware_name,
            "nodes_required": self.nodes_required,
            "gpus_per_node": self.gpus_per_node,
            "total_gpus": self.total_gpus,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "parallelism": self.parallelism,
            "estimated_throughput_tps": self.estimated_throughput_tps,
            "estimated_cost_per_hour": self.estimated_cost_per_hour,
            "utilization_percent": self.utilization_percent,
            "fits": self.fits,
            "optimality": self.optimality,
            "reason": self.reason,
        }


@dataclass
class ClusterFitResult:
    """
    Result of checking if training fits in a specific cluster.
    """

    fits: bool
    memory_per_gpu_gb: float = 0.0
    utilization_percent: float = 0.0
    parallelism: Optional[Dict[str, int]] = None
    reason: Optional[str] = None
    min_gpus_required: int = 1
    estimated_cost_per_hour: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fits": self.fits,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "utilization_percent": self.utilization_percent,
            "parallelism": self.parallelism,
            "reason": self.reason,
            "min_gpus_required": self.min_gpus_required,
            "estimated_cost_per_hour": self.estimated_cost_per_hour,
        }


@dataclass
class TrainingTimeEstimate:
    """
    Training time and cost estimation.
    """

    # Basic metrics
    total_steps: int
    tokens_per_second: float
    estimated_hours: float
    estimated_cost: float

    # Hardware config
    hardware: str
    num_gpus: int
    parallelism: Optional[Dict[str, int]] = None

    # Efficiency metrics
    model_flops_utilization: float = 0.4  # MFU
    flops_per_step: Optional[float] = None

    # Training config
    batch_size: int = 1
    gradient_accumulation: int = 1
    seq_length: int = 2048
    epochs: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_steps": self.total_steps,
            "tokens_per_second": self.tokens_per_second,
            "estimated_hours": self.estimated_hours,
            "estimated_cost": self.estimated_cost,
            "hardware": self.hardware,
            "num_gpus": self.num_gpus,
            "parallelism": self.parallelism,
            "model_flops_utilization": self.model_flops_utilization,
            "flops_per_step": self.flops_per_step,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "seq_length": self.seq_length,
            "epochs": self.epochs,
        }
