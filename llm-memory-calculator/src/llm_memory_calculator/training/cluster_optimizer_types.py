"""
Data structures for Cluster Optimization.

This module defines the core data structures used by the ClusterOptimizer
for cluster selection and optimal cluster design algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class OptimizationTarget(Enum):
    """Optimization objectives for cluster selection."""
    TCO = "tco"                    # Total Cost of Ownership
    THROUGHPUT = "throughput"      # Tokens per second
    MFU = "mfu"                    # Model FLOPS Utilization
    COST_PER_TOKEN = "cost_per_token"  # Cost per million tokens
    LATENCY = "latency"            # Step time in ms
    PARETO = "pareto"              # Multi-objective Pareto frontier


class PricingTier(Enum):
    """Cloud pricing tiers."""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED_1YR = "reserved_1yr"
    RESERVED_3YR = "reserved_3yr"


@dataclass
class ClusterDefinition:
    """
    Definition of an available cluster.

    Represents a specific GPU cluster configuration that can be used for training.
    """
    name: str
    gpu_type: str
    num_gpus: int
    gpus_per_node: int = 8
    intra_node_bandwidth_gbps: float = 600.0   # NVLink
    inter_node_bandwidth_gbps: float = 400.0   # InfiniBand
    provider: str = "unknown"
    region: str = "unknown"
    hourly_rate_per_gpu: float = 0.0
    spot_rate_per_gpu: Optional[float] = None
    reserved_rate_per_gpu: Optional[float] = None

    # Additional metadata
    availability_zone: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return max(1, (self.num_gpus + self.gpus_per_node - 1) // self.gpus_per_node)

    @property
    def total_hourly_rate(self) -> float:
        """Total hourly rate for the cluster."""
        return self.hourly_rate_per_gpu * self.num_gpus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'gpu_type': self.gpu_type,
            'num_gpus': self.num_gpus,
            'gpus_per_node': self.gpus_per_node,
            'num_nodes': self.num_nodes,
            'intra_node_bandwidth_gbps': self.intra_node_bandwidth_gbps,
            'inter_node_bandwidth_gbps': self.inter_node_bandwidth_gbps,
            'provider': self.provider,
            'region': self.region,
            'hourly_rate_per_gpu': self.hourly_rate_per_gpu,
            'spot_rate_per_gpu': self.spot_rate_per_gpu,
            'total_hourly_rate': self.total_hourly_rate,
        }


@dataclass
class TrainingJobSpec:
    """
    Training job specification.

    Defines all parameters for a training job including model, dataset,
    training configuration, and constraints.
    """
    # Model specification
    model: str                          # HF model ID or model config name

    # Dataset specification
    dataset_tokens: int = 0             # Total tokens in dataset
    dataset_samples: int = 0            # Total samples (if tokens not specified)
    avg_sequence_length: int = 2048     # Average sequence length
    num_epochs: float = 1.0             # Number of training epochs

    # Training configuration
    training_type: str = "sft"          # sft, dpo, ppo, kto, rm, grpo, ipo, pt
    method: str = "full"                # full, lora, qlora, dora, pissa, freeze
    batch_size: Optional[int] = None    # Per-GPU batch size (auto if None)
    optimizer: str = "adamw"            # Optimizer type
    precision: str = "bf16"             # Training precision

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: str = "all"            # Modules to apply LoRA to
    use_rslora: bool = False            # Use rank-stabilized LoRA
    use_lora_plus: bool = False         # Use LoRA+ (different LR for A/B)
    lora_plus_lr_ratio: float = 16.0    # LoRA+ learning rate ratio (B/A)

    # Freeze fine-tuning configuration
    freeze_trainable_layers: int = 2    # Number of trainable layers (from end)
    freeze_trainable_modules: str = "all"  # Modules to train: "all", "mlp", "attn"
    freeze_extra_modules: Optional[str] = None  # Additional modules to train

    # GaLore optimizer configuration
    use_galore: bool = False
    galore_rank: int = 128
    galore_target: str = "all"          # Modules for GaLore: "all", "attn", "mlp"
    galore_scale: float = 1.0
    galore_layerwise: bool = False

    # BAdam optimizer configuration
    use_badam: bool = False
    badam_mode: str = "layer"           # "layer" or "ratio"
    badam_switch_mode: str = "ascending"  # "ascending", "descending", "random"
    badam_update_ratio: float = 0.05

    # Additional training options
    flash_attn: str = "auto"            # "auto", "fa2", "sdpa", "disabled"
    quantization_bit: int = 4           # Quantization bits: 2, 3, 4, 5, 6, 8
    quantization_method: str = "bitsandbytes"  # Quantization backend

    # Distributed configuration
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    use_fsdp: bool = False              # Use PyTorch FSDP instead of DeepSpeed

    # Constraints
    max_budget_usd: Optional[float] = None
    max_time_hours: Optional[float] = None
    min_throughput_tps: Optional[float] = None
    max_memory_per_gpu_gb: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens to process."""
        if self.dataset_tokens > 0:
            return int(self.dataset_tokens * self.num_epochs)
        elif self.dataset_samples > 0:
            return int(self.dataset_samples * self.avg_sequence_length * self.num_epochs)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': self.model,
            'dataset_tokens': self.dataset_tokens,
            'dataset_samples': self.dataset_samples,
            'avg_sequence_length': self.avg_sequence_length,
            'num_epochs': self.num_epochs,
            'total_tokens': self.total_tokens,
            'training_type': self.training_type,
            'method': self.method,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing,
            'constraints': {
                'max_budget_usd': self.max_budget_usd,
                'max_time_hours': self.max_time_hours,
                'min_throughput_tps': self.min_throughput_tps,
                'max_memory_per_gpu_gb': self.max_memory_per_gpu_gb,
            },
        }


@dataclass
class TCOBreakdown:
    """
    Total Cost of Ownership breakdown.

    Provides detailed cost breakdown for a training run.
    """
    # Primary cost components
    gpu_compute_cost: float = 0.0      # GPU compute hours cost
    power_cost: float = 0.0            # Electricity cost
    network_cost: float = 0.0          # Network egress/ingress cost
    storage_cost: float = 0.0          # Storage for checkpoints/logs

    # Overhead and adjustments
    operations_overhead: float = 0.0   # 15% typical overhead
    spot_savings: float = 0.0          # Savings from spot instances
    reserved_savings: float = 0.0      # Savings from reserved instances

    # Totals
    total_cost: float = 0.0
    cost_per_million_tokens: float = 0.0
    cost_per_sample: float = 0.0

    # Metadata
    pricing_tier: str = "on_demand"
    provider: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cost_components': {
                'gpu_compute': self.gpu_compute_cost,
                'power': self.power_cost,
                'network': self.network_cost,
                'storage': self.storage_cost,
                'operations_overhead': self.operations_overhead,
            },
            'adjustments': {
                'spot_savings': self.spot_savings,
                'reserved_savings': self.reserved_savings,
            },
            'totals': {
                'total_cost_usd': self.total_cost,
                'cost_per_million_tokens': self.cost_per_million_tokens,
                'cost_per_sample': self.cost_per_sample,
            },
            'metadata': {
                'pricing_tier': self.pricing_tier,
                'provider': self.provider,
            },
        }


@dataclass
class ParallelismStrategy:
    """
    Complete parallelism configuration for training.
    """
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    zero_stage: int = 0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True

    @property
    def total_gpus(self) -> int:
        """Total GPUs required."""
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel * self.expert_parallel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tensor_parallel': self.tensor_parallel,
            'pipeline_parallel': self.pipeline_parallel,
            'data_parallel': self.data_parallel,
            'expert_parallel': self.expert_parallel,
            'zero_stage': self.zero_stage,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing,
            'total_gpus': self.total_gpus,
        }


@dataclass
class ClusterRecommendationResult:
    """
    Complete cluster recommendation with all metrics.

    Result from cluster selection or optimal cluster design algorithms.
    """
    # Cluster configuration
    cluster: ClusterDefinition
    parallelism: ParallelismStrategy

    # Performance metrics
    tokens_per_second: float
    step_time_ms: float
    memory_per_gpu_gb: float
    mfu: float

    # Training duration
    training_hours: float
    training_steps: int = 0

    # Cost analysis
    tco_breakdown: TCOBreakdown = field(default_factory=TCOBreakdown)

    # Generated configurations
    llamafactory_config: Dict[str, Any] = field(default_factory=dict)
    deepspeed_config: Optional[Dict[str, Any]] = None

    # Scoring and ranking
    score: float = 0.0
    rank: int = 0

    # Additional metadata
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def cost_per_million_tokens(self) -> float:
        """Cost per million tokens trained."""
        return self.tco_breakdown.cost_per_million_tokens

    @property
    def effective_throughput(self) -> float:
        """Effective throughput accounting for all GPUs."""
        return self.tokens_per_second

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cluster': self.cluster.to_dict(),
            'parallelism': self.parallelism.to_dict(),
            'performance': {
                'tokens_per_second': self.tokens_per_second,
                'step_time_ms': self.step_time_ms,
                'memory_per_gpu_gb': self.memory_per_gpu_gb,
                'mfu': self.mfu,
            },
            'training': {
                'training_hours': self.training_hours,
                'training_steps': self.training_steps,
            },
            'cost': self.tco_breakdown.to_dict(),
            'configs': {
                'llamafactory': self.llamafactory_config,
                'deepspeed': self.deepspeed_config,
            },
            'scoring': {
                'score': self.score,
                'rank': self.rank,
            },
            'notes': self.notes,
            'warnings': self.warnings,
        }


@dataclass
class OptimalClusterDesignResult:
    """
    Result from optimal cluster design algorithm.

    Contains the optimal configuration and alternatives.
    """
    # Best configuration
    optimal_config: ClusterRecommendationResult

    # Alternative configurations
    alternatives: List[ClusterRecommendationResult] = field(default_factory=list)

    # Pareto frontier (for PARETO optimization)
    pareto_frontier: List[ClusterRecommendationResult] = field(default_factory=list)

    # Search metadata
    configs_evaluated: int = 0
    search_time_seconds: float = 0.0

    # Warnings and notes
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optimal': self.optimal_config.to_dict(),
            'alternatives': [alt.to_dict() for alt in self.alternatives],
            'pareto_frontier_size': len(self.pareto_frontier),
            'metadata': {
                'configs_evaluated': self.configs_evaluated,
                'search_time_seconds': self.search_time_seconds,
            },
            'notes': self.notes,
        }
