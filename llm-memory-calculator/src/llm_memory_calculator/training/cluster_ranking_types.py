"""
Cluster Ranking Data Types.

This module defines data structures for:
- Cluster ranking with batch optimization
- Minimum cluster requirements prediction
- Comprehensive LlamaFactory configuration

These types support the new cluster ranking and requirements prediction functions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from .cluster_optimizer_types import (
    ClusterDefinition,
    ParallelismStrategy,
    TCOBreakdown,
)


class RankingMetric(Enum):
    """Metrics for cluster ranking."""
    THROUGHPUT = "throughput"       # Tokens per second
    ETA = "eta"                     # Estimated time to completion
    COST = "cost"                   # Total training cost
    COST_PER_TOKEN = "cost_per_token"  # Cost per million tokens
    MFU = "mfu"                     # Model FLOPS Utilization
    COMPOSITE = "composite"         # Weighted combination


class ClusterTopology(Enum):
    """Network topologies for clusters."""
    FAT_TREE = "fat-tree"
    DRAGONFLY = "dragonfly"
    TORUS = "torus"
    RING = "ring"
    FULL_MESH = "full_mesh"
    SINGLE_NODE = "single_node"


@dataclass
class GPURequirement:
    """GPU-specific requirements for a training configuration."""
    gpu_type: str
    min_gpus: int
    max_batch_size: int
    memory_per_gpu_gb: float
    estimated_throughput_tps: float
    estimated_training_hours: float
    estimated_cost_usd: float
    recommended_parallelism: ParallelismStrategy
    feasible: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "min_gpus": self.min_gpus,
            "max_batch_size": self.max_batch_size,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "estimated_throughput_tps": self.estimated_throughput_tps,
            "estimated_training_hours": self.estimated_training_hours,
            "estimated_cost_usd": self.estimated_cost_usd,
            "recommended_parallelism": self.recommended_parallelism.to_dict(),
            "feasible": self.feasible,
            "notes": self.notes,
        }


@dataclass
class MemoryBreakdownDetails:
    """Detailed memory breakdown for training."""
    # Weight memory
    weight_memory_gb: float = 0.0
    weight_memory_per_gpu_gb: float = 0.0

    # Gradient memory
    gradient_memory_gb: float = 0.0
    gradient_memory_per_gpu_gb: float = 0.0

    # Optimizer state memory
    optimizer_memory_gb: float = 0.0
    optimizer_memory_per_gpu_gb: float = 0.0

    # Activation memory
    activation_memory_gb: float = 0.0
    activation_memory_per_gpu_gb: float = 0.0

    # Reference model (for DPO, PPO, KTO)
    reference_model_memory_gb: float = 0.0

    # Reward model (for PPO)
    reward_model_memory_gb: float = 0.0

    # Critic model (for PPO)
    critic_model_memory_gb: float = 0.0

    # Total per GPU
    total_per_gpu_gb: float = 0.0

    # Overhead
    system_overhead_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": {
                "total_gb": self.weight_memory_gb,
                "per_gpu_gb": self.weight_memory_per_gpu_gb,
            },
            "gradients": {
                "total_gb": self.gradient_memory_gb,
                "per_gpu_gb": self.gradient_memory_per_gpu_gb,
            },
            "optimizer": {
                "total_gb": self.optimizer_memory_gb,
                "per_gpu_gb": self.optimizer_memory_per_gpu_gb,
            },
            "activations": {
                "total_gb": self.activation_memory_gb,
                "per_gpu_gb": self.activation_memory_per_gpu_gb,
            },
            "reference_model_gb": self.reference_model_memory_gb,
            "reward_model_gb": self.reward_model_memory_gb,
            "critic_model_gb": self.critic_model_memory_gb,
            "total_per_gpu_gb": self.total_per_gpu_gb,
            "system_overhead_gb": self.system_overhead_gb,
        }


@dataclass
class ClusterRankingResult:
    """Result of cluster ranking with batch optimization."""
    # Cluster configuration
    cluster: ClusterDefinition
    optimal_batch_size: int
    parallelism: ParallelismStrategy

    # Performance metrics
    tokens_per_second: float
    step_time_ms: float
    mfu: float
    memory_per_gpu_gb: float

    # Training estimates
    estimated_training_hours: float
    estimated_eta: str  # Human readable (e.g., "2 days 5 hours")
    estimated_cost_usd: float
    cost_per_million_tokens: float
    training_steps: int = 0

    # Cost analysis
    tco_breakdown: Optional[TCOBreakdown] = None

    # Rankings
    throughput_rank: int = 0
    eta_rank: int = 0
    cost_rank: int = 0
    composite_score: float = 0.0

    # Generated configurations
    llamafactory_config: Dict[str, Any] = field(default_factory=dict)
    deepspeed_config: Optional[Dict[str, Any]] = None

    # Additional metadata
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster": self.cluster.to_dict(),
            "optimal_batch_size": self.optimal_batch_size,
            "parallelism": self.parallelism.to_dict(),
            "performance": {
                "tokens_per_second": self.tokens_per_second,
                "step_time_ms": self.step_time_ms,
                "mfu": self.mfu,
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
            },
            "training_estimates": {
                "hours": self.estimated_training_hours,
                "eta": self.estimated_eta,
                "steps": self.training_steps,
            },
            "cost": {
                "total_usd": self.estimated_cost_usd,
                "per_million_tokens": self.cost_per_million_tokens,
                "breakdown": self.tco_breakdown.to_dict() if self.tco_breakdown else None,
            },
            "rankings": {
                "throughput_rank": self.throughput_rank,
                "eta_rank": self.eta_rank,
                "cost_rank": self.cost_rank,
                "composite_score": self.composite_score,
            },
            "configs": {
                "llamafactory": self.llamafactory_config,
                "deepspeed": self.deepspeed_config,
            },
            "notes": self.notes,
            "warnings": self.warnings,
        }


@dataclass
class MinimumClusterRequirements:
    """Minimum cluster requirements output."""
    # Minimum hardware requirements
    min_gpus: int
    min_gpu_memory_gb: float
    min_total_memory_gb: float

    # Per-GPU type requirements
    requirements_by_gpu: Dict[str, GPURequirement] = field(default_factory=dict)

    # Network requirements
    min_inter_node_bandwidth_gbps: float = 0.0
    min_intra_node_bandwidth_gbps: float = 0.0

    # Recommended configuration
    recommended_topology: str = "fat-tree"
    recommended_parallelism: Optional[ParallelismStrategy] = None
    recommended_zero_stage: int = 0
    recommended_method: str = "lora"

    # Memory breakdown
    memory_breakdown: Optional[MemoryBreakdownDetails] = None

    # Feasibility notes
    feasibility_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_hardware": {
                "min_gpus": self.min_gpus,
                "min_gpu_memory_gb": self.min_gpu_memory_gb,
                "min_total_memory_gb": self.min_total_memory_gb,
            },
            "requirements_by_gpu": {
                k: v.to_dict() for k, v in self.requirements_by_gpu.items()
            },
            "network": {
                "min_inter_node_bandwidth_gbps": self.min_inter_node_bandwidth_gbps,
                "min_intra_node_bandwidth_gbps": self.min_intra_node_bandwidth_gbps,
            },
            "recommendations": {
                "topology": self.recommended_topology,
                "parallelism": self.recommended_parallelism.to_dict() if self.recommended_parallelism else None,
                "zero_stage": self.recommended_zero_stage,
                "method": self.recommended_method,
            },
            "memory_breakdown": self.memory_breakdown.to_dict() if self.memory_breakdown else None,
            "feasibility_notes": self.feasibility_notes,
            "warnings": self.warnings,
        }


@dataclass
class PPOModelConfig:
    """Configuration for a single model in PPO training."""
    model_path: str
    adapter_path: Optional[str] = None
    trainable: bool = False
    eval_mode: bool = False
    quantization_bit: Optional[int] = None
    finetuning_type: str = "full"
    memory_estimate_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "trainable": self.trainable,
            "eval_mode": self.eval_mode,
            "quantization_bit": self.quantization_bit,
            "finetuning_type": self.finetuning_type,
            "memory_estimate_gb": self.memory_estimate_gb,
        }


@dataclass
class ComprehensiveLlamaFactoryConfig:
    """Complete LlamaFactory configuration with best practices."""
    # Main configuration
    llamafactory_yaml: Dict[str, Any] = field(default_factory=dict)
    deepspeed_json: Optional[Dict[str, Any]] = None
    accelerate_yaml: Optional[Dict[str, Any]] = None

    # PPO-specific (multiple models)
    ppo_actor_config: Optional[Dict[str, Any]] = None
    ppo_reward_config: Optional[Dict[str, Any]] = None
    ppo_reference_config: Optional[Dict[str, Any]] = None
    vllm_inference_config: Optional[Dict[str, Any]] = None

    # Metadata
    best_practices_applied: List[str] = field(default_factory=list)
    optimization_focus: str = "balanced"

    # Launch commands
    torchrun_command: str = ""
    deepspeed_command: str = ""

    # Expected performance
    expected_throughput_tps: float = 0.0
    expected_training_hours: float = 0.0
    expected_cost_usd: float = 0.0

    # Warnings and notes
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llamafactory_yaml": self.llamafactory_yaml,
            "deepspeed_json": self.deepspeed_json,
            "accelerate_yaml": self.accelerate_yaml,
            "ppo_configs": {
                "actor": self.ppo_actor_config,
                "reward": self.ppo_reward_config,
                "reference": self.ppo_reference_config,
                "vllm_inference": self.vllm_inference_config,
            },
            "best_practices_applied": self.best_practices_applied,
            "optimization_focus": self.optimization_focus,
            "launch_commands": {
                "torchrun": self.torchrun_command,
                "deepspeed": self.deepspeed_command,
            },
            "expected_performance": {
                "throughput_tps": self.expected_throughput_tps,
                "training_hours": self.expected_training_hours,
                "cost_usd": self.expected_cost_usd,
            },
            "notes": self.notes,
            "warnings": self.warnings,
        }


@dataclass
class ClusterRankingRequest:
    """Request for cluster ranking."""
    model: str
    training_type: str
    clusters: List[ClusterDefinition]
    dataset_tokens: int
    avg_sequence_length: int = 4096
    num_epochs: float = 1.0
    method: str = "auto"
    optimizer: str = "adamw"
    precision: str = "bf16"
    max_training_hours: Optional[float] = None
    max_budget_usd: Optional[float] = None
    sort_by: str = "throughput"
    return_top_k: int = 10


@dataclass
class ClusterRequirementsRequest:
    """Request for cluster requirements prediction."""
    training_type: str
    model: str
    dtype: str
    optimizer: str
    dataset_size_tokens: int
    batch_size: int
    network_bandwidth_gbps: Optional[float] = None
    inter_chip_bandwidth_gbps: Optional[float] = None
    topology: str = "fat-tree"
    max_training_hours: Optional[float] = None
    target_mfu: float = 0.4


def format_eta(hours: float) -> str:
    """Format hours into human-readable ETA string."""
    if hours <= 0:
        return "N/A"
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minutes"
    if hours < 24:
        h = int(hours)
        m = int((hours - h) * 60)
        if m > 0:
            return f"{h} hours {m} minutes"
        return f"{h} hours"
    if hours < 24 * 7:
        days = int(hours / 24)
        h = int(hours % 24)
        if h > 0:
            return f"{days} days {h} hours"
        return f"{days} days"
    weeks = hours / (24 * 7)
    if weeks < 4:
        return f"{weeks:.1f} weeks"
    months = hours / (24 * 30)
    return f"{months:.1f} months"
