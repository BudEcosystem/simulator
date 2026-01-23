"""
Training Type Configurations.

Defines memory and resource requirements for different training stages:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization / RLHF)
- KTO (Kahneman-Tversky Optimization)
- RM (Reward Modeling)
- PT (Pre-Training)

Based on analysis of LlamaFactory and published benchmarks.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class TrainingStage(Enum):
    """Training stage types supported by the simulator."""
    PRETRAINING = "pt"
    SFT = "sft"
    DPO = "dpo"
    KTO = "kto"
    PPO = "ppo"
    REWARD_MODELING = "rm"


class DPOLossType(Enum):
    """DPO loss function variants."""
    SIGMOID = "sigmoid"      # Standard DPO, requires ref model
    HINGE = "hinge"          # Margin-based, requires ref model
    IPO = "ipo"              # Identity PO, requires ref model
    KTO_PAIR = "kto_pair"    # KT pairing, requires ref model
    ORPO = "orpo"            # Reference-free odds ratio
    SIMPO = "simpo"          # Reference-free simple PO


@dataclass
class TrainingStageConfig:
    """Configuration for a training stage."""

    # Core properties
    stage: TrainingStage
    name: str
    description: str

    # Model requirements
    num_model_instances: int
    requires_reference_model: bool
    requires_reward_model: bool
    requires_value_head: bool

    # Memory factors (relative to base model)
    base_memory_multiplier: float
    gradient_memory_multiplier: float  # Relative to trainable params
    optimizer_memory_multiplier: float  # Relative to trainable params

    # Training characteristics
    supports_peft: bool = True
    supports_quantization: bool = True
    recommended_batch_size_range: tuple = (1, 16)

    # Data requirements
    data_format: str = "instruction"  # instruction, pairwise, single
    uses_pairwise_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "name": self.name,
            "description": self.description,
            "num_model_instances": self.num_model_instances,
            "requires_reference_model": self.requires_reference_model,
            "requires_reward_model": self.requires_reward_model,
            "requires_value_head": self.requires_value_head,
            "base_memory_multiplier": self.base_memory_multiplier,
            "gradient_memory_multiplier": self.gradient_memory_multiplier,
            "optimizer_memory_multiplier": self.optimizer_memory_multiplier,
            "supports_peft": self.supports_peft,
            "supports_quantization": self.supports_quantization,
            "recommended_batch_size_range": self.recommended_batch_size_range,
            "data_format": self.data_format,
            "uses_pairwise_data": self.uses_pairwise_data,
        }


# Training stage configurations
TRAINING_STAGE_CONFIGS: Dict[str, TrainingStageConfig] = {
    "pt": TrainingStageConfig(
        stage=TrainingStage.PRETRAINING,
        name="Pre-Training",
        description="Causal language modeling on raw text sequences",
        num_model_instances=1,
        requires_reference_model=False,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.0,
        gradient_memory_multiplier=1.0,
        optimizer_memory_multiplier=1.0,
        supports_peft=True,
        supports_quantization=False,  # Usually not for pretraining
        recommended_batch_size_range=(4, 64),
        data_format="raw_text",
        uses_pairwise_data=False,
    ),
    "sft": TrainingStageConfig(
        stage=TrainingStage.SFT,
        name="Supervised Fine-Tuning",
        description="Fine-tuning on instruction-response pairs",
        num_model_instances=1,
        requires_reference_model=False,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.0,
        gradient_memory_multiplier=1.0,
        optimizer_memory_multiplier=1.0,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 32),
        data_format="instruction",
        uses_pairwise_data=False,
    ),
    "dpo": TrainingStageConfig(
        stage=TrainingStage.DPO,
        name="Direct Preference Optimization",
        description="Preference learning with policy and reference models",
        num_model_instances=2,  # policy + reference
        requires_reference_model=True,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.7,  # ref model in eval mode (less memory)
        gradient_memory_multiplier=1.0,  # only policy trains
        optimizer_memory_multiplier=1.0,  # only policy optimized
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 8),
        data_format="pairwise",
        uses_pairwise_data=True,
    ),
    "dpo_orpo": TrainingStageConfig(
        stage=TrainingStage.DPO,
        name="ORPO (Reference-Free DPO)",
        description="Odds Ratio Preference Optimization without reference model",
        num_model_instances=1,
        requires_reference_model=False,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.0,
        gradient_memory_multiplier=1.0,
        optimizer_memory_multiplier=1.0,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 8),
        data_format="pairwise",
        uses_pairwise_data=True,
    ),
    "dpo_simpo": TrainingStageConfig(
        stage=TrainingStage.DPO,
        name="SimPO (Simple Preference Optimization)",
        description="Simplified preference optimization without reference model",
        num_model_instances=1,
        requires_reference_model=False,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.0,
        gradient_memory_multiplier=1.0,
        optimizer_memory_multiplier=1.0,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 8),
        data_format="pairwise",
        uses_pairwise_data=True,
    ),
    "kto": TrainingStageConfig(
        stage=TrainingStage.KTO,
        name="Kahneman-Tversky Optimization",
        description="Preference learning with unpaired data using KTO loss",
        num_model_instances=2,  # policy + reference
        requires_reference_model=True,
        requires_reward_model=False,
        requires_value_head=False,
        base_memory_multiplier=1.7,
        gradient_memory_multiplier=1.0,
        optimizer_memory_multiplier=1.0,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 8),
        data_format="single_tagged",  # Single examples with desirable/undesirable tags
        uses_pairwise_data=False,
    ),
    "ppo": TrainingStageConfig(
        stage=TrainingStage.PPO,
        name="PPO / RLHF",
        description="Proximal Policy Optimization with reward model",
        num_model_instances=3,  # policy + reference + reward
        requires_reference_model=True,
        requires_reward_model=True,
        requires_value_head=True,
        base_memory_multiplier=2.8,  # 3 models with some efficiency
        gradient_memory_multiplier=1.2,  # value head adds gradients
        optimizer_memory_multiplier=1.2,  # value head adds optimizer states
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 4),  # Memory-intensive
        data_format="prompt",
        uses_pairwise_data=False,
    ),
    "ppo_lora_reward": TrainingStageConfig(
        stage=TrainingStage.PPO,
        name="PPO with LoRA Reward Model",
        description="PPO with reward model as LoRA adapter (more efficient)",
        num_model_instances=2,  # policy + reference (reward as LoRA)
        requires_reference_model=True,
        requires_reward_model=True,  # As LoRA adapter
        requires_value_head=True,
        base_memory_multiplier=2.0,
        gradient_memory_multiplier=1.1,
        optimizer_memory_multiplier=1.1,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 8),
        data_format="prompt",
        uses_pairwise_data=False,
    ),
    "rm": TrainingStageConfig(
        stage=TrainingStage.REWARD_MODELING,
        name="Reward Modeling",
        description="Training a reward model for RLHF",
        num_model_instances=1,
        requires_reference_model=False,
        requires_reward_model=False,
        requires_value_head=True,  # Adds value/reward head
        base_memory_multiplier=1.1,  # Value head overhead
        gradient_memory_multiplier=1.1,
        optimizer_memory_multiplier=1.1,
        supports_peft=True,
        supports_quantization=True,
        recommended_batch_size_range=(1, 16),
        data_format="pairwise",
        uses_pairwise_data=True,
    ),
}


@dataclass
class DPOConfig:
    """Configuration specific to DPO training."""

    loss_type: DPOLossType = DPOLossType.SIGMOID
    beta: float = 0.1  # KL penalty strength (0.01 - 1.0)
    label_smoothing: float = 0.0  # Only for sigmoid loss
    simpo_gamma: float = 0.5  # Only for SimPO

    # Auxiliary losses
    ftx_weight: float = 0.0  # FTX auxiliary SFT loss
    bco_weight: float = 0.0  # BCO loss weight

    @property
    def requires_reference_model(self) -> bool:
        """Check if this loss type requires a reference model."""
        return self.loss_type not in (DPOLossType.ORPO, DPOLossType.SIMPO)

    def get_memory_multiplier(self) -> float:
        """Get memory multiplier based on loss type."""
        if self.requires_reference_model:
            return 1.7  # Policy + ref (ref in eval mode)
        return 1.0  # Policy only


@dataclass
class KTOConfig:
    """Configuration specific to KTO training."""

    beta: float = 0.1  # KL penalty strength
    chosen_weight: float = 1.0  # Weight for desirable examples
    rejected_weight: float = 1.0  # Weight for undesirable examples
    ftx_weight: float = 0.0  # Auxiliary SFT loss


@dataclass
class PPOConfig:
    """Configuration specific to PPO/RLHF training."""

    buffer_size: int = 1  # Number of mini-batches in experience buffer
    ppo_epochs: int = 4  # PPO optimization epochs per step
    clip_range: float = 0.2  # PPO clip range
    target_kl: Optional[float] = None  # Target KL divergence
    score_norm: bool = False  # Normalize reward scores
    whiten_rewards: bool = False  # Whiten reward values

    # Reward model configuration
    reward_model_type: str = "full"  # full, lora, api

    def get_reward_model_memory_factor(self) -> float:
        """Get additional memory factor for reward model."""
        if self.reward_model_type == "full":
            return 1.0  # Full additional model
        elif self.reward_model_type == "lora":
            return 0.1  # Just LoRA adapters
        else:  # api
            return 0.0  # External, no memory


@dataclass
class TrainingStageEstimate:
    """
    Memory estimation for a specific training stage.
    """

    # Stage info
    stage: str
    stage_name: str

    # Model instances
    num_model_instances: int
    model_descriptions: List[str]

    # Memory breakdown per GPU (in GB)
    policy_model_memory_gb: float
    reference_model_memory_gb: float = 0.0
    reward_model_memory_gb: float = 0.0
    value_head_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    total_memory_gb: float = 0.0

    # Configuration used
    method: str = "full"
    optimizer: str = "adamw"
    precision: str = "bf16"
    deepspeed_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "num_model_instances": self.num_model_instances,
            "model_descriptions": self.model_descriptions,
            "memory_breakdown": {
                "policy_model_gb": self.policy_model_memory_gb,
                "reference_model_gb": self.reference_model_memory_gb,
                "reward_model_gb": self.reward_model_memory_gb,
                "value_head_gb": self.value_head_memory_gb,
                "gradient_gb": self.gradient_memory_gb,
                "optimizer_gb": self.optimizer_memory_gb,
                "activation_gb": self.activation_memory_gb,
                "total_gb": self.total_memory_gb,
            },
            "configuration": {
                "method": self.method,
                "optimizer": self.optimizer,
                "precision": self.precision,
                "deepspeed_stage": self.deepspeed_stage,
            },
        }


def get_training_stage_config(
    stage: str,
    dpo_loss_type: Optional[str] = None,
    ppo_reward_type: Optional[str] = None,
) -> TrainingStageConfig:
    """
    Get configuration for a training stage.

    Args:
        stage: Training stage (pt, sft, dpo, kto, ppo, rm)
        dpo_loss_type: For DPO, specify loss type (sigmoid, orpo, simpo, etc.)
        ppo_reward_type: For PPO, specify reward model type (full, lora, api)

    Returns:
        TrainingStageConfig for the specified stage
    """
    stage = stage.lower()

    # Handle DPO variants
    if stage == "dpo" and dpo_loss_type:
        dpo_loss_type = dpo_loss_type.lower()
        if dpo_loss_type in ("orpo",):
            return TRAINING_STAGE_CONFIGS["dpo_orpo"]
        elif dpo_loss_type in ("simpo",):
            return TRAINING_STAGE_CONFIGS["dpo_simpo"]

    # Handle PPO variants
    if stage == "ppo" and ppo_reward_type:
        ppo_reward_type = ppo_reward_type.lower()
        if ppo_reward_type in ("lora",):
            return TRAINING_STAGE_CONFIGS["ppo_lora_reward"]

    # Standard lookup
    if stage in TRAINING_STAGE_CONFIGS:
        return TRAINING_STAGE_CONFIGS[stage]

    raise ValueError(
        f"Unknown training stage: {stage}. "
        f"Valid stages: {list(TRAINING_STAGE_CONFIGS.keys())}"
    )


def list_training_stages() -> List[Dict[str, Any]]:
    """List all supported training stages with descriptions."""
    return [
        {
            "stage": key,
            "name": config.name,
            "description": config.description,
            "num_models": config.num_model_instances,
            "requires_reference": config.requires_reference_model,
            "requires_reward": config.requires_reward_model,
        }
        for key, config in TRAINING_STAGE_CONFIGS.items()
    ]
