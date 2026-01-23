"""
Training Stage Implementations for GenZ Training Simulation.

Implements the computation patterns for different training stages:
- SFT (Supervised Fine-Tuning): Standard forward-backward-update
- DPO (Direct Preference Optimization): Policy + Reference forward, Policy backward
- PPO (Proximal Policy Optimization): Actor + Critic + Reward forward, Actor + Critic backward
- KTO (Kahneman-Tversky Optimization): Similar to DPO but single-turn
- RM (Reward Modeling): Classifier head training
- PT (Pre-Training): Same as SFT

Each stage has different:
- Number of forward passes required
- Models loaded in memory
- Communication patterns
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .training_modeling import (
    training_modeling,
    TrainingModelingOutput,
)


class TrainingStageType(Enum):
    """Supported training stages."""
    SFT = "sft"
    DPO = "dpo"
    PPO = "ppo"
    PPO_DETAILED = "ppo_detailed"  # PPO with detailed phase tracking
    KTO = "kto"
    RM = "rm"
    PT = "pt"  # Pre-training
    ORPO = "orpo"
    SIMPO = "simpo"
    GRPO = "grpo"  # Group Relative Policy Optimization (DeepSeek)
    IPO = "ipo"   # Identity Preference Optimization
    RLOO = "rloo"  # REINFORCE Leave-One-Out
    REINFORCE = "reinforce"  # Standard REINFORCE
    CPO = "cpo"   # Contrastive Preference Optimization


@dataclass
class TrainingStageConfig:
    """Configuration for a training stage."""

    stage_type: TrainingStageType
    name: str
    description: str

    # Forward pass requirements
    num_policy_forwards: int = 1  # Number of policy model forward passes
    num_reference_forwards: int = 0  # Number of reference model forward passes
    num_reward_forwards: int = 0  # Number of reward model forward passes
    num_critic_forwards: int = 0  # Number of critic/value model forward passes

    # Backward pass requirements (for detailed tracking)
    num_policy_backwards: int = 1  # Number of policy backward passes
    num_critic_backwards: int = 0  # Number of critic backward passes

    # Generation phase (for RLHF methods like PPO, GRPO)
    generation_forwards: int = 0  # Policy model generates responses (inference mode)
    generation_tokens: int = 0  # Approximate tokens generated per sample

    # Memory requirements
    requires_reference_model: bool = False
    requires_reward_model: bool = False
    requires_critic_model: bool = False
    requires_value_head: bool = False

    # Group/sample requirements (for GRPO-style methods)
    num_samples_per_prompt: int = 1  # Number of samples to generate per prompt

    # Compute multipliers
    forward_multiplier: float = 1.0  # Total forward passes relative to SFT
    backward_multiplier: float = 1.0  # Backward passes relative to SFT

    # Communication pattern
    sync_reference_gradients: bool = False
    sync_reward_gradients: bool = False
    sync_critic_gradients: bool = False

    # Additional compute characteristics
    uses_importance_sampling: bool = False  # PPO-style importance sampling
    uses_kl_penalty: bool = False  # KL divergence penalty
    uses_group_normalization: bool = False  # GRPO-style group normalization

    # Generation phase uses inference FLOPs (2×params×tokens) instead of training (6×)
    # This is critical for RLHF methods where generation is autoregressive decode
    use_inference_flops_for_generation: bool = False

    def validate(self) -> List[str]:
        """Validate configuration for consistency.

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        # Check generation config consistency
        if self.generation_forwards > 0 and self.generation_tokens <= 0:
            warnings.append(
                f"Stage '{self.name}' has generation_forwards={self.generation_forwards} "
                f"but generation_tokens={self.generation_tokens}. Should generation_tokens be set?"
            )

        # Check RLHF stage requirements
        if self.generation_forwards > 0 and not self.use_inference_flops_for_generation:
            warnings.append(
                f"Stage '{self.name}' has generation but use_inference_flops_for_generation=False. "
                f"Generation should use inference FLOPs (2×params) not training (6×params)."
            )

        # Check model requirements consistency
        if self.num_reference_forwards > 0 and not self.requires_reference_model:
            warnings.append(
                f"Stage '{self.name}' has reference forwards but requires_reference_model=False"
            )
        if self.num_reward_forwards > 0 and not self.requires_reward_model:
            warnings.append(
                f"Stage '{self.name}' has reward forwards but requires_reward_model=False"
            )
        if self.num_critic_forwards > 0 and not self.requires_critic_model:
            warnings.append(
                f"Stage '{self.name}' has critic forwards but requires_critic_model=False"
            )

        # Check multipliers are positive
        if self.forward_multiplier <= 0:
            warnings.append(f"Stage '{self.name}' has invalid forward_multiplier={self.forward_multiplier}")
        if self.backward_multiplier <= 0:
            warnings.append(f"Stage '{self.name}' has invalid backward_multiplier={self.backward_multiplier}")

        return warnings


def validate_all_stage_configs() -> Dict[str, List[str]]:
    """Validate all pre-defined stage configurations.

    Returns:
        Dict mapping stage name to list of warnings (empty list if valid)
    """
    results = {}
    for name, config in TRAINING_STAGE_CONFIGS.items():
        warnings = config.validate()
        if warnings:
            results[name] = warnings
    return results


# Pre-defined stage configurations
TRAINING_STAGE_CONFIGS: Dict[str, TrainingStageConfig] = {
    "sft": TrainingStageConfig(
        stage_type=TrainingStageType.SFT,
        name="Supervised Fine-Tuning",
        description="Standard causal language modeling with teacher forcing",
        num_policy_forwards=1,
        forward_multiplier=1.0,
        backward_multiplier=1.0,
    ),

    "pt": TrainingStageConfig(
        stage_type=TrainingStageType.PT,
        name="Pre-Training",
        description="Standard pre-training, same as SFT",
        num_policy_forwards=1,
        forward_multiplier=1.0,
        backward_multiplier=1.0,
    ),

    "dpo": TrainingStageConfig(
        stage_type=TrainingStageType.DPO,
        name="Direct Preference Optimization",
        description="Policy + Reference forward for chosen/rejected pairs",
        num_policy_forwards=2,  # Chosen + rejected
        num_reference_forwards=2,  # Chosen + rejected (frozen)
        requires_reference_model=True,
        forward_multiplier=4.0,  # 2 policy + 2 reference
        backward_multiplier=2.0,  # Only policy backward for both
    ),

    "orpo": TrainingStageConfig(
        stage_type=TrainingStageType.ORPO,
        name="Odds Ratio Preference Optimization",
        description="Reference-free DPO variant",
        num_policy_forwards=2,  # Chosen + rejected
        num_reference_forwards=0,  # No reference needed
        requires_reference_model=False,
        forward_multiplier=2.0,
        backward_multiplier=2.0,
    ),

    "simpo": TrainingStageConfig(
        stage_type=TrainingStageType.SIMPO,
        name="Simple Preference Optimization",
        description="Simplified DPO without reference model",
        num_policy_forwards=2,
        num_reference_forwards=0,
        requires_reference_model=False,
        forward_multiplier=2.0,
        backward_multiplier=2.0,
    ),

    "kto": TrainingStageConfig(
        stage_type=TrainingStageType.KTO,
        name="Kahneman-Tversky Optimization",
        description="Single-turn preference optimization",
        num_policy_forwards=1,
        num_reference_forwards=1,
        requires_reference_model=True,
        forward_multiplier=2.0,
        backward_multiplier=1.0,
    ),

    "ppo": TrainingStageConfig(
        stage_type=TrainingStageType.PPO,
        name="Proximal Policy Optimization",
        description="Full RLHF with actor, critic, reference, and reward models",
        generation_forwards=1,  # Generate responses in inference mode
        generation_tokens=256,  # Default generation length
        num_policy_forwards=1,
        num_reference_forwards=1,
        num_reward_forwards=1,
        num_critic_forwards=1,
        requires_reference_model=True,
        requires_reward_model=True,
        requires_critic_model=True,
        requires_value_head=True,
        forward_multiplier=4.0,  # Policy + Reference + Reward + Critic
        backward_multiplier=2.0,  # Policy + Critic backward
        sync_critic_gradients=True,
        use_inference_flops_for_generation=True,
    ),

    "rm": TrainingStageConfig(
        stage_type=TrainingStageType.RM,
        name="Reward Modeling",
        description="Train reward model on preference data",
        num_policy_forwards=2,  # Chosen + rejected through same model
        requires_value_head=True,
        forward_multiplier=2.0,
        backward_multiplier=2.0,
    ),

    # GRPO - Group Relative Policy Optimization (DeepSeek)
    "grpo": TrainingStageConfig(
        stage_type=TrainingStageType.GRPO,
        name="Group Relative Policy Optimization",
        description="DeepSeek's GRPO - no critic model needed, uses group-based advantage estimation",
        generation_forwards=1,  # Generate responses in inference mode
        generation_tokens=256,  # Default generation length
        num_policy_forwards=2,  # Forward for loss computation
        num_reference_forwards=1,  # For KL computation
        num_policy_backwards=1,
        requires_reference_model=True,
        requires_reward_model=False,  # No separate reward model
        requires_critic_model=False,  # No critic model
        num_samples_per_prompt=8,  # Generate multiple samples per prompt for group comparison
        forward_multiplier=3.0,  # generation + policy forwards + reference
        backward_multiplier=1.0,
        uses_group_normalization=True,
        uses_kl_penalty=True,
        use_inference_flops_for_generation=True,
    ),

    # IPO - Identity Preference Optimization
    "ipo": TrainingStageConfig(
        stage_type=TrainingStageType.IPO,
        name="Identity Preference Optimization",
        description="Reference-free DPO variant with identity mapping",
        num_policy_forwards=2,  # Chosen + rejected
        num_reference_forwards=0,  # No reference model needed
        num_policy_backwards=2,
        requires_reference_model=False,
        forward_multiplier=2.0,
        backward_multiplier=2.0,
    ),

    # Detailed PPO with explicit phase tracking
    "ppo_detailed": TrainingStageConfig(
        stage_type=TrainingStageType.PPO_DETAILED,
        name="PPO (Detailed Phases)",
        description="Full RLHF with explicit generation, reward, and training phases",
        # Generation phase (inference)
        generation_forwards=1,  # Policy generates responses
        generation_tokens=512,  # Approximate tokens per response
        # Reward scoring phase
        num_reward_forwards=1,  # Score with reward model
        num_reference_forwards=1,  # Compute KL divergence baseline
        # Training phase
        num_policy_forwards=1,  # Policy forward for loss
        num_policy_backwards=1,  # Policy backward
        num_critic_forwards=1,  # Value estimation
        num_critic_backwards=1,  # Critic backward
        requires_reference_model=True,
        requires_reward_model=True,
        requires_critic_model=True,
        requires_value_head=True,
        forward_multiplier=5.0,  # 1 gen + 1 reward + 1 ref + 1 policy + 1 critic
        backward_multiplier=2.0,  # policy + critic
        sync_critic_gradients=True,
        uses_importance_sampling=True,
        uses_kl_penalty=True,
        use_inference_flops_for_generation=True,
    ),

    # RLOO - REINFORCE Leave-One-Out
    "rloo": TrainingStageConfig(
        stage_type=TrainingStageType.RLOO,
        name="REINFORCE Leave-One-Out",
        description="REINFORCE with leave-one-out baseline for variance reduction",
        generation_forwards=1,
        generation_tokens=256,  # Default generation length
        num_policy_forwards=1,
        num_reference_forwards=1,
        num_policy_backwards=1,
        num_samples_per_prompt=4,  # Multiple samples for LOO baseline
        requires_reference_model=True,
        requires_reward_model=True,
        requires_critic_model=False,  # Uses LOO baseline instead
        forward_multiplier=3.0,
        backward_multiplier=1.0,
        uses_kl_penalty=True,
        use_inference_flops_for_generation=True,
    ),

    # Standard REINFORCE
    "reinforce": TrainingStageConfig(
        stage_type=TrainingStageType.REINFORCE,
        name="REINFORCE",
        description="Standard REINFORCE policy gradient",
        generation_forwards=1,
        generation_tokens=256,  # Default generation length
        num_policy_forwards=1,
        num_reward_forwards=1,
        num_policy_backwards=1,
        requires_reward_model=True,
        forward_multiplier=3.0,
        backward_multiplier=1.0,
        use_inference_flops_for_generation=True,
    ),

    # CPO - Contrastive Preference Optimization
    "cpo": TrainingStageConfig(
        stage_type=TrainingStageType.CPO,
        name="Contrastive Preference Optimization",
        description="Contrastive learning approach to preference optimization",
        num_policy_forwards=2,  # Chosen + rejected
        num_reference_forwards=0,  # No reference model
        num_policy_backwards=2,
        requires_reference_model=False,
        forward_multiplier=2.0,
        backward_multiplier=2.0,
    ),
}


def get_stage_config(stage: str) -> TrainingStageConfig:
    """Get configuration for a training stage."""
    stage_lower = stage.lower()
    if stage_lower not in TRAINING_STAGE_CONFIGS:
        raise ValueError(
            f"Unknown training stage: {stage}. "
            f"Available: {list(TRAINING_STAGE_CONFIGS.keys())}"
        )
    return TRAINING_STAGE_CONFIGS[stage_lower]


def calculate_stage_memory_multiplier(stage: str) -> float:
    """
    Calculate memory multiplier for a training stage.

    Returns the factor by which memory increases compared to SFT.
    """
    config = get_stage_config(stage)

    multiplier = 1.0  # Base policy model

    if config.requires_reference_model:
        multiplier += 0.8  # Reference model (eval mode, less memory)

    if config.requires_reward_model:
        multiplier += 1.0  # Full reward model

    if config.requires_critic_model:
        multiplier += 1.2  # Critic with optimizer states

    return multiplier


def calculate_stage_compute_multiplier(stage: str) -> Tuple[float, float]:
    """
    Calculate compute multipliers for forward and backward passes.

    Returns:
        Tuple of (forward_multiplier, backward_multiplier)
    """
    config = get_stage_config(stage)
    return config.forward_multiplier, config.backward_multiplier


def training_modeling_for_stage(
    model: str = 'llama-3-8b',
    training_stage: str = 'sft',
    batch_size: int = 4,
    seq_length: int = 4096,
    system_name: str = 'A100_80GB_GPU',
    num_gpus: int = 1,
    tensor_parallel: int = 1,
    data_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    method: str = 'full',
    lora_rank: int = 16,
    optimizer: str = 'adamw',
    zero_stage: int = 0,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    bits: str = 'bf16',
    system_eff: float = 1.0,
    debug: bool = False,
) -> TrainingModelingOutput:
    """
    Training modeling with stage-aware multipliers applied.

    This function wraps training_modeling and adjusts the results
    based on the training stage requirements.

    Args:
        All arguments same as training_modeling

    Returns:
        TrainingModelingOutput with stage-adjusted metrics
    """
    stage_config = get_stage_config(training_stage)

    # Get base training modeling result
    base_result = training_modeling(
        model=model,
        training_stage=training_stage,
        batch_size=batch_size,
        seq_length=seq_length,
        system_name=system_name,
        num_gpus=num_gpus,
        tensor_parallel=tensor_parallel,
        data_parallel=data_parallel,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=expert_parallel,
        method=method,
        lora_rank=lora_rank,
        optimizer=optimizer,
        zero_stage=zero_stage,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bits=bits,
        system_eff=system_eff,
        debug=debug,
    )

    # Apply stage multipliers
    forward_mult, backward_mult = calculate_stage_compute_multiplier(training_stage)
    memory_mult = calculate_stage_memory_multiplier(training_stage)

    # Adjust timing
    adjusted_forward_ms = base_result.forward_time_ms * forward_mult
    adjusted_backward_ms = base_result.backward_time_ms * backward_mult

    # Adjust communication (more forward passes = more TP communication)
    adjusted_comm_ms = base_result.communication_time_ms * ((forward_mult + backward_mult) / 2)

    adjusted_step_ms = (
        adjusted_forward_ms +
        adjusted_backward_ms +
        base_result.optimizer_time_ms +
        adjusted_comm_ms
    )

    # Adjust memory
    adjusted_memory_gb = base_result.memory_per_gpu_gb * memory_mult

    # Recalculate throughput
    total_tokens = batch_size * seq_length
    adjusted_tps = (total_tokens / adjusted_step_ms) * 1000

    # Create adjusted output
    return TrainingModelingOutput(
        step_time_ms=adjusted_step_ms,
        forward_time_ms=adjusted_forward_ms,
        backward_time_ms=adjusted_backward_ms,
        optimizer_time_ms=base_result.optimizer_time_ms,
        communication_time_ms=adjusted_comm_ms,
        tokens_per_second=adjusted_tps,
        samples_per_second=adjusted_tps / seq_length,
        memory_per_gpu_gb=adjusted_memory_gb,
        weight_memory_gb=base_result.weight_memory_gb * memory_mult,
        gradient_memory_gb=base_result.gradient_memory_gb,
        optimizer_memory_gb=base_result.optimizer_memory_gb,
        activation_memory_gb=base_result.activation_memory_gb * forward_mult,
        reference_model_memory_gb=base_result.weight_memory_gb if stage_config.requires_reference_model else 0,
        model_flops_utilization=base_result.model_flops_utilization * (1.0 / forward_mult),
        hardware_flops_utilization=base_result.hardware_flops_utilization,
        communication_overhead=adjusted_comm_ms / adjusted_step_ms if adjusted_step_ms > 0 else 0,
        runtime_breakdown={
            'forward': adjusted_forward_ms / adjusted_step_ms if adjusted_step_ms > 0 else 0,
            'backward': adjusted_backward_ms / adjusted_step_ms if adjusted_step_ms > 0 else 0,
            'optimizer': base_result.optimizer_time_ms / adjusted_step_ms if adjusted_step_ms > 0 else 0,
            'communication': adjusted_comm_ms / adjusted_step_ms if adjusted_step_ms > 0 else 0,
        },
        config={
            **base_result.config,
            'stage_config': {
                'forward_multiplier': forward_mult,
                'backward_multiplier': backward_mult,
                'memory_multiplier': memory_mult,
                'requires_reference': stage_config.requires_reference_model,
                'requires_reward': stage_config.requires_reward_model,
            }
        },
        model_df=base_result.model_df,
        summary_table=base_result.summary_table,
    )


def estimate_dpo_training(
    model: str,
    batch_size: int,
    seq_length: int,
    system_name: str = 'A100_80GB_GPU',
    num_gpus: int = 1,
    **kwargs,
) -> TrainingModelingOutput:
    """
    Convenience function for DPO training estimation.

    DPO requires:
    - 2 forward passes through policy (chosen + rejected)
    - 2 forward passes through reference (chosen + rejected, no gradient)
    - 1 backward pass through policy
    """
    return training_modeling_for_stage(
        model=model,
        training_stage='dpo',
        batch_size=batch_size,
        seq_length=seq_length,
        system_name=system_name,
        num_gpus=num_gpus,
        **kwargs,
    )


def estimate_ppo_training(
    model: str,
    batch_size: int,
    seq_length: int,
    system_name: str = 'A100_80GB_GPU',
    num_gpus: int = 1,
    **kwargs,
) -> TrainingModelingOutput:
    """
    Convenience function for PPO training estimation.

    PPO requires:
    - Actor forward pass
    - Reference forward pass (no gradient)
    - Reward model forward pass (no gradient)
    - Critic forward pass
    - Critic backward pass
    - Actor backward pass
    """
    return training_modeling_for_stage(
        model=model,
        training_stage='ppo',
        batch_size=batch_size,
        seq_length=seq_length,
        system_name=system_name,
        num_gpus=num_gpus,
        **kwargs,
    )


def list_training_stages() -> Dict[str, str]:
    """List available training stages with descriptions."""
    return {
        name: config.description
        for name, config in TRAINING_STAGE_CONFIGS.items()
    }


def get_rlhf_stages() -> Dict[str, TrainingStageConfig]:
    """Get RLHF-specific training stages (PPO, GRPO, RLOO, etc.)."""
    rlhf_types = {
        TrainingStageType.PPO,
        TrainingStageType.PPO_DETAILED,
        TrainingStageType.GRPO,
        TrainingStageType.RLOO,
        TrainingStageType.REINFORCE,
    }
    return {
        name: config
        for name, config in TRAINING_STAGE_CONFIGS.items()
        if config.stage_type in rlhf_types
    }


def get_preference_stages() -> Dict[str, TrainingStageConfig]:
    """Get preference optimization stages (DPO, IPO, ORPO, etc.)."""
    pref_types = {
        TrainingStageType.DPO,
        TrainingStageType.IPO,
        TrainingStageType.ORPO,
        TrainingStageType.SIMPO,
        TrainingStageType.CPO,
        TrainingStageType.KTO,
    }
    return {
        name: config
        for name, config in TRAINING_STAGE_CONFIGS.items()
        if config.stage_type in pref_types
    }


def get_stage_memory_requirements(stage: str) -> Dict[str, bool]:
    """Get memory requirements for a training stage."""
    config = get_stage_config(stage)
    return {
        'policy_model': True,  # Always needed
        'reference_model': config.requires_reference_model,
        'reward_model': config.requires_reward_model,
        'critic_model': config.requires_critic_model,
        'value_head': config.requires_value_head,
    }


def estimate_grpo_training(
    model: str,
    batch_size: int,
    seq_length: int,
    system_name: str = 'A100_80GB_GPU',
    num_gpus: int = 1,
    num_samples_per_prompt: int = 8,
    **kwargs,
) -> TrainingModelingOutput:
    """
    Convenience function for GRPO training estimation.

    GRPO (Group Relative Policy Optimization) from DeepSeek:
    - Generates multiple samples per prompt
    - Uses group-based advantage estimation (no critic needed)
    - Reference model for KL penalty

    Note: num_samples_per_prompt affects the effective batch multiplier
    as GRPO generates multiple responses per prompt for group comparison.
    """
    # Get base result
    result = training_modeling_for_stage(
        model=model,
        training_stage='grpo',
        batch_size=batch_size,
        seq_length=seq_length,
        system_name=system_name,
        num_gpus=num_gpus,
        **kwargs,
    )

    # Adjust for custom num_samples_per_prompt if different from default
    stage_config = get_stage_config('grpo')
    if num_samples_per_prompt != stage_config.num_samples_per_prompt:
        sample_multiplier = num_samples_per_prompt / stage_config.num_samples_per_prompt
        # Adjust generation time proportionally
        # (more samples = more generation time, but parallel generation helps)
        generation_factor = sample_multiplier ** 0.7  # Sub-linear scaling due to batching

        # Update forward time to account for more samples
        adjusted_forward_ms = result.forward_time_ms * generation_factor
        adjusted_step_ms = (
            adjusted_forward_ms +
            result.backward_time_ms +
            result.optimizer_time_ms +
            result.communication_time_ms
        )

        # Recalculate throughput
        total_tokens = batch_size * seq_length
        adjusted_tps = (total_tokens / adjusted_step_ms) * 1000 if adjusted_step_ms > 0 else 0

        return TrainingModelingOutput(
            step_time_ms=adjusted_step_ms,
            forward_time_ms=adjusted_forward_ms,
            backward_time_ms=result.backward_time_ms,
            optimizer_time_ms=result.optimizer_time_ms,
            communication_time_ms=result.communication_time_ms,
            tokens_per_second=adjusted_tps,
            samples_per_second=adjusted_tps / seq_length if seq_length > 0 else 0,
            memory_per_gpu_gb=result.memory_per_gpu_gb,
            weight_memory_gb=result.weight_memory_gb,
            gradient_memory_gb=result.gradient_memory_gb,
            optimizer_memory_gb=result.optimizer_memory_gb,
            activation_memory_gb=result.activation_memory_gb * generation_factor,
            reference_model_memory_gb=result.reference_model_memory_gb,
            model_flops_utilization=result.model_flops_utilization,
            hardware_flops_utilization=result.hardware_flops_utilization,
            communication_overhead=result.communication_overhead,
            runtime_breakdown=result.runtime_breakdown,
            config={
                **result.config,
                'num_samples_per_prompt': num_samples_per_prompt,
            },
            model_df=result.model_df,
            summary_table=result.summary_table,
        )

    return result


def estimate_ipo_training(
    model: str,
    batch_size: int,
    seq_length: int,
    system_name: str = 'A100_80GB_GPU',
    num_gpus: int = 1,
    **kwargs,
) -> TrainingModelingOutput:
    """
    Convenience function for IPO training estimation.

    IPO (Identity Preference Optimization):
    - Reference-free DPO variant
    - No need for reference model in memory
    - Same compute as DPO but lower memory footprint
    """
    return training_modeling_for_stage(
        model=model,
        training_stage='ipo',
        batch_size=batch_size,
        seq_length=seq_length,
        system_name=system_name,
        num_gpus=num_gpus,
        **kwargs,
    )


def compare_training_stages(
    model: str,
    batch_size: int,
    seq_length: int,
    system_name: str,
    stages: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple training stages for the same configuration.

    Args:
        model: Model name
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        system_name: Hardware system
        stages: List of stages to compare (default: all preference stages)
        **kwargs: Additional arguments for training_modeling

    Returns:
        Dictionary mapping stage name to results
    """
    if stages is None:
        stages = ['sft', 'dpo', 'ipo', 'orpo', 'simpo', 'kto']

    results = {}
    for stage in stages:
        try:
            result = training_modeling_for_stage(
                model=model,
                training_stage=stage,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                **kwargs,
            )
            results[stage] = {
                'tokens_per_second': result.tokens_per_second,
                'memory_gb': result.memory_per_gpu_gb,
                'step_time_ms': result.step_time_ms,
                'mfu': result.model_flops_utilization,
                'requires_reference': get_stage_config(stage).requires_reference_model,
            }
        except Exception as e:
            results[stage] = {'error': str(e)}

    return results
