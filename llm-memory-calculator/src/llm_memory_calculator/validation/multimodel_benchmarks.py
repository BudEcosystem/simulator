"""
Multi-Model Training Benchmarks for PPO, DPO, GRPO, and other RLHF methods.

This module contains benchmarks for post-training methods that involve
multiple models with different gradient strategies:

- PPO (3-4 models): policy, reference, reward, value
- DPO (2 models): policy, reference
- KTO (2 models): policy, reference
- GRPO (2+ models): policy, reference with group sampling
- ORPO/SimPO (1 model): reference-free

Key characteristics tracked:
- Memory factor vs single-model training
- Inference vs training phase memory
- Reference model loading strategies
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .benchmark_schema import (
    ExtendedBenchmark,
    HardwareType,
    MultiModelConfig,
    OptimizerType,
    ParallelismConfig,
    PEFTConfig,
    QuantizationConfig,
    QuantizationMethod,
    ReportedMetrics,
    SourceProvenance,
    SourceType,
    TrainingType,
    VerificationStatus,
)


@dataclass
class MultiModelProfile:
    """Profile for multi-model training method."""

    training_type: TrainingType
    name: str
    description: str

    # Model configuration
    num_models: int
    model_roles: List[str]

    # Memory characteristics
    memory_multiplier: float  # vs single-model training
    peak_memory_phase: str  # "generation", "training", "reward_scoring"

    # Compute characteristics
    inference_ratio: float  # Fraction of time in inference mode
    training_ratio: float  # Fraction of time in training mode

    # Reference model handling
    reference_model_mode: str  # "frozen", "offloaded", "quantized", "none"
    reference_in_eval_mode: bool = True

    # KV cache requirements during generation
    requires_kv_cache: bool = True
    kv_cache_multiplier: float = 1.0  # vs single model inference

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_type": self.training_type.value,
            "name": self.name,
            "description": self.description,
            "num_models": self.num_models,
            "model_roles": self.model_roles,
            "memory_multiplier": self.memory_multiplier,
            "peak_memory_phase": self.peak_memory_phase,
            "inference_ratio": self.inference_ratio,
            "training_ratio": self.training_ratio,
            "reference_model_mode": self.reference_model_mode,
            "reference_in_eval_mode": self.reference_in_eval_mode,
            "requires_kv_cache": self.requires_kv_cache,
            "kv_cache_multiplier": self.kv_cache_multiplier,
        }


# Multi-model training profiles
MULTIMODEL_PROFILES: Dict[TrainingType, MultiModelProfile] = {
    TrainingType.PPO: MultiModelProfile(
        training_type=TrainingType.PPO,
        name="PPO / RLHF",
        description="Full PPO with 4 models: policy, reference, reward, value",
        num_models=4,
        model_roles=["policy", "reference", "reward", "value"],
        memory_multiplier=2.8,
        peak_memory_phase="training",  # Policy update phase
        inference_ratio=0.6,  # Generation + reward scoring
        training_ratio=0.4,
        reference_model_mode="frozen",
        requires_kv_cache=True,
        kv_cache_multiplier=1.5,  # Multiple generations
    ),
    TrainingType.DPO: MultiModelProfile(
        training_type=TrainingType.DPO,
        name="Direct Preference Optimization",
        description="DPO with policy and frozen reference model",
        num_models=2,
        model_roles=["policy", "reference"],
        memory_multiplier=1.7,
        peak_memory_phase="training",
        inference_ratio=0.0,  # No generation during training
        training_ratio=1.0,
        reference_model_mode="frozen",
        requires_kv_cache=False,
    ),
    TrainingType.KTO: MultiModelProfile(
        training_type=TrainingType.KTO,
        name="Kahneman-Tversky Optimization",
        description="KTO with policy and frozen reference model",
        num_models=2,
        model_roles=["policy", "reference"],
        memory_multiplier=1.7,
        peak_memory_phase="training",
        inference_ratio=0.0,
        training_ratio=1.0,
        reference_model_mode="frozen",
        requires_kv_cache=False,
    ),
    TrainingType.GRPO: MultiModelProfile(
        training_type=TrainingType.GRPO,
        name="Group Relative Policy Optimization",
        description="GRPO with group sampling for preference learning",
        num_models=2,
        model_roles=["policy", "reference"],
        memory_multiplier=2.0,
        peak_memory_phase="generation",  # Group generation is expensive
        inference_ratio=0.7,  # Heavy generation phase
        training_ratio=0.3,
        reference_model_mode="frozen",
        requires_kv_cache=True,
        kv_cache_multiplier=2.0,  # Multiple samples per group
    ),
    TrainingType.ORPO: MultiModelProfile(
        training_type=TrainingType.ORPO,
        name="Odds Ratio Preference Optimization",
        description="Reference-free preference optimization",
        num_models=1,
        model_roles=["policy"],
        memory_multiplier=1.0,
        peak_memory_phase="training",
        inference_ratio=0.0,
        training_ratio=1.0,
        reference_model_mode="none",
        requires_kv_cache=False,
    ),
    TrainingType.SIMPO: MultiModelProfile(
        training_type=TrainingType.SIMPO,
        name="Simple Preference Optimization",
        description="Simplified reference-free preference optimization",
        num_models=1,
        model_roles=["policy"],
        memory_multiplier=1.0,
        peak_memory_phase="training",
        inference_ratio=0.0,
        training_ratio=1.0,
        reference_model_mode="none",
        requires_kv_cache=False,
    ),
    TrainingType.REINFORCE: MultiModelProfile(
        training_type=TrainingType.REINFORCE,
        name="REINFORCE",
        description="Policy gradient with reward model",
        num_models=2,
        model_roles=["policy", "reward"],
        memory_multiplier=1.8,
        peak_memory_phase="generation",
        inference_ratio=0.5,
        training_ratio=0.5,
        reference_model_mode="none",
        requires_kv_cache=True,
    ),
}


# Multi-model training benchmarks
MULTIMODEL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # PPO Benchmarks
    # =========================================================================
    "ppo_llama_7b_full": {
        "name": "PPO LLaMA-7B Full (4 models)",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.PPO,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 128,
        "seq_length": 2048,
        "memory_per_gpu_gb": 68.0,
        "peak_memory_phase": "training",
        "multi_model": {
            "num_models": 4,
            "has_policy_model": True,
            "has_reference_model": True,
            "has_reward_model": True,
            "has_value_model": True,
            "reference_model_frozen": True,
            "reward_model_frozen": True,
            "share_base_weights": True,
        },
        "source": "OpenRLHF benchmarks",
        "notes": "Full PPO with value head sharing backbone",
    },
    "ppo_llama_7b_lora_reward": {
        "name": "PPO LLaMA-7B with LoRA Reward",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.PPO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 2048,
        "memory_per_gpu_gb": 55.0,
        "multi_model": {
            "num_models": 3,
            "has_policy_model": True,
            "has_reference_model": True,
            "has_reward_model": True,
            "has_value_model": True,
            "reward_model_as_lora": True,
            "share_base_weights": True,
        },
        "source": "TRL benchmarks",
        "notes": "Reward model as LoRA adapter for memory efficiency",
    },
    "ppo_llama_70b_distributed": {
        "name": "PPO LLaMA-70B Distributed",
        "model_name": "LLaMA-70B",
        "model_params_b": 70.0,
        "training_type": TrainingType.PPO,
        "num_gpus": 64,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 512,
        "seq_length": 2048,
        "memory_per_gpu_gb": 75.0,
        "parallelism": {"tp": 4, "pp": 1, "dp": 16},
        "multi_model": {
            "num_models": 4,
            "has_policy_model": True,
            "has_reference_model": True,
            "has_reward_model": True,
            "has_value_model": True,
            "reference_model_frozen": True,
        },
        "source": "OpenRLHF",
        "mfu": 0.35,
        "notes": "Large-scale PPO with tensor parallelism",
    },
    "ppo_mistral_7b_quantized_ref": {
        "name": "PPO Mistral-7B with Quantized Reference",
        "model_name": "Mistral-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.PPO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 2048,
        "memory_per_gpu_gb": 48.0,
        "multi_model": {
            "num_models": 4,
            "has_policy_model": True,
            "has_reference_model": True,
            "has_reward_model": True,
            "has_value_model": True,
            "reference_model_quantized": True,  # INT8 reference
        },
        "source": "Community benchmarks",
        "notes": "Quantized reference model for memory savings",
    },
    # =========================================================================
    # DPO Benchmarks
    # =========================================================================
    "dpo_llama_7b": {
        "name": "DPO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.DPO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 2048,
        "memory_per_gpu_gb": 52.0,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
            "reference_model_frozen": True,
        },
        "source": "TRL benchmarks",
    },
    "dpo_llama_13b": {
        "name": "DPO LLaMA-13B",
        "model_name": "LLaMA-13B",
        "model_params_b": 13.0,
        "training_type": TrainingType.DPO,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 32,
        "seq_length": 2048,
        "memory_per_gpu_gb": 65.0,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
        },
        "source": "Community benchmarks",
    },
    "dpo_llama_70b_quantized": {
        "name": "DPO LLaMA-70B with 4-bit Reference",
        "model_name": "LLaMA-70B",
        "model_params_b": 70.0,
        "training_type": TrainingType.DPO,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 16,
        "seq_length": 2048,
        "memory_per_gpu_gb": 72.0,
        "quantization": QuantizationMethod.NF4,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
            "reference_model_quantized": True,
        },
        "source": "LLaMA-Factory",
        "notes": "4-bit quantized reference model",
    },
    "dpo_mistral_7b_zephyr": {
        "name": "DPO Mistral-7B (Zephyr-style)",
        "model_name": "Mistral-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.DPO,
        "num_gpus": 16,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 128,
        "seq_length": 2048,
        "memory_per_gpu_gb": 45.0,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
        },
        "source": "HuggingFace H4 team",
        "notes": "Zephyr model training configuration",
    },
    # =========================================================================
    # GRPO Benchmarks
    # =========================================================================
    "grpo_deepseek_v3": {
        "name": "GRPO DeepSeek-V3 671B",
        "model_name": "DeepSeek-V3",
        "model_params_b": 671.0,
        "training_type": TrainingType.GRPO,
        "is_moe": True,
        "num_experts": 256,
        "active_experts": 8,
        "num_gpus": 2048,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 2048,
        "seq_length": 4096,
        "memory_per_gpu_gb": 78.0,
        "parallelism": {"tp": 1, "pp": 8, "dp": 32, "ep": 64},
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
            "grpo_num_groups": 8,
            "grpo_samples_per_group": 4,
        },
        "source": "DeepSeek V3 paper",
        "source_url": "https://arxiv.org/abs/2412.19437",
        "mfu": 0.18,
        "notes": "GRPO with FP8 and large group sampling",
    },
    "grpo_llama_7b": {
        "name": "GRPO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.GRPO,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 256,
        "seq_length": 2048,
        "memory_per_gpu_gb": 62.0,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
            "grpo_num_groups": 4,
            "grpo_samples_per_group": 4,
        },
        "source": "GRPO implementation",
        "notes": "Group size 4, 4 samples per group",
    },
    # =========================================================================
    # KTO Benchmarks
    # =========================================================================
    "kto_llama_7b": {
        "name": "KTO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.KTO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 2048,
        "memory_per_gpu_gb": 50.0,
        "multi_model": {
            "num_models": 2,
            "has_policy_model": True,
            "has_reference_model": True,
        },
        "source": "TRL benchmarks",
        "notes": "KTO with unpaired preference data",
    },
    # =========================================================================
    # ORPO/SimPO Benchmarks (Reference-free)
    # =========================================================================
    "orpo_mistral_7b": {
        "name": "ORPO Mistral-7B",
        "model_name": "Mistral-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.ORPO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 16,
        "seq_length": 2048,
        "memory_per_gpu_gb": 32.0,
        "multi_model": {
            "num_models": 1,
            "has_policy_model": True,
            "has_reference_model": False,
        },
        "source": "ORPO paper",
        "source_url": "https://arxiv.org/abs/2403.07691",
        "notes": "Reference-free, same memory as SFT",
    },
    "simpo_llama_7b": {
        "name": "SimPO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.SIMPO,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 16,
        "seq_length": 2048,
        "memory_per_gpu_gb": 32.0,
        "multi_model": {
            "num_models": 1,
            "has_policy_model": True,
            "has_reference_model": False,
        },
        "source": "SimPO paper",
        "notes": "Simple preference optimization without reference",
    },
    # =========================================================================
    # Reward Model Training
    # =========================================================================
    "rm_llama_7b": {
        "name": "Reward Model LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "training_type": TrainingType.REWARD_MODELING,
        "num_gpus": 4,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 2048,
        "memory_per_gpu_gb": 36.0,
        "multi_model": {
            "num_models": 1,
            "has_value_model": True,  # Reward/value head
        },
        "source": "TRL benchmarks",
        "notes": "Reward model with value head",
    },
}


def get_multimodel_profile(training_type: TrainingType) -> Optional[MultiModelProfile]:
    """Get profile for a multi-model training type."""
    return MULTIMODEL_PROFILES.get(training_type)


def estimate_multimodel_memory(
    model_params_b: float,
    training_type: TrainingType,
    optimizer_bytes_per_param: float = 8.0,
    precision_bytes: float = 2.0,
    reference_quantized: bool = False,
) -> Dict[str, float]:
    """
    Estimate memory breakdown for multi-model training.

    Args:
        model_params_b: Model parameters in billions
        training_type: Training type (PPO, DPO, etc.)
        optimizer_bytes_per_param: Optimizer state bytes
        precision_bytes: Model precision bytes (2 for BF16)
        reference_quantized: Whether reference model is quantized

    Returns:
        Memory breakdown by component
    """
    profile = get_multimodel_profile(training_type)
    if profile is None:
        return {"error": f"Unknown training type: {training_type}"}

    # Base model memory
    model_memory = model_params_b * precision_bytes

    # Policy model (with gradients and optimizer)
    policy_memory = model_memory + model_params_b * precision_bytes + model_params_b * optimizer_bytes_per_param

    # Reference model (eval mode, no gradients)
    reference_memory = 0.0
    if profile.reference_model_mode != "none":
        if reference_quantized:
            reference_memory = model_params_b * 0.5  # 4-bit
        else:
            reference_memory = model_memory  # Full precision eval mode

    # Reward model (eval mode if separate)
    reward_memory = 0.0
    if "reward" in profile.model_roles:
        reward_memory = model_memory * 0.7  # Eval mode overhead

    # Value model (shares backbone or separate)
    value_memory = 0.0
    if "value" in profile.model_roles:
        value_memory = model_params_b * 0.1  # Value head only if sharing

    total = policy_memory + reference_memory + reward_memory + value_memory

    return {
        "policy_model_gb": policy_memory,
        "reference_model_gb": reference_memory,
        "reward_model_gb": reward_memory,
        "value_model_gb": value_memory,
        "total_gb": total,
        "memory_multiplier": total / policy_memory if policy_memory > 0 else 0,
    }


def get_multimodel_benchmarks() -> List[ExtendedBenchmark]:
    """Get all multi-model benchmarks as ExtendedBenchmark objects."""
    benchmarks = []

    for bench_id, config in MULTIMODEL_BENCHMARKS.items():
        # Create parallelism config
        p_config = config.get("parallelism", {})
        parallelism = ParallelismConfig(
            tensor_parallel=p_config.get("tp", 1),
            pipeline_parallel=p_config.get("pp", 1),
            data_parallel=p_config.get("dp", config.get("num_gpus", 1)),
            expert_parallel=p_config.get("ep", 1),
        )

        # Create multi-model config
        mm_config = config.get("multi_model", {})
        multi_model = MultiModelConfig(
            num_models=mm_config.get("num_models", 1),
            has_policy_model=mm_config.get("has_policy_model", True),
            has_reference_model=mm_config.get("has_reference_model", False),
            has_reward_model=mm_config.get("has_reward_model", False),
            has_value_model=mm_config.get("has_value_model", False),
            reference_model_frozen=mm_config.get("reference_model_frozen", True),
            reference_model_quantized=mm_config.get("reference_model_quantized", False),
            reward_model_as_lora=mm_config.get("reward_model_as_lora", False),
            share_base_weights=mm_config.get("share_base_weights", False),
            grpo_num_groups=mm_config.get("grpo_num_groups", 1),
            grpo_samples_per_group=mm_config.get("grpo_samples_per_group", 1),
        )

        # Create quantization config
        quant = config.get("quantization")
        quantization = QuantizationConfig(
            model_precision=quant if quant else QuantizationMethod.BF16,
        )

        # Create metrics
        metrics = ReportedMetrics(
            mfu=config.get("mfu"),
            memory_per_gpu_gb=config.get("memory_per_gpu_gb"),
        )

        # Create provenance
        provenance = SourceProvenance(
            source_type=SourceType.GITHUB_REPO if "github" in config.get("source", "").lower() else SourceType.ACADEMIC_PAPER,
            source_url=config.get("source_url"),
            source_title=config.get("source", ""),
            extraction_method="manual",
            extraction_date=datetime.now().strftime("%Y-%m-%d"),
        )

        benchmark = ExtendedBenchmark(
            benchmark_id=bench_id,
            name=config["name"],
            provenance=provenance,
            model_name=config["model_name"],
            model_params_b=config["model_params_b"],
            training_type=config["training_type"],
            is_moe=config.get("is_moe", False),
            num_experts=config.get("num_experts", 1),
            active_experts=config.get("active_experts", 1),
            num_gpus=config.get("num_gpus", 1),
            hardware_type=config.get("hardware_type", HardwareType.A100_80GB),
            batch_size=config.get("batch_size", 1),
            seq_length=config.get("seq_length", 2048),
            parallelism=parallelism,
            quantization=quantization,
            multi_model=multi_model,
            metrics=metrics,
            verification_status=VerificationStatus.UNVERIFIED,
            notes=config.get("notes", ""),
            tags=["multimodel", config["training_type"].value],
        )

        benchmarks.append(benchmark)

    return benchmarks


def validate_multimodel_memory(
    benchmark: ExtendedBenchmark,
    tolerance: float = 0.25,
) -> Dict[str, Any]:
    """
    Validate multi-model training memory against expected values.

    Args:
        benchmark: Benchmark to validate
        tolerance: Acceptable relative error

    Returns:
        Validation result with expected vs actual
    """
    if benchmark.multi_model is None:
        return {"valid": True, "note": "Not a multi-model benchmark"}

    profile = get_multimodel_profile(benchmark.training_type)
    if profile is None:
        return {"valid": None, "error": f"Unknown training type: {benchmark.training_type}"}

    # Estimate expected memory
    expected = estimate_multimodel_memory(
        model_params_b=benchmark.model_params_b,
        training_type=benchmark.training_type,
        reference_quantized=benchmark.multi_model.reference_model_quantized,
    )

    actual = benchmark.metrics.memory_per_gpu_gb

    if actual is None:
        return {
            "valid": None,
            "expected_gb": expected["total_gb"] / benchmark.num_gpus,
            "actual_gb": None,
            "error": "No actual memory reported",
        }

    # Adjust for GPU count
    expected_per_gpu = expected["total_gb"] / benchmark.num_gpus
    relative_error = abs(actual - expected_per_gpu) / expected_per_gpu if expected_per_gpu > 0 else 0

    return {
        "valid": relative_error <= tolerance,
        "expected_gb": expected_per_gpu,
        "actual_gb": actual,
        "relative_error": relative_error,
        "memory_breakdown": expected,
    }
