"""
LlamaFactory Best Practices and Configuration Constants.

Comprehensive best practices derived from LlamaFactory analysis (200+ parameters):
- Training type configurations and memory multipliers
- Learning rate recommendations by model size and method
- Optimizer configurations and memory overhead
- Stability and convergence best practices
- LoRA/QLoRA configuration recommendations
- PPO multi-model configuration
- Distributed training best practices
- Hardware-specific optimizations
- Troubleshooting guides

Sources:
- LlamaFactory source code analysis (model_args.py, data_args.py, training_args.py, finetuning_args.py)
- Published training benchmarks
- Best practices from major training runs
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# TRAINING TYPE CONFIGURATIONS
# ============================================================================

TRAINING_TYPE_MEMORY_MULTIPLIERS: Dict[str, float] = {
    "pt": 1.0,      # Pre-training
    "sft": 1.0,     # Supervised Fine-Tuning
    "dpo": 1.7,     # Direct Preference Optimization (policy + reference)
    "kto": 1.7,     # Kahneman-Tversky Optimization
    "ipo": 1.7,     # Identity Preference Optimization
    "ppo": 2.8,     # PPO (actor + critic + reference + reward)
    "grpo": 1.5,    # Group Relative Policy Optimization
    "orpo": 1.0,    # Odds Ratio Preference Optimization (reference-free)
    "simpo": 1.0,   # Simple Preference Optimization (reference-free)
    "rloo": 1.5,    # REINFORCE Leave-One-Out
    "rm": 1.1,      # Reward Modeling
}

# Training type specific defaults
TRAINING_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sft": {
        "loss": "cross_entropy",
        "train_on_prompt": False,
        "mask_history": False,
    },
    "dpo": {
        "pref_beta": 0.1,
        "pref_loss": "sigmoid",
        "pref_ftx": 0.0,
        "dpo_label_smoothing": 0.0,
        "ref_model_quantization": "same_as_model",
    },
    "kto": {
        "kto_beta": 0.1,
        "kto_chosen_weight": 1.0,
        "kto_rejected_weight": 1.0,
        "kto_ftx": 0.0,
    },
    "orpo": {
        "orpo_beta": 0.1,
        "pref_loss": "orpo",
    },
    "simpo": {
        "simpo_gamma": 1.0,
        "pref_loss": "simpo",
    },
    "ppo": {
        "ppo_epochs": 4,
        "ppo_buffer_size": 1,
        "ppo_target_kl": None,  # None = no early stopping
        "ppo_score_norm": True,
        "ppo_whiten_rewards": True,
    },
    "grpo": {
        "ppo_epochs": 1,
        "num_generations": 8,
    },
    "rm": {
        "loss": "sigmoid",
    },
    "pt": {
        "train_on_prompt": True,
    },
}


# ============================================================================
# LEARNING RATE RECOMMENDATIONS BY MODEL SIZE AND METHOD
# ============================================================================

# Model size bins (in billions of parameters)
MODEL_SIZE_BINS = ["1B", "3B", "7B", "8B", "13B", "14B", "32B", "70B", "72B", "405B"]

LEARNING_RATES: Dict[str, Dict[str, float]] = {
    "lora": {
        "1B": 2e-4,
        "3B": 2e-4,
        "7B": 1e-4,
        "8B": 1e-4,
        "13B": 5e-5,
        "14B": 5e-5,
        "32B": 2e-5,
        "70B": 2e-5,
        "72B": 2e-5,
        "405B": 1e-5,
    },
    "qlora": {
        "1B": 2e-4,
        "3B": 2e-4,
        "7B": 1e-4,
        "8B": 1e-4,
        "13B": 5e-5,
        "14B": 5e-5,
        "32B": 2e-5,
        "70B": 2e-5,
        "72B": 2e-5,
        "405B": 1e-5,
    },
    "full": {
        "1B": 2e-5,
        "3B": 2e-5,
        "7B": 1e-5,
        "8B": 1e-5,
        "13B": 5e-6,
        "14B": 5e-6,
        "32B": 2e-6,
        "70B": 2e-6,
        "72B": 2e-6,
        "405B": 1e-6,
    },
    "dora": {
        "1B": 1e-4,
        "3B": 1e-4,
        "7B": 5e-5,
        "8B": 5e-5,
        "13B": 2e-5,
        "14B": 2e-5,
        "32B": 1e-5,
        "70B": 1e-5,
        "72B": 1e-5,
        "405B": 5e-6,
    },
    "pissa": {
        "1B": 1e-4,
        "3B": 1e-4,
        "7B": 5e-5,
        "8B": 5e-5,
        "13B": 2e-5,
        "14B": 2e-5,
        "32B": 1e-5,
        "70B": 1e-5,
        "72B": 1e-5,
        "405B": 5e-6,
    },
    "freeze": {
        "1B": 2e-5,
        "3B": 2e-5,
        "7B": 1e-5,
        "8B": 1e-5,
        "13B": 5e-6,
        "14B": 5e-6,
        "32B": 2e-6,
        "70B": 2e-6,
        "72B": 2e-6,
        "405B": 1e-6,
    },
}


def get_recommended_learning_rate(method: str, model_params_b: float) -> float:
    """Get recommended learning rate for method and model size."""
    method = method.lower()
    if method not in LEARNING_RATES:
        method = "lora"  # Default to LoRA rates

    rates = LEARNING_RATES[method]

    # Find appropriate size bin
    if model_params_b < 2:
        return rates["1B"]
    elif model_params_b < 5:
        return rates["3B"]
    elif model_params_b < 10:
        return rates["8B"]
    elif model_params_b < 20:
        return rates["14B"]
    elif model_params_b < 50:
        return rates["32B"]
    elif model_params_b < 100:
        return rates["72B"]
    else:
        return rates["405B"]


# ============================================================================
# OPTIMIZER CONFIGURATIONS
# ============================================================================

OPTIMIZER_STATE_MULTIPLIERS: Dict[str, int] = {
    # Standard optimizers (bytes per trainable param)
    "adamw": 8,              # m + v states (FP32)
    "adam": 8,
    "adamw_torch": 8,
    "sgd": 4,                # Just momentum
    "sgd_nesterov": 4,
    "adafactor": 4,          # Row/column factorized
    "lion": 4,               # Momentum only

    # 8-bit quantized optimizers
    "adamw_8bit": 2,         # bitsandbytes
    "adam_8bit": 2,
    "paged_adamw_8bit": 2,   # Paged memory
    "paged_adam_8bit": 2,
    "paged_lion_8bit": 1,

    # Memory-efficient optimizers
    "galore_adamw": 2,       # GaLore low-rank gradient
    "galore_adamw_8bit": 1,
    "apollo": 2,             # Adaptive low-rank
    "apollo_mini": 1,
    "adam_mini": 2,          # Adam-mini
    "muon": 4,               # Muon optimizer
}

OPTIMIZER_BEST_PRACTICES: Dict[str, Dict[str, Any]] = {
    "adamw": {
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "recommended_for": ["full_finetuning", "general"],
    },
    "adamw_8bit": {
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "recommended_for": ["memory_constrained", "lora"],
    },
    "paged_adamw_8bit": {
        "weight_decay": 0.01,
        "recommended_for": ["qlora", "extreme_memory_constrained"],
    },
    "galore_adamw": {
        "galore_rank": 128,
        "galore_update_interval": 200,
        "galore_scale": 0.25,
        "galore_target": "all",
        "recommended_for": ["full_finetuning_memory_limited"],
    },
    "apollo": {
        "apollo_rank": 256,
        "apollo_update_interval": 10,
        "apollo_scale": 1.0,
        "apollo_target": "all",
        "recommended_for": ["large_models", "memory_efficient"],
    },
    "adafactor": {
        "scale_parameter": True,
        "relative_step": True,
        "warmup_init": True,
        "recommended_for": ["pretraining", "memory_efficient"],
    },
    "lion": {
        "weight_decay": 0.1,  # Lion typically uses higher weight decay
        "adam_beta1": 0.95,
        "adam_beta2": 0.98,
        "recommended_for": ["fast_training", "research"],
    },
    "sgd": {
        "momentum": 0.9,
        "weight_decay": 0.01,
        "recommended_for": ["baseline", "simple_tasks"],
    },
}


# ============================================================================
# STABILITY BEST PRACTICES
# ============================================================================

STABILITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "description": "Maximum stability, slower convergence",
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
        "bf16": True,
        "upcast_layernorm": True,
        "upcast_lmhead_output": True,
    },
    "balanced": {
        "description": "Good stability with reasonable speed",
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
        "bf16": True,
    },
    "aggressive": {
        "description": "Fast training, may need tuning",
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "gradient_checkpointing": False,
        "bf16": True,
    },
}


# ============================================================================
# CONVERGENCE BEST PRACTICES
# ============================================================================

CONVERGENCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "neftune": {
        "description": "Noise Embedding Fine-Tuning for better generalization",
        "neftune_noise_alpha": 5.0,
        "applicable_to": ["sft", "pt"],
        "improvement": "Typically 1-2% on downstream tasks",
    },
    "label_smoothing": {
        "description": "Prevents overconfidence",
        "label_smoothing_factor": 0.1,
        "applicable_to": ["sft", "pt"],
    },
    "packing": {
        "description": "Efficient packing of variable-length sequences",
        "packing": True,
        "neat_packing": True,  # Prevent cross-sample attention
        "applicable_to": ["sft", "pt", "dpo"],
        "throughput_improvement": "20-40%",
    },
    "flash_attention": {
        "description": "Memory-efficient attention",
        "flash_attn": "fa2",
        "applicable_to": ["all"],
        "memory_improvement": "2-4x",
        "speed_improvement": "1.5-2x",
    },
}


# ============================================================================
# LORA CONFIGURATION RECOMMENDATIONS
# ============================================================================

LORA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "description": "Minimum compute, quick experiments",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,v_proj",
        "trainable_params_percent": "~0.1%",
    },
    "standard": {
        "description": "Good balance of quality and efficiency",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,k_proj,v_proj,o_proj",
        "trainable_params_percent": "~0.3%",
    },
    "high_quality": {
        "description": "Higher quality, more compute",
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "lora_target": "all",
        "trainable_params_percent": "~1%",
    },
    "dora": {
        "description": "Weight-Decomposed LoRA for better quality",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "use_dora": True,
        "trainable_params_percent": "~0.5%",
    },
    "pissa": {
        "description": "PiSSA initialization for faster convergence",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_target": "all",
        "pissa_init": True,
        "pissa_iter": 16,
        "trainable_params_percent": "~0.5%",
    },
    "rslora": {
        "description": "Rank-Stabilized LoRA for stable training",
        "lora_rank": 64,
        "lora_alpha": 64,  # Same as rank for rsLoRA
        "lora_dropout": 0.05,
        "lora_target": "all",
        "use_rslora": True,
        "trainable_params_percent": "~1%",
    },
}

QLORA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "description": "Standard QLoRA with NF4",
        "quantization_bit": 4,
        "quantization_type": "nf4",
        "double_quantization": True,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_target": "all",
        "optim": "paged_adamw_8bit",
    },
    "extreme_memory": {
        "description": "Maximum memory efficiency",
        "quantization_bit": 4,
        "quantization_type": "nf4",
        "double_quantization": True,
        "lora_rank": 32,
        "lora_alpha": 8,
        "lora_target": "q_proj,v_proj",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
    },
}


# ============================================================================
# PPO MULTI-MODEL CONFIGURATION
# ============================================================================

PPO_BEST_PRACTICES: Dict[str, Any] = {
    # Core PPO settings
    "ppo_epochs": 4,
    "ppo_buffer_size": 1,
    "ppo_target_kl": 0.02,        # Early stopping threshold
    "ppo_score_norm": True,
    "ppo_whiten_rewards": True,

    # Model architecture
    "models_required": ["actor", "critic", "reference", "reward"],
    "actor": {
        "trainable": True,
        "add_value_head": True,
        "finetuning_type": "lora",  # or "full"
    },
    "critic": {
        "trainable": True,
        "separate_from_actor": False,  # Usually shared with actor
        "add_value_head": True,
    },
    "reference": {
        "trainable": False,
        "eval_mode": True,
        "quantization": "same_as_actor_base",  # Can quantize for memory
    },
    "reward": {
        "trainable": False,
        "eval_mode": True,
        "types": ["full", "lora", "api"],  # Model types supported
        "quantization": "optional",
    },

    # vLLM integration for inference during PPO
    "vllm_inference": {
        "vllm_device": "auto",
        "vllm_gpu_util": 0.9,
        "vllm_enforce_eager": False,
        "vllm_maxlen": 4096,
        "use_for": "generation_phase",
    },

    # Memory management
    "memory_tips": [
        "Use LoRA for actor to reduce memory",
        "Quantize reference model (4-bit) to save memory",
        "Use paged attention for generation",
        "Enable gradient checkpointing",
    ],
}

GRPO_BEST_PRACTICES: Dict[str, Any] = {
    "ppo_epochs": 1,
    "num_generations": 8,  # Number of generations per prompt
    "description": "Group Relative Policy Optimization - no reward model needed",
    "advantage": "Uses group-relative rewards instead of reward model",
}


# ============================================================================
# DISTRIBUTED TRAINING BEST PRACTICES
# ============================================================================

DISTRIBUTED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "single_gpu": {
        "description": "Single GPU training",
        "deepspeed": None,
        "fsdp": None,
        "recommended_method": "lora",
    },
    "multi_gpu_small": {
        "description": "2-4 GPUs, model fits in single GPU",
        "deepspeed": "zero_stage_1",
        "fsdp": None,
        "recommended_method": "lora",
        "use_cases": ["7B-13B models with LoRA"],
    },
    "multi_gpu_medium": {
        "description": "4-8 GPUs, need gradient sharding",
        "deepspeed": "zero_stage_2",
        "fsdp": "full_shard",
        "recommended_method": "lora",
        "use_cases": ["13B-70B models with LoRA", "7B full finetuning"],
    },
    "multi_gpu_large": {
        "description": "8+ GPUs, need full sharding",
        "deepspeed": "zero_stage_3",
        "fsdp": "full_shard",
        "recommended_method": "lora",
        "use_cases": ["70B+ models", "Full finetuning of large models"],
    },
    "multi_node": {
        "description": "Multiple nodes",
        "deepspeed": "zero_stage_3",
        "fsdp": "full_shard",
        "network_requirements": {
            "min_bandwidth_gbps": 100,
            "recommended_topology": "fat-tree",
        },
        "ddp_timeout": 180000000,  # 50 hours for large jobs
    },
}

ZERO_STAGE_SELECTION: Dict[str, Dict[str, Any]] = {
    "zero_0": {
        "use_when": "Model fits in single GPU with optimizer states",
        "memory_savings": "0%",
        "comm_overhead": "Minimal",
    },
    "zero_1": {
        "use_when": "Optimizer states don't fit, model fits",
        "memory_savings": "~4x optimizer memory",
        "comm_overhead": "Low",
    },
    "zero_2": {
        "use_when": "Gradients don't fit either",
        "memory_savings": "~8x optimizer + gradient memory",
        "comm_overhead": "Medium",
        "recommended_default": True,
    },
    "zero_3": {
        "use_when": "Model parameters don't fit in single GPU",
        "memory_savings": "Model params distributed",
        "comm_overhead": "High",
        "notes": "Significant communication overhead",
    },
    "zero_2_offload": {
        "use_when": "Need ZeRO-2 but limited GPU memory",
        "trade_off": "CPU<->GPU transfer overhead",
    },
    "zero_3_offload": {
        "use_when": "Very large models, limited GPU memory",
        "trade_off": "Significant CPU<->GPU transfer overhead",
    },
}


# ============================================================================
# DEEPSPEED CONFIGURATION TEMPLATES
# ============================================================================

DEEPSPEED_ZERO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "zero_stage_0": {
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": "auto"},
        "gradient_clipping": "auto",
    },

    "zero_stage_1": {
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },

    "zero_stage_2": {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },

    "zero_stage_2_offload": {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
        },
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },

    "zero_stage_3": {
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },

    "zero_stage_3_offload": {
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
        },
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
}


# ============================================================================
# REAL-WORLD SCENARIOS AND RECOMMENDATIONS
# ============================================================================

SCENARIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "quick_experiment": {
        "description": "Fast iteration for testing",
        "method": "lora",
        "lora_rank": 8,
        "max_samples": 1000,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
    },
    "production_sft": {
        "description": "Production SFT training",
        "method": "lora",
        "lora_config": "standard",
        "stability_config": "balanced",
        "convergence_config": ["neftune", "packing", "flash_attention"],
        "num_train_epochs": 3,
        "save_strategy": "steps",
        "save_steps": 500,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
    },
    "memory_constrained": {
        "description": "Training on limited GPU memory",
        "method": "qlora",
        "qlora_config": "extreme_memory",
        "deepspeed": "zero_stage_2_offload",
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
    },
    "high_quality_alignment": {
        "description": "High-quality DPO/PPO alignment",
        "method": "lora",
        "lora_config": "dora",
        "stability_config": "conservative",
        "num_train_epochs": 1,  # Alignment usually needs fewer epochs
        "warmup_ratio": 0.1,
    },
    "large_scale_pretraining": {
        "description": "Pre-training or continued pre-training",
        "method": "full",
        "deepspeed": "zero_stage_3",
        "stability_config": "balanced",
        "packing": True,
        "streaming": True,  # For large datasets
    },
}


# ============================================================================
# HARDWARE-SPECIFIC OPTIMIZATIONS
# ============================================================================

HARDWARE_SPECIFIC: Dict[str, Dict[str, Any]] = {
    "h100": {
        "flash_attn": "fa2",
        "bf16": True,
        "tf32": True,
        "torch_compile": True,
        "torch_compile_backend": "inductor",
    },
    "h200": {
        "flash_attn": "fa2",
        "bf16": True,
        "tf32": True,
        "torch_compile": True,
        "torch_compile_backend": "inductor",
    },
    "a100": {
        "flash_attn": "fa2",
        "bf16": True,
        "tf32": True,
        "torch_compile": False,  # Less benefit
    },
    "a10g": {
        "flash_attn": "fa2",
        "bf16": True,
        "quantization": "recommended",  # Limited memory
    },
    "v100": {
        "flash_attn": "sdpa",  # No FA2 support
        "fp16": True,
        "bf16": False,
    },
    "l40s": {
        "flash_attn": "fa2",
        "bf16": True,
        "notes": "Good for inference, training limited by memory BW",
    },
    "b100": {
        "flash_attn": "fa2",
        "bf16": True,
        "fp8": True,
        "tf32": True,
        "torch_compile": True,
    },
    "mi300x": {
        "flash_attn": "fa2",
        "bf16": True,
        "notes": "Use ROCm-compatible settings",
    },
}


# ============================================================================
# COMMON ISSUES AND FIXES
# ============================================================================

TROUBLESHOOTING: Dict[str, Dict[str, Any]] = {
    "loss_nan": {
        "causes": ["Learning rate too high", "Gradient explosion", "Data issues"],
        "fixes": [
            "Reduce learning rate by 10x",
            "Enable gradient clipping (max_grad_norm=1.0)",
            "Check for NaN/Inf in dataset",
            "Enable upcast_layernorm",
        ],
    },
    "oom_error": {
        "causes": ["Batch size too large", "Sequence too long", "Model too large"],
        "fixes": [
            "Reduce batch size",
            "Enable gradient checkpointing",
            "Use DeepSpeed ZeRO",
            "Switch to LoRA/QLoRA",
            "Reduce sequence length",
        ],
    },
    "slow_training": {
        "causes": ["No Flash Attention", "No packing", "High communication"],
        "fixes": [
            "Enable flash_attn=fa2",
            "Enable packing",
            "Use overlap_comm in DeepSpeed",
            "Increase batch size (within memory)",
        ],
    },
    "poor_convergence": {
        "causes": ["LR too low", "Too few epochs", "Data quality"],
        "fixes": [
            "Increase learning rate",
            "Add more training epochs",
            "Enable NEFTune",
            "Check data distribution",
        ],
    },
    "validation_loss_increasing": {
        "causes": ["Overfitting", "Learning rate too high late in training"],
        "fixes": [
            "Add dropout or increase dropout",
            "Use cosine scheduler with warm restarts",
            "Early stopping",
            "More regularization (weight decay)",
        ],
    },
}


# ============================================================================
# PRECISION CONFIGURATIONS
# ============================================================================

PRECISION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bf16": {
        "bytes_per_param": 2,
        "supports_training": True,
        "recommended_for": ["H100", "A100", "B100", "H200"],
        "notes": "Best for training, larger dynamic range than FP16",
    },
    "fp16": {
        "bytes_per_param": 2,
        "supports_training": True,
        "recommended_for": ["V100", "older GPUs"],
        "notes": "Requires loss scaling for training stability",
    },
    "fp8": {
        "bytes_per_param": 1,
        "supports_training": True,
        "recommended_for": ["H100", "H200", "B100"],
        "notes": "Experimental, use with Transformer Engine",
    },
    "int8": {
        "bytes_per_param": 1,
        "supports_training": False,
        "recommended_for": ["inference", "qlora base"],
        "notes": "Inference only, not for training",
    },
    "nf4": {
        "bytes_per_param": 0.5,
        "supports_training": False,
        "recommended_for": ["qlora"],
        "notes": "QLoRA quantization format",
    },
    "fp32": {
        "bytes_per_param": 4,
        "supports_training": True,
        "recommended_for": ["debugging", "precision-sensitive"],
        "notes": "Slow and memory-hungry, avoid if possible",
    },
}


def get_precision_bytes(precision: str) -> float:
    """Get bytes per parameter for a precision format."""
    precision = precision.lower()
    if precision in PRECISION_CONFIGS:
        return PRECISION_CONFIGS[precision]["bytes_per_param"]
    # Common aliases
    aliases = {
        "bfloat16": 2, "float16": 2, "float32": 4,
        "f32": 4, "f16": 2, "bf16": 2, "int4": 0.5,
    }
    return aliases.get(precision, 2)  # Default to 2 bytes (bf16)


def get_optimizer_memory_multiplier(optimizer: str) -> int:
    """Get optimizer state memory multiplier (bytes per trainable param)."""
    optimizer = optimizer.lower()
    return OPTIMIZER_STATE_MULTIPLIERS.get(optimizer, 8)  # Default to AdamW


def get_training_type_memory_multiplier(training_type: str) -> float:
    """Get memory multiplier for training type (for multi-model stages)."""
    training_type = training_type.lower()
    return TRAINING_TYPE_MEMORY_MULTIPLIERS.get(training_type, 1.0)
