"""
LlamaFactory Configuration Builder.

This module generates LlamaFactory-compatible configuration files (YAML)
and DeepSpeed configurations for distributed training.

Supports:
- All training stages (sft, dpo, ppo, kto, rm, grpo)
- All fine-tuning methods (full, lora, qlora, dora, freeze)
- DeepSpeed ZeRO stages (0-3)
- Distributed training configurations
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .cluster_optimizer_types import (
    TrainingJobSpec,
    ParallelismStrategy,
    ClusterDefinition,
)


# ============================================================================
# LlamaFactory Feature Mappings
# ============================================================================
# Based on: https://llamafactory.readthedocs.io/en/latest/

# Supported fine-tuning types
SUPPORTED_FINETUNING_TYPES = {
    "full": "Full parameter fine-tuning",
    "lora": "Low-Rank Adaptation",
    "qlora": "Quantized LoRA (4-bit)",
    "dora": "Weight-Decomposed Low-Rank Adaptation",
    "pissa": "Principal Singular Values and Singular Vectors Adaptation",
    "freeze": "Freeze fine-tuning (train selected layers only)",
}

# Supported training stages
SUPPORTED_TRAINING_STAGES = {
    "sft": "Supervised Fine-Tuning",
    "pt": "Pre-Training (incremental)",
    "dpo": "Direct Preference Optimization",
    "kto": "Kahneman-Tversky Optimization",
    "ppo": "Proximal Policy Optimization",
    "rm": "Reward Modeling",
    "orpo": "Odds Ratio Preference Optimization",
    "simpo": "Simple Preference Optimization",
    "grpo": "Group Relative Policy Optimization",
}

# Quantization methods supported by LlamaFactory
QUANTIZATION_METHODS = {
    "bitsandbytes": "BitsAndBytes quantization (default)",
    "hqq": "Half-Quadratic Quantization",
    "eetq": "Efficient 8-bit Quantization",
    "gptq": "GPTQ post-training quantization",
    "awq": "Activation-aware Weight Quantization",
    "aqlm": "Additive Quantization of Language Models",
}

# Quantization bit options
QUANTIZATION_BITS = [2, 3, 4, 5, 6, 8]  # Supported bit widths


def build_llamafactory_config(
    job_spec: TrainingJobSpec,
    parallelism: ParallelismStrategy,
    cluster: ClusterDefinition,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate LlamaFactory-compatible configuration.

    Args:
        job_spec: Training job specification
        parallelism: Parallelism configuration
        cluster: Cluster configuration
        output_dir: Output directory for checkpoints
        run_name: Name for the training run

    Returns:
        Dictionary with LlamaFactory configuration
    """
    # Base configuration
    config = {
        # Model configuration
        "model_name_or_path": job_spec.model,
        "trust_remote_code": True,

        # Stage configuration
        "stage": job_spec.training_type,
        "do_train": True,

        # Fine-tuning type
        "finetuning_type": job_spec.method,
    }

    # Batch size and accumulation
    batch_size = job_spec.batch_size or 4
    config["per_device_train_batch_size"] = batch_size
    config["gradient_accumulation_steps"] = parallelism.gradient_accumulation_steps

    # Learning rate configuration
    config["learning_rate"] = _get_learning_rate(job_spec.method, job_spec.optimizer)
    config["lr_scheduler_type"] = "cosine"
    config["warmup_ratio"] = 0.1

    # Training duration
    if job_spec.num_epochs > 0:
        config["num_train_epochs"] = job_spec.num_epochs
    else:
        # Estimate steps from tokens
        total_tokens = job_spec.total_tokens
        tokens_per_step = batch_size * job_spec.avg_sequence_length * parallelism.data_parallel * parallelism.gradient_accumulation_steps
        if tokens_per_step > 0:
            config["max_steps"] = max(100, total_tokens // tokens_per_step)

    # Precision configuration
    precision = job_spec.precision.lower()
    if precision in ("bf16", "bfloat16"):
        config["bf16"] = True
        config["fp16"] = False
    elif precision in ("fp16", "float16"):
        config["bf16"] = False
        config["fp16"] = True
    elif precision in ("fp8", "fp8_e4m3"):
        config["bf16"] = True
        config["use_fp8"] = True
    else:
        config["bf16"] = True  # Default to bf16

    # Memory optimization
    config["gradient_checkpointing"] = parallelism.gradient_checkpointing
    if parallelism.gradient_checkpointing:
        config["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Sequence length
    config["cutoff_len"] = job_spec.avg_sequence_length

    # LoRA configuration
    if job_spec.method in ("lora", "qlora", "dora", "pissa"):
        config.update(_build_lora_config(job_spec))

    # QLoRA quantization
    if job_spec.method == "qlora":
        quant_bit = getattr(job_spec, 'quantization_bit', 4)
        quant_method = getattr(job_spec, 'quantization_method', 'bitsandbytes')
        config["quantization_bit"] = quant_bit
        config["quantization_method"] = quant_method

    # Freeze fine-tuning configuration
    if job_spec.method == "freeze":
        config.update(_build_freeze_config(job_spec))

    # Flash Attention configuration
    flash_attn = getattr(job_spec, 'flash_attn', 'auto')
    if flash_attn != "auto":
        config["flash_attn"] = flash_attn

    # GaLore optimizer (memory-efficient gradient projection)
    config.update(_build_galore_config(job_spec))

    # BAdam optimizer (block-wise Adam)
    config.update(_build_badam_config(job_spec))

    # Optimizer configuration
    config.update(_build_optimizer_config(job_spec.optimizer))

    # DeepSpeed configuration (if needed)
    if parallelism.zero_stage > 0 or parallelism.data_parallel > 1:
        config["deepspeed"] = f"ds_z{parallelism.zero_stage}_config.json"

    # Output configuration
    if output_dir:
        config["output_dir"] = output_dir
    else:
        config["output_dir"] = f"./outputs/{run_name or 'training'}"

    if run_name:
        config["run_name"] = run_name

    # Logging and saving
    config["logging_steps"] = 10
    config["save_steps"] = 1000
    config["save_total_limit"] = 3

    # Stage-specific configurations
    config.update(_build_stage_config(job_spec.training_type, job_spec.method))

    return config


def _get_learning_rate(method: str, optimizer: str) -> float:
    """Get recommended learning rate based on method and optimizer."""
    # LoRA methods use higher learning rates
    if method in ("lora", "qlora", "dora", "pissa"):
        if optimizer in ("adam_8bit", "adamw_8bit", "paged_adamw_8bit"):
            return 1e-4
        return 2e-4

    # Full fine-tuning uses lower learning rates
    if optimizer in ("adam_8bit", "adamw_8bit", "paged_adamw_8bit"):
        return 5e-6
    return 2e-5


def _build_lora_config(job_spec: TrainingJobSpec) -> Dict[str, Any]:
    """Build LoRA-specific configuration."""
    config = {
        "lora_rank": job_spec.lora_rank,
        "lora_alpha": job_spec.lora_alpha,
        "lora_dropout": job_spec.lora_dropout,
    }

    # Target modules
    if job_spec.lora_target == "all":
        config["lora_target"] = "all"
    else:
        config["lora_target"] = job_spec.lora_target

    # DoRA-specific (Weight-Decomposed Low-Rank Adaptation)
    if job_spec.method == "dora":
        config["use_dora"] = True

    # PiSSA-specific (Principal Singular Values and Singular Vectors Adaptation)
    if job_spec.method == "pissa":
        config["pissa_init"] = True

    # rsLoRA (Rank-Stabilized LoRA) - modifies scaling factor for stability
    if getattr(job_spec, 'use_rslora', False):
        config["use_rslora"] = True

    # LoRA+ (Different learning rates for A and B matrices)
    if getattr(job_spec, 'use_lora_plus', False):
        config["loraplus_lr_ratio"] = getattr(job_spec, 'lora_plus_lr_ratio', 16.0)

    return config


def _build_freeze_config(job_spec: TrainingJobSpec) -> Dict[str, Any]:
    """Build freeze fine-tuning configuration."""
    config = {}

    # Number of trainable layers (from the end of the model)
    freeze_layers = getattr(job_spec, 'freeze_trainable_layers', 2)
    config["freeze_trainable_layers"] = freeze_layers

    # Trainable modules within unfrozen layers
    freeze_modules = getattr(job_spec, 'freeze_trainable_modules', 'all')
    config["freeze_trainable_modules"] = freeze_modules

    # Extra modules to train (e.g., "embed_tokens,lm_head")
    extra_modules = getattr(job_spec, 'freeze_extra_modules', None)
    if extra_modules:
        config["freeze_extra_modules"] = extra_modules

    return config


def _build_galore_config(job_spec: TrainingJobSpec) -> Dict[str, Any]:
    """Build GaLore (Gradient Low-Rank Projection) optimizer configuration."""
    config = {}

    if getattr(job_spec, 'use_galore', False):
        config["use_galore"] = True
        config["galore_rank"] = getattr(job_spec, 'galore_rank', 128)
        config["galore_target"] = getattr(job_spec, 'galore_target', 'all')
        config["galore_scale"] = getattr(job_spec, 'galore_scale', 1.0)

        # Layer-wise GaLore (applies GaLore per-layer for memory efficiency)
        if getattr(job_spec, 'galore_layerwise', False):
            config["galore_layerwise"] = True

    return config


def _build_badam_config(job_spec: TrainingJobSpec) -> Dict[str, Any]:
    """Build BAdam (Block-wise Adam) optimizer configuration."""
    config = {}

    if getattr(job_spec, 'use_badam', False):
        config["use_badam"] = True
        config["badam_mode"] = getattr(job_spec, 'badam_mode', 'layer')
        config["badam_switch_mode"] = getattr(job_spec, 'badam_switch_mode', 'ascending')
        config["badam_update_ratio"] = getattr(job_spec, 'badam_update_ratio', 0.05)

    return config


def _build_optimizer_config(optimizer: str) -> Dict[str, Any]:
    """Build optimizer-specific configuration."""
    optimizer = optimizer.lower()

    # Paged optimizers (memory efficient)
    if optimizer in ("paged_adamw_8bit", "paged_adam_8bit"):
        return {
            "optim": "paged_adamw_8bit",
        }

    # 8-bit optimizers
    if optimizer in ("adam_8bit", "adamw_8bit"):
        return {
            "optim": "adamw_8bit",
        }

    # Standard AdamW
    if optimizer in ("adam", "adamw"):
        return {
            "optim": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.01,
        }

    # Adafactor (memory efficient)
    if optimizer == "adafactor":
        return {
            "optim": "adafactor",
        }

    # Lion optimizer
    if optimizer == "lion":
        return {
            "optim": "adamw_torch",  # Use torch optimizer with Lion settings
            "adam_beta1": 0.95,
            "adam_beta2": 0.98,
        }

    # Default to standard AdamW
    return {
        "optim": "adamw_torch",
    }


def _build_stage_config(training_type: str, method: str) -> Dict[str, Any]:
    """Build training stage-specific configuration."""
    config = {}

    if training_type == "sft":
        config["template"] = "default"  # Can be overridden

    elif training_type == "dpo":
        config["pref_beta"] = 0.1
        config["pref_loss"] = "sigmoid"  # sigmoid, hinge, ipo, orpo, simpo

    elif training_type == "kto":
        config["pref_beta"] = 0.1

    elif training_type == "ppo":
        config["reward_model"] = None  # Must be set by user
        config["ppo_epochs"] = 4

    elif training_type == "grpo":
        config["pref_beta"] = 0.1
        config["num_generations"] = 4

    elif training_type == "rm":
        config["pref_loss"] = "sigmoid"

    return config


def build_deepspeed_config(
    parallelism: ParallelismStrategy,
    precision: str = "bf16",
    gradient_clipping: float = 1.0,
    optimizer: str = "adamw",
) -> Dict[str, Any]:
    """
    Generate DeepSpeed configuration.

    Args:
        parallelism: Parallelism configuration
        precision: Training precision
        gradient_clipping: Gradient clipping value
        optimizer: Optimizer type

    Returns:
        Dictionary with DeepSpeed configuration
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": parallelism.gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
    }

    # Precision configuration
    if precision in ("bf16", "bfloat16"):
        config["bf16"] = {"enabled": True}
        config["fp16"] = {"enabled": False}
    elif precision in ("fp16", "float16"):
        config["bf16"] = {"enabled": False}
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    else:
        config["bf16"] = {"enabled": True}

    # ZeRO optimization based on stage
    if parallelism.zero_stage == 0:
        # No ZeRO - just use communication optimizations
        config["zero_optimization"] = {
            "stage": 0,
        }

    elif parallelism.zero_stage == 1:
        config["zero_optimization"] = {
            "stage": 1,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
        }

    elif parallelism.zero_stage == 2:
        config["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        }

    elif parallelism.zero_stage == 3:
        config["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        }

    # Optimizer configuration for DeepSpeed
    if optimizer in ("adam_8bit", "adamw_8bit"):
        # Use built-in optimizer, not DeepSpeed's
        pass
    else:
        config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto",
            },
        }

    # Scheduler
    config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto",
        },
    }

    return config


def save_llamafactory_config(
    config: Dict[str, Any],
    output_path: str,
    name: str = "training_config",
) -> str:
    """
    Save LlamaFactory config to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output directory
        name: Config file name (without extension)

    Returns:
        Path to saved file
    """
    try:
        import yaml
        file_ext = ".yaml"
        writer = lambda f, c: yaml.dump(c, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON if yaml not available
        file_ext = ".json"
        writer = lambda f, c: json.dump(c, f, indent=2)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{name}{file_ext}"

    with open(file_path, 'w') as f:
        writer(f, config)

    return str(file_path)


def save_deepspeed_config(
    config: Dict[str, Any],
    output_path: str,
    zero_stage: int = 2,
) -> str:
    """
    Save DeepSpeed config to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Output directory
        zero_stage: ZeRO stage (for file naming)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"ds_z{zero_stage}_config.json"

    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(file_path)


def generate_training_configs(
    job_spec: TrainingJobSpec,
    parallelism: ParallelismStrategy,
    cluster: ClusterDefinition,
    output_dir: str,
    run_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate and save all training configuration files.

    Args:
        job_spec: Training job specification
        parallelism: Parallelism configuration
        cluster: Cluster configuration
        output_dir: Output directory for all config files
        run_name: Name for the training run

    Returns:
        Dictionary mapping config type to file path
    """
    saved_files = {}

    # Generate and save LlamaFactory config
    llama_config = build_llamafactory_config(
        job_spec=job_spec,
        parallelism=parallelism,
        cluster=cluster,
        output_dir=f"{output_dir}/checkpoints",
        run_name=run_name,
    )
    saved_files["llamafactory"] = save_llamafactory_config(
        llama_config, output_dir, "training_config"
    )

    # Generate and save DeepSpeed config if needed
    if parallelism.zero_stage > 0 or parallelism.data_parallel > 1:
        ds_config = build_deepspeed_config(
            parallelism=parallelism,
            precision=job_spec.precision,
            optimizer=job_spec.optimizer,
        )
        saved_files["deepspeed"] = save_deepspeed_config(
            ds_config, output_dir, parallelism.zero_stage
        )

    return saved_files


def generate_launch_command(
    config_path: str,
    num_gpus: int,
    num_nodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 29500,
    deepspeed_config: Optional[str] = None,
) -> str:
    """
    Generate the command to launch training.

    Args:
        config_path: Path to LlamaFactory config
        num_gpus: Total number of GPUs
        num_nodes: Number of nodes
        master_addr: Master node address
        master_port: Master port
        deepspeed_config: Path to DeepSpeed config (optional)

    Returns:
        Launch command string
    """
    gpus_per_node = num_gpus // num_nodes

    if deepspeed_config:
        # DeepSpeed launch
        cmd = f"""deepspeed --num_gpus={gpus_per_node} \\
    --num_nodes={num_nodes} \\
    --master_addr={master_addr} \\
    --master_port={master_port} \\
    src/train.py {config_path}"""
    elif num_gpus > 1:
        # Torchrun launch
        cmd = f"""torchrun --nproc_per_node={gpus_per_node} \\
    --nnodes={num_nodes} \\
    --master_addr={master_addr} \\
    --master_port={master_port} \\
    src/train.py {config_path}"""
    else:
        # Single GPU
        cmd = f"python src/train.py {config_path}"

    return cmd
