"""
Comprehensive LlamaFactory Configuration Builder.

This module generates comprehensive training configurations with best practices:
- Full LlamaFactory YAML configuration
- DeepSpeed JSON configuration
- Accelerate YAML configuration
- PPO multi-model configurations (actor, reference, reward)
- vLLM inference configuration for PPO
- Launch commands (torchrun, deepspeed)

Best practices applied based on optimization focus:
- stable: Conservative settings for training stability
- convergence: Settings optimized for model quality
- speed: Settings optimized for throughput
- tco: Settings optimized for cost efficiency
"""

from typing import Dict, Any, List, Optional, Union
from copy import deepcopy

from .cluster_ranking_types import (
    ComprehensiveLlamaFactoryConfig,
    PPOModelConfig,
)
from .cluster_optimizer_types import (
    ClusterDefinition,
    TrainingJobSpec,
    ParallelismStrategy,
)
from .best_practices import (
    TRAINING_TYPE_DEFAULTS,
    LEARNING_RATES,
    STABILITY_CONFIGS,
    CONVERGENCE_CONFIGS,
    LORA_CONFIGS,
    QLORA_CONFIGS,
    PPO_BEST_PRACTICES,
    GRPO_BEST_PRACTICES,
    HARDWARE_SPECIFIC,
    DEEPSPEED_ZERO_CONFIGS,
    get_recommended_learning_rate,
    get_precision_bytes,
)
from .llamafactory_config_builder import (
    build_llamafactory_config,
    build_deepspeed_config,
    generate_launch_command,
)

# Import GenZ for simulation
try:
    from ..genz.LLM_training.training_modeling import training_modeling
    from ..genz.Models import get_configs
    HAS_GENZ = True
except ImportError:
    HAS_GENZ = False


def _estimate_model_size(model: Union[str, Dict[str, Any]]) -> float:
    """Estimate model size in billions of parameters."""
    if isinstance(model, dict):
        # Direct config - estimate from architecture
        hidden = model.get('hidden_size', 4096)
        layers = model.get('num_decoder_layers', model.get('num_layers', 32))
        vocab = model.get('vocab_size', 32000)
        intermediate = model.get('intermediate_size', hidden * 4)
        params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
        total = layers * params_per_layer + 2 * vocab * hidden
        return total / 1e9

    # Try GenZ
    if HAS_GENZ:
        try:
            config = get_configs(model)
            hidden = getattr(config, 'hidden_size', 4096) or 4096
            layers = getattr(config, 'num_decoder_layers', 32) or 32
            vocab = getattr(config, 'vocab_size', 32000) or 32000
            intermediate = getattr(config, 'intermediate_size', None) or (hidden * 4)
            params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
            total = layers * params_per_layer + 2 * vocab * hidden
            return total / 1e9
        except Exception:
            pass

    # Estimate from name
    model_lower = model.lower()
    if '405b' in model_lower or '400b' in model_lower:
        return 405
    elif '70b' in model_lower or '72b' in model_lower:
        return 70
    elif '32b' in model_lower or '34b' in model_lower:
        return 32
    elif '13b' in model_lower or '14b' in model_lower:
        return 13
    elif '8b' in model_lower or '7b' in model_lower:
        return 8
    elif '3b' in model_lower:
        return 3
    elif '1b' in model_lower:
        return 1
    return 8  # Default


def _apply_stability_config(config: Dict[str, Any], focus: str) -> List[str]:
    """Apply stability settings to config."""
    applied = []

    if focus not in STABILITY_CONFIGS:
        focus = "balanced"

    stability = STABILITY_CONFIGS[focus]

    if "warmup_ratio" in stability:
        config["warmup_ratio"] = stability["warmup_ratio"]
        applied.append(f"warmup_ratio={stability['warmup_ratio']}")

    if "lr_scheduler_type" in stability:
        config["lr_scheduler_type"] = stability["lr_scheduler_type"]
        applied.append(f"scheduler={stability['lr_scheduler_type']}")

    if "max_grad_norm" in stability:
        config["max_grad_norm"] = stability["max_grad_norm"]
        applied.append(f"gradient_clipping={stability['max_grad_norm']}")

    if "weight_decay" in stability:
        config["weight_decay"] = stability["weight_decay"]

    if stability.get("gradient_checkpointing", True):
        config["gradient_checkpointing"] = True

    if stability.get("bf16", True):
        config["bf16"] = True
        config["fp16"] = False

    if stability.get("upcast_layernorm"):
        config["upcast_layernorm"] = True
        applied.append("upcast_layernorm")

    if stability.get("upcast_lmhead_output"):
        config["upcast_lmhead_output"] = True
        applied.append("upcast_lmhead_output")

    return applied


def _apply_convergence_config(config: Dict[str, Any], training_type: str, focus: str) -> List[str]:
    """Apply convergence optimization settings."""
    applied = []

    # NEFTune (noise embedding) for SFT
    if training_type in ("sft", "pt") and focus in ("convergence", "balanced"):
        neftune = CONVERGENCE_CONFIGS.get("neftune", {})
        if "neftune_noise_alpha" in neftune:
            config["neftune_noise_alpha"] = neftune["neftune_noise_alpha"]
            applied.append(f"neftune_alpha={neftune['neftune_noise_alpha']}")

    # Packing for efficiency
    if training_type in ("sft", "pt", "dpo") and focus in ("speed", "balanced"):
        packing = CONVERGENCE_CONFIGS.get("packing", {})
        if packing.get("packing"):
            config["packing"] = True
            config["neat_packing"] = packing.get("neat_packing", True)
            applied.append("sequence_packing")

    # Flash attention
    flash = CONVERGENCE_CONFIGS.get("flash_attention", {})
    if flash.get("flash_attn"):
        config["flash_attn"] = flash["flash_attn"]
        applied.append(f"flash_attn={flash['flash_attn']}")

    return applied


def _apply_lora_best_practices(config: Dict[str, Any], method: str, model_size_b: float) -> List[str]:
    """Apply LoRA best practices based on method."""
    applied = []

    if method not in ("lora", "dora", "pissa"):
        return applied

    # Select LoRA config based on model size
    if model_size_b > 70:
        lora_config = LORA_CONFIGS.get("high_quality", LORA_CONFIGS["standard"])
    elif model_size_b > 13:
        lora_config = LORA_CONFIGS.get("standard", {})
    else:
        lora_config = LORA_CONFIGS.get("standard", {})

    # Override with method-specific config
    if method == "dora":
        lora_config = LORA_CONFIGS.get("dora", lora_config)
        config["use_dora"] = True
        applied.append("dora")
    elif method == "pissa":
        lora_config = LORA_CONFIGS.get("pissa", lora_config)
        config["pissa_init"] = True
        config["pissa_iter"] = lora_config.get("pissa_iter", 16)
        applied.append("pissa_init")

    # Apply LoRA settings
    if "lora_rank" in lora_config:
        config["lora_rank"] = lora_config["lora_rank"]
    if "lora_alpha" in lora_config:
        config["lora_alpha"] = lora_config["lora_alpha"]
    if "lora_dropout" in lora_config:
        config["lora_dropout"] = lora_config["lora_dropout"]
    if "lora_target" in lora_config:
        config["lora_target"] = lora_config["lora_target"]

    applied.append(f"lora_rank={config.get('lora_rank', 16)}")

    return applied


def _apply_qlora_best_practices(config: Dict[str, Any], focus: str) -> List[str]:
    """Apply QLoRA best practices."""
    applied = []

    if focus == "tco":
        qlora = QLORA_CONFIGS.get("extreme_memory", QLORA_CONFIGS["standard"])
    else:
        qlora = QLORA_CONFIGS.get("standard", {})

    config["quantization_bit"] = qlora.get("quantization_bit", 4)
    config["quantization_type"] = qlora.get("quantization_type", "nf4")
    config["double_quantization"] = qlora.get("double_quantization", True)

    if "lora_rank" in qlora:
        config["lora_rank"] = qlora["lora_rank"]
    if "lora_alpha" in qlora:
        config["lora_alpha"] = qlora["lora_alpha"]
    if "lora_target" in qlora:
        config["lora_target"] = qlora["lora_target"]
    if "optim" in qlora:
        config["optim"] = qlora["optim"]

    applied.append(f"qlora_nf4_rank={config.get('lora_rank', 64)}")
    applied.append("double_quantization")

    return applied


def _apply_hardware_optimizations(config: Dict[str, Any], gpu_type: str) -> List[str]:
    """Apply hardware-specific optimizations."""
    applied = []

    # Normalize GPU type
    gpu_key = gpu_type.lower().replace("_", "").replace("-", "")

    # Find matching hardware config
    hw_config = None
    for key, conf in HARDWARE_SPECIFIC.items():
        if key.lower().replace("_", "") in gpu_key or gpu_key in key.lower().replace("_", ""):
            hw_config = conf
            break

    if hw_config is None:
        return applied

    if hw_config.get("flash_attn"):
        config["flash_attn"] = hw_config["flash_attn"]
        applied.append(f"flash_attn={hw_config['flash_attn']}")

    if hw_config.get("bf16") and not hw_config.get("fp16"):
        config["bf16"] = True
        config["fp16"] = False
    elif hw_config.get("fp16"):
        config["bf16"] = False
        config["fp16"] = True

    if hw_config.get("tf32"):
        config["tf32"] = True
        applied.append("tf32")

    if hw_config.get("torch_compile"):
        config["torch_compile"] = True
        if "torch_compile_backend" in hw_config:
            config["torch_compile_backend"] = hw_config["torch_compile_backend"]
        applied.append("torch_compile")

    return applied


def _build_ppo_configs(
    job_spec: TrainingJobSpec,
    cluster: ClusterDefinition,
    model_size_b: float,
) -> Dict[str, Dict[str, Any]]:
    """Build PPO-specific multi-model configurations."""
    model = job_spec.model

    # Actor config (trainable)
    actor_config = {
        "model_name_or_path": model,
        "finetuning_type": job_spec.method,
        "trainable": True,
        "add_value_head": True,
    }

    if job_spec.method in ("lora", "qlora", "dora", "pissa"):
        actor_config.update({
            "lora_rank": job_spec.lora_rank,
            "lora_alpha": job_spec.lora_alpha,
            "lora_target": job_spec.lora_target,
        })

    # Reference config (frozen, same as actor base)
    reference_config = {
        "model_name_or_path": model,
        "trainable": False,
        "eval_mode": True,
    }

    # Quantize reference for memory efficiency on large models
    if model_size_b > 13 and job_spec.method in ("lora", "dora", "pissa"):
        reference_config["quantization_bit"] = 4
        reference_config["quantization_type"] = "nf4"

    # Reward config (frozen)
    reward_config = {
        "model_name_or_path": None,  # Must be set by user
        "trainable": False,
        "eval_mode": True,
    }

    # vLLM config for generation phase
    vllm_config = deepcopy(PPO_BEST_PRACTICES.get("vllm_inference", {}))
    vllm_config["model_name_or_path"] = model

    return {
        "actor": actor_config,
        "reference": reference_config,
        "reward": reward_config,
        "vllm": vllm_config,
    }


def _build_deepspeed_with_best_practices(
    parallelism: ParallelismStrategy,
    precision: str,
    optimizer: str,
    focus: str,
) -> Dict[str, Any]:
    """Build DeepSpeed config with best practices."""
    zero_stage = parallelism.zero_stage

    # Select base config
    if zero_stage == 0:
        base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_0", {}))
    elif zero_stage == 1:
        base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_1", {}))
    elif zero_stage == 2:
        if focus == "tco":
            base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_2_offload", {}))
        else:
            base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_2", {}))
    else:
        if focus == "tco":
            base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_3_offload", {}))
        else:
            base = deepcopy(DEEPSPEED_ZERO_CONFIGS.get("zero_stage_3", {}))

    # Precision
    if precision in ("bf16", "bfloat16"):
        base["bf16"] = {"enabled": True}
        base["fp16"] = {"enabled": False}
    elif precision in ("fp16", "float16"):
        base["bf16"] = {"enabled": False}
        base["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }

    # Speed optimizations
    if focus == "speed" and "zero_optimization" in base:
        base["zero_optimization"]["overlap_comm"] = True
        base["zero_optimization"]["contiguous_gradients"] = True

    return base


def _build_accelerate_config(
    cluster: ClusterDefinition,
    parallelism: ParallelismStrategy,
    precision: str,
) -> Dict[str, Any]:
    """Build Accelerate configuration."""
    num_nodes = cluster.num_nodes
    gpus_per_node = cluster.gpus_per_node

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "DEEPSPEED" if parallelism.zero_stage > 0 else "MULTI_GPU",
        "downcast_bf16": False,
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": precision,
        "num_machines": num_nodes,
        "num_processes": cluster.num_gpus,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }

    if parallelism.zero_stage > 0:
        config["deepspeed_config"] = {
            "deepspeed_multinode_launcher": "standard" if num_nodes > 1 else "none",
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "zero3_init_flag": parallelism.zero_stage == 3,
            "zero_stage": parallelism.zero_stage,
        }

    return config


def generate_comprehensive_training_config(
    job_spec: TrainingJobSpec,
    parallelism: ParallelismStrategy,
    cluster: ClusterDefinition,
    optimization_focus: str = "balanced",
    enable_best_practices: bool = True,
) -> ComprehensiveLlamaFactoryConfig:
    """
    Generate comprehensive LlamaFactory configuration with best practices.

    Args:
        job_spec: Training job specification
        parallelism: Parallelism strategy
        cluster: Cluster configuration
        optimization_focus: Focus area (stable, convergence, speed, tco, balanced)
        enable_best_practices: Whether to apply best practices

    Returns:
        ComprehensiveLlamaFactoryConfig with all configurations
    """
    best_practices_applied: List[str] = []
    notes: List[str] = []
    warnings: List[str] = []

    # Estimate model size
    model_size_b = _estimate_model_size(job_spec.model)

    # Start with base LlamaFactory config
    config = build_llamafactory_config(
        job_spec=job_spec,
        parallelism=parallelism,
        cluster=cluster,
    )

    if enable_best_practices:
        # Apply model-size-aware learning rate
        lr = get_recommended_learning_rate(job_spec.method, model_size_b)
        config["learning_rate"] = lr
        best_practices_applied.append(f"lr={lr} for {model_size_b:.0f}B model")

        # Apply stability settings
        stability_applied = _apply_stability_config(config, optimization_focus)
        best_practices_applied.extend(stability_applied)

        # Apply convergence settings
        convergence_applied = _apply_convergence_config(config, job_spec.training_type, optimization_focus)
        best_practices_applied.extend(convergence_applied)

        # Apply method-specific best practices
        if job_spec.method in ("lora", "dora", "pissa"):
            lora_applied = _apply_lora_best_practices(config, job_spec.method, model_size_b)
            best_practices_applied.extend(lora_applied)
        elif job_spec.method == "qlora":
            qlora_applied = _apply_qlora_best_practices(config, optimization_focus)
            best_practices_applied.extend(qlora_applied)

        # Apply hardware-specific optimizations
        hw_applied = _apply_hardware_optimizations(config, cluster.gpu_type)
        best_practices_applied.extend(hw_applied)

        # Apply training type defaults
        if job_spec.training_type in TRAINING_TYPE_DEFAULTS:
            defaults = TRAINING_TYPE_DEFAULTS[job_spec.training_type]
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value

    # Build DeepSpeed config
    ds_config = None
    if parallelism.zero_stage > 0 or parallelism.data_parallel > 1:
        ds_config = _build_deepspeed_with_best_practices(
            parallelism=parallelism,
            precision=job_spec.precision,
            optimizer=job_spec.optimizer,
            focus=optimization_focus,
        )

    # Build Accelerate config
    accel_config = _build_accelerate_config(
        cluster=cluster,
        parallelism=parallelism,
        precision=job_spec.precision,
    )

    # Build PPO-specific configs
    ppo_actor = None
    ppo_reward = None
    ppo_reference = None
    vllm_config = None

    if job_spec.training_type == "ppo":
        ppo_configs = _build_ppo_configs(job_spec, cluster, model_size_b)
        ppo_actor = ppo_configs["actor"]
        ppo_reference = ppo_configs["reference"]
        ppo_reward = ppo_configs["reward"]
        vllm_config = ppo_configs["vllm"]

        # Apply PPO best practices
        ppo_defaults = PPO_BEST_PRACTICES
        config["ppo_epochs"] = ppo_defaults.get("ppo_epochs", 4)
        config["ppo_buffer_size"] = ppo_defaults.get("ppo_buffer_size", 1)
        config["ppo_score_norm"] = ppo_defaults.get("ppo_score_norm", True)
        config["ppo_whiten_rewards"] = ppo_defaults.get("ppo_whiten_rewards", True)

        if ppo_defaults.get("ppo_target_kl"):
            config["ppo_target_kl"] = ppo_defaults["ppo_target_kl"]

        best_practices_applied.extend([
            "ppo_epochs=4",
            "ppo_score_norm",
            "ppo_whiten_rewards",
        ])

        warnings.append("PPO requires reward_model to be specified separately")
        notes.extend(ppo_defaults.get("memory_tips", []))

    elif job_spec.training_type == "grpo":
        grpo_defaults = GRPO_BEST_PRACTICES
        config["ppo_epochs"] = grpo_defaults.get("ppo_epochs", 1)
        config["num_generations"] = grpo_defaults.get("num_generations", 8)
        best_practices_applied.extend([
            "grpo_num_generations=8",
        ])
        notes.append("GRPO uses group-relative rewards (no reward model needed)")

    # Generate launch commands
    torchrun_cmd = generate_launch_command(
        config_path="training_config.yaml",
        num_gpus=cluster.num_gpus,
        num_nodes=cluster.num_nodes,
        deepspeed_config=None,
    )

    deepspeed_cmd = ""
    if ds_config:
        deepspeed_cmd = generate_launch_command(
            config_path="training_config.yaml",
            num_gpus=cluster.num_gpus,
            num_nodes=cluster.num_nodes,
            deepspeed_config=f"ds_z{parallelism.zero_stage}_config.json",
        )

    # Estimate performance
    estimated_tps = 0.0
    estimated_hours = 0.0
    estimated_cost = 0.0

    if HAS_GENZ and isinstance(job_spec.model, str):
        try:
            from ..genz.LLM_training.training_modeling import training_modeling as tm

            # GPU to system mapping
            gpu_to_system = {
                "h100": "H100_GPU", "h100_sxm": "H100_GPU",
                "a100_80gb": "A100_80GB_GPU", "a100_40gb": "A100_40GB_GPU",
                "h200": "H200_GPU",
            }
            system_name = gpu_to_system.get(cluster.gpu_type.lower(), "H100_GPU")

            result = tm(
                model=job_spec.model,
                training_stage=job_spec.training_type,
                batch_size=job_spec.batch_size or 4,
                seq_length=job_spec.avg_sequence_length,
                system_name=system_name,
                num_gpus=cluster.num_gpus,
                tensor_parallel=parallelism.tensor_parallel,
                data_parallel=parallelism.data_parallel,
                pipeline_parallel=parallelism.pipeline_parallel,
                method=job_spec.method,
                optimizer=job_spec.optimizer,
                zero_stage=parallelism.zero_stage,
                gradient_checkpointing=parallelism.gradient_checkpointing,
                bits=job_spec.precision,
            )

            estimated_tps = result.tokens_per_second
            if estimated_tps > 0 and job_spec.total_tokens > 0:
                estimated_hours = job_spec.total_tokens / (estimated_tps * 3600)
                estimated_cost = estimated_hours * cluster.total_hourly_rate

        except Exception:
            pass

    return ComprehensiveLlamaFactoryConfig(
        llamafactory_yaml=config,
        deepspeed_json=ds_config,
        accelerate_yaml=accel_config,
        ppo_actor_config=ppo_actor,
        ppo_reward_config=ppo_reward,
        ppo_reference_config=ppo_reference,
        vllm_inference_config=vllm_config,
        best_practices_applied=best_practices_applied,
        optimization_focus=optimization_focus,
        torchrun_command=torchrun_cmd,
        deepspeed_command=deepspeed_cmd,
        expected_throughput_tps=estimated_tps,
        expected_training_hours=estimated_hours,
        expected_cost_usd=estimated_cost,
        notes=notes,
        warnings=warnings,
    )
