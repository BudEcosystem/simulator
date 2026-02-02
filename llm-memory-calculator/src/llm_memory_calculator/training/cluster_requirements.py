"""
Cluster Requirements Predictor.

This module predicts minimum cluster requirements for training configurations:
- Minimum GPU count and memory requirements
- Network bandwidth requirements
- Recommended parallelism and ZeRO stage
- Per-GPU type feasibility analysis

Integrates with:
- GenZ training_modeling for accurate simulation
- Training types for memory multipliers
- Hardware catalog for GPU specifications
"""

from typing import Dict, Any, List, Optional, Union
import math

from .cluster_ranking_types import (
    MinimumClusterRequirements,
    MemoryBreakdownDetails,
    GPURequirement,
    ClusterTopology,
)
from .cluster_optimizer_types import (
    ParallelismStrategy,
    ClusterDefinition,
    TrainingJobSpec,
)
from .training_types import (
    get_training_stage_config,
    TRAINING_STAGE_CONFIGS,
)
from .hardware_catalog import (
    GPU_SPECS,
    GPU_COSTS,
    get_gpu_spec,
)
from .best_practices import (
    get_precision_bytes,
    get_optimizer_memory_multiplier,
    get_training_type_memory_multiplier,
    ZERO_STAGE_SELECTION,
)
from .tco_calculator import calculate_tco, get_gpu_pricing

# Import GenZ for simulation
try:
    from ..genz.LLM_training.training_modeling import training_modeling
    from ..genz.LLM_training.training_parallelization import get_training_parallelization_options
    from ..genz.Models import get_configs
    HAS_GENZ = True
except ImportError:
    HAS_GENZ = False


# GPU memory mapping
GPU_MEMORY_GB: Dict[str, float] = {
    "v100_16gb": 16, "v100_32gb": 32,
    "a100_40gb": 40, "a100_80gb": 80,
    "l40s": 48, "rtx_4090": 24,
    "h100_sxm": 80, "h100_pcie": 80, "h100": 80,
    "h200": 141,
    "b100": 192, "b200": 192,
    "mi300x": 192,
}

# System name mapping
GPU_TO_SYSTEM_NAME: Dict[str, str] = {
    "v100_16gb": "V100_16GB_GPU", "v100_32gb": "V100_32GB_GPU",
    "a100_40gb": "A100_40GB_GPU", "a100_80gb": "A100_80GB_GPU",
    "l40s": "L40S_48GB_GPU",
    "h100_sxm": "H100_GPU", "h100_pcie": "H100_GPU", "h100": "H100_GPU",
    "h200": "H200_GPU",
    "b100": "B100_GPU", "b200": "B200_GPU",
    "mi300x": "MI300X_GPU",
}


def _get_gpu_memory(gpu_type: str) -> float:
    """Get GPU memory in GB."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[normalized]
    if gpu_type in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[gpu_type]
    # Try to get from hardware catalog
    try:
        spec = get_gpu_spec(gpu_type)
        return spec.memory_gb
    except (ValueError, KeyError):
        return 80.0  # Default to 80GB


def _get_system_name(gpu_type: str) -> str:
    """Map GPU type to GenZ system name."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[normalized]
    if gpu_type in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[gpu_type]
    return gpu_type


def _estimate_model_params(model: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate model parameters from model config or name."""
    if isinstance(model, dict):
        # Direct model config
        hidden = model.get('hidden_size', 4096)
        layers = model.get('num_decoder_layers', model.get('num_layers', 32))
        vocab = model.get('vocab_size', 32000)
        intermediate = model.get('intermediate_size', hidden * 4)
        num_heads = model.get('num_attention_heads', hidden // 128)
        num_kv_heads = model.get('num_key_value_heads', num_heads)
    else:
        # Try GenZ model config
        if HAS_GENZ:
            try:
                config = get_configs(model)
                hidden = getattr(config, 'hidden_size', 4096) or 4096
                layers = getattr(config, 'num_decoder_layers', 32) or 32
                vocab = getattr(config, 'vocab_size', 32000) or 32000
                intermediate = getattr(config, 'intermediate_size', None) or (hidden * 4)
                num_heads = getattr(config, 'num_attention_heads', hidden // 128) or (hidden // 128)
                num_kv_heads = getattr(config, 'num_key_value_heads', num_heads) or num_heads
            except Exception:
                # Fallback to estimation from name
                hidden, layers, vocab, intermediate = _estimate_from_model_name(model)
                num_heads = hidden // 128
                num_kv_heads = num_heads
        else:
            hidden, layers, vocab, intermediate = _estimate_from_model_name(model)
            num_heads = hidden // 128
            num_kv_heads = num_heads

    # Calculate parameters
    params_per_layer = (
        4 * hidden * hidden +  # Attention (Q, K, V, O)
        3 * hidden * intermediate  # FFN (gate, up, down for SwiGLU)
    )
    # Adjust for GQA
    if num_kv_heads < num_heads:
        kv_reduction = num_kv_heads / num_heads
        params_per_layer = (
            hidden * hidden * (1 + 2 * kv_reduction + 1) +  # Q, K, V, O
            3 * hidden * intermediate
        )

    total_params = layers * params_per_layer + 2 * vocab * hidden  # embedding + lm_head

    return {
        'hidden_size': hidden,
        'num_layers': layers,
        'vocab_size': vocab,
        'intermediate_size': intermediate,
        'num_attention_heads': num_heads,
        'num_key_value_heads': num_kv_heads,
        'total_params': total_params,
        'params_billions': total_params / 1e9,
    }


def _estimate_from_model_name(model_name: str) -> tuple:
    """Estimate model architecture from name."""
    model_lower = model_name.lower()

    # Common model sizes
    if '405b' in model_lower or '400b' in model_lower:
        return 16384, 126, 128256, 53248
    elif '70b' in model_lower or '72b' in model_lower:
        return 8192, 80, 128256, 28672
    elif '32b' in model_lower or '34b' in model_lower:
        return 8192, 60, 128256, 28672
    elif '13b' in model_lower or '14b' in model_lower:
        return 5120, 40, 32000, 13824
    elif '8b' in model_lower or '7b' in model_lower:
        return 4096, 32, 128256, 14336
    elif '3b' in model_lower:
        return 3072, 28, 128256, 8192
    elif '1b' in model_lower:
        return 2048, 24, 128256, 5632
    else:
        # Default to 7B-scale
        return 4096, 32, 32000, 11008


def _calculate_memory_breakdown(
    model_params: Dict[str, Any],
    training_type: str,
    method: str,
    optimizer: str,
    precision: str,
    batch_size: int,
    seq_length: int,
    num_gpus: int,
    zero_stage: int,
    tensor_parallel: int = 1,
) -> MemoryBreakdownDetails:
    """Calculate detailed memory breakdown."""
    total_params = model_params['total_params']
    hidden = model_params['hidden_size']
    layers = model_params['num_layers']

    # Precision bytes
    precision_bytes = get_precision_bytes(precision)

    # Calculate trainable params based on method
    if method in ('lora', 'dora', 'pissa'):
        trainable_ratio = 0.01  # ~1% for LoRA
    elif method == 'qlora':
        trainable_ratio = 0.01
        precision_bytes = 0.5  # 4-bit base weights
    elif method == 'freeze':
        trainable_ratio = 0.1  # ~10% for freeze
    else:
        trainable_ratio = 1.0  # Full fine-tuning

    trainable_params = total_params * trainable_ratio

    # Weight memory
    weight_memory_gb = (total_params * precision_bytes) / 1e9
    weight_per_gpu = weight_memory_gb / tensor_parallel
    if zero_stage >= 3:
        weight_per_gpu /= num_gpus

    # Gradient memory (FP32 for trainable params)
    gradient_memory_gb = (trainable_params * 4) / 1e9
    gradient_per_gpu = gradient_memory_gb
    if zero_stage >= 2:
        gradient_per_gpu /= num_gpus

    # Optimizer memory
    opt_bytes = get_optimizer_memory_multiplier(optimizer)
    optimizer_memory_gb = (trainable_params * opt_bytes) / 1e9
    optimizer_per_gpu = optimizer_memory_gb
    if zero_stage >= 1:
        optimizer_per_gpu /= num_gpus

    # Activation memory (rough estimate)
    # Activations scale with batch * seq * hidden * layers
    activation_memory_gb = (batch_size * seq_length * hidden * layers * precision_bytes * 4) / 1e9
    activation_per_gpu = activation_memory_gb / tensor_parallel

    # Apply gradient checkpointing (reduces activations ~70%)
    activation_per_gpu *= 0.3

    # Training type memory multiplier (for reference models, etc.)
    type_multiplier = get_training_type_memory_multiplier(training_type)

    # Reference model memory (DPO, KTO, PPO)
    ref_model_gb = 0.0
    if training_type in ('dpo', 'kto', 'ppo', 'ipo'):
        # Reference model is frozen, often quantized
        ref_model_gb = weight_memory_gb * 0.7  # Assuming some quantization

    # Reward model memory (PPO)
    reward_model_gb = 0.0
    if training_type == 'ppo':
        reward_model_gb = weight_memory_gb * 0.5  # Often smaller or LoRA

    # Critic model memory (PPO)
    critic_model_gb = 0.0
    if training_type == 'ppo':
        critic_model_gb = weight_memory_gb * 0.1  # Value head overhead

    # Total per GPU
    total_per_gpu = (
        weight_per_gpu +
        gradient_per_gpu +
        optimizer_per_gpu +
        activation_per_gpu +
        (ref_model_gb + reward_model_gb + critic_model_gb) / num_gpus
    )

    # System overhead (CUDA, PyTorch, etc.)
    system_overhead_gb = 2.0  # ~2GB overhead

    return MemoryBreakdownDetails(
        weight_memory_gb=weight_memory_gb,
        weight_memory_per_gpu_gb=weight_per_gpu,
        gradient_memory_gb=gradient_memory_gb,
        gradient_memory_per_gpu_gb=gradient_per_gpu,
        optimizer_memory_gb=optimizer_memory_gb,
        optimizer_memory_per_gpu_gb=optimizer_per_gpu,
        activation_memory_gb=activation_memory_gb,
        activation_memory_per_gpu_gb=activation_per_gpu,
        reference_model_memory_gb=ref_model_gb,
        reward_model_memory_gb=reward_model_gb,
        critic_model_memory_gb=critic_model_gb,
        total_per_gpu_gb=total_per_gpu + system_overhead_gb,
        system_overhead_gb=system_overhead_gb,
    )


def predict_cluster_requirements(
    training_type: str,
    model: Union[str, Dict[str, Any]],
    dtype: str = "bf16",
    optimizer: str = "adamw",
    dataset_size_tokens: int = 1_000_000_000,
    batch_size: int = 4,
    seq_length: int = 4096,
    method: str = "auto",
    network_bandwidth_gbps: Optional[float] = None,
    inter_chip_bandwidth_gbps: Optional[float] = None,
    topology: str = "fat-tree",
    max_training_hours: Optional[float] = None,
    target_mfu: float = 0.4,
) -> MinimumClusterRequirements:
    """
    Predict minimum cluster requirements for a training configuration.

    Args:
        training_type: Training stage (sft, dpo, ppo, kto, rm, grpo, ipo, pt)
        model: Model name (HF ID) or model config dict
        dtype: Training precision (bf16, fp16, fp32, fp8, int8, nf4)
        optimizer: Optimizer type (adamw, adam_8bit, paged_adamw_8bit, etc.)
        dataset_size_tokens: Total tokens in dataset
        batch_size: Per-GPU micro batch size
        seq_length: Sequence length
        method: Fine-tuning method (auto, full, lora, qlora, dora, pissa, freeze)
        network_bandwidth_gbps: Inter-node network bandwidth
        inter_chip_bandwidth_gbps: Intra-node interconnect bandwidth
        topology: Network topology (fat-tree, dragonfly, torus, ring)
        max_training_hours: Maximum acceptable training time
        target_mfu: Target Model FLOPS Utilization (0.3-0.5 typical)

    Returns:
        MinimumClusterRequirements with detailed requirements
    """
    # Estimate model parameters
    model_params = _estimate_model_params(model)
    params_b = model_params['params_billions']

    # Auto-select method based on model size
    if method == "auto":
        if params_b > 70:
            method = "qlora"
        elif params_b > 13:
            method = "lora"
        else:
            method = "lora"

    # Get training type config
    try:
        stage_config = get_training_stage_config(training_type)
    except ValueError:
        stage_config = None

    # Calculate base memory requirements
    memory_breakdown = _calculate_memory_breakdown(
        model_params=model_params,
        training_type=training_type,
        method=method,
        optimizer=optimizer,
        precision=dtype,
        batch_size=batch_size,
        seq_length=seq_length,
        num_gpus=1,  # Start with single GPU
        zero_stage=0,
        tensor_parallel=1,
    )

    min_memory_per_gpu = memory_breakdown.total_per_gpu_gb

    # Analyze requirements for each GPU type
    requirements_by_gpu: Dict[str, GPURequirement] = {}
    feasibility_notes: List[str] = []
    warnings: List[str] = []

    gpu_types_to_check = list(GPU_SPECS.keys())

    for gpu_type in gpu_types_to_check:
        gpu_memory = _get_gpu_memory(gpu_type)

        # Calculate minimum GPUs needed
        if min_memory_per_gpu <= gpu_memory * 0.85:
            # Model fits on single GPU
            min_gpus = 1
            zero_stage = 0
        else:
            # Need multiple GPUs
            # Try ZeRO-2 first (optimizer + gradient sharding)
            for zero in [2, 3]:
                test_breakdown = _calculate_memory_breakdown(
                    model_params=model_params,
                    training_type=training_type,
                    method=method,
                    optimizer=optimizer,
                    precision=dtype,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    num_gpus=8,  # Test with 8 GPUs
                    zero_stage=zero,
                    tensor_parallel=1,
                )
                if test_breakdown.total_per_gpu_gb <= gpu_memory * 0.85:
                    zero_stage = zero
                    break
            else:
                zero_stage = 3

            # Calculate minimum GPUs with ZeRO
            if zero_stage == 3:
                # ZeRO-3 shards everything
                min_gpus = max(1, math.ceil(min_memory_per_gpu / (gpu_memory * 0.7)))
            else:
                # ZeRO-2 shards gradients and optimizer
                grad_opt_memory = memory_breakdown.gradient_memory_per_gpu_gb + memory_breakdown.optimizer_memory_per_gpu_gb
                min_gpus = max(1, math.ceil(grad_opt_memory / (gpu_memory * 0.3)))

        # Round to powers of 2
        min_gpus = 2 ** math.ceil(math.log2(max(1, min_gpus)))

        # Calculate throughput and cost estimates
        estimated_tps = 0.0
        estimated_hours = float('inf')
        estimated_cost = 0.0
        feasible = True
        notes: List[str] = []

        if HAS_GENZ and isinstance(model, str):
            try:
                system_name = _get_system_name(gpu_type)
                sim_result = training_modeling(
                    model=model,
                    training_stage=training_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    system_name=system_name,
                    num_gpus=min_gpus,
                    tensor_parallel=1,
                    data_parallel=min_gpus,
                    pipeline_parallel=1,
                    method=method,
                    optimizer=optimizer,
                    zero_stage=zero_stage,
                    gradient_checkpointing=True,
                    bits=dtype,
                )
                estimated_tps = sim_result.tokens_per_second
                if estimated_tps > 0:
                    estimated_hours = dataset_size_tokens / (estimated_tps * 3600)

                # Get cost estimate
                try:
                    tco = calculate_tco(
                        gpu_type=gpu_type,
                        num_gpus=min_gpus,
                        training_hours=estimated_hours,
                        dataset_tokens=dataset_size_tokens,
                    )
                    estimated_cost = tco.total_cost
                except Exception:
                    pass

                # Check if simulation shows memory fit
                if sim_result.memory_per_gpu_gb > gpu_memory * 0.95:
                    feasible = False
                    notes.append(f"Memory exceeds GPU capacity: {sim_result.memory_per_gpu_gb:.1f}GB > {gpu_memory}GB")

            except Exception as e:
                notes.append(f"Simulation failed: {str(e)[:50]}")

        # Check training time constraint
        if max_training_hours and estimated_hours > max_training_hours and estimated_hours != float('inf'):
            # Need more GPUs to meet time constraint
            scale_factor = estimated_hours / max_training_hours
            if scale_factor < 1000:  # Reasonable scaling limit
                min_gpus = int(min_gpus * scale_factor)
                min_gpus = 2 ** math.ceil(math.log2(max(1, min_gpus)))
                notes.append(f"Scaled to {min_gpus} GPUs to meet {max_training_hours}h constraint")

        # Build parallelism strategy
        parallelism = ParallelismStrategy(
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=min_gpus,
            zero_stage=zero_stage,
            gradient_accumulation_steps=max(1, 8 // batch_size),
            gradient_checkpointing=True,
        )

        requirements_by_gpu[gpu_type] = GPURequirement(
            gpu_type=gpu_type,
            min_gpus=min_gpus,
            max_batch_size=batch_size,
            memory_per_gpu_gb=min_memory_per_gpu if min_gpus == 1 else memory_breakdown.total_per_gpu_gb / 2,
            estimated_throughput_tps=estimated_tps,
            estimated_training_hours=estimated_hours,
            estimated_cost_usd=estimated_cost,
            recommended_parallelism=parallelism,
            feasible=feasible,
            notes=notes,
        )

    # Find overall minimum requirements
    feasible_gpus = [r for r in requirements_by_gpu.values() if r.feasible]
    if feasible_gpus:
        best_gpu = min(feasible_gpus, key=lambda x: x.min_gpus)
        min_gpus = best_gpu.min_gpus
        min_gpu_memory_gb = _get_gpu_memory(best_gpu.gpu_type)
        recommended_parallelism = best_gpu.recommended_parallelism
        recommended_zero_stage = best_gpu.recommended_parallelism.zero_stage
    else:
        # No feasible configuration found
        min_gpus = 8
        min_gpu_memory_gb = 80.0
        recommended_parallelism = ParallelismStrategy(
            tensor_parallel=2,
            data_parallel=4,
            zero_stage=3,
            gradient_checkpointing=True,
        )
        recommended_zero_stage = 3
        warnings.append("No single GPU type can fit this configuration. Consider using larger GPUs or more memory-efficient methods.")

    # Network requirements
    # Gradient AllReduce bandwidth requirement
    grad_size_gb = memory_breakdown.gradient_memory_gb
    target_step_time_s = 1.0  # Target 1 second per step
    min_inter_node_bw = (grad_size_gb * 2) / target_step_time_s  # AllReduce = 2x data

    # Intra-node requirements (NVLink)
    min_intra_node_bw = min_inter_node_bw * 2  # Intra-node should be faster

    # Topology recommendation
    if min_gpus <= 8:
        recommended_topology = "single_node"
    elif min_gpus <= 64:
        recommended_topology = "fat-tree"
    else:
        recommended_topology = "fat-tree"  # Still best for most cases

    # Add feasibility notes
    if params_b > 100:
        feasibility_notes.append(f"Very large model ({params_b:.0f}B params) - requires significant resources")
    if training_type == "ppo":
        feasibility_notes.append("PPO requires 3+ models in memory (actor, reference, reward)")
    if method == "full" and params_b > 13:
        feasibility_notes.append(f"Full fine-tuning of {params_b:.0f}B model requires substantial memory")
        warnings.append("Consider using LoRA/QLoRA for memory efficiency")

    return MinimumClusterRequirements(
        min_gpus=min_gpus,
        min_gpu_memory_gb=min_gpu_memory_gb,
        min_total_memory_gb=min_gpus * min_gpu_memory_gb,
        requirements_by_gpu=requirements_by_gpu,
        min_inter_node_bandwidth_gbps=min_inter_node_bw,
        min_intra_node_bandwidth_gbps=min_intra_node_bw,
        recommended_topology=recommended_topology,
        recommended_parallelism=recommended_parallelism,
        recommended_zero_stage=recommended_zero_stage,
        recommended_method=method,
        memory_breakdown=memory_breakdown,
        feasibility_notes=feasibility_notes,
        warnings=warnings,
    )
