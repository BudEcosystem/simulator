"""
Training Parallelization Strategy Predictor for GenZ Framework.

This module provides functions to find optimal parallelization strategies
for distributed training, extending the inference parallelization patterns
to training workloads with ZeRO stages, gradient accumulation, and memory constraints.

Entry points:
- get_training_parallelization_options(): Generate valid parallelism configurations
- get_best_training_parallelization(): Find optimal strategy for throughput/latency/memory
- get_pareto_optimal_training_performance(): Pareto frontier for multi-objective optimization
"""

import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..parallelism import ParallelismConfig
from ..Models import get_configs
from .training_modeling import training_modeling, TrainingModelingOutput

# Make paretoset optional
try:
    from paretoset import paretoset
    HAS_PARETOSET = True
except ImportError:
    HAS_PARETOSET = False
    warnings.warn(
        "paretoset not available. get_pareto_optimal_training_performance will not be available.",
        ImportWarning
    )


def factors(n: int) -> List[int]:
    """Find all factors of n."""
    return sorted(set(
        x for tup in (
            [i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0
        ) for x in tup
    ))


@dataclass
class TrainingParallelismConfig:
    """Configuration for training parallelism with ZeRO support."""
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    zero_stage: int = 0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs used."""
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
        }

    def to_parallelism_config(self) -> ParallelismConfig:
        """Convert to ParallelismConfig."""
        return ParallelismConfig(
            tensor_parallel=self.tensor_parallel,
            pipeline_parallel=self.pipeline_parallel,
            data_parallel=self.data_parallel,
            expert_parallel=self.expert_parallel,
            zero_stage=self.zero_stage,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
        )


def _estimate_memory_per_gpu(
    model_config: Dict[str, Any],
    config: TrainingParallelismConfig,
    batch_size: int,
    seq_length: int,
    method: str,
    optimizer: str,
    bits: str,
) -> float:
    """
    Estimate memory per GPU for a given configuration.

    Returns memory in GB.
    """
    # Extract model params - ModelConfig uses attributes, not dict access
    num_layers = getattr(model_config, 'num_decoder_layers', 32) or 32
    hidden_size = getattr(model_config, 'hidden_size', 4096) or 4096
    intermediate_size = getattr(model_config, 'intermediate_size', None) or (hidden_size * 4)
    vocab_size = getattr(model_config, 'vocab_size', 32000) or 32000
    num_heads = getattr(model_config, 'num_attention_heads', 32) or 32

    # Calculate total parameters
    embedding_params = vocab_size * hidden_size * 2
    attention_params = 4 * hidden_size * hidden_size
    ffn_params = 3 * hidden_size * intermediate_size
    norm_params = 4 * hidden_size
    params_per_layer = attention_params + ffn_params + norm_params
    total_params = embedding_params + num_layers * params_per_layer

    # Calculate trainable params based on method
    if method == 'full':
        trainable_params = total_params
    elif method in ('lora', 'qlora', 'dora'):
        lora_rank = 16  # Default
        attn_lora_params = 4 * 2 * lora_rank * hidden_size * num_layers
        ffn_lora_params = 2 * 2 * lora_rank * (hidden_size + intermediate_size) // 2 * num_layers
        trainable_params = attn_lora_params + ffn_lora_params
    else:
        trainable_params = total_params

    # Precision bytes - extended to match system.py mem_multiplier
    precision_bytes_map = {
        'fp32': 4, 'f32': 4, 'tf32': 4,
        'bf16': 2, 'fp16': 2,
        'mixed_bf16': 2, 'mixed_fp16': 2, 'amp_bf16': 2, 'amp_fp16': 2,
        'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1, 'mixed_fp8': 1, 'mixed_fp8_bf16': 1,
        'int8': 1, 'fp6': 0.75,
        'int4': 0.5, 'fp4': 0.5, 'nf4': 0.5,
        'int2': 0.25,
    }
    precision_bytes = precision_bytes_map.get(bits.lower(), 2)

    # Weight memory
    if method == 'qlora':
        weight_bytes = total_params * 0.5
    else:
        weight_bytes = total_params * precision_bytes

    weight_bytes /= config.tensor_parallel
    if config.zero_stage >= 3:
        weight_bytes /= config.data_parallel

    # Gradient memory
    gradient_bytes = trainable_params * 4  # FP32 gradients
    if config.zero_stage >= 2:
        gradient_bytes /= config.data_parallel

    # Optimizer state memory - complete mapping from OPTIMIZER_PROFILES
    optimizer_bytes_map = {
        # Standard
        'sgd': 0, 'sgd_momentum': 4, 'adam': 8, 'adamw': 8,
        # Memory-efficient
        'adam_8bit': 2, 'adamw_8bit': 2, 'paged_adamw_8bit': 2, 'adafactor': 4,
        # Newer
        'lion': 4, 'lamb': 8, 'lars': 4, 'sophia': 8, 'schedule_free_adamw': 8,
        # Low-rank
        'galore': 2, 'flora': 2,
        # Additional
        'came': 6, 'adan': 12, 'prodigy': 8, 'shampoo': 16, 'muon': 4,
    }
    optimizer_bytes_per_param = optimizer_bytes_map.get(optimizer.lower(), 8)

    optimizer_state_bytes = trainable_params * optimizer_bytes_per_param
    if config.zero_stage >= 1:
        optimizer_state_bytes /= config.data_parallel

    # Activation memory
    effective_layers = int(np.ceil(np.sqrt(num_layers))) if config.gradient_checkpointing else num_layers
    attn_elements = batch_size * seq_length * hidden_size * 4
    attn_scores = batch_size * num_heads * seq_length * seq_length
    ffn_elements = batch_size * seq_length * intermediate_size * 2
    elements_per_layer = attn_elements + attn_scores + ffn_elements
    activation_bytes = elements_per_layer * effective_layers * precision_bytes
    activation_bytes /= config.tensor_parallel

    # Total with 10% overhead
    total_bytes = (weight_bytes + gradient_bytes + optimizer_state_bytes + activation_bytes) * 1.10

    return total_bytes / 1e9


def _check_memory_feasible(
    model_config: Dict[str, Any],
    config: TrainingParallelismConfig,
    gpu_memory_gb: float,
    batch_size: int,
    seq_length: int,
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    memory_margin: float = 0.9,  # Use only 90% of available memory
) -> bool:
    """Check if configuration fits in GPU memory."""
    estimated_memory = _estimate_memory_per_gpu(
        model_config, config, batch_size, seq_length, method, optimizer, bits
    )
    return estimated_memory <= gpu_memory_gb * memory_margin


def get_training_parallelization_options(
    model: Union[str, Dict[str, Any]],
    total_gpus: int,
    gpu_memory_gb: float,
    batch_size: int = 4,
    seq_length: int = 4096,
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    max_tp: int = 8,
    max_pp: int = 16,
    zero_stages: Optional[List[int]] = None,
    include_gradient_checkpointing: bool = True,
) -> List[TrainingParallelismConfig]:
    """
    Generate valid training parallelism configurations.

    Args:
        model: Model name (from MODEL_DICT) or HuggingFace config dict
        total_gpus: Total number of GPUs available
        gpu_memory_gb: Memory per GPU in GB
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        method: Training method (full, lora, qlora)
        optimizer: Optimizer type
        bits: Precision (bf16, fp16, fp32)
        max_tp: Maximum tensor parallel degree
        max_pp: Maximum pipeline parallel degree
        zero_stages: ZeRO stages to consider (default: [0, 1, 2, 3])
        include_gradient_checkpointing: Include configs with gradient checkpointing

    Returns:
        List of valid TrainingParallelismConfig objects
    """
    model_config = get_configs(model)

    if total_gpus == 1:
        # Single GPU - ZeRO doesn't help, just try with/without gradient checkpointing
        # Note: ZeRO is for distributed training, so force stage 0 for single GPU
        configs = []
        for zero_stage in [0]:  # ZeRO doesn't apply to single GPU
            for gc in ([True, False] if include_gradient_checkpointing else [True]):
                config = TrainingParallelismConfig(
                    tensor_parallel=1,
                    pipeline_parallel=1,
                    data_parallel=1,
                    zero_stage=zero_stage,
                    gradient_checkpointing=gc,
                )
                if _check_memory_feasible(
                    model_config, config, gpu_memory_gb, batch_size, seq_length, method, optimizer, bits
                ):
                    configs.append(config)
        return configs

    # Get valid TP options (must divide attention heads)
    # ModelConfig uses attribute access, not dict .get()
    num_heads = getattr(model_config, 'num_attention_heads', 32) or 32
    num_layers = getattr(model_config, 'num_decoder_layers', 32) or 32

    tp_options = [x for x in factors(num_heads) if x <= min(max_tp, total_gpus)]
    pp_options = [x for x in factors(num_layers) if x <= min(max_pp, total_gpus)]

    if zero_stages is None:
        zero_stages = [0, 1, 2, 3]

    configs = []

    for tp, pp in itertools.product(tp_options, pp_options):
        if tp * pp > total_gpus:
            continue

        dp = total_gpus // (tp * pp)
        if dp < 1:
            continue

        # Generate configs with different ZeRO stages
        for zero_stage in zero_stages:
            gc_options = [True, False] if include_gradient_checkpointing else [True]
            for gc in gc_options:
                config = TrainingParallelismConfig(
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    data_parallel=dp,
                    zero_stage=zero_stage,
                    gradient_checkpointing=gc,
                )

                if _check_memory_feasible(
                    model_config, config, gpu_memory_gb, batch_size, seq_length, method, optimizer, bits
                ):
                    configs.append(config)

    return configs


def get_best_training_parallelization(
    model: Union[str, Dict[str, Any]] = 'llama-3-8b',
    total_gpus: int = 8,
    batch_size: int = 4,
    seq_length: int = 4096,
    system_name: str = 'H100_GPU',
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    optimize_for: str = 'throughput',
    gpu_memory_gb: Optional[float] = None,
    max_configs: int = 50,
    debug: bool = False,
) -> Tuple[TrainingParallelismConfig, TrainingModelingOutput]:
    """
    Find optimal parallelization strategy for training.

    Args:
        model: Model name
        total_gpus: Total number of GPUs
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        system_name: Hardware system name
        training_stage: Training type (sft, dpo, ppo, etc.)
        method: Training method (full, lora, qlora)
        optimizer: Optimizer type
        bits: Precision
        optimize_for: Optimization target ('throughput', 'latency', 'memory')
        gpu_memory_gb: Memory per GPU (auto-detected if None)
        max_configs: Maximum configurations to evaluate
        debug: Enable debug output

    Returns:
        Tuple of (best_config, modeling_result)
    """
    # Auto-detect GPU memory if not provided
    if gpu_memory_gb is None:
        gpu_memory_map = {
            'A100_80GB_GPU': 80,
            'A100_40GB_GPU': 40,
            'H100_GPU': 80,
            'H100_80GB_GPU': 80,
            'H200_GPU': 141,
            'B100_GPU': 192,
            'GB200_GPU': 192,
            'MI300X_GPU': 192,
            'MI300A_GPU': 128,
            'TPU_v5e': 16,
            'TPU_v5p': 96,
        }
        gpu_memory_gb = gpu_memory_map.get(system_name, 80)

    # Get valid configurations
    configs = get_training_parallelization_options(
        model=model,
        total_gpus=total_gpus,
        gpu_memory_gb=gpu_memory_gb,
        batch_size=batch_size,
        seq_length=seq_length,
        method=method,
        optimizer=optimizer,
        bits=bits,
    )

    if not configs:
        raise ValueError(
            f"No valid parallelization configs found for model={model}, "
            f"gpus={total_gpus}, memory={gpu_memory_gb}GB. "
            "Try reducing batch_size or using LoRA/QLoRA."
        )

    # Limit configs to evaluate
    if len(configs) > max_configs:
        # Prioritize configs with higher TP (usually faster)
        configs.sort(key=lambda c: (-c.tensor_parallel, c.pipeline_parallel, c.zero_stage))
        configs = configs[:max_configs]

    if debug:
        print(f"Evaluating {len(configs)} parallelization configurations...")

    # Evaluate each configuration
    results = []
    for config in configs:
        try:
            result = training_modeling(
                model=model,
                training_stage=training_stage,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=total_gpus,
                tensor_parallel=config.tensor_parallel,
                data_parallel=config.data_parallel,
                pipeline_parallel=config.pipeline_parallel,
                method=method,
                optimizer=optimizer,
                zero_stage=config.zero_stage,
                gradient_checkpointing=config.gradient_checkpointing,
                bits=bits,
            )
            results.append((config, result))

            if debug:
                print(
                    f"  TP={config.tensor_parallel}, PP={config.pipeline_parallel}, "
                    f"DP={config.data_parallel}, ZeRO={config.zero_stage}: "
                    f"{result.tokens_per_second:.0f} tok/s, {result.memory_per_gpu_gb:.1f} GB"
                )
        except Exception as e:
            if debug:
                print(f"  Config failed: {config.to_dict()} - {e}")
            continue

    if not results:
        raise RuntimeError("All configurations failed evaluation")

    # Sort by optimization target
    if optimize_for == 'throughput':
        results.sort(key=lambda x: -x[1].tokens_per_second)
    elif optimize_for == 'latency':
        results.sort(key=lambda x: x[1].step_time_ms)
    elif optimize_for == 'memory':
        results.sort(key=lambda x: x[1].memory_per_gpu_gb)
    elif optimize_for == 'mfu':
        results.sort(key=lambda x: -x[1].model_flops_utilization)
    else:
        raise ValueError(f"Unknown optimize_for: {optimize_for}")

    best_config, best_result = results[0]

    if debug:
        print(f"\nBest config: TP={best_config.tensor_parallel}, PP={best_config.pipeline_parallel}, "
              f"DP={best_config.data_parallel}, ZeRO={best_config.zero_stage}")
        print(f"  Throughput: {best_result.tokens_per_second:.0f} tok/s")
        print(f"  MFU: {best_result.model_flops_utilization:.1%}")
        print(f"  Memory: {best_result.memory_per_gpu_gb:.1f} GB/GPU")

    return best_config, best_result


def get_pareto_optimal_training_performance(
    model: str,
    total_gpus: int,
    batch_sizes: List[int],
    seq_lengths: List[int],
    system_name: str,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    gpu_memory_gb: Optional[float] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Find Pareto-optimal configurations for throughput vs memory.

    Args:
        model: Model name
        total_gpus: Total number of GPUs
        batch_sizes: List of batch sizes to evaluate
        seq_lengths: List of sequence lengths to evaluate
        system_name: Hardware system name
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        gpu_memory_gb: Memory per GPU
        debug: Enable debug output

    Returns:
        DataFrame with Pareto-optimal configurations
    """
    if not HAS_PARETOSET:
        raise ImportError(
            "paretoset package is required for get_pareto_optimal_training_performance. "
            "Install it with: pip install paretoset"
        )

    # Auto-detect GPU memory
    if gpu_memory_gb is None:
        gpu_memory_map = {
            'A100_80GB_GPU': 80, 'A100_40GB_GPU': 40, 'H100_GPU': 80,
            'H100_80GB_GPU': 80, 'H200_GPU': 141, 'B100_GPU': 192,
        }
        gpu_memory_gb = gpu_memory_map.get(system_name, 80)

    data = []

    for batch_size, seq_length in itertools.product(batch_sizes, seq_lengths):
        try:
            configs = get_training_parallelization_options(
                model=model,
                total_gpus=total_gpus,
                gpu_memory_gb=gpu_memory_gb,
                batch_size=batch_size,
                seq_length=seq_length,
                method=method,
                optimizer=optimizer,
                bits=bits,
            )

            for config in configs[:20]:  # Limit per batch/seq combination
                try:
                    result = training_modeling(
                        model=model,
                        training_stage=training_stage,
                        batch_size=batch_size,
                        seq_length=seq_length,
                        system_name=system_name,
                        num_gpus=total_gpus,
                        tensor_parallel=config.tensor_parallel,
                        data_parallel=config.data_parallel,
                        pipeline_parallel=config.pipeline_parallel,
                        method=method,
                        optimizer=optimizer,
                        zero_stage=config.zero_stage,
                        gradient_checkpointing=config.gradient_checkpointing,
                        bits=bits,
                    )

                    data.append({
                        'batch_size': batch_size,
                        'seq_length': seq_length,
                        'TP': config.tensor_parallel,
                        'PP': config.pipeline_parallel,
                        'DP': config.data_parallel,
                        'ZeRO': config.zero_stage,
                        'GC': config.gradient_checkpointing,
                        'tokens_per_second': result.tokens_per_second,
                        'memory_gb': result.memory_per_gpu_gb,
                        'mfu': result.model_flops_utilization,
                        'step_time_ms': result.step_time_ms,
                    })
                except Exception:
                    continue
        except Exception:
            continue

    if not data:
        raise RuntimeError("No valid configurations found")

    df = pd.DataFrame(data)

    # Find Pareto frontier: maximize throughput, minimize memory
    pareto_data = df[['tokens_per_second', 'memory_gb']].values
    mask = paretoset(pareto_data, sense=["max", "min"])

    return df[mask].sort_values('tokens_per_second', ascending=False)


def get_various_training_parallelization(
    model: str = 'llama-3-8b',
    total_gpus: int = 8,
) -> List[Tuple[int, int, int]]:
    """
    Get various valid (TP, PP, DP) combinations for a model.

    Args:
        model: Model name
        total_gpus: Total number of GPUs

    Returns:
        List of (TP, PP, DP) tuples
    """
    model_config = get_configs(model)

    if total_gpus == 1:
        return [(1, 1, 1)]

    num_heads = getattr(model_config, 'num_attention_heads', 32) or 32
    num_layers = getattr(model_config, 'num_decoder_layers', 32) or 32

    tp_options = [x for x in factors(num_heads) if x <= min(8, total_gpus)]
    pp_options = [x for x in factors(num_layers) if x <= total_gpus]

    combinations = []
    for tp, pp in itertools.product(tp_options, pp_options):
        if tp * pp <= total_gpus:
            dp = total_gpus // (tp * pp)
            if dp >= 1:
                combinations.append((tp, pp, dp))

    return sorted(set(combinations), key=lambda x: (-x[0], x[1], -x[2]))


def recommend_training_config(
    model: str,
    total_gpus: int,
    batch_size: int,
    seq_length: int,
    system_name: str,
    training_stage: str = 'sft',
    target_memory_fraction: float = 0.85,
) -> Dict[str, Any]:
    """
    Provide training configuration recommendations.

    Returns a dictionary with recommended settings and explanations.
    """
    recommendations = {
        'model': model,
        'total_gpus': total_gpus,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'system': system_name,
    }

    # Try to find best config
    try:
        config, result = get_best_training_parallelization(
            model=model,
            total_gpus=total_gpus,
            batch_size=batch_size,
            seq_length=seq_length,
            system_name=system_name,
            training_stage=training_stage,
            method='full',
            optimizer='adamw',
            optimize_for='throughput',
        )

        recommendations['recommended_config'] = config.to_dict()
        recommendations['expected_throughput'] = result.tokens_per_second
        recommendations['expected_memory_gb'] = result.memory_per_gpu_gb
        recommendations['expected_mfu'] = result.model_flops_utilization
        recommendations['method'] = 'full'
        recommendations['status'] = 'success'

    except ValueError:
        # Full fine-tuning doesn't fit, try LoRA
        try:
            config, result = get_best_training_parallelization(
                model=model,
                total_gpus=total_gpus,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                training_stage=training_stage,
                method='lora',
                optimizer='adamw',
                optimize_for='throughput',
            )

            recommendations['recommended_config'] = config.to_dict()
            recommendations['expected_throughput'] = result.tokens_per_second
            recommendations['expected_memory_gb'] = result.memory_per_gpu_gb
            recommendations['expected_mfu'] = result.model_flops_utilization
            recommendations['method'] = 'lora'
            recommendations['note'] = 'Full fine-tuning does not fit in memory. Using LoRA instead.'
            recommendations['status'] = 'partial'

        except ValueError:
            # Even LoRA doesn't fit, try QLoRA
            try:
                reduced_batch_size = max(1, batch_size // 2)  # Ensure at least batch_size=1
                config, result = get_best_training_parallelization(
                    model=model,
                    total_gpus=total_gpus,
                    batch_size=reduced_batch_size,
                    seq_length=seq_length,
                    system_name=system_name,
                    training_stage=training_stage,
                    method='qlora',
                    optimizer='paged_adamw_8bit',
                    optimize_for='throughput',
                )

                recommendations['recommended_config'] = config.to_dict()
                recommendations['expected_throughput'] = result.tokens_per_second
                recommendations['expected_memory_gb'] = result.memory_per_gpu_gb
                recommendations['expected_mfu'] = result.model_flops_utilization
                recommendations['method'] = 'qlora'
                recommendations['adjusted_batch_size'] = reduced_batch_size
                recommendations['note'] = (
                    'Full fine-tuning and LoRA do not fit. '
                    'Using QLoRA with reduced batch size and paged optimizer.'
                )
                recommendations['status'] = 'degraded'

            except Exception as e:
                recommendations['status'] = 'failed'
                recommendations['error'] = str(e)
                recommendations['suggestion'] = (
                    'Consider using more GPUs, a smaller model, '
                    'or reducing sequence length.'
                )

    return recommendations
