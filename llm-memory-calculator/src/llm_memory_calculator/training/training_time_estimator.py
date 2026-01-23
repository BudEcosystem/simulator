"""
Training Time Estimator - Estimate total training time from dataset size.

This module provides functions to estimate total training time, steps,
and cost based on dataset size and hardware configuration.

Main entry points:
- estimate_training_time(): Estimate training time from dataset tokens
- calculate_training_steps(): Calculate number of training steps
- estimate_scaling_curve(): Generate scaling efficiency curve

Example usage:
    >>> from llm_memory_calculator.training import estimate_training_time
    >>>
    >>> estimate = estimate_training_time(
    ...     model='Llama-3.1-8B',
    ...     dataset_tokens=10_000_000_000,  # 10B tokens
    ...     num_epochs=1.0,
    ...     batch_size=4,
    ...     seq_length=4096,
    ...     system_name='H100_GPU',
    ...     num_gpus=8,
    ...     hourly_rate_per_gpu=3.0,
    ... )
    >>>
    >>> print(f"Total hours: {estimate.total_hours:.1f}")
    >>> print(f"Total steps: {estimate.total_steps:,}")
    >>> print(f"Total cost: ${estimate.cost_estimate_usd:,.0f}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
from ..genz.LLM_training.training_parallelization import (
    get_best_training_parallelization,
    TrainingParallelismConfig,
)


@dataclass
class TrainingTimeEstimate:
    """
    Comprehensive training time and cost estimate.

    Contains all metrics needed to understand training duration and economics.
    """
    # Time estimates
    total_hours: float
    total_days: float
    total_steps: int

    # Tokens and batching
    tokens_per_step: int
    steps_per_epoch: int
    num_epochs: float
    total_tokens: int

    # Performance
    tokens_per_second: float
    samples_per_second: float
    step_time_ms: float

    # Hardware utilization
    mfu: float  # Model FLOPs Utilization
    scaling_efficiency: float  # vs single GPU

    # Cost
    cost_estimate_usd: float
    cost_per_million_tokens: float
    cost_per_step: float
    hourly_cost: float

    # Memory
    memory_per_gpu_gb: float

    # Configuration used
    parallelism: Optional[TrainingParallelismConfig] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'time': {
                'total_hours': self.total_hours,
                'total_days': self.total_days,
                'total_steps': self.total_steps,
            },
            'tokens': {
                'tokens_per_step': self.tokens_per_step,
                'steps_per_epoch': self.steps_per_epoch,
                'num_epochs': self.num_epochs,
                'total_tokens': self.total_tokens,
            },
            'performance': {
                'tokens_per_second': self.tokens_per_second,
                'samples_per_second': self.samples_per_second,
                'step_time_ms': self.step_time_ms,
                'mfu': self.mfu,
                'scaling_efficiency': self.scaling_efficiency,
            },
            'cost': {
                'total_usd': self.cost_estimate_usd,
                'per_million_tokens': self.cost_per_million_tokens,
                'per_step': self.cost_per_step,
                'hourly': self.hourly_cost,
            },
            'memory': {
                'per_gpu_gb': self.memory_per_gpu_gb,
            },
            'parallelism': self.parallelism.to_dict() if self.parallelism else None,
            'config': self.config,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "TRAINING TIME ESTIMATE",
            "=" * 60,
            "",
            f"Duration: {self.total_hours:.1f} hours ({self.total_days:.1f} days)",
            f"Total Steps: {self.total_steps:,}",
            f"Total Tokens: {self.total_tokens:,}",
            "",
            f"Throughput: {self.tokens_per_second:,.0f} tokens/sec",
            f"Step Time: {self.step_time_ms:.1f} ms",
            f"MFU: {self.mfu:.1%}",
            f"Scaling Efficiency: {self.scaling_efficiency:.1%}",
            "",
            f"Memory/GPU: {self.memory_per_gpu_gb:.1f} GB",
            "",
            f"Total Cost: ${self.cost_estimate_usd:,.0f}",
            f"Cost/M tokens: ${self.cost_per_million_tokens:.2f}",
            f"Hourly Rate: ${self.hourly_cost:.2f}/hr",
            "=" * 60,
        ]
        return "\n".join(lines)


def estimate_training_time(
    model: str,
    dataset_tokens: int,
    num_epochs: float = 1.0,
    batch_size: int = 4,
    seq_length: int = 4096,
    gradient_accumulation_steps: int = 1,
    system_name: str = 'H100_GPU',
    num_gpus: int = 8,
    parallelism: Optional[TrainingParallelismConfig] = None,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    hourly_rate_per_gpu: float = 3.0,
    auto_find_parallelism: bool = True,
) -> TrainingTimeEstimate:
    """
    Estimate total training time and cost from dataset size.

    Args:
        model: Model name (from MODEL_DICT)
        dataset_tokens: Total tokens in training dataset
        num_epochs: Number of training epochs
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        system_name: Hardware system name (H100_GPU, A100_80GB_GPU, etc.)
        num_gpus: Total number of GPUs
        parallelism: Pre-computed parallelism config (if None, auto-finds)
        training_stage: Training type (sft, dpo, ppo, etc.)
        method: Training method (full, lora, qlora)
        optimizer: Optimizer type (adamw, adamw_8bit, etc.)
        bits: Precision (bf16, fp16, fp32)
        hourly_rate_per_gpu: Cost per GPU per hour (USD)
        auto_find_parallelism: If True and parallelism is None, auto-find best

    Returns:
        TrainingTimeEstimate with all metrics

    Example:
        >>> estimate = estimate_training_time(
        ...     model='Llama-3.1-8B',
        ...     dataset_tokens=100_000_000_000,  # 100B tokens
        ...     num_gpus=64,
        ...     hourly_rate_per_gpu=3.0,
        ... )
        >>> print(f"Training will take {estimate.total_days:.1f} days")
        >>> print(f"Total cost: ${estimate.cost_estimate_usd:,.0f}")
    """
    # Find best parallelism if not provided
    if parallelism is None and auto_find_parallelism:
        parallelism, sim_result = get_best_training_parallelization(
            model=model,
            total_gpus=num_gpus,
            batch_size=batch_size,
            seq_length=seq_length,
            system_name=system_name,
            training_stage=training_stage,
            method=method,
            optimizer=optimizer,
            bits=bits,
            optimize_for='throughput',
        )
    else:
        # Use default parallelism
        if parallelism is None:
            parallelism = TrainingParallelismConfig(
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=num_gpus,
                zero_stage=2 if num_gpus > 1 else 0,
                gradient_checkpointing=True,
            )

        # Run simulation
        sim_result = training_modeling(
            model=model,
            training_stage=training_stage,
            batch_size=batch_size,
            seq_length=seq_length,
            system_name=system_name,
            num_gpus=num_gpus,
            tensor_parallel=parallelism.tensor_parallel,
            data_parallel=parallelism.data_parallel,
            pipeline_parallel=parallelism.pipeline_parallel,
            method=method,
            optimizer=optimizer,
            zero_stage=parallelism.zero_stage,
            gradient_checkpointing=parallelism.gradient_checkpointing,
            bits=bits,
        )

    # Calculate tokens per step
    # tokens_per_step = batch_size * seq_length * data_parallel * gradient_accumulation
    data_parallel = parallelism.data_parallel if parallelism else num_gpus
    tokens_per_step = batch_size * seq_length * data_parallel * gradient_accumulation_steps

    # Calculate total tokens to process
    total_tokens = int(dataset_tokens * num_epochs)

    # Calculate steps
    total_steps = total_tokens // tokens_per_step
    steps_per_epoch = dataset_tokens // tokens_per_step

    # Calculate time
    step_time_sec = sim_result.step_time_ms / 1000
    total_seconds = total_steps * step_time_sec
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    # Calculate cost
    hourly_cost = hourly_rate_per_gpu * num_gpus
    cost_estimate_usd = total_hours * hourly_cost
    cost_per_step = cost_estimate_usd / total_steps if total_steps > 0 else 0
    cost_per_million_tokens = (cost_estimate_usd / total_tokens) * 1_000_000 if total_tokens > 0 else 0

    # Estimate scaling efficiency (vs theoretical single-GPU scaling)
    # In ideal case, throughput scales linearly with GPU count
    # Actual efficiency is reduced by communication overhead
    comm_overhead = sim_result.communication_overhead if hasattr(sim_result, 'communication_overhead') else 0
    scaling_efficiency = 1.0 - comm_overhead if comm_overhead else 0.85  # Default 85% efficiency

    return TrainingTimeEstimate(
        total_hours=total_hours,
        total_days=total_days,
        total_steps=total_steps,
        tokens_per_step=tokens_per_step,
        steps_per_epoch=steps_per_epoch,
        num_epochs=num_epochs,
        total_tokens=total_tokens,
        tokens_per_second=sim_result.tokens_per_second,
        samples_per_second=sim_result.samples_per_second,
        step_time_ms=sim_result.step_time_ms,
        mfu=sim_result.model_flops_utilization,
        scaling_efficiency=scaling_efficiency,
        cost_estimate_usd=cost_estimate_usd,
        cost_per_million_tokens=cost_per_million_tokens,
        cost_per_step=cost_per_step,
        hourly_cost=hourly_cost,
        memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
        parallelism=parallelism,
        config={
            'model': model,
            'dataset_tokens': dataset_tokens,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'system_name': system_name,
            'num_gpus': num_gpus,
            'training_stage': training_stage,
            'method': method,
            'optimizer': optimizer,
            'bits': bits,
        },
    )


def calculate_training_steps(
    dataset_tokens: int,
    batch_size: int,
    seq_length: int,
    data_parallel: int = 1,
    gradient_accumulation_steps: int = 1,
    num_epochs: float = 1.0,
) -> Dict[str, int]:
    """
    Calculate number of training steps from dataset configuration.

    Args:
        dataset_tokens: Total tokens in dataset
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        data_parallel: Number of data parallel replicas (DP degree)
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs

    Returns:
        Dict with:
        - total_steps: Total training steps
        - steps_per_epoch: Steps per epoch
        - tokens_per_step: Tokens processed per step
        - samples_per_step: Samples processed per step
    """
    tokens_per_step = batch_size * seq_length * data_parallel * gradient_accumulation_steps
    samples_per_step = batch_size * data_parallel * gradient_accumulation_steps

    total_tokens = int(dataset_tokens * num_epochs)
    total_steps = total_tokens // tokens_per_step
    steps_per_epoch = dataset_tokens // tokens_per_step

    return {
        'total_steps': total_steps,
        'steps_per_epoch': steps_per_epoch,
        'tokens_per_step': tokens_per_step,
        'samples_per_step': samples_per_step,
        'total_tokens': total_tokens,
    }


def estimate_time_from_throughput(
    dataset_tokens: int,
    tokens_per_second: float,
    num_epochs: float = 1.0,
    hourly_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Simple time estimation from throughput.

    Args:
        dataset_tokens: Total tokens in dataset
        tokens_per_second: Training throughput
        num_epochs: Number of epochs
        hourly_rate: Total cluster hourly rate (USD)

    Returns:
        Dict with hours, days, and cost
    """
    total_tokens = dataset_tokens * num_epochs
    total_seconds = total_tokens / tokens_per_second if tokens_per_second > 0 else float('inf')
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    return {
        'total_hours': total_hours,
        'total_days': total_days,
        'total_cost_usd': total_hours * hourly_rate,
        'total_tokens': total_tokens,
    }


@dataclass
class ScalingPoint:
    """Single point on the scaling curve."""
    num_gpus: int
    throughput_tokens_per_sec: float
    scaling_efficiency: float
    cost_per_million_tokens: float
    total_hours: float
    total_cost_usd: float
    parallelism: TrainingParallelismConfig
    mfu: float


def estimate_scaling_curve(
    model: str,
    dataset_tokens: int,
    system_name: str = 'H100_GPU',
    gpu_counts: Optional[List[int]] = None,
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    hourly_rate_per_gpu: float = 3.0,
    num_epochs: float = 1.0,
) -> List[ScalingPoint]:
    """
    Generate scaling efficiency curve for different GPU counts.

    Args:
        model: Model name
        dataset_tokens: Total tokens in dataset
        system_name: Hardware system name
        gpu_counts: List of GPU counts to evaluate (default: powers of 2)
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        hourly_rate_per_gpu: Cost per GPU per hour
        num_epochs: Number of epochs

    Returns:
        List of ScalingPoint with metrics at each GPU count
    """
    if gpu_counts is None:
        gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    results = []
    single_gpu_tps = None

    for num_gpus in gpu_counts:
        try:
            estimate = estimate_training_time(
                model=model,
                dataset_tokens=dataset_tokens,
                num_epochs=num_epochs,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=num_gpus,
                training_stage=training_stage,
                method=method,
                optimizer=optimizer,
                bits=bits,
                hourly_rate_per_gpu=hourly_rate_per_gpu,
            )

            # Calculate scaling efficiency vs single GPU
            if single_gpu_tps is None and num_gpus == 1:
                single_gpu_tps = estimate.tokens_per_second

            if single_gpu_tps and single_gpu_tps > 0:
                ideal_tps = single_gpu_tps * num_gpus
                actual_efficiency = estimate.tokens_per_second / ideal_tps
            else:
                actual_efficiency = estimate.scaling_efficiency

            point = ScalingPoint(
                num_gpus=num_gpus,
                throughput_tokens_per_sec=estimate.tokens_per_second,
                scaling_efficiency=actual_efficiency,
                cost_per_million_tokens=estimate.cost_per_million_tokens,
                total_hours=estimate.total_hours,
                total_cost_usd=estimate.cost_estimate_usd,
                parallelism=estimate.parallelism,
                mfu=estimate.mfu,
            )
            results.append(point)

        except Exception:
            # Skip GPU counts that don't work (e.g., model too large)
            continue

    return results


def scaling_curve_to_dataframe(points: List[ScalingPoint]):
    """Convert scaling curve to pandas DataFrame for analysis."""
    try:
        import pandas as pd

        data = []
        for p in points:
            data.append({
                'num_gpus': p.num_gpus,
                'throughput_tps': p.throughput_tokens_per_sec,
                'scaling_efficiency': p.scaling_efficiency,
                'cost_per_mtok': p.cost_per_million_tokens,
                'total_hours': p.total_hours,
                'total_cost_usd': p.total_cost_usd,
                'mfu': p.mfu,
                'tp': p.parallelism.tensor_parallel if p.parallelism else 1,
                'pp': p.parallelism.pipeline_parallel if p.parallelism else 1,
                'dp': p.parallelism.data_parallel if p.parallelism else 1,
                'zero': p.parallelism.zero_stage if p.parallelism else 0,
            })

        return pd.DataFrame(data)
    except ImportError:
        return points


def find_optimal_gpu_count(
    model: str,
    dataset_tokens: int,
    system_name: str = 'H100_GPU',
    max_gpus: int = 512,
    max_cost_usd: Optional[float] = None,
    max_hours: Optional[float] = None,
    optimization_target: str = 'cost_efficiency',
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    hourly_rate_per_gpu: float = 3.0,
    num_epochs: float = 1.0,
) -> Tuple[int, TrainingTimeEstimate]:
    """
    Find optimal GPU count for training given constraints.

    Args:
        model: Model name
        dataset_tokens: Total tokens in dataset
        system_name: Hardware system name
        max_gpus: Maximum GPUs to consider
        max_cost_usd: Maximum total cost constraint
        max_hours: Maximum training time constraint
        optimization_target: What to optimize:
            - 'cost_efficiency': Best $/token
            - 'time': Fastest completion
            - 'throughput': Highest tokens/sec
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        hourly_rate_per_gpu: Cost per GPU per hour
        num_epochs: Number of epochs

    Returns:
        Tuple of (optimal_gpu_count, TrainingTimeEstimate)
    """
    # Generate GPU counts to search (powers of 2 + common cluster sizes)
    gpu_counts = set()
    power = 0
    while 2 ** power <= max_gpus:
        gpu_counts.add(2 ** power)
        power += 1

    # Add common sizes
    for size in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        if size <= max_gpus:
            gpu_counts.add(size)

    gpu_counts = sorted(gpu_counts)

    best_estimate = None
    best_gpu_count = 1
    best_score = float('-inf') if optimization_target != 'cost_efficiency' else float('inf')

    for num_gpus in gpu_counts:
        try:
            estimate = estimate_training_time(
                model=model,
                dataset_tokens=dataset_tokens,
                num_epochs=num_epochs,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=num_gpus,
                training_stage=training_stage,
                method=method,
                optimizer=optimizer,
                bits=bits,
                hourly_rate_per_gpu=hourly_rate_per_gpu,
            )

            # Check constraints
            if max_cost_usd is not None and estimate.cost_estimate_usd > max_cost_usd:
                continue
            if max_hours is not None and estimate.total_hours > max_hours:
                continue

            # Score based on optimization target
            if optimization_target == 'cost_efficiency':
                score = estimate.cost_per_million_tokens
                is_better = score < best_score
            elif optimization_target == 'time':
                score = estimate.total_hours
                is_better = score < best_score
            elif optimization_target == 'throughput':
                score = estimate.tokens_per_second
                is_better = score > best_score
            else:
                score = estimate.cost_per_million_tokens
                is_better = score < best_score

            if is_better:
                best_score = score
                best_estimate = estimate
                best_gpu_count = num_gpus

        except Exception:
            continue

    if best_estimate is None:
        raise ValueError(
            f"No valid configuration found within constraints. "
            f"max_gpus={max_gpus}, max_cost=${max_cost_usd}, max_hours={max_hours}"
        )

    return best_gpu_count, best_estimate
