"""
Scale Optimizer - Find optimal number of DP replicas (gangs) for training.

This module provides functions to find the optimal scale (number of GPUs)
for training, balancing throughput, cost, and efficiency.

Main entry points:
- find_optimal_scale(): Find optimal GPU count for throughput/cost target
- find_scaling_frontier(): Get Pareto frontier of scale vs cost
- recommend_gang_configuration(): Recommend number of DP replicas

Example usage:
    >>> from llm_memory_calculator.training import find_optimal_scale
    >>>
    >>> result = find_optimal_scale(
    ...     model='Llama-3.1-70B',
    ...     hardware_type='H100',
    ...     target_throughput=1_000_000,  # 1M tokens/sec
    ...     max_cost_per_hour=1000.0,
    ... )
    >>>
    >>> print(f"Optimal GPUs: {result.optimal_num_gpus}")
    >>> print(f"DP Replicas (gangs): {result.data_parallel}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
from ..genz.LLM_training.training_parallelization import (
    get_best_training_parallelization,
    TrainingParallelismConfig,
    get_training_parallelization_options,
)
from .cluster_optimizer import _get_gpu_memory, _get_system_name, GPU_MEMORY_GB
from .tco_calculator import get_gpu_pricing


@dataclass
class ScalingRecommendation:
    """
    Recommendation for optimal training scale.

    Contains optimal GPU count and parallelism configuration
    with associated performance and cost metrics.
    """
    # Optimal configuration
    optimal_num_gpus: int
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int  # Number of "gangs"
    expert_parallel: int
    zero_stage: int

    # Performance
    throughput_tokens_per_sec: float
    step_time_ms: float
    mfu: float

    # Scaling metrics
    efficiency_vs_single_gpu: float  # Scaling efficiency
    communication_overhead_pct: float

    # Cost
    cost_per_hour: float
    cost_per_million_tokens: float

    # Memory
    memory_per_gpu_gb: float

    # Optional: full parallelism config
    parallelism: Optional[TrainingParallelismConfig] = None

    # Constraints satisfaction
    satisfies_target_throughput: bool = True
    satisfies_cost_constraint: bool = True

    # Additional metadata
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'configuration': {
                'optimal_num_gpus': self.optimal_num_gpus,
                'tensor_parallel': self.tensor_parallel,
                'pipeline_parallel': self.pipeline_parallel,
                'data_parallel': self.data_parallel,
                'expert_parallel': self.expert_parallel,
                'zero_stage': self.zero_stage,
            },
            'performance': {
                'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
                'step_time_ms': self.step_time_ms,
                'mfu': self.mfu,
                'efficiency_vs_single_gpu': self.efficiency_vs_single_gpu,
                'communication_overhead_pct': self.communication_overhead_pct,
            },
            'cost': {
                'cost_per_hour': self.cost_per_hour,
                'cost_per_million_tokens': self.cost_per_million_tokens,
            },
            'memory': {
                'memory_per_gpu_gb': self.memory_per_gpu_gb,
            },
            'constraints': {
                'satisfies_target_throughput': self.satisfies_target_throughput,
                'satisfies_cost_constraint': self.satisfies_cost_constraint,
            },
            'notes': self.notes,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "SCALING RECOMMENDATION",
            "=" * 60,
            "",
            f"Optimal GPUs: {self.optimal_num_gpus}",
            f"Parallelism: TP={self.tensor_parallel}, PP={self.pipeline_parallel}, "
            f"DP={self.data_parallel} (gangs)",
            f"ZeRO Stage: {self.zero_stage}",
            "",
            f"Throughput: {self.throughput_tokens_per_sec:,.0f} tokens/sec",
            f"Scaling Efficiency: {self.efficiency_vs_single_gpu:.1%}",
            f"Communication Overhead: {self.communication_overhead_pct:.1%}",
            f"MFU: {self.mfu:.1%}",
            "",
            f"Memory/GPU: {self.memory_per_gpu_gb:.1f} GB",
            "",
            f"Cost/hour: ${self.cost_per_hour:.2f}",
            f"Cost/M tokens: ${self.cost_per_million_tokens:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def find_optimal_scale(
    model: str,
    hardware_type: str = 'H100',
    min_gpus: int = 1,
    max_gpus: int = 512,
    target_throughput: Optional[float] = None,
    max_cost_per_hour: Optional[float] = None,
    optimization_target: str = 'cost_efficiency',
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    hourly_rate_per_gpu: Optional[float] = None,
) -> ScalingRecommendation:
    """
    Find optimal number of GPUs and DP replicas (gangs) for training.

    Args:
        model: Model name
        hardware_type: GPU type (H100, A100_80GB, etc.)
        min_gpus: Minimum GPUs to consider
        max_gpus: Maximum GPUs to consider
        target_throughput: Target throughput in tokens/sec (optional)
        max_cost_per_hour: Maximum cost per hour constraint (optional)
        optimization_target: What to optimize:
            - 'cost_efficiency': Best $/token (default)
            - 'throughput': Maximum tokens/sec within constraints
            - 'mfu': Maximum hardware utilization
            - 'time': Minimize step time
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        hourly_rate_per_gpu: Cost per GPU per hour (auto-fetched if None)

    Returns:
        ScalingRecommendation with optimal configuration
    """
    # Get GPU specs
    gpu_memory = _get_gpu_memory(hardware_type)
    system_name = _get_system_name(hardware_type)

    # Get hourly rate if not provided
    if hourly_rate_per_gpu is None:
        try:
            pricing = get_gpu_pricing(hardware_type)
            hourly_rate_per_gpu, _, _ = pricing.get_best_rate(allow_spot=False)
        except Exception:
            hourly_rate_per_gpu = 3.0  # Default fallback

    # Generate GPU counts to search
    gpu_counts = _generate_gpu_counts(min_gpus, max_gpus)

    # Track best result and baseline for efficiency calculation
    best_result = None
    # All scores are oriented so higher is better (cost/time are negated)
    best_score = float('-inf')
    baseline_throughput = None  # First successful config's throughput
    baseline_gpus = None

    candidates = []

    for num_gpus in gpu_counts:
        try:
            # Find best parallelism for this GPU count
            config, sim_result = get_best_training_parallelization(
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
                gpu_memory_gb=gpu_memory,
            )

            # Track baseline for efficiency calculation (first successful config)
            if baseline_throughput is None:
                baseline_throughput = sim_result.tokens_per_second
                baseline_gpus = num_gpus

            # Calculate costs
            cost_per_hour = hourly_rate_per_gpu * num_gpus
            tokens_per_hour = sim_result.tokens_per_second * 3600
            cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1_000_000 if tokens_per_hour > 0 else float('inf')

            # Calculate scaling efficiency
            # Use baseline (first successful config) to extrapolate ideal scaling
            if baseline_throughput and baseline_gpus and baseline_throughput > 0:
                # Estimate per-GPU throughput from baseline and extrapolate
                per_gpu_baseline = baseline_throughput / baseline_gpus
                ideal_throughput = per_gpu_baseline * num_gpus
                efficiency = sim_result.tokens_per_second / ideal_throughput
            else:
                efficiency = 1.0 - sim_result.communication_overhead

            # Communication overhead
            comm_overhead = sim_result.communication_overhead if hasattr(sim_result, 'communication_overhead') else (1 - efficiency)

            # Check constraints
            satisfies_throughput = True
            if target_throughput is not None:
                satisfies_throughput = sim_result.tokens_per_second >= target_throughput

            satisfies_cost = True
            if max_cost_per_hour is not None:
                satisfies_cost = cost_per_hour <= max_cost_per_hour

            # Skip if doesn't satisfy constraints
            if not (satisfies_throughput and satisfies_cost):
                continue

            # Calculate score
            if optimization_target == 'cost_efficiency':
                score = -cost_per_million_tokens  # Lower is better, negate for max
            elif optimization_target == 'throughput':
                score = sim_result.tokens_per_second
            elif optimization_target == 'mfu':
                score = sim_result.model_flops_utilization
            elif optimization_target == 'time':
                score = -sim_result.step_time_ms
            else:
                score = -cost_per_million_tokens

            candidate = {
                'num_gpus': num_gpus,
                'config': config,
                'sim_result': sim_result,
                'cost_per_hour': cost_per_hour,
                'cost_per_million_tokens': cost_per_million_tokens,
                'efficiency': efficiency,
                'comm_overhead': comm_overhead,
                'score': score,
                'satisfies_throughput': satisfies_throughput,
                'satisfies_cost': satisfies_cost,
            }
            candidates.append(candidate)

            # Update best if better (all scores are oriented so higher is better)
            if score > best_score:
                best_score = score
                best_result = candidate

        except Exception:
            continue

    if best_result is None:
        raise ValueError(
            f"No valid configuration found. "
            f"model={model}, hardware={hardware_type}, "
            f"target_throughput={target_throughput}, max_cost=${max_cost_per_hour}"
        )

    # Build result
    config = best_result['config']
    sim_result = best_result['sim_result']

    return ScalingRecommendation(
        optimal_num_gpus=best_result['num_gpus'],
        tensor_parallel=config.tensor_parallel,
        pipeline_parallel=config.pipeline_parallel,
        data_parallel=config.data_parallel,
        expert_parallel=config.expert_parallel if hasattr(config, 'expert_parallel') else 1,
        zero_stage=config.zero_stage,
        throughput_tokens_per_sec=sim_result.tokens_per_second,
        step_time_ms=sim_result.step_time_ms,
        mfu=sim_result.model_flops_utilization,
        efficiency_vs_single_gpu=best_result['efficiency'],
        communication_overhead_pct=best_result['comm_overhead'],
        cost_per_hour=best_result['cost_per_hour'],
        cost_per_million_tokens=best_result['cost_per_million_tokens'],
        memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
        parallelism=config,
        satisfies_target_throughput=best_result['satisfies_throughput'],
        satisfies_cost_constraint=best_result['satisfies_cost'],
    )


@dataclass
class ScalingFrontierPoint:
    """Single point on the scaling frontier."""
    num_gpus: int
    throughput_tokens_per_sec: float
    cost_per_hour: float
    cost_per_million_tokens: float
    scaling_efficiency: float
    mfu: float
    parallelism: TrainingParallelismConfig


def find_scaling_frontier(
    model: str,
    hardware_type: str = 'H100',
    min_gpus: int = 1,
    max_gpus: int = 256,
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    hourly_rate_per_gpu: float = 3.0,
) -> List[ScalingFrontierPoint]:
    """
    Find Pareto frontier of scale (GPUs) vs cost efficiency.

    Returns points where increasing GPUs gives meaningful throughput improvement
    that isn't dominated by another configuration.

    Args:
        model: Model name
        hardware_type: GPU type
        min_gpus: Minimum GPUs
        max_gpus: Maximum GPUs
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        hourly_rate_per_gpu: Cost per GPU per hour

    Returns:
        List of ScalingFrontierPoint on the Pareto frontier
    """
    gpu_memory = _get_gpu_memory(hardware_type)
    system_name = _get_system_name(hardware_type)
    gpu_counts = _generate_gpu_counts(min_gpus, max_gpus)

    points = []
    single_gpu_tps = None

    for num_gpus in gpu_counts:
        try:
            config, sim_result = get_best_training_parallelization(
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
                gpu_memory_gb=gpu_memory,
            )

            if single_gpu_tps is None and num_gpus == 1:
                single_gpu_tps = sim_result.tokens_per_second

            # Calculate metrics
            cost_per_hour = hourly_rate_per_gpu * num_gpus
            tokens_per_hour = sim_result.tokens_per_second * 3600
            cost_per_mtok = (cost_per_hour / tokens_per_hour) * 1_000_000 if tokens_per_hour > 0 else float('inf')

            if single_gpu_tps and single_gpu_tps > 0:
                efficiency = sim_result.tokens_per_second / (single_gpu_tps * num_gpus)
            else:
                efficiency = 1.0 - sim_result.communication_overhead

            point = ScalingFrontierPoint(
                num_gpus=num_gpus,
                throughput_tokens_per_sec=sim_result.tokens_per_second,
                cost_per_hour=cost_per_hour,
                cost_per_million_tokens=cost_per_mtok,
                scaling_efficiency=efficiency,
                mfu=sim_result.model_flops_utilization,
                parallelism=config,
            )
            points.append(point)

        except Exception:
            continue

    if not points:
        return []

    # Find Pareto frontier: maximize throughput, minimize cost/token
    frontier = []
    for p in points:
        is_dominated = False
        for other in points:
            if other is p:
                continue
            # Other dominates p if it's better or equal in all objectives
            # and strictly better in at least one
            if (other.throughput_tokens_per_sec >= p.throughput_tokens_per_sec and
                other.cost_per_million_tokens <= p.cost_per_million_tokens and
                (other.throughput_tokens_per_sec > p.throughput_tokens_per_sec or
                 other.cost_per_million_tokens < p.cost_per_million_tokens)):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(p)

    # Sort by GPU count
    frontier.sort(key=lambda x: x.num_gpus)

    return frontier


def recommend_gang_configuration(
    model: str,
    hardware_type: str = 'H100',
    num_gpus: int = 8,
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
    maximize_dp: bool = True,
) -> Dict[str, Any]:
    """
    Recommend optimal gang (DP replica) configuration for given GPUs.

    This focuses on finding the parallelism that maximizes data parallelism
    (number of gangs) while maintaining memory feasibility.

    Args:
        model: Model name
        hardware_type: GPU type
        num_gpus: Total GPUs available
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision
        maximize_dp: If True, prefer configurations with higher DP

    Returns:
        Dict with recommended configuration and analysis
    """
    gpu_memory = _get_gpu_memory(hardware_type)
    system_name = _get_system_name(hardware_type)

    # Get all valid configurations
    configs = get_training_parallelization_options(
        model=model,
        total_gpus=num_gpus,
        gpu_memory_gb=gpu_memory,
        batch_size=batch_size,
        seq_length=seq_length,
        method=method,
        optimizer=optimizer,
        bits=bits,
    )

    if not configs:
        return {
            'success': False,
            'error': f'No valid configuration found for {model} on {num_gpus}x {hardware_type}',
        }

    # Evaluate each config
    results = []
    for config in configs:
        try:
            sim_result = training_modeling(
                model=model,
                training_stage=training_stage,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=num_gpus,
                tensor_parallel=config.tensor_parallel,
                data_parallel=config.data_parallel,
                pipeline_parallel=config.pipeline_parallel,
                method=method,
                optimizer=optimizer,
                zero_stage=config.zero_stage,
                gradient_checkpointing=config.gradient_checkpointing,
                bits=bits,
            )

            results.append({
                'config': config,
                'sim_result': sim_result,
                'dp': config.data_parallel,
                'tp': config.tensor_parallel,
                'pp': config.pipeline_parallel,
                'throughput': sim_result.tokens_per_second,
                'memory': sim_result.memory_per_gpu_gb,
                'mfu': sim_result.model_flops_utilization,
            })
        except Exception:
            continue

    if not results:
        return {
            'success': False,
            'error': 'All configurations failed simulation',
        }

    # Sort by DP (maximize) and throughput (maximize) if maximize_dp
    if maximize_dp:
        results.sort(key=lambda x: (-x['dp'], -x['throughput']))
    else:
        results.sort(key=lambda x: -x['throughput'])

    best = results[0]

    return {
        'success': True,
        'recommended': {
            'tensor_parallel': best['tp'],
            'pipeline_parallel': best['pp'],
            'data_parallel': best['dp'],
            'zero_stage': best['config'].zero_stage,
            'gradient_checkpointing': best['config'].gradient_checkpointing,
        },
        'num_gangs': best['dp'],
        'throughput_tokens_per_sec': best['throughput'],
        'memory_per_gpu_gb': best['memory'],
        'mfu': best['mfu'],
        'all_valid_configs': [
            {
                'tp': r['tp'], 'pp': r['pp'], 'dp': r['dp'],
                'zero': r['config'].zero_stage,
                'throughput': r['throughput'],
                'memory_gb': r['memory'],
            }
            for r in results[:10]  # Top 10
        ],
        'analysis': {
            'total_valid_configs': len(results),
            'max_dp_possible': max(r['dp'] for r in results),
            'max_throughput': max(r['throughput'] for r in results),
            'min_memory': min(r['memory'] for r in results),
        },
    }


def _generate_gpu_counts(min_gpus: int, max_gpus: int) -> List[int]:
    """Generate GPU counts to search (powers of 2 + common sizes)."""
    counts = set()

    # Powers of 2
    power = 0
    while 2 ** power <= max_gpus:
        if 2 ** power >= min_gpus:
            counts.add(2 ** power)
        power += 1

    # Common cluster sizes
    common_sizes = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    for size in common_sizes:
        if min_gpus <= size <= max_gpus:
            counts.add(size)

    return sorted(counts)


def analyze_scaling_efficiency(
    model: str,
    hardware_type: str = 'H100',
    gpu_counts: Optional[List[int]] = None,
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
) -> Dict[str, Any]:
    """
    Analyze scaling efficiency across different GPU counts.

    Returns detailed analysis of how throughput and efficiency scale.

    Args:
        model: Model name
        hardware_type: GPU type
        gpu_counts: GPU counts to analyze (default: powers of 2 up to 256)
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method

    Returns:
        Dict with scaling analysis
    """
    if gpu_counts is None:
        gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    gpu_memory = _get_gpu_memory(hardware_type)
    system_name = _get_system_name(hardware_type)

    data_points = []
    single_gpu_tps = None

    for num_gpus in gpu_counts:
        try:
            config, sim_result = get_best_training_parallelization(
                model=model,
                total_gpus=num_gpus,
                batch_size=batch_size,
                seq_length=seq_length,
                system_name=system_name,
                training_stage=training_stage,
                method=method,
                optimize_for='throughput',
                gpu_memory_gb=gpu_memory,
            )

            if single_gpu_tps is None:
                single_gpu_tps = sim_result.tokens_per_second

            ideal_tps = single_gpu_tps * num_gpus if single_gpu_tps else sim_result.tokens_per_second
            efficiency = sim_result.tokens_per_second / ideal_tps if ideal_tps > 0 else 1.0

            data_points.append({
                'num_gpus': num_gpus,
                'throughput': sim_result.tokens_per_second,
                'ideal_throughput': ideal_tps,
                'efficiency': efficiency,
                'mfu': sim_result.model_flops_utilization,
                'memory_gb': sim_result.memory_per_gpu_gb,
                'tp': config.tensor_parallel,
                'pp': config.pipeline_parallel,
                'dp': config.data_parallel,
                'zero': config.zero_stage,
            })

        except Exception:
            continue

    if not data_points:
        return {'success': False, 'error': 'No valid configurations found'}

    # Calculate summary statistics
    max_efficiency_point = max(data_points, key=lambda x: x['efficiency'])
    max_throughput_point = max(data_points, key=lambda x: x['throughput'])

    return {
        'success': True,
        'scaling_data': data_points,
        'summary': {
            'single_gpu_throughput': single_gpu_tps,
            'best_efficiency': {
                'num_gpus': max_efficiency_point['num_gpus'],
                'efficiency': max_efficiency_point['efficiency'],
                'throughput': max_efficiency_point['throughput'],
            },
            'best_throughput': {
                'num_gpus': max_throughput_point['num_gpus'],
                'efficiency': max_throughput_point['efficiency'],
                'throughput': max_throughput_point['throughput'],
            },
            'efficiency_drop_at_max': data_points[-1]['efficiency'] if data_points else 0,
        },
    }
