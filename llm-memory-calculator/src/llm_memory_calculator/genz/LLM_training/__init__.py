"""
LLM Training Module - Training simulation using GenZ roofline analysis.

This module provides comprehensive training simulation that extends the GenZ
inference framework for training workloads:
- Forward pass timing via existing inference operators
- Backward pass timing via training operators (2x forward FLOPs)
- Communication timing (AllReduce, ReduceScatter, AllGather)
- Optimizer update timing
- Memory usage with ZeRO stages
- Best parallelization strategy prediction
- Validation against published benchmarks

Main entry points:
- training_modeling(): Full training step simulation
- training_modeling_for_stage(): Stage-aware training simulation
- get_best_training_parallelization(): Find optimal parallelism strategy
- validate_against_benchmark(): Validate against published benchmarks

Supported training stages:
- SFT: Supervised Fine-Tuning (standard forward-backward)
- DPO: Direct Preference Optimization (policy + reference)
- PPO: Proximal Policy Optimization (actor + critic + reference + reward)
- PPO_DETAILED: PPO with explicit phase tracking
- GRPO: Group Relative Policy Optimization (DeepSeek)
- IPO: Identity Preference Optimization (reference-free)
- KTO: Kahneman-Tversky Optimization
- ORPO: Odds Ratio Preference Optimization
- SimPO: Simple Preference Optimization
- RLOO: REINFORCE Leave-One-Out
- RM: Reward Modeling

Example usage:
    >>> from llm_memory_calculator.genz.LLM_training import training_modeling
    >>>
    >>> # Basic SFT training simulation
    >>> result = training_modeling(
    ...     model='llama-3-8b',
    ...     training_stage='sft',
    ...     batch_size=4,
    ...     seq_length=4096,
    ...     system_name='H100_GPU',
    ...     num_gpus=8,
    ...     data_parallel=8,
    ...     method='lora',
    ...     optimizer='adamw',
    ...     zero_stage=2,
    ... )
    >>>
    >>> print(f"Step Time: {result.step_time_ms:.1f} ms")
    >>> print(f"Throughput: {result.tokens_per_second:.0f} tok/s")
    >>> print(f"Memory/GPU: {result.memory_per_gpu_gb:.1f} GB")
    >>> print(f"MFU: {result.model_flops_utilization:.1%}")

    >>> # Find best parallelization strategy
    >>> from llm_memory_calculator.genz.LLM_training import get_best_training_parallelization
    >>> config, result = get_best_training_parallelization(
    ...     model='llama-3-70b',
    ...     total_gpus=64,
    ...     batch_size=4,
    ...     seq_length=4096,
    ...     system_name='H100_GPU',
    ... )
    >>> print(f"Best: TP={config.tensor_parallel}, PP={config.pipeline_parallel}, DP={config.data_parallel}")
"""

from .training_modeling import (
    TrainingModelingOutput,
    training_modeling,
)

from .training_stages import (
    TrainingStageType,
    TrainingStageConfig,
    TRAINING_STAGE_CONFIGS,
    get_stage_config,
    calculate_stage_memory_multiplier,
    calculate_stage_compute_multiplier,
    training_modeling_for_stage,
    estimate_dpo_training,
    estimate_ppo_training,
    list_training_stages,
    get_rlhf_stages,
    get_preference_stages,
    get_stage_memory_requirements,
    estimate_grpo_training,
    estimate_ipo_training,
    compare_training_stages,
)

from .training_parallelization import (
    TrainingParallelismConfig,
    get_training_parallelization_options,
    get_best_training_parallelization,
    get_pareto_optimal_training_performance,
    get_various_training_parallelization,
    recommend_training_config,
)

from .validation import (
    PublishedBenchmark,
    ValidationResult,
    PUBLISHED_BENCHMARKS,
    validate_against_benchmark,
    run_validation_suite,
    run_quick_validation,
    get_benchmark_info,
    list_benchmarks,
    CLUSTER_NETWORK_CONFIGS,
    get_network_config,
)

__all__ = [
    # Main entry points
    'training_modeling',
    'training_modeling_for_stage',

    # Output types
    'TrainingModelingOutput',

    # Stage configuration
    'TrainingStageType',
    'TrainingStageConfig',
    'TRAINING_STAGE_CONFIGS',
    'get_stage_config',
    'calculate_stage_memory_multiplier',
    'calculate_stage_compute_multiplier',
    'list_training_stages',
    'get_rlhf_stages',
    'get_preference_stages',
    'get_stage_memory_requirements',
    'compare_training_stages',

    # Convenience functions for stages
    'estimate_dpo_training',
    'estimate_ppo_training',
    'estimate_grpo_training',
    'estimate_ipo_training',

    # Parallelization strategy
    'TrainingParallelismConfig',
    'get_training_parallelization_options',
    'get_best_training_parallelization',
    'get_pareto_optimal_training_performance',
    'get_various_training_parallelization',
    'recommend_training_config',

    # Validation
    'PublishedBenchmark',
    'ValidationResult',
    'PUBLISHED_BENCHMARKS',
    'validate_against_benchmark',
    'run_validation_suite',
    'run_quick_validation',
    'get_benchmark_info',
    'list_benchmarks',
    'CLUSTER_NETWORK_CONFIGS',
    'get_network_config',
]
