"""
Training Memory Calculator - Estimate memory requirements for LLM training.

This module extends the inference memory calculator for training workloads,
including gradient memory, optimizer states, and activation checkpointing.

Supports comprehensive training simulation for:
- Training stages: SFT, DPO, PPO, KTO, RM, Pre-Training
- Fine-tuning methods: Full, LoRA, QLoRA, Freeze, DoRA, PiSSA
- Optimizers: AdamW, 8-bit Adam, GaLore, APOLLO, Adam-mini, etc.
- Distributed: DeepSpeed ZeRO (0-3), FSDP, Tensor/Pipeline Parallelism
"""

from .types import (
    TrainingMethod,
    OptimizerType,
    DeepSpeedStage,
    TrainingMemoryEstimate,
    ClusterRecommendation,
    ClusterFitResult,
    TrainingTimeEstimate,
)

from .calculator import TrainingMemoryCalculator
from .cluster_selector import TrainingClusterSelector
from .time_estimator import TrainingTimeEstimator

# Advanced training types
from .training_types import (
    TrainingStage,
    TrainingStageConfig,
    TrainingStageEstimate,
    DPOConfig,
    KTOConfig,
    PPOConfig,
    DPOLossType,
    get_training_stage_config,
    list_training_stages,
    TRAINING_STAGE_CONFIGS,
)

# Advanced optimizer modeling
from .optimizers import (
    OptimizerCategory,
    OptimizerConfig,
    get_optimizer_config,
    list_optimizers,
    calculate_optimizer_memory,
    get_recommended_optimizer,
    OPTIMIZER_CONFIGS,
)

# Distributed training modeling
from .distributed import (
    DeepSpeedZeROStage,
    FSDPStrategy,
    OffloadTarget,
    DeepSpeedConfig,
    FSDPConfig,
    ParallelismConfig,
    DistributedMemoryEstimate,
    get_deepspeed_config,
    get_fsdp_config,
    calculate_distributed_memory,
    recommend_distributed_strategy,
    DEEPSPEED_CONFIGS,
    FSDP_CONFIGS,
)

# Advanced calculator
from .advanced_calculator import (
    AdvancedTrainingEstimate,
    AdvancedTrainingCalculator,
    calculate_advanced_training_memory,
    list_supported_configurations,
    calculate_training_with_genz,
)

# Hardware catalog
from .hardware_catalog import (
    GPUGeneration,
    MemoryType,
    InterconnectType,
    ComputeCapability,
    GPUSpec,
    GPUCost,
    ClusterSpec,
    GPU_SPECS,
    GPU_COSTS,
    get_gpu_spec,
    get_gpu_cost,
    list_gpus,
    get_optimal_gpu_for_model,
    estimate_cluster_cost,
    calculate_communication_time,
)

# Model characteristics
from .model_characteristics import (
    AttentionType,
    FFNType,
    ModelFamily,
    AttentionConfig,
    MoEConfig,
    ModelCharacteristics,
    MODEL_CHARACTERISTICS,
    get_model_characteristics,
    analyze_model_config,
    list_model_characteristics,
)

# Training optimizer with GenZ integration
from .training_optimizer import (
    OptimizationGoal,
    DatasetConfig,
    TrainingConstraints,
    OptimizerRecommendation,
    ClusterRecommendation,
    TrainingPlan,
    TrainingOptimizer,
    optimize_training,
    compare_training_strategies,
    get_minimum_hardware_for_model,
    # GenZ integration functions
    create_genz_system_from_gpu,
    calculate_training_throughput,
    calculate_communication_overhead,
)

# Cluster optimizer types
from .cluster_optimizer_types import (
    OptimizationTarget,
    PricingTier,
    ClusterDefinition,
    TrainingJobSpec,
    TCOBreakdown,
    ParallelismStrategy,
    ClusterRecommendationResult,
    OptimalClusterDesignResult,
)

# TCO calculator
from .tco_calculator import (
    GPUPricing,
    GPU_PRICING,
    GPU_PRICING_ALIASES,
    get_gpu_pricing,
    calculate_tco,
    estimate_training_cost,
    compare_provider_costs,
)

# LlamaFactory config builder
from .llamafactory_config_builder import (
    build_llamafactory_config,
    build_deepspeed_config,
    save_llamafactory_config,
    save_deepspeed_config,
    generate_training_configs,
    generate_launch_command,
)

# Cluster optimizer
from .cluster_optimizer import (
    ClusterOptimizer,
    select_optimal_cluster,
    design_optimal_training_cluster,
)

# Validation suite
from .validation_suite import (
    ValidationConfig,
    ValidationResult,
    VALIDATION_CONFIGS,
    run_validation_config,
    run_comprehensive_validation,
    run_quick_validation,
    analyze_correlations,
    list_validation_configs,
    get_validation_config,
)

# Cluster optimizer validation
from .cluster_optimizer_validation import (
    validate_tco_calculations,
    validate_parallelism_selection,
    validate_pareto_frontier,
    validate_optimization_targets,
    validate_memory_constraints,
    run_cluster_optimizer_validation,
)

# Training time estimator (NEW)
from .training_time_estimator import (
    TrainingTimeEstimate as DatasetTrainingTimeEstimate,
    estimate_training_time,
    calculate_training_steps,
    estimate_time_from_throughput,
    ScalingPoint,
    estimate_scaling_curve,
    scaling_curve_to_dataframe,
    find_optimal_gpu_count,
)

# Scale optimizer (NEW)
from .scale_optimizer import (
    ScalingRecommendation,
    find_optimal_scale,
    ScalingFrontierPoint,
    find_scaling_frontier,
    recommend_gang_configuration,
    analyze_scaling_efficiency,
)

# Node selector (NEW)
from .node_selector import (
    NodeSpec,
    NodeSelectionResult,
    select_optimal_nodes,
    find_homogeneous_groups,
    evaluate_node_combination,
    rank_node_selections,
)

# Auto-config unified API (NEW)
from .auto_config import (
    OptimalTrainingPlan,
    auto_configure_training,
    quick_configure,
)

__all__ = [
    # Original types
    "TrainingMethod",
    "OptimizerType",
    "DeepSpeedStage",
    "TrainingMemoryEstimate",
    "ClusterRecommendation",
    "ClusterFitResult",
    "TrainingTimeEstimate",

    # Original classes
    "TrainingMemoryCalculator",
    "TrainingClusterSelector",
    "TrainingTimeEstimator",

    # Training stages
    "TrainingStage",
    "TrainingStageConfig",
    "TrainingStageEstimate",
    "DPOConfig",
    "KTOConfig",
    "PPOConfig",
    "DPOLossType",
    "get_training_stage_config",
    "list_training_stages",
    "TRAINING_STAGE_CONFIGS",

    # Optimizer modeling
    "OptimizerCategory",
    "OptimizerConfig",
    "get_optimizer_config",
    "list_optimizers",
    "calculate_optimizer_memory",
    "get_recommended_optimizer",
    "OPTIMIZER_CONFIGS",

    # Distributed training
    "DeepSpeedZeROStage",
    "FSDPStrategy",
    "OffloadTarget",
    "DeepSpeedConfig",
    "FSDPConfig",
    "ParallelismConfig",
    "DistributedMemoryEstimate",
    "get_deepspeed_config",
    "get_fsdp_config",
    "calculate_distributed_memory",
    "recommend_distributed_strategy",
    "DEEPSPEED_CONFIGS",
    "FSDP_CONFIGS",

    # Advanced calculator
    "AdvancedTrainingEstimate",
    "AdvancedTrainingCalculator",
    "calculate_advanced_training_memory",
    "list_supported_configurations",
    "calculate_training_with_genz",

    # Hardware catalog
    "GPUGeneration",
    "MemoryType",
    "InterconnectType",
    "ComputeCapability",
    "GPUSpec",
    "GPUCost",
    "ClusterSpec",
    "GPU_SPECS",
    "GPU_COSTS",
    "get_gpu_spec",
    "get_gpu_cost",
    "list_gpus",
    "get_optimal_gpu_for_model",
    "estimate_cluster_cost",
    "calculate_communication_time",

    # Model characteristics
    "AttentionType",
    "FFNType",
    "ModelFamily",
    "AttentionConfig",
    "MoEConfig",
    "ModelCharacteristics",
    "MODEL_CHARACTERISTICS",
    "get_model_characteristics",
    "analyze_model_config",
    "list_model_characteristics",

    # Training optimizer with GenZ integration
    "OptimizationGoal",
    "DatasetConfig",
    "TrainingConstraints",
    "OptimizerRecommendation",
    "ClusterRecommendation",
    "TrainingPlan",
    "TrainingOptimizer",
    "optimize_training",
    "compare_training_strategies",
    "get_minimum_hardware_for_model",
    "create_genz_system_from_gpu",
    "calculate_training_throughput",
    "calculate_communication_overhead",

    # Cluster optimizer types
    "OptimizationTarget",
    "PricingTier",
    "ClusterDefinition",
    "TrainingJobSpec",
    "TCOBreakdown",
    "ParallelismStrategy",
    "ClusterRecommendationResult",
    "OptimalClusterDesignResult",

    # TCO calculator
    "GPUPricing",
    "GPU_PRICING",
    "GPU_PRICING_ALIASES",
    "get_gpu_pricing",
    "calculate_tco",
    "estimate_training_cost",
    "compare_provider_costs",

    # LlamaFactory config builder
    "build_llamafactory_config",
    "build_deepspeed_config",
    "save_llamafactory_config",
    "save_deepspeed_config",
    "generate_training_configs",
    "generate_launch_command",

    # Cluster optimizer
    "ClusterOptimizer",
    "select_optimal_cluster",
    "design_optimal_training_cluster",

    # Validation suite
    "ValidationConfig",
    "ValidationResult",
    "VALIDATION_CONFIGS",
    "run_validation_config",
    "run_comprehensive_validation",
    "run_quick_validation",
    "analyze_correlations",
    "list_validation_configs",
    "get_validation_config",

    # Cluster optimizer validation
    "validate_tco_calculations",
    "validate_parallelism_selection",
    "validate_pareto_frontier",
    "validate_optimization_targets",
    "validate_memory_constraints",
    "run_cluster_optimizer_validation",

    # Training time estimator (NEW)
    "DatasetTrainingTimeEstimate",
    "estimate_training_time",
    "calculate_training_steps",
    "estimate_time_from_throughput",
    "ScalingPoint",
    "estimate_scaling_curve",
    "scaling_curve_to_dataframe",
    "find_optimal_gpu_count",

    # Scale optimizer (NEW)
    "ScalingRecommendation",
    "find_optimal_scale",
    "ScalingFrontierPoint",
    "find_scaling_frontier",
    "recommend_gang_configuration",
    "analyze_scaling_efficiency",

    # Node selector (NEW)
    "NodeSpec",
    "NodeSelectionResult",
    "select_optimal_nodes",
    "find_homogeneous_groups",
    "evaluate_node_combination",
    "rank_node_selections",

    # Auto-config unified API (NEW)
    "OptimalTrainingPlan",
    "auto_configure_training",
    "quick_configure",
]
