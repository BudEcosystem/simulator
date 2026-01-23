"""
Simulation Accuracy Validation Framework.

This module provides tools to validate the accuracy of LLM training simulation
outputs against real-world benchmarks.

Key Components:
- benchmark_database: Published benchmark data (40+ benchmarks)
- benchmark_schema: Extended benchmark dataclasses and enums
- accuracy_metrics: Statistical accuracy calculations (MAE, MRE, correlation)
- calibration_engine: Efficiency factor tuning
- accuracy_reporter: Report generation
- benchmark_validator: Validation rules and quality checks
- coverage_matrix: Coverage tracking and gap analysis
- optimizer_benchmarks: Optimizer-specific benchmarks
- multimodel_benchmarks: PPO/DPO/GRPO benchmarks
- collectors: Data collection from various sources
"""

# Original benchmark database
from .benchmark_database import (
    PublishedBenchmark,
    PUBLISHED_BENCHMARKS,
    get_benchmark,
    list_benchmarks,
    get_benchmarks_by_category,
    get_high_confidence_benchmarks,
    BenchmarkCategory,
    ConfidenceLevel,
)

# Extended benchmark schema
from .benchmark_schema import (
    # Enums
    SourceType,
    TrainingType,
    QuantizationMethod,
    OptimizerType,
    VerificationStatus,
    HardwareType,
    GPUScale,
    ModelSizeCategory,
    # Dataclasses
    SourceProvenance,
    QuantizationConfig,
    ParallelismConfig,
    MultiModelConfig,
    PEFTConfig,
    ReportedMetrics,
    ExtendedBenchmark,
    # Factory functions
    create_ppo_config,
    create_dpo_config,
    create_grpo_config,
    create_lora_config,
    create_qlora_config,
)

# Accuracy metrics
from .accuracy_metrics import (
    AccuracyMetrics,
    calculate_accuracy_metrics,
    run_accuracy_assessment,
)

# Calibration engine
from .calibration_engine import (
    CalibrationFactors,
    CalibrationEngine,
)

# Accuracy reporter
from .accuracy_reporter import (
    AccuracyReport,
    generate_report,
    generate_summary,
)

# Benchmark validator
from .benchmark_validator import (
    BenchmarkValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_benchmark_plausibility,
    validate_benchmarks_batch,
    filter_valid_benchmarks,
)

# Coverage matrix
from .coverage_matrix import (
    CoverageMatrix,
    CoverageCell,
    CoverageReport,
    analyze_coverage,
    identify_collection_priorities,
)

# Optimizer benchmarks
from .optimizer_benchmarks import (
    OptimizerProfile,
    OPTIMIZER_PROFILES,
    OPTIMIZER_BENCHMARKS,
    get_optimizer_profile,
    estimate_optimizer_memory,
    compare_optimizer_memory,
    get_optimizer_benchmarks,
    validate_optimizer_memory,
)

# Multi-model benchmarks
from .multimodel_benchmarks import (
    MultiModelProfile,
    MULTIMODEL_PROFILES,
    MULTIMODEL_BENCHMARKS,
    get_multimodel_profile,
    estimate_multimodel_memory,
    get_multimodel_benchmarks,
    validate_multimodel_memory,
)

# Collectors (import submodule)
from . import collectors

__all__ = [
    # Original Benchmark Database
    'PublishedBenchmark',
    'PUBLISHED_BENCHMARKS',
    'get_benchmark',
    'list_benchmarks',
    'get_benchmarks_by_category',
    'get_high_confidence_benchmarks',
    'BenchmarkCategory',
    'ConfidenceLevel',

    # Extended Benchmark Schema - Enums
    'SourceType',
    'TrainingType',
    'QuantizationMethod',
    'OptimizerType',
    'VerificationStatus',
    'HardwareType',
    'GPUScale',
    'ModelSizeCategory',

    # Extended Benchmark Schema - Dataclasses
    'SourceProvenance',
    'QuantizationConfig',
    'ParallelismConfig',
    'MultiModelConfig',
    'PEFTConfig',
    'ReportedMetrics',
    'ExtendedBenchmark',

    # Extended Benchmark Schema - Factory functions
    'create_ppo_config',
    'create_dpo_config',
    'create_grpo_config',
    'create_lora_config',
    'create_qlora_config',

    # Accuracy Metrics
    'AccuracyMetrics',
    'calculate_accuracy_metrics',
    'run_accuracy_assessment',

    # Calibration Engine
    'CalibrationFactors',
    'CalibrationEngine',

    # Accuracy Reporter
    'AccuracyReport',
    'generate_report',
    'generate_summary',

    # Benchmark Validator
    'BenchmarkValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'validate_benchmark_plausibility',
    'validate_benchmarks_batch',
    'filter_valid_benchmarks',

    # Coverage Matrix
    'CoverageMatrix',
    'CoverageCell',
    'CoverageReport',
    'analyze_coverage',
    'identify_collection_priorities',

    # Optimizer Benchmarks
    'OptimizerProfile',
    'OPTIMIZER_PROFILES',
    'OPTIMIZER_BENCHMARKS',
    'get_optimizer_profile',
    'estimate_optimizer_memory',
    'compare_optimizer_memory',
    'get_optimizer_benchmarks',
    'validate_optimizer_memory',

    # Multi-model Benchmarks
    'MultiModelProfile',
    'MULTIMODEL_PROFILES',
    'MULTIMODEL_BENCHMARKS',
    'get_multimodel_profile',
    'estimate_multimodel_memory',
    'get_multimodel_benchmarks',
    'validate_multimodel_memory',

    # Collectors module
    'collectors',
]
