"""
Optimizer-Specific Benchmarks for Training Simulation Validation.

This module contains benchmarks specifically for validating optimizer memory
and performance characteristics across different optimizer types.

Optimizers covered:
- AdamW (standard, 8-bit, fused)
- GaLore and variants
- APOLLO
- Adam-mini
- Lion
- Adafactor
- Muon
- BAdam
- LOMO/AdaLOMO
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .benchmark_schema import (
    ExtendedBenchmark,
    HardwareType,
    OptimizerType,
    ParallelismConfig,
    QuantizationConfig,
    QuantizationMethod,
    ReportedMetrics,
    SourceProvenance,
    SourceType,
    TrainingType,
    VerificationStatus,
)


@dataclass
class OptimizerProfile:
    """Profile for an optimizer's memory and compute characteristics."""

    optimizer_type: OptimizerType
    name: str
    description: str

    # Memory characteristics
    bytes_per_param: float  # Optimizer state bytes per parameter
    memory_multiplier: float  # vs AdamW (1.0 = same as AdamW)

    # Computation characteristics
    compute_overhead: float  # Additional compute vs AdamW (1.0 = same)
    supports_quantization: bool = False
    supports_cpu_offload: bool = True

    # Convergence characteristics
    convergence_factor: float = 1.0  # 1.0 = same convergence as AdamW
    recommended_lr_multiplier: float = 1.0

    # Compatibility
    supports_distributed: bool = True
    supports_mixed_precision: bool = True
    min_gpu_memory_gb: float = 0.0  # Minimum GPU memory needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimizer_type": self.optimizer_type.value,
            "name": self.name,
            "description": self.description,
            "bytes_per_param": self.bytes_per_param,
            "memory_multiplier": self.memory_multiplier,
            "compute_overhead": self.compute_overhead,
            "supports_quantization": self.supports_quantization,
            "supports_cpu_offload": self.supports_cpu_offload,
            "convergence_factor": self.convergence_factor,
            "recommended_lr_multiplier": self.recommended_lr_multiplier,
            "supports_distributed": self.supports_distributed,
            "supports_mixed_precision": self.supports_mixed_precision,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
        }


# Optimizer profiles with memory and compute characteristics
OPTIMIZER_PROFILES: Dict[OptimizerType, OptimizerProfile] = {
    OptimizerType.ADAMW: OptimizerProfile(
        optimizer_type=OptimizerType.ADAMW,
        name="AdamW",
        description="Standard AdamW optimizer with fp32 states",
        bytes_per_param=8.0,  # 4 bytes m + 4 bytes v
        memory_multiplier=1.0,
        compute_overhead=1.0,
        supports_quantization=False,
    ),
    OptimizerType.ADAMW_8BIT: OptimizerProfile(
        optimizer_type=OptimizerType.ADAMW_8BIT,
        name="8-bit AdamW",
        description="AdamW with int8 quantized states (bitsandbytes)",
        bytes_per_param=2.0,  # 1 byte m + 1 byte v
        memory_multiplier=0.25,
        compute_overhead=1.05,  # Small dequantization overhead
        supports_quantization=True,
    ),
    OptimizerType.ADAMW_FUSED: OptimizerProfile(
        optimizer_type=OptimizerType.ADAMW_FUSED,
        name="Fused AdamW",
        description="AdamW with fused CUDA kernels (Apex/NVIDIA)",
        bytes_per_param=8.0,
        memory_multiplier=1.0,
        compute_overhead=0.95,  # Faster due to fusion
    ),
    OptimizerType.SGD: OptimizerProfile(
        optimizer_type=OptimizerType.SGD,
        name="SGD",
        description="Stochastic Gradient Descent without momentum",
        bytes_per_param=0.0,  # No optimizer state
        memory_multiplier=0.0,
        compute_overhead=0.8,
        convergence_factor=0.7,  # Slower convergence
    ),
    OptimizerType.SGD_MOMENTUM: OptimizerProfile(
        optimizer_type=OptimizerType.SGD_MOMENTUM,
        name="SGD with Momentum",
        description="SGD with momentum state",
        bytes_per_param=4.0,  # 4 bytes momentum
        memory_multiplier=0.5,
        compute_overhead=0.85,
        convergence_factor=0.9,
    ),
    OptimizerType.LION: OptimizerProfile(
        optimizer_type=OptimizerType.LION,
        name="Lion",
        description="Sign-based optimizer with single momentum",
        bytes_per_param=4.0,  # Only momentum
        memory_multiplier=0.5,
        compute_overhead=0.9,
        convergence_factor=1.0,
        recommended_lr_multiplier=0.1,  # Typically needs 10x lower LR
    ),
    OptimizerType.ADAFACTOR: OptimizerProfile(
        optimizer_type=OptimizerType.ADAFACTOR,
        name="Adafactor",
        description="Factorized second moments for memory efficiency",
        bytes_per_param=4.0,  # Factorized states
        memory_multiplier=0.5,
        compute_overhead=1.1,
        convergence_factor=0.95,
    ),
    OptimizerType.GALORE: OptimizerProfile(
        optimizer_type=OptimizerType.GALORE,
        name="GaLore",
        description="Gradient Low-Rank Projection optimizer",
        bytes_per_param=2.0,  # Low-rank states
        memory_multiplier=0.25,
        compute_overhead=1.2,  # SVD projection overhead
        supports_distributed=True,
    ),
    OptimizerType.GALORE_8BIT: OptimizerProfile(
        optimizer_type=OptimizerType.GALORE_8BIT,
        name="GaLore 8-bit",
        description="GaLore with 8-bit quantized low-rank states",
        bytes_per_param=1.0,
        memory_multiplier=0.125,
        compute_overhead=1.25,
        supports_quantization=True,
    ),
    OptimizerType.APOLLO: OptimizerProfile(
        optimizer_type=OptimizerType.APOLLO,
        name="APOLLO",
        description="Rank-1 approximate second-order optimizer",
        bytes_per_param=2.0,
        memory_multiplier=0.25,
        compute_overhead=1.15,
    ),
    OptimizerType.ADAM_MINI: OptimizerProfile(
        optimizer_type=OptimizerType.ADAM_MINI,
        name="Adam-mini",
        description="Memory-efficient Adam with reduced state",
        bytes_per_param=4.0,
        memory_multiplier=0.5,
        compute_overhead=1.05,
        convergence_factor=0.98,
    ),
    OptimizerType.Q_ADAM_MINI: OptimizerProfile(
        optimizer_type=OptimizerType.Q_ADAM_MINI,
        name="Q-Adam-mini",
        description="Quantized Adam-mini variant",
        bytes_per_param=2.0,
        memory_multiplier=0.25,
        compute_overhead=1.1,
        supports_quantization=True,
    ),
    OptimizerType.MUON: OptimizerProfile(
        optimizer_type=OptimizerType.MUON,
        name="Muon",
        description="Momentum + Unitized gradient optimizer",
        bytes_per_param=4.0,
        memory_multiplier=0.5,
        compute_overhead=0.95,
    ),
    OptimizerType.BADAM: OptimizerProfile(
        optimizer_type=OptimizerType.BADAM,
        name="BAdam",
        description="Block-wise Adam for extreme memory efficiency",
        bytes_per_param=0.3,  # Only active block
        memory_multiplier=0.04,
        compute_overhead=1.5,  # Additional block management
        convergence_factor=0.9,
    ),
    OptimizerType.LOMO: OptimizerProfile(
        optimizer_type=OptimizerType.LOMO,
        name="LOMO",
        description="Low-Memory Optimization (fused forward-backward)",
        bytes_per_param=0.0,  # No optimizer state
        memory_multiplier=0.0,
        compute_overhead=1.2,
        convergence_factor=0.85,
    ),
    OptimizerType.ADALOMO: OptimizerProfile(
        optimizer_type=OptimizerType.ADALOMO,
        name="AdaLOMO",
        description="Adaptive LOMO with running statistics",
        bytes_per_param=1.0,
        memory_multiplier=0.125,
        compute_overhead=1.25,
        convergence_factor=0.92,
    ),
    OptimizerType.SOPHIA: OptimizerProfile(
        optimizer_type=OptimizerType.SOPHIA,
        name="Sophia",
        description="Second-order optimizer using Hessian estimates",
        bytes_per_param=8.0,  # Similar to Adam + Hessian
        memory_multiplier=1.0,
        compute_overhead=1.5,
        convergence_factor=1.2,  # Can converge faster
    ),
}


# Optimizer benchmarks database
OPTIMIZER_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # AdamW Benchmarks (Baseline)
    # =========================================================================
    "adamw_llama_7b_full": {
        "name": "AdamW LLaMA-7B Full Precision",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.ADAMW,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 4,
        "seq_length": 2048,
        "memory_per_gpu_gb": 72.0,
        "optimizer_memory_gb": 56.0,  # 7B * 8 bytes
        "source": "Measured baseline",
        "notes": "Full precision AdamW baseline",
    },
    "adamw_llama_13b_full": {
        "name": "AdamW LLaMA-13B Full Precision",
        "model_name": "LLaMA-13B",
        "model_params_b": 13.0,
        "optimizer": OptimizerType.ADAMW,
        "training_type": TrainingType.SFT,
        "num_gpus": 2,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 4,
        "seq_length": 2048,
        "memory_per_gpu_gb": 75.0,
        "optimizer_memory_gb": 52.0,  # Sharded across 2 GPUs
        "parallelism": {"zero_stage": 2},
        "source": "Measured baseline",
    },
    # =========================================================================
    # 8-bit AdamW Benchmarks
    # =========================================================================
    "adamw_8bit_llama_7b": {
        "name": "8-bit AdamW LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.ADAMW_8BIT,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 8,
        "seq_length": 2048,
        "memory_per_gpu_gb": 28.0,
        "optimizer_memory_gb": 14.0,  # 7B * 2 bytes
        "source": "bitsandbytes benchmarks",
        "notes": "75% optimizer memory reduction",
    },
    "adamw_8bit_llama_13b": {
        "name": "8-bit AdamW LLaMA-13B",
        "model_name": "LLaMA-13B",
        "model_params_b": 13.0,
        "optimizer": OptimizerType.ADAMW_8BIT,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 4,
        "seq_length": 2048,
        "memory_per_gpu_gb": 48.0,
        "optimizer_memory_gb": 26.0,
        "source": "bitsandbytes benchmarks",
    },
    "adamw_8bit_llama_70b": {
        "name": "8-bit AdamW LLaMA-70B ZeRO-3",
        "model_name": "LLaMA-70B",
        "model_params_b": 70.0,
        "optimizer": OptimizerType.ADAMW_8BIT,
        "training_type": TrainingType.SFT,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 8,
        "seq_length": 2048,
        "memory_per_gpu_gb": 70.0,
        "optimizer_memory_gb": 17.5,  # 70B * 2 / 8 GPUs
        "parallelism": {"zero_stage": 3},
        "source": "DeepSpeed + bitsandbytes",
    },
    # =========================================================================
    # GaLore Benchmarks
    # =========================================================================
    "galore_llama_7b": {
        "name": "GaLore LLaMA-7B rank=128",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.GALORE,
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 128,
        "seq_length": 256,
        "memory_per_gpu_gb": 22.0,
        "optimizer_memory_gb": 14.0,
        "source": "GaLore paper",
        "source_url": "https://arxiv.org/abs/2403.03507",
        "notes": "Rank 128, 4x memory reduction",
    },
    "galore_8bit_llama_7b": {
        "name": "GaLore 8-bit LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.GALORE_8BIT,
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 256,
        "seq_length": 256,
        "memory_per_gpu_gb": 15.0,
        "optimizer_memory_gb": 7.0,
        "source": "GaLore paper",
        "notes": "8-bit with bitsandbytes",
    },
    "galore_llama_1b": {
        "name": "GaLore LLaMA-1B Single GPU",
        "model_name": "LLaMA-1B",
        "model_params_b": 1.0,
        "optimizer": OptimizerType.GALORE,
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_40GB,
        "batch_size": 256,
        "seq_length": 256,
        "memory_per_gpu_gb": 6.8,
        "optimizer_memory_gb": 2.0,
        "source": "GaLore GitHub",
    },
    # =========================================================================
    # APOLLO Benchmarks
    # =========================================================================
    "apollo_llama_7b": {
        "name": "APOLLO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.APOLLO,
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 128,
        "seq_length": 256,
        "memory_per_gpu_gb": 18.0,
        "optimizer_memory_gb": 14.0,
        "source": "APOLLO repository",
        "notes": "Rank-1 approximation",
    },
    # =========================================================================
    # Lion Benchmarks
    # =========================================================================
    "lion_llama_7b": {
        "name": "Lion LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.LION,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 8,
        "seq_length": 2048,
        "memory_per_gpu_gb": 42.0,
        "optimizer_memory_gb": 28.0,  # 7B * 4 bytes
        "source": "Lion paper benchmarks",
        "notes": "50% optimizer memory vs AdamW",
    },
    # =========================================================================
    # Adafactor Benchmarks
    # =========================================================================
    "adafactor_llama_7b": {
        "name": "Adafactor LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.ADAFACTOR,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 8,
        "seq_length": 2048,
        "memory_per_gpu_gb": 44.0,
        "optimizer_memory_gb": 28.0,
        "source": "Adafactor benchmarks",
    },
    "adafactor_t5_11b": {
        "name": "Adafactor T5-11B",
        "model_name": "T5-11B",
        "model_params_b": 11.0,
        "optimizer": OptimizerType.ADAFACTOR,
        "training_type": TrainingType.SFT,
        "num_gpus": 8,
        "hardware_type": HardwareType.TPU_V4,
        "batch_size": 128,
        "seq_length": 512,
        "memory_per_gpu_gb": 28.0,
        "source": "T5 paper",
        "notes": "Original Adafactor use case",
    },
    # =========================================================================
    # LOMO Benchmarks
    # =========================================================================
    "lomo_llama_65b": {
        "name": "LOMO LLaMA-65B Single GPU",
        "model_name": "LLaMA-65B",
        "model_params_b": 65.0,
        "optimizer": OptimizerType.LOMO,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 1,
        "seq_length": 512,
        "memory_per_gpu_gb": 78.0,
        "optimizer_memory_gb": 0.0,  # No optimizer states
        "source": "LOMO paper",
        "source_url": "https://arxiv.org/abs/2306.09782",
        "notes": "Full 65B on single GPU with gradient accumulation",
    },
    "adalomo_llama_7b": {
        "name": "AdaLOMO LLaMA-7B",
        "model_name": "LLaMA-7B",
        "model_params_b": 7.0,
        "optimizer": OptimizerType.ADALOMO,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 4,
        "seq_length": 2048,
        "memory_per_gpu_gb": 24.0,
        "optimizer_memory_gb": 7.0,
        "source": "AdaLOMO paper",
        "notes": "Adaptive learning rate with minimal state",
    },
    # =========================================================================
    # BAdam Benchmarks
    # =========================================================================
    "badam_llama_70b": {
        "name": "BAdam LLaMA-70B",
        "model_name": "LLaMA-70B",
        "model_params_b": 70.0,
        "optimizer": OptimizerType.BADAM,
        "training_type": TrainingType.SFT,
        "num_gpus": 1,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 1,
        "seq_length": 2048,
        "memory_per_gpu_gb": 75.0,
        "optimizer_memory_gb": 2.8,  # ~4% of full AdamW
        "source": "BAdam paper",
        "notes": "Block-wise training, 0.03-0.05x memory",
    },
}


def get_optimizer_profile(optimizer: OptimizerType) -> OptimizerProfile:
    """Get profile for an optimizer."""
    if optimizer not in OPTIMIZER_PROFILES:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    return OPTIMIZER_PROFILES[optimizer]


def estimate_optimizer_memory(
    params_b: float,
    optimizer: OptimizerType,
    precision_bytes: float = 2.0,  # BF16
) -> float:
    """
    Estimate optimizer state memory in GB.

    Args:
        params_b: Model parameters in billions
        optimizer: Optimizer type
        precision_bytes: Bytes per parameter for model

    Returns:
        Estimated optimizer memory in GB
    """
    profile = get_optimizer_profile(optimizer)
    return params_b * profile.bytes_per_param


def compare_optimizer_memory(
    params_b: float,
    optimizers: Optional[List[OptimizerType]] = None,
) -> Dict[str, float]:
    """
    Compare memory usage across optimizers.

    Args:
        params_b: Model parameters in billions
        optimizers: List of optimizers to compare (default: all)

    Returns:
        Dictionary mapping optimizer name to memory (GB)
    """
    if optimizers is None:
        optimizers = list(OPTIMIZER_PROFILES.keys())

    results = {}
    for opt in optimizers:
        profile = get_optimizer_profile(opt)
        memory = estimate_optimizer_memory(params_b, opt)
        results[profile.name] = memory

    return results


def get_optimizer_benchmarks() -> List[ExtendedBenchmark]:
    """Get all optimizer benchmarks as ExtendedBenchmark objects."""
    benchmarks = []

    for bench_id, config in OPTIMIZER_BENCHMARKS.items():
        # Create parallelism config
        p_config = config.get("parallelism", {})
        parallelism = ParallelismConfig(
            tensor_parallel=p_config.get("tp", 1),
            pipeline_parallel=p_config.get("pp", 1),
            data_parallel=p_config.get("dp", config.get("num_gpus", 1)),
            zero_stage=p_config.get("zero_stage", 0),
        )

        # Create metrics
        metrics = ReportedMetrics(
            memory_per_gpu_gb=config.get("memory_per_gpu_gb"),
        )

        # Create provenance
        provenance = SourceProvenance(
            source_type=SourceType.GITHUB_REPO if "repository" in config.get("source", "").lower() else SourceType.ACADEMIC_PAPER,
            source_url=config.get("source_url"),
            source_title=config.get("source", ""),
            extraction_method="manual",
            extraction_date=datetime.now().strftime("%Y-%m-%d"),
        )

        benchmark = ExtendedBenchmark(
            benchmark_id=bench_id,
            name=config["name"],
            provenance=provenance,
            model_name=config["model_name"],
            model_params_b=config["model_params_b"],
            training_type=config.get("training_type", TrainingType.SFT),
            num_gpus=config.get("num_gpus", 1),
            hardware_type=config.get("hardware_type", HardwareType.A100_80GB),
            batch_size=config.get("batch_size", 1),
            seq_length=config.get("seq_length", 2048),
            parallelism=parallelism,
            optimizer=config.get("optimizer", OptimizerType.ADAMW),
            optimizer_details={
                "optimizer_memory_gb": config.get("optimizer_memory_gb"),
            },
            metrics=metrics,
            verification_status=VerificationStatus.UNVERIFIED,
            notes=config.get("notes", ""),
            tags=["optimizer", config.get("optimizer", OptimizerType.ADAMW).value],
        )

        benchmarks.append(benchmark)

    return benchmarks


def validate_optimizer_memory(
    benchmark: ExtendedBenchmark,
    tolerance: float = 0.2,
) -> Dict[str, Any]:
    """
    Validate optimizer memory against expected values.

    Args:
        benchmark: Benchmark to validate
        tolerance: Acceptable relative error

    Returns:
        Validation result with expected vs actual
    """
    profile = get_optimizer_profile(benchmark.optimizer)
    expected_memory = estimate_optimizer_memory(
        benchmark.model_params_b,
        benchmark.optimizer,
    )

    actual_memory = benchmark.optimizer_details.get("optimizer_memory_gb") if benchmark.optimizer_details else None

    if actual_memory is None:
        return {
            "valid": None,
            "expected_gb": expected_memory,
            "actual_gb": None,
            "error": "No actual memory reported",
        }

    # Adjust for sharding
    sharding_factor = max(
        benchmark.parallelism.data_parallel if benchmark.parallelism.zero_stage >= 1 else 1,
        1,
    )
    expected_per_gpu = expected_memory / sharding_factor

    relative_error = abs(actual_memory - expected_per_gpu) / expected_per_gpu if expected_per_gpu > 0 else 0

    return {
        "valid": relative_error <= tolerance,
        "expected_gb": expected_per_gpu,
        "actual_gb": actual_memory,
        "relative_error": relative_error,
        "sharding_factor": sharding_factor,
    }
