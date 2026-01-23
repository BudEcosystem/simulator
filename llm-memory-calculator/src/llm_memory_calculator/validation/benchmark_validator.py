"""
Benchmark Validation Rules and Quality Checks.

This module provides comprehensive validation for training benchmarks,
including plausibility checks, consistency verification, and quality scoring.

Key Validation Rules:
1. MFU bounds: 0.10 < MFU < 0.75
2. Memory plausibility: reported <= device capacity
3. Parallelism check: TP × PP × DP × EP <= num_devices
4. Consistency: tokens/sec aligns with step_time and batch_size
5. Scaling laws: larger models should have lower MFU
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .benchmark_schema import (
    ExtendedBenchmark,
    GPUScale,
    HardwareType,
    ModelSizeCategory,
    OptimizerType,
    QuantizationMethod,
    TrainingType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"  # Data is invalid/impossible
    WARNING = "warning"  # Data is suspicious but possible
    INFO = "info"  # Informational observation


@dataclass
class ValidationIssue:
    """A single validation issue found in a benchmark."""
    field: str
    severity: ValidationSeverity
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "severity": self.severity.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "rule": self.rule,
        }


@dataclass
class ValidationResult:
    """Result of benchmark validation."""
    benchmark_id: str
    benchmark_name: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    validation_score: float = 1.0  # 0-1, 1 = perfect

    @property
    def has_errors(self) -> bool:
        return self.errors_count > 0

    @property
    def has_warnings(self) -> bool:
        return self.warnings_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
            "validation_score": self.validation_score,
        }


# Hardware memory limits (GB)
HARDWARE_MEMORY_LIMITS: Dict[HardwareType, float] = {
    HardwareType.A100_40GB: 40.0,
    HardwareType.A100_80GB: 80.0,
    HardwareType.H100_SXM: 80.0,
    HardwareType.H100_PCIE: 80.0,
    HardwareType.H200: 141.0,
    HardwareType.GH200: 480.0,
    HardwareType.B100: 192.0,
    HardwareType.B200: 192.0,
    HardwareType.GB200: 192.0,
    HardwareType.MI250X: 128.0,
    HardwareType.MI300X: 192.0,
    HardwareType.MI300A: 128.0,
    HardwareType.MI325X: 256.0,
    HardwareType.GAUDI2: 96.0,
    HardwareType.GAUDI3: 128.0,
    HardwareType.TPU_V4: 32.0,
    HardwareType.TPU_V5E: 16.0,
    HardwareType.TPU_V5P: 95.0,
    HardwareType.TPU_V6: 128.0,
}

# Hardware peak TFLOPS (BF16)
HARDWARE_PEAK_TFLOPS: Dict[HardwareType, float] = {
    HardwareType.A100_40GB: 312.0,
    HardwareType.A100_80GB: 312.0,
    HardwareType.H100_SXM: 1979.0,  # With FP8 transformer engine
    HardwareType.H100_PCIE: 1513.0,
    HardwareType.H200: 1979.0,
    HardwareType.GH200: 1979.0,
    HardwareType.B100: 3500.0,
    HardwareType.B200: 4500.0,
    HardwareType.GB200: 4500.0,
    HardwareType.MI250X: 383.0,
    HardwareType.MI300X: 1307.0,
    HardwareType.GAUDI2: 432.0,
    HardwareType.GAUDI3: 864.0,
    HardwareType.TPU_V4: 275.0,
    HardwareType.TPU_V5E: 197.0,
    HardwareType.TPU_V5P: 459.0,
}

# Expected MFU ranges by training type and scale
MFU_EXPECTED_RANGES: Dict[TrainingType, Tuple[float, float]] = {
    TrainingType.PRETRAINING: (0.35, 0.65),
    TrainingType.SFT: (0.30, 0.60),
    TrainingType.FULL_FINETUNE: (0.30, 0.60),
    TrainingType.LORA: (0.25, 0.55),
    TrainingType.QLORA: (0.15, 0.45),
    TrainingType.DPO: (0.20, 0.50),
    TrainingType.ORPO: (0.25, 0.55),
    TrainingType.PPO: (0.15, 0.45),
    TrainingType.GRPO: (0.15, 0.45),
    TrainingType.KTO: (0.20, 0.50),
    TrainingType.DISTILLATION: (0.25, 0.55),
}

# Memory bytes per parameter for different optimizers
OPTIMIZER_BYTES_PER_PARAM: Dict[OptimizerType, float] = {
    OptimizerType.ADAMW: 8.0,  # 4+4 for momentum and variance
    OptimizerType.ADAMW_8BIT: 2.0,
    OptimizerType.ADAM: 8.0,
    OptimizerType.ADAM_8BIT: 2.0,
    OptimizerType.SGD: 0.0,  # No state
    OptimizerType.SGD_MOMENTUM: 4.0,
    OptimizerType.LION: 4.0,  # Single momentum
    OptimizerType.ADAFACTOR: 4.0,  # Factorized
    OptimizerType.GALORE: 2.0,  # Low-rank
    OptimizerType.GALORE_8BIT: 1.0,
    OptimizerType.APOLLO: 2.0,
    OptimizerType.ADAM_MINI: 4.0,
    OptimizerType.MUON: 4.0,
    OptimizerType.LOMO: 0.0,  # Fused optimizer
}


class BenchmarkValidator:
    """
    Validates training benchmarks against plausibility rules.

    Performs the following validation checks:
    - MFU bounds validation
    - Memory plausibility
    - Parallelism consistency
    - Throughput consistency
    - Scaling law adherence
    - Configuration completeness
    """

    def __init__(
        self,
        strict_mode: bool = False,
        mfu_lower_bound: float = 0.10,
        mfu_upper_bound: float = 0.75,
    ):
        """
        Initialize validator.

        Args:
            strict_mode: If True, warnings become errors
            mfu_lower_bound: Minimum expected MFU
            mfu_upper_bound: Maximum expected MFU
        """
        self.strict_mode = strict_mode
        self.mfu_lower_bound = mfu_lower_bound
        self.mfu_upper_bound = mfu_upper_bound

    def validate(self, benchmark: ExtendedBenchmark) -> ValidationResult:
        """
        Validate a single benchmark.

        Args:
            benchmark: Benchmark to validate

        Returns:
            ValidationResult with all issues found
        """
        issues: List[ValidationIssue] = []

        # Run all validation checks
        issues.extend(self._validate_mfu(benchmark))
        issues.extend(self._validate_memory(benchmark))
        issues.extend(self._validate_parallelism(benchmark))
        issues.extend(self._validate_throughput_consistency(benchmark))
        issues.extend(self._validate_batch_size(benchmark))
        issues.extend(self._validate_model_config(benchmark))
        issues.extend(self._validate_multi_model(benchmark))

        # Count issues by severity
        errors = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

        # Calculate validation score
        # Errors reduce score by 0.2, warnings by 0.05
        score = max(0.0, 1.0 - (errors * 0.2) - (warnings * 0.05))

        # In strict mode, any warning is treated as an error
        is_valid = errors == 0 and (not self.strict_mode or warnings == 0)

        return ValidationResult(
            benchmark_id=benchmark.benchmark_id,
            benchmark_name=benchmark.name,
            is_valid=is_valid,
            issues=issues,
            warnings_count=warnings,
            errors_count=errors,
            validation_score=score,
        )

    def validate_batch(
        self,
        benchmarks: List[ExtendedBenchmark],
    ) -> List[ValidationResult]:
        """Validate a batch of benchmarks."""
        return [self.validate(b) for b in benchmarks]

    def _validate_mfu(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate MFU is within expected bounds."""
        issues = []

        if benchmark.metrics.mfu is not None:
            mfu = benchmark.metrics.mfu

            # Check absolute bounds
            if mfu < self.mfu_lower_bound:
                issues.append(ValidationIssue(
                    field="metrics.mfu",
                    severity=ValidationSeverity.WARNING,
                    message=f"MFU {mfu:.2%} is below typical minimum {self.mfu_lower_bound:.2%}",
                    expected=f">= {self.mfu_lower_bound:.2%}",
                    actual=f"{mfu:.2%}",
                    rule="mfu_lower_bound",
                ))
            elif mfu > self.mfu_upper_bound:
                issues.append(ValidationIssue(
                    field="metrics.mfu",
                    severity=ValidationSeverity.ERROR,
                    message=f"MFU {mfu:.2%} exceeds maximum possible {self.mfu_upper_bound:.2%}",
                    expected=f"<= {self.mfu_upper_bound:.2%}",
                    actual=f"{mfu:.2%}",
                    rule="mfu_upper_bound",
                ))

            # Check training-type specific bounds
            expected_range = MFU_EXPECTED_RANGES.get(benchmark.training_type)
            if expected_range:
                low, high = expected_range
                if mfu < low * 0.5:  # Allow 50% below expected low
                    issues.append(ValidationIssue(
                        field="metrics.mfu",
                        severity=ValidationSeverity.WARNING,
                        message=f"MFU {mfu:.2%} is unusually low for {benchmark.training_type.value}",
                        expected=f"{low:.2%} - {high:.2%}",
                        actual=f"{mfu:.2%}",
                        rule="mfu_training_type_range",
                    ))

            # MoE models typically have lower MFU
            if benchmark.is_moe and mfu > 0.50:
                issues.append(ValidationIssue(
                    field="metrics.mfu",
                    severity=ValidationSeverity.WARNING,
                    message=f"MoE model with MFU {mfu:.2%} is unusually high",
                    expected="<= 50% for MoE",
                    actual=f"{mfu:.2%}",
                    rule="mfu_moe_ceiling",
                ))

            # Large-scale training typically has lower MFU
            if benchmark.num_gpus > 1000 and mfu > 0.55:
                issues.append(ValidationIssue(
                    field="metrics.mfu",
                    severity=ValidationSeverity.WARNING,
                    message=f"Large-scale ({benchmark.num_gpus} GPUs) with MFU {mfu:.2%} is unusually high",
                    expected="<= 55% for 1000+ GPUs",
                    actual=f"{mfu:.2%}",
                    rule="mfu_scale_ceiling",
                ))

        return issues

    def _validate_memory(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate memory usage is plausible."""
        issues = []

        # Get hardware memory limit
        hw_memory = HARDWARE_MEMORY_LIMITS.get(
            benchmark.hardware_type,
            benchmark.gpu_memory_gb
        )

        # Check reported memory
        if benchmark.metrics.memory_per_gpu_gb is not None:
            if benchmark.metrics.memory_per_gpu_gb > hw_memory:
                issues.append(ValidationIssue(
                    field="metrics.memory_per_gpu_gb",
                    severity=ValidationSeverity.ERROR,
                    message=f"Memory {benchmark.metrics.memory_per_gpu_gb:.1f}GB exceeds hardware capacity {hw_memory:.1f}GB",
                    expected=f"<= {hw_memory:.1f}GB",
                    actual=f"{benchmark.metrics.memory_per_gpu_gb:.1f}GB",
                    rule="memory_capacity",
                ))
            elif benchmark.metrics.memory_per_gpu_gb > hw_memory * 0.99:
                issues.append(ValidationIssue(
                    field="metrics.memory_per_gpu_gb",
                    severity=ValidationSeverity.WARNING,
                    message=f"Memory {benchmark.metrics.memory_per_gpu_gb:.1f}GB is at hardware limit",
                    expected=f"< {hw_memory * 0.99:.1f}GB typical",
                    actual=f"{benchmark.metrics.memory_per_gpu_gb:.1f}GB",
                    rule="memory_headroom",
                ))

        # Estimate minimum memory for model
        bytes_per_param = 2.0  # BF16
        if benchmark.quantization.model_precision == QuantizationMethod.NF4:
            bytes_per_param = 0.5
        elif benchmark.quantization.model_precision in (QuantizationMethod.INT8, QuantizationMethod.FP8_E4M3):
            bytes_per_param = 1.0

        min_model_memory_gb = benchmark.model_params_b * bytes_per_param

        # For full training, add gradients and optimizer states
        if benchmark.peft is None or benchmark.peft.method == "full":
            optimizer_bytes = OPTIMIZER_BYTES_PER_PARAM.get(benchmark.optimizer, 8.0)
            min_training_memory_gb = min_model_memory_gb + benchmark.model_params_b * (2.0 + optimizer_bytes)  # grads + opt

            # Adjust for ZeRO sharding
            sharding_factor = 1
            if benchmark.parallelism.zero_stage == 1:
                sharding_factor = benchmark.parallelism.data_parallel
            elif benchmark.parallelism.zero_stage >= 2:
                sharding_factor = benchmark.parallelism.data_parallel
            elif benchmark.parallelism.fsdp_enabled:
                sharding_factor = benchmark.num_gpus

            min_per_gpu = min_training_memory_gb / sharding_factor

            # Check if model could even fit
            if min_per_gpu > hw_memory * 1.5 and benchmark.parallelism.zero_stage < 3:
                issues.append(ValidationIssue(
                    field="model_params_b",
                    severity=ValidationSeverity.WARNING,
                    message=f"Model {benchmark.model_params_b}B may not fit with reported config",
                    expected=f"Min ~{min_per_gpu:.1f}GB per GPU (estimated)",
                    actual=f"{hw_memory:.1f}GB available",
                    rule="memory_model_fit",
                ))

        return issues

    def _validate_parallelism(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate parallelism configuration is consistent."""
        issues = []

        total_parallel = benchmark.parallelism.total_parallelism

        # Check total parallelism doesn't exceed GPU count
        if total_parallel > benchmark.num_gpus:
            issues.append(ValidationIssue(
                field="parallelism",
                severity=ValidationSeverity.ERROR,
                message=f"Total parallelism {total_parallel} exceeds num_gpus {benchmark.num_gpus}",
                expected=f"TP×PP×DP×EP <= {benchmark.num_gpus}",
                actual=f"{total_parallel}",
                rule="parallelism_total",
            ))

        # Check parallelism factors are powers of 2 (typical requirement)
        for name, value in [
            ("tensor_parallel", benchmark.parallelism.tensor_parallel),
            ("pipeline_parallel", benchmark.parallelism.pipeline_parallel),
            ("expert_parallel", benchmark.parallelism.expert_parallel),
        ]:
            if value > 1 and (value & (value - 1)) != 0:
                issues.append(ValidationIssue(
                    field=f"parallelism.{name}",
                    severity=ValidationSeverity.WARNING,
                    message=f"{name}={value} is not a power of 2",
                    expected="Power of 2",
                    actual=str(value),
                    rule="parallelism_power_of_2",
                ))

        # TP is typically <= 8 (within a node)
        if benchmark.parallelism.tensor_parallel > 8:
            issues.append(ValidationIssue(
                field="parallelism.tensor_parallel",
                severity=ValidationSeverity.WARNING,
                message=f"TP={benchmark.parallelism.tensor_parallel} exceeds typical node size",
                expected="<= 8",
                actual=str(benchmark.parallelism.tensor_parallel),
                rule="parallelism_tp_node",
            ))

        # MoE should have EP > 1 or routing would be inefficient
        if benchmark.is_moe and benchmark.num_experts > 8 and benchmark.parallelism.expert_parallel == 1:
            issues.append(ValidationIssue(
                field="parallelism.expert_parallel",
                severity=ValidationSeverity.WARNING,
                message=f"MoE with {benchmark.num_experts} experts but EP=1",
                expected="EP > 1 for large MoE",
                actual="EP=1",
                rule="parallelism_moe_ep",
            ))

        return issues

    def _validate_throughput_consistency(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate throughput metrics are internally consistent."""
        issues = []

        # Check tokens/sec vs step_time consistency
        if benchmark.metrics.tokens_per_second and benchmark.metrics.step_time_ms:
            # tokens/step = batch_size * seq_length
            tokens_per_step = benchmark.batch_size * benchmark.seq_length
            # Expected tokens/sec from step time
            expected_tps = tokens_per_step / (benchmark.metrics.step_time_ms / 1000.0)

            ratio = benchmark.metrics.tokens_per_second / expected_tps
            if ratio < 0.5 or ratio > 2.0:
                issues.append(ValidationIssue(
                    field="metrics.tokens_per_second",
                    severity=ValidationSeverity.WARNING,
                    message=f"tokens/sec inconsistent with step_time",
                    expected=f"~{expected_tps:.0f} from step_time",
                    actual=f"{benchmark.metrics.tokens_per_second:.0f}",
                    rule="throughput_consistency",
                ))

        # Check tokens/sec per GPU is plausible
        if benchmark.metrics.tokens_per_second_per_gpu:
            # Very rough bound: 100K tokens/sec/GPU is very high
            if benchmark.metrics.tokens_per_second_per_gpu > 200000:
                issues.append(ValidationIssue(
                    field="metrics.tokens_per_second_per_gpu",
                    severity=ValidationSeverity.WARNING,
                    message=f"Per-GPU throughput {benchmark.metrics.tokens_per_second_per_gpu:.0f} seems very high",
                    expected="< 100,000 typical",
                    actual=f"{benchmark.metrics.tokens_per_second_per_gpu:.0f}",
                    rule="throughput_per_gpu_bound",
                ))

        return issues

    def _validate_batch_size(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate batch size configuration."""
        issues = []

        # Check micro_batch * grad_accum * dp = global_batch
        if benchmark.micro_batch_size > 0 and benchmark.gradient_accumulation_steps > 0:
            expected_global = (
                benchmark.micro_batch_size *
                benchmark.gradient_accumulation_steps *
                benchmark.parallelism.data_parallel
            )

            if expected_global != benchmark.batch_size and benchmark.batch_size > 0:
                # Allow some tolerance for padding
                ratio = expected_global / benchmark.batch_size if benchmark.batch_size else 0
                if ratio < 0.9 or ratio > 1.1:
                    issues.append(ValidationIssue(
                        field="batch_size",
                        severity=ValidationSeverity.WARNING,
                        message=f"Global batch size inconsistent with micro_batch × grad_accum × DP",
                        expected=f"{expected_global}",
                        actual=f"{benchmark.batch_size}",
                        rule="batch_size_consistency",
                    ))

        # Sequence length should be power of 2 or typical values
        typical_seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        if benchmark.seq_length not in typical_seq_lengths:
            # Check if it's close to a typical value
            is_close = any(abs(benchmark.seq_length - t) < t * 0.1 for t in typical_seq_lengths)
            if not is_close:
                issues.append(ValidationIssue(
                    field="seq_length",
                    severity=ValidationSeverity.INFO,
                    message=f"Unusual sequence length {benchmark.seq_length}",
                    expected="Typical: 2048, 4096, 8192, etc.",
                    actual=str(benchmark.seq_length),
                    rule="seq_length_typical",
                ))

        return issues

    def _validate_model_config(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate model configuration."""
        issues = []

        # Check model size is positive
        if benchmark.model_params_b <= 0:
            issues.append(ValidationIssue(
                field="model_params_b",
                severity=ValidationSeverity.ERROR,
                message="Model parameters must be positive",
                expected="> 0",
                actual=str(benchmark.model_params_b),
                rule="model_size_positive",
            ))

        # MoE validation
        if benchmark.is_moe:
            if benchmark.num_experts < 2:
                issues.append(ValidationIssue(
                    field="num_experts",
                    severity=ValidationSeverity.ERROR,
                    message="MoE must have at least 2 experts",
                    expected=">= 2",
                    actual=str(benchmark.num_experts),
                    rule="moe_min_experts",
                ))

            if benchmark.active_experts > benchmark.num_experts:
                issues.append(ValidationIssue(
                    field="active_experts",
                    severity=ValidationSeverity.ERROR,
                    message="Active experts cannot exceed total experts",
                    expected=f"<= {benchmark.num_experts}",
                    actual=str(benchmark.active_experts),
                    rule="moe_active_experts",
                ))

        return issues

    def _validate_multi_model(self, benchmark: ExtendedBenchmark) -> List[ValidationIssue]:
        """Validate multi-model training configuration."""
        issues = []

        if benchmark.multi_model is None:
            return issues

        mm = benchmark.multi_model

        # PPO should have all 4 models
        if benchmark.training_type == TrainingType.PPO:
            if not (mm.has_policy_model and mm.has_reference_model and mm.has_reward_model):
                issues.append(ValidationIssue(
                    field="multi_model",
                    severity=ValidationSeverity.WARNING,
                    message="PPO typically requires policy, reference, and reward models",
                    expected="All 3+ models",
                    actual=f"policy={mm.has_policy_model}, ref={mm.has_reference_model}, reward={mm.has_reward_model}",
                    rule="ppo_models",
                ))

        # DPO should have reference model (unless ORPO/SimPO)
        if benchmark.training_type == TrainingType.DPO:
            if not mm.has_reference_model:
                # This is OK for ORPO/SimPO, just note it
                issues.append(ValidationIssue(
                    field="multi_model.has_reference_model",
                    severity=ValidationSeverity.INFO,
                    message="DPO without reference model (ORPO/SimPO variant)",
                    expected="Reference model or reference-free variant",
                    actual="No reference model",
                    rule="dpo_reference",
                ))

        # Validate memory multiplier is reasonable
        multiplier = mm.get_memory_multiplier()
        if multiplier < 1.0:
            issues.append(ValidationIssue(
                field="multi_model",
                severity=ValidationSeverity.ERROR,
                message=f"Multi-model memory multiplier {multiplier:.2f} cannot be < 1.0",
                expected=">= 1.0",
                actual=f"{multiplier:.2f}",
                rule="multi_model_memory_multiplier",
            ))
        elif multiplier > 4.0:
            issues.append(ValidationIssue(
                field="multi_model",
                severity=ValidationSeverity.WARNING,
                message=f"Multi-model memory multiplier {multiplier:.2f} is unusually high",
                expected="<= 3.5 typical",
                actual=f"{multiplier:.2f}",
                rule="multi_model_memory_high",
            ))

        return issues


def validate_benchmark_plausibility(
    benchmark: ExtendedBenchmark,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Quick validation check for benchmark plausibility.

    Args:
        benchmark: Benchmark to validate
        strict: Treat warnings as errors

    Returns:
        Tuple of (is_valid, list of issue messages)
    """
    validator = BenchmarkValidator(strict_mode=strict)
    result = validator.validate(benchmark)
    messages = [f"[{i.severity.value.upper()}] {i.field}: {i.message}" for i in result.issues]
    return result.is_valid, messages


def validate_benchmarks_batch(
    benchmarks: List[ExtendedBenchmark],
    strict: bool = False,
) -> Dict[str, ValidationResult]:
    """
    Validate a batch of benchmarks.

    Args:
        benchmarks: List of benchmarks to validate
        strict: Treat warnings as errors

    Returns:
        Dictionary mapping benchmark_id to ValidationResult
    """
    validator = BenchmarkValidator(strict_mode=strict)
    results = {}
    for benchmark in benchmarks:
        result = validator.validate(benchmark)
        results[benchmark.benchmark_id] = result
    return results


def filter_valid_benchmarks(
    benchmarks: List[ExtendedBenchmark],
    min_score: float = 0.8,
) -> List[ExtendedBenchmark]:
    """
    Filter benchmarks to only those passing validation.

    Args:
        benchmarks: List of benchmarks
        min_score: Minimum validation score (0-1)

    Returns:
        List of valid benchmarks
    """
    validator = BenchmarkValidator()
    valid = []
    for benchmark in benchmarks:
        result = validator.validate(benchmark)
        if result.is_valid and result.validation_score >= min_score:
            valid.append(benchmark)
    return valid
