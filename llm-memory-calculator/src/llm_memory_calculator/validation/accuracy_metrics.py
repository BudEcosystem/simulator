"""
Statistical Accuracy Metrics for Training Simulation Validation.

This module provides comprehensive statistical measures to assess the accuracy
of training simulation predictions against published benchmarks.

Key Metrics:
- Mean Absolute Error (MAE)
- Mean Relative Error (MRE)
- Pearson Correlation Coefficient
- Systematic Bias
- Percentile Error Bounds
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats
import pandas as pd
import warnings

from .benchmark_database import (
    PublishedBenchmark,
    PUBLISHED_BENCHMARKS,
    get_benchmarks_by_category,
    ConfidenceLevel,
    BenchmarkCategory,
)


@dataclass
class AccuracyMetrics:
    """
    Comprehensive accuracy metrics for simulation validation.

    Target values based on industry standards:
    - mean_relative_error: <15% for production-ready simulators
    - pearson_correlation: >0.8 for good prediction accuracy
    - systematic_bias: <5% for unbiased predictions
    """
    # Primary accuracy metrics
    mean_absolute_error: float          # MAE in percentage points
    mean_relative_error: float          # MRE as fraction (target: <0.15)
    median_relative_error: float        # Median RE (more robust than mean)
    pearson_correlation: float          # Linear correlation (target: >0.8)
    spearman_correlation: float         # Rank correlation
    r_squared: float                    # Coefficient of determination

    # Bias analysis
    systematic_bias: float              # Mean signed error (positive = over-predict)
    bias_direction: str                 # "over" or "under" prediction tendency

    # Error distribution
    percentile_25_error: float          # 25th percentile of absolute relative errors
    percentile_75_error: float          # 75th percentile of absolute relative errors
    percentile_90_error: float          # 90th percentile error
    percentile_95_error: float          # Worst-case bound (95th percentile)
    max_error: float                    # Maximum observed error

    # Sample information
    n_samples: int                      # Number of benchmarks used
    n_within_tolerance: int             # Benchmarks within tolerance
    tolerance_rate: float               # Fraction within tolerance

    # Detailed breakdown
    errors_by_category: Dict[str, float] = field(default_factory=dict)
    errors_by_hardware: Dict[str, float] = field(default_factory=dict)
    errors_by_model_size: Dict[str, float] = field(default_factory=dict)

    # Raw data for further analysis
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    relative_errors: List[float] = field(default_factory=list)
    benchmark_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': {
                'mean_absolute_error': self.mean_absolute_error,
                'mean_relative_error': self.mean_relative_error,
                'median_relative_error': self.median_relative_error,
                'pearson_correlation': self.pearson_correlation,
                'spearman_correlation': self.spearman_correlation,
                'r_squared': self.r_squared,
            },
            'bias': {
                'systematic_bias': self.systematic_bias,
                'bias_direction': self.bias_direction,
            },
            'error_distribution': {
                'percentile_25': self.percentile_25_error,
                'percentile_75': self.percentile_75_error,
                'percentile_90': self.percentile_90_error,
                'percentile_95': self.percentile_95_error,
                'max_error': self.max_error,
            },
            'coverage': {
                'n_samples': self.n_samples,
                'n_within_tolerance': self.n_within_tolerance,
                'tolerance_rate': self.tolerance_rate,
            },
            'breakdown': {
                'by_category': self.errors_by_category,
                'by_hardware': self.errors_by_hardware,
                'by_model_size': self.errors_by_model_size,
            },
        }

    def is_production_ready(self) -> bool:
        """Check if accuracy meets production-ready thresholds."""
        return (
            self.mean_relative_error < 0.15 and
            self.pearson_correlation > 0.8 and
            abs(self.systematic_bias) < 0.05 and
            self.tolerance_rate > 0.85
        )

    def get_improvement_suggestions(self) -> List[str]:
        """Suggest areas for improvement based on metrics."""
        suggestions = []

        if self.mean_relative_error >= 0.15:
            suggestions.append(
                f"MRE ({self.mean_relative_error:.1%}) exceeds target (<15%). "
                "Consider calibrating efficiency factors."
            )

        if self.pearson_correlation < 0.8:
            suggestions.append(
                f"Correlation ({self.pearson_correlation:.2f}) below target (>0.8). "
                "Review parallelism and communication overhead models."
            )

        if abs(self.systematic_bias) >= 0.05:
            direction = "over" if self.systematic_bias > 0 else "under"
            suggestions.append(
                f"Systematic {direction}-prediction ({self.systematic_bias:+.1%}). "
                "Adjust base efficiency factor."
            )

        # Check for category-specific issues
        for category, error in self.errors_by_category.items():
            if error > 0.20:  # >20% error in category
                suggestions.append(
                    f"High error in {category} category ({error:.1%}). "
                    f"Review {category}-specific modeling."
                )

        return suggestions


@dataclass
class BenchmarkResult:
    """Result of simulating a single benchmark."""
    benchmark_name: str
    benchmark: PublishedBenchmark
    predicted_mfu: float
    actual_mfu: float
    relative_error: float
    absolute_error: float
    signed_error: float  # positive = over-prediction
    success: bool = True
    error_message: Optional[str] = None


def calculate_accuracy_metrics(
    results: List[BenchmarkResult],
    tolerance: float = 0.15,
) -> AccuracyMetrics:
    """
    Calculate comprehensive accuracy metrics from benchmark results.

    Args:
        results: List of BenchmarkResult from simulations
        tolerance: Acceptable relative error (default 15%)

    Returns:
        AccuracyMetrics with all statistical measures
    """
    # Filter successful results
    successful = [r for r in results if r.success]
    if len(successful) < 2:
        raise ValueError(f"Need at least 2 successful results, got {len(successful)}")

    # Extract arrays
    predictions = np.array([r.predicted_mfu for r in successful])
    actuals = np.array([r.actual_mfu for r in successful])
    relative_errors = np.array([r.relative_error for r in successful])
    signed_errors = np.array([r.signed_error for r in successful])
    absolute_errors = np.abs(relative_errors)

    # Primary metrics
    mae = np.mean(np.abs(predictions - actuals))
    mre = np.mean(absolute_errors)
    median_re = np.median(absolute_errors)

    # Correlation
    pearson_corr, _ = stats.pearsonr(predictions, actuals)
    spearman_corr, _ = stats.spearmanr(predictions, actuals)

    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Bias analysis
    systematic_bias = np.mean(signed_errors)
    bias_direction = "over" if systematic_bias > 0 else "under"

    # Error percentiles
    percentile_25 = np.percentile(absolute_errors, 25)
    percentile_75 = np.percentile(absolute_errors, 75)
    percentile_90 = np.percentile(absolute_errors, 90)
    percentile_95 = np.percentile(absolute_errors, 95)
    max_error = np.max(absolute_errors)

    # Tolerance coverage
    n_within = sum(1 for e in absolute_errors if e <= tolerance)
    tolerance_rate = n_within / len(successful)

    # Category-wise errors
    errors_by_category = _compute_grouped_errors(
        successful, lambda r: r.benchmark.category.value
    )

    # Hardware-wise errors
    errors_by_hardware = _compute_grouped_errors(
        successful, lambda r: r.benchmark.hardware
    )

    # Model size-wise errors
    def get_size_bucket(r: BenchmarkResult) -> str:
        params = r.benchmark.model_params_b
        if params < 10:
            return "small_<10B"
        elif params < 50:
            return "medium_10-50B"
        elif params < 100:
            return "large_50-100B"
        else:
            return "xlarge_>100B"

    errors_by_model_size = _compute_grouped_errors(successful, get_size_bucket)

    return AccuracyMetrics(
        mean_absolute_error=mae,
        mean_relative_error=mre,
        median_relative_error=median_re,
        pearson_correlation=pearson_corr,
        spearman_correlation=spearman_corr,
        r_squared=r_squared,
        systematic_bias=systematic_bias,
        bias_direction=bias_direction,
        percentile_25_error=percentile_25,
        percentile_75_error=percentile_75,
        percentile_90_error=percentile_90,
        percentile_95_error=percentile_95,
        max_error=max_error,
        n_samples=len(successful),
        n_within_tolerance=n_within,
        tolerance_rate=tolerance_rate,
        errors_by_category=errors_by_category,
        errors_by_hardware=errors_by_hardware,
        errors_by_model_size=errors_by_model_size,
        predictions=predictions.tolist(),
        actuals=actuals.tolist(),
        relative_errors=relative_errors.tolist(),
        benchmark_names=[r.benchmark_name for r in successful],
    )


def _compute_grouped_errors(
    results: List[BenchmarkResult],
    group_fn,
) -> Dict[str, float]:
    """Compute mean relative error for each group."""
    groups: Dict[str, List[float]] = {}
    for r in results:
        key = group_fn(r)
        if key not in groups:
            groups[key] = []
        groups[key].append(abs(r.relative_error))
    return {k: np.mean(v) for k, v in groups.items()}


def run_accuracy_assessment(
    benchmarks: Optional[Dict[str, PublishedBenchmark]] = None,
    use_high_confidence_only: bool = True,
    tolerance: float = 0.15,
    debug: bool = False,
) -> AccuracyMetrics:
    """
    Run accuracy assessment against published benchmarks.

    Args:
        benchmarks: Dict of benchmarks to use (default: all or high-confidence)
        use_high_confidence_only: Only use high-confidence benchmarks
        tolerance: Acceptable relative error
        debug: Enable debug output

    Returns:
        AccuracyMetrics with comprehensive accuracy measures
    """
    from ..genz.LLM_training.training_modeling import training_modeling

    # Select benchmarks
    if benchmarks is None:
        if use_high_confidence_only:
            benchmarks = get_benchmarks_by_category(confidence=ConfidenceLevel.HIGH)
        else:
            benchmarks = PUBLISHED_BENCHMARKS

    if debug:
        print(f"Running accuracy assessment on {len(benchmarks)} benchmarks...")

    results: List[BenchmarkResult] = []

    for name, benchmark in benchmarks.items():
        if debug:
            print(f"  Testing {name}...")

        try:
            # Calculate per-GPU batch size
            # Global batch is split across DP ranks only (not TP/PP which are model parallelism)
            # micro_batch_size = global_batch_size / data_parallel
            # Note: TP and PP do NOT reduce batch size - they split the model, not the data
            micro_batch_size = max(1, benchmark.batch_size // max(1, benchmark.data_parallel))

            # Detect training method from benchmark attributes
            # QLoRA benchmarks have precision='nf4' or 'int4' and FINETUNING category
            # LoRA benchmarks have 'lora' in name and FINETUNING category
            method = 'full'
            lora_rank = 16  # Default LoRA rank
            if benchmark.precision.lower() in ('nf4', 'int4', 'fp4'):
                method = 'qlora'
            elif 'lora' in name.lower() and benchmark.category == BenchmarkCategory.FINETUNING:
                method = 'lora'
            elif benchmark.category == BenchmarkCategory.FINETUNING:
                method = 'full'  # Full fine-tuning

            # Run simulation
            result = training_modeling(
                model=benchmark.model,
                # BUG FIX (Phase 5, Bug #7): Correct training stage detection
                # FINETUNING benchmarks use 'sft', PRETRAINING benchmarks use 'pretraining'
                training_stage='sft' if benchmark.category == BenchmarkCategory.FINETUNING else 'pretraining',
                batch_size=micro_batch_size,
                seq_length=benchmark.seq_length,
                system_name=benchmark.hardware,
                num_gpus=benchmark.num_gpus,
                tensor_parallel=benchmark.tensor_parallel,
                pipeline_parallel=benchmark.pipeline_parallel,
                data_parallel=benchmark.data_parallel,
                expert_parallel=benchmark.expert_parallel,
                method=method,
                lora_rank=lora_rank,
                optimizer=benchmark.optimizer,
                zero_stage=benchmark.zero_stage,
                gradient_checkpointing=benchmark.gradient_checkpointing,
                bits=benchmark.precision,
            )

            predicted_mfu = result.model_flops_utilization
            actual_mfu = benchmark.reported_mfu

            # Calculate errors
            absolute_error = abs(predicted_mfu - actual_mfu)
            relative_error = absolute_error / actual_mfu if actual_mfu > 0 else float('inf')
            signed_error = (predicted_mfu - actual_mfu) / actual_mfu if actual_mfu > 0 else 0

            results.append(BenchmarkResult(
                benchmark_name=name,
                benchmark=benchmark,
                predicted_mfu=predicted_mfu,
                actual_mfu=actual_mfu,
                relative_error=relative_error,
                absolute_error=absolute_error,
                signed_error=signed_error,
                success=True,
            ))

            if debug:
                status = "OK" if relative_error <= tolerance else "HIGH ERROR"
                print(f"    Predicted: {predicted_mfu:.2%}, Actual: {actual_mfu:.2%}, "
                      f"Error: {relative_error:.1%} [{status}]")

        except Exception as e:
            if debug:
                print(f"    FAILED: {e}")
            results.append(BenchmarkResult(
                benchmark_name=name,
                benchmark=benchmark,
                predicted_mfu=0.0,
                actual_mfu=benchmark.reported_mfu,
                relative_error=1.0,
                absolute_error=benchmark.reported_mfu,
                signed_error=-1.0,
                success=False,
                error_message=str(e),
            ))

    # Calculate metrics
    metrics = calculate_accuracy_metrics(results, tolerance)

    if debug:
        print(f"\n{'='*60}")
        print("ACCURACY ASSESSMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"Mean Relative Error: {metrics.mean_relative_error:.1%}")
        print(f"Median Relative Error: {metrics.median_relative_error:.1%}")
        print(f"Pearson Correlation: {metrics.pearson_correlation:.3f}")
        print(f"Systematic Bias: {metrics.systematic_bias:+.1%} ({metrics.bias_direction})")
        print(f"Within {tolerance:.0%} Tolerance: {metrics.n_within_tolerance}/{metrics.n_samples} "
              f"({metrics.tolerance_rate:.0%})")
        print(f"\nProduction Ready: {metrics.is_production_ready()}")

        suggestions = metrics.get_improvement_suggestions()
        if suggestions:
            print("\nImprovement Suggestions:")
            for s in suggestions:
                print(f"  - {s}")

    return metrics


def run_quick_accuracy_check(debug: bool = False) -> AccuracyMetrics:
    """
    Run a quick accuracy check on a small set of key benchmarks.

    Useful for fast iteration during development.
    """
    quick_benchmarks = {
        'llama2_70b_meta': PUBLISHED_BENCHMARKS.get('llama2_70b_meta'),
        'llama3_70b_meta': PUBLISHED_BENCHMARKS.get('llama3_70b_meta'),
        'gpt3_175b_megatron': PUBLISHED_BENCHMARKS.get('gpt3_175b_megatron'),
        'llama_7b_a100_baseline': PUBLISHED_BENCHMARKS.get('llama_7b_a100_baseline'),
        'mixtral_8x7b': PUBLISHED_BENCHMARKS.get('mixtral_8x7b'),
    }
    # Filter out None values
    quick_benchmarks = {k: v for k, v in quick_benchmarks.items() if v is not None}

    return run_accuracy_assessment(
        benchmarks=quick_benchmarks,
        use_high_confidence_only=False,
        debug=debug,
    )


def compare_before_after_calibration(
    before_metrics: AccuracyMetrics,
    after_metrics: AccuracyMetrics,
) -> Dict[str, Any]:
    """
    Compare accuracy metrics before and after calibration.

    Returns improvement statistics.
    """
    def improvement(before: float, after: float, lower_better: bool = True) -> Tuple[float, str]:
        """Calculate improvement percentage."""
        if lower_better:
            pct = (before - after) / before if before > 0 else 0
        else:
            pct = (after - before) / before if before > 0 else 0
        direction = "improved" if pct > 0 else "degraded" if pct < 0 else "unchanged"
        return pct, direction

    mre_imp, mre_dir = improvement(
        before_metrics.mean_relative_error,
        after_metrics.mean_relative_error,
    )
    corr_imp, corr_dir = improvement(
        before_metrics.pearson_correlation,
        after_metrics.pearson_correlation,
        lower_better=False,
    )
    bias_imp, bias_dir = improvement(
        abs(before_metrics.systematic_bias),
        abs(after_metrics.systematic_bias),
    )
    tol_imp, tol_dir = improvement(
        before_metrics.tolerance_rate,
        after_metrics.tolerance_rate,
        lower_better=False,
    )

    return {
        'mean_relative_error': {
            'before': before_metrics.mean_relative_error,
            'after': after_metrics.mean_relative_error,
            'improvement': mre_imp,
            'status': mre_dir,
        },
        'pearson_correlation': {
            'before': before_metrics.pearson_correlation,
            'after': after_metrics.pearson_correlation,
            'improvement': corr_imp,
            'status': corr_dir,
        },
        'systematic_bias': {
            'before': before_metrics.systematic_bias,
            'after': after_metrics.systematic_bias,
            'improvement': bias_imp,
            'status': bias_dir,
        },
        'tolerance_rate': {
            'before': before_metrics.tolerance_rate,
            'after': after_metrics.tolerance_rate,
            'improvement': tol_imp,
            'status': tol_dir,
        },
        'production_ready': {
            'before': before_metrics.is_production_ready(),
            'after': after_metrics.is_production_ready(),
        },
    }


def get_outlier_benchmarks(
    metrics: AccuracyMetrics,
    threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Identify benchmarks with high prediction error.

    Returns list of outliers with their details for investigation.
    """
    outliers = []
    for i, name in enumerate(metrics.benchmark_names):
        error = abs(metrics.relative_errors[i])
        if error > threshold:
            outliers.append({
                'name': name,
                'predicted_mfu': metrics.predictions[i],
                'actual_mfu': metrics.actuals[i],
                'relative_error': error,
                'signed_error': metrics.relative_errors[i],
            })

    # Sort by error (highest first)
    outliers.sort(key=lambda x: -x['relative_error'])
    return outliers
