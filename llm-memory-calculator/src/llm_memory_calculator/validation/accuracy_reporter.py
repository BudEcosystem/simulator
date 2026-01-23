"""
Accuracy Report Generation for Training Simulation Validation.

This module generates comprehensive reports on simulation accuracy,
including summary statistics, category breakdowns, and visualizations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import json

from .accuracy_metrics import AccuracyMetrics, run_accuracy_assessment
from .benchmark_database import (
    PUBLISHED_BENCHMARKS,
    list_benchmarks,
    get_mfu_statistics,
    BenchmarkCategory,
    ConfidenceLevel,
)
from .calibration_engine import CalibrationFactors, get_calibrated_factors


@dataclass
class AccuracyReport:
    """Comprehensive accuracy report."""
    # Report metadata
    report_id: str
    generated_at: str
    version: str = "1.0"

    # Summary metrics
    metrics: Optional[AccuracyMetrics] = None
    calibration_factors: Optional[CalibrationFactors] = None

    # Breakdown tables
    category_breakdown: Optional[pd.DataFrame] = None
    hardware_breakdown: Optional[pd.DataFrame] = None
    model_size_breakdown: Optional[pd.DataFrame] = None

    # Detailed results
    benchmark_results: Optional[pd.DataFrame] = None
    outliers: Optional[List[Dict[str, Any]]] = None

    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'metadata': {
                'report_id': self.report_id,
                'generated_at': self.generated_at,
                'version': self.version,
            },
            'summary': self.metrics.to_dict() if self.metrics else None,
            'calibration': self.calibration_factors.to_dict() if self.calibration_factors else None,
            'breakdown': {
                'by_category': self.category_breakdown.to_dict() if self.category_breakdown is not None else None,
                'by_hardware': self.hardware_breakdown.to_dict() if self.hardware_breakdown is not None else None,
                'by_model_size': self.model_size_breakdown.to_dict() if self.model_size_breakdown is not None else None,
            },
            'outliers': self.outliers,
            'recommendations': self.improvement_suggestions,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Header
        lines.append(f"# Simulation Accuracy Report")
        lines.append(f"\n**Report ID:** {self.report_id}")
        lines.append(f"**Generated:** {self.generated_at}")
        lines.append(f"**Version:** {self.version}")

        # Executive Summary
        lines.append(f"\n## Executive Summary")
        if self.metrics:
            production_ready = "Yes" if self.metrics.is_production_ready() else "No"
            lines.append(f"\n| Metric | Value | Target |")
            lines.append(f"|--------|-------|--------|")
            lines.append(f"| Mean Relative Error | {self.metrics.mean_relative_error:.1%} | <15% |")
            lines.append(f"| Pearson Correlation | {self.metrics.pearson_correlation:.3f} | >0.8 |")
            lines.append(f"| Systematic Bias | {self.metrics.systematic_bias:+.1%} | <5% |")
            lines.append(f"| Within Tolerance | {self.metrics.tolerance_rate:.0%} | >85% |")
            lines.append(f"| **Production Ready** | **{production_ready}** | |")

        # Sample Coverage
        if self.metrics:
            lines.append(f"\n## Sample Coverage")
            lines.append(f"\n- **Total Benchmarks:** {self.metrics.n_samples}")
            lines.append(f"- **Within Tolerance:** {self.metrics.n_within_tolerance} ({self.metrics.tolerance_rate:.0%})")

        # Error Distribution
        if self.metrics:
            lines.append(f"\n## Error Distribution")
            lines.append(f"\n| Percentile | Error |")
            lines.append(f"|------------|-------|")
            lines.append(f"| 25th | {self.metrics.percentile_25_error:.1%} |")
            lines.append(f"| Median | {self.metrics.median_relative_error:.1%} |")
            lines.append(f"| 75th | {self.metrics.percentile_75_error:.1%} |")
            lines.append(f"| 90th | {self.metrics.percentile_90_error:.1%} |")
            lines.append(f"| 95th | {self.metrics.percentile_95_error:.1%} |")
            lines.append(f"| Max | {self.metrics.max_error:.1%} |")

        # Category Breakdown
        if self.category_breakdown is not None and len(self.category_breakdown) > 0:
            lines.append(f"\n## Error by Category")
            lines.append(f"\n| Category | Mean Error | Count |")
            lines.append(f"|----------|------------|-------|")
            for _, row in self.category_breakdown.iterrows():
                lines.append(f"| {row['category']} | {row['mean_error']:.1%} | {row['count']} |")

        # Hardware Breakdown
        if self.hardware_breakdown is not None and len(self.hardware_breakdown) > 0:
            lines.append(f"\n## Error by Hardware")
            lines.append(f"\n| Hardware | Mean Error | Count |")
            lines.append(f"|----------|------------|-------|")
            for _, row in self.hardware_breakdown.iterrows():
                lines.append(f"| {row['hardware']} | {row['mean_error']:.1%} | {row['count']} |")

        # Model Size Breakdown
        if self.model_size_breakdown is not None and len(self.model_size_breakdown) > 0:
            lines.append(f"\n## Error by Model Size")
            lines.append(f"\n| Model Size | Mean Error | Count |")
            lines.append(f"|------------|------------|-------|")
            for _, row in self.model_size_breakdown.iterrows():
                lines.append(f"| {row['size_bucket']} | {row['mean_error']:.1%} | {row['count']} |")

        # Outliers
        if self.outliers and len(self.outliers) > 0:
            lines.append(f"\n## High-Error Benchmarks (Outliers)")
            lines.append(f"\n| Benchmark | Predicted | Actual | Error |")
            lines.append(f"|-----------|-----------|--------|-------|")
            for o in self.outliers[:10]:  # Top 10
                lines.append(f"| {o['name']} | {o['predicted_mfu']:.2%} | {o['actual_mfu']:.2%} | {o['relative_error']:.1%} |")

        # Recommendations
        if self.improvement_suggestions:
            lines.append(f"\n## Improvement Recommendations")
            for s in self.improvement_suggestions:
                lines.append(f"\n- {s}")

        # Calibration Factors
        if self.calibration_factors:
            lines.append(f"\n## Calibration Factors Used")
            lines.append(f"\n```json")
            lines.append(self.calibration_factors.to_json())
            lines.append(f"```")

        return "\n".join(lines)


def generate_report(
    use_high_confidence_only: bool = True,
    include_calibration: bool = True,
    tolerance: float = 0.15,
    debug: bool = False,
) -> AccuracyReport:
    """
    Generate a comprehensive accuracy report.

    Args:
        use_high_confidence_only: Only use high-confidence benchmarks
        include_calibration: Include calibration factors in report
        tolerance: Acceptable relative error threshold
        debug: Enable debug output

    Returns:
        AccuracyReport with all metrics and breakdowns
    """
    # Generate report ID
    report_id = f"accuracy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    generated_at = datetime.now().isoformat()

    if debug:
        print(f"Generating accuracy report {report_id}...")

    # Run accuracy assessment
    metrics = run_accuracy_assessment(
        use_high_confidence_only=use_high_confidence_only,
        tolerance=tolerance,
        debug=debug,
    )

    # Get calibration factors if requested
    calibration = get_calibrated_factors() if include_calibration else None

    # Generate category breakdown
    category_data = []
    for category, error in metrics.errors_by_category.items():
        count = sum(1 for i, name in enumerate(metrics.benchmark_names)
                    if PUBLISHED_BENCHMARKS.get(name, None) and
                    PUBLISHED_BENCHMARKS[name].category.value == category)
        category_data.append({
            'category': category,
            'mean_error': error,
            'count': count,
        })
    category_breakdown = pd.DataFrame(category_data).sort_values('mean_error', ascending=False)

    # Generate hardware breakdown
    hardware_data = []
    for hardware, error in metrics.errors_by_hardware.items():
        count = sum(1 for i, name in enumerate(metrics.benchmark_names)
                    if PUBLISHED_BENCHMARKS.get(name, None) and
                    PUBLISHED_BENCHMARKS[name].hardware == hardware)
        hardware_data.append({
            'hardware': hardware,
            'mean_error': error,
            'count': count,
        })
    hardware_breakdown = pd.DataFrame(hardware_data).sort_values('mean_error', ascending=False)

    # Generate model size breakdown
    size_data = []
    for size_bucket, error in metrics.errors_by_model_size.items():
        count = sum(1 for name in metrics.benchmark_names
                    if name in PUBLISHED_BENCHMARKS and
                    _get_size_bucket(PUBLISHED_BENCHMARKS[name].model_params_b) == size_bucket)
        size_data.append({
            'size_bucket': size_bucket,
            'mean_error': error,
            'count': count,
        })
    model_size_breakdown = pd.DataFrame(size_data).sort_values('mean_error', ascending=False)

    # Generate detailed benchmark results
    benchmark_results = pd.DataFrame({
        'benchmark': metrics.benchmark_names,
        'predicted_mfu': metrics.predictions,
        'actual_mfu': metrics.actuals,
        'relative_error': [abs(e) for e in metrics.relative_errors],
        'signed_error': metrics.relative_errors,
    }).sort_values('relative_error', ascending=False)

    # Identify outliers
    outliers = []
    for i, name in enumerate(metrics.benchmark_names):
        error = abs(metrics.relative_errors[i])
        if error > tolerance:
            outliers.append({
                'name': name,
                'predicted_mfu': metrics.predictions[i],
                'actual_mfu': metrics.actuals[i],
                'relative_error': error,
                'signed_error': metrics.relative_errors[i],
            })
    outliers.sort(key=lambda x: -x['relative_error'])

    # Get improvement suggestions
    suggestions = metrics.get_improvement_suggestions()

    return AccuracyReport(
        report_id=report_id,
        generated_at=generated_at,
        metrics=metrics,
        calibration_factors=calibration,
        category_breakdown=category_breakdown,
        hardware_breakdown=hardware_breakdown,
        model_size_breakdown=model_size_breakdown,
        benchmark_results=benchmark_results,
        outliers=outliers,
        improvement_suggestions=suggestions,
    )


def _get_size_bucket(params_b: float) -> str:
    """Get model size bucket."""
    if params_b < 10:
        return "small_<10B"
    elif params_b < 50:
        return "medium_10-50B"
    elif params_b < 100:
        return "large_50-100B"
    else:
        return "xlarge_>100B"


def generate_summary(
    use_high_confidence_only: bool = True,
    debug: bool = False,
) -> str:
    """
    Generate a quick text summary of accuracy.

    Returns a formatted string summary.
    """
    metrics = run_accuracy_assessment(
        use_high_confidence_only=use_high_confidence_only,
        debug=debug,
    )

    lines = [
        "=" * 60,
        "SIMULATION ACCURACY SUMMARY",
        "=" * 60,
        "",
        f"Samples Tested: {metrics.n_samples}",
        "",
        "Primary Metrics:",
        f"  Mean Relative Error:   {metrics.mean_relative_error:>7.1%}  (target: <15%)",
        f"  Pearson Correlation:   {metrics.pearson_correlation:>7.3f}  (target: >0.8)",
        f"  Systematic Bias:       {metrics.systematic_bias:>+7.1%}  (target: <5%)",
        f"  Within Tolerance:      {metrics.tolerance_rate:>7.0%}  (target: >85%)",
        "",
        f"Production Ready: {'YES' if metrics.is_production_ready() else 'NO'}",
        "",
        "Error Percentiles:",
        f"  25th: {metrics.percentile_25_error:>6.1%}",
        f"  50th: {metrics.median_relative_error:>6.1%}",
        f"  75th: {metrics.percentile_75_error:>6.1%}",
        f"  90th: {metrics.percentile_90_error:>6.1%}",
        f"  Max:  {metrics.max_error:>6.1%}",
        "",
    ]

    # Add category breakdown
    if metrics.errors_by_category:
        lines.append("Error by Category:")
        for cat, err in sorted(metrics.errors_by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:.<30} {err:.1%}")
        lines.append("")

    # Add suggestions
    suggestions = metrics.get_improvement_suggestions()
    if suggestions:
        lines.append("Improvement Suggestions:")
        for s in suggestions:
            lines.append(f"  - {s}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_report(report: AccuracyReport, filepath: str, format: str = "markdown"):
    """
    Save report to file.

    Args:
        report: AccuracyReport to save
        filepath: Output file path
        format: Output format ("markdown", "json", or "html")
    """
    if format == "markdown":
        content = report.to_markdown()
    elif format == "json":
        content = report.to_json()
    elif format == "html":
        # Convert markdown to basic HTML
        md = report.to_markdown()
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Accuracy Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
        code {{ background: #f4f4f4; padding: 2px 5px; }}
    </style>
</head>
<body>
<pre>{md}</pre>
</body>
</html>"""
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(filepath, 'w') as f:
        f.write(content)


def compare_reports(
    report1: AccuracyReport,
    report2: AccuracyReport,
) -> Dict[str, Any]:
    """
    Compare two accuracy reports to show improvement.

    Returns comparison statistics.
    """
    if not report1.metrics or not report2.metrics:
        raise ValueError("Both reports must have metrics")

    m1, m2 = report1.metrics, report2.metrics

    def calc_improvement(before: float, after: float, lower_better: bool = True) -> Dict:
        if lower_better:
            pct = (before - after) / before if before > 0 else 0
        else:
            pct = (after - before) / before if before > 0 else 0
        return {
            'before': before,
            'after': after,
            'change': after - before,
            'improvement_pct': pct,
            'improved': pct > 0,
        }

    return {
        'report1_id': report1.report_id,
        'report2_id': report2.report_id,
        'metrics': {
            'mean_relative_error': calc_improvement(m1.mean_relative_error, m2.mean_relative_error),
            'pearson_correlation': calc_improvement(m1.pearson_correlation, m2.pearson_correlation, lower_better=False),
            'systematic_bias': calc_improvement(abs(m1.systematic_bias), abs(m2.systematic_bias)),
            'tolerance_rate': calc_improvement(m1.tolerance_rate, m2.tolerance_rate, lower_better=False),
        },
        'production_ready': {
            'before': m1.is_production_ready(),
            'after': m2.is_production_ready(),
        },
    }
