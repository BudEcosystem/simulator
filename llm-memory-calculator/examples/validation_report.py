#!/usr/bin/env python3
"""
Validation Report Generator for Auto-Parallelism Modules.

This script runs validation tests and generates a comprehensive report
on the accuracy and optimality of the auto-parallelism projections.

Usage:
    python examples/validation_report.py
    python examples/validation_report.py --quick    # Quick validation
    python examples/validation_report.py --full     # Full validation
"""

import sys
import os
import argparse
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import time

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ValidationResult:
    """Result from a validation test."""
    name: str
    passed: bool
    message: str
    value: Any = None
    expected: Any = None
    duration_ms: float = 0.0


class ValidationReport:
    """Generates validation reports for auto-parallelism modules."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)

    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)

    def run_validation(self, name: str, test_func) -> ValidationResult:
        """Run a validation test and record result."""
        start = time.time()
        try:
            passed, message, value, expected = test_func()
            result = ValidationResult(
                name=name,
                passed=passed,
                message=message,
                value=value,
                expected=expected,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                message=f"Exception: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )

        self.add_result(result)
        status = "PASS" if result.passed else "FAIL"
        self.log(f"  [{status}] {name}: {result.message}")
        return result

    def generate_summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 70,
            "VALIDATION REPORT SUMMARY",
            "=" * 70,
            "",
            f"Total Tests: {len(self.results)}",
            f"Passed: {sum(1 for r in self.results if r.passed)}",
            f"Failed: {sum(1 for r in self.results if not r.passed)}",
            "",
        ]

        # Group by category
        categories = {}
        for r in self.results:
            category = r.name.split(':')[0] if ':' in r.name else 'General'
            if category not in categories:
                categories[category] = []
            categories[category].append(r)

        for category, tests in categories.items():
            passed = sum(1 for r in tests if r.passed)
            total = len(tests)
            lines.append(f"{category}: {passed}/{total} passed")

        lines.extend([
            "",
            "-" * 70,
            "DETAILED RESULTS",
            "-" * 70,
        ])

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}")
            lines.append(f"       {r.message}")
            if r.value is not None and r.expected is not None:
                lines.append(f"       Value: {r.value}, Expected: {r.expected}")

        lines.append("=" * 70)
        return "\n".join(lines)


def run_quick_validation() -> ValidationReport:
    """Run quick validation tests."""
    print("Running quick validation...")
    print("-" * 70)

    report = ValidationReport()

    # Import modules
    try:
        from llm_memory_calculator.training import (
            estimate_training_time,
            find_optimal_scale,
            select_optimal_nodes,
            auto_configure_training,
            NodeSpec,
        )
        from llm_memory_calculator.genz.LLM_training import (
            get_best_training_parallelization,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        return report

    # Test 1: Training Time Estimation
    def test_training_time():
        estimate = estimate_training_time(
            model='meta-llama/meta-llama-3.1-8b',
            dataset_tokens=1_000_000_000,
            num_gpus=8,
        )
        passed = estimate.total_hours > 0 and estimate.cost_estimate_usd > 0
        return (
            passed,
            f"Time: {estimate.total_hours:.1f}h, Cost: ${estimate.cost_estimate_usd:.0f}",
            estimate.total_hours,
            "> 0",
        )

    report.run_validation("TrainingTime: Basic estimation", test_training_time)

    # Test 2: MFU Range
    def test_mfu_range():
        estimate = estimate_training_time(
            model='meta-llama/meta-llama-3.1-8b',
            dataset_tokens=1_000_000_000,
            num_gpus=8,
        )
        passed = 0.0 < estimate.mfu <= 1.0
        return (
            passed,
            f"MFU: {estimate.mfu:.1%}",
            estimate.mfu,
            "(0, 1]",
        )

    report.run_validation("TrainingTime: MFU in valid range", test_mfu_range)

    # Test 3: Scale Optimizer
    def test_scale_optimizer():
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )
        tp_pp_dp = result.tensor_parallel * result.pipeline_parallel * result.data_parallel
        passed = tp_pp_dp == result.optimal_num_gpus
        return (
            passed,
            f"GPUs: {result.optimal_num_gpus}, TP×PP×DP={tp_pp_dp}",
            tp_pp_dp,
            result.optimal_num_gpus,
        )

    report.run_validation("ScaleOptimizer: TP×PP×DP = total", test_scale_optimizer)

    # Test 4: Cost efficiency
    def test_cost_efficiency():
        result = find_optimal_scale(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
            optimization_target='cost_efficiency',
        )
        passed = result.cost_per_million_tokens > 0
        return (
            passed,
            f"${result.cost_per_million_tokens:.4f}/Mtok",
            result.cost_per_million_tokens,
            "> 0",
        )

    report.run_validation("ScaleOptimizer: Cost efficiency positive", test_cost_efficiency)

    # Test 5: Node Selector
    def test_node_selector():
        nodes = [
            NodeSpec(node_id='h100-1', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
            NodeSpec(node_id='h100-2', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
        ]
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=nodes,
            max_nodes=2,
        )
        passed = result.total_gpus == 16
        return (
            passed,
            f"Selected {result.total_nodes} nodes, {result.total_gpus} GPUs",
            result.total_gpus,
            16,
        )

    report.run_validation("NodeSelector: Select 2 nodes", test_node_selector)

    # Test 6: Auto Configure
    def test_auto_configure():
        plan = auto_configure_training(
            model='meta-llama/meta-llama-3.1-8b',
            dataset_tokens=1_000_000_000,
            max_gpus=8,
            min_gpus=8,
        )
        passed = plan.total_gpus == 8 and plan.throughput_tokens_per_sec > 0
        return (
            passed,
            f"{plan.total_gpus} GPUs, {plan.throughput_tokens_per_sec:,.0f} tok/s",
            plan.total_gpus,
            8,
        )

    report.run_validation("AutoConfig: Complete plan", test_auto_configure)

    # Test 7: Parallelism valid
    def test_parallelism_valid():
        config, result = get_best_training_parallelization(
            model='meta-llama/meta-llama-3.1-8b',
            total_gpus=8,
            batch_size=4,
            seq_length=4096,
            system_name='H100_GPU',
        )
        total = config.tensor_parallel * config.pipeline_parallel * config.data_parallel
        passed = total == 8
        return (
            passed,
            f"TP={config.tensor_parallel}, PP={config.pipeline_parallel}, DP={config.data_parallel}",
            total,
            8,
        )

    report.run_validation("Parallelism: Valid configuration", test_parallelism_valid)

    return report


def run_full_validation() -> ValidationReport:
    """Run full validation tests (more comprehensive)."""
    print("Running full validation...")
    print("-" * 70)

    report = ValidationReport()

    # Import modules
    try:
        from llm_memory_calculator.training import (
            estimate_training_time,
            find_optimal_scale,
            find_scaling_frontier,
            analyze_scaling_efficiency,
            select_optimal_nodes,
            auto_configure_training,
            NodeSpec,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        return report

    # Quick tests first
    quick_report = run_quick_validation()
    report.results.extend(quick_report.results)

    print("\nRunning additional full validation tests...")
    print("-" * 70)

    # Additional tests for full validation

    # Test: Scaling frontier no dominated points
    def test_pareto_frontier():
        frontier = find_scaling_frontier(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=16,
        )

        if len(frontier) < 2:
            return (True, f"Only {len(frontier)} points, no dominance check needed", len(frontier), ">= 1")

        # Check no dominated points
        for i, point in enumerate(frontier):
            for j, other in enumerate(frontier):
                if i == j:
                    continue
                dominates = (
                    other.throughput_tokens_per_sec >= point.throughput_tokens_per_sec and
                    other.cost_per_million_tokens <= point.cost_per_million_tokens and
                    (other.throughput_tokens_per_sec > point.throughput_tokens_per_sec or
                     other.cost_per_million_tokens < point.cost_per_million_tokens)
                )
                if dominates:
                    return (False, f"Point {i} dominated by point {j}", i, "not dominated")

        return (True, f"{len(frontier)} Pareto-optimal points found", len(frontier), ">= 1")

    report.run_validation("ParetoFrontier: No dominated points", test_pareto_frontier)

    # Test: Scaling efficiency analysis
    def test_scaling_efficiency():
        analysis = analyze_scaling_efficiency(
            model='meta-llama/meta-llama-3.1-8b',
            hardware_type='H100',
            gpu_counts=[1, 2, 4, 8],
        )

        if not analysis['success']:
            return (False, "Analysis failed", None, "success")

        # Check efficiency values are reasonable
        for point in analysis['scaling_data']:
            if not (0 < point['efficiency'] <= 1.5):
                return (
                    False,
                    f"Efficiency {point['efficiency']:.2f} out of range",
                    point['efficiency'],
                    "(0, 1.5]",
                )

        return (True, f"{len(analysis['scaling_data'])} scale points analyzed", len(analysis['scaling_data']), "> 0")

    report.run_validation("ScalingEfficiency: Values in range", test_scaling_efficiency)

    # Test: Network efficiency for same-rack nodes
    def test_network_efficiency():
        nodes = [
            NodeSpec(node_id='n1', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
            NodeSpec(node_id='n2', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
        ]
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=nodes,
            max_nodes=2,
        )

        passed = result.same_rack_fraction == 1.0
        return (
            passed,
            f"Same-rack fraction: {result.same_rack_fraction:.1%}",
            result.same_rack_fraction,
            1.0,
        )

    report.run_validation("NodeSelector: Same-rack efficiency", test_network_efficiency)

    # Test: Multi-GPU config validation
    for num_gpus in [1, 2, 4, 8, 16]:
        def test_gpu_count(n=num_gpus):
            def _test():
                result = find_optimal_scale(
                    model='meta-llama/meta-llama-3.1-8b',
                    hardware_type='H100',
                    min_gpus=n,
                    max_gpus=n,
                )
                passed = result.optimal_num_gpus == n
                return (
                    passed,
                    f"Optimal: {result.optimal_num_gpus} GPUs, MFU: {result.mfu:.1%}",
                    result.optimal_num_gpus,
                    n,
                )
            return _test

        report.run_validation(f"MultiGPU: {num_gpus} GPU configuration", test_gpu_count(num_gpus))

    return report


def main():
    parser = argparse.ArgumentParser(description='Auto-Parallelism Validation Report')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--full', action='store_true', help='Run full validation')
    args = parser.parse_args()

    print("=" * 70)
    print("AUTO-PARALLELISM VALIDATION REPORT")
    print("=" * 70)

    if args.quick:
        report = run_quick_validation()
    elif args.full:
        report = run_full_validation()
    else:
        # Default: run quick
        report = run_quick_validation()

    print("\n")
    print(report.generate_summary())

    # Exit with error code if any tests failed
    failed = sum(1 for r in report.results if not r.passed)
    if failed > 0:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
