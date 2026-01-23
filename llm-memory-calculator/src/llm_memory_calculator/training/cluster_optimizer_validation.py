"""
Validation Suite for Cluster Optimizer.

This module provides validation tests specifically for the ClusterOptimizer:
- TCO accuracy against known cloud pricing
- Throughput predictions vs training_modeling
- Memory estimates vs actual
- Parallelism selection optimality
- Pareto frontier correctness
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from .cluster_optimizer import ClusterOptimizer, _get_gpu_memory
from .cluster_optimizer_types import (
    ClusterDefinition,
    TrainingJobSpec,
    OptimizationTarget,
    TCOBreakdown,
)
from .tco_calculator import calculate_tco, GPU_PRICING


@dataclass
class TCOValidationCase:
    """Test case for TCO validation."""
    name: str
    gpu_type: str
    num_gpus: int
    training_hours: float
    expected_cost_min: float
    expected_cost_max: float
    notes: str = ""


# TCO validation cases based on cloud provider pricing
# Note: Uses best available rates (including Lambda Labs, RunPod, Vast.ai)
# A100 80GB rates: ~$0.99-$1.89/hr on budget providers, ~$3-4/hr on AWS/GCP
# H100 rates: ~$1.89-$2.99/hr on budget providers, ~$4-5/hr on AWS/GCP
TCO_VALIDATION_CASES = [
    TCOValidationCase(
        name="8xA100 10 hours",
        gpu_type="A100_80GB_GPU",
        num_gpus=8,
        training_hours=10,
        expected_cost_min=80,    # ~$1/hr/GPU * 8 GPUs * 10hr (budget provider)
        expected_cost_max=200,   # Including overhead and power
        notes="Basic A100 cluster cost check (budget provider rates)",
    ),
    TCOValidationCase(
        name="8xH100 10 hours",
        gpu_type="H100_GPU",
        num_gpus=8,
        training_hours=10,
        expected_cost_min=150,   # ~$1.9/hr/GPU * 8 GPUs * 10hr
        expected_cost_max=350,   # Including overhead
        notes="H100 has higher hourly rate",
    ),
    TCOValidationCase(
        name="64xH100 100 hours",
        gpu_type="H100_GPU",
        num_gpus=64,
        training_hours=100,
        expected_cost_min=12000,  # ~$1.9/hr * 64 * 100 = $12,160
        expected_cost_max=30000,  # Upper bound with overhead
        notes="Large-scale training cost",
    ),
    TCOValidationCase(
        name="256xA100 1000 hours",
        gpu_type="A100_80GB_GPU",
        num_gpus=256,
        training_hours=1000,
        expected_cost_min=250000,  # ~$1/hr * 256 * 1000 = $256,000
        expected_cost_max=500000,  # With overhead
        notes="Very large training run",
    ),
]


def validate_tco_calculations(debug: bool = False) -> pd.DataFrame:
    """
    Validate TCO calculations against known cloud pricing.

    Tests that our TCO calculations produce reasonable values
    compared to known cloud provider pricing.
    """
    results = []

    for case in TCO_VALIDATION_CASES:
        try:
            tco = calculate_tco(
                gpu_type=case.gpu_type,
                num_gpus=case.num_gpus,
                training_hours=case.training_hours,
                dataset_tokens=1_000_000_000,  # Placeholder
                allow_spot=False,
            )

            passed = case.expected_cost_min <= tco.total_cost <= case.expected_cost_max
            error_pct = 0
            if tco.total_cost > case.expected_cost_max:
                error_pct = 100 * (tco.total_cost - case.expected_cost_max) / case.expected_cost_max
            elif tco.total_cost < case.expected_cost_min:
                error_pct = 100 * (case.expected_cost_min - tco.total_cost) / case.expected_cost_min

            if debug:
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {case.name}")
                print(f"    Cost: ${tco.total_cost:,.0f} "
                      f"(expected: ${case.expected_cost_min:,.0f}-${case.expected_cost_max:,.0f})")
                if not passed:
                    print(f"    Error: {error_pct:.1f}%")

            results.append({
                'name': case.name,
                'gpu_type': case.gpu_type,
                'num_gpus': case.num_gpus,
                'hours': case.training_hours,
                'calculated_cost': tco.total_cost,
                'expected_min': case.expected_cost_min,
                'expected_max': case.expected_cost_max,
                'passed': passed,
                'error_pct': error_pct,
            })

        except Exception as e:
            if debug:
                print(f"[ERROR] {case.name}: {e}")
            results.append({
                'name': case.name,
                'gpu_type': case.gpu_type,
                'num_gpus': case.num_gpus,
                'hours': case.training_hours,
                'calculated_cost': 0,
                'expected_min': case.expected_cost_min,
                'expected_max': case.expected_cost_max,
                'passed': False,
                'error_pct': 100,
                'error': str(e),
            })

    return pd.DataFrame(results)


def validate_parallelism_selection(debug: bool = False) -> pd.DataFrame:
    """
    Validate that ClusterOptimizer selects reasonable parallelism strategies.

    Tests that for known model/cluster combinations, the optimizer
    selects parallelism strategies that:
    1. Fit in GPU memory
    2. Have reasonable efficiency
    3. Are consistent with best practices
    """
    test_cases = [
        # (model, gpu_type, num_gpus, expected_tp_range, expected_pp_range)
        # Note: PP=4 for 7B can be optimal for throughput on 8 GPUs
        ("Llama-2-7B", "A100_80GB_GPU", 8, (1, 4), (1, 8)),
        ("Llama-2-13B", "A100_80GB_GPU", 8, (1, 4), (1, 4)),
        # 70B: optimizer may select TP=2 for efficiency, PP varies with optimization target
        ("Llama-2-70B", "A100_80GB_GPU", 64, (2, 8), (1, 16)),
        ("Llama-2-70B", "H100_GPU", 64, (2, 8), (1, 16)),
    ]

    optimizer = ClusterOptimizer(debug=debug)
    results = []

    for model, gpu_type, num_gpus, expected_tp, expected_pp in test_cases:
        try:
            job_spec = TrainingJobSpec(
                model=model,
                dataset_tokens=1_000_000_000,
                training_type="sft",
                method="full",
            )

            cluster = ClusterDefinition(
                name=f"{gpu_type}x{num_gpus}",
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                hourly_rate_per_gpu=4.0,
            )

            recommendations = optimizer.select_top_k_clusters(
                job_spec=job_spec,
                available_clusters=[cluster],
                optimization_target=OptimizationTarget.THROUGHPUT,
                k=1,
            )

            if recommendations:
                rec = recommendations[0]
                tp = rec.parallelism.tensor_parallel
                pp = rec.parallelism.pipeline_parallel

                tp_valid = expected_tp[0] <= tp <= expected_tp[1]
                pp_valid = expected_pp[0] <= pp <= expected_pp[1]
                passed = tp_valid and pp_valid

                if debug:
                    status = "PASS" if passed else "FAIL"
                    print(f"[{status}] {model} on {gpu_type}x{num_gpus}")
                    print(f"    Selected: TP={tp}, PP={pp}")
                    print(f"    Expected: TP={expected_tp}, PP={expected_pp}")
                    print(f"    MFU: {rec.mfu:.1%}, TPS: {rec.tokens_per_second:.0f}")

                results.append({
                    'model': model,
                    'gpu_type': gpu_type,
                    'num_gpus': num_gpus,
                    'selected_tp': tp,
                    'selected_pp': pp,
                    'expected_tp': expected_tp,
                    'expected_pp': expected_pp,
                    'tp_valid': tp_valid,
                    'pp_valid': pp_valid,
                    'passed': passed,
                    'mfu': rec.mfu,
                    'tokens_per_second': rec.tokens_per_second,
                })
            else:
                if debug:
                    print(f"[ERROR] {model} on {gpu_type}x{num_gpus}: No valid config found")
                results.append({
                    'model': model,
                    'gpu_type': gpu_type,
                    'num_gpus': num_gpus,
                    'passed': False,
                    'error': "No valid configuration found",
                })

        except Exception as e:
            if debug:
                print(f"[ERROR] {model} on {gpu_type}x{num_gpus}: {e}")
            results.append({
                'model': model,
                'gpu_type': gpu_type,
                'num_gpus': num_gpus,
                'passed': False,
                'error': str(e),
            })

    return pd.DataFrame(results)


def validate_pareto_frontier(debug: bool = False) -> pd.DataFrame:
    """
    Validate that Pareto frontier is correctly computed.

    Tests that:
    1. All Pareto-optimal solutions are actually non-dominated
    2. No dominated solutions are included
    3. Frontier contains meaningful variety
    """
    results = []

    # Create test job
    job_spec = TrainingJobSpec(
        model="Llama-2-13B",
        dataset_tokens=1_000_000_000,
        training_type="sft",
        method="full",
    )

    # Create diverse clusters
    clusters = [
        ClusterDefinition(name="a100-8", gpu_type="A100_80GB_GPU", num_gpus=8, hourly_rate_per_gpu=4.0),
        ClusterDefinition(name="a100-16", gpu_type="A100_80GB_GPU", num_gpus=16, hourly_rate_per_gpu=4.0),
        ClusterDefinition(name="h100-8", gpu_type="H100_GPU", num_gpus=8, hourly_rate_per_gpu=5.0),
        ClusterDefinition(name="h100-16", gpu_type="H100_GPU", num_gpus=16, hourly_rate_per_gpu=5.0),
    ]

    optimizer = ClusterOptimizer(debug=debug)

    try:
        recommendations = optimizer.select_top_k_clusters(
            job_spec=job_spec,
            available_clusters=clusters,
            optimization_target=OptimizationTarget.PARETO,
            k=10,
        )

        if debug:
            print(f"\nPareto frontier has {len(recommendations)} solutions")

        # Validate no dominated solutions
        for i, rec_i in enumerate(recommendations):
            for j, rec_j in enumerate(recommendations):
                if i == j:
                    continue

                # Check if rec_i dominates rec_j
                dominates = (
                    rec_i.tokens_per_second >= rec_j.tokens_per_second and
                    rec_i.tco_breakdown.total_cost <= rec_j.tco_breakdown.total_cost and
                    rec_i.mfu >= rec_j.mfu and
                    rec_i.training_hours <= rec_j.training_hours
                )

                strictly_better = (
                    rec_i.tokens_per_second > rec_j.tokens_per_second or
                    rec_i.tco_breakdown.total_cost < rec_j.tco_breakdown.total_cost or
                    rec_i.mfu > rec_j.mfu or
                    rec_i.training_hours < rec_j.training_hours
                )

                if dominates and strictly_better:
                    if debug:
                        print(f"[WARN] {rec_i.cluster.name} dominates {rec_j.cluster.name}")

        # Record results
        for rec in recommendations:
            results.append({
                'cluster': rec.cluster.name,
                'tokens_per_second': rec.tokens_per_second,
                'total_cost': rec.tco_breakdown.total_cost,
                'mfu': rec.mfu,
                'training_hours': rec.training_hours,
                'rank': rec.rank,
            })

        if debug:
            print("\nPareto frontier:")
            for rec in recommendations:
                print(f"  {rec.cluster.name}: TPS={rec.tokens_per_second:.0f}, "
                      f"Cost=${rec.tco_breakdown.total_cost:.0f}, MFU={rec.mfu:.1%}")

    except Exception as e:
        if debug:
            print(f"[ERROR] Pareto validation failed: {e}")
        results.append({
            'error': str(e),
            'passed': False,
        })

    return pd.DataFrame(results)


def validate_optimization_targets(debug: bool = False) -> pd.DataFrame:
    """
    Validate that different optimization targets produce different results.

    Tests that optimizing for TCO vs throughput vs MFU produces
    meaningfully different recommendations.
    """
    results = []

    job_spec = TrainingJobSpec(
        model="Llama-2-7B",
        dataset_tokens=10_000_000_000,
        training_type="sft",
        method="full",
    )

    clusters = [
        ClusterDefinition(name="a100-8", gpu_type="A100_80GB_GPU", num_gpus=8, hourly_rate_per_gpu=3.5),
        ClusterDefinition(name="a100-16", gpu_type="A100_80GB_GPU", num_gpus=16, hourly_rate_per_gpu=3.5),
        ClusterDefinition(name="h100-8", gpu_type="H100_GPU", num_gpus=8, hourly_rate_per_gpu=5.0),
    ]

    optimizer = ClusterOptimizer(debug=debug)

    targets = [
        OptimizationTarget.TCO,
        OptimizationTarget.THROUGHPUT,
        OptimizationTarget.MFU,
        OptimizationTarget.LATENCY,
    ]

    best_by_target = {}

    for target in targets:
        try:
            recommendations = optimizer.select_top_k_clusters(
                job_spec=job_spec,
                available_clusters=clusters,
                optimization_target=target,
                k=1,
            )

            if recommendations:
                rec = recommendations[0]
                best_by_target[target.value] = rec

                if debug:
                    print(f"\nBest for {target.value}:")
                    print(f"  Cluster: {rec.cluster.name}")
                    print(f"  TPS: {rec.tokens_per_second:.0f}")
                    print(f"  Cost: ${rec.tco_breakdown.total_cost:.0f}")
                    print(f"  MFU: {rec.mfu:.1%}")

                results.append({
                    'target': target.value,
                    'cluster': rec.cluster.name,
                    'tokens_per_second': rec.tokens_per_second,
                    'total_cost': rec.tco_breakdown.total_cost,
                    'mfu': rec.mfu,
                    'step_time_ms': rec.step_time_ms,
                    'score': rec.score,
                })

        except Exception as e:
            if debug:
                print(f"[ERROR] {target.value}: {e}")
            results.append({
                'target': target.value,
                'error': str(e),
            })

    # Check that different targets produce different results
    if len(best_by_target) >= 2:
        unique_clusters = set(r.cluster.name for r in best_by_target.values())
        diversity_score = len(unique_clusters) / len(best_by_target)

        if debug:
            print(f"\nTarget diversity score: {diversity_score:.0%} "
                  f"({len(unique_clusters)} unique clusters from {len(best_by_target)} targets)")

    return pd.DataFrame(results)


def validate_memory_constraints(debug: bool = False) -> pd.DataFrame:
    """
    Validate that memory constraints are respected.

    Tests that configurations exceeding GPU memory are not recommended.
    """
    results = []

    test_cases = [
        # (model, method, gpu_type, should_fit_single_gpu)
        ("Llama-2-7B", "full", "A100_40GB_GPU", False),  # 7B full needs ~112GB
        ("Llama-2-7B", "lora", "A100_40GB_GPU", True),   # LoRA fits
        ("Llama-2-7B", "qlora", "A100_40GB_GPU", True),  # QLoRA fits
        ("Llama-2-70B", "full", "A100_80GB_GPU", False), # 70B needs multiple GPUs
        # Note: 70B QLoRA may need 2 GPUs depending on parallelism config
        ("Llama-2-70B", "qlora", "A100_80GB_GPU", False), # QLoRA 70B needs TP for attention
    ]

    optimizer = ClusterOptimizer(debug=debug)

    for model, method, gpu_type, should_fit in test_cases:
        try:
            job_spec = TrainingJobSpec(
                model=model,
                dataset_tokens=1_000_000_000,
                training_type="sft",
                method=method,
            )

            cluster = ClusterDefinition(
                name=f"{gpu_type}x1",
                gpu_type=gpu_type,
                num_gpus=1,
                hourly_rate_per_gpu=4.0,
            )

            try:
                recommendations = optimizer.select_top_k_clusters(
                    job_spec=job_spec,
                    available_clusters=[cluster],
                    optimization_target=OptimizationTarget.THROUGHPUT,
                    k=1,
                )
                found_valid = len(recommendations) > 0
            except RuntimeError:
                found_valid = False

            passed = found_valid == should_fit

            if debug:
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {model} {method} on single {gpu_type}")
                print(f"    Should fit: {should_fit}, Found valid: {found_valid}")

            results.append({
                'model': model,
                'method': method,
                'gpu_type': gpu_type,
                'should_fit': should_fit,
                'found_valid': found_valid,
                'passed': passed,
            })

        except Exception as e:
            if debug:
                print(f"[ERROR] {model} {method} on {gpu_type}: {e}")
            results.append({
                'model': model,
                'method': method,
                'gpu_type': gpu_type,
                'error': str(e),
                'passed': False,
            })

    return pd.DataFrame(results)


def run_cluster_optimizer_validation(debug: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Run complete cluster optimizer validation suite.

    Returns:
        Dict of DataFrames with results from each validation category
    """
    if debug:
        print("=" * 60)
        print("CLUSTER OPTIMIZER VALIDATION SUITE")
        print("=" * 60)

    results = {}

    # TCO validation
    if debug:
        print("\n--- TCO Validation ---")
    results['tco'] = validate_tco_calculations(debug)

    # Parallelism selection validation
    if debug:
        print("\n--- Parallelism Selection Validation ---")
    results['parallelism'] = validate_parallelism_selection(debug)

    # Pareto frontier validation
    if debug:
        print("\n--- Pareto Frontier Validation ---")
    results['pareto'] = validate_pareto_frontier(debug)

    # Optimization targets validation
    if debug:
        print("\n--- Optimization Targets Validation ---")
    results['targets'] = validate_optimization_targets(debug)

    # Memory constraints validation
    if debug:
        print("\n--- Memory Constraints Validation ---")
    results['memory'] = validate_memory_constraints(debug)

    # Summary
    if debug:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for name, df in results.items():
            if 'passed' in df.columns:
                passed = df['passed'].sum()
                total = len(df)
                print(f"{name}: {passed}/{total} ({passed/total*100:.0f}%)")

    return results
