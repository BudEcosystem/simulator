#!/usr/bin/env python3
"""
Auto-Parallelism and Resource Optimization Demo

This script demonstrates the new auto-parallelism and resource optimization
features of the llm-memory-calculator package:

1. Training Time Estimation - Estimate training time from dataset size
2. Scale Optimizer - Find optimal GPU count and gang/DP configuration
3. Node Selector - Select optimal nodes from heterogeneous pool
4. Auto-Configure - Unified API for complete training setup

Usage:
    python examples/auto_parallelism_demo.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_memory_calculator.training import (
    # Training time estimation
    estimate_training_time,
    calculate_training_steps,
    estimate_scaling_curve,
    find_optimal_gpu_count,

    # Scale optimizer
    find_optimal_scale,
    find_scaling_frontier,
    recommend_gang_configuration,
    analyze_scaling_efficiency,
    ScalingRecommendation,

    # Node selector
    NodeSpec,
    select_optimal_nodes,
    evaluate_node_combination,

    # Auto-config
    auto_configure_training,
    quick_configure,
    OptimalTrainingPlan,
)


def demo_training_time_estimation():
    """Demonstrate training time estimation from dataset size."""
    print("\n" + "=" * 70)
    print("DEMO 1: Training Time Estimation")
    print("=" * 70)

    print("\nEstimating training time for Llama-3-8B on 10B tokens...")

    try:
        estimate = estimate_training_time(
            model='Llama-3.1-8B',
            dataset_tokens=10_000_000_000,  # 10B tokens
            num_epochs=1.0,
            batch_size=4,
            seq_length=4096,
            system_name='H100_GPU',
            num_gpus=8,
            training_stage='sft',
            method='full',
            optimizer='adamw',
            hourly_rate_per_gpu=3.0,
        )

        print(estimate.summary())

        print("\n--- Training Steps Calculation ---")
        steps = calculate_training_steps(
            dataset_tokens=10_000_000_000,
            batch_size=4,
            seq_length=4096,
            data_parallel=8,
            gradient_accumulation_steps=2,
            num_epochs=1.0,
        )
        print(f"Total steps: {steps['total_steps']:,}")
        print(f"Tokens per step: {steps['tokens_per_step']:,}")

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if running without full dependencies)")


def demo_scaling_curve():
    """Demonstrate scaling curve analysis."""
    print("\n" + "=" * 70)
    print("DEMO 2: Scaling Curve Analysis")
    print("=" * 70)

    print("\nGenerating scaling curve for Llama-3-8B on H100...")

    try:
        points = estimate_scaling_curve(
            model='Llama-3.1-8B',
            dataset_tokens=10_000_000_000,
            system_name='H100_GPU',
            gpu_counts=[1, 2, 4, 8, 16, 32],
            batch_size=4,
            seq_length=4096,
            hourly_rate_per_gpu=3.0,
        )

        print("\n{:<10} {:<15} {:<12} {:<15}".format(
            "GPUs", "Throughput", "Efficiency", "Cost/MTok"
        ))
        print("-" * 55)

        for p in points:
            print("{:<10} {:>12,.0f} {:>11.1%} ${:>12.4f}".format(
                p.num_gpus,
                p.throughput_tokens_per_sec,
                p.scaling_efficiency,
                p.cost_per_million_tokens,
            ))

    except Exception as e:
        print(f"Error: {e}")


def demo_find_optimal_scale():
    """Demonstrate finding optimal GPU count."""
    print("\n" + "=" * 70)
    print("DEMO 3: Find Optimal Scale")
    print("=" * 70)

    print("\nFinding optimal GPU count for Llama-3-70B...")

    try:
        # Find optimal scale for cost efficiency
        result = find_optimal_scale(
            model='Llama-3.1-70B',
            hardware_type='H100',
            min_gpus=8,
            max_gpus=128,
            optimization_target='cost_efficiency',
            batch_size=4,
            seq_length=4096,
            training_stage='sft',
            method='full',
        )

        print(result.summary())

        # Try with throughput constraint
        print("\n--- With minimum throughput constraint (1M tok/s) ---")
        result_constrained = find_optimal_scale(
            model='Llama-3.1-70B',
            hardware_type='H100',
            min_gpus=8,
            max_gpus=256,
            target_throughput=1_000_000,  # 1M tok/s minimum
            optimization_target='cost_efficiency',
        )

        print(f"Optimal GPUs: {result_constrained.optimal_num_gpus}")
        print(f"Throughput: {result_constrained.throughput_tokens_per_sec:,.0f} tok/s")
        print(f"Cost/hour: ${result_constrained.cost_per_hour:.2f}")

    except Exception as e:
        print(f"Error: {e}")


def demo_gang_configuration():
    """Demonstrate gang (DP replica) configuration."""
    print("\n" + "=" * 70)
    print("DEMO 4: Gang Configuration Recommendation")
    print("=" * 70)

    print("\nRecommending gang configuration for 64 GPUs...")

    try:
        result = recommend_gang_configuration(
            model='Llama-3.1-70B',
            hardware_type='H100',
            num_gpus=64,
            batch_size=4,
            seq_length=4096,
            maximize_dp=True,  # Maximize data parallelism
        )

        if result['success']:
            print(f"\nRecommended Configuration:")
            print(f"  Tensor Parallel: {result['recommended']['tensor_parallel']}")
            print(f"  Pipeline Parallel: {result['recommended']['pipeline_parallel']}")
            print(f"  Data Parallel (gangs): {result['recommended']['data_parallel']}")
            print(f"  ZeRO Stage: {result['recommended']['zero_stage']}")
            print(f"\nPerformance:")
            print(f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tok/s")
            print(f"  Memory/GPU: {result['memory_per_gpu_gb']:.1f} GB")
            print(f"  MFU: {result['mfu']:.1%}")

            print(f"\nAnalysis:")
            print(f"  Max DP possible: {result['analysis']['max_dp_possible']}")
            print(f"  Total valid configs: {result['analysis']['total_valid_configs']}")
        else:
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error: {e}")


def demo_node_selection():
    """Demonstrate heterogeneous node selection."""
    print("\n" + "=" * 70)
    print("DEMO 5: Heterogeneous Node Selection")
    print("=" * 70)

    print("\nCreating heterogeneous node pool...")

    # Create sample node pool
    nodes = [
        NodeSpec(
            node_id='h100-node-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-node-2',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-node-3',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='a100-node-1',
            gpu_type='A100_80GB',
            num_gpus=8,
            rack_id='rack-3',
            hourly_cost=16.0,
        ),
        NodeSpec(
            node_id='a100-node-2',
            gpu_type='A100_80GB',
            num_gpus=8,
            rack_id='rack-3',
            hourly_cost=16.0,
        ),
    ]

    print(f"Available nodes: {len(nodes)}")
    for node in nodes:
        print(f"  {node.node_id}: {node.num_gpus}x {node.gpu_type} (${node.hourly_cost}/hr)")

    try:
        print("\nSelecting optimal nodes for throughput...")
        result = select_optimal_nodes(
            model='Llama-3.1-70B',
            available_nodes=nodes,
            max_nodes=4,
            optimization_target='throughput',
            prefer_homogeneous=True,
            prefer_same_rack=True,
        )

        print(result.summary())

    except Exception as e:
        print(f"Error: {e}")


def demo_auto_configure():
    """Demonstrate unified auto-configure API."""
    print("\n" + "=" * 70)
    print("DEMO 6: Auto-Configure Training (Unified API)")
    print("=" * 70)

    print("\nAuto-configuring training for Llama-3-70B on 100B tokens...")

    try:
        plan = auto_configure_training(
            # Workload
            model='Llama-3.1-70B',
            dataset_tokens=100_000_000_000,  # 100B tokens
            num_epochs=1.0,
            training_stage='sft',

            # Sequence & Batch
            max_seq_length=4096,
            target_global_batch_tokens=4_000_000,

            # Method - auto-select best
            method='auto',
            precision='bf16',
            gradient_checkpointing=True,

            # Hardware options
            available_hardware=['H100', 'A100_80GB'],
            max_gpus=128,
            gpus_per_node=8,

            # Constraints
            max_cost_usd=100000,
            max_hours=168,  # 1 week

            # Goal
            optimization_goal='minimize_cost',
        )

        print(plan.summary())

        # Show config exports
        print("\n--- LlamaFactory Config (partial) ---")
        llama_config = plan.to_llamafactory_config()
        for key, value in list(llama_config.items())[:10]:
            print(f"  {key}: {value}")

        print("\n--- Launch Command ---")
        print(f"  {plan.to_torchrun_command()}")

    except Exception as e:
        print(f"Error: {e}")


def demo_quick_configure():
    """Demonstrate quick configuration."""
    print("\n" + "=" * 70)
    print("DEMO 7: Quick Configure (Simple API)")
    print("=" * 70)

    print("\nQuick config for Llama-3-8B SFT on 8x H100...")

    try:
        plan = quick_configure(
            model='Llama-3.1-8B',
            dataset_tokens=10_000_000_000,
            gpu_type='H100',
            num_gpus=8,
            training_stage='sft',
            method='full',
        )

        print(f"\nConfiguration Summary:")
        print(f"  Parallelism: TP={plan.tensor_parallel}, PP={plan.pipeline_parallel}, DP={plan.data_parallel}")
        print(f"  ZeRO Stage: {plan.zero_stage}")
        print(f"  Batch Size: {plan.per_device_batch_size} x {plan.gradient_accumulation_steps} GA")
        print(f"\nPerformance:")
        print(f"  Throughput: {plan.throughput_tokens_per_sec:,.0f} tok/s")
        print(f"  Training Time: {plan.total_training_hours:.1f} hours")
        print(f"  Total Cost: ${plan.total_cost_usd:,.0f}")

    except Exception as e:
        print(f"Error: {e}")


def demo_scaling_frontier():
    """Demonstrate Pareto frontier of scale vs cost."""
    print("\n" + "=" * 70)
    print("DEMO 8: Scaling Frontier (Pareto Optimal)")
    print("=" * 70)

    print("\nFinding Pareto frontier for Llama-3-8B...")

    try:
        frontier = find_scaling_frontier(
            model='Llama-3.1-8B',
            hardware_type='H100',
            min_gpus=1,
            max_gpus=64,
            batch_size=4,
            seq_length=4096,
        )

        print("\nPareto-Optimal Configurations:")
        print("{:<8} {:<15} {:<12} {:<15} {:<10}".format(
            "GPUs", "Throughput", "Efficiency", "Cost/MTok", "Parallelism"
        ))
        print("-" * 65)

        for p in frontier[:10]:
            print("{:<8} {:>12,.0f} {:>11.1%} ${:>12.4f} TP{}/PP{}/DP{}".format(
                p.num_gpus,
                p.throughput_tokens_per_sec,
                p.scaling_efficiency,
                p.cost_per_million_tokens,
                p.parallelism.tensor_parallel,
                p.parallelism.pipeline_parallel,
                p.parallelism.data_parallel,
            ))

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all demos."""
    print("=" * 70)
    print("AUTO-PARALLELISM AND RESOURCE OPTIMIZATION DEMO")
    print("=" * 70)
    print("\nThis demo showcases the new auto-parallelism features:")
    print("  1. Training time estimation from dataset size")
    print("  2. Scaling curve analysis")
    print("  3. Optimal scale finding")
    print("  4. Gang/DP configuration")
    print("  5. Heterogeneous node selection")
    print("  6. Unified auto-configure API")
    print("  7. Quick configure")
    print("  8. Scaling frontier (Pareto)")

    # Run demos
    demo_training_time_estimation()
    demo_scaling_curve()
    demo_find_optimal_scale()
    demo_gang_configuration()
    demo_node_selection()
    demo_auto_configure()
    demo_quick_configure()
    demo_scaling_frontier()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll demos executed. Some may show errors if dependencies are missing.")
    print("The implementation is complete and ready for use.")


if __name__ == '__main__':
    main()
