"""
Cluster Ranker for Training Workloads.

This module ranks clusters by throughput, ETA, and cost with optimal batch size
determination for each cluster:
- Binary search for max feasible batch size
- Optimal batch size selection (throughput knee)
- Multi-metric ranking (throughput, ETA, cost, composite)
- LlamaFactory config generation for top results

Integrates with:
- GenZ training_modeling for performance simulation
- TCO calculator for cost estimation
- LlamaFactory config builder
"""

from typing import Dict, Any, List, Optional, Union
import math
import time

from .cluster_ranking_types import (
    ClusterRankingResult,
    RankingMetric,
    format_eta,
)
from .cluster_optimizer_types import (
    ClusterDefinition,
    TrainingJobSpec,
    ParallelismStrategy,
    TCOBreakdown,
)
from .tco_calculator import calculate_tco
from .llamafactory_config_builder import build_llamafactory_config, build_deepspeed_config
from .hardware_catalog import GPU_SPECS, get_gpu_spec

# Import GenZ for simulation
try:
    from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
    from ..genz.LLM_training.training_parallelization import get_training_parallelization_options
    from ..genz.Models import get_configs
    HAS_GENZ = True
except ImportError:
    HAS_GENZ = False


# GPU memory mapping
GPU_MEMORY_GB: Dict[str, float] = {
    "v100_16gb": 16, "v100_32gb": 32,
    "a100_40gb": 40, "a100_80gb": 80,
    "l40s": 48, "rtx_4090": 24,
    "h100_sxm": 80, "h100_pcie": 80, "h100": 80,
    "h200": 141,
    "b100": 192, "b200": 192,
    "mi300x": 192,
}

# System name mapping
GPU_TO_SYSTEM_NAME: Dict[str, str] = {
    "v100_16gb": "V100_16GB_GPU", "v100_32gb": "V100_32GB_GPU",
    "a100_40gb": "A100_40GB_GPU", "a100_80gb": "A100_80GB_GPU",
    "l40s": "L40S_48GB_GPU",
    "h100_sxm": "H100_GPU", "h100_pcie": "H100_GPU", "h100": "H100_GPU",
    "h200": "H200_GPU",
    "b100": "B100_GPU", "b200": "B200_GPU",
    "mi300x": "MI300X_GPU",
}


def _get_gpu_memory(gpu_type: str) -> float:
    """Get GPU memory in GB."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[normalized]
    if gpu_type in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[gpu_type]
    try:
        spec = get_gpu_spec(gpu_type)
        return spec.memory_gb
    except (ValueError, KeyError):
        return 80.0


def _get_system_name(gpu_type: str) -> str:
    """Map GPU type to GenZ system name."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[normalized]
    if gpu_type in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[gpu_type]
    return gpu_type


def _find_max_feasible_batch_size(
    model: str,
    training_type: str,
    cluster: ClusterDefinition,
    seq_length: int,
    method: str,
    optimizer: str,
    precision: str,
    parallelism: ParallelismStrategy,
    gpu_memory: float,
    max_batch: int = 64,
) -> int:
    """Binary search for maximum feasible batch size."""
    if not HAS_GENZ:
        return 4  # Default fallback

    system_name = _get_system_name(cluster.gpu_type)
    low, high = 1, max_batch
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            result = training_modeling(
                model=model,
                training_stage=training_type,
                batch_size=mid,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=cluster.num_gpus,
                tensor_parallel=parallelism.tensor_parallel,
                data_parallel=parallelism.data_parallel,
                pipeline_parallel=parallelism.pipeline_parallel,
                method=method,
                optimizer=optimizer,
                zero_stage=parallelism.zero_stage,
                gradient_checkpointing=parallelism.gradient_checkpointing,
                bits=precision,
            )

            if result.memory_per_gpu_gb <= gpu_memory * 0.95:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        except Exception:
            high = mid - 1

    return best


def _find_optimal_batch_size(
    model: str,
    training_type: str,
    cluster: ClusterDefinition,
    seq_length: int,
    method: str,
    optimizer: str,
    precision: str,
    parallelism: ParallelismStrategy,
    max_feasible: int,
) -> tuple:
    """Find optimal batch size (knee of throughput curve)."""
    if not HAS_GENZ:
        return 4, None

    system_name = _get_system_name(cluster.gpu_type)

    # Test batch sizes: powers of 2 up to max_feasible
    batch_sizes = []
    b = 1
    while b <= max_feasible:
        batch_sizes.append(b)
        b *= 2
    if max_feasible not in batch_sizes:
        batch_sizes.append(max_feasible)

    results = []
    for batch in batch_sizes:
        try:
            result = training_modeling(
                model=model,
                training_stage=training_type,
                batch_size=batch,
                seq_length=seq_length,
                system_name=system_name,
                num_gpus=cluster.num_gpus,
                tensor_parallel=parallelism.tensor_parallel,
                data_parallel=parallelism.data_parallel,
                pipeline_parallel=parallelism.pipeline_parallel,
                method=method,
                optimizer=optimizer,
                zero_stage=parallelism.zero_stage,
                gradient_checkpointing=parallelism.gradient_checkpointing,
                bits=precision,
            )
            results.append((batch, result))
        except Exception:
            continue

    if not results:
        return 4, None

    # Find knee of throughput curve (diminishing returns)
    # Use efficiency metric: throughput / memory
    best_batch = results[0][0]
    best_result = results[0][1]
    best_efficiency = 0

    for batch, result in results:
        efficiency = result.tokens_per_second / max(1, result.memory_per_gpu_gb)
        tps_per_batch = result.tokens_per_second / batch

        # Prefer higher throughput with reasonable efficiency
        # Penalize very large batches with diminishing returns
        score = result.tokens_per_second * (efficiency / (efficiency + 100))

        if score > best_efficiency:
            best_efficiency = score
            best_batch = batch
            best_result = result

    return best_batch, best_result


def rank_clusters_for_training(
    model: Union[str, Dict[str, Any]],
    training_type: str,
    clusters: List[ClusterDefinition],
    dataset_tokens: int,
    avg_sequence_length: int = 4096,
    num_epochs: float = 1.0,
    method: str = "auto",
    optimizer: str = "adamw",
    precision: str = "bf16",
    max_training_hours: Optional[float] = None,
    max_budget_usd: Optional[float] = None,
    sort_by: str = "throughput",
    return_top_k: int = 10,
    debug: bool = False,
) -> List[ClusterRankingResult]:
    """
    Rank clusters by throughput, ETA, and cost with optimal batch size per cluster.

    Algorithm:
    1. For each cluster:
       - Binary search for max feasible batch size (memory constraint)
       - Evaluate batch sizes [1, 2, 4, 8, ... max_feasible]
       - Find optimal batch_size (knee of throughput curve)
       - Run training_modeling() with optimal batch_size
       - Calculate TCO and ETA using calculate_tco()
    2. Sort clusters by requested metric
    3. Assign ranks for each metric (throughput_rank, eta_rank, cost_rank)
    4. Calculate composite score: 0.4*throughput + 0.3*cost + 0.3*eta
    5. Generate LlamaFactory configs for top results

    Args:
        model: Model name (HF ID) or model config dict
        training_type: Training stage (sft, dpo, ppo, kto, rm, grpo, ipo, pt)
        clusters: List of available cluster definitions
        dataset_tokens: Total tokens in dataset
        avg_sequence_length: Average sequence length
        num_epochs: Number of training epochs
        method: Fine-tuning method (auto, full, lora, qlora, dora, pissa, freeze)
        optimizer: Optimizer type
        precision: Training precision (bf16, fp16, fp32)
        max_training_hours: Maximum acceptable training time
        max_budget_usd: Maximum acceptable cost
        sort_by: Sorting metric (throughput, eta, cost, composite)
        return_top_k: Number of top results to return
        debug: Enable debug output

    Returns:
        List of ClusterRankingResult sorted by requested metric
    """
    if not clusters:
        raise ValueError("No clusters provided")

    # Model name for simulation
    model_name = model if isinstance(model, str) else "custom-model"

    # Auto-select method if needed
    if method == "auto":
        method = "lora"  # Default to LoRA

    # Total tokens to process
    total_tokens = int(dataset_tokens * num_epochs)

    results: List[ClusterRankingResult] = []
    start_time = time.time()

    for cluster in clusters:
        if debug:
            print(f"Evaluating cluster: {cluster.name}")

        gpu_memory = _get_gpu_memory(cluster.gpu_type)
        system_name = _get_system_name(cluster.gpu_type)

        # Get parallelism options
        if HAS_GENZ:
            try:
                configs = get_training_parallelization_options(
                    model=model_name if isinstance(model, str) else model,
                    total_gpus=cluster.num_gpus,
                    gpu_memory_gb=gpu_memory,
                    batch_size=4,  # Initial batch size for config search
                    seq_length=avg_sequence_length,
                    method=method,
                    optimizer=optimizer,
                    bits=precision,
                )
                if not configs:
                    continue

                # Use first valid config (sorted by preference)
                config = configs[0]
                parallelism = ParallelismStrategy(
                    tensor_parallel=config.tensor_parallel,
                    pipeline_parallel=config.pipeline_parallel,
                    data_parallel=config.data_parallel,
                    zero_stage=config.zero_stage,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    gradient_checkpointing=config.gradient_checkpointing,
                )
            except Exception as e:
                if debug:
                    print(f"  Failed to get parallelism options: {e}")
                continue
        else:
            # Fallback parallelism
            parallelism = ParallelismStrategy(
                tensor_parallel=1,
                data_parallel=cluster.num_gpus,
                zero_stage=2 if cluster.num_gpus > 1 else 0,
                gradient_checkpointing=True,
            )

        # Find maximum feasible batch size
        max_feasible = _find_max_feasible_batch_size(
            model=model_name if isinstance(model, str) else "llama-3-8b",
            training_type=training_type,
            cluster=cluster,
            seq_length=avg_sequence_length,
            method=method,
            optimizer=optimizer,
            precision=precision,
            parallelism=parallelism,
            gpu_memory=gpu_memory,
        )

        if debug:
            print(f"  Max feasible batch size: {max_feasible}")

        # Find optimal batch size
        optimal_batch, sim_result = _find_optimal_batch_size(
            model=model_name if isinstance(model, str) else "llama-3-8b",
            training_type=training_type,
            cluster=cluster,
            seq_length=avg_sequence_length,
            method=method,
            optimizer=optimizer,
            precision=precision,
            parallelism=parallelism,
            max_feasible=max_feasible,
        )

        if sim_result is None:
            if debug:
                print(f"  Simulation failed, skipping cluster")
            continue

        if debug:
            print(f"  Optimal batch size: {optimal_batch}, TPS: {sim_result.tokens_per_second:.0f}")

        # Calculate training duration
        training_hours = total_tokens / (sim_result.tokens_per_second * 3600) if sim_result.tokens_per_second > 0 else float('inf')

        # Check time constraint
        if max_training_hours and training_hours > max_training_hours:
            if debug:
                print(f"  Exceeds time constraint: {training_hours:.1f}h > {max_training_hours}h")
            continue

        # Calculate TCO
        try:
            tco = calculate_tco(
                gpu_type=cluster.gpu_type,
                num_gpus=cluster.num_gpus,
                training_hours=training_hours,
                dataset_tokens=total_tokens,
            )
        except Exception:
            tco = TCOBreakdown(
                gpu_compute_cost=training_hours * cluster.total_hourly_rate,
                total_cost=training_hours * cluster.total_hourly_rate,
                cost_per_million_tokens=(training_hours * cluster.total_hourly_rate) / (total_tokens / 1e6) if total_tokens > 0 else 0,
            )

        # Check budget constraint
        if max_budget_usd and tco.total_cost > max_budget_usd:
            if debug:
                print(f"  Exceeds budget: ${tco.total_cost:.0f} > ${max_budget_usd:.0f}")
            continue

        # Calculate training steps
        tokens_per_step = (
            optimal_batch *
            avg_sequence_length *
            parallelism.data_parallel *
            parallelism.gradient_accumulation_steps
        )
        training_steps = total_tokens // tokens_per_step if tokens_per_step > 0 else 0

        # Build job spec for LlamaFactory config
        job_spec = TrainingJobSpec(
            model=model_name if isinstance(model, str) else model.get('model_name_or_path', 'custom-model'),
            dataset_tokens=dataset_tokens,
            avg_sequence_length=avg_sequence_length,
            num_epochs=num_epochs,
            training_type=training_type,
            method=method,
            batch_size=optimal_batch,
            optimizer=optimizer,
            precision=precision,
            gradient_accumulation_steps=parallelism.gradient_accumulation_steps,
            gradient_checkpointing=parallelism.gradient_checkpointing,
        )

        # Build LlamaFactory config
        llama_config = build_llamafactory_config(
            job_spec=job_spec,
            parallelism=parallelism,
            cluster=cluster,
        )

        # Build DeepSpeed config if needed
        ds_config = None
        if parallelism.zero_stage > 0 or parallelism.data_parallel > 1:
            ds_config = build_deepspeed_config(
                parallelism=parallelism,
                precision=precision,
                optimizer=optimizer,
            )

        # Create result
        result = ClusterRankingResult(
            cluster=cluster,
            optimal_batch_size=optimal_batch,
            parallelism=parallelism,
            tokens_per_second=sim_result.tokens_per_second,
            step_time_ms=sim_result.step_time_ms,
            mfu=sim_result.model_flops_utilization,
            memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
            estimated_training_hours=training_hours,
            estimated_eta=format_eta(training_hours),
            training_steps=int(training_steps),
            estimated_cost_usd=tco.total_cost,
            cost_per_million_tokens=tco.cost_per_million_tokens,
            tco_breakdown=tco,
            llamafactory_config=llama_config,
            deepspeed_config=ds_config,
        )

        results.append(result)

    if debug:
        elapsed = time.time() - start_time
        print(f"Evaluated {len(clusters)} clusters, {len(results)} valid in {elapsed:.2f}s")

    if not results:
        raise RuntimeError(
            "No valid configurations found. Try larger clusters or memory-efficient methods (LoRA/QLoRA)."
        )

    # Assign ranks for each metric
    # Throughput rank (higher is better)
    results_by_tps = sorted(results, key=lambda x: -x.tokens_per_second)
    for i, r in enumerate(results_by_tps):
        r.throughput_rank = i + 1

    # ETA rank (lower is better)
    results_by_eta = sorted(results, key=lambda x: x.estimated_training_hours)
    for i, r in enumerate(results_by_eta):
        r.eta_rank = i + 1

    # Cost rank (lower is better)
    results_by_cost = sorted(results, key=lambda x: x.estimated_cost_usd)
    for i, r in enumerate(results_by_cost):
        r.cost_rank = i + 1

    # Calculate composite score
    # Normalize ranks to 0-1 (lower rank = better = higher score)
    n = len(results)
    for r in results:
        tps_score = (n - r.throughput_rank + 1) / n
        eta_score = (n - r.eta_rank + 1) / n
        cost_score = (n - r.cost_rank + 1) / n
        r.composite_score = 0.4 * tps_score + 0.3 * eta_score + 0.3 * cost_score

    # Sort by requested metric
    sort_by = sort_by.lower()
    if sort_by == "throughput":
        results.sort(key=lambda x: -x.tokens_per_second)
    elif sort_by == "eta":
        results.sort(key=lambda x: x.estimated_training_hours)
    elif sort_by == "cost":
        results.sort(key=lambda x: x.estimated_cost_usd)
    elif sort_by == "composite":
        results.sort(key=lambda x: -x.composite_score)
    elif sort_by == "cost_per_token":
        results.sort(key=lambda x: x.cost_per_million_tokens)
    elif sort_by == "mfu":
        results.sort(key=lambda x: -x.mfu)
    else:
        results.sort(key=lambda x: -x.tokens_per_second)

    return results[:return_top_k]
