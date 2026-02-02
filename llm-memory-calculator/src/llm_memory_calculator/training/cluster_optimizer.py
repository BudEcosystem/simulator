"""
Cluster Optimizer for Training Workloads.

This module implements two core algorithms for training cluster optimization:

1. **Best Top-K Cluster Selector**: Select optimal clusters from available options
   - Evaluates provided clusters against training job requirements
   - Finds best parallelization strategy for each cluster
   - Ranks by optimization target (TCO, throughput, MFU, cost-per-token)

2. **Optimal Cluster Designer**: Design optimal cluster configuration from scratch
   - Searches across GPU types and counts
   - Finds globally optimal configuration
   - Supports Pareto frontier for multi-objective optimization

Both algorithms integrate with LlamaFactory for training configuration generation.
"""

import itertools
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from .cluster_optimizer_types import (
    ClusterDefinition,
    TrainingJobSpec,
    TCOBreakdown,
    ParallelismStrategy,
    ClusterRecommendationResult,
    OptimalClusterDesignResult,
    OptimizationTarget,
    PricingTier,
)
from .tco_calculator import calculate_tco, get_gpu_pricing, GPU_PRICING
from .llamafactory_config_builder import build_llamafactory_config, build_deepspeed_config
from .hardware_catalog import GPU_SPECS, get_gpu_spec

# Import new ranking and requirements modules
from .cluster_ranking_types import (
    ClusterRankingResult,
    MinimumClusterRequirements,
    ComprehensiveLlamaFactoryConfig,
)
from .cluster_ranker import rank_clusters_for_training
from .cluster_requirements import predict_cluster_requirements
from .comprehensive_config_builder import generate_comprehensive_training_config

# Import GenZ training modeling
from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
from ..genz.LLM_training.training_parallelization import (
    get_training_parallelization_options,
    TrainingParallelismConfig,
)
from ..genz.Models import get_configs

# Make paretoset optional
try:
    from paretoset import paretoset
    HAS_PARETOSET = True
except ImportError:
    HAS_PARETOSET = False


# GPU memory mapping for common GPUs
GPU_MEMORY_GB: Dict[str, float] = {
    # NVIDIA
    "v100_16gb": 16, "v100_32gb": 32,
    "a100_40gb": 40, "a100_80gb": 80,
    "l40s": 48,
    "h100_sxm": 80, "h100_pcie": 80, "h100": 80,
    "h200": 141,
    "b100": 192, "b200": 192,
    # AMD
    "mi300x": 192, "mi300a": 128,
    # Aliases
    "A100_40GB_GPU": 40, "A100_80GB_GPU": 80,
    "H100_GPU": 80, "H100_80GB_GPU": 80, "H100_SXM_GPU": 80, "H100_PCIe_GPU": 80,
    "H200_GPU": 141, "B100_GPU": 192, "B200_GPU": 192,
    "MI300X_GPU": 192,
    "V100_16GB_GPU": 16, "V100_32GB_GPU": 32,
    "L40S_48GB_GPU": 48,
}

# System name mapping (GPU type to GenZ system name)
GPU_TO_SYSTEM_NAME: Dict[str, str] = {
    "v100_16gb": "V100_16GB_GPU", "v100_32gb": "V100_32GB_GPU",
    "a100_40gb": "A100_40GB_GPU", "a100_80gb": "A100_80GB_GPU",
    "l40s": "L40S_48GB_GPU",
    "h100_sxm": "H100_GPU", "h100_pcie": "H100_GPU", "h100": "H100_GPU",
    "h200": "H200_GPU",
    "b100": "B100_GPU", "b200": "B200_GPU",
    "mi300x": "MI300X_GPU",
    # Reverse mappings for convenience
    "H100_GPU": "H100_GPU", "A100_80GB_GPU": "A100_80GB_GPU",
}


def _get_gpu_memory(gpu_type: str) -> float:
    """Get GPU memory in GB."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[normalized]
    if gpu_type in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[gpu_type]
    # Default to 80GB (H100)
    return 80.0


def _get_system_name(gpu_type: str) -> str:
    """Map GPU type to GenZ system name."""
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[normalized]
    if gpu_type in GPU_TO_SYSTEM_NAME:
        return GPU_TO_SYSTEM_NAME[gpu_type]
    # Return as-is, might be a valid system name
    return gpu_type


class ClusterOptimizer:
    """
    Unified cluster selection and optimization.

    Provides two main algorithms:
    1. select_top_k_clusters(): Select best clusters from available options
    2. design_optimal_cluster(): Design optimal cluster from scratch
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the cluster optimizer.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def select_top_k_clusters(
        self,
        job_spec: TrainingJobSpec,
        available_clusters: List[ClusterDefinition],
        optimization_target: OptimizationTarget = OptimizationTarget.TCO,
        k: int = 5,
        allow_spot: bool = True,
        max_configs_per_cluster: int = 20,
    ) -> List[ClusterRecommendationResult]:
        """
        Select top-K clusters from available options.

        Algorithm:
        1. For each cluster, generate valid parallelism configs
        2. Run training_modeling() for each config
        3. Calculate TCO and training duration
        4. Filter by constraints (budget, time, throughput)
        5. Score and rank by optimization target
        6. Return top-K with LlamaFactory configs

        Args:
            job_spec: Training job specification
            available_clusters: List of available cluster definitions
            optimization_target: What to optimize for
            k: Number of top clusters to return
            allow_spot: Allow spot/preemptible instances
            max_configs_per_cluster: Max parallelism configs to evaluate per cluster

        Returns:
            List of top-K ClusterRecommendationResult objects
        """
        if not available_clusters:
            raise ValueError("No clusters provided")

        results = []
        start_time = time.time()

        for cluster in available_clusters:
            cluster_results = self._evaluate_cluster(
                job_spec=job_spec,
                cluster=cluster,
                allow_spot=allow_spot,
                max_configs=max_configs_per_cluster,
            )
            results.extend(cluster_results)

        if self.debug:
            elapsed = time.time() - start_time
            print(f"Evaluated {len(available_clusters)} clusters, {len(results)} configs in {elapsed:.2f}s")

        if not results:
            raise RuntimeError(
                "No valid configurations found. Try larger clusters or memory-efficient methods (LoRA/QLoRA)."
            )

        # Handle Pareto optimization
        if optimization_target == OptimizationTarget.PARETO:
            return self._compute_pareto_frontier(results, k)

        # Score results
        for result in results:
            result.score = self._calculate_score(result, optimization_target)

        # Normalize scores to 0-1 range for fair comparison
        self._normalize_scores(results, optimization_target)

        # Sort by normalized score (higher is better)
        results.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks and return top-k
        for i, r in enumerate(results[:k]):
            r.rank = i + 1

        return results[:k]

    def design_optimal_cluster(
        self,
        job_spec: TrainingJobSpec,
        optimization_target: OptimizationTarget = OptimizationTarget.TCO,
        gpu_types: Optional[List[str]] = None,
        max_gpus: int = 512,
        min_gpus: int = 1,
        allow_spot: bool = True,
    ) -> OptimalClusterDesignResult:
        """
        Design optimal cluster configuration from scratch.

        Algorithm:
        1. Calculate minimum memory requirements
        2. Filter GPU types that can fit workload
        3. Search over GPU types x counts x parallelism configs
        4. Simulate each configuration with training_modeling()
        5. Calculate TCO, filter by constraints
        6. Return optimal based on target (or Pareto frontier)

        Args:
            job_spec: Training job specification
            optimization_target: What to optimize for
            gpu_types: GPU types to consider (default: all available)
            max_gpus: Maximum number of GPUs to consider
            min_gpus: Minimum number of GPUs to consider
            allow_spot: Allow spot/preemptible instances

        Returns:
            OptimalClusterDesignResult with optimal and alternative configurations
        """
        start_time = time.time()

        # Default to all available GPU types
        if gpu_types is None:
            gpu_types = list(GPU_PRICING.keys())

        # Filter GPU types that can fit the workload (search space reduction)
        gpu_types = self._filter_gpu_types_for_model(job_spec, gpu_types, max_gpus)

        if self.debug:
            print(f"Evaluating {len(gpu_types)} GPU types: {gpu_types}")

        all_results = []
        configs_evaluated = 0

        for gpu_type in gpu_types:
            gpu_memory = _get_gpu_memory(gpu_type)

            # Generate GPU count options (powers of 2 and common sizes)
            gpu_counts = self._generate_gpu_counts(min_gpus, max_gpus)

            for num_gpus in gpu_counts:
                # Create virtual cluster
                cluster = ClusterDefinition(
                    name=f"{gpu_type}x{num_gpus}",
                    gpu_type=gpu_type,
                    num_gpus=num_gpus,
                    gpus_per_node=min(8, num_gpus),
                    hourly_rate_per_gpu=self._get_hourly_rate(gpu_type, allow_spot),
                )

                # Evaluate cluster
                results = self._evaluate_cluster(
                    job_spec=job_spec,
                    cluster=cluster,
                    allow_spot=allow_spot,
                    max_configs=10,  # Limit per cluster for design search
                )

                configs_evaluated += len(results)
                all_results.extend(results)

        search_time = time.time() - start_time

        if self.debug:
            print(f"Evaluated {configs_evaluated} configurations in {search_time:.2f}s")

        if not all_results:
            raise RuntimeError(
                f"No valid configurations found. Model may be too large for max_gpus={max_gpus}. "
                "Try increasing max_gpus or using memory-efficient methods (LoRA/QLoRA)."
            )

        # Handle Pareto optimization
        if optimization_target == OptimizationTarget.PARETO:
            pareto = self._compute_pareto_frontier(all_results, k=10)
            return OptimalClusterDesignResult(
                optimal_config=pareto[0] if pareto else all_results[0],
                alternatives=pareto[1:5] if len(pareto) > 1 else [],
                pareto_frontier=pareto,
                configs_evaluated=configs_evaluated,
                search_time_seconds=search_time,
            )

        # Score results
        for result in all_results:
            result.score = self._calculate_score(result, optimization_target)

        # Normalize scores to 0-1 range for fair comparison
        self._normalize_scores(all_results, optimization_target)

        # Sort by normalized score (higher is better)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, r in enumerate(all_results):
            r.rank = i + 1

        return OptimalClusterDesignResult(
            optimal_config=all_results[0],
            alternatives=all_results[1:5],
            pareto_frontier=[],
            configs_evaluated=configs_evaluated,
            search_time_seconds=search_time,
        )

    def _evaluate_cluster(
        self,
        job_spec: TrainingJobSpec,
        cluster: ClusterDefinition,
        allow_spot: bool,
        max_configs: int,
    ) -> List[ClusterRecommendationResult]:
        """Evaluate a cluster for the given job specification."""
        results = []
        gpu_memory = _get_gpu_memory(cluster.gpu_type)
        system_name = _get_system_name(cluster.gpu_type)

        # Get valid parallelism configurations
        try:
            configs = get_training_parallelization_options(
                model=job_spec.model,
                total_gpus=cluster.num_gpus,
                gpu_memory_gb=gpu_memory,
                batch_size=job_spec.batch_size or 4,
                seq_length=job_spec.avg_sequence_length,
                method=job_spec.method,
                optimizer=job_spec.optimizer,
                bits=job_spec.precision,
            )
        except Exception as e:
            if self.debug:
                print(f"Failed to get parallelism options for {cluster.name}: {e}")
            return []

        # Limit configs to evaluate
        if len(configs) > max_configs:
            # Prioritize higher TP and lower ZeRO stage
            configs.sort(key=lambda c: (-c.tensor_parallel, c.zero_stage, c.pipeline_parallel))
            configs = configs[:max_configs]

        for parallelism_config in configs:
            # Early constraint pruning - skip expensive simulation if clearly infeasible
            can_satisfy, rejection_reason = self._can_potentially_satisfy_constraints(
                job_spec, cluster, parallelism_config
            )
            if not can_satisfy:
                if self.debug:
                    print(f"Skipped {cluster.name} (TP={parallelism_config.tensor_parallel}, "
                          f"PP={parallelism_config.pipeline_parallel}): {rejection_reason}")
                continue

            try:
                result = self._simulate_and_build_result(
                    job_spec=job_spec,
                    cluster=cluster,
                    parallelism_config=parallelism_config,
                    system_name=system_name,
                    allow_spot=allow_spot,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                if self.debug:
                    print(f"Simulation failed for {cluster.name}: {e}")
                continue

        return results

    def _simulate_and_build_result(
        self,
        job_spec: TrainingJobSpec,
        cluster: ClusterDefinition,
        parallelism_config: TrainingParallelismConfig,
        system_name: str,
        allow_spot: bool,
    ) -> Optional[ClusterRecommendationResult]:
        """Run simulation and build result object."""
        # Run training simulation
        sim_result = training_modeling(
            model=job_spec.model,
            training_stage=job_spec.training_type,
            batch_size=job_spec.batch_size or 4,
            seq_length=job_spec.avg_sequence_length,
            system_name=system_name,
            num_gpus=cluster.num_gpus,
            tensor_parallel=parallelism_config.tensor_parallel,
            data_parallel=parallelism_config.data_parallel,
            pipeline_parallel=parallelism_config.pipeline_parallel,
            expert_parallel=parallelism_config.expert_parallel,
            method=job_spec.method,
            optimizer=job_spec.optimizer,
            zero_stage=parallelism_config.zero_stage,
            gradient_checkpointing=parallelism_config.gradient_checkpointing,
            gradient_accumulation_steps=parallelism_config.gradient_accumulation_steps,
            bits=job_spec.precision,
        )

        # Calculate training duration
        total_tokens = job_spec.total_tokens
        training_hours = total_tokens / (sim_result.tokens_per_second * 3600) if sim_result.tokens_per_second > 0 else float('inf')

        # Calculate TCO
        tco = calculate_tco(
            gpu_type=cluster.gpu_type,
            num_gpus=cluster.num_gpus,
            training_hours=training_hours,
            dataset_tokens=total_tokens,
            allow_spot=allow_spot,
        )

        # Check constraints
        if not self._satisfies_constraints(job_spec, tco, training_hours, sim_result):
            return None

        # Build parallelism strategy
        parallelism = ParallelismStrategy(
            tensor_parallel=parallelism_config.tensor_parallel,
            pipeline_parallel=parallelism_config.pipeline_parallel,
            data_parallel=parallelism_config.data_parallel,
            zero_stage=parallelism_config.zero_stage,
            gradient_accumulation_steps=parallelism_config.gradient_accumulation_steps,
            gradient_checkpointing=parallelism_config.gradient_checkpointing,
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
                precision=job_spec.precision,
                optimizer=job_spec.optimizer,
            )

        # Calculate training steps
        batch_size = job_spec.batch_size or 4
        tokens_per_step = (
            batch_size *
            job_spec.avg_sequence_length *
            parallelism.data_parallel *
            parallelism.gradient_accumulation_steps
        )
        training_steps = total_tokens // tokens_per_step if tokens_per_step > 0 else 0

        return ClusterRecommendationResult(
            cluster=cluster,
            parallelism=parallelism,
            tokens_per_second=sim_result.tokens_per_second,
            step_time_ms=sim_result.step_time_ms,
            memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
            mfu=sim_result.model_flops_utilization,
            training_hours=training_hours,
            training_steps=int(training_steps),
            tco_breakdown=tco,
            llamafactory_config=llama_config,
            deepspeed_config=ds_config,
        )

    def _satisfies_constraints(
        self,
        job_spec: TrainingJobSpec,
        tco: TCOBreakdown,
        training_hours: float,
        sim_result: TrainingModelingOutput,
    ) -> bool:
        """Check if configuration satisfies job constraints."""
        if job_spec.max_budget_usd is not None and tco.total_cost > job_spec.max_budget_usd:
            return False

        if job_spec.max_time_hours is not None and training_hours > job_spec.max_time_hours:
            return False

        if job_spec.min_throughput_tps is not None and sim_result.tokens_per_second < job_spec.min_throughput_tps:
            return False

        if job_spec.max_memory_per_gpu_gb is not None and sim_result.memory_per_gpu_gb > job_spec.max_memory_per_gpu_gb:
            return False

        return True

    def _calculate_score(
        self,
        result: ClusterRecommendationResult,
        optimization_target: OptimizationTarget,
    ) -> float:
        """
        Calculate raw score for a result based on optimization target.

        Note: This returns raw scores that may have different scales.
        Use _normalize_scores() after collecting all results for fair comparison.
        """
        if optimization_target == OptimizationTarget.TCO:
            # Lower cost is better, return negative for consistent "higher is better"
            return -result.tco_breakdown.total_cost

        elif optimization_target == OptimizationTarget.THROUGHPUT:
            return result.tokens_per_second

        elif optimization_target == OptimizationTarget.MFU:
            return result.mfu

        elif optimization_target == OptimizationTarget.COST_PER_TOKEN:
            # Lower cost per token is better, return negative
            return -result.tco_breakdown.cost_per_million_tokens

        elif optimization_target == OptimizationTarget.LATENCY:
            # Lower latency is better, return negative
            return -result.step_time_ms

        else:
            return result.tokens_per_second  # Default to throughput

    def _normalize_scores(
        self,
        results: List[ClusterRecommendationResult],
        optimization_target: OptimizationTarget,
    ) -> None:
        """
        Normalize scores to 0-1 range for fair comparison across results.

        This ensures all scoring functions produce comparable values regardless
        of the underlying metric scale (throughput in millions vs cost in thousands).
        """
        if not results:
            return

        # Collect raw scores
        scores = [r.score for r in results]

        if len(scores) < 2:
            # Single result, assign score of 1.0
            for r in results:
                r.score = 1.0
            return

        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        score_range = max_score - min_score
        if score_range < 1e-9:
            # All scores are equal
            for r in results:
                r.score = 1.0
            return

        # Normalize to 0-1 range (higher is always better after normalization)
        for r in results:
            r.score = (r.score - min_score) / score_range

    def _compute_pareto_frontier(
        self,
        results: List[ClusterRecommendationResult],
        k: int,
    ) -> List[ClusterRecommendationResult]:
        """Compute Pareto frontier for multi-objective optimization."""
        if not HAS_PARETOSET:
            warnings.warn(
                "paretoset not available. Falling back to throughput-based ranking. "
                "Install with: pip install paretoset"
            )
            results.sort(key=lambda x: -x.tokens_per_second)
            return results[:k]

        if len(results) < 2:
            return results[:k]

        import numpy as np

        # Build data array for Pareto: [throughput, -cost, mfu, -time]
        # We want to maximize throughput and MFU, minimize cost and time
        data = np.array([
            [
                r.tokens_per_second,
                -r.tco_breakdown.total_cost,
                r.mfu,
                -r.training_hours,
            ]
            for r in results
        ])

        # Find Pareto frontier (all objectives maximized)
        mask = paretoset(data, sense=["max", "max", "max", "max"])

        pareto_results = [r for r, m in zip(results, mask) if m]

        # Sort by throughput within Pareto frontier
        pareto_results.sort(key=lambda x: -x.tokens_per_second)

        # Assign ranks
        for i, r in enumerate(pareto_results[:k]):
            r.rank = i + 1

        return pareto_results[:k]

    def _generate_gpu_counts(self, min_gpus: int, max_gpus: int) -> List[int]:
        """Generate GPU count options to search."""
        counts = set()

        # Powers of 2
        power = 0
        while 2 ** power <= max_gpus:
            if 2 ** power >= min_gpus:
                counts.add(2 ** power)
            power += 1

        # Common cluster sizes
        common_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        for size in common_sizes:
            if min_gpus <= size <= max_gpus:
                counts.add(size)

        return sorted(counts)

    def _get_hourly_rate(self, gpu_type: str, allow_spot: bool) -> float:
        """Get hourly rate for a GPU type."""
        try:
            pricing = get_gpu_pricing(gpu_type)
            rate, _, _ = pricing.get_best_rate(allow_spot=allow_spot)
            return rate
        except Exception:
            return 5.0  # Default fallback

    # =========================================================================
    # NEW METHODS: Cluster Ranking and Requirements
    # =========================================================================

    def rank_clusters(
        self,
        job_spec: TrainingJobSpec,
        clusters: List[ClusterDefinition],
        sort_by: str = "throughput",
        k: int = 10,
    ) -> List[ClusterRankingResult]:
        """
        Rank clusters by throughput, ETA, and cost with optimal batch size.

        Wrapper for rank_clusters_for_training() that uses job_spec.

        Args:
            job_spec: Training job specification
            clusters: List of available clusters
            sort_by: Sorting metric (throughput, eta, cost, composite)
            k: Number of top results to return

        Returns:
            List of ClusterRankingResult sorted by requested metric
        """
        return rank_clusters_for_training(
            model=job_spec.model,
            training_type=job_spec.training_type,
            clusters=clusters,
            dataset_tokens=job_spec.dataset_tokens,
            avg_sequence_length=job_spec.avg_sequence_length,
            num_epochs=job_spec.num_epochs,
            method=job_spec.method,
            optimizer=job_spec.optimizer,
            precision=job_spec.precision,
            max_training_hours=job_spec.max_time_hours,
            max_budget_usd=job_spec.max_budget_usd,
            sort_by=sort_by,
            return_top_k=k,
            debug=self.debug,
        )

    def predict_requirements(
        self,
        job_spec: TrainingJobSpec,
        max_training_hours: Optional[float] = None,
        target_mfu: float = 0.4,
    ) -> MinimumClusterRequirements:
        """
        Predict minimum cluster requirements for a training job.

        Wrapper for predict_cluster_requirements() that uses job_spec.

        Args:
            job_spec: Training job specification
            max_training_hours: Maximum acceptable training time
            target_mfu: Target Model FLOPS Utilization

        Returns:
            MinimumClusterRequirements with detailed requirements
        """
        return predict_cluster_requirements(
            training_type=job_spec.training_type,
            model=job_spec.model,
            dtype=job_spec.precision,
            optimizer=job_spec.optimizer,
            dataset_size_tokens=job_spec.dataset_tokens,
            batch_size=job_spec.batch_size or 4,
            seq_length=job_spec.avg_sequence_length,
            method=job_spec.method,
            max_training_hours=max_training_hours or job_spec.max_time_hours,
            target_mfu=target_mfu,
        )

    def generate_comprehensive_config(
        self,
        job_spec: TrainingJobSpec,
        parallelism: ParallelismStrategy,
        cluster: ClusterDefinition,
        focus: str = "balanced",
    ) -> ComprehensiveLlamaFactoryConfig:
        """
        Generate comprehensive LlamaFactory configuration with best practices.

        Wrapper for generate_comprehensive_training_config().

        Args:
            job_spec: Training job specification
            parallelism: Parallelism strategy
            cluster: Cluster configuration
            focus: Optimization focus (stable, convergence, speed, tco, balanced)

        Returns:
            ComprehensiveLlamaFactoryConfig with all configurations
        """
        return generate_comprehensive_training_config(
            job_spec=job_spec,
            parallelism=parallelism,
            cluster=cluster,
            optimization_focus=focus,
            enable_best_practices=True,
        )

    def _can_potentially_satisfy_constraints(
        self,
        job_spec: TrainingJobSpec,
        cluster: ClusterDefinition,
        parallelism_config: TrainingParallelismConfig,
    ) -> Tuple[bool, Optional[str]]:
        """
        Quick check if config can potentially satisfy constraints before full simulation.

        This performs fast estimates to avoid expensive training_modeling() calls
        for configurations that will clearly fail.

        Returns:
            Tuple of (can_satisfy, rejection_reason)
        """
        gpu_memory = _get_gpu_memory(cluster.gpu_type)

        # Quick memory estimate (simplified model)
        try:
            from ..genz.Models import get_configs
            model_config = get_configs(job_spec.model)

            # Estimate model parameters
            hidden = getattr(model_config, 'hidden_size', 4096) or 4096
            layers = getattr(model_config, 'num_decoder_layers', 32) or 32
            vocab = getattr(model_config, 'vocab_size', 32000) or 32000
            intermediate = getattr(model_config, 'intermediate_size', None) or (hidden * 4)

            # Rough parameter estimate
            params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
            total_params = layers * params_per_layer + 2 * vocab * hidden

            # Memory per GPU estimate (simplified)
            precision_bytes = 2  # BF16 default
            if job_spec.precision in ('fp32', 'f32'):
                precision_bytes = 4
            elif job_spec.precision in ('int8', 'int4', 'nf4'):
                precision_bytes = 1

            # Weight memory (divided by TP)
            weight_memory_gb = (total_params * precision_bytes) / parallelism_config.tensor_parallel / 1e9

            # For LoRA/QLoRA, weights are frozen in lower precision
            if job_spec.method in ('qlora', 'gptq'):
                weight_memory_gb = (total_params * 0.5) / parallelism_config.tensor_parallel / 1e9
            elif job_spec.method in ('lora', 'dora', 'pissa'):
                # LoRA has small additional overhead
                weight_memory_gb = (total_params * precision_bytes) / parallelism_config.tensor_parallel / 1e9

            # Gradient memory (FP32, only for trainable params)
            trainable_ratio = 1.0 if job_spec.method == 'full' else 0.01  # ~1% for LoRA
            gradient_memory_gb = (total_params * trainable_ratio * 4) / 1e9

            # ZeRO-2+ shards gradients
            if parallelism_config.zero_stage >= 2:
                gradient_memory_gb /= parallelism_config.data_parallel

            # ZeRO-3 shards weights
            if parallelism_config.zero_stage >= 3:
                weight_memory_gb /= parallelism_config.data_parallel

            # Optimizer memory (8 bytes for AdamW per trainable param)
            opt_bytes = 8 if job_spec.optimizer in ('adamw', 'adam', 'lamb') else 4
            if job_spec.optimizer in ('adamw_8bit', 'paged_adamw_8bit', 'adam_8bit'):
                opt_bytes = 2
            optimizer_memory_gb = (total_params * trainable_ratio * opt_bytes) / 1e9

            # ZeRO-1+ shards optimizer
            if parallelism_config.zero_stage >= 1:
                optimizer_memory_gb /= parallelism_config.data_parallel

            # Activation memory (very rough estimate)
            batch = job_spec.batch_size or 4
            seq = job_spec.avg_sequence_length
            activation_memory_gb = (batch * seq * hidden * layers * precision_bytes * 4) / 1e9

            # Gradient checkpointing reduces activations
            if parallelism_config.gradient_checkpointing:
                activation_memory_gb *= 0.3  # ~70% reduction

            # Divide by TP
            activation_memory_gb /= parallelism_config.tensor_parallel

            # Total estimate
            total_estimate_gb = (
                weight_memory_gb +
                gradient_memory_gb +
                optimizer_memory_gb +
                activation_memory_gb
            ) * 1.15  # 15% overhead

            # Check memory constraint
            if total_estimate_gb > gpu_memory * 0.95:
                return False, f"Estimated memory {total_estimate_gb:.1f}GB > GPU memory {gpu_memory}GB"

            # Check user-specified memory constraint
            if job_spec.max_memory_per_gpu_gb is not None:
                if total_estimate_gb > job_spec.max_memory_per_gpu_gb * 1.1:
                    return False, f"Estimated memory {total_estimate_gb:.1f}GB > max {job_spec.max_memory_per_gpu_gb}GB"

        except Exception:
            # If estimation fails, don't reject - let full simulation decide
            pass

        # Quick throughput estimate for min_throughput_tps constraint
        if job_spec.min_throughput_tps is not None:
            try:
                # Very rough throughput estimate based on GPU peak performance
                gpu_spec = get_gpu_spec(cluster.gpu_type)
                if gpu_spec:
                    peak_tflops = gpu_spec.fp16_tflops
                    # Rough estimate: ~30% efficiency, 6 FLOPs per token per param
                    estimated_tps = (peak_tflops * 1e12 * 0.3) / (6 * total_params / cluster.num_gpus)
                    if estimated_tps < job_spec.min_throughput_tps * 0.3:  # 30% tolerance for rough estimate
                        return False, f"Estimated TPS {estimated_tps:.0f} << min {job_spec.min_throughput_tps:.0f}"
            except Exception:
                pass

        return True, None

    def _filter_gpu_types_for_model(
        self,
        job_spec: TrainingJobSpec,
        gpu_types: List[str],
        max_gpus: int,
    ) -> List[str]:
        """
        Filter GPU types based on model requirements to reduce search space.

        Removes GPU types that clearly can't fit the workload even with maximum parallelism.
        """
        filtered = []

        try:
            from ..genz.Models import get_configs
            model_config = get_configs(job_spec.model)

            # Estimate total parameters
            hidden = getattr(model_config, 'hidden_size', 4096) or 4096
            layers = getattr(model_config, 'num_decoder_layers', 32) or 32
            vocab = getattr(model_config, 'vocab_size', 32000) or 32000
            intermediate = getattr(model_config, 'intermediate_size', None) or (hidden * 4)

            params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
            total_params = layers * params_per_layer + 2 * vocab * hidden

            # Minimum memory needed per GPU (with maximum parallelism)
            if job_spec.method in ('qlora', 'gptq'):
                bytes_per_param = 0.5  # 4-bit
            else:
                bytes_per_param = 2  # BF16

            min_memory_per_gpu = (total_params * bytes_per_param) / max_gpus / 1e9

            for gpu_type in gpu_types:
                gpu_memory = _get_gpu_memory(gpu_type)
                # Need at least 20% headroom for activations
                if gpu_memory * 0.8 >= min_memory_per_gpu:
                    filtered.append(gpu_type)
                elif self.debug:
                    print(f"Filtered out {gpu_type}: {gpu_memory}GB < min needed {min_memory_per_gpu:.1f}GB")

            return filtered if filtered else gpu_types

        except Exception:
            # If estimation fails, return all GPUs
            return gpu_types


def select_optimal_cluster(
    job_spec: TrainingJobSpec,
    available_clusters: List[ClusterDefinition],
    optimization_target: OptimizationTarget = OptimizationTarget.TCO,
    debug: bool = False,
) -> ClusterRecommendationResult:
    """
    Convenience function to select the single best cluster.

    Args:
        job_spec: Training job specification
        available_clusters: List of available clusters
        optimization_target: What to optimize for
        debug: Enable debug output

    Returns:
        Best ClusterRecommendationResult
    """
    optimizer = ClusterOptimizer(debug=debug)
    results = optimizer.select_top_k_clusters(
        job_spec=job_spec,
        available_clusters=available_clusters,
        optimization_target=optimization_target,
        k=1,
    )
    return results[0]


def design_optimal_training_cluster(
    job_spec: TrainingJobSpec,
    optimization_target: OptimizationTarget = OptimizationTarget.TCO,
    max_gpus: int = 128,
    debug: bool = False,
) -> OptimalClusterDesignResult:
    """
    Convenience function to design optimal cluster from scratch.

    Args:
        job_spec: Training job specification
        optimization_target: What to optimize for
        max_gpus: Maximum GPUs to consider
        debug: Enable debug output

    Returns:
        OptimalClusterDesignResult
    """
    optimizer = ClusterOptimizer(debug=debug)
    return optimizer.design_optimal_cluster(
        job_spec=job_spec,
        optimization_target=optimization_target,
        max_gpus=max_gpus,
    )
