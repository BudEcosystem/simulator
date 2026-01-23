"""
Training Cluster Selector.

Recommends optimal cluster configurations for training workloads.
"""

from typing import Dict, Any, List, Optional

from ..hardware import HARDWARE_CONFIGS, get_hardware_by_type
from .types import (
    TrainingMemoryEstimate,
    ClusterRecommendation,
    ClusterFitResult,
)


class TrainingClusterSelector:
    """
    Select optimal cluster configurations for training.

    Features:
    - Recommend clusters sorted by cost or speed
    - Check if training fits in specific cluster
    - Suggest parallelism strategies (TP, PP, DP)
    - Estimate utilization and cost
    """

    # GPU cost per hour (approximate cloud pricing)
    GPU_COST_PER_HOUR = {
        'A100_40GB_GPU': 2.21,
        'A100_80GB_GPU': 3.67,
        'H100_GPU': 4.76,
        'GH200_GPU': 5.50,
        'B100': 7.00,
        'GB200': 8.00,
        'V100_16GB_GPU': 0.90,
        'V100_32GB_GPU': 1.21,
        'L40S_48GB_GPU': 1.98,
        'MI300X': 3.50,
        'MI325X': 4.00,
        'TPUv4': 2.00,
        'TPUv5e': 2.50,
        'TPUv5p': 4.50,
        'TPUv6': 5.50,
        'Gaudi3': 3.00,
        'Trainium1': 2.00,
    }

    # TFLOPs efficiency for training (typically 30-50% of peak)
    TRAINING_EFFICIENCY = 0.35

    def __init__(self):
        """Initialize the cluster selector."""
        self._hardware_cache: Dict[str, Dict[str, Any]] = {}

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """List all available hardware profiles including GPUs, ASICs, accelerators, and CPUs."""
        hardware_list = []
        for name, config in HARDWARE_CONFIGS.items():
            # Include all hardware types: gpu, asic, accelerator, cpu
            hw_type = config.get('type')
            if hw_type in ('gpu', 'asic', 'accelerator', 'cpu'):
                hardware_list.append({
                    "name": name,
                    "type": hw_type,
                    "manufacturer": config.get('manufacturer'),
                    "memory_gb": config.get('Memory_size', 0),
                    "flops_tflops": config.get('Flops', 0),
                    "memory_bw_gbps": config.get('Memory_BW', 0),
                    "cost_per_hour": self.GPU_COST_PER_HOUR.get(name, 0),
                })
        return hardware_list

    def recommend_clusters(
        self,
        training_estimate: Dict[str, Any],
        prefer_cost: bool = True,
        max_budget_per_hour: Optional[float] = None,
        available_hardware: Optional[List[str]] = None,
        max_gpus: int = 32,
    ) -> List[ClusterRecommendation]:
        """
        Recommend cluster configurations for training.

        Args:
            training_estimate: Training memory estimate dict or TrainingMemoryEstimate
            prefer_cost: Sort by cost (True) or speed (False)
            max_budget_per_hour: Maximum hourly budget
            available_hardware: Limit to specific hardware types
            max_gpus: Maximum number of GPUs to consider

        Returns:
            List of cluster recommendations sorted by preference
        """
        # Convert to dict if needed
        if isinstance(training_estimate, TrainingMemoryEstimate):
            estimate = training_estimate.to_dict()
        else:
            estimate = training_estimate

        total_memory_gb = estimate.get('total_memory_gb', 0)
        trainable_params = estimate.get('trainable_params', 0)

        recommendations = []

        # Get hardware to consider
        hardware_list = available_hardware or list(self.GPU_COST_PER_HOUR.keys())

        for hw_name in hardware_list:
            hw_config = HARDWARE_CONFIGS.get(hw_name)
            if not hw_config:
                continue

            memory_per_gpu = hw_config.get('Memory_size', 0)
            if memory_per_gpu <= 0:
                continue

            # Try different GPU counts
            for num_gpus in [1, 2, 4, 8, 16, 32]:
                if num_gpus > max_gpus:
                    break

                # Find best parallelism strategy
                fit_result = self._find_best_parallelism(
                    total_memory_gb=total_memory_gb,
                    trainable_params=trainable_params,
                    memory_per_gpu=memory_per_gpu,
                    num_gpus=num_gpus,
                    hw_config=hw_config,
                )

                if not fit_result.fits:
                    continue

                # Calculate cost
                cost_per_hour = self.GPU_COST_PER_HOUR.get(hw_name, 0) * num_gpus

                # Apply budget filter
                if max_budget_per_hour and cost_per_hour > max_budget_per_hour:
                    continue

                # Calculate throughput estimate
                throughput = self._estimate_throughput(
                    hw_config=hw_config,
                    num_gpus=num_gpus,
                    parallelism=fit_result.parallelism,
                    trainable_params=trainable_params,
                )

                # Determine optimality
                optimality = "optimal" if fit_result.utilization_percent >= 70 else "good"
                if fit_result.utilization_percent < 50:
                    optimality = "suboptimal"

                recommendations.append(ClusterRecommendation(
                    hardware_name=hw_name,
                    nodes_required=max(1, num_gpus // 8),
                    gpus_per_node=min(8, num_gpus),
                    total_gpus=num_gpus,
                    memory_per_gpu_gb=memory_per_gpu,
                    parallelism=fit_result.parallelism or {"tp": 1, "pp": 1, "dp": num_gpus},
                    estimated_throughput_tps=throughput,
                    estimated_cost_per_hour=cost_per_hour,
                    utilization_percent=fit_result.utilization_percent,
                    fits=True,
                    optimality=optimality,
                ))

        # Sort by preference
        if prefer_cost:
            recommendations.sort(key=lambda r: r.estimated_cost_per_hour)
        else:
            recommendations.sort(key=lambda r: -r.estimated_throughput_tps)

        return recommendations

    def check_fit(
        self,
        training_estimate: Dict[str, Any],
        hardware: str,
        num_gpus: int,
    ) -> ClusterFitResult:
        """
        Check if training fits in specific cluster configuration.

        Args:
            training_estimate: Training memory estimate
            hardware: Hardware name
            num_gpus: Number of GPUs

        Returns:
            ClusterFitResult with fit status and details
        """
        # Convert to dict if needed
        if isinstance(training_estimate, TrainingMemoryEstimate):
            estimate = training_estimate.to_dict()
        else:
            estimate = training_estimate

        hw_config = HARDWARE_CONFIGS.get(hardware)
        if not hw_config:
            available = list(self.GPU_COST_PER_HOUR.keys())
            return ClusterFitResult(
                fits=False,
                reason=f"Unknown hardware: '{hardware}'. Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}",
            )

        memory_per_gpu = hw_config.get('Memory_size', 0)
        total_memory_gb = estimate.get('total_memory_gb', 0)
        trainable_params = estimate.get('trainable_params', 0)

        # Find best parallelism
        fit_result = self._find_best_parallelism(
            total_memory_gb=total_memory_gb,
            trainable_params=trainable_params,
            memory_per_gpu=memory_per_gpu,
            num_gpus=num_gpus,
            hw_config=hw_config,
        )

        # Calculate minimum GPUs needed
        if not fit_result.fits:
            min_gpus = self._calculate_min_gpus(
                total_memory_gb=total_memory_gb,
                memory_per_gpu=memory_per_gpu,
            )
            fit_result.min_gpus_required = min_gpus

        # Calculate cost
        cost_per_hour = self.GPU_COST_PER_HOUR.get(hardware, 0) * num_gpus
        fit_result.estimated_cost_per_hour = cost_per_hour

        return fit_result

    def _find_best_parallelism(
        self,
        total_memory_gb: float,
        trainable_params: int,
        memory_per_gpu: float,
        num_gpus: int,
        hw_config: Dict[str, Any],
    ) -> ClusterFitResult:
        """Find optimal parallelism strategy for given configuration."""
        # Safety margin (90% of GPU memory)
        available_memory = memory_per_gpu * 0.9

        # Try different parallelism strategies
        for tp in [1, 2, 4, 8]:
            if tp > num_gpus:
                continue

            for pp in [1, 2, 4]:
                if tp * pp > num_gpus:
                    continue

                dp = num_gpus // (tp * pp)
                if dp == 0:
                    continue

                # Calculate memory per GPU with this parallelism
                # Weight memory divided by TP
                # Optimizer/gradient memory divided by DP (ZeRO-2 assumed)
                # Activation memory divided by TP

                # Simplified memory model
                memory_per_device = total_memory_gb / (tp * pp * dp)

                # With ZeRO-2, optimizer states are sharded
                # This reduces memory by ~40% on average
                if dp > 1:
                    memory_per_device *= 0.7

                if memory_per_device <= available_memory:
                    utilization = (memory_per_device / memory_per_gpu) * 100

                    return ClusterFitResult(
                        fits=True,
                        memory_per_gpu_gb=memory_per_device,
                        utilization_percent=min(100, utilization),
                        parallelism={"tp": tp, "pp": pp, "dp": dp},
                    )

        # No valid configuration found
        return ClusterFitResult(
            fits=False,
            reason=f"Insufficient memory: {total_memory_gb:.1f}GB required, {memory_per_gpu * num_gpus:.1f}GB available across {num_gpus} GPUs",
        )

    def _calculate_min_gpus(
        self,
        total_memory_gb: float,
        memory_per_gpu: float,
    ) -> int:
        """Calculate minimum GPUs needed for training."""
        available_memory = memory_per_gpu * 0.9

        # Start with simple calculation
        min_gpus = int(total_memory_gb / available_memory) + 1

        # Consider ZeRO-2 effect (optimizer states sharded)
        # With ZeRO-2, we can fit more with fewer GPUs
        if min_gpus >= 2:
            # ZeRO-2 reduces optimizer memory by DP factor
            # Approximate: need ~60% of naive calculation
            min_gpus = max(2, int(min_gpus * 0.6))

        # Round to power of 2 for efficient parallelism
        for gpus in [1, 2, 4, 8, 16, 32, 64]:
            if gpus >= min_gpus:
                return gpus

        return 64

    def _estimate_throughput(
        self,
        hw_config: Dict[str, Any],
        num_gpus: int,
        parallelism: Optional[Dict[str, int]],
        trainable_params: int,
    ) -> float:
        """Estimate training throughput in tokens per second."""
        # Get hardware specs
        tflops = hw_config.get('Flops', 100)  # Peak TFLOPs
        memory_bw = hw_config.get('Memory_BW', 1000)  # GB/s

        # Training FLOPs per token: ~6 × params (forward + backward + optimizer)
        flops_per_token = 6 * trainable_params

        # Compute-bound estimate
        effective_tflops = tflops * self.TRAINING_EFFICIENCY * num_gpus
        compute_tps = (effective_tflops * 1e12) / flops_per_token

        # Memory-bound estimate (gradient sync, etc.)
        memory_tps = (memory_bw * 1e9 * num_gpus) / (trainable_params * 4)

        # Take minimum (bottleneck)
        theoretical_tps = min(compute_tps, memory_tps)

        # Apply DP efficiency (communication overhead)
        dp = parallelism.get('dp', 1) if parallelism else 1
        if dp > 1:
            efficiency = 0.9 ** (dp - 1)  # ~10% overhead per DP rank
            theoretical_tps *= efficiency

        return max(100, theoretical_tps)  # Minimum 100 TPS
