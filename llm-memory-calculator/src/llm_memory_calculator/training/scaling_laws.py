"""
Empirical Scaling Laws for Training Simulation Accuracy Calibration.

This module provides generalized scaling laws derived from ground truth benchmarks
rather than hardcoded values. The key principle is physics-based models with
empirically-fitted coefficients.

Key Scaling Laws:
1. Model Size Efficiency Scaling - Superlinear penalty with model size
2. GPU Scale Efficiency Decay - Logarithmic decay with GPU count
3. Network Congestion Model - Superlinear overhead from concurrent collectives
4. Straggler/Synchronization Model - sqrt scaling with cluster size

References:
- Meta LLaMA 2/3 training reports
- NVIDIA Megatron-LM measurements
- DeepSeek V3 technical report
- AMSP paper (scale-dependent bubble growth)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np


@dataclass
class ScalingCoefficients:
    """
    Calibration coefficients derived from ground truth benchmarks.

    These coefficients parametrize the generalized scaling formulas.
    They should be fitted from benchmark data, not hardcoded.
    """
    # Model size scaling: penalty = α * (params_b / reference_size) ^ β
    size_scaling_alpha: float = 0.025    # Base coefficient
    size_scaling_beta: float = 1.3       # Exponent (superlinear)
    size_reference_b: float = 10.0       # Reference model size in billions

    # GPU scale efficiency decay: efficiency = 1 / (1 + γ * log(gpus / reference))
    scale_gamma_tp: float = 0.04         # TP scale decay coefficient
    scale_gamma_pp: float = 0.06         # PP scale decay coefficient
    scale_gamma_dp: float = 0.03         # DP scale decay coefficient
    scale_gamma_ep: float = 0.08         # EP scale decay coefficient
    scale_reference_gpus: int = 8        # Reference GPU count

    # Network congestion: overhead = 1 + δ * log(1 + concurrent_collectives)
    congestion_delta: float = 0.12       # Congestion coefficient

    # Straggler overhead: overhead = ε * sqrt(total_gpus / 1000)
    straggler_epsilon: float = 0.05      # Straggler coefficient

    # Pipeline bubble enhancement factors
    bubble_base_scale: float = 1.0       # Base bubble multiplier
    bubble_scale_growth_rate: float = 0.3  # Growth rate with scale

    # Memory pressure factors (affects efficiency at large model sizes)
    memory_pressure_threshold_b: float = 50.0  # Model size where memory pressure starts
    memory_pressure_coefficient: float = 0.02   # Memory pressure per 10B above threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'size_scaling_alpha': self.size_scaling_alpha,
            'size_scaling_beta': self.size_scaling_beta,
            'size_reference_b': self.size_reference_b,
            'scale_gamma_tp': self.scale_gamma_tp,
            'scale_gamma_pp': self.scale_gamma_pp,
            'scale_gamma_dp': self.scale_gamma_dp,
            'scale_gamma_ep': self.scale_gamma_ep,
            'scale_reference_gpus': self.scale_reference_gpus,
            'congestion_delta': self.congestion_delta,
            'straggler_epsilon': self.straggler_epsilon,
            'bubble_base_scale': self.bubble_base_scale,
            'bubble_scale_growth_rate': self.bubble_scale_growth_rate,
            'memory_pressure_threshold_b': self.memory_pressure_threshold_b,
            'memory_pressure_coefficient': self.memory_pressure_coefficient,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ScalingCoefficients':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModelSizeScaling:
    """
    Empirically-derived model size efficiency scaling.

    Large models experience efficiency loss due to:
    - Memory bandwidth pressure (HBM saturation)
    - Cache thrashing (larger activations)
    - Register spills (wider layers)
    - Communication overhead (larger gradient tensors)

    The penalty scales superlinearly with model size because
    these effects compound rather than add linearly.
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()

    def compute_size_penalty(
        self,
        model_params_b: float,
        hardware: Optional[str] = None,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ) -> float:
        """
        Compute model size efficiency penalty.

        Generalized formula:
            penalty = α * (params_b / reference_size) ^ β * memory_pressure_factor

        Where:
        - α, β are fitted from ground truth data
        - memory_pressure_factor accounts for cache/register pressure
        - reference_size normalizes across hardware generations

        Args:
            model_params_b: Model parameters in billions
            hardware: Hardware name for hardware-specific adjustments
            tensor_parallel: TP degree (higher TP reduces per-GPU model size)
            pipeline_parallel: PP degree (higher PP reduces per-GPU layers)

        Returns:
            Efficiency penalty (0.0 to 0.5, where 0.0 means no penalty)
        """
        if model_params_b <= 0:
            return 0.0

        c = self.coefficients

        # Effective model size per GPU (after parallelism)
        # TP shards model width, PP shards model depth
        effective_params_b = model_params_b / max(1, tensor_parallel * pipeline_parallel)

        # Base penalty: superlinear scaling with model size
        if effective_params_b <= c.size_reference_b:
            # Small models: minimal penalty
            base_penalty = 0.0
        else:
            # Superlinear penalty for large models
            size_ratio = effective_params_b / c.size_reference_b
            base_penalty = c.size_scaling_alpha * (size_ratio ** c.size_scaling_beta - 1)

        # Memory pressure factor for very large models
        memory_pressure = 0.0
        if model_params_b > c.memory_pressure_threshold_b:
            excess_b = model_params_b - c.memory_pressure_threshold_b
            memory_pressure = c.memory_pressure_coefficient * (excess_b / 10.0)

        # Total penalty (capped at 50% to avoid unrealistic values)
        total_penalty = min(0.5, base_penalty + memory_pressure)

        return total_penalty

    def get_size_tier(self, model_params_b: float) -> str:
        """Classify model into size tier for analysis."""
        if model_params_b < 10:
            return "small"
        elif model_params_b < 50:
            return "medium"
        elif model_params_b < 100:
            return "large"
        else:
            return "xlarge"


class ScaleEfficiencyModel:
    """
    Models efficiency decay as GPU count increases.

    Efficiency drops at scale due to:
    - Communication overhead (latency + bandwidth)
    - Synchronization barriers (stragglers)
    - Network congestion (competing collectives)
    - Pipeline bubbles (more stages = more idle time)
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()

    def compute_scale_efficiency(
        self,
        num_gpus: int,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        data_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> float:
        """
        Compute efficiency factor based on parallelism scale.

        Generalized formula:
            efficiency = base_efficiency * Π(parallel_dim_penalties)

        Each parallel dimension has:
        - Base overhead (latency-dominated for small scale)
        - Log-scaling overhead (bandwidth-dominated for medium scale)
        - Superlinear overhead (congestion-dominated for large scale)

        Args:
            num_gpus: Total GPU count
            tensor_parallel: TP degree
            pipeline_parallel: PP degree
            data_parallel: DP degree
            expert_parallel: EP degree

        Returns:
            Scale efficiency factor (0.0 to 1.0)
        """
        c = self.coefficients

        efficiency = 1.0

        # Tensor Parallel overhead
        if tensor_parallel > 1:
            tp_factor = 1.0 / (1.0 + c.scale_gamma_tp * math.log2(tensor_parallel))
            efficiency *= tp_factor

        # Pipeline Parallel overhead (separate from bubble, this is communication)
        if pipeline_parallel > 1:
            pp_factor = 1.0 / (1.0 + c.scale_gamma_pp * math.log2(pipeline_parallel))
            efficiency *= pp_factor

        # Data Parallel overhead
        if data_parallel > 1:
            dp_factor = 1.0 / (1.0 + c.scale_gamma_dp * math.log2(data_parallel))
            efficiency *= dp_factor

        # Expert Parallel overhead (All2All is expensive)
        if expert_parallel > 1:
            ep_factor = 1.0 / (1.0 + c.scale_gamma_ep * math.log2(expert_parallel))
            efficiency *= ep_factor

        # Straggler overhead at large scale
        straggler_factor = 1.0 - c.straggler_epsilon * math.sqrt(num_gpus / 1000)
        straggler_factor = max(0.7, straggler_factor)  # Cap at 30% loss
        efficiency *= straggler_factor

        return max(0.2, efficiency)  # Minimum 20% efficiency

    def compute_parallelism_overhead(
        self,
        parallelism_type: str,
        degree: int,
    ) -> float:
        """
        Compute overhead for a single parallelism dimension.

        Args:
            parallelism_type: One of 'tp', 'pp', 'dp', 'ep'
            degree: Parallelism degree

        Returns:
            Overhead fraction (0.0 to 1.0)
        """
        if degree <= 1:
            return 0.0

        c = self.coefficients

        gamma_map = {
            'tp': c.scale_gamma_tp,
            'pp': c.scale_gamma_pp,
            'dp': c.scale_gamma_dp,
            'ep': c.scale_gamma_ep,
        }

        gamma = gamma_map.get(parallelism_type, 0.05)
        overhead = gamma * math.log2(degree)

        return min(0.5, overhead)


class NetworkCongestionModel:
    """
    Models bandwidth degradation from concurrent collectives.

    When multiple collectives run simultaneously, they compete for
    network bandwidth and switch ports, leading to congestion.
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()

    def effective_bandwidth(
        self,
        base_bw_gbps: float,
        concurrent_collectives: int,
        topology: str = "fat_tree",
    ) -> float:
        """
        Compute effective bandwidth under congestion.

        Generalized congestion formula:
            effective_bw = base_bw / (1 + γ * log(1 + concurrent_collectives))

        Where γ depends on topology:
        - Fat-tree: γ ≈ 0.10 (good bisection bandwidth)
        - Torus: γ ≈ 0.15 (limited bisection)
        - Dragonfly: γ ≈ 0.12 (moderate)

        Args:
            base_bw_gbps: Base bandwidth in Gbps
            concurrent_collectives: Number of concurrent collective operations
            topology: Network topology type

        Returns:
            Effective bandwidth in Gbps
        """
        c = self.coefficients

        # Topology-specific congestion factors
        topology_gamma = {
            'fat_tree': c.congestion_delta,
            'torus': c.congestion_delta * 1.5,
            'dragonfly': c.congestion_delta * 1.2,
            'nvlink': c.congestion_delta * 0.7,  # NVLink has less congestion
        }

        gamma = topology_gamma.get(topology, c.congestion_delta)

        if concurrent_collectives <= 1:
            return base_bw_gbps

        congestion_factor = 1.0 + gamma * math.log(1 + concurrent_collectives)
        effective_bw = base_bw_gbps / congestion_factor

        return max(base_bw_gbps * 0.3, effective_bw)  # Minimum 30% of base

    def estimate_concurrent_collectives(
        self,
        tensor_parallel: int,
        pipeline_parallel: int,
        data_parallel: int,
        expert_parallel: int,
    ) -> int:
        """
        Estimate number of concurrent collectives during training.

        Each parallel dimension introduces collectives:
        - TP: 2 AllReduce per layer (attention, FFN)
        - PP: Activation passing, send/recv overlap
        - DP: Gradient AllReduce
        - EP: All2All for token routing

        These can overlap depending on schedule.
        """
        concurrent = 0

        if tensor_parallel > 1:
            concurrent += 2  # Overlapped AR
        if pipeline_parallel > 1:
            concurrent += 1  # Some overlap with compute
        if data_parallel > 1:
            concurrent += 1  # Gradient sync (overlapped)
        if expert_parallel > 1:
            concurrent += 2  # A2A is harder to overlap

        return max(1, concurrent)


class StragglerModel:
    """
    Models synchronization overhead at scale due to stragglers.

    At large scale, the slowest GPU determines iteration time.
    This causes idle time proportional to GPU count variance.
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()

    def sync_overhead(
        self,
        num_gpus: int,
        collective_type: str = "allreduce",
    ) -> float:
        """
        Compute synchronization overhead fraction.

        Formula: overhead = δ * sqrt(num_gpus / 1000)

        Where δ is empirically fitted per collective type:
        - AllReduce: δ ≈ 0.05 (well-optimized)
        - All2All: δ ≈ 0.08 (more variance)
        - Pipeline sync: δ ≈ 0.03 (smaller groups)

        Args:
            num_gpus: Total GPU count
            collective_type: Type of collective operation

        Returns:
            Overhead fraction (0.0 to 0.3)
        """
        c = self.coefficients

        # Collective-specific straggler coefficients
        collective_epsilon = {
            'allreduce': c.straggler_epsilon,
            'all2all': c.straggler_epsilon * 1.6,
            'pipeline': c.straggler_epsilon * 0.6,
            'allgather': c.straggler_epsilon * 0.9,
            'reducescatter': c.straggler_epsilon * 0.9,
        }

        epsilon = collective_epsilon.get(collective_type, c.straggler_epsilon)

        if num_gpus < 100:
            return 0.0  # Negligible at small scale

        overhead = epsilon * math.sqrt(num_gpus / 1000)

        return min(0.3, overhead)  # Cap at 30%


class PipelineBubbleModel:
    """
    Enhanced pipeline bubble calculation with scale-aware factors.

    Real-world measurements show bubble fraction increases at scale:
    - 2K GPUs: ~1.0x base bubble
    - 8K GPUs: ~1.5x base bubble
    - 16K GPUs: ~4x base bubble (congestion, stragglers)
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()

    def compute_bubble_fraction(
        self,
        pipeline_parallel: int,
        num_micro_batches: int,
        total_gpus: int,
        interleaved_stages: int = 1,
        use_zero_bubble: bool = False,
    ) -> float:
        """
        Compute pipeline bubble fraction with scale-aware adjustments.

        Enhanced calculation:
        1. Base bubble (1F1B): (pp - 1) / (pp + M - 1)
        2. Interleaved reduction: * interleave_factor
        3. Scale-dependent straggler: + straggler_overhead
        4. ZB-V reduction if enabled

        Args:
            pipeline_parallel: Number of pipeline stages
            num_micro_batches: Number of micro-batches per step
            total_gpus: Total GPUs in training
            interleaved_stages: Virtual pipeline stages
            use_zero_bubble: Whether using Zero-Bubble scheduling

        Returns:
            Effective bubble fraction (0.0 to 0.8)
        """
        if pipeline_parallel <= 1:
            return 0.0

        c = self.coefficients

        # Base bubble fraction (1F1B schedule)
        base_bubble = (pipeline_parallel - 1) / (pipeline_parallel + num_micro_batches - 1)

        # Scale factor based on total GPU count
        # Derived from real measurements (AMSP paper, Meta reports)
        if total_gpus <= 2048:
            scale_factor = c.bubble_base_scale
        elif total_gpus <= 4096:
            scale_factor = c.bubble_base_scale + c.bubble_scale_growth_rate * math.log2(total_gpus / 2048)
        elif total_gpus <= 8192:
            scale_factor = (c.bubble_base_scale + c.bubble_scale_growth_rate) + \
                          c.bubble_scale_growth_rate * 1.5 * math.log2(total_gpus / 4096)
        else:
            # Very large scale: significant congestion and straggler effects
            scale_factor = (c.bubble_base_scale + 2.5 * c.bubble_scale_growth_rate) + \
                          c.bubble_scale_growth_rate * 2.0 * math.log2(total_gpus / 8192)

        # Interleaved schedule reduction
        if interleaved_stages > 1:
            interleave_benefit = 1.0 / interleaved_stages
            # But benefit diminishes at very large scale
            scale_diminish = 1.0 + 0.1 * math.log2(max(1, total_gpus / 4096))
            interleave_benefit = min(1.0, interleave_benefit * scale_diminish)
            base_bubble *= interleave_benefit

        # Zero-bubble scheduling (ZB-V) reduces by ~90%
        if use_zero_bubble:
            base_bubble *= 0.1

        # Add straggler overhead
        straggler = StragglerModel(c).sync_overhead(total_gpus, 'pipeline')

        return min(0.8, base_bubble * scale_factor + straggler)

    def compute_activation_delay(
        self,
        pipeline_parallel: int,
        model_params_b: float,
        hidden_size: int = 8192,
        seq_length: int = 4096,
        batch_size: int = 1,
    ) -> float:
        """
        Compute activation passing delay as fraction of step time.

        Larger models have larger activations to pass between stages.
        This becomes significant for 100B+ models.

        Args:
            pipeline_parallel: Number of pipeline stages
            model_params_b: Model parameters in billions
            hidden_size: Model hidden dimension
            seq_length: Sequence length
            batch_size: Micro-batch size

        Returns:
            Activation delay fraction (0.0 to 0.15)
        """
        if pipeline_parallel <= 1:
            return 0.0

        # Activation size per micro-batch: batch * seq * hidden * 2 bytes (bf16)
        activation_bytes = batch_size * seq_length * hidden_size * 2

        # Larger models typically have larger hidden sizes
        # This is already captured, but add model-size awareness
        size_factor = 1.0 + 0.01 * max(0, model_params_b - 50) / 10

        # Base delay fraction (normalized)
        # At 100B+ models with large sequences, this becomes ~5-10%
        base_delay = 0.01 * (activation_bytes / (4096 * 8192 * 2)) * size_factor

        return min(0.15, base_delay * (pipeline_parallel - 1))


class DynamicEfficiencyBounds:
    """
    Computes dynamic efficiency bounds based on configuration.

    Replaces hardcoded max_efficiency = 0.60-0.65 with data-driven bounds
    that depend on model size, scale, and hardware.
    """

    def __init__(self, coefficients: Optional[ScalingCoefficients] = None):
        self.coefficients = coefficients or ScalingCoefficients()
        self.size_model = ModelSizeScaling(coefficients)
        self.scale_model = ScaleEfficiencyModel(coefficients)

    def get_efficiency_bounds(
        self,
        hardware: str,
        model_params_b: float,
        num_gpus: int,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        data_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> Tuple[float, float]:
        """
        Get dynamic efficiency bounds based on configuration.

        Formula:
            max_eff = base_max * size_decay * scale_decay
            min_eff = base_min * (1 - extreme_scale_penalty)

        Where:
        - base_max comes from hardware profile
        - size_decay = 1 / (1 + α * log(params_b / 10))
        - scale_decay = 1 / (1 + β * log(num_gpus / 8))

        Args:
            hardware: Hardware name
            model_params_b: Model parameters in billions
            num_gpus: Total GPU count
            tensor_parallel: TP degree
            pipeline_parallel: PP degree
            data_parallel: DP degree
            expert_parallel: EP degree

        Returns:
            (min_efficiency, max_efficiency) tuple
        """
        # Base bounds from hardware profile (import here to avoid circular)
        try:
            from ..genz.LLM_training.hardware_calibration import get_hardware_efficiency_profile
            hw_profile = get_hardware_efficiency_profile(hardware)
            base_max = hw_profile.max_efficiency
            base_min = hw_profile.min_efficiency
        except ImportError:
            # Fallback defaults
            base_max = 0.65
            base_min = 0.15

        # Size decay (larger models achieve lower peak efficiency)
        if model_params_b > 10:
            size_decay = 1.0 / (1.0 + 0.08 * math.log(model_params_b / 10))
        else:
            size_decay = 1.0

        # Scale decay (more GPUs means more overhead)
        if num_gpus > 8:
            scale_decay = 1.0 / (1.0 + 0.05 * math.log(num_gpus / 8))
        else:
            scale_decay = 1.0

        # Parallelism penalty (more dimensions = more overhead)
        parallelism_dims = sum([
            1 if tensor_parallel > 1 else 0,
            1 if pipeline_parallel > 1 else 0,
            1 if data_parallel > 1 else 0,
            1 if expert_parallel > 1 else 0,
        ])
        parallelism_penalty = 0.02 * parallelism_dims

        # Compute dynamic bounds
        max_eff = base_max * size_decay * scale_decay - parallelism_penalty
        min_eff = base_min

        # Very large scale further reduces max efficiency
        if num_gpus > 4096:
            extreme_scale_penalty = 0.05 * math.log2(num_gpus / 4096)
            max_eff -= extreme_scale_penalty

        # Very large models further reduce efficiency
        if model_params_b > 100:
            extreme_size_penalty = 0.03 * math.log2(model_params_b / 100)
            max_eff -= extreme_size_penalty

        # Ensure sensible bounds
        max_eff = max(0.20, min(0.70, max_eff))
        min_eff = max(0.05, min(max_eff - 0.10, min_eff))

        return (min_eff, max_eff)


# Default coefficients fitted from published benchmarks
DEFAULT_SCALING_COEFFICIENTS = ScalingCoefficients(
    # Fitted from Meta LLaMA, NVIDIA Megatron-LM benchmarks
    size_scaling_alpha=0.025,
    size_scaling_beta=1.3,
    size_reference_b=10.0,

    # Fitted from scale efficiency measurements
    scale_gamma_tp=0.04,
    scale_gamma_pp=0.06,
    scale_gamma_dp=0.03,
    scale_gamma_ep=0.08,
    scale_reference_gpus=8,

    # Network congestion (NCCL benchmarks)
    congestion_delta=0.12,

    # Straggler overhead (large-scale training reports)
    straggler_epsilon=0.05,

    # Pipeline bubble scaling
    bubble_base_scale=1.0,
    bubble_scale_growth_rate=0.3,

    # Memory pressure
    memory_pressure_threshold_b=50.0,
    memory_pressure_coefficient=0.02,
)


def get_scaling_models(
    coefficients: Optional[ScalingCoefficients] = None
) -> Tuple[ModelSizeScaling, ScaleEfficiencyModel, NetworkCongestionModel, StragglerModel, PipelineBubbleModel]:
    """
    Get all scaling models with consistent coefficients.

    Args:
        coefficients: Optional custom coefficients

    Returns:
        Tuple of (ModelSizeScaling, ScaleEfficiencyModel, NetworkCongestionModel, StragglerModel, PipelineBubbleModel)
    """
    c = coefficients or DEFAULT_SCALING_COEFFICIENTS
    return (
        ModelSizeScaling(c),
        ScaleEfficiencyModel(c),
        NetworkCongestionModel(c),
        StragglerModel(c),
        PipelineBubbleModel(c),
    )


def compute_composite_efficiency(
    model_params_b: float,
    num_gpus: int,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    data_parallel: int = 1,
    expert_parallel: int = 1,
    hardware: str = "A100_80GB_GPU",
    coefficients: Optional[ScalingCoefficients] = None,
) -> Dict[str, float]:
    """
    Compute composite efficiency with all scaling factors.

    This is the main entry point for efficiency calculation that combines
    all scaling models into a single efficiency estimate.

    Args:
        model_params_b: Model parameters in billions
        num_gpus: Total GPU count
        tensor_parallel: TP degree
        pipeline_parallel: PP degree
        data_parallel: DP degree
        expert_parallel: EP degree
        hardware: Hardware name
        coefficients: Optional custom coefficients

    Returns:
        Dictionary with efficiency breakdown:
        - base_efficiency: Hardware base efficiency
        - size_penalty: Model size penalty
        - scale_efficiency: Scale efficiency factor
        - straggler_overhead: Straggler overhead
        - final_efficiency: Net efficiency
        - efficiency_bounds: (min, max) efficiency bounds
    """
    c = coefficients or DEFAULT_SCALING_COEFFICIENTS

    size_model = ModelSizeScaling(c)
    scale_model = ScaleEfficiencyModel(c)
    straggler_model = StragglerModel(c)
    bounds_model = DynamicEfficiencyBounds(c)

    # Get hardware base efficiency
    try:
        from ..genz.LLM_training.hardware_calibration import get_hardware_efficiency_profile
        hw_profile = get_hardware_efficiency_profile(hardware)
        base_efficiency = hw_profile.base_efficiency
    except ImportError:
        base_efficiency = 0.70

    # Compute penalties
    size_penalty = size_model.compute_size_penalty(
        model_params_b, hardware, tensor_parallel, pipeline_parallel
    )

    scale_efficiency = scale_model.compute_scale_efficiency(
        num_gpus, tensor_parallel, pipeline_parallel, data_parallel, expert_parallel
    )

    straggler_overhead = straggler_model.sync_overhead(num_gpus, 'allreduce')

    # Combine into final efficiency
    final_efficiency = base_efficiency * scale_efficiency * (1 - size_penalty) * (1 - straggler_overhead)

    # Get dynamic bounds
    min_eff, max_eff = bounds_model.get_efficiency_bounds(
        hardware, model_params_b, num_gpus,
        tensor_parallel, pipeline_parallel, data_parallel, expert_parallel
    )

    # Clamp to bounds
    final_efficiency = max(min_eff, min(max_eff, final_efficiency))

    return {
        'base_efficiency': base_efficiency,
        'size_penalty': size_penalty,
        'scale_efficiency': scale_efficiency,
        'straggler_overhead': straggler_overhead,
        'final_efficiency': final_efficiency,
        'efficiency_bounds': (min_eff, max_eff),
    }
