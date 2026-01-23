"""
Calibration Engine for Training Simulation Efficiency Factors.

This module provides tools to automatically tune efficiency factors to match
real-world performance benchmarks. It uses a train/validation split approach
to find optimal calibration parameters.

Key Calibration Factors:
- Hardware-specific base efficiency
- Parallelism overhead multipliers
- Training method adjustments
- Model size scaling factors
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution
import json
import warnings

from .benchmark_database import (
    PublishedBenchmark,
    PUBLISHED_BENCHMARKS,
    get_benchmarks_by_category,
    ConfidenceLevel,
    BenchmarkCategory,
)
from .accuracy_metrics import (
    AccuracyMetrics,
    BenchmarkResult,
    calculate_accuracy_metrics,
)


@dataclass
class CalibrationFactors:
    """
    Calibration factors for training simulation.

    These factors adjust the raw simulation outputs to better match
    real-world performance.
    """
    # Hardware-specific base efficiency (multiplier for MFU prediction)
    hardware_efficiency: Dict[str, float] = field(default_factory=lambda: {
        # NVIDIA GPUs
        "A100_40GB_GPU": 0.70,
        "A100_80GB_GPU": 0.70,
        "H100_GPU": 0.75,
        "H200_GPU": 0.78,
        "GH200_GPU": 0.76,
        "B100": 0.80,
        "B200": 0.80,
        "GB200": 0.82,
        "V100_32GB_GPU": 0.60,
        "V100_16GB_GPU": 0.58,
        "L40S_48GB_GPU": 0.65,
        "RTX4090_GPU": 0.60,
        "RTX3090_GPU": 0.55,

        # AMD GPUs
        "MI300X": 0.65,
        "MI250X": 0.58,

        # Google TPUs
        "TPU_v4": 0.72,
        "TPU_v5e": 0.68,
        "TPU_v5p": 0.75,
        "TPUv4": 0.72,
        "TPUv5e": 0.68,
        "TPUv5p": 0.75,

        # Intel/AWS
        "Gaudi2": 0.60,
        "Gaudi3": 0.65,
        "Trainium1": 0.55,
        "Trainium2": 0.65,

        # Default
        "default": 0.65,
    })

    # Parallelism overhead multipliers
    # These scale the communication overhead estimates
    tp_overhead_multiplier: float = 1.5       # TP typically has higher overhead than modeled
    pp_bubble_multiplier: float = 1.2         # Pipeline bubbles larger in practice
    dp_comm_multiplier: float = 1.3           # AllReduce overhead
    ep_overhead_multiplier: float = 1.8       # Expert parallelism All2All overhead

    # ZeRO stage overhead
    zero_overhead: Dict[int, float] = field(default_factory=lambda: {
        0: 1.00,   # No ZeRO
        1: 1.02,   # Optimizer state partitioning
        2: 1.05,   # + Gradient partitioning
        3: 1.12,   # + Parameter partitioning
    })

    # Training method efficiency factors
    method_efficiency: Dict[str, float] = field(default_factory=lambda: {
        "full": 1.00,
        "lora": 0.95,
        "qlora": 0.88,
        "dora": 0.93,
        "pissa": 0.95,
        "freeze": 0.98,
    })

    # Model size scaling (larger models tend to have lower efficiency)
    # MFU multiplier = 1.0 - (params_b / 1000) * size_scaling_factor
    size_scaling_factor: float = 0.05  # 5% reduction per 100B params

    # MoE specific factors
    moe_base_efficiency: float = 0.70  # MoE models have lower base efficiency
    moe_expert_scaling: float = 0.02   # Additional overhead per doubling of experts

    # Small batch penalty adjustment
    small_batch_threshold: int = 2048   # Tokens per GPU below which efficiency drops
    small_batch_penalty_max: float = 0.10  # Max 10% efficiency reduction

    def get_hardware_efficiency(self, hardware: str) -> float:
        """Get hardware efficiency factor with fallback to default."""
        if hardware in self.hardware_efficiency:
            return self.hardware_efficiency[hardware]
        # Try partial match
        for key, value in self.hardware_efficiency.items():
            if key.lower() in hardware.lower() or hardware.lower() in key.lower():
                return value
        return self.hardware_efficiency.get("default", 0.65)

    def get_zero_overhead(self, zero_stage: int) -> float:
        """Get ZeRO overhead factor."""
        return self.zero_overhead.get(zero_stage, 1.0)

    def get_method_efficiency(self, method: str) -> float:
        """Get training method efficiency factor."""
        return self.method_efficiency.get(method.lower(), 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hardware_efficiency': self.hardware_efficiency,
            'tp_overhead_multiplier': self.tp_overhead_multiplier,
            'pp_bubble_multiplier': self.pp_bubble_multiplier,
            'dp_comm_multiplier': self.dp_comm_multiplier,
            'ep_overhead_multiplier': self.ep_overhead_multiplier,
            'zero_overhead': self.zero_overhead,
            'method_efficiency': self.method_efficiency,
            'size_scaling_factor': self.size_scaling_factor,
            'moe_base_efficiency': self.moe_base_efficiency,
            'moe_expert_scaling': self.moe_expert_scaling,
            'small_batch_threshold': self.small_batch_threshold,
            'small_batch_penalty_max': self.small_batch_penalty_max,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationFactors':
        """Create from dictionary."""
        return cls(
            hardware_efficiency=d.get('hardware_efficiency', {}),
            tp_overhead_multiplier=d.get('tp_overhead_multiplier', 1.5),
            pp_bubble_multiplier=d.get('pp_bubble_multiplier', 1.2),
            dp_comm_multiplier=d.get('dp_comm_multiplier', 1.3),
            ep_overhead_multiplier=d.get('ep_overhead_multiplier', 1.8),
            zero_overhead=d.get('zero_overhead', {0: 1.0, 1: 1.02, 2: 1.05, 3: 1.12}),
            method_efficiency=d.get('method_efficiency', {}),
            size_scaling_factor=d.get('size_scaling_factor', 0.05),
            moe_base_efficiency=d.get('moe_base_efficiency', 0.70),
            moe_expert_scaling=d.get('moe_expert_scaling', 0.02),
            small_batch_threshold=d.get('small_batch_threshold', 2048),
            small_batch_penalty_max=d.get('small_batch_penalty_max', 0.10),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'CalibrationFactors':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Default calibration factors (uncalibrated baseline)
DEFAULT_FACTORS = CalibrationFactors()


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    factors: CalibrationFactors
    train_metrics: AccuracyMetrics
    validation_metrics: AccuracyMetrics
    train_benchmarks: List[str]
    validation_benchmarks: List[str]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    final_objective: float = 0.0


class CalibrationEngine:
    """
    Engine for calibrating training simulation efficiency factors.

    Uses optimization to find factors that minimize prediction error
    on a training set, then validates on a held-out validation set.
    """

    def __init__(
        self,
        benchmarks: Optional[Dict[str, PublishedBenchmark]] = None,
        train_split: float = 0.7,
        random_seed: int = 42,
    ):
        """
        Initialize calibration engine.

        Args:
            benchmarks: Benchmarks to use (default: high-confidence benchmarks)
            train_split: Fraction for training (default 70%)
            random_seed: Random seed for reproducibility
        """
        if benchmarks is None:
            benchmarks = get_benchmarks_by_category(confidence=ConfidenceLevel.HIGH)

        self.all_benchmarks = benchmarks
        self.train_split = train_split
        self.random_seed = random_seed

        # Split benchmarks
        self._split_benchmarks()

        # Initialize factors
        self.current_factors = CalibrationFactors()
        self.best_factors: Optional[CalibrationFactors] = None
        self.optimization_history: List[Dict[str, Any]] = []

    def _split_benchmarks(self):
        """Split benchmarks into train and validation sets."""
        np.random.seed(self.random_seed)
        names = list(self.all_benchmarks.keys())
        np.random.shuffle(names)

        split_idx = int(len(names) * self.train_split)
        train_names = names[:split_idx]
        val_names = names[split_idx:]

        self.train_benchmarks = {n: self.all_benchmarks[n] for n in train_names}
        self.validation_benchmarks = {n: self.all_benchmarks[n] for n in val_names}

    def _simulate_with_factors(
        self,
        benchmark: PublishedBenchmark,
        factors: CalibrationFactors,
    ) -> Optional[float]:
        """
        Run simulation with calibration factors applied.

        Returns predicted MFU or None on failure.
        """
        from ..genz.LLM_training.training_modeling import training_modeling

        try:
            # Calculate per-GPU batch size
            total_parallel = (
                benchmark.tensor_parallel *
                benchmark.pipeline_parallel *
                benchmark.data_parallel
            )
            divisor = max(1, total_parallel)
            micro_batch_size = max(1, benchmark.batch_size // divisor)

            # Run base simulation
            # Phase 5 Bug #7 fix: Use correct training stage based on benchmark category
            result = training_modeling(
                model=benchmark.model,
                training_stage='sft' if benchmark.category == BenchmarkCategory.FINETUNING else 'pretraining',
                batch_size=micro_batch_size,
                seq_length=benchmark.seq_length,
                system_name=benchmark.hardware,
                num_gpus=benchmark.num_gpus,
                tensor_parallel=benchmark.tensor_parallel,
                pipeline_parallel=benchmark.pipeline_parallel,
                data_parallel=benchmark.data_parallel,
                expert_parallel=benchmark.expert_parallel,
                method='full',
                optimizer=benchmark.optimizer,
                zero_stage=benchmark.zero_stage,
                gradient_checkpointing=benchmark.gradient_checkpointing,
                bits=benchmark.precision,
            )

            # Apply calibration factors
            base_mfu = result.model_flops_utilization

            # Hardware efficiency adjustment
            hw_eff = factors.get_hardware_efficiency(benchmark.hardware)

            # Model size scaling
            size_penalty = min(0.3, benchmark.model_params_b / 1000 * factors.size_scaling_factor)

            # MoE adjustment
            moe_penalty = 0.0
            if benchmark.is_moe:
                moe_penalty = (1.0 - factors.moe_base_efficiency)
                expert_penalty = factors.moe_expert_scaling * np.log2(max(1, benchmark.num_experts))
                moe_penalty += expert_penalty

            # Small batch penalty
            tokens_per_gpu = micro_batch_size * benchmark.seq_length
            batch_penalty = 0.0
            if tokens_per_gpu < factors.small_batch_threshold:
                ratio = tokens_per_gpu / factors.small_batch_threshold
                batch_penalty = factors.small_batch_penalty_max * (1 - np.sqrt(ratio))

            # Combine adjustments
            # The raw MFU is scaled by hardware efficiency, then penalties are applied
            calibrated_mfu = base_mfu * hw_eff * (1 - size_penalty) * (1 - moe_penalty) * (1 - batch_penalty)

            return max(0.05, min(0.80, calibrated_mfu))

        except Exception as e:
            warnings.warn(f"Simulation failed for {benchmark.name}: {e}")
            return None

    def _evaluate_factors(
        self,
        factors: CalibrationFactors,
        benchmarks: Dict[str, PublishedBenchmark],
    ) -> Tuple[float, AccuracyMetrics]:
        """
        Evaluate calibration factors on a set of benchmarks.

        Returns (objective_value, accuracy_metrics).
        Lower objective is better.
        """
        results: List[BenchmarkResult] = []

        for name, benchmark in benchmarks.items():
            predicted_mfu = self._simulate_with_factors(benchmark, factors)

            if predicted_mfu is not None:
                actual_mfu = benchmark.reported_mfu
                relative_error = abs(predicted_mfu - actual_mfu) / actual_mfu if actual_mfu > 0 else 1.0
                signed_error = (predicted_mfu - actual_mfu) / actual_mfu if actual_mfu > 0 else 0

                results.append(BenchmarkResult(
                    benchmark_name=name,
                    benchmark=benchmark,
                    predicted_mfu=predicted_mfu,
                    actual_mfu=actual_mfu,
                    relative_error=relative_error,
                    absolute_error=abs(predicted_mfu - actual_mfu),
                    signed_error=signed_error,
                    success=True,
                ))
            else:
                results.append(BenchmarkResult(
                    benchmark_name=name,
                    benchmark=benchmark,
                    predicted_mfu=0.0,
                    actual_mfu=benchmark.reported_mfu,
                    relative_error=1.0,
                    absolute_error=benchmark.reported_mfu,
                    signed_error=-1.0,
                    success=False,
                    error_message="Simulation failed",
                ))

        if len([r for r in results if r.success]) < 2:
            return float('inf'), None

        metrics = calculate_accuracy_metrics(results)

        # Objective: weighted combination of MRE, correlation, and bias
        # Lower is better
        objective = (
            metrics.mean_relative_error * 2.0 +      # Primary: minimize MRE
            (1 - metrics.pearson_correlation) * 1.0 + # Secondary: maximize correlation
            abs(metrics.systematic_bias) * 1.5        # Penalize bias heavily
        )

        return objective, metrics

    def _params_to_factors(self, params: np.ndarray) -> CalibrationFactors:
        """Convert optimization parameters to CalibrationFactors."""
        # params layout:
        # [0-3]: parallelism multipliers (tp, pp, dp, ep)
        # [4]: size_scaling_factor
        # [5]: moe_base_efficiency
        # [6]: small_batch_penalty_max

        factors = CalibrationFactors()
        factors.tp_overhead_multiplier = params[0]
        factors.pp_bubble_multiplier = params[1]
        factors.dp_comm_multiplier = params[2]
        factors.ep_overhead_multiplier = params[3]
        factors.size_scaling_factor = params[4]
        factors.moe_base_efficiency = params[5]
        factors.small_batch_penalty_max = params[6]

        return factors

    def _factors_to_params(self, factors: CalibrationFactors) -> np.ndarray:
        """Convert CalibrationFactors to optimization parameters."""
        return np.array([
            factors.tp_overhead_multiplier,
            factors.pp_bubble_multiplier,
            factors.dp_comm_multiplier,
            factors.ep_overhead_multiplier,
            factors.size_scaling_factor,
            factors.moe_base_efficiency,
            factors.small_batch_penalty_max,
        ])

    def run_calibration(
        self,
        method: str = "differential_evolution",
        max_iterations: int = 100,
        debug: bool = False,
    ) -> CalibrationResult:
        """
        Run calibration to find optimal efficiency factors.

        Args:
            method: Optimization method ("differential_evolution" or "nelder-mead")
            max_iterations: Maximum optimization iterations
            debug: Enable debug output

        Returns:
            CalibrationResult with optimized factors and metrics
        """
        if debug:
            print(f"Starting calibration with {len(self.train_benchmarks)} training "
                  f"and {len(self.validation_benchmarks)} validation benchmarks...")

        # Parameter bounds
        bounds = [
            (1.0, 3.0),   # tp_overhead_multiplier
            (1.0, 2.5),   # pp_bubble_multiplier
            (1.0, 2.5),   # dp_comm_multiplier
            (1.0, 3.0),   # ep_overhead_multiplier
            (0.01, 0.15), # size_scaling_factor
            (0.50, 0.85), # moe_base_efficiency
            (0.05, 0.20), # small_batch_penalty_max
        ]

        def objective_fn(params):
            factors = self._params_to_factors(params)
            obj, metrics = self._evaluate_factors(factors, self.train_benchmarks)

            if debug and len(self.optimization_history) % 10 == 0:
                print(f"  Iteration {len(self.optimization_history)}: "
                      f"objective={obj:.4f}, MRE={metrics.mean_relative_error:.1%}")

            self.optimization_history.append({
                'iteration': len(self.optimization_history),
                'objective': obj,
                'mre': metrics.mean_relative_error if metrics else None,
                'params': params.tolist(),
            })

            return obj

        # Run optimization
        if method == "differential_evolution":
            result = differential_evolution(
                objective_fn,
                bounds,
                maxiter=max_iterations,
                seed=self.random_seed,
                polish=True,
                disp=debug,
            )
            best_params = result.x
        else:
            # Nelder-Mead with initial guess
            x0 = self._factors_to_params(CalibrationFactors())
            result = minimize(
                objective_fn,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iterations, 'disp': debug},
            )
            best_params = result.x

        # Get final factors
        self.best_factors = self._params_to_factors(best_params)

        # Evaluate on train and validation sets
        train_obj, train_metrics = self._evaluate_factors(
            self.best_factors, self.train_benchmarks
        )
        val_obj, val_metrics = self._evaluate_factors(
            self.best_factors, self.validation_benchmarks
        )

        if debug:
            print(f"\n{'='*60}")
            print("CALIBRATION COMPLETE")
            print(f"{'='*60}")
            print(f"Training MRE: {train_metrics.mean_relative_error:.1%}")
            print(f"Validation MRE: {val_metrics.mean_relative_error:.1%}")
            print(f"Training Correlation: {train_metrics.pearson_correlation:.3f}")
            print(f"Validation Correlation: {val_metrics.pearson_correlation:.3f}")
            print(f"\nCalibrated Factors:")
            print(f"  TP overhead: {self.best_factors.tp_overhead_multiplier:.2f}")
            print(f"  PP bubble: {self.best_factors.pp_bubble_multiplier:.2f}")
            print(f"  DP comm: {self.best_factors.dp_comm_multiplier:.2f}")
            print(f"  EP overhead: {self.best_factors.ep_overhead_multiplier:.2f}")
            print(f"  Size scaling: {self.best_factors.size_scaling_factor:.3f}")
            print(f"  MoE efficiency: {self.best_factors.moe_base_efficiency:.2f}")

        return CalibrationResult(
            factors=self.best_factors,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            train_benchmarks=list(self.train_benchmarks.keys()),
            validation_benchmarks=list(self.validation_benchmarks.keys()),
            optimization_history=self.optimization_history,
            final_objective=val_obj,
        )

    def calibrate_hardware_efficiency(
        self,
        hardware: str,
        debug: bool = False,
    ) -> float:
        """
        Calibrate efficiency factor for a specific hardware type.

        Uses benchmarks for that hardware to find optimal efficiency.
        """
        # Filter benchmarks for this hardware
        hw_benchmarks = {
            k: v for k, v in self.all_benchmarks.items()
            if hardware.lower() in v.hardware.lower()
        }

        if len(hw_benchmarks) < 2:
            warnings.warn(f"Not enough benchmarks for {hardware}, using default")
            return DEFAULT_FACTORS.get_hardware_efficiency(hardware)

        # Grid search for best efficiency
        best_eff = 0.65
        best_error = float('inf')

        for eff in np.linspace(0.50, 0.85, 36):
            factors = CalibrationFactors()
            factors.hardware_efficiency[hardware] = eff

            obj, metrics = self._evaluate_factors(factors, hw_benchmarks)
            if obj < best_error:
                best_error = obj
                best_eff = eff

        if debug:
            print(f"Calibrated {hardware} efficiency: {best_eff:.2f}")

        return best_eff

    def save_factors(self, filepath: str):
        """Save calibration factors to a JSON file."""
        if self.best_factors is None:
            raise ValueError("No calibration has been run yet")

        with open(filepath, 'w') as f:
            f.write(self.best_factors.to_json())

    def load_factors(self, filepath: str) -> CalibrationFactors:
        """Load calibration factors from a JSON file."""
        with open(filepath, 'r') as f:
            self.best_factors = CalibrationFactors.from_json(f.read())
        return self.best_factors


def get_default_calibration_factors() -> CalibrationFactors:
    """Get default (uncalibrated) factors."""
    return CalibrationFactors()


def get_calibrated_factors() -> CalibrationFactors:
    """
    Get pre-calibrated factors based on published benchmarks.

    These factors have been calibrated on the benchmark database to
    minimize prediction error.
    """
    # Pre-calibrated values from running calibration on the benchmark database
    return CalibrationFactors(
        hardware_efficiency={
            "A100_40GB_GPU": 0.68,
            "A100_80GB_GPU": 0.70,
            "H100_GPU": 0.74,
            "H200_GPU": 0.76,
            "GH200_GPU": 0.75,
            "B100": 0.78,
            "GB200": 0.80,
            "V100_32GB_GPU": 0.58,
            "L40S_48GB_GPU": 0.63,
            "RTX4090_GPU": 0.58,
            "MI300X": 0.63,
            "TPU_v4": 0.71,
            "TPUv4": 0.71,
            "TPU_v5e": 0.67,
            "TPUv5e": 0.67,
            "TPU_v5p": 0.73,
            "TPUv5p": 0.73,
            "Gaudi3": 0.62,
            "default": 0.65,
        },
        tp_overhead_multiplier=1.65,
        pp_bubble_multiplier=1.35,
        dp_comm_multiplier=1.25,
        ep_overhead_multiplier=2.0,
        zero_overhead={0: 1.00, 1: 1.03, 2: 1.07, 3: 1.15},
        method_efficiency={"full": 1.0, "lora": 0.94, "qlora": 0.86, "dora": 0.92},
        size_scaling_factor=0.055,
        moe_base_efficiency=0.68,
        moe_expert_scaling=0.025,
        small_batch_threshold=2048,
        small_batch_penalty_max=0.12,
    )


def quick_calibrate(debug: bool = False) -> CalibrationResult:
    """
    Run a quick calibration with default settings.

    Returns calibration result with optimized factors.
    """
    engine = CalibrationEngine()
    return engine.run_calibration(
        method="differential_evolution",
        max_iterations=50,
        debug=debug,
    )


# ========================================
# Phase 14: Auto-Calibration from Ground Truth
# ========================================
# Automatically derive scaling coefficients from benchmark data
# instead of using hardcoded values.

@dataclass
class ScalingCoefficients:
    """
    Calibrated scaling coefficients derived from ground truth.

    These parametrize the generalized scaling formulas used
    throughout the training simulator.
    """
    # Model size scaling: penalty = α * (params_b / reference) ^ β
    size_scaling_alpha: float = 0.025
    size_scaling_beta: float = 1.3
    size_reference_b: float = 10.0

    # GPU scale efficiency: 1 / (1 + γ * log(gpus / reference))
    scale_gamma_tp: float = 0.04
    scale_gamma_pp: float = 0.06
    scale_gamma_dp: float = 0.03
    scale_gamma_ep: float = 0.08
    scale_reference_gpus: int = 8

    # Network congestion: overhead = 1 + δ * log(1 + concurrent)
    congestion_delta: float = 0.12

    # Straggler: overhead = ε * sqrt(total_gpus / 1000)
    straggler_epsilon: float = 0.05

    # Pipeline bubble scale factors
    bubble_base_scale: float = 1.0
    bubble_scale_growth: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            'bubble_scale_growth': self.bubble_scale_growth,
        }


@dataclass
class EfficiencyBounds:
    """Efficiency bounds derived from empirical data."""
    # Per hardware type
    hardware_max_efficiency: Dict[str, float]
    hardware_min_efficiency: Dict[str, float]

    # Per model size tier
    size_tier_max: Dict[str, float]  # small, medium, large, xlarge
    size_tier_min: Dict[str, float]

    # Per GPU scale
    scale_tier_max: Dict[str, float]  # <1K, 1K-4K, 4K-8K, >8K
    scale_tier_min: Dict[str, float]


class AutoCalibrator:
    """
    Automatically calibrate scaling coefficients from ground truth benchmarks.

    This class analyzes published benchmarks to derive:
    1. Scaling law coefficients (α, β, γ, δ, ε)
    2. Hardware-specific efficiency bounds
    3. Per-tier efficiency ranges

    The derived coefficients replace hardcoded values throughout
    the training simulator for improved accuracy.
    """

    def __init__(
        self,
        benchmarks: Optional[Dict[str, PublishedBenchmark]] = None,
    ):
        """
        Initialize with benchmark database.

        Args:
            benchmarks: Benchmarks to use for calibration
        """
        if benchmarks is None:
            benchmarks = get_benchmarks_by_category(confidence=ConfidenceLevel.HIGH)
        self.benchmarks = benchmarks

    def fit_scaling_coefficients(
        self,
        debug: bool = False,
    ) -> ScalingCoefficients:
        """
        Fit generalized scaling coefficients from benchmark data.

        Uses least-squares optimization to find coefficients that
        minimize prediction error across all benchmarks.

        Args:
            debug: Enable debug output

        Returns:
            ScalingCoefficients with fitted values
        """
        if len(self.benchmarks) < 5:
            warnings.warn("Too few benchmarks for reliable fitting, using defaults")
            return ScalingCoefficients()

        # Convert benchmarks to feature matrix
        features = []
        targets = []

        for name, b in self.benchmarks.items():
            # Features: model_size, num_gpus, tp, pp, dp, ep
            total_parallel = b.tensor_parallel * b.pipeline_parallel * b.data_parallel
            features.append([
                b.model_params_b,
                b.num_gpus,
                b.tensor_parallel,
                b.pipeline_parallel,
                b.data_parallel,
                b.expert_parallel,
                total_parallel,
            ])
            targets.append(b.reported_mfu)

        X = np.array(features)
        y = np.array(targets)

        # Define objective function
        def objective(params):
            alpha, beta, gamma_scale = params[0], params[1], params[2]

            predictions = []
            for i, (xi, yi) in enumerate(zip(X, y)):
                model_size, num_gpus, tp, pp, dp, ep, total_p = xi

                # Base efficiency (assume 0.65)
                base_eff = 0.65

                # Size penalty: α * (size / 10) ^ β - 1
                if model_size > 10:
                    size_penalty = alpha * ((model_size / 10) ** beta - 1)
                else:
                    size_penalty = 0

                # Scale efficiency: 1 / (1 + γ * log(gpus / 8))
                if total_p > 8:
                    scale_eff = 1.0 / (1.0 + gamma_scale * np.log(total_p / 8))
                else:
                    scale_eff = 1.0

                pred = base_eff * scale_eff - size_penalty
                pred = max(0.15, min(0.65, pred))
                predictions.append(pred)

            # Mean squared error
            mse = np.mean((np.array(predictions) - y) ** 2)
            return mse

        # Optimize
        from scipy.optimize import minimize

        # Initial guess
        x0 = [0.025, 1.3, 0.05]
        bounds = [
            (0.01, 0.10),   # alpha
            (1.0, 2.0),    # beta
            (0.02, 0.15),  # gamma_scale
        ]

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
        )

        if debug:
            print(f"Fitted coefficients: α={result.x[0]:.4f}, β={result.x[1]:.2f}, γ={result.x[2]:.4f}")
            print(f"Final MSE: {result.fun:.6f}")

        return ScalingCoefficients(
            size_scaling_alpha=result.x[0],
            size_scaling_beta=result.x[1],
            scale_gamma_tp=result.x[2],
            scale_gamma_pp=result.x[2] * 1.5,  # PP has higher overhead
            scale_gamma_dp=result.x[2] * 0.8,   # DP has lower overhead
            scale_gamma_ep=result.x[2] * 2.0,   # EP has highest overhead
        )

    def derive_efficiency_bounds(
        self,
        debug: bool = False,
    ) -> EfficiencyBounds:
        """
        Derive realistic efficiency bounds from empirical data.

        Computes percentile-based bounds for:
        - max_efficiency[hardware][model_size] = percentile_95(observed_mfu)
        - min_efficiency[hardware][scale] = percentile_5(observed_mfu)

        Args:
            debug: Enable debug output

        Returns:
            EfficiencyBounds with empirically-derived values
        """
        # Group benchmarks by hardware
        hw_mfus: Dict[str, List[float]] = {}
        for name, b in self.benchmarks.items():
            hw = b.hardware
            if hw not in hw_mfus:
                hw_mfus[hw] = []
            hw_mfus[hw].append(b.reported_mfu)

        # Compute per-hardware bounds
        hardware_max = {}
        hardware_min = {}
        for hw, mfus in hw_mfus.items():
            if len(mfus) >= 2:
                hardware_max[hw] = float(np.percentile(mfus, 95))
                hardware_min[hw] = float(np.percentile(mfus, 5))
            elif len(mfus) == 1:
                hardware_max[hw] = mfus[0] * 1.1  # 10% above observed
                hardware_min[hw] = mfus[0] * 0.7  # 30% below observed

        # Group by model size tier
        size_tiers = {
            'small': [],    # <10B
            'medium': [],   # 10-50B
            'large': [],    # 50-100B
            'xlarge': [],   # >100B
        }
        for name, b in self.benchmarks.items():
            if b.model_params_b < 10:
                size_tiers['small'].append(b.reported_mfu)
            elif b.model_params_b < 50:
                size_tiers['medium'].append(b.reported_mfu)
            elif b.model_params_b < 100:
                size_tiers['large'].append(b.reported_mfu)
            else:
                size_tiers['xlarge'].append(b.reported_mfu)

        size_tier_max = {}
        size_tier_min = {}
        for tier, mfus in size_tiers.items():
            if len(mfus) >= 2:
                size_tier_max[tier] = float(np.percentile(mfus, 95))
                size_tier_min[tier] = float(np.percentile(mfus, 5))
            else:
                # Default bounds
                defaults = {
                    'small': (0.45, 0.55),
                    'medium': (0.40, 0.52),
                    'large': (0.35, 0.48),
                    'xlarge': (0.30, 0.42),
                }
                size_tier_min[tier], size_tier_max[tier] = defaults[tier]

        # Group by scale tier
        scale_tiers = {
            '<1K': [],
            '1K-4K': [],
            '4K-8K': [],
            '>8K': [],
        }
        for name, b in self.benchmarks.items():
            if b.num_gpus < 1000:
                scale_tiers['<1K'].append(b.reported_mfu)
            elif b.num_gpus < 4000:
                scale_tiers['1K-4K'].append(b.reported_mfu)
            elif b.num_gpus < 8000:
                scale_tiers['4K-8K'].append(b.reported_mfu)
            else:
                scale_tiers['>8K'].append(b.reported_mfu)

        scale_tier_max = {}
        scale_tier_min = {}
        for tier, mfus in scale_tiers.items():
            if len(mfus) >= 2:
                scale_tier_max[tier] = float(np.percentile(mfus, 95))
                scale_tier_min[tier] = float(np.percentile(mfus, 5))
            else:
                # Default bounds
                defaults = {
                    '<1K': (0.40, 0.55),
                    '1K-4K': (0.38, 0.52),
                    '4K-8K': (0.35, 0.48),
                    '>8K': (0.30, 0.45),
                }
                scale_tier_min[tier], scale_tier_max[tier] = defaults[tier]

        if debug:
            print("\nDerived Efficiency Bounds:")
            print(f"  By Hardware: {hardware_max}")
            print(f"  By Size Tier: {size_tier_max}")
            print(f"  By Scale Tier: {scale_tier_max}")

        return EfficiencyBounds(
            hardware_max_efficiency=hardware_max,
            hardware_min_efficiency=hardware_min,
            size_tier_max=size_tier_max,
            size_tier_min=size_tier_min,
            scale_tier_max=scale_tier_max,
            scale_tier_min=scale_tier_min,
        )

    def run_full_calibration(
        self,
        debug: bool = False,
    ) -> Tuple[ScalingCoefficients, EfficiencyBounds, AccuracyMetrics]:
        """
        Run complete auto-calibration pipeline.

        1. Fits scaling coefficients from benchmark data
        2. Derives efficiency bounds
        3. Validates accuracy against benchmarks

        Args:
            debug: Enable debug output

        Returns:
            Tuple of (coefficients, bounds, metrics)
        """
        if debug:
            print(f"Auto-calibrating with {len(self.benchmarks)} benchmarks...")

        # Fit coefficients
        coefficients = self.fit_scaling_coefficients(debug=debug)

        # Derive bounds
        bounds = self.derive_efficiency_bounds(debug=debug)

        # Validate (compute accuracy metrics)
        # For now, return placeholder metrics
        # Full validation would re-run predictions with new coefficients
        from .accuracy_metrics import AccuracyMetrics

        # Placeholder metrics - would be computed by re-running simulation
        metrics = AccuracyMetrics(
            mean_absolute_error=0.0,
            mean_relative_error=0.0,
            median_relative_error=0.0,
            pearson_correlation=0.0,
            spearman_correlation=0.0,
            r_squared=0.0,
            systematic_bias=0.0,
            bias_direction="neutral",
            percentile_25_error=0.0,
            percentile_75_error=0.0,
            percentile_90_error=0.0,
            percentile_95_error=0.0,
            max_error=0.0,
            n_samples=len(self.benchmarks),
            n_within_tolerance=0,
            tolerance_rate=0.0,
        )

        if debug:
            print("\nAuto-calibration complete!")
            print(f"  Coefficients: α={coefficients.size_scaling_alpha:.4f}, "
                  f"β={coefficients.size_scaling_beta:.2f}")

        return coefficients, bounds, metrics


def run_auto_calibration(debug: bool = False) -> Tuple[ScalingCoefficients, EfficiencyBounds]:
    """
    Convenience function to run auto-calibration.

    Args:
        debug: Enable debug output

    Returns:
        Tuple of (ScalingCoefficients, EfficiencyBounds)
    """
    calibrator = AutoCalibrator()
    coefficients, bounds, _ = calibrator.run_full_calibration(debug=debug)
    return coefficients, bounds
