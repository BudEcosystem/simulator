"""
Training Time Estimator.

Estimates training time and cost based on hardware and workload.
"""

from typing import Dict, Any, Optional

from ..hardware import HARDWARE_CONFIGS
from .types import TrainingTimeEstimate


class TrainingTimeEstimator:
    """
    Estimate training time and cost.

    Features:
    - Tokens per second estimation
    - Training step calculation
    - Cost estimation
    - MFU (Model FLOPs Utilization) calculation
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

    # Baseline TPS for different model sizes (on A100 80GB)
    # Based on empirical measurements from training runs
    BASELINE_TPS = {
        'A100_80GB_GPU': {
            '1B': 15000,
            '7B': 4000,
            '8B': 3500,
            '13B': 2000,
            '70B': 400,
        },
        'H100_GPU': {
            '1B': 30000,
            '7B': 8000,
            '8B': 7000,
            '13B': 4000,
            '70B': 900,
        },
    }

    def __init__(self):
        """Initialize the time estimator."""
        pass

    def estimate_training_time(
        self,
        model_config: Dict[str, Any],
        dataset_tokens: int,
        batch_size: int,
        gradient_accumulation: int,
        epochs: float,
        hardware: str,
        num_gpus: int,
        seq_length: int = 2048,
        parallelism: Optional[Dict[str, int]] = None,
    ) -> TrainingTimeEstimate:
        """
        Estimate training time and cost.

        Args:
            model_config: Model configuration
            dataset_tokens: Total tokens in dataset
            batch_size: Per-device batch size
            gradient_accumulation: Gradient accumulation steps
            epochs: Number of training epochs
            hardware: Hardware type
            num_gpus: Number of GPUs
            seq_length: Sequence length
            parallelism: Parallelism strategy {"tp": X, "pp": Y, "dp": Z}

        Returns:
            TrainingTimeEstimate
        """
        # Validate inputs
        self._validate_inputs(dataset_tokens, hardware)

        # Get hardware config
        hw_config = HARDWARE_CONFIGS.get(hardware)
        if not hw_config:
            raise ValueError(f"Unknown hardware: {hardware}")

        # Get model parameters
        total_params = self._get_total_params(model_config)

        # Calculate parallelism
        if parallelism is None:
            parallelism = {"tp": 1, "pp": 1, "dp": num_gpus}

        dp = parallelism.get('dp', num_gpus)

        # Calculate effective batch size
        effective_batch = batch_size * gradient_accumulation * dp

        # Calculate total training steps
        total_tokens = int(dataset_tokens * epochs)
        tokens_per_step = effective_batch * seq_length
        # Ensure at least 1 step for small datasets
        total_steps = max(1, total_tokens // tokens_per_step)

        # Estimate tokens per second
        tps = self._estimate_tps(
            model_config=model_config,
            hw_config=hw_config,
            hardware=hardware,
            num_gpus=num_gpus,
            batch_size=batch_size,
            parallelism=parallelism,
        )

        # Calculate training time
        estimated_seconds = total_tokens / tps
        estimated_hours = estimated_seconds / 3600

        # Calculate cost
        cost_per_hour = self.GPU_COST_PER_HOUR.get(hardware, 3.0) * num_gpus
        estimated_cost = cost_per_hour * estimated_hours

        # Calculate MFU
        mfu = self._calculate_mfu(
            tps=tps,
            total_params=total_params,
            hw_config=hw_config,
            num_gpus=num_gpus,
            batch_size=batch_size,
            seq_length=seq_length,
        )

        # Calculate FLOPs per step
        flops_per_step = 6 * total_params * effective_batch * seq_length

        return TrainingTimeEstimate(
            total_steps=total_steps,
            tokens_per_second=tps,
            estimated_hours=estimated_hours,
            estimated_cost=estimated_cost,
            hardware=hardware,
            num_gpus=num_gpus,
            parallelism=parallelism,
            model_flops_utilization=mfu,
            flops_per_step=flops_per_step,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            seq_length=seq_length,
            epochs=epochs,
        )

    def _validate_inputs(
        self,
        dataset_tokens: int,
        hardware: str,
    ) -> None:
        """Validate input parameters."""
        if dataset_tokens <= 0:
            raise ValueError("dataset_tokens must be positive")

        if hardware not in HARDWARE_CONFIGS:
            available = list(HARDWARE_CONFIGS.keys())
            raise ValueError(
                f"Unknown hardware: '{hardware}'. "
                f"Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}"
            )

    def _get_total_params(self, config: Dict[str, Any]) -> int:
        """Get total parameter count from config."""
        if 'num_parameters' in config:
            return config['num_parameters']

        # Estimate from config
        hidden = config.get('hidden_size', 4096)
        layers = config.get('num_hidden_layers', 32)
        vocab = config.get('vocab_size', 50000)
        intermediate = config.get('intermediate_size', hidden * 4)

        # Rough estimate: embedding + layers × (attention + FFN) + output
        embedding_params = vocab * hidden
        attention_params = 4 * hidden * hidden  # Q, K, V, O
        ffn_params = 3 * hidden * intermediate  # up, gate, down (SwiGLU)
        layer_params = attention_params + ffn_params

        return embedding_params + layers * layer_params + vocab * hidden

    def _estimate_tps(
        self,
        model_config: Dict[str, Any],
        hw_config: Dict[str, Any],
        hardware: str,
        num_gpus: int,
        batch_size: int,
        parallelism: Dict[str, int],
    ) -> float:
        """Estimate tokens per second for training."""
        total_params = self._get_total_params(model_config)

        # Get hardware performance
        peak_tflops = hw_config.get('Flops', 312)
        memory_bw = hw_config.get('Memory_BW', 2000)

        # Get baseline TPS if available
        size_category = self._get_size_category(total_params)
        baseline_hw = self.BASELINE_TPS.get(hardware, self.BASELINE_TPS.get('A100_80GB_GPU', {}))
        baseline_tps = baseline_hw.get(size_category, 0)

        if baseline_tps > 0:
            # Use empirical baseline
            tps = baseline_tps * num_gpus
        else:
            # Calculate from first principles
            # Training FLOPs per token: ~6 × params
            flops_per_token = 6 * total_params

            # Efficiency: typically 30-50% of peak
            efficiency = 0.35

            # Compute bound TPS
            effective_tflops = peak_tflops * efficiency * num_gpus
            tps = (effective_tflops * 1e12) / flops_per_token

        # Apply parallelism overhead
        dp = int(parallelism.get('dp', num_gpus))
        tp = int(parallelism.get('tp', 1))
        pp = int(parallelism.get('pp', 1))

        # DP communication overhead (~10% per rank after 1)
        if dp > 1:
            # Use log2 approximation for communication overhead
            dp_ranks = dp.bit_length() - 1  # log2 of dp
            tps *= 0.9 ** dp_ranks

        # TP communication overhead (~5% per rank)
        if tp > 1:
            tp_ranks = tp.bit_length() - 1  # log2 of tp
            tps *= 0.95 ** tp_ranks

        # PP adds pipeline bubbles (~15% overhead)
        if pp > 1:
            tps *= 0.85

        # Scale with batch size (larger batches are more efficient)
        if batch_size >= 8:
            tps *= 1.1
        elif batch_size <= 1:
            tps *= 0.8

        return max(100, tps)  # Minimum 100 TPS

    def _get_size_category(self, params: int) -> str:
        """Get size category for baseline lookup."""
        billions = params / 1e9
        if billions < 2:
            return '1B'
        elif billions < 10:
            return '7B' if billions < 8 else '8B'
        elif billions < 20:
            return '13B'
        else:
            return '70B'

    def _calculate_mfu(
        self,
        tps: float,
        total_params: int,
        hw_config: Dict[str, Any],
        num_gpus: int,
        batch_size: int,
        seq_length: int,
    ) -> float:
        """Calculate Model FLOPs Utilization (MFU)."""
        peak_tflops = hw_config.get('Flops', 312) * num_gpus

        # Actual FLOPs per second
        # Training: ~6 FLOPs per parameter per token
        flops_per_token = 6 * total_params
        actual_tflops = (tps * flops_per_token) / 1e12

        # MFU = actual / peak
        mfu = actual_tflops / peak_tflops if peak_tflops > 0 else 0

        return min(1.0, max(0.0, mfu))
