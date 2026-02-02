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
        'RTX4090_GPU': 0.50,
        'RTX4080_GPU': 0.40,
        'RTX4070Ti_GPU': 0.35,
        'RTX4070_GPU': 0.30,
        'RTX3090_GPU': 0.45,
        'RTX3080Ti_GPU': 0.40,
        'RTX3080_GPU': 0.35,
        'RTX3070_GPU': 0.25,
        'RTX3060_GPU': 0.20,
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
        'RTX4090_GPU': {
            '1B': 12000,
            '7B': 850,
            '8B': 750,
            '13B': 400,
            '70B': 80,
        },
        'RTX3090_GPU': {
            '1B': 7000,
            '7B': 450,
            '8B': 400,
            '13B': 200,
            '70B': 40,
        },
        'RTX3080_GPU': {
            '1B': 5500,
            '7B': 350,
            '8B': 300,
            '13B': 150,
            '70B': 30,
        },
        'RTX3070_GPU': {
            '1B': 4000,
            '7B': 250,
            '8B': 220,
            '13B': 100,
            '70B': 20,
        },
        'RTX3060_GPU': {
            '1B': 3000,
            '7B': 180,
            '8B': 160,
            '13B': 80,
            '70B': 15,
        },
        'RTX4080_GPU': {
            '1B': 10000,
            '7B': 700,
            '8B': 620,
            '13B': 350,
            '70B': 65,
        },
        'RTX4070Ti_GPU': {
            '1B': 8000,
            '7B': 550,
            '8B': 480,
            '13B': 280,
            '70B': 50,
        },
        'RTX4070_GPU': {
            '1B': 6500,
            '7B': 450,
            '8B': 400,
            '13B': 220,
            '70B': 40,
        },
    }

    # Per-GPU Model FLOPs Utilization (MFU) for first-principles estimation
    GPU_MFU = {
        'H100_GPU': 0.50,
        'GH200_GPU': 0.50,
        'B100': 0.50,
        'GB200': 0.50,
        'A100_80GB_GPU': 0.47,
        'A100_40GB_GPU': 0.47,
        'L40S_48GB_GPU': 0.35,
        'RTX4090_GPU': 0.30,
        'RTX4080_GPU': 0.28,
        'RTX4070Ti_GPU': 0.25,
        'RTX4070_GPU': 0.23,
        'RTX3090_GPU': 0.18,
        'RTX3080_GPU': 0.16,
        'RTX3080Ti_GPU': 0.17,
        'RTX3070_GPU': 0.15,
        'RTX3060_GPU': 0.12,
        'MI300X': 0.40,
        'MI325X': 0.40,
        'TPUv4': 0.45,
        'TPUv5e': 0.45,
        'TPUv5p': 0.50,
        'TPUv6': 0.50,
        'V100_16GB_GPU': 0.30,
        'V100_32GB_GPU': 0.30,
        'Gaudi3': 0.35,
    }

    # Optimizer compute overhead factors
    OPTIMIZER_OVERHEAD = {
        'adamw': 1.0,
        'adam': 1.0,
        'adamw_8bit': 1.05,
        'paged_adamw_8bit': 1.05,
        'sgd': 0.95,
        'adafactor': 1.02,
        'lion': 0.98,
        'galore': 1.15,
        'galore_8bit': 1.18,
        'apollo': 1.12,
        'adam_mini': 1.03,
        'muon': 1.08,
        'lomo': 0.90,
        'adalomo': 0.95,
        'badam_layer': 1.0,
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
        optimizer: str = "adamw",
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

        # Apply optimizer compute overhead
        opt_overhead = self.OPTIMIZER_OVERHEAD.get(optimizer.lower(), 1.0)
        tps /= opt_overhead

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

            # Use per-GPU MFU efficiency
            efficiency = self.GPU_MFU.get(hardware, 0.35)

            # Compute bound TPS
            effective_tflops = peak_tflops * efficiency * num_gpus
            tps = (effective_tflops * 1e12) / flops_per_token

        # Apply parallelism overhead
        dp = int(parallelism.get('dp', num_gpus))
        tp = int(parallelism.get('tp', 1))
        pp = int(parallelism.get('pp', 1))

        # Multi-GPU scaling with interconnect-aware penalties
        if dp > 1:
            dp_log2 = max(1, dp.bit_length() - 1)
            # Check interconnect type from hardware config
            interconnect = hw_config.get('interconnect', 'pcie4')
            if interconnect == 'nvlink' or hw_config.get('ICN', 0) >= 300:
                tps *= 0.92 ** dp_log2  # NVLink: better scaling
            else:
                tps *= 0.70 ** dp_log2  # PCIe: worse scaling

        # TP communication overhead (~5% per rank)
        if tp > 1:
            tp_log2 = max(1, tp.bit_length() - 1)
            tps *= 0.95 ** tp_log2

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
