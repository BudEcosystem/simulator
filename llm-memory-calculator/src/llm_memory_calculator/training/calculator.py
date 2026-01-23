"""
Training Memory Calculator.

Calculates memory requirements for LLM training including:
- Weight memory
- Gradient memory
- Optimizer state memory
- Activation memory
- Framework overhead
"""

import math
from typing import Dict, Any, Optional

from ..calculator import ModelMemoryCalculator
from ..parameter_counter import UniversalParameterCounter
from .types import (
    TrainingMethod,
    OptimizerType,
    DeepSpeedStage,
    TrainingMemoryEstimate,
)


class TrainingMemoryCalculator:
    """
    Calculate training memory requirements for LLM fine-tuning.

    Extends inference memory calculation with:
    - Gradient memory for trainable parameters
    - Optimizer state memory (momentum, variance, etc.)
    - Activation memory with gradient checkpointing support
    - DeepSpeed ZeRO sharding effects
    """

    # Precision to bytes mapping
    PRECISION_BYTES = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
        'int4': 0.5, 'uint4': 0.5,
        'nf4': 0.5,  # NormalFloat4 (QLoRA)
    }

    # Optimizer state multipliers (number of states per param)
    OPTIMIZER_STATE_MULTIPLIERS = {
        'adamw': 2,       # momentum + variance
        'adam': 2,
        'adamw_8bit': 2,  # 2 states in 8-bit each
        'sgd': 1,         # only momentum
        'galore': 0.25,   # low-rank approximation
        'apollo': 0.25,   # low-rank
        'adafactor': 0.5, # factored states
        'lion': 1,        # only momentum
    }

    # Cost per GPU-hour (approximate cloud pricing)
    GPU_COST_PER_HOUR = {
        'A100_40GB_GPU': 2.21,
        'A100_80GB_GPU': 3.67,
        'H100_GPU': 4.76,
        'V100_16GB_GPU': 0.90,
        'V100_32GB_GPU': 1.21,
        'L40S_48GB_GPU': 1.98,
        'MI300X': 3.50,
        'TPUv5e': 2.50,
        'TPUv5p': 4.50,
    }

    def __init__(self):
        """Initialize the training calculator."""
        self.param_counter = UniversalParameterCounter()
        self.inference_calculator = ModelMemoryCalculator()

    def calculate_training_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str = "bf16",
        method: str = "lora",
        optimizer: str = "adamw",
        gradient_checkpointing: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        freeze_layers: int = 0,
        deepspeed_stage: Optional[str] = None,
        tensor_parallel: int = 1,
        data_parallel: int = 1,
        framework_overhead_percent: float = 10.0,
    ) -> TrainingMemoryEstimate:
        """
        Calculate complete training memory requirements.

        Args:
            config: Model configuration dictionary
            batch_size: Per-device batch size
            seq_length: Sequence length
            precision: Weight precision (fp32, bf16, fp16, int4, etc.)
            method: Training method (full, lora, qlora, freeze)
            optimizer: Optimizer type
            gradient_checkpointing: Enable gradient checkpointing
            lora_rank: LoRA rank (if using LoRA)
            lora_alpha: LoRA alpha (if using LoRA)
            freeze_layers: Number of layers to freeze (if using freeze)
            deepspeed_stage: DeepSpeed ZeRO stage (zero2, zero3)
            tensor_parallel: Tensor parallelism degree
            data_parallel: Data parallelism degree
            framework_overhead_percent: Additional overhead percentage

        Returns:
            TrainingMemoryEstimate with complete breakdown
        """
        # Validate inputs
        self._validate_inputs(batch_size, seq_length, method, optimizer)

        # Get model parameters
        total_params = self._get_total_params(config)

        # Calculate trainable parameters based on method
        trainable_params = self._calculate_trainable_params(
            config, method, lora_rank, freeze_layers, total_params
        )

        # Calculate weight memory (divided by TP)
        weight_memory = self._calculate_weight_memory(
            config, precision, tensor_parallel, deepspeed_stage, data_parallel
        )

        # Calculate gradient memory
        gradient_memory = self._calculate_gradient_memory(
            trainable_params, deepspeed_stage, data_parallel
        )

        # Calculate optimizer memory
        optimizer_memory = self._calculate_optimizer_memory(
            trainable_params, optimizer, deepspeed_stage, data_parallel
        )

        # Calculate activation memory
        activation_memory = self._calculate_activation_memory(
            config, batch_size, seq_length, precision,
            gradient_checkpointing, tensor_parallel
        )

        # Calculate total with overhead
        components_sum = (
            weight_memory + gradient_memory +
            optimizer_memory + activation_memory
        )
        total_memory = components_sum * (1 + framework_overhead_percent / 100)

        return TrainingMemoryEstimate(
            weight_memory_gb=weight_memory,
            gradient_memory_gb=gradient_memory,
            optimizer_memory_gb=optimizer_memory,
            activation_memory_gb=activation_memory,
            total_memory_gb=total_memory,
            trainable_params=trainable_params,
            total_params=total_params,
            method=method,
            optimizer=optimizer,
            precision=precision,
            batch_size=batch_size,
            seq_length=seq_length,
            lora_rank=lora_rank if method in ('lora', 'qlora') else None,
            gradient_checkpointing=gradient_checkpointing,
            deepspeed_stage=deepspeed_stage,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            framework_overhead_percent=framework_overhead_percent,
        )

    def _validate_inputs(
        self,
        batch_size: int,
        seq_length: int,
        method: str,
        optimizer: str,
    ) -> None:
        """Validate input parameters."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")

        valid_methods = {'full', 'lora', 'qlora', 'freeze', 'dora', 'pissa'}
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid training method: {method}. "
                f"Valid methods: {valid_methods}"
            )

        valid_optimizers = set(self.OPTIMIZER_STATE_MULTIPLIERS.keys())
        if optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {optimizer}. "
                f"Valid optimizers: {valid_optimizers}"
            )

    def _get_total_params(self, config: Dict[str, Any]) -> int:
        """Get total parameter count from config."""
        if 'num_parameters' in config:
            return config['num_parameters']
        return self.param_counter.count_parameters(config)

    def _calculate_trainable_params(
        self,
        config: Dict[str, Any],
        method: str,
        lora_rank: int,
        freeze_layers: int,
        total_params: int,
    ) -> int:
        """Calculate trainable parameters based on method."""
        method = method.lower()

        if method == 'full':
            return total_params

        elif method in ('lora', 'qlora', 'dora', 'pissa'):
            # LoRA params: 2 × rank × (hidden_size + intermediate_size/3) × num_layers × num_adapters
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', 32)

            # Typical LoRA targets: q_proj, k_proj, v_proj, o_proj + up/down projections
            # Each adapter: rank × in_dim + rank × out_dim ≈ 2 × rank × hidden
            num_adapters = 4  # Q, K, V, O projections
            lora_params_per_layer = num_adapters * 2 * lora_rank * hidden_size

            # Add MLP adapters (optional, depends on config)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            mlp_adapters = 2  # gate_proj, up_proj typically
            mlp_params_per_layer = mlp_adapters * 2 * lora_rank * (hidden_size + intermediate_size) // 2

            return num_layers * (lora_params_per_layer + mlp_params_per_layer)

        elif method == 'freeze':
            # Only train last N layers
            num_layers = config.get('num_hidden_layers', 32)
            unfrozen_layers = num_layers - freeze_layers
            if unfrozen_layers <= 0:
                unfrozen_layers = 1

            # Approximate: each layer has roughly equal params
            params_per_layer = total_params / num_layers
            return int(unfrozen_layers * params_per_layer)

        return total_params

    def _calculate_weight_memory(
        self,
        config: Dict[str, Any],
        precision: str,
        tensor_parallel: int,
        deepspeed_stage: Optional[str],
        data_parallel: int,
    ) -> float:
        """Calculate weight memory in GB."""
        total_params = self._get_total_params(config)
        bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)

        # Base weight memory
        weight_memory = (total_params * bytes_per_param) / 1e9

        # Divide by tensor parallelism
        weight_memory /= tensor_parallel

        # ZeRO-3 shards weights across DP ranks
        if deepspeed_stage == 'zero3':
            weight_memory /= data_parallel

        return weight_memory

    def _calculate_gradient_memory(
        self,
        trainable_params: int,
        deepspeed_stage: Optional[str],
        data_parallel: int,
    ) -> float:
        """
        Calculate gradient memory in GB.

        Gradients are typically stored in fp32 for numerical stability.
        """
        # Gradients in fp32 (4 bytes per param)
        gradient_memory = (trainable_params * 4) / 1e9

        # ZeRO-3 shards gradients across DP ranks
        if deepspeed_stage == 'zero3':
            gradient_memory /= data_parallel

        return gradient_memory

    def _calculate_optimizer_memory(
        self,
        trainable_params: int,
        optimizer: str,
        deepspeed_stage: Optional[str],
        data_parallel: int,
    ) -> float:
        """
        Calculate optimizer state memory in GB.

        AdamW: 2 states (momentum + variance) in fp32
        SGD: 1 state (momentum) in fp32
        8-bit optimizers: reduced precision states
        GaLore/Apollo: low-rank approximation
        """
        optimizer = optimizer.lower()
        multiplier = self.OPTIMIZER_STATE_MULTIPLIERS.get(optimizer, 2)

        # Optimizer states in fp32 (4 bytes per param per state)
        if optimizer == 'adamw_8bit':
            # 8-bit states: 1 byte per param per state
            bytes_per_param = 1 * multiplier
        else:
            bytes_per_param = 4 * multiplier

        optimizer_memory = (trainable_params * bytes_per_param) / 1e9

        # ZeRO-2 and ZeRO-3 shard optimizer states
        if deepspeed_stage in ('zero2', 'zero3'):
            optimizer_memory /= data_parallel

        return optimizer_memory

    def _calculate_activation_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str,
        gradient_checkpointing: bool,
        tensor_parallel: int,
    ) -> float:
        """
        Calculate activation memory in GB.

        Without checkpointing: Store all layer activations
        With checkpointing: Store sqrt(num_layers) checkpoints
        """
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_hidden_layers', 32)
        intermediate_size = config.get('intermediate_size', hidden_size * 4)
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)

        # Activations per layer:
        # - Hidden states: batch × seq × hidden
        # - Attention: batch × heads × seq × seq (for attention scores)
        # - FFN: batch × seq × intermediate

        # Approximate total activations per layer
        attention_elements = batch_size * seq_length * hidden_size  # Q, K, V, output
        attention_scores = batch_size * config.get('num_attention_heads', 32) * seq_length * seq_length
        ffn_elements = batch_size * seq_length * intermediate_size

        elements_per_layer = attention_elements * 4 + attention_scores + ffn_elements * 2

        if gradient_checkpointing:
            # With checkpointing, only store sqrt(n) layers worth of activations
            # Plus recomputation overhead during backward pass
            effective_layers = math.sqrt(num_layers)
            # Add some overhead for checkpoint boundaries
            effective_layers = max(effective_layers, 2)
        else:
            effective_layers = num_layers

        total_elements = elements_per_layer * effective_layers
        activation_memory = (total_elements * bytes_per_element) / 1e9

        # Divide by tensor parallelism (attention is split)
        activation_memory /= tensor_parallel

        return activation_memory


# Convenience function
def calculate_training_memory(
    model_id_or_config,
    batch_size: int = 4,
    seq_length: int = 2048,
    **kwargs,
) -> TrainingMemoryEstimate:
    """
    Convenience function to calculate training memory.

    Args:
        model_id_or_config: HuggingFace model ID or config dict
        batch_size: Per-device batch size
        seq_length: Sequence length
        **kwargs: Additional arguments for calculate_training_memory

    Returns:
        TrainingMemoryEstimate
    """
    from ..huggingface_loader import HuggingFaceConfigLoader

    calculator = TrainingMemoryCalculator()

    if isinstance(model_id_or_config, str):
        loader = HuggingFaceConfigLoader()
        config = loader.get_model_config(model_id_or_config)
    else:
        config = model_id_or_config

    return calculator.calculate_training_memory(
        config=config,
        batch_size=batch_size,
        seq_length=seq_length,
        **kwargs,
    )
