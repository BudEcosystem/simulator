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
from .optimizers import OPTIMIZER_CONFIGS, get_optimizer_config, ADAMW_BASELINE_BYTES, OPTIMIZER_ALIASES


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

    # Optimizer state multipliers - legacy dict kept for backward compatibility.
    # The actual calculation now uses OPTIMIZER_CONFIGS from optimizers.py
    # which has precise bytes_per_param for all supported optimizers.
    OPTIMIZER_STATE_MULTIPLIERS = {
        name: config.state_count * (config.state_precision_bytes / 4.0)
        for name, config in OPTIMIZER_CONFIGS.items()
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
        # Consumer GPUs (estimated electricity cost for local hardware)
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
        flash_attention: bool = True,
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
            config, precision, tensor_parallel, deepspeed_stage, data_parallel,
            method, trainable_params=trainable_params
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
            gradient_checkpointing, tensor_parallel, flash_attention
        )

        # Peak-of-phases model: memory phases don't fully overlap.
        # Forward pass:  weights + activations
        # Backward pass: weights + activations + gradients  (dominates for LoRA)
        # Optimizer step: weights + gradients + optimizer states (dominates for full FT + checkpointing)
        forward_peak = weight_memory + activation_memory
        backward_peak = weight_memory + activation_memory + gradient_memory
        optimizer_peak = weight_memory + gradient_memory + optimizer_memory
        peak_memory = max(forward_peak, backward_peak, optimizer_peak)
        total_memory = peak_memory * (1 + framework_overhead_percent / 100)

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

        optimizer_lower = optimizer.lower()
        # Accept any optimizer known to OPTIMIZER_CONFIGS or OPTIMIZER_STATE_MULTIPLIERS.
        # For truly unknown optimizers, fall back to AdamW-equivalent memory.
        if optimizer_lower not in OPTIMIZER_CONFIGS and optimizer_lower not in self.OPTIMIZER_STATE_MULTIPLIERS:
            if optimizer_lower not in OPTIMIZER_ALIASES:
                import warnings
                warnings.warn(
                    f"Unknown optimizer '{optimizer}', using AdamW memory estimate. "
                    f"Known optimizers: {sorted(OPTIMIZER_CONFIGS.keys())}"
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
            # LlamaFactory defaults to 7 targets: q,k,v,o projections + gate,up,down MLP
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', 32)
            num_heads = config.get('num_attention_heads', 32)
            num_kv_heads = config.get('num_key_value_heads', num_heads)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)

            # KV dimension for GQA models
            kv_dim = hidden_size * num_kv_heads // num_heads

            # Attention LoRA params per layer:
            # q_proj: 2r(h+h), k_proj: 2r(h+kv_dim), v_proj: 2r(h+kv_dim), o_proj: 2r(h+h)
            attn_params = (
                2 * lora_rank * (hidden_size + hidden_size) +      # q_proj
                2 * lora_rank * (hidden_size + kv_dim) +           # k_proj
                2 * lora_rank * (hidden_size + kv_dim) +           # v_proj
                2 * lora_rank * (hidden_size + hidden_size)        # o_proj
            )

            # MLP LoRA params per layer (gate, up, down projections):
            mlp_params = (
                2 * lora_rank * (hidden_size + intermediate_size) +  # gate_proj
                2 * lora_rank * (hidden_size + intermediate_size) +  # up_proj
                2 * lora_rank * (hidden_size + intermediate_size)    # down_proj
            )

            return num_layers * (attn_params + mlp_params)

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
        method: str = "lora",
        trainable_params: int = 0,
    ) -> float:
        """Calculate weight memory in GB.

        Args:
            trainable_params: Number of trainable parameters. Used to compute
                fp32 master copy size for LoRA-family methods (only adapter
                params need a master copy, not the frozen base model).
        """
        total_params = self._get_total_params(config)

        if method.lower() == 'qlora':
            # QLoRA: NF4 + double quantization ≈ 0.516 bytes per param
            weight_memory = (total_params * 0.516) / 1e9
            # Dequantization buffer: only ONE layer dequantized at a time
            hidden_size = config.get('hidden_size', 4096)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            dequant_buffer = hidden_size * intermediate_size * 2 / 1e9
            weight_memory += dequant_buffer
            # fp32 master copy for LoRA adapter params only
            if trainable_params > 0:
                weight_memory += (trainable_params * 4) / 1e9
        else:
            bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)
            weight_memory = (total_params * bytes_per_param) / 1e9

            # fp32 master weight copy for mixed-precision training
            if precision.lower() in ('fp16', 'float16', 'bf16', 'bfloat16'):
                # For LoRA-family: master copy only for trainable adapter params
                # For full/freeze: master copy for all trainable params
                if method.lower() in ('lora', 'dora', 'pissa', 'loraplus', 'oft'):
                    master_params = trainable_params
                else:
                    master_params = trainable_params if trainable_params > 0 else total_params
                master_copy_memory = (master_params * 4) / 1e9
                weight_memory += master_copy_memory

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

        # ZeRO-2 and ZeRO-3 shard gradients across DP ranks
        if deepspeed_stage in ('zero2', 'zero3'):
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

        # Resolve aliases for common Transformers/LlamaFactory optimizer names
        resolved = OPTIMIZER_ALIASES.get(optimizer, optimizer)

        # Use OPTIMIZER_CONFIGS for precise bytes_per_param
        if resolved in OPTIMIZER_CONFIGS:
            opt_config = OPTIMIZER_CONFIGS[resolved]
            bytes_per_param = opt_config.total_bytes_per_param
        else:
            # Unknown optimizer: assume AdamW-like (8 bytes/param)
            bytes_per_param = ADAMW_BASELINE_BYTES

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
        flash_attention: bool = True,
    ) -> float:
        """
        Calculate activation memory in GB using config-aware Megatron-LM formula.

        Extends the standard Megatron-LM ``sbh(10 + 24/T)`` structure by
        deriving the TP-parallelized coefficient from actual model dimensions
        (GQA ratio, SwiGLU vs GeLU, real intermediate_size) and by optionally
        eliminating the O(s^2) attention-score term when flash attention is used.

        With gradient checkpointing, only ceil(sqrt(L)) layers are stored.
        """
        h = config.get('hidden_size', 4096)
        L = config.get('num_hidden_layers', 32)
        a = config.get('num_attention_heads', 32)
        kv_heads = config.get('num_key_value_heads', a)
        ffn = config.get('intermediate_size', h * 4)
        hidden_act = config.get('hidden_act', 'silu')

        s, b, T = seq_length, batch_size, tensor_parallel
        kv_ratio = kv_heads / a  # 1.0 for MHA, <1 for GQA/MQA

        # --- Non-TP terms (bytes per token, per layer) ---
        # LayerNorm inputs (2), attn dropout mask (1), MLP dropout mask (1),
        # two residual connections (2+2), two LayerNorm outputs (2) = 10
        non_tp_coeff = 10.0

        # --- TP-parallelized attention terms ---
        # Q(2) + K(2*kv_ratio) + V(2*kv_ratio) + O(2) = 4 + 4*kv_ratio
        attn_coeff = 4.0 + 4.0 * kv_ratio

        # --- TP-parallelized MLP terms ---
        ffn_mult = ffn / h
        if hidden_act in ('silu', 'swish', 'swiglu', 'gelu_new'):
            # SwiGLU/GeGLU: gate + up + activation input = 3 projections stored
            mlp_coeff = 6.0 * ffn_mult
        else:
            # Standard GeLU/ReLU: up + activation input = 2 projections stored
            mlp_coeff = 4.0 * ffn_mult

        tp_coeff = attn_coeff + mlp_coeff

        # --- Attention score term (quadratic in seq_length) ---
        if flash_attention:
            score_coeff = 0.0  # FlashAttention eliminates O(s^2) materialization
        else:
            score_coeff = 5.0 * a * s / h  # Q*K^T scores + softmax + dropout mask

        # Per-layer bytes
        per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff / T + score_coeff / T)

        # Gradient checkpointing: store only ceil(sqrt(L)) layers
        if gradient_checkpointing:
            effective_layers = math.ceil(math.sqrt(L))
            effective_layers = max(effective_layers, 2)
        else:
            effective_layers = L

        total_bytes = per_layer_bytes * effective_layers
        return total_bytes / 1e9


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
