"""
Advanced Training Memory Calculator.

Integrates all training components for comprehensive simulation:
- Training types (SFT, DPO, PPO, KTO, RM)
- Multiple optimizer variants
- Distributed training (DeepSpeed, FSDP)
- Quantization-aware calculations

This provides production-ready estimates validated against published benchmarks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math

from ..parameter_counter import UniversalParameterCounter
from .types import TrainingMemoryEstimate
from .training_types import (
    TrainingStage,
    TrainingStageConfig,
    TrainingStageEstimate,
    DPOConfig,
    KTOConfig,
    PPOConfig,
    get_training_stage_config,
    TRAINING_STAGE_CONFIGS,
)
from .optimizers import (
    OptimizerConfig,
    get_optimizer_config,
    calculate_optimizer_memory,
    OPTIMIZER_CONFIGS,
)
from .distributed import (
    ParallelismConfig,
    DistributedMemoryEstimate,
    get_deepspeed_config,
    get_fsdp_config,
    calculate_distributed_memory,
    recommend_distributed_strategy,
)


@dataclass
class AdvancedTrainingEstimate:
    """
    Comprehensive training memory and performance estimate.

    Combines training type, optimizer, quantization, and distributed
    training configurations into a single estimate.
    """

    # Training configuration
    training_stage: str
    training_method: str  # full, lora, qlora, etc.
    optimizer: str
    precision: str

    # Model information
    model_params: int
    trainable_params: int
    trainable_percent: float

    # Memory breakdown (per GPU, in GB)
    weight_memory_gb: float
    gradient_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    reference_model_memory_gb: float = 0.0
    reward_model_memory_gb: float = 0.0
    framework_overhead_gb: float = 0.0
    total_memory_gb: float = 0.0

    # Distributed configuration
    num_gpus: int = 1
    deepspeed_stage: Optional[str] = None
    fsdp_strategy: Optional[str] = None
    parallelism: Optional[Dict[str, int]] = None

    # Performance estimates
    throughput_factor: float = 1.0  # Relative to single GPU baseline
    communication_overhead: float = 0.0
    estimated_tps: Optional[float] = None  # Tokens per second

    # Fit recommendations
    fits_in_memory: bool = True
    recommended_gpu: Optional[str] = None
    min_gpus_required: int = 1

    # Quantization
    base_model_quantization: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_config": {
                "stage": self.training_stage,
                "method": self.training_method,
                "optimizer": self.optimizer,
                "precision": self.precision,
            },
            "model_info": {
                "total_params": self.model_params,
                "trainable_params": self.trainable_params,
                "trainable_percent": self.trainable_percent,
            },
            "memory_breakdown": {
                "weights_gb": self.weight_memory_gb,
                "gradients_gb": self.gradient_memory_gb,
                "optimizer_gb": self.optimizer_memory_gb,
                "activations_gb": self.activation_memory_gb,
                "reference_model_gb": self.reference_model_memory_gb,
                "reward_model_gb": self.reward_model_memory_gb,
                "framework_overhead_gb": self.framework_overhead_gb,
                "total_gb": self.total_memory_gb,
            },
            "distributed_config": {
                "num_gpus": self.num_gpus,
                "deepspeed_stage": self.deepspeed_stage,
                "fsdp_strategy": self.fsdp_strategy,
                "parallelism": self.parallelism,
            },
            "performance": {
                "throughput_factor": self.throughput_factor,
                "communication_overhead": self.communication_overhead,
                "estimated_tps": self.estimated_tps,
            },
            "fit_analysis": {
                "fits_in_memory": self.fits_in_memory,
                "recommended_gpu": self.recommended_gpu,
                "min_gpus_required": self.min_gpus_required,
            },
        }


class AdvancedTrainingCalculator:
    """
    Advanced calculator for training memory and performance estimation.

    Supports:
    - All training stages (SFT, DPO, PPO, KTO, RM, PT)
    - All fine-tuning methods (full, LoRA, QLoRA, freeze)
    - All optimizer variants (AdamW, 8-bit, GaLore, APOLLO, etc.)
    - Distributed training (DeepSpeed ZeRO, FSDP)
    - Quantization-aware calculations
    """

    # Precision to bytes mapping
    PRECISION_BYTES = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
        'int4': 0.5, 'uint4': 0.5,
        'nf4': 0.5,  # NormalFloat4 (QLoRA)
        'fp8': 1,
    }

    # GPU memory configurations
    GPU_MEMORY = {
        'RTX_3090': 24,
        'RTX_4090': 24,
        'A100_40GB': 40,
        'A100_80GB': 80,
        'H100': 80,
        'H200': 141,
        'MI300X': 192,
        'L40S': 48,
        'V100_16GB': 16,
        'V100_32GB': 32,
    }

    def __init__(self):
        """Initialize the advanced training calculator."""
        self.param_counter = UniversalParameterCounter()

    def calculate_advanced_training_memory(
        self,
        config: Dict[str, Any],
        training_stage: str = "sft",
        method: str = "lora",
        optimizer: str = "adamw",
        precision: str = "bf16",
        batch_size: int = 4,
        seq_length: int = 2048,
        num_gpus: int = 1,
        gradient_checkpointing: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        freeze_layers: int = 0,
        deepspeed_stage: Optional[str] = None,
        fsdp_strategy: Optional[str] = None,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        data_parallel: Optional[int] = None,
        base_model_quantization: Optional[str] = None,
        dpo_loss_type: Optional[str] = None,
        ppo_reward_type: Optional[str] = None,
        framework_overhead_percent: float = 10.0,
    ) -> AdvancedTrainingEstimate:
        """
        Calculate comprehensive training memory requirements.

        Args:
            config: Model configuration dictionary
            training_stage: Training stage (sft, dpo, ppo, kto, rm, pt)
            method: Fine-tuning method (full, lora, qlora, freeze)
            optimizer: Optimizer name
            precision: Training precision
            batch_size: Per-device batch size
            seq_length: Sequence length
            num_gpus: Total number of GPUs
            gradient_checkpointing: Enable gradient checkpointing
            lora_rank: LoRA rank (if using LoRA)
            lora_alpha: LoRA alpha (if using LoRA)
            freeze_layers: Layers to freeze (if using freeze method)
            deepspeed_stage: DeepSpeed ZeRO stage
            fsdp_strategy: FSDP sharding strategy
            tensor_parallel: Tensor parallelism degree
            pipeline_parallel: Pipeline parallelism degree
            data_parallel: Data parallelism degree (auto-calculated if None)
            base_model_quantization: Quantization for base model (int4, int8)
            dpo_loss_type: DPO loss type (for DPO training)
            ppo_reward_type: PPO reward model type (for PPO training)
            framework_overhead_percent: Framework memory overhead

        Returns:
            AdvancedTrainingEstimate with comprehensive breakdown
        """
        # Get training stage configuration
        stage_config = get_training_stage_config(
            training_stage,
            dpo_loss_type=dpo_loss_type,
            ppo_reward_type=ppo_reward_type,
        )

        # Get optimizer configuration
        opt_config = get_optimizer_config(optimizer)

        # Calculate data parallel if not specified
        if data_parallel is None:
            data_parallel = num_gpus // (tensor_parallel * pipeline_parallel)
            data_parallel = max(data_parallel, 1)

        # Create parallelism config
        parallelism = ParallelismConfig(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=data_parallel,
        )

        # Get model parameters
        total_params = self._get_total_params(config)

        # Calculate trainable parameters
        trainable_params = self._calculate_trainable_params(
            config, method, lora_rank, freeze_layers, total_params
        )

        # Get precision bytes
        precision_bytes = self.PRECISION_BYTES.get(precision.lower(), 2)

        # Apply quantization to base model if specified
        if base_model_quantization:
            quant_precision = self.PRECISION_BYTES.get(base_model_quantization.lower(), 2)
            weight_precision_bytes = quant_precision
        else:
            weight_precision_bytes = precision_bytes

        # Calculate base weight memory
        weight_memory = self._calculate_weight_memory(
            total_params,
            weight_precision_bytes,
            tensor_parallel,
            deepspeed_stage,
            data_parallel,
        )

        # Calculate gradient memory
        gradient_memory = self._calculate_gradient_memory(
            trainable_params,
            deepspeed_stage,
            data_parallel,
        )

        # Calculate optimizer memory
        optimizer_memory = calculate_optimizer_memory(
            optimizer,
            trainable_params,
            rank=lora_rank if method in ('lora', 'qlora') else None,
            deepspeed_stage=deepspeed_stage,
            data_parallel=data_parallel,
        )

        # Calculate activation memory
        activation_memory = self._calculate_activation_memory(
            config,
            batch_size,
            seq_length,
            precision_bytes,
            gradient_checkpointing,
            tensor_parallel,
        )

        # Calculate additional memory for training stage
        reference_memory = 0.0
        reward_memory = 0.0

        if stage_config.requires_reference_model:
            # Reference model in eval mode (less memory, no gradients/optimizer)
            reference_memory = weight_memory * 0.8  # Slightly less due to eval mode

        if stage_config.requires_reward_model:
            # Reward model memory (depends on PPO config)
            if ppo_reward_type == "lora":
                # LoRA reward model: just adapter params
                reward_memory = (trainable_params * precision_bytes) / 1e9 / tensor_parallel
            elif ppo_reward_type == "api":
                # API reward model: no GPU memory
                reward_memory = 0.0
            else:
                # Full reward model
                reward_memory = weight_memory

        # Value head overhead for RM and PPO
        if stage_config.requires_value_head:
            hidden_size = config.get('hidden_size', 4096)
            value_head_params = hidden_size  # Simple linear head
            value_head_memory = (value_head_params * precision_bytes) / 1e9
            weight_memory += value_head_memory

        # Calculate total with overhead
        components_sum = (
            weight_memory +
            gradient_memory +
            optimizer_memory +
            activation_memory +
            reference_memory +
            reward_memory
        )
        framework_overhead = components_sum * (framework_overhead_percent / 100)
        total_memory = components_sum + framework_overhead

        # Calculate performance factors
        throughput_factor = 1.0
        communication_overhead = parallelism.get_communication_overhead()

        if deepspeed_stage:
            ds_config = get_deepspeed_config(deepspeed_stage)
            communication_overhead += ds_config.communication_overhead
            throughput_factor = 1.0 - communication_overhead

        elif fsdp_strategy:
            fsdp_config = get_fsdp_config(fsdp_strategy)
            ds_equiv = get_deepspeed_config(fsdp_config.equivalent_zero)
            communication_overhead += ds_equiv.communication_overhead
            throughput_factor = fsdp_config.throughput_factor * (1.0 - ds_equiv.communication_overhead)

        throughput_factor = max(0.4, throughput_factor)

        # Fit analysis
        fits_in_memory = True
        recommended_gpu = None
        min_gpus = 1

        for gpu_name, gpu_mem in sorted(self.GPU_MEMORY.items(), key=lambda x: x[1]):
            if total_memory <= gpu_mem * 0.85:  # 85% utilization target
                recommended_gpu = gpu_name
                fits_in_memory = True
                break

        if recommended_gpu is None:
            fits_in_memory = False
            # Calculate minimum GPUs needed
            min_memory = min(self.GPU_MEMORY.values())
            min_gpus = math.ceil(total_memory / (min_memory * 0.85))
            recommended_gpu = f"{min_gpus}x A100_80GB (or larger)"

        # Estimate TPS (basic estimate, can be refined)
        estimated_tps = self._estimate_tps(
            total_params,
            num_gpus,
            throughput_factor,
            batch_size,
        )

        return AdvancedTrainingEstimate(
            training_stage=training_stage,
            training_method=method,
            optimizer=optimizer,
            precision=precision,
            model_params=total_params,
            trainable_params=trainable_params,
            trainable_percent=100 * trainable_params / total_params,
            weight_memory_gb=weight_memory,
            gradient_memory_gb=gradient_memory,
            optimizer_memory_gb=optimizer_memory,
            activation_memory_gb=activation_memory,
            reference_model_memory_gb=reference_memory,
            reward_model_memory_gb=reward_memory,
            framework_overhead_gb=framework_overhead,
            total_memory_gb=total_memory,
            num_gpus=num_gpus,
            deepspeed_stage=deepspeed_stage,
            fsdp_strategy=fsdp_strategy,
            parallelism=parallelism.to_dict(),
            throughput_factor=throughput_factor,
            communication_overhead=communication_overhead,
            estimated_tps=estimated_tps,
            fits_in_memory=fits_in_memory,
            recommended_gpu=recommended_gpu,
            min_gpus_required=min_gpus,
            base_model_quantization=base_model_quantization,
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
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', 32)

            # LoRA typically targets Q, K, V, O projections
            num_adapters = 4
            lora_params_per_layer = num_adapters * 2 * lora_rank * hidden_size

            # Optional MLP adapters
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            mlp_adapters = 2
            mlp_params_per_layer = mlp_adapters * 2 * lora_rank * (hidden_size + intermediate_size) // 2

            return num_layers * (lora_params_per_layer + mlp_params_per_layer)

        elif method == 'freeze':
            num_layers = config.get('num_hidden_layers', 32)
            unfrozen_layers = max(1, num_layers - freeze_layers)
            params_per_layer = total_params / num_layers
            return int(unfrozen_layers * params_per_layer)

        return total_params

    def _calculate_weight_memory(
        self,
        total_params: int,
        precision_bytes: float,
        tensor_parallel: int,
        deepspeed_stage: Optional[str],
        data_parallel: int,
    ) -> float:
        """Calculate weight memory in GB."""
        weight_memory = (total_params * precision_bytes) / 1e9
        weight_memory /= tensor_parallel

        if deepspeed_stage == 'zero3':
            weight_memory /= data_parallel

        return weight_memory

    def _calculate_gradient_memory(
        self,
        trainable_params: int,
        deepspeed_stage: Optional[str],
        data_parallel: int,
    ) -> float:
        """Calculate gradient memory in GB (FP32 gradients)."""
        gradient_memory = (trainable_params * 4) / 1e9

        if deepspeed_stage in ('zero2', 'zero3'):
            gradient_memory /= data_parallel

        return gradient_memory

    def _calculate_activation_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision_bytes: float,
        gradient_checkpointing: bool,
        tensor_parallel: int,
    ) -> float:
        """Calculate activation memory in GB."""
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_hidden_layers', 32)
        num_heads = config.get('num_attention_heads', 32)
        intermediate_size = config.get('intermediate_size', hidden_size * 4)

        # Per-layer activations
        attention_elements = batch_size * seq_length * hidden_size * 4
        attention_scores = batch_size * num_heads * seq_length * seq_length
        ffn_elements = batch_size * seq_length * intermediate_size * 2

        elements_per_layer = attention_elements + attention_scores + ffn_elements

        if gradient_checkpointing:
            effective_layers = math.sqrt(num_layers)
            effective_layers = max(effective_layers, 2)
        else:
            effective_layers = num_layers

        total_elements = elements_per_layer * effective_layers
        activation_memory = (total_elements * precision_bytes) / 1e9
        activation_memory /= tensor_parallel

        return activation_memory

    def _estimate_tps(
        self,
        total_params: int,
        num_gpus: int,
        throughput_factor: float,
        batch_size: int,
    ) -> float:
        """Estimate tokens per second for training."""
        # Base estimate: ~6 FLOPs per param per token for training
        # Assume H100-like performance: ~2000 TFLOPS theoretical, 40% MFU
        effective_tflops = 2000 * 0.4 * num_gpus * throughput_factor
        flops_per_token = 6 * total_params
        base_tps = (effective_tflops * 1e12) / flops_per_token

        # Batch size scaling
        if batch_size >= 8:
            base_tps *= 1.1
        elif batch_size <= 1:
            base_tps *= 0.8

        return max(100, base_tps)

    def compare_configurations(
        self,
        config: Dict[str, Any],
        configurations: List[Dict[str, Any]],
    ) -> List[AdvancedTrainingEstimate]:
        """
        Compare multiple training configurations.

        Args:
            config: Model configuration
            configurations: List of training configurations to compare

        Returns:
            List of estimates for each configuration
        """
        results = []
        for cfg in configurations:
            estimate = self.calculate_advanced_training_memory(
                config=config,
                **cfg,
            )
            results.append(estimate)
        return results


# Convenience functions
def calculate_advanced_training_memory(
    model_id_or_config,
    **kwargs,
) -> AdvancedTrainingEstimate:
    """
    Convenience function to calculate advanced training memory.

    Args:
        model_id_or_config: HuggingFace model ID or config dict
        **kwargs: Arguments for calculate_advanced_training_memory

    Returns:
        AdvancedTrainingEstimate
    """
    from ..huggingface_loader import HuggingFaceConfigLoader

    calculator = AdvancedTrainingCalculator()

    if isinstance(model_id_or_config, str):
        loader = HuggingFaceConfigLoader()
        config = loader.get_model_config(model_id_or_config)
    else:
        config = model_id_or_config

    return calculator.calculate_advanced_training_memory(
        config=config,
        **kwargs,
    )


def list_supported_configurations() -> Dict[str, List[str]]:
    """List all supported training configurations."""
    return {
        "training_stages": list(TRAINING_STAGE_CONFIGS.keys()),
        "methods": ["full", "lora", "qlora", "freeze", "dora", "pissa"],
        "optimizers": list(OPTIMIZER_CONFIGS.keys()),
        "precisions": list(AdvancedTrainingCalculator.PRECISION_BYTES.keys()),
        "deepspeed_stages": ["zero0", "zero1", "zero2", "zero2_offload", "zero3", "zero3_offload"],
        "fsdp_strategies": ["no_shard", "shard_grad_op", "full_shard"],
    }


def calculate_training_with_genz(
    model_id_or_config,
    training_stage: str = "sft",
    method: str = "lora",
    optimizer: str = "adamw",
    precision: str = "bf16",
    batch_size: int = 4,
    seq_length: int = 2048,
    num_gpus: int = 1,
    system_name: str = "A100_80GB_GPU",
    tensor_parallel: int = 1,
    data_parallel: Optional[int] = None,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    lora_rank: int = 16,
    zero_stage: int = 0,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Calculate training metrics using GenZ roofline analysis for accurate timing.

    This function integrates the GenZ training simulation with the memory
    calculator for comprehensive training estimates that include:
    - Accurate step timing based on hardware roofline analysis
    - Communication overhead from collective operations
    - Memory breakdown with ZeRO optimization stages

    Args:
        model_id_or_config: HuggingFace model ID or config dict
        training_stage: Training type (sft, dpo, ppo, kto, rm)
        method: Training method (full, lora, qlora)
        optimizer: Optimizer type (adamw, adam_8bit, etc.)
        precision: Training precision (bf16, fp16, fp32)
        batch_size: Per-device batch size
        seq_length: Sequence length
        num_gpus: Total number of GPUs
        system_name: Hardware system name from GenZ catalog
        tensor_parallel: Tensor parallelism degree
        data_parallel: Data parallelism degree (auto if None)
        pipeline_parallel: Pipeline parallelism degree
        expert_parallel: Expert parallelism degree
        lora_rank: LoRA rank (if using LoRA)
        zero_stage: ZeRO optimization stage (0-3)
        gradient_checkpointing: Enable gradient checkpointing
        gradient_accumulation_steps: Micro-batch count

    Returns:
        Dictionary with comprehensive training metrics:
        - timing: Step, forward, backward, optimizer, communication times
        - throughput: Tokens and samples per second
        - memory: Per-GPU memory breakdown
        - utilization: MFU, HFU, communication overhead
        - config: Training configuration used
    """
    from ..huggingface_loader import HuggingFaceConfigLoader
    from ..genz.LLM_training import training_modeling

    # Load model config if string
    if isinstance(model_id_or_config, str):
        loader = HuggingFaceConfigLoader()
        config = loader.get_model_config(model_id_or_config)
        model_name = model_id_or_config
    else:
        config = model_id_or_config
        model_name = config.get('_name_or_path', 'custom')

    # Auto-calculate data parallel if not specified
    if data_parallel is None:
        data_parallel = num_gpus // (tensor_parallel * pipeline_parallel * expert_parallel)
        data_parallel = max(1, data_parallel)

    # Map deepspeed stage string to int
    zero_stage_int = zero_stage
    if isinstance(zero_stage, str):
        zero_map = {'zero0': 0, 'zero1': 1, 'zero2': 2, 'zero3': 3}
        zero_stage_int = zero_map.get(zero_stage.lower().replace('_offload', ''), 0)

    # Try to find model in GenZ catalog, otherwise use config directly
    from ..genz.Models import MODEL_DICT
    genz_model = None
    for model_key in MODEL_DICT:
        if model_name.lower() in model_key.lower() or model_key.lower() in model_name.lower():
            genz_model = model_key
            break

    try:
        # Use GenZ training modeling for accurate timing
        genz_result = training_modeling(
            model=genz_model if genz_model else config,
            training_stage=training_stage,
            batch_size=batch_size,
            seq_length=seq_length,
            system_name=system_name,
            num_gpus=num_gpus,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            method=method,
            lora_rank=lora_rank,
            optimizer=optimizer,
            zero_stage=zero_stage_int,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bits=precision,
        )

        return genz_result.to_dict()

    except Exception as e:
        # Fallback to standard calculator if GenZ fails
        import warnings
        warnings.warn(f"GenZ training modeling failed: {e}. Using fallback calculator.")

        calculator = AdvancedTrainingCalculator()
        estimate = calculator.calculate_advanced_training_memory(
            config=config,
            training_stage=training_stage,
            method=method,
            optimizer=optimizer,
            precision=precision,
            batch_size=batch_size,
            seq_length=seq_length,
            num_gpus=num_gpus,
            gradient_checkpointing=gradient_checkpointing,
            lora_rank=lora_rank,
            deepspeed_stage=f"zero{zero_stage_int}" if zero_stage_int > 0 else None,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            pipeline_parallel=pipeline_parallel,
        )

        return estimate.to_dict()
