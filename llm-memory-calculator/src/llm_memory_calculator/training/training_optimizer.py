"""
Training Optimizer - Comprehensive Training Configuration Optimization.

Provides optimal training configurations based on:
- Model characteristics (architecture, size, MoE, attention type)
- Dataset size and token count
- Hardware constraints and costs
- Quality vs cost tradeoffs

Outputs optimal settings for:
- Training method (full, LoRA, QLoRA)
- Optimizer selection
- Quantization strategy
- Parallelism configuration
- Cluster sizing and cost

Sources:
- Scaling laws (Chinchilla, Kaplan)
- Published training benchmarks
- Hardware efficiency studies
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import math

# GenZ integration for precise hardware calculations
from ..genz.system import System
from ..genz.unit import Unit
from ..genz.collective_times import get_AR_time, get_AG_time, get_A2A_time, get_message_pass_time

from .hardware_catalog import (
    GPUSpec, GPUCost, ClusterSpec, InterconnectType,
    get_gpu_spec, get_gpu_cost, list_gpus,
    GPU_SPECS, GPU_COSTS,
)
from .model_characteristics import (
    ModelCharacteristics, AttentionType, FFNType, ModelFamily,
    get_model_characteristics, analyze_model_config,
    MODEL_CHARACTERISTICS,
)
from .optimizers import (
    OptimizerConfig, get_optimizer_config, calculate_optimizer_memory,
    OPTIMIZER_CONFIGS,
)
from .training_types import (
    TrainingStage, get_training_stage_config, TRAINING_STAGE_CONFIGS,
)
from .distributed import (
    DeepSpeedConfig, get_deepspeed_config, ParallelismConfig,
    calculate_distributed_memory, DEEPSPEED_CONFIGS,
)


def create_genz_system_from_gpu(
    gpu_spec: GPUSpec,
    num_gpus: int = 1,
    precision: str = "bf16",
    parallelism_hierarchy: str = "TP{1}_EP{1}_PP{1}",
) -> System:
    """
    Create a GenZ System from GPUSpec for precise roofline calculations.

    This bridges the training optimizer with GenZ's hardware modeling.

    Args:
        gpu_spec: GPU specification from hardware catalog
        num_gpus: Number of GPUs in cluster
        precision: Training precision (bf16, fp16, fp32, fp8)
        parallelism_hierarchy: Parallelism configuration string

    Returns:
        GenZ System object for roofline analysis
    """
    # Map precision to GenZ bits format
    precision_map = {
        "bf16": "bf16",
        "fp16": "bf16",  # GenZ uses bf16 for half precision
        "fp32": "f32",
        "fp8": "fp8",
        "int8": "int8",
    }
    bits = precision_map.get(precision, "bf16")

    # Get compute TFLOPS for the precision
    compute_tflops = gpu_spec.get_compute_tflops(precision)

    # Fall back to fp16 if bf16 not supported (e.g., V100)
    if compute_tflops == 0 and precision in ("bf16", "bfloat16"):
        compute_tflops = gpu_spec.get_compute_tflops("fp16")
        bits = "bf16"  # GenZ treats fp16 similar to bf16

    # Fall back to fp32 if still 0
    if compute_tflops == 0:
        compute_tflops = gpu_spec.get_compute_tflops("fp32")
        bits = "f32"

    # Determine interconnect bandwidth based on GPU type
    if gpu_spec.nvlink_bandwidth_gbps > 0 and num_gpus > 1:
        interchip_link_bw = gpu_spec.nvlink_bandwidth_gbps
    else:
        # PCIe bandwidth
        interchip_link_bw = 64.0  # PCIe 4.0 x16

    # Create GenZ System
    # GenZ expects TFLOPS for flops parameter (it converts internally using unit_to_raw with type='C' = 1e12)
    # Memory bandwidth in GB/s, memory size in MB
    system = System(
        flops=compute_tflops,  # TFLOPS directly - GenZ converts internally
        offchip_mem_bw=gpu_spec.memory_bandwidth_gbps,
        off_chip_mem_size=gpu_spec.memory_gb * 1024,  # Convert GB to MB
        bits=bits,
        num_nodes=num_gpus,
        interchip_link_bw=interchip_link_bw,
        interchip_link_latency=1.9,  # Typical NVLink latency in us
        parallelism_heirarchy=parallelism_hierarchy,
    )

    return system


def calculate_training_throughput(
    model: ModelCharacteristics,
    system: System,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Calculate training throughput using GenZ roofline analysis.

    Args:
        model: Model characteristics
        system: GenZ System object
        batch_size: Training batch size per GPU
        sequence_length: Sequence length
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Dict with throughput metrics:
        - tokens_per_second: Training tokens per second
        - compute_time_ms: Compute time per step
        - memory_time_ms: Memory time per step
        - mfu: Model FLOPS Utilization
    """
    # Calculate FLOPs per token for forward pass
    # Approximation: 2 * params per token for forward, 4x for backward
    params = model.active_parameters
    flops_per_token = 2 * params
    total_tokens = batch_size * sequence_length

    # Forward + backward = 3x forward FLOPs (with gradient checkpointing)
    # Without checkpointing it's 2x, with checkpointing it's ~3x due to recomputation
    if gradient_checkpointing:
        training_flops = flops_per_token * total_tokens * 3
    else:
        training_flops = flops_per_token * total_tokens * 2

    # Compute time (ms)
    compute_time_ms = (training_flops * system.get_bit_multiplier(type='C')) / (system.op_per_sec) * 1000

    # Memory time - weight access + gradient access + optimizer access
    # Forward: read weights once
    # Backward: read weights, write gradients
    # Optimizer: read gradients, update weights
    weight_bytes = params * system.get_bit_multiplier(type='M')

    # With gradient checkpointing, we re-read weights for recomputation
    if gradient_checkpointing:
        weight_accesses = 3 + model.num_layers  # Forward + backward + recomputation
    else:
        weight_accesses = 3  # Forward + backward + optimizer

    memory_time_ms = (weight_bytes * weight_accesses / system.offchip_mem_bw) * 1000

    # Total step time (compute-bound or memory-bound)
    step_time_ms = max(compute_time_ms, memory_time_ms)

    # Calculate throughput
    tokens_per_second = total_tokens / (step_time_ms / 1000)

    # MFU: achieved FLOPS / peak FLOPS
    # GenZ stores flops as raw FLOPS (after unit_to_raw with TFLOPS -> 1e12 multiplier)
    achieved_flops = training_flops / (step_time_ms / 1000)
    peak_flops = system.flops  # Already in raw FLOPS
    mfu = achieved_flops / peak_flops if peak_flops > 0 else 0

    return {
        "tokens_per_second": tokens_per_second,
        "compute_time_ms": compute_time_ms,
        "memory_time_ms": memory_time_ms,
        "step_time_ms": step_time_ms,
        "mfu": min(mfu, 1.0),
    }


def calculate_communication_overhead(
    model: ModelCharacteristics,
    system: System,
    parallelism: Dict[str, int],
    batch_size: int,
    sequence_length: int,
) -> Dict[str, float]:
    """
    Calculate communication overhead for distributed training using GenZ.

    Args:
        model: Model characteristics
        parallelism: Dict with tp, dp, pp, ep settings
        system: GenZ System object
        batch_size: Per-GPU batch size
        sequence_length: Sequence length

    Returns:
        Dict with communication metrics:
        - total_comm_time_ms: Total communication time per step
        - allreduce_time_ms: AllReduce time for gradients
        - tp_comm_time_ms: Tensor parallel communication
        - pp_comm_time_ms: Pipeline parallel communication
    """
    tp = parallelism.get("tensor_parallel", 1)
    dp = parallelism.get("data_parallel", 1)
    pp = parallelism.get("pipeline_parallel", 1)
    ep = parallelism.get("expert_parallel", 1)

    # AllReduce time for data parallel gradients
    if dp > 1:
        gradient_bytes = model.active_parameters * system.get_bit_multiplier(type='M')
        allreduce_time_ms = get_AR_time(gradient_bytes, dp, system)
    else:
        allreduce_time_ms = 0

    # Tensor parallel communication (2 AllReduces per layer)
    if tp > 1:
        hidden_size = model.hidden_size
        tp_message_size = batch_size * sequence_length * hidden_size * system.get_bit_multiplier(type='M')
        tp_comm_time_ms = get_AR_time(tp_message_size, tp, system) * model.num_layers * 2
    else:
        tp_comm_time_ms = 0

    # Pipeline parallel communication (activation passing)
    if pp > 1:
        activation_size = batch_size * sequence_length * model.hidden_size * system.get_bit_multiplier(type='M')
        pp_comm_time_ms = get_message_pass_time(activation_size, system) * (pp - 1) * 2
    else:
        pp_comm_time_ms = 0

    # Expert parallel communication (All2All for MoE)
    if ep > 1 and model.moe is not None:
        # All2All for routing to experts
        expert_message_size = batch_size * sequence_length * model.hidden_size * system.get_bit_multiplier(type='M')
        ep_comm_time_ms = get_A2A_time(expert_message_size, ep, system) * model.num_layers * 2
    else:
        ep_comm_time_ms = 0

    total_comm_time_ms = allreduce_time_ms + tp_comm_time_ms + pp_comm_time_ms + ep_comm_time_ms

    return {
        "total_comm_time_ms": total_comm_time_ms,
        "allreduce_time_ms": allreduce_time_ms,
        "tp_comm_time_ms": tp_comm_time_ms,
        "pp_comm_time_ms": pp_comm_time_ms,
        "ep_comm_time_ms": ep_comm_time_ms,
    }


class OptimizationGoal(Enum):
    """Optimization goals."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCE = "balance"
    MINIMIZE_TIME = "minimize_time"


@dataclass
class DatasetConfig:
    """Dataset configuration for training."""

    total_tokens: int
    num_examples: int = 0
    avg_sequence_length: int = 2048

    # Data quality factors
    deduplication_done: bool = True
    quality_filtered: bool = True

    # For fine-tuning datasets
    instruction_following: bool = False
    preference_pairs: int = 0  # For DPO/PPO

    @property
    def estimated_examples(self) -> int:
        if self.num_examples > 0:
            return self.num_examples
        return self.total_tokens // self.avg_sequence_length


@dataclass
class TrainingConstraints:
    """Constraints for training optimization."""

    # Hardware constraints
    max_gpus: int = 8
    min_gpus: int = 1
    max_gpu_memory_gb: float = 80.0
    available_gpus: Optional[List[str]] = None  # List of GPU types to consider

    # Budget constraints
    max_cost_usd: Optional[float] = None
    max_hours: Optional[float] = None

    # Quality constraints
    min_quality_score: float = 0.90  # Relative to full fine-tuning

    # Training constraints
    max_batch_size: int = 64
    min_batch_size: int = 1
    target_batch_tokens: int = 2**20  # ~1M tokens effective batch

    # Allow spot/preemptible instances
    allow_spot: bool = True


@dataclass
class OptimizerRecommendation:
    """Recommendation for optimizer selection."""

    optimizer: str
    precision: str
    quantization: Optional[str]

    memory_per_param_bytes: float
    quality_factor: float
    convergence_factor: float  # 1.0 = baseline, <1 = faster

    rationale: str


@dataclass
class ClusterRecommendation:
    """Recommendation for cluster configuration."""

    gpu_type: str
    num_gpus: int
    gpus_per_node: int

    # Parallelism
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    expert_parallel: int

    # DeepSpeed/FSDP
    deepspeed_stage: Optional[str]

    # Memory
    memory_per_gpu_gb: float
    memory_utilization: float

    # Performance
    estimated_tps: float  # Tokens per second
    estimated_mfu: float
    communication_overhead: float

    # Cost
    hourly_cost_usd: float
    provider: str


@dataclass
class TrainingPlan:
    """Complete training plan with all recommendations."""

    # Model info
    model_name: str
    model_params: int
    active_params: int

    # Dataset
    total_tokens: int
    num_epochs: float

    # Training method
    training_stage: str
    training_method: str  # full, lora, qlora

    # Optimizer
    optimizer: OptimizerRecommendation

    # Cluster
    cluster: ClusterRecommendation

    # LoRA config (if applicable)
    lora_rank: int = 0
    lora_alpha: int = 0
    lora_target_modules: List[str] = field(default_factory=list)

    # Batch configuration
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 0

    # Training settings
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    gradient_checkpointing: bool = True

    # Estimates
    estimated_time_hours: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_quality_score: float = 1.0

    # Alternatives
    alternative_plans: List['TrainingPlan'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": {
                "name": self.model_name,
                "total_params": self.model_params,
                "active_params": self.active_params,
            },
            "dataset": {
                "total_tokens": self.total_tokens,
                "epochs": self.num_epochs,
            },
            "training": {
                "stage": self.training_stage,
                "method": self.training_method,
                "optimizer": self.optimizer.optimizer,
                "precision": self.optimizer.precision,
                "quantization": self.optimizer.quantization,
            },
            "cluster": {
                "gpu_type": self.cluster.gpu_type,
                "num_gpus": self.cluster.num_gpus,
                "tensor_parallel": self.cluster.tensor_parallel,
                "pipeline_parallel": self.cluster.pipeline_parallel,
                "data_parallel": self.cluster.data_parallel,
                "deepspeed_stage": self.cluster.deepspeed_stage,
                "provider": self.cluster.provider,
            },
            "batch_config": {
                "per_device_batch_size": self.per_device_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
            },
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.lora_target_modules,
            } if self.lora_rank > 0 else None,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "gradient_checkpointing": self.gradient_checkpointing,
            },
            "estimates": {
                "time_hours": self.estimated_time_hours,
                "cost_usd": self.estimated_cost_usd,
                "quality_score": self.estimated_quality_score,
            },
        }


class TrainingOptimizer:
    """
    Comprehensive training configuration optimizer.

    Given model config, dataset, and constraints, finds optimal:
    - Training method (full, LoRA, QLoRA)
    - Optimizer (AdamW, 8-bit, GaLore, etc.)
    - Cluster configuration (GPU type, count, parallelism)
    - Hyperparameters
    """

    # Quality factors for different configurations
    TRAINING_METHOD_QUALITY = {
        "full": 1.0,
        "lora": 0.95,
        "qlora": 0.93,
        "freeze": 0.90,
        "dora": 0.96,
        "pissa": 0.95,
    }

    OPTIMIZER_QUALITY = {
        "adamw": 1.0,
        "adam": 1.0,
        "adamw_8bit": 0.995,
        "paged_adamw_8bit": 0.995,
        "galore": 0.98,
        "galore_8bit": 0.97,
        "apollo": 0.98,
        "adafactor": 0.97,
        "lion": 0.98,
        "sgd": 0.95,
        "adam_mini": 0.97,
    }

    QUANTIZATION_QUALITY = {
        None: 1.0,
        "int8": 0.99,
        "nf4": 0.98,
        "int4": 0.97,
        "fp4": 0.96,
    }

    def __init__(self):
        pass

    def optimize(
        self,
        model: Union[str, Dict[str, Any], ModelCharacteristics],
        dataset: DatasetConfig,
        training_stage: str = "sft",
        constraints: Optional[TrainingConstraints] = None,
        goal: OptimizationGoal = OptimizationGoal.BALANCE,
    ) -> TrainingPlan:
        """
        Generate optimal training plan.

        Args:
            model: Model name, config dict, or ModelCharacteristics
            dataset: Dataset configuration
            training_stage: Training type (sft, dpo, ppo, etc.)
            constraints: Hardware and budget constraints
            goal: Optimization goal

        Returns:
            TrainingPlan with complete configuration
        """
        if constraints is None:
            constraints = TrainingConstraints()

        # Get model characteristics
        if isinstance(model, str):
            try:
                model_char = get_model_characteristics(model)
            except ValueError:
                # Assume it's a HuggingFace model ID
                model_char = self._load_hf_model_characteristics(model)
        elif isinstance(model, dict):
            model_char = analyze_model_config(model)
        else:
            model_char = model

        # Get training stage config
        stage_config = get_training_stage_config(training_stage)

        # Determine best training method
        training_method = self._select_training_method(
            model_char, dataset, constraints, goal
        )

        # Select optimizer
        optimizer_rec = self._select_optimizer(
            model_char, training_method, constraints, goal
        )

        # Configure cluster
        cluster_rec = self._configure_cluster(
            model_char, dataset, training_method, optimizer_rec,
            training_stage, constraints, goal
        )

        # Calculate batch configuration
        batch_config = self._calculate_batch_config(
            model_char, cluster_rec, dataset, constraints
        )

        # Calculate training time and cost
        time_hours, cost_usd = self._estimate_time_and_cost(
            model_char, dataset, cluster_rec, batch_config
        )

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            model_char, training_method, optimizer_rec.optimizer,
            optimizer_rec.quantization
        )

        # LoRA configuration
        lora_rank, lora_alpha, lora_targets = self._configure_lora(
            model_char, training_method, constraints
        )

        # Learning rate
        learning_rate = self._calculate_learning_rate(
            model_char, training_method, optimizer_rec.optimizer,
            batch_config["effective_batch_size"]
        )

        plan = TrainingPlan(
            model_name=model_char.name,
            model_params=model_char.num_parameters,
            active_params=model_char.active_parameters,
            total_tokens=dataset.total_tokens,
            num_epochs=dataset.total_tokens / (dataset.estimated_examples * dataset.avg_sequence_length),
            training_stage=training_stage,
            training_method=training_method,
            optimizer=optimizer_rec,
            cluster=cluster_rec,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_targets,
            per_device_batch_size=batch_config["per_device_batch_size"],
            gradient_accumulation_steps=batch_config["gradient_accumulation_steps"],
            effective_batch_size=batch_config["effective_batch_size"],
            learning_rate=learning_rate,
            warmup_steps=min(1000, dataset.total_tokens // batch_config["effective_batch_size"] // 10),
            gradient_checkpointing=True,
            estimated_time_hours=time_hours,
            estimated_cost_usd=cost_usd,
            estimated_quality_score=quality_score,
        )

        # Generate alternatives if requested
        if goal == OptimizationGoal.BALANCE:
            plan.alternative_plans = self._generate_alternatives(
                model_char, dataset, training_stage, constraints
            )

        return plan

    def _load_hf_model_characteristics(self, model_id: str) -> ModelCharacteristics:
        """Load model characteristics from HuggingFace."""
        # This would typically use HuggingFaceConfigLoader
        # For now, estimate based on model name
        params_b = 7  # Default
        if "70b" in model_id.lower():
            params_b = 70
        elif "13b" in model_id.lower():
            params_b = 13
        elif "8b" in model_id.lower():
            params_b = 8
        elif "3b" in model_id.lower():
            params_b = 3
        elif "1b" in model_id.lower():
            params_b = 1

        # Create basic characteristics
        return ModelCharacteristics(
            name=model_id,
            family=ModelFamily.LLAMA,
            num_parameters=int(params_b * 1e9),
            num_layers=32 if params_b < 20 else 80,
            hidden_size=4096 if params_b < 20 else 8192,
            intermediate_size=14336 if params_b < 20 else 28672,
            vocab_size=128256,
            attention=AttentionConfig(
                attention_type=AttentionType.GQA,
                num_attention_heads=32 if params_b < 20 else 64,
                num_kv_heads=8,
                head_dim=128,
                hidden_size=4096 if params_b < 20 else 8192,
            ),
            ffn_type=FFNType.SWIGLU,
        )

    def _select_training_method(
        self,
        model: ModelCharacteristics,
        dataset: DatasetConfig,
        constraints: TrainingConstraints,
        goal: OptimizationGoal,
    ) -> str:
        """Select optimal training method."""
        params_b = model.num_parameters / 1e9

        # Estimate memory for full fine-tuning
        full_memory_gb = (model.num_parameters * 16) / 1e9  # Mixed precision AdamW

        # Check if full fine-tuning is feasible
        max_memory = constraints.max_gpu_memory_gb * constraints.max_gpus * 0.85

        if goal == OptimizationGoal.MAXIMIZE_QUALITY:
            # Prefer full fine-tuning if possible
            if full_memory_gb <= max_memory:
                return "full"
            return "lora"

        if goal == OptimizationGoal.MINIMIZE_COST:
            # Always use most efficient method
            if params_b >= 7:
                return "qlora"
            return "lora"

        # Balanced approach
        lora_memory_gb = (model.num_parameters * 2) / 1e9 + 2  # Weights + overhead

        if full_memory_gb <= max_memory * 0.5:
            # Full fine-tuning fits comfortably
            return "full"
        elif lora_memory_gb <= constraints.max_gpu_memory_gb:
            # LoRA fits on single GPU
            return "lora"
        else:
            # Need QLoRA for memory
            return "qlora"

    def _select_optimizer(
        self,
        model: ModelCharacteristics,
        training_method: str,
        constraints: TrainingConstraints,
        goal: OptimizationGoal,
    ) -> OptimizerRecommendation:
        """Select optimal optimizer."""
        params_b = model.num_parameters / 1e9

        # Default recommendations by goal
        if goal == OptimizationGoal.MAXIMIZE_QUALITY:
            # Standard AdamW for best quality
            optimizer = "adamw"
            precision = "bf16"
            quantization = None if training_method != "qlora" else "nf4"
            rationale = "Standard AdamW for maximum quality"

        elif goal == OptimizationGoal.MINIMIZE_COST:
            # Most memory-efficient options
            if training_method == "qlora":
                optimizer = "paged_adamw_8bit"
                precision = "bf16"
                quantization = "nf4"
                rationale = "8-bit paged optimizer with 4-bit quantization for minimum memory"
            elif params_b > 13:
                optimizer = "adamw_8bit"
                precision = "bf16"
                quantization = None
                rationale = "8-bit AdamW reduces memory by 75%"
            else:
                optimizer = "adamw"
                precision = "bf16"
                quantization = None
                rationale = "Standard optimizer for smaller models"

        else:
            # Balanced
            if training_method == "qlora":
                optimizer = "paged_adamw_8bit"
                precision = "bf16"
                quantization = "nf4"
                rationale = "QLoRA with 8-bit paged optimizer for good balance"
            elif training_method == "full" and params_b > 30:
                optimizer = "adamw_8bit"
                precision = "bf16"
                quantization = None
                rationale = "8-bit AdamW for large model full fine-tuning"
            else:
                optimizer = "adamw"
                precision = "bf16"
                quantization = None
                rationale = "Standard AdamW for good quality-efficiency balance"

        opt_config = get_optimizer_config(optimizer)

        return OptimizerRecommendation(
            optimizer=optimizer,
            precision=precision,
            quantization=quantization,
            memory_per_param_bytes=opt_config.total_bytes_per_param,
            quality_factor=self.OPTIMIZER_QUALITY.get(optimizer, 1.0),
            convergence_factor=1.0,
            rationale=rationale,
        )

    def _configure_cluster(
        self,
        model: ModelCharacteristics,
        dataset: DatasetConfig,
        training_method: str,
        optimizer: OptimizerRecommendation,
        training_stage: str,
        constraints: TrainingConstraints,
        goal: OptimizationGoal,
    ) -> ClusterRecommendation:
        """Configure optimal cluster."""
        # Calculate memory requirements
        precision_bytes = 2 if optimizer.precision in ("bf16", "fp16") else 4
        quant_bytes = 0.5 if optimizer.quantization in ("nf4", "int4") else precision_bytes

        if training_method == "qlora":
            weight_memory = (model.num_parameters * quant_bytes) / 1e9
        elif training_method in ("lora", "dora", "pissa"):
            weight_memory = (model.num_parameters * precision_bytes) / 1e9
        else:
            weight_memory = (model.num_parameters * precision_bytes) / 1e9

        # Training stage multiplier
        stage_config = get_training_stage_config(training_stage)
        stage_multiplier = 1.0
        if stage_config.requires_reference_model:
            stage_multiplier = 1.7
        if stage_config.requires_reward_model:
            stage_multiplier = 2.8

        # Optimizer memory
        if training_method in ("lora", "qlora", "dora", "pissa"):
            # Only LoRA params need optimizer states
            trainable_ratio = 0.01  # ~1% for LoRA
        else:
            trainable_ratio = 1.0

        opt_memory = (model.num_parameters * trainable_ratio * optimizer.memory_per_param_bytes) / 1e9

        # Activation memory (rough estimate)
        activation_memory = weight_memory * 0.2  # With gradient checkpointing

        # Total per-GPU memory needed
        total_base_memory = (weight_memory + opt_memory + activation_memory) * stage_multiplier

        # Find suitable GPU configurations
        candidates = []

        gpu_list = constraints.available_gpus or list(GPU_SPECS.keys())

        for gpu_name in gpu_list:
            try:
                gpu_spec = get_gpu_spec(gpu_name)
                gpu_cost = get_gpu_cost(gpu_name)
            except ValueError:
                continue

            if gpu_spec.memory_gb < constraints.max_gpu_memory_gb:
                pass  # Check if this GPU is suitable

            # Determine parallelism needed
            parallelism = model.get_optimal_parallelism(
                constraints.max_gpus,
                gpu_spec.memory_gb,
                training_method,
            )

            # Calculate memory per GPU with parallelism
            tp = parallelism["tp"]
            dp = parallelism["dp"]
            pp = parallelism["pp"]
            ep = parallelism.get("ep", 1)

            num_gpus = tp * dp * pp * ep
            if num_gpus > constraints.max_gpus or num_gpus < constraints.min_gpus:
                continue

            # Distributed memory
            memory_per_gpu = weight_memory / tp
            if dp > 1:
                # ZeRO-3 for large models
                deepspeed_stage = "zero3" if weight_memory / tp > gpu_spec.memory_gb * 0.5 else "zero2"
                if deepspeed_stage == "zero3":
                    memory_per_gpu /= dp
                    opt_memory_per_gpu = opt_memory / dp
                else:
                    opt_memory_per_gpu = opt_memory / dp
            else:
                deepspeed_stage = None
                opt_memory_per_gpu = opt_memory

            total_per_gpu = memory_per_gpu + opt_memory_per_gpu + activation_memory

            # Check if fits
            if total_per_gpu > gpu_spec.memory_gb * 0.85:
                continue

            # Calculate performance using GenZ roofline analysis
            parallelism_str = f"TP{{{tp}}}_EP{{{ep}}}_PP{{{pp}}}"
            genz_system = create_genz_system_from_gpu(
                gpu_spec, num_gpus, optimizer.precision, parallelism_str
            )

            # Use GenZ-based throughput calculation
            batch_size = 4  # Typical per-device batch size
            seq_len = dataset.avg_sequence_length
            throughput_metrics = calculate_training_throughput(
                model, genz_system, batch_size, seq_len, gradient_checkpointing=True
            )

            # Use GenZ-based communication overhead calculation
            parallelism_dict = {
                "tensor_parallel": tp,
                "data_parallel": dp,
                "pipeline_parallel": pp,
                "expert_parallel": ep,
            }
            comm_metrics = calculate_communication_overhead(
                model, genz_system, parallelism_dict, batch_size, seq_len
            )

            # Calculate effective MFU and throughput
            compute_time = throughput_metrics["compute_time_ms"]
            comm_time = comm_metrics["total_comm_time_ms"]
            total_step_time = throughput_metrics["step_time_ms"] + comm_time

            mfu = throughput_metrics["mfu"]
            comm_overhead = comm_time / total_step_time if total_step_time > 0 else 0.0

            # Calculate tokens per second across all GPUs
            tokens_per_step = batch_size * seq_len * num_gpus
            tps = tokens_per_step / (total_step_time / 1000) if total_step_time > 0 else 0

            # Cost
            rate, provider = gpu_cost.get_best_cloud_rate(constraints.allow_spot)
            hourly_cost = rate * num_gpus

            candidates.append(ClusterRecommendation(
                gpu_type=gpu_name,
                num_gpus=num_gpus,
                gpus_per_node=min(8, num_gpus),
                tensor_parallel=tp,
                pipeline_parallel=pp,
                data_parallel=dp,
                expert_parallel=ep,
                deepspeed_stage=deepspeed_stage,
                memory_per_gpu_gb=total_per_gpu,
                memory_utilization=total_per_gpu / gpu_spec.memory_gb,
                estimated_tps=tps,
                estimated_mfu=mfu,
                communication_overhead=comm_overhead,
                hourly_cost_usd=hourly_cost,
                provider=provider,
            ))

        if not candidates:
            # Fallback to default
            return self._fallback_cluster(model, constraints)

        # Sort by goal
        if goal == OptimizationGoal.MINIMIZE_COST:
            candidates.sort(key=lambda x: x.hourly_cost_usd / max(x.estimated_tps, 1))
        elif goal == OptimizationGoal.MINIMIZE_TIME:
            candidates.sort(key=lambda x: -x.estimated_tps)
        else:
            # Balance: cost per token throughput
            candidates.sort(key=lambda x: x.hourly_cost_usd / max(x.estimated_tps, 1))

        return candidates[0]

    def _fallback_cluster(
        self,
        model: ModelCharacteristics,
        constraints: TrainingConstraints,
    ) -> ClusterRecommendation:
        """Fallback cluster configuration."""
        return ClusterRecommendation(
            gpu_type="a100_80gb",
            num_gpus=constraints.max_gpus,
            gpus_per_node=8,
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=constraints.max_gpus,
            expert_parallel=1,
            deepspeed_stage="zero3",
            memory_per_gpu_gb=40.0,
            memory_utilization=0.5,
            estimated_tps=1000.0,
            estimated_mfu=0.4,
            communication_overhead=0.2,
            hourly_cost_usd=constraints.max_gpus * 3.67,
            provider="AWS",
        )

    def _calculate_batch_config(
        self,
        model: ModelCharacteristics,
        cluster: ClusterRecommendation,
        dataset: DatasetConfig,
        constraints: TrainingConstraints,
    ) -> Dict[str, int]:
        """Calculate optimal batch configuration."""
        # Target effective batch size based on tokens
        target_batch_tokens = constraints.target_batch_tokens
        seq_length = dataset.avg_sequence_length

        target_batch_size = target_batch_tokens // seq_length

        # Per-device batch size (start conservative)
        per_device = 4
        if cluster.memory_utilization > 0.7:
            per_device = 2
        elif cluster.memory_utilization < 0.5:
            per_device = 8

        per_device = max(constraints.min_batch_size, min(per_device, constraints.max_batch_size))

        # Gradient accumulation to reach target
        total_devices = cluster.data_parallel  # Only DP contributes to effective batch
        grad_accum = max(1, target_batch_size // (per_device * total_devices))

        effective = per_device * total_devices * grad_accum

        return {
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective,
        }

    def _estimate_time_and_cost(
        self,
        model: ModelCharacteristics,
        dataset: DatasetConfig,
        cluster: ClusterRecommendation,
        batch_config: Dict[str, int],
    ) -> Tuple[float, float]:
        """Estimate training time and cost."""
        total_tokens = dataset.total_tokens
        tps = cluster.estimated_tps

        # Training steps
        tokens_per_step = batch_config["effective_batch_size"] * dataset.avg_sequence_length
        total_steps = total_tokens // tokens_per_step

        # Time
        seconds = total_tokens / max(tps, 100)
        hours = seconds / 3600

        # Cost
        cost = hours * cluster.hourly_cost_usd

        return hours, cost

    def _calculate_quality_score(
        self,
        model: ModelCharacteristics,
        training_method: str,
        optimizer: str,
        quantization: Optional[str],
    ) -> float:
        """Calculate estimated quality score."""
        score = 1.0

        # Training method
        score *= self.TRAINING_METHOD_QUALITY.get(training_method, 1.0)

        # Optimizer
        score *= self.OPTIMIZER_QUALITY.get(optimizer.lower(), 1.0)

        # Quantization
        score *= self.QUANTIZATION_QUALITY.get(quantization, 1.0)

        # Model architecture
        score *= model.estimate_quality_score(training_method, optimizer, "bf16", quantization)

        return score

    def _configure_lora(
        self,
        model: ModelCharacteristics,
        training_method: str,
        constraints: TrainingConstraints,
    ) -> Tuple[int, int, List[str]]:
        """Configure LoRA parameters."""
        if training_method not in ("lora", "qlora", "dora", "pissa"):
            return 0, 0, []

        # Rank based on model size
        params_b = model.num_parameters / 1e9

        if params_b > 30:
            rank = 32
        elif params_b > 10:
            rank = 16
        else:
            rank = 8

        alpha = rank * 2  # Standard: alpha = 2 * rank

        # Target modules
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if params_b > 7:
            targets.extend(["gate_proj", "up_proj", "down_proj"])

        return rank, alpha, targets

    def _calculate_learning_rate(
        self,
        model: ModelCharacteristics,
        training_method: str,
        optimizer: str,
        effective_batch_size: int,
    ) -> float:
        """Calculate optimal learning rate."""
        # Base learning rates
        base_lr = {
            "full": 2e-5,
            "lora": 1e-4,
            "qlora": 2e-4,
            "dora": 1e-4,
            "pissa": 1e-4,
            "freeze": 5e-5,
        }

        lr = base_lr.get(training_method, 2e-5)

        # Scale with batch size (linear scaling)
        if effective_batch_size > 32:
            lr *= min(effective_batch_size / 32, 4)

        # Adjust for optimizer
        if optimizer in ("lion", "adafactor"):
            lr *= 0.3  # These optimizers prefer lower LR

        return lr

    def _generate_alternatives(
        self,
        model: ModelCharacteristics,
        dataset: DatasetConfig,
        training_stage: str,
        constraints: TrainingConstraints,
    ) -> List[TrainingPlan]:
        """Generate alternative training plans."""
        alternatives = []

        # Cost-optimized alternative
        cost_constraints = TrainingConstraints(
            max_gpus=constraints.max_gpus,
            min_gpus=1,
            max_gpu_memory_gb=24,  # Consumer GPU
            allow_spot=True,
        )

        try:
            cost_plan = self.optimize(
                model, dataset, training_stage, cost_constraints,
                OptimizationGoal.MINIMIZE_COST
            )
            if cost_plan.estimated_cost_usd < constraints.max_cost_usd if constraints.max_cost_usd else True:
                alternatives.append(cost_plan)
        except Exception:
            pass

        # Quality-optimized alternative
        quality_constraints = TrainingConstraints(
            max_gpus=constraints.max_gpus * 2,  # Allow more GPUs
            min_gpus=constraints.min_gpus,
            max_gpu_memory_gb=80,
            allow_spot=False,
        )

        try:
            quality_plan = self.optimize(
                model, dataset, training_stage, quality_constraints,
                OptimizationGoal.MAXIMIZE_QUALITY
            )
            alternatives.append(quality_plan)
        except Exception:
            pass

        return alternatives[:2]  # Max 2 alternatives


# Import needed for type hints
from .model_characteristics import AttentionConfig, ModelFamily


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_training(
    model: Union[str, Dict[str, Any]],
    total_tokens: int,
    training_stage: str = "sft",
    max_gpus: int = 8,
    max_cost_usd: Optional[float] = None,
    goal: str = "balance",
) -> TrainingPlan:
    """
    Convenience function to optimize training configuration.

    Args:
        model: Model name or config dict
        total_tokens: Total tokens in dataset
        training_stage: Training type (sft, dpo, ppo, etc.)
        max_gpus: Maximum GPUs available
        max_cost_usd: Budget constraint
        goal: Optimization goal (minimize_cost, maximize_quality, balance)

    Returns:
        TrainingPlan with optimal configuration
    """
    optimizer = TrainingOptimizer()

    dataset = DatasetConfig(total_tokens=total_tokens)

    constraints = TrainingConstraints(
        max_gpus=max_gpus,
        max_cost_usd=max_cost_usd,
    )

    goal_enum = {
        "minimize_cost": OptimizationGoal.MINIMIZE_COST,
        "maximize_quality": OptimizationGoal.MAXIMIZE_QUALITY,
        "balance": OptimizationGoal.BALANCE,
        "minimize_time": OptimizationGoal.MINIMIZE_TIME,
    }.get(goal, OptimizationGoal.BALANCE)

    return optimizer.optimize(model, dataset, training_stage, constraints, goal_enum)


def compare_training_strategies(
    model: Union[str, Dict[str, Any]],
    total_tokens: int,
    training_stage: str = "sft",
    max_gpus: int = 8,
) -> List[Dict[str, Any]]:
    """
    Compare different training strategies for a model.

    Returns list of strategies with estimated time, cost, and quality.
    """
    optimizer = TrainingOptimizer()
    dataset = DatasetConfig(total_tokens=total_tokens)

    strategies = []

    for method in ["full", "lora", "qlora"]:
        for opt in ["adamw", "adamw_8bit"]:
            try:
                if method == "qlora" and opt == "adamw":
                    opt = "paged_adamw_8bit"

                constraints = TrainingConstraints(max_gpus=max_gpus)
                plan = optimizer.optimize(
                    model, dataset, training_stage, constraints,
                    OptimizationGoal.BALANCE
                )

                strategies.append({
                    "method": method,
                    "optimizer": opt,
                    "estimated_hours": plan.estimated_time_hours,
                    "estimated_cost": plan.estimated_cost_usd,
                    "quality_score": plan.estimated_quality_score,
                    "gpu_type": plan.cluster.gpu_type,
                    "num_gpus": plan.cluster.num_gpus,
                })
            except Exception:
                pass

    return sorted(strategies, key=lambda x: x["estimated_cost"])


def get_minimum_hardware_for_model(
    model: Union[str, Dict[str, Any]],
    training_method: str = "lora",
) -> Dict[str, Any]:
    """
    Get minimum hardware requirements for training a model.

    Returns:
        Dict with minimum GPU memory, recommended GPU, and estimated cost.
    """
    # Get model characteristics
    if isinstance(model, str):
        try:
            model_char = get_model_characteristics(model)
        except ValueError:
            # Estimate from name
            params_b = 7
            if "70b" in model.lower():
                params_b = 70
            elif "13b" in model.lower():
                params_b = 13
            elif "8b" in model.lower():
                params_b = 8

            model_char = ModelCharacteristics(
                name=model,
                family=ModelFamily.LLAMA,
                num_parameters=int(params_b * 1e9),
                num_layers=32,
                hidden_size=4096,
                intermediate_size=14336,
                vocab_size=128256,
                attention=AttentionConfig(
                    attention_type=AttentionType.GQA,
                    num_attention_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    hidden_size=4096,
                ),
                ffn_type=FFNType.SWIGLU,
            )
    elif isinstance(model, dict):
        model_char = analyze_model_config(model)
    else:
        model_char = model

    params_b = model_char.num_parameters / 1e9

    # Calculate minimum memory
    if training_method == "qlora":
        min_memory_gb = (model_char.num_parameters * 0.5) / 1e9 * 2  # 4-bit + overhead
    elif training_method in ("lora", "dora", "pissa"):
        min_memory_gb = (model_char.num_parameters * 2) / 1e9 * 1.3  # bf16 + overhead
    else:
        min_memory_gb = (model_char.num_parameters * 16) / 1e9  # Full fine-tuning

    # Find recommended GPU
    recommended_gpu = None
    min_gpus = 1

    for gpu_name in ["rtx_4090", "l40s", "a100_40gb", "a100_80gb", "h100_sxm"]:
        try:
            spec = get_gpu_spec(gpu_name)
            if spec.memory_gb * 0.85 >= min_memory_gb:
                recommended_gpu = gpu_name
                break
        except ValueError:
            continue

    if recommended_gpu is None:
        # Need multi-GPU
        min_gpus = math.ceil(min_memory_gb / (80 * 0.85))
        recommended_gpu = "a100_80gb"

    # Get cost
    try:
        cost = get_gpu_cost(recommended_gpu)
        rate, provider = cost.get_best_cloud_rate()
    except ValueError:
        rate, provider = 2.0, "Unknown"

    return {
        "model_params_b": params_b,
        "training_method": training_method,
        "min_memory_gb": min_memory_gb,
        "recommended_gpu": recommended_gpu,
        "min_gpus": min_gpus,
        "hourly_cost_per_gpu": rate,
        "provider": provider,
        "total_hourly_cost": rate * min_gpus,
    }
