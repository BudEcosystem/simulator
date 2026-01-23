"""
Auto-Configure Training - Unified API for optimal training configuration.

This module provides a single entry point for complete training configuration
that combines all optimization aspects:
- Hardware selection
- Parallelism configuration
- Training method selection (full/LoRA/QLoRA)
- Time and cost estimation
- Config generation for LlamaFactory/DeepSpeed

Main entry point:
- auto_configure_training(): One-stop function for complete training setup

Example usage:
    >>> from llm_memory_calculator.training import auto_configure_training
    >>>
    >>> plan = auto_configure_training(
    ...     model='llama-3-70b',
    ...     dataset_tokens=100_000_000_000,  # 100B tokens
    ...     available_hardware=['H100', 'A100_80GB'],
    ...     max_gpus=128,
    ...     max_cost_usd=50000,
    ...     optimization_goal='minimize_cost',
    ... )
    >>>
    >>> print(plan.summary())
    >>> print(f"Launch: {plan.to_torchrun_command()}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import yaml
import json

from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
from ..genz.LLM_training.training_parallelization import (
    get_best_training_parallelization,
    TrainingParallelismConfig,
    get_training_parallelization_options,
)
from .cluster_optimizer import (
    ClusterOptimizer,
    _get_gpu_memory,
    _get_system_name,
    GPU_MEMORY_GB,
)
from .cluster_optimizer_types import (
    ClusterDefinition,
    TrainingJobSpec,
    ParallelismStrategy,
    OptimizationTarget,
)
from .tco_calculator import get_gpu_pricing, calculate_tco
from .llamafactory_config_builder import build_llamafactory_config, build_deepspeed_config
from .node_selector import NodeSpec
from .training_time_estimator import estimate_training_time, TrainingTimeEstimate


@dataclass
class OptimalTrainingPlan:
    """
    Comprehensive optimal training plan.

    Contains everything needed to launch and complete training with
    optimal hardware, parallelism, and configuration.
    """
    # ========================================
    # Hardware Selection
    # ========================================
    selected_hardware: str  # GPU type (e.g., 'H100', 'A100_80GB')
    total_gpus: int
    total_nodes: int
    gpus_per_node: int
    gpu_memory_gb: float

    # ========================================
    # Parallelism Configuration
    # ========================================
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int  # "gangs"
    expert_parallel: int
    zero_stage: int
    gradient_checkpointing: bool

    # ========================================
    # Batch & Sequence Configuration
    # ========================================
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int  # per_device * DP * grad_accum
    effective_batch_tokens: int  # global_batch * seq_length
    max_seq_length: int

    # ========================================
    # Training Method
    # ========================================
    training_method: str  # 'full', 'lora', 'qlora', 'dora', 'pissa', 'freeze'
    precision: str  # 'bf16', 'fp16', 'fp8', etc.
    optimizer_name: str
    learning_rate: float

    # LoRA details (if applicable)
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_target_modules: Optional[List[str]] = None

    # ========================================
    # Performance Estimates
    # ========================================
    throughput_tokens_per_sec: float = 0.0
    step_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    memory_per_gpu_gb: float = 0.0
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    mfu: float = 0.0  # Model FLOPs Utilization
    hfu: float = 0.0  # Hardware FLOPs Utilization

    # ========================================
    # Training Time Estimation
    # ========================================
    total_training_hours: float = 0.0
    total_training_days: float = 0.0
    total_training_steps: int = 0
    steps_per_epoch: int = 0
    tokens_processed_per_step: int = 0

    # ========================================
    # Cost Analysis
    # ========================================
    hourly_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_million_tokens: float = 0.0
    cost_per_step: float = 0.0

    # ========================================
    # Scaling Efficiency
    # ========================================
    scaling_efficiency: float = 0.0  # vs single-GPU throughput
    communication_overhead_pct: float = 0.0
    bubble_overhead_pct: float = 0.0  # Pipeline parallel bubble

    # ========================================
    # Input Configuration
    # ========================================
    model: str = ""
    dataset_tokens: int = 0
    training_stage: str = "sft"
    optimization_goal: str = "minimize_cost"

    # ========================================
    # Generated Configs
    # ========================================
    _llamafactory_config: Dict[str, Any] = field(default_factory=dict)
    _deepspeed_config: Optional[Dict[str, Any]] = None

    def to_llamafactory_config(self) -> Dict[str, Any]:
        """Export as LLaMA Factory YAML config."""
        if self._llamafactory_config:
            return self._llamafactory_config

        config = {
            # Model
            'model_name_or_path': self.model,

            # Training stage
            'stage': self.training_stage,

            # Method
            'finetuning_type': self.training_method,

            # Data
            'cutoff_len': self.max_seq_length,
            'per_device_train_batch_size': self.per_device_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,

            # Precision
            'bf16': self.precision == 'bf16',
            'fp16': self.precision == 'fp16',

            # Optimizer
            'optim': self.optimizer_name,
            'learning_rate': self.learning_rate,

            # Gradient checkpointing
            'gradient_checkpointing': self.gradient_checkpointing,

            # DeepSpeed
            'deepspeed': f'ds_z{self.zero_stage}_config.json' if self.zero_stage > 0 else None,
        }

        # Add LoRA config if applicable
        if self.training_method in ('lora', 'qlora', 'dora'):
            config['lora_rank'] = self.lora_rank or 8
            config['lora_alpha'] = self.lora_alpha or 16
            if self.lora_target_modules:
                config['lora_target'] = ','.join(self.lora_target_modules)

        # QLoRA specific
        if self.training_method == 'qlora':
            config['quantization_bit'] = 4

        return config

    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Export as DeepSpeed JSON config."""
        if self._deepspeed_config:
            return self._deepspeed_config

        if self.zero_stage == 0:
            return {}

        config = {
            'bf16': {
                'enabled': self.precision == 'bf16',
            },
            'fp16': {
                'enabled': self.precision == 'fp16',
            },
            'zero_optimization': {
                'stage': self.zero_stage,
                'overlap_comm': True,
                'contiguous_gradients': True,
                'reduce_bucket_size': 5e8,
                'allgather_bucket_size': 5e8,
            },
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'train_micro_batch_size_per_gpu': self.per_device_batch_size,
            'gradient_clipping': 1.0,
        }

        # ZeRO-3 specific
        if self.zero_stage == 3:
            config['zero_optimization'].update({
                'stage3_prefetch_bucket_size': 5e8,
                'stage3_param_persistence_threshold': 1e6,
                'sub_group_size': 1e9,
            })

        return config

    def to_megatron_config(self) -> Dict[str, Any]:
        """Export as Megatron-LM arguments."""
        return {
            '--tensor-model-parallel-size': self.tensor_parallel,
            '--pipeline-model-parallel-size': self.pipeline_parallel,
            '--micro-batch-size': self.per_device_batch_size,
            '--global-batch-size': self.global_batch_size,
            '--seq-length': self.max_seq_length,
            '--bf16': self.precision == 'bf16',
            '--fp16': self.precision == 'fp16',
            '--use-flash-attn': True,
            '--recompute-activations': self.gradient_checkpointing,
        }

    def to_torchrun_command(self) -> str:
        """Generate torchrun launch command."""
        cmd_parts = [
            'torchrun',
            f'--nproc_per_node={self.gpus_per_node}',
            f'--nnodes={self.total_nodes}',
        ]

        if self.total_nodes > 1:
            cmd_parts.extend([
                '--node_rank=$NODE_RANK',
                '--master_addr=$MASTER_ADDR',
                '--master_port=$MASTER_PORT',
            ])

        cmd_parts.append('train.py')

        return ' '.join(cmd_parts)

    def summary(self) -> str:
        """Human-readable summary of the training plan."""
        lines = [
            "=" * 70,
            "OPTIMAL TRAINING PLAN",
            "=" * 70,
            "",
            f"Model: {self.model}",
            f"Training Stage: {self.training_stage}",
            f"Optimization Goal: {self.optimization_goal}",
            "",
            "-" * 70,
            "HARDWARE",
            "-" * 70,
            f"GPU Type: {self.selected_hardware}",
            f"Total GPUs: {self.total_gpus} ({self.total_nodes} nodes x {self.gpus_per_node} GPUs)",
            f"GPU Memory: {self.gpu_memory_gb:.0f} GB/GPU",
            "",
            "-" * 70,
            "PARALLELISM",
            "-" * 70,
            f"Tensor Parallel (TP): {self.tensor_parallel}",
            f"Pipeline Parallel (PP): {self.pipeline_parallel}",
            f"Data Parallel (DP/gangs): {self.data_parallel}",
            f"ZeRO Stage: {self.zero_stage}",
            f"Gradient Checkpointing: {self.gradient_checkpointing}",
            "",
            "-" * 70,
            "BATCH CONFIGURATION",
            "-" * 70,
            f"Per-Device Batch Size: {self.per_device_batch_size}",
            f"Gradient Accumulation: {self.gradient_accumulation_steps}",
            f"Global Batch Size: {self.global_batch_size}",
            f"Tokens per Step: {self.effective_batch_tokens:,}",
            f"Max Sequence Length: {self.max_seq_length}",
            "",
            "-" * 70,
            "TRAINING METHOD",
            "-" * 70,
            f"Method: {self.training_method}",
            f"Precision: {self.precision}",
            f"Optimizer: {self.optimizer_name}",
            f"Learning Rate: {self.learning_rate}",
        ]

        if self.training_method in ('lora', 'qlora', 'dora'):
            lines.extend([
                f"LoRA Rank: {self.lora_rank}",
                f"LoRA Alpha: {self.lora_alpha}",
            ])

        lines.extend([
            "",
            "-" * 70,
            "PERFORMANCE ESTIMATES",
            "-" * 70,
            f"Throughput: {self.throughput_tokens_per_sec:,.0f} tokens/sec",
            f"Step Time: {self.step_time_ms:.1f} ms",
            f"MFU: {self.mfu:.1%}",
            f"Memory/GPU: {self.memory_per_gpu_gb:.1f} GB",
            f"Scaling Efficiency: {self.scaling_efficiency:.1%}",
            "",
            "-" * 70,
            "TRAINING TIME",
            "-" * 70,
            f"Total Steps: {self.total_training_steps:,}",
            f"Total Hours: {self.total_training_hours:.1f}",
            f"Total Days: {self.total_training_days:.1f}",
            "",
            "-" * 70,
            "COST ANALYSIS",
            "-" * 70,
            f"Hourly Cost: ${self.hourly_cost_usd:.2f}",
            f"Total Cost: ${self.total_cost_usd:,.0f}",
            f"Cost per Million Tokens: ${self.cost_per_million_tokens:.4f}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hardware': {
                'gpu_type': self.selected_hardware,
                'total_gpus': self.total_gpus,
                'total_nodes': self.total_nodes,
                'gpus_per_node': self.gpus_per_node,
                'gpu_memory_gb': self.gpu_memory_gb,
            },
            'parallelism': {
                'tensor_parallel': self.tensor_parallel,
                'pipeline_parallel': self.pipeline_parallel,
                'data_parallel': self.data_parallel,
                'expert_parallel': self.expert_parallel,
                'zero_stage': self.zero_stage,
                'gradient_checkpointing': self.gradient_checkpointing,
            },
            'batch': {
                'per_device_batch_size': self.per_device_batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'global_batch_size': self.global_batch_size,
                'effective_batch_tokens': self.effective_batch_tokens,
                'max_seq_length': self.max_seq_length,
            },
            'method': {
                'training_method': self.training_method,
                'precision': self.precision,
                'optimizer': self.optimizer_name,
                'learning_rate': self.learning_rate,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
            },
            'performance': {
                'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
                'step_time_ms': self.step_time_ms,
                'mfu': self.mfu,
                'memory_per_gpu_gb': self.memory_per_gpu_gb,
                'scaling_efficiency': self.scaling_efficiency,
            },
            'time': {
                'total_hours': self.total_training_hours,
                'total_days': self.total_training_days,
                'total_steps': self.total_training_steps,
            },
            'cost': {
                'hourly_usd': self.hourly_cost_usd,
                'total_usd': self.total_cost_usd,
                'per_million_tokens': self.cost_per_million_tokens,
            },
            'input': {
                'model': self.model,
                'dataset_tokens': self.dataset_tokens,
                'training_stage': self.training_stage,
                'optimization_goal': self.optimization_goal,
            },
        }


def auto_configure_training(
    # ========================================
    # Workload Configuration
    # ========================================
    model: Union[str, Dict[str, Any]],
    dataset_tokens: int,
    num_epochs: float = 1.0,
    training_stage: str = 'sft',

    # ========================================
    # Sequence & Batch Configuration
    # ========================================
    max_seq_length: int = 4096,
    per_device_batch_size: Optional[int] = None,
    max_batch_size: int = 64,
    min_batch_size: int = 1,
    target_global_batch_tokens: int = 1_000_000,
    gradient_accumulation_steps: Optional[int] = None,
    max_gradient_accumulation: int = 128,

    # ========================================
    # Training Method & Precision
    # ========================================
    method: str = 'auto',
    precision: str = 'bf16',
    gradient_checkpointing: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: Optional[List[str]] = None,

    # ========================================
    # Optimizer Configuration
    # ========================================
    optimizer: str = 'adamw',
    learning_rate: float = 2e-5,

    # ========================================
    # Hardware Options
    # ========================================
    available_hardware: Optional[Union[List[NodeSpec], List[str], str]] = None,
    max_gpus: Optional[int] = None,
    min_gpus: int = 1,
    gpus_per_node: int = 8,
    memory_utilization_target: float = 0.85,

    # ========================================
    # Resource Constraints
    # ========================================
    max_cost_usd: Optional[float] = None,
    max_hours: Optional[float] = None,
    min_throughput: Optional[float] = None,

    # ========================================
    # Parallelism Constraints
    # ========================================
    max_tensor_parallel: Optional[int] = None,
    max_pipeline_parallel: Optional[int] = None,
    prefer_zero_stage: Optional[int] = None,

    # ========================================
    # Optimization Goal
    # ========================================
    optimization_goal: str = 'minimize_cost',

    # ========================================
    # Advanced Options
    # ========================================
    default_hourly_rate: float = 3.0,
    flash_attention: bool = True,
) -> OptimalTrainingPlan:
    """
    One-stop function for complete training configuration.

    This function analyzes the model, hardware, and constraints to produce
    an optimal training plan including:
    - Best hardware selection
    - Optimal parallelism strategy
    - Training method recommendation (full/LoRA/QLoRA)
    - Time and cost estimates
    - Ready-to-use configuration files

    Args:
        model: Model name (e.g., 'Llama-3.1-8B', 'Llama-3.1-70B') or HuggingFace config dict
        dataset_tokens: Total tokens in training dataset
        num_epochs: Number of training epochs
        training_stage: Training type ('sft', 'dpo', 'ppo', 'kto', etc.)
        max_seq_length: Maximum sequence length
        per_device_batch_size: Per-GPU batch size (auto if None)
        max_batch_size: Maximum per-device batch size
        min_batch_size: Minimum per-device batch size
        target_global_batch_tokens: Target effective batch in tokens
        gradient_accumulation_steps: Gradient accumulation (auto if None)
        max_gradient_accumulation: Maximum gradient accumulation
        method: Training method ('auto', 'full', 'lora', 'qlora', 'dora')
        precision: Training precision ('bf16', 'fp16', 'fp8')
        gradient_checkpointing: Enable gradient checkpointing
        lora_rank: LoRA rank (if using LoRA methods)
        lora_alpha: LoRA alpha (if using LoRA methods)
        lora_target_modules: LoRA target modules (auto if None)
        optimizer: Optimizer name ('adamw', 'adamw_8bit', etc.)
        learning_rate: Learning rate
        available_hardware: Hardware options (list of GPU types or NodeSpecs)
        max_gpus: Maximum GPUs to use
        min_gpus: Minimum GPUs to use
        gpus_per_node: GPUs per node
        memory_utilization_target: Target memory usage (0-1)
        max_cost_usd: Maximum total cost constraint
        max_hours: Maximum training time constraint
        min_throughput: Minimum throughput constraint (tokens/sec)
        max_tensor_parallel: Limit TP degree
        max_pipeline_parallel: Limit PP degree
        prefer_zero_stage: Prefer specific ZeRO stage
        optimization_goal: What to optimize:
            - 'minimize_cost': Lowest total training cost
            - 'minimize_time': Fastest training completion
            - 'maximize_throughput': Highest tokens/sec
            - 'maximize_mfu': Best hardware utilization
            - 'balance': Pareto-optimal cost vs time
        default_hourly_rate: Default cost per GPU per hour
        flash_attention: Enable Flash Attention

    Returns:
        OptimalTrainingPlan with complete configuration
    """
    # Normalize hardware options
    if available_hardware is None:
        available_hardware = ['H100', 'A100_80GB']
    elif isinstance(available_hardware, str):
        available_hardware = [available_hardware]

    # Extract GPU types from hardware options
    gpu_types = []
    for hw in available_hardware:
        if isinstance(hw, NodeSpec):
            gpu_types.append(hw.gpu_type)
        else:
            gpu_types.append(hw)
    gpu_types = list(set(gpu_types))  # Deduplicate

    # Set default max_gpus
    if max_gpus is None:
        max_gpus = 256

    # Generate GPU counts to search
    gpu_counts = _generate_gpu_counts(min_gpus, max_gpus)

    # Track best configuration
    best_plan = None
    best_score = float('-inf') if optimization_goal in ('maximize_throughput', 'maximize_mfu') else float('inf')

    # Search over GPU types and counts
    for gpu_type in gpu_types:
        gpu_memory = _get_gpu_memory(gpu_type)
        system_name = _get_system_name(gpu_type)

        # Get hourly rate
        try:
            pricing = get_gpu_pricing(gpu_type)
            hourly_rate, _, _ = pricing.get_best_rate(allow_spot=False)
        except Exception:
            hourly_rate = default_hourly_rate

        for num_gpus in gpu_counts:
            # Try different methods if auto
            methods_to_try = ['full', 'lora', 'qlora'] if method == 'auto' else [method]

            for training_method in methods_to_try:
                try:
                    plan = _evaluate_configuration(
                        model=model,
                        dataset_tokens=dataset_tokens,
                        num_epochs=num_epochs,
                        training_stage=training_stage,
                        max_seq_length=max_seq_length,
                        per_device_batch_size=per_device_batch_size,
                        max_batch_size=max_batch_size,
                        min_batch_size=min_batch_size,
                        target_global_batch_tokens=target_global_batch_tokens,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        max_gradient_accumulation=max_gradient_accumulation,
                        method=training_method,
                        precision=precision,
                        gradient_checkpointing=gradient_checkpointing,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_target_modules=lora_target_modules,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        gpu_type=gpu_type,
                        num_gpus=num_gpus,
                        gpus_per_node=gpus_per_node,
                        system_name=system_name,
                        gpu_memory=gpu_memory,
                        hourly_rate=hourly_rate,
                        max_tensor_parallel=max_tensor_parallel,
                        max_pipeline_parallel=max_pipeline_parallel,
                        prefer_zero_stage=prefer_zero_stage,
                        optimization_goal=optimization_goal,
                    )

                    if plan is None:
                        continue

                    # Check constraints
                    if max_cost_usd is not None and plan.total_cost_usd > max_cost_usd:
                        continue
                    if max_hours is not None and plan.total_training_hours > max_hours:
                        continue
                    if min_throughput is not None and plan.throughput_tokens_per_sec < min_throughput:
                        continue

                    # Calculate score
                    score = _calculate_score(plan, optimization_goal)

                    # Check if better
                    is_better = False
                    if optimization_goal in ('maximize_throughput', 'maximize_mfu'):
                        is_better = score > best_score
                    else:
                        is_better = score < best_score

                    if is_better:
                        best_score = score
                        best_plan = plan

                except Exception:
                    continue

    if best_plan is None:
        raise ValueError(
            f"No valid configuration found for {model}. "
            f"Constraints: max_gpus={max_gpus}, max_cost=${max_cost_usd}, max_hours={max_hours}. "
            "Try relaxing constraints or using a smaller model."
        )

    return best_plan


def _evaluate_configuration(
    model: Union[str, Dict[str, Any]],
    dataset_tokens: int,
    num_epochs: float,
    training_stage: str,
    max_seq_length: int,
    per_device_batch_size: Optional[int],
    max_batch_size: int,
    min_batch_size: int,
    target_global_batch_tokens: int,
    gradient_accumulation_steps: Optional[int],
    max_gradient_accumulation: int,
    method: str,
    precision: str,
    gradient_checkpointing: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: Optional[List[str]],
    optimizer: str,
    learning_rate: float,
    gpu_type: str,
    num_gpus: int,
    gpus_per_node: int,
    system_name: str,
    gpu_memory: float,
    hourly_rate: float,
    max_tensor_parallel: Optional[int],
    max_pipeline_parallel: Optional[int],
    prefer_zero_stage: Optional[int],
    optimization_goal: str,
) -> Optional[OptimalTrainingPlan]:
    """Evaluate a specific configuration."""
    # Determine batch size
    batch_size = per_device_batch_size or 4

    # Find best parallelism
    try:
        config, sim_result = get_best_training_parallelization(
            model=model,
            total_gpus=num_gpus,
            batch_size=batch_size,
            seq_length=max_seq_length,
            system_name=system_name,
            training_stage=training_stage,
            method=method,
            optimizer=optimizer,
            bits=precision,
            optimize_for='throughput',
            gpu_memory_gb=gpu_memory,
        )
    except Exception:
        return None

    # Apply parallelism constraints
    if max_tensor_parallel and config.tensor_parallel > max_tensor_parallel:
        return None
    if max_pipeline_parallel and config.pipeline_parallel > max_pipeline_parallel:
        return None

    # Calculate batch configuration
    data_parallel = config.data_parallel

    # Auto-calculate gradient accumulation to hit target batch tokens
    if gradient_accumulation_steps is None:
        tokens_per_step_without_ga = batch_size * max_seq_length * data_parallel
        gradient_accumulation_steps = max(
            1,
            min(
                max_gradient_accumulation,
                target_global_batch_tokens // tokens_per_step_without_ga
            )
        )

    global_batch_size = batch_size * data_parallel * gradient_accumulation_steps
    effective_batch_tokens = global_batch_size * max_seq_length

    # Calculate training time
    total_tokens = int(dataset_tokens * num_epochs)
    tokens_per_step = effective_batch_tokens
    total_steps = total_tokens // tokens_per_step
    steps_per_epoch = dataset_tokens // tokens_per_step

    step_time_sec = sim_result.step_time_ms / 1000
    total_seconds = total_steps * step_time_sec
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    # Calculate costs
    hourly_cost = hourly_rate * num_gpus
    total_cost = total_hours * hourly_cost
    cost_per_mtok = (total_cost / total_tokens) * 1_000_000 if total_tokens > 0 else float('inf')
    cost_per_step = total_cost / total_steps if total_steps > 0 else 0

    # Estimate scaling efficiency
    comm_overhead = sim_result.communication_overhead if hasattr(sim_result, 'communication_overhead') else 0
    scaling_efficiency = 1.0 - comm_overhead if comm_overhead else 0.85

    # Calculate nodes
    total_nodes = (num_gpus + gpus_per_node - 1) // gpus_per_node

    # Build memory breakdown
    memory_breakdown = {
        'weights': sim_result.weight_memory_gb if hasattr(sim_result, 'weight_memory_gb') else 0,
        'gradients': sim_result.gradient_memory_gb if hasattr(sim_result, 'gradient_memory_gb') else 0,
        'optimizer': sim_result.optimizer_memory_gb if hasattr(sim_result, 'optimizer_memory_gb') else 0,
        'activations': sim_result.activation_memory_gb if hasattr(sim_result, 'activation_memory_gb') else 0,
    }

    return OptimalTrainingPlan(
        # Hardware
        selected_hardware=gpu_type,
        total_gpus=num_gpus,
        total_nodes=total_nodes,
        gpus_per_node=min(gpus_per_node, num_gpus),
        gpu_memory_gb=gpu_memory,

        # Parallelism
        tensor_parallel=config.tensor_parallel,
        pipeline_parallel=config.pipeline_parallel,
        data_parallel=data_parallel,
        expert_parallel=config.expert_parallel if hasattr(config, 'expert_parallel') else 1,
        zero_stage=config.zero_stage,
        gradient_checkpointing=gradient_checkpointing,

        # Batch
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=global_batch_size,
        effective_batch_tokens=effective_batch_tokens,
        max_seq_length=max_seq_length,

        # Method
        training_method=method,
        precision=precision,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        lora_rank=lora_rank if method in ('lora', 'qlora', 'dora') else None,
        lora_alpha=lora_alpha if method in ('lora', 'qlora', 'dora') else None,
        lora_target_modules=lora_target_modules,

        # Performance
        throughput_tokens_per_sec=sim_result.tokens_per_second,
        step_time_ms=sim_result.step_time_ms,
        forward_time_ms=sim_result.forward_time_ms if hasattr(sim_result, 'forward_time_ms') else 0,
        backward_time_ms=sim_result.backward_time_ms if hasattr(sim_result, 'backward_time_ms') else 0,
        communication_time_ms=sim_result.communication_time_ms if hasattr(sim_result, 'communication_time_ms') else 0,
        optimizer_time_ms=sim_result.optimizer_time_ms if hasattr(sim_result, 'optimizer_time_ms') else 0,
        memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
        memory_breakdown=memory_breakdown,
        mfu=sim_result.model_flops_utilization,
        hfu=sim_result.hardware_flops_utilization if hasattr(sim_result, 'hardware_flops_utilization') else 0,

        # Time
        total_training_hours=total_hours,
        total_training_days=total_days,
        total_training_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        tokens_processed_per_step=tokens_per_step,

        # Cost
        hourly_cost_usd=hourly_cost,
        total_cost_usd=total_cost,
        cost_per_million_tokens=cost_per_mtok,
        cost_per_step=cost_per_step,

        # Scaling
        scaling_efficiency=scaling_efficiency,
        communication_overhead_pct=comm_overhead,
        bubble_overhead_pct=0,  # TODO: Calculate PP bubble

        # Input
        model=model,
        dataset_tokens=dataset_tokens,
        training_stage=training_stage,
        optimization_goal=optimization_goal,
    )


def _calculate_score(plan: OptimalTrainingPlan, optimization_goal: str) -> float:
    """Calculate optimization score for a plan."""
    if optimization_goal == 'minimize_cost':
        return plan.total_cost_usd
    elif optimization_goal == 'minimize_time':
        return plan.total_training_hours
    elif optimization_goal == 'maximize_throughput':
        return plan.throughput_tokens_per_sec
    elif optimization_goal == 'maximize_mfu':
        return plan.mfu
    elif optimization_goal == 'cost_per_token':
        return plan.cost_per_million_tokens
    elif optimization_goal == 'balance':
        # Balanced: sqrt(cost * time) - lower is better
        return (plan.total_cost_usd * plan.total_training_hours) ** 0.5
    else:
        return plan.total_cost_usd


def _generate_gpu_counts(min_gpus: int, max_gpus: int) -> List[int]:
    """Generate GPU counts to search."""
    counts = set()

    # Powers of 2
    power = 0
    while 2 ** power <= max_gpus:
        if 2 ** power >= min_gpus:
            counts.add(2 ** power)
        power += 1

    # Common cluster sizes
    for size in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        if min_gpus <= size <= max_gpus:
            counts.add(size)

    return sorted(counts)


def quick_configure(
    model: str,
    dataset_tokens: int,
    gpu_type: str = 'H100',
    num_gpus: int = 8,
    training_stage: str = 'sft',
    method: str = 'full',
) -> OptimalTrainingPlan:
    """
    Quick configuration with minimal parameters.

    Convenience wrapper for auto_configure_training with sensible defaults.

    Args:
        model: Model name
        dataset_tokens: Total tokens in dataset
        gpu_type: GPU type to use
        num_gpus: Number of GPUs
        training_stage: Training type
        method: Training method

    Returns:
        OptimalTrainingPlan
    """
    return auto_configure_training(
        model=model,
        dataset_tokens=dataset_tokens,
        training_stage=training_stage,
        method=method,
        available_hardware=[gpu_type],
        max_gpus=num_gpus,
        min_gpus=num_gpus,
    )
