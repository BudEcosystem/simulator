import numpy as np
from typing import Dict, Any, Optional


class ParallelismConfig():
    r"""
    This is the configuration class to store the configuration of a Model Splitting.
    It is used to instantiate an LLM into multiple parallel units
    according to the specified arguments, defining the degree of various parallelism.

    Args:
        tensor_parallel: Tensor parallelism degree (splits model across GPUs within a layer)
        pipeline_parallel: Pipeline parallelism degree (splits model layers across GPUs)
        data_parallel: Data parallelism degree (replicates model across GPUs)
        expert_parallel: Expert parallelism degree for MoE models
        sequence_parallel: Sequence parallelism degree (splits sequence dimension)
        zero_stage: ZeRO optimization stage (0=disabled, 1=optimizer, 2=optimizer+gradients, 3=full sharding)
        gradient_accumulation_steps: Number of micro-batches before weight update
        gradient_checkpointing: Enable gradient/activation checkpointing to save memory
        context_parallel: Context parallelism for long sequences (ulysses/ring attention)
    """
    def __init__(
        self,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        data_parallel: int = 1,
        expert_parallel: int = 1,
        sequence_parallel: int = 1,
        zero_stage: int = 0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        context_parallel: int = 1,
        **kwargs,
    ):
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.data_parallel = data_parallel
        self.expert_parallel = expert_parallel
        self.sequence_parallel = sequence_parallel
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.context_parallel = context_parallel
        self.total_chips = np.prod([
                            self.data_parallel,
                            self.expert_parallel,
                            self.sequence_parallel,
                            self.pipeline_parallel,
                            self.tensor_parallel,
                            self.context_parallel])

        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'tensor_parallel': self.tensor_parallel,
            'pipeline_parallel': self.pipeline_parallel,
            'data_parallel': self.data_parallel,
            'expert_parallel': self.expert_parallel,
            'sequence_parallel': self.sequence_parallel,
            'zero_stage': self.zero_stage,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing,
            'context_parallel': self.context_parallel,
            'total_chips': int(self.total_chips),
        }

    def get_parallelism_hierarchy_str(self) -> str:
        """Generate parallelism hierarchy string for GenZ system."""
        return f"TP{{{self.tensor_parallel}}}_EP{{{self.expert_parallel}}}_PP{{{self.pipeline_parallel}}}"

    def get_communication_overhead(self) -> float:
        """
        Estimate communication overhead based on parallelism configuration.

        Uses physics-based models instead of linear approximations:
        - TP overhead: log-scaling based on AllReduce algorithm complexity
        - PP overhead: bubble fraction with scale-aware adjustments
        - DP overhead: log-scaling with congestion at large scale
        - EP overhead: All2All with superlinear congestion

        Returns:
            Fraction of time spent in communication (0.0 to 1.0)
        """
        import math
        overhead = 0.0

        # Tensor parallel overhead: 2 AllReduces per layer
        # Physics: Ring AR has O(N-1)/N bandwidth efficiency
        # At small scale, latency dominates; at large scale, congestion grows
        if self.tensor_parallel > 1:
            # Log-scaling: overhead = α * log2(tp) + β * tp / 8 for congestion
            base_tp = 0.03 * math.log2(self.tensor_parallel)
            # Superlinear congestion for large TP (multiple NVLink hops)
            congestion_tp = 0.005 * (self.tensor_parallel / 8) ** 1.3 if self.tensor_parallel > 4 else 0
            overhead += base_tp + congestion_tp

        # Pipeline parallel overhead: activation passing + bubble
        if self.pipeline_parallel > 1:
            # Base bubble fraction (1F1B schedule)
            num_micro_batches = max(self.gradient_accumulation_steps, self.pipeline_parallel)
            base_bubble = (self.pipeline_parallel - 1) / (self.pipeline_parallel + num_micro_batches - 1)

            # Scale-aware factor: bubbles grow at large scale due to stragglers
            total_gpus = self.total_chips
            if total_gpus > 2048:
                scale_factor = 1.0 + 0.2 * math.log2(total_gpus / 2048)
            else:
                scale_factor = 1.0

            bubble_overhead = base_bubble * scale_factor
            # Activation passing overhead (grows with PP stages)
            activation_overhead = 0.01 * math.log2(self.pipeline_parallel)
            overhead += min(bubble_overhead + activation_overhead, 0.4)

        # Data parallel overhead: gradient AllReduce
        # Physics: communication volume is fixed, time = volume / (effective_bw)
        # Effective BW degrades with log(DP) due to network congestion
        if self.data_parallel > 1:
            # Log-scaling base overhead
            base_dp = 0.04 * math.log2(self.data_parallel)

            # Congestion factor at large scale (superlinear growth)
            if self.data_parallel > 64:
                congestion_dp = 0.02 * math.log2(self.data_parallel / 64) ** 1.5
            else:
                congestion_dp = 0

            dp_overhead = base_dp + congestion_dp

            # ZeRO stages increase communication
            # ZeRO-2: +ReduceScatter, ZeRO-3: +AllGather per layer
            if self.zero_stage >= 2:
                dp_overhead *= 1.25  # ReduceScatter + AllGather
            if self.zero_stage >= 3:
                dp_overhead *= 1.4   # Weight gathering per forward/backward

            overhead += dp_overhead

        # Expert parallel overhead: All2All for MoE
        # All2All is O(N^2) in worst case, typically O(N*log(N)) with good routing
        if self.expert_parallel > 1:
            # All2All has higher base overhead than AllReduce
            base_ep = 0.06 * math.log2(self.expert_parallel)
            # Superlinear congestion for A2A (routing conflicts)
            congestion_ep = 0.01 * (self.expert_parallel / 8) ** 1.5 if self.expert_parallel > 4 else 0
            overhead += base_ep + congestion_ep

        # Context parallel overhead (if enabled)
        if self.context_parallel > 1:
            # Ring attention or Ulysses attention overhead
            cp_overhead = 0.03 * math.log2(self.context_parallel)
            overhead += cp_overhead

        return min(overhead, 0.6)  # Cap at 60% overhead

    @property
    def effective_batch_multiplier(self) -> int:
        """Get effective batch size multiplier from data parallelism."""
        return self.data_parallel * self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ParallelismConfig':
        """Create ParallelismConfig from dictionary."""
        return cls(
            tensor_parallel=config.get('tensor_parallel', 1),
            pipeline_parallel=config.get('pipeline_parallel', 1),
            data_parallel=config.get('data_parallel', 1),
            expert_parallel=config.get('expert_parallel', 1),
            sequence_parallel=config.get('sequence_parallel', 1),
            zero_stage=config.get('zero_stage', 0),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=config.get('gradient_checkpointing', True),
            context_parallel=config.get('context_parallel', 1),
        )

    @classmethod
    def for_training(
        cls,
        num_gpus: int,
        model_params_b: float,
        gpu_memory_gb: float = 80,
        training_method: str = "lora",
        sequence_length: int = 4096,
    ) -> 'ParallelismConfig':
        """
        Create optimal parallelism configuration for training.

        Args:
            num_gpus: Total number of GPUs available
            model_params_b: Model parameters in billions
            gpu_memory_gb: Memory per GPU in GB
            training_method: Training method (full, lora, qlora)
            sequence_length: Training sequence length

        Returns:
            Optimized ParallelismConfig for training
        """
        # Estimate memory requirements
        if training_method == "qlora":
            weight_mem_gb = model_params_b * 0.5  # 4-bit
            optimizer_mem_gb = model_params_b * 0.01 * 12  # 1% trainable, 12 bytes/param
        elif training_method in ("lora", "dora"):
            weight_mem_gb = model_params_b * 2  # bf16
            optimizer_mem_gb = model_params_b * 0.01 * 12  # 1% trainable
        else:  # full
            weight_mem_gb = model_params_b * 2  # bf16
            optimizer_mem_gb = model_params_b * 12  # Full optimizer states

        total_mem_gb = weight_mem_gb + optimizer_mem_gb
        available_mem = gpu_memory_gb * 0.85

        tp, pp, dp = 1, 1, num_gpus
        zero_stage = 0

        # If model doesn't fit, use tensor parallelism first
        if total_mem_gb / tp > available_mem:
            while total_mem_gb / tp > available_mem and tp < min(8, num_gpus):
                tp *= 2

        # If still doesn't fit, try ZeRO
        if total_mem_gb / tp > available_mem:
            zero_stage = 2  # ZeRO-2 for gradients + optimizer
            dp = num_gpus // tp
            if (total_mem_gb / tp / dp) > available_mem:
                zero_stage = 3  # ZeRO-3 for full sharding

        # Remaining GPUs for data parallel
        dp = max(1, num_gpus // tp)

        # Gradient accumulation for effective batch size
        grad_accum = max(1, 32 // dp)  # Target ~32 effective batch size

        return cls(
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            zero_stage=zero_stage,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
        )