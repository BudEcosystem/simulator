"""Type definitions for LLM Memory Calculator."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MemoryReport:
    """Detailed memory breakdown for model inference."""
    model_type: str
    attention_type: Optional[str]
    precision: str
    parameter_count: int
    weight_memory_bytes: float
    kv_cache_bytes: float
    activation_memory_bytes: float
    state_memory_bytes: float
    image_memory_bytes: float
    extra_work_bytes: float
    lora_adapter_memory_bytes: float = 0.0
    # Punica prefill working buffers -- the TRANSIENT, ~50x the storage above and
    # the term that actually sizes a LoRA serving pod. Kept separate because
    # folding it into the storage figure is how it stayed invisible.
    lora_prefill_scratch_bytes: float = 0.0
    # Sampler working set: logits + log_softmax + sorted values + an int64 index,
    # live per in-flight sequence. Its width is the VOCABULARY, not the hidden
    # size, so a small-hidden/large-vocab model (Qwen3-0.6B: vocab 151936 vs
    # hidden 1024) carries a real per-sequence term that no weights+KV formula
    # sees. Model geometry x workload concurrency -- the same shape as KV and
    # activation, so it belongs here rather than in a deployment layer that would
    # have to re-open config.json to get the vocabulary.
    sampler_logits_bytes: float = 0.0
    # Non-fatal calibration caveats (empty when every term is inside its measured
    # envelope). Callers should surface these, not drop them.
    notes: List[str] = field(default_factory=list)
    # How weight_memory_bytes was determined: a checkpoint measurement, or the
    # parameter-counting estimate. A caller sizing a pod should know which it got --
    # the estimate carries architecture risk, the measurement does not.
    weight_source: str = "estimated"

    
    @property
    def total_memory_bytes(self) -> float:
        """Total memory in bytes."""
        return (
            self.weight_memory_bytes +
            self.kv_cache_bytes +
            self.activation_memory_bytes +
            self.state_memory_bytes +
            self.image_memory_bytes +
            self.lora_adapter_memory_bytes +
            self.lora_prefill_scratch_bytes +
            self.sampler_logits_bytes +
            self.extra_work_bytes
        )
    
    @property
    def weight_memory_gb(self) -> float:
        """Weight memory in GB."""
        return self.weight_memory_bytes / 1e9
    
    @property
    def kv_cache_gb(self) -> float:
        """KV cache memory in GB."""
        return self.kv_cache_bytes / 1e9
    
    @property
    def activation_memory_gb(self) -> float:
        """Activation memory in GB."""
        return self.activation_memory_bytes / 1e9
    
    @property
    def state_memory_gb(self) -> float:
        """State memory in GB."""
        return self.state_memory_bytes / 1e9
    
    @property
    def image_memory_gb(self) -> float:
        """Image memory in GB."""
        return self.image_memory_bytes / 1e9
    
    @property
    def extra_work_gb(self) -> float:
        """Extra work memory in GB."""
        return self.extra_work_bytes / 1e9

    @property
    def sampler_logits_gb(self) -> float:
        """Sampler working set in decimal GB."""
        return self.sampler_logits_bytes / 1e9

    @property
    def lora_adapter_memory_gb(self) -> float:
        """LoRA adapter memory in GB."""
        return self.lora_adapter_memory_bytes / 1e9
    
    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory_bytes / 1e9
    
    @property
    def recommended_gpu_memory_gb(self) -> int:
        """Recommended GPU memory in GB."""
        total_gb = self.total_memory_gb
        overhead_gb = total_gb * 1.2  # 20% overhead
        
        # Round up to standard GPU sizes
        gpu_sizes = [8, 12, 16, 24, 40, 48, 80, 160]
        for size in gpu_sizes:
            if overhead_gb <= size:
                return size
        
        # For very large models, round to nearest 80GB increment
        return int((overhead_gb + 79) // 80) * 80
    
    @property
    def can_fit_24gb_gpu(self) -> bool:
        """Check if model fits in 24GB GPU."""
        return self.total_memory_gb < 22  # Leave some headroom
    
    @property
    def can_fit_80gb_gpu(self) -> bool:
        """Check if model fits in 80GB GPU."""
        return self.total_memory_gb < 75  # Leave some headroom
    
    def __str__(self) -> str:
        """String representation of memory report."""
        lines = [
            f"Model Type: {self.model_type}",
            f"Attention: {self.attention_type or 'N/A'}",
            f"Parameters: {self.parameter_count:,}",
            f"Precision: {self.precision}",
            "",
            "Memory Breakdown:",
            f"  Weights: {self.weight_memory_gb:.2f} GB",
            f"  KV Cache: {self.kv_cache_gb:.2f} GB",
            f"  Activations: {self.activation_memory_gb:.2f} GB",
        ]
        
        if self.state_memory_gb > 0:
            lines.append(f"  State: {self.state_memory_gb:.2f} GB")
        if self.image_memory_gb > 0:
            lines.append(f"  Images: {self.image_memory_gb:.2f} GB")
        if self.lora_adapter_memory_gb > 0:
            lines.append(f"  LoRA Adapters: {self.lora_adapter_memory_gb:.2f} GB")
        if self.extra_work_gb > 0:
            lines.append(f"  Extra/Overhead: {self.extra_work_gb:.2f} GB")
        
        lines.extend([
            "",
            f"Total Memory: {self.total_memory_gb:.2f} GB",
            f"Recommended GPU: {self.recommended_gpu_memory_gb} GB",
        ])
        
        return "\n".join(lines)