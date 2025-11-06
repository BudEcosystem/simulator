from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class LoraConfig:
    """
    Configuration for LoRA simulation and memory calculation.

    Args:
        enabled: Whether LoRA is enabled
        rank: Rank for performance simulation (used in GenZ operations)
        target_modules: List of module types to apply LoRA (e.g., ['attn', 'ffn'])
        strategy: Strategy for performance simulation ('dynamic' or 'merge')
        max_loras: Maximum number of LoRA adapters to serve simultaneously (vLLM)
        max_lora_rank: Maximum rank across all adapters (vLLM)
        lora_dtype: Data type for LoRA weights ('auto', 'fp16', 'bf16', 'fp32')
        fully_sharded_loras: Whether to shard both A and B matrices with tensor parallelism
    """
    enabled: bool = False
    rank: int = 0
    target_modules: List[str] = field(default_factory=list)
    strategy: Literal['dynamic', 'merge'] = 'dynamic'

    # vLLM-style memory calculation parameters
    max_loras: int = 1
    max_lora_rank: int = 256
    lora_dtype: str = 'auto'
    fully_sharded_loras: bool = False 