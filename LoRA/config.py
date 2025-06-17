from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class LoraConfig:
    """Configuration for LoRA simulation (enabled, rank, strategy, targets)."""
    enabled: bool = False
    rank: int = 0
    target_modules: List[str] = field(default_factory=list)
    strategy: Literal['dynamic', 'merge'] = 'dynamic' 