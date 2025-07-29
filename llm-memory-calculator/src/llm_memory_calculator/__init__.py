"""
LLM Memory Calculator - Calculate memory requirements for Large Language Models.

This package provides accurate memory estimation for various LLM architectures
including transformers, diffusion models, and state-space models.
"""

__version__ = "0.1.0"

# Import main classes
from .calculator import ModelMemoryCalculator
from .parameter_counter import UniversalParameterCounter
from .huggingface_loader import HuggingFaceConfigLoader
from .types import MemoryReport

# Import convenience functions
from .utils import (
    calculate_memory,
    estimate_memory,
    analyze_hf_model,
    compare_models,
    estimate_max_batch_size,
    analyze_attention_efficiency,
)

# Define public API
__all__ = [
    # Classes
    "ModelMemoryCalculator",
    "UniversalParameterCounter",
    "HuggingFaceConfigLoader",
    "MemoryReport",
    
    # Functions
    "calculate_memory",
    "estimate_memory",
    "analyze_hf_model",
    "compare_models",
    "estimate_max_batch_size",
    "analyze_attention_efficiency",
]