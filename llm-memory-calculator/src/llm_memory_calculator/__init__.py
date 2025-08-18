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

# Import hardware management functions
from .hardware import (
    HardwareManager,
    get_hardware_config,
    get_all_hardware,
    get_hardware_by_type,
    get_hardware_by_manufacturer,
    search_hardware,
    set_hardware_db_path,
    HARDWARE_CONFIGS,
)

# Import parallelism functions if available
try:
    from .parallelism_optimizer import (
        get_various_parallelization,
        get_best_parallelization_strategy,
        get_pareto_optimal_performance,
        get_minimum_system_size,
    )
    _PARALLELISM_AVAILABLE = True
except ImportError:
    _PARALLELISM_AVAILABLE = False

# Import performance estimation functions if available
try:
    from .performance_estimator import (
        estimate_prefill_performance,
        estimate_decode_performance,
        estimate_end_to_end_performance,
        estimate_chunked_performance,
        compare_performance_configurations,
    )
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    _PERFORMANCE_AVAILABLE = False

# Define public API
__all__ = [
    # Classes
    "ModelMemoryCalculator",
    "UniversalParameterCounter",
    "HuggingFaceConfigLoader",
    "MemoryReport",
    "HardwareManager",
    
    # Functions
    "calculate_memory",
    "estimate_memory",
    "analyze_hf_model",
    "compare_models",
    "estimate_max_batch_size",
    "analyze_attention_efficiency",
    
    # Hardware functions
    "get_hardware_config",
    "get_all_hardware",
    "get_hardware_by_type",
    "get_hardware_by_manufacturer",
    "search_hardware",
    "set_hardware_db_path",
    "HARDWARE_CONFIGS",
]

# Add parallelism functions if available
if _PARALLELISM_AVAILABLE:
    __all__.extend([
        "get_various_parallelization",
        "get_best_parallelization_strategy",
        "get_pareto_optimal_performance",
        "get_minimum_system_size",
    ])

# Add performance estimation functions if available
if _PERFORMANCE_AVAILABLE:
    __all__.extend([
        "estimate_prefill_performance",
        "estimate_decode_performance", 
        "estimate_end_to_end_performance",
        "estimate_chunked_performance",
        "compare_performance_configurations",
    ])