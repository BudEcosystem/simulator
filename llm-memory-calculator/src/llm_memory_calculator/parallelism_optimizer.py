"""
Parallelism optimization for LLM deployment using GenZ.

This module provides a clean interface to GenZ's parallelism optimization
functions, with graceful fallbacks when GenZ is not available.
"""

from typing import Dict, Optional, Set, Tuple, Union
import warnings

# Import GenZ functions from internal copy
try:
    from .genz.LLM_inference import (
        get_best_parallization_strategy as _get_best_parallelization_strategy_genz,
        get_various_parallization as _get_various_parallelization_genz,
        get_pareto_optimal_performance as _get_pareto_optimal_performance_genz,
        get_minimum_system_size as _get_minimum_system_size_genz
    )
    HAS_GENZ = True
except ImportError as e:
    HAS_GENZ = False
    warnings.warn(
        f"Internal GenZ not available: {e}. This should not happen in a properly installed package.",
        ImportWarning
    )


def get_various_parallelization(model: str = 'llama2_7b', total_nodes: int = 8) -> Set[Tuple[int, int]]:
    """
    Get various parallelization combinations for a model and node count.
    
    Args:
        model: Model identifier (e.g., 'llama2_7b', 'llama2_70b')
        total_nodes: Total number of nodes/devices available
        
    Returns:
        Set of (tensor_parallel, pipeline_parallel) tuples
        
    Raises:
        ImportError: If GenZ is not installed
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    return _get_various_parallelization_genz(model=model, total_nodes=total_nodes)


def get_best_parallelization_strategy(
    stage: str = 'decode',
    model: str = 'llama2_7b',
    total_nodes: int = 8,
    batch_size: int = 1,
    beam_size: int = 1,
    input_tokens: int = 2000,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    debug: bool = False,
    **kwargs
) -> Union[Dict, 'pandas.DataFrame']:
    """
    Find the best parallelization strategy for given constraints.
    
    Args:
        stage: 'prefill' or 'decode'
        model: Model identifier (e.g., 'llama2_7b')
        total_nodes: Number of available devices
        batch_size: Batch size for inference
        beam_size: Beam size for generation
        input_tokens: Input sequence length
        output_tokens: Output sequence length
        system_name: Hardware configuration dict with keys:
            - Flops: TFLOPS
            - Memory_size: Memory in GB
            - Memory_BW: Memory bandwidth in GB/s
            - ICN: Interconnect bandwidth in GB/s
            - real_values: Whether to use real hardware values
        bits: Precision ('fp32', 'bf16', 'int8', 'int4')
        debug: Whether to print debug information
        
    Returns:
        DataFrame with best configuration or dict with results
        
    Raises:
        ImportError: If GenZ is not installed
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        # Default to A100 80GB configuration
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    return _get_best_parallelization_strategy_genz(
        stage=stage,
        model=model,
        total_nodes=total_nodes,
        batch_size=batch_size,
        beam_size=beam_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=system_name,
        bits=bits,
        debug=debug,
        **kwargs
    )


def get_pareto_optimal_performance(
    stage: str = 'decode',
    model: str = 'llama2_7b',
    total_nodes: int = 8,
    batch_list: Union[int, list] = 1,
    beam_size: int = 1,
    input_tokens: int = 2000,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    debug: bool = False,
    **kwargs
) -> 'pandas.DataFrame':
    """
    Get Pareto-optimal configurations balancing latency and throughput.
    
    This function requires the paretoset package to be installed.
    
    Args:
        stage: 'prefill' or 'decode'
        model: Model identifier
        total_nodes: Number of available devices
        batch_list: Single batch size or list of batch sizes to try
        beam_size: Beam size for generation
        input_tokens: Input sequence length
        output_tokens: Output sequence length
        system_name: Hardware configuration dict
        bits: Precision
        debug: Whether to print debug information
        
    Returns:
        DataFrame with Pareto-optimal configurations
        
    Raises:
        ImportError: If GenZ or paretoset is not installed
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    try:
        return _get_pareto_optimal_performance_genz(
            stage=stage,
            model=model,
            total_nodes=total_nodes,
            batch_list=batch_list,
            beam_size=beam_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_name=system_name,
            bits=bits,
            debug=debug,
            **kwargs
        )
    except ImportError as e:
        if "paretoset" in str(e):
            raise ImportError(
                "paretoset package is required for Pareto optimization. "
                "Install it with: pip install paretoset"
            )
        raise


def get_minimum_system_size(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    input_tokens: int = 2000,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    **kwargs
) -> int:
    """
    Get the minimum number of devices needed for a model configuration.
    
    Args:
        model: Model identifier
        batch_size: Batch size
        input_tokens: Input sequence length
        output_tokens: Output sequence length
        system_name: Hardware configuration dict
        bits: Precision
        
    Returns:
        Minimum number of devices required
        
    Raises:
        ImportError: If GenZ is not installed
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    return _get_minimum_system_size_genz(
        model=model,
        max_batch_size=batch_size,  # Internal function uses max_batch_size
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=system_name,
        bits=bits,
        **kwargs
    )


# Pre-defined hardware configurations for convenience
HARDWARE_CONFIGS = {
    'A100_40GB': {
        'Flops': 312,
        'Memory_size': 40,
        'Memory_BW': 1555,
        'ICN': 600,
        'real_values': True
    },
    'A100_80GB': {
        'Flops': 312,
        'Memory_size': 80,
        'Memory_BW': 2039,
        'ICN': 600,
        'real_values': True
    },
    'H100_80GB': {
        'Flops': 989,
        'Memory_size': 80,
        'Memory_BW': 3352,
        'ICN': 900,
        'real_values': True
    },
    'MI300X': {
        'Flops': 1307,
        'Memory_size': 192,
        'Memory_BW': 5300,
        'ICN': 896,
        'real_values': True
    },
    'TPUv4': {
        'Flops': 275,
        'Memory_size': 32,
        'Memory_BW': 1200,
        'ICN': 300,
        'real_values': True
    },
    'TPUv5e': {
        'Flops': 197,
        'Memory_size': 16,
        'Memory_BW': 819,
        'ICN': 300,
        'real_values': True
    }
}


def get_hardware_config(hardware_name: str) -> Dict:
    """
    Get pre-defined hardware configuration.
    
    Args:
        hardware_name: Name of hardware (e.g., 'A100_80GB', 'H100_80GB')
        
    Returns:
        Hardware configuration dict
        
    Raises:
        ValueError: If hardware name is not recognized
    """
    if hardware_name not in HARDWARE_CONFIGS:
        raise ValueError(
            f"Unknown hardware: {hardware_name}. "
            f"Available options: {list(HARDWARE_CONFIGS.keys())}"
        )
    
    return HARDWARE_CONFIGS[hardware_name].copy()