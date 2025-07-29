"""
GenZ CPU Support Package
"""
from .cpu_system import CPUSystem
from .cpu_configs import CPUConfig, get_cpu_preset, CPU_PRESETS
from .operator_enhancement import enhance_operators_for_cpu, restore_original_operators
from .llm_cpu_integration import (
    cpu_aware_prefill_moddeling,
    cpu_aware_decode_moddeling,
    enable_cpu_aware_inference,
    disable_cpu_aware_inference,
    is_cpu_system
)


def create_cpu_system(cpu_model='intel_xeon_8380', **kwargs):
    """
    Create a CPU system with predefined configurations
    
    Args:
        cpu_model: One of 'intel_xeon_8380', 'amd_epyc_7763', 'aws_graviton3', etc.
        **kwargs: Additional parameters to override defaults
        
    Returns:
        CPUSystem: Configured CPU system
    """
    from llm_memory_calculator.genz.system import System
    from llm_memory_calculator.genz.unit import Unit
    
    # Handle case-insensitive lookup
    cpu_model_lower = cpu_model.lower()
    matching_key = None
    for key in CPU_PRESETS.keys():
        if key.lower() == cpu_model_lower:
            matching_key = key
            break
    
    if matching_key is None:
        raise ValueError(f"Unknown CPU model: {cpu_model}. Available: {list(CPU_PRESETS.keys())}")
    
    preset = CPU_PRESETS[matching_key]
    
    # Create base system
    unit = Unit()
    base_params = preset['base_params'].copy()
    base_params['unit'] = unit
    
    # Override base params with kwargs if provided
    for key in ['frequency', 'bits', 'compute_efficiency', 'memory_efficiency']:
        if key in kwargs:
            base_params[key] = kwargs[key]
    
    base_system = System(**base_params)
    
    # Get CPU config
    cpu_config = preset['cpu_specific']
    
    # Override CPU config with kwargs if provided
    for key, value in kwargs.items():
        if hasattr(cpu_config, key):
            setattr(cpu_config, key, value)
    
    return CPUSystem(base_system, cpu_config)


__all__ = [
    'CPUSystem',
    'create_cpu_system',
    'CPUConfig',
    'get_cpu_preset',
    'enhance_operators_for_cpu',
    'restore_original_operators',
    'cpu_aware_prefill_moddeling',
    'cpu_aware_decode_moddeling',
    'enable_cpu_aware_inference',
    'disable_cpu_aware_inference',
    'is_cpu_system'
] 