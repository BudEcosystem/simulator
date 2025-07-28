"""
CPU Integration for GenZ LLM Inference
This module provides seamless integration of CPU systems with existing GenZ prefill and decode methods.
"""
from typing import Union, Dict, Any
from GenZ.system import System
from GenZ.cpu.cpu_system import CPUSystem
from GenZ.cpu.operator_enhancement import enhance_operators_for_cpu, restore_original_operators
from GenZ.LLM_inference import prefill_moddeling, decode_moddeling
from functools import wraps
import warnings
from .cpu_configs import CPU_PRESETS


def is_cpu_system(system: Union[str, Dict, System]) -> bool:
    """
    Check if the provided system is a CPU system.
    
    Args:
        system: System specification (string, dict, or System object)
        
    Returns:
        bool: True if it's a CPU system
    """
    if isinstance(system, CPUSystem):
        return True
    
    if isinstance(system, dict):
        # Check for CPU-specific keys
        return any(key in system for key in ['cache_configs', 'numa_config', 'isa_support'])
    
    if isinstance(system, str):
        # Check if it's in CPU_PRESETS
        return system.lower() in [key.lower() for key in CPU_PRESETS.keys()]
    
    return False


def cpu_aware_prefill_moddeling(model='BERT', batch_size=1, input_tokens=4096,
                               system_name='A100_40GB_GPU', system_eff=1, bits='bf16', 
                               debug=False, model_profilling=False,
                               tensor_parallel=1, pipeline_parallel=1, expert_parallel=1,
                               collective_strategy='GenZ', network_config=None,
                               parallelism_heirarchy="TP{1}_EP{1}_PP{1}",
                               model_offload=False, **kwargs):
    """
    CPU-aware wrapper for prefill_moddeling that automatically handles CPU systems.
    
    This function detects if a CPU system is being used and automatically:
    1. Enhances operators with CPU-specific implementations
    2. Calls the original prefill_moddeling
    3. Restores original operators afterward
    
    All parameters are passed through to the original prefill_moddeling function.
    """
    # Check if CPU system is being used
    if is_cpu_system(system_name):
        # Enable CPU support
        enhance_operators_for_cpu()
        
        # Convert string CPU preset to CPUSystem if needed
        if isinstance(system_name, str) and is_cpu_system(system_name):
            from GenZ.cpu import create_cpu_system
            system_name = create_cpu_system(system_name)
            
        if debug:
            print(f"CPU system detected. Using CPU-enhanced operators.")
        
        try:
            # Call original function with CPU enhancements
            result = prefill_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                system_name=system_name,
                system_eff=system_eff,
                bits=bits,
                debug=debug,
                model_profilling=model_profilling,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
                expert_parallel=expert_parallel,
                collective_strategy=collective_strategy,
                network_config=network_config,
                parallelism_heirarchy=parallelism_heirarchy,
                model_offload=model_offload,
                **kwargs
            )
            # Apply calibration
            if isinstance(system_name, CPUSystem):
                calibrated_latency = result['Latency'] - system_name.cpu_config.framework_overhead_ms
                result['Latency'] = max(0, calibrated_latency)
        finally:
            # Always restore original operators
            restore_original_operators()
            
        return result
    else:
        # Not a CPU system, use original function
        return prefill_moddeling(
            model=model,
            batch_size=batch_size,
            input_tokens=input_tokens,
            system_name=system_name,
            system_eff=system_eff,
            bits=bits,
            debug=debug,
            model_profilling=model_profilling,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            collective_strategy=collective_strategy,
            network_config=network_config,
            parallelism_heirarchy=parallelism_heirarchy,
            model_offload=model_offload,
            **kwargs
        )


def cpu_aware_decode_moddeling(model='BERT', batch_size=1, input_tokens=4096,
                              output_tokens=0, Bb=4,
                              system_name='A100_40GB_GPU', system_eff=1, bits='bf16', 
                              debug=False, model_profilling=False,
                              tensor_parallel=1, pipeline_parallel=1, expert_parallel=1,
                              collective_strategy='GenZ', network_config=None,
                              parallelism_heirarchy="TP{1}_EP{1}_PP{1}",
                              model_offload=False, ceff=None, meff=None, **kwargs):
    """
    CPU-aware wrapper for decode_moddeling that automatically handles CPU systems.
    
    This function detects if a CPU system is being used and automatically:
    1. Enhances operators with CPU-specific implementations
    2. Calls the original decode_moddeling
    3. Restores original operators afterward
    
    All parameters are passed through to the original decode_moddeling function.
    """
    # Check if CPU system is being used
    if is_cpu_system(system_name):
        # Enable CPU support
        enhance_operators_for_cpu()
        
        # Convert string CPU preset to CPUSystem if needed
        if isinstance(system_name, str) and is_cpu_system(system_name):
            from GenZ.cpu import create_cpu_system
            system_name = create_cpu_system(system_name)
            
        if debug:
            print(f"CPU system detected. Using CPU-enhanced operators.")
        
        try:
            # Call original function with CPU enhancements
            result = decode_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                Bb=Bb,
                system_name=system_name,
                system_eff=system_eff,
                bits=bits,
                debug=debug,
                model_profilling=model_profilling,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
                expert_parallel=expert_parallel,
                collective_strategy=collective_strategy,
                network_config=network_config,
                parallelism_heirarchy=parallelism_heirarchy,
                model_offload=model_offload,
                ceff=ceff,
                meff=meff,
                **kwargs
            )
            # Apply calibration
            if isinstance(system_name, CPUSystem):
                calibrated_latency = result['Latency'] - system_name.cpu_config.framework_overhead_ms
                result['Latency'] = max(0, calibrated_latency)
        finally:
            # Always restore original operators
            restore_original_operators()
            
        return result
    else:
        # Not a CPU system, use original function
        return decode_moddeling(
            model=model,
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            Bb=Bb,
            system_name=system_name,
            system_eff=system_eff,
            bits=bits,
            debug=debug,
            model_profilling=model_profilling,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            collective_strategy=collective_strategy,
            network_config=network_config,
            parallelism_heirarchy=parallelism_heirarchy,
            model_offload=model_offload,
            ceff=ceff,
            meff=meff,
            **kwargs
        )


def auto_detect_system_decorator(func):
    """
    Decorator that automatically detects CPU systems and applies enhancements.
    Can be used to wrap any function that takes a system_name parameter.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to find system_name in args or kwargs
        system_name = None
        if 'system_name' in kwargs:
            system_name = kwargs['system_name']
        elif len(args) >= 4:  # Assuming system_name is 4th positional arg
            system_name = args[3]
            
        if system_name and is_cpu_system(system_name):
            enhance_operators_for_cpu()
            try:
                result = func(*args, **kwargs)
            finally:
                restore_original_operators()
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper


# Monkey-patch the original functions to add CPU awareness
def enable_cpu_aware_inference():
    """
    Enable CPU-aware inference by patching the original GenZ functions.
    This allows existing code to automatically use CPU enhancements without changes.
    """
    import GenZ.LLM_inference.llm_prefill as prefill_module
    import GenZ.LLM_inference.llm_decode as decode_module
    
    # Store original functions
    if not hasattr(prefill_module, '_original_prefill_moddeling'):
        prefill_module._original_prefill_moddeling = prefill_module.prefill_moddeling
        prefill_module.prefill_moddeling = cpu_aware_prefill_moddeling
        
    if not hasattr(decode_module, '_original_decode_moddeling'):
        decode_module._original_decode_moddeling = decode_module.decode_moddeling
        decode_module.decode_moddeling = cpu_aware_decode_moddeling
        
    warnings.warn("CPU-aware inference enabled. GenZ will automatically detect and use CPU systems.")


def disable_cpu_aware_inference():
    """
    Disable CPU-aware inference by restoring original GenZ functions.
    """
    import GenZ.LLM_inference.llm_prefill as prefill_module
    import GenZ.LLM_inference.llm_decode as decode_module
    
    # Restore original functions if they were patched
    if hasattr(prefill_module, '_original_prefill_moddeling'):
        prefill_module.prefill_moddeling = prefill_module._original_prefill_moddeling
        delattr(prefill_module, '_original_prefill_moddeling')
        
    if hasattr(decode_module, '_original_decode_moddeling'):
        decode_module.decode_moddeling = decode_module._original_decode_moddeling
        delattr(decode_module, '_original_decode_moddeling')
        
    warnings.warn("CPU-aware inference disabled. GenZ will use original implementations.") 