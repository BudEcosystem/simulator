"""Enhance existing GenZ operators with CPU support"""
from typing import Dict, Any
import numpy as np


def enhance_operators_for_cpu():
    """Enhance existing operators with CPU support"""
    
    # Import all operator classes
    from llm_memory_calculator.genz.operators import GEMM, Logit, Attend, FC, CONV2D
    from .cpu_operator import CPUOperatorMixin
    
    # List of operator classes to enhance
    operator_classes = [GEMM, Logit, Attend, FC, CONV2D]
    
    for op_class in operator_classes:
        # Add CPU mixin to the class
        if CPUOperatorMixin not in op_class.__bases__:
            op_class.__bases__ = (CPUOperatorMixin,) + op_class.__bases__
        
        # Save original methods
        if not hasattr(op_class, '_original_get_roofline'):
            op_class._original_get_roofline = op_class.get_roofline
            op_class._original_get_memory_time = op_class.get_memory_time
            op_class._original_get_compute_time = op_class.get_compute_time
            op_class._original_get_model_characterstics = op_class.get_model_characterstics
        
        # Override with CPU-aware versions
        op_class.get_roofline = cpu_aware_get_roofline
        op_class.get_memory_time = cpu_aware_get_memory_time
        op_class.get_compute_time = cpu_aware_get_compute_time
        op_class.get_model_characterstics = cpu_aware_get_model_characterstics


def cpu_aware_get_roofline(self, system, unit):
    """CPU-aware roofline method"""
    if hasattr(system, '__class__') and system.__class__.__name__ == 'CPUSystem':
        return get_cpu_roofline(self, system, unit)
    else:
        return self._original_get_roofline(system, unit)


def cpu_aware_get_memory_time(self, system):
    """CPU-aware memory time calculation"""
    if hasattr(system, '__class__') and system.__class__.__name__ == 'CPUSystem':
        return self.get_cpu_memory_time(system)
    else:
        return self._original_get_memory_time(system)


def cpu_aware_get_compute_time(self, system):
    """CPU-aware compute time calculation"""
    if hasattr(system, '__class__') and system.__class__.__name__ == 'CPUSystem':
        return self.get_cpu_compute_time(system)
    else:
        return self._original_get_compute_time(system)


def cpu_aware_get_model_characterstics(self, system, unit):
    """CPU-aware model characteristics method"""
    if hasattr(system, '__class__') and system.__class__.__name__ == 'CPUSystem':
        # For CPU systems, return the same format as original but without timing info
        result = self._original_get_model_characterstics(system.base_system, unit)
        # Don't add CPU-specific columns to model characteristics
        return result
    else:
        return self._original_get_model_characterstics(system, unit)


def get_cpu_roofline(self, system, unit) -> Dict[str, Any]:
    """CPU-specific roofline calculation"""
    
    # Get base roofline first to ensure we have all required columns
    base_result = self._original_get_roofline(system.base_system, unit)
    
    # Get optimal configuration for this operator
    isa, isa_efficiency = system.isa_selector.select_isa(self, system.base_system.bits)
    thread_config = system.threading_model.get_optimal_thread_config(self, system)
    
    # Calculate compute time with all CPU effects
    compute_time = self.get_cpu_compute_time(system)
    
    # Calculate memory time through cache hierarchy
    memory_time = self.get_cpu_memory_time(system)
    
    # Communication time (if multi-socket)
    # For CPU systems, we don't have collective communication
    comm_time = 0
    
    # Apply parallel efficiency (already included in compute_time)
    parallel_efficiency = system.threading_model.calculate_parallel_efficiency(
        self, thread_config, system
    )
    
    # Update timing values: on CPUs compute and memory are not fully overlapped.
    # Use additive model (compute + memory + comm) to better approximate latency.
    exec_time = compute_time + memory_time + comm_time
    
    # Determine boundedness
    com_to_mem_ratio = compute_time / memory_time if memory_time else 0
    if com_to_mem_ratio > 1.5:
        boundedness = 'Compute'
    elif com_to_mem_ratio < 0.67:
        boundedness = 'Memory'
    else:
        boundedness = 'Balanced'
        
    # Get cache hit rates
    cache_sim = system.cache_hierarchy
    accesses = cache_sim.analyze_operator_access_pattern(self)
    cache_results = cache_sim.simulate_data_access(accesses) if accesses else {
        'l1_hit_rate': 0.9,
        'l2_hit_rate': 0.05,
        'l3_hit_rate': 0.04,
        'dram_access_rate': 0.01,
        'total_cycles': 100
    }
    
    # Update base result with CPU-specific values
    base_result.update({
        'Bound': boundedness,
        'C/M ratio': com_to_mem_ratio,
        f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
        f'Compute time ({unit.unit_time})': unit.raw_to_unit(compute_time, type='T'),
        f'Memory time ({unit.unit_time})': unit.raw_to_unit(memory_time, type='T'),
        f'Communication time ({unit.unit_time})': unit.raw_to_unit(comm_time, type='T'),
        # CPU-specific metrics
        'ISA_used': isa.value,
        'ISA_efficiency': isa_efficiency,
        'Thread_count': thread_config.num_threads,
        'Threads_per_core': thread_config.threads_per_core,
        'Parallel_efficiency': parallel_efficiency,
        'Frequency_GHz': system.frequency_governor.get_frequency(
            isa, thread_config.num_threads, 'compute'
        ) / 1e9,
        'L1_hit_rate': cache_results['l1_hit_rate'],
        'L2_hit_rate': cache_results['l2_hit_rate'],
        'L3_hit_rate': cache_results['l3_hit_rate'],
        'DRAM_access_rate': cache_results['dram_access_rate']
    })
    
    return base_result 


def restore_original_operators():
    """
    Restore original operator implementations
    """
    from llm_memory_calculator.genz.operators import GEMM, Logit, FC, Attend, CONV2D
    
    # List of operator classes to restore
    operator_classes = [GEMM, Logit, Attend, FC, CONV2D]
    
    for op_class in operator_classes:
        # Restore original methods
        if hasattr(op_class, '_original_get_roofline'):
            op_class.get_roofline = op_class._original_get_roofline
            delattr(op_class, '_original_get_roofline')
            
        if hasattr(op_class, '_original_get_memory_time'):
            op_class.get_memory_time = op_class._original_get_memory_time
            delattr(op_class, '_original_get_memory_time')
            
        if hasattr(op_class, '_original_get_compute_time'):
            op_class.get_compute_time = op_class._original_get_compute_time
            delattr(op_class, '_original_get_compute_time')
            
        if hasattr(op_class, '_original_get_model_characterstics'):
            op_class.get_model_characterstics = op_class._original_get_model_characterstics
            delattr(op_class, '_original_get_model_characterstics')
            
        # Remove CPU mixin from bases if it was added
        from .cpu_operator import CPUOperatorMixin
        if CPUOperatorMixin in op_class.__bases__:
            op_class.__bases__ = tuple(b for b in op_class.__bases__ if b != CPUOperatorMixin) 