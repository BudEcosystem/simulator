import logging
import os
from typing import Optional, Dict, Tuple
import numpy as np
from .cache_model import AccessPattern

# Setup a dedicated logger for CPU performance debugging
cpu_log_file = '/tmp/cpu_perf_log.txt'
cpu_logger = logging.getLogger('cpu_perf')
cpu_logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if the module is reloaded
if not cpu_logger.handlers:
    # Clear log file at the start of a run
    if os.path.exists(cpu_log_file):
        os.remove(cpu_log_file)
    handler = logging.FileHandler(cpu_log_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    cpu_logger.addHandler(handler)


class CPUOperatorMixin:
    """Mixin to add CPU-specific functionality to operators"""
    
    def get_cpu_memory_time(self, system: 'CPUSystem') -> float:
        """Calculate memory access time through cache hierarchy"""
        
        # Get access pattern for this operator
        access_pattern = self._determine_access_pattern()
        
        # Simulate through cache hierarchy
        cache_sim = system.cache_hierarchy
        accesses = cache_sim.analyze_operator_access_pattern(self)
        results = cache_sim.simulate_data_access(accesses)
        
        # Convert cycles to time
        total_time = results['total_cycles'] / system.cpu_config.base_frequency
        
        # Add bandwidth constraints
        data_volume = self._calculate_data_volume()
        bandwidth_time = self._calculate_bandwidth_limited_time(
            data_volume, results, system
        )
        
        final_memory_time = max(total_time, bandwidth_time)
        cpu_logger.debug(f"MEMORY_TIME;{self.__class__.__name__};{final_memory_time}")
        return final_memory_time
        
    def _determine_access_pattern(self) -> AccessPattern:
        """Determine the memory access pattern for this operator"""
        op_type = self.__class__.__name__
        
        if op_type in ['GEMM', 'FC']:
            return AccessPattern.TILED  # Assumes cache-blocked GEMM
        elif op_type in ['Logit', 'Attend']:
            # Attention has less regular patterns
            return AccessPattern.STRIDED
        else:
            return AccessPattern.SEQUENTIAL
            
    def _calculate_data_volume(self) -> Dict[str, float]:
        """Calculate data volume for this operator"""
        # Get tensors
        tensors = self.get_tensors()
        if len(tensors) >= 3:
            input_a, input_w, output = tensors[:3]
            
            # Calculate sizes in bytes (assuming fp32)
            bytes_per_element = 4
            input_a_size = np.prod(input_a) * bytes_per_element
            input_w_size = np.prod(input_w) * bytes_per_element
            output_size = np.prod(output) * bytes_per_element
            
            return {
                'input_a': input_a_size,
                'input_w': input_w_size,
                'output': output_size,
                'total': input_a_size + input_w_size + output_size
            }
        else:
            # Fallback for operators without standard tensor structure
            num_ops = self.get_num_ops()
            return {'total': num_ops * 4}  # Rough estimate
            
    def _calculate_bandwidth_limited_time(self, data_volume: Dict[str, float],
                                        cache_results: Dict[str, float],
                                        system: 'CPUSystem') -> float:
        """Calculate time limited by bandwidth at each level"""
        time = 0
        
        # L1 bandwidth
        l1_volume = data_volume['total'] * cache_results['l1_hit_rate']
        time += l1_volume / (system.cpu_config.l1d_config.bandwidth * 1e9)
        
        # L2 bandwidth  
        l2_volume = data_volume['total'] * cache_results['l2_hit_rate']
        time += l2_volume / (system.cpu_config.l2_config.bandwidth * 1e9)
        
        # L3 bandwidth (shared)
        l3_volume = data_volume['total'] * cache_results['l3_hit_rate']
        active_cores = system.get_active_cores()
        l3_bandwidth = system.cpu_config.l3_config.bandwidth / np.sqrt(active_cores)
        time += l3_volume / (l3_bandwidth * 1e9)
        
        # DRAM bandwidth – for compute-heavy GEMM we assume most data eventually hits DRAM.
        dram_access_rate = cache_results['dram_access_rate']
        if self.__class__.__name__ == 'GEMM' and dram_access_rate < 0.8:
            dram_access_rate = 1.0  # worst-case – all data fetched
        dram_volume = data_volume['total'] * dram_access_rate
        time += dram_volume / (system.peak_memory_bandwidth * 1e9)
        
        return time
        
    def get_cpu_compute_time(self, system: 'CPUSystem') -> float:
        """Calculate compute time on CPU system"""
        # Select optimal ISA
        data_type = system.base_system.bits if hasattr(system, 'base_system') else 'fp32'
        isa, efficiency = system.isa_selector.select_isa(self, data_type)
        
        # Get optimal thread configuration
        thread_config = system.threading_model.get_optimal_thread_config(self, system) if hasattr(system, 'threading_model') else None
        
        if thread_config:
            num_threads = thread_config.num_threads
            # Base compute time assuming ideal utilisation
            base_time = system.isa_selector.get_compute_time(
                self, isa, num_threads, system
            )
            # Parallel-efficiency (threading) penalty
            parallel_eff = system.threading_model.calculate_parallel_efficiency(
                self, thread_config, system
            )
        else:
            num_threads = system.get_active_cores()
            base_time = system.isa_selector.get_compute_time(
                self, isa, num_threads, system
            )
            parallel_eff = 1.0

        # ---------------------------------------------------------------------------------
        # NEW: incorporate ISA-level efficiency returned by select_isa (e.g. 0.85 or 0.70).
        # This accounts for pipeline bubbles, vector utilisation and non-FMA instructions
        # that make real kernels slower than the theoretical peak.
        # ---------------------------------------------------------------------------------
        isa_efficiency = efficiency if efficiency > 0 else 0.5  # safety guard

        # Effective compute time is longer when efficiency < 1 and/or parallel_eff < 1.
        effective_time = base_time / (isa_efficiency * parallel_eff)

        cpu_logger.debug(f"COMPUTE_TIME;{self.__class__.__name__};{effective_time}")
        return effective_time
            
    def optimize_for_cpu_cache(self, system: 'CPUSystem') -> 'Operator':
        """Return cache-optimized version of this operator"""
        op_type = self.__class__.__name__
        
        if op_type == 'GEMM':
            return self._optimize_gemm_tiling(system)
        elif op_type in ['Logit', 'Attend']:
            return self._optimize_attention_blocking(system)
        else:
            return self  # No optimization
            
    def _optimize_gemm_tiling(self, system: 'CPUSystem') -> 'Operator':
        """Optimize GEMM tiling for cache hierarchy"""
        # This is a placeholder - in a real implementation,
        # this would return a modified operator with optimal tiling
        return self
        
    def _optimize_attention_blocking(self, system: 'CPUSystem') -> 'Operator':
        """Optimize attention blocking for cache"""
        # This is a placeholder - in a real implementation,
        # this would return a modified operator with optimal blocking
        return self 