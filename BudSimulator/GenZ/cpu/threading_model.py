from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class ThreadConfig:
    num_threads: int
    threads_per_core: int  # For SMT
    thread_to_core_mapping: Dict[int, int]
    memory_affinity: Dict[int, int]  # thread -> NUMA node


class ThreadingModel:
    """Model threading behavior and parallel efficiency"""
    
    def __init__(self, cpu_config, numa_topology):
        self.cpu_config = cpu_config
        self.numa_topology = numa_topology
        
    def get_optimal_thread_config(self, operator, system) -> ThreadConfig:
        """Determine optimal threading configuration"""
        
        # Get operator characteristics
        compute_intensity = self._get_compute_intensity(operator)
        memory_footprint = self._estimate_memory_footprint(operator)
        
        # Check if this is a decode operation (small M dimension)
        is_decode = self._is_decode_operation(operator)
        
        if is_decode:
            # Decode operations: much smaller parallelism for M=1 workloads
            # M=1 means very small per-thread work → limit threads aggressively
            batch_size = getattr(operator, 'dim', [1])[0] if hasattr(operator, 'dim') else 1
            kv_size = getattr(operator, 'dim', [1, 1, 1, 128])[3] if hasattr(operator, 'dim') else 128
            
            # For decode, threads should scale with batch size primarily 
            # Each batch element can be processed independently
            
            # Base threads: scale with batch size but cap for efficiency
            # Rule: ~1 thread per 4 batch elements, but min 4 threads, max 32 threads
            optimal_threads = max(4, min(32, int(batch_size / 4)))
            
            # For very large batches, use more parallelism
            if batch_size >= 64:
                optimal_threads = min(48, int(batch_size / 2))
            
            # Use physical cores only (no SMT) for better efficiency with decode
            num_threads = optimal_threads
            threads_per_core = 1
            
        elif compute_intensity > 10:  # Compute bound (prefill)
            # Use all physical cores
            num_threads = (self.cpu_config.cores_per_socket * 
                          self.cpu_config.sockets)
            threads_per_core = 1  # No SMT for compute-bound
        else:  # Memory bound
            # May benefit from SMT to hide memory latency
            num_threads = (self.cpu_config.cores_per_socket * 
                          self.cpu_config.sockets * 
                          self.cpu_config.threads_per_core)
            threads_per_core = self.cpu_config.threads_per_core
            
        # Determine memory affinity
        memory_affinity = self._determine_memory_affinity(
            operator, num_threads
        )
        
        # Optimize thread placement
        thread_mapping = self.numa_topology.optimize_thread_placement(
            memory_affinity
        )
        
        return ThreadConfig(
            num_threads=num_threads,
            threads_per_core=threads_per_core,
            thread_to_core_mapping=thread_mapping,
            memory_affinity=memory_affinity
        )
        
    def calculate_parallel_efficiency(self, operator, thread_config: ThreadConfig,
                                    system) -> float:
        """Calculate efficiency of parallel execution"""
        
        # Base parallel fraction (Amdahl's law)
        parallel_fraction = self._get_parallel_fraction(operator)
        
        # Synchronization overhead
        sync_overhead = self._calculate_sync_overhead(
            thread_config.num_threads, operator
        )
        
        # Memory bandwidth contention
        bandwidth_saturation = self._calculate_bandwidth_saturation(
            operator, thread_config, system
        )
        
        # NUMA effects
        numa_overhead = self._calculate_numa_overhead(
            thread_config, operator
        )
        
        # SMT resource contention
        smt_efficiency = 1.0
        if thread_config.threads_per_core > 1:
            smt_efficiency = self._calculate_smt_efficiency(operator)
            
        # Combined efficiency
        efficiency = (
            parallel_fraction *
            (1 - sync_overhead) *
            (1 - bandwidth_saturation) *
            (1 - numa_overhead) *
            smt_efficiency
        )
        
        return min(efficiency, 1.0)
        
    def _get_compute_intensity(self, operator) -> float:
        """Calculate compute intensity (ops/byte)"""
        num_ops = operator.get_num_ops()
        num_bytes = operator.get_num_data() if hasattr(operator, 'get_num_data') else num_ops / 10
        return num_ops / num_bytes if num_bytes > 0 else 1.0
    
    def _is_decode_operation(self, operator) -> bool:
        """Check if this is a decode operation (M=1 for attention/GEMM)"""
        op_type = operator.__class__.__name__
        
        if op_type in ['GEMM', 'FC'] and hasattr(operator, 'get_gemms'):
            M, N, K = operator.get_gemms()[:3]
            return M == 1  # Decode has M=1
        elif op_type in ['Logit', 'Attend'] and hasattr(operator, 'dim'):
            # For attention operators, M=1 indicates decode
            M = operator.dim[2] if len(operator.dim) > 2 else 1
            return M == 1
        
        return False
        
    def _estimate_memory_footprint(self, operator) -> Dict[int, int]:
        """Estimate memory footprint for threads"""
        # Simplified: assume round-robin distribution
        num_threads = self.cpu_config.cores_per_socket * self.cpu_config.sockets
        memory_affinity = {}
        
        for i in range(num_threads):
            # Assign to NUMA node in round-robin
            memory_affinity[i] = i % self.numa_topology.num_nodes
            
        return memory_affinity
        
    def _determine_memory_affinity(self, operator, num_threads: int) -> Dict[int, int]:
        """Determine memory affinity for threads"""
        memory_affinity = {}
        
        # Simple strategy: distribute threads across NUMA nodes
        for i in range(num_threads):
            memory_affinity[i] = i % self.numa_topology.num_nodes
            
        return memory_affinity
        
    def _get_parallel_fraction(self, operator) -> float:
        """Get parallelizable fraction of the operator"""
        op_type = operator.__class__.__name__
        
        # Different operators have different parallelization potential
        if op_type == 'GEMM':
            return 0.95  # Highly parallel
        elif op_type in ['Logit', 'Attend']:
            return 0.90  # Good parallelism
        else:
            return 0.85  # Default
            
    def _calculate_sync_overhead(self, num_threads: int, operator) -> float:
        """Calculate synchronization overhead"""
        op_type = operator.__class__.__name__
        
        # Different operators have different sync requirements
        if op_type == 'GEMM':
            # GEMM has minimal sync (just work distribution)
            base_overhead = 0.001
        elif op_type in ['Logit', 'Attend']:
            # Attention may need more synchronization
            base_overhead = 0.01
        else:
            base_overhead = 0.005
            
        # Overhead increases with thread count
        return base_overhead * np.log2(num_threads)
        
    def _calculate_bandwidth_saturation(self, operator, thread_config: ThreadConfig,
                                      system) -> float:
        """Calculate memory bandwidth saturation effects"""
        
        # Check if this is a decode operation
        is_decode = self._is_decode_operation(operator)
        
        # Estimate bandwidth requirement per thread
        data_volume = operator.get_num_data() if hasattr(operator, 'get_num_data') else operator.get_num_ops() * 4  # Assume 4 bytes per op
        compute_time = operator.get_num_ops() / (system.peak_fp32_tflops * 1e12) if hasattr(system, 'peak_fp32_tflops') else 1.0
        
        # Bandwidth per thread in GB/s
        bandwidth_per_thread = (data_volume / 1e9) / compute_time / thread_config.num_threads
        
        # Total bandwidth requirement in GB/s
        total_bandwidth_required = bandwidth_per_thread * thread_config.num_threads
        
        # Available bandwidth in GB/s
        available_bandwidth = system.peak_memory_bandwidth if hasattr(system, 'peak_memory_bandwidth') else 200
        
        # For decode operations, apply much lower bandwidth penalties
        # M=1 operations have high compute intensity per byte accessed
        if is_decode:
            # Decode operations are more compute-bound per memory access
            # Reduce bandwidth requirements significantly
            total_bandwidth_required *= 0.3  # Decode uses bandwidth more efficiently
            
        # Saturation model (queuing theory)
        utilization = total_bandwidth_required / available_bandwidth
        
        # Clamp utilization to reasonable range
        utilization = min(utilization, 2.0)
        
        if utilization < 0.5:
            return 0.0
        elif utilization < 0.8:
            return (utilization - 0.5) * 0.5 / 0.3  # Linear from 0 to 0.5
        else:
            # Severe saturation
            penalty = min(0.5 + (utilization - 0.8) * 0.5 / 1.2, 0.9)  # Cap at 0.9
            # Further reduce penalty for decode
            if is_decode:
                penalty *= 0.5  # Less sensitive to bandwidth saturation
            return penalty
            
    def _calculate_numa_overhead(self, thread_config: ThreadConfig, operator) -> float:
        """Calculate NUMA-related overhead"""
        
        # Get memory access pattern
        local_accesses = 0
        remote_accesses = 0
        
        for thread_id, core_id in thread_config.thread_to_core_mapping.items():
            thread_node = self.numa_topology.get_numa_node(core_id)
            memory_node = thread_config.memory_affinity.get(thread_id, 0)
            
            if thread_node == memory_node:
                local_accesses += 1
            else:
                remote_accesses += 1
                
        # Remote access penalty
        remote_penalty = 0.5  # 50% slower
        numa_overhead = (remote_accesses / (local_accesses + remote_accesses)) * remote_penalty if (local_accesses + remote_accesses) > 0 else 0
        
        return numa_overhead
        
    def _calculate_smt_efficiency(self, operator) -> float:
        """Calculate SMT (Hyper-Threading) efficiency"""
        
        # SMT efficiency depends on instruction mix
        op_type = operator.__class__.__name__
        
        if op_type == 'GEMM':
            # Compute-heavy, limited benefit from SMT
            return 1.15  # 15% benefit
        elif op_type in ['Logit', 'Attend']:
            # More memory access, better SMT benefit
            return 1.3   # 30% benefit
        else:
            return 1.2   # 20% benefit 