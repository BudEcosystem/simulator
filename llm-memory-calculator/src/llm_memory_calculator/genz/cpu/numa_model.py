from typing import Dict, Tuple
import numpy as np


class NUMATopology:
    """Model NUMA topology and memory access costs"""
    
    def __init__(self, cpu_config):
        self.cpu_config = cpu_config
        self.num_nodes = cpu_config.numa_nodes
        self.cores_per_node = cpu_config.cores_per_numa
        self.distance_matrix = cpu_config.numa_distance_matrix
        
        # Memory allocation tracker
        self.memory_allocation = {}  # address_range -> numa_node
        self._next_address = 0
        
    def get_numa_node(self, core_id: int) -> int:
        """Get NUMA node for given core"""
        return core_id // self.cores_per_node
        
    def get_memory_node(self, address: int) -> int:
        """Get NUMA node where memory is allocated"""
        for (start, end), node in self.memory_allocation.items():
            if start <= address < end:
                return node
        # Default: first-touch policy
        return 0
        
    def get_access_penalty(self, core_id: int, memory_address: int) -> float:
        """Calculate NUMA penalty for memory access"""
        core_node = self.get_numa_node(core_id)
        mem_node = self.get_memory_node(memory_address)
        
        distance = self.distance_matrix[core_node, mem_node]
        
        # Convert distance to penalty
        # Local access (distance=10) -> 1.0x
        # Remote access (distance=21) -> ~2.1x
        return distance / 10.0
        
    def allocate_memory(self, size: int, preferred_node: int = -1) -> Tuple[int, int]:
        """
        Allocate memory on NUMA node
        Returns: (start_address, end_address)
        """
        if preferred_node == -1:
            # Round-robin allocation
            preferred_node = len(self.memory_allocation) % self.num_nodes
            
        # Simplified: just track allocation
        start = self._next_address
        end = start + size
        self._next_address = end
        
        self.memory_allocation[(start, end)] = preferred_node
        return start, end
        
    def optimize_thread_placement(self, memory_footprint: Dict[int, int]) -> Dict[int, int]:
        """
        Optimize thread to core mapping based on memory access
        memory_footprint: thread_id -> numa_node of most accessed memory
        Returns: thread_id -> core_id mapping
        """
        thread_to_core = {}
        used_cores = set()
        
        # Group threads by their preferred NUMA node
        threads_by_node = {}
        for thread_id, mem_node in memory_footprint.items():
            if mem_node not in threads_by_node:
                threads_by_node[mem_node] = []
            threads_by_node[mem_node].append(thread_id)
            
        # Assign threads to cores on their preferred NUMA node
        for node, threads in threads_by_node.items():
            node_cores = list(range(node * self.cores_per_node, 
                                  (node + 1) * self.cores_per_node))
            
            for i, thread_id in enumerate(threads):
                if i < len(node_cores):
                    core_id = node_cores[i]
                    thread_to_core[thread_id] = core_id
                    used_cores.add(core_id)
                else:
                    # Spill to another node
                    total_cores = self.cpu_config.cores_per_socket * self.cpu_config.sockets
                    for c in range(total_cores):
                        if c not in used_cores:
                            thread_to_core[thread_id] = c
                            used_cores.add(c)
                            break
                            
        return thread_to_core 