"""Tests for CPU NUMA and threading models"""
import unittest
import numpy as np
from GenZ.cpu.numa_model import NUMATopology
from GenZ.cpu.threading_model import ThreadingModel, ThreadConfig
from GenZ.cpu import create_cpu_system
from GenZ.operators import GEMM, Logit


class TestNUMATopology(unittest.TestCase):
    """Test NUMA topology modeling"""
    
    def setUp(self):
        """Set up test CPU system"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        self.numa = NUMATopology(self.cpu_system.cpu_config)
        
    def test_numa_initialization(self):
        """Test NUMA topology initialization"""
        self.assertEqual(self.numa.num_nodes, 2)
        self.assertEqual(self.numa.cores_per_node, 40)
        self.assertEqual(self.numa.distance_matrix.shape, (2, 2))
        
    def test_numa_node_mapping(self):
        """Test core to NUMA node mapping"""
        # First 40 cores on node 0
        self.assertEqual(self.numa.get_numa_node(0), 0)
        self.assertEqual(self.numa.get_numa_node(39), 0)
        
        # Next 40 cores on node 1
        self.assertEqual(self.numa.get_numa_node(40), 1)
        self.assertEqual(self.numa.get_numa_node(79), 1)
        
    def test_numa_distance_matrix(self):
        """Test NUMA distance matrix"""
        # Local access
        self.assertEqual(self.numa.distance_matrix[0, 0], 10)
        self.assertEqual(self.numa.distance_matrix[1, 1], 10)
        
        # Remote access
        self.assertEqual(self.numa.distance_matrix[0, 1], 21)
        self.assertEqual(self.numa.distance_matrix[1, 0], 21)
        
    def test_memory_allocation(self):
        """Test memory allocation on NUMA nodes"""
        # Allocate on node 0
        start1, end1 = self.numa.allocate_memory(1024*1024, preferred_node=0)
        self.assertEqual(self.numa.get_memory_node(start1), 0)
        
        # Allocate on node 1
        start2, end2 = self.numa.allocate_memory(1024*1024, preferred_node=1)
        self.assertEqual(self.numa.get_memory_node(start2), 1)
        
        # Round-robin allocation
        start3, end3 = self.numa.allocate_memory(1024*1024)
        # Should allocate on different nodes in round-robin
        
    def test_access_penalty(self):
        """Test NUMA access penalty calculation"""
        # Allocate memory on node 0
        start, end = self.numa.allocate_memory(1024*1024, preferred_node=0)
        
        # Local access (core 0 accessing node 0 memory)
        local_penalty = self.numa.get_access_penalty(0, start)
        self.assertAlmostEqual(local_penalty, 1.0)
        
        # Remote access (core 40 on node 1 accessing node 0 memory)
        remote_penalty = self.numa.get_access_penalty(40, start)
        self.assertAlmostEqual(remote_penalty, 2.1)
        
    def test_thread_placement_optimization(self):
        """Test thread placement optimization"""
        # Create memory footprint - threads 0-3 access node 0, threads 4-7 access node 1
        memory_footprint = {
            0: 0, 1: 0, 2: 0, 3: 0,  # Threads accessing node 0
            4: 1, 5: 1, 6: 1, 7: 1   # Threads accessing node 1
        }
        
        mapping = self.numa.optimize_thread_placement(memory_footprint)
        
        # Threads 0-3 should be placed on cores from node 0
        for thread in range(4):
            core = mapping[thread]
            self.assertEqual(self.numa.get_numa_node(core), 0)
            
        # Threads 4-7 should be placed on cores from node 1
        for thread in range(4, 8):
            core = mapping[thread]
            self.assertEqual(self.numa.get_numa_node(core), 1)
            
    def test_amd_numa_topology(self):
        """Test AMD EPYC NUMA topology (8 nodes)"""
        amd_system = create_cpu_system('amd_epyc_7763')
        amd_numa = NUMATopology(amd_system.cpu_config)
        
        self.assertEqual(amd_numa.num_nodes, 8)
        self.assertEqual(amd_numa.cores_per_node, 16)
        
        # Check distance matrix shape
        self.assertEqual(amd_numa.distance_matrix.shape, (8, 8))
        
        # Local access
        self.assertEqual(amd_numa.distance_matrix[0, 0], 10)
        
        # Different distances for different nodes
        self.assertGreater(amd_numa.distance_matrix[0, 4], amd_numa.distance_matrix[0, 1])


class TestThreadingModel(unittest.TestCase):
    """Test threading and parallelism model"""
    
    def setUp(self):
        """Set up test CPU system"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        self.numa = NUMATopology(self.cpu_system.cpu_config)
        self.threading = ThreadingModel(self.cpu_system.cpu_config, self.numa)
        
    def test_thread_config_creation(self):
        """Test thread configuration creation"""
        config = ThreadConfig(
            num_threads=40,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(40)},
            memory_affinity={i: 0 for i in range(40)}
        )
        
        self.assertEqual(config.num_threads, 40)
        self.assertEqual(config.threads_per_core, 1)
        self.assertEqual(len(config.thread_to_core_mapping), 40)
        
    def test_optimal_thread_config_compute_bound(self):
        """Test optimal thread configuration for compute-bound workload"""
        # Create compute-intensive GEMM
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        config = self.threading.get_optimal_thread_config(gemm, self.cpu_system)
        
        # Should use physical cores only (no SMT) for compute-bound
        self.assertEqual(config.threads_per_core, 1)
        self.assertEqual(config.num_threads, 80)  # 2 sockets * 40 cores
        
    def test_optimal_thread_config_memory_bound(self):
        """Test optimal thread configuration for memory-bound workload"""
        # Create memory-intensive operation (small compute, large data)
        small_gemm = GEMM(['gemm', 1, 100, 100, 10], density=(1.0, 1.0, 1.0))
        
        # Mock compute intensity calculation
        self.threading._get_compute_intensity = lambda op: 0.5  # Low intensity
        
        config = self.threading.get_optimal_thread_config(small_gemm, self.cpu_system)
        
        # May use SMT for memory-bound workloads
        self.assertGreaterEqual(config.threads_per_core, 1)
        
    def test_parallel_efficiency_calculation(self):
        """Test parallel efficiency calculation"""
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        # Test with different thread counts
        config_8 = ThreadConfig(
            num_threads=8,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(8)},
            memory_affinity={i: 0 for i in range(8)}
        )
        
        config_80 = ThreadConfig(
            num_threads=80,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(80)},
            memory_affinity={i: i // 40 for i in range(80)}
        )
        
        eff_8 = self.threading.calculate_parallel_efficiency(gemm, config_8, self.cpu_system)
        eff_80 = self.threading.calculate_parallel_efficiency(gemm, config_80, self.cpu_system)
        
        # Efficiency should be between 0 and 1
        self.assertGreater(eff_8, 0)
        self.assertLessEqual(eff_8, 1.0)
        
        # Efficiency should decrease with more threads (Amdahl's law)
        self.assertGreater(eff_8, eff_80)
        
    def test_synchronization_overhead(self):
        """Test synchronization overhead calculation"""
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        # Sync overhead should increase with thread count
        overhead_8 = self.threading._calculate_sync_overhead(8, gemm)
        overhead_64 = self.threading._calculate_sync_overhead(64, gemm)
        
        self.assertLess(overhead_8, overhead_64)
        self.assertLess(overhead_8, 0.1)  # Should be small for GEMM
        
    def test_bandwidth_saturation(self):
        """Test memory bandwidth saturation effects"""
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        config = ThreadConfig(
            num_threads=80,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(80)},
            memory_affinity={i: i // 40 for i in range(80)}
        )
        
        saturation = self.threading._calculate_bandwidth_saturation(
            gemm, config, self.cpu_system
        )
        
        # Should be between 0 and 1
        self.assertGreaterEqual(saturation, 0)
        self.assertLess(saturation, 1.0)
        
    def test_numa_overhead_calculation(self):
        """Test NUMA overhead calculation"""
        # All threads on same node
        local_config = ThreadConfig(
            num_threads=40,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(40)},
            memory_affinity={i: 0 for i in range(40)}  # All access node 0
        )
        
        # Threads split across nodes
        split_config = ThreadConfig(
            num_threads=40,
            threads_per_core=1,
            thread_to_core_mapping={i: i for i in range(40)},
            memory_affinity={i: 1 for i in range(40)}  # All access node 1 (remote)
        )
        
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        local_overhead = self.threading._calculate_numa_overhead(local_config, gemm)
        remote_overhead = self.threading._calculate_numa_overhead(split_config, gemm)
        
        # Local access should have no overhead
        self.assertAlmostEqual(local_overhead, 0.0)
        
        # Remote access should have overhead
        self.assertGreater(remote_overhead, 0)
        
    def test_smt_efficiency(self):
        """Test SMT (Hyper-Threading) efficiency calculation"""
        # GEMM should have limited SMT benefit
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        gemm_smt_eff = self.threading._calculate_smt_efficiency(gemm)
        self.assertGreater(gemm_smt_eff, 1.0)
        self.assertLess(gemm_smt_eff, 1.3)
        
        # Attention should have better SMT benefit
        attention = Logit(['logit', 1, 8, 128, 128, 64, 8], density=(1.0, 1.0, 1.0))
        attention_smt_eff = self.threading._calculate_smt_efficiency(attention)
        self.assertGreater(attention_smt_eff, gemm_smt_eff)
        
    def test_arm_no_smt(self):
        """Test ARM system without SMT"""
        arm_system = create_cpu_system('aws_graviton3')
        arm_numa = NUMATopology(arm_system.cpu_config)
        arm_threading = ThreadingModel(arm_system.cpu_config, arm_numa)
        
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        config = arm_threading.get_optimal_thread_config(gemm, arm_system)
        
        # ARM Graviton3 has no SMT
        self.assertEqual(config.threads_per_core, 1)
        self.assertEqual(arm_system.cpu_config.threads_per_core, 1)


if __name__ == '__main__':
    unittest.main() 