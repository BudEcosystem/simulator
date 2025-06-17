"""Tests for CPU system model"""
import unittest
import numpy as np
from GenZ.system import System
from GenZ.cpu import CPUSystem, CPUConfig, create_cpu_system
from GenZ.cpu.cache_model import CacheConfig


class TestCPUSystem(unittest.TestCase):
    """Test CPU system initialization and configuration"""
    
    def test_cpu_config_creation(self):
        """Test creating a CPU configuration"""
        cache_config = CacheConfig(
            size=32*1024,  # 32KB
            latency=4,
            bandwidth=3200,  # GB/s
            associativity=8,
            line_size=64,
            is_shared=False,
            is_inclusive=True
        )
        
        cpu_config = CPUConfig(
            cores_per_socket=40,
            sockets=2,
            threads_per_core=2,
            l1i_config=cache_config,
            l1d_config=cache_config,
            l2_config=CacheConfig(size=1280*1024, latency=14, bandwidth=1600),
            l3_config=CacheConfig(size=60*1024*1024, latency=42, bandwidth=800, is_shared=True),
            numa_nodes=2,
            cores_per_numa=40,
            numa_distance_matrix=np.array([[10, 21], [21, 10]]),
            isa_support=['amx', 'avx512', 'avx2', 'sse'],
            base_frequency=2.3e9,
            turbo_frequency_curve={1: 3.4e9, 2: 3.4e9, 4: 3.2e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=25.6,
            memory_channels_per_socket=8,
            vendor='intel',
            microarchitecture='icelake'
        )
        
        self.assertEqual(cpu_config.cores_per_socket, 40)
        self.assertEqual(cpu_config.sockets, 2)
        self.assertEqual(cpu_config.l1d_config.size, 32*1024)
        
    def test_cpu_system_creation(self):
        """Test creating a CPU system from base system"""
        base_system = System(
            flops=123,
            frequency=2.3e9,
            bits='fp32',
            compute_efficiency=0.85,
            memory_efficiency=0.75
        )
        
        cpu_config = CPUConfig(
            cores_per_socket=40,
            sockets=2,
            threads_per_core=2,
            l1i_config=CacheConfig(size=32*1024, latency=4, bandwidth=3200),
            l1d_config=CacheConfig(size=48*1024, latency=5, bandwidth=3200),
            l2_config=CacheConfig(size=1280*1024, latency=14, bandwidth=1600),
            l3_config=CacheConfig(size=60*1024*1024, latency=42, bandwidth=800, is_shared=True),
            numa_nodes=2,
            cores_per_numa=40,
            numa_distance_matrix=np.array([[10, 21], [21, 10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.3e9,
            turbo_frequency_curve={1: 3.4e9, 40: 2.4e9},
            avx_frequency_offset={'avx512': -200e6},
            dram_bandwidth_per_channel=25.6,
            memory_channels_per_socket=8,
            vendor='intel',
            microarchitecture='icelake'
        )
        
        cpu_system = CPUSystem(base_system, cpu_config)
        
        # Test that CPU system has base system attributes
        self.assertIsNotNone(cpu_system.base_system)
        self.assertEqual(cpu_system.cpu_config.cores_per_socket, 40)
        
        # Test peak performance calculations
        self.assertGreater(cpu_system.peak_memory_bandwidth, 0)
        
    def test_create_cpu_system_factory(self):
        """Test factory function for creating CPU systems"""
        cpu_system = create_cpu_system('intel_xeon_8380')
        
        self.assertIsInstance(cpu_system, CPUSystem)
        self.assertEqual(cpu_system.cpu_config.vendor, 'intel')
        self.assertEqual(cpu_system.cpu_config.cores_per_socket, 40)
        
    def test_cpu_system_bandwidth_calculation(self):
        """Test effective bandwidth calculation with cache hierarchy"""
        cpu_system = create_cpu_system('intel_xeon_8380')
        
        # Test L1 bandwidth
        l1_bw = cpu_system.get_effective_bandwidth('L1', 0)
        self.assertEqual(l1_bw, cpu_system.cpu_config.l1d_config.bandwidth)
        
        # Test L3 bandwidth (should be reduced with more active cores)
        cpu_system.get_active_cores = lambda: 40
        l3_bw = cpu_system.get_effective_bandwidth('L3', 0)
        self.assertLess(l3_bw, cpu_system.cpu_config.l3_config.bandwidth)
        
    def test_cpu_system_flops_calculation(self):
        """Test effective FLOPS calculation for different ISAs"""
        cpu_system = create_cpu_system('intel_xeon_8380')
        
        # Mock ISA selector and frequency governor
        cpu_system.isa_selector = type('obj', (object,), {
            'get_peak_performance': lambda self, isa, dtype: 16 * 2 if isa == 'avx512' else 8 * 2
        })()
        cpu_system.frequency_governor = type('obj', (object,), {
            'get_frequency': lambda self, isa, cores: 2.3e9 if isa == 'avx512' else 2.5e9
        })()
        
        # Test AVX-512 FLOPS
        flops = cpu_system.get_effective_flops('GEMM', 'fp32', 'avx512', 40)
        self.assertGreater(flops, 0)
        
        # Save original efficiency function
        original_efficiency = cpu_system._get_vectorization_efficiency
        
        # Test that vectorization efficiency affects FLOPS
        cpu_system._get_vectorization_efficiency = lambda op, isa: 0.5
        flops_reduced = cpu_system.get_effective_flops('GEMM', 'fp32', 'avx512', 40)
        
        # Calculate expected reduced flops
        # Original efficiency for GEMM with avx512 is 0.85
        expected_reduced = flops * (0.5 / 0.85)
        self.assertAlmostEqual(flops_reduced, expected_reduced, places=5)


class TestCPUPresets(unittest.TestCase):
    """Test CPU preset configurations"""
    
    def test_intel_xeon_preset(self):
        """Test Intel Xeon preset configuration"""
        cpu_system = create_cpu_system('intel_xeon_8380')
        
        self.assertEqual(cpu_system.cpu_config.vendor, 'intel')
        self.assertEqual(cpu_system.cpu_config.microarchitecture, 'icelake')
        self.assertEqual(cpu_system.cpu_config.cores_per_socket, 40)
        self.assertEqual(cpu_system.cpu_config.sockets, 2)
        self.assertIn('amx', cpu_system.cpu_config.isa_support)
        self.assertIn('avx512', cpu_system.cpu_config.isa_support)
        
    def test_amd_epyc_preset(self):
        """Test AMD EPYC preset configuration"""
        cpu_system = create_cpu_system('amd_epyc_7763')
        
        self.assertEqual(cpu_system.cpu_config.vendor, 'amd')
        self.assertEqual(cpu_system.cpu_config.microarchitecture, 'zen3')
        self.assertEqual(cpu_system.cpu_config.cores_per_socket, 64)
        self.assertEqual(cpu_system.cpu_config.numa_nodes, 8)
        self.assertNotIn('avx512', cpu_system.cpu_config.isa_support)
        
    def test_arm_graviton_preset(self):
        """Test ARM Graviton preset configuration"""
        cpu_system = create_cpu_system('aws_graviton3')
        
        self.assertEqual(cpu_system.cpu_config.vendor, 'arm')
        self.assertEqual(cpu_system.cpu_config.microarchitecture, 'neoverse-v1')
        self.assertEqual(cpu_system.cpu_config.threads_per_core, 1)  # No SMT
        self.assertIn('sve2', cpu_system.cpu_config.isa_support)
        self.assertIn('neon', cpu_system.cpu_config.isa_support)
        
    def test_unknown_cpu_model(self):
        """Test error handling for unknown CPU model"""
        with self.assertRaises(ValueError):
            create_cpu_system('unknown_cpu')


if __name__ == '__main__':
    unittest.main() 