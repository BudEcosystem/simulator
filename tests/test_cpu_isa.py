"""Tests for CPU ISA modeling"""
import unittest
import numpy as np
from GenZ.cpu.isa_model import ISASelector, ISAConfig, ISAType
from GenZ.cpu import create_cpu_system
from GenZ.operators import GEMM, Logit, FC


class TestISAConfig(unittest.TestCase):
    """Test ISA configuration"""
    
    def test_isa_config_creation(self):
        """Test creating ISA configuration"""
        config = ISAConfig(
            name=ISAType.AVX512,
            vector_width={'fp32': 16, 'fp16': 32, 'int8': 64},
            throughput={'fma': 2.0, 'add': 2.0, 'mul': 2.0},
            supported_types=['fp32', 'fp16', 'int8'],
            latency={'fma': 4, 'add': 3, 'mul': 3},
            alignment_required=64
        )
        
        self.assertEqual(config.name, ISAType.AVX512)
        self.assertEqual(config.vector_width['fp32'], 16)
        self.assertEqual(config.throughput['fma'], 2.0)
        self.assertIn('fp32', config.supported_types)
        
    def test_amx_config(self):
        """Test AMX-specific configuration"""
        config = ISAConfig(
            name=ISAType.AMX,
            vector_width={'int8': 1024, 'bf16': 512},
            throughput={'tilemmul': 1.0},
            supported_types=['int8', 'bf16'],
            latency={'tilemmul': 12},
            special_constraints={
                'tile_m': 16,
                'tile_n': 64,
                'tile_k': 64,
                'num_tiles': 8
            }
        )
        
        self.assertEqual(config.special_constraints['tile_m'], 16)
        self.assertEqual(config.vector_width['int8'], 1024)


class TestISASelector(unittest.TestCase):
    """Test ISA selection logic"""
    
    def setUp(self):
        """Set up test CPU system"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        self.isa_selector = ISASelector(self.cpu_system.cpu_config)
        
    def test_isa_initialization(self):
        """Test ISA selector initialization"""
        self.assertIsNotNone(self.isa_selector.isa_configs)
        self.assertIn(ISAType.AVX512, self.isa_selector.isa_configs)
        self.assertIn(ISAType.AMX, self.isa_selector.isa_configs)
        
    def test_gemm_isa_selection_amx(self):
        """Test ISA selection for GEMM with AMX"""
        # Create GEMM operator suitable for AMX
        gemm = GEMM(['gemm', 1, 64, 64, 64], density=(1.0, 1.0, 1.0))
        
        # Select ISA for int8
        isa, efficiency = self.isa_selector.select_isa(gemm, 'int8')
        self.assertEqual(isa, ISAType.AMX)
        self.assertGreater(efficiency, 0.8)
        
    def test_gemm_isa_selection_avx512(self):
        """Test ISA selection for GEMM with AVX-512"""
        # Create GEMM operator
        gemm = GEMM(['gemm', 1, 100, 100, 100], density=(1.0, 1.0, 1.0))
        
        # Select ISA for fp32
        isa, efficiency = self.isa_selector.select_isa(gemm, 'fp32')
        self.assertEqual(isa, ISAType.AVX512)
        self.assertGreater(efficiency, 0.7)
        
    def test_attention_isa_selection(self):
        """Test ISA selection for attention operators"""
        # Create attention operator
        logit = Logit(['logit', 1, 8, 128, 128, 64, 8], density=(1.0, 1.0, 1.0))
        
        # Select ISA
        isa, efficiency = self.isa_selector.select_isa(logit, 'fp32')
        self.assertIn(isa, [ISAType.AVX512, ISAType.AVX2])
        
    def test_small_operator_isa_selection(self):
        """Test ISA selection for small operators"""
        # Small GEMM that might not benefit from wide vectors
        small_gemm = GEMM(['gemm', 1, 4, 4, 4], density=(1.0, 1.0, 1.0))
        
        isa, efficiency = self.isa_selector.select_isa(small_gemm, 'fp32')
        # Small operators might use narrower ISA or have lower efficiency
        self.assertLess(efficiency, 0.9)
        
    def test_unsupported_data_type(self):
        """Test ISA selection with unsupported data type"""
        gemm = GEMM(['gemm', 1, 64, 64, 64], density=(1.0, 1.0, 1.0))
        
        # AMD EPYC doesn't support AMX
        amd_system = create_cpu_system('amd_epyc_7763')
        amd_selector = ISASelector(amd_system.cpu_config)
        
        isa, efficiency = amd_selector.select_isa(gemm, 'int8')
        self.assertNotEqual(isa, ISAType.AMX)  # Should not select AMX
        
    def test_compute_time_calculation(self):
        """Test compute time calculation with ISA"""
        gemm = GEMM(['gemm', 1, 128, 128, 128], density=(1.0, 1.0, 1.0))
        
        # Calculate compute time
        compute_time = self.isa_selector.get_compute_time(
            gemm, ISAType.AVX512, 40, self.cpu_system
        )
        
        self.assertGreater(compute_time, 0)
        self.assertIsInstance(compute_time, float)
        
    def test_amx_tile_constraints(self):
        """Test AMX tile size constraints"""
        # Perfect tile size (multiples of tile dimensions)
        good_gemm = GEMM(['gemm', 1, 64, 64, 64], density=(1.0, 1.0, 1.0))
        self.assertTrue(self.isa_selector._check_amx_constraints(64, 64, 64))
        
        # Larger dimensions (still valid, will use multiple tiles)
        large_gemm = GEMM(['gemm', 1, 128, 128, 128], density=(1.0, 1.0, 1.0))
        self.assertTrue(self.isa_selector._check_amx_constraints(128, 128, 128))
        
        # Too small for AMX (smaller than minimum tile size)
        tiny_gemm = GEMM(['gemm', 1, 8, 8, 8], density=(1.0, 1.0, 1.0))
        self.assertFalse(self.isa_selector._check_amx_constraints(8, 8, 8))
        
        # Mixed sizes - some dimensions too small
        mixed_gemm = GEMM(['gemm', 1, 64, 64, 8], density=(1.0, 1.0, 1.0))
        self.assertFalse(self.isa_selector._check_amx_constraints(64, 64, 8))


class TestFrequencyGovernor(unittest.TestCase):
    """Test frequency scaling model"""
    
    def setUp(self):
        """Set up test CPU system"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        # Initialize frequency governor
        from GenZ.cpu.frequency_model import FrequencyGovernor
        self.freq_gov = FrequencyGovernor(self.cpu_system.cpu_config)
        
    def test_base_frequency(self):
        """Test base frequency retrieval"""
        freq = self.freq_gov.get_frequency(ISAType.SCALAR, 80, 'compute')
        self.assertGreaterEqual(freq, self.cpu_system.cpu_config.base_frequency * 0.8)
        
    def test_turbo_frequency(self):
        """Test turbo frequency for low core counts"""
        freq = self.freq_gov.get_frequency(ISAType.SCALAR, 1, 'compute')
        self.assertGreater(freq, self.cpu_system.cpu_config.base_frequency)
        self.assertLessEqual(freq, self.cpu_system.cpu_config.turbo_frequency_curve[1])
        
    def test_avx_frequency_reduction(self):
        """Test frequency reduction with AVX instructions"""
        # Use fewer cores to get higher turbo frequencies where offset matters
        cores = 8
        scalar_freq = self.freq_gov.get_frequency(ISAType.SCALAR, cores, 'compute')
        avx512_freq = self.freq_gov.get_frequency(ISAType.AVX512, cores, 'compute')
        amx_freq = self.freq_gov.get_frequency(ISAType.AMX, cores, 'compute')
        
        # AVX-512 should have lower frequency than scalar
        self.assertLess(avx512_freq, scalar_freq)
        # AMX should have even lower frequency
        self.assertLess(amx_freq, avx512_freq)
        
        # Verify the frequency reduction amounts
        avx512_reduction = scalar_freq - avx512_freq
        amx_reduction = scalar_freq - amx_freq
        
        # Should match the configured offsets
        self.assertAlmostEqual(avx512_reduction, 200e6, delta=1e6)
        self.assertAlmostEqual(amx_reduction, 300e6, delta=1e6)
        
    def test_thermal_throttling(self):
        """Test thermal throttling behavior"""
        # Normal temperature
        self.freq_gov.thermal_state.temperature = 60
        normal_freq = self.freq_gov.get_frequency(ISAType.AVX512, 40, 'compute')
        
        # High temperature
        self.freq_gov.thermal_state.temperature = 80
        throttled_freq = self.freq_gov.get_frequency(ISAType.AVX512, 40, 'compute')
        
        self.assertLess(throttled_freq, normal_freq)
        
        # Reset temperature
        self.freq_gov.thermal_state.temperature = 50
        
    def test_memory_bound_frequency(self):
        """Test frequency for memory-bound workloads"""
        compute_freq = self.freq_gov.get_frequency(ISAType.AVX512, 40, 'compute')
        memory_freq = self.freq_gov.get_frequency(ISAType.AVX512, 40, 'memory')
        
        # Memory-bound workloads can maintain slightly higher frequency
        self.assertGreaterEqual(memory_freq, compute_freq)
        
    def test_frequency_interpolation(self):
        """Test frequency interpolation for core counts"""
        # Test interpolation between defined points
        freq_20_cores = self.freq_gov.get_frequency(ISAType.SCALAR, 20, 'compute')
        freq_16_cores = self.freq_gov.get_frequency(ISAType.SCALAR, 16, 'compute')
        freq_32_cores = self.freq_gov.get_frequency(ISAType.SCALAR, 32, 'compute')
        
        # Frequency should decrease with more cores
        self.assertGreater(freq_16_cores, freq_20_cores)
        self.assertGreater(freq_20_cores, freq_32_cores)


class TestISAIntegration(unittest.TestCase):
    """Test ISA integration with operators"""
    
    def test_arm_isa_support(self):
        """Test ARM ISA support"""
        arm_system = create_cpu_system('aws_graviton3')
        isa_selector = ISASelector(arm_system.cpu_config)
        
        # Check SVE2 support
        self.assertIn('sve2', arm_system.cpu_config.isa_support)
        self.assertIn('neon', arm_system.cpu_config.isa_support)
        
        # ARM shouldn't have x86 ISAs
        self.assertNotIn('avx512', arm_system.cpu_config.isa_support)
        self.assertNotIn('amx', arm_system.cpu_config.isa_support)
        
    def test_amd_no_avx512(self):
        """Test AMD EPYC without AVX-512"""
        amd_system = create_cpu_system('amd_epyc_7763')
        isa_selector = ISASelector(amd_system.cpu_config)
        
        # AMD Zen3 doesn't have AVX-512
        self.assertNotIn('avx512', amd_system.cpu_config.isa_support)
        self.assertIn('avx2', amd_system.cpu_config.isa_support)


if __name__ == '__main__':
    unittest.main() 