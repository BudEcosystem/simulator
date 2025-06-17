"""Integration tests for CPU support with GenZ operators"""
import unittest
import pandas as pd
from GenZ.operators import GEMM, Logit, FC
from GenZ.unit import Unit
from GenZ.cpu import create_cpu_system
from GenZ.cpu.operator_enhancement import enhance_operators_for_cpu
from GenZ.analyse_model import analysis_model


class TestCPUIntegration(unittest.TestCase):
    """Test CPU integration with GenZ operators"""
    
    @classmethod
    def setUpClass(cls):
        """Set up CPU enhancement once for all tests"""
        enhance_operators_for_cpu()
        
    def setUp(self):
        """Set up test environment"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        self.unit = Unit()
        
    def test_gemm_cpu_roofline(self):
        """Test GEMM operator with CPU system"""
        gemm = GEMM(['gemm', 1, 512, 512, 512], density=(1.0, 1.0, 1.0))
        
        # Get roofline analysis
        result = gemm.get_roofline(self.cpu_system, self.unit)
        
        # Check basic results
        self.assertIn('Layer Name', result)
        self.assertIn('Op Type', result)
        self.assertEqual(result['Op Type'], 'GEMM')
        
        # Check CPU-specific metrics
        self.assertIn('ISA_used', result)
        self.assertIn('Thread_count', result)
        self.assertIn('Parallel_efficiency', result)
        self.assertIn('L1_hit_rate', result)
        
        # Verify ISA selection
        self.assertIn(result['ISA_used'], ['avx512', 'avx2', 'amx'])
        
        # Verify thread count is reasonable
        self.assertGreater(result['Thread_count'], 0)
        self.assertLessEqual(result['Thread_count'], 160)  # 2 sockets * 40 cores * 2 threads
        
    def test_attention_cpu_roofline(self):
        """Test attention operator with CPU system"""
        logit = Logit(['logit', 1, 8, 128, 128, 64, 8], density=(1.0, 1.0, 1.0))
        
        result = logit.get_roofline(self.cpu_system, self.unit)
        
        # Check results
        self.assertEqual(result['Op Type'], 'Logit')
        self.assertIn('ISA_used', result)
        self.assertIn('L1_hit_rate', result)
        
    def test_fc_cpu_roofline(self):
        """Test FC operator with CPU system"""
        fc = FC(['fc', 1, 1024, 768], density=(1.0, 1.0, 1.0))
        
        result = fc.get_roofline(self.cpu_system, self.unit)
        
        # Check results
        self.assertEqual(result['Op Type'], 'FC')
        self.assertIn('Bound', result)
        self.assertIn(result['Bound'], ['Compute', 'Memory', 'Balanced'])
        
    def skip_test_cpu_vs_gpu_comparison(self):
        """Test that CPU and GPU systems produce different results"""
        from GenZ.system import System
        
        # Create GPU system
        gpu_system = System(
            flops=123,  # TFLOPS
            onchip_mem_bw=18000,  # GB/s
            offchip_mem_bw=900,   # GB/s
            frequency=940,
            bits='f32'  # GenZ uses 'f32' not 'fp32'
        )
        
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        # Get results from both systems
        cpu_result = gemm.get_roofline(self.cpu_system, self.unit)
        gpu_result = gemm.get_roofline(gpu_system, self.unit)
        
        # CPU should have additional metrics
        self.assertIn('ISA_used', cpu_result)
        self.assertNotIn('ISA_used', gpu_result)
        
        # Performance characteristics should differ
        self.assertNotEqual(
            cpu_result[f'Latency ({self.unit.unit_time})'],
            gpu_result[f'Latency ({self.unit.unit_time})']
        )
        
    def test_model_analysis_with_cpu(self):
        """Test full model analysis with CPU system"""
        from GenZ.Models import OpType, ResidencyInfo
        
        # Simple model with a few layers - need to add OpType and ResidencyInfo
        model_dims = [
            ['gemm', 1, 512, 512, 512, ResidencyInfo.All_offchip, OpType.GEMM],
            ['logit', 1, 8, 128, 128, 64, 8, ResidencyInfo.All_offchip, OpType.Logit],
            ['fc', 1, 768, 3072, ResidencyInfo.All_offchip, OpType.FC]
        ]
        
        # Run analysis
        results = analysis_model(
            model_dims,
            system=self.cpu_system,
            unit=self.unit
        )
        
        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 3)
        
        # All layers should have CPU metrics
        self.assertIn('ISA_used', results.columns)
        self.assertIn('Thread_count', results.columns)
        self.assertIn('Parallel_efficiency', results.columns)
        self.assertIn('L1_hit_rate', results.columns)
        
    def test_amd_cpu_system(self):
        """Test with AMD CPU system"""
        amd_system = create_cpu_system('amd_epyc_7763')
        
        gemm = GEMM(['gemm', 1, 512, 512, 512], density=(1.0, 1.0, 1.0))
        result = gemm.get_roofline(amd_system, self.unit)
        
        # AMD doesn't have AVX-512 or AMX
        self.assertNotIn(result['ISA_used'], ['avx512', 'amx'])
        self.assertEqual(result['ISA_used'], 'avx2')
        
    def test_arm_cpu_system(self):
        """Test with ARM CPU system"""
        arm_system = create_cpu_system('aws_graviton3')
        
        gemm = GEMM(['gemm', 1, 512, 512, 512], density=(1.0, 1.0, 1.0))
        result = gemm.get_roofline(arm_system, self.unit)
        
        # ARM should use SVE2 or NEON
        self.assertIn(result['ISA_used'], ['sve2', 'neon'])
        
        # ARM Graviton3 has no SMT
        self.assertEqual(result['Threads_per_core'], 1)
        
    def test_cache_optimization(self):
        """Test cache optimization functionality"""
        gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
        
        # Optimize for CPU cache
        optimized = gemm.optimize_for_cpu_cache(self.cpu_system)
        
        # Should return an operator (even if just the same one for now)
        self.assertIsNotNone(optimized)
        self.assertIsInstance(optimized, GEMM)
        
    def test_memory_time_calculation(self):
        """Test CPU-specific memory time calculation"""
        gemm = GEMM(['gemm', 1, 256, 256, 256], density=(1.0, 1.0, 1.0))
        
        # Get memory time
        mem_time = gemm.get_memory_time(self.cpu_system)
        
        # Should be positive
        self.assertGreater(mem_time, 0)
        
        # Should be different from compute time
        compute_time = gemm.get_compute_time(self.cpu_system)
        self.assertNotEqual(mem_time, compute_time)
        
    def test_parallel_efficiency_impact(self):
        """Test impact of parallel efficiency on performance"""
        gemm = GEMM(['gemm', 1, 2048, 2048, 2048], density=(1.0, 1.0, 1.0))
        
        result = gemm.get_roofline(self.cpu_system, self.unit)
        
        # Parallel efficiency should be less than 1
        self.assertLess(result['Parallel_efficiency'], 1.0)
        self.assertGreater(result['Parallel_efficiency'], 0.5)


if __name__ == '__main__':
    unittest.main() 