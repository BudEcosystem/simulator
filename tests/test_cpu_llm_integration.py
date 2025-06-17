"""
Tests for CPU LLM Integration
"""
import pytest
import pandas as pd
from GenZ.cpu import (
    create_cpu_system,
    cpu_aware_prefill_moddeling,
    cpu_aware_decode_moddeling,
    enable_cpu_aware_inference,
    disable_cpu_aware_inference,
    is_cpu_system
)
from GenZ.LLM_inference import prefill_moddeling, decode_moddeling


class TestCPUSystemDetection:
    """Test CPU system detection logic"""
    
    def test_detect_cpu_system_object(self):
        """Test detection of CPUSystem objects"""
        cpu_system = create_cpu_system('intel_xeon_8380')
        assert is_cpu_system(cpu_system) == True
        
    def test_detect_cpu_preset_string(self):
        """Test detection of CPU preset strings"""
        assert is_cpu_system('intel_xeon_8380') == True
        assert is_cpu_system('amd_epyc_7763') == True
        assert is_cpu_system('aws_graviton3') == True
        assert is_cpu_system('Intel_Xeon_8380') == True  # Case insensitive
        
    def test_detect_cpu_dict(self):
        """Test detection of CPU configuration dicts"""
        cpu_dict = {
            'cache_configs': {'L1': 32, 'L2': 256},
            'numa_config': {'nodes': 2}
        }
        assert is_cpu_system(cpu_dict) == True
        
    def test_not_cpu_system(self):
        """Test non-CPU systems are not detected"""
        assert is_cpu_system('A100_40GB_GPU') == False
        assert is_cpu_system('H100_80GB_GPU') == False
        assert is_cpu_system({'Flops': 320, 'Memory_BW': 40}) == False


class TestCPUAwareFunctions:
    """Test CPU-aware wrapper functions"""
    
    def test_cpu_aware_prefill_with_string(self):
        """Test CPU-aware prefill with string preset"""
        result = cpu_aware_prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        assert 'Latency' in result
        assert 'Throughput' in result
        assert result['Latency'] > 0
        assert result['Throughput'] > 0
        
        # Check for CPU-specific columns in model_df
        assert 'ISA_used' in result['model_df'].columns
        assert 'Thread_count' in result['model_df'].columns
        assert 'Parallel_efficiency' in result['model_df'].columns
        
    def test_cpu_aware_prefill_with_object(self):
        """Test CPU-aware prefill with CPU system object"""
        cpu_system = create_cpu_system('amd_epyc_7763')
        
        result = cpu_aware_prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name=cpu_system,
            debug=False
        )
        
        assert 'Latency' in result
        assert 'Throughput' in result
        assert result['Latency'] > 0
        
    def test_cpu_aware_decode(self):
        """Test CPU-aware decode function"""
        result = cpu_aware_decode_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            output_tokens=32,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        assert 'Latency' in result
        assert 'Throughput' in result
        assert result['Latency'] > 0
        assert result['Throughput'] > 0
        
    def test_gpu_system_passthrough(self):
        """Test that GPU systems work normally"""
        result = cpu_aware_prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name='A100_40GB_GPU',
            debug=False
        )
        
        assert 'Latency' in result
        assert 'Throughput' in result
        # Should not have CPU-specific columns
        assert 'ISA_used' not in result['model_df'].columns


class TestAutomaticDetection:
    """Test automatic CPU detection feature"""
    
    def test_enable_disable_cpu_aware_inference(self):
        """Test enabling and disabling automatic CPU detection"""
        # Enable CPU-aware inference
        enable_cpu_aware_inference()
        
        # Test with CPU system - should use CPU enhancements
        result = prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        assert 'ISA_used' in result['model_df'].columns
        
        # Disable CPU-aware inference
        disable_cpu_aware_inference()
        
        # Test again - should not use CPU enhancements
        result2 = prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name='A100_40GB_GPU',
            debug=False
        )
        
        assert 'ISA_used' not in result2['model_df'].columns


class TestPrefillDecodeMetrics:
    """Test prefill and decode metrics calculation"""
    
    def test_prefill_metrics(self):
        """Test prefill phase metrics"""
        result = cpu_aware_prefill_moddeling(
            model='llama',
            batch_size=1,
            input_tokens=512,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        # Check latency is reasonable for prefill
        assert result['Latency'] > 0
        assert result['Latency'] < 10000  # Less than 10 seconds
        
        # Check throughput calculation
        expected_throughput = 1000 * 1 / result['Latency']  # tokens/sec
        assert abs(result['Throughput'] - expected_throughput) < 0.1
        
    def test_decode_metrics(self):
        """Test decode phase metrics"""
        result = cpu_aware_decode_moddeling(
            model='llama',
            batch_size=1,
            input_tokens=512,
            output_tokens=1,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        # Decode should be faster than prefill for single token
        assert result['Latency'] > 0
        assert result['Latency'] < 1000  # Less than 1 second per token
        
    def test_decode_scaling(self):
        """Test decode performance scaling with KV cache size"""
        latencies = []
        
        for kv_size in [128, 256, 512]:
            result = cpu_aware_decode_moddeling(
                model='gpt2',
                batch_size=1,
                input_tokens=kv_size,
                output_tokens=1,
                system_name='intel_xeon_8380',
                debug=False
            )
            latencies.append(result['Latency'])
        
        # Latency should increase with KV cache size
        assert latencies[1] >= latencies[0]
        assert latencies[2] >= latencies[1]


class TestCPUSpecificMetrics:
    """Test CPU-specific metrics in results"""
    
    def test_cpu_metrics_in_results(self):
        """Test that CPU-specific metrics are included"""
        result = cpu_aware_prefill_moddeling(
            model='gpt2',
            batch_size=1,
            input_tokens=128,
            system_name='intel_xeon_8380',
            debug=False
        )
        
        model_df = result['model_df']
        
        # Check CPU-specific columns exist
        cpu_columns = ['ISA_used', 'Thread_count', 'Parallel_efficiency', 
                      'L1_hit_rate', 'L2_hit_rate', 'L3_hit_rate', 
                      'DRAM_access_rate', 'Frequency_GHz']
        
        for col in cpu_columns:
            assert col in model_df.columns
            
        # Check ISA selection
        assert model_df['ISA_used'].iloc[0] in ['avx2', 'avx512', 'amx', 'neon', 'sve', 'sve2']
        
        # Check thread count is reasonable
        assert model_df['Thread_count'].iloc[0] > 0
        assert model_df['Thread_count'].iloc[0] <= 200  # Reasonable upper bound
        
        # Check cache hit rates
        assert 0 <= model_df['L1_hit_rate'].iloc[0] <= 1
        assert 0 <= model_df['L2_hit_rate'].iloc[0] <= 1
        assert 0 <= model_df['L3_hit_rate'].iloc[0] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 