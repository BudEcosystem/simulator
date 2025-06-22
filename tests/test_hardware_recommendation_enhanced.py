"""
Test suite for enhanced hardware recommendation system.
Tests utilization-based sorting, CPU/GPU separation, and multi-node configurations.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hardware_recommendation import HardwareRecommendation
from src.hardware import BudHardware


class TestEnhancedHardwareRecommendation:
    """Test enhanced hardware recommendation features."""
    
    @pytest.fixture
    def recommender(self):
        """Create HardwareRecommendation instance."""
        return HardwareRecommendation()
    
    @pytest.fixture
    def mock_hardware_data(self):
        """Mock hardware data for testing."""
        return [
            {
                'name': 'NVIDIA_A100_80GB',
                'type': 'gpu',
                'Memory_size': 80,
                'manufacturer': 'NVIDIA',
                'flops': 19.5e12,
                'memory_bw': 2000
            },
            {
                'name': 'NVIDIA_H100_80GB',
                'type': 'gpu',
                'Memory_size': 80,
                'manufacturer': 'NVIDIA',
                'flops': 60e12,
                'memory_bw': 3000
            },
            {
                'name': 'AMD_MI250X',
                'type': 'gpu',
                'Memory_size': 128,
                'manufacturer': 'AMD',
                'flops': 47.9e12,
                'memory_bw': 3200
            },
            {
                'name': 'Intel_Xeon_8480+',
                'type': 'cpu',
                'Memory_size': 512,
                'manufacturer': 'Intel',
                'flops': 3.5e12,
                'memory_bw': 350
            },
            {
                'name': 'AMD_EPYC_9654',
                'type': 'cpu',
                'Memory_size': 768,
                'manufacturer': 'AMD',
                'flops': 5.0e12,
                'memory_bw': 460
            }
        ]
    
    def test_utilization_based_sorting(self, recommender, mock_hardware_data):
        """Test that hardware is sorted by nodes required first, then utilization."""
        with patch.object(recommender.hardware, 'get_all_hardwares', return_value=mock_hardware_data):
            result = recommender.recommend_hardware(
                total_memory_gb=150.0,
                model_params_b=70.0
            )
            
            # Check GPU recommendations are sorted by nodes required first, then utilization
            gpu_recs = result['gpu_recommendations']
            assert len(gpu_recs) > 0
            
            # Verify sorting: nodes_required ascending, then utilization descending
            for i in range(len(gpu_recs) - 1):
                current = gpu_recs[i]
                next_rec = gpu_recs[i + 1]
                
                # If nodes are different, current should have fewer or equal nodes
                if current['nodes_required'] != next_rec['nodes_required']:
                    assert current['nodes_required'] <= next_rec['nodes_required']
                # If nodes are the same, current should have higher or equal utilization
                else:
                    assert current['utilization'] >= next_rec['utilization']
            
            # Check CPU recommendations follow same sorting
            cpu_recs = result['cpu_recommendations']
            if len(cpu_recs) > 1:
                for i in range(len(cpu_recs) - 1):
                    current = cpu_recs[i]
                    next_rec = cpu_recs[i + 1]
                    
                    if current['nodes_required'] != next_rec['nodes_required']:
                        assert current['nodes_required'] <= next_rec['nodes_required']
                    else:
                        assert current['utilization'] >= next_rec['utilization']
    
    def test_cpu_gpu_separation(self, recommender, mock_hardware_data):
        """Test that CPUs and GPUs are properly separated."""
        with patch.object(recommender.hardware, 'get_all_hardwares', return_value=mock_hardware_data):
            result = recommender.recommend_hardware(
                total_memory_gb=50.0,
                model_params_b=13.0
            )
            
            # Verify structure
            assert 'cpu_recommendations' in result
            assert 'gpu_recommendations' in result
            assert 'model_info' in result
            
            # Check CPU recommendations only contain CPUs
            for cpu_rec in result['cpu_recommendations']:
                assert cpu_rec['type'] == 'cpu'
            
            # Check GPU recommendations only contain GPUs/accelerators
            for gpu_rec in result['gpu_recommendations']:
                assert gpu_rec['type'] in ['gpu', 'accelerator', 'asic']
    
    def test_small_model_detection(self, recommender):
        """Test detection of small models (<13-14B params)."""
        # Test small model (7B)
        result = recommender.recommend_hardware(
            total_memory_gb=15.0,
            model_params_b=7.0
        )
        assert result['model_info']['is_small_model'] is True
        assert result['model_info']['cpu_compatible'] is True
        
        # Test medium model (13B)
        result = recommender.recommend_hardware(
            total_memory_gb=26.0,
            model_params_b=13.0
        )
        assert result['model_info']['is_small_model'] is True
        
        # Test large model (70B)
        result = recommender.recommend_hardware(
            total_memory_gb=140.0,
            model_params_b=70.0
        )
        assert result['model_info']['is_small_model'] is False
    
    def test_multi_node_cpu_batch_recommendations(self, recommender):
        """Test multi-node CPU configurations with batch recommendations."""
        # Small model with low memory requirements
        result = recommender.recommend_hardware(
            total_memory_gb=15.0,
            model_params_b=7.0
        )
        
        # Should have CPU recommendations
        assert len(result['cpu_recommendations']) > 0
        
        # Check for batch recommendations in CPU configs
        cpu_rec = result['cpu_recommendations'][0]
        assert 'batch_recommendations' in cpu_rec
        
        if cpu_rec['batch_recommendations']:
            # Verify batch recommendation structure
            batch_rec = cpu_rec['batch_recommendations'][0]
            assert 'nodes' in batch_rec
            assert 'recommended_batch_size' in batch_rec
            assert 'utilization_at_batch' in batch_rec
            assert 'total_memory' in batch_rec
            
            # Verify multi-node configs
            assert any(br['nodes'] > 1 for br in cpu_rec['batch_recommendations'])
    
    def test_kv_cache_consideration(self, recommender):
        """Test that KV cache size affects batch recommendations."""
        # Model with different sequence lengths affecting KV cache
        # Short sequence (small KV cache)
        result_short = recommender.recommend_hardware(
            total_memory_gb=20.0,  # Includes small KV cache
            model_params_b=7.0
        )
        
        # Long sequence (large KV cache)
        result_long = recommender.recommend_hardware(
            total_memory_gb=50.0,  # Includes large KV cache
            model_params_b=7.0
        )
        
        # Both should have CPU recommendations for small model
        assert len(result_short['cpu_recommendations']) > 0
        assert len(result_long['cpu_recommendations']) > 0
        
        # Model info should reflect the memory requirements
        assert result_short['model_info']['total_memory_gb'] == 20.0
        assert result_long['model_info']['total_memory_gb'] == 50.0
    
    def test_response_structure(self, recommender):
        """Test the complete response structure."""
        result = recommender.recommend_hardware(
            total_memory_gb=30.0,
            model_params_b=13.0
        )
        
        # Check top-level structure
        assert isinstance(result, dict)
        assert 'cpu_recommendations' in result
        assert 'gpu_recommendations' in result
        assert 'model_info' in result
        
        # Check model_info structure
        model_info = result['model_info']
        assert 'is_small_model' in model_info
        assert 'cpu_compatible' in model_info
        assert 'total_memory_gb' in model_info
        assert 'model_params_b' in model_info
        
        # Check recommendation structure
        if len(result['gpu_recommendations']) > 0:
            gpu_rec = result['gpu_recommendations'][0]
            assert 'hardware_name' in gpu_rec
            assert 'nodes_required' in gpu_rec
            assert 'memory_per_chip' in gpu_rec
            assert 'utilization' in gpu_rec
            assert 'type' in gpu_rec
            assert 'manufacturer' in gpu_rec
            assert 'total_memory_available' in gpu_rec
    
    def test_edge_cases(self, recommender):
        """Test edge cases and error handling."""
        # Very small model
        result = recommender.recommend_hardware(
            total_memory_gb=1.0,
            model_params_b=0.5
        )
        assert result['model_info']['is_small_model'] is True
        assert len(result['cpu_recommendations']) > 0
        
        # Very large model
        result = recommender.recommend_hardware(
            total_memory_gb=1000.0,
            model_params_b=500.0
        )
        assert result['model_info']['is_small_model'] is False
        assert result['model_info']['cpu_compatible'] is False
        
        # Zero memory (should handle gracefully)
        result = recommender.recommend_hardware(
            total_memory_gb=0.0,
            model_params_b=0.0
        )
        assert isinstance(result, dict)
        assert 'model_info' in result
    
    def test_utilization_calculation_accuracy(self, recommender):
        """Test that utilization calculations are accurate."""
        # Test with known values
        total_memory = 40.0  # GB
        
        with patch.object(recommender.hardware, 'get_all_hardwares', return_value=[
            {
                'name': 'Test_GPU_80GB',
                'type': 'gpu',
                'Memory_size': 80,
                'manufacturer': 'Test',
                'flops': 10e12,
                'memory_bw': 1000
            }
        ]):
            result = recommender.recommend_hardware(
                total_memory_gb=total_memory,
                model_params_b=20.0
            )
            
            gpu_rec = result['gpu_recommendations'][0]
            # 40GB needed / 80GB available = 50% utilization
            assert gpu_rec['utilization'] == 50
            assert gpu_rec['nodes_required'] == 1
            assert gpu_rec['total_memory_available'] == 80


class TestBatchSizeOptimization:
    """Test batch size optimization for multi-node configurations."""
    
    @pytest.fixture
    def recommender(self):
        return HardwareRecommendation()
    
    def test_batch_size_scaling(self, recommender):
        """Test that batch sizes scale appropriately with node count."""
        # Small model that fits in CPU memory
        result = recommender.recommend_hardware(
            total_memory_gb=10.0,
            model_params_b=3.0
        )
        
        cpu_rec = next((r for r in result['cpu_recommendations'] if r['batch_recommendations']), None)
        if cpu_rec:
            batch_recs = cpu_rec['batch_recommendations']
            
            # Verify increasing node counts
            node_counts = [br['nodes'] for br in batch_recs]
            assert node_counts == sorted(node_counts)
            
            # Verify batch sizes make sense
            for br in batch_recs:
                assert br['recommended_batch_size'] > 0
                assert br['utilization_at_batch'] > 0
                assert br['utilization_at_batch'] <= 100
    
    def test_kv_cache_threshold(self, recommender):
        """Test 40GB KV cache threshold for CPU recommendations."""
        # Model with KV cache under 40GB
        result_under = recommender.recommend_hardware(
            total_memory_gb=35.0,  # Assuming ~20GB KV cache
            model_params_b=10.0
        )
        
        # Model with KV cache over 40GB
        result_over = recommender.recommend_hardware(
            total_memory_gb=80.0,  # Assuming ~50GB KV cache
            model_params_b=10.0
        )
        
        # Both should have recommendations, but behavior might differ
        assert 'cpu_recommendations' in result_under
        assert 'cpu_recommendations' in result_over
        
        # Under 40GB should be more suitable for CPU
        assert result_under['model_info']['cpu_compatible'] is True 