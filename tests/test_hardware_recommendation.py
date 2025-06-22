"""Test cases for hardware recommendation functionality."""
import unittest
import math
from unittest.mock import Mock, patch
from BudSimulator.src.hardware_recommendation import HardwareRecommendation


class TestHardwareRecommendation(unittest.TestCase):
    """Test hardware recommendation logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.recommender = HardwareRecommendation()
        
        # Mock hardware data
        self.mock_hardware = [
            {
                'name': 'H100_GPU',
                'type': 'gpu',
                'Memory_size': 80,
                'manufacturer': 'NVIDIA',
                'Flops': 1979,
                'Memory_BW': 3400,
                'ICN': 900
            },
            {
                'name': 'A100_GPU',
                'type': 'gpu', 
                'Memory_size': 40,
                'manufacturer': 'NVIDIA',
                'Flops': 312,
                'Memory_BW': 1600,
                'ICN': 150
            },
            {
                'name': 'MI300X',
                'type': 'gpu',
                'Memory_size': 192,
                'manufacturer': 'AMD',
                'Flops': 1307,
                'Memory_BW': 5300,
                'ICN': 400
            },
            {
                'name': 'SapphireRapids_CPU',
                'type': 'cpu',
                'Memory_size': 300,
                'manufacturer': 'Intel',
                'Flops': 33,
                'Memory_BW': 180,
                'ICN': 100
            },
            {
                'name': 'TPUv4',
                'type': 'asic',
                'Memory_size': 32,
                'manufacturer': 'Google',
                'Flops': 275,
                'Memory_BW': 1200,
                'ICN': 300
            }
        ]
    
    @patch.object(HardwareRecommendation, '__init__', lambda x: None)
    def test_recommend_hardware_basic(self):
        """Test basic hardware recommendation."""
        self.recommender = HardwareRecommendation()
        self.recommender.hardware = Mock()
        self.recommender.hardware.get_all_hardwares.return_value = self.mock_hardware
        
        # Test with 150GB requirement
        result = self.recommender.recommend_hardware(150.0)
        
        # Should return dictionary with cpu and gpu recommendations
        self.assertIsInstance(result, dict)
        self.assertIn('cpu_recommendations', result)
        self.assertIn('gpu_recommendations', result)
        self.assertIn('model_info', result)
        
        # Get all recommendations
        all_recommendations = result['cpu_recommendations'] + result['gpu_recommendations']
        self.assertEqual(len(all_recommendations), 5)
        
        # Check MI300X needs only 1 node (192GB > 150GB)
        mi300x = next((r for r in result['gpu_recommendations'] if r['hardware_name'] == 'MI300X'), None)
        self.assertIsNotNone(mi300x)
        self.assertEqual(mi300x['nodes_required'], 1)
        self.assertEqual(mi300x['memory_per_chip'], 192)
        self.assertEqual(mi300x['manufacturer'], 'AMD')
        self.assertEqual(mi300x['type'], 'gpu')
        
        # Check H100 needs 2 nodes (80GB * 2 = 160GB > 150GB)
        h100 = next((r for r in result['gpu_recommendations'] if r['hardware_name'] == 'H100_GPU'), None)
        self.assertIsNotNone(h100)
        self.assertEqual(h100['nodes_required'], 2)
        
        # Check A100 needs 4 nodes (40GB * 4 = 160GB > 150GB)
        a100 = next((r for r in result['gpu_recommendations'] if r['hardware_name'] == 'A100_GPU'), None)
        self.assertIsNotNone(a100)
        self.assertEqual(a100['nodes_required'], 4)
    
    @patch.object(HardwareRecommendation, '__init__', lambda x: None)
    def test_cpu_filtering_large_models(self):
        """Test CPU filtering for models > 13B parameters."""
        self.recommender = HardwareRecommendation()
        self.recommender.hardware = Mock()
        self.recommender.hardware.get_all_hardwares.return_value = self.mock_hardware
        
        # Test with 70B model
        result = self.recommender.recommend_hardware(150.0, model_params_b=70.0)
        
        # CPU recommendations will still be returned but model_info will show it's not cpu_compatible
        self.assertIn('cpu_recommendations', result)
        self.assertIn('gpu_recommendations', result)
        
        # Model info should indicate it's not a small model
        self.assertFalse(result['model_info']['is_small_model'])
        self.assertFalse(result['model_info']['cpu_compatible'])
        
        # Should have GPU recommendations
        self.assertGreater(len(result['gpu_recommendations']), 0)
    
    @patch.object(HardwareRecommendation, '__init__', lambda x: None)
    def test_cpu_included_small_models(self):
        """Test CPU included for models <= 13B parameters."""
        self.recommender = HardwareRecommendation()
        self.recommender.hardware = Mock()
        self.recommender.hardware.get_all_hardwares.return_value = self.mock_hardware
        
        # Test with 7B model
        result = self.recommender.recommend_hardware(20.0, model_params_b=7.0)
        
        # Should include CPU recommendations
        self.assertGreater(len(result['cpu_recommendations']), 0)
        
        # Check for SapphireRapids_CPU
        cpu_rec = next((r for r in result['cpu_recommendations'] if r['hardware_name'] == 'SapphireRapids_CPU'), None)
        self.assertIsNotNone(cpu_rec)
        
        # Model info should indicate it's a small model and CPU compatible
        self.assertTrue(result['model_info']['is_small_model'])
        self.assertTrue(result['model_info']['cpu_compatible'])
    
    @patch.object(HardwareRecommendation, '__init__', lambda x: None)
    def test_sorting_by_utilization(self):
        """Test recommendations are sorted by utilization in descending order."""
        self.recommender = HardwareRecommendation()
        self.recommender.hardware = Mock()
        self.recommender.hardware.get_all_hardwares.return_value = self.mock_hardware
        
        result = self.recommender.recommend_hardware(500.0)
        
        # Check GPU recommendations are sorted by utilization (descending)
        gpu_utilizations = [r['utilization'] for r in result['gpu_recommendations']]
        self.assertEqual(gpu_utilizations, sorted(gpu_utilizations, reverse=True))
        
        # Check CPU recommendations are sorted by utilization (descending)
        if len(result['cpu_recommendations']) > 1:
            cpu_utilizations = [r['utilization'] for r in result['cpu_recommendations']]
            self.assertEqual(cpu_utilizations, sorted(cpu_utilizations, reverse=True))
        
        # Verify specific calculations
        # CPU: 500/300 = 2 nodes
        # MI300X: 500/192 = 3 nodes
        # H100: 500/80 = 7 nodes
        cpu_rec = next((r for r in result['cpu_recommendations'] if r['hardware_name'] == 'SapphireRapids_CPU'), None)
        if cpu_rec:
            self.assertEqual(cpu_rec['nodes_required'], 2)
        
        mi300x_rec = next((r for r in result['gpu_recommendations'] if r['hardware_name'] == 'MI300X'), None)
        if mi300x_rec:
            self.assertEqual(mi300x_rec['nodes_required'], 3)
    
    @patch.object(HardwareRecommendation, '__init__', lambda x: None)
    def test_zero_memory_hardware(self):
        """Test handling of hardware with zero memory."""
        self.recommender = HardwareRecommendation()
        self.recommender.hardware = Mock()
        
        hardware_with_zero_mem = self.mock_hardware + [{
            'name': 'ZeroMemHW',
            'type': 'accelerator',
            'Memory_size': 0,
            'manufacturer': 'Test',
            'Flops': 100,
            'Memory_BW': 100,
            'ICN': 100
        }]
        
        self.recommender.hardware.get_all_hardwares.return_value = hardware_with_zero_mem
        
        result = self.recommender.recommend_hardware(100.0)
        
        # Should not include hardware with zero memory
        all_recommendations = result['cpu_recommendations'] + result['gpu_recommendations']
        zero_mem_recs = [r for r in all_recommendations if r['hardware_name'] == 'ZeroMemHW']
        self.assertEqual(len(zero_mem_recs), 0)


if __name__ == '__main__':
    unittest.main() 