"""Test cases for hardware API endpoints."""
import unittest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from BudSimulator.apis.routers.hardware import router as hardware_router


class TestHardwareAPI(unittest.TestCase):
    """Test hardware API endpoints."""
    
    def setUp(self):
        """Set up test FastAPI app."""
        self.app = FastAPI()
        self.app.include_router(hardware_router, prefix="/api")
        self.client = TestClient(self.app)
    
    @patch('BudSimulator.apis.routers.hardware.BudHardware')
    def test_add_hardware_success(self, mock_bud_hardware):
        """Test successful hardware addition."""
        # Setup mock
        mock_hw_instance = Mock()
        mock_hw_instance.add_hardware.return_value = True
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Test data
        hardware_data = {
            'name': 'TestGPU',
            'type': 'gpu',
            'flops': 500,
            'memory_size': 64,
            'memory_bw': 1500,
            'manufacturer': 'TestCorp'
        }
        
        # Make request
        response = self.client.post('/api/hardware', json=hardware_data)
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['message'], 'Hardware added successfully')
        
        # Verify mock was called (check that the required fields are present)
        call_args = mock_hw_instance.add_hardware.call_args[0][0]
        self.assertEqual(call_args['name'], 'TestGPU')
        self.assertEqual(call_args['type'], 'gpu')
        self.assertEqual(call_args['flops'], 500.0)
        self.assertEqual(call_args['memory_size'], 64.0)
        self.assertEqual(call_args['memory_bw'], 1500.0)
        self.assertEqual(call_args['manufacturer'], 'TestCorp')
    
    @patch('BudSimulator.apis.routers.hardware.BudHardware')
    def test_add_hardware_validation_error(self, mock_bud_hardware):
        """Test hardware addition with validation error."""
        # Setup mock to raise ValueError
        mock_hw_instance = Mock()
        mock_hw_instance.add_hardware.side_effect = ValueError("Missing required field: flops")
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Test data (missing flops)
        hardware_data = {
            'name': 'TestGPU',
            'type': 'gpu',
            'memory_size': 64,
            'memory_bw': 1500
        }
        
        # Make request
        response = self.client.post('/api/hardware', json=hardware_data)
        
        # Verify response (FastAPI returns 422 for validation errors)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        # Check that it's a validation error for missing flops field
        self.assertIn('detail', data)
        self.assertTrue(any('flops' in str(error) for error in data['detail']))
    
    @patch('BudSimulator.apis.routers.hardware.BudHardware')
    def test_list_hardware(self, mock_bud_hardware):
        """Test listing hardware with filters."""
        # Setup mock
        mock_hw_instance = Mock()
        mock_hw_instance.search_hardware.return_value = [
            {'name': 'GPU1', 'type': 'gpu', 'manufacturer': 'NVIDIA'},
            {'name': 'GPU2', 'type': 'gpu', 'manufacturer': 'AMD'}
        ]
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Make request with filters
        response = self.client.get('/api/hardware?type=gpu&sort_by=name')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['hardware']), 2)
        
        # Verify mock was called with correct params
        mock_hw_instance.search_hardware.assert_called_once()
        call_args = mock_hw_instance.search_hardware.call_args[1]
        self.assertEqual(call_args['type'], 'gpu')
        self.assertEqual(call_args['sort_by'], 'name')
    
    @patch('BudSimulator.apis.routers.hardware.HardwareRecommender')
    def test_recommend_hardware(self, mock_hw_recommendation):
        """Test hardware recommendation endpoint."""
        # Setup mock
        mock_recommender = Mock()
        mock_recommender.recommend_hardware.return_value = [
            {
                'hardware_name': 'H100_GPU',
                'nodes_required': 2,
                'memory_per_chip': 80,
                'manufacturer': 'NVIDIA',
                'type': 'gpu'
            },
            {
                'hardware_name': 'MI300X',
                'nodes_required': 1,
                'memory_per_chip': 192,
                'manufacturer': 'AMD',
                'type': 'gpu'
            }
        ]
        mock_hw_recommendation.return_value = mock_recommender
        
        # Test data
        request_data = {
            'total_memory_gb': 150.0,
            'model_params_b': 70.0
        }
        
        # Make request
        response = self.client.post('/api/hardware/recommend', json=request_data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['recommendations']), 2)
        self.assertEqual(data['recommendations'][0]['hardware_name'], 'H100_GPU')
        
        # Verify mock was called
        mock_recommender.recommend_hardware.assert_called_once_with(150.0, model_params_b=70.0)
    
    @patch('BudSimulator.apis.routers.hardware.HardwareRecommender')
    def test_recommend_hardware_missing_memory(self, mock_hw_recommendation):
        """Test recommendation endpoint with missing required field."""
        # Test data (missing total_memory_gb)
        request_data = {
            'model_params_b': 70.0
        }
        
        # Make request
        response = self.client.post('/api/hardware/recommend', json=request_data)
        
        # Verify response
        self.assertEqual(response.status_code, 422)  # FastAPI validation error
    
    @patch('BudSimulator.apis.routers.hardware.BudHardware')
    def test_filter_hardware(self, mock_bud_hardware):
        """Test hardware filtering endpoint."""
        # Setup mock
        mock_hw_instance = Mock()
        mock_hw_instance.search_hardware.return_value = [
            {'name': 'A100', 'type': 'gpu', 'memory_size': 40},
            {'name': 'H100', 'type': 'gpu', 'memory_size': 80}
        ]
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Make request with multiple filters
        response = self.client.get('/api/hardware/filter?type=gpu&min_memory=40&max_memory=100')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['hardware']), 2)
        
        # Verify mock was called with correct params
        call_args = mock_hw_instance.search_hardware.call_args[1]
        self.assertEqual(call_args['type'], 'gpu')
        self.assertEqual(call_args['min_memory'], 40.0)
        self.assertEqual(call_args['max_memory'], 100.0)


if __name__ == '__main__':
    unittest.main() 