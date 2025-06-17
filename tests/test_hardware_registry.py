"""Test cases for hardware registry functionality."""
import unittest
from unittest.mock import Mock, patch
from BudSimulator.src.hardware_registry import HardwareRegistry


class TestHardwareRegistry(unittest.TestCase):
    """Test hardware registry for runtime integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset initialization flag
        HardwareRegistry._initialized = False
        
        # Mock hardware data
        self.mock_db_hardware = [
            {
                'name': 'NewGPU_100',
                'type': 'gpu',
                'Flops': 500,
                'Memory_size': 64,
                'Memory_BW': 1500,
                'ICN': 300,
                'ICN_LL': None,
                'real_values': True,
                'manufacturer': 'TestCorp'
            },
            {
                'name': 'NewCPU_200',
                'type': 'cpu',
                'Flops': 50,
                'Memory_size': 256,
                'Memory_BW': 200,
                'ICN': 100,
                'ICN_LL': 0.5,
                'real_values': False,
                'manufacturer': 'Intel'
            }
        ]
    
    @patch('BudSimulator.src.hardware_registry.BudHardware')
    @patch('BudSimulator.src.hardware_registry.system_configs')
    def test_initialize_adds_new_hardware(self, mock_system_configs, mock_bud_hardware):
        """Test that initialize adds new hardware to system_configs."""
        # Setup mocks
        mock_hw_instance = Mock()
        mock_hw_instance.get_all_hardwares.return_value = self.mock_db_hardware
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Mock system_configs as a dict with existing hardware
        mock_system_configs.__contains__ = lambda self, key: key == 'A100_40GB_GPU'
        mock_system_configs.__setitem__ = Mock()
        
        # Initialize
        HardwareRegistry.initialize()
        
        # Verify new hardware was added
        self.assertEqual(mock_system_configs.__setitem__.call_count, 2)
        
        # Check first hardware
        first_call = mock_system_configs.__setitem__.call_args_list[0]
        self.assertEqual(first_call[0][0], 'NewGPU_100')
        self.assertEqual(first_call[0][1]['Flops'], 500)
        self.assertEqual(first_call[0][1]['Memory_size'], 64)
        self.assertEqual(first_call[0][1]['Memory_BW'], 1500)
        self.assertEqual(first_call[0][1]['ICN'], 300)
        self.assertEqual(first_call[0][1]['real_values'], True)
        
        # Check second hardware
        second_call = mock_system_configs.__setitem__.call_args_list[1]
        self.assertEqual(second_call[0][0], 'NewCPU_200')
        self.assertEqual(second_call[0][1]['ICN_LL'], 0.5)
    
    @patch('BudSimulator.src.hardware_registry.BudHardware')
    @patch('BudSimulator.src.hardware_registry.system_configs')
    def test_initialize_skips_existing_hardware(self, mock_system_configs, mock_bud_hardware):
        """Test that initialize doesn't overwrite existing hardware."""
        # Setup mocks
        mock_hw_instance = Mock()
        mock_hw_instance.get_all_hardwares.return_value = self.mock_db_hardware
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Mock system_configs to say all hardware already exists
        mock_system_configs.__contains__ = lambda self, key: True
        mock_system_configs.__setitem__ = Mock()
        
        # Initialize
        HardwareRegistry.initialize()
        
        # Verify no hardware was added
        mock_system_configs.__setitem__.assert_not_called()
    
    @patch('BudSimulator.src.hardware_registry.BudHardware')
    def test_initialize_only_runs_once(self, mock_bud_hardware):
        """Test that initialize only runs once."""
        # Setup mock
        mock_hw_instance = Mock()
        mock_hw_instance.get_all_hardwares.return_value = []
        mock_bud_hardware.return_value = mock_hw_instance
        
        # First initialization
        HardwareRegistry.initialize()
        self.assertTrue(HardwareRegistry._initialized)
        self.assertEqual(mock_bud_hardware.call_count, 1)
        
        # Second initialization should not create new instance
        HardwareRegistry.initialize()
        self.assertEqual(mock_bud_hardware.call_count, 1)
    
    @patch('BudSimulator.src.hardware_registry.BudHardware')
    @patch('BudSimulator.src.hardware_registry.system_configs')
    @patch('BudSimulator.src.hardware_registry.cpu_configs', create=True)
    def test_get_cpu_configs(self, mock_cpu_configs, mock_system_configs, mock_bud_hardware):
        """Test getting CPU configurations."""
        # Setup mocks
        mock_hw_instance = Mock()
        mock_hw_instance.get_all_hardwares.return_value = self.mock_db_hardware
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Initialize first
        HardwareRegistry.initialize()
        
        # Get CPU configs
        cpu_configs = HardwareRegistry.get_cpu_configs()
        
        # Should return only CPU hardware
        self.assertEqual(len(cpu_configs), 1)
        self.assertEqual(cpu_configs[0]['name'], 'NewCPU_200')
        self.assertEqual(cpu_configs[0]['type'], 'cpu')
    
    @patch('BudSimulator.src.hardware_registry.BudHardware')
    def test_get_hardware_by_type(self, mock_bud_hardware):
        """Test getting hardware by type."""
        # Setup mocks
        mock_hw_instance = Mock()
        mock_hw_instance.get_all_hardwares.return_value = []  # Empty list for initialization
        mock_hw_instance.search_hardware.return_value = [
            {'name': 'GPU1', 'type': 'gpu'},
            {'name': 'GPU2', 'type': 'gpu'}
        ]
        mock_bud_hardware.return_value = mock_hw_instance
        
        # Get GPUs
        gpus = HardwareRegistry.get_hardware_by_type('gpu')
        
        # Verify search was called correctly
        mock_hw_instance.search_hardware.assert_called_once_with(type='gpu')
        self.assertEqual(len(gpus), 2)


if __name__ == '__main__':
    unittest.main() 