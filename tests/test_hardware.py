"""
Test suite for BudHardware class using Test-Driven Development.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware import BudHardware
from src.db import DatabaseConnection
from Systems.system_configs import system_configs


class TestBudHardwareBasics(unittest.TestCase):
    """Test basic functionality of BudHardware class."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize BudHardware with test database
        self.hardware = BudHardware(db_path=self.db_path)
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'hardware'):
            del self.hardware
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_initialization(self):
        """Test BudHardware initialization."""
        self.assertIsNotNone(self.hardware)
        self.assertIsNotNone(self.hardware.db)
    
    def test_load_system_configs(self):
        """Test loading hardware from system_configs."""
        all_hardware = self.hardware.get_all_hardwares()
        
        # Check that system_configs hardware is loaded
        for hw_name in system_configs:
            self.assertIn(hw_name, [hw['name'] for hw in all_hardware])
    
    def test_get_hardware_by_name(self):
        """Test getting specific hardware by name."""
        # Test existing hardware from system_configs
        a100_config = self.hardware.get_hardware_by_name('A100_40GB_GPU')
        self.assertIsNotNone(a100_config)
        self.assertEqual(a100_config['Flops'], 312)
        self.assertEqual(a100_config['Memory_size'], 40)
        
        # Test non-existent hardware
        non_existent = self.hardware.get_hardware_by_name('NonExistentGPU')
        self.assertFalse(non_existent)
    
    def test_get_config_genz_compatibility(self):
        """Test GenZ compatibility with get_config method."""
        config = self.hardware.get_config('H100_GPU')
        self.assertIsNotNone(config)
        self.assertIn('Flops', config)
        self.assertIn('Memory_size', config)
        self.assertIn('Memory_BW', config)
        self.assertIn('ICN', config)
    
    def test_format_memory_size(self):
        """Test memory size formatting."""
        test_cases = [
            (512, "512 MB"),
            (1024, "1.02 GB"),
            (1000, "1 GB"),
            (1048576, "1.05 TB"),
            (1073741824, "1.07 PB"),
        ]
        
        for mb, expected in test_cases:
            result = self.hardware.format_memory_size(mb)
            self.assertEqual(result, expected)


class TestBudHardwareCRUD(unittest.TestCase):
    """Test CRUD operations for BudHardware."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        self.hardware = BudHardware(db_path=self.db_path)
        
        # Test hardware data
        self.test_hardware = {
            'name': 'TestGPU_100',
            'type': 'gpu',
            'manufacturer': 'TestCorp',
            'flops': 500,
            'memory_size': 64,
            'memory_bw': 1500,
            'icn': 300,
            'power': 350,
            'real_values': True,
            'description': 'Test GPU for unit tests',
            'url': 'https://example.com/testgpu'
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'hardware'):
            del self.hardware
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_add_hardware(self):
        """Test adding new hardware."""
        # Add hardware
        result = self.hardware.add_hardware(self.test_hardware)
        self.assertTrue(result)
        
        # Verify it was added
        hw = self.hardware.get_hardware_by_name('TestGPU_100')
        self.assertIsNotNone(hw)
        self.assertEqual(hw['flops'], 500)
        self.assertEqual(hw['memory_size'], 64)
        
        # Test duplicate addition
        with self.assertRaises(ValueError):
            self.hardware.add_hardware(self.test_hardware)
    
    def test_update_hardware(self):
        """Test updating existing hardware."""
        # Add hardware first with different name
        update_test_hw = self.test_hardware.copy()
        update_test_hw['name'] = 'TestGPU_Update'
        self.hardware.add_hardware(update_test_hw)
        
        # Update it
        updates = {
            'flops': 600,
            'memory_size': 80,
            'power': 400
        }
        result = self.hardware.update_hardware('TestGPU_Update', updates)
        self.assertTrue(result)
        
        # Verify updates
        hw = self.hardware.get_hardware_by_name('TestGPU_Update')
        self.assertEqual(hw['flops'], 600)
        self.assertEqual(hw['memory_size'], 80)
        self.assertEqual(hw['power'], 400)
        
        # Test updating non-existent hardware
        result = self.hardware.update_hardware('NonExistent', updates)
        self.assertFalse(result)
    
    def test_delete_hardware(self):
        """Test deleting hardware."""
        # Add hardware first with different name
        delete_test_hw = self.test_hardware.copy()
        delete_test_hw['name'] = 'TestGPU_Delete'
        self.hardware.add_hardware(delete_test_hw)
        
        # Soft delete
        result = self.hardware.delete_hardware('TestGPU_Delete')
        self.assertTrue(result)
        
        # Verify it's not returned in normal queries
        hw = self.hardware.get_hardware_by_name('TestGPU_Delete')
        self.assertFalse(hw)
        
        # Test deleting non-existent hardware
        result = self.hardware.delete_hardware('NonExistent')
        self.assertFalse(result)
    
    def test_validation(self):
        """Test hardware validation."""
        # Test missing required fields
        invalid_hw = {'name': 'Invalid', 'type': 'gpu'}
        with self.assertRaises(ValueError):
            self.hardware.add_hardware(invalid_hw)
        
        # Test invalid type
        invalid_hw = {
            'name': 'Invalid',
            'type': 'invalid_type',
            'flops': 100,
            'memory_size': 32,
            'memory_bw': 500
        }
        with self.assertRaises(ValueError):
            self.hardware.add_hardware(invalid_hw)
        
        # Test negative values
        invalid_hw = {
            'name': 'Invalid',
            'type': 'gpu',
            'flops': -100,
            'memory_size': 32,
            'memory_bw': 500
        }
        with self.assertRaises(ValueError):
            self.hardware.add_hardware(invalid_hw)


class TestBudHardwareSearch(unittest.TestCase):
    """Test search, filter, and sort functionality."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        self.hardware = BudHardware(db_path=self.db_path)
        
        # Use timestamp to ensure unique names
        import time
        self.test_prefix = f"Search_{int(time.time() * 1000)}"
        
        # Add test hardware
        test_hardware_list = [
            {
                'name': f'{self.test_prefix}_GPU_High_Mem',
                'type': 'gpu',
                'manufacturer': 'NVIDIA',
                'flops': 1000,
                'memory_size': 128,
                'memory_bw': 2000,
                'power': 500,
                'on_prem_vendors': [
                    {'name': 'Vendor1', 'price_lower': 25000, 'price_upper': 30000}
                ]
            },
            {
                'name': f'{self.test_prefix}_GPU_Mid_Range',
                'type': 'gpu',
                'manufacturer': 'AMD',
                'flops': 500,
                'memory_size': 64,
                'memory_bw': 1000,
                'power': 300,
                'on_prem_vendors': [
                    {'name': 'Vendor1', 'price_lower': 10000, 'price_upper': 15000}
                ]
            },
            {
                'name': f'{self.test_prefix}_CPU_Server',
                'type': 'cpu',
                'manufacturer': 'Intel',
                'flops': 50,
                'memory_size': 512,
                'memory_bw': 200,
                'power': 250,
                'on_prem_vendors': [
                    {'name': 'Vendor1', 'price_lower': 5000, 'price_upper': 7000}
                ]
            },
            {
                'name': f'{self.test_prefix}_Accelerator_AI',
                'type': 'accelerator',
                'manufacturer': 'Google',
                'flops': 800,
                'memory_size': 32,
                'memory_bw': 1500,
                'power': 200,
                'on_prem_vendors': [
                    {'name': 'Vendor1', 'price_lower': 15000, 'price_upper': 20000}
                ]
            }
        ]
        
        for hw in test_hardware_list:
            self.hardware.add_hardware(hw)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'hardware'):
            del self.hardware
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_search_by_type(self):
        """Test searching by hardware type."""
        # Search for GPUs
        all_gpus = self.hardware.search_hardware(type='gpu')
        gpus = [hw for hw in all_gpus if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(gpus), 2)
        for gpu in gpus:
            self.assertEqual(gpu['type'], 'gpu')
        
        # Search for multiple types
        all_results = self.hardware.search_hardware(type=['gpu', 'cpu'])
        results = [hw for hw in all_results if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(results), 3)
    
    def test_search_by_manufacturer(self):
        """Test searching by manufacturer."""
        all_nvidia_hw = self.hardware.search_hardware(manufacturer='NVIDIA')
        nvidia_hw = [hw for hw in all_nvidia_hw if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(nvidia_hw), 1)
        self.assertEqual(nvidia_hw[0]['name'], f'{self.test_prefix}_GPU_High_Mem')
    
    def test_search_by_performance_range(self):
        """Test searching by performance metrics."""
        # Search by FLOPS range - filter by our test hardware names
        high_perf = self.hardware.search_hardware(min_flops=500)
        # Filter to only our test hardware
        test_high_perf = [hw for hw in high_perf if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(test_high_perf), 3)
        
        # Search by memory size
        high_mem = self.hardware.search_hardware(min_memory=64)
        test_high_mem = [hw for hw in high_mem if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(test_high_mem), 3)
        
        # Combined filters
        all_results = self.hardware.search_hardware(
            min_flops=500,
            max_power=400,
            type='gpu'
        )
        results = [hw for hw in all_results if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], f'{self.test_prefix}_GPU_Mid_Range')
    
    def test_search_by_price_range(self):
        """Test searching by price range."""
        budget_hw = self.hardware.search_hardware(
            min_price=5000,
            max_price=15000
        )
        test_budget_hw = [hw for hw in budget_hw if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(len(test_budget_hw), 2)
    
    def test_sorting(self):
        """Test sorting functionality."""
        # Sort by FLOPS descending - filter to test hardware only
        all_results = self.hardware.search_hardware(
            sort_by='flops',
            sort_order='desc'
        )
        results = [hw for hw in all_results if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(results[0]['name'], f'{self.test_prefix}_GPU_High_Mem')
        self.assertEqual(results[-1]['name'], f'{self.test_prefix}_CPU_Server')
        
        # Sort by price ascending
        all_results = self.hardware.search_hardware(
            sort_by='price',
            sort_order='asc'
        )
        results = [hw for hw in all_results if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(results[0]['name'], f'{self.test_prefix}_CPU_Server')
        
        # Sort by performance per watt
        all_results = self.hardware.search_hardware(
            sort_by='perf_per_watt',
            sort_order='desc'
        )
        results = [hw for hw in all_results if hw['name'].startswith(self.test_prefix)]
        self.assertEqual(results[0]['name'], f'{self.test_prefix}_Accelerator_AI')  # 800/200 = 4.0
    
    def test_pagination(self):
        """Test pagination in search results."""
        # Get first 2 results
        page1 = self.hardware.search_hardware(limit=2, offset=0)
        self.assertEqual(len(page1), 2)
        
        # Get next 2 results
        page2 = self.hardware.search_hardware(limit=2, offset=2)
        self.assertEqual(len(page2), 2)
        
        # Verify no overlap
        page1_names = [hw['name'] for hw in page1]
        page2_names = [hw['name'] for hw in page2]
        self.assertEqual(len(set(page1_names) & set(page2_names)), 0)


class TestBudHardwareImportExport(unittest.TestCase):
    """Test import/export functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        self.hardware = BudHardware(db_path=self.db_path)
        
        # Create temporary JSON file
        self.temp_json = tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            suffix='.json'
        )
        
        # Sample hardware data from hardware.json
        self.sample_data = {
            "TestGPU": {
                "Flops": 1000,
                "Memory_size": 80,
                "Memory_BW": 2000,
                "ICN": 600,
                "Power": 400,
                "real_values": True,
                "name": "Test GPU Model",
                "manufacturer": "TestCorp",
                "type": "gpu",
                "url": "https://example.com",
                "description": "Test GPU for import",
                "advantages": ["Fast", "Efficient"],
                "disadvantages": ["Expensive"],
                "on_prem_price": {
                    "lower_bound": 20000,
                    "upper_bound": 25000
                },
                "on_prem_vendors": ["Dell", "HPE"],
                "cloud_pricing": {
                    "lower_bound": 3.0,
                    "upper_bound": 5.0
                },
                "supported_clouds": [
                    {
                        "cloud_name": "AWS",
                        "regions": ["us-east-1"],
                        "instance_name": ["p4d"]
                    }
                ]
            }
        }
        
        json.dump(self.sample_data, self.temp_json)
        self.temp_json.close()
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'hardware'):
            del self.hardware
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        if os.path.exists(self.temp_json.name):
            os.unlink(self.temp_json.name)
    
    def test_import_from_json(self):
        """Test importing hardware from JSON file."""
        # Import from JSON
        result = self.hardware.import_from_json(self.temp_json.name)
        self.assertTrue(result['success'])
        self.assertEqual(result['imported'], 1)
        self.assertEqual(result['failed'], 0)
        
        # Verify imported hardware
        hw = self.hardware.get_hardware_by_name('TestGPU')
        self.assertIsNotNone(hw)
        self.assertEqual(hw['flops'], 1000)
        self.assertEqual(hw['manufacturer'], 'TestCorp')
        
        # Verify vendors were imported
        vendors = self.hardware.get_hardware_vendors('TestGPU')
        vendor_names = [v['vendor_name'] for v in vendors]
        self.assertIn('Dell', vendor_names)
        self.assertIn('HPE', vendor_names)
        
        # Verify cloud info was imported
        clouds = self.hardware.get_hardware_clouds('TestGPU')
        cloud_names = [c['cloud_name'] for c in clouds]
        self.assertIn('AWS', cloud_names)
    
    def test_export_to_json(self):
        """Test exporting hardware to JSON."""
        # Add some hardware
        self.hardware.add_hardware({
            'name': 'ExportTest',
            'type': 'gpu',
            'flops': 500,
            'memory_size': 32,
            'memory_bw': 1000
        })
        
        # Export to JSON
        export_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        export_file.close()
        
        try:
            result = self.hardware.export_to_json(export_file.name)
            self.assertTrue(result)
            
            # Verify exported data
            with open(export_file.name, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn('ExportTest', exported_data)
            self.assertEqual(exported_data['ExportTest']['Flops'], 500)
        finally:
            os.unlink(export_file.name)


class TestBudHardwareRegression(unittest.TestCase):
    """Regression tests to ensure existing functionality isn't broken."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_1_system_configs_compatibility(self):
        """Test that system_configs hardware still works."""
        # Create a fresh hardware instance
        hardware = BudHardware(db_path=self.db_path)
        
        # All system_configs hardware should be accessible
        for hw_name, hw_config in system_configs.items():
            retrieved = hardware.get_hardware_by_name(hw_name)
            self.assertIsNotNone(retrieved, f"Hardware {hw_name} not found")
            
            # Verify key fields match
            self.assertEqual(retrieved['Flops'], hw_config['Flops'], 
                           f"Flops mismatch for {hw_name}: {retrieved['Flops']} != {hw_config['Flops']}")
            self.assertEqual(retrieved['Memory_size'], hw_config['Memory_size'])
            self.assertEqual(retrieved['Memory_BW'], hw_config['Memory_BW'])
            self.assertEqual(retrieved['ICN'], hw_config['ICN'])
    
    def test_2_genz_interface_compatibility(self):
        """Test GenZ interface compatibility."""
        # Create a fresh hardware instance
        hardware = BudHardware(db_path=self.db_path)
        
        # Test get_config method (used by GenZ)
        for hw_name in system_configs:
            config = hardware.get_config(hw_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, dict)
            
            # Verify it has all required GenZ fields
            required_fields = ['Flops', 'Memory_size', 'Memory_BW', 'ICN', 'real_values']
            for field in required_fields:
                self.assertIn(field, config)
    
    def test_3_case_sensitivity(self):
        """Test that hardware names are case-sensitive (as in original)."""
        # Create a fresh hardware instance
        hardware = BudHardware(db_path=self.db_path)
        
        # Original system_configs is case-sensitive
        hw = hardware.get_hardware_by_name('A100_40GB_GPU')
        self.assertIsNotNone(hw)
        
        # Different case should not work
        hw_lower = hardware.get_hardware_by_name('a100_40gb_gpu')
        self.assertFalse(hw_lower)
    
    def test_4_immutability_of_system_configs(self):
        """Test that system_configs cannot be modified through BudHardware."""
        # Create a fresh hardware instance
        hardware = BudHardware(db_path=self.db_path)
        
        # Get a hardware config
        original_flops = system_configs['A100_40GB_GPU']['Flops']
        
        # Try to update it (should only update in DB, not system_configs)
        hardware.update_hardware('A100_40GB_GPU', {'flops': 999})
        
        # Verify system_configs is unchanged
        self.assertEqual(system_configs['A100_40GB_GPU']['Flops'], original_flops)
        
        # But BudHardware should return updated value
        hw = hardware.get_hardware_by_name('A100_40GB_GPU')
        self.assertEqual(hw['flops'], 999)
        self.assertEqual(hw['Flops'], 999)  # Also check uppercase key


if __name__ == '__main__':
    unittest.main() 