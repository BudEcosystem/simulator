"""
Test the new hardware schema with instance types and vendor-specific pricing.
"""

import pytest
import tempfile
import json
from pathlib import Path

from BudSimulator.src.hardware import BudHardware
from BudSimulator.src.db.connection import DatabaseConnection


class TestHardwareNewSchema:
    """Test the new hardware schema functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def hardware_manager(self, temp_db):
        """Create a BudHardware instance with temporary database."""
        return BudHardware(db_path=temp_db)
    
    def test_add_hardware_with_vendor_pricing(self, hardware_manager):
        """Test adding hardware with vendor-specific pricing."""
        hardware_data = {
            'name': 'TestGPU_VendorPricing',
            'type': 'gpu',
            'manufacturer': 'TestCorp',
            'flops': 1000,
            'memory_size': 80,
            'memory_bw': 2000,
            'on_prem_vendors': [
                {'name': 'Vendor1', 'price_lower': 10000, 'price_upper': 12000},
                {'name': 'Vendor2', 'price_lower': 11000, 'price_upper': 13000},
                'Vendor3'  # Vendor without pricing
            ]
        }
        
        # Add hardware
        assert hardware_manager.add_hardware(hardware_data)
        
        # Get vendor details
        vendors = hardware_manager.get_hardware_vendors('TestGPU_VendorPricing')
        assert len(vendors) == 3
        
        # Check vendor pricing
        vendor_dict = {v['vendor_name']: v for v in vendors}
        assert vendor_dict['Vendor1']['price_lower'] == 10000
        assert vendor_dict['Vendor1']['price_upper'] == 12000
        assert vendor_dict['Vendor2']['price_lower'] == 11000
        assert vendor_dict['Vendor2']['price_upper'] == 13000
        assert vendor_dict['Vendor3']['price_lower'] is None
        assert vendor_dict['Vendor3']['price_upper'] is None
    
    def test_add_hardware_with_cloud_instances(self, hardware_manager):
        """Test adding hardware with cloud instance types and pricing."""
        hardware_data = {
            'name': 'TestGPU_CloudInstances',
            'type': 'gpu',
            'manufacturer': 'TestCorp',
            'flops': 1500,
            'memory_size': 80,
            'memory_bw': 3000,
            'clouds': [
                {
                    'name': 'AWS',
                    'regions': ['us-east-1', 'us-west-2'],
                    'instances': [
                        {'name': 'p5.48xlarge', 'price_lower': 3.5, 'price_upper': 4.0},
                        {'name': 'p5.24xlarge', 'price_lower': 2.0, 'price_upper': 2.5}
                    ]
                },
                {
                    'name': 'Azure',
                    'regions': ['eastus', 'westus'],
                    'instances': [
                        {'name': 'ND_H100_v5', 'price_lower': 3.8, 'price_upper': 4.2}
                    ]
                }
            ]
        }
        
        # Add hardware
        assert hardware_manager.add_hardware(hardware_data)
        
        # Get cloud details
        clouds = hardware_manager.get_hardware_clouds('TestGPU_CloudInstances')
        assert len(clouds) == 3  # 2 AWS instances + 1 Azure instance
        
        # Check cloud instance details
        aws_instances = [c for c in clouds if c['cloud_name'] == 'AWS']
        assert len(aws_instances) == 2
        
        # Check instance pricing
        p5_48xl = next(c for c in aws_instances if c['instance_name'] == 'p5.48xlarge')
        assert p5_48xl['price_lower'] == 3.5
        assert p5_48xl['price_upper'] == 4.0
        assert 'us-east-1' in p5_48xl['regions']
        assert 'us-west-2' in p5_48xl['regions']
    
    def test_import_json_with_new_structure(self, hardware_manager):
        """Test importing hardware.json with the new structure."""
        # Create test JSON with new structure
        test_json = {
            "TestHardware1": {
                "Flops": 500,
                "Memory_size": 32,
                "Memory_BW": 1000,
                "ICN": 200,
                "type": "gpu",
                "manufacturer": "TestCorp",
                "on_prem_price": {
                    "lower_bound": 5000,
                    "upper_bound": 6000
                },
                "on_prem_vendors": ["Dell", "HPE"],
                "cloud_pricing": {
                    "lower_bound": 1.5,
                    "upper_bound": 2.0
                },
                "supported_clouds": [
                    {
                        "cloud_name": "AWS",
                        "regions": ["us-east-1"],
                        "instance_name": ["g5.xlarge", "g5.2xlarge"]
                    }
                ]
            }
        }
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_json, f)
            temp_path = f.name
        
        try:
            # Import
            result = hardware_manager.import_from_json(temp_path)
            assert result['success']
            assert result['imported'] == 1
            
            # Verify vendor pricing
            vendors = hardware_manager.get_hardware_vendors('TestHardware1')
            assert len(vendors) == 2
            for vendor in vendors:
                assert vendor['price_lower'] == 5000
                assert vendor['price_upper'] == 6000
            
            # Verify cloud instances
            clouds = hardware_manager.get_hardware_clouds('TestHardware1')
            assert len(clouds) == 2  # 2 instance types
            for cloud in clouds:
                assert cloud['cloud_name'] == 'AWS'
                assert cloud['price_lower'] == 1.5
                assert cloud['price_upper'] == 2.0
                assert cloud['instance_name'] in ['g5.xlarge', 'g5.2xlarge']
        
        finally:
            Path(temp_path).unlink()
    
    def test_search_with_vendor_pricing(self, hardware_manager):
        """Test searching hardware with vendor price filters."""
        # Add test hardware with different pricing
        hardware1 = {
            'name': 'CheapGPU',
            'type': 'gpu',
            'flops': 100,
            'memory_size': 16,
            'memory_bw': 500,
            'on_prem_vendors': [
                {'name': 'BudgetVendor', 'price_lower': 1000, 'price_upper': 1500}
            ]
        }
        
        hardware2 = {
            'name': 'ExpensiveGPU',
            'type': 'gpu',
            'flops': 1000,
            'memory_size': 80,
            'memory_bw': 2000,
            'on_prem_vendors': [
                {'name': 'PremiumVendor', 'price_lower': 10000, 'price_upper': 15000}
            ]
        }
        
        hardware_manager.add_hardware(hardware1)
        hardware_manager.add_hardware(hardware2)
        
        # Search with price filter
        results = hardware_manager.search_hardware(min_price=500, max_price=2000)
        assert len(results) == 1
        assert results[0]['name'] == 'CheapGPU'
        
        # Search expensive hardware
        results = hardware_manager.search_hardware(min_price=8000)
        assert len(results) == 1
        assert results[0]['name'] == 'ExpensiveGPU'
    
    def test_export_with_new_structure(self, hardware_manager):
        """Test exporting hardware with the new structure."""
        # Add hardware with full details
        hardware_data = {
            'name': 'ExportTestGPU',
            'type': 'gpu',
            'manufacturer': 'ExportCorp',
            'flops': 750,
            'memory_size': 48,
            'memory_bw': 1500,
            'on_prem_vendors': [
                {'name': 'Vendor1', 'price_lower': 7000, 'price_upper': 8000},
                {'name': 'Vendor2', 'price_lower': 7500, 'price_upper': 8500}
            ],
            'clouds': [
                {
                    'name': 'GCP',
                    'regions': ['us-central1', 'europe-west1'],
                    'instances': [
                        {'name': 'a2-highgpu-1g', 'price_lower': 2.5, 'price_upper': 3.0}
                    ]
                }
            ]
        }
        
        hardware_manager.add_hardware(hardware_data)
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export
            assert hardware_manager.export_to_json(temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                exported = json.load(f)
            
            assert 'ExportTestGPU' in exported
            hw = exported['ExportTestGPU']
            
            # Check vendor info
            assert 'on_prem_vendors' in hw
            assert set(hw['on_prem_vendors']) == {'Vendor1', 'Vendor2'}
            assert hw['on_prem_price']['lower_bound'] == 7000  # Min of all vendors
            assert hw['on_prem_price']['upper_bound'] == 8500  # Max of all vendors
            
            # Check cloud info
            assert 'supported_clouds' in hw
            assert len(hw['supported_clouds']) == 1
            cloud = hw['supported_clouds'][0]
            assert cloud['cloud_name'] == 'GCP'
            assert 'a2-highgpu-1g' in cloud['instance_name']
            assert hw['cloud_pricing']['lower_bound'] == 2.5
            assert hw['cloud_pricing']['upper_bound'] == 3.0
        
        finally:
            Path(temp_path).unlink()
    
    def test_backward_compatibility(self, hardware_manager):
        """Test that old format still works (backward compatibility)."""
        # Old format with single pricing
        old_format = {
            'name': 'OldFormatGPU',
            'type': 'gpu',
            'flops': 200,
            'memory_size': 24,
            'memory_bw': 800,
            'on_prem_vendors': ['Dell', 'HPE', 'Lenovo']  # Simple list of vendors
        }
        
        # Should still work
        assert hardware_manager.add_hardware(old_format)
        
        # Verify vendors were added without pricing
        vendors = hardware_manager.get_hardware_vendors('OldFormatGPU')
        assert len(vendors) == 3
        for vendor in vendors:
            assert vendor['price_lower'] is None
            assert vendor['price_upper'] is None 