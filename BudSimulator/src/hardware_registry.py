"""Hardware registry for runtime integration of database hardware into system_configs."""
from typing import List, Dict, Any, Optional
from .hardware import BudHardware
from Systems.system_configs import system_configs


class HardwareRegistry:
    """Registry to inject database hardware into system configs at runtime."""
    
    _initialized = False
    _hardware_instance = None
    
    @classmethod
    def initialize(cls):
        """Initialize hardware in system configs at startup."""
        if cls._initialized:
            return
        
        # Create hardware instance
        cls._hardware_instance = BudHardware()
        
        # Get all hardware from database
        db_hardware = cls._hardware_instance.get_all_hardwares()
        
        # Add to system_configs if not already present
        for hardware in db_hardware:
            if hardware['name'] not in system_configs:
                # Convert to GenZ format
                system_configs[hardware['name']] = {
                    'Flops': hardware['Flops'],
                    'Memory_size': hardware['Memory_size'],
                    'Memory_BW': hardware['Memory_BW'],
                    'ICN': hardware['ICN'],
                    'ICN_LL': hardware.get('ICN_LL'),
                    'real_values': hardware.get('real_values', True)
                }
        
        cls._initialized = True
    
    @classmethod
    def get_hardware_instance(cls) -> BudHardware:
        """Get the BudHardware instance, initializing if needed."""
        if not cls._initialized:
            cls.initialize()
        return cls._hardware_instance
    
    @classmethod
    def get_cpu_configs(cls) -> List[Dict[str, Any]]:
        """Get all CPU configurations from the database."""
        if not cls._initialized:
            cls.initialize()
        
        # Get all hardware and filter CPUs
        all_hardware = cls._hardware_instance.get_all_hardwares()
        return [hw for hw in all_hardware if hw.get('type') == 'cpu']
    
    @classmethod
    def get_hardware_by_type(cls, hw_type: str) -> List[Dict[str, Any]]:
        """Get hardware by type (cpu, gpu, accelerator, asic)."""
        if not cls._initialized:
            cls.initialize()
        
        return cls._hardware_instance.search_hardware(type=hw_type)
    
    @classmethod
    def refresh(cls):
        """Force refresh of hardware data."""
        cls._initialized = False
        cls.initialize() 