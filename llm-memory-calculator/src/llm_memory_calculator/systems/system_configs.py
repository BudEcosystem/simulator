"""
Backward compatibility module for system_configs.

This module provides the system_configs dictionary for backward compatibility
with existing code that imports from this location.
"""

from typing import Any, Dict
from llm_memory_calculator.hardware import HardwareManager

# Create default hardware manager
_hw_manager = HardwareManager()

# Get all hardware as a dictionary for backward compatibility
system_configs: Dict[str, Dict[str, Any]] = _hw_manager.get_all_hardware_dict()


def set_hardware_db_path(db_path: str):
    """Set the database path for hardware loading.
    
    This allows dynamic loading of hardware from a database while maintaining
    backward compatibility with the system_configs dictionary.
    
    Args:
        db_path: Path to hardware database.
    """
    global _hw_manager, system_configs
    _hw_manager = HardwareManager(db_path)
    system_configs = _hw_manager.get_all_hardware_dict()