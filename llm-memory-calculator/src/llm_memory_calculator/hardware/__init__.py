"""
Hardware configuration management for LLM Memory Calculator.

This module provides read-only access to hardware configurations,
supporting both static configs and optional database loading.
It also includes device matching capabilities for cluster integration.
"""

from .configs import HARDWARE_CONFIGS
from .manager import (
    HardwareManager,
    get_hardware_config,
    get_all_hardware,
    get_hardware_by_type,
    get_hardware_by_manufacturer,
    search_hardware,
    set_hardware_db_path,
)
from .device_matcher import (
    DeviceIdentity,
    DeviceParser,
    DeviceMatcher,
    match_device,
)

__all__ = [
    "HardwareManager",
    "HARDWARE_CONFIGS",
    "get_hardware_config",
    "get_all_hardware",
    "get_hardware_by_type",
    "get_hardware_by_manufacturer",
    "search_hardware",
    "set_hardware_db_path",
    "DeviceIdentity",
    "DeviceParser",
    "DeviceMatcher",
    "match_device",
]