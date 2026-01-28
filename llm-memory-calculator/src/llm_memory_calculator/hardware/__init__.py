"""
Hardware configuration management for LLM Memory Calculator.

This module provides read-only access to hardware configurations,
supporting both static configs and optional database loading.
It also includes device matching capabilities for cluster integration
and cost utility functions for pricing analysis.
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
from .cost_utils import (
    get_best_rate,
    has_cost_data,
    get_all_provider_rates,
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
    "get_best_rate",
    "has_cost_data",
    "get_all_provider_rates",
]