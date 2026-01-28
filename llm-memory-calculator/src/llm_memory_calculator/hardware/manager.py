"""
Hardware manager for LLM Memory Calculator.

This module provides the main HardwareManager class for read-only access
to hardware configurations from both static configs and optional database.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from functools import lru_cache

from .configs import HARDWARE_CONFIGS
from .db_connection import ReadOnlyDatabaseConnection
from .device_matcher import match_device, DeviceParser, DeviceMatcher
from .cost_utils import get_best_rate, has_cost_data

logger = logging.getLogger(__name__)

# Global instance for module-level functions
_default_manager = None


class HardwareManager:
    """
    Read-only hardware management system that:
    - Loads hardware from static configs
    - Optionally loads hardware from database (read-only)
    - Provides search and retrieval operations
    - Maintains compatibility with GenZ hardware structure
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize HardwareManager with optional database.
        
        Args:
            db_path: Optional path to database. If None, uses only static configs.
        """
        # Load static configs
        self._static_configs = HARDWARE_CONFIGS.copy()
        
        # Optional database connection
        self._db = None
        if db_path:
            try:
                self._db = ReadOnlyDatabaseConnection(db_path)
                logger.info(f"Connected to hardware database at {db_path}")
            except (FileNotFoundError, ConnectionError) as e:
                logger.warning(f"Failed to connect to database: {e}. Using static configs only.")
        
        # Cache for merged hardware data
        self._hardware_cache = {}
        self._alias_map = {}  # alias (lowercase) -> canonical hardware name
        self._cache_valid = False

        # Cache for device matching (cluster_info_hash -> hardware_config)
        self._match_cache = {}
    
    def _refresh_cache(self):
        """Refresh the hardware cache by merging static configs and database."""
        self._hardware_cache = {}
        self._alias_map = {}

        # First, add all static configs
        for hw_name, hw_config in self._static_configs.items():
            config = hw_config.copy()
            # Ensure 'name' field exists
            if 'name' not in config:
                config['name'] = hw_name
            self._hardware_cache[hw_name] = config

            # Build alias map (case-insensitive)
            for alias in config.get('aliases', []):
                self._alias_map[alias.lower()] = hw_name

        # Then, overlay database hardware if available
        if self._db:
            try:
                db_hardware = self._db.query_hardware()
                for hw in db_hardware:
                    hw_name = hw['name']
                    # Database hardware overwrites static if names match
                    self._hardware_cache[hw_name] = hw
                    # Add aliases from database hardware
                    for alias in hw.get('aliases', []):
                        self._alias_map[alias.lower()] = hw_name
            except Exception as e:
                logger.error(f"Failed to load hardware from database: {e}")

        self._cache_valid = True
    
    def get_hardware_config(self, name: str) -> Union[Dict[str, Any], bool]:
        """Get specific hardware configuration by name or alias.

        Args:
            name: Hardware name or alias (case-sensitive for exact match,
                  case-insensitive for alias lookup).

        Returns:
            Hardware dictionary if found, False otherwise (for GenZ compatibility).
        """
        if not self._cache_valid:
            self._refresh_cache()

        # Direct lookup first (case-sensitive)
        config = self._hardware_cache.get(name)
        if config:
            return config

        # Try alias lookup (case-insensitive)
        canonical_name = self._alias_map.get(name.lower())
        if canonical_name:
            return self._hardware_cache.get(canonical_name)

        return False
    
    def get_all_hardware(self) -> List[Dict[str, Any]]:
        """Get all hardware from both static configs and database.
        
        Returns:
            List of hardware dictionaries with unified format.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return list(self._hardware_cache.values())
    
    def get_all_hardware_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get all hardware as a dictionary (for system_configs compatibility).
        
        Returns:
            Dictionary mapping hardware names to configurations.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return self._hardware_cache.copy()
    
    def get_hardware_by_type(self, hw_type: str) -> List[Dict[str, Any]]:
        """Get all hardware of a specific type.
        
        Args:
            hw_type: Hardware type (gpu, cpu, asic, accelerator).
            
        Returns:
            List of matching hardware configurations.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return [
            hw for hw in self._hardware_cache.values()
            if hw.get('type', '').lower() == hw_type.lower()
        ]
    
    def get_hardware_by_manufacturer(self, manufacturer: str) -> List[Dict[str, Any]]:
        """Get all hardware from a specific manufacturer.
        
        Args:
            manufacturer: Manufacturer name.
            
        Returns:
            List of matching hardware configurations.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return [
            hw for hw in self._hardware_cache.values()
            if hw.get('manufacturer', '').lower() == manufacturer.lower()
        ]
    
    def search_hardware(self,
                       query: Optional[str] = None,
                       hw_type: Optional[Union[str, List[str]]] = None,
                       manufacturer: Optional[Union[str, List[str]]] = None,
                       min_flops: Optional[float] = None,
                       max_flops: Optional[float] = None,
                       min_memory: Optional[float] = None,
                       max_memory: Optional[float] = None,
                       min_memory_bw: Optional[float] = None,
                       max_memory_bw: Optional[float] = None,
                       has_real_values: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters.
        
        Args:
            query: Text search in name and description.
            hw_type: Filter by type(s).
            manufacturer: Filter by manufacturer(s).
            min_flops: Minimum FLOPS.
            max_flops: Maximum FLOPS.
            min_memory: Minimum memory in GB.
            max_memory: Maximum memory in GB.
            min_memory_bw: Minimum memory bandwidth in GB/s.
            max_memory_bw: Maximum memory bandwidth in GB/s.
            has_real_values: Filter by real_values flag.
            
        Returns:
            List of matching hardware configurations.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        results = []
        
        # Convert single values to lists for uniform processing
        if isinstance(hw_type, str):
            hw_type = [hw_type]
        if isinstance(manufacturer, str):
            manufacturer = [manufacturer]
        
        for hw in self._hardware_cache.values():
            # Text search
            if query:
                query_lower = query.lower()
                name_match = query_lower in hw.get('name', '').lower()
                desc_match = query_lower in hw.get('description', '').lower()
                if not (name_match or desc_match):
                    continue
            
            # Type filter
            if hw_type and hw.get('type', '').lower() not in [t.lower() for t in hw_type]:
                continue
            
            # Manufacturer filter
            if manufacturer and hw.get('manufacturer', '').lower() not in [m.lower() for m in manufacturer]:
                continue
            
            # Performance filters
            hw_flops = hw.get('Flops', hw.get('flops', 0))
            if min_flops is not None and hw_flops < min_flops:
                continue
            if max_flops is not None and hw_flops > max_flops:
                continue
            
            hw_memory = hw.get('Memory_size', hw.get('memory_size', 0))
            if min_memory is not None and hw_memory < min_memory:
                continue
            if max_memory is not None and hw_memory > max_memory:
                continue
            
            hw_memory_bw = hw.get('Memory_BW', hw.get('memory_bw', 0))
            if min_memory_bw is not None and hw_memory_bw < min_memory_bw:
                continue
            if max_memory_bw is not None and hw_memory_bw > max_memory_bw:
                continue
            
            # Real values filter
            if has_real_values is not None:
                hw_real = hw.get('real_values', True)
                if hw_real != has_real_values:
                    continue
            
            results.append(hw)
        
        return results
    
    def get_config(self, system_name: str) -> Optional[Dict[str, Any]]:
        """Get hardware config in GenZ-compatible format.
        
        Args:
            system_name: Hardware system name.
            
        Returns:
            Configuration dictionary with GenZ-required fields, or None if not found.
        """
        hw = self.get_hardware_config(system_name)
        if not hw:
            return None
        
        # Return GenZ-compatible format
        return {
            'Flops': hw.get('Flops', hw.get('flops', 0)),
            'Memory_size': hw.get('Memory_size', hw.get('memory_size', 0)),
            'Memory_BW': hw.get('Memory_BW', hw.get('memory_bw', 0)),
            'ICN': hw.get('ICN', hw.get('icn', 0)),
            'ICN_LL': hw.get('ICN_LL', hw.get('icn_ll')),
            'real_values': hw.get('real_values', True)
        }
    
    def format_memory_size(self, mb: float) -> str:
        """Format memory size with appropriate units.
        
        Args:
            mb: Memory size in MB.
            
        Returns:
            Formatted string with appropriate unit.
        """
        units = ["MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        size = mb
        unit_index = 0
        
        while size >= 1000 and unit_index < len(units) - 1:
            size /= 1000
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}" if size != int(size) else f"{int(size)} {units[unit_index]}"
    
    def match_cluster_device(self, device_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Match cluster-reported device information to hardware configuration.
        
        This method attempts to match device information from a cluster
        (e.g., from NFD/node-info) to a known hardware configuration using
        various identifiers like PCI IDs, device names, and memory sizes.
        
        Args:
            device_info: Device information from cluster containing:
                - raw_name: Full device name string (e.g., "NVIDIA A100-SXM4-40GB")
                - pci_vendor: PCI vendor ID (e.g., "10de" for NVIDIA)
                - pci_device: PCI device ID (e.g., "20f1" for A100)
                - memory_mb: Memory size in MB (e.g., 40960)
                - gpu_count: Number of GPUs (optional)
                
        Returns:
            Matched hardware configuration dictionary with performance specs,
            or None if no match found.
            
        Example:
            >>> device_info = {
            ...     "raw_name": "NVIDIA A100-SXM4-40GB",
            ...     "pci_vendor": "10de",
            ...     "pci_device": "20f1",
            ...     "memory_mb": 40960
            ... }
            >>> config = manager.match_cluster_device(device_info)
            >>> print(config['name'], config['Flops'])
            A100_40GB_GPU 312
        """
        # Create cache key from device info
        cache_key = (
            device_info.get('pci_vendor', ''),
            device_info.get('pci_device', ''),
            device_info.get('memory_mb', 0),
            device_info.get('raw_name', '')
        )
        
        # Check cache first
        if cache_key in self._match_cache:
            logger.debug(f"Found cached match for device: {cache_key}")
            return self._match_cache[cache_key]
        
        # Ensure hardware cache is loaded
        if not self._cache_valid:
            self._refresh_cache()
        
        # Use device matcher to find configuration
        matched_config = match_device(device_info, self._hardware_cache)
        
        # Cache the result (even if None to avoid repeated lookups)
        self._match_cache[cache_key] = matched_config
        
        if matched_config:
            logger.info(f"Matched cluster device to: {matched_config.get('name')}")
        else:
            logger.warning(f"No match found for cluster device: {device_info}")
        
        return matched_config
    
    def match_cluster_devices_batch(self, devices: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Match multiple cluster devices in batch.
        
        Args:
            devices: List of device information dictionaries from cluster.
            
        Returns:
            List of matched hardware configurations (None for unmatched devices).
        """
        results = []
        for device_info in devices:
            results.append(self.match_cluster_device(device_info))
        return results
    
    def get_cluster_hardware_specs(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get hardware specifications for a cluster device.

        This method matches a cluster device and returns the complete
        performance specifications needed for LLM calculations, including
        cost data when available.

        Args:
            device_info: Device information from cluster.

        Returns:
            Dictionary containing:
                - device_name: Matched configuration name
                - flops_fp16: FP16 FLOPS (in TFLOPS)
                - memory_size_gb: Memory size in GB
                - memory_bandwidth_gbs: Memory bandwidth in GB/s
                - interconnect_bandwidth_gbs: Interconnect bandwidth in GB/s
                - matched: Boolean indicating if match was successful
                - cost: Full cost dictionary (or None if unavailable)
                - has_cost_data: Boolean indicating if cost data is available
                - best_hourly_rate: Best available hourly rate (or None)
                - best_provider: Provider with best rate (or None)
                - best_tier: Pricing tier for best rate (or None)
                - tdp_watts: Power consumption in watts (or None)
                - purchase_price_usd: Purchase price in USD (or None)

        Example:
            >>> specs = manager.get_cluster_hardware_specs(device_info)
            >>> if specs['matched']:
            ...     print(f"Device: {specs['device_name']}")
            ...     print(f"FP16 Performance: {specs['flops_fp16']} TFLOPS")
            >>> if specs['has_cost_data']:
            ...     print(f"Best rate: ${specs['best_hourly_rate']}/hr from {specs['best_provider']}")
        """
        matched_config = self.match_cluster_device(device_info)

        if matched_config:
            cost_data = matched_config.get('cost')
            best_rate, best_provider, best_tier = get_best_rate(cost_data)

            return {
                # Existing performance fields
                'device_name': matched_config.get('name'),
                'flops_fp16': matched_config.get('Flops', matched_config.get('flops', 0)),
                'memory_size_gb': matched_config.get('Memory_size', matched_config.get('memory_size', 0)),
                'memory_bandwidth_gbs': matched_config.get('Memory_BW', matched_config.get('memory_bw', 0)),
                'interconnect_bandwidth_gbs': matched_config.get('ICN', matched_config.get('icn', 0)),
                'real_values': matched_config.get('real_values', True),
                'matched': True,

                # New cost fields
                'cost': cost_data,
                'has_cost_data': has_cost_data(matched_config),
                'best_hourly_rate': best_rate,
                'best_provider': best_provider,
                'best_tier': best_tier,
                'tdp_watts': cost_data.get('tdp_watts') if cost_data else None,
                'purchase_price_usd': cost_data.get('purchase_price_usd') if cost_data else None,
            }
        else:
            # Return empty specs for unmatched device
            return {
                'device_name': device_info.get('raw_name', 'Unknown'),
                'flops_fp16': 0,
                'memory_size_gb': device_info.get('memory_mb', 0) / 1024 if 'memory_mb' in device_info else 0,
                'memory_bandwidth_gbs': 0,
                'interconnect_bandwidth_gbs': 0,
                'real_values': False,
                'matched': False,
                'cost': None,
                'has_cost_data': False,
                'best_hourly_rate': None,
                'best_provider': None,
                'best_tier': None,
                'tdp_watts': None,
                'purchase_price_usd': None,
            }
    
    def clear_match_cache(self):
        """Clear the device matching cache."""
        self._match_cache = {}
        logger.debug("Cleared device matching cache")


# Module-level functions for convenience
def get_hardware_config(name: str) -> Union[Dict[str, Any], bool]:
    """Get specific hardware configuration by name.
    
    Args:
        name: Hardware name.
        
    Returns:
        Hardware dictionary if found, False otherwise.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HardwareManager()
    return _default_manager.get_hardware_config(name)


def get_all_hardware() -> List[Dict[str, Any]]:
    """Get all available hardware configurations.
    
    Returns:
        List of hardware dictionaries.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HardwareManager()
    return _default_manager.get_all_hardware()


def get_hardware_by_type(hw_type: str) -> List[Dict[str, Any]]:
    """Get all hardware of a specific type.
    
    Args:
        hw_type: Hardware type (gpu, cpu, asic, accelerator).
        
    Returns:
        List of matching hardware configurations.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HardwareManager()
    return _default_manager.get_hardware_by_type(hw_type)


def get_hardware_by_manufacturer(manufacturer: str) -> List[Dict[str, Any]]:
    """Get all hardware from a specific manufacturer.
    
    Args:
        manufacturer: Manufacturer name.
        
    Returns:
        List of matching hardware configurations.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HardwareManager()
    return _default_manager.get_hardware_by_manufacturer(manufacturer)


def search_hardware(**kwargs) -> List[Dict[str, Any]]:
    """Search hardware with multiple filters.
    
    Args:
        **kwargs: Search filters (see HardwareManager.search_hardware).
        
    Returns:
        List of matching hardware configurations.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HardwareManager()
    return _default_manager.search_hardware(**kwargs)


def set_hardware_db_path(db_path: str):
    """Set the database path for hardware loading.
    
    Args:
        db_path: Path to hardware database.
    """
    global _default_manager
    _default_manager = HardwareManager(db_path)