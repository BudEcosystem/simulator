"""
BudHardware class for unified hardware management.
Supports both static system_configs and dynamic database-stored hardware.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import sqlite3

from .db.connection import DatabaseConnection
from .db.hardware_schema import (
    get_hardware_schema_statements, 
    get_migration_statements,
    HARDWARE_SCHEMA_VERSION
)
from Systems.system_configs import system_configs


logger = logging.getLogger(__name__)


class BudHardware:
    """
    Unified hardware management system that:
    - Loads hardware from system_configs.py (static)
    - Loads hardware from database (dynamic)
    - Provides CRUD operations for hardware
    - Maintains compatibility with GenZ hardware structure
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize BudHardware with database connection.
        
        Args:
            db_path: Optional path to database. If None, uses default location.
        """
        # Initialize database connection
        self.db = DatabaseConnection(db_path)
        
        # Initialize hardware schema
        self._initialize_hardware_schema()
        
        # Load system configs into memory (read-only)
        self._system_configs = system_configs.copy()
        
        # Cache for merged hardware data
        self._hardware_cache = {}
        self._cache_valid = False
        
        # Load initial data
        self._ensure_system_configs_in_db()
    
    def _initialize_hardware_schema(self):
        """Initialize hardware database schema."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check current schema version
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='hardware_schema_version'
            """)
            
            current_version = 0
            if cursor.fetchone():
                cursor.execute(
                    "SELECT version FROM hardware_schema_version ORDER BY version DESC LIMIT 1"
                )
                result = cursor.fetchone()
                if result:
                    current_version = result[0]
            
            # Apply schema updates if needed
            if current_version < HARDWARE_SCHEMA_VERSION:
                # Create new schema
                for statement in get_hardware_schema_statements():
                    cursor.executescript(statement)
                
                # Apply migrations if upgrading
                if current_version > 0:
                    migration_sql = get_migration_statements(current_version, HARDWARE_SCHEMA_VERSION)
                    if migration_sql:
                        cursor.executescript(migration_sql)
                
                # Update schema version
                cursor.execute(
                    "INSERT OR REPLACE INTO hardware_schema_version (version) VALUES (?)",
                    (HARDWARE_SCHEMA_VERSION,)
                )
                conn.commit()
    
    def _ensure_system_configs_in_db(self):
        """Ensure system_configs hardware exists in database for metadata storage."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for hw_name, hw_config in self._system_configs.items():
                # Check if already exists
                cursor.execute(
                    "SELECT id FROM hardware WHERE name = ? AND source = 'system_config'",
                    (hw_name,)
                )
                
                if not cursor.fetchone():
                    # Add minimal record for system_config hardware
                    cursor.execute("""
                        INSERT INTO hardware (
                            name, type, flops, memory_size, memory_bw, 
                            icn, icn_ll, real_values, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'system_config')
                    """, (
                        hw_name,
                        self._detect_hardware_type(hw_name),
                        hw_config['Flops'],
                        hw_config['Memory_size'],
                        hw_config['Memory_BW'],
                        hw_config['ICN'],
                        hw_config.get('ICN_LL'),
                        hw_config.get('real_values', True)
                    ))
            
            conn.commit()
    
    def _detect_hardware_type(self, name: str) -> str:
        """Detect hardware type from name."""
        name_lower = name.lower()
        if 'cpu' in name_lower:
            return 'cpu'
        elif 'gpu' in name_lower:
            return 'gpu'
        elif 'tpu' in name_lower:
            return 'asic'
        else:
            return 'accelerator'
    
    def get_all_hardwares(self) -> List[Dict[str, Any]]:
        """Get all hardware from both system_configs and database.
        
        Returns:
            List of hardware dictionaries with unified format.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return list(self._hardware_cache.values())
    
    def _refresh_cache(self):
        """Refresh the hardware cache by merging system_configs and database."""
        self._hardware_cache = {}
        
        # First, add all system_configs hardware
        for hw_name, hw_config in self._system_configs.items():
            self._hardware_cache[hw_name] = {
                'name': hw_name,
                'Flops': hw_config['Flops'],
                'Memory_size': hw_config['Memory_size'],
                'Memory_BW': hw_config['Memory_BW'],
                'ICN': hw_config['ICN'],
                'ICN_LL': hw_config.get('ICN_LL'),
                'real_values': hw_config.get('real_values', True),
                'source': 'system_config'
            }
        
        # Then, overlay database hardware (which may include metadata for system_configs)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT h.*, 
                       GROUP_CONCAT(DISTINCT hopv.vendor_name) as on_prem_vendors,
                       GROUP_CONCAT(DISTINCT hc.cloud_name) as clouds,
                       MIN(hopv.price_lower) as min_on_prem_price,
                       MAX(hopv.price_upper) as max_on_prem_price
                FROM hardware h
                LEFT JOIN hardware_on_prem_vendors hopv ON h.id = hopv.hardware_id
                LEFT JOIN hardware_clouds hc ON h.id = hc.hardware_id
                WHERE h.is_active = 1
                GROUP BY h.id
            """)
            
            for row in cursor.fetchall():
                hw_data = dict(row)
                hw_name = hw_data['name']
                
                # Convert database format to GenZ format
                formatted_hw = {
                    'name': hw_name,
                    'type': hw_data['type'],
                    'manufacturer': hw_data['manufacturer'],
                    'flops': hw_data['flops'],
                    'memory_size': hw_data['memory_size'],
                    'memory_bw': hw_data['memory_bw'],
                    'icn': hw_data['icn'],
                    'icn_ll': hw_data['icn_ll'],
                    'power': hw_data['power'],
                    'real_values': bool(hw_data['real_values']),
                    'url': hw_data['url'],
                    'description': hw_data['description'],
                    'on_prem_vendors': hw_data['on_prem_vendors'].split(',') if hw_data['on_prem_vendors'] else [],
                    'clouds': hw_data['clouds'].split(',') if hw_data['clouds'] else [],
                    'min_on_prem_price': hw_data['min_on_prem_price'],
                    'max_on_prem_price': hw_data['max_on_prem_price'],
                    'source': hw_data['source']
                }
                
                # For GenZ compatibility, also include uppercase keys
                formatted_hw['Flops'] = formatted_hw['flops']
                formatted_hw['Memory_size'] = formatted_hw['memory_size']
                formatted_hw['Memory_BW'] = formatted_hw['memory_bw']
                formatted_hw['ICN'] = formatted_hw['icn']
                formatted_hw['ICN_LL'] = formatted_hw['icn_ll']
                
                # For system_config hardware, the database values take precedence
                # This allows updates to system_config hardware to be reflected
                # while the original system_configs dictionary remains unchanged
                
                self._hardware_cache[hw_name] = formatted_hw
        
        self._cache_valid = True
    
    def get_hardware_by_name(self, name: str) -> Union[Dict[str, Any], bool]:
        """Get specific hardware by name.
        
        Args:
            name: Hardware name (case-sensitive)
            
        Returns:
            Hardware dictionary if found, False otherwise.
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return self._hardware_cache.get(name, False)
    
    def get_config(self, system_name: str) -> Dict[str, Any]:
        """Get hardware config in GenZ-compatible format.
        
        Args:
            system_name: Hardware system name
            
        Returns:
            Configuration dictionary with GenZ-required fields.
        """
        hw = self.get_hardware_by_name(system_name)
        if not hw:
            return None
        
        # Return GenZ-compatible format
        return {
            'Flops': hw['Flops'],
            'Memory_size': hw['Memory_size'],
            'Memory_BW': hw['Memory_BW'],
            'ICN': hw['ICN'],
            'ICN_LL': hw.get('ICN_LL'),
            'real_values': hw.get('real_values', True)
        }
    
    def format_memory_size(self, mb: float) -> str:
        """Format memory size with appropriate units.
        
        Args:
            mb: Memory size in MB
            
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
    
    def add_hardware(self, hardware_data: Dict[str, Any]) -> bool:
        """Add new hardware to the database.
        
        Args:
            hardware_data: Dictionary containing hardware specifications
            
        Returns:
            True if successful, raises ValueError if validation fails.
        """
        # Validate hardware data
        self._validate_hardware(hardware_data)
        
        # Check if already exists
        if self.get_hardware_by_name(hardware_data['name']):
            raise ValueError(f"Hardware {hardware_data['name']} already exists")
        
        # Insert into database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert main hardware record
            cursor.execute("""
                INSERT INTO hardware (
                    name, manufacturer, type, flops, memory_size, memory_bw,
                    icn, icn_ll, power, real_values, url, description, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hardware_data['name'],
                hardware_data.get('manufacturer'),
                hardware_data['type'],
                hardware_data['flops'],
                hardware_data['memory_size'],
                hardware_data['memory_bw'],
                hardware_data.get('icn'),
                hardware_data.get('icn_ll'),
                hardware_data.get('power'),
                hardware_data.get('real_values', True),
                hardware_data.get('url'),
                hardware_data.get('description'),
                hardware_data.get('source', 'manual')
            ))
            
            hardware_id = cursor.lastrowid
            
            # Add on-prem vendors with pricing
            if 'on_prem_vendors' in hardware_data:
                for vendor_info in hardware_data['on_prem_vendors']:
                    if isinstance(vendor_info, dict):
                        cursor.execute("""
                            INSERT INTO hardware_on_prem_vendors 
                            (hardware_id, vendor_name, price_lower, price_upper)
                            VALUES (?, ?, ?, ?)
                        """, (
                            hardware_id,
                            vendor_info['name'],
                            vendor_info.get('price_lower'),
                            vendor_info.get('price_upper')
                        ))
                    else:
                        # Simple vendor name without pricing
                        cursor.execute("""
                            INSERT INTO hardware_on_prem_vendors 
                            (hardware_id, vendor_name)
                            VALUES (?, ?)
                        """, (hardware_id, vendor_info))
            
            # Add cloud support with instance types
            if 'clouds' in hardware_data:
                for cloud_info in hardware_data['clouds']:
                    # Insert cloud
                    cursor.execute("""
                        INSERT INTO hardware_clouds (hardware_id, cloud_name)
                        VALUES (?, ?)
                    """, (hardware_id, cloud_info['name']))
                    
                    cloud_id = cursor.lastrowid
                    
                    # Add regions
                    if 'regions' in cloud_info:
                        for region in cloud_info['regions']:
                            cursor.execute("""
                                INSERT INTO hardware_cloud_regions (cloud_id, region)
                                VALUES (?, ?)
                            """, (cloud_id, region))
                    
                    # Add instance types with pricing
                    if 'instances' in cloud_info:
                        for instance in cloud_info['instances']:
                            cursor.execute("""
                                INSERT INTO hardware_cloud_instances 
                                (cloud_id, instance_name, price_lower, price_upper)
                                VALUES (?, ?, ?, ?)
                            """, (
                                cloud_id,
                                instance['name'],
                                instance.get('price_lower'),
                                instance.get('price_upper')
                            ))
            
            conn.commit()
        
        # Invalidate cache
        self._cache_valid = False
        
        return True
    
    def update_hardware(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update existing hardware.
        
        Args:
            name: Hardware name
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if hardware not found.
        """
        # Check if hardware exists
        hw = self.get_hardware_by_name(name)
        if not hw:
            return False
        
        # For system_config hardware, updates only affect the database copy
        # The original system_configs dictionary remains unchanged
        
        # Validate updates
        if 'type' in updates and updates['type'] not in ['gpu', 'cpu', 'accelerator', 'asic']:
            raise ValueError(f"Invalid hardware type: {updates['type']}")
        
        # Build update query
        update_fields = []
        update_values = []
        
        field_mapping = {
            'flops': 'flops',
            'memory_size': 'memory_size',
            'memory_bw': 'memory_bw',
            'icn': 'icn',
            'icn_ll': 'icn_ll',
            'power': 'power',
            'type': 'type',
            'manufacturer': 'manufacturer',
            'url': 'url',
            'description': 'description'
        }
        
        for key, value in updates.items():
            if key in field_mapping:
                update_fields.append(f"{field_mapping[key]} = ?")
                update_values.append(value)
        
        if not update_fields:
            return True  # Nothing to update
        
        update_values.append(name)  # For WHERE clause
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE hardware SET {', '.join(update_fields)} WHERE name = ?"
            cursor.execute(query, update_values)
            conn.commit()
        
        # Invalidate cache
        self._cache_valid = False
        
        return True
    
    def delete_hardware(self, name: str, hard_delete: bool = False) -> bool:
        """Delete hardware (soft delete by default).
        
        Args:
            name: Hardware name
            hard_delete: If True, permanently delete
            
        Returns:
            True if successful, False if not found.
        """
        # Don't allow deleting system_configs
        if name in self._system_configs:
            return False
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if hard_delete:
                cursor.execute("DELETE FROM hardware WHERE name = ?", (name,))
            else:
                cursor.execute(
                    "UPDATE hardware SET is_active = 0 WHERE name = ? AND source != 'system_config'",
                    (name,)
                )
            
            if cursor.rowcount > 0:
                conn.commit()
                self._cache_valid = False
                return True
        
        return False
    
    def _validate_hardware(self, hardware_data: Dict[str, Any]):
        """Validate hardware data before insertion/update.
        
        Args:
            hardware_data: Hardware data to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Required fields
        required_fields = ['name', 'type', 'flops', 'memory_size', 'memory_bw']
        for field in required_fields:
            if field not in hardware_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Type validation
        valid_types = ['gpu', 'cpu', 'accelerator', 'asic']
        if hardware_data['type'] not in valid_types:
            raise ValueError(f"Invalid hardware type: {hardware_data['type']}")
        
        # Numeric validation
        numeric_fields = ['flops', 'memory_size', 'memory_bw']
        for field in numeric_fields:
            if field in hardware_data and hardware_data[field] <= 0:
                raise ValueError(f"{field} must be positive")
        
        # Optional numeric fields
        optional_numeric = ['icn', 'power']
        for field in optional_numeric:
            if field in hardware_data and hardware_data[field] is not None and hardware_data[field] < 0:
                raise ValueError(f"{field} cannot be negative")
    
    def search_hardware(self,
                       query: Optional[str] = None,
                       type: Optional[Union[str, List[str]]] = None,
                       manufacturer: Optional[Union[str, List[str]]] = None,
                       min_flops: Optional[float] = None,
                       max_flops: Optional[float] = None,
                       min_memory: Optional[float] = None,
                       max_memory: Optional[float] = None,
                       min_memory_bw: Optional[float] = None,
                       max_memory_bw: Optional[float] = None,
                       min_power: Optional[float] = None,
                       max_power: Optional[float] = None,
                       min_price: Optional[float] = None,
                       max_price: Optional[float] = None,
                       vendor: Optional[Union[str, List[str]]] = None,
                       cloud: Optional[Union[str, List[str]]] = None,
                       region: Optional[str] = None,
                       has_real_values: Optional[bool] = None,
                       sort_by: str = 'name',
                       sort_order: str = 'asc',
                       limit: Optional[int] = None,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters and sorting options."""
        
        # Build query
        base_query = """
            SELECT DISTINCT h.*,
                   GROUP_CONCAT(DISTINCT hopv.vendor_name) as vendors,
                   GROUP_CONCAT(DISTINCT hc.cloud_name) as clouds
            FROM hardware h
            LEFT JOIN hardware_on_prem_vendors hopv ON h.id = hopv.hardware_id
            LEFT JOIN hardware_clouds hc ON h.id = hc.hardware_id
            LEFT JOIN hardware_cloud_regions hcr ON hc.id = hcr.cloud_id
            WHERE h.is_active = 1
        """
        
        conditions = []
        params = []
        
        # Type filter
        if type:
            if isinstance(type, str):
                conditions.append("h.type = ?")
                params.append(type)
            else:
                placeholders = ','.join(['?' for _ in type])
                conditions.append(f"h.type IN ({placeholders})")
                params.extend(type)
        
        # Manufacturer filter
        if manufacturer:
            if isinstance(manufacturer, str):
                conditions.append("h.manufacturer = ?")
                params.append(manufacturer)
            else:
                placeholders = ','.join(['?' for _ in manufacturer])
                conditions.append(f"h.manufacturer IN ({placeholders})")
                params.extend(manufacturer)
        
        # Performance filters
        if min_flops is not None:
            conditions.append("h.flops >= ?")
            params.append(min_flops)
        if max_flops is not None:
            conditions.append("h.flops <= ?")
            params.append(max_flops)
        
        if min_memory is not None:
            conditions.append("h.memory_size >= ?")
            params.append(min_memory)
        if max_memory is not None:
            conditions.append("h.memory_size <= ?")
            params.append(max_memory)
        
        if min_memory_bw is not None:
            conditions.append("h.memory_bw >= ?")
            params.append(min_memory_bw)
        if max_memory_bw is not None:
            conditions.append("h.memory_bw <= ?")
            params.append(max_memory_bw)
        
        if min_power is not None:
            conditions.append("h.power >= ?")
            params.append(min_power)
        if max_power is not None:
            conditions.append("h.power <= ?")
            params.append(max_power)
        
        # Price filters (now check vendor pricing)
        if min_price is not None:
            conditions.append("hopv.price_lower >= ?")
            params.append(min_price)
        if max_price is not None:
            conditions.append("hopv.price_upper <= ?")
            params.append(max_price)
        
        # Add conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        # Group by
        base_query += " GROUP BY h.id"
        
        # Sorting
        sort_mapping = {
            'name': 'h.name',
            'flops': 'h.flops',
            'memory_size': 'h.memory_size',
            'memory_bw': 'h.memory_bw',
            'power': 'h.power',
            'price': 'MIN(hopv.price_lower)',
            'perf_per_watt': 'CAST(h.flops AS REAL) / NULLIF(h.power, 0)',
            'perf_per_dollar': 'CAST(h.flops AS REAL) / NULLIF(MIN(hopv.price_lower), 0)'
        }
        
        sort_column = sort_mapping.get(sort_by, 'h.name')
        sort_direction = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
        base_query += f" ORDER BY {sort_column} {sort_direction}"
        
        # Pagination
        if limit:
            base_query += f" LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                hw_data = dict(row)
                # Convert to standard format
                formatted = {
                    'name': hw_data['name'],
                    'type': hw_data['type'],
                    'manufacturer': hw_data['manufacturer'],
                    'flops': hw_data['flops'],
                    'memory_size': hw_data['memory_size'],
                    'memory_bw': hw_data['memory_bw'],
                    'icn': hw_data['icn'],
                    'power': hw_data['power'],
                    'vendors': hw_data['vendors'].split(',') if hw_data['vendors'] else [],
                    'clouds': hw_data['clouds'].split(',') if hw_data['clouds'] else []
                }
                results.append(formatted)
        
        return results
    
    def get_hardware_vendors(self, hardware_name: str) -> List[Dict[str, Any]]:
        """Get list of vendors with pricing for a specific hardware."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT hopv.vendor_name, hopv.price_lower, hopv.price_upper
                FROM hardware_on_prem_vendors hopv
                JOIN hardware h ON hopv.hardware_id = h.id
                WHERE h.name = ?
            """, (hardware_name,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_hardware_clouds(self, hardware_name: str) -> List[Dict[str, Any]]:
        """Get cloud availability with instance types for a specific hardware."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT hc.cloud_name, hci.instance_name, 
                       hci.price_lower, hci.price_upper,
                       GROUP_CONCAT(hcr.region) as regions
                FROM hardware h
                JOIN hardware_clouds hc ON h.id = hc.hardware_id
                LEFT JOIN hardware_cloud_instances hci ON hc.id = hci.cloud_id
                LEFT JOIN hardware_cloud_regions hcr ON hc.id = hcr.cloud_id
                WHERE h.name = ?
                GROUP BY hc.cloud_name, hci.instance_name
            """, (hardware_name,))
            
            results = []
            for row in cursor.fetchall():
                cloud_data = dict(row)
                cloud_data['regions'] = cloud_data['regions'].split(',') if cloud_data['regions'] else []
                results.append(cloud_data)
            
            return results
    
    def import_from_json(self, filepath: str) -> Dict[str, Any]:
        """Import hardware from JSON file (hardware.json format)."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            imported = 0
            failed = 0
            errors = []
            
            for hw_key, hw_data in data.items():
                try:
                    # Skip hardware with zero memory
                    if hw_data.get('Memory_size', 0) <= 0:
                        failed += 1
                        errors.append(f"{hw_key}: Memory size must be positive (got {hw_data.get('Memory_size', 0)})")
                        continue
                    
                    # Convert from hardware.json format to our format
                    hardware_info = {
                        'name': hw_key,
                        'type': hw_data.get('type', self._detect_hardware_type(hw_key)),
                        'manufacturer': hw_data.get('manufacturer'),
                        'flops': hw_data['Flops'],
                        'memory_size': hw_data['Memory_size'],
                        'memory_bw': hw_data['Memory_BW'],
                        'icn': hw_data.get('ICN'),
                        'icn_ll': hw_data.get('ICN_LL'),
                        'power': hw_data.get('Power'),
                        'real_values': hw_data.get('real_values', True),
                        'url': hw_data.get('url'),
                        'description': hw_data.get('description'),
                        'source': 'json_import'
                    }
                    
                    # Add base hardware
                    self.add_hardware(hardware_info)
                    
                    # Get hardware ID for related data
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM hardware WHERE name = ?", (hw_key,))
                        hw_id = cursor.fetchone()['id']
                        
                        # Add on-prem vendors with pricing
                        if 'on_prem_vendors' in hw_data and hw_data['on_prem_vendors']:
                            # Get pricing info
                            price_lower = None
                            price_upper = None
                            if 'on_prem_price' in hw_data:
                                lower = hw_data['on_prem_price'].get('lower_bound')
                                upper = hw_data['on_prem_price'].get('upper_bound')
                                
                                # Convert string prices to numbers or None
                                if isinstance(lower, str):
                                    price_lower = float(lower) if lower.replace('.', '').isdigit() else None
                                else:
                                    price_lower = lower
                                    
                                if isinstance(upper, str):
                                    price_upper = float(upper) if upper.replace('.', '').isdigit() else None
                                else:
                                    price_upper = upper
                            
                            for vendor in hw_data['on_prem_vendors']:
                                if vendor:  # Skip empty/null vendors
                                    cursor.execute("""
                                        INSERT INTO hardware_on_prem_vendors 
                                        (hardware_id, vendor_name, price_lower, price_upper) 
                                        VALUES (?, ?, ?, ?)
                                    """, (hw_id, vendor, price_lower, price_upper))
                        
                        # Add cloud support with instance types
                        if 'supported_clouds' in hw_data:
                            # Get cloud pricing info
                            cloud_price_lower = None
                            cloud_price_upper = None
                            if 'cloud_pricing' in hw_data:
                                lower = hw_data['cloud_pricing'].get('lower_bound')
                                upper = hw_data['cloud_pricing'].get('upper_bound')
                                
                                # Convert string prices to numbers or None
                                if isinstance(lower, str):
                                    cloud_price_lower = float(lower) if lower.replace('.', '').isdigit() else None
                                else:
                                    cloud_price_lower = lower
                                    
                                if isinstance(upper, str):
                                    cloud_price_upper = float(upper) if upper.replace('.', '').isdigit() else None
                                else:
                                    cloud_price_upper = upper
                            
                            for cloud in hw_data['supported_clouds']:
                                cloud_name = cloud.get('cloud_name')
                                if cloud_name:
                                    # Insert cloud
                                    cursor.execute("""
                                        INSERT INTO hardware_clouds (hardware_id, cloud_name)
                                        VALUES (?, ?)
                                    """, (hw_id, cloud_name))
                                    
                                    cloud_id = cursor.lastrowid
                                    
                                    # Add regions
                                    regions = cloud.get('regions', [])
                                    for region in regions:
                                        cursor.execute("""
                                            INSERT INTO hardware_cloud_regions (cloud_id, region)
                                            VALUES (?, ?)
                                        """, (cloud_id, region))
                                    
                                    # Add instance types with pricing
                                    instance_names = cloud.get('instance_name', [])
                                    if instance_names:
                                        for instance in instance_names:
                                            cursor.execute("""
                                                INSERT INTO hardware_cloud_instances 
                                                (cloud_id, instance_name, price_lower, price_upper)
                                                VALUES (?, ?, ?, ?)
                                            """, (cloud_id, instance, cloud_price_lower, cloud_price_upper))
                                    else:
                                        # Add default instance if no specific instances
                                        cursor.execute("""
                                            INSERT INTO hardware_cloud_instances 
                                            (cloud_id, instance_name, price_lower, price_upper)
                                            VALUES (?, ?, ?, ?)
                                        """, (cloud_id, 'default', cloud_price_lower, cloud_price_upper))
                        
                        # Add advantages
                        if 'advantages' in hw_data:
                            for advantage in hw_data['advantages']:
                                cursor.execute(
                                    "INSERT INTO hardware_advantages (hardware_id, advantage) VALUES (?, ?)",
                                    (hw_id, advantage)
                                )
                        
                        # Add disadvantages
                        if 'disadvantages' in hw_data:
                            for disadvantage in hw_data['disadvantages']:
                                cursor.execute(
                                    "INSERT INTO hardware_disadvantages (hardware_id, disadvantage) VALUES (?, ?)",
                                    (hw_id, disadvantage)
                                )
                        
                        conn.commit()
                    
                    imported += 1
                    
                except Exception as e:
                    failed += 1
                    errors.append(f"{hw_key}: {str(e)}")
            
            return {
                'success': failed == 0,
                'imported': imported,
                'failed': failed,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'imported': 0,
                'failed': 0,
                'errors': [str(e)]
            }
    
    def export_to_json(self, filepath: str) -> bool:
        """Export all hardware to JSON file."""
        try:
            all_hardware = self.get_all_hardwares()
            
            export_data = {}
            for hw in all_hardware:
                # Convert to hardware.json format
                hw_export = {
                    'Flops': hw['Flops'],
                    'Memory_size': hw['Memory_size'],
                    'Memory_BW': hw['Memory_BW'],
                    'ICN': hw['ICN'],
                    'real_values': hw.get('real_values', True)
                }
                
                # Add optional fields
                if hw.get('ICN_LL'):
                    hw_export['ICN_LL'] = hw['ICN_LL']
                if hw.get('power'):
                    hw_export['Power'] = hw['power']
                if hw.get('manufacturer'):
                    hw_export['manufacturer'] = hw['manufacturer']
                if hw.get('type'):
                    hw_export['type'] = hw['type']
                if hw.get('url'):
                    hw_export['url'] = hw['url']
                if hw.get('description'):
                    hw_export['description'] = hw['description']
                
                # Get vendor and cloud details
                vendors = self.get_hardware_vendors(hw['name'])
                clouds = self.get_hardware_clouds(hw['name'])
                
                # Add on-prem vendors
                if vendors:
                    hw_export['on_prem_vendors'] = [v['vendor_name'] for v in vendors]
                    # Get price range from vendors
                    price_lowers = [v['price_lower'] for v in vendors if v['price_lower'] is not None]
                    price_uppers = [v['price_upper'] for v in vendors if v['price_upper'] is not None]
                    if price_lowers or price_uppers:
                        hw_export['on_prem_price'] = {
                            'lower_bound': min(price_lowers) if price_lowers else None,
                            'upper_bound': max(price_uppers) if price_uppers else None
                        }
                
                # Add cloud support
                if clouds:
                    # Group by cloud name
                    cloud_dict = {}
                    for cloud in clouds:
                        cloud_name = cloud['cloud_name']
                        if cloud_name not in cloud_dict:
                            cloud_dict[cloud_name] = {
                                'cloud_name': cloud_name,
                                'regions': cloud['regions'],
                                'instance_name': []
                            }
                        if cloud['instance_name'] and cloud['instance_name'] != 'default':
                            cloud_dict[cloud_name]['instance_name'].append(cloud['instance_name'])
                    
                    hw_export['supported_clouds'] = list(cloud_dict.values())
                    
                    # Get cloud price range
                    cloud_price_lowers = [c['price_lower'] for c in clouds if c['price_lower'] is not None]
                    cloud_price_uppers = [c['price_upper'] for c in clouds if c['price_upper'] is not None]
                    if cloud_price_lowers or cloud_price_uppers:
                        hw_export['cloud_pricing'] = {
                            'lower_bound': min(cloud_price_lowers) if cloud_price_lowers else None,
                            'upper_bound': max(cloud_price_uppers) if cloud_price_uppers else None
                        }
                
                export_data[hw['name']] = hw_export
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export hardware: {e}")
            return False

    def get_recommendations(
        self, 
        required_memory_gb: float,
        model_config: Dict[str, Any],
        usecase_config: Dict[str, Any],
        batch_size: int,
        include_pricing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get hardware recommendations that meet memory and performance SLOs.
        """
        all_hardware = self.get_all_hardwares()
        
        recommendations = []
        
        for hw in all_hardware:
            memory_per_chip = hw.get('Memory_size', 0)
            if memory_per_chip <= 0:
                continue
            
            nodes_required = max(1, int((required_memory_gb + memory_per_chip - 1) // memory_per_chip))
            total_memory = nodes_required * memory_per_chip
            utilization = (required_memory_gb / total_memory) * 100
            
            if utilization < 10:
                continue

            # Estimate performance on this hardware
            # Note: This is a simplified estimation. A real implementation would be more complex.
            flops = hw.get('Flops', 0)
            if flops <= 0:
                continue # Cannot estimate performance without FLOPS

            # Simplified performance model
            param_count = model_config.get('num_parameters', 0)
            base_ttft = 0.1 * (param_count / 7e9) # Scale with model size
            compute_time_per_token = (param_count * 2) / (flops * 1e12) # Simplified ops per token
            
            estimated_ttft = base_ttft + (batch_size * compute_time_per_token)
            estimated_e2e = estimated_ttft + (usecase_config.get('output_tokens_max', 100) * compute_time_per_token)

            # Check if it meets SLOs
            ttft_ok = usecase_config.get('ttft_max') is None or estimated_ttft <= usecase_config['ttft_max']
            e2e_ok = usecase_config.get('e2e_max') is None or estimated_e2e <= usecase_config['e2e_max']

            if not (ttft_ok and e2e_ok):
                continue # Skip hardware that doesn't meet performance SLOs

            price_per_hour = None
            if include_pricing:
                price_per_hour = self.calculate_price_indicator(
                    flops=flops,
                    memory_gb=memory_per_chip,
                    bandwidth_gbs=hw.get('Memory_BW', 0)
                )

            recommendations.append({
                'hardware_name': hw['name'],
                'nodes_required': nodes_required,
                'memory_per_chip': memory_per_chip,
                'utilization': round(utilization, 1),
                'price_per_hour': price_per_hour,
                'total_cost_per_hour': price_per_hour * nodes_required if price_per_hour else None,
                'optimality': 'optimal' if 80 <= utilization <= 95 else 'good'
            })
            
        # Sort by fewest nodes, then highest utilization
        recommendations.sort(key=lambda x: (x['nodes_required'], -x['utilization']))
        
        return recommendations[:5]

    @staticmethod
    def calculate_price_indicator(flops: float, memory_gb: float, bandwidth_gbs: float) -> float:
        """
        Calculate approximate price indicator for hardware comparison.
        Note: This is NOT actual pricing, only for relative comparison.
        
        Based on a simplified model correlating hardware specifications
        to relative cost indicators. The formula uses logarithmic scaling
        to account for the non-linear relationship between performance
        and cost in the hardware market.
        
        Args:
            flops: Floating point operations per second (FLOPS)
            memory_gb: Memory size in GB
            bandwidth_gbs: Memory bandwidth in GB/s
            
        Returns:
            Approximate price indicator (relative scale, not actual dollars)
            Returns 0.0 for invalid inputs.
            
        Example:
            >>> BudHardware.calculate_price_indicator(1000, 80, 2000)
            45.67  # Relative price indicator
        """
        import math
        
        if flops <= 0 or memory_gb <= 0 or bandwidth_gbs <= 0:
            return 0.0
            
        try:
            # Logarithmic pricing model based on hardware specification scaling
            # Formula: log10(price) = -3.46 - 0.05*log10(flops) + 0.49*log10(memory) + 1.03*log10(bandwidth)
            # This creates a relative scale where higher-spec hardware gets higher indicators
            log_price = (-3.46
                         - 0.05 * math.log10(flops)
                         + 0.49 * math.log10(memory_gb)
                         + 1.03 * math.log10(bandwidth_gbs))
            
            price_indicator = 10 ** log_price
            
            # Round to 2 decimal places for clean display
            return round(price_indicator, 2)
            
        except (ValueError, OverflowError, ZeroDivisionError):
            # Handle edge cases (negative values, overflow, etc.)
            return 0.0 