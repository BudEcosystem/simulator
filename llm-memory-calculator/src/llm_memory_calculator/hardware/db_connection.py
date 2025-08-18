"""
Read-only database connection for hardware configurations.

This module provides a lightweight, read-only interface to hardware databases,
allowing external repositories to access extended hardware configurations.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ReadOnlyDatabaseConnection:
    """Read-only database connection for hardware configurations."""
    
    def __init__(self, db_path: str):
        """Initialize read-only database connection.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        
        # Verify database exists and is readable
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        # Test connection
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
        except sqlite3.Error as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get a read-only database connection context manager."""
        # Open in read-only mode using URI
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def query_hardware(self) -> List[Dict[str, Any]]:
        """Query all active hardware from database.
        
        Returns:
            List of hardware dictionaries.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        name,
                        manufacturer,
                        type,
                        flops,
                        memory_size,
                        memory_bw,
                        icn,
                        icn_ll,
                        power,
                        real_values,
                        url,
                        description,
                        source
                    FROM hardware
                    WHERE is_active = 1
                """)
                
                results = []
                for row in cursor.fetchall():
                    hw_dict = dict(row)
                    # Convert to standard format with uppercase keys for GenZ compatibility
                    hw_dict['Flops'] = hw_dict['flops']
                    hw_dict['Memory_size'] = hw_dict['memory_size']
                    hw_dict['Memory_BW'] = hw_dict['memory_bw']
                    hw_dict['ICN'] = hw_dict['icn']
                    if hw_dict.get('icn_ll'):
                        hw_dict['ICN_LL'] = hw_dict['icn_ll']
                    if hw_dict.get('power'):
                        hw_dict['Power'] = hw_dict['power']
                    results.append(hw_dict)
                
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Database query failed: {e}")
            return []
    
    def get_hardware_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific hardware by name from database.
        
        Args:
            name: Hardware name.
            
        Returns:
            Hardware dictionary if found, None otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        name,
                        manufacturer,
                        type,
                        flops,
                        memory_size,
                        memory_bw,
                        icn,
                        icn_ll,
                        power,
                        real_values,
                        url,
                        description,
                        source
                    FROM hardware
                    WHERE name = ? AND is_active = 1
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    hw_dict = dict(row)
                    # Convert to standard format
                    hw_dict['Flops'] = hw_dict['flops']
                    hw_dict['Memory_size'] = hw_dict['memory_size']
                    hw_dict['Memory_BW'] = hw_dict['memory_bw']
                    hw_dict['ICN'] = hw_dict['icn']
                    if hw_dict.get('icn_ll'):
                        hw_dict['ICN_LL'] = hw_dict['icn_ll']
                    if hw_dict.get('power'):
                        hw_dict['Power'] = hw_dict['power']
                    return hw_dict
                
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Database query failed: {e}")
            return None
    
    def search_hardware(self,
                       hw_type: Optional[str] = None,
                       manufacturer: Optional[str] = None,
                       min_flops: Optional[float] = None,
                       max_flops: Optional[float] = None,
                       min_memory: Optional[float] = None,
                       max_memory: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search hardware with filters.
        
        Args:
            hw_type: Filter by hardware type (gpu, cpu, asic, accelerator).
            manufacturer: Filter by manufacturer.
            min_flops: Minimum FLOPS requirement.
            max_flops: Maximum FLOPS limit.
            min_memory: Minimum memory requirement in GB.
            max_memory: Maximum memory limit in GB.
            
        Returns:
            List of matching hardware dictionaries.
        """
        try:
            query = """
                SELECT 
                    name,
                    manufacturer,
                    type,
                    flops,
                    memory_size,
                    memory_bw,
                    icn,
                    icn_ll,
                    power,
                    real_values,
                    url,
                    description,
                    source
                FROM hardware
                WHERE is_active = 1
            """
            
            conditions = []
            params = []
            
            if hw_type:
                conditions.append("type = ?")
                params.append(hw_type)
            
            if manufacturer:
                conditions.append("manufacturer = ?")
                params.append(manufacturer)
            
            if min_flops is not None:
                conditions.append("flops >= ?")
                params.append(min_flops)
            
            if max_flops is not None:
                conditions.append("flops <= ?")
                params.append(max_flops)
            
            if min_memory is not None:
                conditions.append("memory_size >= ?")
                params.append(min_memory)
            
            if max_memory is not None:
                conditions.append("memory_size <= ?")
                params.append(max_memory)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY name"
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    hw_dict = dict(row)
                    # Convert to standard format
                    hw_dict['Flops'] = hw_dict['flops']
                    hw_dict['Memory_size'] = hw_dict['memory_size']
                    hw_dict['Memory_BW'] = hw_dict['memory_bw']
                    hw_dict['ICN'] = hw_dict['icn']
                    if hw_dict.get('icn_ll'):
                        hw_dict['ICN_LL'] = hw_dict['icn_ll']
                    if hw_dict.get('power'):
                        hw_dict['Power'] = hw_dict['power']
                    results.append(hw_dict)
                
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check.
            
        Returns:
            True if table exists, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error:
            return False