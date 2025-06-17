"""
Database connection manager with singleton pattern.
"""

import sqlite3
import os
import threading
from typing import Optional, Any, List, Tuple
from contextlib import contextmanager
import json
from pathlib import Path

from .schema import get_schema_statements, SCHEMA_VERSION


class DatabaseConnection:
    """Singleton database connection manager."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if hasattr(self, '_initialized'):
            return
            
        if db_path is None:
            # Default database location
            db_dir = Path.home() / '.genz_simulator' / 'db'
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / 'models.db')
            
        self.db_path = db_path
        self._conn = None
        self._initialized = True
        
        # Initialize database schema
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema if needed."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Execute schema creation statements
            for statement in get_schema_statements():
                cursor.executescript(statement)
            
            # Check and update schema version
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            result = cursor.fetchone()
            
            if result is None or result[0] < SCHEMA_VERSION:
                # Insert or update schema version
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
                conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> List[sqlite3.Row]:
        """Execute a query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """Execute a query multiple times with different parameters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
    
    def execute_one(self, query: str, params: Optional[Tuple] = None) -> Optional[sqlite3.Row]:
        """Execute a query and return the first result."""
        results = self.execute(query, params)
        return results[0] if results else None
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script (multiple statements)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(script)
    
    def insert(self, table: str, data: dict) -> int:
        """Insert a record and return the last inserted ID."""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid
    
    def update(self, table: str, data: dict, where: str, where_params: Tuple) -> int:
        """Update records and return the number of affected rows."""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + where_params
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
    
    def delete(self, table: str, where: str, where_params: Tuple) -> int:
        """Delete records and return the number of affected rows."""
        query = f"DELETE FROM {table} WHERE {where}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, where_params)
            return cursor.rowcount
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_one(query, (table_name,))
        return result is not None
    
    def get_db_info(self) -> dict:
        """Get information about the database."""
        info = {
            'db_path': self.db_path,
            'schema_version': SCHEMA_VERSION,
            'tables': []
        }
        
        # Get list of tables
        tables = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        info['tables'] = [row['name'] for row in tables]
        
        # Get row counts for each table
        table_counts = {}
        for table in info['tables']:
            count = self.execute_one(f"SELECT COUNT(*) as count FROM {table}")
            table_counts[table] = count['count'] if count else 0
        info['table_counts'] = table_counts
        
        return info
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
    
    def create_schema(self):
        """Create all database tables (useful for tests)."""
        for statement in get_schema_statements():
            self.execute_script(statement) 