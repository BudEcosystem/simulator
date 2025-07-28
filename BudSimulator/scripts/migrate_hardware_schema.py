#!/usr/bin/env python3
"""
Script to migrate hardware database to the new schema with instance types and vendor-specific pricing.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connection import DatabaseConnection
from src.db.hardware_schema import (
    get_hardware_schema_statements,
    get_migration_statements,
    HARDWARE_SCHEMA_VERSION
)


def check_current_schema_version(db_path: str) -> int:
    """Check the current schema version."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if schema version table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='hardware_schema_version'
        """)
        
        if not cursor.fetchone():
            # Check if hardware table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='hardware'
            """)
            
            if cursor.fetchone():
                # Old schema (version 1)
                return 1
            else:
                # No schema
                return 0
        
        # Get version
        cursor.execute(
            "SELECT version FROM hardware_schema_version ORDER BY version DESC LIMIT 1"
        )
        result = cursor.fetchone()
        return result[0] if result else 0
        
    finally:
        conn.close()


def backup_database(db_path: str) -> str:
    """Create a backup of the database."""
    backup_path = f"{db_path}.backup_{Path(db_path).stat().st_mtime_ns}"
    
    print(f"Creating backup at: {backup_path}")
    
    # Copy database
    import shutil
    shutil.copy2(db_path, backup_path)
    
    return backup_path


def migrate_database(db_path: str, from_version: int, to_version: int):
    """Migrate database schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        print(f"Migrating from version {from_version} to version {to_version}")
        
        if from_version == 0:
            # Fresh installation
            print("Creating new schema...")
            for statement in get_hardware_schema_statements():
                cursor.executescript(statement)
        
        elif from_version == 1 and to_version == 2:
            # Migrate from v1 to v2
            print("Migrating from schema v1 to v2...")
            
            # Create new tables
            print("Creating new tables...")
            cursor.executescript("""
                -- On-premise vendors with pricing
                CREATE TABLE IF NOT EXISTS hardware_on_prem_vendors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hardware_id INTEGER NOT NULL,
                    vendor_name TEXT NOT NULL,
                    price_lower REAL CHECK(price_lower IS NULL OR price_lower >= 0),
                    price_upper REAL CHECK(price_upper IS NULL OR price_upper >= 0),
                    currency TEXT DEFAULT 'USD',
                    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE,
                    CHECK(price_upper IS NULL OR price_lower IS NULL OR price_upper >= price_lower),
                    UNIQUE(hardware_id, vendor_name)
                );
                
                -- Cloud providers
                CREATE TABLE IF NOT EXISTS hardware_clouds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hardware_id INTEGER NOT NULL,
                    cloud_name TEXT NOT NULL,
                    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE,
                    UNIQUE(hardware_id, cloud_name)
                );
                
                -- Cloud instance types with pricing
                CREATE TABLE IF NOT EXISTS hardware_cloud_instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cloud_id INTEGER NOT NULL,
                    instance_name TEXT NOT NULL,
                    price_lower REAL CHECK(price_lower IS NULL OR price_lower >= 0),
                    price_upper REAL CHECK(price_upper IS NULL OR price_upper >= 0),
                    price_unit TEXT DEFAULT 'per_hour',
                    currency TEXT DEFAULT 'USD',
                    FOREIGN KEY (cloud_id) REFERENCES hardware_clouds(id) ON DELETE CASCADE,
                    CHECK(price_upper IS NULL OR price_lower IS NULL OR price_upper >= price_lower)
                );
                
                -- Migration table
                CREATE TABLE IF NOT EXISTS hardware_legacy_pricing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hardware_id INTEGER NOT NULL,
                    on_prem_price_lower REAL,
                    on_prem_price_upper REAL,
                    cloud_price_lower REAL,
                    cloud_price_upper REAL,
                    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE
                );
            """)
            
            # Migrate data
            print("Migrating vendor data...")
            cursor.execute("""
                INSERT INTO hardware_on_prem_vendors (hardware_id, vendor_name, price_lower, price_upper)
                SELECT DISTINCT hv.hardware_id, hv.vendor_name, h.on_prem_price_lower, h.on_prem_price_upper
                FROM hardware_vendors hv
                JOIN hardware h ON hv.hardware_id = h.id
                WHERE hv.vendor_type = 'on_prem'
            """)
            
            print("Migrating cloud data...")
            cursor.execute("""
                INSERT INTO hardware_clouds (hardware_id, cloud_name)
                SELECT DISTINCT hardware_id, cloud_name
                FROM hardware_cloud_pricing
            """)
            
            cursor.execute("""
                INSERT INTO hardware_cloud_instances (cloud_id, instance_name, price_lower, price_upper, price_unit, currency)
                SELECT hc.id, 'default', hcp.price_lower, hcp.price_upper, hcp.price_unit, hcp.currency
                FROM hardware_clouds hc
                JOIN hardware_cloud_pricing hcp ON hc.hardware_id = hcp.hardware_id AND hc.cloud_name = hcp.cloud_name
            """)
            
            # Update cloud regions
            print("Updating cloud regions...")
            cursor.execute("""
                ALTER TABLE hardware_cloud_regions ADD COLUMN cloud_id INTEGER
            """)
            
            cursor.execute("""
                UPDATE hardware_cloud_regions 
                SET cloud_id = (
                    SELECT hc.id 
                    FROM hardware_clouds hc 
                    WHERE hc.hardware_id = hardware_cloud_regions.hardware_id 
                    AND hc.cloud_name = hardware_cloud_regions.cloud_name
                )
            """)
            
            # Save legacy pricing
            print("Saving legacy pricing data...")
            cursor.execute("""
                INSERT INTO hardware_legacy_pricing (hardware_id, on_prem_price_lower, on_prem_price_upper, cloud_price_lower, cloud_price_upper)
                SELECT h.id, h.on_prem_price_lower, h.on_prem_price_upper, 
                       MIN(hcp.price_lower), MAX(hcp.price_upper)
                FROM hardware h
                LEFT JOIN hardware_cloud_pricing hcp ON h.id = hcp.hardware_id
                GROUP BY h.id
            """)
            
            # Create indexes
            print("Creating indexes...")
            cursor.executescript("""
                CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_hardware ON hardware_on_prem_vendors(hardware_id);
                CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_name ON hardware_on_prem_vendors(vendor_name);
                CREATE INDEX IF NOT EXISTS idx_clouds_hardware ON hardware_clouds(hardware_id);
                CREATE INDEX IF NOT EXISTS idx_cloud_instances_cloud ON hardware_cloud_instances(cloud_id);
                CREATE INDEX IF NOT EXISTS idx_cloud_regions_cloud ON hardware_cloud_regions(cloud_id);
            """)
        
        # Update schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute(
            "INSERT OR REPLACE INTO hardware_schema_version (version) VALUES (?)",
            (to_version,)
        )
        
        conn.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


def main():
    """Main function."""
    # Get database path
    db_path = os.environ.get('GENZ_DB_PATH')
    if not db_path:
        db_path = os.path.expanduser('~/.genz_simulator/db/models.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        print("Creating new database with latest schema...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create with latest schema
        migrate_database(db_path, 0, HARDWARE_SCHEMA_VERSION)
        return
    
    # Check current version
    current_version = check_current_schema_version(db_path)
    
    print(f"Current schema version: {current_version}")
    print(f"Target schema version: {HARDWARE_SCHEMA_VERSION}")
    
    if current_version >= HARDWARE_SCHEMA_VERSION:
        print("Database is already up to date!")
        return
    
    # Create backup
    backup_path = backup_database(db_path)
    
    try:
        # Perform migration
        migrate_database(db_path, current_version, HARDWARE_SCHEMA_VERSION)
        
        print(f"\nMigration successful!")
        print(f"Backup saved at: {backup_path}")
        print("\nNote: The old tables (hardware_vendors, hardware_cloud_pricing) are still present.")
        print("You can manually drop them after verifying the migration:")
        print("  DROP TABLE hardware_vendors;")
        print("  DROP TABLE hardware_cloud_pricing;")
        print("  ALTER TABLE hardware DROP COLUMN on_prem_price_lower;")
        print("  ALTER TABLE hardware DROP COLUMN on_prem_price_upper;")
        
    except Exception as e:
        print(f"\nMigration failed: {e}")
        print(f"Backup available at: {backup_path}")
        print("You can restore the backup if needed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 