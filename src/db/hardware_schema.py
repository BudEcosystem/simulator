"""
Database schema for hardware management.
"""

HARDWARE_SCHEMA_VERSION = 2

# SQL statements for creating hardware tables
CREATE_HARDWARE_TABLES = """
-- Main hardware table
CREATE TABLE IF NOT EXISTS hardware (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    manufacturer TEXT,
    type TEXT NOT NULL CHECK(type IN ('gpu', 'cpu', 'accelerator', 'asic')),
    flops REAL NOT NULL CHECK(flops > 0),
    memory_size REAL NOT NULL CHECK(memory_size > 0),  -- in GB
    memory_bw REAL NOT NULL CHECK(memory_bw > 0),      -- in GB/s
    icn REAL,                    -- interconnect bandwidth
    icn_ll REAL,                 -- interconnect latency (optional)
    power REAL CHECK(power IS NULL OR power > 0),      -- power consumption in watts
    real_values BOOLEAN DEFAULT 1,
    
    -- Additional metadata
    url TEXT,
    description TEXT,
    
    -- System fields
    source TEXT DEFAULT 'manual' CHECK(source IN ('system_config', 'json_import', 'manual', 'database')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Separate tables for normalized data
CREATE TABLE IF NOT EXISTS hardware_advantages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hardware_id INTEGER NOT NULL,
    advantage TEXT NOT NULL,
    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hardware_disadvantages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hardware_id INTEGER NOT NULL,
    disadvantage TEXT NOT NULL,
    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE
);

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

-- Cloud regions
CREATE TABLE IF NOT EXISTS hardware_cloud_regions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cloud_id INTEGER NOT NULL,
    region TEXT NOT NULL,
    FOREIGN KEY (cloud_id) REFERENCES hardware_clouds(id) ON DELETE CASCADE
);

-- Hardware schema version table
CREATE TABLE IF NOT EXISTS hardware_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Migration table to track old pricing data (for backward compatibility)
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
"""

# Create indexes for performance
CREATE_HARDWARE_INDEXES = """
-- Indexes on main hardware table
CREATE INDEX IF NOT EXISTS idx_hardware_type ON hardware(type);
CREATE INDEX IF NOT EXISTS idx_hardware_manufacturer ON hardware(manufacturer);
CREATE INDEX IF NOT EXISTS idx_hardware_flops ON hardware(flops);
CREATE INDEX IF NOT EXISTS idx_hardware_memory_size ON hardware(memory_size);
CREATE INDEX IF NOT EXISTS idx_hardware_memory_bw ON hardware(memory_bw);
CREATE INDEX IF NOT EXISTS idx_hardware_power ON hardware(power);
CREATE INDEX IF NOT EXISTS idx_hardware_source ON hardware(source);
CREATE INDEX IF NOT EXISTS idx_hardware_active ON hardware(is_active);

-- Indexes on vendor tables
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_hardware ON hardware_on_prem_vendors(hardware_id);
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_name ON hardware_on_prem_vendors(vendor_name);
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_price_lower ON hardware_on_prem_vendors(price_lower);

-- Indexes on cloud tables
CREATE INDEX IF NOT EXISTS idx_clouds_hardware ON hardware_clouds(hardware_id);
CREATE INDEX IF NOT EXISTS idx_clouds_name ON hardware_clouds(cloud_name);

CREATE INDEX IF NOT EXISTS idx_cloud_instances_cloud ON hardware_cloud_instances(cloud_id);
CREATE INDEX IF NOT EXISTS idx_cloud_instances_name ON hardware_cloud_instances(instance_name);
CREATE INDEX IF NOT EXISTS idx_cloud_instances_price_lower ON hardware_cloud_instances(price_lower);

CREATE INDEX IF NOT EXISTS idx_cloud_regions_cloud ON hardware_cloud_regions(cloud_id);
CREATE INDEX IF NOT EXISTS idx_cloud_regions_region ON hardware_cloud_regions(region);

-- Indexes on other tables
CREATE INDEX IF NOT EXISTS idx_advantages_hardware ON hardware_advantages(hardware_id);
CREATE INDEX IF NOT EXISTS idx_disadvantages_hardware ON hardware_disadvantages(hardware_id);
"""

# Trigger to update the updated_at timestamp
UPDATE_HARDWARE_TIMESTAMP_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS update_hardware_timestamp 
AFTER UPDATE ON hardware
BEGIN
    UPDATE hardware SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""

# Views for common queries
CREATE_HARDWARE_VIEWS = """
-- View for hardware with on-prem vendor information
CREATE VIEW IF NOT EXISTS hardware_with_on_prem_vendors AS
SELECT 
    h.*,
    GROUP_CONCAT(DISTINCT hopv.vendor_name) as on_prem_vendors,
    COUNT(DISTINCT hopv.id) as on_prem_vendor_count,
    MIN(hopv.price_lower) as min_on_prem_price,
    MAX(hopv.price_upper) as max_on_prem_price
FROM hardware h
LEFT JOIN hardware_on_prem_vendors hopv ON h.id = hopv.hardware_id
WHERE h.is_active = 1
GROUP BY h.id;

-- View for hardware with cloud availability
CREATE VIEW IF NOT EXISTS hardware_with_clouds AS
SELECT 
    h.*,
    GROUP_CONCAT(DISTINCT hc.cloud_name) as clouds,
    COUNT(DISTINCT hc.cloud_name) as cloud_count,
    MIN(hci.price_lower) as min_cloud_price,
    MAX(hci.price_upper) as max_cloud_price
FROM hardware h
LEFT JOIN hardware_clouds hc ON h.id = hc.hardware_id
LEFT JOIN hardware_cloud_instances hci ON hc.id = hci.cloud_id
WHERE h.is_active = 1
GROUP BY h.id;

-- View for cloud instance details
CREATE VIEW IF NOT EXISTS hardware_cloud_instance_details AS
SELECT 
    h.id as hardware_id,
    h.name as hardware_name,
    h.type as hardware_type,
    hc.cloud_name,
    hci.instance_name,
    hci.price_lower,
    hci.price_upper,
    hci.price_unit,
    GROUP_CONCAT(hcr.region) as regions
FROM hardware h
JOIN hardware_clouds hc ON h.id = hc.hardware_id
LEFT JOIN hardware_cloud_instances hci ON hc.id = hci.cloud_id
LEFT JOIN hardware_cloud_regions hcr ON hc.id = hcr.cloud_id
WHERE h.is_active = 1
GROUP BY h.id, hc.cloud_name, hci.instance_name;

-- View for performance metrics
CREATE VIEW IF NOT EXISTS hardware_performance_metrics AS
SELECT 
    h.*,
    CASE WHEN h.power > 0 THEN h.flops / h.power ELSE NULL END as perf_per_watt,
    CASE WHEN hopv.price_lower > 0 THEN h.flops / hopv.price_lower ELSE NULL END as perf_per_dollar,
    CASE WHEN h.memory_size > 0 THEN h.memory_bw / h.memory_size ELSE NULL END as bandwidth_per_gb
FROM hardware h
LEFT JOIN hardware_on_prem_vendors hopv ON h.id = hopv.hardware_id
WHERE h.is_active = 1;
"""

# Migration queries for schema updates
MIGRATE_FROM_V1_TO_V2 = """
-- Create new tables first
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

CREATE TABLE IF NOT EXISTS hardware_clouds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hardware_id INTEGER NOT NULL,
    cloud_name TEXT NOT NULL,
    FOREIGN KEY (hardware_id) REFERENCES hardware(id) ON DELETE CASCADE,
    UNIQUE(hardware_id, cloud_name)
);

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

-- Create new cloud regions table
CREATE TABLE IF NOT EXISTS hardware_cloud_regions_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cloud_id INTEGER NOT NULL,
    region TEXT NOT NULL,
    FOREIGN KEY (cloud_id) REFERENCES hardware_clouds(id) ON DELETE CASCADE
);

-- Migrate existing vendor data
INSERT OR IGNORE INTO hardware_on_prem_vendors (hardware_id, vendor_name, price_lower, price_upper)
SELECT DISTINCT hv.hardware_id, hv.vendor_name, h.on_prem_price_lower, h.on_prem_price_upper
FROM hardware_vendors hv
JOIN hardware h ON hv.hardware_id = h.id
WHERE hv.vendor_type = 'on_prem';

-- Migrate existing cloud data
INSERT OR IGNORE INTO hardware_clouds (hardware_id, cloud_name)
SELECT DISTINCT hardware_id, cloud_name
FROM hardware_cloud_pricing;

-- Migrate cloud pricing to instance level (assuming single instance per cloud for now)
INSERT OR IGNORE INTO hardware_cloud_instances (cloud_id, instance_name, price_lower, price_upper, price_unit, currency)
SELECT hc.id, 'default', hcp.price_lower, hcp.price_upper, hcp.price_unit, hcp.currency
FROM hardware_clouds hc
JOIN hardware_cloud_pricing hcp ON hc.hardware_id = hcp.hardware_id AND hc.cloud_name = hcp.cloud_name;

-- Migrate regions to new table
INSERT OR IGNORE INTO hardware_cloud_regions_new (cloud_id, region)
SELECT DISTINCT hc.id, hcr.region
FROM hardware_clouds hc
JOIN hardware_cloud_regions hcr ON hc.hardware_id = hcr.hardware_id AND hc.cloud_name = hcr.cloud_name;

-- Drop old regions table and rename new one
DROP TABLE IF EXISTS hardware_cloud_regions;
ALTER TABLE hardware_cloud_regions_new RENAME TO hardware_cloud_regions;

-- Save legacy pricing data
INSERT OR IGNORE INTO hardware_legacy_pricing (hardware_id, on_prem_price_lower, on_prem_price_upper, cloud_price_lower, cloud_price_upper)
SELECT h.id, h.on_prem_price_lower, h.on_prem_price_upper, 
       MIN(hcp.price_lower), MAX(hcp.price_upper)
FROM hardware h
LEFT JOIN hardware_cloud_pricing hcp ON h.id = hcp.hardware_id
GROUP BY h.id;

-- Create indexes for new tables
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_hardware ON hardware_on_prem_vendors(hardware_id);
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_name ON hardware_on_prem_vendors(vendor_name);
CREATE INDEX IF NOT EXISTS idx_on_prem_vendors_price_lower ON hardware_on_prem_vendors(price_lower);

CREATE INDEX IF NOT EXISTS idx_clouds_hardware ON hardware_clouds(hardware_id);
CREATE INDEX IF NOT EXISTS idx_clouds_name ON hardware_clouds(cloud_name);

CREATE INDEX IF NOT EXISTS idx_cloud_instances_cloud ON hardware_cloud_instances(cloud_id);
CREATE INDEX IF NOT EXISTS idx_cloud_instances_name ON hardware_cloud_instances(instance_name);
CREATE INDEX IF NOT EXISTS idx_cloud_instances_price_lower ON hardware_cloud_instances(price_lower);

CREATE INDEX IF NOT EXISTS idx_cloud_regions_cloud ON hardware_cloud_regions(cloud_id);
CREATE INDEX IF NOT EXISTS idx_cloud_regions_region ON hardware_cloud_regions(region);

-- Note: The following cleanup commands should be run manually after verification:
-- DROP TABLE hardware_vendors;
-- DROP TABLE hardware_cloud_pricing;
-- ALTER TABLE hardware DROP COLUMN on_prem_price_lower;
-- ALTER TABLE hardware DROP COLUMN on_prem_price_upper;
"""

def get_hardware_schema_statements():
    """Get all SQL statements needed to create the hardware schema."""
    return [
        CREATE_HARDWARE_TABLES,
        CREATE_HARDWARE_INDEXES,
        UPDATE_HARDWARE_TIMESTAMP_TRIGGER,
        CREATE_HARDWARE_VIEWS
    ]

def get_migration_statements(from_version, to_version):
    """Get migration statements for schema updates."""
    if from_version == 1 and to_version == 2:
        return MIGRATE_FROM_V1_TO_V2
    return None 