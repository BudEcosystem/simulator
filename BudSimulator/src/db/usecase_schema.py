"""
Database schema for usecase management.
"""

USECASE_SCHEMA_VERSION = 1

# SQL statements for creating usecase tables
CREATE_USECASE_TABLES = """
-- Main usecases table
CREATE TABLE IF NOT EXISTS usecases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_id TEXT UNIQUE NOT NULL,       -- e.g., "uc_001", "usecase_013"
    name TEXT NOT NULL,                   -- e.g., "Question Answering"
    industry TEXT NOT NULL,               -- e.g., "Healthcare", "Finance"
    description TEXT,                     -- Optional detailed description
    
    -- Configuration parameters
    batch_size INTEGER NOT NULL DEFAULT 1,
    beam_size INTEGER NOT NULL DEFAULT 1,
    
    -- Token ranges
    input_tokens_min INTEGER NOT NULL CHECK(input_tokens_min >= 0),
    input_tokens_max INTEGER NOT NULL CHECK(input_tokens_max >= input_tokens_min),
    output_tokens_min INTEGER NOT NULL CHECK(output_tokens_min >= 0),
    output_tokens_max INTEGER NOT NULL CHECK(output_tokens_max >= output_tokens_min),
    
    -- Performance SLOs (Service Level Objectives)
    ttft_min REAL CHECK(ttft_min >= 0),           -- Time to First Token min (seconds)
    ttft_max REAL CHECK(ttft_max >= ttft_min),    -- Time to First Token max (seconds)
    e2e_min REAL CHECK(e2e_min >= 0),             -- End-to-end latency min (seconds)
    e2e_max REAL CHECK(e2e_max >= e2e_min),       -- End-to-end latency max (seconds)
    inter_token_min REAL CHECK(inter_token_min >= 0),    -- Inter-token latency min (seconds)
    inter_token_max REAL CHECK(inter_token_max >= inter_token_min),  -- Inter-token latency max (seconds)
    
    -- System fields
    source TEXT DEFAULT 'manual' CHECK(source IN ('json_import', 'manual', 'api')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Usecase tags for categorization
CREATE TABLE IF NOT EXISTS usecase_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usecase_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (usecase_id) REFERENCES usecases(id) ON DELETE CASCADE,
    UNIQUE(usecase_id, tag)
);

-- Usecase versions table for tracking updates
CREATE TABLE IF NOT EXISTS usecase_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usecase_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    config_json TEXT NOT NULL,      -- Full configuration as JSON
    change_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (usecase_id) REFERENCES usecases(unique_id) ON DELETE CASCADE
);

-- Usecase cache table for computed values
CREATE TABLE IF NOT EXISTS usecase_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usecase_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,        -- e.g., "hardware_recommendations_a100"
    cache_value TEXT NOT NULL,      -- JSON encoded value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (usecase_id) REFERENCES usecases(unique_id) ON DELETE CASCADE,
    UNIQUE(usecase_id, cache_key)
);

-- Schema version table
CREATE TABLE IF NOT EXISTS usecase_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_usecases_name ON usecases(name);
CREATE INDEX IF NOT EXISTS idx_usecases_industry ON usecases(industry);
CREATE INDEX IF NOT EXISTS idx_usecases_unique_id ON usecases(unique_id);
CREATE INDEX IF NOT EXISTS idx_usecases_active ON usecases(is_active);
CREATE INDEX IF NOT EXISTS idx_usecases_source ON usecases(source);

-- Indexes for token ranges (useful for filtering)
CREATE INDEX IF NOT EXISTS idx_usecases_input_tokens ON usecases(input_tokens_min, input_tokens_max);
CREATE INDEX IF NOT EXISTS idx_usecases_output_tokens ON usecases(output_tokens_min, output_tokens_max);

-- Indexes for performance requirements
CREATE INDEX IF NOT EXISTS idx_usecases_ttft ON usecases(ttft_min, ttft_max);
CREATE INDEX IF NOT EXISTS idx_usecases_e2e ON usecases(e2e_min, e2e_max);

-- Index for tags
CREATE INDEX IF NOT EXISTS idx_usecase_tags_tag ON usecase_tags(tag);

-- Index for cache expiration
CREATE INDEX IF NOT EXISTS idx_usecase_cache_expires ON usecase_cache(expires_at);
"""

# Trigger to update the updated_at timestamp
UPDATE_USECASE_TIMESTAMP_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS update_usecases_timestamp 
AFTER UPDATE ON usecases
BEGIN
    UPDATE usecases SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""

# Views for common queries
CREATE_USECASE_VIEWS = """
-- View for usecases with tag aggregation
CREATE VIEW IF NOT EXISTS usecases_with_tags AS
SELECT 
    u.*,
    GROUP_CONCAT(ut.tag) as tags
FROM usecases u
LEFT JOIN usecase_tags ut ON u.id = ut.usecase_id
WHERE u.is_active = 1
GROUP BY u.id;

-- View for usecase performance profiles
CREATE VIEW IF NOT EXISTS usecase_performance_profiles AS
SELECT 
    unique_id,
    name,
    industry,
    input_tokens_max as max_input_tokens,
    output_tokens_max as max_output_tokens,
    ttft_max as max_ttft,
    e2e_max as max_e2e,
    inter_token_max as max_inter_token,
    CASE 
        WHEN e2e_max <= 2 THEN 'real-time'
        WHEN e2e_max <= 5 THEN 'interactive'
        WHEN e2e_max <= 10 THEN 'responsive'
        ELSE 'batch'
    END as latency_profile,
    CASE
        WHEN input_tokens_max <= 1000 THEN 'short'
        WHEN input_tokens_max <= 10000 THEN 'medium'
        ELSE 'long'
    END as input_length_profile
FROM usecases
WHERE is_active = 1;

-- View for industry-specific usecases
CREATE VIEW IF NOT EXISTS usecases_by_industry AS
SELECT 
    industry,
    COUNT(*) as usecase_count,
    AVG(input_tokens_max) as avg_max_input_tokens,
    AVG(output_tokens_max) as avg_max_output_tokens,
    AVG(e2e_max) as avg_max_e2e_latency
FROM usecases
WHERE is_active = 1
GROUP BY industry;
"""

def get_usecase_schema_statements():
    """Get all SQL statements needed to create the usecase schema."""
    return [
        CREATE_USECASE_TABLES,
        UPDATE_USECASE_TIMESTAMP_TRIGGER,
        CREATE_USECASE_VIEWS
    ]

def get_migration_statements(from_version, to_version):
    """Get migration statements for schema updates."""
    # Placeholder for future migrations
    return None 