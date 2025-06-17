"""
Database schema for model management.
"""

SCHEMA_VERSION = 1

# SQL statements for creating tables
CREATE_TABLES = """
-- Models table to store model configurations
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,  -- e.g., "mistralai/Mistral-7B"
    model_name TEXT NOT NULL,       -- e.g., "Mistral-7B"
    source TEXT NOT NULL,           -- "huggingface" or "manual"
    config_json TEXT NOT NULL,      -- Full configuration as JSON
    model_type TEXT,                -- "decoder-only", "encoder-decoder", etc.
    attention_type TEXT,            -- "mha", "gqa", "mqa", "mla"
    parameter_count INTEGER,        -- Total number of parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    logo TEXT,                      -- URL to the model's logo
    model_analysis TEXT             -- JSON blob with analysis, advantages, etc.
);

-- Model cache table for computed values
CREATE TABLE IF NOT EXISTS model_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,        -- e.g., "memory_2048_fp16"
    cache_value TEXT NOT NULL,      -- JSON encoded value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
    UNIQUE(model_id, cache_key)
);

-- Model versions table for tracking updates
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    change_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Model quality metrics table
CREATE TABLE IF NOT EXISTS model_quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,      -- e.g., "MMLU", "GSM8K"
    metric_value REAL,              -- e.g., 0.625
    shots INTEGER,                  -- Number of shots used
    metadata TEXT,                  -- Additional metadata as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Schema version table for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_models_model_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_attention_type ON models(attention_type);
CREATE INDEX IF NOT EXISTS idx_models_parameter_count ON models(parameter_count);
CREATE INDEX IF NOT EXISTS idx_model_cache_expires ON model_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_model_metric ON model_quality_metrics(model_id, metric_name);
"""

# Trigger to update the updated_at timestamp
UPDATE_TIMESTAMP_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS update_models_timestamp 
AFTER UPDATE ON models
BEGIN
    UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""

def get_schema_statements():
    """Get all SQL statements needed to create the schema."""
    return [CREATE_TABLES, UPDATE_TIMESTAMP_TRIGGER] 