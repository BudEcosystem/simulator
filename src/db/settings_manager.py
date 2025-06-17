"""
Simple settings manager for LLM configuration.
"""

import json
from typing import Any, Dict, Optional
import logging

from .connection import DatabaseConnection


logger = logging.getLogger(__name__)


class LLMSettingsManager:
    """Manager class for LLM settings stored in database."""
    
    # Default settings
    DEFAULT_SETTINGS = {
        'llm_base_url': {
            'value': 'http://20.66.97.208/v1/',
            'type': 'string',
            'description': 'Base URL for the primary LLM API endpoint'
        },
        'llm_model': {
            'value': 'qwen3-32b',
            'type': 'string',
            'description': 'Default LLM model to use for API calls'
        }
    }
    
    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """Initialize settings manager.
        
        Args:
            db_connection: Database connection instance. If None, creates a new one.
        """
        self.db = db_connection or DatabaseConnection()
        self._cache = {}
        self._cache_loaded = False
        
        # Initialize default settings if needed
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default settings if they don't exist."""
        try:
            # Create table if not exists
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    setting_type TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add trigger for updated_at
            self.db.execute("""
                CREATE TRIGGER IF NOT EXISTS update_app_settings_timestamp 
                AFTER UPDATE ON app_settings
                BEGIN
                    UPDATE app_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
            """)
            
            # Initialize default settings
            for key, config in self.DEFAULT_SETTINGS.items():
                existing = self.db.execute_one(
                    "SELECT id FROM app_settings WHERE setting_key = ?",
                    (key,)
                )
                
                if not existing:
                    self.db.insert('app_settings', {
                        'setting_key': key,
                        'setting_value': config['value'],
                        'setting_type': config['type'],
                        'description': config['description']
                    })
                    logger.info(f"Initialized default setting: {key}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize default settings: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value
        """
        # Load cache if needed
        if not self._cache_loaded:
            self._load_cache()
        
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Query database
        result = self.db.execute_one(
            "SELECT setting_value FROM app_settings WHERE setting_key = ?",
            (key,)
        )
        
        if result:
            value = result['setting_value']
            self._cache[key] = value
            return value
            
        return default
    
    def set(self, key: str, value: str) -> bool:
        """Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if successful
        """
        # Check if setting exists
        existing = self.db.execute_one(
            "SELECT id FROM app_settings WHERE setting_key = ?",
            (key,)
        )
        
        if existing:
            # Update existing
            rows_affected = self.db.update(
                'app_settings',
                {'setting_value': value},
                'setting_key = ?',
                (key,)
            )
            success = rows_affected > 0
        else:
            # Insert new
            try:
                self.db.insert('app_settings', {
                    'setting_key': key,
                    'setting_value': value,
                    'setting_type': 'string',
                    'description': self.DEFAULT_SETTINGS.get(key, {}).get('description', '')
                })
                success = True
            except Exception as e:
                logger.error(f"Failed to insert setting {key}: {e}")
                success = False
        
        if success:
            # Update cache
            self._cache[key] = value
            logger.info(f"Set setting {key} = {value}")
            
        return success
    
    def _load_cache(self):
        """Load all settings into cache."""
        try:
            results = self.db.execute(
                "SELECT setting_key, setting_value FROM app_settings"
            )
            
            self._cache = {
                row['setting_key']: row['setting_value']
                for row in results
            }
            self._cache_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load settings cache: {e}")
    
    def clear_cache(self):
        """Clear the settings cache."""
        self._cache = {}
        self._cache_loaded = False 