"""
Model manager for CRUD operations on models in the database.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from .connection import DatabaseConnection


logger = logging.getLogger(__name__)

# Import ModelMemoryCalculator for parameter calculation
try:
    from ..bud_models import ModelMemoryCalculator
    _calculator = None
    
    def get_calculator():
        global _calculator
        if _calculator is None:
            _calculator = ModelMemoryCalculator()
        return _calculator
except ImportError:
    logger.warning("Could not import ModelMemoryCalculator")
    def get_calculator():
        return None


class ModelManager:
    """Manager class for model CRUD operations."""
    
    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """Initialize model manager.
        
        Args:
            db_connection: Database connection instance. If None, creates a new one.
        """
        self.db = db_connection or DatabaseConnection()
    
    def add_model(self, model_id: str, config: Dict[str, Any], 
                  source: str = "huggingface", model_type: Optional[str] = None,
                  attention_type: Optional[str] = None, 
                  parameter_count: Optional[int] = None,
                  logo: Optional[str] = None,
                  model_analysis: Optional[Dict[str, Any]] = None) -> int:
        """Add a new model to the database.
        
        Args:
            model_id: Unique model identifier (e.g., "mistralai/Mistral-7B")
            config: Model configuration dictionary
            source: Model source ("huggingface" or "manual")
            model_type: Type of model (e.g., "decoder-only")
            attention_type: Attention mechanism type (e.g., "gqa")
            parameter_count: Total number of parameters
            logo: URL to the model's logo
            model_analysis: Detailed analysis of the model as a dictionary
            
        Returns:
            ID of the inserted model
        """
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        data = {
            'model_id': model_id,
            'model_name': model_name,
            'source': source,
            'config_json': json.dumps(config),
            'model_type': model_type,
            'attention_type': attention_type,
            'parameter_count': parameter_count,
            'logo': logo,
            'model_analysis': json.dumps(model_analysis) if model_analysis else None
        }
        
        try:
            model_db_id = self.db.insert('models', data)
            
            # Add initial version
            self._add_model_version(model_id, 1, config, "Initial version")
            
            logger.info(f"Added model {model_id} to database")
            return model_db_id
            
        except Exception as e:
            logger.error(f"Failed to add model {model_id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by its ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model data dictionary or None if not found
        """
        query = "SELECT * FROM models WHERE model_id = ? AND is_active = 1"
        result = self.db.execute_one(query, (model_id,))
        
        if result:
            return self._row_to_dict(result)
        return None
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get just the configuration of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration dictionary or None if not found
        """
        config = None
        
        # First check user-provided configs for gated models
        user_config = self.get_user_model_config(model_id)
        if user_config:
            config = user_config.copy()
        else:
            # Then check regular models table
            model = self.get_model(model_id)
            if model and model.get('config_json'):
                config = json.loads(model['config_json'])
                if model.get('model_analysis'):
                    config['model_analysis'] = json.loads(model['model_analysis'])
                if model.get('logo'):
                    config['logo'] = model['logo']
        
        if config is None:
            return None
            
        # Calculate missing parameter count if needed
        if config.get('num_parameters') is None:
            calculator = get_calculator()
            if calculator:
                try:
                    # Check if we have enough architectural data to calculate parameters
                    required_fields = ['hidden_size', 'num_hidden_layers', 'num_attention_heads']
                    if all(field in config for field in required_fields):
                        param_count = calculator._calculate_transformer_params(config)
                        if param_count > 0:
                            config['num_parameters'] = param_count
                            logger.debug(f"Calculated parameter count for {model_id}: {param_count:,}")
                except Exception as e:
                    logger.warning(f"Failed to calculate parameter count for {model_id}: {e}")
        
        return config
    
    def update_model(self, model_id: str, config: Dict[str, Any],
                    model_type: Optional[str] = None,
                    attention_type: Optional[str] = None,
                    parameter_count: Optional[int] = None,
                    logo: Optional[str] = None,
                    model_analysis: Optional[Dict[str, Any]] = None,
                    change_description: Optional[str] = None) -> bool:
        """Update an existing model.
        
        Args:
            model_id: Model identifier
            config: New configuration dictionary
            model_type: Updated model type
            attention_type: Updated attention type
            parameter_count: Updated parameter count
            logo: Updated URL to the model's logo
            model_analysis: Updated detailed analysis of the model
            change_description: Description of changes
            
        Returns:
            True if successful, False otherwise
        """
        current = self.get_model(model_id)
        if not current:
            logger.error(f"Model {model_id} not found")
            return False
        
        version_query = "SELECT MAX(version) as max_version FROM model_versions WHERE model_id = ?"
        version_result = self.db.execute_one(version_query, (model_id,))
        next_version = (version_result['max_version'] or 0) + 1
        
        data = {'config_json': json.dumps(config)}
        if model_type is not None: data['model_type'] = model_type
        if attention_type is not None: data['attention_type'] = attention_type
        if parameter_count is not None: data['parameter_count'] = parameter_count
        if logo is not None: data['logo'] = logo
        if model_analysis is not None: data['model_analysis'] = json.dumps(model_analysis)
        
        rows_affected = self.db.update('models', data, 'model_id = ?', (model_id,))
        
        if rows_affected > 0:
            self._add_model_version(
                model_id, next_version, config, 
                change_description or f"Updated to version {next_version}"
            )
            logger.info(f"Updated model {model_id}")
            return True
            
        return False
    
    def delete_model(self, model_id: str, hard_delete: bool = False) -> bool:
        """Delete a model (soft delete by default).
        
        Args:
            model_id: Model identifier
            hard_delete: If True, permanently delete from database
            
        Returns:
            True if successful, False otherwise
        """
        if hard_delete:
            rows_affected = self.db.delete('models', 'model_id = ?', (model_id,))
        else:
            # Soft delete
            rows_affected = self.db.update(
                'models', {'is_active': 0}, 'model_id = ?', (model_id,)
            )
        
        if rows_affected > 0:
            logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted model {model_id}")
            return True
        return False
    
    def list_models(self, model_type: Optional[str] = None,
                   attention_type: Optional[str] = None,
                   min_params: Optional[int] = None,
                   max_params: Optional[int] = None,
                   source: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            attention_type: Filter by attention type
            min_params: Minimum parameter count
            max_params: Maximum parameter count
            source: Filter by source
            
        Returns:
            List of model dictionaries
        """
        query = "SELECT * FROM models WHERE is_active = 1"
        params = []
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        if attention_type:
            query += " AND attention_type = ?"
            params.append(attention_type)
        
        if min_params is not None:
            query += " AND parameter_count >= ?"
            params.append(min_params)
        
        if max_params is not None:
            query += " AND parameter_count <= ?"
            params.append(max_params)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY model_id"
        
        results = self.db.execute(query, tuple(params) if params else None)
        return [self._row_to_dict(row) for row in results]
    
    def search_models(self, search_term: str) -> List[Dict[str, Any]]:
        """Search models by name or ID.
        
        Args:
            search_term: Search term
            
        Returns:
            List of matching models
        """
        query = """
            SELECT * FROM models 
            WHERE is_active = 1 
            AND (model_id LIKE ? OR model_name LIKE ?)
            ORDER BY model_id
        """
        search_pattern = f"%{search_term}%"
        results = self.db.execute(query, (search_pattern, search_pattern))
        return [self._row_to_dict(row) for row in results]
    
    def add_quality_metric(self, model_id: str, metric_name: str,
                          metric_value: float, shots: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a quality metric for a model.
        
        Args:
            model_id: Model identifier
            metric_name: Name of the metric (e.g., "MMLU")
            metric_value: Metric value
            shots: Number of shots used
            metadata: Additional metadata
            
        Returns:
            ID of the inserted metric
        """
        data = {
            'model_id': model_id,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'shots': shots,
            'metadata': json.dumps(metadata) if metadata else None
        }
        
        return self.db.insert('model_quality_metrics', data)
    
    def get_quality_metrics(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all quality metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of quality metrics
        """
        query = """
            SELECT * FROM model_quality_metrics 
            WHERE model_id = ? 
            ORDER BY metric_name
        """
        results = self.db.execute(query, (model_id,))
        
        metrics = []
        for row in results:
            metric = self._row_to_dict(row)
            if metric['metadata']:
                metric['metadata'] = json.loads(metric['metadata'])
            metrics.append(metric)
        
        return metrics
    
    def cache_value(self, model_id: str, cache_key: str, 
                   cache_value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Cache a computed value for a model.
        
        Args:
            model_id: Model identifier
            cache_key: Cache key
            cache_value: Value to cache (will be JSON encoded)
            ttl_seconds: Time to live in seconds
        """
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        # Try to update existing cache entry
        existing = self.db.execute_one(
            "SELECT id FROM model_cache WHERE model_id = ? AND cache_key = ?",
            (model_id, cache_key)
        )
        
        if existing:
            self.db.update(
                'model_cache',
                {
                    'cache_value': json.dumps(cache_value),
                    'created_at': datetime.now().isoformat(),
                    'expires_at': expires_at
                },
                'id = ?',
                (existing['id'],)
            )
        else:
            self.db.insert('model_cache', {
                'model_id': model_id,
                'cache_key': cache_key,
                'cache_value': json.dumps(cache_value),
                'expires_at': expires_at
            })
    
    def get_cached_value(self, model_id: str, cache_key: str) -> Optional[Any]:
        """Get a cached value for a model.
        
        Args:
            model_id: Model identifier
            cache_key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        query = """
            SELECT cache_value, expires_at 
            FROM model_cache 
            WHERE model_id = ? AND cache_key = ?
        """
        result = self.db.execute_one(query, (model_id, cache_key))
        
        if result:
            # Check expiration
            if result['expires_at']:
                expires_at = datetime.fromisoformat(result['expires_at'])
                if expires_at < datetime.now():
                    # Expired, delete it
                    self.db.delete(
                        'model_cache',
                        'model_id = ? AND cache_key = ?',
                        (model_id, cache_key)
                    )
                    return None
            
            return json.loads(result['cache_value'])
        
        return None
    
    def clear_cache(self, model_id: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            model_id: If provided, clear only for this model
            
        Returns:
            Number of entries cleared
        """
        if model_id:
            return self.db.delete('model_cache', 'model_id = ?', (model_id,))
        else:
            # Clear all expired entries
            return self.db.delete(
                'model_cache',
                'expires_at IS NOT NULL AND expires_at < ?',
                (datetime.now().isoformat(),)
            )
    
    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get version history for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of versions
        """
        query = """
            SELECT * FROM model_versions 
            WHERE model_id = ? 
            ORDER BY version DESC
        """
        results = self.db.execute(query, (model_id,))
        
        versions = []
        for row in results:
            version = self._row_to_dict(row)
            version['config_json'] = json.loads(version['config_json'])
            versions.append(version)
        
        return versions
    
    def _add_model_version(self, model_id: str, version: int,
                          config: Dict[str, Any], description: str) -> None:
        """Add a version entry for a model."""
        self.db.insert('model_versions', {
            'model_id': model_id,
            'version': version,
            'config_json': json.dumps(config),
            'change_description': description
        })
    
    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        return dict(row) if row else {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the models in the database."""
        stats = {
            'total_models': 0,
            'by_type': {},
            'by_attention': {},
            'by_source': {},
            'parameter_ranges': {
                'min': None,
                'max': None,
                'avg': None
            }
        }
        
        # Total models
        count_result = self.db.execute_one(
            "SELECT COUNT(*) as count FROM models WHERE is_active = 1"
        )
        stats['total_models'] = count_result['count'] if count_result else 0
        
        # By type
        type_results = self.db.execute("""
            SELECT model_type, COUNT(*) as count 
            FROM models 
            WHERE is_active = 1 AND model_type IS NOT NULL
            GROUP BY model_type
        """)
        stats['by_type'] = {row['model_type']: row['count'] for row in type_results}
        
        # By attention
        attention_results = self.db.execute("""
            SELECT attention_type, COUNT(*) as count 
            FROM models 
            WHERE is_active = 1 AND attention_type IS NOT NULL
            GROUP BY attention_type
        """)
        stats['by_attention'] = {row['attention_type']: row['count'] for row in attention_results}
        
        # By source
        source_results = self.db.execute("""
            SELECT source, COUNT(*) as count 
            FROM models 
            WHERE is_active = 1
            GROUP BY source
        """)
        stats['by_source'] = {row['source']: row['count'] for row in source_results}
        
        # Parameter ranges
        param_result = self.db.execute_one("""
            SELECT 
                MIN(parameter_count) as min_params,
                MAX(parameter_count) as max_params,
                AVG(parameter_count) as avg_params
            FROM models 
            WHERE is_active = 1 AND parameter_count IS NOT NULL
        """)
        
        if param_result:
            stats['parameter_ranges'] = {
                'min': param_result['min_params'],
                'max': param_result['max_params'],
                'avg': round(param_result['avg_params'], 2) if param_result['avg_params'] else None
            }
        
        return stats
    
    def save_user_model_config(self, model_uri: str, config: Dict[str, Any]) -> int:
        """Save a user-provided configuration for a gated model.
        
        Args:
            model_uri: Model URI (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            config: Model configuration dictionary
            
        Returns:
            ID of the inserted/updated record
        """
        # Check if config already exists
        existing = self.db.execute_one(
            "SELECT id FROM user_model_configs WHERE model_uri = ?",
            (model_uri,)
        )
        
        config_json = json.dumps(config)
        
        if existing:
            # Update existing config
            self.db.update(
                'user_model_configs',
                {
                    'config_json': config_json,
                    'validated': 1,
                    'updated_at': datetime.now().isoformat()
                },
                'id = ?',
                (existing['id'],)
            )
            logger.info(f"Updated user config for {model_uri}")
            return existing['id']
        else:
            # Insert new config
            data = {
                'model_uri': model_uri,
                'config_json': config_json,
                'validated': 1
            }
            config_id = self.db.insert('user_model_configs', data)
            logger.info(f"Saved new user config for {model_uri}")
            return config_id
    
    def get_user_model_config(self, model_uri: str) -> Optional[Dict[str, Any]]:
        """Get a user-provided configuration for a gated model.
        
        Args:
            model_uri: Model URI
            
        Returns:
            Configuration dictionary or None if not found
        """
        result = self.db.execute_one(
            "SELECT config_json FROM user_model_configs WHERE model_uri = ? AND validated = 1",
            (model_uri,)
        )
        
        if result and result['config_json']:
            return json.loads(result['config_json'])
        return None
    
    def list_user_configs(self) -> List[Dict[str, Any]]:
        """List all user-provided model configurations.
        
        Returns:
            List of user config records
        """
        query = """
            SELECT model_uri, validated, created_at, updated_at 
            FROM user_model_configs 
            ORDER BY updated_at DESC
        """
        results = self.db.execute(query)
        return [self._row_to_dict(row) for row in results]
    
    def delete_user_config(self, model_uri: str) -> bool:
        """Delete a user-provided configuration.
        
        Args:
            model_uri: Model URI
            
        Returns:
            True if deleted, False if not found
        """
        rows_affected = self.db.delete(
            'user_model_configs',
            'model_uri = ?',
            (model_uri,)
        )
        
        if rows_affected > 0:
            logger.info(f"Deleted user config for {model_uri}")
            return True
        return False 