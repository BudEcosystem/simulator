"""
BudUsecases class for unified usecase and SLO management.
Provides CRUD operations for usecases with performance requirements.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from .db.connection import DatabaseConnection
from .db.usecase_schema import (
    get_usecase_schema_statements,
    get_migration_statements,
    USECASE_SCHEMA_VERSION
)


logger = logging.getLogger(__name__)


class BudUsecases:
    """
    Unified usecase management system that:
    - Provides CRUD operations for usecases
    - Manages SLOs (Service Level Objectives)
    - Supports tagging and categorization
    - Maintains version history
    - Provides caching for expensive computations
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize BudUsecases with database connection.
        
        Args:
            db_path: Optional path to database. If None, uses default location.
        """
        # Initialize database connection
        self.db = DatabaseConnection(db_path)
        
        # Initialize usecase schema
        self._initialize_usecase_schema()
    
    def _initialize_usecase_schema(self):
        """Initialize usecase database schema."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check current schema version
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='usecase_schema_version'
            """)
            
            current_version = 0
            if cursor.fetchone():
                cursor.execute(
                    "SELECT version FROM usecase_schema_version ORDER BY version DESC LIMIT 1"
                )
                result = cursor.fetchone()
                if result:
                    current_version = result[0]
            
            # Apply schema updates if needed
            if current_version < USECASE_SCHEMA_VERSION:
                # Create new schema
                for statement in get_usecase_schema_statements():
                    cursor.executescript(statement)
                
                # Apply migrations if upgrading
                if current_version > 0:
                    migration_sql = get_migration_statements(current_version, USECASE_SCHEMA_VERSION)
                    if migration_sql:
                        cursor.executescript(migration_sql)
                
                # Update schema version
                cursor.execute(
                    "INSERT OR REPLACE INTO usecase_schema_version (version) VALUES (?)",
                    (USECASE_SCHEMA_VERSION,)
                )
                conn.commit()
    
    def add_usecase(self, usecase_data: Dict[str, Any]) -> str:
        """Add new usecase to the database.
        
        Args:
            usecase_data: Dictionary containing usecase specifications
            
        Returns:
            unique_id of the created usecase
            
        Raises:
            ValueError: If validation fails or usecase already exists
        """
        # Validate usecase data
        self._validate_usecase(usecase_data)
        
        # Check if already exists
        unique_id = usecase_data['unique_id']
        if self.get_usecase(unique_id):
            raise ValueError(f"Usecase {unique_id} already exists")
        
        # Convert numeric strings to appropriate types
        numeric_fields = ['ttft_min', 'ttft_max', 'e2e_min', 'e2e_max', 
                         'inter_token_min', 'inter_token_max']
        for field in numeric_fields:
            if field in usecase_data and isinstance(usecase_data[field], str):
                usecase_data[field] = float(usecase_data[field])
        
        # Insert into database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert main usecase record
            cursor.execute("""
                INSERT INTO usecases (
                    unique_id, name, industry, description,
                    batch_size, beam_size,
                    input_tokens_min, input_tokens_max,
                    output_tokens_min, output_tokens_max,
                    ttft_min, ttft_max,
                    e2e_min, e2e_max,
                    inter_token_min, inter_token_max,
                    source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                unique_id,
                usecase_data['name'],
                usecase_data['industry'],
                usecase_data.get('description'),
                usecase_data.get('batch_size', 1),
                usecase_data.get('beam_size', 1),
                usecase_data['input_tokens_min'],
                usecase_data['input_tokens_max'],
                usecase_data['output_tokens_min'],
                usecase_data['output_tokens_max'],
                usecase_data.get('ttft_min'),
                usecase_data.get('ttft_max'),
                usecase_data.get('e2e_min'),
                usecase_data.get('e2e_max'),
                usecase_data.get('inter_token_min'),
                usecase_data.get('inter_token_max'),
                usecase_data.get('source', 'manual')
            ))
            
            usecase_id = cursor.lastrowid
            
            # Add tags if provided
            if 'tags' in usecase_data:
                for tag in usecase_data['tags']:
                    cursor.execute("""
                        INSERT INTO usecase_tags (usecase_id, tag)
                        VALUES (?, ?)
                    """, (usecase_id, tag))
            
            # Add initial version
            cursor.execute("""
                INSERT INTO usecase_versions (usecase_id, version, config_json, change_description)
                VALUES (?, ?, ?, ?)
            """, (unique_id, 1, json.dumps(usecase_data), "Initial version"))
            
            conn.commit()
        
        logger.info(f"Added usecase {unique_id}: {usecase_data['name']}")
        return unique_id
    
    def get_usecase(self, unique_id: str) -> Optional[Dict[str, Any]]:
        """Get specific usecase by unique_id.
        
        Args:
            unique_id: Usecase unique identifier
            
        Returns:
            Usecase dictionary if found, None otherwise.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.*, GROUP_CONCAT(ut.tag) as tags
                FROM usecases u
                LEFT JOIN usecase_tags ut ON u.id = ut.usecase_id
                WHERE u.unique_id = ? AND u.is_active = 1
                GROUP BY u.id
            """, (unique_id,))
            
            row = cursor.fetchone()
            if row:
                usecase = dict(row)
                # Convert tags string to list
                if usecase['tags']:
                    usecase['tags'] = usecase['tags'].split(',')
                else:
                    usecase['tags'] = []
                return usecase
            
        return None
    
    def get_all_usecases(self) -> List[Dict[str, Any]]:
        """Get all active usecases.
        
        Returns:
            List of usecase dictionaries.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.*, GROUP_CONCAT(ut.tag) as tags
                FROM usecases u
                LEFT JOIN usecase_tags ut ON u.id = ut.usecase_id
                WHERE u.is_active = 1
                GROUP BY u.id
                ORDER BY u.unique_id
            """)
            
            usecases = []
            for row in cursor.fetchall():
                usecase = dict(row)
                # Convert tags string to list
                if usecase['tags']:
                    usecase['tags'] = usecase['tags'].split(',')
                else:
                    usecase['tags'] = []
                usecases.append(usecase)
            
        return usecases
    
    def update_usecase(self, unique_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing usecase.
        
        Args:
            unique_id: Usecase unique identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if usecase not found.
        """
        # Check if usecase exists
        usecase = self.get_usecase(unique_id)
        if not usecase:
            return False
        
        # Convert numeric strings to appropriate types
        numeric_fields = ['ttft_min', 'ttft_max', 'e2e_min', 'e2e_max', 
                         'inter_token_min', 'inter_token_max']
        for field in numeric_fields:
            if field in updates and isinstance(updates[field], str):
                updates[field] = float(updates[field])
        
        # Validate updates
        if 'industry' in updates and not updates['industry']:
            raise ValueError("Industry cannot be empty")
        
        # Build update query
        update_fields = []
        update_values = []
        
        # Map of allowed update fields
        field_mapping = {
            'name': 'name',
            'industry': 'industry',
            'description': 'description',
            'batch_size': 'batch_size',
            'beam_size': 'beam_size',
            'input_tokens_min': 'input_tokens_min',
            'input_tokens_max': 'input_tokens_max',
            'output_tokens_min': 'output_tokens_min',
            'output_tokens_max': 'output_tokens_max',
            'ttft_min': 'ttft_min',
            'ttft_max': 'ttft_max',
            'e2e_min': 'e2e_min',
            'e2e_max': 'e2e_max',
            'inter_token_min': 'inter_token_min',
            'inter_token_max': 'inter_token_max'
        }
        
        for key, value in updates.items():
            if key in field_mapping:
                update_fields.append(f"{field_mapping[key]} = ?")
                update_values.append(value)
        
        if not update_fields:
            return True  # Nothing to update
        
        update_values.append(unique_id)  # For WHERE clause
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update main record
            query = f"UPDATE usecases SET {', '.join(update_fields)} WHERE unique_id = ?"
            cursor.execute(query, update_values)
            
            # Handle tags update if provided
            if 'tags' in updates:
                # Delete existing tags
                cursor.execute(
                    "DELETE FROM usecase_tags WHERE usecase_id = (SELECT id FROM usecases WHERE unique_id = ?)",
                    (unique_id,)
                )
                
                # Add new tags
                cursor.execute("SELECT id FROM usecases WHERE unique_id = ?", (unique_id,))
                usecase_id = cursor.fetchone()[0]
                
                for tag in updates['tags']:
                    cursor.execute(
                        "INSERT INTO usecase_tags (usecase_id, tag) VALUES (?, ?)",
                        (usecase_id, tag)
                    )
            
            # Add version record
            cursor.execute(
                "SELECT MAX(version) FROM usecase_versions WHERE usecase_id = ?",
                (unique_id,)
            )
            result = cursor.fetchone()
            current_version = result[0] if result and result[0] else 0
            
            # Get current full config for version history
            cursor.execute("""
                SELECT u.*, GROUP_CONCAT(ut.tag) as tags
                FROM usecases u
                LEFT JOIN usecase_tags ut ON u.id = ut.usecase_id
                WHERE u.unique_id = ?
                GROUP BY u.id
            """, (unique_id,))
            
            updated_row = cursor.fetchone()
            if updated_row:
                updated_usecase = dict(updated_row)
                if updated_usecase['tags']:
                    updated_usecase['tags'] = updated_usecase['tags'].split(',')
                else:
                    updated_usecase['tags'] = []
                
                cursor.execute("""
                    INSERT INTO usecase_versions (usecase_id, version, config_json, change_description)
                    VALUES (?, ?, ?, ?)
                """, (
                    unique_id, 
                    current_version + 1, 
                    json.dumps(updated_usecase),
                    f"Updated fields: {', '.join(update_fields)}"
                ))
            
            conn.commit()
        
        logger.info(f"Updated usecase {unique_id}")
        return True
    
    def delete_usecase(self, unique_id: str, hard_delete: bool = False) -> bool:
        """Delete usecase (soft delete by default).
        
        Args:
            unique_id: Usecase unique identifier
            hard_delete: If True, permanently delete
            
        Returns:
            True if successful, False if not found.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if hard_delete:
                cursor.execute("DELETE FROM usecases WHERE unique_id = ?", (unique_id,))
            else:
                cursor.execute(
                    "UPDATE usecases SET is_active = 0 WHERE unique_id = ?",
                    (unique_id,)
                )
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted usecase {unique_id}")
                return True
        
        return False
    
    def search_usecases(self,
                       query: Optional[str] = None,
                       industry: Optional[Union[str, List[str]]] = None,
                       tags: Optional[List[str]] = None,
                       min_input_tokens: Optional[int] = None,
                       max_input_tokens: Optional[int] = None,
                       min_output_tokens: Optional[int] = None,
                       max_output_tokens: Optional[int] = None,
                       max_e2e_latency: Optional[float] = None,
                       max_ttft: Optional[float] = None,
                       batch_size: Optional[int] = None,
                       sort_by: str = 'unique_id',
                       sort_order: str = 'asc',
                       limit: Optional[int] = None,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """Search usecases with various filters.
        
        Args:
            query: Text search in name and description
            industry: Filter by industry (single or list)
            tags: Filter by tags (must have all specified tags)
            min_input_tokens: Minimum input token requirement
            max_input_tokens: Maximum input token requirement
            min_output_tokens: Minimum output token requirement
            max_output_tokens: Maximum output token requirement
            max_e2e_latency: Maximum end-to-end latency requirement
            max_ttft: Maximum time to first token requirement
            batch_size: Filter by specific batch size
            sort_by: Field to sort by
            sort_order: 'asc' or 'desc'
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching usecases
        """
        # Build query
        base_query = """
            SELECT DISTINCT u.*, GROUP_CONCAT(DISTINCT ut.tag) as tags
            FROM usecases u
            LEFT JOIN usecase_tags ut ON u.id = ut.usecase_id
            WHERE u.is_active = 1
        """
        
        conditions = []
        params = []
        
        # Text search
        if query:
            conditions.append("(u.name LIKE ? OR u.description LIKE ?)")
            search_term = f"%{query}%"
            params.extend([search_term, search_term])
        
        # Industry filter
        if industry:
            if isinstance(industry, str):
                conditions.append("u.industry = ?")
                params.append(industry)
            else:
                placeholders = ','.join(['?' for _ in industry])
                conditions.append(f"u.industry IN ({placeholders})")
                params.extend(industry)
        
        # Token filters
        if min_input_tokens is not None:
            conditions.append("u.input_tokens_max >= ?")
            params.append(min_input_tokens)
        
        if max_input_tokens is not None:
            conditions.append("u.input_tokens_min <= ?")
            params.append(max_input_tokens)
        
        if min_output_tokens is not None:
            conditions.append("u.output_tokens_max >= ?")
            params.append(min_output_tokens)
        
        if max_output_tokens is not None:
            conditions.append("u.output_tokens_min <= ?")
            params.append(max_output_tokens)
        
        # Performance filters
        if max_e2e_latency is not None:
            conditions.append("u.e2e_min <= ?")
            params.append(max_e2e_latency)
        
        if max_ttft is not None:
            conditions.append("u.ttft_min <= ?")
            params.append(max_ttft)
        
        if batch_size is not None:
            conditions.append("u.batch_size = ?")
            params.append(batch_size)
        
        # Tag filter (handled separately due to complexity)
        tag_query = ""
        if tags:
            tag_query = f"""
                AND u.id IN (
                    SELECT usecase_id 
                    FROM usecase_tags 
                    WHERE tag IN ({','.join(['?' for _ in tags])})
                    GROUP BY usecase_id
                    HAVING COUNT(DISTINCT tag) = ?
                )
            """
            params.extend(tags)
            params.append(len(tags))
        
        # Combine conditions
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        base_query += tag_query
        
        # Group by
        base_query += " GROUP BY u.id"
        
        # Sorting
        valid_sort_fields = {
            'unique_id': 'u.unique_id',
            'name': 'u.name',
            'industry': 'u.industry',
            'e2e_max': 'u.e2e_max',
            'ttft_max': 'u.ttft_max',
            'input_tokens_max': 'u.input_tokens_max',
            'output_tokens_max': 'u.output_tokens_max',
            'created_at': 'u.created_at'
        }
        
        sort_field = valid_sort_fields.get(sort_by, 'u.unique_id')
        sort_direction = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
        base_query += f" ORDER BY {sort_field} {sort_direction}"
        
        # Pagination
        if limit:
            base_query += f" LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                usecase = dict(row)
                # Convert tags string to list
                if usecase['tags']:
                    usecase['tags'] = usecase['tags'].split(',')
                else:
                    usecase['tags'] = []
                results.append(usecase)
        
        return results
    
    def import_from_json(self, filepath: str) -> Dict[str, Any]:
        """Import usecases from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with import statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of usecases")
        
        stats = {
            'total': len(data),
            'imported': 0,
            'skipped': 0,
            'errors': []
        }
        
        for usecase_data in data:
            try:
                # Set source
                usecase_data['source'] = 'json_import'
                
                # Add usecase
                self.add_usecase(usecase_data)
                stats['imported'] += 1
                
            except ValueError as e:
                if "already exists" in str(e):
                    stats['skipped'] += 1
                else:
                    stats['errors'].append({
                        'usecase': usecase_data.get('unique_id', 'unknown'),
                        'error': str(e)
                    })
            except Exception as e:
                stats['errors'].append({
                    'usecase': usecase_data.get('unique_id', 'unknown'),
                    'error': str(e)
                })
        
        logger.info(f"Import complete: {stats['imported']} imported, {stats['skipped']} skipped")
        return stats
    
    def export_to_json(self, filepath: str, industry: Optional[str] = None) -> bool:
        """Export usecases to JSON file.
        
        Args:
            filepath: Path to output file
            industry: Optional filter by industry
            
        Returns:
            True if successful
        """
        # Get usecases
        if industry:
            usecases = self.search_usecases(industry=industry)
        else:
            usecases = self.get_all_usecases()
        
        # Clean up for export
        export_data = []
        for usecase in usecases:
            # Remove database-specific fields
            export_usecase = {
                'unique_id': usecase['unique_id'],
                'name': usecase['name'],
                'industry': usecase['industry'],
                'batch_size': usecase['batch_size'],
                'beam_size': usecase['beam_size'],
                'input_tokens_min': usecase['input_tokens_min'],
                'input_tokens_max': usecase['input_tokens_max'],
                'output_tokens_min': usecase['output_tokens_min'],
                'output_tokens_max': usecase['output_tokens_max'],
                'ttft_min': usecase['ttft_min'],
                'ttft_max': usecase['ttft_max'],
                'e2e_min': str(usecase['e2e_min']),  # Convert to string for compatibility
                'e2e_max': str(usecase['e2e_max']),
                'inter_token_min': str(usecase['inter_token_min']),
                'inter_token_max': str(usecase['inter_token_max'])
            }
            
            if usecase.get('description'):
                export_usecase['description'] = usecase['description']
            
            if usecase.get('tags'):
                export_usecase['tags'] = usecase['tags']
            
            export_data.append(export_usecase)
        
        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} usecases to {filepath}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about usecases.
        
        Returns:
            Dictionary with various statistics
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM usecases WHERE is_active = 1")
            total = cursor.fetchone()[0]
            
            # By industry
            cursor.execute("""
                SELECT industry, COUNT(*) as count
                FROM usecases
                WHERE is_active = 1
                GROUP BY industry
                ORDER BY count DESC
            """)
            by_industry = {row['industry']: row['count'] for row in cursor.fetchall()}
            
            # By latency profile
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN e2e_max <= 2 THEN 'real-time'
                        WHEN e2e_max <= 5 THEN 'interactive'
                        WHEN e2e_max <= 10 THEN 'responsive'
                        ELSE 'batch'
                    END as profile,
                    COUNT(*) as count
                FROM usecases
                WHERE is_active = 1
                GROUP BY profile
            """)
            by_latency = {row['profile']: row['count'] for row in cursor.fetchall()}
            
            # Popular tags
            cursor.execute("""
                SELECT tag, COUNT(*) as count
                FROM usecase_tags ut
                JOIN usecases u ON ut.usecase_id = u.id
                WHERE u.is_active = 1
                GROUP BY tag
                ORDER BY count DESC
                LIMIT 10
            """)
            popular_tags = [(row['tag'], row['count']) for row in cursor.fetchall()]
        
        return {
            'total_usecases': total,
            'by_industry': by_industry,
            'by_latency_profile': by_latency,
            'popular_tags': popular_tags
        }
    
    def cache_value(self, usecase_id: str, cache_key: str, 
                   cache_value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Cache a computed value for a usecase.
        
        Args:
            usecase_id: Usecase unique identifier
            cache_key: Cache key
            cache_value: Value to cache (will be JSON encoded)
            ttl_seconds: Time to live in seconds
        """
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to update existing cache entry
            cursor.execute(
                "SELECT id FROM usecase_cache WHERE usecase_id = ? AND cache_key = ?",
                (usecase_id, cache_key)
            )
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("""
                    UPDATE usecase_cache
                    SET cache_value = ?, created_at = CURRENT_TIMESTAMP, expires_at = ?
                    WHERE id = ?
                """, (json.dumps(cache_value), expires_at, existing[0]))
            else:
                cursor.execute("""
                    INSERT INTO usecase_cache (usecase_id, cache_key, cache_value, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (usecase_id, cache_key, json.dumps(cache_value), expires_at))
            
            conn.commit()
    
    def get_cached_value(self, usecase_id: str, cache_key: str) -> Optional[Any]:
        """Get a cached value for a usecase.
        
        Args:
            usecase_id: Usecase unique identifier
            cache_key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cache_value, expires_at 
                FROM usecase_cache 
                WHERE usecase_id = ? AND cache_key = ?
            """, (usecase_id, cache_key))
            
            result = cursor.fetchone()
            if result:
                # Check expiration
                if result['expires_at']:
                    expires_at = datetime.fromisoformat(result['expires_at'])
                    if expires_at < datetime.now():
                        # Expired, delete it
                        cursor.execute(
                            "DELETE FROM usecase_cache WHERE usecase_id = ? AND cache_key = ?",
                            (usecase_id, cache_key)
                        )
                        conn.commit()
                        return None
                
                return json.loads(result['cache_value'])
        
        return None
    
    def _validate_usecase(self, usecase_data: Dict[str, Any]):
        """Validate usecase data before insertion/update.
        
        Args:
            usecase_data: Usecase data to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Required fields
        required_fields = [
            'unique_id', 'name', 'industry',
            'input_tokens_min', 'input_tokens_max',
            'output_tokens_min', 'output_tokens_max'
        ]
        
        for field in required_fields:
            if field not in usecase_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate token ranges
        if usecase_data['input_tokens_min'] > usecase_data['input_tokens_max']:
            raise ValueError("input_tokens_min cannot be greater than input_tokens_max")
        
        if usecase_data['output_tokens_min'] > usecase_data['output_tokens_max']:
            raise ValueError("output_tokens_min cannot be greater than output_tokens_max")
        
        # Validate SLOs if provided
        if 'ttft_min' in usecase_data and 'ttft_max' in usecase_data:
            if usecase_data['ttft_min'] > usecase_data['ttft_max']:
                raise ValueError("ttft_min cannot be greater than ttft_max")
        
        if 'e2e_min' in usecase_data and 'e2e_max' in usecase_data:
            if usecase_data['e2e_min'] > usecase_data['e2e_max']:
                raise ValueError("e2e_min cannot be greater than e2e_max")
        
        if 'inter_token_min' in usecase_data and 'inter_token_max' in usecase_data:
            if usecase_data['inter_token_min'] > usecase_data['inter_token_max']:
                raise ValueError("inter_token_min cannot be greater than inter_token_max") 