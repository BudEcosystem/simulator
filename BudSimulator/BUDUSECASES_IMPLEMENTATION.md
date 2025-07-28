# BudUsecases Implementation Documentation

## Overview

BudUsecases is a comprehensive usecase and SLO (Service Level Objectives) management system for the BudSimulator platform. It provides CRUD operations for managing AI/ML usecases with their performance requirements, token constraints, and latency targets.

## Architecture

### 1. Database Schema (`src/db/usecase_schema.py`)

The implementation uses SQLite with the following tables:

- **usecases**: Main table storing usecase configurations
  - Unique ID, name, industry, description
  - Token ranges (input/output min/max)
  - Batch and beam size configurations
  - SLO metrics (TTFT, E2E, inter-token latencies)
  - Soft delete support with `is_active` flag

- **usecase_tags**: Many-to-many relationship for tagging
  - Enables flexible categorization and filtering

- **usecase_versions**: Version history tracking
  - Stores complete configuration snapshots
  - Tracks changes over time

- **usecase_cache**: Performance optimization
  - Caches computed values (e.g., hardware recommendations)
  - TTL-based expiration support

### 2. Core Class (`src/usecases.py`)

The `BudUsecases` class provides:

#### CRUD Operations
- `add_usecase()`: Create new usecases with validation
- `get_usecase()`: Retrieve by unique ID
- `get_all_usecases()`: List all active usecases
- `update_usecase()`: Modify existing usecases with versioning
- `delete_usecase()`: Soft/hard delete support

#### Search & Filtering
- `search_usecases()`: Advanced search with multiple criteria
  - Text search in name/description
  - Filter by industry, tags, token ranges
  - Performance-based filtering (latency, TTFT)
  - Sorting and pagination support

#### Data Management
- `import_from_json()`: Bulk import from JSON files
- `export_to_json()`: Export filtered usecases
- `get_stats()`: Aggregate statistics and insights

#### Performance Features
- `cache_value()`: Store computed results with TTL
- `get_cached_value()`: Retrieve cached data
- Automatic cache invalidation on updates

### 3. API Layer (`apis/routers/usecases.py`)

RESTful API endpoints:

- **POST /api/usecases**: Create new usecase
- **GET /api/usecases**: List with filters
- **GET /api/usecases/search**: Advanced search
- **GET /api/usecases/{unique_id}**: Get specific usecase
- **PUT /api/usecases/{unique_id}**: Update usecase
- **DELETE /api/usecases/{unique_id}**: Delete usecase
- **GET /api/usecases/stats**: Get statistics
- **POST /api/usecases/import**: Import from JSON
- **POST /api/usecases/export**: Export to JSON
- **GET /api/usecases/industries/list**: List all industries
- **GET /api/usecases/tags/list**: List all tags with counts

### 4. Data Models

#### Request/Response Models (Pydantic)
- `UsecaseCreate`: Validation for new usecases
- `UsecaseUpdate`: Partial update support
- `UsecaseResponse`: Enhanced response with computed fields
  - `latency_profile`: Automatic categorization (real-time, interactive, responsive, batch)
  - `input_length_profile`: Token range classification (short, medium, long)

## Key Features

### 1. Validation
- Token range validation (min <= max)
- SLO constraint validation
- Required field enforcement
- Numeric string conversion support

### 2. Performance Optimization
- In-memory caching with `_usecase_cache`
- Database-level caching with TTL
- Efficient batch operations
- Optimized queries with proper indexing

### 3. Flexibility
- Tag-based categorization
- Industry-based grouping
- Version history tracking
- Soft delete for data recovery

### 4. Integration
- Seamless integration with existing BudSimulator infrastructure
- Uses shared database connection management
- Compatible with existing authentication/authorization

## Usage Examples

### 1. Creating a Usecase
```python
usecase_manager = BudUsecases()
usecase_data = {
    'unique_id': 'healthcare_001',
    'name': 'Patient Summarization',
    'industry': 'Healthcare',
    'input_tokens_min': 1000,
    'input_tokens_max': 2000,
    'output_tokens_min': 100,
    'output_tokens_max': 200,
    'ttft_min': 1.0,
    'ttft_max': 2.0,
    'e2e_min': 4.0,
    'e2e_max': 6.0,
    'tags': ['medical', 'nlp']
}
unique_id = usecase_manager.add_usecase(usecase_data)
```

### 2. Searching Usecases
```python
# Find healthcare usecases with real-time requirements
results = usecase_manager.search_usecases(
    industry='Healthcare',
    max_e2e_latency=2.0,
    tags=['real-time'],
    limit=10
)

# Find usecases supporting long inputs
results = usecase_manager.search_usecases(
    min_input_tokens=5000,
    sort_by='input_tokens_max',
    sort_order='desc'
)
```

### 3. Using Cache
```python
# Cache hardware recommendations
usecase_manager.cache_value(
    'healthcare_001',
    'hardware_recs',
    {'gpus': ['A100', 'H100'], 'config': {...}},
    ttl_seconds=3600
)

# Retrieve cached value
cached = usecase_manager.get_cached_value('healthcare_001', 'hardware_recs')
```

### 4. API Usage
```bash
# Create usecase
curl -X POST http://localhost:8000/api/usecases \
  -H "Content-Type: application/json" \
  -d '{
    "unique_id": "test_001",
    "name": "Test Usecase",
    "industry": "Testing",
    "input_tokens_min": 100,
    "input_tokens_max": 500,
    "output_tokens_min": 50,
    "output_tokens_max": 200
  }'

# Search with filters
curl "http://localhost:8000/api/usecases/search?industry=Healthcare&max_e2e_latency=5.0&limit=10"

# Get statistics
curl http://localhost:8000/api/usecases/stats
```

## Testing

Comprehensive test suite in `tests/test_usecases.py`:

### Unit Tests
- CRUD operations validation
- Search functionality
- Import/export capabilities
- Cache management
- Statistics generation

### API Tests
- Endpoint functionality
- Request/response validation
- Error handling
- Integration with FastAPI

### Test Coverage
- Database operations
- Validation logic
- Edge cases and error conditions
- Performance optimizations

## Data Import

Initial data import from `src/utils/usecases.json`:
- 112 predefined usecases across multiple industries
- Comprehensive SLO configurations
- Real-world performance requirements

Import script: `scripts/import_usecases.py`

## Future Enhancements

1. **Performance Profiling**: Integration with hardware recommendation engine
2. **ML-based Predictions**: Predict SLOs based on model characteristics
3. **Benchmarking Integration**: Link to actual benchmark results
4. **Multi-tenancy**: Organization-specific usecase management
5. **Advanced Analytics**: Trend analysis and optimization suggestions

## Backward Compatibility

The implementation maintains full backward compatibility:
- No changes to existing database schemas
- No modifications to existing APIs
- Additive changes only
- Shared infrastructure usage

## Migration Guide

For existing installations:
1. Run database migrations (automatic on first use)
2. Import predefined usecases: `python scripts/import_usecases.py`
3. Update API routes in main.py (already included)
4. No changes required to existing functionality 