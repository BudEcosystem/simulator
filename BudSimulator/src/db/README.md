# Dynamic Model Management for GenZ Simulator

This module provides dynamic model management capabilities for the GenZ simulator, allowing models to be loaded from both static definitions and a SQLite database. It supports automatic importing of models from HuggingFace Hub.

## Features

- **Dynamic Model Loading**: Seamlessly load models from static definitions or database
- **HuggingFace Integration**: Import any model from HuggingFace Hub with a single command
- **Persistent Storage**: Models are stored in SQLite database for offline access
- **Caching**: Computed values (e.g., memory analysis) are cached for performance
- **Version Control**: Track model updates and changes over time
- **Quality Metrics**: Store and retrieve model quality metrics (MMLU, GSM8K, etc.)
- **Backward Compatibility**: Works alongside existing static MODEL_DICT
- **CLI Tools**: Command-line interface for model management

## Installation

The database module is automatically available when you install BudSimulator. The required dependencies are:
- SQLite (built into Python)
- huggingface_hub
- tabulate (for CLI)

## Usage

### Automatic Dynamic Loading

When you import GenZ Models, dynamic loading is automatically enabled if the database module is available:

```python
from Models import MODEL_DICT

# This now supports both static and database models
model = MODEL_DICT.get_model("mistralai/Mistral-7B-v0.1")

# If the model isn't found locally, it will automatically 
# try to import from HuggingFace (if auto_import is enabled)
model = MODEL_DICT.get_model("meta-llama/Llama-2-7b-hf")
```

### Programmatic Model Import

```python
from Models import import_model_from_hf

# Import a model from HuggingFace
success = import_model_from_hf("mistralai/Mistral-7B-v0.1")

# Force update an existing model
success = import_model_from_hf("mistralai/Mistral-7B-v0.1", force_update=True)
```

### Using the Model Manager

```python
from src.db import ModelManager

manager = ModelManager()

# List all models
models = manager.list_models()

# Filter models
gqa_models = manager.list_models(attention_type="gqa")
large_models = manager.list_models(min_params=10_000_000_000)  # 10B+

# Get model details
model = manager.get_model("mistralai/Mistral-7B-v0.1")
config = manager.get_model_config("mistralai/Mistral-7B-v0.1")

# Add quality metrics
manager.add_quality_metric(
    model_id="mistralai/Mistral-7B-v0.1",
    metric_name="MMLU",
    metric_value=0.625,
    shots=5
)
```

### Command Line Interface

The module includes a comprehensive CLI for model management:

```bash
# List all models
python -m src.db.cli list

# Filter models
python -m src.db.cli list --type decoder-only --attention gqa

# Show model details
python -m src.db.cli show mistralai/Mistral-7B-v0.1
python -m src.db.cli show mistralai/Mistral-7B-v0.1 --config

# Import a model from HuggingFace
python -m src.db.cli import mistralai/Mistral-7B-v0.1
python -m src.db.cli import meta-llama/Llama-2-7b-hf --force

# Delete a model
python -m src.db.cli delete old-model --hard

# Sync models
python -m src.db.cli sync --static  # Sync static models to DB
python -m src.db.cli sync           # Update HF models

# Show statistics
python -m src.db.cli stats

# Clear cache
python -m src.db.cli clear-cache
python -m src.db.cli clear-cache mistralai/Mistral-7B-v0.1
```

## Database Schema

The module uses SQLite with the following tables:

### models
- Stores model configurations and metadata
- Fields: model_id, model_name, source, config_json, model_type, attention_type, parameter_count

### model_cache
- Caches computed values (e.g., memory analysis)
- Supports TTL-based expiration

### model_versions
- Tracks model version history
- Allows rollback to previous versions

### model_quality_metrics
- Stores quality metrics (MMLU, GSM8K, etc.)
- Supports multiple metrics per model with shot information

## Architecture

```
src/db/
├── __init__.py           # Module exports
├── schema.py            # Database schema definition
├── connection.py        # Database connection manager (singleton)
├── model_manager.py     # CRUD operations for models
├── model_converter.py   # Convert between HF and GenZ formats
├── hf_integration.py    # HuggingFace Hub integration
├── model_loader.py      # Dynamic MODEL_DICT implementation
└── cli.py              # Command-line interface

GenZ/Models/
└── dynamic_loader.py    # Integration with GenZ MODEL_DICT
```

## How It Works

1. **Database Initialization**: On first use, creates SQLite database at `~/.genz_simulator/db/models.db`

2. **Model Loading**: When `MODEL_DICT.get_model()` is called:
   - First checks static definitions
   - Then checks database
   - If auto_import is enabled and model looks like HF ID, imports from HuggingFace

3. **Conversion**: HuggingFace configs are automatically converted to GenZ ModelConfig format

4. **Caching**: Expensive computations (like memory analysis) are cached with optional TTL

## Examples

### Example 1: Import and Use a New Model

```python
from Models import MODEL_DICT, import_model_from_hf

# Import a model not in static definitions
import_model_from_hf("microsoft/phi-2")

# Now use it like any other model
model = MODEL_DICT.get_model("microsoft/phi-2")
print(f"Phi-2 has {model.num_decoder_layers} layers")
```

### Example 2: Analyze Models by Criteria

```python
from src.db import ModelManager

manager = ModelManager()

# Find all models with GQA attention
gqa_models = manager.list_models(attention_type="gqa")

# Find large models (>50B parameters)
large_models = manager.list_models(min_params=50_000_000_000)

# Get statistics
stats = manager.get_stats()
print(f"Total models: {stats['total_models']}")
print(f"By attention type: {stats['by_attention']}")
```

### Example 3: Sync Static Models to Database

```python
from Models import sync_static_models_to_db

# Sync all static MODEL_DICT entries to database
synced = sync_static_models_to_db()
print(f"Synced {synced} models to database")
```

## Database Location

By default, the database is stored at:
- Linux/Mac: `~/.genz_simulator/db/models.db`
- Windows: `%USERPROFILE%\.genz_simulator\db\models.db`

You can specify a custom location:

```python
from src.db import DatabaseConnection

db = DatabaseConnection("/path/to/custom/models.db")
```

## Troubleshooting

### Database not available
If you see "Database modules not available", ensure:
1. You're in the correct directory
2. The `src/db` module is in your Python path

### Import failures
If model import fails:
1. Check your internet connection
2. For gated models, set HF_TOKEN environment variable
3. Check if the model ID is correct

### Performance
- Use caching for expensive operations
- Clear expired cache entries periodically
- Consider using filters when listing many models

## Future Enhancements

- [ ] Bulk import from model lists
- [ ] Export/import database backups
- [ ] Model comparison tools
- [ ] Integration with model benchmarking
- [ ] Web UI for model management
- [ ] Automatic model discovery from HF Hub 