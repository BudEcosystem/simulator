# BudSimulator Scripts

This directory contains utility scripts for managing the BudSimulator system.

## update_missing_data.py

A batch script that checks all models in MODEL_DICT and the database for missing logos and LLM-based analysis, then fetches them automatically.

### Key Features

- **Non-invasive**: Does NOT modify the static MODEL_DICT
- **Supplementary Records**: Creates database records for MODEL_DICT models to store logos/analysis
- **Smart Merging**: The API automatically merges data from both sources when serving requests
- **Efficient**: Only processes models with missing data
- **Flexible**: Supports filtering and limiting for targeted updates

### Usage

```bash
# Update all models with missing data
python scripts/update_missing_data.py

# Process only the first 10 models
python scripts/update_missing_data.py --limit 10

# Process only models matching a pattern
python scripts/update_missing_data.py --filter "mistral"

# Combine limit and filter
python scripts/update_missing_data.py --limit 5 --filter "meta-llama"

# Dry run mode (shows what would be updated without making changes)
python scripts/update_missing_data.py --dry-run
```

### How It Works

1. **Model Discovery**: 
   - Fetches all models from MODEL_DICT (static configuration)
   - Fetches all models from the database
   - Creates a unified list, avoiding duplicates

2. **For Each Model**:
   - Checks if it has a logo and analysis in the database
   - For MODEL_DICT models without DB records, creates a supplementary record
   - Fetches missing logos from HuggingFace organization pages
   - Generates missing analysis using LLM (via `call_bud_LLM`)

3. **Data Storage**:
   - Logos are saved to the `logos/` directory
   - Analysis is stored as JSON in the database
   - MODEL_DICT models get supplementary DB records (source: 'model_dict_supplement')

4. **API Integration**:
   - The `/api/models/list` endpoint automatically merges data
   - When a model exists in both sources, database data (logo/analysis) takes precedence
   - Frontend receives complete model information regardless of source

### Example Output

```
Starting batch update process...
Found 20 models in MODEL_DICT
Found 15 models in database

[1/35] Processing mistral_7b (source: model_dict)
Creating supplementary DB record for MODEL_DICT model: mistral_7b
Fetching logo for mistral_7b
✓ Added logo for mistral_7b
Generating analysis for mistral_7b
✓ Added analysis for mistral_7b
Updated: logo, analysis

[2/35] Processing meta-llama/Llama-2-7b-hf (source: database)
Generating analysis for meta-llama/Llama-2-7b-hf
✓ Added analysis for meta-llama/Llama-2-7b-hf
Updated: analysis

============================================================
BATCH UPDATE SUMMARY
============================================================
Total models found:     35
Models checked:         35
Logos added:           12
Analysis added:        28
Errors encountered:    2
============================================================
```

### Testing

Use the `test_update.py` script to verify functionality:

```bash
python scripts/test_update.py
```

This will:
- Process a small subset of models
- Verify supplementary records are created
- Test the filter functionality
- Display the results

### Notes

- The script respects rate limits when calling HuggingFace and the LLM
- Errors are logged but don't stop the batch process
- Progress is displayed in real-time
- The script is idempotent - running it multiple times is safe 

# BudSimulator Database Setup

This directory contains scripts for setting up and managing the BudSimulator database.

## Quick Start

To set up the complete database with all data:

```bash
cd BudSimulator/scripts
python setup_database.py
```

## Setup Script (`setup_database.py`)

The main setup script handles complete database initialization and data population.

### What it does:

1. **🔧 Database Schema Creation**
   - Creates SQLite database with all required tables
   - Initializes proper indexes and constraints

2. **🖥️ Hardware Data Import**
   - Imports from `Systems/system_configs.py` (GPUs, accelerators)
   - Imports from `GenZ/cpu/cpu_configs.py` (CPU configurations)
   - Imports from `src/utils/hardware.json` (comprehensive hardware database)

3. **🤖 Model Data Import**
   - Imports all models from GenZ `MODEL_DICT`
   - Extracts basic model attributes (type, parameters, etc.)

4. **📊 Model Parameter Extraction**
   - Detects model types (transformer, mamba, etc.)
   - Identifies attention mechanisms (mha, gqa, mqa)
   - Calculates parameter counts from model configs

5. **🖼️ Logo Download**
   - Downloads model logos from HuggingFace
   - Saves locally and updates database paths

6. **🧠 LLM-based Model Analysis**
   - Scrapes model descriptions from HuggingFace
   - Generates detailed analysis using LLM
   - Extracts advantages, disadvantages, use cases

### Usage Options:

```bash
# Full setup (recommended for first time)
python setup_database.py

# Skip time-consuming operations
python setup_database.py --skip-analysis --skip-logos

# Skip only LLM analysis (keeps logo download)
python setup_database.py --skip-analysis

# Skip only logo download (keeps LLM analysis)
python setup_database.py --skip-logos

# Force recreate database (WARNING: destroys existing data)
python setup_database.py --force-recreate
```

### Expected Output:

```
🚀 Starting complete database setup...

🔧 Creating database schema...
✅ Database schema created successfully

🖥️  Importing hardware data...
  📊 Importing from system_configs.py...
    ✓ Added A100_40GB_GPU
    ✓ Added H100_GPU
    [... more hardware ...]
  🧠 Importing CPU configurations...
    ✓ Added intel_xeon_8380
    [... more CPUs ...]
  📁 Importing from hardware.json...
    ✓ Added NVIDIA H100 (Hopper) - SXM
    [... more hardware ...]
✅ Hardware data import completed

🤖 Importing model data from MODEL_DICT...
  Found 150+ models in MODEL_DICT
    ✓ Added meta-llama/Llama-2-7b-hf
    [... more models ...]
✅ Model data import completed

📊 Extracting model parameters...
    ✓ Updated meta-llama/Llama-2-7b-hf: ['model_type', 'attention_type']
    [... more updates ...]
✅ Model parameter extraction completed (120 models updated)

🖼️  Downloading model logos...
    📥 Downloading logo for meta-llama/Llama-2-7b-hf
    ✅ Downloaded logo for meta-llama/Llama-2-7b-hf
    [... more logos ...]
✅ Logo download completed (150 logos downloaded)

🧠 Generating model analysis...
    🤔 Generating analysis for meta-llama/Llama-2-7b-hf
    ✅ Generated analysis for meta-llama/Llama-2-7b-hf
    [... more analysis ...]
✅ Model analysis generation completed (150 analyses generated)

🎉 Database setup completed successfully!

Next steps:
1. Start the backend API: python -m uvicorn apis.main:app --reload
2. Start the frontend: cd frontend && npm start
3. Open http://localhost:3000 in your browser
```

## Other Scripts

### `update_missing_data.py`
Simplified script to update only missing logos and analysis for existing models in the database.

```bash
python update_missing_data.py
```

## Troubleshooting

### Common Issues:

1. **Import Errors**
   - Make sure you're in the correct directory: `BudSimulator/scripts/`
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Database Lock Errors**
   - Close any DB browser applications
   - Make sure no other processes are using the database

3. **Network Errors (Logo/Analysis)**
   - Check internet connection
   - HuggingFace might be temporarily unavailable
   - Use `--skip-logos` or `--skip-analysis` to bypass

4. **LLM API Errors**
   - Check your LLM configuration in `src/bud_ai.py`
   - Ensure API keys are properly set
   - Use `--skip-analysis` to bypass LLM generation

### Database Location:
- Default: `~/.genz_simulator/db/models.db`
- Hardware: `~/.genz_simulator/db/hardware.db`

### Performance Notes:
- **Full setup**: 30-60 minutes (depending on network and LLM speed)
- **Without analysis**: 5-10 minutes
- **Without logos**: 10-15 minutes
- **Schema + data only**: 2-3 minutes

## Development

To add new data sources or modify the setup process, edit the `DatabaseSetup` class in `setup_database.py`.

Key methods:
- `import_hardware_data()`: Add new hardware sources
- `import_model_data()`: Modify model import logic
- `extract_model_parameters()`: Add new parameter extraction
- `generate_model_analysis()`: Customize analysis generation 