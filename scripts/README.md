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