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

# GenZ Simulation Scripts

This directory contains example and validation scripts for the GenZ unified simulation interface.

## Scripts Overview

### 1. `validation_comparison.py` - Core Validation
**Purpose**: Validates that the unified interface produces identical results to original GenZ functions.

**Features**:
- Compares `prefill_moddeling`, `decode_moddeling`, and `chunked_moddeling` with unified interface
- **Enhanced Metrics**: Now includes advanced performance metrics:
  - **TTFT (Time to First Token)**: Critical for interactive applications
  - **TPOT (Time Per Output Token)**: Determines streaming generation speed
  - **E2E Latency**: End-to-end latency for complete inference
  - **Prefill Throughput**: Tokens processed per second during prefill
  - **Decode Throughput**: Tokens generated per second during decode
  - **Total Throughput**: Combined throughput for entire inference
- Validates perfect accuracy (0.00e+00 relative error)
- Comprehensive test coverage with multiple configurations

**Usage**:
```bash
python scripts/validation_comparison.py
```

**Sample Output**:
```
Basic Decode:
  Original Results:
    Latency:    8.895 ms
    Throughput: 112.424 tokens/s
    Tpot        : 0.035 ms
    Ttft        : 0.035 ms
    E2E Latency : 8.895 ms
    Decode Throughput: 28780.4 tokens/s
    Total Throughput: 143902.2 tokens/s
  Unified Results:
    [Identical metrics...]
  Comparison:
    Match: ✓
    Max Relative Error: 0.00e+00
```

### 2. `metrics_showcase.py` - Comprehensive Metrics Analysis
**Purpose**: Demonstrates advanced performance metrics for LLM inference analysis.

**Key Metrics Explained**:
- **TTFT (Time to First Token)**: 
  - For prefill: Total prefill latency
  - For decode: Time per output token (≈ TPOT)
  - Critical for user experience in interactive applications
  
- **TPOT (Time Per Output Token)**:
  - Average time to generate each output token
  - Key metric for streaming generation performance
  - Calculated as: `total_decode_latency / output_tokens`
  
- **E2E Latency (End-to-End)**:
  - Complete inference time including prefill + decode
  - Total user-perceived latency
  
- **Throughput Breakdowns**:
  - **Prefill Throughput**: `(input_tokens * batch_size) / (prefill_time_sec)`
  - **Decode Throughput**: `(output_tokens * batch_size) / (decode_time_sec)`
  - **Total Throughput**: `(total_tokens * batch_size) / (total_time_sec)`

**Features**:
- Prefill metrics analysis (different context lengths and batch sizes)
- Decode metrics analysis (various generation lengths)
- Optimization impact comparison (tensor parallelism, LoRA, flash attention)
- Model size comparison (8B vs 13B models)
- JSON metrics report generation

**Usage**:
```bash
python scripts/metrics_showcase.py
```

### 3. `unified_simulation_examples.py` - Feature Demonstrations
**Purpose**: Comprehensive examples showing various feature combinations.

**Features**:
- Basic simulations (prefill, decode, chunked)
- Parallelism features (tensor_parallel, pipeline_parallel)
- Optimization features (lora, flash_attention, memory_offload)
- Feature discovery and compatibility checking
- Error handling demonstrations

**Usage**:
```bash
python scripts/unified_simulation_examples.py
```

### 4. `quick_demo.py` - Quick Start Guide
**Purpose**: Concise demonstration of key capabilities.

**Features**:
- Basic prefill simulation
- Decode with LoRA optimization
- Chunked with tensor parallelism
- Available features overview

**Usage**:
```bash
python scripts/quick_demo.py
```

## Performance Metrics Deep Dive

### Understanding the Metrics

#### 1. TTFT (Time to First Token)
- **Definition**: Time from request start to first token generation
- **Importance**: Critical for interactive applications (chatbots, code completion)
- **Typical Values**: 
  - Prefill-heavy: 20-200ms
  - Decode-optimized: 0.01-1ms per token
- **Optimization**: Reduce through efficient prefill, smaller models, or hardware acceleration

#### 2. TPOT (Time Per Output Token) 
- **Definition**: Average time to generate each output token
- **Importance**: Determines streaming generation speed
- **Calculation**: `total_decode_latency / output_tokens`
- **Typical Values**: 0.005-0.1ms for modern GPUs
- **Optimization**: Tensor parallelism, optimized attention mechanisms

#### 3. E2E Latency (End-to-End)
- **Definition**: Complete inference time (prefill + decode)
- **Importance**: Total user-perceived latency
- **Components**: `prefill_time + (output_tokens * TPOT)`
- **Optimization**: Balance between prefill efficiency and decode speed

#### 4. Throughput Metrics
- **Prefill Throughput**: Input processing speed (tokens/sec)
- **Decode Throughput**: Output generation speed (tokens/sec)  
- **Total Throughput**: Overall processing efficiency
- **Batch Impact**: Higher batch sizes typically increase throughput but may increase latency

### Practical Use Cases

#### Interactive Applications (Chatbots, Assistants)
- **Priority**: Low TTFT (< 100ms)
- **Secondary**: Consistent TPOT for smooth streaming
- **Optimization**: Smaller models, aggressive caching, speculative decoding

#### Batch Processing (Document Analysis, Translation)
- **Priority**: High total throughput
- **Strategy**: Large batch sizes, longer contexts
- **Optimization**: Pipeline parallelism, memory offloading

#### Real-time Applications (Code Completion, Auto-suggestions)
- **Priority**: Ultra-low TTFT (< 50ms)
- **Constraint**: Limited output tokens
- **Optimization**: Tiny models, edge deployment, caching

## Validation Results

The validation script confirms perfect accuracy across all test cases:

```
Validation Summary:
  Prefill        : ✓ PASS (0.00e+00 relative error)
  Decode         : ✓ PASS (0.00e+00 relative error)
  Chunked        : ✓ PASS (0.00e+00 relative error)

Overall Result: ✓ ALL TESTS PASSED
```

## Key Benefits

### 1. **Comprehensive Metrics**
- Industry-standard performance metrics (TTFT, TPOT, E2E)
- Separate prefill/decode throughput analysis
- Performance ratio calculations

### 2. **Consistent Interface**
- Single API for all simulation types
- Unified configuration and result formats
- Feature composition capabilities

### 3. **Validation & Reliability**
- Perfect accuracy validation (0.00e+00 error)
- Comprehensive test coverage
- Backward compatibility maintained

### 4. **Practical Insights**
- Optimization impact quantification
- Model size performance comparison
- Hardware configuration analysis

### 5. **Production Ready**
- JSON report generation for integration
- Error handling and edge cases
- Extensible architecture for new features

## Getting Started

1. **Quick validation**: `python scripts/validation_comparison.py`
2. **Explore metrics**: `python scripts/metrics_showcase.py`
3. **Learn features**: `python scripts/unified_simulation_examples.py`
4. **Quick demo**: `python scripts/quick_demo.py`

## Advanced Usage

### Custom Metrics Analysis
```python
from GenZ.simulation.engine import SimulationEngine
from GenZ.simulation.config import SimulationConfig

engine = SimulationEngine()
config = SimulationConfig(
    model="meta-llama/Llama-3.1-8B",
    features=["decode", "tensor_parallel"],
    simulation_params={
        "batch_size": 4,
        "input_tokens": 2048,
        "output_tokens": 1024,
        "system_name": "A100_40GB_GPU",
        "bits": "bf16",
        "tensor_parallel": 2
    }
)

result = engine.simulate(config)

# Calculate custom metrics
tpot = result.latency / config.simulation_params["output_tokens"]
decode_throughput = (config.simulation_params["output_tokens"] * 
                    config.simulation_params["batch_size"]) / (result.latency / 1000)

print(f"TPOT: {tpot:.3f} ms")
print(f"Decode Throughput: {decode_throughput:.1f} tokens/s")
```

### Performance Optimization Workflow
1. **Baseline measurement** with basic configuration
2. **Feature impact analysis** using optimization features
3. **Model size comparison** for accuracy vs speed tradeoffs
4. **Hardware scaling** with parallelism features
5. **Production tuning** based on specific use case requirements

The unified interface provides a comprehensive foundation for LLM inference performance analysis and optimization. 