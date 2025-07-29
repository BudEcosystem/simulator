# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BudSimulator is a comprehensive AI model simulation and benchmarking platform built on top of the GenZ-LLM Analyzer framework. It provides analytical performance estimates for Large Language Model (LLM) inference on various hardware platforms through a full-stack application with React frontend and FastAPI backend.

Key components:
- **GenZ-LLM Analyzer (genz_llm)**: Core Python package for LLM performance modeling
- **FastAPI Backend**: REST API with hardware recommendations and model analysis
- **React Frontend**: Interactive UI for hardware selection and use case management
- **Streamlit Interface**: Alternative web dashboard for model comparison

## Development Commands

### Installation
```bash
# Automated setup (recommended)
python BudSimulator/setup.py

# Manual installation in development mode  
pip install -e BudSimulator/

# Or install from PyPI
pip install genz-llm
```

### Running the Application
```bash
# Backend API (stable mode)
cd BudSimulator
python run_api.py

# Backend API (with hot-reload for development)
cd BudSimulator
RELOAD=true python run_api.py  # Unix/Linux/macOS
set RELOAD=true && python run_api.py  # Windows

# Frontend (in frontend directory)
cd BudSimulator/frontend
npm install  # First time only
npm start

# Alternative Streamlit interface
streamlit run BudSimulator/Website/Home.py
```

### Testing
```bash
# Run all tests
pytest BudSimulator/tests/ -v

# Run specific test file
pytest BudSimulator/comprehensive_api_test.py

# Test CPU functionality
python BudSimulator/cpu_test.py
python BudSimulator/cpu_test_detailed.py

# Test model parameters
python BudSimulator/model_param_test.py
```

### Linting and Type Checking
```bash
# Python linting (if ruff is installed)
ruff check BudSimulator/

# Frontend linting
cd BudSimulator/frontend
npm run lint  # Note: Currently no explicit lint script, uses eslint via react-scripts

# Python type checking (if mypy is installed)
mypy BudSimulator/
```

## High-Level Architecture

### Core Components

1. **System Abstraction** (`BudSimulator/GenZ/system.py`)
   - Represents hardware accelerators with compute/memory capabilities
   - Handles different precision formats (fp32, bf16, int8) with compute/memory multipliers
   - Integrates with compute engines (GenZ, Scale-sim) and collective strategies (GenZ, ASTRA-SIM)

2. **Operator Framework** (`BudSimulator/GenZ/operator_base.py`, `BudSimulator/GenZ/operators.py`)
   - Base class for all computational operations with roofline analysis
   - Concrete operators: FC, GEMM, Logit, Attend, CONV2D, Einsum
   - Special operators for synchronization and layer repetition

3. **Parallelism Management** (`BudSimulator/GenZ/parallelism.py`)
   - ParallelismConfig handles: tensor_parallel, pipeline_parallel, data_parallel, expert_parallel
   - Supports hierarchical parallelism (e.g., "TP{4}_EP{2}_PP{1}")
   - Collective operations: AllReduce, All2All, AllGather for inter-accelerator communication

4. **Performance Modeling** (`BudSimulator/GenZ/LLM_inference/`)
   - Models prefill and decode phases separately (`llm_prefill.py`, `llm_decode.py`)
   - Calculates memory requirements (weights + KV cache)
   - Handles memory offloading when capacity exceeded
   - Returns latency, throughput, and runtime breakdown

5. **Model Management** (`BudSimulator/GenZ/Models/`)
   - Registry of pre-configured models (Meta, Google, Microsoft, etc.)
   - Creates model definitions with specified parallelism configurations
   - Supports MHA (Multi-Head Attention) and Mamba architectures

### API Layer (`BudSimulator/apis/`)
- FastAPI application with automatic OpenAPI documentation (/docs)
- Routers for models, hardware, and usecases
- CORS-enabled for frontend integration
- Static file serving for logos

### Database (`BudSimulator/src/db/`)
- SQLAlchemy models for hardware and model data
- Pre-populated SQLite database at `BudSimulator/data/prepopulated.db`
- Alembic for migrations

### Frontend (`BudSimulator/frontend/`)
- React 18.2 with TypeScript
- Tailwind CSS for styling
- Components for hardware browsing, usecase management, and AI memory calculation
- Proxy configuration to backend API (port 8000)

### Analysis Flow

1. Model Definition → CSV representation of operator sequences
2. Performance Analysis → Per-operator roofline analysis
3. Aggregation → Runtime breakdown by component (MHA, FFN, Embedding, Collective)
4. Visualization → Web interface for interactive exploration

### Key Interfaces

- `get_model_df()`: Load model and compute performance metrics
- `prefill_moddeling()` / `decode_moddeling()`: End-to-end performance estimation
- `System()`: Hardware platform definition
- `ParallelismConfig()`: Distributed execution configuration

## Common Development Tasks

### Adding a New Model
1. Create model definition in `BudSimulator/GenZ/Models/Model_sets/`
2. Register in model registry
3. Update database if needed via migration
4. Add frontend display logic if required

### Adding a New Hardware Platform
1. Define system specs in `BudSimulator/GenZ/system.py` format
2. Add to database via migration or seed script
3. Update hardware recommendation logic in `BudSimulator/src/services/`
4. Test with existing models

### Modifying API Endpoints
1. Update router in `BudSimulator/apis/routers/`
2. Modify corresponding service in `BudSimulator/src/`
3. Update frontend API service and TypeScript types
4. Run tests to ensure compatibility

### Working with the Database
1. Create new Alembic migration: `alembic revision -m "description"`
2. Apply migrations: `alembic upgrade head`
3. Database is at `BudSimulator/data/prepopulated.db`

## Important Files and Directories

- `BudSimulator/run_api.py` - Main API entry point
- `BudSimulator/setup.py` - Automated setup script
- `BudSimulator/apis/` - FastAPI application and routers
- `BudSimulator/src/` - Core business logic and services
- `BudSimulator/GenZ/` - Performance modeling framework
- `BudSimulator/frontend/` - React application
- `BudSimulator/Website/` - Streamlit alternative UI
- `BudSimulator/tests/` - Test suite
- `BudSimulator/data/prepopulated.db` - SQLite database
- `BudSimulator/config/env.template` - Environment configuration template

## Code Style Guidelines

- **Python**: Follow PEP 8, use type hints where appropriate
- **TypeScript**: Use strict mode, define interfaces for all API responses
- **React**: Functional components with hooks
- **API**: RESTful conventions, consistent error handling
- Document complex algorithms with inline comments
- Keep functions focused and single-purpose

## Important Implementation Details

### Usecase ID Handling
The codebase uses `unique_id` (string) for usecases, not numeric `id`. This is critical for proper functionality:
```python
# Correct
usecase = db.query(Usecase).filter(Usecase.unique_id == unique_id).first()

# Incorrect (will fail)
usecase = db.query(Usecase).filter(Usecase.id == id).first()
```

### Environment Configuration
Create a `.env` file based on `BudSimulator/config/env.template`:
- LLM_PROVIDER: openai, anthropic, ollama, or custom
- LLM_API_KEY: Your provider API key
- LLM_MODEL: Model name (e.g., gpt-4, claude-3-opus-20240229)
- LLM_API_URL: Provider endpoint URL

### Package Management
The project uses:
- Python dependencies in `BudSimulator/requirements.txt`
- Frontend dependencies in `BudSimulator/frontend/package.json`
- Core package configuration in `BudSimulator/pyproject.toml` (genz_llm v0.0.16)

### Testing Approach
- **Backend**: pytest with test files in various locations
- **Frontend**: React Testing Library and Jest via react-scripts
- **Integration**: API testing with `comprehensive_api_test.py`
- **CPU Testing**: Specialized tests in `cpu_test.py` and `cpu_test_detailed.py`

## Current Development Context

- Repository is at `/home/budadmin/simulator`
- Main project directory is `BudSimulator/`
- Currently on branch `main`
- Uses virtual environment at `BudSimulator/env/`