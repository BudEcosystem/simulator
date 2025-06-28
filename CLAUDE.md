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
python setup.py

# Manual installation in development mode
pip install -e .

# Or install from PyPI
pip install genz-llm
```

### Running the Application
```bash
# Backend API (stable mode)
python run_api.py

# Backend API (with hot-reload for development)
RELOAD=true python run_api.py  # Unix/Linux/macOS
set RELOAD=true && python run_api.py  # Windows

# Frontend (in frontend directory)
cd frontend
npm start

# Alternative Streamlit interface
streamlit run BudSimulator/Website/Home.py
```

### Testing
```bash
# Run all tests
pytest BudSimulator/tests/ -v

# Run a specific test file
pytest BudSimulator/tests/test_prefill_time.py

# Run tests with coverage
pytest BudSimulator/tests/ --cov=BudSimulator
```

### Frontend Development
```bash
cd frontend
npm install       # Install dependencies
npm run build     # Production build
npm test          # Run tests
npm run lint      # Run linter
```

## High-Level Architecture

### Core Components

1. **System Abstraction** (`GenZ/system.py`)
   - Represents hardware accelerators with compute/memory capabilities
   - Handles different precision formats (fp32, bf16, int8) with compute/memory multipliers
   - Integrates with compute engines (GenZ, Scale-sim) and collective strategies (GenZ, ASTRA-SIM)

2. **Operator Framework** (`GenZ/operator_base.py`, `GenZ/operators.py`)
   - Base class for all computational operations with roofline analysis
   - Concrete operators: FC, GEMM, Logit, Attend, CONV2D, Einsum
   - Special operators for synchronization and layer repetition

3. **Parallelism Management** (`GenZ/parallelism.py`)
   - ParallelismConfig handles: tensor_parallel, pipeline_parallel, data_parallel, expert_parallel
   - Supports hierarchical parallelism (e.g., "TP{4}_EP{2}_PP{1}")
   - Collective operations: AllReduce, All2All, AllGather for inter-accelerator communication

4. **Performance Modeling** (`GenZ/llm_prefill.py`, `GenZ/llm_decode.py`)
   - Models prefill and decode phases separately
   - Calculates memory requirements (weights + KV cache)
   - Handles memory offloading when capacity exceeded
   - Returns latency, throughput, and runtime breakdown

5. **Model Management** (`GenZ/Models/`)
   - Registry of pre-configured models (Meta, Google, Microsoft, etc.)
   - Creates model definitions with specified parallelism configurations
   - Supports MHA (Multi-Head Attention) and Mamba architectures

### API Layer (`apis/`)
- FastAPI application with automatic OpenAPI documentation (/docs)
- Routers for models, hardware, and usecases
- CORS-enabled for frontend integration
- Static file serving for logos

### Database (`src/db/`)
- SQLAlchemy models for hardware and model data
- Pre-populated SQLite database
- Alembic for migrations

### Frontend (`frontend/`)
- React 18.2 with TypeScript
- Tailwind CSS for styling
- Components for hardware browsing, usecase management, and AI memory calculation
- Proxy configuration to backend API

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
1. Create model definition in `GenZ/Models/Model_sets/`
2. Register in model registry
3. Update database if needed via migration
4. Add frontend display logic if required

### Adding a New Hardware Platform
1. Define system specs in `GenZ/system.py` format
2. Add to database via migration or seed script
3. Update hardware recommendation logic in `src/services/`
4. Test with existing models

### Modifying API Endpoints
1. Update router in `apis/routers/`
2. Modify corresponding service in `src/`
3. Update frontend API service and TypeScript types
4. Run tests to ensure compatibility

### Working with the Database
1. Create new Alembic migration: `alembic revision -m "description"`
2. Apply migrations: `alembic upgrade head`
3. Database is at `data/prepopulated.db`

## Important Files and Directories

- `run_api.py` - Main API entry point
- `setup.py` - Automated setup script
- `apis/` - FastAPI application and routers
- `src/` - Core business logic and services
- `GenZ/` - Performance modeling framework
- `frontend/` - React application
- `Website/` - Streamlit alternative UI
- `tests/` - Test suite with golden files
- `data/prepopulated.db` - SQLite database
- `.env.example` - Environment configuration template

## Testing Approach

- **Backend**: pytest with fixtures and golden file testing
- **Frontend**: React Testing Library and Jest
- **Integration**: API endpoint testing with FastAPI TestClient
- **Performance**: Golden file comparison for model predictions

When working with tests:
- Run existing tests before making changes
- Add tests for new functionality
- Use golden test files in `tests/golden/` for expected outputs
- Test both unit and integration levels

## Code Style Guidelines

- **Python**: Follow PEP 8, use type hints where appropriate
- **TypeScript**: Use strict mode, define interfaces for all API responses
- **React**: Functional components with hooks
- **API**: RESTful conventions, consistent error handling
- Document complex algorithms with inline comments
- Keep functions focused and single-purpose

## Recent Changes Context

The current branch `feature/usecase-details-fix` includes fixes for the usecase details page to use `unique_id` instead of numeric `id`. When working with usecases, ensure you're using the string-based unique identifiers.

## Documentation Location

Documentation should be placed in the @docs directory (per .cursor/rules/documentation.mdc).