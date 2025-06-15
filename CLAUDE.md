# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenZ-LLM Analyzer (genz_llm) is a Python package that models Large Language Model (LLM) performance on various hardware platforms. It provides analytical performance estimates through a three-component approach: Model Profiler, NPU Characterizer, and Platform Characterizer.

## Development Commands

### Installation
```bash
# Install in development mode from BudSimulator directory
pip install -e .

# Or install from PyPI
pip install genz-llm
```

### Running the Application
```bash
# Start the Streamlit web interface
streamlit run BudSimulator/Website/Home.py
```

### Testing
```bash
# Run tests (using pytest convention)
pytest BudSimulator/tests/

# Run a specific test file
pytest BudSimulator/tests/test_prefill_time.py
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

## Documentation Location

Documentation should be placed in the @docs directory (per .cursor/rules/documentation.mdc).