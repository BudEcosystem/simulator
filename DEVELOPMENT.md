# GenZ Development Setup Guide

This guide will help you set up GenZ locally with automatic build refreshing for continuous development.

## Quick Start

1. **Run the setup script**:
   ```bash
   ./setup_dev.sh
   ```

2. **Start the file watcher** (in a separate terminal):
   ```bash
   python watch_and_rebuild.py
   ```

3. **Run the web interface**:
   ```bash
   streamlit run Website/Home.py
   ```

## Manual Setup

If you prefer manual setup:

### 1. Install Development Dependencies
```bash
pip install watchdog pytest pytest-cov streamlit
```

### 2. Install GenZ in Development Mode
```bash
pip install -e .
```

This creates a link to your local code, so changes are immediately reflected without reinstalling.

### 3. Start the Automatic Build Watcher
```bash
python watch_and_rebuild.py
```

The watcher monitors:
- All Python files in `GenZ/` directory
- All Python files in `Systems/` directory  
- The `pyproject.toml` file

When changes are detected, it automatically reinstalls the package.

## Development Workflow

### Running the Application
```bash
# Start the Streamlit web interface
streamlit run Website/Home.py
```

The web interface includes:
- Home page with overview
- Usecase Comparison
- Model Comparison
- Platform Comparison
- Technology Comparison

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_prefill_time.py

# Run with coverage
pytest --cov=GenZ tests/
```

### Key Development Commands

| Command | Description |
|---------|-------------|
| `pip install -e .` | Install GenZ in editable mode |
| `python watch_and_rebuild.py` | Start file watcher for auto-rebuild |
| `streamlit run Website/Home.py` | Launch web interface |
| `pytest tests/` | Run all tests |
| `pytest tests/test_<name>.py` | Run specific test |

## Project Structure

```
BudSimulator/
├── GenZ/                    # Core GenZ package
│   ├── system.py           # Hardware abstraction
│   ├── operators.py        # Computational operators
│   ├── parallelism.py      # Parallelism configurations
│   ├── llm_prefill.py      # Prefill phase modeling
│   ├── llm_decode.py       # Decode phase modeling
│   └── Models/             # Pre-configured models
├── Systems/                 # Hardware configurations
├── Website/                 # Streamlit web interface
├── tests/                   # Test suite
└── watch_and_rebuild.py    # Development watcher
```

## Making Changes

1. **Edit source files** in `GenZ/` or `Systems/`
2. **Watcher detects changes** and rebuilds automatically
3. **Refresh web interface** or re-run scripts to see changes
4. **Run tests** to ensure nothing is broken

## Common Development Tasks

### Adding a New Model
1. Add model definition to appropriate file in `GenZ/Models/Model_sets/`
2. Import in `GenZ/Models/default_models.py`
3. Test with: `python -c "from GenZ.Models import get_model_df; df = get_model_df('your_model_name')"`

### Adding a New Operator
1. Create operator class in `GenZ/operators.py`
2. Inherit from `operator_base.BaseOperator`
3. Implement required methods
4. Add tests in `tests/`

### Modifying System Configurations
1. Edit `Systems/system_configs.py`
2. Changes are immediately available after rebuild

## Troubleshooting

### Import Errors
- Ensure you've installed in editable mode: `pip install -e .`
- Check that the watcher is running if you expect auto-rebuild

### Watcher Not Detecting Changes
- Verify watchdog is installed: `pip install watchdog`
- Check that you're editing files in watched directories
- Restart the watcher if needed

### Tests Failing
- Run `pytest -v` for verbose output
- Check golden test files in `tests/golden/`
- Ensure all dependencies are installed

## Tips for Efficient Development

1. **Keep the watcher running** in a dedicated terminal
2. **Use the web interface** for quick visual feedback
3. **Write tests** for new features before implementing
4. **Check existing patterns** in the codebase before adding new code
5. **Run relevant tests** after making changes

## Additional Resources

- [CLAUDE.md](../CLAUDE.md) - AI assistant instructions
- [pyproject.toml](pyproject.toml) - Package configuration
- [requirements.txt](requirements.txt) - Python dependencies