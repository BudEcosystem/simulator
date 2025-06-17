# BudSimulator Setup Guide

## Overview

BudSimulator provides a fully automated setup process that handles everything from environment setup to launching the complete system. The setup script ensures that even non-technical users can get the system running with minimal effort.

## What the Setup Script Does

The automated setup script (`setup.py`) performs the following tasks:

1. **System Requirements Check**
   - Verifies Python 3.8+ is installed
   - Checks for Node.js and npm availability
   - Validates system compatibility

2. **Environment Setup**
   - Creates a Python virtual environment
   - Isolates dependencies from system Python

3. **Dependency Installation**
   - Installs all Python packages from requirements.txt
   - Installs frontend npm dependencies
   - Handles platform-specific requirements

4. **Database Setup**
   - Creates database directory structure
   - Copies pre-populated database with 45 AI models
   - Includes model configurations and analysis data

5. **LLM Configuration**
   - Interactive configuration for LLM provider
   - Supports OpenAI, Anthropic, Ollama, and custom APIs
   - Saves configuration to .env file

6. **System Testing**
   - Runs unit tests to verify installation
   - Performs system diagnostics
   - Validates all components are working

7. **Server Launch**
   - Starts backend API server (with automatic port selection)
   - Launches frontend development server
   - Opens the application in your default browser

8. **Health Verification**
   - Runs comprehensive health checks
   - Validates database connectivity
   - Ensures all endpoints are accessible

## Quick Start

### Option 1: One-Line Setup (Recommended)

```bash
python setup.py
```

That's it! Follow the interactive prompts and the system will be fully configured and running.

### Option 2: Platform-Specific Scripts

**Unix/Linux/macOS:**
```bash
./setup.sh
```

**Windows:**
```batch
setup.bat
```

## Configuration Options

During setup, you'll be asked to configure your LLM provider:

### OpenAI Configuration
- **Provider**: OpenAI
- **API Key**: Your OpenAI API key
- **Model**: gpt-4, gpt-3.5-turbo, etc.
- **API URL**: Automatically configured

### Anthropic Configuration
- **Provider**: Anthropic
- **API Key**: Your Anthropic API key
- **Model**: claude-3-opus-20240229, claude-3-sonnet-20240229
- **API URL**: Automatically configured

### Ollama Configuration (Local)
- **Provider**: Ollama
- **API Key**: Not required
- **Model**: llama2, mistral, codellama, etc.
- **API URL**: http://localhost:11434 (default)

### Custom API Configuration
- **Provider**: Custom
- **API Key**: As required by your provider
- **Model**: Your model name
- **API URL**: Your API endpoint

## Post-Setup Verification

After setup completes, you can verify the installation:

```bash
python verify_setup.py
```

This will check:
- Python version and dependencies
- Database integrity
- Environment configuration
- API server status
- Frontend server status

## Accessing the System

Once setup is complete:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Directory Structure

After setup, your directory will contain:

```
BudSimulator/
├── env/                # Virtual environment
├── data/               # Pre-populated database
├── logos/              # Model logos
├── frontend/           # React frontend
├── apis/               # FastAPI backend
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── .env                # Environment configuration
└── *.log               # Log files
```

## Restarting the System

To restart BudSimulator after stopping:

1. **Activate virtual environment:**
   ```bash
   # Unix/Linux/macOS
   source env/bin/activate
   
   # Windows
   env\Scripts\activate
   ```

2. **Start backend:**
   ```bash
   python run_api.py
   ```

3. **Start frontend (new terminal):**
   ```bash
   cd frontend
   npm start
   ```

## Troubleshooting

### Port Conflicts
The setup script automatically finds available ports if defaults (3000, 8000) are in use.

### Database Issues
```bash
# Reset database
rm -rf ~/.genz_simulator/db
python scripts/setup_database.py
```

### Dependency Issues
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## Environment Variables

The `.env` file contains:
- `LLM_PROVIDER`: Your LLM provider
- `LLM_API_KEY`: API key for authentication
- `LLM_MODEL`: Model to use
- `LLM_API_URL`: API endpoint

Additional optional variables:
- `BACKEND_PORT`: Custom backend port (default: 8000)
- `FRONTEND_PORT`: Custom frontend port (default: 3000)
- `LOG_LEVEL`: Logging level (default: INFO)

## System Requirements

### Minimum Requirements
- Python 3.8+
- Node.js 14+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.10+
- Node.js 18+
- 8GB RAM
- 5GB free disk space
- Modern web browser

## Security Notes

- API keys are stored locally in `.env`
- Never commit `.env` to version control
- Use environment-specific configurations for production
- Enable CORS restrictions for production deployments

## Getting Help

1. Run verification script: `python verify_setup.py`
2. Check API docs: http://localhost:8000/docs
3. Review logs in the console output
4. Check the README.md for additional information

## Next Steps

After successful setup:
1. Browse available AI models
2. Get hardware recommendations
3. Run performance benchmarks
4. Analyze model architectures
5. Compare different configurations

Enjoy using BudSimulator! 