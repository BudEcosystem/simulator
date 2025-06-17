#!/bin/bash

# Development setup script for BudSimulator
echo "🚀 Setting up BudSimulator development environment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Ensure we're in the virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -d "env" ]; then
        print_info "Activating virtual environment..."
        source env/bin/activate
    else
        print_info "Creating virtual environment..."
        python3 -m venv env
        source env/bin/activate
    fi
fi

# Install critical dependencies first
print_info "Installing critical dependencies..."
pip install psutil python-multipart aiofiles colorama

# Install development dependencies
print_info "Installing development dependencies..."
pip install watchdog pytest pytest-cov pytest-asyncio python-dotenv

# Install all dependencies except problematic ones
print_info "Installing project dependencies..."
grep -v '^genz>=' requirements.txt | grep -v '^genz-llm' > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# Install GenZ in editable mode
print_info "Installing GenZ in development mode..."
pip install -e .

# Install Streamlit if not already installed
print_info "Ensuring Streamlit is installed..."
pip install streamlit

# Make scripts executable
chmod +x watch_and_rebuild.py
chmod +x run_api.py

print_success "Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  - Run API server (stable): python run_api.py"
echo "  - Run API server (hot-reload): RELOAD=true python run_api.py"
echo "  - Run watcher: python watch_and_rebuild.py"
echo "  - Run web UI: streamlit run Website/Home.py"
echo "  - Run tests: pytest tests/"
echo "  - Run specific test: pytest tests/test_prefill_time.py"
echo ""
echo "💡 Tips:"
echo "  - The server runs without hot-reload by default to avoid virtual env issues"
echo "  - Use RELOAD=true only when actively developing"
echo "  - The virtual environment is now activated"