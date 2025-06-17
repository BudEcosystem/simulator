#!/bin/bash
# BudSimulator Setup Script for Unix-based systems

echo "🚀 Starting BudSimulator Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv env
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source env/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
print_info "Installing dependencies from requirements.txt..."
# Install critical dependencies first
pip install psutil python-multipart aiofiles colorama

# Filter out problematic requirements and install the rest
grep -v '^genz>=' requirements.txt | grep -v '^genz-llm' > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# Install the local GenZ package in editable mode
print_info "Installing local GenZ package..."
if [ -d "GenZ" ]; then
    pip install -e .
    print_success "GenZ package installed in editable mode"
fi

# Run the main setup script
print_info "Running main setup script..."
python setup.py

# Verify installation
print_info "Verifying installation..."
python verify_setup.py

print_success "Setup complete!"
echo ""
echo "🔧 Next steps:"
echo "  1. To start the API server: python run_api.py"
echo "  2. To enable hot-reload (development): RELOAD=true python run_api.py"
echo "  3. To run tests: pytest tests/"
echo ""
echo "⚠️  Note: The virtual environment is activated. To deactivate, run: deactivate" 