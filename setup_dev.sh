#!/bin/bash

# Development setup script for GenZ
echo "🚀 Setting up GenZ development environment..."

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install watchdog pytest pytest-cov

# Install GenZ in editable mode
echo "📦 Installing GenZ in development mode..."
pip install -e .

# Install Streamlit if not already installed
echo "📦 Ensuring Streamlit is installed..."
pip install streamlit

# Make watcher script executable
chmod +x watch_and_rebuild.py

echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  - Run watcher: python watch_and_rebuild.py"
echo "  - Run web UI: streamlit run Website/Home.py"
echo "  - Run tests: pytest tests/"
echo "  - Run specific test: pytest tests/test_prefill_time.py"