# BudSimulator

BudSimulator is an advanced AI model simulation and benchmarking platform built on top of the GenZ framework. It provides comprehensive tools for evaluating, comparing, and optimizing AI models with hardware recommendations.

## Features

- 🚀 **Model Benchmarking**: Comprehensive performance analysis across different hardware configurations
- 💻 **Hardware Recommendations**: Intelligent suggestions for optimal hardware based on your requirements
- 📊 **Interactive Dashboard**: Beautiful web interface for visualizing model performance
- 🔍 **Model Analysis**: Detailed insights into model architecture and capabilities
- 🛠️ **Easy Setup**: One-click automated installation and configuration

## Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm
- Git

## Quick Start

### Automated Setup (Recommended)

Simply run the automated setup script:

```bash
python setup.py
```

This will:
1. Check system requirements
2. Create a virtual environment
3. Install all Python and npm dependencies
4. Set up the database with pre-populated model data
5. Configure your LLM provider (OpenAI, Anthropic, Ollama, etc.)
6. Run system tests
7. Start both backend and frontend servers
8. Open the application in your browser

### Manual Setup

If you prefer manual setup:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BudSimulator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Set up database**
   ```bash
   python scripts/setup_database.py
   ```

6. **Configure environment**
   Create a `.env` file with your LLM configuration:
   ```env
   LLM_PROVIDER=openai
   LLM_API_KEY=your-api-key
   LLM_MODEL=gpt-4
   LLM_API_URL=https://api.openai.com/v1/chat/completions
   ```

7. **Start the servers**
   
   Backend:
   ```bash
   python run_api.py
   ```
   
   Frontend (new terminal):
   ```bash
   cd frontend
   npm start
   ```

## Usage

Once the setup is complete:

1. **Access the application**: http://localhost:3000
2. **API Documentation**: http://localhost:8000/docs
3. **Model Dashboard**: Browse and analyze AI models
4. **Hardware Recommendations**: Get optimal hardware suggestions
5. **Benchmarking**: Run performance tests on different configurations

## Project Structure

```
BudSimulator/
├── apis/               # Backend API endpoints
├── frontend/           # React frontend application
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── data/               # Pre-populated database
├── models/             # Model definitions
├── services/           # Business logic services
├── utils/              # Utility functions
├── setup.py            # Automated setup script
└── requirements.txt    # Python dependencies
```

## Configuration

### LLM Providers

BudSimulator supports multiple LLM providers:

- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude models
- **Ollama**: Local models
- **Custom**: Any OpenAI-compatible API

### Environment Variables

- `LLM_PROVIDER`: Your LLM provider (openai, anthropic, ollama, custom)
- `LLM_API_KEY`: API key for your provider
- `LLM_MODEL`: Model name to use
- `LLM_API_URL`: API endpoint URL

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use, the setup script will automatically find available ports.

### Database Issues

If you encounter database issues:
```bash
rm -rf ~/.genz_simulator/db
python scripts/setup_database.py
```

### Frontend Build Issues

Clear npm cache and reinstall:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation at http://localhost:8000/docs
- Review the logs in the console output

## Acknowledgments

Built on top of the GenZ framework for advanced AI model management. 