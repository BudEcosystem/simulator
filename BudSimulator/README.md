# BudSimulator

BudSimulator is a comprehensive AI model simulation and benchmarking platform that provides analytical performance estimates for Large Language Model (LLM) inference on various hardware platforms. Built on top of the [llm-memory-calculator](https://github.com/yourusername/llm-memory-calculator) engine, it offers a full-stack application with React frontend and FastAPI backend for interactive exploration of LLM deployment scenarios.

## Features

### 🚀 Core Capabilities
- **Memory Estimation**: Calculate precise memory requirements for LLMs across different architectures
- **Performance Modeling**: Estimate inference latency and throughput for prefill and decode phases
- **Hardware Recommendations**: Get optimal hardware suggestions based on model requirements
- **Multi-Model Comparison**: Compare multiple models side-by-side for memory and performance
- **Use Case Management**: Create and manage different deployment scenarios
- **Interactive Dashboard**: Beautiful web interface for visualizing model performance

### 🎯 Key Components
- **Web Dashboard**: Interactive React frontend for visual exploration
- **REST API**: FastAPI backend with comprehensive endpoints
- **Streamlit Interface**: Alternative dashboard for model comparisons
- **Database Management**: SQLite database for hardware and model configurations
- **HuggingFace Integration**: Direct loading of model configurations from HuggingFace Hub

### 🔧 Technical Features
- Support for various model architectures (Transformers, Mamba, Hybrid, Diffusion)
- Multiple precision formats (fp32, fp16, bf16, int8, int4, etc.)
- Parallelism strategies (Tensor, Pipeline, Data, Expert parallelism)
- CPU and GPU performance modeling
- LoRA adapter support
- Batch processing capabilities

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

5. **Initialize database**
   ```bash
   python scripts/setup_database.py
   ```

6. **Configure environment** (optional)
   Create a `.env` file based on `config/env.template`:
   ```env
   LLM_PROVIDER=openai
   LLM_API_KEY=your-api-key-here
   LLM_MODEL=gpt-4
   ```

## Usage

### Running the Application

#### Start the Backend API:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`. Access the interactive API documentation at `http://localhost:8000/docs`.

#### Start the Frontend (in a new terminal):
```bash
cd frontend
npm start
```

The React app will open at `http://localhost:3000`.

#### Alternative: Streamlit Interface
```bash
streamlit run Website/Home.py
```

### API Examples

#### Calculate Memory for a Model
```python
import requests

response = requests.post("http://localhost:8000/api/models/calculate-memory", json={
    "model_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "model_type": "llama"
    },
    "batch_size": 1,
    "sequence_length": 2048,
    "precision": "fp16"
})

print(response.json())
```

#### Get Hardware Recommendations
```python
response = requests.post("http://localhost:8000/api/hardware/recommend", json={
    "model_size_gb": 13.5,
    "required_memory_gb": 27.0,
    "use_case": "inference"
})

print(response.json())
```

### Command Line Interface

For development and testing:

```bash
# Test memory calculation
python -c "
from src.bud_models import estimate_memory
config = {'hidden_size': 4096, 'num_hidden_layers': 32, 'vocab_size': 32000}
report = estimate_memory(config, seq_length=2048)
print(f'Total memory: {report.total_memory_gb:.2f} GB')
"

# Test CPU inference
python cpu_test.py

# Run comprehensive API tests
python comprehensive_api_test.py
```

## Architecture

### High-Level Overview

```
BudSimulator/
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── services/      # API services
│   │   └── types/         # TypeScript types
│   └── public/            # Static assets
├── apis/                  # FastAPI backend
│   ├── routers/          # API endpoints
│   └── schemas.py        # Pydantic schemas
├── src/                   # Core business logic
│   ├── db/               # Database models and management
│   ├── utils/            # Utility functions
│   └── bud_models.py     # Model interfaces
├── Website/              # Streamlit interface
│   └── pages/            # Streamlit pages
├── data/                 # SQLite database
├── scripts/              # Utility scripts
└── tests/                # Test files
```

### Core Dependencies

- **llm-memory-calculator**: Core LLM performance modeling engine
- **FastAPI**: Modern web API framework
- **React**: Frontend framework with TypeScript
- **SQLAlchemy**: Database ORM
- **Streamlit**: Alternative web interface
- **HuggingFace Hub**: Model configuration loading

## Development

### Adding a New Model

1. Import from HuggingFace:
```python
from src.db import HuggingFaceModelImporter
importer = HuggingFaceModelImporter()
importer.import_model("meta-llama/Llama-3-8B")
```

2. Or add manually to the database via the API or scripts.

### Adding Hardware Configurations

1. Create a hardware configuration:
```python
hardware = {
    "name": "NVIDIA H200",
    "memory_gb": 141,
    "memory_bandwidth_gb_s": 4800,
    "fp16_tflops": 1979,
    "int8_tops": 3958
}
```

2. Add via API or database scripts.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
python cpu_test.py
python comprehensive_api_test.py
python model_param_test.py
```

## API Documentation

The API provides comprehensive endpoints for model analysis and hardware recommendations:

### Key Endpoints

- `POST /api/models/calculate-memory` - Calculate memory requirements
- `POST /api/models/analyze` - Analyze model from HuggingFace
- `GET /api/models/list` - List available models
- `POST /api/hardware/recommend` - Get hardware recommendations
- `GET /api/hardware` - List available hardware
- `POST /api/usecases` - Create deployment scenarios
- `GET /api/usecases/{id}/recommendations` - Get recommendations for a use case

Full API documentation is available at `http://localhost:8000/docs` when running the backend.

## Configuration

### Environment Variables

Create a `.env` file based on `config/env.template`:

```env
# LLM Provider settings (optional, for AI-powered features)
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4

# Database settings
DATABASE_PATH=data/prepopulated.db

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Frontend settings
REACT_APP_API_URL=http://localhost:8000
```

### Database Configuration

The SQLite database contains:
- Pre-populated hardware configurations
- Model metadata and configurations
- Use case scenarios
- Performance benchmarks

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **Database errors**: Run `python scripts/setup_database.py` to reinitialize
3. **Frontend not connecting**: Check API is running on port 8000
4. **Memory calculation errors**: Verify model configuration format

### Debug Mode

Run the API with debug logging:
```bash
RELOAD=true python run_api.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Python: Follow PEP 8, use type hints
- TypeScript: Use strict mode, define interfaces
- React: Functional components with hooks
- API: RESTful conventions, consistent error handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [llm-memory-calculator](https://github.com/yourusername/llm-memory-calculator)
- Inspired by the GenZ-LLM Analyzer framework
- Thanks to the HuggingFace team for model hosting
- Community contributors and testers

## Support

- **Documentation**: See the `/docs` directory for detailed guides
- **Issues**: Report bugs at [GitHub Issues](https://github.com/yourusername/BudSimulator/issues)
- **Questions**: Ask in [Discussions](https://github.com/yourusername/BudSimulator/discussions)

## Roadmap

- [ ] Support for more model architectures
- [ ] Enhanced visualization capabilities  
- [ ] Multi-GPU simulation support
- [ ] Cost optimization features
- [ ] Cloud deployment templates
- [ ] Real-time monitoring integration
- [ ] Model fine-tuning recommendations
- [ ] Energy efficiency metrics

---

**Note**: This project is under active development. Features and APIs may change.