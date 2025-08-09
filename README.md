# LLM Performance Simulator

A comprehensive ecosystem for simulating and analyzing Large Language Model (LLM) performance across diverse hardware platforms. This repository contains two main projects that work together to provide accurate memory estimation and performance modeling for LLM deployment.

## 🏗️ Repository Structure

```
simulator/
├── BudSimulator/           # Full-stack web application for LLM analysis
│   ├── frontend/          # React TypeScript UI
│   ├── apis/              # FastAPI backend
│   └── Website/           # Streamlit dashboard
│
└── llm-memory-calculator/  # Core LLM performance modeling engine
    └── src/               # Python package with GenZ framework
```

## 📦 Projects Overview

### 1. llm-memory-calculator
The core engine that provides:
- **Memory Calculation**: Precise memory requirements for any LLM architecture
- **Performance Modeling**: Latency and throughput estimation
- **Architecture Support**: Transformers, Mamba, Hybrid, Diffusion models
- **Hardware Abstraction**: CPU and GPU performance modeling
- **Parallelism Strategies**: Tensor, Pipeline, Data, and Expert parallelism

[View llm-memory-calculator README](./llm-memory-calculator/README.md)

### 2. BudSimulator
A full-stack web application built on top of llm-memory-calculator:
- **Web Dashboard**: Interactive React frontend
- **REST API**: FastAPI backend with comprehensive endpoints
- **Database**: SQLite with pre-populated hardware configs
- **Model Management**: HuggingFace integration
- **Use Case Scenarios**: Deployment planning tools

[View BudSimulator README](./BudSimulator/README.md)

## 🚀 Quick Start

### Option 1: Full Stack Application (BudSimulator)
```bash
cd BudSimulator
python setup.py  # Automated setup script
```

This will install everything and launch the web application.

### Option 2: Python Package Only (llm-memory-calculator)
```bash
cd llm-memory-calculator
pip install -e .
```

Then use in Python:
```python
from llm_memory_calculator import estimate_memory

config = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 32000
}
report = estimate_memory(config, seq_length=2048)
print(f"Total memory: {report.total_memory_gb:.2f} GB")
```

### Option 3: Docker Deployment (Recommended for Production)
```bash
# Build and run with Docker
docker build -t budsimulator .
docker run -p 8000:8000 -p 3000:3000 budsimulator

# Or use docker-compose
docker-compose up --build

# For domain deployment (e.g., simulator.bud.studio)
./deploy-domain.sh
```

## 🎯 Use Cases

### For ML Engineers
- Estimate GPU memory requirements before training/deployment
- Compare memory usage across different model architectures
- Plan distributed training strategies

### For Infrastructure Teams
- Get hardware recommendations for specific models
- Optimize resource allocation
- Plan scaling strategies

### For Researchers
- Analyze performance characteristics of new architectures
- Compare efficiency across different model designs
- Benchmark hardware platforms

## 📊 Key Features

### Memory Analysis
- Weight memory calculation
- KV cache estimation
- Activation memory tracking
- Gradient memory (for training)
- Optimizer state sizing

### Performance Metrics
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- Throughput (tokens/second)
- Latency breakdown by component

### Hardware Support
- NVIDIA GPUs (A100, H100, etc.)
- AMD GPUs (MI300X, etc.)
- Google TPUs
- Intel/AMD CPUs
- Custom hardware configs

### Model Architectures
- Transformer models (GPT, LLaMA, etc.)
- Mixture of Experts (MoE)
- State Space Models (Mamba)
- Hybrid architectures
- Diffusion models

## 🔧 Architecture

### System Design
```
┌─────────────────┐     ┌──────────────────┐
│   React Web UI  │────▶│  FastAPI Backend │
└─────────────────┘     └──────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ llm-memory-calculator  │
                    │   (Core Engine)        │
                    └────────────────────────┘
                                 │
                    ┌────────────┴───────────┐
                    ▼                        ▼
            ┌──────────────┐        ┌──────────────┐
            │ GenZ Models  │        │  Hardware    │
            │              │        │  Configs     │
            └──────────────┘        └──────────────┘
```

### Technology Stack
- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Core Engine**: Python, NumPy, Pandas
- **Visualization**: Plotly, Streamlit
- **Database**: SQLite
- **Testing**: Pytest, Jest

## 📖 Documentation

- [API Documentation](./BudSimulator/apis/README.md)
- [Frontend Guide](./BudSimulator/frontend/README.md)
- [Adding Custom Hardware](./llm-memory-calculator/docs/ADD_NEW_CPU_GUIDE.md)
- [Performance API Guide](./llm-memory-calculator/docs/PERFORMANCE_API_GUIDE.md)
- [Deployment Guide](./DEPLOYMENT.md)

## 🧪 Examples

### Calculate Memory for LLaMA Model
```python
from llm_memory_calculator import ModelMemoryCalculator

calculator = ModelMemoryCalculator()
config = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 32000
}

report = calculator.calculate_memory(
    config,
    seq_length=2048,
    batch_size=4,
    precision="fp16"
)

print(f"Total Memory: {report.total_memory_gb:.2f} GB")
print(f"Breakdown: {report.memory_breakdown}")
```

### Performance Estimation
```python
from llm_memory_calculator import estimate_prefill_performance

result = estimate_prefill_performance(
    model="llama2_7b",
    batch_size=1,
    input_tokens=2048,
    system_name="A100_80GB",
    tensor_parallel=2
)

print(f"Latency: {result['Latency']:.2f} ms")
print(f"Throughput: {result['Throughput']:.2f} tokens/s")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/BudEcosystem/simulator.git
cd simulator

# Install development dependencies
pip install -e llm-memory-calculator/
cd BudSimulator && pip install -r requirements.txt
```

### Running Tests
```bash
# Test llm-memory-calculator
cd llm-memory-calculator
pytest tests/

# Test BudSimulator
cd BudSimulator
python comprehensive_api_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the GenZ-LLM Analyzer framework
- Inspired by MLPerf benchmarks
- Hardware specs from official documentation
- Model configs from HuggingFace

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/BudEcosystem/simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BudEcosystem/simulator/discussions)
- **Email**: support@budecosystem.com

## 🗺️ Roadmap

### Near Term
- [ ] Support for quantized models (GGUF, AWQ, GPTQ)
- [ ] Cloud cost estimation
- [ ] Inference server integration (vLLM, TGI)
- [ ] Multi-node simulation

### Long Term
- [ ] Training performance estimation
- [ ] Fine-tuning recommendations
- [ ] Energy efficiency metrics
- [ ] Real-time monitoring integration

---

**Built with ❤️ by the Bud Ecosystem team**