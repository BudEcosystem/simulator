# BudSimulator REST API

This is a FastAPI-based REST API for the BudSimulator that provides endpoints for model validation, configuration retrieval, memory calculation, and analysis.

## Features

- **Model Validation**: Check if a HuggingFace model URL/ID is valid
- **Configuration Retrieval**: Get detailed model configurations and metadata
- **Memory Calculation**: Calculate memory requirements for specific model configurations
- **Model Comparison**: Compare memory requirements across multiple models
- **Efficiency Analysis**: Analyze how memory scales with sequence length
- **Popular Models**: Get a curated list of popular models with analysis
- **Model Management**: List, filter, and add models from various sources

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. POST /api/models/validate
Validate if a model URL/ID is valid and accessible.

**Request:**
```json
{
  "model_url": "meta-llama/Llama-2-7b-hf"
}
```

**Response:**
```json
{
  "valid": true,
  "error": null
}
```

### 2. GET /api/models/{model_id}/config
Get detailed configuration for a specific model.

**Response:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "model_type": "decoder-only",
  "attention_type": "gqa",
  "parameter_count": 7000000000,
  "architecture": "LlamaForCausalLM",
  "config": {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "activation_function": "silu"
  },
  "metadata": {
    "downloads": 1500000,
    "likes": 25000,
    "size_gb": 13.5,
    "tags": ["text-generation", "llama"]
  }
}
```

### 3. POST /api/models/calculate
Calculate memory requirements for a model with specific parameters.

**Request:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "precision": "fp16",
  "batch_size": 1,
  "seq_length": 2048,
  "num_images": 0,
  "include_gradients": false,
  "decode_length": 0
}
```

**Response:**
```json
{
  "model_type": "decoder-only",
  "attention_type": "gqa",
  "precision": "fp16",
  "parameter_count": 7000000000,
  "memory_breakdown": {
    "weight_memory_gb": 13.5,
    "kv_cache_gb": 0.5,
    "activation_memory_gb": 2.1,
    "state_memory_gb": 0.0,
    "image_memory_gb": 0.0,
    "extra_work_gb": 0.5
  },
  "total_memory_gb": 16.6,
  "recommendations": {
    "recommended_gpu_memory_gb": 24,
    "can_fit_24gb_gpu": true,
    "can_fit_80gb_gpu": true,
    "min_gpu_memory_gb": 16.6
  }
}
```

### 4. POST /api/models/compare
Compare memory requirements for multiple models.

**Request:**
```json
{
  "models": [
    {
      "model_id": "meta-llama/Llama-2-7b-hf",
      "precision": "fp16",
      "batch_size": 1,
      "seq_length": 2048
    },
    {
      "model_id": "mistralai/Mistral-7B-v0.1",
      "precision": "fp16",
      "batch_size": 1,
      "seq_length": 2048
    }
  ]
}
```

### 5. POST /api/models/analyze
Analyze model efficiency across different sequence lengths.

**Request:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "precision": "fp16",
  "batch_size": 1,
  "sequence_lengths": [1024, 4096, 16384, 32768]
}
```

**Response:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "attention_type": "gqa",
  "analysis": {
    "1024": {
      "total_memory_gb": 14.2,
      "kv_cache_gb": 0.25,
      "kv_cache_percent": 1.8
    },
    "4096": {
      "total_memory_gb": 16.6,
      "kv_cache_gb": 1.0,
      "kv_cache_percent": 6.0
    }
  },
  "insights": {
    "memory_per_token_bytes": 256,
    "efficiency_rating": "high",
    "recommendations": [
      "GQA provides 4x KV cache compression compared to MHA"
    ]
  }
}
```

### 6. GET /api/models/popular
Get a list of popular models.

**Query Parameters:**
- `limit` (optional): Number of models to return (default: 10)

**Response:**
```json
{
  "models": [
    {
      "model_id": "meta-llama/Llama-2-7b-hf",
      "name": "Llama 2 7B",
      "parameters": "7B",
      "model_type": "decoder-only",
      "attention_type": "gqa",
      "downloads": 1500000,
      "likes": 25000,
      "description": "High-performance 7B language model with GQA"
    }
  ]
}
```

### 7. GET /api/models/list
Get a comprehensive list of all models from both MODEL_DICT and database.

**Response:**
```json
{
  "total_count": 42,
  "model_dict_count": 20,
  "database_count": 25,
  "models": [
    {
      "model_id": "gpt2",
      "name": "GPT2",
      "author": null,
      "model_type": "decoder-only",
      "attention_type": "mha",
      "parameter_count": 137022720,
      "source": "both",
      "in_model_dict": true,
      "in_database": true
    }
  ]
}
```

### 8. GET /api/models/filter
Filter models by various criteria.

**Query Parameters:**
- `author`: Filter by model author/organization (e.g., "meta-llama", "microsoft")
- `model_type`: Filter by type (decoder-only, encoder-decoder, etc.)
- `attention_type`: Filter by attention mechanism (mha, gqa, mqa, mla)
- `min_parameters`: Minimum parameter count
- `max_parameters`: Maximum parameter count
- `source`: Filter by source (model_dict, database, or both)
- `tags`: Comma-separated tags to filter by

**Example:** `/api/models/filter?author=microsoft&model_type=decoder-only`

**Response:**
```json
{
  "total_count": 2,
  "filters_applied": {
    "author": "microsoft",
    "model_type": "decoder-only"
  },
  "models": [
    {
      "model_id": "microsoft/phi-2",
      "name": "PHI 2",
      "author": "microsoft",
      "model_type": "decoder-only",
      "attention_type": "mha",
      "parameter_count": 2779684864,
      "source": "database",
      "in_model_dict": false,
      "in_database": true
    }
  ]
}
```

### 9. POST /api/models/add/huggingface
Add a model from HuggingFace Hub.

**Request:**
```json
{
  "model_uri": "distilgpt2",
  "auto_import": true
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "distilgpt2",
  "message": "Successfully imported model from HuggingFace",
  "source": "database",
  "already_existed": false
}
```

### 10. POST /api/models/add/config
Add a model from a configuration dictionary.

**Request:**
```json
{
  "model_id": "custom-model-7b",
  "config": {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "model_type": "llama"
  },
  "metadata": {
    "description": "Custom 7B model",
    "created_by": "Research Team"
  }
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "custom-model-7b",
  "message": "Successfully added model from configuration",
  "source": "database",
  "already_existed": false
}
```

## Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

This will test all endpoints and display the responses.

## Error Handling

The API uses standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (model not found)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

Error responses include a `detail` field with the error message:
```json
{
  "detail": "Model not found: meta-llama/Llama-2-7b-hf"
}
```

## Precision Formats

Supported precision formats:
- `fp32`: 32-bit floating point
- `fp16`: 16-bit floating point
- `bf16`: Brain floating point 16-bit
- `int8`: 8-bit integer quantization
- `int4`: 4-bit integer quantization
- `fp8`: 8-bit floating point

## Model Types

The API recognizes various model types:
- `decoder-only`: GPT-style models
- `encoder-decoder`: T5-style models
- `encoder-only`: BERT-style models
- `moe`: Mixture of Experts models
- `multimodal`: Vision-language models
- `mamba`: State-space models
- `hybrid`: Combined architectures

## Attention Types

- `mha`: Multi-Head Attention
- `mqa`: Multi-Query Attention
- `gqa`: Grouped-Query Attention
- `mla`: Multi-Latent Attention

## Model Sources

Models can come from different sources:
- `model_dict`: Built-in models from GenZ MODEL_DICT
- `database`: Models imported and stored in SQLite database
- `both`: Models that exist in both sources

## Development

The API is built with:
- **FastAPI**: Modern web framework
- **Pydantic**: Data validation using Python type annotations
- **Uvicorn**: ASGI server

To run in development mode with auto-reload:
```bash
python run_api.py
```

## Integration

The API integrates with:
- **HuggingFace Hub**: For fetching model configurations
- **BudSimulator Database**: For persistent model storage
- **GenZ Memory Calculator**: For accurate memory estimations
- **Dynamic Model Loading**: Seamlessly combines static and dynamic models

## Performance

- Results are cached in the database to improve response times
- Popular models are pre-analyzed for quick access
- The API supports concurrent requests
- Database queries are optimized with indexes

## Future Enhancements

- [ ] Authentication and rate limiting
- [ ] Batch model import endpoint
- [ ] Model search with filters
- [ ] Memory optimization suggestions
- [ ] Cost estimation based on cloud GPU pricing
- [ ] WebSocket support for real-time analysis
- [ ] Model versioning and rollback
- [ ] Export/import model collections 