#!/usr/bin/env python3
"""
Test script to demonstrate the BudSimulator API endpoints.
"""

import requests
import json
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8000"

def print_response(response: requests.Response, endpoint: str):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"Endpoint: {endpoint}")
    print(f"Status: {response.status_code}")
    print(f"Response:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
    print('='*60)

def test_validate_model():
    """Test the validate model endpoint."""
    endpoint = "/api/models/validate"
    
    # Test valid model
    data = {"model_url": "gpt2"}
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint} (valid model)")
    
    # Test invalid model
    data = {"model_url": "invalid/model-name-123"}
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint} (invalid model)")

def test_get_model_config():
    """Test the get model config endpoint."""
    model_id = "gpt2"
    endpoint = f"/api/models/{model_id}/config"
    
    response = requests.get(BASE_URL + endpoint)
    print_response(response, f"GET {endpoint}")

def test_calculate_memory():
    """Test the calculate memory endpoint."""
    endpoint = "/api/models/calculate"
    
    data = {
        "model_id": "gpt2",
        "precision": "fp16",
        "batch_size": 1,
        "seq_length": 2048,
        "num_images": 0,
        "include_gradients": False,
        "decode_length": 0
    }
    
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint}")

def test_compare_models():
    """Test the compare models endpoint."""
    endpoint = "/api/models/compare"
    
    data = {
        "models": [
            {
                "model_id": "gpt2",
                "precision": "fp16",
                "batch_size": 1,
                "seq_length": 2048
            },
            {
                "model_id": "microsoft/phi-2",
                "precision": "fp16",
                "batch_size": 1,
                "seq_length": 2048
            }
        ]
    }
    
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint}")

def test_analyze_model():
    """Test the analyze model endpoint."""
    endpoint = "/api/models/analyze"
    
    data = {
        "model_id": "gpt2",
        "precision": "fp16",
        "batch_size": 1,
        "sequence_lengths": [1024, 4096, 16384, 32768]
    }
    
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint}")

def test_get_popular_models():
    """Test the get popular models endpoint."""
    endpoint = "/api/models/popular"
    
    response = requests.get(BASE_URL + endpoint + "?limit=5")
    print_response(response, f"GET {endpoint}?limit=5")

def test_list_all_models():
    """Test the list all models endpoint."""
    endpoint = "/api/models/list"
    
    response = requests.get(BASE_URL + endpoint)
    print_response(response, f"GET {endpoint}")

def test_filter_models():
    """Test the filter models endpoint."""
    endpoint = "/api/models/filter"
    
    # Test filtering by author
    response = requests.get(BASE_URL + endpoint + "?author=microsoft")
    print_response(response, f"GET {endpoint}?author=microsoft")
    
    # Test filtering by model type
    response = requests.get(BASE_URL + endpoint + "?model_type=decoder-only")
    print_response(response, f"GET {endpoint}?model_type=decoder-only")
    
    # Test filtering by source
    response = requests.get(BASE_URL + endpoint + "?source=database")
    print_response(response, f"GET {endpoint}?source=database")

def test_add_model_from_huggingface():
    """Test adding a model from HuggingFace."""
    endpoint = "/api/models/add/huggingface"
    
    # Try to add a small model
    data = {
        "model_uri": "distilgpt2",
        "auto_import": True
    }
    
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint}")

def test_add_model_from_config():
    """Test adding a model from configuration."""
    endpoint = "/api/models/add/config"
    
    # Add a custom model
    data = {
        "model_id": "custom-test-model",
        "config": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "model_type": "gpt2"
        },
        "metadata": {
            "description": "Test custom model",
            "created_by": "API test"
        }
    }
    
    response = requests.post(BASE_URL + endpoint, json=data)
    print_response(response, f"POST {endpoint}")

def main():
    """Run all API tests."""
    print("BudSimulator API Test Suite")
    print("Make sure the API server is running on http://localhost:8000")
    print("Run with: python run_api.py")
    
    # Check if API is running
    try:
        response = requests.get(BASE_URL + "/health")
        if response.status_code != 200:
            print("\nError: API server is not responding. Please start it first.")
            return
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API server. Please start it with:")
        print("  python run_api.py")
        return
    
    print("\nAPI server is running. Starting tests...")
    
    # Run all tests
    print("\n### BASIC ENDPOINTS ###")
    test_validate_model()
    test_get_model_config()
    test_calculate_memory()
    test_compare_models()
    test_analyze_model()
    test_get_popular_models()
    
    print("\n### NEW ENDPOINTS ###")
    test_list_all_models()
    test_filter_models()
    test_add_model_from_huggingface()
    test_add_model_from_config()
    
    print("\nAll tests completed!")
    print("\nYou can also explore the API documentation at:")
    print(f"  - Swagger UI: {BASE_URL}/docs")
    print(f"  - ReDoc: {BASE_URL}/redoc")

if __name__ == "__main__":
    main() 