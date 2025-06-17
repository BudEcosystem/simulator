#!/usr/bin/env python3
"""
Test script to verify the batch update functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from update_missing_data import ModelDataUpdater
from src.db import ModelManager

def test_batch_updater():
    """Test the batch updater with a small subset of models."""
    print("Testing batch updater...")
    
    # Initialize
    updater = ModelDataUpdater()
    model_manager = ModelManager()
    
    # Test with just a few models
    print("\n1. Testing with limit=2 to process only 2 models:")
    updater.run(limit=2)
    
    # Check a specific model
    print("\n2. Checking if supplementary records were created for MODEL_DICT models:")
    test_models = ['mistral_7b', 'gpt2', 'meta-llama/Llama-2-7b-hf']
    
    for model_id in test_models:
        db_model = model_manager.get_model(model_id)
        if db_model:
            print(f"\n✓ {model_id}:")
            print(f"  - Source: {db_model.get('source', 'unknown')}")
            print(f"  - Has logo: {'Yes' if db_model.get('logo') else 'No'}")
            print(f"  - Has analysis: {'Yes' if db_model.get('model_analysis') else 'No'}")
        else:
            print(f"\n✗ {model_id}: Not found in database")
    
    print("\n3. Testing filter functionality:")
    updater_filtered = ModelDataUpdater()
    updater_filtered.run(limit=1, model_filter="mistral")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_batch_updater() 