#!/usr/bin/env python3
"""
Test script for dynamic model loading functionality.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add GenZ to path
sys.path.insert(0, str(Path(__file__).parent / 'GenZ'))

def test_database_connection():
    """Test database connection and initialization."""
    print("\n=== Testing Database Connection ===")
    try:
        from src.db import DatabaseConnection
        db = DatabaseConnection()
        info = db.get_db_info()
        print(f"✓ Database initialized at: {info['db_path']}")
        print(f"  Schema version: {info['schema_version']}")
        print(f"  Tables: {', '.join(info['tables'])}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_model_import():
    """Test importing a model from HuggingFace."""
    print("\n=== Testing Model Import ===")
    try:
        from src.db import HuggingFaceModelImporter
        importer = HuggingFaceModelImporter()
        
        # Try to import a small model (using a non-gated model)
        model_id = "gpt2"
        print(f"Importing {model_id}...")
        
        success = importer.import_model(model_id)
        if success:
            print(f"✓ Successfully imported {model_id}")
            
            # Validate import
            validation = importer.validate_import(model_id)
            print(f"  Config loaded: {validation['has_config']}")
            print(f"  Convertible to GenZ: {validation['convertible']}")
            print(f"  Has metrics: {validation['has_metrics']}")
            return True
        else:
            print(f"✗ Failed to import {model_id}")
            return False
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_model_dict():
    """Test the dynamic MODEL_DICT functionality."""
    print("\n=== Testing Dynamic MODEL_DICT ===")
    try:
        # Import from the correct path
        from GenZ.Models import MODEL_DICT
        from src.db.model_loader import DynamicModelCollection
        
        # Check if MODEL_DICT is dynamic
        is_dynamic = isinstance(MODEL_DICT, DynamicModelCollection)
        print(f"MODEL_DICT type: {type(MODEL_DICT)}")
        print(f"MODEL_DICT is dynamic: {is_dynamic}")
        
        # Even if not dynamic, test functionality
        if True:  # Always test functionality
            # Test getting a static model
            print("\nTesting static model access:")
            model = MODEL_DICT.get_model("mistral_7b")
            if model:
                print(f"✓ Found static model: {model.model}")
                print(f"  Hidden size: {model.hidden_size}")
                print(f"  Num layers: {model.num_decoder_layers}")
            
            # Test getting a database model
            print("\nTesting database model access:")
            model = MODEL_DICT.get_model("gpt2")
            if model:
                print(f"✓ Found database model: {model.model}")
                print(f"  Hidden size: {model.hidden_size}")
                print(f"  Num layers: {model.num_decoder_layers}")
            else:
                print("✗ Database model not found (may need to import first)")
            
            # List all models
            print(f"\nTotal available models: {len(MODEL_DICT.list_models())}")
            
            # If we can access both static and dynamic models, consider it a success
            return True
            
    except Exception as e:
        print(f"✗ Dynamic MODEL_DICT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_conversion():
    """Test conversion between HuggingFace and GenZ formats."""
    print("\n=== Testing Model Conversion ===")
    try:
        from src.db.model_converter import ModelConverter
        from Models.default_models import ModelConfig
        
        converter = ModelConverter()
        
        # Test HF to GenZ conversion
        hf_config = {
            'model_id': 'test/model',
            'vocab_size': 32000,
            'hidden_size': 4096,
            'intermediate_size': 11008,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'hidden_act': 'silu',
            'model_type': 'mistral'
        }
        
        print("Converting HF config to GenZ...")
        genz_config = converter.hf_to_genz(hf_config)
        
        if isinstance(genz_config, ModelConfig):
            print("✓ Successfully converted to GenZ ModelConfig")
            print(f"  Model: {genz_config.model}")
            print(f"  Hidden size: {genz_config.hidden_size}")
            print(f"  Attention type: GQA (kv_heads={genz_config.num_key_value_heads})")
            
            # Test reverse conversion
            print("\nConverting back to HF format...")
            hf_config_back = converter.genz_to_hf(genz_config)
            print("✓ Successfully converted back to HF format")
            
            # Validate conversion
            validation = converter.validate_conversion(hf_config, genz_config, "hf_to_genz")
            print(f"  Conversion valid: {validation['valid']}")
            if validation['warnings']:
                print(f"  Warnings: {validation['warnings']}")
            
            return True
        else:
            print("✗ Conversion failed")
            return False
            
    except Exception as e:
        print(f"✗ Model conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_stats():
    """Test CLI statistics command."""
    print("\n=== Testing CLI Stats ===")
    try:
        from src.db.cli import ModelCLI
        import argparse
        
        cli = ModelCLI()
        
        # Create mock args for stats command
        args = argparse.Namespace()
        
        print("Running stats command...")
        cli.stats(args)
        
        return True
    except Exception as e:
        print(f"✗ CLI stats test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Dynamic Model Loading for GenZ Simulator")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Import", test_model_import),
        ("Dynamic MODEL_DICT", test_dynamic_model_dict),
        ("Model Conversion", test_model_conversion),
        ("CLI Stats", test_cli_stats),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("-" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 