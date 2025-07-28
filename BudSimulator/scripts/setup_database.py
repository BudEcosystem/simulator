#!/usr/bin/env python3
"""
Complete database setup script for BudSimulator.

This script handles:
1. Database creation and schema setup
2. Hardware data import from multiple sources
3. Model data import from MODEL_DICT
4. Model parameter extraction and analysis
5. Logo downloading
6. LLM-based model analysis generation

Usage:
    python setup_database.py [--skip-analysis] [--skip-logos] [--force-recreate]
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import ModelManager, HuggingFaceModelImporter, DatabaseConnection
from src.hardware_registry import HardwareRegistry
from src.bud_ai import call_bud_LLM
from src.utils.text_extraction import extract_text_from_huggingface
from src.utils.llm_integration import parse_model_analysis, validate_analysis
from src.prompts import MODEL_ANALYSIS_PROMPT
from src.bud_models import ModelMemoryCalculator
from GenZ.Models import MODEL_DICT
from Systems.system_configs import system_configs
from GenZ.cpu.cpu_configs import CPU_PRESETS


class DatabaseSetup:
    """Handles complete database setup and data population."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.hf_importer = HuggingFaceModelImporter()
        self.hardware_registry = HardwareRegistry()
        self.bud_hardware = None
        self.memory_calculator = ModelMemoryCalculator()
        
    def create_database_schema(self):
        """Initialize database and create all tables."""
        print("🔧 Creating database schema...")
        # Initialize database connection (this will create schema automatically)
        db_conn = DatabaseConnection()
        print("✅ Database schema created successfully")
        
    def import_hardware_data(self):
        """Import hardware data from all sources."""
        print("\n🖥️  Importing hardware data...")
        
        # Initialize BudHardware if not already done
        if not self.bud_hardware:
            from src.hardware import BudHardware
            self.bud_hardware = BudHardware()
        
        # Import from system_configs.py
        print("  📊 Importing from system_configs.py...")
        for hw_id, hw_data in system_configs.items():
            try:
                # Convert system_configs format to database format
                hardware_data = {
                    'name': hw_id.replace('_', ' '),
                    'type': 'gpu' if 'GPU' in hw_id else 'accelerator',
                    'manufacturer': self._extract_manufacturer(hw_id),
                    'flops': hw_data['Flops'],
                    'memory_size': hw_data['Memory_size'],
                    'memory_bw': hw_data['Memory_BW'],
                    'icn': hw_data.get('ICN', 0),
                    'power': hw_data.get('Power'),
                    'real_values': hw_data.get('real_values', True),
                    'source': 'json_import'
                }
                
                self.bud_hardware.add_hardware(hardware_data)
                print(f"    ✓ Added {hw_id}")
                
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"    ❌ Failed to add {hw_id}: {e}")
                else:
                    print(f"    ⚠️  {hw_id} already exists, skipping")
        
        # Import from CPU configs
        print("  🧠 Importing CPU configurations...")
        for cpu_id, cpu_data in CPU_PRESETS.items():
            try:
                base_params = cpu_data['base_params']
                cpu_config = cpu_data['cpu_specific']
                
                # Calculate approximate FLOPS for CPU
                total_cores = cpu_config.cores_per_socket * cpu_config.sockets
                approx_flops = (total_cores * base_params['frequency'] * 8) / 1e12  # Rough estimate
                
                hardware_data = {
                    'name': cpu_id.replace('_', ' ').title(),
                    'type': 'cpu',
                    'manufacturer': cpu_config.vendor.title(),
                    'flops': approx_flops,
                    'memory_size': 512,  # Typical server memory capacity
                    'memory_bw': cpu_config.dram_bandwidth_per_channel * cpu_config.memory_channels_per_socket,
                    'icn': 50,  # Typical CPU interconnect
                    'power': 200,  # Typical CPU power
                    'real_values': True,
                    'source': 'json_import'
                }
                
                self.bud_hardware.add_hardware(hardware_data)
                print(f"    ✓ Added {cpu_id}")
                
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"    ❌ Failed to add {cpu_id}: {e}")
                else:
                    print(f"    ⚠️  {cpu_id} already exists, skipping")
        
        # Import from hardware.json
        print("  📁 Importing from hardware.json...")
        hardware_json_path = Path(__file__).parent.parent / "src/utils/hardware.json"
        
        if hardware_json_path.exists():
            with open(hardware_json_path, 'r') as f:
                hardware_json = json.load(f)
            
            for hw_id, hw_data in hardware_json.items():
                try:
                    # Convert pricing data
                    min_price = None
                    max_price = None
                    
                    if 'on_prem_price' in hw_data:
                        price_data = hw_data['on_prem_price']
                        if isinstance(price_data, dict):
                            min_price = price_data.get('lower_bound')
                            max_price = price_data.get('upper_bound')
                            # Convert string prices to numbers
                            if isinstance(min_price, str) and min_price.isdigit():
                                min_price = int(min_price)
                            if isinstance(max_price, str) and max_price.isdigit():
                                max_price = int(max_price)
                    
                    hardware_data = {
                        'name': hw_data.get('name', hw_id),
                        'type': hw_data.get('type', 'unknown'),
                        'manufacturer': hw_data.get('manufacturer'),
                        'flops': hw_data.get('Flops', 0),
                        'memory_size': hw_data.get('Memory_size', 0),
                        'memory_bw': hw_data.get('Memory_BW', 0),
                        'icn': hw_data.get('ICN', 0),
                        'power': hw_data.get('Power'),
                        'real_values': hw_data.get('real_values', False),
                        'url': hw_data.get('url'),
                        'description': hw_data.get('description'),
                        'source': 'json_import'
                    }
                    
                    self.bud_hardware.add_hardware(hardware_data)
                    print(f"    ✓ Added {hw_data.get('name', hw_id)}")
                    
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"    ❌ Failed to add {hw_id}: {e}")
                    else:
                        print(f"    ⚠️  {hw_data.get('name', hw_id)} already exists, skipping")
        
        print("✅ Hardware data import completed")
    
    def import_model_data(self):
        """Import model data from MODEL_DICT."""
        print("\n🤖 Importing model data from MODEL_DICT...")
        
        try:
            model_list = MODEL_DICT.list_models()
            print(f"  Found {len(model_list)} models in MODEL_DICT")
            
            for model_id in model_list:
                try:
                    model = MODEL_DICT.get_model(model_id)
                    if model:
                        # Extract basic attributes
                        model_type = getattr(model, 'model_type', 'unknown')
                        attention_type = getattr(model, 'attention_type', None)
                        parameter_count = getattr(model, 'num_parameters', None)
                        
                        # Get model config
                        config = {}
                        if hasattr(model, 'config'):
                            config = model.config
                        
                        # Add model to database
                        self.model_manager.add_model(
                            model_id=model_id,
                            config=config,
                            source='MODEL_DICT',
                            model_type=model_type,
                            attention_type=attention_type,
                            parameter_count=parameter_count
                        )
                        print(f"    ✓ Added {model_id}")
                        
                except Exception as e:
                    if "UNIQUE constraint failed" not in str(e):
                        print(f"    ❌ Failed to add {model_id}: {e}")
                    else:
                        print(f"    ⚠️  {model_id} already exists, skipping")
                    
        except Exception as e:
            print(f"❌ Error accessing MODEL_DICT: {e}")
        
        print("✅ Model data import completed")
    
    def extract_model_parameters(self):
        """Extract and populate missing model parameters."""
        print("\n📊 Extracting model parameters...")
        
        models = self.model_manager.list_models()
        updated_count = 0
        
        for model in models:
            model_id = model['model_id']
            needs_update = False
            updates = {}
            
            # Check what needs to be updated
            if not model.get('model_type') or model['model_type'] == 'unknown':
                try:
                    model_type = self.memory_calculator.detect_model_type(model_id)
                    if model_type != 'unknown':
                        updates['model_type'] = model_type
                        needs_update = True
                except Exception:
                    pass
            
            if not model.get('attention_type'):
                try:
                    attention_type = self.memory_calculator.detect_attention_type(model_id)
                    if attention_type:
                        updates['attention_type'] = attention_type
                        needs_update = True
                except Exception:
                    pass
            
            if not model.get('parameter_count'):
                try:
                    config = json.loads(model.get('config_json', '{}'))
                    if config:
                        param_count = self.memory_calculator._calculate_transformer_params(config)
                        if param_count > 0:
                            updates['parameter_count'] = param_count
                            needs_update = True
                except Exception:
                    pass
            
            # Update if needed
            if needs_update:
                try:
                    existing_config = json.loads(model.get('config_json', '{}'))
                    
                    # Update model with new parameters
                    self.model_manager.update_model(
                        model_id=model_id,
                        config=existing_config,
                        **updates
                    )
                    updated_count += 1
                    print(f"    ✓ Updated {model_id}: {list(updates.keys())}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to update {model_id}: {e}")
        
        print(f"✅ Model parameter extraction completed ({updated_count} models updated)")
    
    def download_logos(self, skip_existing=True):
        """Download logos for all models."""
        print("\n🖼️  Downloading model logos...")
        
        models = self.model_manager.list_models()
        downloaded_count = 0
        
        for model in models:
            model_id = model['model_id']
            
            # Skip if logo already exists
            if skip_existing and model.get('logo'):
                continue
            
            try:
                print(f"    📥 Downloading logo for {model_id}")
                logo_url = self.hf_importer.scrap_hf_logo(model_id)
                
                if logo_url:
                    local_logo_path = self.hf_importer._save_logo(model_id, logo_url)
                    
                    if local_logo_path:
                        existing_config = json.loads(model.get('config_json', '{}'))
                        self.model_manager.update_model(
                            model_id=model_id,
                            config=existing_config,
                            logo=local_logo_path
                        )
                        downloaded_count += 1
                        print(f"    ✅ Downloaded logo for {model_id}")
                    else:
                        print(f"    ⚠️  Failed to save logo for {model_id}")
                else:
                    print(f"    ⚠️  No logo found for {model_id}")
                    
            except Exception as e:
                print(f"    ❌ Error downloading logo for {model_id}: {e}")
        
        print(f"✅ Logo download completed ({downloaded_count} logos downloaded)")
    
    def generate_model_analysis(self, skip_existing=True):
        """Generate LLM-based analysis for all models."""
        print("\n🧠 Generating model analysis...")
        
        models = self.model_manager.list_models()
        generated_count = 0
        
        for model in models:
            model_id = model['model_id']
            
            # Check if analysis already exists
            if skip_existing:
                model_analysis = model.get('model_analysis')
                if model_analysis:
                    try:
                        analysis_data = json.loads(model_analysis)
                        if analysis_data.get('description'):
                            continue  # Skip if valid analysis exists
                    except:
                        pass  # Continue if analysis is invalid
            
            try:
                print(f"    🤔 Generating analysis for {model_id}")
                
                # Extract text from HuggingFace
                hf_url = f"https://huggingface.co/{model_id}"
                model_description = extract_text_from_huggingface(hf_url)
                
                if not model_description:
                    print(f"    ⚠️  No description found for {model_id}")
                    continue
                
                # Generate analysis with LLM
                full_prompt = MODEL_ANALYSIS_PROMPT + model_description
                llm_response = call_bud_LLM(full_prompt)
                
                # Parse and validate
                analysis = parse_model_analysis(llm_response)
                validated_analysis = validate_analysis(analysis)
                
                # Update model in DB
                existing_config = json.loads(model.get('config_json', '{}'))
                self.model_manager.update_model(
                    model_id=model_id,
                    config=existing_config,
                    model_analysis=validated_analysis
                )
                
                generated_count += 1
                print(f"    ✅ Generated analysis for {model_id}")
                
            except Exception as e:
                print(f"    ❌ Error generating analysis for {model_id}: {e}")
        
        print(f"✅ Model analysis generation completed ({generated_count} analyses generated)")
    
    def _extract_manufacturer(self, hw_id):
        """Extract manufacturer from hardware ID."""
        hw_id_lower = hw_id.lower()
        if 'nvidia' in hw_id_lower or 'h100' in hw_id_lower or 'a100' in hw_id_lower or 'v100' in hw_id_lower:
            return 'NVIDIA'
        elif 'amd' in hw_id_lower or 'mi300' in hw_id_lower:
            return 'AMD'
        elif 'intel' in hw_id_lower:
            return 'Intel'
        elif 'tpu' in hw_id_lower:
            return 'Google'
        elif 'b100' in hw_id_lower or 'gb200' in hw_id_lower:
            return 'NVIDIA'
        else:
            return None
    
    def run_complete_setup(self, skip_analysis=False, skip_logos=False, force_recreate=False):
        """Run the complete database setup process."""
        print("🚀 Starting complete database setup...\n")
        
        if force_recreate:
            print("⚠️  Force recreate enabled - existing data will be lost!")
            # Here you could add database drop/recreate logic if needed
        
        # Step 1: Create database schema
        self.create_database_schema()
        
        # Step 2: Import hardware data
        self.import_hardware_data()
        
        # Step 3: Import model data
        self.import_model_data()
        
        # Step 4: Extract model parameters
        self.extract_model_parameters()
        
        # Step 5: Download logos (optional)
        if not skip_logos:
            self.download_logos()
        else:
            print("\n🖼️  Skipping logo download (--skip-logos)")
        
        # Step 6: Generate model analysis (optional)
        if not skip_analysis:
            self.generate_model_analysis()
        else:
            print("\n🧠 Skipping model analysis generation (--skip-analysis)")
        
        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the backend API: python -m uvicorn apis.main:app --reload")
        print("2. Start the frontend: cd frontend && npm start")
        print("3. Open http://localhost:3000 in your browser")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete database setup for BudSimulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_database.py                    # Full setup
  python setup_database.py --skip-analysis   # Skip LLM analysis
  python setup_database.py --skip-logos      # Skip logo download
  python setup_database.py --force-recreate  # Recreate database
        """
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip LLM-based model analysis generation'
    )
    
    parser.add_argument(
        '--skip-logos',
        action='store_true',
        help='Skip logo downloading'
    )
    
    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='Force recreate database (WARNING: existing data will be lost)'
    )
    
    args = parser.parse_args()
    
    # Create setup instance and run
    setup = DatabaseSetup()
    setup.run_complete_setup(
        skip_analysis=args.skip_analysis,
        skip_logos=args.skip_logos,
        force_recreate=args.force_recreate
    )


if __name__ == "__main__":
    main() 