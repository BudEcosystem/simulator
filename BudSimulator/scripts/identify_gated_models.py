#!/usr/bin/env python3
"""
Script to identify gated models in existing MODEL_DICT and database.

This script:
1. Checks all models in MODEL_DICT and database for gated status
2. Updates database with gated flag
3. Fetches appropriate information for gated models
4. Handles edge cases where models might be gated but exist in our data

Usage:
    python identify_gated_models.py [--dry-run] [--force-check]
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import ModelManager, DatabaseConnection
from src.bud_models import HuggingFaceConfigLoader, ModelMemoryCalculator
from src.utils.text_extraction import extract_text_from_huggingface
from src.utils.llm_integration import parse_model_analysis, validate_analysis
from src.prompts import MODEL_ANALYSIS_PROMPT
from src.bud_ai import call_bud_LLM
from llm_memory_calculator.genz.Models import MODEL_DICT


class GatedModelIdentifier:
    """Identifies and processes gated models in existing data."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.model_manager = ModelManager()
        self.hf_loader = HuggingFaceConfigLoader()
        self.memory_calculator = ModelMemoryCalculator()
        
        # Statistics
        self.stats = {
            'total_checked': 0,
            'gated_found': 0,
            'gated_updated': 0,
            'analysis_generated': 0,
            'errors': 0
        }
    
    def check_if_gated(self, model_id: str) -> bool:
        """
        Check if a model is gated by trying to fetch its config.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            True if model is gated, False otherwise
        """
        try:
            # Try to fetch config from HuggingFace
            config = self.hf_loader.fetch_model_config(model_id)
            return False  # If successful, not gated
            
        except Exception as e:
            error_msg = str(e).lower()
            # Check for gated model indicators
            if any(indicator in error_msg for indicator in [
                'model_gated', 'gated', '403', 'forbidden', 
                'authentication', 'access', 'private'
            ]):
                return True
            return False  # Other errors (404, network, etc.)
    
    def update_database_schema(self):
        """Add gated flag column to models table if it doesn't exist."""
        try:
            # Check if gated column exists
            db = self.model_manager.db
            result = db.execute_one("PRAGMA table_info(models)")
            columns = [col['name'] for col in db.execute("PRAGMA table_info(models)")]
            
            if 'is_gated' not in columns:
                print("🔧 Adding 'is_gated' column to models table...")
                if not self.dry_run:
                    db.execute("ALTER TABLE models ADD COLUMN is_gated BOOLEAN DEFAULT 0")
                    print("✅ Added 'is_gated' column")
                else:
                    print("🔍 [DRY RUN] Would add 'is_gated' column")
            else:
                print("✅ 'is_gated' column already exists")
                
        except Exception as e:
            print(f"❌ Error updating database schema: {e}")
    
    def process_model(self, model_id: str, source: str = "unknown") -> dict:
        """
        Process a single model to check if it's gated and update accordingly.
        
        Args:
            model_id: Model identifier
            source: Source of the model (database, model_dict, etc.)
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'model_id': model_id,
            'source': source,
            'was_gated': False,
            'updated': False,
            'analysis_generated': False,
            'error': None
        }
        
        try:
            print(f"  🔍 Checking {model_id} from {source}...")
            self.stats['total_checked'] += 1
            
            # Check if model is gated
            is_gated = self.check_if_gated(model_id)
            result['was_gated'] = is_gated
            
            if is_gated:
                print(f"    🚫 Found gated model: {model_id}")
                self.stats['gated_found'] += 1
                
                if not self.dry_run:
                    # Update database with gated flag
                    db_model = self.model_manager.get_model(model_id)
                    if db_model:
                        # Update existing model
                        existing_config = json.loads(db_model.get('config_json', '{}'))
                        self.model_manager.db.update(
                            'models',
                            {'is_gated': 1},
                            'model_id = ?',
                            (model_id,)
                        )
                        print(f"    ✅ Updated gated flag for {model_id}")
                        result['updated'] = True
                        self.stats['gated_updated'] += 1
                    else:
                        # Add new model with gated flag
                        try:
                            # Try to get basic info from MODEL_DICT if available
                            model_info = {}
                            try:
                                model_config = MODEL_DICT.get_model(model_id)
                                if model_config:
                                    model_info = {
                                        'model_type': getattr(model_config, 'model_type', 'unknown'),
                                        'attention_type': getattr(model_config, 'attention_type', None),
                                        'parameter_count': getattr(model_config, 'num_parameters', None)
                                    }
                            except:
                                pass
                            
                            # Add minimal model record
                            self.model_manager.db.insert('models', {
                                'model_id': model_id,
                                'model_name': model_id.split('/')[-1] if '/' in model_id else model_id,
                                'source': f'{source}_gated',
                                'config_json': json.dumps({}),
                                'model_type': model_info.get('model_type', 'unknown'),
                                'attention_type': model_info.get('attention_type'),
                                'parameter_count': model_info.get('parameter_count'),
                                'is_gated': 1
                            })
                            print(f"    ✅ Added gated model record for {model_id}")
                            result['updated'] = True
                            self.stats['gated_updated'] += 1
                            
                        except Exception as e:
                            print(f"    ⚠️  Could not add model record for {model_id}: {e}")
                    
                    # Try to generate analysis for gated model (best effort)
                    try:
                        print(f"    🧠 Attempting to generate analysis for {model_id}...")
                        
                        # Extract description from HuggingFace (may work even for gated models)
                        hf_url = f"https://huggingface.co/{model_id}"
                        model_description = extract_text_from_huggingface(hf_url)
                        
                        if model_description and "Error" not in model_description:
                            # Generate analysis with LLM
                            full_prompt = MODEL_ANALYSIS_PROMPT + model_description
                            llm_response = call_bud_LLM(full_prompt)
                            
                            if llm_response:
                                analysis = parse_model_analysis(llm_response)
                                validated_analysis = validate_analysis(analysis)
                                
                                # Update model with analysis
                                db_model = self.model_manager.get_model(model_id)
                                if db_model:
                                    existing_config = json.loads(db_model.get('config_json', '{}'))
                                    self.model_manager.update_model(
                                        model_id=model_id,
                                        config=existing_config,
                                        model_analysis=validated_analysis
                                    )
                                    print(f"    ✅ Generated analysis for {model_id}")
                                    result['analysis_generated'] = True
                                    self.stats['analysis_generated'] += 1
                    except Exception as e:
                        print(f"    ⚠️  Could not generate analysis for {model_id}: {e}")
                
                else:
                    print(f"    🔍 [DRY RUN] Would update gated flag for {model_id}")
            else:
                print(f"    ✅ {model_id} is not gated")
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ❌ Error processing {model_id}: {e}")
            result['error'] = str(e)
            self.stats['errors'] += 1
        
        return result
    
    def process_model_dict_models(self):
        """Process all models from MODEL_DICT."""
        print("\n🔍 Checking models from MODEL_DICT...")
        
        results = []
        try:
            # Get models from MODEL_DICT
            if hasattr(MODEL_DICT, 'models'):
                model_ids = list(MODEL_DICT.models.keys())
            else:
                model_ids = MODEL_DICT.list_models() if hasattr(MODEL_DICT, 'list_models') else []
            
            print(f"Found {len(model_ids)} models in MODEL_DICT")
            
            for model_id in model_ids:
                result = self.process_model(model_id, "model_dict")
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error processing MODEL_DICT: {e}")
            
        return results
    
    def process_database_models(self):
        """Process all models from database."""
        print("\n🔍 Checking models from database...")
        
        results = []
        try:
            models = self.model_manager.list_models()
            print(f"Found {len(models)} models in database")
            
            for model in models:
                model_id = model['model_id']
                # Skip if already marked as gated
                if model.get('is_gated'):
                    print(f"  ⏭️  Skipping {model_id} (already marked as gated)")
                    continue
                    
                result = self.process_model(model_id, "database")
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error processing database models: {e}")
            
        return results
    
    def run(self, force_check=False):
        """
        Run the gated model identification process.
        
        Args:
            force_check: If True, check all models even if already marked as gated
        """
        print("🔍 Starting gated model identification...")
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        # Update database schema
        self.update_database_schema()
        
        # Process models from different sources
        model_dict_results = self.process_model_dict_models()
        database_results = self.process_database_models()
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total models checked: {self.stats['total_checked']}")
        print(f"  Gated models found: {self.stats['gated_found']}")
        print(f"  Database records updated: {self.stats['gated_updated']}")
        print(f"  Analyses generated: {self.stats['analysis_generated']}")
        print(f"  Errors encountered: {self.stats['errors']}")
        
        if self.stats['gated_found'] > 0:
            print(f"\n✅ Identified {self.stats['gated_found']} gated models")
            if not self.dry_run:
                print("🔧 Database updated with gated flags")
                print("💡 Users will now be prompted for configs only if not already provided")
            else:
                print("🔍 Run without --dry-run to apply changes")
        else:
            print("\n✅ No new gated models found")


def main():
    parser = argparse.ArgumentParser(description="Identify gated models in existing data")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--force-check', action='store_true',
                       help='Check all models even if already marked as gated')
    
    args = parser.parse_args()
    
    identifier = GatedModelIdentifier(dry_run=args.dry_run)
    identifier.run(force_check=args.force_check)


if __name__ == "__main__":
    main() 