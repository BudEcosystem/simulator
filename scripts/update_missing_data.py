#!/usr/bin/env python3
"""
Simple script to update missing logos and analysis for models in the database.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import ModelManager, HuggingFaceModelImporter
from src.bud_ai import call_bud_LLM
from src.utils.text_extraction import extract_text_from_huggingface
from src.utils.llm_integration import parse_model_analysis, validate_analysis
from src.prompts import MODEL_ANALYSIS_PROMPT


def main():
    # Initialize components
    model_manager = ModelManager()
    hf_importer = HuggingFaceModelImporter()
    
    # Get all models from database only
    models = model_manager.list_models()
    print(f"Found {len(models)} models in database")
    
    # Process each model
    for i, model in enumerate(models, 1):
        model_id = model['model_id']
        print(f"\n[{i}/{len(models)}] Processing {model_id}")
        
        # 2.1 Check and update logo
        if not model.get('logo'):
            print(f"  Downloading logo for {model_id}")
            logo_url = hf_importer.scrap_hf_logo(model_id)
            local_logo_path = hf_importer._save_logo(model_id, logo_url)
            
            existing_config = json.loads(model.get('config_json', '{}'))
            model_manager.update_model(
                model_id=model_id,
                config=existing_config,
                logo=local_logo_path
            )
            print(f"  ✓ Logo updated")
        
        # 2.2 Check and update analysis
        model_analysis = model.get('model_analysis')
        needs_analysis = False
        
        if not model_analysis:
            needs_analysis = True
        elif model_analysis == '{}' or model_analysis == '':
            needs_analysis = True
        else:
            # Check if it's valid JSON with description
            analysis_data = json.loads(model_analysis)
            if not analysis_data.get('description'):
                needs_analysis = True
        
        if needs_analysis:
            print(f"  Generating analysis for {model_id}")
            
            # Create HuggingFace URL and scrape data
            hf_url = f"https://huggingface.co/{model_id}"
            model_description = extract_text_from_huggingface(hf_url)
            
            # Pass to LLM for analysis
            full_prompt = MODEL_ANALYSIS_PROMPT + model_description
            llm_response = call_bud_LLM(full_prompt)
            
            # Convert to JSON
            analysis = parse_model_analysis(llm_response)
            validated_analysis = validate_analysis(analysis)
            
            # Update model in DB
            existing_config = json.loads(model.get('config_json', '{}'))
            model_manager.update_model(
                model_id=model_id,
                config=existing_config,
                model_analysis=validated_analysis
            )
            print(f"  ✓ Analysis updated")
    
    print(f"\nCompleted processing {len(models)} models")


if __name__ == "__main__":
    main() 