"""
LLM integration utilities for model analysis.
"""

import json
import logging
from typing import Dict, Any, Optional
from json_repair import repair_json

logger = logging.getLogger(__name__)


def extract_json_from_string(text: str) -> Optional[str]:
    """
    Extract JSON content from a string that may contain JSON within <json> tags.
    
    Args:
        text: String that may contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    # Try to find JSON between <json> tags
    import re
    json_match = re.search(r'<json>(.*?)</json>', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # If no tags, try to find JSON-like content
    # Look for content starting with { and ending with }
    json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # As a last resort, return the whole text if it looks like JSON
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    
    return None


def parse_model_analysis(llm_response: str) -> Dict[str, Any]:
    """
    Parse the LLM response to extract model analysis data.
    
    Args:
        llm_response: Raw response from LLM
        
    Returns:
        Parsed model analysis dictionary
    """
    try:
        # Extract JSON from response
        json_str = extract_json_from_string(llm_response)
        if not json_str:
            logger.error("No JSON found in LLM response")
            return get_empty_analysis()
        
        # Repair and parse JSON
        repaired_json = repair_json(json_str)
        analysis_data = json.loads(repaired_json)
        
        # Extract the model analysis from the nested structure
        # The response format has model_id as key with model_analysis inside
        if isinstance(analysis_data, dict):
            # Get the first key (model_id)
            first_key = next(iter(analysis_data), None)
            if first_key and 'model_analysis' in analysis_data[first_key]:
                return analysis_data[first_key]['model_analysis']
            elif 'model_analysis' in analysis_data:
                return analysis_data['model_analysis']
            else:
                # If it's already in the correct format
                if all(key in analysis_data for key in ['description', 'advantages', 'disadvantages', 'usecases', 'evals']):
                    return analysis_data
        
        logger.error("Invalid analysis data structure")
        return get_empty_analysis()
        
    except Exception as e:
        logger.error(f"Error parsing model analysis: {e}")
        return get_empty_analysis()


def get_empty_analysis() -> Dict[str, Any]:
    """
    Return an empty analysis structure.
    
    Returns:
        Empty analysis dictionary with all required fields
    """
    return {
        "description": "",
        "advantages": [],
        "disadvantages": [],
        "usecases": [],
        "evals": []
    }


def validate_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the analysis data.
    
    Args:
        analysis: Raw analysis dictionary
        
    Returns:
        Validated and cleaned analysis
    """
    # Ensure all required fields exist
    validated = get_empty_analysis()
    
    # Copy valid fields
    if isinstance(analysis.get('description'), str):
        validated['description'] = analysis['description']
    
    for field in ['advantages', 'disadvantages', 'usecases']:
        if isinstance(analysis.get(field), list):
            # Ensure all items are strings
            validated[field] = [str(item) for item in analysis[field] if item]
    
    # Validate evals
    if isinstance(analysis.get('evals'), list):
        validated_evals = []
        for eval_item in analysis['evals']:
            if isinstance(eval_item, dict) and 'name' in eval_item and 'score' in eval_item:
                try:
                    validated_evals.append({
                        'name': str(eval_item['name']),
                        'score': float(eval_item['score'])
                    })
                except (ValueError, TypeError):
                    continue
        validated['evals'] = validated_evals
    
    return validated 