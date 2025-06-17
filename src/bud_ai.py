import os
import openai
import requests
from typing import Optional, Dict, Any, List
import logging

# Import settings manager
from .db import LLMSettingsManager

# Initialize settings manager
settings_manager = LLMSettingsManager()

# Set up logging
logger = logging.getLogger(__name__)


def call_bud_LLM(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
    temperature: float = 0.1,
    max_tokens: int = 3000,
    top_p: float = 1.0,
    frequency_penalty: float = 0.7,
    presence_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Call the LLM API with the given prompt and parameters.
    Tries the primary server first, then falls back to OpenAI if configured.
    
    Args:
        prompt: The prompt to send to the API
        model: The model to use (if None, uses default from settings)
        system_prompt: System prompt for the conversation
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        top_p: Controls diversity via nucleus sampling
        frequency_penalty: Penalizes repeated tokens
        presence_penalty: Penalizes repeated topics
        stop: List of strings that stop generation when encountered
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        The generated text response
    """
    # Get base URL and model from settings with fallback to defaults
    base_url = settings_manager.get('llm_base_url', '')
    default_model = settings_manager.get('llm_model', '')
    
    # Use provided model or default from settings
    if model is None:
        model = default_model
    
    # Try primary server first
    try:
        # Create a client instance for the primary server
        logger.info(f"Calling LLM at {base_url} with model {model}")
        primary_client = openai.OpenAI(base_url=base_url, api_key="sk-xxx")
        
        # Prepare the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Create the completion with the new API format
        response = primary_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    
    except Exception as primary_error:
        logger.error(f"Failed to call primary LLM: {primary_error}")
        return None


def update_llm_url_and_model(base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, bool]:
    """
    Update LLM base URL and model in the database.
    
    Args:
        base_url: New base URL for LLM API
        model: New default model
        
    Returns:
        Dictionary of setting_key -> success status
    """
    results = {}
    
    if base_url is not None:
        results['llm_base_url'] = settings_manager.set('llm_base_url', base_url)
    
    if model is not None:
        results['llm_model'] = settings_manager.set('llm_model', model)
    
    # Clear cache to ensure fresh values are loaded
    settings_manager.clear_cache()
    
    return results


def get_llm_url_and_model() -> Dict[str, str]:
    """
    Get current LLM base URL and model from the database.
    
    Returns:
        Dictionary with base_url and model
    """
    return {
        'base_url': settings_manager.get('llm_base_url', 'http://20.66.97.208/v1/'),
        'model': settings_manager.get('llm_model', 'Qwen/QwQ-32B')
    }