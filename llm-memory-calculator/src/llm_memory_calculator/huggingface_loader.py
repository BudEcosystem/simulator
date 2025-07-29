"""HuggingFace model configuration loader."""

import json
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, ModelCard
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    raise ImportError(
        "Please install huggingface_hub: pip install huggingface_hub"
    )

from .calculator import ModelMemoryCalculator
from .types import MemoryReport


class HuggingFaceConfigLoader:
    """Load and analyze model configs from HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            token: Optional HuggingFace API token for private models.
                   Can also be set via HF_TOKEN environment variable.
        """
        self.api = HfApi(token=token)
        self.token = token
        
    def fetch_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Fetch model configuration from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            
        Returns:
            Raw config dictionary from HuggingFace
            
        Raises:
            RepositoryNotFoundError: If model doesn't exist
            GatedRepoError: If model requires access request
            Exception: For other errors
        """
        # List of possible config file names in order of preference
        config_filenames = [
            "config.json",           # Standard transformers config
            "model_config.json",     # Alternative naming
            "configuration.json",    # Another alternative
        ]
        
        last_error = None
        
        for filename in config_filenames:
            try:
                # Download config file
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=self.token
                )
                
                # Load and parse config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                return config
                
            except RepositoryNotFoundError:
                raise  # Re-raise immediately for repo not found
                
            except GatedRepoError as e:
                # For gated repos, we want to provide a clear error
                raise GatedRepoError(
                    f"Model {model_id} is gated. Please request access on HuggingFace "
                    f"and provide your token. Error: {str(e)}"
                )
                
            except Exception as e:
                # Store error and try next filename
                last_error = e
                continue
        
        # If we get here, no config file was found
        if last_error:
            raise Exception(f"Could not find config file for {model_id}. Last error: {last_error}")
        else:
            raise Exception(f"No config file found for {model_id}")
    
    def get_model_config(self, model_id: str, add_param_count: bool = True) -> Dict[str, Any]:
        """
        Get model configuration with enhanced parameter counting.
        
        This method fetches the config and optionally adds parameter count from model metadata.
        
        Args:
            model_id: HuggingFace model identifier
            add_param_count: Whether to try to add parameter count from model info
            
        Returns:
            Enhanced config dictionary with num_parameters
        """
        config = self.fetch_model_config(model_id)
        
        # Try to add parameter count from model info if requested
        if add_param_count and 'num_parameters' not in config:
            try:
                from huggingface_hub import model_info
                info = model_info(model_id, token=self.token)
                
                # Check safetensors metadata for total parameter count
                if hasattr(info, 'safetensors') and info.safetensors:
                    if 'total' in info.safetensors:
                        config['num_parameters'] = info.safetensors['total']
            except Exception:
                # If we can't get the info, just continue without it
                pass
        
        # Calculate parameter count if still not present
        if 'num_parameters' not in config:
            from .parameter_counter import UniversalParameterCounter
            counter = UniversalParameterCounter()
            config['num_parameters'] = counter.count_parameters(config)
            
        return config
    
    def get_model_info(self, model_id: str) -> Any:
        """
        Get model information from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            ModelInfo object from HuggingFace Hub API
        """
        try:
            return self.api.model_info(model_id, token=self.token)
        except Exception as e:
            warnings.warn(f"Could not fetch model info for {model_id}: {e}")
            return None
    
    def analyze_model(
        self,
        model_id: str,
        seq_length: int = 2048,
        batch_size: int = 1,
        precision: str = 'fp16',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze a model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model identifier
            seq_length: Sequence length for memory calculation
            batch_size: Batch size for memory calculation
            precision: Model precision
            **kwargs: Additional arguments for memory calculation
            
        Returns:
            Dictionary with model info and memory analysis
        """
        # Get config
        config = self.get_model_config(model_id)
        
        # Create calculator and run analysis
        calculator = ModelMemoryCalculator()
        result = calculator.calculate_total_memory(
            config,
            batch_size=batch_size,
            seq_length=seq_length,
            precision=precision,
            **kwargs
        )
        
        # Get model info
        model_info = self.get_model_info(model_id)
        
        # Build analysis dictionary
        analysis = {
            'model_id': model_id,
            'model_type': result.model_type,
            'attention_type': result.attention_type,
            'parameter_count': result.parameter_count,
            'memory_report': result,
            'config': config,
        }
        
        # Add model info if available
        if model_info:
            analysis['model_info'] = {
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'tags': getattr(model_info, 'tags', []),
                'created_at': str(getattr(model_info, 'created_at', '')),
                'last_modified': str(getattr(model_info, 'last_modified', '')),
            }
        
        return analysis
    
    def compare_models(
        self,
        model_ids: List[str],
        seq_length: int = 2048,
        precision: str = 'fp16',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Compare memory requirements for multiple models.
        
        Args:
            model_ids: List of HuggingFace model identifiers
            seq_length: Sequence length for comparison
            precision: Model precision for comparison
            **kwargs: Additional arguments for memory calculation
            
        Returns:
            List of analysis results
        """
        results = []
        
        for model_id in model_ids:
            try:
                analysis = self.analyze_model(
                    model_id, 
                    seq_length=seq_length,
                    precision=precision,
                    **kwargs
                )
                results.append(analysis)
            except Exception as e:
                warnings.warn(f"Failed to analyze {model_id}: {e}")
                results.append({
                    'model_id': model_id,
                    'error': str(e)
                })
        
        return results