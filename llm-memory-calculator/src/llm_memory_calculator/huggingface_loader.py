"""HuggingFace model configuration loader with local path support."""

import json
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

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
    
    def _is_local_path(self, model_id_or_path: str) -> bool:
        """
        Check if the input is a local path or a HuggingFace model ID.
        
        Args:
            model_id_or_path: Either a HuggingFace model ID or a local path
            
        Returns:
            True if it's a local path, False if it's a HuggingFace model ID
        """
        # Check if it's an existing local path
        path = Path(model_id_or_path)
        if path.exists():
            return True
        
        # Check if it looks like a file path (contains / or \ but not in HF format)
        # HuggingFace IDs typically have format "organization/model-name"
        if ('/' in model_id_or_path or '\\' in model_id_or_path):
            # If it has more than one slash or contains backslashes, it's likely a path
            if model_id_or_path.count('/') > 1 or '\\' in model_id_or_path:
                return True
            # If it starts with ./ or ../ or / it's a path
            if model_id_or_path.startswith('./') or model_id_or_path.startswith('../') or model_id_or_path.startswith('/'):
                return True
        
        # Otherwise assume it's a HuggingFace model ID
        return False
    
    def _load_local_config(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a local path.
        
        Args:
            path: Local path to the model directory or config file
            
        Returns:
            Config dictionary loaded from local file
            
        Raises:
            FileNotFoundError: If no config file is found
            json.JSONDecodeError: If config file is not valid JSON
        """
        path_obj = Path(path)
        
        # If path is a directory, look for config files
        if path_obj.is_dir():
            config_filenames = [
                "config.json",
                "model_config.json",
                "configuration.json",
            ]
            
            for filename in config_filenames:
                config_path = path_obj / filename
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        return json.load(f)
            
            raise FileNotFoundError(f"No config file found in directory: {path}")
        
        # If path is a file, load it directly
        elif path_obj.is_file():
            if path_obj.suffix == '.json':
                with open(path_obj, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Config file must be a JSON file, got: {path_obj.suffix}")
        
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
        
    def fetch_model_config(self, model_id_or_path: str) -> Dict[str, Any]:
        """
        Fetch model configuration from HuggingFace Hub or local path.
        
        Args:
            model_id_or_path: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
                             or local path to model directory/config file
            
        Returns:
            Raw config dictionary
            
        Raises:
            RepositoryNotFoundError: If HuggingFace model doesn't exist
            GatedRepoError: If HuggingFace model requires access request
            FileNotFoundError: If local config file doesn't exist
            Exception: For other errors
        """
        # Check if it's a local path
        if self._is_local_path(model_id_or_path):
            return self._load_local_config(model_id_or_path)
        
        # Otherwise, treat as HuggingFace model ID
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
                    repo_id=model_id_or_path,
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
                    f"Model {model_id_or_path} is gated. Please request access on HuggingFace "
                    f"and provide your token. Error: {str(e)}"
                )
                
            except Exception as e:
                # Store error and try next filename
                last_error = e
                continue
        
        # If we get here, no config file was found
        if last_error:
            raise Exception(f"Could not find config file for {model_id_or_path}. Last error: {last_error}")
        else:
            raise Exception(f"No config file found for {model_id_or_path}")
    
    def get_model_config(self, model_id_or_path: str, add_param_count: bool = True, respect_weight_tying: bool = True) -> Dict[str, Any]:
        """
        Get model configuration with enhanced parameter counting.

        This method fetches the config and optionally adds parameter count from model metadata.

        Args:
            model_id_or_path: HuggingFace model identifier or local path
            add_param_count: Whether to try to add parameter count from model info
            respect_weight_tying: Whether to respect tie_word_embeddings in parameter counting

        Returns:
            Enhanced config dictionary with num_parameters
        """
        config = self.fetch_model_config(model_id_or_path)
        
        # Try to add parameter count from model info if requested (only for HuggingFace models)
        if add_param_count and 'num_parameters' not in config and not self._is_local_path(model_id_or_path):
            try:
                from huggingface_hub import model_info
                info = model_info(model_id_or_path, token=self.token)
                
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
            config['num_parameters'] = counter.count_parameters(config, respect_weight_tying=respect_weight_tying)

        return config
    
    def get_model_info(self, model_id_or_path: str) -> Any:
        """
        Get model information from HuggingFace Hub.
        
        Args:
            model_id_or_path: HuggingFace model identifier or local path
            
        Returns:
            ModelInfo object from HuggingFace Hub API, or None for local paths
        """
        # Local paths don't have HuggingFace model info
        if self._is_local_path(model_id_or_path):
            return None
            
        try:
            return self.api.model_info(model_id_or_path, token=self.token)
        except Exception as e:
            warnings.warn(f"Could not fetch model info for {model_id_or_path}: {e}")
            return None
    
    def analyze_model(
        self,
        model_id_or_path: str,
        seq_length: int = 2048,
        batch_size: int = 1,
        precision: str = 'fp16',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze a model from HuggingFace Hub.
        
        Args:
            model_id_or_path: HuggingFace model identifier or local path
            seq_length: Sequence length for memory calculation
            batch_size: Batch size for memory calculation
            precision: Model precision
            **kwargs: Additional arguments for memory calculation
            
        Returns:
            Dictionary with model info and memory analysis
        """
        # Get config
        config = self.get_model_config(model_id_or_path)
        
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
        model_info = self.get_model_info(model_id_or_path)
        
        # Build analysis dictionary
        analysis = {
            'model_id': model_id_or_path,
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
        model_ids_or_paths: List[str],
        seq_length: int = 2048,
        precision: str = 'fp16',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Compare memory requirements for multiple models.
        
        Args:
            model_ids_or_paths: List of HuggingFace model identifiers or local paths
            seq_length: Sequence length for comparison
            precision: Model precision for comparison
            **kwargs: Additional arguments for memory calculation
            
        Returns:
            List of analysis results
        """
        results = []
        
        for model_id_or_path in model_ids_or_paths:
            try:
                analysis = self.analyze_model(
                    model_id_or_path, 
                    seq_length=seq_length,
                    precision=precision,
                    **kwargs
                )
                results.append(analysis)
            except Exception as e:
                warnings.warn(f"Failed to analyze {model_id_or_path}: {e}")
                results.append({
                    'model_id': model_id_or_path,
                    'error': str(e)
                })
        
        return results