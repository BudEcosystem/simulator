import math
from typing import Dict, Any, Optional, Tuple, List, Union
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    from huggingface_hub import HfApi, hf_hub_download, ModelCard
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    raise ImportError(
        "Please install huggingface_hub: pip install huggingface_hub"
    )

# Import from llm-memory-calculator package
from llm_memory_calculator import (
    UniversalParameterCounter,
    HuggingFaceConfigLoader,
    ModelMemoryCalculator,
    MemoryReport,
    estimate_memory,
    analyze_hf_model,
    estimate_max_batch_size,
    analyze_attention_efficiency,
)

# Re-export for backward compatibility
__all__ = [
    'UniversalParameterCounter',
    'HuggingFaceConfigLoader', 
    'ModelMemoryCalculator',
    'MemoryReport',
    'estimate_memory',
    'analyze_hf_model',
    'get_model_config_from_hf',
    'compare_hf_models',
    'estimate_max_sequence_length',
    'estimate_max_batch_size',
    'analyze_attention_efficiency',
]


def estimate_max_sequence_length(config: Dict[str, Any], 
                                 gpu_memory_gb: float,
                                 batch_size: int = 1,
                                 precision: str = 'fp16',
                                 overhead_percent: float = 0.1) -> int:
    """
    Estimate the maximum sequence length that can fit in GPU memory.
    
    Args:
        config: Model configuration
        gpu_memory_gb: Available GPU memory in GB
        batch_size: Batch size
        precision: Precision format
        overhead_percent: Reserve for overhead (0.1 = 10%)
        
    Returns:
        Maximum sequence length
    """
    calculator = ModelMemoryCalculator()
    
    # Binary search for max sequence length
    low, high = 1, 128000
    available_memory = gpu_memory_gb * (1 - overhead_percent)
    
    result = 0
    while low <= high:
        mid = (low + high) // 2
        report = calculator.calculate_memory(
            config, 
            seq_length=mid, 
            batch_size=batch_size,
            precision=precision
        )
        
        if report.total_memory_gb <= available_memory:
            result = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return result


# Convenience functions for backward compatibility
def get_model_config_from_hf(
    model_id: str, 
    token: Optional[str] = None,
    add_param_count: bool = True
) -> Dict[str, Any]:
    """
    Fetch and prepare model config from HuggingFace for memory estimation.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        token: Optional HuggingFace API token for private models
        add_param_count: Whether to try to find and add parameter count
        
    Returns:
        Config dictionary ready for memory estimation
    """
    loader = HuggingFaceConfigLoader(token=token)
    return loader.get_model_config(model_id, add_param_count=add_param_count)


def compare_hf_models(
    model_ids: List[str],
    seq_length: int = 2048,
    precision: str = 'fp16',
    token: Optional[str] = None,
    print_results: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Compare memory requirements of multiple HuggingFace models.
    
    Args:
        model_ids: List of HuggingFace model identifiers
        seq_length: Sequence length for comparison
        precision: Precision format
        token: Optional HuggingFace API token
        print_results: Whether to print a comparison table
        **kwargs: Additional arguments for estimate_memory
        
    Returns:
        List of analysis results
    """
    loader = HuggingFaceConfigLoader(token=token)
    results = loader.compare_models(
        model_ids,
        seq_length=seq_length,
        precision=precision,
        **kwargs
    )
    
    if print_results:
        print(f"\nComparing models at {seq_length} token context, {precision} precision:")
        print("=" * 90)
        print(f"{'Model':<40} {'Type':<15} {'Attention':<10} {'Memory':<10} {'GPU':<15}")
        print("-" * 90)
        
        for result in results:
            if 'error' in result:
                print(f"{result['model_id']:<40} Error: {result['error'][:40]}...")
            else:
                model_name = result['model_id'].split('/')[-1][:35]
                model_type = result['model_type'][:12]
                attention = result['attention_type'] or 'n/a'
                memory = result['memory_analysis']['total_memory_gb']
                gpu = f"{result['deployment_recommendations']['min_gpu_memory_gb']}GB"
                
                print(f"{model_name:<40} {model_type:<15} {attention:<10} {memory:<10.1f} {gpu:<15}")
        
        print("=" * 90)
    
    return results


if __name__ == "__main__":
    print("HuggingFace Model Config Loader (using HfApi)")
    print("=" * 50)
    
    # Initialize loader (can also use environment variable HF_TOKEN)
    loader = HuggingFaceConfigLoader()
    
    # Example 1: Load a single model
    print("\n1. Loading model config:")
    try:
        config = loader.get_model_config("mistralai/Magistral-Small-2506")
        print(f"✓ Loaded config for mistralai/Magistral-Small-2506")
        print(f"  Model type: {config.get('model_type')}")
        print(f"  Parameters: {config.get('num_parameters', 'not found'):,}")
        
        # Estimate memory
        try:
            result = estimate_memory(config, seq_length=2048, batch_size=8)
            print(f"  Memory at 2K context: {result.total_memory_gb:.1f} GB")
            print(result)
        except Exception as mem_error:
            print(f"  Memory estimation error: {mem_error}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Analyze a model
    print("\n2. Analyzing a model:")
    try:
        analysis = loader.analyze_model("mistralai/Pixtral-12B-Base-2409", seq_length=4096)
        print(analysis)
        print(f"✓ Analysis complete for {analysis['model_id']}")
        print(f"  Type: {analysis['model_type']}")
        print(f"  Attention: {analysis['attention_type']}")
        print(f"  Memory: {analysis['memory_analysis']['total_memory_gb']:.1f} GB")
        print(f"  Downloads: {analysis['model_popularity']['downloads']:,}")
        
    except Exception as e:
        print(f"Error: {e}")