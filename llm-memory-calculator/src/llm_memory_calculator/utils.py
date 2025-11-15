"""Utility functions for memory calculation."""

from typing import Dict, Any, List, Optional, Union

from .calculator import ModelMemoryCalculator
from .types import MemoryReport
from .huggingface_loader import HuggingFaceConfigLoader
from .lora.config import LoraConfig


def calculate_memory(
    model_id_or_config: Union[str, Dict[str, Any]],
    batch_size: int = 1,
    seq_length: int = 2048,
    precision: str = 'fp16',
    tensor_parallel: int = 1,
    framework_overhead: float = 1.2,
    include_gradients: bool = False,
    token: Optional[str] = None,
    respect_weight_tying: bool = True,
    # LoRA parameters (simplified API)
    max_loras: Optional[int] = None,
    max_lora_rank: Optional[int] = None,
    lora_dtype: Optional[str] = None,
    fully_sharded_loras: Optional[bool] = None,
    target_modules: Optional[List[str]] = None,
    # Advanced: pass LoraConfig directly (overrides individual parameters)
    lora_config: Optional[LoraConfig] = None,
    **kwargs
) -> MemoryReport:
    """
    Calculate memory requirements for a model.

    This is a convenience function that handles both HuggingFace model IDs
    and direct config dictionaries.

    Args:
        model_id_or_config: HuggingFace model ID or config dictionary
        batch_size: Batch size for inference
        seq_length: Maximum sequence length
        precision: Model precision (fp32, fp16, bf16, int8, int4)
        tensor_parallel: Tensor parallelism degree
        framework_overhead: Multiplicative overhead for framework memory
        include_gradients: Include gradient memory (for training)
        token: Optional HuggingFace API token
        respect_weight_tying: Whether to respect tie_word_embeddings config for parameter
                             counting (default True). Set to False for accurate memory
                             estimation that counts all physical tensors.

        max_loras: Maximum number of LoRA adapters to serve simultaneously
        max_lora_rank: Maximum rank across all LoRA adapters
        lora_dtype: Data type for LoRA weights ('auto', 'fp16', 'bf16', 'fp32')
        fully_sharded_loras: Whether to shard both A and B matrices with TP
        target_modules: Which modules to apply LoRA to (default: ['attn', 'ffn'])

        lora_config: Advanced - pass LoraConfig object directly (overrides other LoRA params)
        **kwargs: Additional arguments passed to calculator

    Returns:
        MemoryReport with detailed breakdown

    Examples:
        # Basic usage
        report = calculate_memory("meta-llama/Llama-2-7b-hf")

        # With LoRA adapters (simplified API)
        report = calculate_memory(
            "meta-llama/Llama-2-7b-hf",
            max_loras=5,
            max_lora_rank=256
        )

        # With custom LoRA configuration
        report = calculate_memory(
            "meta-llama/Llama-2-7b-hf",
            max_loras=5,
            max_lora_rank=256,
            target_modules=['attn'],  # attention only
            fully_sharded_loras=True
        )

        # Advanced: using LoraConfig object
        from llm_memory_calculator.lora.config import LoraConfig
        lora_cfg = LoraConfig(enabled=True, max_loras=5, max_lora_rank=256)
        report = calculate_memory("meta-llama/Llama-2-7b-hf", lora_config=lora_cfg)
    """
    # Handle config vs model ID
    if isinstance(model_id_or_config, str):
        # It's a HuggingFace model ID
        loader = HuggingFaceConfigLoader(token=token)
        config = loader.get_model_config(model_id_or_config, respect_weight_tying=respect_weight_tying)
    else:
        # It's already a config dictionary
        config = model_id_or_config

    # Auto-create LoraConfig if individual parameters are provided
    if lora_config is None and (max_loras is not None or max_lora_rank is not None):
        lora_config = LoraConfig(
            enabled=True,
            max_loras=max_loras or 1,
            max_lora_rank=max_lora_rank or 256,
            target_modules=target_modules or ['attn', 'ffn'],
            lora_dtype=lora_dtype or 'auto',
            fully_sharded_loras=fully_sharded_loras or False
        )

    # Create calculator and compute memory
    calculator = ModelMemoryCalculator()
    return calculator.calculate_total_memory(
        config,
        batch_size=batch_size,
        seq_length=seq_length,
        precision=precision,
        tensor_parallel=tensor_parallel,
        framework_overhead=framework_overhead,
        include_gradients=include_gradients,
        lora_config=lora_config,
        respect_weight_tying=respect_weight_tying,
        **kwargs
    )


def estimate_memory(config: Dict[str, Any], **kwargs) -> MemoryReport:
    """
    Convenience function matching the original BudSimulator API.
    
    Args:
        config: Model configuration dictionary
        **kwargs: Arguments passed to calculator
        
    Returns:
        MemoryReport with detailed breakdown
    """
    calculator = ModelMemoryCalculator()
    return calculator.calculate_total_memory(config, **kwargs)


def analyze_hf_model(
    model_id: str,
    seq_length: int = 2048,
    batch_size: int = 1,
    precision: str = 'fp16',
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model identifier
        seq_length: Sequence length for memory calculation
        batch_size: Batch size for memory calculation
        precision: Model precision
        token: Optional HuggingFace API token
        **kwargs: Additional arguments for memory calculation
        
    Returns:
        Dictionary with model info and memory analysis
    """
    loader = HuggingFaceConfigLoader(token=token)
    return loader.analyze_model(
        model_id,
        seq_length=seq_length,
        batch_size=batch_size,
        precision=precision,
        **kwargs
    )


def compare_models(
    model_ids: List[str],
    seq_length: int = 2048,
    precision: str = 'fp16',
    token: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Compare memory requirements for multiple models.
    
    Args:
        model_ids: List of HuggingFace model identifiers
        seq_length: Sequence length for comparison
        precision: Model precision for comparison
        token: Optional HuggingFace API token
        **kwargs: Additional arguments for memory calculation
        
    Returns:
        List of analysis results
    """
    loader = HuggingFaceConfigLoader(token=token)
    return loader.compare_models(
        model_ids,
        seq_length=seq_length,
        precision=precision,
        **kwargs
    )


def estimate_max_batch_size(
    model_id_or_config: Union[str, Dict[str, Any]],
    gpu_memory_gb: float,
    seq_length: int = 2048,
    precision: str = 'fp16',
    token: Optional[str] = None,
    respect_weight_tying: bool = True
) -> int:
    """
    Estimate maximum batch size that fits in given GPU memory.
    
    Args:
        model_id_or_config: HuggingFace model ID or config dictionary
        gpu_memory_gb: Available GPU memory in GB
        seq_length: Sequence length
        precision: Model precision
        token: Optional HuggingFace API token
        respect_weight_tying: Whether to respect tie_word_embeddings config (default True)

    Returns:
        Maximum batch size that fits in memory
    """
    # Get config
    if isinstance(model_id_or_config, str):
        loader = HuggingFaceConfigLoader(token=token)
        config = loader.get_model_config(model_id_or_config, respect_weight_tying=respect_weight_tying)
    else:
        config = model_id_or_config
    
    calculator = ModelMemoryCalculator()
    
    # Binary search for optimal batch size
    low, high = 1, 1024
    result = 1
    
    while low <= high:
        mid = (low + high) // 2
        report = calculator.calculate_total_memory(
            config, 
            batch_size=mid, 
            seq_length=seq_length, 
            precision=precision
        )
        
        if report.total_memory_gb <= gpu_memory_gb * 0.9:  # Leave 10% buffer
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return result


def analyze_attention_efficiency(
    model_id_or_config: Union[str, Dict[str, Any]],
    seq_lengths: List[int] = [1024, 4096, 16384, 32768],
    batch_size: int = 1,
    precision: str = 'fp16',
    token: Optional[str] = None,
    respect_weight_tying: bool = True
) -> Dict[str, Any]:
    """
    Analyze KV cache memory efficiency for different sequence lengths.
    
    Args:
        model_id_or_config: HuggingFace model ID or config dictionary
        seq_lengths: List of sequence lengths to analyze
        batch_size: Batch size
        precision: Model precision
        token: Optional HuggingFace API token
        respect_weight_tying: Whether to respect tie_word_embeddings config (default True)

    Returns:
        Dictionary with efficiency analysis
    """
    # Get config
    if isinstance(model_id_or_config, str):
        loader = HuggingFaceConfigLoader(token=token)
        config = loader.get_model_config(model_id_or_config, respect_weight_tying=respect_weight_tying)
        model_name = model_id_or_config
    else:
        config = model_id_or_config
        model_name = config.get('_name_or_path', 'Custom Model')
    
    calculator = ModelMemoryCalculator()
    attention_type = calculator.detect_attention_type(config)
    
    results = {}
    for seq_len in seq_lengths:
        report = calculator.calculate_total_memory(
            config,
            batch_size=batch_size,
            seq_length=seq_len,
            precision=precision
        )
        
        results[seq_len] = {
            'total_memory_gb': report.total_memory_gb,
            'kv_cache_gb': report.kv_cache_gb,
            'kv_cache_percent': (report.kv_cache_gb / report.total_memory_gb * 100) if report.total_memory_gb > 0 else 0
        }
    
    # Calculate memory per token
    if len(seq_lengths) >= 2:
        seq_diff = seq_lengths[-1] - seq_lengths[0]
        kv_diff = results[seq_lengths[-1]]['kv_cache_gb'] - results[seq_lengths[0]]['kv_cache_gb']
        memory_per_token_bytes = int((kv_diff * 1e9) / seq_diff) if seq_diff > 0 else 0
    else:
        memory_per_token_bytes = 0
    
    return {
        'model': model_name,
        'attention_type': attention_type,
        'results': results,
        'memory_per_token_bytes': memory_per_token_bytes,
        'efficiency_rating': _get_efficiency_rating(attention_type)
    }


def _get_efficiency_rating(attention_type: Optional[str]) -> str:
    """Get efficiency rating based on attention type."""
    if attention_type == 'mla':
        return 'excellent'
    elif attention_type in ['gqa', 'mqa']:
        return 'good'
    elif attention_type == 'mha':
        return 'moderate'
    else:
        return 'unknown'