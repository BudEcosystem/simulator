"""
Performance estimation for LLM inference using GenZ modeling.

This module provides functions to estimate detailed performance metrics
for LLM inference on various hardware configurations.
"""

from typing import Dict, Optional, Union, List
import warnings

# Import GenZ performance modeling functions
try:
    from .genz.LLM_inference import (
        prefill_moddeling,
        decode_moddeling,
        chunked_moddeling
    )
    HAS_GENZ = True
except ImportError as e:
    HAS_GENZ = False
    warnings.warn(
        f"Internal GenZ not available: {e}. Performance estimation not available.",
        ImportWarning
    )


def estimate_prefill_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    input_tokens: int = 2048,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False,
    **kwargs
) -> Dict:
    """
    Estimate prefill phase performance for LLM inference.
    
    The prefill phase processes the input prompt and generates the first token.
    
    Args:
        model: Model identifier (e.g., 'llama2_7b', 'llama2_70b')
        batch_size: Number of sequences to process in parallel
        input_tokens: Length of input sequence (prompt length)
        system_name: Hardware configuration dict with keys:
            - Flops: TFLOPS
            - Memory_size: Memory in GB
            - Memory_BW: Memory bandwidth in GB/s
            - ICN: Interconnect bandwidth in GB/s
            - real_values: Whether to use real hardware values
        bits: Precision ('fp32', 'bf16', 'int8', 'int4')
        tensor_parallel: Tensor parallelism degree
        pipeline_parallel: Pipeline parallelism degree
        expert_parallel: Expert parallelism degree (for MoE models)
        debug: Whether to print debug information
        
    Returns:
        Dictionary with performance metrics:
        - Latency: Total latency in milliseconds
        - Throughput: Throughput in tokens/second
        - TTFT: Time to first token in milliseconds
        - Memory_used: Memory usage breakdown
        - Compute_breakdown: Detailed compute analysis
        - Communication_breakdown: Communication overhead analysis
        
    Raises:
        ImportError: If GenZ is not available
        ValueError: If model or configuration is invalid
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        # Default to A100 80GB configuration
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    try:
        result = prefill_moddeling(
            model=model,
            batch_size=batch_size,
            input_tokens=input_tokens,
            system_name=system_name,
            bits=bits,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            debug=debug
        )
        
        # Add computed metrics
        if 'Latency' in result and result['Latency'] > 0:
            # Throughput calculation: tokens processed / time
            total_tokens = batch_size * input_tokens
            result['Input_throughput'] = total_tokens / (result['Latency'] / 1000)  # tokens/s
            result['TTFT'] = result['Latency']  # Time to first token
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to estimate prefill performance: {str(e)}")


def estimate_decode_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    beam_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False
) -> Dict:
    """
    Estimate decode phase performance for LLM inference.
    
    The decode phase generates output tokens one by one after the prefill phase.
    
    Args:
        model: Model identifier
        batch_size: Number of sequences to process in parallel
        beam_size: Beam search width (1 for greedy decoding)
        input_tokens: Length of input sequence (context length)
        output_tokens: Number of tokens to generate
        system_name: Hardware configuration dict
        bits: Precision
        tensor_parallel: Tensor parallelism degree
        pipeline_parallel: Pipeline parallelism degree
        expert_parallel: Expert parallelism degree
        debug: Whether to print debug information
        
    Returns:
        Dictionary with performance metrics:
        - Latency: Per-token latency in milliseconds
        - Throughput: Throughput in tokens/second
        - Total_latency: Total generation time in milliseconds
        - TPOT: Time per output token in milliseconds
        - Memory_used: Memory usage breakdown
        - Compute_breakdown: Detailed compute analysis
        - Communication_breakdown: Communication overhead analysis
        
    Raises:
        ImportError: If GenZ is not available
        ValueError: If model or configuration is invalid
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    try:
        result = decode_moddeling(
            model=model,
            batch_size=batch_size,
            Bb=beam_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_name=system_name,
            bits=bits,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            debug=debug
        )
        
        # Add computed metrics
        if 'Latency' in result:
            result['TPOT'] = result['Latency']  # Time per output token
            result['Total_latency'] = result['Latency'] * output_tokens  # Total generation time
            
            # Effective throughput considering batch size and beam size
            effective_batch = batch_size * beam_size
            if result['Latency'] > 0:
                result['Effective_throughput'] = effective_batch * 1000 / result['Latency']  # tokens/s
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to estimate decode performance: {str(e)}")


def estimate_end_to_end_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    beam_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False
) -> Dict:
    """
    Estimate end-to-end performance combining prefill and decode phases.
    
    Args:
        model: Model identifier
        batch_size: Number of sequences to process in parallel
        beam_size: Beam search width
        input_tokens: Length of input sequence
        output_tokens: Number of tokens to generate  
        system_name: Hardware configuration dict
        bits: Precision
        tensor_parallel: Tensor parallelism degree
        pipeline_parallel: Pipeline parallelism degree
        expert_parallel: Expert parallelism degree
        debug: Whether to print debug information
        
    Returns:
        Dictionary with comprehensive performance metrics:
        - prefill: Prefill phase results
        - decode: Decode phase results  
        - total_latency: End-to-end latency in milliseconds
        - ttft: Time to first token in milliseconds
        - average_tpot: Average time per output token in milliseconds
        - total_throughput: Overall throughput in tokens/second
        - memory_peak: Peak memory usage
        
    Raises:
        ImportError: If GenZ is not available
        ValueError: If model or configuration is invalid
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    # Get prefill performance
    prefill_result = estimate_prefill_performance(
        model=model,
        batch_size=batch_size,
        input_tokens=input_tokens,
        system_name=system_name,
        bits=bits,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=expert_parallel,
        debug=debug
    )
    
    # Get decode performance
    decode_result = estimate_decode_performance(
        model=model,
        batch_size=batch_size,
        beam_size=beam_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=system_name,
        bits=bits,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=expert_parallel,
        debug=debug
    )
    
    # Combine results
    total_latency = prefill_result.get('Latency', 0) + decode_result.get('Total_latency', 0)
    ttft = prefill_result.get('TTFT', prefill_result.get('Latency', 0))
    tpot = decode_result.get('TPOT', decode_result.get('Latency', 0))
    
    total_tokens = batch_size * (input_tokens + output_tokens)
    total_throughput = 0
    if total_latency > 0:
        total_throughput = total_tokens * 1000 / total_latency  # tokens/s
    
    return {
        'prefill': prefill_result,
        'decode': decode_result,
        'total_latency': total_latency,
        'ttft': ttft,
        'average_tpot': tpot,
        'total_throughput': total_throughput,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'batch_size': batch_size,
        'beam_size': beam_size,
        'model': model,
        'parallelism': {
            'tensor_parallel': tensor_parallel,
            'pipeline_parallel': pipeline_parallel,
            'expert_parallel': expert_parallel
        }
    }


def estimate_chunked_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    chunk_size: int = 512,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    debug: bool = False
) -> Dict:
    """
    Estimate performance for chunked prefill processing.
    
    Chunked prefill processes long sequences in smaller chunks to optimize
    memory usage and enable overlapping of compute and communication.
    
    Args:
        model: Model identifier
        batch_size: Number of sequences to process
        input_tokens: Total input sequence length
        output_tokens: Number of tokens to generate
        chunk_size: Size of each chunk for processing
        system_name: Hardware configuration dict
        bits: Precision
        tensor_parallel: Tensor parallelism degree
        pipeline_parallel: Pipeline parallelism degree
        debug: Whether to print debug information
        
    Returns:
        Dictionary with chunked processing performance metrics
        
    Raises:
        ImportError: If GenZ is not available
        ValueError: If model or configuration is invalid
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    if system_name is None:
        system_name = {
            'Flops': 312,
            'Memory_size': 80,
            'Memory_BW': 2039,
            'ICN': 600,
            'real_values': True
        }
    
    try:
        result = chunked_moddeling(
            model=model,
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            chunk_size=chunk_size,
            system_name=system_name,
            bits=bits,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            debug=debug
        )
        
        # Add chunked-specific metrics
        num_chunks = (input_tokens + chunk_size - 1) // chunk_size
        result['num_chunks'] = num_chunks
        result['chunk_size'] = chunk_size
        result['chunked_processing'] = True
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to estimate chunked performance: {str(e)}")


def compare_performance_configurations(
    model: str,
    configurations: List[Dict],
    batch_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    debug: bool = False
) -> List[Dict]:
    """
    Compare performance across multiple hardware and parallelism configurations.
    
    Args:
        model: Model identifier
        configurations: List of configuration dicts, each containing:
            - name: Configuration name for identification
            - system_name: Hardware configuration
            - tensor_parallel: Tensor parallelism degree
            - pipeline_parallel: Pipeline parallelism degree
            - bits: Precision
            - Any other parameters for performance estimation
        batch_size: Batch size for comparison
        input_tokens: Input sequence length
        output_tokens: Number of tokens to generate
        debug: Whether to print debug information
        
    Returns:
        List of dictionaries with performance results for each configuration,
        sorted by total throughput (descending)
        
    Example:
        configs = [
            {
                'name': 'A100_80GB_TP4',
                'system_name': get_hardware_config('A100_80GB'),
                'tensor_parallel': 4,
                'pipeline_parallel': 1,
                'bits': 'bf16'
            },
            {
                'name': 'H100_80GB_TP8', 
                'system_name': get_hardware_config('H100_80GB'),
                'tensor_parallel': 8,
                'pipeline_parallel': 1,
                'bits': 'bf16'
            }
        ]
        results = compare_performance_configurations('llama2_7b', configs)
    """
    if not HAS_GENZ:
        raise ImportError(
            "Internal GenZ not available. This indicates a package installation issue."
        )
    
    results = []
    
    for config in configurations:
        config_name = config.get('name', 'unnamed_config')
        
        try:
            # Extract configuration parameters
            system_name = config.get('system_name')
            tensor_parallel = config.get('tensor_parallel', 1)
            pipeline_parallel = config.get('pipeline_parallel', 1)
            # data_parallel not used in GenZ functions
            expert_parallel = config.get('expert_parallel', 1)
            bits = config.get('bits', 'bf16')
            beam_size = config.get('beam_size', 1)
            
            # Get end-to-end performance
            perf = estimate_end_to_end_performance(
                model=model,
                batch_size=batch_size,
                beam_size=beam_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=system_name,
                bits=bits,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
                expert_parallel=expert_parallel,
                debug=debug
            )
            
            # Add configuration info to results
            result = {
                'config_name': config_name,
                'performance': perf,
                'total_throughput': perf.get('total_throughput', 0),
                'ttft': perf.get('ttft', 0),
                'tpot': perf.get('average_tpot', 0),
                'total_latency': perf.get('total_latency', 0),
                'configuration': {
                    'tensor_parallel': tensor_parallel,
                    'pipeline_parallel': pipeline_parallel,
                    'expert_parallel': expert_parallel,
                    'bits': bits,
                    'beam_size': beam_size
                }
            }
            
            results.append(result)
            
        except Exception as e:
            # Add failed configuration to results with error info
            result = {
                'config_name': config_name,
                'error': str(e),
                'total_throughput': 0,
                'ttft': float('inf'),
                'tpot': float('inf'),
                'total_latency': float('inf'),
                'configuration': config
            }
            results.append(result)
    
    # Sort by total throughput (descending)
    results.sort(key=lambda x: x.get('total_throughput', 0), reverse=True)
    
    return results