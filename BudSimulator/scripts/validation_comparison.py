#!/usr/bin/env python3
"""
Validation Comparison Script

This script compares the results between the original GenZ functions 
(prefill_moddeling, decode_moddeling, chunked_moddeling) and the new 
unified simulation interface to ensure they produce identical results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZ.simulation import SimulationEngine, SimulationConfig, SimulationResult
from GenZ.LLM_inference import prefill_moddeling, decode_moddeling, chunked_moddeling
import numpy as np
import json
from typing import Dict, Any, Tuple


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def extract_metrics(result: Any) -> Dict[str, float]:
    """Extract key metrics from any result format."""
    metrics = {}
    
    if isinstance(result, SimulationResult):
        metrics["latency"] = float(result.latency)
        metrics["throughput"] = float(result.throughput)
        
        # Extract additional metrics from raw output if available
        if result.raw_output:
            raw = result.raw_output
            metrics.update(_extract_additional_metrics(raw))
            
    elif isinstance(result, dict):
        metrics["latency"] = float(result.get("Latency", result.get("latency", 0)))
        metrics["throughput"] = float(result.get("Throughput", result.get("throughput", 0)))
        metrics.update(_extract_additional_metrics(result))
        
    else:
        # Handle ModdelingOutput or other objects
        metrics["latency"] = float(getattr(result, "Latency", getattr(result, "latency", 0)))
        metrics["throughput"] = float(getattr(result, "Throughput", getattr(result, "throughput", 0)))
        
        # Try to get additional metrics from the object
        if hasattr(result, '__dict__'):
            metrics.update(_extract_additional_metrics(result.__dict__))
    
    return metrics


def _extract_additional_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract additional performance metrics from result data."""
    additional_metrics = {}
    
    # Extract TTFT (Time to First Token) - usually the prefill latency
    ttft = data.get("TTFT", data.get("ttft", data.get("prefill_latency", 0)))
    if ttft:
        additional_metrics["ttft"] = float(ttft)
    
    # Extract TPOT (Time Per Output Token) - decode latency per token
    tpot = data.get("TPOT", data.get("tpot", data.get("decode_latency_per_token", 0)))
    if tpot:
        additional_metrics["tpot"] = float(tpot)
    
    # Extract E2E latency if available
    e2e = data.get("E2E", data.get("e2e_latency", data.get("end_to_end_latency", 0)))
    if e2e:
        additional_metrics["e2e_latency"] = float(e2e)
    
    # Extract separate throughputs
    prefill_throughput = data.get("prefill_throughput", data.get("Prefill_Throughput", 0))
    if prefill_throughput:
        additional_metrics["prefill_throughput"] = float(prefill_throughput)
    
    decode_throughput = data.get("decode_throughput", data.get("Decode_Throughput", 0))
    if decode_throughput:
        additional_metrics["decode_throughput"] = float(decode_throughput)
    
    # Extract tokens per second metrics
    tokens_per_sec = data.get("Throughput_tokens_per_sec", data.get("tokens_per_second", 0))
    if tokens_per_sec:
        additional_metrics["tokens_per_sec"] = float(tokens_per_sec)
    
    # Extract memory metrics
    _extract_memory_metrics(data, additional_metrics)
    
    # Calculate derived metrics if we have the raw data
    _calculate_derived_metrics(data, additional_metrics)
    
    return additional_metrics


def _extract_memory_metrics(data: Dict[str, Any], metrics: Dict[str, float]):
    """Extract memory-related metrics from result data."""
    
    # KV Cache size
    kv_cache_size = data.get("KV_cache_size", data.get("kv_cache_size", 
                            data.get("KV_Cache_Size", data.get("kv_cache_memory", 0))))
    if kv_cache_size:
        # Convert to GB if needed (assume input is in bytes or MB)
        kv_cache_gb = float(kv_cache_size)
        if kv_cache_gb > 1000:  # Likely in MB
            kv_cache_gb = kv_cache_gb / 1024  # Convert MB to GB
        if kv_cache_gb > 1000000:  # Likely in bytes
            kv_cache_gb = kv_cache_gb / (1024 * 1024 * 1024)  # Convert bytes to GB
        metrics["kv_cache_gb"] = kv_cache_gb
    
    # Model weights size
    model_weights = data.get("Model_weights", data.get("model_weights", 
                            data.get("Model_Size", data.get("model_size", 0))))
    if model_weights:
        # Convert to GB if needed
        model_weights_gb = float(model_weights)
        if model_weights_gb > 1000:  # Likely in MB
            model_weights_gb = model_weights_gb / 1024  # Convert MB to GB
        if model_weights_gb > 1000000:  # Likely in bytes
            model_weights_gb = model_weights_gb / (1024 * 1024 * 1024)  # Convert bytes to GB
        metrics["model_weights_gb"] = model_weights_gb
    
    # Total memory usage
    total_memory = data.get("Total_memory", data.get("total_memory", 
                           data.get("Memory_usage", data.get("memory_usage", 0))))
    if total_memory:
        # Convert to GB if needed
        total_memory_gb = float(total_memory)
        if total_memory_gb > 1000:  # Likely in MB
            total_memory_gb = total_memory_gb / 1024  # Convert MB to GB
        if total_memory_gb > 1000000:  # Likely in bytes
            total_memory_gb = total_memory_gb / (1024 * 1024 * 1024)  # Convert bytes to GB
        metrics["total_memory_gb"] = total_memory_gb
    
    # Peak memory usage
    peak_memory = data.get("Peak_memory", data.get("peak_memory", 
                          data.get("Max_memory", data.get("max_memory", 0))))
    if peak_memory:
        # Convert to GB if needed
        peak_memory_gb = float(peak_memory)
        if peak_memory_gb > 1000:  # Likely in MB
            peak_memory_gb = peak_memory_gb / 1024  # Convert MB to GB
        if peak_memory_gb > 1000000:  # Likely in bytes
            peak_memory_gb = peak_memory_gb / (1024 * 1024 * 1024)  # Convert bytes to GB
        metrics["peak_memory_gb"] = peak_memory_gb
    
    # Memory efficiency metrics
    if "model_weights_gb" in metrics and "total_memory_gb" in metrics:
        if metrics["total_memory_gb"] > 0:
            metrics["memory_efficiency"] = metrics["model_weights_gb"] / metrics["total_memory_gb"]
    
    # Try to extract from system info if available
    system_info = data.get("System_info", data.get("system_info", {}))
    if isinstance(system_info, dict):
        _extract_system_memory_metrics(system_info, metrics)


def _extract_system_memory_metrics(system_info: Dict[str, Any], metrics: Dict[str, float]):
    """Extract memory metrics from system information."""
    
    # GPU memory info
    gpu_memory = system_info.get("gpu_memory", system_info.get("GPU_memory", 0))
    if gpu_memory:
        metrics["gpu_memory_gb"] = float(gpu_memory) / 1024 if gpu_memory > 1000 else float(gpu_memory)
    
    # Available memory
    available_memory = system_info.get("available_memory", system_info.get("Available_memory", 0))
    if available_memory:
        metrics["available_memory_gb"] = float(available_memory) / 1024 if available_memory > 1000 else float(available_memory)
    
    # Memory utilization
    if "total_memory_gb" in metrics and "available_memory_gb" in metrics:
        if metrics["available_memory_gb"] > 0:
            metrics["memory_utilization"] = metrics["total_memory_gb"] / metrics["available_memory_gb"]


def _calculate_derived_metrics(data: Dict[str, Any], metrics: Dict[str, float]):
    """Calculate derived metrics like TTFT, TPOT, E2E latency."""
    
    # Get input/output token counts
    input_tokens = data.get("input_tokens", data.get("Input_tokens", 0))
    output_tokens = data.get("output_tokens", data.get("Output_tokens", 0))
    batch_size = data.get("batch_size", data.get("Batch_size", 1))
    
    # Get latency components from runtime breakdown
    runtime_breakdown = data.get("Runtime_breakdown", data.get("runtime_breakdown", {}))
    
    if runtime_breakdown:
        # Handle RuntimeBreakdown object
        if hasattr(runtime_breakdown, 'to_dict'):
            breakdown_dict = runtime_breakdown.to_dict()
        elif hasattr(runtime_breakdown, '__dict__'):
            breakdown_dict = runtime_breakdown.__dict__
        elif isinstance(runtime_breakdown, dict):
            breakdown_dict = runtime_breakdown
        else:
            breakdown_dict = {}
        
        # Calculate TTFT (Time to First Token) - typically prefill time
        if "ttft" not in metrics:
            # TTFT is usually the time for prefill + first decode step
            embedding_time = breakdown_dict.get("Embedding", 0)
            mha_time = breakdown_dict.get("MHA", 0) 
            ffn_time = breakdown_dict.get("FFN", 0)
            
            # For prefill, TTFT is the total prefill latency
            if input_tokens and not output_tokens:  # Prefill only
                metrics["ttft"] = metrics.get("latency", 0)
            elif output_tokens:  # Decode case
                # TTFT is approximately the per-token decode time
                total_latency = metrics.get("latency", 0)
                if output_tokens > 0:
                    metrics["tpot"] = total_latency / output_tokens
                    metrics["ttft"] = metrics["tpot"]  # First token time
    
    # Calculate E2E latency if we have both prefill and decode info
    if "e2e_latency" not in metrics:
        # For decode simulations, the latency already includes E2E
        if output_tokens and input_tokens:
            metrics["e2e_latency"] = metrics.get("latency", 0)
    
    # Calculate throughput metrics
    total_latency = metrics.get("latency", 0)
    if total_latency > 0:
        # Prefill throughput (tokens processed / time)
        if input_tokens and not output_tokens:  # Prefill only
            metrics["prefill_throughput"] = (input_tokens * batch_size) / (total_latency / 1000)  # tokens/sec
        
        # Decode throughput (tokens generated / time)  
        if output_tokens:  # Decode case
            metrics["decode_throughput"] = (output_tokens * batch_size) / (total_latency / 1000)  # tokens/sec
            
            # Total tokens throughput
            total_tokens = (input_tokens + output_tokens) * batch_size
            metrics["total_throughput"] = total_tokens / (total_latency / 1000)


def compare_results(original: Dict[str, float], unified: Dict[str, float], tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare two result dictionaries and return comparison details."""
    comparison = {
        "match": True,
        "differences": {},
        "relative_errors": {},
        "max_error": 0.0
    }
    
    # Primary metrics that must match
    primary_metrics = ["latency", "throughput"]
    
    # Additional metrics to compare if available
    additional_metrics = [
        "ttft", "tpot", "e2e_latency", "prefill_throughput", "decode_throughput", "tokens_per_sec",
        "kv_cache_gb", "model_weights_gb", "total_memory_gb", "peak_memory_gb", 
        "memory_efficiency", "gpu_memory_gb", "available_memory_gb", "memory_utilization"
    ]
    
    all_metrics = primary_metrics + additional_metrics
    
    for metric in all_metrics:
        if metric in original and metric in unified:
            orig_val = original[metric]
            unif_val = unified[metric]
            
            if orig_val == 0 and unif_val == 0:
                relative_error = 0.0
            elif orig_val == 0:
                relative_error = float('inf') if unif_val != 0 else 0.0
            else:
                relative_error = abs(orig_val - unif_val) / abs(orig_val)
            
            comparison["differences"][metric] = unif_val - orig_val
            comparison["relative_errors"][metric] = relative_error
            comparison["max_error"] = max(comparison["max_error"], relative_error)
            
            # Only fail on primary metrics
            if relative_error > tolerance and metric in primary_metrics:
                comparison["match"] = False
    
    return comparison


def print_comparison(test_name: str, original: Dict[str, float], unified: Dict[str, float], comparison: Dict[str, Any]):
    """Print detailed comparison results."""
    print(f"\n{test_name}:")
    print(f"  Original Results:")
    print(f"    Latency:    {original['latency']:.6f} ms")
    print(f"    Throughput: {original['throughput']:.6f} tokens/s")
    
    # Print additional metrics if available
    additional_orig = {k: v for k, v in original.items() if k not in ['latency', 'throughput']}
    if additional_orig:
        for metric, value in additional_orig.items():
            metric_name = metric.replace('_', ' ').title()
            unit = _get_metric_unit(metric)
            print(f"    {metric_name:<12}: {value:.6f} {unit}")
    
    print(f"  Unified Results:")
    print(f"    Latency:    {unified['latency']:.6f} ms")
    print(f"    Throughput: {unified['throughput']:.6f} tokens/s")
    
    # Print additional metrics if available
    additional_unif = {k: v for k, v in unified.items() if k not in ['latency', 'throughput']}
    if additional_unif:
        for metric, value in additional_unif.items():
            metric_name = metric.replace('_', ' ').title()
            unit = _get_metric_unit(metric)
            print(f"    {metric_name:<12}: {value:.6f} {unit}")
    
    print(f"  Comparison:")
    print(f"    Match: {'✓' if comparison['match'] else '✗'}")
    print(f"    Max Relative Error: {comparison['max_error']:.2e}")
    
    if not comparison['match'] or comparison['max_error'] > 1e-9:
        print(f"    Differences:")
        for metric, diff in comparison['differences'].items():
            rel_err = comparison['relative_errors'][metric]
            unit = _get_metric_unit(metric)
            print(f"      {metric}: {diff:.6f} {unit} (rel: {rel_err:.2e})")


def _get_metric_unit(metric: str) -> str:
    """Get the appropriate unit for a metric."""
    if 'latency' in metric or 'ttft' in metric or 'tpot' in metric:
        return 'ms'
    elif 'throughput' in metric or 'tokens_per_sec' in metric:
        return 'tokens/s'
    elif 'memory_gb' in metric or 'cache_gb' in metric or 'weights_gb' in metric:
        return 'GB'
    elif 'efficiency' in metric or 'utilization' in metric:
        return ''  # Ratios are unitless
    else:
        return ''


def test_prefill_comparison():
    """Test prefill modeling comparison."""
    print_section("Prefill Modeling Comparison")
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Prefill",
            "model": "meta-llama/Llama-3.1-8B",
            "batch_size": 1,
            "input_tokens": 1024,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1
        },
        {
            "name": "Large Batch Prefill",
            "model": "meta-llama/Llama-3.1-8B",
            "batch_size": 4,
            "input_tokens": 2048,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1
        }
    ]
    
    engine = SimulationEngine()
    all_match = True
    
    for config in test_configs:
        # Original function
        original_result = prefill_moddeling(
            model=config["model"],
            batch_size=config["batch_size"],
            input_tokens=config["input_tokens"],
            system_name=config["system_name"],
            bits=config["bits"],
            tensor_parallel=config["tensor_parallel"],
            pipeline_parallel=config["pipeline_parallel"],
            expert_parallel=config["expert_parallel"],
            debug=False
        )
        
        # Unified interface
        unified_config = SimulationConfig(
            model=config["model"],
            features=["prefill"],
            simulation_params={
                "batch_size": config["batch_size"],
                "input_tokens": config["input_tokens"],
                "system_name": config["system_name"],
                "bits": config["bits"],
                "tensor_parallel": config["tensor_parallel"],
                "pipeline_parallel": config["pipeline_parallel"],
                "expert_parallel": config["expert_parallel"]
            }
        )
        
        unified_result = engine.simulate(unified_config)
        
        # Add configuration info to results for metric calculation
        config_info = {
            "input_tokens": config["input_tokens"],
            "batch_size": config["batch_size"],
            "output_tokens": 0  # Prefill has no output tokens
        }
        
        # Compare results
        original_metrics = extract_metrics(original_result)
        unified_metrics = extract_metrics(unified_result)
        
        # Add config info for derived metric calculations
        original_metrics.update(_calculate_metrics_from_config(original_metrics, config_info))
        unified_metrics.update(_calculate_metrics_from_config(unified_metrics, config_info))
        
        comparison = compare_results(original_metrics, unified_metrics)
        
        print_comparison(config["name"], original_metrics, unified_metrics, comparison)
        
        if not comparison["match"]:
            all_match = False
    
    return all_match


def test_decode_comparison():
    """Test decode modeling comparison."""
    print_section("Decode Modeling Comparison")
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Decode",
            "model": "meta-llama/Llama-3.1-8B",
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 256,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1
        },
        {
            "name": "Large Output Decode",
            "model": "meta-llama/Llama-3.1-8B",
            "batch_size": 1,
            "input_tokens": 512,
            "output_tokens": 1024,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1
        }
    ]
    
    engine = SimulationEngine()
    all_match = True
    
    for config in test_configs:
        # Original function
        original_result = decode_moddeling(
            model=config["model"],
            batch_size=config["batch_size"],
            input_tokens=config["input_tokens"],
            output_tokens=config["output_tokens"],
            system_name=config["system_name"],
            bits=config["bits"],
            tensor_parallel=config["tensor_parallel"],
            pipeline_parallel=config["pipeline_parallel"],
            expert_parallel=config["expert_parallel"],
            debug=False
        )
        
        # Unified interface
        unified_config = SimulationConfig(
            model=config["model"],
            features=["decode"],
            simulation_params={
                "batch_size": config["batch_size"],
                "input_tokens": config["input_tokens"],
                "output_tokens": config["output_tokens"],
                "system_name": config["system_name"],
                "bits": config["bits"],
                "tensor_parallel": config["tensor_parallel"],
                "pipeline_parallel": config["pipeline_parallel"],
                "expert_parallel": config["expert_parallel"]
            }
        )
        
        unified_result = engine.simulate(unified_config)
        
        # Add configuration info to results for metric calculation
        config_info = {
            "input_tokens": config["input_tokens"],
            "output_tokens": config["output_tokens"],
            "batch_size": config["batch_size"]
        }
        
        # Compare results
        original_metrics = extract_metrics(original_result)
        unified_metrics = extract_metrics(unified_result)
        
        # Add config info for derived metric calculations
        original_metrics.update(_calculate_metrics_from_config(original_metrics, config_info))
        unified_metrics.update(_calculate_metrics_from_config(unified_metrics, config_info))
        
        comparison = compare_results(original_metrics, unified_metrics)
        
        print_comparison(config["name"], original_metrics, unified_metrics, comparison)
        
        if not comparison["match"]:
            all_match = False
    
    return all_match


def test_chunked_comparison():
    """Test chunked modeling comparison."""
    print_section("Chunked Modeling Comparison")
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Chunked",
            "model": "meta-llama/Llama-3.1-8B",
            "prefill_kv_sizes": [(2048, 1024)],
            "decode_kv_sizes": [3072, 3073, 3074],
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1
        }
    ]
    
    engine = SimulationEngine()
    all_match = True
    
    for config in test_configs:
        # Original function
        original_result = chunked_moddeling(
            model=config["model"],
            prefill_kv_sizes=config["prefill_kv_sizes"],
            decode_kv_sizes=config["decode_kv_sizes"],
            system_name=config["system_name"],
            bits=config["bits"],
            tensor_parallel=config["tensor_parallel"],
            pipeline_parallel=config["pipeline_parallel"],
            expert_parallel=config["expert_parallel"],
            debug=False
        )
        
        # Unified interface
        unified_config = SimulationConfig(
            model=config["model"],
            features=["chunked"],
            simulation_params={
                "prefill_kv_sizes": config["prefill_kv_sizes"],
                "decode_kv_sizes": config["decode_kv_sizes"],
                "system_name": config["system_name"],
                "bits": config["bits"],
                "tensor_parallel": config["tensor_parallel"],
                "pipeline_parallel": config["pipeline_parallel"],
                "expert_parallel": config["expert_parallel"]
            }
        )
        
        unified_result = engine.simulate(unified_config)
        
        # Calculate token counts from chunked configuration
        total_prefill_tokens = sum(batch_size * seq_len for batch_size, seq_len in config["prefill_kv_sizes"])
        total_decode_tokens = len(config["decode_kv_sizes"])
        
        config_info = {
            "input_tokens": total_prefill_tokens,
            "output_tokens": total_decode_tokens,
            "batch_size": 1,  # Chunked typically processes one chunk at a time
            "is_chunked": True
        }
        
        # Compare results
        original_metrics = extract_metrics(original_result)
        unified_metrics = extract_metrics(unified_result)
        
        # Add config info for derived metric calculations
        original_metrics.update(_calculate_chunked_metrics(original_metrics, config_info))
        unified_metrics.update(_calculate_chunked_metrics(unified_metrics, config_info))
        
        comparison = compare_results(original_metrics, unified_metrics)
        
        print_comparison(config["name"], original_metrics, unified_metrics, comparison)
        
        if not comparison["match"]:
            all_match = False
    
    return all_match


def _calculate_chunked_metrics(metrics: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate additional metrics for chunked simulations."""
    derived = {}
    
    input_tokens = config.get("input_tokens", 0)
    output_tokens = config.get("output_tokens", 0)
    batch_size = config.get("batch_size", 1)
    latency = metrics.get("latency", 0)
    
    if latency > 0:
        # For chunked simulations, metrics are more complex
        # TTFT includes prefill time + first decode chunk
        if output_tokens > 0:
            # Estimate TPOT from total decode time
            derived["tpot"] = latency / max(output_tokens, 1)
            derived["ttft"] = derived["tpot"]  # Approximation
            derived["e2e_latency"] = latency
            
            # Throughput calculations
            if input_tokens > 0:
                # Combined prefill + decode throughput
                total_tokens = input_tokens + output_tokens
                derived["total_throughput"] = total_tokens / (latency / 1000)
                
                # Separate estimates (rough approximation)
                prefill_ratio = input_tokens / total_tokens
                decode_ratio = output_tokens / total_tokens
                
                derived["prefill_throughput"] = input_tokens / ((latency * prefill_ratio) / 1000)
                derived["decode_throughput"] = output_tokens / ((latency * decode_ratio) / 1000)
        else:
            # Pure prefill chunked
            derived["ttft"] = latency
            derived["prefill_throughput"] = input_tokens / (latency / 1000)
    
    return derived


def _calculate_metrics_from_config(metrics: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate additional metrics from configuration and existing metrics."""
    derived = {}
    
    input_tokens = config.get("input_tokens", 0)
    output_tokens = config.get("output_tokens", 0)
    batch_size = config.get("batch_size", 1)
    latency = metrics.get("latency", 0)
    
    if latency > 0:
        # Calculate TTFT (Time to First Token)
        if output_tokens > 0:  # Decode case
            # TPOT (Time Per Output Token)
            derived["tpot"] = latency / output_tokens
            # TTFT is approximately the same as TPOT for decode
            derived["ttft"] = derived["tpot"]
            # E2E latency for decode is the total latency
            derived["e2e_latency"] = latency
            # Decode throughput
            derived["decode_throughput"] = (output_tokens * batch_size) / (latency / 1000)
        else:  # Prefill case
            # TTFT for prefill is the total prefill time
            derived["ttft"] = latency
            # Prefill throughput
            derived["prefill_throughput"] = (input_tokens * batch_size) / (latency / 1000)
        
        # Total throughput calculation
        if input_tokens and output_tokens:
            total_tokens = (input_tokens + output_tokens) * batch_size
            derived["total_throughput"] = total_tokens / (latency / 1000)
    
    return derived


def run_comprehensive_validation():
    """Run comprehensive validation suite."""
    print("GenZ Unified Simulation Interface Validation")
    print("=" * 80)
    print("Comparing original functions vs unified interface...")
    
    results = {
        "prefill": test_prefill_comparison(),
        "decode": test_decode_comparison(),
        "chunked": test_chunked_comparison()
    }
    
    print_section("Validation Summary")
    
    all_passed = True
    for test_type, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_type.capitalize():<15}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 The unified simulation interface produces identical results to the original functions!")
        print("   The implementation is validated and ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please review the differences above.")
        print("   The unified interface may need adjustments to match original behavior.")
    
    return all_passed


def main():
    """Main function to run validation."""
    try:
        success = run_comprehensive_validation()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
