#!/usr/bin/env python3
"""
Unified Simulation Interface Examples

This script demonstrates how to use the new unified simulation interface
for GenZ with various feature combinations and simulation modes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZ.simulation import SimulationEngine, SimulationConfig, SimulationResult
from GenZ.features.registry import FeatureRegistry
import json
import time
from typing import Any, Dict


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(result: SimulationResult, title: str):
    """Print simulation result in a formatted way."""
    print(f"\n{title}:")
    print(f"  Latency: {result.latency:.4f} ms")
    print(f"  Throughput: {result.throughput:.2f} tokens/s")
    print(f"  Features: {list(result.feature_metrics.keys())}")
    
    # Extract and display memory metrics
    memory_metrics = {}
    if hasattr(result, 'raw_output') and result.raw_output:
        memory_metrics = _extract_memory_from_raw_output(result.raw_output)
    
    if memory_metrics:
        print("  Memory Usage:")
        if "kv_cache_gb" in memory_metrics:
            print(f"    KV Cache: {memory_metrics['kv_cache_gb']:.3f} GB")
        if "model_weights_gb" in memory_metrics:
            print(f"    Model Weights: {memory_metrics['model_weights_gb']:.3f} GB")
        if "total_memory_gb" in memory_metrics:
            print(f"    Total Memory: {memory_metrics['total_memory_gb']:.3f} GB")
        if "peak_memory_gb" in memory_metrics:
            print(f"    Peak Memory: {memory_metrics['peak_memory_gb']:.3f} GB")
        if "memory_efficiency" in memory_metrics:
            print(f"    Memory Efficiency: {memory_metrics['memory_efficiency']:.3f}")
    
    if result.runtime_breakdown:
        print("  Runtime Breakdown:")
        # Handle both dict and RuntimeBreakdown objects
        if hasattr(result.runtime_breakdown, 'to_dict'):
            breakdown_dict = result.runtime_breakdown.to_dict()
        elif hasattr(result.runtime_breakdown, '__dict__'):
            breakdown_dict = result.runtime_breakdown.__dict__
        elif isinstance(result.runtime_breakdown, dict):
            breakdown_dict = result.runtime_breakdown
        else:
            breakdown_dict = {"total": result.runtime_breakdown}
        
        for component, time_ms in breakdown_dict.items():
            if isinstance(time_ms, (int, float)) and time_ms > 0:
                print(f"    {component}: {time_ms:.2f} ms")


def _extract_memory_from_raw_output(raw_output: Any) -> Dict[str, float]:
    """Extract memory metrics from raw simulation output."""
    memory_metrics = {}
    
    # Handle different result formats
    data = {}
    if hasattr(raw_output, '__dict__'):
        data = raw_output.__dict__
    elif isinstance(raw_output, dict):
        data = raw_output
    else:
        return memory_metrics
    
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
        memory_metrics["kv_cache_gb"] = kv_cache_gb
    
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
        memory_metrics["model_weights_gb"] = model_weights_gb
    
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
        memory_metrics["total_memory_gb"] = total_memory_gb
    
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
        memory_metrics["peak_memory_gb"] = peak_memory_gb
    
    # Memory efficiency metrics
    if "model_weights_gb" in memory_metrics and "total_memory_gb" in memory_metrics:
        if memory_metrics["total_memory_gb"] > 0:
            memory_metrics["memory_efficiency"] = memory_metrics["model_weights_gb"] / memory_metrics["total_memory_gb"]
    
    return memory_metrics


def example_1_basic_simulations():
    """Example 1: Basic simulation modes (prefill, decode, chunked)."""
    print_section("Example 1: Basic Simulation Modes")
    
    engine = SimulationEngine()
    model = "meta-llama/Llama-3.1-8B"
    
    # 1. Prefill simulation
    prefill_config = SimulationConfig(
        model=model,
        features=["prefill"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 2048,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    prefill_result = engine.simulate(prefill_config)
    print_result(prefill_result, "Prefill Simulation")
    
    # 2. Decode simulation
    decode_config = SimulationConfig(
        model=model,
        features=["decode"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 2048,
            "output_tokens": 512,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    decode_result = engine.simulate(decode_config)
    print_result(decode_result, "Decode Simulation")
    
    # 3. Chunked simulation
    chunked_config = SimulationConfig(
        model=model,
        features=["chunked"],
        simulation_params={
            "prefill_kv_sizes": [(2048, 1024)],  # (total_tokens, prefill_tokens)
            "decode_kv_sizes": [3072, 3073, 3074, 3075],  # decode steps
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    chunked_result = engine.simulate(chunked_config)
    print_result(chunked_result, "Chunked Simulation")


def example_2_parallelism_features():
    """Example 2: Parallelism features (tensor parallel, pipeline parallel)."""
    print_section("Example 2: Parallelism Features")
    
    engine = SimulationEngine()
    model = "meta-llama/Llama-3.1-8B"  # Use 8B model to avoid memory issues
    
    # 1. Tensor parallelism
    tp_config = SimulationConfig(
        model=model,
        features=["decode", "tensor_parallel"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 256,
            "tensor_parallel": 4,  # Reduced from 8 to 4
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    tp_result = engine.simulate(tp_config)
    print_result(tp_result, "Tensor Parallel (4-way)")
    
    # 2. Pipeline parallelism
    pp_config = SimulationConfig(
        model=model,
        features=["decode", "pipeline_parallel"],
        simulation_params={
            "batch_size": 4,  # Larger batch for pipeline efficiency
            "input_tokens": 1024,
            "output_tokens": 256,
            "pipeline_parallel": 2,  # Reduced from 4 to 2
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    pp_result = engine.simulate(pp_config)
    print_result(pp_result, "Pipeline Parallel (2-stage)")


def example_3_optimization_features():
    """Example 3: Optimization features (LoRA, Flash Attention)."""
    print_section("Example 3: Optimization Features")
    
    engine = SimulationEngine()
    model = "meta-llama/Llama-3.1-8B"
    
    # 1. LoRA optimization
    lora_config = SimulationConfig(
        model=model,
        features=["prefill", "lora"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 2048,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "lora": {
                "enabled": True,
                "rank": 16,
                "strategy": "dynamic",
                "target_modules": ["attn", "ffn"]
            }
        }
    )
    
    lora_result = engine.simulate(lora_config)
    print_result(lora_result, "LoRA Optimization (rank=16)")
    
    # 2. Flash Attention
    flash_config = SimulationConfig(
        model=model,
        features=["decode", "flash_attention"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 4096,  # Longer sequence for flash attention benefits
            "output_tokens": 512,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    flash_result = engine.simulate(flash_config)
    print_result(flash_result, "Flash Attention")


def example_4_chunked_with_features():
    """Example 4: Chunked inference with various features."""
    print_section("Example 4: Chunked Inference with Features")
    
    engine = SimulationEngine()
    model = "meta-llama/Llama-3.1-8B"  # Use 8B model to avoid memory issues
    
    # 1. Chunked with LoRA
    chunked_lora_config = SimulationConfig(
        model=model,
        features=["chunked", "lora"],
        simulation_params={
            "prefill_kv_sizes": [(2048, 1024)],  # Reduced size
            "decode_kv_sizes": [3072, 3073, 3074],  # Fewer decode steps
            "system_name": "A100_40GB_GPU",
            "bits": "bf16",
            "lora": {
                "enabled": True,
                "rank": 32,  # Reduced from 64 to 32
                "strategy": "dynamic"
            }
        }
    )
    
    chunked_lora_result = engine.simulate(chunked_lora_config)
    print_result(chunked_lora_result, "Chunked + LoRA")


def example_5_memory_analysis():
    """Example 5: Memory analysis across different configurations."""
    print_section("Example 5: Memory Analysis")
    
    engine = SimulationEngine()
    
    # Memory analysis scenarios
    scenarios = [
        {
            "name": "Small Model Memory Usage",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 1024,
                    "output_tokens": 256,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Large Context Memory Usage",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["prefill"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 8192,  # Large context
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Batch Processing Memory",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode"],
                "simulation_params": {
                    "batch_size": 8,  # Large batch
                    "input_tokens": 1024,
                    "output_tokens": 512,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Tensor Parallel Memory",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode", "tensor_parallel"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 1024,
                    "output_tokens": 256,
                    "tensor_parallel": 2,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        }
    ]
    
    print("Analyzing memory usage patterns across different configurations...")
    
    for scenario in scenarios:
        try:
            config = SimulationConfig(**scenario["config"])
            result = engine.simulate(config)
            print_result(result, scenario["name"])
        except Exception as e:
            print(f"\nError in {scenario['name']}: {e}")


def example_6_feature_discovery():
    """Example 6: Feature discovery and validation."""
    print_section("Example 6: Feature Discovery and Validation")
    
    registry = FeatureRegistry()
    
    # List all available features
    features = registry.get_available_features()
    print(f"Available features ({len(features)}):")
    for feature in sorted(features):
        metadata = registry.get_feature_metadata(feature)
        is_builtin = registry.is_builtin_feature(feature)
        print(f"  {feature:<20} - {metadata.description} {'(builtin)' if is_builtin else ''}")
    
    # Get features by category
    from GenZ.features.base import FeatureCategory
    
    for category in FeatureCategory:
        category_features = registry.get_features_by_category(category)
        if category_features:
            print(f"\n{category.value.title()} features:")
            for feature in category_features:
                print(f"  - {feature}")


def main():
    """Run all examples."""
    print("GenZ Unified Simulation Interface Examples")
    print("==========================================")
    
    examples = [
        example_1_basic_simulations,
        example_2_parallelism_features,
        example_3_optimization_features,
        example_4_chunked_with_features,
        example_5_memory_analysis,
        example_6_feature_discovery
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nError in Example {i}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    print_section("Examples Complete")
    print("All examples completed successfully!")
    print("\nFor more information, see:")
    print("- GenZ/simulation/ - Core simulation interface")
    print("- GenZ/features/ - Feature system")
    print("- tests/test_unified_simulation_interface.py - Test examples")


if __name__ == "__main__":
    main()
