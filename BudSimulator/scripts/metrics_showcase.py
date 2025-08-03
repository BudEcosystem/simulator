#!/usr/bin/env python3
"""
Comprehensive Metrics Showcase for GenZ Unified Simulation Interface

This script demonstrates how to extract and analyze key performance metrics:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- E2E Latency (End-to-End Latency)
- Prefill Throughput
- Decode Throughput
- Total Throughput

These metrics are crucial for understanding LLM inference performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_memory_calculator.genz.simulation.engine import SimulationEngine
from llm_memory_calculator.genz.simulation.config import SimulationConfig
from typing import Dict, Any
import json


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_metrics_analysis(name: str, result: Any, config: Dict[str, Any]):
    """Print detailed metrics analysis for a simulation result."""
    print(f"\n{name}:")
    print("-" * 60)
    
    # Basic metrics
    print(f"  Basic Metrics:")
    print(f"    Latency:    {result.latency:.3f} ms")
    print(f"    Throughput: {result.throughput:.3f} tokens/s")
    
    # Configuration info
    input_tokens = config.get("input_tokens", 0)
    output_tokens = config.get("output_tokens", 0)
    batch_size = config.get("batch_size", 1)
    
    print(f"  Configuration:")
    print(f"    Input Tokens:  {input_tokens}")
    print(f"    Output Tokens: {output_tokens}")
    print(f"    Batch Size:    {batch_size}")
    
    # Extract memory metrics from raw output
    memory_metrics = {}
    if hasattr(result, 'raw_output') and result.raw_output:
        memory_metrics = _extract_memory_from_result(result.raw_output)
    
    # Display memory metrics
    if memory_metrics:
        print(f"  Memory Metrics:")
        if "kv_cache_gb" in memory_metrics:
            print(f"    KV Cache:      {memory_metrics['kv_cache_gb']:.3f} GB")
        if "model_weights_gb" in memory_metrics:
            print(f"    Model Weights: {memory_metrics['model_weights_gb']:.3f} GB")
        if "total_memory_gb" in memory_metrics:
            print(f"    Total Memory:  {memory_metrics['total_memory_gb']:.3f} GB")
        if "peak_memory_gb" in memory_metrics:
            print(f"    Peak Memory:   {memory_metrics['peak_memory_gb']:.3f} GB")
        if "memory_efficiency" in memory_metrics:
            print(f"    Memory Efficiency: {memory_metrics['memory_efficiency']:.3f}")
    
    # Calculate and display advanced metrics
    if result.latency > 0:
        print(f"  Advanced Metrics:")
        
        if output_tokens > 0:  # Decode case
            tpot = result.latency / output_tokens
            ttft = tpot  # For decode, TTFT ≈ TPOT
            e2e_latency = result.latency
            decode_throughput = (output_tokens * batch_size) / (result.latency / 1000)
            
            print(f"    TTFT (Time to First Token): {ttft:.3f} ms")
            print(f"    TPOT (Time Per Output Token): {tpot:.3f} ms")
            print(f"    E2E Latency: {e2e_latency:.3f} ms")
            print(f"    Decode Throughput: {decode_throughput:.1f} tokens/s")
            
            if input_tokens > 0:
                total_tokens = (input_tokens + output_tokens) * batch_size
                total_throughput = total_tokens / (result.latency / 1000)
                print(f"    Total Throughput: {total_throughput:.1f} tokens/s")
                
                # Performance ratios
                print(f"  Performance Ratios:")
                print(f"    Tokens/ms: {total_tokens / result.latency:.2f}")
                print(f"    ms/Token: {result.latency / total_tokens:.4f}")
                
                # Memory efficiency ratios
                if "total_memory_gb" in memory_metrics and memory_metrics["total_memory_gb"] > 0:
                    tokens_per_gb = total_tokens / memory_metrics["total_memory_gb"]
                    print(f"    Tokens/GB: {tokens_per_gb:.1f}")
                
        else:  # Prefill case
            ttft = result.latency  # For prefill, TTFT is the total prefill time
            prefill_throughput = (input_tokens * batch_size) / (result.latency / 1000)
            
            print(f"    TTFT (Time to First Token): {ttft:.3f} ms")
            print(f"    Prefill Throughput: {prefill_throughput:.1f} tokens/s")
            
            # Performance ratios
            print(f"  Performance Ratios:")
            print(f"    Tokens/ms: {input_tokens / result.latency:.2f}")
            print(f"    ms/Token: {result.latency / input_tokens:.4f}")
            
            # Memory efficiency ratios
            if "total_memory_gb" in memory_metrics and memory_metrics["total_memory_gb"] > 0:
                tokens_per_gb = input_tokens / memory_metrics["total_memory_gb"]
                print(f"    Tokens/GB: {tokens_per_gb:.1f}")


def _extract_memory_from_result(raw_output: Any) -> Dict[str, float]:
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


def showcase_prefill_metrics():
    """Showcase prefill metrics with different configurations."""
    print_section("Prefill Metrics Analysis")
    
    engine = SimulationEngine()
    
    # Different prefill scenarios
    scenarios = [
        {
            "name": "Short Context Prefill",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["prefill"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 512,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Long Context Prefill",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["prefill"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 4096,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Batch Prefill",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["prefill"],
                "simulation_params": {
                    "batch_size": 8,
                    "input_tokens": 1024,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        }
    ]
    
    for scenario in scenarios:
        config = SimulationConfig(**scenario["config"])
        result = engine.simulate(config)
        print_metrics_analysis(scenario["name"], result, scenario["config"]["simulation_params"])


def showcase_decode_metrics():
    """Showcase decode metrics with different configurations."""
    print_section("Decode Metrics Analysis")
    
    engine = SimulationEngine()
    
    # Different decode scenarios
    scenarios = [
        {
            "name": "Short Generation",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 1024,
                    "output_tokens": 100,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Long Generation",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode"],
                "simulation_params": {
                    "batch_size": 1,
                    "input_tokens": 1024,
                    "output_tokens": 2048,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        },
        {
            "name": "Batch Decode",
            "config": {
                "model": "meta-llama/Llama-3.1-8B",
                "features": ["decode"],
                "simulation_params": {
                    "batch_size": 4,
                    "input_tokens": 512,
                    "output_tokens": 256,
                    "system_name": "A100_40GB_GPU",
                    "bits": "bf16"
                }
            }
        }
    ]
    
    for scenario in scenarios:
        config = SimulationConfig(**scenario["config"])
        result = engine.simulate(config)
        print_metrics_analysis(scenario["name"], result, scenario["config"]["simulation_params"])


def showcase_optimization_impact():
    """Showcase how different optimizations impact metrics."""
    print_section("Optimization Impact on Metrics")
    
    engine = SimulationEngine()
    
    # Base configuration
    base_params = {
        "model": "meta-llama/Llama-3.1-8B",
        "simulation_params": {
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 512,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    }
    
    # Different optimization scenarios
    optimizations = [
        {
            "name": "Baseline (No Optimizations)",
            "features": ["decode"]
        },
        {
            "name": "With Tensor Parallelism",
            "features": ["decode", "tensor_parallel"],
            "extra_params": {"tensor_parallel": 2}
        },
        {
            "name": "With LoRA",
            "features": ["decode", "lora"],
            "extra_params": {"lora_rank": 16}
        },
        {
            "name": "With Flash Attention",
            "features": ["decode", "flash_attention"]
        }
    ]
    
    baseline_result = None
    
    for opt in optimizations:
        # Merge parameters
        params = base_params["simulation_params"].copy()
        if "extra_params" in opt:
            params.update(opt["extra_params"])
        
        config = SimulationConfig(
            model=base_params["model"],
            features=opt["features"],
            simulation_params=params
        )
        
        try:
            result = engine.simulate(config)
            print_metrics_analysis(opt["name"], result, params)
            
            # Compare with baseline
            if baseline_result is None:
                baseline_result = result
            else:
                print(f"    Improvement vs Baseline:")
                latency_improvement = ((baseline_result.latency - result.latency) / baseline_result.latency) * 100
                throughput_improvement = ((result.throughput - baseline_result.throughput) / baseline_result.throughput) * 100
                print(f"      Latency: {latency_improvement:+.1f}%")
                print(f"      Throughput: {throughput_improvement:+.1f}%")
                
        except Exception as e:
            print(f"    Error with {opt['name']}: {e}")


def showcase_model_comparison():
    """Compare metrics across different model sizes."""
    print_section("Model Size Impact on Metrics")
    
    engine = SimulationEngine()
    
    # Different model sizes
    models = [
        {
            "name": "Small Model (Llama-3.1-8B)",
            "model": "meta-llama/Llama-3.1-8B"
        },
        {
            "name": "Medium Model (Llama-2-13B)",
            "model": "meta-llama/Llama-2-13B"
        }
    ]
    
    # Standard configuration
    standard_params = {
        "features": ["decode"],
        "simulation_params": {
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 256,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    }
    
    for model_info in models:
        config = SimulationConfig(
            model=model_info["model"],
            **standard_params
        )
        
        try:
            result = engine.simulate(config)
            print_metrics_analysis(model_info["name"], result, standard_params["simulation_params"])
        except Exception as e:
            print(f"    Error with {model_info['name']}: {e}")


def generate_metrics_report():
    """Generate a comprehensive metrics report."""
    print_section("Comprehensive Metrics Report")
    
    engine = SimulationEngine()
    
    # Test configuration
    config = SimulationConfig(
        model="meta-llama/Llama-3.1-8B",
        features=["decode"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 512,
            "system_name": "A100_40GB_GPU",
            "bits": "bf16"
        }
    )
    
    result = engine.simulate(config)
    
    # Generate comprehensive report
    report = {
        "model": config.model,
        "configuration": config.simulation_params,
        "basic_metrics": {
            "latency_ms": result.latency,
            "throughput_tokens_per_sec": result.throughput
        },
        "advanced_metrics": {},
        "performance_insights": {}
    }
    
    # Calculate advanced metrics
    input_tokens = config.simulation_params["input_tokens"]
    output_tokens = config.simulation_params["output_tokens"]
    batch_size = config.simulation_params["batch_size"]
    
    if output_tokens > 0:
        tpot = result.latency / output_tokens
        ttft = tpot
        e2e_latency = result.latency
        decode_throughput = (output_tokens * batch_size) / (result.latency / 1000)
        total_throughput = ((input_tokens + output_tokens) * batch_size) / (result.latency / 1000)
        
        report["advanced_metrics"] = {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_latency_ms": e2e_latency,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "total_throughput_tokens_per_sec": total_throughput
        }
        
        # Performance insights
        report["performance_insights"] = {
            "tokens_per_millisecond": (input_tokens + output_tokens) / result.latency,
            "milliseconds_per_token": result.latency / (input_tokens + output_tokens),
            "efficiency_score": total_throughput / 1000,  # Arbitrary efficiency metric
            "prefill_to_decode_ratio": input_tokens / output_tokens if output_tokens > 0 else float('inf')
        }
    
    # Print formatted report
    print("\nGenerated Metrics Report:")
    print(json.dumps(report, indent=2))
    
    return report


def main():
    """Main function to run all metric showcases."""
    print("GenZ Unified Simulation Interface - Comprehensive Metrics Showcase")
    print("="*80)
    print("This script demonstrates advanced performance metrics for LLM inference:")
    print("• TTFT (Time to First Token) - Time to generate the first token")
    print("• TPOT (Time Per Output Token) - Average time per generated token")
    print("• E2E Latency - End-to-end latency for complete inference")
    print("• Throughput Breakdowns - Separate prefill and decode throughputs")
    print("• Performance Comparisons - Impact of optimizations and model sizes")
    
    try:
        # Run all showcases
        showcase_prefill_metrics()
        showcase_decode_metrics()
        showcase_optimization_impact()
        showcase_model_comparison()
        generate_metrics_report()
        
        print_section("Summary")
        print("✅ All metric showcases completed successfully!")
        print("\nKey Takeaways:")
        print("• TTFT is crucial for interactive applications")
        print("• TPOT determines streaming generation speed")
        print("• E2E latency includes both prefill and decode phases")
        print("• Throughput breakdowns help identify bottlenecks")
        print("• Different optimizations have varying impacts on metrics")
        
    except Exception as e:
        print(f"❌ Error during showcase: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 