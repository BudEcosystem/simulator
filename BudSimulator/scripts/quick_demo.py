#!/usr/bin/env python3
"""
Quick Demo of Unified Simulation Interface

A concise demonstration of the new unified simulation interface capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZ.simulation import SimulationEngine, SimulationConfig
from GenZ.features.registry import FeatureRegistry


def main():
    """Quick demo of unified interface."""
    print("🚀 GenZ Unified Simulation Interface - Quick Demo")
    print("=" * 60)
    
    engine = SimulationEngine()
    
    # 1. Basic Prefill
    print("\n1. Basic Prefill Simulation:")
    config = SimulationConfig(
        model="meta-llama/Llama-3.1-8B",
        features=["prefill"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 1024,
            "system_name": "A100_40GB_GPU"
        }
    )
    result = engine.simulate(config)
    print(f"   Latency: {result.latency:.2f} ms | Throughput: {result.throughput:.2f} tokens/s")
    
    # 2. Decode with LoRA
    print("\n2. Decode with LoRA:")
    config = SimulationConfig(
        model="meta-llama/Llama-3.1-8B",
        features=["decode", "lora"],
        simulation_params={
            "batch_size": 1,
            "input_tokens": 1024,
            "output_tokens": 256,
            "system_name": "A100_40GB_GPU",
            "lora": {"rank": 16}
        }
    )
    result = engine.simulate(config)
    print(f"   Latency: {result.latency:.2f} ms | Throughput: {result.throughput:.2f} tokens/s")
    print(f"   Features: {list(result.feature_metrics.keys())}")
    
    # 3. Chunked with Tensor Parallelism
    print("\n3. Chunked with Tensor Parallelism:")
    config = SimulationConfig(
        model="meta-llama/Llama-3.1-8B",
        features=["chunked", "tensor_parallel"],
        simulation_params={
            "prefill_kv_sizes": [(1024, 512)],
            "decode_kv_sizes": [1536, 1537, 1538],
            "tensor_parallel": 2,
            "system_name": "A100_40GB_GPU"
        }
    )
    result = engine.simulate(config)
    print(f"   Latency: {result.latency:.2f} ms | Throughput: {result.throughput:.2f} tokens/s")
    print(f"   Features: {list(result.feature_metrics.keys())}")
    
    # 4. Available Features
    print("\n4. Available Features:")
    registry = FeatureRegistry()
    features = registry.get_available_features()
    print(f"   Total: {len(features)} features")
    print(f"   Examples: {', '.join(features[:8])}...")
    
    print("\n✅ Demo Complete! The unified interface provides:")
    print("   • Consistent API for all simulation types")
    print("   • Easy feature combination")
    print("   • Identical results to original functions")
    print("   • Extensible architecture for new features")


if __name__ == "__main__":
    main() 