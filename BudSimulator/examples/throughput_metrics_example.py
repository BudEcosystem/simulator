"""Example: Prefill and decode throughput metrics using the SimulationEngine.

This script demonstrates how to use the llm-memory-calculator's GenZ
SimulationEngine to estimate prefill and decode throughput for an LLM
on a given hardware configuration.

Usage:
    python examples/throughput_metrics_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    # Ensure llm-memory-calculator is importable
    calc_src = Path(__file__).resolve().parent.parent.parent / "llm-memory-calculator" / "src"
    if calc_src.is_dir() and str(calc_src) not in sys.path:
        sys.path.insert(0, str(calc_src))

    try:
        from llm_memory_calculator.genz.simulation.engine import SimulationEngine
        from llm_memory_calculator.genz.simulation.config import SimulationConfig
    except ImportError:
        print(
            "Error: llm-memory-calculator is not installed.\n"
            "Install it with: pip install -e ../llm-memory-calculator/"
        )
        sys.exit(1)

    engine = SimulationEngine()

    # --- Prefill throughput ---
    prefill_config = SimulationConfig(
        model="meta-llama/Llama-2-7b",
        features=["prefill"],
        simulation_params={
            "batch_size": 1,
            "seq_length": 2048,
            "precision": "bf16",
        },
        system_config={},
    )
    prefill_result = engine.simulate(prefill_config)

    print("=== Prefill Throughput ===")
    print(f"  Latency : {prefill_result.latency:.3f} s")
    print(f"  Throughput: {prefill_result.throughput:.1f} tokens/s")
    if hasattr(prefill_result, "breakdown"):
        print(f"  Breakdown : {json.dumps(prefill_result.breakdown, indent=2)}")
    print()

    # --- Decode throughput ---
    decode_config = SimulationConfig(
        model="meta-llama/Llama-2-7b",
        features=["decode"],
        simulation_params={
            "batch_size": 1,
            "output_tokens": 256,
            "precision": "bf16",
        },
        system_config={},
    )
    decode_result = engine.simulate(decode_config)

    print("=== Decode Throughput ===")
    print(f"  Latency : {decode_result.latency:.3f} s")
    print(f"  Throughput: {decode_result.throughput:.1f} tokens/s")
    if hasattr(decode_result, "breakdown"):
        print(f"  Breakdown : {json.dumps(decode_result.breakdown, indent=2)}")
    print()

    # --- Combined end-to-end ---
    combined_config = SimulationConfig(
        model="meta-llama/Llama-2-7b",
        features=["prefill", "decode"],
        simulation_params={
            "batch_size": 1,
            "seq_length": 2048,
            "output_tokens": 256,
            "precision": "bf16",
        },
        system_config={},
    )
    combined_result = engine.simulate(combined_config)

    total_tokens = 2048 + 256
    print("=== End-to-End (Prefill + Decode) ===")
    print(f"  Total latency : {combined_result.latency:.3f} s")
    print(f"  Overall throughput: {combined_result.throughput:.1f} tokens/s")
    print(f"  Total tokens     : {total_tokens}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
