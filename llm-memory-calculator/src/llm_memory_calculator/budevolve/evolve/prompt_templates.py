"""Roofline-enriched prompt templates for OpenEvolve."""
from typing import Dict, Optional

from ..types import RooflineReport
from ..roofline_analyzer import RooflineAnalyzer


def build_scheduler_prompt(
    current_code: str,
    metrics: Dict[str, float],
    roofline: Optional[RooflineReport] = None,
) -> str:
    """Build a prompt for evolving a batch scheduling algorithm.

    Includes current code, performance metrics, and roofline bottleneck
    analysis so the LLM can make informed mutations.
    """
    lines = [
        "You are evolving a batch scheduling algorithm for LLM serving.",
        "",
        "Current algorithm:",
        "```python",
        current_code,
        "```",
        "",
        f"Performance: throughput={metrics.get('throughput_rps', 0):.1f} req/s, "
        f"TTFT={metrics.get('ttft_ms', 0):.1f}ms, "
        f"TPOT={metrics.get('tpot_ms', 0):.1f}ms",
    ]

    if roofline:
        analyzer = RooflineAnalyzer()
        roofline_text = analyzer.format_for_prompt(roofline)
        lines.extend(["", "Roofline Analysis:", roofline_text])

    lines.extend([
        "",
        "Generate an improved scheduling algorithm that addresses the bottlenecks above.",
        "The function must have the same signature. Return only the Python code.",
    ])

    return "\n".join(lines)


def build_cache_policy_prompt(
    current_code: str,
    metrics: Dict[str, float],
    roofline: Optional[RooflineReport] = None,
) -> str:
    """Build a prompt for evolving a KV cache eviction policy."""
    lines = [
        "You are evolving a KV cache eviction policy for LLM serving.",
        "",
        "Current policy:",
        "```python",
        current_code,
        "```",
        "",
        f"Performance: throughput={metrics.get('throughput_rps', 0):.1f} req/s, "
        f"cache_hit_rate={metrics.get('cache_hit_rate', 0):.1%}",
    ]

    if roofline:
        analyzer = RooflineAnalyzer()
        lines.extend(["", "Roofline Analysis:", analyzer.format_for_prompt(roofline)])

    lines.extend([
        "",
        "Generate an improved eviction policy. Return only the Python code.",
    ])
    return "\n".join(lines)


def build_cpu_scheduler_prompt(
    current_code: str,
    metrics: Dict[str, float],
    roofline: Optional[RooflineReport] = None,
    cpu_profile: Optional[Dict[str, float]] = None,
) -> str:
    """Build a prompt for evolving a CPU-specific batch scheduling algorithm.

    Includes CPU-specific constraints and analysis: compute-bound nature,
    NUMA topology, cache hierarchy, ISA-aware scheduling.

    Args:
        current_code: Current algorithm source code.
        metrics: Performance metrics dict.
        roofline: Optional roofline analysis report.
        cpu_profile: Optional CPU-specific profiling data.
    """
    lines = [
        "You are evolving a batch scheduling algorithm optimized for CPU-based LLM inference.",
        "",
        "CPU inference characteristics (from BudSim GenZ analysis):",
        "- CPU inference is 100% COMPUTE-BOUND (not memory-bound like GPU decode)",
        "- Embeddings consume ~41% of total compute time",
        "- Batch size is the primary throughput lever (BS=32 → 30x throughput vs BS=1)",
        "- INT8 quantization doubles throughput on CPU (use AMX/AVX-512 for BF16/INT8)",
        "- TTFT scales linearly with batch size (~114ms per request at 512 input tokens)",
        "- Pipeline parallelism HURTS throughput on CPU (always use PP=1)",
        "- Memory bandwidth becomes important only at small batch sizes (BS=1: 30% mem util)",
        "- L3 cache: KV cache should fit in L3 to avoid DRAM spills (288MB on GraniteRapids)",
        "- NUMA: group similar-length requests on same NUMA node to reduce cross-node traffic",
        "",
        "Current algorithm:",
        "```python",
        current_code,
        "```",
        "",
        f"Performance: throughput={metrics.get('throughput_rps', 0):.1f} req/s, "
        f"TTFT={metrics.get('ttft_ms', 0):.1f}ms, "
        f"TPOT={metrics.get('tpot_ms', 0):.1f}ms, "
        f"memory={metrics.get('memory_gb', 0):.1f}GB, "
        f"power={metrics.get('power_w', 0):.0f}W",
    ]

    if cpu_profile:
        lines.extend([
            "",
            "CPU profiling data:",
            f"  Compute utilization: {cpu_profile.get('compute_util', 0):.0%}",
            f"  Memory BW utilization: {cpu_profile.get('mem_bw_util', 0):.0%}",
            f"  L3 cache size: {cpu_profile.get('l3_cache_mb', 0):.0f} MB",
            f"  NUMA nodes: {cpu_profile.get('numa_nodes', 1)}",
            f"  ISA: {cpu_profile.get('isa', 'unknown')}",
        ])

    if roofline:
        analyzer = RooflineAnalyzer()
        roofline_text = analyzer.format_for_prompt(roofline)
        lines.extend(["", "Roofline Analysis:", roofline_text])

    lines.extend([
        "",
        "Key optimization targets for CPU inference:",
        "1. Maximize batch utilization while respecting TTFT SLO",
        "2. Keep KV cache within L3 cache capacity",
        "3. Group similar-length requests for minimal padding waste",
        "4. NUMA-aware request placement",
        "5. Priority-aware scheduling with SLO deadline tracking",
        "",
        "Generate an improved CPU-optimized scheduling algorithm.",
        "The function signature must be: schedule_batch_cpu(queue, **kwargs) -> List",
        "Return only the Python code.",
    ])

    return "\n".join(lines)
