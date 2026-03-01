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
