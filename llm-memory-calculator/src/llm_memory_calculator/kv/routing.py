"""The routing policy class (FR-ROUTE-4): from the reuse profile, never from the replica count."""

from __future__ import annotations

from typing import Any, Dict


def routing_policy(profile: str) -> Dict[str, Any]:
    """Agentic and batch traffic gets maximal affinity; everything else is capped by load so that
    packing onto warm replicas doesn't cost decode latency (+22-45% ITL p50 in llm-d's benchmark).
    """
    return {
        "strategy": "prefix-cache",
        "affinity": "max" if profile in ("agentic", "batch") else "capped",
        "load_factor": 1.25,
    }
