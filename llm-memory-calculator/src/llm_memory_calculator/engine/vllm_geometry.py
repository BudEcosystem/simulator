"""vLLM KV-cache geometry, owned by the engine layer (FRD-027 §8; IMPLEMENTATION_PLAN 0.3).

``kv/geometry.py`` remains the implementation for FRD-023's planner; this module is the engine layer's name for
it, so both paths read one geometry.
"""

from ..kv.geometry import KVGeometry, kv_bytes_per_token_per_rank

__all__ = ["KVGeometry", "kv_bytes_per_token_per_rank"]
