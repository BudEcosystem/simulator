"""Utility helpers for formatting hardware payloads."""
from typing import Any, Dict


def normalize_hardware_keys(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure lowercase hardware keys exist for their uppercase counterparts.

    Some hardware records may come from sources that use uppercase keys (e.g.,
    ``FLOPS`` instead of ``flops``). This helper preserves the original keys
    while populating the expected lowercase versions when they are missing so
    that API serializers can rely on a consistent shape.
    """
    if hardware is None:
        return {}

    normalized = dict(hardware)
    for key, value in hardware.items():
        if isinstance(key, str) and key.isupper():
            lower_key = key.lower()
            if lower_key not in normalized:
                normalized[lower_key] = value
    return normalized
