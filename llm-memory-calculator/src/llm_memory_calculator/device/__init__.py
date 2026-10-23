"""FRD-027 Part A, layer 0: device facts (§6)."""

from . import catalogue
from .facts import GIB, MIB, DeviceFacts, HardwareMode, Interconnect, Virtualization
from .gates import allows_dtype, canonical_format, host_register_allowed, kernel_path, max_tp
from .live import facts_from_node

__all__ = [
    "GIB",
    "MIB",
    "DeviceFacts",
    "HardwareMode",
    "Interconnect",
    "Virtualization",
    "allows_dtype",
    "canonical_format",
    "catalogue",
    "facts_from_node",
    "host_register_allowed",
    "kernel_path",
    "max_tp",
]
