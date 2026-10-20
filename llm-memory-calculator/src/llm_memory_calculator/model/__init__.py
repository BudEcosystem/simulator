"""FRD-027 Part A, layer 1: model facts (§7)."""

from .classes import ARCH_CLASSES, SIZE_BUCKETS, arch_class, size_bucket
from .facts import Component, LayerCounts, ModelFacts, canonical_format, dtype_bytes
from .recognise import KNOWN_MODEL_TYPES, recognised

__all__ = [
    "ARCH_CLASSES",
    "KNOWN_MODEL_TYPES",
    "SIZE_BUCKETS",
    "Component",
    "LayerCounts",
    "ModelFacts",
    "arch_class",
    "canonical_format",
    "dtype_bytes",
    "recognised",
    "size_bucket",
]
