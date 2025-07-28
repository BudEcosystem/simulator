"""
GenZ Features Package

This package provides a unified interface for all GenZ simulation features,
including inference modes, hardware optimizations, model enhancements, and parallelism.
"""

from .registry import FeatureRegistry
from .base import BaseFeature, FeatureMetadata, FeatureCategory
from .decorators import register_feature, feature_compatibility

__all__ = [
    'FeatureRegistry',
    'BaseFeature', 
    'FeatureMetadata',
    'FeatureCategory',
    'register_feature',
    'feature_compatibility'
]

# Global feature registry instance
_global_registry = None

def get_global_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FeatureRegistry()
        _global_registry.discover_features()
    return _global_registry 