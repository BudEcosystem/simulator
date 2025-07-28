"""
Feature registry for discovering and managing simulation features.
"""

from typing import Dict, List, Type, Optional, Any
import importlib
import inspect
import pkgutil
from pathlib import Path

from .base import BaseFeature, FeatureMetadata, FeatureCategory


class FeatureRegistry:
    """
    Registry for discovering and managing simulation features.
    
    This class automatically discovers features in the features directory
    and provides methods to access and instantiate them.
    """
    
    def __init__(self):
        self._features: Dict[str, Type[BaseFeature]] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}
        self._discover_features()
    
    def _discover_features(self):
        """Discover all available features in the features package."""
        # Get the features package
        features_package = importlib.import_module("GenZ.features")
        features_path = Path(features_package.__file__).parent
        
        # Built-in features that map to existing functionality
        self._register_builtin_features()
        
        # Discover feature modules
        for module_info in pkgutil.iter_modules([str(features_path)]):
            if module_info.name in ["base", "registry", "decorators", "__init__"]:
                continue
            
            try:
                module = importlib.import_module(f"GenZ.features.{module_info.name}")
                
                # Find all feature classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseFeature) and 
                        obj is not BaseFeature and
                        not inspect.isabstract(obj)):
                        
                        # Create instance to get metadata
                        if hasattr(obj, 'get_metadata'):
                            metadata = obj.get_metadata()
                        else:
                            # Try to instantiate with a dummy metadata
                            try:
                                dummy_metadata = FeatureMetadata(
                                    name=name.lower(),
                                    category=FeatureCategory.OPTIMIZATION,
                                    version="1.0.0",
                                    description=f"Feature {name}"
                                )
                                instance = obj(dummy_metadata)
                                metadata = instance.metadata
                            except Exception:
                                continue
                        
                        self._features[metadata.name] = obj
                        self._metadata[metadata.name] = metadata
                        
            except Exception as e:
                # Log error but continue discovering other features
                print(f"Error discovering features in {module_info.name}: {e}")
    
    def _register_builtin_features(self):
        """Register built-in features that wrap existing functionality."""
        # These are pseudo-features that map to existing GenZ functions
        builtin_features = [
            FeatureMetadata(
                name="prefill",
                category=FeatureCategory.INFERENCE,
                version="1.0.0",
                description="Standard prefill inference mode",
                incompatible_with=["decode", "chunked"]
            ),
            FeatureMetadata(
                name="decode",
                category=FeatureCategory.INFERENCE,
                version="1.0.0",
                description="Standard decode inference mode",
                incompatible_with=["prefill", "chunked"]
            ),
            FeatureMetadata(
                name="chunked",
                category=FeatureCategory.INFERENCE,
                version="1.0.0",
                description="Chunked prefill/decode inference mode",
                incompatible_with=["prefill", "decode"]
            ),
            FeatureMetadata(
                name="lora",
                category=FeatureCategory.MODEL,
                version="1.0.0",
                description="Low-Rank Adaptation (LoRA) optimization"
            ),
            FeatureMetadata(
                name="flash_attention",
                category=FeatureCategory.OPTIMIZATION,
                version="1.0.0",
                description="Flash Attention optimization"
            ),
            FeatureMetadata(
                name="tensor_parallel",
                category=FeatureCategory.PARALLELISM,
                version="1.0.0",
                description="Tensor parallelism"
            ),
            FeatureMetadata(
                name="pipeline_parallel",
                category=FeatureCategory.PARALLELISM,
                version="1.0.0",
                description="Pipeline parallelism"
            ),
            FeatureMetadata(
                name="cpu_optimization",
                category=FeatureCategory.HARDWARE,
                version="1.0.0",
                description="CPU-specific optimizations"
            ),
            FeatureMetadata(
                name="memory_offload",
                category=FeatureCategory.OPTIMIZATION,
                version="1.0.0",
                description="Memory offloading to external storage"
            ),
            FeatureMetadata(
                name="speculative_decode",
                category=FeatureCategory.OPTIMIZATION,
                version="1.0.0",
                description="Speculative decoding optimization",
                dependencies=["decode"]
            )
        ]
        
        for metadata in builtin_features:
            # Create a placeholder class for built-in features
            # These will be handled specially by the SimulationEngine
            self._metadata[metadata.name] = metadata
            self._features[metadata.name] = None  # Special marker for built-in
    
    def get_available_features(self) -> List[str]:
        """Get list of all available feature names."""
        return list(self._features.keys())
    
    def get_feature_metadata(self, name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a specific feature."""
        return self._metadata.get(name)
    
    def get_features_by_category(self, category: FeatureCategory) -> List[str]:
        """Get all features in a specific category."""
        return [
            name for name, metadata in self._metadata.items()
            if metadata.category == category
        ]
    
    def is_builtin_feature(self, name: str) -> bool:
        """Check if a feature is a built-in feature."""
        return name in self._features and self._features[name] is None
    
    def create_feature(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseFeature]:
        """
        Create an instance of a feature.
        
        Args:
            name: Feature name
            config: Feature configuration
            
        Returns:
            Feature instance or None if feature is built-in
        """
        if name not in self._features:
            raise ValueError(f"Unknown feature: {name}")
        
        feature_class = self._features[name]
        if feature_class is None:
            # Built-in feature, return None
            return None
        
        # Create feature instance
        metadata = self._metadata[name]
        feature = feature_class(metadata)
        
        # Initialize with config if provided
        if config:
            feature.initialize(config)
        
        return feature
    
    def validate_feature_combination(self, features: List[str]) -> bool:
        """
        Validate that a combination of features is compatible.
        
        Args:
            features: List of feature names
            
        Returns:
            True if combination is valid
            
        Raises:
            ValueError: If features are incompatible
        """
        for feature in features:
            if feature not in self._features:
                raise ValueError(f"Unknown feature: {feature}")
        
        # Check incompatibilities
        for i, feat1 in enumerate(features):
            metadata1 = self._metadata[feat1]
            
            for feat2 in features[i+1:]:
                # Check if features are incompatible
                if feat2 in metadata1.incompatible_with:
                    raise ValueError(
                        f"Feature '{feat1}' is incompatible with '{feat2}'"
                    )
                
                metadata2 = self._metadata[feat2]
                if feat1 in metadata2.incompatible_with:
                    raise ValueError(
                        f"Feature '{feat2}' is incompatible with '{feat1}'"
                    )
        
        # Check dependencies
        for feature in features:
            metadata = self._metadata[feature]
            for dep in metadata.dependencies:
                if dep not in features:
                    raise ValueError(
                        f"Feature '{feature}' requires '{dep}' to be enabled"
                    )
        
        return True
    
    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a feature."""
        if name not in self._metadata:
            raise ValueError(f"Unknown feature: {name}")
        
        metadata = self._metadata[name]
        return {
            "name": metadata.name,
            "category": metadata.category.value,
            "version": metadata.version,
            "description": metadata.description,
            "dependencies": metadata.dependencies,
            "incompatible_with": metadata.incompatible_with,
            "is_builtin": self.is_builtin_feature(name)
        } 