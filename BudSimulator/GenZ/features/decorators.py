"""
Decorators for feature registration and compatibility management.
"""

from functools import wraps
from typing import Dict, Any, List, Optional, Callable
from .base import FeatureMetadata, FeatureCategory


def register_feature(
    name: str,
    version: str = "1.0.0",
    category: FeatureCategory = FeatureCategory.OPTIMIZATION,
    description: str = "",
    dependencies: Optional[List[str]] = None,
    incompatible_with: Optional[List[str]] = None,
    required_params: Optional[List[str]] = None,
    optional_params: Optional[List[str]] = None,
    min_genz_version: str = "1.0.0",
    max_genz_version: Optional[str] = None
):
    """
    Decorator to register a feature with the global registry.
    
    Args:
        name: Feature name
        version: Feature version
        category: Feature category
        description: Feature description
        dependencies: List of required features
        incompatible_with: List of incompatible features
        required_params: List of required configuration parameters
        optional_params: List of optional configuration parameters
        min_genz_version: Minimum GenZ version required
        max_genz_version: Maximum GenZ version supported
    """
    def decorator(cls):
        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            version=version,
            category=category,
            description=description,
            dependencies=dependencies or [],
            incompatible_with=incompatible_with or [],
            required_params=required_params or [],
            optional_params=optional_params or [],
            min_genz_version=min_genz_version,
            max_genz_version=max_genz_version
        )
        
        # Store metadata on the class
        cls._feature_metadata = metadata
        
        # Register with global registry (lazy registration)
        if not hasattr(cls, '_registered'):
            cls._registered = True
            
            # Import here to avoid circular imports
            from . import get_global_registry
            registry = get_global_registry()
            registry.register_feature_class(cls)
        
        return cls
    
    return decorator


def feature_compatibility(*compatible_features: str):
    """
    Decorator to specify feature compatibility.
    
    Args:
        compatible_features: Names of features this feature is compatible with
    """
    def decorator(cls):
        if hasattr(cls, '_feature_metadata'):
            # Remove from incompatible list if present
            for feature in compatible_features:
                if feature in cls._feature_metadata.incompatible_with:
                    cls._feature_metadata.incompatible_with.remove(feature)
        
        # Store compatibility info
        if not hasattr(cls, '_compatible_features'):
            cls._compatible_features = set()
        cls._compatible_features.update(compatible_features)
        
        return cls
    
    return decorator


def requires_params(*param_names: str):
    """
    Decorator to specify required parameters for a feature.
    
    Args:
        param_names: Names of required parameters
    """
    def decorator(cls):
        if hasattr(cls, '_feature_metadata'):
            cls._feature_metadata.required_params.extend(param_names)
        else:
            # Store for later use
            if not hasattr(cls, '_required_params'):
                cls._required_params = []
            cls._required_params.extend(param_names)
        
        return cls
    
    return decorator


def optional_params(*param_names: str):
    """
    Decorator to specify optional parameters for a feature.
    
    Args:
        param_names: Names of optional parameters
    """
    def decorator(cls):
        if hasattr(cls, '_feature_metadata'):
            cls._feature_metadata.optional_params.extend(param_names)
        else:
            # Store for later use
            if not hasattr(cls, '_optional_params'):
                cls._optional_params = []
            cls._optional_params.extend(param_names)
        
        return cls
    
    return decorator


def depends_on(*feature_names: str):
    """
    Decorator to specify feature dependencies.
    
    Args:
        feature_names: Names of features this feature depends on
    """
    def decorator(cls):
        if hasattr(cls, '_feature_metadata'):
            cls._feature_metadata.dependencies.extend(feature_names)
        else:
            # Store for later use
            if not hasattr(cls, '_dependencies'):
                cls._dependencies = []
            cls._dependencies.extend(feature_names)
        
        return cls
    
    return decorator


def incompatible_with(*feature_names: str):
    """
    Decorator to specify feature incompatibilities.
    
    Args:
        feature_names: Names of features this feature is incompatible with
    """
    def decorator(cls):
        if hasattr(cls, '_feature_metadata'):
            cls._feature_metadata.incompatible_with.extend(feature_names)
        else:
            # Store for later use
            if not hasattr(cls, '_incompatible_with'):
                cls._incompatible_with = []
            cls._incompatible_with.extend(feature_names)
        
        return cls
    
    return decorator


def validate_config(validation_func: Callable[[Dict[str, Any]], bool]):
    """
    Decorator to add custom configuration validation to a feature.
    
    Args:
        validation_func: Function that takes config dict and returns bool
    """
    def decorator(cls):
        original_validate = cls.validate_config
        
        @wraps(original_validate)
        def wrapper(self, config: Dict[str, Any]) -> bool:
            # Run original validation first
            if not original_validate(self, config):
                return False
            
            # Run custom validation
            return validation_func(config)
        
        cls.validate_config = wrapper
        return cls
    
    return decorator


def post_process_results(post_process_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
    """
    Decorator to add post-processing to feature results.
    
    Args:
        post_process_func: Function that takes and returns result dict
    """
    def decorator(cls):
        original_apply = cls.apply
        
        @wraps(original_apply)
        def wrapper(self, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
            # Run original apply
            result = original_apply(self, simulation_context)
            
            # Post-process results
            return post_process_func(result)
        
        cls.apply = wrapper
        return cls
    
    return decorator 