"""
Base feature interface and metadata classes for GenZ simulation features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Union
from packaging import version


class FeatureCategory(Enum):
    """Categories of simulation features."""
    INFERENCE = "inference"
    HARDWARE = "hardware" 
    MODEL = "model"
    PARALLELISM = "parallelism"
    OPTIMIZATION = "optimization"


@dataclass
class FeatureMetadata:
    """Metadata for a simulation feature."""
    name: str
    version: str
    category: FeatureCategory
    description: str
    dependencies: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    min_genz_version: str = "1.0.0"
    max_genz_version: Optional[str] = None


class BaseFeature(ABC):
    """Base class for all simulation features."""
    
    def __init__(self, metadata: FeatureMetadata):
        self.metadata = metadata
        self._is_initialized = False
        self._config = {}
    
    @property
    def name(self) -> str:
        """Get feature name."""
        return self.metadata.name
    
    @property
    def category(self) -> FeatureCategory:
        """Get feature category."""
        return self.metadata.category
    
    @property
    def version(self) -> str:
        """Get feature version."""
        return self.metadata.version
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate feature configuration.
        
        Args:
            config: Feature configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def apply(self, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the feature to the simulation context.
        
        Args:
            simulation_context: Current simulation context
            
        Returns:
            Modified simulation context with feature applied
        """
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the feature with configuration.
        
        Args:
            config: Feature configuration
        """
        if not self.validate_config(config):
            raise ValueError(f"Invalid configuration for feature {self.name}")
        
        self._config = config.copy()
        self._is_initialized = True
    
    def cleanup(self) -> None:
        """Clean up feature resources."""
        self._is_initialized = False
        self._config.clear()
    
    def is_compatible_with(self, other_feature: 'BaseFeature') -> bool:
        """
        Check if this feature is compatible with another feature.
        
        Args:
            other_feature: Another feature to check compatibility with
            
        Returns:
            True if features are compatible
        """
        return other_feature.name not in self.metadata.incompatible_with
    
    def get_required_params(self) -> List[str]:
        """Get list of required configuration parameters."""
        return self.metadata.required_params.copy()
    
    def get_optional_params(self) -> List[str]:
        """Get list of optional configuration parameters."""
        return self.metadata.optional_params.copy()
    
    def get_dependencies(self) -> List[str]:
        """Get list of feature dependencies."""
        return self.metadata.dependencies.copy()
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.category.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} v{self.version}>"


class InferenceFeature(BaseFeature):
    """Base class for inference mode features."""
    
    def __init__(self, metadata: FeatureMetadata):
        super().__init__(metadata)
        if metadata.category != FeatureCategory.INFERENCE:
            raise ValueError("InferenceFeature must have INFERENCE category")
    
    @abstractmethod
    def run_inference(self, 
                     model_config: Dict[str, Any],
                     system_config: Dict[str, Any],
                     simulation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the inference simulation.
        
        Args:
            model_config: Model configuration
            system_config: System configuration  
            simulation_params: Simulation parameters
            
        Returns:
            Simulation results
        """
        pass


class ModelFeature(BaseFeature):
    """Base class for model enhancement features."""
    
    def __init__(self, metadata: FeatureMetadata):
        super().__init__(metadata)
        if metadata.category != FeatureCategory.MODEL:
            raise ValueError("ModelFeature must have MODEL category")
    
    @abstractmethod
    def modify_model(self, model_operations: List[List]) -> List[List]:
        """
        Modify model operations to apply the feature.
        
        Args:
            model_operations: List of model operations
            
        Returns:
            Modified list of operations
        """
        pass


class HardwareFeature(BaseFeature):
    """Base class for hardware optimization features."""
    
    def __init__(self, metadata: FeatureMetadata):
        super().__init__(metadata)
        if metadata.category != FeatureCategory.HARDWARE:
            raise ValueError("HardwareFeature must have HARDWARE category")
    
    @abstractmethod
    def optimize_system(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize system configuration for the feature.
        
        Args:
            system_config: System configuration
            
        Returns:
            Optimized system configuration
        """
        pass


class ParallelismFeature(BaseFeature):
    """Base class for parallelism features."""
    
    def __init__(self, metadata: FeatureMetadata):
        super().__init__(metadata)
        if metadata.category != FeatureCategory.PARALLELISM:
            raise ValueError("ParallelismFeature must have PARALLELISM category")
    
    @abstractmethod
    def configure_parallelism(self, parallelism_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure parallelism settings.
        
        Args:
            parallelism_config: Current parallelism configuration
            
        Returns:
            Updated parallelism configuration
        """
        pass 