"""
Simulation configuration module for unified interface.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class SimulationType(Enum):
    """Types of simulation that can be run."""
    PREFILL = "prefill"
    DECODE = "decode"
    CHUNKED = "chunked"
    CONTINUOUS = "continuous"


@dataclass
class SimulationConfig:
    """
    Configuration for running simulations through the unified interface.
    
    Attributes:
        model: Model name or configuration
        features: List of features to enable for the simulation
        simulation_params: Parameters specific to the simulation type
        system_config: System/hardware configuration override
    """
    
    model: Union[str, Dict[str, Any]]
    features: List[str] = field(default_factory=list)
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    system_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Determine simulation type from features
        self._determine_simulation_type()
        
        # Validate feature compatibility
        self._validate_features()
        
        # Check if we have required parameters before setting defaults
        missing_before_defaults = self._get_missing_required_params()
        
        # Set default parameters if not provided
        self._set_defaults()
        
        # If we had missing required parameters before defaults, raise error
        if missing_before_defaults:
            raise ValueError(f"Missing required parameters: {missing_before_defaults}")
    
    def _determine_simulation_type(self):
        """Determine the primary simulation type from features."""
        sim_types = {
            "prefill": SimulationType.PREFILL,
            "decode": SimulationType.DECODE,
            "chunked": SimulationType.CHUNKED,
            "continuous": SimulationType.CONTINUOUS
        }
        
        self.simulation_type = None
        for feature in self.features:
            if feature in sim_types:
                if self.simulation_type is not None:
                    raise ValueError(
                        f"Multiple simulation types specified: {self.simulation_type.value} and {feature}. "
                        "Only one primary simulation type is allowed."
                    )
                self.simulation_type = sim_types[feature]
        
        if self.simulation_type is None:
            # Default to prefill if no specific type is mentioned
            self.simulation_type = SimulationType.PREFILL
    
    def _validate_features(self):
        """Validate that the specified features are compatible."""
        # Check for incompatible feature combinations
        incompatible_pairs = [
            ("prefill", "decode"),  # Can't do both in single simulation
            ("chunked", "continuous"),  # Mutually exclusive modes
        ]
        
        for feat1, feat2 in incompatible_pairs:
            if feat1 in self.features and feat2 in self.features:
                raise ValueError(f"Features '{feat1}' and '{feat2}' are incompatible")
        
        # Validate feature dependencies
        feature_dependencies = {
            "speculative_decode": ["decode"],
            "memory_offload": ["chunked", "decode"],
        }
        
        for feature, deps in feature_dependencies.items():
            if feature in self.features:
                if not any(dep in self.features for dep in deps):
                    raise ValueError(
                        f"Feature '{feature}' requires one of: {', '.join(deps)}"
                    )
    
    def _set_defaults(self):
        """Set default parameters based on simulation type."""
        defaults = {
            SimulationType.PREFILL: {
                "batch_size": 1,
                "input_tokens": 1024,
                "system_name": "A100_40GB_GPU",
                "bits": "bf16",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "expert_parallel": 1
            },
            SimulationType.DECODE: {
                "batch_size": 1,
                "input_tokens": 1024,
                "output_tokens": 256,
                "system_name": "A100_40GB_GPU",
                "bits": "bf16",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "expert_parallel": 1
            },
            SimulationType.CHUNKED: {
                "prefill_kv_sizes": [(1024, 511)],
                "decode_kv_sizes": [1600],
                "system_name": "A100_40GB_GPU",
                "bits": "bf16",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "expert_parallel": 1
            }
        }
        
        # Apply defaults for the simulation type
        if self.simulation_type in defaults:
            for key, value in defaults[self.simulation_type].items():
                if key not in self.simulation_params:
                    self.simulation_params[key] = value
    
    def _get_missing_required_params(self) -> List[str]:
        """Get list of missing required parameters."""
        required_params = {
            SimulationType.PREFILL: ["batch_size", "input_tokens"],
            SimulationType.DECODE: ["batch_size", "input_tokens", "output_tokens"],
            SimulationType.CHUNKED: ["prefill_kv_sizes", "decode_kv_sizes"]
        }
        
        missing = []
        if self.simulation_type in required_params:
            for param in required_params[self.simulation_type]:
                if param not in self.simulation_params:
                    missing.append(param)
        
        return missing
    
    def is_valid(self) -> bool:
        """
        Check if the configuration is valid.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required parameters for each simulation type
            required_params = {
                SimulationType.PREFILL: ["batch_size", "input_tokens"],
                SimulationType.DECODE: ["batch_size", "input_tokens", "output_tokens"],
                SimulationType.CHUNKED: ["prefill_kv_sizes", "decode_kv_sizes"]
            }
            
            if self.simulation_type in required_params:
                for param in required_params[self.simulation_type]:
                    if param not in self.simulation_params:
                        return False
            
            # Validate model specification
            if not self.model:
                return False
            
            return True
        except Exception:
            return False
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get parameters formatted for the specific simulation function."""
        params = self.simulation_params.copy()
        
        # Add model to params
        params["model"] = self.model
        
        # Add debug flag if not present
        if "debug" not in params:
            params["debug"] = False
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model,
            "features": self.features,
            "simulation_params": self.simulation_params,
            "system_config": self.system_config,
            "simulation_type": self.simulation_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from dictionary."""
        # Remove simulation_type as it's determined from features
        data.pop("simulation_type", None)
        return cls(**data) 