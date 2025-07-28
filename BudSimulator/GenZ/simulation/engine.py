"""
Simulation engine for unified interface.
"""

from typing import Dict, Any, List, Optional
import logging

from .config import SimulationConfig, SimulationType
from .results import SimulationResult
from ..features.registry import FeatureRegistry
from ..features.base import BaseFeature, FeatureCategory

# Import existing GenZ functions
from ..LLM_inference import prefill_moddeling, decode_moddeling, chunked_moddeling

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Main simulation engine that provides a unified interface for all simulation types.
    
    This engine orchestrates the execution of simulations with various features
    and provides a consistent interface regardless of the underlying implementation.
    """
    
    def __init__(self):
        self.feature_registry = FeatureRegistry()
        self._active_features: Dict[str, BaseFeature] = {}
    
    def get_available_features(self) -> List[str]:
        """Get list of all available features."""
        return self.feature_registry.get_available_features()
    
    def simulate(self, config: SimulationConfig) -> SimulationResult:
        """
        Run a simulation with the given configuration.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Simulation result
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        if not config.is_valid():
            raise ValueError("Invalid simulation configuration")
        
        # Validate feature combination
        self.feature_registry.validate_feature_combination(config.features)
        
        # Initialize features
        self._initialize_features(config)
        
        try:
            # Apply pre-simulation features
            simulation_context = self._apply_pre_simulation_features(config)
            
            # Run the main simulation
            raw_result = self._run_simulation(config, simulation_context)
            
            # Convert to unified result format
            result = self._convert_to_unified_result(config, raw_result)
            
            # Apply post-simulation features
            result = self._apply_post_simulation_features(result, simulation_context)
            
            return result
            
        finally:
            # Clean up features
            self._cleanup_features()
    
    def _initialize_features(self, config: SimulationConfig):
        """Initialize features for the simulation."""
        self._active_features.clear()
        
        for feature_name in config.features:
            # Skip built-in features (handled by simulation functions)
            if self.feature_registry.is_builtin_feature(feature_name):
                continue
            
            # Get feature configuration
            feature_config = config.simulation_params.get(feature_name, {})
            
            # Create and initialize feature
            feature = self.feature_registry.create_feature(feature_name, feature_config)
            if feature:
                self._active_features[feature_name] = feature
    
    def _cleanup_features(self):
        """Clean up active features."""
        for feature in self._active_features.values():
            feature.cleanup()
        self._active_features.clear()
    
    def _apply_pre_simulation_features(self, config: SimulationConfig) -> Dict[str, Any]:
        """Apply features that modify the simulation context before running."""
        simulation_context = {
            "config": config,
            "model": config.model,
            "system": config.simulation_params.get("system_name", "A100_40GB_GPU"),
            "params": config.get_simulation_params()
        }
        
        # Apply each active feature
        for feature_name, feature in self._active_features.items():
            if feature.category in [FeatureCategory.MODEL, FeatureCategory.HARDWARE]:
                simulation_context = feature.apply(simulation_context)
        
        return simulation_context
    
    def _apply_post_simulation_features(self, 
                                      result: SimulationResult, 
                                      simulation_context: Dict[str, Any]) -> SimulationResult:
        """Apply features that modify the result after simulation."""
        # Apply optimization features
        for feature_name, feature in self._active_features.items():
            if feature.category == FeatureCategory.OPTIMIZATION:
                # Apply feature and extract metrics
                modified_context = feature.apply(simulation_context)
                
                # Add feature-specific metrics to result
                if "feature_metrics" in modified_context:
                    result.add_feature_metrics(
                        feature_name, 
                        modified_context["feature_metrics"]
                    )
        
        return result
    
    def _run_simulation(self, config: SimulationConfig, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the actual simulation based on the configuration."""
        params = context["params"]
        
        # Handle different simulation types
        if config.simulation_type == SimulationType.PREFILL:
            return self._run_prefill_simulation(params)
        elif config.simulation_type == SimulationType.DECODE:
            return self._run_decode_simulation(params)
        elif config.simulation_type == SimulationType.CHUNKED:
            return self._run_chunked_simulation(params)
        else:
            raise ValueError(f"Unsupported simulation type: {config.simulation_type}")
    
    def _run_prefill_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run prefill simulation."""
        # Extract parameters with defaults
        result = prefill_moddeling(
            model=params.get("model"),
            batch_size=params.get("batch_size", 1),
            input_tokens=params.get("input_tokens", 1024),
            system_name=params.get("system_name", "A100_40GB_GPU"),
            bits=params.get("bits", "bf16"),
            tensor_parallel=params.get("tensor_parallel", 1),
            pipeline_parallel=params.get("pipeline_parallel", 1),
            expert_parallel=params.get("expert_parallel", 1),
            debug=params.get("debug", False)
        )
        
        # Ensure consistent structure
        if isinstance(result, dict):
            return result
        else:
            # Handle ModdelingOutput or other return types
            return result.__dict__ if hasattr(result, '__dict__') else {"result": result}
    
    def _run_decode_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run decode simulation."""
        result = decode_moddeling(
            model=params.get("model"),
            batch_size=params.get("batch_size", 1),
            input_tokens=params.get("input_tokens", 1024),
            output_tokens=params.get("output_tokens", 256),
            system_name=params.get("system_name", "A100_40GB_GPU"),
            bits=params.get("bits", "bf16"),
            tensor_parallel=params.get("tensor_parallel", 1),
            pipeline_parallel=params.get("pipeline_parallel", 1),
            expert_parallel=params.get("expert_parallel", 1),
            debug=params.get("debug", False)
        )
        
        if isinstance(result, dict):
            return result
        else:
            return result.__dict__ if hasattr(result, '__dict__') else {"result": result}
    
    def _run_chunked_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run chunked simulation."""
        result = chunked_moddeling(
            model=params.get("model"),
            prefill_kv_sizes=params.get("prefill_kv_sizes", [(1024, 511)]),
            decode_kv_sizes=params.get("decode_kv_sizes", [1600]),
            system_name=params.get("system_name", "A100_40GB_GPU"),
            bits=params.get("bits", "bf16"),
            tensor_parallel=params.get("tensor_parallel", 1),
            pipeline_parallel=params.get("pipeline_parallel", 1),
            expert_parallel=params.get("expert_parallel", 1),
            debug=params.get("debug", False)
        )
        
        # Chunked modeling returns a different structure
        # Need to adapt it to our unified format
        return self._adapt_chunked_result(result)
    
    def _adapt_chunked_result(self, raw_result: Any) -> Dict[str, Any]:
        """Adapt chunked modeling result to unified format."""
        if isinstance(raw_result, dict):
            return raw_result
        
        # Handle different possible return formats from chunked_moddeling
        adapted = {
            "prefill_results": [],
            "decode_results": []
        }
        
        # This will need to be adjusted based on actual chunked_moddeling output
        if hasattr(raw_result, 'prefill_results'):
            adapted["prefill_results"] = raw_result.prefill_results
        if hasattr(raw_result, 'decode_results'):
            adapted["decode_results"] = raw_result.decode_results
        
        # Calculate totals
        total_latency = 0
        if "prefill_results" in adapted:
            for result in adapted["prefill_results"]:
                if isinstance(result, dict) and "latency" in result:
                    total_latency += result["latency"]
        
        if "decode_results" in adapted:
            for result in adapted["decode_results"]:
                if isinstance(result, dict) and "latency" in result:
                    total_latency += result["latency"]
        
        adapted["total_latency"] = total_latency
        
        return adapted
    
    def _convert_to_unified_result(self, config: SimulationConfig, raw_result: Dict[str, Any]) -> SimulationResult:
        """Convert raw simulation output to unified result format."""
        if config.simulation_type == SimulationType.PREFILL:
            result = SimulationResult.from_prefill_output(raw_result)
        elif config.simulation_type == SimulationType.DECODE:
            result = SimulationResult.from_decode_output(raw_result)
        elif config.simulation_type == SimulationType.CHUNKED:
            result = SimulationResult.from_chunked_output(raw_result)
        else:
            # Generic conversion
            result = SimulationResult(
                latency=raw_result.get("latency", 0),
                throughput=raw_result.get("throughput", 0),
                runtime_breakdown=raw_result.get("runtime_breakdown", {}),
                memory_usage=raw_result.get("memory_usage", {}),
                hardware_utilization=raw_result.get("hardware_utilization", {}),
                raw_output=raw_result
            )
        
        # Add metrics for built-in features
        for feature in config.features:
            if self.feature_registry.is_builtin_feature(feature):
                if feature == "tensor_parallel":
                    result.add_feature_metrics("tensor_parallel", {
                        "degree": config.simulation_params.get("tensor_parallel", 1),
                        "enabled": config.simulation_params.get("tensor_parallel", 1) > 1
                    })
                elif feature == "pipeline_parallel":
                    result.add_feature_metrics("pipeline_parallel", {
                        "degree": config.simulation_params.get("pipeline_parallel", 1),
                        "enabled": config.simulation_params.get("pipeline_parallel", 1) > 1
                    })
                elif feature == "lora":
                    lora_config = config.simulation_params.get("lora", {})
                    result.add_feature_metrics("lora", {
                        "enabled": lora_config.get("enabled", True),
                        "rank": lora_config.get("rank", 16),
                        "strategy": lora_config.get("strategy", "default")
                    })
                elif feature == "flash_attention":
                    result.add_feature_metrics("flash_attention", {
                        "enabled": True
                    })
        
        return result
    
    def validate_configuration(self, config: SimulationConfig) -> Dict[str, Any]:
        """
        Validate a simulation configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with details
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check basic configuration validity
        if not config.is_valid():
            result["valid"] = False
            result["errors"].append("Configuration failed basic validation")
        
        # Validate feature combination
        try:
            self.feature_registry.validate_feature_combination(config.features)
        except ValueError as e:
            result["valid"] = False
            result["errors"].append(str(e))
        
        # Check for missing optional parameters
        if config.simulation_type == SimulationType.DECODE:
            if "output_tokens" not in config.simulation_params:
                result["warnings"].append("output_tokens not specified, using default: 256")
        
        return result 