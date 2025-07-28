"""
Simulation results module for unified interface.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class SimulationResult:
    """
    Unified result object for all simulation types.
    
    Attributes:
        latency: Total latency in seconds
        throughput: Throughput in tokens/second
        runtime_breakdown: Breakdown of runtime by component
        feature_metrics: Metrics specific to enabled features
        memory_usage: Memory usage information
        hardware_utilization: Hardware utilization metrics
        raw_output: Raw output from underlying simulation function
    """
    
    latency: float
    throughput: float
    runtime_breakdown: Dict[str, float] = field(default_factory=dict)
    feature_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    hardware_utilization: Dict[str, float] = field(default_factory=dict)
    raw_output: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_prefill_output(cls, output: Dict[str, Any]) -> "SimulationResult":
        """Create result from prefill modeling output."""
        # Extract key metrics
        latency = output.get("Latency", output.get("latency", 0))
        throughput = output.get("Throughput", output.get("throughput", 0))
        
        # Extract runtime breakdown
        runtime_breakdown = {}
        if "Runtime_breakdown" in output:
            runtime_breakdown = output["Runtime_breakdown"]
        elif "runtime_breakdown" in output:
            runtime_breakdown = output["runtime_breakdown"]
        
        # Extract memory usage
        memory_usage = {}
        if "Memory" in output:
            memory_usage = output["Memory"]
        elif "memory_usage" in output:
            memory_usage = output["memory_usage"]
        
        # Extract hardware utilization
        hardware_utilization = {}
        if "compute_utilization" in output:
            hardware_utilization["compute"] = output["compute_utilization"]
        if "memory_utilization" in output:
            hardware_utilization["memory"] = output["memory_utilization"]
        
        return cls(
            latency=latency,
            throughput=throughput,
            runtime_breakdown=runtime_breakdown,
            memory_usage=memory_usage,
            hardware_utilization=hardware_utilization,
            raw_output=output,
            feature_metrics={"prefill": {"tokens": output.get("input_tokens", 0)}}
        )
    
    @classmethod
    def from_decode_output(cls, output: Dict[str, Any]) -> "SimulationResult":
        """Create result from decode modeling output."""
        # Similar to prefill but with decode-specific metrics
        latency = output.get("Latency", output.get("latency", 0))
        throughput = output.get("Throughput", output.get("throughput", 0))
        
        runtime_breakdown = {}
        if "Runtime_breakdown" in output:
            runtime_breakdown = output["Runtime_breakdown"]
        elif "runtime_breakdown" in output:
            runtime_breakdown = output["runtime_breakdown"]
        
        memory_usage = {}
        if "Memory" in output:
            memory_usage = output["Memory"]
        elif "memory_usage" in output:
            memory_usage = output["memory_usage"]
        
        hardware_utilization = {}
        if "compute_utilization" in output:
            hardware_utilization["compute"] = output["compute_utilization"]
        if "memory_utilization" in output:
            hardware_utilization["memory"] = output["memory_utilization"]
        
        return cls(
            latency=latency,
            throughput=throughput,
            runtime_breakdown=runtime_breakdown,
            memory_usage=memory_usage,
            hardware_utilization=hardware_utilization,
            raw_output=output,
            feature_metrics={
                "decode": {
                    "input_tokens": output.get("input_tokens", 0),
                    "output_tokens": output.get("output_tokens", 0)
                }
            }
        )
    
    @classmethod
    def from_chunked_output(cls, output: Dict[str, Any]) -> "SimulationResult":
        """Create result from chunked modeling output."""
        # Chunked output has the same structure as prefill/decode
        # Extract key metrics
        latency = output.get("Latency", output.get("latency", 0))
        throughput = output.get("Throughput", output.get("throughput", 0))
        
        # Extract runtime breakdown
        runtime_breakdown = {}
        if "Runtime_breakdown" in output:
            breakdown = output["Runtime_breakdown"]
            if hasattr(breakdown, 'to_dict'):
                runtime_breakdown = breakdown.to_dict()
            elif isinstance(breakdown, dict):
                runtime_breakdown = breakdown
            else:
                runtime_breakdown = {"total": breakdown}
        elif "runtime_breakdown" in output:
            runtime_breakdown = output["runtime_breakdown"]
        
        # Extract memory usage
        memory_usage = {}
        if "Memory" in output:
            memory_usage = output["Memory"]
        elif "memory_usage" in output:
            memory_usage = output["memory_usage"]
        
        # Extract hardware utilization
        hardware_utilization = {}
        if "compute_utilization" in output:
            hardware_utilization["compute"] = output["compute_utilization"]
        if "memory_utilization" in output:
            hardware_utilization["memory"] = output["memory_utilization"]
        
        # For chunked, we don't have detailed prefill/decode breakdown in the standard output
        # But we can infer some information
        feature_metrics = {
            "chunked": {
                "total_latency": latency,
                "throughput": throughput,
                "is_offload": output.get("is_offload", False)
            }
        }
        
        return cls(
            latency=latency,
            throughput=throughput,
            runtime_breakdown=runtime_breakdown,
            memory_usage=memory_usage,
            hardware_utilization=hardware_utilization,
            raw_output=output,
            feature_metrics=feature_metrics
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "latency": self.latency,
            "throughput": self.throughput,
            "runtime_breakdown": self.runtime_breakdown,
            "feature_metrics": self.feature_metrics,
            "memory_usage": self.memory_usage,
            "hardware_utilization": self.hardware_utilization
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationResult":
        """Create result from dictionary."""
        return cls(
            latency=data.get("latency", 0),
            throughput=data.get("throughput", 0),
            runtime_breakdown=data.get("runtime_breakdown", {}),
            feature_metrics=data.get("feature_metrics", {}),
            memory_usage=data.get("memory_usage", {}),
            hardware_utilization=data.get("hardware_utilization", {})
        )
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SimulationResult":
        """Create result from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_feature_metric(self, feature: str, metric: str, default: Any = None) -> Any:
        """Get a specific metric for a feature."""
        if feature in self.feature_metrics:
            return self.feature_metrics[feature].get(metric, default)
        return default
    
    def add_feature_metrics(self, feature: str, metrics: Dict[str, Any]) -> None:
        """Add metrics for a specific feature."""
        if feature not in self.feature_metrics:
            self.feature_metrics[feature] = {}
        self.feature_metrics[feature].update(metrics)
    
    def __str__(self) -> str:
        """String representation of the result."""
        return (
            f"SimulationResult(\n"
            f"  latency={self.latency:.4f}s,\n"
            f"  throughput={self.throughput:.2f} tokens/s,\n"
            f"  features={list(self.feature_metrics.keys())}\n"
            f")"
        ) 