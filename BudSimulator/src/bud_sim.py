from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

try:
    from llm_memory_calculator.genz.simulation import SimulationEngine, SimulationConfig as GenZSimulationConfig
except ImportError:
    SimulationEngine = None
    GenZSimulationConfig = None


class SimType(Enum):
    USECASE_SIM = "usecase_sim"
    BEST_MODEL_SIM = "best_model_sim"
    BEST_HARDWARE_SIM = "best_hardware_sim"
    PARALLELISATION_STRATEGY_SIM = "parallelisation_strategy_sim"
    HETEROGENEOUS_SIM = "heterogeneous_sim"
    POWER_CONSUMPTION_SIM = "power_consumption_sim"
    COST_SIM = "cost_sim"
    YTD_SIM = "ytd_sim"


# Mapping from SimType to the primary simulation feature used by GenZ
_SIM_TYPE_TO_FEATURES = {
    SimType.USECASE_SIM: ["prefill"],
    SimType.BEST_MODEL_SIM: ["prefill"],
    SimType.BEST_HARDWARE_SIM: ["prefill"],
    SimType.PARALLELISATION_STRATEGY_SIM: ["prefill"],
    SimType.HETEROGENEOUS_SIM: ["prefill"],
    SimType.POWER_CONSUMPTION_SIM: ["prefill"],
    SimType.COST_SIM: ["prefill"],
    SimType.YTD_SIM: ["prefill", "decode"],
}


class SimulationConfig(BaseModel):
    models: Optional[list[dict]] = Field(description="The model id to simulate", default=None)
    batch_size: Optional[int] = Field(description="The batch size to simulate", default=None)
    precision: Optional[str] = Field(description="The precision to simulate", default=None)
    decode_length: Optional[int] = Field(description="The decode length to simulate", default=None)
    usecases: Optional[list[dict]] = Field(description="The usecases to simulate", default=None)
    hardwares: Optional[list[dict]] = Field(description="The hardwares to simulate", default=None)
    features: list[dict] = Field(description="The features to simulate", default=[])

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, v: Any) -> list[dict]:
        if not isinstance(v, list):
            raise ValueError("features must be a list")
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"features[{i}] must be a dict, got {type(item).__name__}")
            if "name" not in item:
                raise ValueError(f"features[{i}] must contain a 'name' key")
        return v


class BudSimulator:

    sim_type: Optional[SimType] = None

    def __init__(self, sim_type: Optional[SimType] = None):
        self.sim_type = sim_type
        self._engine: Optional[Any] = None

    def _get_engine(self) -> Any:
        """Get or create the SimulationEngine instance."""
        if SimulationEngine is None:
            raise ImportError(
                "SimulationEngine is not available. "
                "Install llm-memory-calculator: pip install -e llm-memory-calculator/"
            )
        if self._engine is None:
            self._engine = SimulationEngine()
        return self._engine

    def set_simulation_type(self, sim_type: SimType) -> None:
        self.sim_type = sim_type

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run a simulation using the GenZ SimulationEngine.

        Args:
            **kwargs: Simulation parameters. Expected keys depend on sim_type:
                - model (str): Model name or identifier (required)
                - batch_size (int): Batch size for simulation
                - input_tokens (int): Number of input tokens
                - output_tokens (int): Number of output tokens (for decode)
                - system_name (str): Hardware system name
                - bits (str): Precision format (e.g., 'bf16', 'fp16', 'int8')
                - tensor_parallel (int): Tensor parallelism degree
                - pipeline_parallel (int): Pipeline parallelism degree
                - expert_parallel (int): Expert parallelism degree
                - features (list[str]): Additional GenZ features to enable

        Returns:
            Dictionary with simulation results including latency, throughput,
            runtime breakdown, memory usage, and hardware utilization.

        Raises:
            ImportError: If SimulationEngine is not available.
            ValueError: If sim_type is not set or configuration is invalid.
        """
        if self.sim_type is None:
            raise ValueError("Simulation type not set. Call set_simulation_type() or pass sim_type to __init__.")

        engine = self._get_engine()
        config = self._build_config(**kwargs)
        result = engine.simulate(config)

        return {
            "latency": result.latency,
            "throughput": result.throughput,
            "runtime_breakdown": result.runtime_breakdown,
            "memory_usage": result.memory_usage,
            "hardware_utilization": result.hardware_utilization,
            "feature_metrics": result.feature_metrics,
            "raw_output": result.raw_output,
        }

    def _build_config(self, **kwargs: Any) -> Any:
        """Build a GenZ SimulationConfig from the sim_type and kwargs.

        Maps BudSimulator's SimType to GenZ features and simulation_params.
        """
        if GenZSimulationConfig is None:
            raise ImportError("GenZSimulationConfig is not available.")

        model = kwargs.pop("model", None)
        if model is None:
            raise ValueError("'model' parameter is required for simulation.")

        # Determine features from sim_type
        base_features = list(_SIM_TYPE_TO_FEATURES.get(self.sim_type, ["prefill"]))

        # Allow overriding or extending features
        extra_features = kwargs.pop("features", [])
        if isinstance(extra_features, list):
            for f in extra_features:
                if f not in base_features:
                    base_features.append(f)

        # Build simulation params from remaining kwargs
        simulation_params: dict[str, Any] = {}
        param_keys = [
            "batch_size", "input_tokens", "output_tokens",
            "system_name", "bits", "tensor_parallel",
            "pipeline_parallel", "expert_parallel", "debug",
        ]
        for key in param_keys:
            if key in kwargs:
                simulation_params[key] = kwargs[key]

        system_config = kwargs.get("system_config")

        return GenZSimulationConfig(
            model=model,
            features=base_features,
            simulation_params=simulation_params,
            system_config=system_config,
        )

    def get_supported_features(self) -> list[dict[str, Any]]:
        """Return supported simulation features from the GenZ engine.

        Returns a list of feature info dicts, each containing at minimum
        a 'name' key. Returns an empty list if the engine is unavailable.
        """
        if SimulationEngine is None:
            return []

        engine = self._get_engine()
        feature_names = engine.get_available_features()

        features = []
        for name in feature_names:
            try:
                info = engine.feature_registry.get_feature_info(name)
                if isinstance(info, dict):
                    features.append(info)
                else:
                    features.append({"name": name})
            except (AttributeError, KeyError):
                features.append({"name": name})

        return features
