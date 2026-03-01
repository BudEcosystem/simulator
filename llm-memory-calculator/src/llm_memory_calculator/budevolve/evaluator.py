"""BudSim evaluation API -- wraps GenZ as a fitness function.

Provides BudSimEvaluator, which wraps the GenZ analytical engine's
prefill_moddeling and decode_moddeling functions into a clean fitness
evaluation interface for BudEvolve's optimization loops.

Two evaluation modes:
    1. evaluate_config: Evaluate a named hardware + model configuration
    2. evaluate_hardware: Evaluate a hypothetical hardware specification
       by constructing a System object from raw parameters
"""
import warnings
from typing import Optional

from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling
from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling
from llm_memory_calculator.genz.unit import Unit

from .types import ServingConfig, HardwareSpec, EvalResult


# Cloud pricing: $/GPU-hour by hardware name.
# Sources: AWS on-demand, Lambda Labs, Coreweave (2024-2025 averages).
_CLOUD_PRICING_PER_GPU_HOUR = {
    "A100_40GB_GPU": 1.10,
    "A100_80GB_GPU": 1.60,
    "H100_GPU": 2.49,
    "H100_PCIe_GPU": 2.06,
    "H200_GPU": 3.99,
    "B200_GPU": 5.50,
    "L40S_48GB_GPU": 0.80,
    "V100_16GB_GPU": 0.65,
    "V100_32GB_GPU": 0.74,
    "MI300X": 2.49,
    "GH200_GPU": 3.49,
    "RTX4090_GPU": 0.44,
    "RTX3090_GPU": 0.28,
    "A10_GPU": 0.60,
}


def _extract_memory_gb(summary_table, unit: Unit) -> float:
    """Extract total memory (weights + KV cache) from GenZ summary_table.

    Args:
        summary_table: DataFrame from GenZ get_summary_table().
        unit: GenZ Unit instance (for unit labels).

    Returns:
        Memory in GB. Returns 0.0 if extraction fails.
    """
    try:
        weights_col = f"Total Weights ({unit.unit_mem})"
        kv_col = f"KV Cache ({unit.unit_mem})"
        weights_mb = float(summary_table[weights_col].values[0])
        kv_mb = float(summary_table[kv_col].values[0])
        return (weights_mb + kv_mb) / 1024.0  # MB -> GB
    except (KeyError, IndexError, ValueError):
        return 0.0


def _compute_cost_per_million_tokens(
    hardware: str, throughput_tps: float,
) -> float:
    """Compute $/million tokens from cloud pricing and throughput.

    Cost = (price_per_hour / tokens_per_hour) * 1_000_000

    Args:
        hardware: Hardware name for pricing lookup.
        throughput_tps: Tokens per second throughput.

    Returns:
        Cost in $/million tokens. Returns 0.0 if pricing unavailable.
    """
    price = _CLOUD_PRICING_PER_GPU_HOUR.get(hardware, 0.0)
    if price <= 0 or throughput_tps <= 0:
        return 0.0
    tokens_per_hour = throughput_tps * 3600
    return (price / tokens_per_hour) * 1_000_000


def _estimate_power_w(hardware: str, tdp_override: float = 0.0) -> float:
    """Estimate power draw from hardware TDP.

    Uses the hardware config's TDP if available, otherwise falls back
    to tdp_override. Applies a utilization factor (75% of TDP for
    typical inference workloads) rather than reporting raw TDP.

    Args:
        hardware: Hardware name for TDP lookup.
        tdp_override: Override TDP in watts (used for HardwareSpec evaluation).

    Returns:
        Estimated power in watts.
    """
    _TDP_MAP = {
        "A100_40GB_GPU": 250, "A100_80GB_GPU": 300, "H100_GPU": 700,
        "H100_PCIe_GPU": 350, "H200_GPU": 700, "B200_GPU": 1000,
        "L40S_48GB_GPU": 350, "V100_16GB_GPU": 250, "V100_32GB_GPU": 300,
        "MI300X": 750, "GH200_GPU": 900, "RTX4090_GPU": 450,
        "RTX3090_GPU": 350, "A10_GPU": 150,
    }
    tdp = _TDP_MAP.get(hardware, tdp_override)
    if tdp <= 0:
        return 0.0
    return tdp * 0.75  # Typical inference utilization factor


class BudSimEvaluator:
    """Wraps BudSim's GenZ analytical engine as a fitness function.

    This evaluator translates ServingConfig / HardwareSpec dataclasses
    into GenZ API calls and returns standardized EvalResult objects with
    real metrics extracted from GenZ: throughput, latency, memory, power,
    and cost.

    All GenZ exceptions are caught and returned as infeasible results
    so that optimization loops never crash on invalid configurations.
    """

    def __init__(self):
        self._unit = Unit()

    def evaluate_config(
        self,
        config: ServingConfig,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> EvalResult:
        """Evaluate a serving configuration using a named hardware platform.

        Runs GenZ prefill + decode modeling and extracts real metrics:
        throughput, latency, memory footprint, estimated power, and
        cost per million tokens.

        Args:
            config: ServingConfig with model name, hardware name, parallelism,
                    batch size, and precision settings.
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (generation) tokens.

        Returns:
            EvalResult with throughput, latency, memory, power, cost, and
            feasibility. If GenZ raises any exception (e.g. model not found,
            memory overflow), returns an infeasible EvalResult.
        """
        try:
            prefill = prefill_moddeling(
                model=config.model,
                batch_size=config.batch_size,
                input_tokens=input_tokens,
                system_name=config.hardware,
                bits=config.precision,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel,
            )
            decode = decode_moddeling(
                model=config.model,
                batch_size=config.batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=config.hardware,
                bits=config.precision,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel,
            )

            memory_gb = _extract_memory_gb(prefill.summary_table, self._unit)
            token_tps = decode.Throughput_tokens_per_sec
            cost = _compute_cost_per_million_tokens(config.hardware, token_tps)
            power = _estimate_power_w(config.hardware)
            num_devices = config.tensor_parallel * config.pipeline_parallel

            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=token_tps,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
                memory_gb=memory_gb,
                power_w=power * num_devices,
                cost_per_million_tokens=cost * num_devices,
                feasible=True,
                config={
                    "model": config.model,
                    "hardware": config.hardware,
                    "tensor_parallel": config.tensor_parallel,
                    "pipeline_parallel": config.pipeline_parallel,
                    "batch_size": config.batch_size,
                    "precision": config.precision,
                },
            )
        except Exception as e:
            warnings.warn(f"BudSim evaluation failed for {config}: {e}")
            return EvalResult(feasible=False, config={"error": str(e)})

    def evaluate_hardware(
        self,
        hw_spec: HardwareSpec,
        model: str = "meta-llama/Meta-Llama-3.1-8B",
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ) -> EvalResult:
        """Evaluate a hypothetical hardware specification against a model.

        Constructs a GenZ System object from the raw HardwareSpec parameters
        and runs the prefill + decode modeling pipeline.

        Args:
            hw_spec: HardwareSpec with FLOPS, memory bandwidth, memory size, etc.
            model: Model identifier (HuggingFace-style or GenZ model name).
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (generation) tokens.
            batch_size: Batch size for inference.
            precision: Weight/compute precision (e.g. "bf16", "fp8").
            tensor_parallel: Tensor parallelism degree.
            pipeline_parallel: Pipeline parallelism degree.

        Returns:
            EvalResult with throughput, latency, memory, power, cost, and
            feasibility fields.
        """
        from llm_memory_calculator.genz.system import System

        sys_kwargs = hw_spec.to_system_kwargs()
        sys_kwargs["bits"] = precision
        try:
            system = System(**sys_kwargs)
            prefill = prefill_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                system_name=system,
                bits=precision,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
            )
            decode = decode_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=system,
                bits=precision,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
            )

            memory_gb = _extract_memory_gb(prefill.summary_table, self._unit)
            token_tps = decode.Throughput_tokens_per_sec
            power = hw_spec.tdp_watts * 0.75  # Inference utilization factor

            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=token_tps,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
                memory_gb=memory_gb,
                power_w=power,
                cost_per_million_tokens=0.0,  # No cloud pricing for hypothetical HW
                feasible=True,
                config={
                    "flops_tflops": hw_spec.flops_tflops,
                    "offchip_mem_bw_gbps": hw_spec.offchip_mem_bw_gbps,
                    "off_chip_mem_size_gb": hw_spec.off_chip_mem_size_gb,
                    "tdp_watts": hw_spec.tdp_watts,
                    "estimated_cost_usd": hw_spec.estimated_cost_usd,
                },
            )
        except Exception as e:
            warnings.warn(f"Hardware evaluation failed: {e}")
            return EvalResult(feasible=False, config={"error": str(e)})
