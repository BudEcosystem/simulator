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
from llm_memory_calculator.genz.system import System

from .types import ServingConfig, HardwareSpec, EvalResult


class BudSimEvaluator:
    """Wraps BudSim's GenZ analytical engine as a fitness function.

    This evaluator translates ServingConfig / HardwareSpec dataclasses
    into GenZ API calls and returns standardized EvalResult objects.
    All GenZ exceptions are caught and returned as infeasible results
    so that optimization loops never crash on invalid configurations.
    """

    def evaluate_config(
        self,
        config: ServingConfig,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> EvalResult:
        """Evaluate a serving configuration using a named hardware platform.

        Args:
            config: ServingConfig with model name, hardware name, parallelism,
                    batch size, and precision settings.
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (generation) tokens.

        Returns:
            EvalResult with throughput, latency, and feasibility fields.
            If the GenZ engine raises any exception (e.g. model not found,
            memory overflow), returns an infeasible EvalResult with the
            error captured in config["error"].
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
            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=decode.Throughput_tokens_per_sec,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
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

        Constructs a System object from the raw HardwareSpec parameters
        and runs the GenZ prefill + decode modeling pipeline.

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
            EvalResult with throughput, latency, power, and feasibility fields.
            If the GenZ engine raises any exception, returns an infeasible
            EvalResult with the error captured in config["error"].
        """
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
            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=decode.Throughput_tokens_per_sec,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
                power_w=hw_spec.tdp_watts,
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
