"""Cluster-level analysis for multi-instance LLM serving."""
import math
from typing import Dict, Any, List, Optional
import warnings

from .constants import NS_PER_MS, NS_PER_S, GB_TO_BYTES
from .workload import WorkloadConfig
from .batch_scheduler import SchedulerConfig


class ClusterAnalyzer:
    """Analyze cluster-level scaling and parallelism for LLM serving.

    Models how throughput scales with the number of independent serving
    instances, and finds optimal tensor-parallel / pipeline-parallel
    configurations for a given device count.
    """

    def __init__(self, model: str, hardware: str, precision: str = "bf16"):
        self._model = model
        self._hardware = hardware
        self._precision = precision
        # SV6 diagnostics: the memory-budget-bounded steady-state concurrency and the weight
        # figure it was derived from (exposed for tests / introspection).
        self.last_b_eff: int = 1
        self.last_weight_bytes: int = 0

    def _build_b_eff(self, context: int, tensor_parallel: int = 1,
                     max_batch_size: Optional[int] = None) -> int:
        """SV6: steady-state continuous-batching concurrency the memory budget pins.

        ``B_eff = floor((primary_tier_capacity - weight_bytes) / (bytes_per_token_kv * context))``,
        capped by ``SchedulerConfig.max_batch_size``. ``bytes_per_token_kv`` is the canonical
        ``MemoryModel`` formula; ``weight_bytes`` the single-source estimator. Returns 1 (the old
        serial behaviour) when a MemoryModel cannot be built (unknown model / mocked config), so
        no magic number is ever substituted and mocked tests are preserved.
        """
        if max_batch_size is None:
            max_batch_size = SchedulerConfig().max_batch_size
        try:
            from llm_memory_calculator.genz.Models import get_configs
            from .memory_model import MemoryModel, MemoryTierConfig, estimate_weight_bytes
            from .constants import MemoryTier
            from llm_memory_calculator.hardware import get_hardware_config

            model_config = get_configs(self._model)
            hw = get_hardware_config(self._hardware) or {}
            mem_gb = hw.get("Memory_size")
            if not mem_gb:
                return 1  # honest: no device capacity -> cannot derive a budget, stay serial
            primary_capacity = int(mem_gb * GB_TO_BYTES)

            mm = MemoryModel(
                model_config=model_config,
                tier_configs=[MemoryTierConfig(MemoryTier.DEVICE_HBM, primary_capacity, 0.0)],
                tensor_parallel=tensor_parallel,
            )
            weight_bytes = estimate_weight_bytes(model_config, self._precision, tensor_parallel)
            self.last_weight_bytes = weight_bytes
            footprint = mm.bytes_per_token_kv * max(context, 1)
            if footprint <= 0:
                return 1
            avail = primary_capacity - weight_bytes
            if avail <= 0:
                return 1  # weights alone fill the tier -> at most one in-flight request
            return max(1, min(max_batch_size, avail // footprint))
        except Exception:
            return 1

    def analyze_scaling(
        self,
        instance_counts: List[int],
        workload: Optional[WorkloadConfig] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze throughput scaling across instance counts.

        Models sub-linear scaling for data-parallel instances that share
        memory bus and PCIe bandwidth. Efficiency degrades logarithmically
        with the number of instances.
        """
        prefill_lat, decode_lat = self._get_latencies(input_tokens, output_tokens)
        time_per_request = prefill_lat + decode_lat * output_tokens
        # SV6: an instance serves B_eff requests concurrently (continuous batching), not 1
        # serial request. B_eff is pinned by the memory budget over the representative context.
        b_eff = self._build_b_eff(input_tokens + output_tokens, tensor_parallel=1)
        self.last_b_eff = b_eff
        single_instance_tps = (
            b_eff * 1000.0 / time_per_request if time_per_request > 0 else 0
        )

        results = []
        for n in instance_counts:
            # Data parallel scaling efficiency: independent instances share memory bus
            # and PCIe bandwidth. Efficiency degrades logarithmically with instance count.
            # Model: efficiency = 1 / (1 + 0.02 * log2(n))
            # Yields: n=1->100%, n=2->98%, n=4->96%, n=8->94%
            if n <= 1:
                efficiency = 1.0
            else:
                efficiency = 1.0 / (1.0 + 0.02 * math.log2(n))
            throughput = single_instance_tps * n * efficiency
            results.append(
                {
                    "instances": n,
                    "throughput_rps": throughput,
                    "tokens_per_second": throughput * (input_tokens + output_tokens),
                    "scaling_efficiency": efficiency,
                    "time_per_request_ms": time_per_request,
                }
            )

        return {
            "model": self._model,
            "hardware": self._hardware,
            "single_instance_throughput_rps": single_instance_tps,
            "single_instance_latency_ms": time_per_request,
            "scaling_results": results,
        }

    def optimize_parallelism(
        self,
        num_devices: int,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Find optimal parallelism strategy for a given number of devices.

        Tests combinations of TP and PP that divide num_devices evenly
        and returns the configuration that maximises total throughput.

        Note: TP scaling efficiency (communication overhead from AllReduce
        collectives across tensor-parallel ranks) is already modeled by the
        GenZ engine when computing prefill/decode latencies at each TP value.
        The latencies returned by prefill_moddeling/decode_moddeling naturally
        include the cost of inter-device communication, so no additional
        efficiency factor is applied here.
        """
        configs: List[Dict[str, Any]] = []

        from llm_memory_calculator.hardware.configs import is_cpu_hardware
        cpu_mode = is_cpu_hardware(self._hardware)
        # CPU systems: TP limited to socket count (1-2), PP limited to 1
        tp_values = [1, 2] if cpu_mode else [1, 2, 4, 8]
        pp_values = [1] if cpu_mode else [1, 2, 4]

        for tp in tp_values:
            if tp > num_devices:
                continue
            for pp in pp_values:
                if tp * pp > num_devices:
                    continue
                num_instances = num_devices // (tp * pp)
                if num_instances < 1:
                    continue

                try:
                    from llm_memory_calculator.genz.serving import get_modeling_functions
                    prefill_fn, decode_fn = get_modeling_functions(self._hardware)

                    prefill = prefill_fn(
                        model=self._model,
                        batch_size=1,
                        input_tokens=input_tokens,
                        system_name=self._hardware,
                        bits=self._precision,
                        tensor_parallel=tp,
                        pipeline_parallel=pp,
                    )
                    decode = decode_fn(
                        model=self._model,
                        batch_size=1,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        system_name=self._hardware,
                        bits=self._precision,
                        tensor_parallel=tp,
                        pipeline_parallel=pp,
                    )

                    prefill_lat = prefill.Latency
                    decode_lat = decode.Latency
                except Exception:
                    # SV6 / ROOT-B: no magic fallback. If the engine cannot model this config,
                    # skip it honestly rather than fabricating input_tokens*0.01 / 0.5 latencies.
                    continue

                # SV6: each instance serves B_eff requests concurrently; the budget shrinks with
                # tp (weights /tp, but kv/token also /tp), evaluated over this config's context.
                b_eff = self._build_b_eff(input_tokens + output_tokens, tensor_parallel=tp)
                time_per_request = prefill_lat + decode_lat * output_tokens
                instance_throughput = (
                    b_eff * 1000.0 / time_per_request if time_per_request > 0 else 0
                )
                total_throughput = instance_throughput * num_instances

                configs.append(
                    {
                        "tensor_parallel": tp,
                        "pipeline_parallel": pp,
                        "num_instances": num_instances,
                        "devices_per_instance": tp * pp,
                        "prefill_latency_ms": prefill_lat,
                        "decode_latency_ms": decode_lat,
                        "time_per_request_ms": time_per_request,
                        "instance_throughput_rps": instance_throughput,
                        "total_throughput_rps": total_throughput,
                    }
                )

        if not configs:
            return {
                "optimal": None,
                "all_configs": [],
                "num_devices": num_devices,
            }

        best = max(configs, key=lambda c: c["total_throughput_rps"])

        return {
            "optimal": best,
            "all_configs": configs,
            "num_devices": num_devices,
        }

    def _get_latencies(self, input_tokens: int, output_tokens: int):
        """Get prefill and per-token decode latency from GenZ."""
        try:
            from llm_memory_calculator.genz.serving import get_modeling_functions
            prefill_fn, decode_fn = get_modeling_functions(self._hardware)

            prefill = prefill_fn(
                model=self._model,
                batch_size=1,
                input_tokens=input_tokens,
                system_name=self._hardware,
                bits=self._precision,
            )
            decode = decode_fn(
                model=self._model,
                batch_size=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=self._hardware,
                bits=self._precision,
            )
            return prefill.Latency, decode.Latency
        except Exception:
            # SV6 / ROOT-B: fail honestly. The fabricated (input_tokens*0.01, 0.5) heuristic
            # silently substituted physics-free numbers; re-raise so the caller sees the failure.
            raise
