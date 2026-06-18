"""Prefill-decode disaggregation analysis for LLM serving."""
from typing import Dict, Any, Optional, List
import warnings

from .constants import NS_PER_MS, NS_PER_S, GB_TO_BYTES
from .workload import WorkloadConfig
from .batch_scheduler import SchedulerConfig


class DisaggregationAnalyzer:
    """Analyze prefill-decode disaggregation strategies.

    In disaggregated serving, prefill and decode run on separate
    instance pools, connected by KV cache transfer over network.
    """

    def __init__(self, model: str, hardware: str, precision: str = "bf16"):
        self._model = model
        self._hardware = hardware
        self._precision = precision
        self.last_b_eff: int = 1  # SV6 diagnostic

    def _build_b_eff(self, context: int, tensor_parallel: int = 1,
                     max_batch_size: Optional[int] = None) -> int:
        """SV6: memory-budget-bounded steady-state concurrency per instance.

        Same derivation as ``ClusterAnalyzer._build_b_eff``:
        ``floor((primary_capacity - weight_bytes) / (bytes_per_token_kv * context))`` capped by
        ``max_batch_size``; returns 1 when a MemoryModel cannot be built. Applied IDENTICALLY to
        the prefill and decode pools, so it cancels in the bottleneck/utilization inequalities.
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
                return 1
            primary_capacity = int(mem_gb * GB_TO_BYTES)
            mm = MemoryModel(
                model_config=model_config,
                tier_configs=[MemoryTierConfig(MemoryTier.DEVICE_HBM, primary_capacity, 0.0)],
                tensor_parallel=tensor_parallel,
            )
            weight_bytes = estimate_weight_bytes(model_config, self._precision, tensor_parallel)
            footprint = mm.bytes_per_token_kv * max(context, 1)
            if footprint <= 0:
                return 1
            avail = primary_capacity - weight_bytes
            if avail <= 0:
                return 1
            return max(1, min(max_batch_size, avail // footprint))
        except Exception:
            return 1

    def analyze(
        self,
        prefill_instances: int,
        decode_instances: int,
        workload_config: Optional[WorkloadConfig] = None,
        kv_transfer_bw_gbps: float = 100.0,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze a specific P:D disaggregation configuration.

        Args:
            prefill_instances: Number of prefill-dedicated instances.
            decode_instances: Number of decode-dedicated instances.
            workload_config: Workload for analysis (optional).
            kv_transfer_bw_gbps: Network bandwidth for KV transfer (GB/s).
            input_tokens: Representative input length.
            output_tokens: Representative output length.

        Returns:
            Dictionary with throughput, latency, bottleneck, and utilization metrics.
        """
        prefill_lat, decode_lat = self._get_latencies(input_tokens, output_tokens)

        # SV6: each instance serves B_eff requests concurrently (continuous batching), not one
        # serial request. The SAME B_eff applies to the prefill and decode pools (both bound by
        # the per-instance memory budget over the same context), so it scales system throughput
        # but CANCELS in the bottleneck ratio and the utilization fractions below.
        b_eff = self._build_b_eff(input_tokens + output_tokens, tensor_parallel=1)
        self.last_b_eff = b_eff

        # Compute throughput first (needed for M/M/1 queuing model)
        prefill_throughput_rps = (
            b_eff * prefill_instances * (1000.0 / prefill_lat) if prefill_lat > 0 else 0
        )
        decode_time_per_req = decode_lat * output_tokens
        decode_throughput_rps = (
            b_eff * decode_instances * (1000.0 / decode_time_per_req)
            if decode_time_per_req > 0
            else 0
        )
        system_throughput_rps = min(prefill_throughput_rps, decode_throughput_rps)

        # Compute KV transfer with M/M/1 queuing contention
        kv_transfer_bytes = self._estimate_kv_bytes(input_tokens, output_tokens)
        serialization_time_s = kv_transfer_bytes / (kv_transfer_bw_gbps * 1e9)
        serialization_time_ms = serialization_time_s * 1000
        # Link utilization = arrival_rate * service_time (capped for stability)
        effective_arrival_rate = system_throughput_rps if system_throughput_rps > 0 else 1.0
        link_utilization = min(effective_arrival_rate * serialization_time_s, 0.95)
        queuing_factor = 1.0 / (1.0 - link_utilization) if link_utilization < 1.0 else 20.0
        propagation_latency_ms = 0.01  # 10 microseconds intra-rack
        kv_transfer_time_ms = serialization_time_ms * queuing_factor + propagation_latency_ms

        effective_ttft_ms = prefill_lat + kv_transfer_time_ms

        bottleneck = "prefill" if prefill_throughput_rps < decode_throughput_rps else "decode"

        return {
            "prefill_instances": prefill_instances,
            "decode_instances": decode_instances,
            "total_instances": prefill_instances + decode_instances,
            "prefill_latency_ms": prefill_lat,
            "decode_latency_ms": decode_lat,
            "kv_transfer_time_ms": kv_transfer_time_ms,
            "kv_transfer_bytes": kv_transfer_bytes,
            "effective_ttft_ms": effective_ttft_ms,
            "prefill_throughput_rps": prefill_throughput_rps,
            "decode_throughput_rps": decode_throughput_rps,
            "system_throughput_rps": system_throughput_rps,
            "bottleneck": bottleneck,
            "prefill_utilization": (
                system_throughput_rps / prefill_throughput_rps
                if prefill_throughput_rps > 0
                else 0
            ),
            "decode_utilization": (
                system_throughput_rps / decode_throughput_rps
                if decode_throughput_rps > 0
                else 0
            ),
            "link_utilization": link_utilization,
            "queuing_factor": queuing_factor,
        }

    def find_optimal_ratio(
        self,
        total_instances: int,
        workload_config: Optional[WorkloadConfig] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
        kv_transfer_bw_gbps: float = 100.0,
    ) -> Dict[str, Any]:
        """Find the optimal prefill:decode ratio for a given total instance count.

        Sweeps all valid splits from 1:N-1 to N-1:1 and returns the
        configuration that maximises system throughput.
        """
        best_ratio = None
        best_throughput = 0.0
        all_ratios: List[Dict[str, Any]] = []

        for prefill_n in range(1, total_instances):
            decode_n = total_instances - prefill_n
            result = self.analyze(
                prefill_instances=prefill_n,
                decode_instances=decode_n,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                kv_transfer_bw_gbps=kv_transfer_bw_gbps,
            )
            all_ratios.append(
                {
                    "prefill": prefill_n,
                    "decode": decode_n,
                    "ratio": f"{prefill_n}:{decode_n}",
                    "throughput_rps": result["system_throughput_rps"],
                    "bottleneck": result["bottleneck"],
                }
            )
            if result["system_throughput_rps"] > best_throughput:
                best_throughput = result["system_throughput_rps"]
                best_ratio = result

        return {
            "optimal": best_ratio,
            "all_ratios": all_ratios,
            "total_instances": total_instances,
        }

    def compare_colocated_vs_disaggregated(
        self,
        total_instances: int,
        workload_config: Optional[WorkloadConfig] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
        kv_transfer_bw_gbps: float = 100.0,
    ) -> Dict[str, Any]:
        """Compare colocated (standard) vs disaggregated serving.

        Colocated: every instance runs both prefill and decode.
        Disaggregated: instances are split into prefill-only and decode-only pools.
        """
        prefill_lat, decode_lat = self._get_latencies(input_tokens, output_tokens)

        # SV6: colocated instances also serve B_eff requests concurrently (continuous batching).
        b_eff = self._build_b_eff(input_tokens + output_tokens, tensor_parallel=1)
        self.last_b_eff = b_eff
        time_per_request = prefill_lat + decode_lat * output_tokens
        colocated_throughput = (
            b_eff * total_instances * (1000.0 / time_per_request)
            if time_per_request > 0
            else 0
        )
        colocated_ttft = prefill_lat  # No KV transfer in colocated

        # Disaggregated: find optimal split
        optimal = self.find_optimal_ratio(
            total_instances,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            kv_transfer_bw_gbps=kv_transfer_bw_gbps,
        )
        disagg_result = optimal.get("optimal")
        disagg_throughput = (
            disagg_result["system_throughput_rps"] if disagg_result else 0
        )
        disagg_ttft = (
            disagg_result["effective_ttft_ms"] if disagg_result else 0
        )

        speedup = (
            disagg_throughput / colocated_throughput
            if colocated_throughput > 0
            else 0
        )

        return {
            "colocated": {
                "instances": total_instances,
                "throughput_rps": colocated_throughput,
                "ttft_ms": colocated_ttft,
                "time_per_request_ms": time_per_request,
            },
            "disaggregated": {
                "prefill_instances": (
                    disagg_result["prefill_instances"] if disagg_result else 0
                ),
                "decode_instances": (
                    disagg_result["decode_instances"] if disagg_result else 0
                ),
                "throughput_rps": disagg_throughput,
                "ttft_ms": disagg_ttft,
            },
            "speedup": speedup,
            "recommendation": (
                "disaggregated" if speedup > 1.1 else "colocated"
            ),
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
            # silently produced physics-free latencies; re-raise so the failure is visible.
            raise

    def _estimate_kv_bytes(self, input_tokens: int, output_tokens: int = 0) -> int:
        """Estimate KV cache bytes for transfer using the canonical per-token KV formula."""
        from llm_memory_calculator.genz.Models import get_configs

        # No magic fallback: if the model config cannot be resolved we cannot derive the
        # per-token KV footprint honestly, so let the exception propagate (ROOT-B).
        config = get_configs(self._model)
        num_kv = getattr(config, "num_key_value_heads", None)
        if num_kv is None:
            num_kv = getattr(config, "num_attention_heads", 32)
        hdim = getattr(config, "head_dim", None)
        if hdim is None:
            hidden = getattr(config, "hidden_size", 4096)
            n_heads = getattr(config, "num_attention_heads", 32)
            hdim = hidden // n_heads
        layers = getattr(config, "num_decoder_layers", 32)
        # K + V, bf16 = 2 bytes per element (canonical bytes_per_token_kv, TP=1 for transfer).
        bytes_per_token = 2 * num_kv * hdim * layers * 2
        return bytes_per_token * (input_tokens + output_tokens)
