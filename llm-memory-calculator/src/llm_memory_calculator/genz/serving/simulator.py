"""Main serving simulation orchestrator.

Combines workload generation, batch scheduling, memory management,
SLO tracking, and power modeling into an event-driven simulation loop.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import heapq
import warnings

from .constants import (
    MemoryTier, NS_PER_MS, NS_PER_S, GB_TO_BYTES,
    DEFAULT_BLOCK_SIZE, DEFAULT_PRECISION_BYTES,
)
from .request import Request, Batch
from .slo_tracker import SLOTargets, SLOTracker
from .memory_model import MemoryModel, MemoryTierConfig
from .power_model import PowerModel
from .workload import WorkloadConfig, WorkloadGenerator
from .batch_scheduler import SchedulerConfig, BatchScheduler


# Event types for the simulation event queue
_EVENT_ARRIVAL = 0
_EVENT_BATCH_COMPLETE = 1


@dataclass
class ServingSimulationResult:
    """Complete results from a serving simulation run."""
    total_duration_ms: float = 0.0
    total_requests_completed: int = 0
    total_requests_failed: int = 0
    overall_throughput_rps: float = 0.0
    overall_token_throughput_tps: float = 0.0
    slo_summary: Dict[str, Any] = field(default_factory=dict)
    power_summary: Dict[str, Any] = field(default_factory=dict)
    memory_peak: Dict[str, Any] = field(default_factory=dict)
    time_series: Dict[str, List] = field(default_factory=lambda: {
        "time_ms": [], "batch_size": [], "throughput_rps": [],
        "queue_depth": [], "memory_util": [],
    })
    per_request_metrics: Dict[str, float] = field(default_factory=dict)
    raw_requests: List[Dict] = field(default_factory=list)


class ServingSimulator:
    """Event-driven serving simulation using GenZ analytical engine.

    Orchestrates workload generation, memory management, batch scheduling,
    SLO tracking, and power modeling to simulate LLM serving behavior.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ):
        self._model = model
        self._hardware = hardware
        self._precision = precision
        self._tensor_parallel = tensor_parallel
        self._pipeline_parallel = pipeline_parallel

    def simulate(
        self,
        workload_config: Optional[WorkloadConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        slo_targets: Optional[SLOTargets] = None,
        memory_tiers: Optional[List[MemoryTierConfig]] = None,
        enable_power_tracking: bool = False,
    ) -> ServingSimulationResult:
        """Run a full serving simulation.

        Args:
            workload_config: Workload generation parameters. Uses defaults if None.
            scheduler_config: Batch scheduling parameters. Uses defaults if None.
            slo_targets: SLO targets for quality tracking. Uses defaults if None.
            memory_tiers: Memory tier configs. Auto-derived from hardware if None.
            enable_power_tracking: Whether to track power/energy consumption.

        Returns:
            ServingSimulationResult with aggregate and per-request metrics.
        """
        workload_config = workload_config or WorkloadConfig(model=self._model)
        scheduler_config = scheduler_config or SchedulerConfig()
        slo_targets = slo_targets or SLOTargets()

        # Generate workload
        wl_config = WorkloadConfig(
            arrival_rate_rps=workload_config.arrival_rate_rps,
            arrival_pattern=workload_config.arrival_pattern,
            num_requests=workload_config.num_requests,
            input_length_distribution=workload_config.input_length_distribution,
            output_length_distribution=workload_config.output_length_distribution,
            model=self._model,
            random_seed=workload_config.random_seed,
            gamma_shape=workload_config.gamma_shape,
            burst_period_s=workload_config.burst_period_s,
            burst_amplitude=workload_config.burst_amplitude,
        )
        generator = WorkloadGenerator(wl_config)
        requests = generator.generate()

        # Initialize memory model
        model_config = self._get_model_config()
        if memory_tiers is None:
            memory_tiers = self._default_memory_tiers()
        memory_model = MemoryModel(
            model_config=model_config,
            tier_configs=memory_tiers,
            tensor_parallel=self._tensor_parallel,
        )

        # Load model weights
        weight_bytes = self._estimate_weight_bytes(model_config)
        try:
            memory_model.load_weights(weight_bytes)
        except Exception:
            warnings.warn("Could not load full model weights into memory tier.")

        # Initialize scheduler
        scheduler = BatchScheduler(
            model=self._model,
            hardware=self._hardware,
            memory_model=memory_model,
            config=scheduler_config,
            precision=self._precision,
            tensor_parallel=self._tensor_parallel,
            pipeline_parallel=self._pipeline_parallel,
        )

        # Initialize trackers
        slo_tracker = SLOTracker(slo_targets)
        power_model = None
        if enable_power_tracking:
            from llm_memory_calculator.hardware.configs import is_cpu_hardware
            if is_cpu_hardware(self._hardware):
                power_model = PowerModel.from_cpu_hardware(self._hardware)
            else:
                power_model = PowerModel.from_hardware_name(self._hardware)

        # Run event-driven simulation
        result = self._run_simulation_loop(
            requests, scheduler, memory_model, slo_tracker, power_model,
        )
        return result

    def sweep(
        self,
        parameter: str,
        values: List,
        **kwargs,
    ) -> Dict[str, List]:
        """Sweep a parameter over multiple values, running simulation for each.

        Args:
            parameter: Name of the parameter to sweep (e.g., "arrival_rate_rps",
                      "max_batch_size", "tensor_parallel").
            values: List of values to test.
            **kwargs: Base simulation parameters.

        Returns:
            Dict with parameter values and corresponding result metrics.
        """
        results = {
            "parameter_values": list(values),
            "throughput_rps": [],
            "token_throughput_tps": [],
            "ttft_p50_ms": [],
            "ttft_p99_ms": [],
            "tpot_p50_ms": [],
            "goodput": [],
            "total_duration_ms": [],
        }

        workload_config = kwargs.get("workload_config", WorkloadConfig(model=self._model))
        scheduler_config = kwargs.get("scheduler_config", SchedulerConfig())

        for val in values:
            # Apply parameter override
            wl = dataclasses.replace(workload_config, model=self._model)
            sc = dataclasses.replace(scheduler_config)

            if parameter == "arrival_rate_rps":
                wl.arrival_rate_rps = val
            elif parameter == "max_batch_size":
                sc.max_batch_size = val
            elif parameter == "max_num_batched_tokens":
                sc.max_num_batched_tokens = val
            elif parameter == "num_requests":
                wl.num_requests = val
            else:
                warnings.warn(f"Unknown sweep parameter: {parameter}")

            sim_result = self.simulate(
                workload_config=wl,
                scheduler_config=sc,
                slo_targets=kwargs.get("slo_targets"),
                memory_tiers=kwargs.get("memory_tiers"),
                enable_power_tracking=kwargs.get("enable_power_tracking", False),
            )

            results["throughput_rps"].append(sim_result.overall_throughput_rps)
            results["token_throughput_tps"].append(sim_result.overall_token_throughput_tps)
            results["goodput"].append(sim_result.slo_summary.get("goodput", 0))
            results["total_duration_ms"].append(sim_result.total_duration_ms)

            ttft_pcts = sim_result.slo_summary.get("ttft", {}).get("percentiles_ns", {})
            tpot_pcts = sim_result.slo_summary.get("tpot", {}).get("percentiles_ns", {})
            results["ttft_p50_ms"].append(ttft_pcts.get("p50", 0) / NS_PER_MS)
            results["ttft_p99_ms"].append(ttft_pcts.get("p99", 0) / NS_PER_MS)
            results["tpot_p50_ms"].append(tpot_pcts.get("p50", 0) / NS_PER_MS)

        return results

    def _run_simulation_loop(
        self,
        requests: List[Request],
        scheduler: BatchScheduler,
        memory_model: MemoryModel,
        slo_tracker: SLOTracker,
        power_model: Optional[PowerModel],
    ) -> ServingSimulationResult:
        """Event-driven simulation loop."""
        result = ServingSimulationResult()

        # Build event queue: (time_ns, event_type, data)
        event_queue: List[Tuple[int, int, Any]] = []
        for req in requests:
            heapq.heappush(event_queue, (req.arrival_time_ns, _EVENT_ARRIVAL, req))

        completed_requests: List[Request] = []
        current_time_ns = 0
        last_batch_end_ns = 0
        last_power_calc_ns = 0
        total_tokens_generated = 0
        peak_memory: Dict[str, int] = {}
        batch_in_flight = False
        time_series_interval_ns = 100 * NS_PER_MS  # Sample every 100ms
        next_sample_ns = time_series_interval_ns

        # Safety limit to prevent infinite loops
        max_events = len(requests) * 500 + 1000

        event_count = 0
        while event_queue and event_count < max_events:
            event_count += 1
            event_time, event_type, event_data = heapq.heappop(event_queue)
            current_time_ns = event_time

            if event_type == _EVENT_ARRIVAL:
                req = event_data
                scheduler.add_request(req)

            elif event_type == _EVENT_BATCH_COMPLETE:
                batch = event_data
                batch_in_flight = False

                # Complete the batch — decode requests get a token
                newly_completed = scheduler.complete_batch(batch, current_time_ns)

                for req in newly_completed:
                    slo_tracker.record_completed_request(req)
                    completed_requests.append(req)
                    total_tokens_generated += req.tokens_generated

                # Track power for batch duration using full component model
                if power_model and batch:
                    batch_latency_ns = current_time_ns - last_batch_end_ns
                    if batch_latency_ns > 0:
                        # Use estimate_from_simulation_result for multi-component tracking
                        batch_latency_ms = batch_latency_ns / NS_PER_MS
                        batch_util = len(batch.requests) / max(scheduler._config.max_batch_size, 1)
                        power_model.estimate_from_simulation_result(
                            latency_ms=batch_latency_ms,
                            compute_util=min(batch_util, 1.0),
                            mem_util=min(batch_util * 0.8, 1.0),
                            num_accel=self._tensor_parallel,
                        )
                    # Track idle time since last batch
                    if last_batch_end_ns > 0 and last_batch_end_ns < current_time_ns:
                        power_model.add_accelerator_standby_energy(
                            current_time_ns, last_batch_end_ns, last_power_calc_ns,
                        )
                    last_power_calc_ns = current_time_ns

                last_batch_end_ns = current_time_ns

            # Try scheduling a new batch if nothing in flight
            if not batch_in_flight:
                batch = scheduler.schedule(current_time_ns)
                if batch is not None:
                    batch_in_flight = True
                    latency_ms = scheduler.estimate_batch_latency_ms(batch)
                    completion_time = current_time_ns + int(latency_ms * NS_PER_MS)
                    heapq.heappush(event_queue, (completion_time, _EVENT_BATCH_COMPLETE, batch))

            # Record time series data
            while current_time_ns >= next_sample_ns:
                snap = memory_model.memory_snapshot()
                result.time_series["time_ms"].append(next_sample_ns / NS_PER_MS)
                result.time_series["batch_size"].append(scheduler.inflight_count + scheduler.pending_count)
                result.time_series["queue_depth"].append(scheduler.pending_count)

                # Use primary tier (DEVICE_HBM for GPU, DEVICE_DRAM for CPU)
                primary_tier_name = memory_model.primary_tier.value
                primary_snap = snap.get(primary_tier_name, {})
                result.time_series["memory_util"].append(primary_snap.get("utilization", 0))

                duration_so_far_s = next_sample_ns / NS_PER_S
                if duration_so_far_s > 0:
                    result.time_series["throughput_rps"].append(
                        len(completed_requests) / duration_so_far_s
                    )
                else:
                    result.time_series["throughput_rps"].append(0)

                # Track peak memory
                for tier_name, tier_data in snap.items():
                    used = tier_data.get("used_bytes", 0)
                    if tier_name not in peak_memory or used > peak_memory[tier_name]:
                        peak_memory[tier_name] = used

                next_sample_ns += time_series_interval_ns

        # Compute final results
        if current_time_ns > 0:
            duration_s = current_time_ns / NS_PER_S
            result.total_duration_ms = current_time_ns / NS_PER_MS
            result.total_requests_completed = len(completed_requests)
            result.overall_throughput_rps = len(completed_requests) / duration_s if duration_s > 0 else 0
            result.overall_token_throughput_tps = total_tokens_generated / duration_s if duration_s > 0 else 0

        result.slo_summary = slo_tracker.summary()

        if power_model:
            result.power_summary = power_model.summary(
                current_time_ns, total_tokens=total_tokens_generated
            )

        result.memory_peak = {
            tier: {"peak_bytes": peak_bytes}
            for tier, peak_bytes in peak_memory.items()
        }

        # Per-request detail
        for req in completed_requests:
            result.raw_requests.append({
                "request_id": req.request_id,
                "input_tokens": req.input_tokens,
                "tokens_generated": req.tokens_generated,
                "ttft_ns": req.ttft_ns,
                "tpot_ns": req.tpot_ns,
                "e2e_ns": req.e2e_ns,
                "queuing_delay_ns": req.queuing_delay_ns,
            })

        # Per-request aggregate metrics
        if completed_requests:
            ttft_vals = [r.ttft_ns for r in completed_requests]
            tpot_vals = [r.tpot_ns for r in completed_requests]
            e2e_vals = [r.e2e_ns for r in completed_requests]
            result.per_request_metrics = {
                "ttft_avg_ms": sum(ttft_vals) / len(ttft_vals) / NS_PER_MS,
                "tpot_avg_ms": sum(tpot_vals) / len(tpot_vals) / NS_PER_MS,
                "e2e_avg_ms": sum(e2e_vals) / len(e2e_vals) / NS_PER_MS,
            }

        return result

    def _get_model_config(self):
        """Get model config from GenZ registry."""
        try:
            from llm_memory_calculator.genz.Models import get_configs
            return get_configs(self._model)
        except Exception:
            # Fallback to a default config for unknown models
            from types import SimpleNamespace
            warnings.warn(f"Model {self._model} not found. Using default 7B config.")
            return SimpleNamespace(
                num_key_value_heads=32, head_dim=128,
                num_decoder_layers=32, num_attention_heads=32,
                hidden_size=4096, vocab_size=32000,
                intermediate_size=11008,
            )

    def _default_memory_tiers(self) -> List[MemoryTierConfig]:
        """Create default memory tier config from hardware."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS, is_cpu_hardware
        try:
            if self._hardware in HARDWARE_CONFIGS:
                hw = HARDWARE_CONFIGS[self._hardware]
                mem_gb = hw.get("Memory_size", 80)
                mem_bw = hw.get("Memory_BW", 2000)
            else:
                # Try CPU_PRESETS for memory specs
                from llm_memory_calculator.genz.cpu.cpu_configs import CPU_PRESETS
                preset_key = next(
                    (k for k in CPU_PRESETS if k.lower() == self._hardware.lower()),
                    None,
                )
                if preset_key:
                    params = CPU_PRESETS[preset_key]['base_params']
                    mem_gb = params.get('off_chip_mem_size', 512 * 1024) / 1024
                    mem_bw = params.get('offchip_mem_bw', 200)
                else:
                    mem_gb, mem_bw = 80, 2000
        except Exception:
            mem_gb, mem_bw = 80, 2000

        tier = (
            MemoryTier.DEVICE_DRAM
            if is_cpu_hardware(self._hardware)
            else MemoryTier.DEVICE_HBM
        )

        return [MemoryTierConfig(
            tier=tier,
            capacity_bytes=int(mem_gb * GB_TO_BYTES),
            bandwidth_gbps=float(mem_bw),
        )]

    def _estimate_weight_bytes(self, model_config) -> int:
        """Estimate model weight size in bytes."""
        hidden = getattr(model_config, "hidden_size", 4096)
        layers = getattr(model_config, "num_decoder_layers", 32)
        intermediate = getattr(model_config, "intermediate_size", 11008)
        vocab = getattr(model_config, "vocab_size", 32000)
        n_heads = getattr(model_config, "num_attention_heads", 32)
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            head_dim = hidden // n_heads

        # Approximate parameter count
        # Attention: Q, K, V, O projections per layer (GQA-aware)
        num_kv_heads = getattr(model_config, 'num_key_value_heads', n_heads)
        # Q and O projections use full n_heads; K and V use num_kv_heads
        attn_params = (
            2 * hidden * n_heads * head_dim +        # Q and O projections
            2 * hidden * num_kv_heads * head_dim      # K and V projections
        ) * layers
        # FFN: gate + up + down per layer
        ffn_params = 3 * hidden * intermediate * layers
        # Embeddings
        embed_params = vocab * hidden
        total_params = attn_params + ffn_params + embed_params

        # 2 bytes per parameter for bf16
        precision_bytes = 2 if self._precision in ("bf16", "fp16") else (
            1 if self._precision in ("int8", "fp8") else 4
        )
        return total_params * precision_bytes // self._tensor_parallel
