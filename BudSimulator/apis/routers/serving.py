"""Serving simulation API endpoints (v2)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api/v2", tags=["serving"])


# === Pydantic Schemas ===

class MemoryTierRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'meta-llama/Meta-Llama-3.1-8B'")
    hardware: str = Field(..., description="Hardware name, e.g. 'A100_80GB_GPU'")
    precision_bytes: int = Field(2, description="Bytes per element (2=bf16, 1=int8)")
    tensor_parallel: int = Field(1, ge=1)
    block_size: int = Field(16, ge=1)
    host_ddr_gb: float = Field(0, ge=0, description="Host DDR capacity in GB")
    cxl_gb: float = Field(0, ge=0, description="CXL memory capacity in GB")
    nvme_gb: float = Field(0, ge=0, description="NVMe capacity in GB")

class MemoryTierResponse(BaseModel):
    model: str
    hardware: str
    bytes_per_token_kv: int
    bytes_per_block: int
    tiers: Dict[str, Any]

class PowerEstimateRequest(BaseModel):
    model: str = Field(..., description="Model name")
    hardware: str = Field(..., description="Hardware name")
    batch_size: int = Field(1, ge=1)
    input_tokens: int = Field(1024, ge=1)
    output_tokens: int = Field(128, ge=1)
    tensor_parallel: int = Field(1, ge=1)
    num_accelerators: int = Field(1, ge=1)
    precision: str = Field("bf16")

class PowerEstimateResponse(BaseModel):
    model: str
    hardware: str
    base_power_w: float
    estimated_power: Dict[str, Any]
    energy_summary: Dict[str, Any]


# === Endpoints ===

@router.post("/memory/tiers", response_model=MemoryTierResponse)
async def analyze_memory_tiers(req: MemoryTierRequest):
    """Multi-tier memory analysis for KV cache management."""
    try:
        from llm_memory_calculator.genz.Models import get_configs
        from llm_memory_calculator.genz.serving import (
            MemoryModel, MemoryTierConfig, MemoryTier, GB_TO_BYTES,
        )
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS

        model_config = get_configs(req.model)

        if req.hardware not in HARDWARE_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown hardware: {req.hardware}")
        hw = HARDWARE_CONFIGS[req.hardware]

        tier_configs = [
            MemoryTierConfig(
                tier=MemoryTier.DEVICE_HBM,
                capacity_bytes=int(hw['Memory_size'] * GB_TO_BYTES),
                bandwidth_gbps=float(hw['Memory_BW']),
            )
        ]
        if req.host_ddr_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.HOST_DDR,
                capacity_bytes=int(req.host_ddr_gb * GB_TO_BYTES),
            ))
        if req.cxl_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.CXL,
                capacity_bytes=int(req.cxl_gb * GB_TO_BYTES),
            ))
        if req.nvme_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.NVME,
                capacity_bytes=int(req.nvme_gb * GB_TO_BYTES),
            ))

        mm = MemoryModel(
            model_config=model_config,
            tier_configs=tier_configs,
            block_size=req.block_size,
            tensor_parallel=req.tensor_parallel,
            precision_bytes=req.precision_bytes,
        )

        return MemoryTierResponse(
            model=req.model,
            hardware=req.hardware,
            bytes_per_token_kv=mm.bytes_per_token_kv,
            bytes_per_block=mm.bytes_per_block,
            tiers=mm.memory_snapshot(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/power/estimate", response_model=PowerEstimateResponse)
async def estimate_power(req: PowerEstimateRequest):
    """7-component power breakdown estimation."""
    try:
        from llm_memory_calculator.genz.serving import PowerModel
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS

        if req.hardware not in HARDWARE_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown hardware: {req.hardware}")

        pm = PowerModel.from_hardware_name(req.hardware, num_accel=req.num_accelerators)

        # Estimate power using analytical mode
        compute_util = min(0.95, 0.3 + 0.05 * req.batch_size)
        mem_util = min(0.95, 0.4 + 0.03 * req.batch_size)

        # Rough latency estimate: prefill + decode
        prefill_latency_ms = req.input_tokens * 0.01 * req.batch_size / req.tensor_parallel
        decode_latency_ms = req.output_tokens * 0.5 / req.tensor_parallel
        total_latency_ms = prefill_latency_ms + decode_latency_ms

        estimated = pm.estimate_from_simulation_result(
            latency_ms=total_latency_ms,
            compute_util=compute_util,
            mem_util=mem_util,
            num_accel=req.num_accelerators,
        )

        total_tokens = req.batch_size * (req.input_tokens + req.output_tokens)
        duration_ns = total_latency_ms * 1_000_000

        return PowerEstimateResponse(
            model=req.model,
            hardware=req.hardware,
            base_power_w=pm.get_base_power_w(),
            estimated_power=estimated,
            energy_summary=pm.summary(duration_ns, total_tokens),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Phase 2: Serving Simulation Endpoints ===

class ServingSimulationRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'meta-llama/Meta-Llama-3.1-8B'")
    hardware: str = Field(..., description="Hardware name, e.g. 'A100_80GB_GPU'")
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    pipeline_parallel: int = Field(1, ge=1)
    arrival_rate_rps: float = Field(10.0, gt=0)
    arrival_pattern: str = Field("poisson")
    num_requests: int = Field(100, ge=1, le=10000)
    input_length_mean: int = Field(512, ge=1)
    output_length_mean: int = Field(128, ge=1)
    random_seed: int = Field(42)
    max_batch_size: int = Field(256, ge=1)
    max_num_batched_tokens: int = Field(8192, ge=1)
    enable_chunked_prefill: bool = Field(False)
    ttft_target_ms: float = Field(500.0, gt=0)
    tpot_target_ms: float = Field(100.0, gt=0)
    e2e_target_ms: float = Field(30000.0, gt=0)
    enable_power_tracking: bool = Field(False)

class ServingSimulationResponse(BaseModel):
    model: str
    hardware: str
    total_duration_ms: float
    total_requests_completed: int
    overall_throughput_rps: float
    overall_token_throughput_tps: float
    slo_summary: Dict[str, Any]
    power_summary: Dict[str, Any]
    memory_peak: Dict[str, Any]
    per_request_metrics: Dict[str, float]

class BatchAnalysisRequest(BaseModel):
    model: str = Field(..., description="Model name")
    hardware: str = Field(..., description="Hardware name")
    precision: str = Field("bf16")
    tensor_parallel: int = Field(1, ge=1)
    batch_size: int = Field(1, ge=1)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)

class BatchAnalysisResponse(BaseModel):
    model: str
    hardware: str
    batch_size: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float


@router.post("/simulate/serving", response_model=ServingSimulationResponse)
async def simulate_serving(req: ServingSimulationRequest):
    """Full serving simulation with workload generation, scheduling, and SLO tracking."""
    try:
        from llm_memory_calculator.genz.serving import (
            ServingSimulator, WorkloadConfig, SchedulerConfig, SLOTargets,
        )

        simulator = ServingSimulator(
            model=req.model, hardware=req.hardware, precision=req.precision,
            tensor_parallel=req.tensor_parallel, pipeline_parallel=req.pipeline_parallel,
        )

        workload = WorkloadConfig(
            arrival_rate_rps=req.arrival_rate_rps, arrival_pattern=req.arrival_pattern,
            num_requests=req.num_requests,
            input_length_distribution={
                "dist": "lognormal", "mean": req.input_length_mean,
                "std": max(req.input_length_mean // 3, 1), "min": 32,
                "max": req.input_length_mean * 4,
            },
            output_length_distribution={
                "dist": "lognormal", "mean": req.output_length_mean,
                "std": max(req.output_length_mean // 3, 1), "min": 8,
                "max": req.output_length_mean * 4,
            },
            random_seed=req.random_seed,
        )

        scheduler = SchedulerConfig(
            max_batch_size=req.max_batch_size,
            max_num_batched_tokens=req.max_num_batched_tokens,
            enable_chunked_prefill=req.enable_chunked_prefill,
        )

        slo_targets = SLOTargets.from_ms(
            ttft_ms=req.ttft_target_ms, tpot_ms=req.tpot_target_ms, e2e_ms=req.e2e_target_ms,
        )

        result = simulator.simulate(
            workload_config=workload, scheduler_config=scheduler,
            slo_targets=slo_targets, enable_power_tracking=req.enable_power_tracking,
        )

        return ServingSimulationResponse(
            model=req.model, hardware=req.hardware,
            total_duration_ms=result.total_duration_ms,
            total_requests_completed=result.total_requests_completed,
            overall_throughput_rps=result.overall_throughput_rps,
            overall_token_throughput_tps=result.overall_token_throughput_tps,
            slo_summary=result.slo_summary, power_summary=result.power_summary,
            memory_peak=result.memory_peak, per_request_metrics=result.per_request_metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(req: BatchAnalysisRequest):
    """Single batch analysis using GenZ analytical engine."""
    try:
        from llm_memory_calculator.genz import prefill_moddeling, decode_moddeling

        prefill_result = prefill_moddeling(
            model=req.model, batch_size=req.batch_size, input_tokens=req.input_tokens,
            system_name=req.hardware, bits=req.precision, tensor_parallel=req.tensor_parallel,
        )
        prefill_latency = prefill_result.Latency

        decode_result = decode_moddeling(
            model=req.model, batch_size=req.batch_size, input_tokens=req.input_tokens,
            output_tokens=req.output_tokens, system_name=req.hardware,
            bits=req.precision, tensor_parallel=req.tensor_parallel,
        )
        decode_latency = decode_result.Latency

        total_latency = prefill_latency + decode_latency * req.output_tokens
        total_tokens = req.batch_size * (req.input_tokens + req.output_tokens)
        tokens_per_sec = total_tokens / (total_latency / 1000.0) if total_latency > 0 else 0

        return BatchAnalysisResponse(
            model=req.model, hardware=req.hardware, batch_size=req.batch_size,
            prefill_latency_ms=prefill_latency, decode_latency_ms=decode_latency,
            total_latency_ms=total_latency, tokens_per_second=tokens_per_sec,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Phase 3: Configuration Optimization Endpoints ===

class OptimizeConfigRequest(BaseModel):
    model: str = Field(..., description="Model name")
    hardware: str = Field(..., description="Hardware name")
    target: str = Field("throughput", description="Optimization target: throughput, latency, efficiency")
    num_devices: int = Field(1, ge=1)
    budget: int = Field(20, ge=1, le=200)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)
    max_ttft_ms: float = Field(float('inf'), gt=0)
    max_tpot_ms: float = Field(float('inf'), gt=0)

class ParetoRequest(BaseModel):
    model: str = Field(...)
    hardware: str = Field(...)
    objectives: List[str] = Field(default=["throughput", "latency"])
    num_devices: int = Field(1, ge=1)
    budget: int = Field(30, ge=1, le=200)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)

class SensitivityRequest(BaseModel):
    model: str = Field(...)
    hardware: str = Field(...)
    target: str = Field("throughput")
    num_devices: int = Field(1, ge=1)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)


@router.post("/optimize/config")
async def optimize_config(req: OptimizeConfigRequest):
    """Find optimal serving configuration for a single objective."""
    try:
        from llm_memory_calculator.genz.serving import ConfigOptimizer, OptimizationConstraints
        optimizer = ConfigOptimizer(req.model, req.hardware, num_devices=req.num_devices)
        constraints = OptimizationConstraints(
            max_ttft_ms=req.max_ttft_ms, max_tpot_ms=req.max_tpot_ms,
            max_total_devices=req.num_devices,
        )
        return optimizer.optimize(
            target=req.target, constraints=constraints, budget=req.budget,
            input_tokens=req.input_tokens, output_tokens=req.output_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/pareto")
async def pareto_optimize(req: ParetoRequest):
    """Multi-objective Pareto frontier optimization."""
    try:
        from llm_memory_calculator.genz.serving import ConfigOptimizer
        optimizer = ConfigOptimizer(req.model, req.hardware, num_devices=req.num_devices)
        return optimizer.pareto_optimize(
            objectives=req.objectives, budget=req.budget,
            input_tokens=req.input_tokens, output_tokens=req.output_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/sensitivity")
async def analyze_sensitivity(req: SensitivityRequest):
    """Parameter sensitivity analysis."""
    try:
        from llm_memory_calculator.genz.serving import ConfigOptimizer
        optimizer = ConfigOptimizer(req.model, req.hardware, num_devices=req.num_devices)
        return optimizer.analyze_sensitivity(
            target=req.target, input_tokens=req.input_tokens, output_tokens=req.output_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Phase 4: Advanced Feature Endpoints ===

class CacheAnalyzeRequest(BaseModel):
    model: str = Field(...)
    hardware: str = Field(...)
    cache_capacity_gb: float = Field(10.0, gt=0)
    num_requests: int = Field(100, ge=1, le=1000)
    input_length_mean: int = Field(512, ge=1)

class DisaggregationRequest(BaseModel):
    model: str = Field(...)
    hardware: str = Field(...)
    total_instances: int = Field(8, ge=2)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)
    kv_transfer_bw_gbps: float = Field(100.0, gt=0)

class ClusterTopologyRequest(BaseModel):
    model: str = Field(...)
    hardware: str = Field(...)
    num_devices: int = Field(8, ge=1)
    input_tokens: int = Field(512, ge=1)
    output_tokens: int = Field(128, ge=1)


@router.post("/cache/analyze")
async def analyze_cache(req: CacheAnalyzeRequest):
    """Prefix cache effectiveness analysis."""
    try:
        from llm_memory_calculator.genz.serving import PrefixCacheAnalyzer, Request
        from llm_memory_calculator.genz.Models import get_configs
        import random

        config = get_configs(req.model)
        analyzer = PrefixCacheAnalyzer(config, cache_capacity_gb=req.cache_capacity_gb)

        # Generate synthetic workload
        workload = []
        for i in range(req.num_requests):
            workload.append(Request(
                request_id=i, model=req.model,
                input_tokens=req.input_length_mean,
                max_output_tokens=128, arrival_time_ns=i * 100_000_000,
            ))

        return analyzer.analyze(workload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster/disaggregate")
async def disaggregate(req: DisaggregationRequest):
    """Prefill-decode disaggregation analysis."""
    try:
        from llm_memory_calculator.genz.serving import DisaggregationAnalyzer
        analyzer = DisaggregationAnalyzer(req.model, req.hardware)
        return analyzer.compare_colocated_vs_disaggregated(
            total_instances=req.total_instances,
            input_tokens=req.input_tokens, output_tokens=req.output_tokens,
            kv_transfer_bw_gbps=req.kv_transfer_bw_gbps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster/topology")
async def cluster_topology(req: ClusterTopologyRequest):
    """Cluster topology and parallelism optimization."""
    try:
        from llm_memory_calculator.genz.serving import ClusterAnalyzer
        analyzer = ClusterAnalyzer(req.model, req.hardware)
        return analyzer.optimize_parallelism(
            num_devices=req.num_devices,
            input_tokens=req.input_tokens, output_tokens=req.output_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
