"""
Training API Router.

Provides endpoints for training memory estimation, cluster recommendations,
and hardware fit checking using the training calculator module.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# Import training calculator module
try:
    from llm_memory_calculator.training import (
        TrainingMemoryCalculator,
        TrainingClusterSelector,
        TrainingTimeEstimator,
    )
    from llm_memory_calculator import HuggingFaceConfigLoader
    TRAINING_MODULE_AVAILABLE = True
except ImportError:
    TRAINING_MODULE_AVAILABLE = False


router = APIRouter(prefix="/api/simulator", tags=["training"])


# ==================== Request Schemas ====================

class TrainingEstimateRequest(BaseModel):
    """Request model for training memory estimation."""
    model: str = Field(
        ...,
        description="HuggingFace model ID or path",
        examples=["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.1"]
    )
    method: str = Field(
        "lora",
        description="Training method: full, lora, qlora, freeze, dora, pissa",
        examples=["lora", "full", "qlora"]
    )
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    seq_length: int = Field(2048, ge=1, description="Sequence length")
    precision: str = Field("bf16", description="Weight precision: fp32, bf16, fp16, int8, int4")
    optimizer: str = Field(
        "adamw",
        description="Optimizer type: adamw, adamw_8bit, sgd, galore, apollo, adafactor"
    )
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    lora_rank: int = Field(16, ge=1, description="LoRA rank (if using LoRA method)")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha (if using LoRA method)")
    freeze_layers: int = Field(0, ge=0, description="Number of layers to freeze (if using freeze method)")
    deepspeed_stage: Optional[str] = Field(None, description="DeepSpeed ZeRO stage: zero2, zero3")
    tensor_parallel: int = Field(1, ge=1, description="Tensor parallelism degree")
    data_parallel: int = Field(1, ge=1, description="Data parallelism degree")
    framework_overhead_percent: float = Field(10.0, ge=0, description="Framework overhead percentage")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        valid_methods = {'full', 'lora', 'qlora', 'freeze', 'dora', 'pissa'}
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v.lower()

    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v):
        valid_precisions = {'fp32', 'bf16', 'fp16', 'int8', 'int4', 'nf4'}
        if v.lower() not in valid_precisions:
            raise ValueError(f"Precision must be one of {valid_precisions}")
        return v.lower()

    @field_validator('optimizer')
    @classmethod
    def validate_optimizer(cls, v):
        valid_optimizers = {'adamw', 'adam', 'adamw_8bit', 'sgd', 'galore', 'apollo', 'adafactor', 'lion'}
        if v.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")
        return v.lower()


class ClusterRecommendRequest(BaseModel):
    """Request model for cluster recommendations."""
    model: str = Field(..., description="HuggingFace model ID")
    method: str = Field("lora", description="Training method")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    seq_length: int = Field(2048, ge=1, description="Sequence length")
    precision: str = Field("bf16", description="Weight precision")
    optimizer: str = Field("adamw", description="Optimizer type")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    lora_rank: int = Field(16, ge=1, description="LoRA rank")
    deepspeed_stage: Optional[str] = Field(None, description="DeepSpeed ZeRO stage")
    prefer_cost: bool = Field(True, description="Sort by cost (True) or speed (False)")
    max_budget_per_hour: Optional[float] = Field(None, ge=0, description="Maximum hourly budget in USD")
    available_hardware: Optional[List[str]] = Field(None, description="Limit to specific hardware types")
    max_gpus: int = Field(32, ge=1, description="Maximum number of GPUs to consider")


class CheckFitRequest(BaseModel):
    """Request model for checking if training fits in a cluster."""
    model: str = Field(..., description="HuggingFace model ID")
    method: str = Field("lora", description="Training method")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    seq_length: int = Field(2048, ge=1, description="Sequence length")
    precision: str = Field("bf16", description="Weight precision")
    optimizer: str = Field("adamw", description="Optimizer type")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    lora_rank: int = Field(16, ge=1, description="LoRA rank")
    deepspeed_stage: Optional[str] = Field(None, description="DeepSpeed ZeRO stage")
    hardware: str = Field(..., description="Hardware type to check", examples=["A100_80GB_GPU", "H100_GPU"])
    num_gpus: int = Field(..., ge=1, description="Number of GPUs")


class TimeEstimateRequest(BaseModel):
    """Request model for training time estimation."""
    model: str = Field(..., description="HuggingFace model ID")
    dataset_tokens: int = Field(..., gt=0, description="Total tokens in dataset")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation: int = Field(4, ge=1, description="Gradient accumulation steps")
    epochs: float = Field(1.0, gt=0, description="Number of training epochs")
    hardware: str = Field(..., description="Hardware type", examples=["A100_80GB_GPU", "H100_GPU"])
    num_gpus: int = Field(1, ge=1, description="Number of GPUs")
    seq_length: int = Field(2048, ge=1, description="Sequence length")
    parallelism: Optional[Dict[str, int]] = Field(
        None,
        description="Parallelism strategy: {tp: X, pp: Y, dp: Z}"
    )


# ==================== Response Schemas ====================

class MemoryBreakdownResponse(BaseModel):
    """Memory breakdown by component."""
    weight_memory_gb: float = Field(..., description="Model weights memory in GB")
    gradient_memory_gb: float = Field(..., description="Gradient memory in GB")
    optimizer_memory_gb: float = Field(..., description="Optimizer state memory in GB")
    activation_memory_gb: float = Field(..., description="Activation memory in GB")
    total_memory_gb: float = Field(..., description="Total memory requirement in GB")


class TrainingEstimateResponse(BaseModel):
    """Response model for training memory estimation."""
    model: str = Field(..., description="Model identifier")
    method: str = Field(..., description="Training method")
    precision: str = Field(..., description="Weight precision")
    optimizer: str = Field(..., description="Optimizer type")
    batch_size: int = Field(..., description="Batch size")
    seq_length: int = Field(..., description="Sequence length")

    # Memory breakdown
    memory_breakdown: MemoryBreakdownResponse = Field(..., description="Memory breakdown by component")

    # Parameter counts
    total_params: int = Field(..., description="Total model parameters")
    trainable_params: int = Field(..., description="Trainable parameters")
    trainable_percent: float = Field(..., description="Percentage of trainable parameters")

    # Configuration
    gradient_checkpointing: bool = Field(..., description="Whether gradient checkpointing is enabled")
    lora_rank: Optional[int] = Field(None, description="LoRA rank if using LoRA")
    deepspeed_stage: Optional[str] = Field(None, description="DeepSpeed ZeRO stage")

    # Fit recommendations
    fits_single_gpu_24gb: bool = Field(..., description="Fits on single 24GB GPU")
    fits_single_gpu_40gb: bool = Field(..., description="Fits on single 40GB GPU")
    fits_single_gpu_80gb: bool = Field(..., description="Fits on single 80GB GPU")
    min_gpus_80gb: int = Field(..., description="Minimum A100 80GB GPUs needed")


class ParallelismStrategy(BaseModel):
    """Parallelism strategy configuration."""
    tp: int = Field(..., description="Tensor parallelism degree")
    pp: int = Field(..., description="Pipeline parallelism degree")
    dp: int = Field(..., description="Data parallelism degree")


class ClusterRecommendationResponse(BaseModel):
    """Single cluster recommendation."""
    hardware_name: str = Field(..., description="Hardware type name")
    nodes_required: int = Field(..., description="Number of nodes required")
    gpus_per_node: int = Field(..., description="GPUs per node")
    total_gpus: int = Field(..., description="Total number of GPUs")
    memory_per_gpu_gb: float = Field(..., description="Memory per GPU in GB")
    parallelism: ParallelismStrategy = Field(..., description="Parallelism strategy")
    estimated_throughput_tps: float = Field(..., description="Estimated tokens per second")
    estimated_cost_per_hour: float = Field(..., description="Estimated cost per hour in USD")
    utilization_percent: float = Field(..., description="GPU memory utilization percentage")
    optimality: str = Field(..., description="Optimality rating: optimal, good, suboptimal")
    fits: bool = Field(..., description="Whether training fits in this cluster")


class ClusterRecommendResponse(BaseModel):
    """Response model for cluster recommendations."""
    recommendations: List[ClusterRecommendationResponse] = Field(
        ...,
        description="List of cluster recommendations sorted by preference"
    )
    total_options: int = Field(..., description="Total number of viable options")


class CheckFitResponse(BaseModel):
    """Response model for fit check."""
    fits: bool = Field(..., description="Whether training fits in the specified cluster")
    memory_per_gpu_gb: float = Field(..., description="Memory per GPU required")
    utilization_percent: float = Field(..., description="GPU memory utilization percentage")
    parallelism: Optional[ParallelismStrategy] = Field(None, description="Recommended parallelism strategy")
    reason: Optional[str] = Field(None, description="Reason if training doesn't fit")
    min_gpus_required: int = Field(..., description="Minimum GPUs required")
    estimated_cost_per_hour: float = Field(..., description="Estimated cost per hour in USD")


class TimeEstimateResponse(BaseModel):
    """Response model for training time estimation."""
    total_steps: int = Field(..., description="Total training steps")
    tokens_per_second: float = Field(..., description="Estimated tokens per second")
    estimated_hours: float = Field(..., description="Estimated training time in hours")
    estimated_cost: float = Field(..., description="Estimated total cost in USD")
    hardware: str = Field(..., description="Hardware type")
    num_gpus: int = Field(..., description="Number of GPUs")
    parallelism: Optional[ParallelismStrategy] = Field(None, description="Parallelism strategy")
    model_flops_utilization: float = Field(..., description="Model FLOPs Utilization (MFU)")


class HardwareProfileResponse(BaseModel):
    """Hardware profile information."""
    name: str = Field(..., description="Hardware name")
    type: str = Field(..., description="Hardware type (gpu, asic, accelerator)")
    manufacturer: Optional[str] = Field(None, description="Hardware manufacturer")
    memory_gb: float = Field(..., description="Memory size in GB")
    flops_tflops: float = Field(..., description="Peak FLOPs in TFLOPS")
    memory_bw_gbps: float = Field(..., description="Memory bandwidth in GB/s")
    cost_per_hour: float = Field(..., description="Cost per hour in USD")


class HardwareListResponse(BaseModel):
    """Response model for hardware list."""
    hardware: List[HardwareProfileResponse] = Field(..., description="List of available hardware")
    total_count: int = Field(..., description="Total number of hardware profiles")


# ==================== API Endpoints ====================

def _get_model_config(model_id: str) -> Dict[str, Any]:
    """Load model configuration from HuggingFace or cache."""
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    try:
        loader = HuggingFaceConfigLoader()
        return loader.get_model_config(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not load model config for {model_id}: {str(e)}"
        )


@router.post("/estimate-training", response_model=TrainingEstimateResponse)
async def estimate_training(request: TrainingEstimateRequest) -> TrainingEstimateResponse:
    """
    Estimate training memory requirements.

    Returns detailed memory breakdown including weights, gradients,
    optimizer states, and activations.
    """
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    # Load model config
    config = _get_model_config(request.model)

    # Calculate training memory
    calculator = TrainingMemoryCalculator()
    estimate = calculator.calculate_training_memory(
        config=config,
        batch_size=request.batch_size,
        seq_length=request.seq_length,
        precision=request.precision,
        method=request.method,
        optimizer=request.optimizer,
        gradient_checkpointing=request.gradient_checkpointing,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
        freeze_layers=request.freeze_layers,
        deepspeed_stage=request.deepspeed_stage,
        tensor_parallel=request.tensor_parallel,
        data_parallel=request.data_parallel,
        framework_overhead_percent=request.framework_overhead_percent,
    )

    # Calculate fit recommendations
    total_mem = estimate.total_memory_gb
    trainable_percent = (estimate.trainable_params / estimate.total_params * 100) if estimate.total_params > 0 else 0

    return TrainingEstimateResponse(
        model=request.model,
        method=request.method,
        precision=request.precision,
        optimizer=request.optimizer,
        batch_size=request.batch_size,
        seq_length=request.seq_length,
        memory_breakdown=MemoryBreakdownResponse(
            weight_memory_gb=estimate.weight_memory_gb,
            gradient_memory_gb=estimate.gradient_memory_gb,
            optimizer_memory_gb=estimate.optimizer_memory_gb,
            activation_memory_gb=estimate.activation_memory_gb,
            total_memory_gb=estimate.total_memory_gb,
        ),
        total_params=estimate.total_params,
        trainable_params=estimate.trainable_params,
        trainable_percent=round(trainable_percent, 2),
        gradient_checkpointing=estimate.gradient_checkpointing,
        lora_rank=estimate.lora_rank,
        deepspeed_stage=estimate.deepspeed_stage,
        fits_single_gpu_24gb=total_mem <= 24 * 0.9,
        fits_single_gpu_40gb=total_mem <= 40 * 0.9,
        fits_single_gpu_80gb=total_mem <= 80 * 0.9,
        min_gpus_80gb=max(1, int(total_mem / (80 * 0.9)) + 1) if total_mem > 80 * 0.9 else 1,
    )


@router.post("/recommend-cluster", response_model=ClusterRecommendResponse)
async def recommend_cluster(request: ClusterRecommendRequest) -> ClusterRecommendResponse:
    """
    Get cluster recommendations for training.

    Returns ranked list of cluster configurations sorted by cost or speed.
    """
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    # Load model config
    config = _get_model_config(request.model)

    # Calculate training memory estimate
    calculator = TrainingMemoryCalculator()
    estimate = calculator.calculate_training_memory(
        config=config,
        batch_size=request.batch_size,
        seq_length=request.seq_length,
        precision=request.precision,
        method=request.method,
        optimizer=request.optimizer,
        gradient_checkpointing=request.gradient_checkpointing,
        lora_rank=request.lora_rank,
        deepspeed_stage=request.deepspeed_stage,
    )

    # Get cluster recommendations
    selector = TrainingClusterSelector()
    recommendations = selector.recommend_clusters(
        training_estimate=estimate.to_dict(),
        prefer_cost=request.prefer_cost,
        max_budget_per_hour=request.max_budget_per_hour,
        available_hardware=request.available_hardware,
        max_gpus=request.max_gpus,
    )

    # Convert to response format
    rec_responses = []
    for rec in recommendations:
        rec_responses.append(ClusterRecommendationResponse(
            hardware_name=rec.hardware_name,
            nodes_required=rec.nodes_required,
            gpus_per_node=rec.gpus_per_node,
            total_gpus=rec.total_gpus,
            memory_per_gpu_gb=rec.memory_per_gpu_gb,
            parallelism=ParallelismStrategy(
                tp=rec.parallelism.get("tp", 1),
                pp=rec.parallelism.get("pp", 1),
                dp=rec.parallelism.get("dp", 1),
            ),
            estimated_throughput_tps=rec.estimated_throughput_tps,
            estimated_cost_per_hour=rec.estimated_cost_per_hour,
            utilization_percent=rec.utilization_percent,
            optimality=rec.optimality,
            fits=rec.fits,
        ))

    return ClusterRecommendResponse(
        recommendations=rec_responses,
        total_options=len(rec_responses),
    )


@router.post("/check-fit", response_model=CheckFitResponse)
async def check_fit(request: CheckFitRequest) -> CheckFitResponse:
    """
    Check if training fits in a specific cluster configuration.

    Returns fit status and recommended parallelism strategy.
    """
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    # Load model config
    config = _get_model_config(request.model)

    # Calculate training memory estimate
    calculator = TrainingMemoryCalculator()
    estimate = calculator.calculate_training_memory(
        config=config,
        batch_size=request.batch_size,
        seq_length=request.seq_length,
        precision=request.precision,
        method=request.method,
        optimizer=request.optimizer,
        gradient_checkpointing=request.gradient_checkpointing,
        lora_rank=request.lora_rank,
        deepspeed_stage=request.deepspeed_stage,
    )

    # Check fit
    selector = TrainingClusterSelector()
    result = selector.check_fit(
        training_estimate=estimate.to_dict(),
        hardware=request.hardware,
        num_gpus=request.num_gpus,
    )

    parallelism = None
    if result.parallelism:
        parallelism = ParallelismStrategy(
            tp=result.parallelism.get("tp", 1),
            pp=result.parallelism.get("pp", 1),
            dp=result.parallelism.get("dp", 1),
        )

    return CheckFitResponse(
        fits=result.fits,
        memory_per_gpu_gb=result.memory_per_gpu_gb,
        utilization_percent=result.utilization_percent,
        parallelism=parallelism,
        reason=result.reason,
        min_gpus_required=result.min_gpus_required,
        estimated_cost_per_hour=result.estimated_cost_per_hour,
    )


@router.post("/estimate-time", response_model=TimeEstimateResponse)
async def estimate_time(request: TimeEstimateRequest) -> TimeEstimateResponse:
    """
    Estimate training time and cost.

    Returns estimated training duration and total cost.
    """
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    # Load model config
    config = _get_model_config(request.model)

    # Estimate training time
    estimator = TrainingTimeEstimator()
    estimate = estimator.estimate_training_time(
        model_config=config,
        dataset_tokens=request.dataset_tokens,
        batch_size=request.batch_size,
        gradient_accumulation=request.gradient_accumulation,
        epochs=request.epochs,
        hardware=request.hardware,
        num_gpus=request.num_gpus,
        seq_length=request.seq_length,
        parallelism=request.parallelism,
    )

    parallelism = None
    if estimate.parallelism:
        parallelism = ParallelismStrategy(
            tp=estimate.parallelism.get("tp", 1),
            pp=estimate.parallelism.get("pp", 1),
            dp=estimate.parallelism.get("dp", 1),
        )

    return TimeEstimateResponse(
        total_steps=estimate.total_steps,
        tokens_per_second=estimate.tokens_per_second,
        estimated_hours=estimate.estimated_hours,
        estimated_cost=estimate.estimated_cost,
        hardware=estimate.hardware,
        num_gpus=estimate.num_gpus,
        parallelism=parallelism,
        model_flops_utilization=estimate.model_flops_utilization,
    )


@router.get("/hardware", response_model=HardwareListResponse)
async def list_hardware() -> HardwareListResponse:
    """
    List available hardware profiles.

    Returns all supported hardware configurations with specifications.
    """
    if not TRAINING_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training module not available. Please install llm-memory-calculator."
        )

    selector = TrainingClusterSelector()
    hardware_list = selector.list_available_hardware()

    profiles = []
    for hw in hardware_list:
        profiles.append(HardwareProfileResponse(
            name=hw["name"],
            type=hw.get("type", "gpu"),
            manufacturer=hw.get("manufacturer"),
            memory_gb=hw.get("memory_gb", 0),
            flops_tflops=hw.get("flops_tflops", 0),
            memory_bw_gbps=hw.get("memory_bw_gbps", 0),
            cost_per_hour=hw.get("cost_per_hour", 0),
        ))

    return HardwareListResponse(
        hardware=profiles,
        total_count=len(profiles),
    )
