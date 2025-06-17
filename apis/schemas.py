"""
Pydantic models for API request and response schemas.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator


# Request Models

class ValidateModelRequest(BaseModel):
    """Request model for validating a model URL."""
    model_url: str = Field(..., description="HuggingFace model URL or ID", example="meta-llama/Llama-2-7b-hf")


class CalculateMemoryRequest(BaseModel):
    """Request model for calculating memory requirements."""
    model_id: str = Field(..., description="HuggingFace model ID", example="meta-llama/Llama-2-7b-hf")
    precision: str = Field("fp16", description="Precision format", example="fp16")
    batch_size: int = Field(1, ge=1, description="Batch size", example=1)
    seq_length: int = Field(2048, ge=1, description="Sequence length", example=2048)
    num_images: int = Field(0, ge=0, description="Number of images (for multimodal models)", example=0)
    include_gradients: bool = Field(False, description="Include gradient memory (for training)", example=False)
    decode_length: int = Field(0, ge=0, description="Number of tokens to generate", example=0)
    
    @validator('precision')
    def validate_precision(cls, v):
        valid_precisions = ['fp32', 'fp16', 'bf16', 'int8', 'int4', 'fp8']
        if v not in valid_precisions:
            raise ValueError(f"Precision must be one of {valid_precisions}")
        return v


class CompareModelConfig(BaseModel):
    """Configuration for a single model in comparison."""
    model_id: str = Field(..., description="HuggingFace model ID", example="meta-llama/Llama-2-7b-hf")
    precision: str = Field("fp16", description="Precision format", example="fp16")
    batch_size: int = Field(1, ge=1, description="Batch size", example=1)
    seq_length: int = Field(2048, ge=1, description="Sequence length", example=2048)


class CompareModelsRequest(BaseModel):
    """Request model for comparing multiple models."""
    models: List[CompareModelConfig] = Field(..., description="List of models to compare", min_items=1)


class AnalyzeModelRequest(BaseModel):
    """Request model for analyzing model efficiency."""
    model_id: str = Field(..., description="HuggingFace model ID", example="meta-llama/Llama-2-7b-hf")
    precision: str = Field("fp16", description="Precision format", example="fp16")
    batch_size: int = Field(1, ge=1, description="Batch size", example=1)
    sequence_lengths: List[int] = Field(
        [1024, 4096, 16384, 32768],
        description="List of sequence lengths to analyze",
        example=[1024, 4096, 16384, 32768]
    )


class AddModelFromHFRequest(BaseModel):
    """Request model for adding a model from HuggingFace."""
    model_uri: str = Field(..., description="HuggingFace model URI", example="meta-llama/Llama-2-7b-hf")
    auto_import: bool = Field(True, description="Automatically import if not found", example=True)


class AddModelFromConfigRequest(BaseModel):
    """Request model for adding a model from configuration."""
    model_id: str = Field(..., description="Unique model identifier", example="custom-model-7b")
    config: Dict[str, Any] = Field(..., description="Model configuration dictionary")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ConfigSubmitRequest(BaseModel):
    """Request model for submitting a gated model configuration."""
    model_uri: str = Field(..., description="HuggingFace model URI", example="meta-llama/Llama-3.1-8B-Instruct")
    config: Dict[str, Any] = Field(..., description="Model configuration from config.json")


class ConfigSubmitResponse(BaseModel):
    """Response model for config submission."""
    success: bool = Field(..., description="Whether the config was saved successfully")
    message: str = Field(..., description="Success or error message")
    model_id: str = Field(..., description="The model ID")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation results if successful")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    missing_fields: Optional[List[str]] = Field(None, description="List of missing required fields")


# Response Models

class ValidateModelResponse(BaseModel):
    """Response model for model validation."""
    valid: bool = Field(..., description="Whether the model URL is valid")
    error: Optional[str] = Field(None, description="Error message if invalid")
    error_code: Optional[str] = Field(None, description="Error code for specific error types")
    model_id: Optional[str] = Field(None, description="Model ID if validation succeeded or gated")
    requires_config: bool = Field(False, description="Whether model requires manual config")
    config_submission_url: Optional[str] = Field(None, description="URL for config submission")


class ModelConfig(BaseModel):
    """Model configuration details."""
    hidden_size: int = Field(..., description="Hidden dimension size", example=4096)
    num_hidden_layers: int = Field(..., description="Number of hidden layers", example=32)
    num_attention_heads: int = Field(..., description="Number of attention heads", example=32)
    num_key_value_heads: Optional[int] = Field(None, description="Number of KV heads (for GQA)", example=8)
    intermediate_size: int = Field(..., description="FFN intermediate size", example=11008)
    vocab_size: int = Field(..., description="Vocabulary size", example=32000)
    max_position_embeddings: int = Field(..., description="Maximum sequence length", example=4096)
    activation_function: str = Field(..., description="Activation function", example="silu")


class ModelMetadata(BaseModel):
    """Model metadata from HuggingFace."""
    downloads: Optional[int] = Field(None, description="Number of downloads", example=1500000)
    likes: Optional[int] = Field(None, description="Number of likes", example=25000)
    size_gb: Optional[float] = Field(None, description="Model size in GB", example=13.5)
    tags: List[str] = Field(default_factory=list, description="Model tags", example=["text-generation", "llama"])


class ModelAnalysisEval(BaseModel):
    """Evaluation score for a model."""
    name: str = Field(..., description="Benchmark name", example="MMLU")
    score: float = Field(..., description="Benchmark score", example=66.3)


class ModelAnalysis(BaseModel):
    """Detailed analysis of a model."""
    description: str = Field(..., description="Model description")
    advantages: List[str] = Field(default_factory=list, description="List of advantages")
    disadvantages: List[str] = Field(default_factory=list, description="List of disadvantages")
    usecases: List[str] = Field(default_factory=list, description="List of use cases")
    evals: List[ModelAnalysisEval] = Field(default_factory=list, description="List of evaluation scores")


class ModelConfigResponse(BaseModel):
    """Response model for model configuration."""
    model_id: str = Field(..., description="Model identifier", example="meta-llama/Llama-2-7b-hf")
    model_type: str = Field(..., description="Model type", example="decoder-only")
    attention_type: Optional[str] = Field(None, description="Attention mechanism type", example="gqa")
    parameter_count: Optional[int] = Field(None, description="Total parameter count", example=7000000000)
    architecture: Optional[str] = Field(None, description="Model architecture", example="LlamaForCausalLM")
    logo: Optional[str] = Field(None, description="URL to the model's logo")
    model_analysis: Optional[ModelAnalysis] = Field(None, description="Detailed model analysis")
    config: ModelConfig = Field(..., description="Model configuration details")
    metadata: ModelMetadata = Field(..., description="Model metadata")


class MemoryBreakdown(BaseModel):
    """Memory breakdown by component."""
    weight_memory_gb: float = Field(..., description="Model weights memory", example=13.5)
    kv_cache_gb: float = Field(..., description="KV cache memory", example=0.5)
    activation_memory_gb: float = Field(..., description="Activation memory", example=2.1)
    state_memory_gb: float = Field(..., description="State memory (for SSM models)", example=0.0)
    image_memory_gb: float = Field(..., description="Image memory (for multimodal)", example=0.0)
    extra_work_gb: float = Field(..., description="Extra work buffer memory", example=0.5)


class MemoryRecommendations(BaseModel):
    """GPU memory recommendations."""
    recommended_gpu_memory_gb: int = Field(..., description="Recommended GPU memory", example=24)
    can_fit_24gb_gpu: bool = Field(..., description="Can fit in 24GB GPU", example=True)
    can_fit_80gb_gpu: bool = Field(..., description="Can fit in 80GB GPU", example=True)
    min_gpu_memory_gb: float = Field(..., description="Minimum required GPU memory", example=16.6)


class CalculateMemoryResponse(BaseModel):
    """Response model for memory calculation."""
    model_type: str = Field(..., description="Model type", example="decoder-only")
    attention_type: Optional[str] = Field(None, description="Attention mechanism type", example="gqa")
    precision: str = Field(..., description="Precision format", example="fp16")
    parameter_count: Optional[int] = Field(None, description="Total parameter count", example=7000000000)
    memory_breakdown: MemoryBreakdown = Field(..., description="Memory breakdown by component")
    total_memory_gb: float = Field(..., description="Total memory requirement", example=16.6)
    recommendations: MemoryRecommendations = Field(..., description="GPU memory recommendations")


class ModelComparison(BaseModel):
    """Single model comparison result."""
    model_id: str = Field(..., description="Model identifier", example="meta-llama/Llama-2-7b-hf")
    model_name: str = Field(..., description="Model display name", example="Llama 2 7B")
    total_memory_gb: float = Field(..., description="Total memory requirement", example=16.6)
    memory_breakdown: MemoryBreakdown = Field(..., description="Memory breakdown by component")
    recommendations: MemoryRecommendations = Field(..., description="GPU memory recommendations")


class CompareModelsResponse(BaseModel):
    """Response model for model comparison."""
    comparisons: List[ModelComparison] = Field(..., description="List of model comparisons")


class SequenceAnalysis(BaseModel):
    """Analysis for a specific sequence length."""
    total_memory_gb: float = Field(..., description="Total memory requirement", example=14.2)
    kv_cache_gb: float = Field(..., description="KV cache memory", example=0.25)
    kv_cache_percent: float = Field(..., description="KV cache percentage of total", example=1.8)


class AnalysisInsights(BaseModel):
    """Analysis insights and recommendations."""
    memory_per_token_bytes: int = Field(..., description="Memory per token in bytes", example=256)
    efficiency_rating: Literal["low", "medium", "high"] = Field(..., description="Efficiency rating", example="high")
    recommendations: List[str] = Field(..., description="List of recommendations")


class AnalyzeModelResponse(BaseModel):
    """Response model for model analysis."""
    model_id: str = Field(..., description="Model identifier", example="meta-llama/Llama-2-7b-hf")
    attention_type: Optional[str] = Field(None, description="Attention mechanism type", example="gqa")
    analysis: Dict[str, SequenceAnalysis] = Field(..., description="Analysis by sequence length")
    insights: AnalysisInsights = Field(..., description="Analysis insights and recommendations")


class PopularModel(BaseModel):
    """Popular model information."""
    model_id: str = Field(..., description="Model identifier", example="meta-llama/Llama-2-7b-hf")
    name: str = Field(..., description="Model display name", example="Llama 2 7B")
    parameters: str = Field(..., description="Parameter count display", example="7B")
    model_type: str = Field(..., description="Model type", example="decoder-only")
    attention_type: Optional[str] = Field(None, description="Attention mechanism type", example="gqa")
    downloads: int = Field(..., description="Number of downloads", example=1500000)
    likes: int = Field(..., description="Number of likes", example=25000)
    description: str = Field(..., description="Model description", example="High-performance language model")
    logo: Optional[str] = Field(None, description="URL to the model's logo")


class PopularModelsResponse(BaseModel):
    """Response model for popular models."""
    models: List[PopularModel] = Field(..., description="List of popular models")


class ModelSummary(BaseModel):
    """Summary information for a model."""
    model_id: str = Field(..., description="Model identifier", example="meta-llama/Llama-2-7b-hf")
    name: str = Field(..., description="Model display name", example="Llama 2 7B")
    author: Optional[str] = Field(None, description="Model author/organization", example="meta-llama")
    model_type: str = Field(..., description="Model type", example="decoder-only")
    attention_type: Optional[str] = Field(None, description="Attention mechanism type", example="gqa")
    parameter_count: Optional[int] = Field(None, description="Total parameter count", example=7000000000)
    logo: Optional[str] = Field(None, description="URL to the model's logo")
    model_analysis: Optional[ModelAnalysis] = Field(None, description="Detailed model analysis")
    source: str = Field(..., description="Model source (model_dict, database, or both)", example="database")
    in_model_dict: bool = Field(..., description="Whether model is in MODEL_DICT", example=False)
    in_database: bool = Field(..., description="Whether model is in database", example=True)


class ListModelsResponse(BaseModel):
    """Response model for listing all models."""
    total_count: int = Field(..., description="Total number of models", example=42)
    model_dict_count: int = Field(..., description="Number of models in MODEL_DICT", example=20)
    database_count: int = Field(..., description="Number of models in database", example=25)
    models: List[ModelSummary] = Field(..., description="List of all models")


class FilterModelsResponse(BaseModel):
    """Response model for filtered models."""
    total_count: int = Field(..., description="Total number of matching models", example=10)
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")
    models: List[ModelSummary] = Field(..., description="List of filtered models")


class AddModelResponse(BaseModel):
    """Response model for adding a model."""
    success: bool = Field(..., description="Whether the model was added successfully")
    model_id: str = Field(..., description="The ID of the added model", example="meta-llama/Llama-2-7b-hf")
    message: str = Field(..., description="Success or error message")
    source: Optional[str] = Field(None, description="Where the model was added (database/model_dict)")
    already_existed: bool = Field(False, description="Whether the model already existed")


class ModelDetailResponse(ModelSummary):
    """Detailed information for a single model."""
    config: Dict[str, Any] = Field(..., description="Full model configuration")
    logo: Optional[str] = Field(None, description="URL to the model's logo")
    analysis: Optional[ModelAnalysis] = Field(None, description="Detailed model analysis") 