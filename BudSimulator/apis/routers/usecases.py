"""
Usecase management API routes.
Provides endpoints for CRUD operations on usecases and SLOs.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

from src.usecases import BudUsecases
from src.bud_models import ModelMemoryCalculator
from src.hardware import BudHardware
from src.db.model_manager import ModelManager

router = APIRouter(prefix="/api/usecases", tags=["usecases"])

# Initialize managers
usecase_manager = BudUsecases()
memory_calculator = ModelMemoryCalculator()
hardware_manager = BudHardware()
model_manager = ModelManager()

# Default model categories — single source of truth for both table creation and population.
DEFAULT_MODEL_CATEGORIES: Dict[str, List[str]] = {
    '3B': [
        'google/gemma-3-4B',
        'microsoft/Phi-3-mini',
        'meta-llama/Llama-3.2-3B',
    ],
    '8B': [
        'meta-llama/meta-llama-3.1-8b',
        'google/gemma-2-9b',
        'Qwen/Qwen2.5-7B',
    ],
    '32B': [
        'Qwen/Qwen2.5-32B',
        'Qwen/Qwen1.5-32B',
        'google/gemma-3-27B',
    ],
    '72B': [
        'Qwen/Qwen2.5-72B',
        'meta-llama/meta-llama-3.1-70b',
        'meta-llama/Llama-4-Scout-17B-16E',
    ],
    '200B+': [
        'meta-llama/Llama-3.1-405B',
        'meta-llama/Llama-4-Maverick-17B-128E',
        'deepseek-ai/DeepSeek-V3-Base',
    ],
}


def _populate_model_categories(db: Any, model_categories: Dict[str, List[str]]) -> None:
    """Insert model categories into the database table."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM model_recommendation_categories")
        insert_sql = """
        INSERT INTO model_recommendation_categories (category, model_id, display_order)
        VALUES (?, ?, ?)
        """
        for category, models in model_categories.items():
            for order, model_id in enumerate(models, 1):
                cursor.execute(insert_sql, (category, model_id, order))
        conn.commit()


def _ensure_model_categories_table():
    """Ensure the model_recommendation_categories table exists and is populated."""
    try:
        from src.db.connection import DatabaseConnection

        db = DatabaseConnection()

        if not db.table_exists('model_recommendation_categories'):
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS model_recommendation_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category VARCHAR(10) NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                display_order INTEGER NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, model_id)
            );
            """
            create_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_model_categories_category
            ON model_recommendation_categories(category, display_order);
            """
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                cursor.execute(create_index_sql)
                conn.commit()

            _populate_model_categories(db, DEFAULT_MODEL_CATEGORIES)

        # Repopulate if empty
        count_result = db.execute_one(
            "SELECT COUNT(*) as count FROM model_recommendation_categories WHERE is_active = 1"
        )
        if not count_result or count_result['count'] == 0:
            _populate_model_categories(db, DEFAULT_MODEL_CATEGORIES)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to ensure model categories table: %s", e)

_model_categories_initialized = False


def _lazy_ensure_model_categories():
    """Lazily ensure model categories table on first use."""
    global _model_categories_initialized
    if not _model_categories_initialized:
        _ensure_model_categories_table()
        _model_categories_initialized = True


# Pydantic models for request/response
class UsecaseCreate(BaseModel):
    """Request model for creating a usecase."""
    unique_id: str = Field(..., description="Unique identifier for the usecase")
    name: str = Field(..., description="Name of the usecase")
    industry: str = Field(..., description="Industry category")
    description: Optional[str] = Field(None, description="Detailed description")
    
    # Configuration
    batch_size: int = Field(1, ge=1, description="Batch size for processing")
    beam_size: int = Field(1, ge=1, description="Beam size for generation")
    
    # Token ranges
    input_tokens_min: int = Field(..., ge=0, description="Minimum input tokens")
    input_tokens_max: int = Field(..., ge=0, description="Maximum input tokens")
    output_tokens_min: int = Field(..., ge=0, description="Minimum output tokens")
    output_tokens_max: int = Field(..., ge=0, description="Maximum output tokens")
    
    # SLOs (Service Level Objectives)
    ttft_min: Optional[float] = Field(None, ge=0, description="Minimum time to first token (seconds)")
    ttft_max: Optional[float] = Field(None, ge=0, description="Maximum time to first token (seconds)")
    e2e_min: Optional[float] = Field(None, ge=0, description="Minimum end-to-end latency (seconds)")
    e2e_max: Optional[float] = Field(None, ge=0, description="Maximum end-to-end latency (seconds)")
    inter_token_min: Optional[float] = Field(None, ge=0, description="Minimum inter-token latency (seconds)")
    inter_token_max: Optional[float] = Field(None, ge=0, description="Maximum inter-token latency (seconds)")
    
    # Tags for categorization
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    @validator('input_tokens_max')
    def validate_input_tokens(cls, v, values):
        if 'input_tokens_min' in values and v < values['input_tokens_min']:
            raise ValueError('input_tokens_max must be >= input_tokens_min')
        return v
    
    @validator('output_tokens_max')
    def validate_output_tokens(cls, v, values):
        if 'output_tokens_min' in values and v < values['output_tokens_min']:
            raise ValueError('output_tokens_max must be >= output_tokens_min')
        return v
    
    @validator('ttft_max')
    def validate_ttft(cls, v, values):
        if v is not None and 'ttft_min' in values and values['ttft_min'] is not None and v < values['ttft_min']:
            raise ValueError('ttft_max must be >= ttft_min')
        return v
    
    @validator('e2e_max')
    def validate_e2e(cls, v, values):
        if v is not None and 'e2e_min' in values and values['e2e_min'] is not None and v < values['e2e_min']:
            raise ValueError('e2e_max must be >= e2e_min')
        return v
    
    @validator('inter_token_max')
    def validate_inter_token(cls, v, values):
        if v is not None and 'inter_token_min' in values and values['inter_token_min'] is not None and v < values['inter_token_min']:
            raise ValueError('inter_token_max must be >= inter_token_min')
        return v


class UsecaseUpdate(BaseModel):
    """Request model for updating a usecase."""
    name: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    batch_size: Optional[int] = Field(None, ge=1)
    beam_size: Optional[int] = Field(None, ge=1)
    input_tokens_min: Optional[int] = Field(None, ge=0)
    input_tokens_max: Optional[int] = Field(None, ge=0)
    output_tokens_min: Optional[int] = Field(None, ge=0)
    output_tokens_max: Optional[int] = Field(None, ge=0)
    ttft_min: Optional[float] = Field(None, ge=0)
    ttft_max: Optional[float] = Field(None, ge=0)
    e2e_min: Optional[float] = Field(None, ge=0)
    e2e_max: Optional[float] = Field(None, ge=0)
    inter_token_min: Optional[float] = Field(None, ge=0)
    inter_token_max: Optional[float] = Field(None, ge=0)
    tags: Optional[List[str]] = None


class UsecaseResponse(BaseModel):
    """Response model for usecase."""
    id: int
    unique_id: str
    name: str
    industry: str
    description: Optional[str] = None
    batch_size: int
    beam_size: int
    input_tokens_min: int
    input_tokens_max: int
    output_tokens_min: int
    output_tokens_max: int
    ttft_min: Optional[float] = None
    ttft_max: Optional[float] = None
    e2e_min: Optional[float] = None
    e2e_max: Optional[float] = None
    inter_token_min: Optional[float] = None
    inter_token_max: Optional[float] = None
    tags: List[str] = []
    source: str
    created_at: str
    updated_at: str
    is_active: bool
    
    # Computed fields
    latency_profile: Optional[str] = None
    input_length_profile: Optional[str] = None
    
    @validator('latency_profile', always=True)
    def compute_latency_profile(cls, v, values):
        if 'e2e_max' in values and values['e2e_max'] is not None:
            if values['e2e_max'] <= 2:
                return 'real-time'
            elif values['e2e_max'] <= 5:
                return 'interactive'
            elif values['e2e_max'] <= 10:
                return 'responsive'
            else:
                return 'batch'
        return None
    
    @validator('input_length_profile', always=True)
    def compute_input_profile(cls, v, values):
        if 'input_tokens_max' in values:
            if values['input_tokens_max'] <= 1000:
                return 'short'
            elif values['input_tokens_max'] <= 10000:
                return 'medium'
            else:
                return 'long'
        return None


class UsecaseStats(BaseModel):
    """Response model for usecase statistics."""
    total_usecases: int
    by_industry: Dict[str, int]
    by_latency_profile: Dict[str, int]
    popular_tags: List[tuple]


class ImportResponse(BaseModel):
    """Response model for import operation."""
    total: int
    imported: int
    skipped: int
    errors: List[Dict[str, str]]


# New recommendation models
class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    batch_sizes: List[int] = Field([1, 8, 16, 32, 64], description="Batch sizes to analyze")
    model_categories: List[str] = Field(['3B', '8B', '32B', '72B', '200B+'], description="Model parameter categories")
    precision: str = Field('fp16', description="Precision for calculations")
    include_pricing: bool = Field(True, description="Include pricing estimates")

class HardwareOption(BaseModel):
    """Hardware recommendation option."""
    hardware_name: str
    nodes_required: int
    memory_per_chip: float
    utilization: float
    price_per_hour: Optional[float] = None
    total_cost_per_hour: Optional[float] = None
    optimality: str

class BatchConfiguration(BaseModel):
    """Configuration for a specific batch size."""
    batch_size: int
    memory_required_gb: float
    meets_slo: bool
    estimated_ttft: float
    estimated_e2e: float
    hardware_options: List[HardwareOption]

class ModelRecommendation(BaseModel):
    """Model recommendation with configurations."""
    model_id: str
    parameter_count: int
    model_type: str
    attention_type: Optional[str] = None
    batch_configurations: List[BatchConfiguration]

class CategoryRecommendation(BaseModel):
    """Recommendations for a model category."""
    model_category: str
    recommended_models: List[ModelRecommendation]

class RecommendationsResponse(BaseModel):
    """Complete recommendations response."""
    usecase: UsecaseResponse
    recommendations: List[CategoryRecommendation]


@router.post("", response_model=UsecaseResponse)
async def create_usecase(usecase: UsecaseCreate):
    """Create a new usecase."""
    try:
        # Convert Pydantic model to dict
        usecase_data = usecase.dict()
        
        # Add usecase
        unique_id = usecase_manager.add_usecase(usecase_data)
        
        # Get the created usecase
        created = usecase_manager.get_usecase(unique_id)
        if not created:
            raise HTTPException(status_code=500, detail="Failed to create usecase")
        
        return UsecaseResponse(**created)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("", response_model=List[UsecaseResponse])
async def list_usecases(
    industry: Optional[str] = None,
    max_e2e_latency: Optional[float] = Query(None, gt=0),
    max_ttft: Optional[float] = Query(None, gt=0),
    min_input_tokens: Optional[int] = Query(None, ge=0),
    max_input_tokens: Optional[int] = Query(None, ge=0),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all usecases with optional filters."""
    try:
        # Search with filters
        results = usecase_manager.search_usecases(
            industry=industry,
            max_e2e_latency=max_e2e_latency,
            max_ttft=max_ttft,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            limit=limit,
            offset=offset
        )
        
        return [UsecaseResponse(**usecase) for usecase in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/search", response_model=List[UsecaseResponse])
async def search_usecases(
    query: Optional[str] = Query(None, description="Text search in name and description"),
    industry: Optional[List[str]] = Query(None, description="Filter by industries"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags (must have all)"),
    min_input_tokens: Optional[int] = Query(None, ge=0),
    max_input_tokens: Optional[int] = Query(None, ge=0),
    min_output_tokens: Optional[int] = Query(None, ge=0),
    max_output_tokens: Optional[int] = Query(None, ge=0),
    max_e2e_latency: Optional[float] = Query(None, gt=0),
    max_ttft: Optional[float] = Query(None, gt=0),
    batch_size: Optional[int] = Query(None, ge=1),
    sort_by: str = Query("unique_id", regex="^(unique_id|name|industry|e2e_max|ttft_max|input_tokens_max|output_tokens_max|created_at)$"),
    sort_order: str = Query("asc", regex="^(asc|desc)$"),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Advanced usecase search with multiple criteria."""
    try:
        results = usecase_manager.search_usecases(
            query=query,
            industry=industry,
            tags=tags,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
            max_e2e_latency=max_e2e_latency,
            max_ttft=max_ttft,
            batch_size=batch_size,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        return [UsecaseResponse(**usecase) for usecase in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/stats", response_model=UsecaseStats)
async def get_usecase_stats():
    """Get statistics about usecases."""
    try:
        stats = usecase_manager.get_stats()
        return UsecaseStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{unique_id}", response_model=UsecaseResponse)
async def get_usecase(unique_id: str):
    """Get a specific usecase by unique ID."""
    try:
        usecase = usecase_manager.get_usecase(unique_id)
        if not usecase:
            raise HTTPException(status_code=404, detail="Usecase not found")
        
        return UsecaseResponse(**usecase)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.put("/{unique_id}", response_model=UsecaseResponse)
async def update_usecase(unique_id: str, updates: UsecaseUpdate):
    """Update an existing usecase."""
    try:
        # Convert to dict and remove None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        # Update usecase
        success = usecase_manager.update_usecase(unique_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Usecase not found")
        
        # Get updated usecase
        updated = usecase_manager.get_usecase(unique_id)
        return UsecaseResponse(**updated)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.delete("/{unique_id}")
async def delete_usecase(unique_id: str, hard_delete: bool = False):
    """Delete a usecase (soft delete by default)."""
    try:
        success = usecase_manager.delete_usecase(unique_id, hard_delete)
        if not success:
            raise HTTPException(status_code=404, detail="Usecase not found")
        
        return {"message": f"Usecase {unique_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/import", response_model=ImportResponse)
async def import_usecases(
    file_path: str = Body(..., description="Path to JSON file containing usecases")
):
    """Import usecases from a JSON file."""
    try:
        stats = usecase_manager.import_from_json(file_path)
        return ImportResponse(**stats)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/export")
async def export_usecases(
    file_path: str = Body(..., description="Path to export JSON file"),
    industry: Optional[str] = Body(None, description="Optional industry filter")
):
    """Export usecases to a JSON file."""
    try:
        success = usecase_manager.export_to_json(file_path, industry)
        if success:
            return {"message": f"Usecases exported successfully to {file_path}"}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/industries/list")
async def list_industries():
    """Get list of all unique industries."""
    try:
        all_usecases = usecase_manager.get_all_usecases()
        industries = sorted(list(set(uc['industry'] for uc in all_usecases)))
        return {"industries": industries}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/tags/list")
async def list_tags():
    """Get list of all unique tags with counts."""
    try:
        stats = usecase_manager.get_stats()
        tags = [{"tag": tag, "count": count} for tag, count in stats['popular_tags']]
        return {"tags": tags}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/{unique_id}/recommendations", response_model=RecommendationsResponse)
async def get_usecase_recommendations(unique_id: str, request: RecommendationRequest):
    """Get model and hardware recommendations for a specific usecase."""
    try:
        usecase = usecase_manager.get_usecase(unique_id)
        if not usecase:
            raise HTTPException(status_code=404, detail="Usecase not found")

        model_categories = _get_models_by_category(request.model_categories)
        recommendations = []
        
        for category, models in model_categories.items():
            category_models = []
            
            # Take top 2 models per category for recommendations
            for model_data in models[:2]:
                try:
                    config = model_manager.get_model_config(model_data['model_id'])
                    if not config:
                        continue
                    
                    batch_configs = []
                    for batch_size in request.batch_sizes:
                        # Use the maximum input tokens for memory calculation
                        seq_length = max(usecase['input_tokens_max'], 2048)
                        
                        memory_result = memory_calculator.calculate_total_memory(
                            config=config,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            precision=request.precision
                        )
                        
                        # Enhanced performance estimation
                        estimated_ttft, estimated_e2e = _estimate_performance(
                            config=config,
                            batch_size=batch_size,
                            usecase=usecase,
                            model_data=model_data
                        )
                        
                        # Check SLO compliance
                        meets_ttft_slo = (usecase.get('ttft_max') is None or 
                                        estimated_ttft <= usecase['ttft_max'])
                        meets_e2e_slo = (usecase.get('e2e_max') is None or 
                                       estimated_e2e <= usecase['e2e_max'])
                        meets_slo = meets_ttft_slo and meets_e2e_slo
                        
                        # Get hardware recommendations with improved sorting
                        hardware_options = _get_hardware_recommendations(
                            required_memory_gb=memory_result.total_memory_gb,
                            model_config=config,
                            usecase_config=usecase,
                            batch_size=batch_size,
                            include_pricing=request.include_pricing,
                            meets_slo=meets_slo,
                            model_data=model_data
                        )

                        batch_configs.append(BatchConfiguration(
                            batch_size=batch_size,
                            memory_required_gb=memory_result.total_memory_gb,
                            meets_slo=meets_slo,
                            estimated_ttft=estimated_ttft,
                            estimated_e2e=estimated_e2e,
                            hardware_options=[HardwareOption(**hw) for hw in hardware_options]
                        ))
                    
                    if batch_configs:
                        category_models.append(ModelRecommendation(
                            model_id=model_data['model_id'],
                            parameter_count=model_data.get('parameter_count', 0),
                            model_type=model_data.get('model_type', 'unknown'),
                            attention_type=model_data.get('attention_type'),
                            batch_configurations=batch_configs
                        ))
                        
                except Exception as e:
                    logging.getLogger(__name__).warning("Error processing model %s: %s", model_data.get('model_id', 'unknown'), e)
                    continue
            
            if category_models:
                recommendations.append(CategoryRecommendation(
                    model_category=category,
                    recommended_models=category_models
                ))
        
        return RecommendationsResponse(
            usecase=UsecaseResponse(**usecase),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def _estimate_performance(config: Dict[str, Any], batch_size: int, 
                         usecase: Dict[str, Any], model_data: Dict[str, Any]) -> tuple[float, float]:
    """
    Enhanced performance estimation for TTFT and E2E latency.
    
    Args:
        config: Model configuration
        batch_size: Batch size for inference
        usecase: Usecase configuration with SLO requirements
        model_data: Model metadata
        
    Returns:
        Tuple of (estimated_ttft, estimated_e2e) in seconds
    """
    # Base TTFT estimation (time to first token)
    base_ttft = 0.1  # 100ms baseline
    
    # Model size scaling factor
    param_count = model_data.get('parameter_count', 0)
    if param_count > 70e9:  # 70B+ models
        size_multiplier = 3.0
    elif param_count > 30e9:  # 30B+ models  
        size_multiplier = 2.0
    elif param_count > 7e9:   # 7B+ models
        size_multiplier = 1.5
    else:
        size_multiplier = 1.0
    
    # Batch size penalty (additional latency for larger batches)
    batch_penalty = 1.0 + (batch_size - 1) * 0.1  # 10% penalty per additional batch item
    
    # Calculate TTFT
    estimated_ttft = base_ttft * size_multiplier * batch_penalty
    
    # Inter-token latency estimation
    base_inter_token = 0.02  # 20ms baseline
    inter_token_latency = base_inter_token * size_multiplier
    
    # Attention efficiency bonuses
    attention_type = model_data.get('attention_type', 'mha')
    if attention_type == 'gqa':
        inter_token_latency *= 0.8  # 20% faster with GQA
    elif attention_type == 'mqa':
        inter_token_latency *= 0.8  # 20% faster with MQA  
    elif attention_type == 'mla':
        inter_token_latency *= 0.6  # 40% faster with MLA
    
    # Calculate E2E latency
    output_tokens = usecase.get('output_tokens_max', 100)
    estimated_e2e = estimated_ttft + (output_tokens * inter_token_latency)
    
    return estimated_ttft, estimated_e2e


def _get_hardware_recommendations(required_memory_gb: float, model_config: Dict[str, Any],
                                usecase_config: Dict[str, Any], batch_size: int,
                                include_pricing: bool = True, meets_slo: bool = True,
                                model_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Get hardware recommendations with improved sorting and filtering.
    
    Sorting Priority:
    1. Fewest number of nodes (ascending)
    2. Highest utilization for same node count (descending)
    
    CPU Filtering:
    - Exclude CPUs for models with >14B parameters
    - Exclude CPUs for total memory >40GB
    """
    all_hardware = hardware_manager.get_all_hardwares()
    recommendations = []
    
    # Determine if CPUs should be excluded
    model_params = model_data.get('parameter_count', 0) if model_data else 0
    if model_params == 0:
        # Try to calculate parameters from config if not available in model_data
        try:
            model_params = memory_calculator.calculate_parameters(model_config)
        except Exception:
            model_params = 0
    
    # CPU exclusion criteria
    exclude_cpus = (
        model_params > 14e9 or  # More than 14B parameters
        required_memory_gb > 40  # More than 40GB total memory
    )
    
    if exclude_cpus:
        logger.info(f"Excluding CPUs: Model has {model_params/1e9:.1f}B params, requires {required_memory_gb:.1f}GB memory")
    else:
        logger.info(f"Including CPUs: Model has {model_params/1e9:.1f}B params, requires {required_memory_gb:.1f}GB memory")
    
    for hw in all_hardware:
        # Get hardware type
        hw_type = hw.get('type', '').lower()
        
        # Skip CPUs if they should be excluded
        if exclude_cpus and hw_type == 'cpu':
            continue
            
        memory_per_chip = hw.get('Memory_size', hw.get('memory_size', 0))
        if memory_per_chip <= 0:
            continue
        
        # Calculate nodes required (round up)
        nodes_required = max(1, int((required_memory_gb + memory_per_chip - 1) // memory_per_chip))
        total_memory = nodes_required * memory_per_chip
        utilization = (required_memory_gb / total_memory) * 100
        
        # Skip hardware with very low utilization (< 10%)
        if utilization < 10:
            continue
        
        # Calculate pricing if requested
        price_per_hour = None
        total_cost_per_hour = None
        if include_pricing:
            flops = hw.get('Flops', hw.get('flops', 0))
            memory_bw = hw.get('Memory_BW', hw.get('memory_bw', 0))
            
            if flops > 0 and memory_per_chip > 0 and memory_bw > 0:
                price_per_hour = hardware_manager.calculate_price_indicator(
                    flops=flops,
                    memory_gb=memory_per_chip,
                    bandwidth_gbs=memory_bw
                )
                total_cost_per_hour = price_per_hour * nodes_required
        
        # Determine optimality based on utilization
        if utilization >= 80:
            optimality = 'optimal'
        elif utilization >= 60:
            optimality = 'good'
        else:
            optimality = 'ok'
        
        recommendations.append({
            'hardware_name': hw['name'],
            'nodes_required': nodes_required,
            'memory_per_chip': memory_per_chip,
            'utilization': round(utilization, 1),
            'price_per_hour': price_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'optimality': optimality
        })
    
    # Sort by nodes required (ascending), then utilization (descending)
    recommendations.sort(key=lambda x: (x['nodes_required'], -x['utilization']))
    
    # Return top 5 recommendations
    return recommendations[:5]

def _get_models_by_category(categories: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get models organized by categories using the model_recommendation_categories table."""
    try:
        from src.db.connection import DatabaseConnection
        
        db = DatabaseConnection()
        categorized = {}
        
        # Try to use the dedicated table first
        try:
            for category in categories:
                # Get models for this category from the table
                category_model_ids = db.execute("""
                    SELECT model_id, display_order
                    FROM model_recommendation_categories 
                    WHERE category = ? AND is_active = 1
                    ORDER BY display_order
                """, (category,))
                
                if category_model_ids:
                    category_models = []
                    
                    # Get model details from the models table
                    for row in category_model_ids:
                        model_id = row['model_id']
                        model_data = model_manager.get_model(model_id)
                        
                        if model_data:
                            # Ensure we have the necessary fields
                            if not model_data.get('parameter_count'):
                                # Try to get parameter count from config
                                config = model_manager.get_model_config(model_id)
                                if config:
                                    # Use ModelMemoryCalculator to get accurate parameter count
                                    try:
                                        param_count = memory_calculator.calculate_parameters(config)
                                        model_data['parameter_count'] = param_count
                                    except Exception:
                                        model_data['parameter_count'] = 0
                            
                            # Ensure attention type is detected
                            if not model_data.get('attention_type'):
                                config = model_manager.get_model_config(model_id)
                                if config:
                                    try:
                                        attention_type = memory_calculator.detect_attention_type(config)
                                        model_data['attention_type'] = attention_type
                                    except Exception:
                                        model_data['attention_type'] = 'mha'
                            
                            category_models.append(model_data)
                        else:
                            # If model not found in database, create a minimal entry
                            logger.warning(f"Model {model_id} not found in database, creating minimal entry")
                            category_models.append({
                                'model_id': model_id,
                                'parameter_count': 0,
                                'model_type': 'unknown',
                                'attention_type': 'mha'
                            })
                    
                    categorized[category] = category_models
                else:
                    # Category not found in table, use empty list
                    logger.warning(f"No models found for category {category} in model_recommendation_categories table")
                    categorized[category] = []
            
            # If we got any results from the table, return them
            if any(categorized.values()):
                return categorized
            else:
                logger.warning("No models found in model_recommendation_categories table, falling back to parameter ranges")
                
        except Exception as e:
            logger.warning(f"Failed to use model categories table: {e}")
            # Fall back to parameter ranges
            
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
    
    # Fallback: use parameter ranges (original implementation)
    logger.info("Using parameter ranges fallback for model categorization")
    all_models = model_manager.list_models()
    
    # Define parameter ranges (in billions) - updated to match new categories
    param_ranges = {
        '3B': (1e9, 5e9),
        '8B': (5e9, 15e9),
        '32B': (15e9, 50e9),
        '72B': (50e9, 100e9),  # Changed from '70B' to '72B'
        '200B+': (100e9, float('inf'))  # Changed from '600B+' to '200B+'
    }
    
    categorized = {}
    
    for category in categories:
        if category not in param_ranges:
            logger.warning(f"Unknown category {category}, skipping")
            categorized[category] = []
            continue
            
        min_params, max_params = param_ranges[category]
        category_models = []
        
        for model in all_models:
            param_count = model.get('parameter_count')
            if param_count and min_params <= param_count < max_params:
                # Ensure attention type is available
                if not model.get('attention_type'):
                    config = model_manager.get_model_config(model['model_id'])
                    if config:
                        try:
                            attention_type = memory_calculator.detect_attention_type(config)
                            model['attention_type'] = attention_type
                        except Exception:
                            model['attention_type'] = 'mha'
                
                category_models.append(model)
        
        # Sort by parameter count and take top models
        category_models.sort(key=lambda x: x.get('parameter_count', 0), reverse=True)
        categorized[category] = category_models[:5]  # Top 5 per category
    
    return categorized 