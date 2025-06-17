"""
Models router for FastAPI application.
"""

import sys
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import logging
import json

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import schemas
from ..schemas import (
    ValidateModelRequest, ValidateModelResponse,
    CalculateMemoryRequest, CalculateMemoryResponse,
    CompareModelsRequest, CompareModelsResponse,
    AnalyzeModelRequest, AnalyzeModelResponse,
    ModelConfigResponse, PopularModelsResponse,
    ModelConfig, ModelMetadata, MemoryBreakdown,
    MemoryRecommendations, ModelComparison,
    SequenceAnalysis, AnalysisInsights, PopularModel,
    ListModelsResponse, FilterModelsResponse, ModelSummary,
    AddModelFromHFRequest, AddModelFromConfigRequest, AddModelResponse,
    ModelDetailResponse, ConfigSubmitRequest, ConfigSubmitResponse
)

# Import BudSimulator modules
try:
    from src.bud_models import (
        HuggingFaceConfigLoader, 
        ModelMemoryCalculator,
        estimate_memory,
        analyze_hf_model
    )
    from src.db import ModelManager, HuggingFaceModelImporter
    from GenZ.Models import MODEL_DICT
except ImportError as e:
    logging.error(f"Failed to import BudSimulator modules: {e}")
    raise

# Create router
router = APIRouter()

# Initialize services
hf_loader = HuggingFaceConfigLoader()
memory_calculator = ModelMemoryCalculator()
model_manager = ModelManager()
hf_importer = HuggingFaceModelImporter()


def format_parameter_count(count: Optional[int]) -> str:
    """Format parameter count for display."""
    if count is None:
        return "Unknown"
    
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def extract_model_name(model_id: str) -> str:
    """Extract a display name from model ID."""
    # Remove organization prefix
    name = model_id.split('/')[-1] if '/' in model_id else model_id
    
    # Clean up common patterns
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize appropriately
    parts = name.split()
    formatted_parts = []
    for part in parts:
        if part.lower() in ['llama', 'gpt', 'bert', 'phi']:
            formatted_parts.append(part.upper())
        elif part[0].isdigit() and part[-1].lower() == 'b':
            formatted_parts.append(part.upper())
        else:
            formatted_parts.append(part.capitalize())
    
    return ' '.join(formatted_parts)


@router.post("/validate", response_model=ValidateModelResponse)
async def validate_model(request: ValidateModelRequest):
    """
    Validate if a model URL/ID is valid and accessible.
    
    This endpoint checks if the provided model URL or ID corresponds to a valid
    model on HuggingFace Hub or in the local database.
    """
    try:
        # Check if it's already in the database
        db_model = model_manager.get_model(request.model_url)
        if db_model:
            return ValidateModelResponse(valid=True, error=None)
        
        # Try to fetch from HuggingFace
        try:
            config = hf_loader.fetch_model_config(request.model_url)
            return ValidateModelResponse(
                valid=True, 
                error=None,
                model_id=request.model_url
            )
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "403" in error_msg:
                return ValidateModelResponse(
                    valid=False,
                    error="This model is gated and requires authentication. Please provide the config.json manually.",
                    error_code="MODEL_GATED",
                    model_id=request.model_url,
                    requires_config=True,
                    config_submission_url="/api/models/config/submit"
                )
            elif "not found" in error_msg.lower() or "404" in error_msg:
                return ValidateModelResponse(
                    valid=False,
                    error="Model not found on HuggingFace Hub",
                    error_code="NOT_FOUND"
                )
            else:
                return ValidateModelResponse(
                    valid=False,
                    error=f"Failed to access model: {error_msg}",
                    error_code="ACCESS_ERROR"
                )
                
    except Exception as e:
        logging.error(f"Error validating model {request.model_url}: {e}")
        return ValidateModelResponse(
            valid=False,
            error=f"Internal error: {str(e)}"
        )


@router.get("/{model_id:path}/config", response_model=ModelConfigResponse)
async def get_model_config(model_id: str, request: Request):
    """
    Get detailed configuration for a specific model.
    
    This endpoint returns the model's architecture details, configuration parameters,
    and metadata from HuggingFace Hub.
    """
    try:
        # Try to get from database first
        db_model = model_manager.get_model(model_id)
        
        if db_model:
            config = model_manager.get_model_config(model_id)
            model_type = db_model['model_type'] or 'unknown'
            attention_type = db_model['attention_type']
            parameter_count = db_model['parameter_count']
            
            # Get logo from database
            logo = db_model.get('logo')
            if logo and logo.startswith('logos/'):
                # Convert relative path to full URL
                logo = str(request.url_for('logos', path=logo.split('/')[-1]))
            
            # Get model analysis from database
            model_analysis = None
            if db_model.get('model_analysis'):
                try:
                    analysis_data = json.loads(db_model['model_analysis'])
                    # Simply check if it's a valid dict with at least description
                    if analysis_data and isinstance(analysis_data, dict) and 'description' in analysis_data:
                        model_analysis = analysis_data
                except:
                    model_analysis = None
        else:
            # Fetch from HuggingFace
            try:
                config = hf_loader.get_model_config(model_id)
                model_type = memory_calculator.detect_model_type(config)
                attention_type = memory_calculator.detect_attention_type(config)
                parameter_count = config.get('num_parameters')
                logo = None  # No logo for models not in database
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "403" in error_msg:
                    raise HTTPException(
                        status_code=403, 
                        detail={
                            "error": "This model is gated and requires authentication. Please provide the config.json manually.",
                            "error_code": "MODEL_GATED",
                            "model_id": model_id
                        }
                    )
                raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        
        # Get model info for metadata
        try:
            model_info = hf_loader.get_model_info(model_id)
            downloads = getattr(model_info, 'downloads', 0)
            likes = getattr(model_info, 'likes', 0)
            tags = getattr(model_info, 'tags', [])
            
            # Estimate size
            if parameter_count:
                # Rough estimate: 2 bytes per parameter for fp16
                size_gb = (parameter_count * 2) / (1024**3)
            else:
                size_gb = None
        except:
            downloads = 0
            likes = 0
            tags = []
            size_gb = None
        
        # Extract architecture
        architectures = config.get('architectures', [])
        architecture = architectures[0] if architectures else None
        
        # Get model analysis from database
        model_analysis = None
        if db_model and db_model.get('model_analysis'):
            try:
                model_analysis = json.loads(db_model['model_analysis'])
            except:
                model_analysis = None
        
        return ModelConfigResponse(
            model_id=model_id,
            model_type=model_type,
            attention_type=attention_type,
            parameter_count=parameter_count,
            architecture=architecture,
            logo=logo,
            model_analysis=model_analysis,
            config=ModelConfig(
                hidden_size=config.get('hidden_size', 0),
                num_hidden_layers=config.get('num_hidden_layers', 0),
                num_attention_heads=config.get('num_attention_heads', 0),
                num_key_value_heads=config.get('num_key_value_heads'),
                intermediate_size=config.get('intermediate_size', 0),
                vocab_size=config.get('vocab_size', 0),
                max_position_embeddings=config.get('max_position_embeddings', 0),
                activation_function=config.get('activation_function', config.get('hidden_act', 'unknown'))
            ),
            metadata=ModelMetadata(
                downloads=downloads,
                likes=likes,
                size_gb=size_gb,
                tags=tags
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting config for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/calculate", response_model=CalculateMemoryResponse)
async def calculate_memory(request: CalculateMemoryRequest):
    """
    Calculate memory requirements for a model with specific parameters.
    
    This endpoint calculates detailed memory breakdown including weights, KV cache,
    activations, and provides GPU recommendations.
    """
    try:
        # Get model configuration
        try:
            # Try database first (including user configs)
            config = model_manager.get_model_config(request.model_id)
            if not config:
                # Fetch from HuggingFace
                config = hf_loader.get_model_config(request.model_id)
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "403" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "This model is gated and requires authentication. Please provide the config.json manually.",
                        "error_code": "MODEL_GATED",
                        "model_id": request.model_id
                    }
                )
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        
        # Calculate memory
        result = estimate_memory(
            config,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            num_images=request.num_images,
            precision=request.precision,
            include_gradients=request.include_gradients,
            decode_length=request.decode_length
        )
        
        return CalculateMemoryResponse(
            model_type=result.model_type,
            attention_type=result.attention_type,
            precision=request.precision,
            parameter_count=result.parameter_count,
            memory_breakdown=MemoryBreakdown(
                weight_memory_gb=round(result.weight_memory_gb, 3),
                kv_cache_gb=round(result.kv_cache_gb, 3),
                activation_memory_gb=round(result.activation_memory_gb, 3),
                state_memory_gb=round(result.state_memory_gb, 3),
                image_memory_gb=round(result.image_memory_gb, 3),
                extra_work_gb=round(result.extra_work_gb, 3)
            ),
            total_memory_gb=round(result.total_memory_gb, 1),
            recommendations=MemoryRecommendations(
                recommended_gpu_memory_gb=result.recommended_gpu_memory_gb,
                can_fit_24gb_gpu=result.can_fit_24gb_gpu,
                can_fit_80gb_gpu=result.can_fit_80gb_gpu,
                min_gpu_memory_gb=round(result.total_memory_gb, 1)
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error calculating memory for {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/compare", response_model=CompareModelsResponse)
async def compare_models(request: CompareModelsRequest):
    """
    Compare memory requirements for multiple models.
    
    This endpoint allows comparing memory requirements across different models
    with potentially different configurations.
    """
    try:
        comparisons = []
        
        for model_config in request.models:
            try:
                # Get model configuration
                config = model_manager.get_model_config(model_config.model_id)
                if not config:
                    config = hf_loader.get_model_config(model_config.model_id)
                
                # Calculate memory
                result = estimate_memory(
                    config,
                    batch_size=model_config.batch_size,
                    seq_length=model_config.seq_length,
                    precision=model_config.precision
                )
                
                comparisons.append(ModelComparison(
                    model_id=model_config.model_id,
                    model_name=extract_model_name(model_config.model_id),
                    total_memory_gb=round(result.total_memory_gb, 1),
                    memory_breakdown=MemoryBreakdown(
                        weight_memory_gb=round(result.weight_memory_gb, 3),
                        kv_cache_gb=round(result.kv_cache_gb, 3),
                        activation_memory_gb=round(result.activation_memory_gb, 3),
                        state_memory_gb=round(result.state_memory_gb, 3),
                        image_memory_gb=round(result.image_memory_gb, 3),
                        extra_work_gb=round(result.extra_work_gb, 3)
                    ),
                    recommendations=MemoryRecommendations(
                        recommended_gpu_memory_gb=result.recommended_gpu_memory_gb,
                        can_fit_24gb_gpu=result.can_fit_24gb_gpu,
                        can_fit_80gb_gpu=result.can_fit_80gb_gpu,
                        min_gpu_memory_gb=round(result.total_memory_gb, 1)
                    )
                ))
                
            except Exception as e:
                logging.error(f"Error comparing model {model_config.model_id}: {e}")
                # Continue with other models
                
        if not comparisons:
            raise HTTPException(status_code=400, detail="No valid models to compare")
            
        return CompareModelsResponse(comparisons=comparisons)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/analyze", response_model=AnalyzeModelResponse)
async def analyze_model(request: AnalyzeModelRequest):
    """
    Analyze model efficiency across different sequence lengths.
    
    This endpoint provides detailed analysis of how memory requirements scale
    with sequence length and provides efficiency insights.
    """
    try:
        # Get model configuration
        try:
            config = model_manager.get_model_config(request.model_id)
            if not config:
                config = hf_loader.get_model_config(request.model_id)
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "403" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "This model is gated and requires authentication. Please provide the config.json manually.",
                        "error_code": "MODEL_GATED",
                        "model_id": request.model_id
                    }
                )
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        
        # Detect attention type
        attention_type = memory_calculator.detect_attention_type(config)
        
        # Analyze for each sequence length
        analysis = {}
        total_memory_values = []
        kv_cache_values = []
        
        for seq_len in request.sequence_lengths:
            result = estimate_memory(
                config,
                batch_size=request.batch_size,
                seq_length=seq_len,
                precision=request.precision
            )
            
            kv_cache_percent = (result.kv_cache_bytes / result.total_memory_bytes * 100) if result.total_memory_bytes > 0 else 0
            
            analysis[str(seq_len)] = SequenceAnalysis(
                total_memory_gb=round(result.total_memory_gb, 1),
                kv_cache_gb=round(result.kv_cache_gb, 3),
                kv_cache_percent=round(kv_cache_percent, 1)
            )
            
            total_memory_values.append(result.total_memory_gb)
            kv_cache_values.append(result.kv_cache_gb)
        
        # Calculate memory per token
        if len(request.sequence_lengths) >= 2:
            # Use difference between two sequence lengths
            seq_diff = request.sequence_lengths[-1] - request.sequence_lengths[0]
            kv_diff = kv_cache_values[-1] - kv_cache_values[0]
            memory_per_token_bytes = int((kv_diff * 1e9) / seq_diff) if seq_diff > 0 else 0
        else:
            memory_per_token_bytes = 0
        
        # Determine efficiency rating
        if attention_type == "mla":
            efficiency_rating = "high"
        elif attention_type == "gqa":
            efficiency_rating = "high"
        elif attention_type == "mqa":
            efficiency_rating = "medium"
        else:  # mha
            efficiency_rating = "low"
        
        # Generate recommendations
        recommendations = []
        
        if attention_type == "mha":
            recommendations.append("Consider using a model with GQA or MQA for better memory efficiency")
        elif attention_type == "gqa":
            num_heads = config.get('num_attention_heads', 1)
            num_kv_heads = config.get('num_key_value_heads', num_heads)
            compression = num_heads / num_kv_heads if num_kv_heads > 0 else 1
            recommendations.append(f"GQA provides {compression:.0f}x KV cache compression compared to MHA")
        elif attention_type == "mqa":
            recommendations.append("MQA provides maximum KV cache compression with single KV head")
        elif attention_type == "mla":
            recommendations.append("MLA provides state-of-the-art memory efficiency through latent attention")
        
        # Add sequence length recommendations
        if max(total_memory_values) > 80:
            recommendations.append("Consider using shorter sequence lengths or quantization for this model")
        elif max(total_memory_values) > 24:
            recommendations.append("This model requires high-end GPUs for long sequences")
        
        return AnalyzeModelResponse(
            model_id=request.model_id,
            attention_type=attention_type,
            analysis=analysis,
            insights=AnalysisInsights(
                memory_per_token_bytes=memory_per_token_bytes,
                efficiency_rating=efficiency_rating,
                recommendations=recommendations
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error analyzing model {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/config/submit", response_model=ConfigSubmitResponse)
async def submit_model_config(request: ConfigSubmitRequest):
    """
    Submit a configuration for a gated model.
    
    This endpoint allows users to manually provide configurations for gated models
    that require authentication to access on HuggingFace Hub.
    """
    try:
        # Validate required fields in config
        required_fields = ['model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads']
        missing_fields = []
        
        for field in required_fields:
            if field not in request.config:
                missing_fields.append(field)
        
        if missing_fields:
            return ConfigSubmitResponse(
                success=False,
                message="Configuration is missing required fields",
                model_id=request.model_uri,
                error_code="INVALID_CONFIG",
                missing_fields=missing_fields
            )
        
        # Save the configuration
        try:
            model_manager.save_user_model_config(request.model_uri, request.config)
        except Exception as e:
            logging.error(f"Failed to save config for {request.model_uri}: {e}")
            return ConfigSubmitResponse(
                success=False,
                message=f"Failed to save configuration: {str(e)}",
                model_id=request.model_uri,
                error_code="SAVE_ERROR"
            )
        
        # Try to detect model type and attention type
        try:
            model_type = memory_calculator.detect_model_type(request.config)
            attention_type = memory_calculator.detect_attention_type(request.config)
            parameter_count = request.config.get('num_parameters')
            
            validation = {
                'valid': True,
                'model_type': model_type,
                'attention_type': attention_type,
                'parameter_count': parameter_count
            }
        except:
            validation = {
                'valid': True,
                'model_type': 'unknown',
                'attention_type': 'unknown',
                'parameter_count': None
            }
        
        return ConfigSubmitResponse(
            success=True,
            message="Configuration saved successfully",
            model_id=request.model_uri,
            validation=validation
        )
        
    except Exception as e:
        logging.error(f"Error submitting config for {request.model_uri}: {e}")
        return ConfigSubmitResponse(
            success=False,
            message=f"Internal error: {str(e)}",
            model_id=request.model_uri,
            error_code="INTERNAL_ERROR"
        )


@router.get("/popular", response_model=PopularModelsResponse)
async def get_popular_models(request: Request, limit: int = 10):
    """
    Get a list of popular models.
    
    This endpoint returns popular models based on downloads and likes from
    HuggingFace Hub, enriched with memory analysis data.
    """
    try:
        # Get models that exist in both MODEL_DICT and database (they have logos)
        all_models_response = await list_all_models(request)
        all_models = all_models_response.models
        
        # Filter for models that exist in both sources (they have logos)
        models_with_logos = [m for m in all_models if m.source == "both" and m.logo]
        
        # Sort by parameter count (descending) to get larger models first
        models_with_logos.sort(key=lambda x: x.parameter_count or 0, reverse=True)
        
        # If we don't have enough models with logos, add some from MODEL_DICT
        if len(models_with_logos) < limit:
            model_dict_models = [m for m in all_models if m.source == "model_dict" and m not in models_with_logos]
            models_with_logos.extend(model_dict_models[:limit - len(models_with_logos)])
        
        models = []
        
        for model_summary in models_with_logos[:limit]:
            try:
                model_id = model_summary.model_id
                
                # Get model info
                db_model = model_manager.get_model(model_id)
                
                if db_model:
                    model_type = db_model['model_type'] or 'unknown'
                    attention_type = db_model['attention_type']
                    parameter_count = db_model['parameter_count']
                    
                    # Get logo from database
                    logo = db_model.get('logo')
                    if logo and logo.startswith('logos/'):
                        # Convert relative path to full URL
                        logo = str(request.url_for('logos', path=logo.split('/')[-1]))
                else:
                    # Use data from MODEL_DICT
                    model_type = model_summary.model_type
                    attention_type = model_summary.attention_type
                    parameter_count = model_summary.parameter_count
                    logo = model_summary.logo
                
                # Get metadata
                try:
                    model_info = hf_loader.get_model_info(model_id)
                    downloads = getattr(model_info, 'downloads', 0)
                    likes = getattr(model_info, 'likes', 0)
                except:
                    downloads = 0
                    likes = 0
                
                # Generate description
                param_str = format_parameter_count(parameter_count)
                if attention_type == "gqa":
                    description = f"High-performance {param_str} language model with GQA"
                elif attention_type == "mqa":
                    description = f"Efficient {param_str} language model with MQA"
                else:
                    description = f"Powerful {param_str} language model"
                
                models.append(PopularModel(
                    model_id=model_id,
                    name=extract_model_name(model_id),
                    parameters=param_str,
                    model_type=model_type,
                    attention_type=attention_type,
                    downloads=downloads,
                    likes=likes,
                    description=description,
                    logo=logo
                ))
                
            except Exception as e:
                logging.error(f"Error processing popular model {model_id}: {e}")
                continue
        
        return PopularModelsResponse(models=models)
        
    except Exception as e:
        logging.error(f"Error getting popular models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/list", response_model=ListModelsResponse)
async def list_all_models(request: Request):
    """
    Get a comprehensive list of all models from both MODEL_DICT and database.
    
    This endpoint returns all available models, indicating their source and
    providing summary information for each.
    """
    try:
        models_map = {}  # Use dict to avoid duplicates
        
        # Get models from MODEL_DICT
        model_dict_count = 0
        try:
            # Check if MODEL_DICT has been patched to DynamicModelCollection
            if hasattr(MODEL_DICT, 'models'):
                # Access the models dictionary directly
                for model_name, model_config in MODEL_DICT.models.items():
                    try:
                        model_id = getattr(model_config, 'model', model_name)
                        author = model_id.split('/')[0] if '/' in model_id else None
                        logo = getattr(model_config, 'logo', None)
                        if logo and logo.startswith('logos/'):
                            # Convert relative path to full URL
                            logo = str(request.url_for('logos', path=logo.split('/')[-1]))
                        models_map[model_id] = ModelSummary(
                            model_id=model_id,
                            name=extract_model_name(model_id),
                            author=author,
                            model_type=getattr(model_config, 'model_type', 'unknown'),
                            attention_type=getattr(model_config, 'attention_type', None),
                            parameter_count=getattr(model_config, 'num_parameters', None),
                            logo=logo,
                            model_analysis=None,
                            source="model_dict",
                            in_model_dict=True,
                            in_database=False
                        )
                        model_dict_count += 1
                    except Exception as e:
                        logging.error(f"Error processing MODEL_DICT model {model_name}: {e}")
            else:
                # Fallback: try to list models using list_models method
                model_list = MODEL_DICT.list_models() if hasattr(MODEL_DICT, 'list_models') else []
                for model_id in model_list:
                    try:
                        model = MODEL_DICT.get_model(model_id)
                        if model:
                            author = model_id.split('/')[0] if '/' in model_id else None
                            logo = getattr(model, 'logo', None)
                            if logo and logo.startswith('logos/'):
                                # Convert relative path to full URL
                                logo = str(request.url_for('logos', path=logo.split('/')[-1]))
                            models_map[model_id] = ModelSummary(
                                model_id=model_id,
                                name=extract_model_name(model_id),
                                author=author,
                                model_type=getattr(model, 'model_type', 'unknown'),
                                attention_type=getattr(model, 'attention_type', None),
                                parameter_count=getattr(model, 'num_parameters', None),
                                logo=logo,
                                model_analysis=None,
                                source="model_dict",
                                in_model_dict=True,
                                in_database=False
                            )
                            model_dict_count += 1
                    except Exception as e:
                        logging.error(f"Error processing MODEL_DICT model {model_id}: {e}")
        except Exception as e:
            logging.error(f"Error accessing MODEL_DICT: {e}")
        
        # Get models from database
        database_count = 0
        db_models = model_manager.list_models()
        for db_model in db_models:
            model_id = db_model['model_id']
            author = model_id.split('/')[0] if '/' in model_id else None
            
            logo = db_model.get('logo')
            if logo and logo.startswith('logos/'):
                # Convert relative path to full URL
                logo = str(request.url_for('logos', path=logo.split('/')[-1]))
            
            # Get model analysis from database
            model_analysis = None
            if db_model.get('model_analysis'):
                try:
                    analysis_data = json.loads(db_model['model_analysis'])
                    # Simply check if it's a valid dict with at least description
                    if analysis_data and isinstance(analysis_data, dict) and 'description' in analysis_data:
                        model_analysis = analysis_data
                except:
                    model_analysis = None
            
            summary = ModelSummary(
                    model_id=model_id,
                    name=extract_model_name(model_id),
                    author=author,
                model_type=db_model.get('model_type') or 'unknown',
                attention_type=db_model.get('attention_type'),
                parameter_count=db_model.get('parameter_count'),
                logo=logo,
                model_analysis=model_analysis,
                    source="database",
                    in_model_dict=False,
                    in_database=True
                )

            if model_id in models_map:
                # Model exists in both - update the entry
                models_map[model_id].source = "both"
                models_map[model_id].in_database = True
                models_map[model_id].logo = summary.logo
                models_map[model_id].model_analysis = summary.model_analysis
            else:
                # Model only in database
                models_map[model_id] = summary
            database_count += 1
        
        # Convert to list and sort by model_id
        models_list = sorted(models_map.values(), key=lambda x: x.model_id)
        
        return ListModelsResponse(
            total_count=len(models_list),
            model_dict_count=model_dict_count,
            database_count=database_count,
            models=models_list
        )
        
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/filter", response_model=FilterModelsResponse)
async def filter_models(
    author: Optional[str] = None,
    model_type: Optional[str] = None,
    attention_type: Optional[str] = None,
    min_parameters: Optional[int] = None,
    max_parameters: Optional[int] = None,
    source: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    Filter models by various criteria.
    
    Query Parameters:
    - author: Filter by model author/organization (e.g., "meta-llama", "microsoft")
    - model_type: Filter by type (decoder-only, encoder-decoder, etc.)
    - attention_type: Filter by attention mechanism (mha, gqa, mqa, mla)
    - min_parameters: Minimum parameter count
    - max_parameters: Maximum parameter count
    - source: Filter by source (model_dict, database, or both)
    - tags: Comma-separated tags to filter by
    """
    try:
        # First get all models
        all_models_response = await list_all_models()
        models = all_models_response.models
        
        # Apply filters
        filtered_models = []
        filters_applied = {}
        
        for model in models:
            # Author filter
            if author and model.author != author:
                continue
            if author:
                filters_applied['author'] = author
            
            # Model type filter
            if model_type and model.model_type != model_type:
                continue
            if model_type:
                filters_applied['model_type'] = model_type
            
            # Attention type filter
            if attention_type and model.attention_type != attention_type:
                continue
            if attention_type:
                filters_applied['attention_type'] = attention_type
            
            # Parameter count filters
            if min_parameters and (model.parameter_count is None or model.parameter_count < min_parameters):
                continue
            if min_parameters:
                filters_applied['min_parameters'] = min_parameters
                
            if max_parameters and (model.parameter_count is None or model.parameter_count > max_parameters):
                continue
            if max_parameters:
                filters_applied['max_parameters'] = max_parameters
            
            # Source filter
            if source and model.source != source:
                continue
            if source:
                filters_applied['source'] = source
            
            # Tags filter (only for database models)
            if tags and model.in_database:
                tag_list = [t.strip() for t in tags.split(',')]
                filters_applied['tags'] = tag_list
                
                # Get model metadata to check tags
                try:
                    db_model = model_manager.get_model(model.model_id)
                    if db_model and db_model.get('metadata'):
                        model_tags = db_model['metadata'].get('tags', [])
                        if not any(tag in model_tags for tag in tag_list):
                            continue
                    else:
                        continue
                except:
                    continue
            
            filtered_models.append(model)
        
        return FilterModelsResponse(
            total_count=len(filtered_models),
            filters_applied=filters_applied,
            models=filtered_models
        )
        
    except Exception as e:
        logging.error(f"Error filtering models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/add/huggingface", response_model=AddModelResponse)
async def add_model_from_huggingface(request: AddModelFromHFRequest):
    """
    Add a model from HuggingFace Hub.
    
    This endpoint imports a model from HuggingFace and adds it to the database.
    If auto_import is True, it will automatically import the model if not found.
    """
    try:
        # Check if model already exists
        existing_model = model_manager.get_model(request.model_uri)
        if existing_model:
            return AddModelResponse(
                success=True,
                model_id=request.model_uri,
                message="Model already exists in database",
                source="database",
                already_existed=True
            )
        
        # Check if in MODEL_DICT
        try:
            model_dict_model = MODEL_DICT.get_model(request.model_uri)
            if model_dict_model:
                return AddModelResponse(
                    success=True,
                    model_id=request.model_uri,
                    message="Model already exists in MODEL_DICT",
                    source="model_dict",
                    already_existed=True
                )
        except Exception as e:
            logging.debug(f"Error checking MODEL_DICT: {e}")
        
        # Import from HuggingFace
        try:
            success = hf_importer.import_model(request.model_uri)
            if success:
                return AddModelResponse(
                    success=True,
                    model_id=request.model_uri,
                    message=f"Successfully imported model from HuggingFace",
                    source="database",
                    already_existed=False
                )
            else:
                return AddModelResponse(
                    success=False,
                    model_id=request.model_uri,
                    message="Failed to import model",
                    source=None,
                    already_existed=False
                )
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower():
                message = "Model is gated and requires access approval"
            elif "not found" in error_msg.lower():
                message = "Model not found on HuggingFace Hub"
            else:
                message = f"Failed to import model: {error_msg}"
            
            return AddModelResponse(
                success=False,
                model_id=request.model_uri,
                message=message,
                source=None,
                already_existed=False
            )
            
    except Exception as e:
        logging.error(f"Error adding model from HuggingFace {request.model_uri}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/add/config", response_model=AddModelResponse)
async def add_model_from_config(request: AddModelFromConfigRequest):
    """
    Add a model from a configuration dictionary.
    
    This endpoint allows adding custom models by providing their configuration
    directly, useful for models not on HuggingFace or custom architectures.
    """
    try:
        # Check if model already exists
        existing_model = model_manager.get_model(request.model_id)
        if existing_model:
            return AddModelResponse(
                success=True,
                model_id=request.model_id,
                message="Model already exists in database",
                source="database",
                already_existed=True
            )
        
        # Validate required config fields
        required_fields = ['hidden_size', 'num_hidden_layers', 'num_attention_heads']
        missing_fields = [f for f in required_fields if f not in request.config]
        if missing_fields:
            return AddModelResponse(
                success=False,
                model_id=request.model_id,
                message=f"Missing required config fields: {', '.join(missing_fields)}",
                source=None,
                already_existed=False
            )
        
        # Detect model type and attention type
        model_type = memory_calculator.detect_model_type(request.config)
        attention_type = memory_calculator.detect_attention_type(request.config)
        
        # Calculate parameter count if not provided
        parameter_count = request.config.get('num_parameters')
        if not parameter_count:
            try:
                # Estimate based on architecture
                hidden_size = request.config['hidden_size']
                num_layers = request.config['num_hidden_layers']
                vocab_size = request.config.get('vocab_size', 50000)
                intermediate_size = request.config.get('intermediate_size', hidden_size * 4)
                
                # Rough estimation
                embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
                attention_params = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O projections
                ffn_params = num_layers * (2 * hidden_size * intermediate_size)  # FFN layers
                layer_norm_params = num_layers * 4 * hidden_size  # Multiple LayerNorms per layer
                
                parameter_count = embedding_params + attention_params + ffn_params + layer_norm_params
            except:
                parameter_count = None
        
        # Add metadata to config if provided
        if request.metadata:
            request.config['_metadata'] = request.metadata
            request.config['_metadata']['source'] = 'custom_config'
            request.config['_metadata']['added_via_api'] = True
        
        # Add to database
        model_id = model_manager.add_model(
            model_id=request.model_id,
            model_type=model_type,
            attention_type=attention_type,
            parameter_count=parameter_count,
            config=request.config,
            source='custom'
        )
        
        if model_id:
            return AddModelResponse(
                success=True,
                model_id=request.model_id,
                message="Successfully added model from configuration",
                source="database",
                already_existed=False
            )
        else:
            return AddModelResponse(
                success=False,
                model_id=request.model_id,
                message="Failed to add model to database",
                source=None,
                already_existed=False
            )
            
    except Exception as e:
        logging.error(f"Error adding model from config {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 


@router.get("/{model_id:path}", response_model=ModelDetailResponse)
async def get_model_details(model_id: str, request: Request):
    """
    Get full details for a specific model, including config and analysis.
    """
    try:
        # First try to get from database
        model_data = model_manager.get_model(model_id)
        
        if model_data:
            # Model found in database
            config = json.loads(model_data.get('config_json', '{}'))
            analysis = json.loads(model_data.get('model_analysis', '{}')) if model_data.get('model_analysis') else None
            
            logo = model_data.get('logo')
            if logo and logo.startswith('logos/'):
                # Convert relative path to full URL
                logo = str(request.url_for('logos', path=logo.split('/')[-1]))

            return ModelDetailResponse(
                model_id=model_data['model_id'],
                name=model_data['model_name'],
                author=model_data['model_id'].split('/')[0] if '/' in model_data['model_id'] else None,
                model_type=model_data.get('model_type') or 'unknown',
                attention_type=model_data.get('attention_type'),
                parameter_count=model_data.get('parameter_count'),
                logo=logo,
                source=model_data.get('source', 'database'),
                in_database=True,
                in_model_dict=False, # This would require checking MODEL_DICT again
                config=config,
                analysis=analysis
            )
        
        # Try to get from MODEL_DICT
        try:
            model_config = MODEL_DICT.get_model(model_id)
            if model_config:
                # Get config from HuggingFace if available
                try:
                    hf_config = hf_loader.get_model_config(model_id)
                    config = hf_config
                except:
                    config = {}
                
                # Extract basic info
                author = model_id.split('/')[0] if '/' in model_id else None
                model_type = getattr(model_config, 'model_type', 'unknown')
                attention_type = getattr(model_config, 'attention_type', None)
                parameter_count = getattr(model_config, 'num_parameters', None)
                
                logo = getattr(model_config, 'logo', None)
                if logo and logo.startswith('logos/'):
                    # Convert relative path to full URL
                    logo = str(request.url_for('logos', path=logo.split('/')[-1]))

                return ModelDetailResponse(
                    model_id=model_id,
                    name=extract_model_name(model_id),
                    author=author,
                    model_type=model_type,
                    attention_type=attention_type,
                    parameter_count=parameter_count,
                    logo=logo,
                    source="model_dict",
                    in_database=False,
                    in_model_dict=True,
                    config=config,
                    analysis=None
                )
        except Exception as e:
            logging.debug(f"Error getting model from MODEL_DICT: {e}")
        
        # Model not found in either source
        raise HTTPException(status_code=404, detail="Model not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting model details for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")