"""
Hardware optimization endpoints for usecases using GenZ
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.optimization.hardware_optimizer import (
    find_best_hardware_for_usecase,
    HardwareOptimizer
)
from src.usecases import BudUsecases

router = APIRouter(prefix="/api/usecases", tags=["usecases"])

# Initialize managers
usecase_manager = BudUsecases()
optimizer = HardwareOptimizer()


class OptimizationRequest(BaseModel):
    """Request model for hardware optimization"""
    batch_sizes: List[int] = Field(default=[1, 4, 8, 16], description="Batch sizes to evaluate")
    model_sizes: List[str] = Field(default=["1B", "3B", "8B", "32B", "70B"], description="Model size categories")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    optimization_mode: str = Field(default="balanced", description="cost, performance, or balanced")


class OptimizationResult(BaseModel):
    """Result of hardware optimization"""
    model_id: str
    model_size: str
    hardware_type: str
    num_nodes: int
    parallelism: str
    batch_size: int
    achieved_ttft: float
    achieved_e2e: float
    required_ttft: float
    required_e2e: float
    meets_slo: bool
    cost_per_hour: float
    cost_per_request: float
    throughput: float
    utilization: float
    efficiency_score: float


class OptimizationResponse(BaseModel):
    """Response for optimization endpoint"""
    usecase: Dict[str, Any]
    configurations: List[OptimizationResult]
    optimization_mode: str
    summary: Dict[str, Any]


@router.post("/{unique_id}/optimize-hardware", response_model=OptimizationResponse)
async def optimize_hardware_for_usecase(
    unique_id: str,
    request: OptimizationRequest = OptimizationRequest()
):
    """
    Find optimal hardware configurations for a usecase using GenZ performance modeling.
    
    This endpoint:
    - Tests multiple model sizes (1B to 100B+)
    - Evaluates different batch sizes
    - Finds minimum hardware requirements
    - Returns cost-optimized configurations that meet SLO requirements
    
    The optimization uses GenZ for accurate performance modeling considering:
    - Memory bandwidth limitations
    - Compute requirements
    - Network communication overhead
    - Parallelism strategies (TP/PP/EP)
    """
    
    # Get usecase
    try:
        usecase = usecase_manager.get_usecase(unique_id)
        if not usecase:
            raise HTTPException(status_code=404, detail=f"Usecase {unique_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching usecase: {str(e)}")
    
    # Convert usecase to dict if needed
    if hasattr(usecase, 'to_dict'):
        usecase_dict = usecase.to_dict()
    else:
        usecase_dict = usecase
    
    # Ensure required fields
    required_fields = ['input_tokens_max', 'output_tokens_max', 'ttft_max', 'e2e_max']
    for field in required_fields:
        if field not in usecase_dict:
            raise HTTPException(
                status_code=400, 
                detail=f"Usecase missing required field: {field}"
            )
    
    # Run optimization for each batch size
    all_configs = []
    
    for batch_size in request.batch_sizes:
        try:
            configs = find_best_hardware_for_usecase(
                usecase=usecase_dict,
                batch_size=batch_size,
                model_sizes=request.model_sizes
            )
            
            # Convert to response models
            for config in configs:
                all_configs.append(OptimizationResult(**config))
                
        except Exception as e:
            print(f"Error optimizing for batch size {batch_size}: {str(e)}")
            continue
    
    # Apply optimization mode filtering
    if request.optimization_mode == "cost":
        # Sort by cost per request
        all_configs.sort(key=lambda x: x.cost_per_request)
    elif request.optimization_mode == "performance":
        # Sort by TTFT (lowest latency)
        all_configs.sort(key=lambda x: x.achieved_ttft)
    else:  # balanced
        # Sort by efficiency score
        all_configs.sort(key=lambda x: x.efficiency_score)
    
    # Limit results
    final_configs = all_configs[:request.max_results]
    
    # Generate summary
    summary = {
        "total_configurations_found": len(all_configs),
        "configurations_returned": len(final_configs),
        "model_sizes_evaluated": list(set(c.model_size for c in final_configs)),
        "hardware_types_found": list(set(c.hardware_type for c in final_configs)),
        "batch_sizes_evaluated": list(set(c.batch_size for c in final_configs)),
        "min_cost_per_request": min([c.cost_per_request for c in final_configs]) if final_configs else None,
        "min_ttft_achieved": min([c.achieved_ttft for c in final_configs]) if final_configs else None,
        "max_throughput": max([c.throughput for c in final_configs]) if final_configs else None
    }
    
    return OptimizationResponse(
        usecase=usecase_dict,
        configurations=final_configs,
        optimization_mode=request.optimization_mode,
        summary=summary
    )


@router.get("/{unique_id}/quick-optimization")
async def quick_hardware_optimization(
    unique_id: str,
    batch_size: int = Query(default=1, ge=1, le=128),
    model_size: str = Query(default="8B", description="Model size category")
):
    """
    Quick optimization for a single model size and batch size.
    
    Useful for real-time recommendations without full optimization sweep.
    """
    
    # Validate model size
    if model_size not in optimizer.model_registry:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model size. Options: {list(optimizer.model_registry.keys())}"
        )
    
    # Get usecase
    try:
        usecase = usecase_manager.get_usecase(unique_id)
        if not usecase:
            raise HTTPException(status_code=404, detail=f"Usecase {unique_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching usecase: {str(e)}")
    
    # Convert to dict
    if hasattr(usecase, 'to_dict'):
        usecase_dict = usecase.to_dict()
    else:
        usecase_dict = usecase
    
    # Run optimization for single configuration
    try:
        configs = find_best_hardware_for_usecase(
            usecase=usecase_dict,
            batch_size=batch_size,
            model_sizes=[model_size]
        )
        
        if not configs:
            return {
                "message": "No configurations found that meet SLO requirements",
                "usecase": usecase_dict,
                "model_size": model_size,
                "batch_size": batch_size
            }
        
        # Return best configuration
        best = configs[0]
        return {
            "usecase": usecase_dict,
            "optimal_configuration": OptimizationResult(**best),
            "alternatives": [OptimizationResult(**c) for c in configs[1:5]]  # Top 5
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )