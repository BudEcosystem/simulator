"""
Hardware management API routes.
Supports the new data structure with instance types and vendor-specific pricing.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.hardware import BudHardware
from src.hardware_recommendation import HardwareRecommendation


router = APIRouter(prefix="/api/hardware", tags=["hardware"])

# Initialize hardware manager
hardware_manager = BudHardware()
recommendation_engine = HardwareRecommendation()


# Pydantic models for request/response
class VendorPricing(BaseModel):
    """On-premise vendor with pricing information."""
    name: str
    price_lower: Optional[float] = None
    price_upper: Optional[float] = None


class CloudInstance(BaseModel):
    """Cloud instance type with pricing."""
    name: str
    price_lower: Optional[float] = None
    price_upper: Optional[float] = None


class CloudSupport(BaseModel):
    """Cloud provider with instance types and regions."""
    name: str
    regions: List[str] = []
    instances: List[CloudInstance] = []


class HardwareCreate(BaseModel):
    """Request model for creating hardware."""
    name: str
    type: str = Field(..., pattern="^(gpu|cpu|accelerator|asic)$")
    manufacturer: Optional[str] = None
    flops: float = Field(..., gt=0)
    memory_size: float = Field(..., gt=0)
    memory_bw: float = Field(..., gt=0)
    icn: Optional[float] = None
    icn_ll: Optional[float] = None
    power: Optional[float] = Field(None, gt=0)
    real_values: bool = True
    url: Optional[str] = None
    description: Optional[str] = None
    on_prem_vendors: List[Union[str, VendorPricing]] = []
    clouds: List[CloudSupport] = []


class HardwareUpdate(BaseModel):
    """Request model for updating hardware."""
    type: Optional[str] = Field(None, pattern="^(gpu|cpu|accelerator|asic)$")
    manufacturer: Optional[str] = None
    flops: Optional[float] = Field(None, gt=0)
    memory_size: Optional[float] = Field(None, gt=0)
    memory_bw: Optional[float] = Field(None, gt=0)
    icn: Optional[float] = None
    icn_ll: Optional[float] = None
    power: Optional[float] = Field(None, gt=0)
    url: Optional[str] = None
    description: Optional[str] = None


class HardwareResponse(BaseModel):
    """Response model for hardware."""
    name: str
    type: str
    manufacturer: Optional[str] = None
    flops: float
    memory_size: float
    memory_bw: float
    icn: Optional[float] = None
    icn_ll: Optional[float] = None
    power: Optional[float] = None
    real_values: bool = True
    url: Optional[str] = None
    description: Optional[str] = None
    on_prem_vendors: List[str] = []
    clouds: List[str] = []
    min_on_prem_price: Optional[float] = None
    max_on_prem_price: Optional[float] = None
    source: str = "manual"


class HardwareDetailResponse(BaseModel):
    """Detailed response including vendor and cloud pricing."""
    name: str
    type: str
    manufacturer: Optional[str] = None
    flops: float
    memory_size: float
    memory_bw: float
    icn: Optional[float] = None
    icn_ll: Optional[float] = None
    power: Optional[float] = None
    real_values: bool = True
    url: Optional[str] = None
    description: Optional[str] = None
    vendors: List[Dict[str, Any]] = []
    clouds: List[Dict[str, Any]] = []
    source: str = "manual"


class RecommendationRequest(BaseModel):
    """Request model for hardware recommendations."""
    total_memory_gb: float = Field(..., gt=0, description="Total memory required in GB")
    model_params_b: Optional[float] = Field(None, gt=0, description="Model parameters in billions")


class RecommendationResponse(BaseModel):
    """Response model for hardware recommendations."""
    hardware_name: str
    nodes_required: int
    memory_per_chip: float
    manufacturer: Optional[str] = None
    type: str


@router.post("", response_model=HardwareResponse)
async def create_hardware(hardware: HardwareCreate):
    """Create new hardware entry."""
    try:
        # Convert Pydantic model to dict for hardware manager
        hardware_data = hardware.dict()
        
        # Convert vendor and cloud data to expected format
        vendors_data = []
        for vendor in hardware_data.get('on_prem_vendors', []):
            if isinstance(vendor, str):
                vendors_data.append(vendor)
            else:
                vendors_data.append(vendor)
        hardware_data['on_prem_vendors'] = vendors_data
        
        # Add hardware
        hardware_manager.add_hardware(hardware_data)
        
        # Get the created hardware
        created = hardware_manager.get_hardware_by_name(hardware.name)
        if not created:
            raise HTTPException(status_code=500, detail="Failed to create hardware")
        
        return HardwareResponse(**created)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("", response_model=List[HardwareResponse])
async def list_hardware(
    type: Optional[str] = Query(None, pattern="^(gpu|cpu|accelerator|asic)$"),
    manufacturer: Optional[str] = None,
    min_memory: Optional[float] = Query(None, gt=0),
    max_memory: Optional[float] = Query(None, gt=0),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all hardware with optional filters."""
    try:
        # Search with filters
        results = hardware_manager.search_hardware(
            type=type,
            manufacturer=manufacturer,
            min_memory=min_memory,
            max_memory=max_memory,
            limit=limit,
            offset=offset
        )
        
        return [HardwareResponse(**hw) for hw in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/filter", response_model=List[HardwareResponse])
async def filter_hardware(
    query: Optional[str] = None,
    type: Optional[List[str]] = Query(None),
    manufacturer: Optional[List[str]] = Query(None),
    min_flops: Optional[float] = Query(None, gt=0),
    max_flops: Optional[float] = Query(None, gt=0),
    min_memory: Optional[float] = Query(None, gt=0),
    max_memory: Optional[float] = Query(None, gt=0),
    min_memory_bw: Optional[float] = Query(None, gt=0),
    max_memory_bw: Optional[float] = Query(None, gt=0),
    min_power: Optional[float] = Query(None, gt=0),
    max_power: Optional[float] = Query(None, gt=0),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    vendor: Optional[List[str]] = Query(None),
    cloud: Optional[List[str]] = Query(None),
    has_real_values: Optional[bool] = None,
    sort_by: str = Query("name", pattern="^(name|flops|memory_size|memory_bw|power|price|perf_per_watt|perf_per_dollar)$"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Advanced hardware filtering with multiple criteria."""
    try:
        results = hardware_manager.search_hardware(
            query=query,
            type=type,
            manufacturer=manufacturer,
            min_flops=min_flops,
            max_flops=max_flops,
            min_memory=min_memory,
            max_memory=max_memory,
            min_memory_bw=min_memory_bw,
            max_memory_bw=max_memory_bw,
            min_power=min_power,
            max_power=max_power,
            min_price=min_price,
            max_price=max_price,
            vendor=vendor,
            cloud=cloud,
            has_real_values=has_real_values,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        return [HardwareResponse(**hw) for hw in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{hardware_name}", response_model=HardwareDetailResponse)
async def get_hardware(hardware_name: str):
    """Get detailed hardware information including vendor and cloud pricing."""
    try:
        # Get basic hardware info
        hardware = hardware_manager.get_hardware_by_name(hardware_name)
        if not hardware:
            raise HTTPException(status_code=404, detail="Hardware not found")
        
        # Get vendor details with pricing
        vendors = hardware_manager.get_hardware_vendors(hardware_name)
        
        # Get cloud details with instance pricing
        clouds = hardware_manager.get_hardware_clouds(hardware_name)
        
        # Build detailed response
        response = HardwareDetailResponse(
            name=hardware['name'],
            type=hardware['type'],
            manufacturer=hardware.get('manufacturer'),
            flops=hardware['flops'],
            memory_size=hardware['memory_size'],
            memory_bw=hardware['memory_bw'],
            icn=hardware.get('icn'),
            icn_ll=hardware.get('icn_ll'),
            power=hardware.get('power'),
            real_values=hardware.get('real_values', True),
            url=hardware.get('url'),
            description=hardware.get('description'),
            vendors=vendors,
            clouds=clouds,
            source=hardware.get('source', 'manual')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.put("/{hardware_name}", response_model=HardwareResponse)
async def update_hardware(hardware_name: str, updates: HardwareUpdate):
    """Update existing hardware."""
    try:
        # Convert to dict and remove None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        # Update hardware
        success = hardware_manager.update_hardware(hardware_name, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Hardware not found")
        
        # Get updated hardware
        updated = hardware_manager.get_hardware_by_name(hardware_name)
        return HardwareResponse(**updated)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.delete("/{hardware_name}")
async def delete_hardware(hardware_name: str, hard_delete: bool = False):
    """Delete hardware (soft delete by default)."""
    try:
        success = hardware_manager.delete_hardware(hardware_name, hard_delete)
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Hardware not found or cannot be deleted"
            )
        
        return {"message": f"Hardware {hardware_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/recommend", response_model=List[RecommendationResponse])
async def recommend_hardware(request: RecommendationRequest):
    """Get hardware recommendations based on memory requirements."""
    try:
        recommendations = recommendation_engine.recommend_hardware(
            total_memory_gb=request.total_memory_gb,
            model_params_b=request.model_params_b
        )
        
        return [RecommendationResponse(**rec) for rec in recommendations]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/vendors/{hardware_name}")
async def get_hardware_vendors(hardware_name: str):
    """Get vendor information with pricing for specific hardware."""
    try:
        vendors = hardware_manager.get_hardware_vendors(hardware_name)
        if not vendors:
            raise HTTPException(
                status_code=404, 
                detail="Hardware not found or no vendors available"
            )
        
        return vendors
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/clouds/{hardware_name}")
async def get_hardware_clouds(hardware_name: str):
    """Get cloud availability with instance types and pricing for specific hardware."""
    try:
        clouds = hardware_manager.get_hardware_clouds(hardware_name)
        if not clouds:
            raise HTTPException(
                status_code=404, 
                detail="Hardware not found or not available on any cloud"
            )
        
        return clouds
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 