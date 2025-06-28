from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Any, Dict
import httpx
import json
from datetime import datetime
from database import get_db
from utils.auth import get_current_active_user
from utils.audit import log_action
from utils.admin_storage import admin_storage
from models import AdminUser
from config import settings

router = APIRouter(prefix="/crud", tags=["CRUD Operations"])

# Generic CRUD handler that uses the main API
async def proxy_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
):
    """Proxy requests to the main API"""
    async with httpx.AsyncClient() as client:
        url = f"{settings.MAIN_API_URL}{endpoint}"
        
        try:
            if method == "GET":
                response = await client.get(url, params=params)
            elif method == "POST":
                response = await client.post(url, json=data)
            elif method == "PUT":
                response = await client.put(url, json=data)
            elif method == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Error connecting to main API: {str(e)}")

# Models CRUD
@router.get("/models")
async def list_models(
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: AdminUser = Depends(get_current_active_user)
):
    """List all models with search and pagination"""
    # Get models from the main API
    response = await proxy_request("/api/models/list")
    
    # Extract models array from response
    models = response.get("models", [])
    
    # Apply overrides from admin storage
    overrides = admin_storage.get_model_overrides()
    model_dict = {m["model_id"]: m for m in models}
    
    # Update with overrides
    for model_id, override in overrides.items():
        if model_id in model_dict:
            model_dict[model_id].update(override)
        else:
            # Add new models created in admin
            model_dict[model_id] = override
    
    # Convert to list and filter out deleted models
    models = [m for m in model_dict.values() if not m.get("is_deleted", False)]
    
    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        models = [m for m in models if 
                  search_lower in m.get("model_id", "").lower() or 
                  search_lower in m.get("model_name", "").lower() or
                  search_lower in m.get("name", "").lower() or
                  search_lower in m.get("author", "").lower()]
    
    # Apply pagination
    total = len(models)
    models = models[offset:offset + limit]
    
    return {"models": models, "total": total, "limit": limit, "offset": offset}

@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    current_user: AdminUser = Depends(get_current_active_user)
):
    """Get specific model details"""
    return await proxy_request(f"/api/models/{model_id}")

@router.post("/models")
async def create_model(
    model_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new model"""
    # Use the add config endpoint
    result = await proxy_request("/api/models/add/config", method="POST", data=model_data)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="create",
        resource_type="model",
        resource_id=model_data.get("model_id", "unknown"),
        new_value=model_data
    )
    
    return result

@router.put("/models/{model_id}")
async def update_model(
    model_id: str,
    model_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update an existing model"""
    # Since main API doesn't support updates, use admin storage
    try:
        # Get current model data
        response = await proxy_request("/api/models/list")
        models = response.get("models", [])
        current_model = next((m for m in models if m["model_id"] == model_id), None)
        
        if not current_model:
            # Check if it's an admin-created model
            override = admin_storage.get_model_override(model_id)
            if not override:
                raise HTTPException(status_code=404, detail="Model not found")
            current_model = override
        
        # Save the update as an override
        admin_storage.save_model_override(model_id, model_data)
        
        # Log the action
        log_action(
            db=db,
            admin_user=current_user,
            action="update",
            resource_type="model",
            resource_id=model_id,
            old_value=current_model,
            new_value=model_data
        )
        
        return {"message": "Model updated successfully", "model": model_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a model"""
    # Since main API doesn't support delete, use admin storage to mark as deleted
    try:
        # Get current model data for audit
        response = await proxy_request("/api/models/list")
        models = response.get("models", [])
        current_model = next((m for m in models if m["model_id"] == model_id), None)
        
        if not current_model:
            # Check if it's an admin-created model
            override = admin_storage.get_model_override(model_id)
            if not override:
                raise HTTPException(status_code=404, detail="Model not found")
            current_model = override
        
        # Mark as deleted in admin storage
        deleted_data = {**current_model, "is_deleted": True, "deleted_at": datetime.utcnow().isoformat()}
        admin_storage.save_model_override(model_id, deleted_data)
        
        # Log the action
        log_action(
            db=db,
            admin_user=current_user,
            action="delete",
            resource_type="model",
            resource_id=model_id,
            old_value=current_model
        )
        
        return {"message": "Model deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Hardware CRUD
@router.get("/hardware")
async def list_hardware(
    search: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    manufacturer: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: AdminUser = Depends(get_current_active_user)
):
    """List all hardware with filters"""
    params = {"limit": limit, "offset": offset}
    if search:
        params["search"] = search
    if type:
        params["type"] = type
    if manufacturer:
        params["manufacturer"] = manufacturer
    
    return await proxy_request("/api/hardware", params=params)

@router.get("/hardware/{hardware_name}")
async def get_hardware(
    hardware_name: str,
    current_user: AdminUser = Depends(get_current_active_user)
):
    """Get specific hardware details"""
    return await proxy_request(f"/api/hardware/{hardware_name}")

@router.post("/hardware")
async def create_hardware(
    hardware_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new hardware"""
    result = await proxy_request("/api/hardware", method="POST", data=hardware_data)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="create",
        resource_type="hardware",
        resource_id=hardware_data.get("name", "unknown"),
        new_value=hardware_data
    )
    
    return result

@router.put("/hardware/{hardware_name}")
async def update_hardware(
    hardware_name: str,
    hardware_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update existing hardware"""
    # Get old value for audit
    old_value = await proxy_request(f"/api/hardware/{hardware_name}")
    
    result = await proxy_request(f"/api/hardware/{hardware_name}", method="PUT", data=hardware_data)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="update",
        resource_type="hardware",
        resource_id=hardware_name,
        old_value=old_value,
        new_value=hardware_data
    )
    
    return result

@router.delete("/hardware/{hardware_name}")
async def delete_hardware(
    hardware_name: str,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete hardware"""
    # Get old value for audit
    old_value = await proxy_request(f"/api/hardware/{hardware_name}")
    
    result = await proxy_request(f"/api/hardware/{hardware_name}", method="DELETE")
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="delete",
        resource_type="hardware",
        resource_id=hardware_name,
        old_value=old_value
    )
    
    return result

# Usecases CRUD
@router.get("/usecases")
async def list_usecases(
    industry: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: AdminUser = Depends(get_current_active_user)
):
    """List all usecases with filters"""
    params = {"limit": limit, "offset": offset}
    if industry:
        params["industry"] = industry
    if search:
        params["search"] = search
    
    return await proxy_request("/api/usecases", params=params)

@router.get("/usecases/{usecase_id}")
async def get_usecase(
    usecase_id: str,
    current_user: AdminUser = Depends(get_current_active_user)
):
    """Get specific usecase details"""
    return await proxy_request(f"/api/usecases/{usecase_id}")

@router.post("/usecases")
async def create_usecase(
    usecase_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new usecase"""
    result = await proxy_request("/api/usecases", method="POST", data=usecase_data)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="create",
        resource_type="usecase",
        resource_id=usecase_data.get("unique_id", "unknown"),
        new_value=usecase_data
    )
    
    return result

@router.put("/usecases/{usecase_id}")
async def update_usecase(
    usecase_id: str,
    usecase_data: Dict[str, Any],
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update existing usecase"""
    # Get old value for audit
    old_value = await proxy_request(f"/api/usecases/{usecase_id}")
    
    result = await proxy_request(f"/api/usecases/{usecase_id}", method="PUT", data=usecase_data)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="update",
        resource_type="usecase",
        resource_id=usecase_id,
        old_value=old_value,
        new_value=usecase_data
    )
    
    return result

@router.delete("/usecases/{usecase_id}")
async def delete_usecase(
    usecase_id: str,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete usecase"""
    # Get old value for audit
    old_value = await proxy_request(f"/api/usecases/{usecase_id}")
    
    result = await proxy_request(f"/api/usecases/{usecase_id}", method="DELETE")
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="delete",
        resource_type="usecase",
        resource_id=usecase_id,
        old_value=old_value
    )
    
    return result