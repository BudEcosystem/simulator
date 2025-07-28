"""Health check endpoints for BudSimulator."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import sys
import sqlite3
from pathlib import Path
import psutil
import platform
import time
from datetime import datetime

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    system: Dict[str, Any]
    database: Dict[str, Any]
    services: Dict[str, Any]
    dependencies: Dict[str, Any]


class SystemDiagnostics(BaseModel):
    """Detailed system diagnostics."""
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, Any]
    python_version: str
    platform: str
    uptime: float


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check endpoint."""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {},
        "database": {},
        "services": {},
        "dependencies": {}
    }
    
    try:
        # System information
        health_data["system"] = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # Database check
        db_path = Path(__file__).parent.parent / "data" / "prepopulated.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get model count
                cursor.execute("SELECT COUNT(*) FROM models")
                model_count = cursor.fetchone()[0]
                
                # Get models with complete data
                cursor.execute("""
                    SELECT COUNT(*) FROM models 
                    WHERE config_json IS NOT NULL 
                    AND config_json != ''
                    AND config_json != '{}'
                """)
                complete_models = cursor.fetchone()[0]
                
                conn.close()
                
                health_data["database"] = {
                    "status": "connected",
                    "path": str(db_path),
                    "total_models": model_count,
                    "models_with_config": complete_models,
                    "completeness_percent": round((complete_models / model_count * 100) if model_count > 0 else 0, 2)
                }
            except Exception as e:
                health_data["database"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_data["status"] = "degraded"
        else:
            health_data["database"] = {
                "status": "not_found",
                "path": str(db_path)
            }
            health_data["status"] = "degraded"
            
        # Check services
        health_data["services"] = {
            "api": {
                "status": "running",
                "endpoints": [
                    "/api/models",
                    "/api/hardware/recommendations",
                    "/api/analysis",
                    "/api/logos"
                ]
            }
        }
        
        # Check dependencies
        dependencies = {
            "fastapi": "installed",
            "uvicorn": "installed",
            "pydantic": "installed",
            "genz": "checking..."
        }
        
        try:
            import genz
            dependencies["genz"] = "installed"
        except ImportError:
            dependencies["genz"] = "not_installed"
            health_data["status"] = "degraded"
            
        health_data["dependencies"] = dependencies
        
    except Exception as e:
        health_data["status"] = "unhealthy"
        health_data["error"] = str(e)
        
    return health_data


@router.get("/health/diagnostics", response_model=SystemDiagnostics)
async def system_diagnostics():
    """Detailed system diagnostics."""
    try:
        # Get process info
        process = psutil.Process()
        
        return SystemDiagnostics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage={
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "used_percent": psutil.disk_usage('/').percent
            },
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=f"{platform.system()} {platform.release()}",
            uptime=time.time() - process.create_time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")


@router.get("/health/ready")
async def readiness_check():
    """Simple readiness check."""
    # Check if database exists
    db_path = Path(__file__).parent.parent / "data" / "prepopulated.db"
    
    if not db_path.exists():
        return {"ready": False, "reason": "Database not initialized"}
        
    try:
        # Try to connect to database
        conn = sqlite3.connect(db_path)
        conn.close()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "reason": f"Database error: {str(e)}"} 