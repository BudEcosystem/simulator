"""
Main FastAPI application for BudSimulator API.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routers import models, hardware, usecases, usecases_optimization, training
from .health import router as health_router
from src.hardware_registry import HardwareRegistry

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    _SLOWAPI_AVAILABLE = True
except ImportError:
    limiter = None
    _SLOWAPI_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="BudSimulator Model API",
    description="API for analyzing and managing AI model configurations and memory requirements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

if _SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get allowed origins from environment or use defaults
import os
_CORS_ORIGINS = os.getenv(
    "BUD_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:8000,http://localhost:8001"
).split(",")

# Add TrustedHost middleware for proxy support
# In production, set BUD_ALLOWED_HOSTS environment variable
_ALLOWED_HOSTS = os.getenv("BUD_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_ALLOWED_HOSTS
)

# Configure CORS - NEVER use wildcards in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "Accept", "Origin"],
)

# Mount static files for logos
logos_dir = Path(__file__).parent.parent / "logos"
logos_dir.mkdir(exist_ok=True)
app.mount("/logos", StaticFiles(directory=str(logos_dir)), name="logos")

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(hardware.router)
app.include_router(usecases.router)
app.include_router(usecases_optimization.router)
app.include_router(training.router)  # Training/simulator endpoints
app.include_router(health_router, prefix="/api", tags=["health"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "BudSimulator Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }



# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize hardware registry and apply runtime patches on startup."""
    HardwareRegistry.initialize()

    # Patch MODEL_DICT at startup instead of at import time
    try:
        from .routers.models import apply_model_dict_patch
        apply_model_dict_patch()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"MODEL_DICT patching failed: {e}")