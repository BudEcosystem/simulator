"""
Main FastAPI application for BudSimulator API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routers import models, hardware
from src.hardware_registry import HardwareRegistry

# Create FastAPI app
app = FastAPI(
    title="BudSimulator Model API",
    description="API for analyzing and managing AI model configurations and memory requirements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for logos
logos_dir = Path(__file__).parent.parent / "logos"
logos_dir.mkdir(exist_ok=True)
app.mount("/logos", StaticFiles(directory=str(logos_dir)), name="logos")

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(hardware.router, tags=["hardware"])  # hardware router already has /api/hardware prefix

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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize hardware registry on startup."""
    HardwareRegistry.initialize() 