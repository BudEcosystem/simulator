from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
from datetime import datetime
from user_agents import parse
from config import settings
from database import init_db, create_admin_user, get_db
from sqlalchemy.orm import Session
from api import auth, analytics, crud, feedback
from utils.analytics import track_api_usage, get_or_create_session

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing admin panel database...")
    init_db()
    create_admin_user()
    print("Admin panel ready!")
    
    yield
    
    # Shutdown
    print("Shutting down admin panel...")

# Create FastAPI app
app = FastAPI(
    title="BudSimulator Admin Panel",
    description="Administration panel for BudSimulator platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analytics middleware
@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    """Track API usage for analytics"""
    start_time = time.time()
    
    # Get request details
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "")
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Track usage (skip auth endpoints and static files)
    if not request.url.path.startswith("/auth") and not request.url.path.startswith("/static"):
        try:
            db = next(get_db())
            
            # Get or create session
            session_id = get_or_create_session(db, {
                "ip_address": ip_address,
                "user_agent": user_agent,
                "country": "Unknown",  # Would use GeoIP in production
                "city": "Unknown"
            })
            
            # Extract feature info from path
            feature_type = None
            feature_id = None
            
            if "/models/" in request.url.path:
                feature_type = "model"
                parts = request.url.path.split("/models/")
                if len(parts) > 1:
                    feature_id = parts[1].split("/")[0]
            elif "/hardware/" in request.url.path:
                feature_type = "hardware"
                parts = request.url.path.split("/hardware/")
                if len(parts) > 1:
                    feature_id = parts[1].split("/")[0]
            elif "/usecases/" in request.url.path:
                feature_type = "usecase"
                parts = request.url.path.split("/usecases/")
                if len(parts) > 1:
                    feature_id = parts[1].split("/")[0]
            
            # Track the usage
            track_api_usage(db, {
                "endpoint": request.url.path,
                "method": request.method,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "response_time": response_time,
                "status_code": response.status_code,
                "session_id": session_id,
                "feature_type": feature_type,
                "feature_id": feature_id
            })
            
            db.close()
        except Exception as e:
            print(f"Error tracking analytics: {e}")
    
    return response

# Include routers
app.include_router(auth.router)
app.include_router(analytics.router)
app.include_router(crud.router)
app.include_router(feedback.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "BudSimulator Admin Panel API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# Public endpoint for feedback submission (no auth required)
@app.post("/public/feedback")
async def submit_feedback(
    feedback_data: dict,
    request: Request,
    db: Session = Depends(get_db)
):
    """Public endpoint for users to submit feedback"""
    from models import UserFeedback
    
    # Get session info
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "")
    
    session_id = get_or_create_session(db, {
        "ip_address": ip_address,
        "user_agent": user_agent,
        "country": "Unknown",
        "city": "Unknown"
    })
    
    # Create feedback
    feedback = UserFeedback(
        session_id=session_id,
        **feedback_data
    )
    
    db.add(feedback)
    db.commit()
    
    return {"message": "Feedback submitted successfully", "id": feedback.id}