from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
from database import get_db
from schemas.analytics import AnalyticsSummary, SystemHealth, AuditLogResponse
from utils.auth import get_current_active_user
from utils.analytics import get_analytics_summary, get_system_health
from utils.audit import get_audit_logs
from models import AdminUser

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get analytics summary for dashboard"""
    summary = get_analytics_summary(db, days)
    return summary

@router.get("/health", response_model=SystemHealth)
async def get_health(
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get system health metrics"""
    health = get_system_health(db)
    return health

@router.get("/audit-logs", response_model=AuditLogResponse)
async def get_logs(
    admin_id: Optional[int] = Query(None),
    resource_type: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get audit logs with filtering"""
    logs = get_audit_logs(
        db=db,
        admin_id=admin_id,
        resource_type=resource_type,
        action=action,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    return logs

@router.get("/usage/timeline")
async def get_usage_timeline(
    days: int = Query(30, description="Number of days to analyze"),
    feature_type: Optional[str] = Query(None, description="Filter by feature type"),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get API usage timeline"""
    from sqlalchemy import func
    from models import ApiUsage
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(
        func.date(ApiUsage.timestamp).label('date'),
        func.count().label('count')
    ).filter(ApiUsage.timestamp >= start_date)
    
    if feature_type:
        query = query.filter(ApiUsage.feature_type == feature_type)
    
    timeline = query.group_by(func.date(ApiUsage.timestamp)).all()
    
    return [{"date": str(d), "count": c} for d, c in timeline]

@router.get("/usage/top-endpoints")
async def get_top_endpoints(
    limit: int = Query(10, le=50),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get most used API endpoints"""
    from sqlalchemy import func
    from models import ApiUsage
    
    endpoints = db.query(
        ApiUsage.endpoint,
        ApiUsage.method,
        func.count().label('count'),
        func.avg(ApiUsage.response_time).label('avg_response_time')
    ).group_by(
        ApiUsage.endpoint,
        ApiUsage.method
    ).order_by(func.count().desc()).limit(limit).all()
    
    return [{
        "endpoint": e.endpoint,
        "method": e.method,
        "count": e.count,
        "avg_response_time": round(e.avg_response_time, 2) if e.avg_response_time else 0
    } for e in endpoints]

@router.get("/users/active")
async def get_active_users(
    hours: int = Query(24, description="Hours to look back"),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get currently active users"""
    from models import UserSession
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    active_sessions = db.query(UserSession).filter(
        UserSession.start_time >= cutoff_time,
        UserSession.is_active == True
    ).all()
    
    return [{
        "session_id": s.session_id,
        "country": s.country,
        "city": s.city,
        "device_type": s.device_type,
        "browser": s.browser,
        "os": s.os,
        "start_time": s.start_time.isoformat(),
        "page_views": s.page_views
    } for s in active_sessions]