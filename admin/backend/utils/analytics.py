from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, distinct
from user_agents import parse
import hashlib
import json
from models import ApiUsage, UserSession, SystemMetrics, AnalyticsCache

def get_or_create_session(db: Session, request_data: dict) -> str:
    """Get or create user session based on IP and user agent"""
    ip_address = request_data.get("ip_address", "")
    user_agent = request_data.get("user_agent", "")
    
    # Create session ID from IP and user agent
    session_data = f"{ip_address}:{user_agent}"
    session_id = hashlib.md5(session_data.encode()).hexdigest()
    
    # Check if session exists and is active
    session = db.query(UserSession).filter(
        UserSession.session_id == session_id,
        UserSession.is_active == True
    ).first()
    
    if not session:
        # Parse user agent
        ua = parse(user_agent)
        
        # Create new session
        session = UserSession(
            session_id=session_id,
            ip_address=ip_address,
            device_type=get_device_type(ua),
            browser=ua.browser.family,
            os=ua.os.family,
            country=request_data.get("country", "Unknown"),
            city=request_data.get("city", "Unknown")
        )
        db.add(session)
        db.commit()
    else:
        # Update page views
        session.page_views += 1
        db.commit()
    
    return session_id

def get_device_type(user_agent) -> str:
    """Determine device type from user agent"""
    if user_agent.is_mobile:
        return "mobile"
    elif user_agent.is_tablet:
        return "tablet"
    elif user_agent.is_pc:
        return "desktop"
    else:
        return "other"

def track_api_usage(db: Session, request_data: dict):
    """Track API usage"""
    ua = parse(request_data.get("user_agent", ""))
    
    usage = ApiUsage(
        endpoint=request_data.get("endpoint"),
        method=request_data.get("method"),
        ip_address=request_data.get("ip_address"),
        user_agent=request_data.get("user_agent"),
        country=request_data.get("country", "Unknown"),
        city=request_data.get("city", "Unknown"),
        device_type=get_device_type(ua),
        browser=ua.browser.family,
        os=ua.os.family,
        response_time=request_data.get("response_time", 0),
        status_code=request_data.get("status_code", 200),
        session_id=request_data.get("session_id"),
        feature_type=request_data.get("feature_type"),
        feature_id=request_data.get("feature_id"),
        custom_slos=request_data.get("custom_slos")
    )
    
    db.add(usage)
    db.commit()

def get_analytics_summary(db: Session, days: int = 30) -> Dict:
    """Get analytics summary for dashboard"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Check cache first
    cache_key = f"analytics_summary_{days}"
    cached = db.query(AnalyticsCache).filter(
        AnalyticsCache.cache_key == cache_key,
        AnalyticsCache.expires_at > datetime.utcnow()
    ).first()
    
    if cached:
        return cached.cache_value
    
    # Total users (unique sessions)
    total_users = db.query(func.count(distinct(UserSession.session_id))).scalar()
    
    # Active users (in last 24 hours)
    active_users = db.query(func.count(distinct(UserSession.session_id))).filter(
        UserSession.start_time >= datetime.utcnow() - timedelta(hours=24)
    ).scalar()
    
    # Users by country
    users_by_country = db.query(
        UserSession.country,
        func.count(distinct(UserSession.session_id)).label('count')
    ).group_by(UserSession.country).all()
    
    # Popular models
    popular_models = db.query(
        ApiUsage.feature_id,
        func.count().label('count')
    ).filter(
        ApiUsage.feature_type == 'model',
        ApiUsage.timestamp >= start_date
    ).group_by(ApiUsage.feature_id).order_by(func.count().desc()).limit(10).all()
    
    # Popular hardware
    popular_hardware = db.query(
        ApiUsage.feature_id,
        func.count().label('count')
    ).filter(
        ApiUsage.feature_type == 'hardware',
        ApiUsage.timestamp >= start_date
    ).group_by(ApiUsage.feature_id).order_by(func.count().desc()).limit(10).all()
    
    # Popular use cases
    popular_usecases = db.query(
        ApiUsage.feature_id,
        func.count().label('count')
    ).filter(
        ApiUsage.feature_type == 'usecase',
        ApiUsage.timestamp >= start_date
    ).group_by(ApiUsage.feature_id).order_by(func.count().desc()).limit(10).all()
    
    # Device breakdown
    device_breakdown = db.query(
        UserSession.device_type,
        func.count(distinct(UserSession.session_id)).label('count')
    ).group_by(UserSession.device_type).all()
    
    # Browser breakdown
    browser_breakdown = db.query(
        UserSession.browser,
        func.count(distinct(UserSession.session_id)).label('count')
    ).group_by(UserSession.browser).all()
    
    # API usage over time
    api_usage_timeline = db.query(
        func.date(ApiUsage.timestamp).label('date'),
        func.count().label('count')
    ).filter(
        ApiUsage.timestamp >= start_date
    ).group_by(func.date(ApiUsage.timestamp)).all()
    
    # Custom SLOs usage
    custom_slos_count = db.query(func.count()).filter(
        ApiUsage.custom_slos != None,
        ApiUsage.timestamp >= start_date
    ).scalar()
    
    result = {
        "total_users": total_users or 0,
        "active_users": active_users or 0,
        "users_by_country": [{"country": c, "count": cnt} for c, cnt in users_by_country],
        "popular_models": [{"id": id, "count": cnt} for id, cnt in popular_models if id],
        "popular_hardware": [{"id": id, "count": cnt} for id, cnt in popular_hardware if id],
        "popular_usecases": [{"id": id, "count": cnt} for id, cnt in popular_usecases if id],
        "device_breakdown": [{"device": d, "count": cnt} for d, cnt in device_breakdown],
        "browser_breakdown": [{"browser": b, "count": cnt} for b, cnt in browser_breakdown],
        "api_usage_timeline": [{"date": str(d), "count": cnt} for d, cnt in api_usage_timeline],
        "custom_slos_usage": custom_slos_count or 0
    }
    
    # Update or create cache entry
    try:
        # Try to update existing cache
        existing_cache = db.query(AnalyticsCache).filter(
            AnalyticsCache.cache_key == cache_key
        ).first()
        
        if existing_cache:
            # Update the existing entry
            db.query(AnalyticsCache).filter(
                AnalyticsCache.cache_key == cache_key
            ).update({
                "cache_value": result,
                "expires_at": datetime.utcnow() + timedelta(hours=1)
            })
        else:
            # Create new entry
            cache_entry = AnalyticsCache(
                cache_key=cache_key,
                cache_value=result,
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.add(cache_entry)
        
        db.commit()
    except Exception as e:
        db.rollback()
        # If there's still an error, just return the result without caching
        print(f"Warning: Failed to cache analytics: {e}")
    
    return result

def get_system_health(db: Session) -> Dict:
    """Get system health metrics"""
    # Get latest metrics
    latest_metrics = {}
    metric_types = ['cpu', 'memory', 'disk', 'response_time']
    
    for metric_type in metric_types:
        metric = db.query(SystemMetrics).filter(
            SystemMetrics.metric_type == metric_type
        ).order_by(SystemMetrics.timestamp.desc()).first()
        
        if metric:
            latest_metrics[metric_type] = {
                "value": metric.metric_value,
                "unit": metric.metric_unit,
                "timestamp": metric.timestamp.isoformat()
            }
    
    # Calculate average response time for last hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    avg_response_time = db.query(func.avg(ApiUsage.response_time)).filter(
        ApiUsage.timestamp >= one_hour_ago
    ).scalar()
    
    # Count errors in last hour
    error_count = db.query(func.count()).filter(
        ApiUsage.timestamp >= one_hour_ago,
        ApiUsage.status_code >= 400
    ).scalar()
    
    return {
        "metrics": latest_metrics,
        "avg_response_time": avg_response_time or 0,
        "error_count": error_count or 0,
        "status": "healthy" if error_count < 10 else "degraded"
    }