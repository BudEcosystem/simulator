from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class AdminUser(Base):
    __tablename__ = "admin_users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    audit_logs = relationship("AuditLog", back_populates="admin_user")
    feedback_responses = relationship("FeedbackResponse", back_populates="admin_user")

class ApiUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    method = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    country = Column(String, index=True)
    city = Column(String)
    device_type = Column(String)  # desktop, mobile, tablet
    browser = Column(String)
    os = Column(String)
    response_time = Column(Float)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String, index=True)
    
    # Track specific feature usage
    feature_type = Column(String)  # model, hardware, usecase, custom
    feature_id = Column(String)
    custom_slos = Column(JSON)  # For tracking custom SLO usage

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    ip_address = Column(String)
    country = Column(String)
    city = Column(String)
    device_type = Column(String)
    browser = Column(String)
    os = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    page_views = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String, index=True)  # cpu, memory, disk, response_time
    metric_value = Column(Float)
    metric_unit = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metric_metadata = Column(JSON)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("admin_users.id"))
    action = Column(String, index=True)  # create, update, delete, login, logout
    resource_type = Column(String)  # model, hardware, usecase, user
    resource_id = Column(String)
    old_value = Column(JSON)
    new_value = Column(JSON)
    ip_address = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    admin_user = relationship("AdminUser", back_populates="audit_logs")

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    feedback_type = Column(String)  # bug, feature_request, general
    category = Column(String)  # ui, performance, data, other
    rating = Column(Integer)  # 1-5
    title = Column(String)
    message = Column(Text)
    email = Column(String)
    screenshot_url = Column(String)
    status = Column(String, default="pending")  # pending, in_progress, resolved, closed
    priority = Column(String, default="medium")  # low, medium, high, critical
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    responses = relationship("FeedbackResponse", back_populates="feedback")

class FeedbackResponse(Base):
    __tablename__ = "feedback_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    feedback_id = Column(Integer, ForeignKey("user_feedback.id"))
    admin_id = Column(Integer, ForeignKey("admin_users.id"))
    message = Column(Text)
    is_internal = Column(Boolean, default=False)  # Internal notes vs user-visible
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    feedback = relationship("UserFeedback", back_populates="responses")
    admin_user = relationship("AdminUser", back_populates="feedback_responses")

class AnalyticsCache(Base):
    __tablename__ = "analytics_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String, unique=True, index=True)
    cache_value = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)