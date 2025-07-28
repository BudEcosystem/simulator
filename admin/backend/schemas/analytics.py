from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class AnalyticsSummary(BaseModel):
    total_users: int
    active_users: int
    users_by_country: List[Dict[str, Any]]
    popular_models: List[Dict[str, Any]]
    popular_hardware: List[Dict[str, Any]]
    popular_usecases: List[Dict[str, Any]]
    device_breakdown: List[Dict[str, Any]]
    browser_breakdown: List[Dict[str, Any]]
    api_usage_timeline: List[Dict[str, Any]]
    custom_slos_usage: int

class SystemHealth(BaseModel):
    metrics: Dict[str, Any]
    avg_response_time: float
    error_count: int
    status: str

class AuditLogEntry(BaseModel):
    id: int
    admin_id: int
    action: str
    resource_type: str
    resource_id: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    ip_address: Optional[str] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True

class AuditLogResponse(BaseModel):
    total: int
    logs: List[AuditLogEntry]
    limit: int
    offset: int

class FeedbackCreate(BaseModel):
    session_id: Optional[str] = None
    feedback_type: str
    category: str
    rating: Optional[int] = None
    title: str
    message: str
    email: Optional[str] = None
    screenshot_url: Optional[str] = None

class FeedbackUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None

class FeedbackResponseCreate(BaseModel):
    message: str
    is_internal: bool = False

class Feedback(BaseModel):
    id: int
    session_id: Optional[str]
    feedback_type: str
    category: str
    rating: Optional[int]
    title: str
    message: str
    email: Optional[str]
    screenshot_url: Optional[str]
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime
    responses: List[Any] = []
    
    class Config:
        from_attributes = True