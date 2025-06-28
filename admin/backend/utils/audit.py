from typing import Optional, Any, Dict
from sqlalchemy.orm import Session
from datetime import datetime
import json
from models import AuditLog, AdminUser

def log_action(
    db: Session,
    admin_user: AdminUser,
    action: str,
    resource_type: str,
    resource_id: str,
    old_value: Optional[Any] = None,
    new_value: Optional[Any] = None,
    ip_address: Optional[str] = None
):
    """Log an admin action for audit trail"""
    
    # Convert values to JSON if they're dictionaries
    if isinstance(old_value, dict):
        old_value = json.dumps(old_value)
    if isinstance(new_value, dict):
        new_value = json.dumps(new_value)
    
    audit = AuditLog(
        admin_id=admin_user.id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        old_value=old_value,
        new_value=new_value,
        ip_address=ip_address,
        timestamp=datetime.utcnow()
    )
    
    db.add(audit)
    db.commit()
    
    return audit

def get_audit_logs(
    db: Session,
    admin_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    action: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict:
    """Get audit logs with filtering"""
    
    query = db.query(AuditLog)
    
    if admin_id:
        query = query.filter(AuditLog.admin_id == admin_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if action:
        query = query.filter(AuditLog.action == action)
    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)
    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)
    
    total = query.count()
    logs = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "logs": logs,
        "limit": limit,
        "offset": offset
    }