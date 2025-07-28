from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from database import get_db
from schemas.analytics import Feedback, FeedbackCreate, FeedbackUpdate, FeedbackResponseCreate
from utils.auth import get_current_active_user
from utils.audit import log_action
from models import AdminUser, UserFeedback, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback Management"])

@router.get("/", response_model=List[Feedback])
async def list_feedback(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    feedback_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all feedback with filters"""
    query = db.query(UserFeedback)
    
    if status:
        query = query.filter(UserFeedback.status == status)
    if priority:
        query = query.filter(UserFeedback.priority == priority)
    if category:
        query = query.filter(UserFeedback.category == category)
    if feedback_type:
        query = query.filter(UserFeedback.feedback_type == feedback_type)
    
    feedback_list = query.order_by(UserFeedback.created_at.desc()).offset(offset).limit(limit).all()
    
    # Load responses for each feedback
    for feedback in feedback_list:
        feedback.responses = db.query(FeedbackResponse).filter(
            FeedbackResponse.feedback_id == feedback.id
        ).all()
    
    return feedback_list

@router.get("/{feedback_id}", response_model=Feedback)
async def get_feedback(
    feedback_id: int,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific feedback details"""
    feedback = db.query(UserFeedback).filter(UserFeedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Load responses
    feedback.responses = db.query(FeedbackResponse).filter(
        FeedbackResponse.feedback_id == feedback.id
    ).all()
    
    return feedback

@router.post("/", response_model=Feedback)
async def create_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """Create new feedback (public endpoint for users)"""
    feedback = UserFeedback(**feedback_data.dict())
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    
    return feedback

@router.put("/{feedback_id}", response_model=Feedback)
async def update_feedback(
    feedback_id: int,
    feedback_update: FeedbackUpdate,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update feedback status/priority"""
    feedback = db.query(UserFeedback).filter(UserFeedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Track old values for audit
    old_values = {
        "status": feedback.status,
        "priority": feedback.priority
    }
    
    # Update fields
    if feedback_update.status:
        feedback.status = feedback_update.status
    if feedback_update.priority:
        feedback.priority = feedback_update.priority
    
    feedback.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(feedback)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="update",
        resource_type="feedback",
        resource_id=str(feedback_id),
        old_value=old_values,
        new_value={"status": feedback.status, "priority": feedback.priority}
    )
    
    return feedback

@router.post("/{feedback_id}/respond")
async def respond_to_feedback(
    feedback_id: int,
    response_data: FeedbackResponseCreate,
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add response to feedback"""
    feedback = db.query(UserFeedback).filter(UserFeedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Create response
    response = FeedbackResponse(
        feedback_id=feedback_id,
        admin_id=current_user.id,
        message=response_data.message,
        is_internal=response_data.is_internal
    )
    
    db.add(response)
    
    # Update feedback status if it's the first response
    if feedback.status == "pending":
        feedback.status = "in_progress"
    
    db.commit()
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="respond",
        resource_type="feedback",
        resource_id=str(feedback_id),
        new_value={
            "message": response_data.message[:100] + "..." if len(response_data.message) > 100 else response_data.message,
            "is_internal": response_data.is_internal
        }
    )
    
    return {"message": "Response added successfully"}

@router.get("/stats/summary")
async def get_feedback_stats(
    current_user: AdminUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get feedback statistics"""
    from sqlalchemy import func
    
    # Count by status
    status_counts = db.query(
        UserFeedback.status,
        func.count().label('count')
    ).group_by(UserFeedback.status).all()
    
    # Count by priority
    priority_counts = db.query(
        UserFeedback.priority,
        func.count().label('count')
    ).group_by(UserFeedback.priority).all()
    
    # Count by category
    category_counts = db.query(
        UserFeedback.category,
        func.count().label('count')
    ).group_by(UserFeedback.category).all()
    
    # Average rating
    avg_rating = db.query(func.avg(UserFeedback.rating)).filter(
        UserFeedback.rating != None
    ).scalar()
    
    # Recent feedback (last 7 days)
    recent_count = db.query(func.count()).filter(
        UserFeedback.created_at >= datetime.utcnow() - timedelta(days=7)
    ).scalar()
    
    return {
        "status_breakdown": {s: c for s, c in status_counts},
        "priority_breakdown": {p: c for p, c in priority_counts},
        "category_breakdown": {c: cnt for c, cnt in category_counts},
        "average_rating": round(avg_rating, 2) if avg_rating else None,
        "recent_feedback_count": recent_count
    }