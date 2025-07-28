from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from database import get_db
from schemas.auth import Token, AdminUser, AdminUserCreate, LoginRequest
from utils.auth import (
    authenticate_user, create_access_token, get_current_active_user,
    get_password_hash
)
from models import AdminUser as AdminUserModel
from utils.audit import log_action

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login endpoint for admin users"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Log the login
    log_action(
        db=db,
        admin_user=user,
        action="login",
        resource_type="auth",
        resource_id=str(user.id)
    )
    
    access_token_expires = timedelta(minutes=1440)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Alternative login endpoint that accepts JSON"""
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Log the login
    log_action(
        db=db,
        admin_user=user,
        action="login",
        resource_type="auth",
        resource_id=str(user.id)
    )
    
    access_token_expires = timedelta(minutes=1440)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=AdminUser)
async def get_current_user(
    current_user: AdminUserModel = Depends(get_current_active_user)
):
    """Get current authenticated user"""
    return current_user

@router.post("/logout")
async def logout(
    current_user: AdminUserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Logout endpoint"""
    log_action(
        db=db,
        admin_user=current_user,
        action="logout",
        resource_type="auth",
        resource_id=str(current_user.id)
    )
    
    return {"message": "Successfully logged out"}

@router.post("/users", response_model=AdminUser)
async def create_admin_user(
    user_data: AdminUserCreate,
    db: Session = Depends(get_db),
    current_user: AdminUserModel = Depends(get_current_active_user)
):
    """Create a new admin user (superuser only)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can create new admin users"
        )
    
    # Check if user already exists
    existing_user = db.query(AdminUserModel).filter(
        (AdminUserModel.username == user_data.username) |
        (AdminUserModel.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    new_user = AdminUserModel(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        is_active=user_data.is_active,
        is_superuser=user_data.is_superuser
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Log the action
    log_action(
        db=db,
        admin_user=current_user,
        action="create",
        resource_type="admin_user",
        resource_id=str(new_user.id),
        new_value={"username": new_user.username, "email": new_user.email}
    )
    
    return new_user