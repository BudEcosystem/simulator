from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class AdminUserBase(BaseModel):
    username: str
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False

class AdminUserCreate(AdminUserBase):
    password: str

class AdminUserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class AdminUser(AdminUserBase):
    id: int
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    username: str
    password: str