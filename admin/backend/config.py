from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # Database
    DATABASE_URL: str = "sqlite:///../../../data/prepopulated.db"
    
    # Admin Credentials
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme123"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Main API URL
    MAIN_API_URL: str = "http://localhost:8000"
    
    # GeoIP Database
    GEOIP_DB_PATH: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3001", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure database directory exists
db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
db_path.parent.mkdir(parents=True, exist_ok=True)