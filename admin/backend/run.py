#!/usr/bin/env python3
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8001))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting Admin Panel API on {host}:{port}")
    print(f"Reload mode: {reload}")
    print(f"API Documentation: http://localhost:{port}/docs")
    
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )