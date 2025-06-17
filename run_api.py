#!/usr/bin/env python3
"""
Script to run the BudSimulator API server.
"""

import uvicorn
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("BACKEND_PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(
        "apis.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 