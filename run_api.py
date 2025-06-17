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
    # Run the FastAPI app
    uvicorn.run(
        "apis.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 