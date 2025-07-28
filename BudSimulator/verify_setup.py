#!/usr/bin/env python3
"""
Verification script for BudSimulator setup.
Run this after setup to ensure everything is working correctly.
"""

import sys
import os
import sqlite3
import requests
import time
from pathlib import Path
from typing import Tuple, List

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test(test_name: str):
    """Print test name."""
    print(f"\n{Colors.BLUE}Testing: {test_name}{Colors.ENDC}")


def print_pass(message: str):
    """Print pass message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_fail(message: str):
    """Print fail message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_warn(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    print_test("Python Version")
    
    if sys.version_info >= (3, 8):
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print_pass(f"Python {version} is compatible")
        return True, version
    else:
        print_fail("Python 3.8 or higher is required")
        return False, ""


def check_database() -> Tuple[bool, dict]:
    """Check database setup."""
    print_test("Database")
    
    db_path = Path(__file__).parent / "data" / "prepopulated.db"
    
    if not db_path.exists():
        print_fail(f"Database not found at {db_path}")
        return False, {}
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check models table
        cursor.execute("SELECT COUNT(*) FROM models")
        model_count = cursor.fetchone()[0]
        
        # Check models with config
        cursor.execute("""
            SELECT COUNT(*) FROM models 
            WHERE config_json IS NOT NULL 
            AND config_json != ''
            AND config_json != '{}'
        """)
        config_count = cursor.fetchone()[0]
        
        conn.close()
        
        print_pass(f"Database found with {model_count} models")
        print_pass(f"{config_count} models have configuration data")
        
        return True, {
            "total_models": model_count,
            "models_with_config": config_count,
            "completeness": round((config_count / model_count * 100) if model_count > 0 else 0, 2)
        }
        
    except Exception as e:
        print_fail(f"Database error: {str(e)}")
        return False, {}


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check Python dependencies."""
    print_test("Python Dependencies")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "beautifulsoup4",
        "PIL",
        "numpy",
        "genz"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL")
            else:
                __import__(package)
            print_pass(f"{package} is installed")
        except ImportError:
            print_fail(f"{package} is NOT installed")
            missing.append(package)
            
    return len(missing) == 0, missing


def check_api_server(port: int = 8000) -> bool:
    """Check if API server is running."""
    print_test("API Server")
    
    try:
        # Check health endpoint
        response = requests.get(f"http://localhost:{port}/api/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print_pass(f"API server is running on port {port}")
            print_pass(f"Status: {health_data.get('status', 'unknown')}")
            
            # Check specific endpoints
            endpoints = [
                "/docs",
                "/api/models",
                "/api/health/ready"
            ]
            
            for endpoint in endpoints:
                try:
                    resp = requests.get(f"http://localhost:{port}{endpoint}", timeout=3)
                    if resp.status_code == 200:
                        print_pass(f"Endpoint {endpoint} is accessible")
                    else:
                        print_warn(f"Endpoint {endpoint} returned status {resp.status_code}")
                except:
                    print_fail(f"Endpoint {endpoint} is not accessible")
                    
            return True
        else:
            print_fail(f"API server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_fail(f"API server is not running on port {port}")
        print_warn("Start the server with: python run_api.py")
        return False
    except Exception as e:
        print_fail(f"Error checking API server: {str(e)}")
        return False


def check_frontend(port: int = 3000) -> bool:
    """Check if frontend is running."""
    print_test("Frontend Server")
    
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        
        if response.status_code == 200:
            print_pass(f"Frontend is running on port {port}")
            return True
        else:
            print_fail(f"Frontend returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_fail(f"Frontend is not running on port {port}")
        print_warn("Start the frontend with: cd frontend && npm start")
        return False
    except Exception as e:
        print_fail(f"Error checking frontend: {str(e)}")
        return False


def check_environment_config() -> bool:
    """Check environment configuration."""
    print_test("Environment Configuration")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print_fail(".env file not found")
        print_warn("Run setup.py to configure your environment")
        return False
        
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            
        required_vars = ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_URL"]
        missing = []
        
        for var in required_vars:
            if var in content:
                print_pass(f"{var} is configured")
            else:
                print_fail(f"{var} is NOT configured")
                missing.append(var)
                
        return len(missing) == 0
        
    except Exception as e:
        print_fail(f"Error reading .env file: {str(e)}")
        return False


def main():
    """Run all verification checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}BudSimulator Setup Verification{Colors.ENDC}")
    print("=" * 50)
    
    results = {
        "python": check_python_version()[0],
        "database": check_database()[0],
        "dependencies": check_dependencies()[0],
        "environment": check_environment_config(),
        "api": check_api_server(),
        "frontend": check_frontend()
    }
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print("-" * 30)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.RED}FAIL{Colors.ENDC}"
        print(f"{check.capitalize():<15} {status}")
        
    print("-" * 30)
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All checks passed! BudSimulator is ready to use.{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  {passed}/{total} checks passed.{Colors.ENDC}")
        print("Run setup.py to fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 