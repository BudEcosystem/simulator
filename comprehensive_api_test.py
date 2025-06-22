#!/usr/bin/env python3
"""
Comprehensive API Test Suite for BudSimulator

Tests all API endpoints to ensure database consolidation hasn't broken functionality.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
from urllib.parse import quote

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class APITester:
    """Comprehensive API testing suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BLUE}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        
    def print_test(self, test_name: str):
        """Print test name."""
        print(f"\n{Colors.CYAN}🔍 Testing: {test_name}{Colors.ENDC}")
        
    def print_success(self, message: str):
        """Print success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
        self.results['passed'] += 1
        
    def print_failure(self, message: str):
        """Print failure message."""
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
        self.results['failed'] += 1
        
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
        self.results['skipped'] += 1
        
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"   Request failed: {e}")
            return None

    def test_basic_connectivity(self):
        """Test basic API connectivity."""
        self.print_header("BASIC CONNECTIVITY TESTS")
        
        # Test root endpoint
        self.print_test("Root endpoint (/)")
        response = self.make_request("GET", "/")
        if response and response.status_code == 200:
            data = response.json()
            if "name" in data and "version" in data:
                self.print_success(f"Root endpoint working - {data.get('name')} v{data.get('version')}")
            else:
                self.print_failure("Root endpoint missing required fields")
        else:
            self.print_failure("Root endpoint not accessible")
            
    def test_health_endpoints(self):
        """Test health check endpoints."""
        self.print_header("HEALTH CHECK TESTS")
        
        # Test main health endpoint
        self.print_test("Health check (/api/health)")
        response = self.make_request("GET", "/api/health")
        if response and response.status_code == 200:
            data = response.json()
            if data.get('status') in ['healthy', 'degraded']:
                db_info = data.get('database', {})
                if db_info.get('status') == 'connected':
                    model_count = db_info.get('total_models', 0)
                    self.print_success(f"Health check passed - {model_count} models in database")
                    
                    # Verify database path
                    db_path = db_info.get('path', '')
                    if 'prepopulated.db' in db_path:
                        self.print_success("Using correct consolidated database (prepopulated.db)")
                    else:
                        self.print_failure(f"Wrong database path: {db_path}")
                else:
                    self.print_failure(f"Database not connected: {db_info.get('status')}")
            else:
                self.print_failure(f"Health check failed with status: {data.get('status')}")
        else:
            self.print_failure("Health endpoint not accessible")
            
    def test_models_list_endpoints(self):
        """Test model listing endpoints."""
        self.print_header("MODEL LISTING TESTS")
        
        # Test list all models
        self.print_test("List all models (/api/models/list)")
        response = self.make_request("GET", "/api/models/list")
        if response and response.status_code == 200:
            data = response.json()
            total_count = data.get('total_count', 0)
            models = data.get('models', [])
            
            if total_count > 0 and len(models) > 0:
                self.print_success(f"Model list working - {total_count} total models")
                
                # Test specific models we know should exist
                model_ids = {m['model_id'] for m in models}
                test_models = ['Qwen/Qwen-72B', 'THUDM/glm-4-9b-chat', 'microsoft/Phi-3-mini']
                
                for model_id in test_models:
                    if model_id in model_ids:
                        model = next(m for m in models if m['model_id'] == model_id)
                        param_count = model.get('parameter_count')
                        if param_count:
                            self.print_success(f"✓ {model_id}: {param_count/1e9:.1f}B parameters")
                        else:
                            self.print_failure(f"✗ {model_id}: Missing parameter count")
                    else:
                        self.print_failure(f"✗ {model_id}: Not found in list")
            else:
                self.print_failure("Model list empty or invalid")
        else:
            self.print_failure("Model list endpoint not accessible")
            
    def test_individual_model_endpoints(self):
        """Test individual model endpoints."""
        self.print_header("INDIVIDUAL MODEL TESTS")
        
        # Test models with known parameter counts
        test_cases = [
            ("Qwen/Qwen-72B", 88393400320, "88.4B"),
            ("THUDM/glm-4-9b-chat", 7000000000, "7.0B"),
            ("microsoft/Phi-3-mini", 3800000000, "3.8B")
        ]
        
        for model_id, expected_params, expected_display in test_cases:
            self.print_test(f"Individual model: {model_id}")
            encoded_id = quote(model_id, safe='')
            response = self.make_request("GET", f"/api/models/{encoded_id}")
            
            if response and response.status_code == 200:
                data = response.json()
                actual_params = data.get('parameter_count')
                
                if actual_params == expected_params:
                    self.print_success(f"✓ Parameter count correct: {expected_display}")
                elif actual_params:
                    error_pct = abs(actual_params - expected_params) / expected_params * 100
                    if error_pct < 5:  # Allow 5% variance
                        self.print_success(f"✓ Parameter count close: {actual_params/1e9:.1f}B (within 5%)")
                    else:
                        self.print_failure(f"✗ Parameter count wrong: {actual_params/1e9:.1f}B (expected {expected_display})")
                else:
                    self.print_failure(f"✗ No parameter count returned")
                    
                # Check other fields
                if data.get('model_id') == model_id:
                    self.print_success(f"✓ Model ID correct: {model_id}")
                else:
                    self.print_failure(f"✗ Model ID mismatch")
                    
                if data.get('in_database'):
                    self.print_success(f"✓ Model found in database")
                else:
                    self.print_failure(f"✗ Model not found in database")
                    
            else:
                self.print_failure(f"✗ Model endpoint not accessible")
                
    def test_hardware_endpoints(self):
        """Test hardware-related endpoints."""
        self.print_header("HARDWARE API TESTS")
        
        # Test hardware list
        self.print_test("Hardware list (/api/hardware/list)")
        response = self.make_request("GET", "/api/hardware/list")
        
        if response and response.status_code == 200:
            data = response.json()
            hardware_items = data.get('hardware', [])
            
            if len(hardware_items) > 0:
                self.print_success(f"✓ Hardware list working: {len(hardware_items)} items")
            else:
                self.print_warning("No hardware items found")
        else:
            self.print_failure("✗ Hardware list endpoint not accessible")
            
    def run_all_tests(self):
        """Run all test suites."""
        print(f"{Colors.BOLD}{Colors.CYAN}🚀 Starting Comprehensive API Tests{Colors.ENDC}")
        print(f"Testing API at: {self.base_url}")
        
        # Run test suites
        test_suites = [
            self.test_basic_connectivity,
            self.test_health_endpoints,
            self.test_models_list_endpoints,
            self.test_individual_model_endpoints,
            self.test_hardware_endpoints
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                self.print_failure(f"Test suite failed with exception: {e}")
                
        # Print final results
        self.print_results()
        
    def print_results(self):
        """Print final test results."""
        self.print_header("TEST RESULTS SUMMARY")
        
        total_tests = self.results['passed'] + self.results['failed'] + self.results['skipped']
        
        print(f"{Colors.BOLD}Total Tests: {total_tests}{Colors.ENDC}")
        print(f"{Colors.GREEN}✅ Passed: {self.results['passed']}{Colors.ENDC}")
        print(f"{Colors.RED}❌ Failed: {self.results['failed']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}⚠️  Skipped: {self.results['skipped']}{Colors.ENDC}")
        
        if self.results['failed'] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! API is working correctly.{Colors.ENDC}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Some tests failed. Check output above for details.{Colors.ENDC}")
            return 1


def main():
    """Main test runner."""
    tester = APITester()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
