#!/usr/bin/env python3
"""
Hardware Recommendation Sorting Validation Script

This script validates the new intelligent hardware sorting system including:
1. CPU compatibility logic (model params < 14B OR total memory < 35GB)
2. Memory size descending sort order
3. Optimality indicators (optimal/good/ok)
4. Backward compatibility with existing endpoints
5. All edge cases and scenarios

Usage:
    python validate_hardware_sorting.py [--host localhost:8000]
"""

import requests
import json
import argparse
from typing import List, Dict, Any


class HardwareSortingValidator:
    def __init__(self, host: str = "localhost:8000"):
        self.base_url = f"http://{host}"
        self.passed_tests = 0
        self.failed_tests = 0
        
    def test_scenario(self, name: str, total_memory_gb: float, model_params_b: float, 
                     expected_cpu_count: int, expected_first_type: str):
        """Test a specific scenario and validate results."""
        print(f"\n🧪 {name}")
        print(f"   Input: {model_params_b}B params, {total_memory_gb}GB memory")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/hardware/recommend",
                json={
                    "total_memory_gb": total_memory_gb,
                    "model_params_b": model_params_b
                }
            )
            
            if response.status_code != 200:
                print(f"   ❌ API Error: {response.status_code}")
                self.failed_tests += 1
                return
                
            recommendations = response.json()
            
            # Validate structure
            if not isinstance(recommendations, list) or len(recommendations) == 0:
                print(f"   ❌ Invalid response format")
                self.failed_tests += 1
                return
                
            # Count CPUs
            cpu_count = sum(1 for r in recommendations if r.get('type') == 'cpu')
            
            # Check first hardware type
            first_type = recommendations[0].get('type') if recommendations else None
            
            # Validate CPU logic: CPUs should be included if model < 14B AND memory < 35GB
            cpu_should_be_included = (model_params_b < 14) and (total_memory_gb < 35)
            cpu_logic_correct = (cpu_count > 0) == cpu_should_be_included
            
            # Validate sorting - memory should be descending within each type group
            memory_descending = True
            cpu_memories = [r.get('memory_per_chip', 0) for r in recommendations if r.get('type') == 'cpu']
            non_cpu_memories = [r.get('memory_per_chip', 0) for r in recommendations if r.get('type') != 'cpu']
            
            # Check CPU memory sorting (if any CPUs)
            if len(cpu_memories) > 1:
                for i in range(1, len(cpu_memories)):
                    if cpu_memories[i] > cpu_memories[i-1]:
                        memory_descending = False
                        break
            
            # Check non-CPU memory sorting (if any non-CPUs)
            if len(non_cpu_memories) > 1:
                for i in range(1, len(non_cpu_memories)):
                    if non_cpu_memories[i] > non_cpu_memories[i-1]:
                        memory_descending = False
                        break
            
            # Check that CPUs come before non-CPUs (if both exist)
            cpu_indices = [i for i, r in enumerate(recommendations) if r.get('type') == 'cpu']
            non_cpu_indices = [i for i, r in enumerate(recommendations) if r.get('type') != 'cpu']
            
            if cpu_indices and non_cpu_indices:
                if max(cpu_indices) > min(non_cpu_indices):
                    memory_descending = False
            
            # Validate optimality indicators
            optimality_correct = True
            for i, rec in enumerate(recommendations[:8]):
                expected_optimality = 'optimal' if i < 2 else 'good' if i < 5 else 'ok'
                if rec.get('optimality') != expected_optimality:
                    optimality_correct = False
                    break
            
            # Validate required fields
            required_fields = ['hardware_name', 'nodes_required', 'memory_per_chip', 
                             'type', 'optimality', 'utilization']
            fields_present = all(field in recommendations[0] for field in required_fields)
            
            # Results
            tests = [
                ("CPU Logic", cpu_logic_correct, f"Expected CPUs: {cpu_should_be_included}, Got CPUs: {cpu_count > 0}"),
                ("Memory Sorting", memory_descending, "Memory should be descending within types"),
                ("Optimality Indicators", optimality_correct, "First 2 optimal, next 3 good, rest ok"),
                ("Required Fields", fields_present, f"Missing fields: {[f for f in required_fields if f not in recommendations[0]]}"),
                ("First Type", first_type == expected_first_type, f"Expected {expected_first_type}, got {first_type}")
            ]
            
            all_passed = all(passed for _, passed, _ in tests)
            
            for test_name, passed, details in tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}: {details}")
            
            if all_passed:
                print(f"   🎉 ALL TESTS PASSED - {len(recommendations)} recommendations")
                self.passed_tests += 1
            else:
                print(f"   💥 SOME TESTS FAILED")
                self.failed_tests += 1
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.failed_tests += 1
    
    def validate_backward_compatibility(self):
        """Test that existing endpoints still work."""
        print(f"\n🔄 BACKWARD COMPATIBILITY TESTS")
        
        endpoints = [
            ("GET /api/hardware", "GET", "/api/hardware?limit=5"),
            ("GET /api/hardware/filter", "GET", "/api/hardware/filter?type=gpu&limit=3"),
            ("Hardware Details", "GET", "/api/hardware/NVIDIA%20H100%20(Hopper)%20-%20SXM")
        ]
        
        for name, method, endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) or isinstance(data, dict):
                        print(f"   ✅ {name}: Working")
                        self.passed_tests += 1
                    else:
                        print(f"   ❌ {name}: Invalid response format")
                        self.failed_tests += 1
                else:
                    print(f"   ❌ {name}: HTTP {response.status_code}")
                    self.failed_tests += 1
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                self.failed_tests += 1
    
    def run_all_tests(self):
        """Run comprehensive validation suite."""
        print("🚀 HARDWARE RECOMMENDATION SORTING VALIDATION")
        print("=" * 60)
        
        # Test Scenarios (AND logic: CPUs shown if model < 14B AND memory < 35GB)
        scenarios = [
            # Small models with small memory - should show CPUs first
            ("Small Model (GPT-2)", 2.1, 1.5, ">0", "cpu"),
            ("Medium Model (7B)", 14.2, 7.0, ">0", "cpu"), 
            ("Edge Case (13.9B)", 28.4, 13.9, ">0", "cpu"),
            ("Tiny Model", 0.5, 0.1, ">0", "cpu"),
            
            # Large models - should NOT show CPUs (params >= 14B)
            ("Large Model (14B)", 45.6, 14.0, "0", "gpu"),
            ("Very Large (70B)", 150.8, 70.0, "0", "gpu"),
            ("Large Params Small Memory", 30.0, 50.0, "0", "gpu"),
            
            # Small models with large memory - should NOT show CPUs (memory >= 35GB)
            ("Small Model Large Memory", 45.0, 5.0, "0", "gpu"),
            ("Memory Boundary", 35.0, 13.9, "0", "gpu"),
            
            # Boundary cases - should NOT show CPUs
            ("Param Boundary", 33.0, 14.0, "0", "gpu"),
            ("Memory Edge (35GB)", 35.0, 10.0, "0", "gpu"),
        ]
        
        for name, memory, params, expected_cpus, expected_first in scenarios:
            expected_cpu_count = 1 if expected_cpus == ">0" else 0
            self.test_scenario(name, memory, params, expected_cpu_count, expected_first)
        
        # Backward compatibility
        self.validate_backward_compatibility()
        
        # Summary
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 60)
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Hardware sorting is working perfectly!")
            return True
        else:
            print(f"\n💥 {self.failed_tests} tests failed. Please review the issues above.")
            return False


def main():
    parser = argparse.ArgumentParser(description="Validate hardware recommendation sorting")
    parser.add_argument("--host", default="localhost:8000", help="API host:port")
    args = parser.parse_args()
    
    validator = HardwareSortingValidator(args.host)
    success = validator.run_all_tests()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 