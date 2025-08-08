#!/usr/bin/env python3
"""Simple test runner for hardware optimization tests"""

import sys
import unittest
from tests.test_hardware_optimization import (
    TestHardwareOptimizer,
    TestOptimizationAlgorithm,
    TestGenZIntegration,
    TestCostOptimization,
    TestEdgeCases,
    TestModelRegistry
)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHardwareOptimizer,
        TestOptimizationAlgorithm,
        TestGenZIntegration,
        TestCostOptimization,
        TestEdgeCases,
        TestModelRegistry
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)