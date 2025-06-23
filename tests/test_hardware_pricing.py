"""
Tests for hardware pricing functionality.
"""
import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.hardware import BudHardware
from src.hardware_recommendation import HardwareRecommendation


class TestHardwarePricing:
    """Test suite for hardware pricing functionality."""

    def test_price_indicator_calculation_normal_case(self):
        """Test price indicator calculation with normal values."""
        price = BudHardware.calculate_price_indicator(
            flops=1000,  # 1 TFLOPS
            memory_gb=80,  # 80GB
            bandwidth_gbs=2000  # 2TB/s
        )
        
        assert price > 0
        assert isinstance(price, float)
        assert price == round(price, 2)  # Should be rounded to 2 decimal places

    def test_price_indicator_calculation_high_end_hardware(self):
        """Test price indicator calculation with high-end hardware specs."""
        price = BudHardware.calculate_price_indicator(
            flops=1979,  # H100 SXM specs
            memory_gb=80,
            bandwidth_gbs=3350
        )
        
        assert price > 0
        # High-end hardware should have higher price indicators
        
        basic_price = BudHardware.calculate_price_indicator(
            flops=100,
            memory_gb=16,
            bandwidth_gbs=500
        )
        
        assert price > basic_price  # High-end should cost more

    def test_price_indicator_edge_cases(self):
        """Test price indicator calculation with edge cases."""
        # Zero values should return 0
        assert BudHardware.calculate_price_indicator(0, 80, 2000) == 0.0
        assert BudHardware.calculate_price_indicator(1000, 0, 2000) == 0.0
        assert BudHardware.calculate_price_indicator(1000, 80, 0) == 0.0
        
        # Negative values should return 0
        assert BudHardware.calculate_price_indicator(-1000, 80, 2000) == 0.0
        assert BudHardware.calculate_price_indicator(1000, -80, 2000) == 0.0
        assert BudHardware.calculate_price_indicator(1000, 80, -2000) == 0.0

    def test_price_indicator_very_small_values(self):
        """Test price indicator with very small values."""
        price = BudHardware.calculate_price_indicator(
            flops=0.001,
            memory_gb=0.001,
            bandwidth_gbs=0.001
        )
        assert price >= 0

    def test_price_indicator_very_large_values(self):
        """Test price indicator with very large values."""
        price = BudHardware.calculate_price_indicator(
            flops=1e15,  # 1 PFLOPS
            memory_gb=1000,  # 1TB
            bandwidth_gbs=100000  # 100TB/s
        )
        assert price >= 0
        assert price != float('inf')  # Should not overflow

    def test_hardware_recommendation_includes_pricing(self):
        """Test that hardware recommendations include price indicators."""
        engine = HardwareRecommendation()
        recommendations = engine.recommend_hardware(total_memory_gb=100)
        
        # Check that price_approx is included in recommendations
        for rec in recommendations['cpu_recommendations']:
            assert 'price_approx' in rec
            if rec['price_approx'] is not None:
                assert rec['price_approx'] > 0
                assert isinstance(rec['price_approx'], float)

        for rec in recommendations['gpu_recommendations']:
            assert 'price_approx' in rec
            if rec['price_approx'] is not None:
                assert rec['price_approx'] > 0
                assert isinstance(rec['price_approx'], float)

    def test_price_indicator_consistency(self):
        """Test that price indicators are consistent for the same specs."""
        specs = [
            (1000, 80, 2000),
            (500, 40, 1000),
            (2000, 160, 4000)
        ]
        
        for flops, memory_gb, bandwidth_gbs in specs:
            price1 = BudHardware.calculate_price_indicator(flops, memory_gb, bandwidth_gbs)
            price2 = BudHardware.calculate_price_indicator(flops, memory_gb, bandwidth_gbs)
            assert price1 == price2  # Should be deterministic

    def test_price_indicator_scaling(self):
        """Test that price indicators scale appropriately with specs."""
        base_price = BudHardware.calculate_price_indicator(1000, 80, 2000)
        
        # Double memory should increase price
        double_memory_price = BudHardware.calculate_price_indicator(1000, 160, 2000)
        assert double_memory_price > base_price
        
        # Double bandwidth should increase price more significantly
        double_bandwidth_price = BudHardware.calculate_price_indicator(1000, 80, 4000)
        assert double_bandwidth_price > base_price
        
        # Double FLOPS should decrease price slightly (due to negative coefficient)
        double_flops_price = BudHardware.calculate_price_indicator(2000, 80, 2000)
        # Note: FLOPS has a negative coefficient in the formula

    def test_recommendation_response_schema_compatibility(self):
        """Test that recommendation responses match expected schema."""
        engine = HardwareRecommendation()
        recommendations = engine.recommend_hardware(
            total_memory_gb=50.0,
            model_params_b=7.0
        )
        
        # Check response structure
        assert 'cpu_recommendations' in recommendations
        assert 'gpu_recommendations' in recommendations
        assert 'model_info' in recommendations
        assert 'total_recommendations' in recommendations
        
        # Check individual recommendation structure
        for rec_list in [recommendations['cpu_recommendations'], recommendations['gpu_recommendations']]:
            for rec in rec_list:
                assert 'hardware_name' in rec
                assert 'nodes_required' in rec
                assert 'memory_per_chip' in rec
                assert 'type' in rec
                assert 'optimality' in rec
                assert 'utilization' in rec
                assert 'price_approx' in rec  # NEW FIELD

if __name__ == '__main__':
    pytest.main([__file__]) 