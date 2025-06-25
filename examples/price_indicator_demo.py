#!/usr/bin/env python3
"""
Price Indicator Feature Demonstration

This script demonstrates the new price indicator functionality for hardware recommendations.
It shows how the system calculates relative price indicators based on hardware specifications.
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.hardware import BudHardware
from src.hardware_recommendation import HardwareRecommendation


def demonstrate_price_calculation():
    """Demonstrate the price indicator calculation for various hardware types."""
    print("🔧 Price Indicator Calculation Demonstration")
    print("=" * 60)
    
    hardware_examples = [
        {
            "name": "NVIDIA H100 SXM",
            "flops": 1979,
            "memory_gb": 80,
            "bandwidth_gbs": 3350
        },
        {
            "name": "NVIDIA A100 80GB",
            "flops": 312,
            "memory_gb": 80,
            "bandwidth_gbs": 2000
        },
        {
            "name": "NVIDIA RTX 4090",
            "flops": 661,
            "memory_gb": 24,
            "bandwidth_gbs": 1008
        },
        {
            "name": "AMD MI300X",
            "flops": 1600,
            "memory_gb": 192,
            "bandwidth_gbs": 5300
        },
        {
            "name": "Intel Xeon CPU",
            "flops": 3.8912,
            "memory_gb": 512,
            "bandwidth_gbs": 358.4
        }
    ]
    
    for hw in hardware_examples:
        price_indicator = BudHardware.calculate_price_indicator(
            flops=hw["flops"],
            memory_gb=hw["memory_gb"],
            bandwidth_gbs=hw["bandwidth_gbs"]
        )
        
        print(f"📦 {hw['name']}")
        print(f"   Performance: {hw['flops']} TFLOPS")
        print(f"   Memory: {hw['memory_gb']} GB")
        print(f"   Bandwidth: {hw['bandwidth_gbs']} GB/s")
        print(f"   💰 Price Indicator: ${price_indicator:.2f}")
        print()


def demonstrate_hardware_recommendations():
    """Demonstrate hardware recommendations with price indicators."""
    print("🎯 Hardware Recommendations with Price Indicators")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {"memory_gb": 20, "model_params_b": 7, "description": "7B parameter model (e.g., Llama-2-7B)"},
        {"memory_gb": 150, "model_params_b": 70, "description": "70B parameter model (e.g., Llama-2-70B)"},
        {"memory_gb": 50, "model_params_b": None, "description": "Custom workload requiring 50GB"}
    ]
    
    engine = HardwareRecommendation()
    
    for scenario in scenarios:
        print(f"📋 Scenario: {scenario['description']}")
        print(f"   Memory Required: {scenario['memory_gb']} GB")
        if scenario['model_params_b']:
            print(f"   Model Size: {scenario['model_params_b']}B parameters")
        print()
        
        recommendations = engine.recommend_hardware(
            total_memory_gb=scenario['memory_gb'],
            model_params_b=scenario['model_params_b']
        )
        
        # Show top 3 GPU recommendations
        gpu_recs = recommendations['gpu_recommendations'][:3]
        if gpu_recs:
            print("   🔥 Top GPU Recommendations:")
            for i, rec in enumerate(gpu_recs, 1):
                price_str = f"${rec['price_approx']:.2f}" if rec['price_approx'] else "N/A"
                print(f"   {i}. {rec['hardware_name']}")
                print(f"      Nodes: {rec['nodes_required']}, Utilization: {rec['utilization']}%")
                print(f"      Optimality: {rec['optimality']}, Price Indicator: {price_str}")
        
        # Show top 2 CPU recommendations if available
        cpu_recs = recommendations['cpu_recommendations'][:2]
        if cpu_recs and scenario['model_params_b'] and scenario['model_params_b'] < 14:
            print("   💻 CPU Recommendations:")
            for i, rec in enumerate(cpu_recs, 1):
                price_str = f"${rec['price_approx']:.2f}" if rec['price_approx'] else "N/A"
                print(f"   {i}. {rec['hardware_name']}")
                print(f"      Nodes: {rec['nodes_required']}, Utilization: {rec['utilization']}%")
                print(f"      Price Indicator: {price_str}")
        
        print("-" * 50)
        print()


def demonstrate_api_response_format():
    """Show the API response format with price indicators."""
    print("📡 API Response Format Example")
    print("=" * 60)
    
    engine = HardwareRecommendation()
    recommendations = engine.recommend_hardware(
        total_memory_gb=100,
        model_params_b=13
    )
    
    # Show a sample response in JSON format
    sample_response = {
        "cpu_recommendations": recommendations['cpu_recommendations'][:1],
        "gpu_recommendations": recommendations['gpu_recommendations'][:2],
        "model_info": recommendations['model_info'],
        "total_recommendations": recommendations['total_recommendations'],
        "cost_disclaimer": "Pricing information is estimated and may vary. Contact vendors for current pricing."
    }
    
    print("Sample API Response:")
    print(json.dumps(sample_response, indent=2, default=str))


def main():
    """Run all demonstrations."""
    print("🚀 BudSimulator Price Indicator Feature Demo")
    print("=" * 60)
    print("This demonstration showcases the new price indicator functionality.")
    print("Price indicators provide relative cost estimates for hardware comparison.")
    print("⚠️  Note: These are NOT actual prices, only relative indicators!")
    print()
    
    try:
        demonstrate_price_calculation()
        demonstrate_hardware_recommendations()
        demonstrate_api_response_format()
        
        print("✅ Demonstration completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("• Price indicator calculation based on hardware specs")
        print("• Integration with hardware recommendations")
        print("• API response format with pricing information")
        print("• Backward compatibility (optional price fields)")
        print()
        print("Next Steps:")
        print("• Test the API endpoints with tools like curl or Postman")
        print("• Check the frontend for price display in hardware cards")
        print("• Review the price disclaimer component in the UI")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 