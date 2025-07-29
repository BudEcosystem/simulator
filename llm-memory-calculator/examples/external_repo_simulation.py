#!/usr/bin/env python3
"""
External Repository Simulation

This script simulates how an external repository would use llm-memory-calculator
for LLM performance analysis. It demonstrates the key use cases that external
developers would need.

This can be copied to any external repository and run with:
  pip install llm-memory-calculator
  python external_repo_simulation.py
"""

def test_external_repo_usage():
    """Test as if this were an external repository importing the package."""
    
    print("🧪 Testing llm-memory-calculator as External Dependency")
    print("=" * 60)
    
    # Test imports (this is what external repos will do)
    try:
        from llm_memory_calculator import (
            # Core performance functions
            estimate_prefill_performance,
            estimate_decode_performance,
            estimate_end_to_end_performance,
            compare_performance_configurations,
            
            # Hardware and parallelism optimization
            get_hardware_config,
            get_best_parallelization_strategy,
            get_various_parallelization,
            HARDWARE_CONFIGS,
            
            # Memory calculation
            calculate_memory
        )
        print("✅ All imports successful!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Install with: pip install llm-memory-calculator")
        return False
    
    # Test 1: Basic performance estimation
    print(f"\n🔬 Test 1: Basic Performance Estimation")
    try:
        hardware = get_hardware_config('A100_80GB')
        result = estimate_end_to_end_performance(
            model='llama2_7b',
            batch_size=4,
            input_tokens=2048,
            output_tokens=256,
            system_name=hardware,
            bits='bf16',
            tensor_parallel=2
        )
        
        throughput = result.get('total_throughput', 0)
        ttft = result.get('ttft', 0)
        print(f"   ✅ Llama2-7B performance: {throughput:.1f} tokens/s, TTFT: {ttft:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ Performance estimation failed: {e}")
        return False
    
    # Test 2: Hardware comparison
    print(f"\n🔬 Test 2: Hardware Comparison")
    try:
        configs = [
            {
                'name': 'A100_80GB',
                'system_name': get_hardware_config('A100_80GB'),
                'tensor_parallel': 2,
                'bits': 'bf16'
            },
            {
                'name': 'H100_80GB',
                'system_name': get_hardware_config('H100_80GB'),
                'tensor_parallel': 2,
                'bits': 'bf16'
            }
        ]
        
        comparison = compare_performance_configurations(
            model='llama2_7b',
            configurations=configs,
            batch_size=4,
            input_tokens=2048,
            output_tokens=256
        )
        
        best = comparison[0]
        best_name = best['config_name']
        best_throughput = best['total_throughput']
        print(f"   ✅ Best hardware: {best_name} with {best_throughput:.1f} tokens/s")
        
    except Exception as e:
        print(f"   ❌ Hardware comparison failed: {e}")
        return False
    
    # Test 3: Parallelism optimization
    print(f"\n🔬 Test 3: Parallelism Optimization")
    try:
        # Get available options
        options = get_various_parallelization('llama2_7b', total_nodes=4)
        print(f"   📊 Available TP/PP options for 4 GPUs: {len(options)} combinations")
        
        # Get best strategy
        strategy = get_best_parallelization_strategy(
            model='llama2_7b',
            total_nodes=4,
            batch_size=8,
            system_name=get_hardware_config('A100_80GB'),
            bits='bf16'
        )
        
        best_tp = strategy['TP'].iloc[0]
        best_pp = strategy['PP'].iloc[0]
        best_perf = strategy['Tokens/s'].iloc[0]
        print(f"   ✅ Best strategy: TP={best_tp}, PP={best_pp} → {best_perf:.1f} tokens/s")
        
    except Exception as e:
        print(f"   ❌ Parallelism optimization failed: {e}")
        return False
    
    # Test 4: Memory calculation
    print(f"\n🔬 Test 4: Memory Calculation")
    try:
        memory = calculate_memory(
            "meta-llama/Llama-2-7b-hf",
            batch_size=4,
            seq_length=2048,
            precision="bf16"
        )
        
        total_gb = memory.total_memory_gb
        recommended_gpu = memory.recommended_gpu_memory_gb
        print(f"   ✅ Memory needed: {total_gb:.1f} GB, recommended GPU: {recommended_gpu} GB")
        
    except Exception as e:
        print(f"   ❌ Memory calculation failed: {e}")
        return False
    
    # Test 5: Available hardware configurations
    print(f"\n🔬 Test 5: Available Hardware Configurations")
    try:
        print(f"   📊 Available hardware:")
        for hw_name, config in HARDWARE_CONFIGS.items():
            memory = config['Memory_size']
            flops = config['Flops']
            print(f"      • {hw_name}: {memory} GB, {flops} TFLOPS")
        print(f"   ✅ {len(HARDWARE_CONFIGS)} hardware configurations available")
        
    except Exception as e:
        print(f"   ❌ Hardware config access failed: {e}")
        return False
    
    return True


def example_use_cases():
    """Show practical use cases for external repositories."""
    
    print(f"\n🎯 Practical Use Cases for External Repositories")
    print("=" * 55)
    
    from llm_memory_calculator import (
        estimate_end_to_end_performance,
        get_hardware_config,
        get_best_parallelization_strategy,
        calculate_memory
    )
    
    # Use Case 1: ML Infrastructure team choosing hardware
    print(f"\n💼 Use Case 1: Hardware Selection for Deployment")
    print("   Scenario: ML team needs to deploy Llama2-7B for production")
    
    target_throughput = 10000  # tokens/s
    workload = {
        'model': 'llama2_7b',
        'batch_size': 8,
        'input_tokens': 2048,
        'output_tokens': 256
    }
    
    hardware_options = ['A100_80GB', 'H100_80GB', 'MI300X']
    
    print(f"   🎯 Target: ≥{target_throughput} tokens/s")
    print(f"   📋 Results:")
    
    for hw_name in hardware_options:
        try:
            hardware = get_hardware_config(hw_name)
            result = estimate_end_to_end_performance(
                system_name=hardware,
                tensor_parallel=4,
                bits='bf16',
                **workload
            )
            
            throughput = result.get('total_throughput', 0)
            meets_target = "✅" if throughput >= target_throughput else "❌"
            cost_per_hour = {"A100_80GB": 3.2, "H100_80GB": 4.6, "MI300X": 3.8}.get(hw_name, 0)
            
            print(f"      {meets_target} {hw_name}: {throughput:.0f} tok/s (${cost_per_hour}/h)")
            
        except Exception as e:
            print(f"      ❌ {hw_name}: Error - {str(e)[:30]}...")
    
    # Use Case 2: Research team optimizing parallelism
    print(f"\n💼 Use Case 2: Research Team Parallelism Optimization")
    print("   Scenario: Researchers with 8 GPUs want optimal configuration")
    
    try:
        best_config = get_best_parallelization_strategy(
            model='llama2_7b',
            total_nodes=8,
            batch_size=16,
            system_name=get_hardware_config('A100_80GB'),
            bits='bf16'
        )
        
        tp = best_config['TP'].iloc[0]
        pp = best_config['PP'].iloc[0]
        perf = best_config['Tokens/s'].iloc[0]
        latency = best_config['Latency(ms)'].iloc[0]
        
        print(f"   🎯 Optimal for 8x A100: TP={tp}, PP={pp}")
        print(f"   📊 Expected: {perf:.0f} tokens/s, {latency:.1f}ms latency")
        print(f"   💡 Recommendation: Use tensor parallelism for this model size")
        
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
    
    # Use Case 3: DevOps team planning memory requirements
    print(f"\n💼 Use Case 3: DevOps Memory Planning")
    print("   Scenario: DevOps needs memory estimates for container sizing")
    
    models_to_deploy = [
        ("meta-llama/Llama-2-7b-hf", 4, 2048),
        ("meta-llama/Llama-2-13b-hf", 2, 4096),
    ]
    
    print(f"   📋 Memory Requirements:")
    
    for model_id, batch_size, seq_len in models_to_deploy:
        try:
            memory = calculate_memory(
                model_id,
                batch_size=batch_size,
                seq_length=seq_len,
                precision="bf16"
            )
            
            model_name = model_id.split('/')[-1]
            total_gb = memory.total_memory_gb
            gpu_rec = memory.recommended_gpu_memory_gb
            
            print(f"      • {model_name} (B={batch_size}, L={seq_len}): {total_gb:.1f} GB → {gpu_rec} GB GPU")
            
        except Exception as e:
            print(f"      • {model_id}: Error - {str(e)[:30]}...")


def main():
    """Main demo function."""
    
    print("🚀 LLM Memory Calculator - External Repository Demo")
    print("=" * 65)
    print("This demonstrates how external repositories can use the package")
    print("for LLM performance analysis and optimization.")
    
    # Run tests
    success = test_external_repo_usage()
    
    if success:
        example_use_cases()
        
        print(f"\n🎉 Demo Complete - All Tests Passed!")
        print(f"=" * 40)
        print(f"✅ llm-memory-calculator is ready for external repository use")
        print(f"📦 Installation: pip install llm-memory-calculator")
        print(f"📖 Key imports:")
        print(f"   from llm_memory_calculator import (")
        print(f"       estimate_end_to_end_performance,")
        print(f"       get_best_parallelization_strategy,")
        print(f"       get_hardware_config,")
        print(f"       calculate_memory")
        print(f"   )")
        print(f"\n🎯 Main use cases:")
        print(f"   • Hardware selection and comparison")
        print(f"   • Parallelism strategy optimization")
        print(f"   • Memory requirement planning") 
        print(f"   • Performance estimation and analysis")
        
    else:
        print(f"\n❌ Demo Failed - Package Not Ready")
        print(f"Please check installation and dependencies")


if __name__ == "__main__":
    main()