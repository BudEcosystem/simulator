"""
Example script demonstrating CPU support in GenZ/BudSimulator
"""
import pandas as pd
from llm_memory_calculator.genz.operators import GEMM, Logit, FC
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.genz.cpu import create_cpu_system
from llm_memory_calculator.genz.cpu.operator_enhancement import enhance_operators_for_cpu
from llm_memory_calculator.genz.analyse_model import analysis_model
from llm_memory_calculator.genz.Models import OpType, ResidencyInfo


def main():
    # Enable CPU support for operators
    enhance_operators_for_cpu()
    
    # Create different CPU systems
    intel_cpu = create_cpu_system('intel_xeon_8380')
    amd_cpu = create_cpu_system('amd_epyc_7763')
    arm_cpu = create_cpu_system('aws_graviton3')
    
    # Create unit for display
    unit = Unit()
    
    # Example 1: Single operator analysis
    print("=== Single Operator Analysis ===")
    gemm = GEMM(['gemm', 1, 1024, 1024, 1024], density=(1.0, 1.0, 1.0))
    
    # Analyze on Intel CPU
    result = gemm.get_roofline(intel_cpu, unit)
    print(f"\nGEMM on Intel Xeon 8380:")
    print(f"  ISA used: {result['ISA_used']}")
    print(f"  Thread count: {result['Thread_count']}")
    print(f"  Frequency: {result['Frequency_GHz']:.2f} GHz")
    print(f"  Latency: {result[f'Latency ({unit.unit_time})']} {unit.unit_time}")
    print(f"  Throughput: {result[f'Throughput ({unit.unit_compute})']} {unit.unit_compute}")
    print(f"  L1 hit rate: {result['L1_hit_rate']:.2%}")
    print(f"  Parallel efficiency: {result['Parallel_efficiency']:.2%}")
    
    # Example 2: Compare different CPUs
    print("\n\n=== CPU Comparison ===")
    cpus = {
        'Intel Xeon 8380': intel_cpu,
        'AMD EPYC 7763': amd_cpu,
        'AWS Graviton3': arm_cpu
    }
    
    comparison_results = []
    for cpu_name, cpu_system in cpus.items():
        result = gemm.get_roofline(cpu_system, unit)
        comparison_results.append({
            'CPU': cpu_name,
            'ISA': result['ISA_used'],
            'Threads': result['Thread_count'],
            f'Latency ({unit.unit_time})': result[f'Latency ({unit.unit_time})'],
            'Bound': result['Bound']
        })
    
    df_comparison = pd.DataFrame(comparison_results)
    print(df_comparison.to_string(index=False))
    
    # Example 3: Model analysis
    print("\n\n=== Model Analysis ===")
    # Define a simple transformer layer
    model_dims = [
        ['qkv_proj', 1, 768, 2304, ResidencyInfo.All_offchip, OpType.FC],  # Q,K,V projection
        ['logit', 1, 12, 128, 128, 64, 12, ResidencyInfo.All_offchip, OpType.Logit],  # Attention scores
        ['attend', 1, 12, 128, 128, 64, 12, ResidencyInfo.All_offchip, OpType.Attend],  # Attention
        ['out_proj', 1, 768, 768, ResidencyInfo.All_offchip, OpType.FC],  # Output projection
        ['ffn_1', 1, 768, 3072, ResidencyInfo.All_offchip, OpType.FC],  # FFN layer 1
        ['ffn_2', 1, 3072, 768, ResidencyInfo.All_offchip, OpType.FC],  # FFN layer 2
    ]
    
    # Analyze on Intel CPU
    results_df = analysis_model(model_dims, system=intel_cpu, unit=unit)
    
    print("Transformer layer on Intel Xeon 8380:")
    print(results_df[['Layer Name', 'Op Type', 'ISA_used', 
                     f'Latency ({unit.unit_time})', 'Bound']].to_string(index=False))
    
    # Calculate total latency
    total_latency = results_df[f'Latency ({unit.unit_time})'].sum()
    print(f"\nTotal latency: {total_latency} {unit.unit_time}")
    
    # Example 4: Memory-bound vs Compute-bound analysis
    print("\n\n=== Workload Characterization ===")
    workloads = [
        ('Small GEMM', GEMM(['small_gemm', 1, 64, 64, 64], density=(1.0, 1.0, 1.0))),
        ('Large GEMM', GEMM(['large_gemm', 1, 2048, 2048, 2048], density=(1.0, 1.0, 1.0))),
        ('Attention', Logit(['attention', 1, 16, 512, 512, 64, 16], density=(1.0, 1.0, 1.0))),
        ('FC Layer', FC(['fc', 1, 1024, 4096], density=(1.0, 1.0, 1.0)))
    ]
    
    characterization = []
    for name, op in workloads:
        result = op.get_roofline(intel_cpu, unit)
        characterization.append({
            'Workload': name,
            'Bound': result['Bound'],
            'C/M ratio': f"{result['C/M ratio']:.2f}",
            'ISA': result['ISA_used'],
            'Parallel Eff': f"{result['Parallel_efficiency']:.2%}"
        })
    
    df_char = pd.DataFrame(characterization)
    print(df_char.to_string(index=False))


if __name__ == '__main__':
    main() 