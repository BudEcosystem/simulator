# Intel Xeon 6747P CPU Simulation Demo

This directory contains comprehensive demo scripts for analyzing LLM inference performance on the Intel Xeon 6747P (Granite Rapids) processor using the meta-llama/Llama-3.1-8B model.

## 🖥️ Hardware Specifications

**Intel Xeon 6747P (Granite Rapids)**
- **Cores**: 48 cores, 96 threads (Hyper-Threading)
- **Frequency**: 2.7 GHz base, 3.9 GHz max turbo
- **Cache**: 288 MB L3 cache (shared)
- **Memory**: DDR5-6400, 8-channel (409.6 GB/s bandwidth)
- **TDP**: 330W
- **Socket**: Intel Socket 4710
- **ISA Support**: AMX, AVX-512, AVX2, SSE
- **Launch**: February 2025 (MSRP: $6,497)

## 📁 Files Overview

### 1. CPU Configuration
- **`GenZ/cpu/cpu_configs.py`** - Contains the Intel Xeon 6747P configuration added to the GenZ CPU presets

### 2. Demo Scripts
- **`test_intel_xeon_6747p.py`** - Quick validation and basic performance test
- **`demo_intel_xeon_6747p_llama31_8b.py`** - Comprehensive performance analysis demo

### 3. Generated Reports (after running demo)
- **`intel_xeon_6747p_llama31_8b_report.md`** - Performance summary report
- **`intel_xeon_6747p_results.json`** - Detailed results in JSON format

## 🚀 Quick Start

### Prerequisites
1. Ensure GenZ is properly installed and configured
2. BudSimulator environment is set up
3. Required Python packages are installed

### Step 1: Quick Validation
Run the quick test to validate the CPU configuration:

```bash
cd BudSimulator
python test_intel_xeon_6747p.py
```

**Expected Output:**
```
Intel Xeon 6747P + Llama 3.1 8B - Quick Test & Validation
============================================================
🔍 Testing Intel Xeon 6747P CPU Configuration
==================================================
✅ CPU System Created Successfully!
   Cores: 48 cores, 96 threads
   Base Frequency: 2.7 GHz
   Max Turbo: 3.9 GHz
   L3 Cache: 0.27 GB
   Memory BW: 409.6 GB/s
   Vendor: intel, Architecture: granite_rapids

🔄 Testing Basic Prefill Operation
------------------------------
✅ Prefill Test Successful!
   Latency: XX.XX ms
   Throughput: XXXX.X tokens/s
```

### Step 2: Comprehensive Analysis
Run the full demo for detailed performance analysis:

```bash
python demo_intel_xeon_6747p_llama31_8b.py
```

This will perform:
- Basic prefill and decode analysis
- Batch size scaling analysis
- Sequence length impact assessment
- CPU vs GPU performance comparison
- Detailed CPU metrics analysis
- Generate comprehensive report

## 📊 Analysis Features

### 1. Basic Performance Analysis
- **TTFT (Time to First Token)**: Measures interactive responsiveness
- **TPOT (Time Per Output Token)**: Measures streaming generation speed
- **E2E Latency**: Complete inference time for full generation
- **Throughput**: Tokens processed per second

### 2. Batch Size Scaling
Tests performance with different batch sizes (1, 2, 4, 8, 16) to identify:
- Optimal batch size for throughput
- Scaling efficiency
- Latency per sample trends

### 3. Sequence Length Impact
Analyzes how context length affects performance:
- Memory scaling characteristics
- Compute vs memory bottlenecks
- Cache utilization patterns

### 4. CPU vs GPU Comparison
Compares Intel Xeon 6747P against:
- NVIDIA A100 80GB
- Performance ratios and speedup factors
- Use case recommendations

### 5. Detailed CPU Metrics
- Cache hierarchy performance
- Frequency scaling analysis
- ISA utilization patterns
- NUMA effects
- Theoretical vs actual performance

## 🎯 Key Performance Insights

### Optimal Use Cases
- **✅ CPU-only environments** (cloud, edge, on-premise)
- **✅ Cost-sensitive deployments** (lower infrastructure costs)
- **✅ Interactive applications** (excellent TTFT performance)
- **✅ Moderate throughput workloads** (good balance of latency/throughput)

### Performance Characteristics
- **Strong single-threaded performance** due to high turbo frequencies
- **Excellent memory bandwidth** with DDR5-6400 and 8-channel config
- **Large L3 cache** reduces DRAM access for moderate model sizes
- **AMX acceleration** provides significant speedup for BF16 operations

### Optimization Recommendations
1. **Enable AMX instructions** for maximum performance
2. **Use batch sizes 4-16** for optimal throughput/latency balance
3. **Consider NUMA placement** for multi-socket configurations
4. **Monitor memory bandwidth** utilization
5. **Leverage turbo boost** for single-threaded workloads

## 📋 Sample Results

Based on typical runs, you can expect performance characteristics like:

```
Key Performance Metrics:
- TTFT (Time to First Token): ~XX ms
- TPOT (Time Per Output Token): ~X.X ms  
- E2E Latency: ~XXX ms (for 512 token generation)
- Prefill Throughput: ~XXXX tokens/s
- Decode Throughput: ~XXXX tokens/s

CPU vs GPU Comparison:
- A100 is typically X-Xx faster for prefill
- A100 is typically X-Xx faster for decode
- CPU offers better cost/performance for certain use cases
```

## ⚙️ Technical Details

### CPU Configuration Parameters
```python
'intel_xeon_6747p': {
    'base_params': {
        'frequency': 2.7e9,          # 2.7 GHz base
        'bits': 'bf16',              # BF16 with AMX
        'compute_efficiency': 0.90,   # High efficiency
        'memory_efficiency': 0.82,    # DDR5 efficiency
    },
    'cpu_specific': CPUConfig(
        cores_per_socket=48,
        threads_per_core=2,
        # ... detailed cache, NUMA, ISA configurations
    )
}
```

### Cache Hierarchy
- **L1**: 32KB I-cache + 48KB D-cache per core
- **L2**: 2MB per core (estimated)
- **L3**: 288MB shared across all cores
- **Memory**: DDR5-6400, 8-channel, 409.6 GB/s

### ISA and Frequency Scaling
- **Base ISAs**: SSE, AVX2, AVX-512, AMX
- **Frequency penalties**: AVX-512 (-100MHz), AMX (-200MHz)
- **Turbo scaling**: From 3.9 GHz (1-2 cores) to 2.8 GHz (96 threads)

## 🔧 Customization

### Modifying CPU Configuration
To customize the CPU configuration, edit `GenZ/cpu/cpu_configs.py`:

```python
# Example: Modify cache sizes
'intel_xeon_6747p': {
    # ... existing config ...
    'cpu_specific': CPUConfig(
        # Modify L3 cache size
        l3_config=CacheConfig(
            size=384*1024*1024,  # Increase to 384MB
            # ... other parameters
        ),
        # ... rest of config
    )
}
```

### Adding Custom Test Scenarios
Extend the demo scripts with custom workloads:

```python
# Custom sequence length test
custom_lengths = [1024, 4096, 8192, 16384]
for length in custom_lengths:
    result = cpu_aware_prefill_moddeling(
        model='meta-llama/Llama-3.1-8B',
        batch_size=1,
        input_tokens=length,
        system_name='intel_xeon_6747p'
    )
    print(f"Length {length}: {result['Latency']:.2f} ms")
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Ensure GenZ CPU module is properly installed
   ```bash
   # Check if GenZ CPU module exists
   python -c "from GenZ.cpu import create_cpu_system; print('✅ CPU module available')"
   ```

2. **Model not found**: Verify Llama 3.1 8B model is configured in GenZ
   ```bash
   # Test with a known model first
   python -c "from GenZ.Models import get_configs; print(get_configs('llama'))"
   ```

3. **Performance seems unrealistic**: Check system configuration
   - Verify CPU specifications match expected values
   - Ensure proper frequency scaling is configured
   - Check cache sizes and memory bandwidth

### Debug Mode
Enable debug mode for detailed analysis:

```python
result = cpu_aware_prefill_moddeling(
    model='meta-llama/Llama-3.1-8B',
    batch_size=1,
    input_tokens=1024,
    system_name='intel_xeon_6747p',
    debug=True  # Enable detailed output
)
```

## 📚 Additional Resources

### Intel Xeon 6747P Documentation
- [Intel Xeon 6000 Series Product Brief](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html)
- [Granite Rapids Architecture Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/granite-rapids-architecture.html)
- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions.html)

### GenZ CPU Module
- [GenZ CPU Documentation](GenZ/cpu/README.md)
- [CPU Configuration Guide](GenZ/cpu/ADD_NEW_CPU_GUIDE.md)
- [Performance Analysis Examples](examples/cpu_llm_inference_example.py)

### Related Examples
- [CPU vs GPU Comparison](examples/cpu_analysis_example.py)
- [Throughput Metrics Analysis](examples/throughput_metrics_example.py)
- [LoRA Performance with CPU](examples/lora_example.py)

## 🤝 Contributing

To add support for additional CPUs or improve the analysis:

1. **Add CPU configurations** to `GenZ/cpu/cpu_configs.py`
2. **Create test scripts** following the pattern in this demo
3. **Update documentation** with new findings and optimizations
4. **Submit performance reports** for different workloads

## 📝 License

This demo is part of the BudSimulator project. See the main project license for terms and conditions.

---

**Note**: Performance results may vary based on system configuration, memory setup, cooling, and workload characteristics. The analysis provides estimates based on architectural models and should be validated against actual hardware when available. 