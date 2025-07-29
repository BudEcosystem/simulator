# Performance Estimation API Reference

This document provides detailed API reference for the performance estimation features in llm-memory-calculator.

## Core Performance Functions

### estimate_prefill_performance()

Estimates performance for the prefill phase (processing input prompt).

```python
estimate_prefill_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    input_tokens: int = 2048,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False,
    **kwargs
) -> Dict
```

**Parameters:**
- `model`: Model identifier (e.g., 'llama2_7b', 'llama2_70b')
- `batch_size`: Number of sequences to process in parallel
- `input_tokens`: Length of input sequence (prompt length)
- `system_name`: Hardware configuration dict with keys:
  - `Flops`: TFLOPS
  - `Memory_size`: Memory in GB
  - `Memory_BW`: Memory bandwidth in GB/s
  - `ICN`: Interconnect bandwidth in GB/s
  - `real_values`: Whether to use real hardware values
- `bits`: Precision ('fp32', 'bf16', 'int8', 'int4')
- `tensor_parallel`: Tensor parallelism degree
- `pipeline_parallel`: Pipeline parallelism degree
- `expert_parallel`: Expert parallelism degree (for MoE models)
- `debug`: Whether to print debug information

**Returns:**
Dictionary with performance metrics:
- `Latency`: Total latency in milliseconds
- `Throughput`: Throughput in tokens/second
- `TTFT`: Time to first token in milliseconds
- `Memory_used`: Memory usage breakdown
- `Compute_breakdown`: Detailed compute analysis
- `Communication_breakdown`: Communication overhead analysis

### estimate_decode_performance()

Estimates performance for the decode phase (generating output tokens).

```python
estimate_decode_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    beam_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False
) -> Dict
```

**Parameters:**
- `beam_size`: Beam search width (1 for greedy decoding)
- `output_tokens`: Number of tokens to generate
- Other parameters same as `estimate_prefill_performance()`

**Returns:**
- `Latency`: Per-token latency in milliseconds
- `Total_latency`: Total generation time in milliseconds
- `TPOT`: Time per output token in milliseconds
- `Effective_throughput`: Throughput considering batch and beam size

### estimate_end_to_end_performance()

Combines prefill and decode for complete inference analysis.

```python
estimate_end_to_end_performance(
    model: str = 'llama2_7b',
    batch_size: int = 1,
    beam_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    system_name: Optional[Dict] = None,
    bits: str = 'bf16',
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    debug: bool = False
) -> Dict
```

**Returns:**
Comprehensive metrics including:
- `total_latency`: End-to-end latency in milliseconds
- `ttft`: Time to first token in milliseconds
- `average_tpot`: Average time per output token in milliseconds
- `total_throughput`: Overall throughput in tokens/second
- `prefill`: Detailed prefill phase results
- `decode`: Detailed decode phase results

### compare_performance_configurations()

Compare multiple hardware and parallelism configurations.

```python
compare_performance_configurations(
    model: str,
    configurations: List[Dict],
    batch_size: int = 1,
    input_tokens: int = 2048,
    output_tokens: int = 256,
    debug: bool = False
) -> List[Dict]
```

**Parameters:**
- `configurations`: List of configuration dicts, each containing:
  - `name`: Configuration name for identification
  - `system_name`: Hardware configuration
  - `tensor_parallel`: Tensor parallelism degree
  - `pipeline_parallel`: Pipeline parallelism degree
  - `bits`: Precision
  - Any other parameters for performance estimation

**Returns:**
List of dictionaries with performance results for each configuration, sorted by total throughput (descending).

## Practical Use Cases

### Use Case 1: Batch Size Optimization
```python
# Find optimal batch size for given hardware
hardware = get_hardware_config('A100_80GB')
batch_sizes = [1, 2, 4, 8, 16, 32]

for batch_size in batch_sizes:
    result = estimate_decode_performance(
        model='llama2_7b',
        batch_size=batch_size,
        input_tokens=2048,
        output_tokens=1,  # Single token for per-token metrics
        system_name=hardware,
        bits='bf16',
        tensor_parallel=4
    )
    
    latency = result['Latency']
    throughput = result.get('Effective_throughput', 0)
    print(f"Batch {batch_size}: {latency:.1f} ms/token, {throughput:.0f} tok/s")
```

### Use Case 2: Precision Impact Analysis
```python
# Compare different precisions
precisions = ['bf16', 'int8', 'int4']
hardware = get_hardware_config('A100_80GB')

for precision in precisions:
    try:
        result = estimate_end_to_end_performance(
            model='llama2_7b',
            batch_size=4,
            input_tokens=2048,
            output_tokens=128,
            system_name=hardware,
            bits=precision,
            tensor_parallel=4
        )
        
        throughput = result['total_throughput']
        ttft = result['ttft']
        print(f"{precision}: {throughput:.0f} tok/s, TTFT: {ttft:.1f} ms")
        
    except Exception as e:
        print(f"{precision}: Not supported - {e}")
```

### Use Case 3: Multi-Model Comparison
```python
# Compare different model sizes
models = ['llama2_7b', 'llama2_13b', 'llama2_70b']
hardware = get_hardware_config('H100_80GB')

for model in models:
    # Adjust TP based on model size
    tp = {'llama2_7b': 2, 'llama2_13b': 4, 'llama2_70b': 8}[model]
    
    try:
        result = estimate_end_to_end_performance(
            model=model,
            batch_size=4,
            input_tokens=2048,
            output_tokens=256,
            system_name=hardware,
            bits='bf16',
            tensor_parallel=tp
        )
        
        throughput = result['total_throughput']
        ttft = result['ttft']
        print(f"{model} (TP={tp}): {throughput:.0f} tok/s, TTFT: {ttft:.1f} ms")
        
    except Exception as e:
        print(f"{model}: Error - {e}")
```

## Performance Tips

1. **Use appropriate tensor parallelism**: 
   - 7B models: TP=2-4
   - 13B models: TP=4-8
   - 70B models: TP=8-16

2. **Cache hardware configs**: Reuse `get_hardware_config()` results

3. **Batch requests**: Use `compare_performance_configurations()` for multiple configs

4. **Monitor precision tradeoffs**: 
   - INT8 can provide 1.5-2x speedup vs BF16
   - INT4 can provide 2-3x speedup but may impact quality

5. **Consider memory constraints**: Larger batch sizes need more GPU memory

## Advanced Usage

For direct access to GenZ modeling functions:

```python
from llm_memory_calculator.genz.LLM_inference import (
    prefill_moddeling,
    decode_moddeling
)

# Direct access to GenZ modeling functions
# (Use wrapper functions above for easier API)
```