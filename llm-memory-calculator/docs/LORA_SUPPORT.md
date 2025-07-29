# LoRA (Low-Rank Adaptation) Support

This document describes the LoRA simulation capabilities integrated into llm-memory-calculator.

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to frozen pre-trained weights. This implementation simulates the performance impact of LoRA in transformer models.

## Features

- **Two strategies supported:**
  - **Merge Strategy**: One-time cost of merging adapters before inference
  - **Dynamic Strategy**: Per-token computational overhead during inference

- **Flexible targeting**: Apply LoRA to specific module types (attention, FFN, etc.)
- **Configurable rank**: Control the rank of adaptation matrices
- **Parallelism support**: Works with tensor, expert, and sequence parallelism

## Usage

### Basic Configuration

```python
from llm_memory_calculator.lora.config import LoraConfig
from llm_memory_calculator.genz.Models import get_configs, create_full_prefill_model

# Get a model configuration
model_config = get_configs('llama-7b')

# Configure LoRA
model_config.lora_config = LoraConfig(
    enabled=True,
    rank=16,                          # Low-rank dimension
    target_modules=['attn', 'ffn'],   # Apply to attention and FFN
    strategy='dynamic'                # or 'merge'
)

# Generate model with LoRA
csv_file = create_full_prefill_model(
    name=model_config,
    input_sequence_length=512,
    data_path='/tmp/output'
)
```

### Strategy Details

#### Merge Strategy
- Adds `LORA_MERGE` operations before base GEMM operations
- Simulates the one-time cost of merging LoRA weights: W' = W₀ + B @ A
- Best for production inference where adapters are pre-merged

#### Dynamic Strategy
- Replaces each targeted GEMM with a sequence:
  1. Base GEMM (original computation)
  2. `GEMM_LORA_A`: Input @ A (down-projection to rank r)
  3. `GEMM_LORA_B`: LoRA_A output @ B (up-projection)
  4. `ADD`: Base output + LoRA output
- Simulates runtime overhead of LoRA computation
- Useful for multi-adapter scenarios or development

### Operation Targeting

The implementation recognizes common operation names and maps them to module types:

- **Attention operations**: 'qkv', 'out proj', 'query', 'key', 'value'
- **FFN operations**: 'up+gate', 'down', 'up', 'gate', 'w1', 'w2', 'w3'

## Implementation Details

### New OpTypes
- `LORA_MERGE` (18): Merge operation for adapter weights
- `GEMM_LORA_A` (19): LoRA down-projection
- `GEMM_LORA_B` (20): LoRA up-projection  
- `ADD` (21): Element-wise addition

### Files
- `lora/config.py`: LoraConfig dataclass definition
- `lora/injection.py`: Core logic for injecting LoRA operations

### Integration Points
- Modified `ModelConfig` to include `lora_config` field
- Updated `get_language_model.py` to inject LoRA operations
- Added operation name matching for targeting specific layers

## Performance Considerations

- **Memory**: LoRA weights are marked as `All_offchip` residency
- **Compute**: Dynamic strategy adds 3 extra operations per adapted layer
- **Parallelism**: Dimensions are automatically adjusted for tensor parallelism

## Example: Performance Impact Analysis

```python
from llm_memory_calculator import estimate_end_to_end_performance
from llm_memory_calculator.lora.config import LoraConfig
from llm_memory_calculator.genz.Models import get_configs

# Compare performance with and without LoRA
model_config = get_configs('llama2_7b')

# Without LoRA
result_base = estimate_end_to_end_performance(
    model='llama2_7b',
    batch_size=4,
    input_tokens=2048,
    output_tokens=256,
    system_name=hardware,
    bits='bf16'
)

# With LoRA (dynamic strategy)
model_config.lora_config = LoraConfig(
    enabled=True,
    rank=16,
    target_modules=['attn', 'ffn'],
    strategy='dynamic'
)

# Note: Performance estimation with LoRA requires creating custom model
# This is an advanced use case requiring direct GenZ access
```

## Future Enhancements

- Support for different LoRA variants (QLoRA, ReLoRA)
- Per-layer rank configuration
- Adapter switching simulation
- Memory footprint analysis