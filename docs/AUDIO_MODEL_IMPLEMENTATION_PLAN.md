# Audio/Speech Model Support Implementation Plan

## Overview

This document outlines the TDD-based implementation plan for adding audio/speech model support to the llm-memory-calculator. The implementation follows a test-driven approach where tests are written first, then implementation follows.

## Target Models

| Model | Type | Encoder | Decoder | Total Params |
|-------|------|---------|---------|--------------|
| Whisper Large v3 Turbo | Encoder-Decoder | 32 layers, 1280 dim | 4 layers, 1280 dim | 809M |
| Whisper Large v3 | Encoder-Decoder | 32 layers, 1280 dim | 32 layers, 1280 dim | 1.55B |
| Qwen2-Audio-7B | Audio-LLM | 32 layers, 1280 dim | 28 layers, 3584 dim (GQA) | 8.2B |
| Ultravox v0.5 | Audio-LLM | 32 layers, 1280 dim | 16 layers, 2048 dim (GQA) | 1.84B |

## Memory Components for Audio Models

### 1. Model Weights

```
Total Weights = Encoder Weights + Decoder Weights + Embeddings + Projector (if present)

Encoder Weights per layer:
- Self-attention: 4 × d_model² (Q, K, V, O projections)
- FFN: 2 × d_model × ffn_dim (or 3× for SwiGLU)
- LayerNorm: 2 × d_model

Decoder Weights per layer:
- Self-attention: 4 × d_model² (or less with GQA/MQA)
- Cross-attention: 4 × d_model² (Q from decoder, K/V from encoder)
- FFN: 2 × d_model × ffn_dim
- LayerNorm: 3 × d_model (before self-attn, cross-attn, FFN)

Embeddings:
- Input: vocab_size × d_model
- Output: vocab_size × d_model (often tied)
- Positional: max_positions × d_model (if learned)

Audio-specific:
- Conv layers: kernel_size × in_channels × out_channels × num_conv_layers
- Projector (Audio-LLM): audio_dim × text_dim × num_projector_layers
```

### 2. KV Cache Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache for Encoder-Decoder                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Encoder Self-Attention KV (STATIC after encoding)       │   │
│  │ Size: 2 × B × enc_layers × enc_seq_len × d_model        │   │
│  │ Computed once, reused for all decoder steps             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Cross-Attention KV (STATIC, from encoder output)        │   │
│  │ Size: 2 × B × dec_layers × enc_seq_len × d_model        │   │
│  │ Encoder output projected to K,V for each decoder layer  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Decoder Self-Attention KV (GROWS during generation)     │   │
│  │ Size: 2 × B × dec_layers × dec_seq_len × d_model        │   │
│  │ Increases by 1 position per generated token             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total KV = Encoder KV + Cross-Attention KV + Decoder KV
```

### 3. Activation Memory

```
Encoder Activations:
- Input: B × enc_seq_len × d_model
- Per layer: B × enc_seq_len × d_model × activation_multiplier
- Peak during FFN: B × enc_seq_len × ffn_dim

Decoder Activations:
- Input: B × dec_seq_len × d_model
- Per layer: B × dec_seq_len × d_model × activation_multiplier
- Cross-attention: B × dec_seq_len × enc_seq_len (attention weights)

Audio Preprocessing:
- Mel spectrogram: B × num_mel_bins × audio_frames
- Conv outputs: B × d_model × (audio_frames / stride)
```

## Implementation Phases

### Phase 1: Core Encoder-Decoder Support (Week 1-2)

#### 1.1 Model Type Detection
**File:** `calculator.py`

```python
def detect_model_type(self, config: dict) -> str:
    # Add encoder-decoder detection
    if config.get("is_encoder_decoder", False):
        return "encoder-decoder"
    if "encoder_layers" in config and "decoder_layers" in config:
        return "encoder-decoder"
    if "audio_config" in config:
        if "text_config" in config:
            return "audio-llm"
        return "encoder-decoder"
    # ... existing detection logic
```

#### 1.2 Encoder Parameter Counting
**File:** `parameter_counter.py`

```python
def calculate_encoder_params(self, config: dict) -> int:
    """Calculate encoder-only parameters."""
    d_model = config.get("d_model", config.get("hidden_size"))
    encoder_layers = config.get("encoder_layers", 0)
    encoder_heads = config.get("encoder_attention_heads")
    encoder_ffn = config.get("encoder_ffn_dim")

    params = 0

    # Self-attention per layer
    # Q, K, V, O projections
    params += encoder_layers * 4 * d_model * d_model

    # FFN per layer
    params += encoder_layers * 2 * d_model * encoder_ffn

    # LayerNorm per layer (2 per layer: pre-attention, pre-FFN)
    params += encoder_layers * 2 * d_model

    return params
```

#### 1.3 Decoder Parameter Counting (with Cross-Attention)
**File:** `parameter_counter.py`

```python
def calculate_decoder_params(self, config: dict) -> int:
    """Calculate decoder parameters including cross-attention."""
    d_model = config.get("d_model", config.get("hidden_size"))
    decoder_layers = config.get("decoder_layers", config.get("num_hidden_layers"))

    # Check for nested config (audio-llm models)
    if "text_config" in config:
        text_config = config["text_config"]
        d_model = text_config.get("hidden_size", d_model)
        decoder_layers = text_config.get("num_hidden_layers", decoder_layers)

    params = 0

    # Self-attention (with GQA support)
    num_heads = config.get("decoder_attention_heads", config.get("num_attention_heads"))
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = d_model // num_heads

    # Q projection
    params += decoder_layers * d_model * d_model
    # K, V projections (may be smaller with GQA)
    params += decoder_layers * 2 * d_model * (num_kv_heads * head_dim)
    # O projection
    params += decoder_layers * d_model * d_model

    # Cross-attention (if encoder-decoder)
    if config.get("is_encoder_decoder") or "encoder_layers" in config:
        encoder_d_model = config.get("d_model", d_model)
        if "audio_config" in config:
            encoder_d_model = config["audio_config"].get("d_model", encoder_d_model)

        # Q from decoder, K/V from encoder
        params += decoder_layers * d_model * d_model  # Q
        params += decoder_layers * 2 * encoder_d_model * d_model  # K, V
        params += decoder_layers * d_model * d_model  # O

    # FFN
    ffn_dim = config.get("decoder_ffn_dim", config.get("intermediate_size"))
    params += decoder_layers * 2 * d_model * ffn_dim

    # LayerNorm (3 per layer for encoder-decoder: self-attn, cross-attn, FFN)
    norm_count = 3 if config.get("is_encoder_decoder") else 2
    params += decoder_layers * norm_count * d_model

    return params
```

#### 1.4 KV Cache Calculation Methods
**File:** `calculator.py`

```python
def calculate_encoder_kv_cache(
    self,
    config: dict,
    batch_size: int,
    precision: str = "fp16"
) -> float:
    """Calculate encoder self-attention KV cache (static)."""
    d_model = self._get_encoder_d_model(config)
    encoder_layers = config.get("encoder_layers", 0)
    max_source_positions = config.get("max_source_positions", 1500)

    bytes_per_element = self._get_bytes_per_element(precision)

    # 2 (K+V) × batch × layers × seq_len × d_model
    elements = 2 * batch_size * encoder_layers * max_source_positions * d_model
    return (elements * bytes_per_element) / 1e9

def calculate_decoder_kv_cache(
    self,
    config: dict,
    batch_size: int,
    seq_length: int,
    precision: str = "fp16"
) -> float:
    """Calculate decoder self-attention KV cache (grows)."""
    d_model = self._get_decoder_d_model(config)
    decoder_layers = self._get_decoder_layers(config)
    num_kv_heads = self._get_num_kv_heads(config)
    head_dim = d_model // self._get_num_attention_heads(config)

    bytes_per_element = self._get_bytes_per_element(precision)

    # With GQA: 2 × batch × layers × seq_len × num_kv_heads × head_dim
    elements = 2 * batch_size * decoder_layers * seq_length * num_kv_heads * head_dim
    return (elements * bytes_per_element) / 1e9

def calculate_cross_attention_kv_cache(
    self,
    config: dict,
    batch_size: int,
    encoder_seq_length: int,
    precision: str = "fp16"
) -> float:
    """Calculate cross-attention KV cache (static, from encoder)."""
    decoder_layers = self._get_decoder_layers(config)
    encoder_d_model = self._get_encoder_d_model(config)

    bytes_per_element = self._get_bytes_per_element(precision)

    # 2 × batch × decoder_layers × encoder_seq_len × encoder_d_model
    elements = 2 * batch_size * decoder_layers * encoder_seq_length * encoder_d_model
    return (elements * bytes_per_element) / 1e9
```

### Phase 2: Audio-Specific Components (Week 3)

#### 2.1 Audio Input Memory
**File:** `calculator.py`

```python
def calculate_audio_input_memory(
    self,
    num_mel_bins: int = 128,
    audio_frames: int = 3000,
    batch_size: int = 1,
    precision: str = "fp32"
) -> float:
    """Calculate memory for mel spectrogram input."""
    bytes_per_element = self._get_bytes_per_element(precision)
    elements = batch_size * num_mel_bins * audio_frames
    return (elements * bytes_per_element) / 1e9

def calculate_audio_conv_memory(
    self,
    config: dict,
    audio_frames: int,
    batch_size: int,
    precision: str = "fp16"
) -> float:
    """Calculate memory for audio conv layers (Whisper-style)."""
    num_mel_bins = config.get("num_mel_bins", 128)
    d_model = self._get_encoder_d_model(config)

    bytes_per_element = self._get_bytes_per_element(precision)

    # Whisper: 2 Conv1D layers
    # Conv1: mel_bins -> d_model, stride=1
    # Conv2: d_model -> d_model, stride=2
    conv1_output = batch_size * d_model * audio_frames
    conv2_output = batch_size * d_model * (audio_frames // 2)

    elements = conv1_output + conv2_output
    return (elements * bytes_per_element) / 1e9
```

#### 2.2 Projector Memory (Audio-LLM)
**File:** `calculator.py`

```python
def calculate_projector_params(self, config: dict) -> int:
    """Calculate projector parameters for audio-LLM models."""
    if "projector_config" not in config and "audio_config" not in config:
        return 0

    audio_d_model = self._get_encoder_d_model(config)
    text_d_model = self._get_decoder_d_model(config)

    # Simple linear projection
    projector_params = audio_d_model * text_d_model

    # Check for multi-layer projector
    if "projector_config" in config:
        proj_config = config["projector_config"]
        if proj_config.get("projector_act") == "swiglu":
            # SwiGLU has 3x parameters
            projector_params *= 3

        stack_factor = proj_config.get("stack_factor", 1)
        # Stack factor affects input dimension
        projector_params = (audio_d_model * stack_factor) * text_d_model

    return projector_params
```

### Phase 3: Integration (Week 4)

#### 3.1 Update calculate_total_memory
**File:** `calculator.py`

```python
def calculate_total_memory(
    self,
    config: dict,
    batch_size: int = 1,
    seq_length: int = 2048,
    encoder_seq_length: int = None,  # NEW: for audio models
    precision: str = "fp16",
    tensor_parallel: int = 1,
    **kwargs
) -> MemoryReport:
    """Calculate total memory with encoder-decoder support."""

    self.model_type = self.detect_model_type(config)

    # Calculate weights
    param_count, weight_gb = self.calculate_model_weights(config, precision)

    # Calculate KV cache based on model type
    if self.model_type in ["encoder-decoder", "audio-llm"]:
        enc_seq_len = encoder_seq_length or config.get("max_source_positions", 1500)

        encoder_kv = self.calculate_encoder_kv_cache(config, batch_size, precision)
        decoder_kv = self.calculate_decoder_kv_cache(config, batch_size, seq_length, precision)
        cross_attn_kv = self.calculate_cross_attention_kv_cache(config, batch_size, enc_seq_len, precision)

        total_kv = encoder_kv + decoder_kv + cross_attn_kv
    else:
        total_kv = self.calculate_kv_cache(config, batch_size, seq_length, precision)
        encoder_kv = decoder_kv = cross_attn_kv = 0

    # Calculate activations
    activation_gb = self.calculate_activation_memory(config, batch_size, seq_length, precision)

    # Audio-specific memory
    audio_input_gb = 0
    if self.model_type in ["encoder-decoder", "audio-llm"]:
        if config.get("num_mel_bins"):
            audio_frames = (encoder_seq_length or 1500) * 2  # Approximate
            audio_input_gb = self.calculate_audio_input_memory(
                config.get("num_mel_bins", 128),
                audio_frames,
                batch_size,
                precision
            )

    # Apply tensor parallelism
    weight_gb_tp = weight_gb / tensor_parallel
    total_kv_tp = total_kv / tensor_parallel

    # Calculate total
    total_gb = weight_gb_tp + total_kv_tp + activation_gb + audio_input_gb

    return MemoryReport(
        weight_memory_gb=weight_gb_tp,
        kv_cache_gb=total_kv_tp,
        activation_memory_gb=activation_gb,
        state_memory_gb=0,
        image_memory_gb=audio_input_gb,  # Reuse for audio
        extra_work_gb=0,
        total_memory_gb=total_gb,
        # Extended fields for encoder-decoder
        encoder_kv_cache_gb=encoder_kv / tensor_parallel if encoder_kv else None,
        decoder_kv_cache_gb=decoder_kv / tensor_parallel if decoder_kv else None,
        cross_attention_kv_cache_gb=cross_attn_kv / tensor_parallel if cross_attn_kv else None,
    )
```

#### 3.2 Update MemoryReport
**File:** `types.py`

```python
@dataclass
class MemoryReport:
    # Existing fields...
    weight_memory_gb: float
    kv_cache_gb: float
    activation_memory_gb: float
    state_memory_gb: float
    image_memory_gb: float
    extra_work_gb: float
    total_memory_gb: float

    # NEW: Encoder-decoder breakdown (optional)
    encoder_weight_memory_gb: float = None
    decoder_weight_memory_gb: float = None
    encoder_kv_cache_gb: float = None
    decoder_kv_cache_gb: float = None
    cross_attention_kv_cache_gb: float = None
    audio_memory_gb: float = None
    projector_memory_gb: float = None
```

## Test Execution Plan

### Run Tests to Establish Baseline

```bash
# Run all audio model tests (expect failures initially)
cd /home/budadmin/simulator/llm-memory-calculator
pytest tests/test_audio_models.py -v

# Run specific test class
pytest tests/test_audio_models.py::TestEncoderWeightCalculation -v

# Run with coverage
pytest tests/test_audio_models.py --cov=llm_memory_calculator --cov-report=term-missing
```

### Expected Test Progression

| Phase | Tests Passing | Implementation |
|-------|---------------|----------------|
| Baseline | 0/40 | No changes |
| Phase 1.1 | 5/40 | Model type detection |
| Phase 1.2 | 12/40 | Encoder params |
| Phase 1.3 | 20/40 | Decoder params |
| Phase 1.4 | 30/40 | KV cache methods |
| Phase 2 | 35/40 | Audio components |
| Phase 3 | 40/40 | Integration |

## Validation Against Reference

### Memory Verification Table

| Model | Component | Expected | Calculated | Error |
|-------|-----------|----------|------------|-------|
| Whisper Turbo | Weights (fp16) | 1.51 GB | TBD | < 5% |
| Whisper Turbo | Encoder KV | 245 MB | TBD | < 5% |
| Whisper Turbo | Decoder KV | 9.2 MB | TBD | < 5% |
| Whisper Turbo | Cross-Attn KV | 30.7 MB | TBD | < 5% |
| Whisper v3 | Weights (fp16) | 2.88 GB | TBD | < 5% |
| Qwen2-Audio | Weights (bf16) | 15.28 GB | TBD | < 5% |
| Ultravox | Weights (bf16) | 3.43 GB | TBD | < 5% |

### External Validation

1. **vLLM Memory Profiler**: Compare with vLLM's memory usage
2. **HuggingFace Accelerate**: Use `estimate_memory_size()`
3. **PyTorch Memory Stats**: Profile actual inference

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `tests/test_audio_models.py` | NEW | TDD test cases |
| `calculator.py` | MODIFY | Add encoder-decoder methods |
| `parameter_counter.py` | MODIFY | Add encoder/decoder param counting |
| `types.py` | MODIFY | Extend MemoryReport |
| `config_normalizer.py` | MODIFY | Handle audio model configs |

## Dependencies

- No new external dependencies required
- Uses existing transformers for config loading (optional)
- pytest for testing

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Inaccurate param counting | Medium | High | Validate against HF model cards |
| Missing config variations | Medium | Medium | Add config normalization |
| Cross-attention complexity | Low | Medium | Start with standard patterns |
| Audio preprocessing variance | Medium | Low | Document assumptions |

## Success Criteria

1. All 40+ test cases passing
2. Memory estimates within 10% of actual usage
3. Support for all 11 vLLM audio models
4. No regression in existing decoder-only tests
5. Documentation complete
