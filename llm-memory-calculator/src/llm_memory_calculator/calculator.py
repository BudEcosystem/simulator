"""Core memory calculator for LLM models."""

import math
from typing import Dict, Any, Optional, List

from .types import MemoryReport
from .parameter_counter import UniversalParameterCounter
from .config_normalizer import ConfigNormalizer


class ModelMemoryCalculator:
    """
    Comprehensive memory calculator for various model architectures during inference.
    
    Handles all edge cases from production deployments.
    
    Supported attention mechanisms:
    - MHA (Multi-Head Attention): Standard attention with Q, K, V projections
    - MQA (Multi-Query Attention): Single K, V head shared across all Q heads
    - GQA (Grouped-Query Attention): Groups of Q heads share K, V heads
    - MLA (Multi-head Latent Attention): Compresses K, V into low-rank latent space
    
    MLA provides superior KV cache compression by projecting keys and values into a 
    lower-dimensional latent space before attention computation, reducing memory by 80-90%.
    """
    
    # Precision to bytes mapping
    PRECISION_BYTES = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2, 
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
        'int4': 0.5, 'uint4': 0.5,
        # Advanced quantization methods
        'mxfp4': 0.5,    # Microsoft MX-FP4
        'fp4': 0.5,      # 4-bit float
        'nf4': 0.5,      # NormalFloat4 (QLoRA)
        'fp8': 1.0,      # FP8 formats
        'fp8_e4m3': 1.0,
        'fp8_e5m2': 1.0,
        'awq': 0.5,      # Activation-aware Weight Quantization
        'gptq': 0.5,     # GPTQ 4-bit
        'squeezellm': 0.5,
    }
    
    def __init__(self):
        """Initialize the calculator."""
        self.param_counter = UniversalParameterCounter()
        self.model_type = None
        self.attention_type = None
    
    def detect_model_type(self, config: Dict[str, Any]) -> str:
        """Detect the model type from configuration."""
        # Check model_type field
        model_type = config.get('model_type', '').lower()

        # Audio-LLM models (check first)
        if 'audio_config' in config:
            # Check for text_config OR top-level LLM parameters (Ultravox style)
            if 'text_config' in config:
                return 'audio-llm'
            # Ultravox has hidden_size at top level instead of text_config
            if 'hidden_size' in config and config.get('hidden_size', 0) > 2000:
                return 'audio-llm'
            return 'encoder-decoder'

        # Pure audio model types
        if model_type in ['whisper', 'speech_to_text', 'wav2vec2', 'hubert']:
            return 'encoder-decoder'

        # Audio-LLM model types
        if model_type in ['qwen2_audio', 'ultravox', 'audio-flamingo']:
            return 'audio-llm'

        # Check for encoder-decoder by structure
        if 'encoder_layers' in config and 'decoder_layers' in config:
            return 'encoder-decoder'

        # Multimodal models (vision + text)
        if 'vision_config' in config and 'text_config' in config:
            return 'multimodal'

        # Mamba/SSM models
        if model_type in ['mamba', 's4', 'ssm', 'state-space']:
            return 'state-space'

        # Hybrid models
        if model_type in ['jamba', 'hybrid'] or 'mamba_config' in config:
            return 'hybrid'

        # Text-to-speech
        if model_type in ['bark', 'vall-e', 'tortoise-tts', 'xtts']:
            return 'text-to-speech'

        # Diffusion models
        if model_type in ['unet', 'diffusion', 'stable-diffusion', 'dit']:
            return 'diffusion'

        # Check for encoder-decoder
        if config.get('is_encoder_decoder', False):
            return 'encoder-decoder'

        # Check for hybrid architectures
        if 'mamba_config' in config or 'attention_layers' in config:
            return 'hybrid'

        # Infer from config structure if model_type not specified
        if 'num_attention_heads' in config and 'hidden_size' in config and 'num_hidden_layers' in config:
            # It's likely a transformer model
            if config.get('is_decoder', True) and not config.get('is_encoder_decoder', False):
                return 'decoder-only'
            elif config.get('is_encoder_decoder', False):
                return 'encoder-decoder'
            else:
                return 'encoder-only'

        return 'unknown'
    
    def detect_attention_type(self, config: Dict[str, Any]) -> Optional[str]:
        """Detect the attention mechanism type."""
        # For multimodal models, check text_config
        if 'text_config' in config and isinstance(config['text_config'], dict):
            config = config['text_config']
            
        # Check for MLA first (DeepSeek V2/V3 uses this)
        if any(key in config for key in ['q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim', 'qk_nope_head_dim']):
            return 'mla'  # Multi-Latent Attention
        if 'latent_attention_dim' in config or 'compressed_kv_dim' in config:
            return 'mla'  # Multi-Latent Attention
            
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 0))
        num_key_value_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
        
        if num_key_value_heads == 0 or num_attention_heads == 0:
            return None
        elif num_key_value_heads == 1:
            return 'mqa'  # Multi-Query Attention
        elif num_key_value_heads < num_attention_heads:
            return 'gqa'  # Grouped-Query Attention
        else:
            return 'mha'  # Multi-Head Attention
    
    def _get_activation_function_multiplier(self, config: Dict[str, Any]) -> int:
        """Get FFN matrix count based on activation function."""
        act_fn = (config.get('activation_function', config.get('hidden_act', 'gelu'))).lower()
        # SwiGLU, SiLU use 3 matrices (gate, up, down), others use 2 (up, down)
        if any(x in act_fn for x in ['swiglu', 'silu', 'swish']):
            return 3
        return 2
    
    def calculate_model_weights(self, config: Dict[str, Any], precision: str, respect_weight_tying: bool = True) -> tuple[int, float]:
        """Calculate model weight memory in GB and return (param_count, memory_gb)."""
        # Normalize config for consistent key handling
        config = ConfigNormalizer.normalize_config(config)

        # Get parameter count
        param_count = config.get('num_parameters')
        if not param_count:
            # Check for encoder-decoder / audio models
            model_type = self.detect_model_type(config)
            if model_type in ['encoder-decoder', 'audio-llm']:
                param_count = self._calculate_encoder_decoder_params(config)
            else:
                param_count = self.param_counter.count_parameters(config, respect_weight_tying=respect_weight_tying)

        # Check for mixed precision quantization
        if '_quantization' in config and config['_quantization'].get('has_mixed_precision'):
            weight_memory_gb = self._calculate_mixed_precision_memory(
                config, param_count, precision
            )
        else:
            # Standard calculation
            bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)
            weight_memory_gb = (param_count * bytes_per_param) / 1e9  # Use decimal GB to match API

        return param_count, weight_memory_gb

    def _calculate_encoder_decoder_params(self, config: Dict[str, Any]) -> int:
        """Calculate total parameters for encoder-decoder models."""
        encoder_params = self.calculate_encoder_params(config)
        decoder_params = self.calculate_decoder_params(config)
        projector_params = self.calculate_projector_params(config)

        return encoder_params + decoder_params + projector_params
    
    def _calculate_mixed_precision_memory(
        self, 
        config: Dict[str, Any],
        param_count: int,
        default_precision: str
    ) -> float:
        """
        Calculate weight memory with mixed precision quantization.
        
        Handles cases where different modules use different precisions,
        such as quantized experts but fp16 attention/embeddings.
        
        Args:
            config: Normalized config with _quantization metadata
            param_count: Total parameter count
            default_precision: Default precision for non-quantized modules
            
        Returns:
            Memory in GB
        """
        quant_info = config.get('_quantization', {})
        
        # Get quantization settings
        skip_patterns = quant_info.get('skip_modules', [])
        quant_bytes = quant_info.get('bytes_per_param', 0.5)
        default_bytes = self.PRECISION_BYTES.get(default_precision.lower(), 2)
        
        # Estimate proportion of non-quantized parameters
        skip_ratio = self._estimate_skip_ratio(config, skip_patterns, param_count)
        
        # Calculate mixed memory
        quantized_params = param_count * (1 - skip_ratio)
        non_quantized_params = param_count * skip_ratio
        
        memory_gb = (quantized_params * quant_bytes + 
                     non_quantized_params * default_bytes) / 1e9
        
        return memory_gb
    
    def _estimate_skip_ratio(
        self, 
        config: Dict[str, Any], 
        skip_patterns: List[str],
        total_params: int
    ) -> float:
        """
        Estimate proportion of parameters that are NOT quantized.
        
        Uses heuristics based on module patterns:
        - "*.self_attn" -> attention layers
        - "*.mlp.router" -> MoE router
        - "*embed*" -> embeddings
        - "*lm_head*" -> output head
        
        Args:
            config: Model configuration
            skip_patterns: List of module patterns to skip
            total_params: Total parameter count
            
        Returns:
            Ratio of non-quantized parameters (0.0 to 1.0)
        """
        if not skip_patterns:
            return 0.0
        
        skip_param_count = 0
        
        # Extract config values
        vocab_size = config.get('vocab_size', 50000)
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_hidden_layers', 24)
        num_heads = config.get('num_attention_heads', 32)
        num_kv_heads = config.get('num_key_value_heads', num_heads)
        head_dim = config.get('head_dim', hidden_size // num_heads)
        
        # Check if embeddings are tied
        tie_embeddings = config.get('tie_word_embeddings', False)
        
        for pattern in skip_patterns:
            pattern_lower = pattern.lower()
            
            if 'embed' in pattern_lower:
                # Embeddings: input + output (if not tied)
                embedding_params = vocab_size * hidden_size
                if not tie_embeddings:
                    embedding_params *= 2
                skip_param_count += embedding_params
            
            elif 'lm_head' in pattern_lower or ('output' in pattern_lower and 'layer' not in pattern_lower):
                # Output head (if not already counted in embeddings)
                if tie_embeddings:
                    head_params = vocab_size * hidden_size
                    skip_param_count += head_params
            
            elif 'attn' in pattern_lower or 'attention' in pattern_lower:
                # Attention layers: Q, K, V, O projections
                # Q projection: hidden -> num_heads * head_dim
                q_params = hidden_size * num_heads * head_dim
               # K, V projections: hidden -> num_kv_heads * head_dim
                kv_params = 2 * hidden_size * num_kv_heads * head_dim
                # O projection: num_heads * head_dim -> hidden
                o_params = num_heads * head_dim * hidden_size
                
                attn_params_per_layer = q_params + kv_params + o_params
                skip_param_count += num_layers * attn_params_per_layer
            
            elif 'router' in pattern_lower:
                # MoE router
                num_experts = config.get('n_routed_experts', 1)
                if num_experts > 1:
                    moe_freq = config.get('moe_layer_freq', 1)
                    num_moe_layers = num_layers // moe_freq if moe_freq > 0 else 0
                    router_params = num_moe_layers * hidden_size * num_experts
                    skip_param_count += router_params
        
        # Calculate ratio, capped at 1.0
        ratio = min(skip_param_count / total_params, 1.0) if total_params > 0 else 0.0
        
        return ratio

    
    def calculate_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str
    ) -> float:
        """Calculate KV cache memory in GB with per-layer attention type support."""
        # Normalize config for consistent key handling
        config = ConfigNormalizer.normalize_config(config)
        
        # For multimodal models, use text_config
        if self.model_type == 'multimodal' and 'text_config' in config:
            text_config = config['text_config']
            # Normalize text_config as well
            text_config = ConfigNormalizer.normalize_config(text_config)
        else:
            text_config = config
            
        attention_type = self.detect_attention_type(config)
        
        if not attention_type:
            return 0.0
        
        # Get bytes per element
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # Check for per-layer attention types (e.g., mixed sliding/full)
        if '_layer_metadata' in config and config['_layer_metadata'].get('has_mixed_attention'):
            return self._calculate_kv_cache_per_layer(
                text_config, config['_layer_metadata'], 
                batch_size, seq_length, bytes_per_element, attention_type
            )
        
        # Standard calculation: global sliding window or full seq
        # Handle sliding window attention
        if 'sliding_window' in text_config and text_config['sliding_window'] is not None:
            seq_length = min(seq_length, text_config['sliding_window'])
        
        # Calculate based on attention type
        if attention_type == 'mla':
            return self._calculate_kv_cache_mla(text_config, batch_size, seq_length, bytes_per_element)
        elif attention_type == 'mqa':
            return self._calculate_kv_cache_mqa(text_config, batch_size, seq_length, bytes_per_element)
        elif attention_type == 'gqa':
            return self._calculate_kv_cache_gqa(text_config, batch_size, seq_length, bytes_per_element)
        else:  # mha
            return self._calculate_kv_cache_mha(text_config, batch_size, seq_length, bytes_per_element)
    
    def _calculate_kv_cache_per_layer(
        self,
        config: Dict[str, Any],
        layer_metadata: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
        attention_type: str
    ) -> float:
        """
        Calculate KV cache for models with per-layer attention types.
        
        Handles mixed sliding/full attention layers (e.g., GPT-OSS-20B).
        
        Args:
            config: Model configuration
            layer_metadata: Layer metadata from config normalization
            batch_size: Batch size
            seq_length: Full sequence length
            bytes_per_element: Bytes per KV element
            attention_type: Attention mechanism type
            
        Returns:
            Total KV cache memory in GB
        """
        num_sliding_layers = layer_metadata.get('num_sliding_layers', 0)
        num_full_layers = layer_metadata.get('num_full_layers', 0)
        
        sliding_window = config.get('sliding_window', seq_length)
        
        total_kv_cache = 0.0
        
        # Calculate KV cache for sliding attention layers
        if num_sliding_layers > 0:
            effective_seq = min(seq_length, sliding_window)
            sliding_kv = self._calculate_kv_for_layers(
                config, num_sliding_layers, batch_size, 
                effective_seq, bytes_per_element, attention_type
            )
            total_kv_cache += sliding_kv
        
        # Calculate KV cache for full attention layers
        if num_full_layers > 0:
            full_kv = self._calculate_kv_for_layers(
                config, num_full_layers, batch_size,
                seq_length, bytes_per_element, attention_type
            )
            total_kv_cache += full_kv
        
        return total_kv_cache
    
    def _calculate_kv_for_layers(
        self,
        config: Dict[str, Any],
        num_layers: int,
        batch_size: int,
        seq_length: int,
        bytes_per_element: float,
        attention_type: str
    ) -> float:
        """Calculate KV cache for a specific number of layers.
        
        Args:
            config: Model configuration
            num_layers: Number of layers to calculate for
            batch_size: Batch size
            seq_length: Sequence length for these layers
            bytes_per_element: Bytes per KV element
            attention_type: Attention mechanism type
            
        Returns:
            KV cache memory in GB
        """
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_attention_heads = config.get('num_attention_heads', 12)
        num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
        head_dim = config.get('head_dim', hidden_size // num_attention_heads)
        
        if attention_type == 'mla':
            kv_lora_rank = config.get('kv_lora_rank', 512)
            compressed_kv_dim = config.get('compressed_kv_dim', kv_lora_rank)
            kv_elements = 2 * batch_size * num_layers * seq_length * compressed_kv_dim
        elif attention_type == 'mqa':
            kv_elements = 2 * batch_size * num_layers * seq_length * head_dim
        elif attention_type == 'gqa':
            kv_elements = 2 * batch_size * num_layers * seq_length * num_key_value_heads * head_dim
        else:  # mha
            kv_elements = 2 * batch_size * num_layers * seq_length * hidden_size
        
        return (kv_elements * bytes_per_element) / 1e9

    
    def _calculate_kv_cache_mha(self, config: Dict[str, Any], batch_size: int, seq_length: int, bytes_per_element: float) -> float:
        """Calculate KV cache for Multi-Head Attention."""
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        
        # Count only attention layers for hybrid models
        if self.model_type == 'hybrid':
            num_layers = self._count_attention_layers(config, num_layers)
        
        # 2 for K and V, full hidden size for each
        kv_elements = 2 * batch_size * num_layers * seq_length * hidden_size
        return (kv_elements * bytes_per_element) / 1e9
    
    def _calculate_kv_cache_mqa(self, config: Dict[str, Any], batch_size: int, seq_length: int, bytes_per_element: float) -> float:
        """Calculate KV cache for Multi-Query Attention."""
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        head_dim = hidden_size // num_attention_heads
        
        # Count only attention layers for hybrid models
        if self.model_type == 'hybrid':
            num_layers = self._count_attention_layers(config, num_layers)
        
        # MQA has single K, V head
        kv_elements = 2 * batch_size * num_layers * seq_length * head_dim
        return (kv_elements * bytes_per_element) / 1e9
    
    def _calculate_kv_cache_gqa(self, config: Dict[str, Any], batch_size: int, seq_length: int, bytes_per_element: float) -> float:
        """Calculate KV cache for Grouped-Query Attention."""
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        num_key_value_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
        head_dim = hidden_size // num_attention_heads
        
        # Count only attention layers for hybrid models
        if self.model_type == 'hybrid':
            num_layers = self._count_attention_layers(config, num_layers)
        
        # GQA has num_key_value_heads K, V heads
        kv_elements = 2 * batch_size * num_layers * seq_length * num_key_value_heads * head_dim
        return (kv_elements * bytes_per_element) / 1e9
    
    def _calculate_kv_cache_mla(self, config: Dict[str, Any], batch_size: int, seq_length: int, bytes_per_element: float) -> float:
        """Calculate KV cache for Multi-Latent Attention (DeepSeek V2/V3)."""
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        
        # MLA compresses KV into latent dimensions
        kv_lora_rank = config.get('kv_lora_rank', 512)  # DeepSeek V3 default
        compressed_kv_dim = config.get('compressed_kv_dim', kv_lora_rank)
        
        # Count only attention layers for hybrid models
        if self.model_type == 'hybrid':
            num_layers = self._count_attention_layers(config, num_layers)
        
        # Latent KV cache is much smaller
        kv_elements = 2 * batch_size * num_layers * seq_length * compressed_kv_dim
        return (kv_elements * bytes_per_element) / 1e9
    
    def _count_attention_layers(self, config: Dict[str, Any], total_layers: int) -> int:
        """Count the number of attention layers in hybrid models."""
        # Check for explicit layer types
        layer_types = config.get('layer_types', [])
        if layer_types:
            return sum(1 for lt in layer_types if 'attention' in str(lt).lower())

        # Use attention ratio if specified
        attention_ratio = config.get('attention_ratio', 0.5)
        return int(total_layers * attention_ratio)

    # =========================================================================
    # Encoder-Decoder / Audio Model Support
    # =========================================================================

    def _get_encoder_d_model(self, config: Dict[str, Any]) -> int:
        """Get encoder hidden dimension."""
        # Check for audio_config (Qwen2-Audio, Ultravox style)
        if 'audio_config' in config:
            audio_config = config['audio_config']
            return audio_config.get('d_model', audio_config.get('hidden_size', 1280))
        # Whisper-style config
        return config.get('d_model', config.get('hidden_size', 1280))

    def _get_decoder_d_model(self, config: Dict[str, Any]) -> int:
        """Get decoder hidden dimension."""
        # Check for text_config (Qwen2-Audio style)
        if 'text_config' in config:
            text_config = config['text_config']
            # text_config may be incomplete, infer from intermediate_size if needed
            if 'hidden_size' in text_config:
                return text_config['hidden_size']
            # Infer from intermediate_size (typically 4x or 2.67x hidden_size)
            if 'intermediate_size' in text_config:
                # Qwen2 uses ~2.75x ratio: 4096 * 2.6875 ≈ 11008
                return int(text_config['intermediate_size'] / 2.6875)
            return text_config.get('d_model', 4096)
        # Ultravox style: audio_config + top-level LLM params
        if 'audio_config' in config and 'hidden_size' in config:
            return config['hidden_size']
        # Whisper-style (shared d_model for encoder/decoder)
        return config.get('d_model', config.get('hidden_size', 1280))

    def _get_encoder_layers(self, config: Dict[str, Any]) -> int:
        """Get number of encoder layers."""
        if 'audio_config' in config:
            return config['audio_config'].get('encoder_layers', 32)
        return config.get('encoder_layers', 0)

    def _get_decoder_layers(self, config: Dict[str, Any]) -> int:
        """Get number of decoder layers."""
        if 'text_config' in config:
            text_cfg = config['text_config']
            return text_cfg.get('num_hidden_layers', text_cfg.get('decoder_layers', 32))
        # For encoder-decoder models (Whisper), prefer decoder_layers over num_hidden_layers
        # num_hidden_layers in Whisper config refers to encoder, not decoder
        if 'decoder_layers' in config:
            return config['decoder_layers']
        # For audio-LLM with top-level params (Ultravox), use num_hidden_layers
        if 'audio_config' in config and 'hidden_size' in config:
            return config.get('num_hidden_layers', 32)
        return config.get('num_hidden_layers', 4)

    def _get_num_attention_heads(self, config: Dict[str, Any]) -> int:
        """Get number of attention heads for decoder."""
        if 'text_config' in config:
            text_cfg = config['text_config']
            if 'num_attention_heads' in text_cfg:
                return text_cfg['num_attention_heads']
            # Infer from intermediate_size / hidden_size ratio for Qwen2 family
            hidden_size = self._get_decoder_d_model(config)
            # Qwen2-7B: 4096 hidden, 28 heads -> head_dim ~146 (non-standard)
            # Common: head_dim = 128 -> num_heads = hidden_size / 128
            return hidden_size // 128  # Default to 128 head_dim
        # Ultravox style: audio_config + top-level LLM params
        if 'audio_config' in config and 'num_attention_heads' in config:
            return config['num_attention_heads']
        return config.get('decoder_attention_heads', config.get('num_attention_heads', 20))

    def _get_num_kv_heads(self, config: Dict[str, Any]) -> int:
        """Get number of key-value heads for decoder (for GQA)."""
        if 'text_config' in config:
            text_cfg = config['text_config']
            if 'num_key_value_heads' in text_cfg:
                return text_cfg['num_key_value_heads']
            # Qwen2 family typically uses GQA with 4 KV heads
            if text_cfg.get('model_type', '').lower().startswith('qwen'):
                return 4
            num_heads = self._get_num_attention_heads(config)
            return num_heads
        # Ultravox style: audio_config + top-level LLM params
        if 'audio_config' in config and 'num_attention_heads' in config:
            num_heads = config['num_attention_heads']
            return config.get('num_key_value_heads', num_heads)
        num_heads = config.get('decoder_attention_heads', config.get('num_attention_heads', 20))
        return config.get('num_key_value_heads', num_heads)

    def _get_bytes_per_element(self, precision: str) -> float:
        """Get bytes per element for given precision."""
        return self.PRECISION_BYTES.get(precision.lower(), 2)

    def calculate_encoder_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate encoder-only parameters for encoder-decoder models.

        For Whisper-like encoders:
        - Self-attention: 4 × d_model² per layer (Q, K, V, O)
        - FFN: 2 × d_model × ffn_dim per layer
        - LayerNorm: 2 × d_model per layer
        - Conv layers (audio): 2 conv1d layers
        """
        # Get encoder config
        if 'audio_config' in config:
            enc_config = config['audio_config']
        else:
            enc_config = config

        d_model = enc_config.get('d_model', enc_config.get('hidden_size', 1280))
        encoder_layers = enc_config.get('encoder_layers', 32)
        encoder_ffn_dim = enc_config.get('encoder_ffn_dim', d_model * 4)
        num_mel_bins = enc_config.get('num_mel_bins', 128)

        params = 0

        # Audio conv layers (Whisper-style: 2 Conv1D)
        # Conv1: num_mel_bins -> d_model, kernel=3
        # Conv2: d_model -> d_model, kernel=3
        conv1_params = num_mel_bins * d_model * 3 + d_model  # weights + bias
        conv2_params = d_model * d_model * 3 + d_model
        params += conv1_params + conv2_params

        # Position embedding
        max_source_positions = enc_config.get('max_source_positions', 1500)
        params += max_source_positions * d_model

        # Self-attention per layer (Q, K, V, O)
        params += encoder_layers * 4 * d_model * d_model

        # FFN per layer (up + down)
        params += encoder_layers * 2 * d_model * encoder_ffn_dim

        # LayerNorm per layer (2 per layer: pre-attn, pre-ffn) + final
        params += (2 * encoder_layers + 1) * d_model

        return params

    def calculate_decoder_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate decoder-only parameters for encoder-decoder models.

        Includes:
        - Self-attention
        - Cross-attention (for encoder-decoder)
        - FFN
        - LayerNorm
        - Embeddings
        """
        # Use helper methods for consistent extraction
        d_model = self._get_decoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)
        num_heads = self._get_num_attention_heads(config)
        num_kv_heads = self._get_num_kv_heads(config)
        head_dim = d_model // num_heads if num_heads > 0 else 64

        # Determine if this is an audio-LLM (uses projector) vs encoder-decoder (uses cross-attention)
        is_audio_llm = (
            ('text_config' in config) or
            ('audio_config' in config and 'hidden_size' in config and config['hidden_size'] > 2000)
        )

        # Get decoder-specific config for other params
        if 'text_config' in config:
            dec_config = config['text_config']
        elif 'audio_config' in config and 'hidden_size' in config:
            # Ultravox style: top-level params
            dec_config = config
        else:
            dec_config = config

        vocab_size = dec_config.get('vocab_size', config.get('vocab_size', 51866))
        decoder_ffn_dim = dec_config.get('intermediate_size', dec_config.get('decoder_ffn_dim', d_model * 4))

        params = 0

        # Embeddings (input + output, often tied)
        tie_embeddings = dec_config.get('tie_word_embeddings', True)
        params += vocab_size * d_model
        if not tie_embeddings:
            params += vocab_size * d_model

        # Position embeddings (if not RoPE)
        max_positions = dec_config.get('max_position_embeddings', dec_config.get('max_target_positions', 448))
        use_rope = dec_config.get('rope_theta', 0) > 0 or dec_config.get('rope_scaling') is not None
        if not use_rope and max_positions > 0:
            params += max_positions * d_model

        # Self-attention per layer
        # Q projection: d_model -> num_heads * head_dim
        params += decoder_layers * d_model * (num_heads * head_dim)
        # K, V projections: d_model -> num_kv_heads * head_dim (with GQA)
        params += decoder_layers * 2 * d_model * (num_kv_heads * head_dim)
        # O projection
        params += decoder_layers * (num_heads * head_dim) * d_model

        # Cross-attention per layer (if encoder-decoder, not audio-LLM)
        # Audio-LLM models typically fuse via projector, not cross-attention
        if not is_audio_llm and (config.get('is_encoder_decoder') or 'encoder_layers' in config):
            encoder_d_model = self._get_encoder_d_model(config)
            # Q from decoder, K/V from encoder
            params += decoder_layers * d_model * d_model  # Q
            params += decoder_layers * 2 * encoder_d_model * d_model  # K, V from encoder
            params += decoder_layers * d_model * d_model  # O

        # FFN per layer
        # Check for SwiGLU
        act_fn = dec_config.get('hidden_act', dec_config.get('activation_function', 'gelu')).lower()
        if any(x in act_fn for x in ['swiglu', 'silu', 'swish']):
            params += decoder_layers * 3 * d_model * decoder_ffn_dim
        else:
            params += decoder_layers * 2 * d_model * decoder_ffn_dim

        # LayerNorm (3 per layer for encoder-decoder: self-attn, cross-attn, FFN; 2 for LLM)
        norms_per_layer = 3 if not is_audio_llm else 2
        params += (norms_per_layer * decoder_layers + 1) * d_model

        return params

    def calculate_projector_params(self, config: Dict[str, Any]) -> int:
        """
        Calculate projector parameters for audio-LLM models.

        The projector maps audio encoder output to text decoder input space.
        """
        # Check if this is an audio-LLM model
        if 'audio_config' not in config:
            return 0
        # Must have text_config OR top-level hidden_size (Ultravox style)
        if 'text_config' not in config and 'hidden_size' not in config:
            return 0

        audio_d_model = self._get_encoder_d_model(config)
        text_d_model = self._get_decoder_d_model(config)

        # Base linear projection
        params = audio_d_model * text_d_model

        # Check for projector config (could be nested or at top level)
        proj_config = config.get('projector_config', {})

        # Stack factor - check both projector_config and top level
        stack_factor = proj_config.get('stack_factor', config.get('stack_factor', 1))
        if stack_factor > 1:
            params = (audio_d_model * stack_factor) * text_d_model

        # SwiGLU projector has 3x parameters - check both locations
        projector_act = proj_config.get('projector_act', config.get('projector_act', '')).lower()
        if projector_act == 'swiglu':
            params *= 3

        # Layer norm in projector - check both locations
        if proj_config.get('projector_ln_mid', config.get('projector_ln_mid', False)):
            params += text_d_model

        return params

    def calculate_encoder_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        precision: str = "fp16"
    ) -> float:
        """
        Calculate encoder self-attention KV cache (STATIC after encoding).

        This is computed once when processing audio input and reused
        during all decoder steps.

        Size: 2 × batch × encoder_layers × encoder_seq_len × d_model
        """
        encoder_d_model = self._get_encoder_d_model(config)
        encoder_layers = self._get_encoder_layers(config)

        # Get max source positions (audio sequence length)
        if 'audio_config' in config:
            max_source_positions = config['audio_config'].get('max_source_positions', 1500)
        else:
            max_source_positions = config.get('max_source_positions', 1500)

        bytes_per_element = self._get_bytes_per_element(precision)

        # 2 (K+V) × batch × layers × seq_len × d_model
        elements = 2 * batch_size * encoder_layers * max_source_positions * encoder_d_model
        return (elements * bytes_per_element) / 1e9

    def calculate_decoder_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str = "fp16"
    ) -> float:
        """
        Calculate decoder self-attention KV cache (GROWS during generation).

        This grows by one position per generated token.
        Supports GQA where num_kv_heads < num_attention_heads.

        Size: 2 × batch × decoder_layers × seq_len × num_kv_heads × head_dim
        """
        decoder_d_model = self._get_decoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)
        num_heads = self._get_num_attention_heads(config)
        num_kv_heads = self._get_num_kv_heads(config)
        head_dim = decoder_d_model // num_heads

        bytes_per_element = self._get_bytes_per_element(precision)

        # With GQA: 2 × batch × layers × seq_len × num_kv_heads × head_dim
        elements = 2 * batch_size * decoder_layers * seq_length * num_kv_heads * head_dim
        return (elements * bytes_per_element) / 1e9

    def calculate_cross_attention_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        encoder_seq_length: int,
        precision: str = "fp16"
    ) -> float:
        """
        Calculate cross-attention KV cache (STATIC, from encoder output).

        This is the encoder output projected to K,V for each decoder layer.
        It's computed once and reused for all decoder steps.

        Size: 2 × batch × decoder_layers × encoder_seq_len × d_model
        """
        # Cross-attention uses encoder hidden size for K,V
        encoder_d_model = self._get_encoder_d_model(config)
        decoder_layers = self._get_decoder_layers(config)

        bytes_per_element = self._get_bytes_per_element(precision)

        # 2 × batch × decoder_layers × encoder_seq_len × encoder_d_model
        elements = 2 * batch_size * decoder_layers * encoder_seq_length * encoder_d_model
        return (elements * bytes_per_element) / 1e9

    def calculate_audio_input_memory(
        self,
        num_mel_bins: int = 128,
        audio_frames: int = 3000,
        batch_size: int = 1,
        precision: str = "fp32"
    ) -> float:
        """
        Calculate memory for mel spectrogram input.

        Whisper uses 128 mel bins × ~3000 frames for 30s audio.
        """
        bytes_per_element = self._get_bytes_per_element(precision)
        elements = batch_size * num_mel_bins * audio_frames
        return (elements * bytes_per_element) / 1e9

    def calculate_audio_conv_memory(
        self,
        config: Dict[str, Any],
        audio_frames: int,
        batch_size: int,
        precision: str = "fp16"
    ) -> float:
        """
        Calculate memory for audio conv layer activations.

        Whisper uses 2 Conv1D layers:
        - Conv1: mel_bins -> d_model, stride=1
        - Conv2: d_model -> d_model, stride=2
        """
        encoder_d_model = self._get_encoder_d_model(config)
        bytes_per_element = self._get_bytes_per_element(precision)

        # Conv1 output: batch × d_model × audio_frames
        conv1_elements = batch_size * encoder_d_model * audio_frames
        # Conv2 output: batch × d_model × (audio_frames // 2)
        conv2_elements = batch_size * encoder_d_model * (audio_frames // 2)

        total_elements = conv1_elements + conv2_elements
        return (total_elements * bytes_per_element) / 1e9
    
    def calculate_activation_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str
    ) -> float:
        """Calculate activation memory for forward pass."""
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        if self.model_type == 'multimodal':
            # Multimodal models have separate text and vision activations
            text_config = config.get('text_config', config)
            vision_config = config.get('vision_config', {})
            
            # Text activations
            text_hidden = text_config.get('hidden_size', 768)
            text_layers = text_config.get('num_hidden_layers', 12)
            text_activation_multiplier = 10
            if seq_length > 8192:
                text_activation_multiplier = 15
            text_elements = batch_size * seq_length * text_hidden * text_activation_multiplier
            
            # Vision activations
            vision_hidden = vision_config.get('hidden_size', 768)
            image_size = vision_config.get('image_size', 224)
            patch_size = vision_config.get('patch_size', 16)
            num_patches = (image_size // patch_size) ** 2
            vision_elements = batch_size * num_patches * vision_hidden * 4
            
            total_elements = text_elements + vision_elements
            return (total_elements * bytes_per_element) / 1e9
        else:
            # Standard calculation for non-multimodal
            hidden_size = config.get('hidden_size', config.get('d_model', 768))
            num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
            
            # Base activation memory (hidden states through the network)
            # Typically need to keep activations for: residual connections, attention outputs, FFN outputs
            activation_multiplier = 10  # Conservative estimate for peak activation memory
            
            # For very long sequences, activation memory can be dominant
            if seq_length > 8192:
                activation_multiplier = 15  # More buffer for long sequences
            
            activation_elements = batch_size * seq_length * hidden_size * activation_multiplier
            return (activation_elements * bytes_per_element) / 1e9
    
    def calculate_state_memory(
        self,
        config: Dict[str, Any],
        batch_size: int,
        precision: str
    ) -> float:
        """Calculate state memory for SSM/Mamba models."""
        if self.model_type not in ['state-space', 'hybrid']:
            return 0.0
        
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        
        # For hybrid models, count only SSM layers
        if self.model_type == 'hybrid':
            layer_types = config.get('layer_types', [])
            if layer_types:
                num_layers = sum(1 for lt in layer_types if any(x in str(lt).lower() for x in ['mamba', 'ssm']))
            else:
                attention_ratio = config.get('attention_ratio', 0.5)
                num_layers = int(num_layers * (1 - attention_ratio))
        
        # SSM state dimensions
        state_size = config.get('state_size', config.get('d_state', 16))
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        expand_factor = config.get('expand_factor', config.get('expand', 2))
        
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # State memory is constant regardless of sequence length
        state_elements = batch_size * num_layers * state_size * hidden_size * expand_factor
        return (state_elements * bytes_per_element) / 1e9

    def calculate_lora_adapter_memory(
        self,
        config: Dict[str, Any],
        lora_config: Any,
        precision: str,
        tensor_parallel: int = 1
    ) -> float:
        """
        Calculate memory for LoRA adapters using vLLM-style allocation.

        Following vLLM's approach:
        - Pre-allocates memory for max_loras adapters
        - Each LoRA layer has A and B matrices
        - LoRA A: [max_loras, 1, max_lora_rank, input_size]
        - LoRA B: [max_loras, 1, output_size, max_lora_rank]

        Args:
            config: Model configuration
            lora_config: LoRA configuration with max_loras, max_lora_rank, etc.
            precision: Model precision for base model
            tensor_parallel: Tensor parallelism degree

        Returns:
            Total LoRA adapter memory in GB
        """
        if not lora_config or not lora_config.enabled:
            return 0.0

        # For multimodal models, use text_config
        if self.model_type == 'multimodal' and 'text_config' in config:
            text_config = config['text_config']
        else:
            text_config = config

        # Get LoRA dtype (defaults to model precision if 'auto')
        lora_dtype = lora_config.lora_dtype
        if lora_dtype == 'auto':
            lora_dtype = precision
        bytes_per_element = self.PRECISION_BYTES.get(lora_dtype.lower(), 2)

        # Extract model dimensions
        hidden_size = text_config.get('hidden_size', text_config.get('d_model', 768))
        intermediate_size = text_config.get('intermediate_size',
                                           text_config.get('ffn_dim', hidden_size * 4))
        num_layers = text_config.get('num_hidden_layers', text_config.get('n_layers', 12))

        # Count only attention layers for hybrid models
        if self.model_type == 'hybrid':
            num_attention_layers = self._count_attention_layers(text_config, num_layers)
        else:
            num_attention_layers = num_layers

        max_loras = lora_config.max_loras
        max_lora_rank = lora_config.max_lora_rank
        target_modules = lora_config.target_modules

        total_elements = 0

        # Calculate memory for each target module type
        for module in target_modules:
            module_lower = module.lower()

            # Attention modules: Q, K, V, O projections
            if any(x in module_lower for x in ['attn', 'attention', 'qkv', 'query', 'key', 'value', 'out']):
                # Each attention layer typically has 4 projections: Q, K, V, O
                # Each projection: hidden_size -> hidden_size
                num_projections = 4

                for _ in range(num_projections):
                    # LoRA A: [max_loras, 1, max_lora_rank, input_size]
                    lora_a_elements = max_loras * 1 * max_lora_rank * hidden_size

                    # LoRA B: [max_loras, 1, output_size, max_lora_rank]
                    lora_b_elements = max_loras * 1 * hidden_size * max_lora_rank

                    # Apply tensor parallelism sharding
                    if lora_config.fully_sharded_loras:
                        # Both A and B are sharded
                        lora_a_elements = lora_a_elements / tensor_parallel
                        lora_b_elements = lora_b_elements / tensor_parallel
                    else:
                        # Only B is sharded by default
                        lora_b_elements = lora_b_elements / tensor_parallel

                    total_elements += (lora_a_elements + lora_b_elements) * num_attention_layers

            # FFN modules: up, down, gate projections
            if any(x in module_lower for x in ['ffn', 'mlp', 'feed_forward', 'up', 'down', 'gate']):
                # Typical FFN has 3 projections for SwiGLU: up, down, gate
                # or 2 for standard: up, down
                act_fn = text_config.get('activation_function',
                                        text_config.get('hidden_act', 'gelu')).lower()
                num_ffn_projections = 3 if any(x in act_fn for x in ['swiglu', 'silu', 'swish']) else 2

                for i in range(num_ffn_projections):
                    if i < num_ffn_projections - 1:  # up/gate projections
                        input_dim = hidden_size
                        output_dim = intermediate_size
                    else:  # down projection
                        input_dim = intermediate_size
                        output_dim = hidden_size

                    # LoRA A: [max_loras, 1, max_lora_rank, input_size]
                    lora_a_elements = max_loras * 1 * max_lora_rank * input_dim

                    # LoRA B: [max_loras, 1, output_size, max_lora_rank]
                    lora_b_elements = max_loras * 1 * output_dim * max_lora_rank

                    # Apply tensor parallelism sharding
                    if lora_config.fully_sharded_loras:
                        lora_a_elements = lora_a_elements / tensor_parallel
                        lora_b_elements = lora_b_elements / tensor_parallel
                    else:
                        lora_b_elements = lora_b_elements / tensor_parallel

                    total_elements += (lora_a_elements + lora_b_elements) * num_layers

        # Convert to GB
        return (total_elements * bytes_per_element) / 1e9

    def calculate_total_memory(
        self,
        config: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 2048,
        precision: str = 'fp16',
        tensor_parallel: int = 1,
        framework_overhead: float = 1.2,
        include_gradients: bool = False,
        decode_length: Optional[int] = None,
        num_images: Optional[int] = None,
        image_resolution: int = 1024,
        lora_config: Optional[Any] = None,
        respect_weight_tying: bool = True,
        encoder_seq_length: Optional[int] = None
    ) -> MemoryReport:
        """
        Calculate total memory requirements for model inference.

        Args:
            config: Model configuration dictionary
            batch_size: Batch size for inference
            seq_length: Maximum sequence length (context + generation) or decoder output length
            precision: Model precision (fp32, fp16, bf16, int8, int4)
            tensor_parallel: Tensor parallelism degree
            framework_overhead: Multiplicative overhead for framework/kernel memory (default 1.2)
            include_gradients: Include gradient memory (for training)
            decode_length: Length of tokens to generate (defaults to seq_length)
            num_images: Number of images for multimodal models
            image_resolution: Image resolution for vision models
            lora_config: Optional LoRA configuration for adapter memory calculation
            respect_weight_tying: Whether to respect tie_word_embeddings config (default True)
            encoder_seq_length: Encoder sequence length for encoder-decoder models (audio/speech)

        Returns:
            MemoryReport with detailed breakdown
        """
        # Detect model and attention types
        self.model_type = self.detect_model_type(config)
        self.attention_type = self.detect_attention_type(config)

        # Calculate model weights
        param_count, weight_memory = self.calculate_model_weights(config, precision, respect_weight_tying)

        # Divide weights by tensor parallelism
        weight_memory = weight_memory / tensor_parallel

        # Calculate KV cache based on model type
        encoder_kv_cache = 0.0
        decoder_kv_cache = 0.0
        cross_attn_kv_cache = 0.0

        if self.model_type in ['encoder-decoder', 'audio-llm']:
            # Get encoder sequence length (default from config or parameter)
            enc_seq_len = encoder_seq_length
            if enc_seq_len is None:
                if 'audio_config' in config:
                    enc_seq_len = config['audio_config'].get('max_source_positions', 1500)
                else:
                    enc_seq_len = config.get('max_source_positions', 1500)

            # Encoder self-attention KV (static)
            encoder_kv_cache = self.calculate_encoder_kv_cache(config, batch_size, precision)

            # Decoder self-attention KV (grows with output)
            decoder_kv_cache = self.calculate_decoder_kv_cache(config, batch_size, seq_length, precision)

            # Cross-attention KV (static, based on encoder output)
            cross_attn_kv_cache = self.calculate_cross_attention_kv_cache(
                config, batch_size, enc_seq_len, precision
            )

            kv_cache = encoder_kv_cache + decoder_kv_cache + cross_attn_kv_cache
        else:
            # Standard decoder-only model
            kv_cache = self.calculate_kv_cache(config, batch_size, seq_length, precision)

        # Divide KV cache by tensor parallelism (each device holds a slice)
        kv_cache = kv_cache / tensor_parallel

        # Calculate activations
        activations = self.calculate_activation_memory(config, batch_size, seq_length, precision)
        
        # Calculate state memory (for SSM models)
        state_memory = self.calculate_state_memory(config, batch_size, precision)

        # Calculate LoRA adapter memory (if enabled)
        lora_memory = 0.0
        if lora_config:
            lora_memory = self.calculate_lora_adapter_memory(
                config, lora_config, precision, tensor_parallel
            )

        # Calculate image memory (for multimodal models)
        image_memory = 0.0
        if num_images and num_images > 0:
            # Estimate based on common vision encoder sizes
            patches_per_image = (image_resolution // 16) ** 2  # Assuming 16x16 patches
            hidden_size = config.get('vision_config', {}).get('hidden_size', config.get('hidden_size', 768))
            bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
            image_memory = (num_images * patches_per_image * hidden_size * bytes_per_value) / 1e9

        # Calculate audio input memory (for encoder-decoder/audio models)
        audio_memory = 0.0
        if self.model_type in ['encoder-decoder', 'audio-llm']:
            # Get audio config
            if 'audio_config' in config:
                audio_config = config['audio_config']
            else:
                audio_config = config

            num_mel_bins = audio_config.get('num_mel_bins', 128)
            if num_mel_bins > 0:
                # Approximate audio frames from encoder sequence length
                enc_seq_len = encoder_seq_length or audio_config.get('max_source_positions', 1500)
                audio_frames = enc_seq_len * 2  # Approximate: ~2 frames per position
                audio_memory = self.calculate_audio_input_memory(
                    num_mel_bins, audio_frames, batch_size, precision
                )
                # Add conv layer activations
                audio_memory += self.calculate_audio_conv_memory(
                    config, audio_frames, batch_size, precision
                )

        # Extra work buffers (important for TTS, diffusion, etc.)
        extra_work_bytes = config.get('extra_work_bytes', 0)
        
        # Default work buffer for TTS models
        if self.model_type == 'text-to-speech' and 'extra_work_bytes' not in config:
            extra_work_bytes = 2 * 1024**3  # 2 GiB default for latents/overlap-add
        
        # For diffusion models, add latent buffer estimate if not specified
        if self.model_type == 'diffusion' and 'extra_work_bytes' not in config:
            # More generous estimate for SDXL (includes attention maps, etc.)
            latent_size = image_resolution // 8
            latent_channels = config.get('in_channels', 4)
            # Base latent buffer + attention maps + intermediate features
            latent_elements = batch_size * latent_size * latent_size * latent_channels * 50  # Increased multiplier
            extra_work_bytes = latent_elements * self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # Convert all to bytes
        weight_bytes = weight_memory * 1e9
        kv_bytes = kv_cache * 1e9
        activation_bytes = activations * 1e9
        state_bytes = state_memory * 1e9
        lora_bytes = lora_memory * 1e9
        image_bytes = image_memory * 1e9
        audio_bytes = audio_memory * 1e9

        # Apply framework overhead to runtime components (not weights)
        runtime_bytes = (kv_bytes + activation_bytes + state_bytes + image_bytes + audio_bytes)
        runtime_bytes_with_overhead = runtime_bytes * framework_overhead + extra_work_bytes
        
        # Add gradients if training
        if include_gradients:
            weight_bytes *= 2  # Gradients are same size as weights
        
        # Create report
        return MemoryReport(
            model_type=self.model_type,
            attention_type=self.attention_type,
            precision=precision,
            parameter_count=param_count,
            weight_memory_bytes=weight_bytes,
            kv_cache_bytes=kv_bytes,
            activation_memory_bytes=activation_bytes,
            state_memory_bytes=state_bytes,
            image_memory_bytes=image_bytes,
            lora_adapter_memory_bytes=lora_bytes,
            extra_work_bytes=extra_work_bytes + (runtime_bytes_with_overhead - runtime_bytes),
        )