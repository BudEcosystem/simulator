"""Core memory calculator for LLM models."""

import math
from typing import Dict, Any, Optional, List

from .types import MemoryReport
from .parameter_counter import UniversalParameterCounter


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
        'int4': 0.5, 'uint4': 0.5
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
        
        # Multimodal models (check first as they often have nested configs)
        if 'vision_config' in config or 'text_config' in config:
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
    
    def calculate_model_weights(self, config: Dict[str, Any], precision: str) -> tuple[int, float]:
        """Calculate model weight memory in GB and return (param_count, memory_gb)."""
        # Get parameter count
        param_count = config.get('num_parameters')
        if not param_count:
            param_count = self.param_counter.count_parameters(config)
        
        # Calculate memory
        bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)
        weight_memory_gb = (param_count * bytes_per_param) / 1e9  # Use decimal GB to match API
        
        return param_count, weight_memory_gb
    
    def calculate_kv_cache(
        self,
        config: Dict[str, Any],
        batch_size: int,
        seq_length: int,
        precision: str
    ) -> float:
        """Calculate KV cache memory in GB based on attention type."""
        # For multimodal models, use text_config
        if self.model_type == 'multimodal' and 'text_config' in config:
            text_config = config['text_config']
        else:
            text_config = config
            
        attention_type = self.detect_attention_type(config)
        
        if not attention_type:
            return 0.0
        
        # Get bytes per element
        bytes_per_element = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # Handle sliding window attention
        if 'sliding_window' in text_config:
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
        image_resolution: int = 1024
    ) -> MemoryReport:
        """
        Calculate total memory requirements for model inference.
        
        Args:
            config: Model configuration dictionary
            batch_size: Batch size for inference
            seq_length: Maximum sequence length (context + generation)
            precision: Model precision (fp32, fp16, bf16, int8, int4)
            tensor_parallel: Tensor parallelism degree
            framework_overhead: Multiplicative overhead for framework/kernel memory (default 1.2)
            include_gradients: Include gradient memory (for training)
            decode_length: Length of tokens to generate (defaults to seq_length)
            num_images: Number of images for multimodal models
            image_resolution: Image resolution for vision models
            
        Returns:
            MemoryReport with detailed breakdown
        """
        # Detect model and attention types
        self.model_type = self.detect_model_type(config)
        self.attention_type = self.detect_attention_type(config)
        
        # Calculate model weights
        param_count, weight_memory = self.calculate_model_weights(config, precision)
        
        # Divide weights by tensor parallelism
        weight_memory = weight_memory / tensor_parallel
        
        # Calculate KV cache
        kv_cache = self.calculate_kv_cache(config, batch_size, seq_length, precision)
        
        # Divide KV cache by tensor parallelism (each device holds a slice)
        kv_cache = kv_cache / tensor_parallel
        
        # Calculate activations
        activations = self.calculate_activation_memory(config, batch_size, seq_length, precision)
        
        # Calculate state memory (for SSM models)
        state_memory = self.calculate_state_memory(config, batch_size, precision)
        
        # Calculate image memory (for multimodal models)
        image_memory = 0.0
        if num_images and num_images > 0:
            # Estimate based on common vision encoder sizes
            patches_per_image = (image_resolution // 16) ** 2  # Assuming 16x16 patches
            hidden_size = config.get('vision_config', {}).get('hidden_size', config.get('hidden_size', 768))
            bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
            image_memory = (num_images * patches_per_image * hidden_size * bytes_per_value) / 1e9
        
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
        image_bytes = image_memory * 1e9
        
        # Apply framework overhead to runtime components (not weights)
        runtime_bytes = (kv_bytes + activation_bytes + state_bytes + image_bytes)
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
            extra_work_bytes=extra_work_bytes + (runtime_bytes_with_overhead - runtime_bytes),
        )