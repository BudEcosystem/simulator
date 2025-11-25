"""Config normalizer for handling diverse model configurations.

Normalizes various model config formats into a standard internal representation,
supporting advanced features like mixed precision quantization and per-layer attention types.
"""

from typing import Dict, Any, List, Optional


class ConfigNormalizer:
    """Normalize various model configs to standard format.
    
    Handles:
    - MoE key variants (num_local_experts, num_experts, n_routed_experts)
    - Intermediate size variants for dense vs MoE layers
    - Quantization config parsing
    - Per-layer attention type metadata
    """
    
    # MoE expert count key variants
    MOE_EXPERT_KEYS = [
        'num_local_experts',
        'num_experts', 
        'n_routed_experts',
        'moe_num_experts',
        'num_experts_per_tok',  # Alternative naming
    ]
    
    # Expert selection key variants
    MOE_TOP_K_KEYS = [
        'expert_top_k',
        'experts_per_token',
        'num_experts_per_tok',
        'top_k_experts',
    ]
    
    # Intermediate size key variants
    INTERMEDIATE_KEYS = {
        'dense': ['intermediate_size', 'ffn_dim', 'd_ff', 'ffn_hidden_size'],
        'moe': [
            'moe_intermediate_size', 
            'expert_intermediate_size',
            'moe_ffn_dim',
            'expert_ffn_dim',
        ]
    }
    
    # Quantization method to bytes per parameter mapping
    QUANT_METHODS = {
        'mxfp4': 0.5,     # Microsoft MX-FP4 format
        'fp4': 0.5,       # 4-bit float
        'nf4': 0.5,       # NormalFloat4 (QLoRA)
        'int4': 0.5,      # 4-bit integer
        'int8': 1.0,      # 8-bit integer
        'fp8': 1.0,       # FP8 E4M3/E5M2
        'fp8_e4m3': 1.0,  # FP8 E4M3
        'fp8_e5m2': 1.0,  # FP8 E5M2
        'awq': 0.5,       # Activation-aware Weight Quantization (4-bit)
        'gptq': 0.5,      # GPTQ (4-bit default)
        'squeezellm': 0.5, # SqueezeLLM
    }
    
    @staticmethod
    def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config to standard internal format.
        
        Creates a copy of the config and adds standardized keys without
        modifying the original config.
        
        Args:
            config: Original model configuration
            
        Returns:
            Normalized config with standardized keys
        """
        # Create a copy to avoid modifying original
        normalized = config.copy()
        
        # 1. Normalize MoE expert count keys
        if 'n_routed_experts' not in normalized:
            for key in ConfigNormalizer.MOE_EXPERT_KEYS:
                if key in config:
                    normalized['n_routed_experts'] = config[key]
                    break
        
        # 2. Normalize MoE top-k keys
        if 'expert_top_k' not in normalized:
            for key in ConfigNormalizer.MOE_TOP_K_KEYS:
                if key in config:
                    normalized['expert_top_k'] = config[key]
                    break
        
        # 3. Normalize intermediate sizes
        # Check if this is an MoE model
        is_moe = normalized.get('n_routed_experts', 1) > 1
        
        if is_moe and 'moe_intermediate_size' not in normalized:
            # Try to find MoE-specific intermediate size
            for key in ConfigNormalizer.INTERMEDIATE_KEYS['moe']:
                if key in config:
                    normalized['moe_intermediate_size'] = config[key]
                    break
            
            # Fallback: use dense intermediate size if no MoE-specific size found
            if 'moe_intermediate_size' not in normalized:
                for key in ConfigNormalizer.INTERMEDIATE_KEYS['dense']:
                    if key in config:
                        normalized['moe_intermediate_size'] = config[key]
                        break
        
        # Ensure dense intermediate_size exists
        if 'intermediate_size' not in normalized:
            for key in ConfigNormalizer.INTERMEDIATE_KEYS['dense']:
                if key in config:
                    normalized['intermediate_size'] = config[key]
                    break
        
        # 4. Parse quantization config if present
        if 'quantization_config' in config:
            normalized['_quantization'] = ConfigNormalizer._parse_quantization(
                config['quantization_config']
            )
        
        # 5. Parse layer types for per-layer handling
        if 'layer_types' in config:
            normalized['_layer_metadata'] = ConfigNormalizer._parse_layer_types(
                config['layer_types'],
                config.get('num_hidden_layers', len(config['layer_types']))
            )
        
        return normalized
    
    @staticmethod
    def _parse_quantization(quant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse quantization config into usable format.
        
        Args:
            quant_config: Quantization configuration dict
            
        Returns:
            Parsed quantization metadata
        """
        quant_method = quant_config.get('quant_method', 'none').lower()
        modules_skip = quant_config.get('modules_to_not_convert', [])
        
        # Get bytes per parameter for this quantization method
        bytes_per_param = ConfigNormalizer.QUANT_METHODS.get(quant_method, 2.0)
        
        # Also check for explicit bits
        if 'bits' in quant_config:
            bits = quant_config['bits']
            bytes_per_param = bits / 8.0
        
        return {
            'method': quant_method,
            'bytes_per_param': bytes_per_param,
            'skip_modules': modules_skip,
            'has_mixed_precision': len(modules_skip) > 0,
            'bits': quant_config.get('bits'),
        }
    
    @staticmethod
    def _parse_layer_types(
        layer_types: List[str], 
        num_layers: int
    ) -> Dict[str, Any]:
        """Parse layer types for per-layer attention handling.
        
        Extracts metadata about which layers use which attention mechanisms,
        supporting mixed sliding/full attention patterns.
        
        Args:
            layer_types: List of layer type strings (e.g., ["sliding_attention", "full_attention", ...])
            num_layers: Total number of layers
            
        Returns:
            Layer metadata dict with attention layer indices and counts
        """
        attention_layers = []
        sliding_layers = []
        full_layers = []
        mamba_layers = []
        
        for idx, layer_type in enumerate(layer_types):
            layer_type_lower = layer_type.lower()
            
            # Check for attention layers
            if 'attention' in layer_type_lower:
                attention_layers.append(idx)
                
                # Distinguish sliding vs full attention
                if 'sliding' in layer_type_lower:
                    sliding_layers.append(idx)
                elif 'full' in layer_type_lower or 'global' in layer_type_lower:
                    full_layers.append(idx)
                else:
                    # Default to full attention if not specified
                    full_layers.append(idx)
            
            # Check for Mamba/SSM layers
            elif 'mamba' in layer_type_lower or 'ssm' in layer_type_lower:
                mamba_layers.append(idx)
        
        return {
            'attention_layers': attention_layers,
            'sliding_attention_layers': sliding_layers,
            'full_attention_layers': full_layers,
            'mamba_layers': mamba_layers,
            'num_attention_layers': len(attention_layers),
            'num_sliding_layers': len(sliding_layers),
            'num_full_layers': len(full_layers),
            'num_mamba_layers': len(mamba_layers),
            'has_mixed_attention': len(sliding_layers) > 0 and len(full_layers) > 0,
            'has_hybrid_architecture': len(mamba_layers) > 0 and len(attention_layers) > 0,
        }
    
    @staticmethod
    def get_effective_precision(
        config: Dict[str, Any],
        module_type: str = 'default'
    ) -> float:
        """Get effective bytes per parameter for a specific module type.
        
        Handles mixed precision quantization where different modules
        may have different precisions.
        
        Args:
            config: Normalized config with _quantization metadata
            module_type: Type of module ('embedding', 'attention', 'ffn', 'router', 'default')
            
        Returns:
            Bytes per parameter for this module type
        """
        if '_quantization' not in config:
            return 2.0  # Default fp16
        
        quant_info = config['_quantization']
        
        if not quant_info.get('has_mixed_precision'):
            # Uniform quantization
            return quant_info.get('bytes_per_param', 2.0)
        
        # Check if this module type is in skip list
        skip_modules = quant_info.get('skip_modules', [])
        
        for pattern in skip_modules:
            pattern_lower = pattern.lower()
            
            if module_type == 'embedding' and 'embed' in pattern_lower:
                return 2.0  # Not quantized
            elif module_type == 'attention' and 'attn' in pattern_lower:
                return 2.0  # Not quantized
            elif module_type == 'router' and 'router' in pattern_lower:
                return 2.0  # Not quantized
            elif module_type == 'lm_head' and ('lm_head' in pattern_lower or 'output' in pattern_lower):
                return 2.0  # Not quantized
        
        # Not in skip list, use quantized precision
        return quant_info.get('bytes_per_param', 2.0)
