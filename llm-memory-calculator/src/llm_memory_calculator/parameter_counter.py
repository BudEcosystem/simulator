"""Universal parameter counter for all model architectures."""

import warnings
from typing import Dict, Any, Union


class UniversalParameterCounter:
    """Universal parameter counter for all model architectures."""
    
    def __init__(self):
        """Initialize the parameter counter."""
        # Model patterns that always tie embeddings regardless of config
        self.always_tied_models = ['llama', 'opt', 'gptj', 'gpt_neox', 'falcon', 'mpt']
        self.never_tied_models = ['t5', 'bart']  # Encoder-decoders typically don't
        
        # GLU variant patterns - includes 'silu' for modern models
        self.glu_variants = ['swiglu', 'geglu', 'reglu', 'glu', 'swish', 'silu', 'silu_and_mul']
        
        # Keys that indicate shared down projection in MoE models
        self.shared_down_keys = {
            'shared_expert_down_proj', 'ffn_shared_down',
            'experts_share_output', 'share_expert_down_proj'
        }
    
    def count_parameters(self, config: Union[Dict[str, Any], Any], respect_weight_tying: bool = True) -> int:
        """
        Main entry point to count parameters from a config.

        Args:
            config: Model configuration (dict or HuggingFace config object)
            respect_weight_tying: Whether to respect tie_word_embeddings config (default True).
                                 Set to False for accurate memory estimation.

        Returns:
            Total parameter count
        """
        # Convert to dict if needed
        if hasattr(config, '__dict__'):
            config = vars(config)
        elif not isinstance(config, dict):
            raise ValueError("Config must be a dict or have __dict__ attribute")
        
        # Detect and calculate based on architecture
        arch_type = self._detect_architecture_type(config)
        
        try:
            if arch_type == "multimodal":
                return self._calculate_multimodal_params(config, respect_weight_tying)
            elif arch_type == "mamba":
                return self._calculate_mamba_params(config)
            elif arch_type == "hybrid":
                return self._calculate_hybrid_params(config, respect_weight_tying)
            elif arch_type == "diffusion":
                return self._calculate_diffusion_params(config)
            else:
                # Standard transformer with potential modifications
                return self._calculate_transformer_params(config, respect_weight_tying)
        except Exception as e:
            warnings.warn(f"Error calculating params for {arch_type}: {e}. Using fallback.")
            return self._fallback_calculation(config)
    
    def _detect_architecture_type(self, config: Dict[str, Any]) -> str:
        """Detect the primary architecture type from config."""
        # Check for multimodal first (they have nested configs)
        if 'vision_config' in config and 'text_config' in config:
            return "multimodal"
            
        # Check model_type first
        model_type = config.get('model_type', '').lower()
        
        # Direct mappings
        if model_type in ['mamba', 's4', 'ssm']:
            return "mamba"
        elif model_type in ['jamba', 'hybrid']:
            return "hybrid"
        elif model_type in ['unet', 'diffusion', 'stable-diffusion']:
            return "diffusion"
        
        # Detect from config structure
        # SSM/Mamba indicators
        if any(key in config for key in ['state_size', 'd_state', 'dt_rank', 'conv_kernel']):
            if 'num_attention_heads' in config:  # Hybrid
                return "hybrid"
            return "mamba"
        
        # Diffusion indicators
        if any(key in config for key in ['block_out_channels', 'down_block_types', 'up_block_types']):
            return "diffusion"
        
        # Default to transformer
        return "transformer"
    
    def _get_actual_tie_embeddings(self, config: Dict[str, Any]) -> bool:
        """Determine if embeddings are actually tied, not just what the flag says."""
        model_type = config.get('model_type', '').lower()
        architectures = config.get('architectures', [])
        
        # First, check explicit tie_word_embeddings flag
        tie_flag = config.get('tie_word_embeddings', True)
        
        # Models that NEVER tie embeddings
        if model_type in self.never_tied_models:
            return False
            
        # Special cases where config overrides defaults
        # Llama 3+ models explicitly set tie_word_embeddings
        if model_type == 'llama' and 'tie_word_embeddings' in config:
            return config['tie_word_embeddings']
            
        # For models like DeepSeek V3 that explicitly set tie_word_embeddings=False
        if model_type in ['deepseek_v3', 'deepseek'] and not tie_flag:
            return False
            
        # Models that ALWAYS tie embeddings (if not explicitly set)
        if model_type in self.always_tied_models and 'tie_word_embeddings' not in config:
            return True
            
        # Check architecture strings
        for arch in architectures:
            arch_lower = arch.lower()
            if any(model in arch_lower for model in self.never_tied_models):
                return False
            if any(model in arch_lower for model in self.always_tied_models) and 'tie_word_embeddings' not in config:
                return True
        
        # Default to the config flag
        return tie_flag
    
    def _is_down_shared(self, config: Dict[str, Any]) -> bool:
        """Determine if experts share the down projection matrix."""
        # Check explicit keys first
        if any(config.get(k) for k in self.shared_down_keys):
            return True
        
        # Heuristic fallback
        n_routed_experts = config.get('n_routed_experts', 1)
        if n_routed_experts > 32 and config.get('moe_layer_freq', 1) == 1:
            moe_intermediate_size = config.get('moe_intermediate_size', 0)
            hidden_size = config.get('hidden_size', 1)
            ratio = moe_intermediate_size / hidden_size
            # Small experts often share down projection
            return ratio < 0.6
        return False
    
    def _aux_seq_head_params(self, config: Dict[str, Any], hidden_size: int) -> int:
        """Calculate auxiliary sequence head parameters."""
        total_aux = 0
        
        # Check various keys that indicate auxiliary heads
        has_aux = (config.get('seq_aux') or 
                  config.get('use_aux_head') or 
                  config.get('aux_loss_alpha', 0) > 0)
        
        if has_aux:
            vocab_size = config.get('vocab_size', 1)
            total_aux += hidden_size * vocab_size
            
        # Router auxiliary loss head for MoE models
        if config.get('router_aux_loss_coef', 0) > 0 and config.get('n_routed_experts', 0) > 1:
            # Router aux loss typically uses a projection from hidden_size to vocab_size
            # This is used to predict the next token from router hidden states
            vocab_size = config.get('vocab_size', 1)
            total_aux += hidden_size * vocab_size
            
        return total_aux
    
    def _calculate_transformer_params(self, config: Dict[str, Any], respect_weight_tying: bool = True) -> int:
        """Calculate parameters for transformer models."""
        # Basic dimensions
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        vocab_size = config.get('vocab_size', config.get('n_vocab', 50257))

        # Determine if embeddings are tied
        if respect_weight_tying:
            tie_embeddings = self._get_actual_tie_embeddings(config)
        else:
            # For accurate memory estimation, always count both embeddings separately
            tie_embeddings = False
        
        # Embeddings
        embeddings = vocab_size * hidden_size
        if not tie_embeddings:
            embeddings *= 2  # Input and output embeddings
        
        # Position embeddings (if not using RoPE/ALiBi)
        max_position_embeddings = config.get('max_position_embeddings', 0)
        use_rope = any([
            config.get('rope_scaling') is not None,
            config.get('rotary_pct', 0) > 0,
            config.get('rotary_emb_base', 0) > 0,
            config.get('rope_theta', 0) > 0,  # Llama 3 uses rope_theta
            config.get('position_embedding_type', '').lower() in ['rope', 'rotary']
        ])
        use_alibi = config.get('alibi', False) or config.get('use_alibi', False)
        
        if max_position_embeddings > 0 and not use_rope and not use_alibi:
            embeddings += max_position_embeddings * hidden_size
        
        # Attention parameters
        attention_params = self._calculate_attention_params(config, num_layers, hidden_size)
        
        # FFN parameters
        ffn_params = self._calculate_ffn_params(config, num_layers, hidden_size)
        
        # Layer norms
        norm_params = self._calculate_norm_params(config, num_layers, hidden_size)
        
        # MoE parameters (if applicable)
        moe_params = self._calculate_moe_params(config, num_layers, hidden_size)
        
        # Auxiliary sequence head (if applicable)
        aux_params = self._aux_seq_head_params(config, hidden_size)
        
        # Total
        return embeddings + attention_params + ffn_params + norm_params + moe_params + aux_params
    
    def _calculate_attention_params(self, config: Dict[str, Any], num_layers: int, hidden_size: int) -> int:
        """Calculate attention-related parameters."""
        # Check for MLA (Multi-Latent Attention) first
        if any(key in config for key in ['q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim']):
            # DeepSeek V2/V3 style MLA
            q_lora_rank = config.get('q_lora_rank', 0)
            kv_lora_rank = config.get('kv_lora_rank', 0)
            qk_rope_head_dim = config.get('qk_rope_head_dim', 0)
            qk_nope_head_dim = config.get('qk_nope_head_dim', 0)
            v_head_dim = config.get('v_head_dim', 128)
            num_attention_heads = config.get('num_attention_heads', 1)
            
            # Q projection: hidden -> q_lora + rope + nope
            q_proj = hidden_size * (q_lora_rank + qk_rope_head_dim + qk_nope_head_dim)
            
            # KV projections: hidden -> kv_lora_rank
            kv_proj = 2 * hidden_size * kv_lora_rank
            
            # Output projection
            o_proj = num_attention_heads * v_head_dim * hidden_size
            
            return num_layers * (q_proj + kv_proj + o_proj)
        
        # Standard attention (MHA/MQA/GQA)
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        num_key_value_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
        head_dim = config.get('head_dim', hidden_size // num_attention_heads)
        
        # Q, K, V, O projections
        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim
        
        # Some models have different projection sizes
        q_proj_size = config.get('q_proj_size', q_size)
        kv_proj_size = config.get('kv_proj_size', kv_size)
        
        return num_layers * (
            hidden_size * q_proj_size +  # Q projection
            hidden_size * kv_proj_size * 2 +  # K, V projections
            q_proj_size * hidden_size  # O projection
        )
    
    def _calculate_ffn_params(self, config: Dict[str, Any], num_layers: int, hidden_size: int) -> int:
        """Calculate FFN parameters."""
        # Check if using MoE layers
        n_routed_experts = config.get('n_routed_experts', config.get('num_experts', 1))
        moe_layer_freq = config.get('moe_layer_freq', 1)
        
        if n_routed_experts > 1 and moe_layer_freq > 0:
            # MoE layers are handled separately
            non_moe_layers = num_layers - (num_layers // moe_layer_freq if moe_layer_freq > 0 else 0)
        else:
            non_moe_layers = num_layers
        
        # Intermediate size
        intermediate_size = config.get('intermediate_size', config.get('ffn_dim', hidden_size * 4))
        
        # Check activation function for GLU variants
        act_fn = config.get('activation_function', config.get('hidden_act', 'gelu')).lower()
        
        if any(variant in act_fn for variant in self.glu_variants):
            # GLU variants use 3 matrices: gate, up, down
            return non_moe_layers * (3 * hidden_size * intermediate_size)
        else:
            # Standard FFN uses 2 matrices: up, down
            return non_moe_layers * (2 * hidden_size * intermediate_size)
    
    def _calculate_moe_params(self, config: Dict[str, Any], num_layers: int, hidden_size: int) -> int:
        """Calculate MoE-specific parameters."""
        n_routed_experts = config.get('n_routed_experts', config.get('num_experts', 1))
        if n_routed_experts <= 1:
            return 0
        
        moe_layer_freq = config.get('moe_layer_freq', 1)
        if moe_layer_freq <= 0:
            return 0
        
        num_moe_layers = num_layers // moe_layer_freq
        moe_intermediate_size = config.get('moe_intermediate_size', hidden_size * 4)
        
        # Router parameters
        router_params = num_moe_layers * hidden_size * n_routed_experts
        
        # Expert FFN parameters
        act_fn = config.get('activation_function', config.get('hidden_act', 'gelu')).lower()
        is_glu = any(variant in act_fn for variant in self.glu_variants)
        
        if is_glu:
            # GLU: gate + up projections (separate)
            expert_ffn = 2 * hidden_size * moe_intermediate_size
        else:
            # Standard: single up projection
            expert_ffn = hidden_size * moe_intermediate_size
        
        # Down projection (might be shared)
        if self._is_down_shared(config):
            # Shared down projection across all experts
            down_proj = hidden_size * moe_intermediate_size
            expert_params = num_moe_layers * (n_routed_experts * expert_ffn + down_proj)
        else:
            # Each expert has its own down projection
            down_proj = hidden_size * moe_intermediate_size
            expert_params = num_moe_layers * n_routed_experts * (expert_ffn + down_proj)
        
        # Shared experts (if any)
        n_shared_experts = config.get('n_shared_experts', 0)
        if n_shared_experts > 0:
            shared_expert_params = num_moe_layers * n_shared_experts * (
                (3 if is_glu else 2) * hidden_size * moe_intermediate_size
            )
            expert_params += shared_expert_params
        
        return router_params + expert_params
    
    def _calculate_norm_params(self, config: Dict[str, Any], num_layers: int, hidden_size: int) -> int:
        """Calculate normalization layer parameters."""
        # Most models have 2 norms per layer (pre-attention, pre-ffn) + final norm
        norm_type = config.get('normalization_type', config.get('norm_type', 'layernorm')).lower()
        
        # Check model type for defaults
        model_type = config.get('model_type', '').lower()
        
        # Llama models use RMSNorm
        if model_type in ['llama', 'mistral', 'mixtral'] or 'rmsnorm' in norm_type or 'rms' in norm_type:
            # RMSNorm only has scale parameters
            return (2 * num_layers + 1) * hidden_size
        else:
            # LayerNorm has scale and bias
            return (2 * num_layers + 1) * hidden_size * 2
    
    def _calculate_mamba_params(self, config: Dict[str, Any]) -> int:
        """Calculate parameters for Mamba/SSM models."""
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        vocab_size = config.get('vocab_size', 50257)
        
        # SSM-specific dimensions
        state_size = config.get('state_size', config.get('d_state', 16))
        expand_factor = config.get('expand_factor', config.get('expand', 2))
        dt_rank = config.get('dt_rank', hidden_size // 16)
        conv_kernel = config.get('conv_kernel', config.get('d_conv', 4))
        
        # Embeddings
        embeddings = vocab_size * hidden_size * 2  # Usually not tied in SSMs
        
        # Per-layer SSM parameters
        inner_size = hidden_size * expand_factor
        
        ssm_params_per_layer = (
            hidden_size * inner_size +  # in_proj
            conv_kernel * inner_size +  # conv1d
            inner_size * state_size +  # x_proj (to dt, B, C)
            inner_size * dt_rank +  # dt_proj
            inner_size * state_size +  # A
            inner_size * state_size +  # B
            state_size * inner_size +  # C
            state_size +  # D
            inner_size * hidden_size  # out_proj
        )
        
        # Norms (usually RMSNorm)
        norm_params = (2 * num_layers + 1) * hidden_size
        
        return embeddings + num_layers * ssm_params_per_layer + norm_params
    
    def _calculate_hybrid_params(self, config: Dict[str, Any], respect_weight_tying: bool = True) -> int:
        """Calculate parameters for hybrid models (e.g., Jamba with both attention and Mamba)."""
        # Get layer configuration
        num_layers = config.get('num_hidden_layers', 12)

        # Some hybrids specify layer types explicitly
        layer_types = config.get('layer_types', [])
        if layer_types:
            attention_layers = sum(1 for t in layer_types if 'attention' in str(t).lower())
            mamba_layers = sum(1 for t in layer_types if 'mamba' in str(t).lower() or 'ssm' in str(t).lower())
        else:
            # Use ratio if specified
            attention_ratio = config.get('attention_ratio', 0.5)
            attention_layers = int(num_layers * attention_ratio)
            mamba_layers = num_layers - attention_layers

        # Calculate transformer params for attention layers
        transformer_config = config.copy()
        transformer_config['num_hidden_layers'] = attention_layers
        attention_params = self._calculate_transformer_params(transformer_config, respect_weight_tying) if attention_layers > 0 else 0
        
        # Calculate Mamba params for SSM layers
        mamba_config = config.copy()
        mamba_config['num_hidden_layers'] = mamba_layers
        # Get Mamba config if nested
        if 'mamba_config' in config:
            mamba_config.update(config['mamba_config'])
        mamba_params = self._calculate_mamba_params(mamba_config) if mamba_layers > 0 else 0
        
        # Avoid double-counting embeddings
        hidden_size = config.get('hidden_size', 768)
        vocab_size = config.get('vocab_size', 50257)
        embedding_params = vocab_size * hidden_size * 2
        
        # Subtract embeddings from both to avoid double counting, then add back once
        if attention_layers > 0:
            attention_params -= embedding_params
        if mamba_layers > 0:
            mamba_params -= embedding_params
        
        return embedding_params + attention_params + mamba_params
    
    def _calculate_multimodal_params(self, config: Dict[str, Any], respect_weight_tying: bool = True) -> int:
        """Calculate parameters for multimodal models (e.g., LLaVA, Llama-4-Scout)."""
        total_params = 0

        # Process text config
        if 'text_config' in config and isinstance(config['text_config'], dict):
            text_config = config['text_config'].copy()

            # Ensure model_type is set for proper calculation
            if 'model_type' not in text_config:
                text_config['model_type'] = text_config.get('model_type', 'decoder-only')

            # Handle MoE parameters from text config
            if 'num_local_experts' in text_config:
                text_config['n_routed_experts'] = text_config.get('num_local_experts')
            if 'num_experts' in text_config:
                text_config['n_routed_experts'] = text_config.get('num_experts')

            # For Llama-4 style configs:
            # - intermediate_size is the MoE expert size
            # - intermediate_size_mlp is the dense FFN size (if any)
            # - interleave_moe_layer_step determines MoE frequency
            if 'intermediate_size' in text_config and 'intermediate_size_mlp' in text_config:
                # Keep moe_intermediate_size as the smaller value
                text_config['moe_intermediate_size'] = text_config['intermediate_size']
                # Set intermediate_size to the larger value for dense layers
                text_config['intermediate_size'] = text_config['intermediate_size_mlp']

            # Check MoE frequency
            interleave_step = text_config.get('interleave_moe_layer_step', 1)
            if interleave_step == 1:
                # All layers are MoE, no dense FFN
                text_config['moe_layer_freq'] = 1
                text_config['first_k_dense_replace'] = 0
            else:
                # Some layers are dense
                text_config['moe_layer_freq'] = interleave_step

            # Calculate text model parameters
            text_params = self._calculate_transformer_params(text_config, respect_weight_tying)
            total_params += text_params
        
        # Process vision config
        if 'vision_config' in config and isinstance(config['vision_config'], dict):
            vision_config = config['vision_config']
            
            # Vision transformer parameters
            hidden_size = vision_config.get('hidden_size', 768)
            num_layers = vision_config.get('num_hidden_layers', 12)
            image_size = vision_config.get('image_size', 224)
            patch_size = vision_config.get('patch_size', 16)
            num_patches = (image_size // patch_size) ** 2
            
            # Patch embedding
            channels = vision_config.get('num_channels', 3)
            vision_params = channels * patch_size * patch_size * hidden_size
            
            # Position embeddings
            vision_params += (num_patches + 1) * hidden_size  # +1 for CLS token
            
            # Vision transformer layers
            for _ in range(num_layers):
                # Attention
                vision_params += 4 * hidden_size * hidden_size  # Q, K, V, O
                # FFN
                intermediate_size = vision_config.get('intermediate_size', hidden_size * 4)
                vision_params += 2 * hidden_size * intermediate_size
                # Layer norms
                vision_params += 2 * hidden_size
            
            # Final norm
            vision_params += hidden_size
            
            # Projector (if specified)
            if 'projector_input_dim' in vision_config and 'projector_output_dim' in vision_config:
                vision_params += vision_config['projector_input_dim'] * vision_config['projector_output_dim']
            
            total_params += vision_params
        
        # Cross-modal projector (if not already counted)
        if 'multi_modal_projector_dim' in config:
            proj_in = config.get('vision_hidden_size', 768)
            proj_out = config['multi_modal_projector_dim']
            total_params += proj_in * proj_out
        
        return total_params
    
    def _calculate_diffusion_params(self, config: Dict[str, Any]) -> int:
        """Calculate parameters for diffusion models."""
        # This is a simplified estimation for UNet-based diffusion models
        # Real architectures can be much more complex
        
        in_channels = config.get('in_channels', 4)
        out_channels = config.get('out_channels', 4)
        
        # Block configurations
        block_out_channels = config.get('block_out_channels', [320, 640, 1280, 1280])
        layers_per_block = config.get('layers_per_block', 2)
        
        params = 0
        
        # Conv in/out
        if block_out_channels:
            params += in_channels * block_out_channels[0] * 3 * 3  # Conv in
            params += block_out_channels[-1] * out_channels * 3 * 3  # Conv out
        
        # Down/Up blocks
        for i, channels in enumerate(block_out_channels):
            # Each block typically has multiple ResNet layers
            for _ in range(layers_per_block):
                # Simplified ResNet block calculation
                params += channels * channels * 3 * 3 * 2  # Two conv layers
                params += channels * 2  # Norms
            
            # Attention blocks (usually in lower resolution layers)
            if i >= len(block_out_channels) // 2:
                head_dim = 64
                num_heads = channels // head_dim
                params += channels * channels * 4  # Rough attention param estimate
        
        # Middle block
        if block_out_channels:
            mid_channels = block_out_channels[-1]
            params += mid_channels * mid_channels * 3 * 3 * 4  # Multiple convs
            params += mid_channels * mid_channels * 4  # Attention
        
        # Time/Class embeddings
        time_embed_dim = config.get('time_embed_dim', 1280)
        params += 1280 * time_embed_dim * 4  # MLP layers
        
        return params
    
    def _fallback_calculation(self, config: Dict[str, Any]) -> int:
        """Fallback calculation when architecture-specific calculation fails."""
        # Very rough estimation based on common patterns
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
        vocab_size = config.get('vocab_size', config.get('n_vocab', 50257))
        
        # Rough transformer estimate
        embeddings = vocab_size * hidden_size
        per_layer = hidden_size * hidden_size * 12  # Very rough
        
        return embeddings + num_layers * per_layer