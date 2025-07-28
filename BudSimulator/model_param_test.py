"""
UniversalParameterCounter - A comprehensive parameter counter for all model architectures.

FIXED: Now correctly handles 'silu' as a gated activation (SwiGLU variant with 3 matrices).

Supports:
- Standard Transformers (encoder, decoder, encoder-decoder)
- State Space Models (Mamba, S4)
- Hybrid architectures (Jamba)
- Diffusion models (U-Net)
- Vision models (ViT)
- Multimodal models (CLIP)
- Sparse attention (BigBird, Longformer)
- Parameter-efficient methods (LoRA, Adapters, Prefix Tuning)
- MoE models with proper layer frequency
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import warnings


class UniversalParameterCounter:
    """Universal parameter counter for all model architectures."""
    
    def __init__(self):
        """Initialize the parameter counter."""
        # Model patterns that always tie embeddings regardless of config
        self.always_tied_models = ['llama', 'opt', 'gptj', 'gpt_neox', 'falcon', 'mpt']
        self.never_tied_models = ['t5', 'bart']  # Encoder-decoders typically don't
        
        # GLU variant patterns - FIXED: Added 'silu' 
        # Note: Many modern models (DeepSeek, Llama3, Qwen) use "silu" to indicate
        # the SwiGLU variant which requires 3 matrices (gate, up, down)
        self.glu_variants = ['swiglu', 'geglu', 'reglu', 'glu', 'swish', 'silu', 'silu_and_mul']
        
        # Keys that indicate shared down projection in MoE models
        self.shared_down_keys = {
            'shared_expert_down_proj', 'ffn_shared_down',
            'experts_share_output', 'share_expert_down_proj'
        }
    
    def count_parameters(self, config: Union[Dict[str, Any], Any]) -> int:
        """
        Main entry point to count parameters from a config.
        
        Args:
            config: Model configuration (dict or HuggingFace config object)
            
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
            if arch_type == "mamba":
                return self._calculate_mamba_params(config)
            elif arch_type == "hybrid":
                return self._calculate_hybrid_params(config)
            elif arch_type == "diffusion":
                return self._calculate_diffusion_params(config)
            else:
                # Standard transformer with potential modifications
                return self._calculate_transformer_params(config)
        except Exception as e:
            warnings.warn(f"Error calculating params for {arch_type}: {e}. Using fallback.")
            return self._fallback_calculation(config)
    
    def _detect_architecture_type(self, config: Dict[str, Any]) -> str:
        """Detect the primary architecture type from config."""
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
        
        # Models that ALWAYS tie embeddings regardless of flag
        if model_type in self.always_tied_models:
            return True
            
        # Models that NEVER tie embeddings
        if model_type in self.never_tied_models:
            return False
            
        # For models like DeepSeek V3 that explicitly set tie_word_embeddings=False
        if model_type in ['deepseek_v3', 'deepseek'] and not tie_flag:
            return False
            
        # Check architecture strings
        for arch in architectures:
            arch_lower = arch.lower()
            if any(model in arch_lower for model in self.always_tied_models):
                return True
            if any(model in arch_lower for model in self.never_tied_models):
                return False
        
        # Default to the config flag
        return tie_flag
    
    def _is_down_shared(self, config: Dict[str, Any]) -> bool:
        """Determine if experts share the down projection matrix."""
        # Check explicit keys first
        if any(config.get(k) for k in self.shared_down_keys):
            return True
        
        # Heuristic fallback - still model-agnostic
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
        # Check various keys that indicate auxiliary heads
        has_aux = (config.get('seq_aux') or 
                  config.get('use_aux_head') or 
                  config.get('aux_loss_alpha', 0) > 0)
        
        if has_aux:
            # The aux head is a single component, not per-layer.
            # The original implementation incorrectly multiplied by num_hidden_layers.
            # A typical aux head might project hidden_size -> 1 or a few labels.
            # A simple hidden_size -> hidden_size projection is a safe estimate.
            return hidden_size * hidden_size + hidden_size  # Projection + bias
        return 0
    
    def _calculate_transformer_params(self, config: Dict[str, Any]) -> int:
        """Calculate transformer model parameters with all modern variants."""
        # Extract dimensions with fallbacks
        vocab_size = config.get('vocab_size', config.get('n_vocab', 32000))
        hidden_size = config.get('hidden_size', config.get('d_model', config.get('n_embd', 768)))
        num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 12)))
        intermediate_size = config.get('intermediate_size', config.get('ffn_dim', config.get('d_ff', hidden_size * 4)))
        
        # Attention configuration
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        num_key_value_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
        
        total_params = 0
        
        # ============= EMBEDDINGS =============
        embedding_params = self._calculate_embedding_params(config, vocab_size, hidden_size)
        total_params += embedding_params
        
        # ============= ATTENTION =============
        attention_params = self._calculate_attention_params(config, hidden_size, num_attention_heads, num_key_value_heads)
        
        # ============= FFN =============
        ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
        
        # ============= NORMALIZATION =============
        ln_params_per_layer = self._calculate_norm_params(config, hidden_size, num_norms=2)
        
        # ============= LAYER ASSEMBLY =============
        total_layer_params = 0
        
        # Check if this is an MoE model
        n_routed_experts = config.get('n_routed_experts', config.get('num_experts', config.get('num_local_experts', 1)))
        is_moe = n_routed_experts > 1
        
        if is_moe:
            # MoE model - calculate layer-specific parameters
            moe_components = self._calculate_moe_layer_params(
                config, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size
            )
            
            # Get layer configuration
            num_layers = config.get('num_hidden_layers', 12)
            first_k_dense_replace = config.get('first_k_dense_replace', 0)
            moe_layer_freq = config.get('moe_layer_freq', 1)
            
            # Calculate total parameters for all layers
            for layer_idx in range(num_layers):
                # All layers have attention and normalization
                layer_params = attention_params + ln_params_per_layer
                
                # Determine if this is a dense or MoE layer
                if layer_idx < first_k_dense_replace:
                    # Dense layer
                    if 'dense_ffn' in moe_components:
                        layer_params += moe_components['dense_ffn'] / first_k_dense_replace
                    else:
                        # Use standard FFN
                        layer_params += self._calculate_ffn_params(config, hidden_size, intermediate_size)
                else:
                    # Check if this should be an MoE layer based on frequency
                    if moe_layer_freq == 1 or (layer_idx - first_k_dense_replace) % moe_layer_freq == 0:
                        # MoE layer
                        if 'routed_experts' in moe_components:
                            layer_params += moe_components['routed_experts']
                        if 'shared_experts' in moe_components:
                            layer_params += moe_components['shared_experts']
                        if 'router' in moe_components:
                            layer_params += moe_components['router']
                    else:
                        # Dense layer (in sparse models)
                        layer_params += self._calculate_ffn_params(config, hidden_size, intermediate_size)
                
                total_layer_params += layer_params
        else:
            # Standard model - all layers identical
            ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
            layer_params = attention_params + ffn_params + ln_params_per_layer
            total_layer_params = num_layers * layer_params

        total_params += total_layer_params
        
        # Final layer norm
        total_params += self._calculate_norm_params(config, hidden_size, num_norms=1)
        
        # ============= ENCODER-DECODER =============
        if config.get('is_encoder_decoder', False):
            # Recalculate for encoder-decoder architecture
            return self._calculate_encoder_decoder_params(config)
        
        # ============= TASK HEADS =============
        total_params += self._calculate_task_heads(config, hidden_size)
        
        # ============= SPARSE ATTENTION =============
        total_params += self._calculate_sparse_attention_params(config, hidden_size, num_layers)
        
        # ============= ADAPTERS =============
        total_params += self._calculate_adapter_params(config, hidden_size, num_layers)
        
        # ============= AUXILIARY SEQUENCE HEADS =============
        total_params += self._aux_seq_head_params(config, hidden_size)
        
        # ============= MULTI-TOKEN PREDICTION (MTP) =============
        num_mtp_layers = config.get('num_nextn_predict_layers', 0)
        if num_mtp_layers > 0:
            # Each MTP layer has:
            # - Projection matrix: 2 * hidden_size -> hidden_size
            # - Transformer block (similar to a standard layer)
            # Note: Embedding and output head are shared with main model
            for _ in range(num_mtp_layers):
                # Projection matrix
                total_params += 2 * hidden_size * hidden_size
                
                # Transformer block (attention + FFN + norms)
                # Use standard layer params (not MoE) for MTP
                mtp_attention_params = self._calculate_attention_params(
                    config, hidden_size, num_attention_heads, num_key_value_heads
                )
                mtp_ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
                mtp_norm_params = self._calculate_norm_params(config, hidden_size, num_norms=2)
                
                total_params += mtp_attention_params + mtp_ffn_params + mtp_norm_params
        
        return int(total_params)
    
    def _calculate_embedding_params(self, config: Dict[str, Any], vocab_size: int, hidden_size: int) -> int:
        """Calculate embedding parameters."""
        # Token embeddings
        embedding_params = vocab_size * hidden_size
        
        # Check if embeddings are actually tied
        if not self._get_actual_tie_embeddings(config):
            embedding_params *= 2
        
        # Position embeddings (only for models that don't use RoPE)
        # Check for RoPE indicators
        has_rope = (
            config.get('rope_theta') is not None or
            config.get('rope_scaling') is not None or
            config.get('rotary_emb_base') is not None or
            config.get('rotary_pct') is not None or
            config.get('rotary_dim') is not None
        )
        
        if not has_rope:
            position_embedding_type = config.get('position_embedding_type', 'absolute').lower()
            if position_embedding_type in ['absolute', 'learned']:
                max_position_embeddings = config.get('max_position_embeddings', config.get('n_positions', 512))
                embedding_params += max_position_embeddings * hidden_size
            elif position_embedding_type == 'relative_key_query':
                num_buckets = config.get('relative_attention_num_buckets', 32)
                num_heads = config.get('num_attention_heads', 12)
                embedding_params += num_buckets * num_heads
        
        # Token type embeddings
        if config.get('type_vocab_size', 0) > 0:
            embedding_params += config['type_vocab_size'] * hidden_size
        
        return embedding_params
    
    def _calculate_attention_params(self, config: Dict[str, Any], hidden_size: int, 
                                   num_attention_heads: int, num_key_value_heads: int) -> int:
        """Calculate attention parameters with MLA support."""
        
        # Check if using MLA (Multi-head Latent Attention)
        if 'q_lora_rank' in config and 'kv_lora_rank' in config and 'qk_rope_head_dim' in config:
            # MLA architecture (DeepSeek V3 style)
            q_lora_rank = config['q_lora_rank']
            kv_lora_rank = config['kv_lora_rank']
            
            # Get head dimensions for MLA
            qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
            qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
            v_head_dim = config.get('v_head_dim', 128)
            
            # Q projection: hidden -> q_lora_rank -> heads * (rope + nope)
            q_params = (
                hidden_size * q_lora_rank +  # Down projection
                q_lora_rank * num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)  # Up projection
            )
            
            # KV projection in DeepSeek V3 MLA:
            # - KV jointly compressed: hidden -> kv_lora_rank
            # - K (non-RoPE part): kv_lora_rank -> heads * qk_nope_head_dim
            # - V: kv_lora_rank -> heads * v_head_dim  
            # - K (RoPE part): hidden -> heads * qk_rope_head_dim (separate projection)
            kv_params = (
                hidden_size * kv_lora_rank +  # Down projection (shared for K and V)
                kv_lora_rank * num_attention_heads * qk_nope_head_dim +  # Up projection for K (non-RoPE)
                kv_lora_rank * num_attention_heads * v_head_dim +  # Up projection for V
                hidden_size * num_attention_heads * qk_rope_head_dim  # Separate K RoPE projection
            )
            
            # O projection: v_head_dim * heads -> hidden (not full hidden_size)
            o_params = num_attention_heads * v_head_dim * hidden_size
            
            attention_params = q_params + kv_params + o_params
            
            # MLA typically doesn't use biases
            if config.get('use_bias', config.get('attention_bias', False)):
                attention_params += hidden_size  # Only output bias if any
                
            return attention_params
            
        else:
            # Standard attention or GQA/MQA (original implementation)
            head_dim = config.get('head_dim', hidden_size // num_attention_heads)
            
            # Handle GQA/MQA
            if 'head_dim' in config:
                q_dim = head_dim * num_attention_heads
                kv_dim = head_dim * num_key_value_heads
            else:
                q_dim = hidden_size
                kv_dim = (hidden_size // num_attention_heads) * num_key_value_heads
            
            # Check for LoRA (adds to, not replaces)
            has_lora = 'q_lora_rank' in config or 'lora_alpha' in config
            
            # Base attention parameters
            attention_params = (
                hidden_size * q_dim +      # Q
                hidden_size * kv_dim +     # K
                hidden_size * kv_dim +     # V
                q_dim * hidden_size        # O
            )
            
            # Add LoRA if present (but not MLA)
            if has_lora and config.get('q_lora_rank') and 'qk_rope_head_dim' not in config:
                q_lora_rank = config.get('q_lora_rank', 1536)
                kv_lora_rank = config.get('kv_lora_rank', 512)
                attention_params += (
                    hidden_size * q_lora_rank + q_lora_rank * hidden_size +
                    hidden_size * kv_lora_rank + kv_lora_rank * kv_dim
                )
            
            # Biases
            if config.get('use_bias', config.get('bias', False)):
                attention_params += q_dim + kv_dim + kv_dim + hidden_size
            
            return attention_params
    
    def _calculate_ffn_params(self, config: Dict[str, Any], hidden_size: int, intermediate_size: int) -> int:
        """Calculate FFN parameters."""
        activation = str(config.get('hidden_act', config.get('activation_function', 'gelu'))).lower()
        
        # Check if gated - FIXED: Now correctly identifies 'silu' as gated
        # Note: 'silu' in modern models (DeepSeek, Llama3, Qwen) means SwiGLU variant
        is_gated = any(pattern in activation for pattern in self.glu_variants)
        
        if is_gated:
            # Gated: 3 matrices (gate, up, down) - This is the SwiGLU/SiLU-and-Multiply pattern
            ffn_params = hidden_size * intermediate_size * 2 + intermediate_size * hidden_size
        else:
            # Standard: 2 matrices (up, down)
            ffn_params = hidden_size * intermediate_size + intermediate_size * hidden_size
        
        # Biases (rare in modern models)
        if config.get('use_bias', config.get('mlp_bias', False)):
            if is_gated:
                ffn_params += 2 * intermediate_size + hidden_size  # gate+up bias, down bias
            else:
                ffn_params += intermediate_size + hidden_size  # up bias, down bias
        
        return ffn_params
    
    def _calculate_norm_params(self, config: Dict[str, Any], hidden_size: int, num_norms: int) -> int:
        """Calculate normalization parameters."""
        norm_type = config.get('norm_type', config.get('normalization_type', 'layernorm')).lower()
        
        # RMSNorm (most modern LLMs)
        if 'rmsnorm' in norm_type or 'rms_norm' in norm_type or config.get('rms_norm_eps') is not None:
            # RMSNorm only has scale parameters, no bias
            return num_norms * hidden_size
        elif 'scalenorm' in norm_type:
            # ScaleNorm has a single scalar per norm
            return num_norms
        else:
            # LayerNorm has scale + optional bias
            has_bias = config.get('layer_norm_bias', True)
            return num_norms * hidden_size * (2 if has_bias else 1)
    
    def _calculate_moe_layer_params(self, config: Dict[str, Any], hidden_size: int, 
                                    num_attention_heads: int, num_key_value_heads: int,
                                    intermediate_size: int) -> Dict[str, int]:
        """Calculate MoE layer parameters with correct expert sizing."""
        
        params = {}
        
        # Get MoE configuration
        n_routed_experts = config.get('n_routed_experts', config.get('num_experts', 0))
        n_shared_experts = config.get('n_shared_experts', 0)
        num_experts_per_tok = config.get('num_experts_per_tok', config.get('top_k', 2))
        
        # CRITICAL FIX: moe_intermediate_size is PER EXPERT, not total
        moe_intermediate_size = config.get('moe_intermediate_size', intermediate_size)
        
        # Router parameters (only for routed experts)
        if n_routed_experts > 0:
            # Standard router: hidden -> num_experts scores
            router_params = hidden_size * n_routed_experts
            params['router'] = router_params
        
        # Routed expert FFN parameters
        if n_routed_experts > 0:
            # Each expert has its own FFN with moe_intermediate_size
            expert_ffn_params = self._calculate_ffn_params(config, hidden_size, moe_intermediate_size)
            # Total for all routed experts
            params['routed_experts'] = expert_ffn_params * n_routed_experts
        
        # Shared expert FFN parameters
        if n_shared_experts > 0:
            # Shared experts in DeepSeek use full intermediate_size, not moe_intermediate_size
            # This is a key difference from routed experts
            shared_intermediate = config.get('shared_expert_intermediate_size', intermediate_size)
            shared_expert_params = self._calculate_ffn_params(config, hidden_size, shared_intermediate)
            params['shared_experts'] = shared_expert_params * n_shared_experts
        
        # Dense FFN (if any - only in hybrid architectures)
        first_k_dense_replace = config.get('first_k_dense_replace', 0)
        if first_k_dense_replace > 0:
            # These layers use full intermediate_size, not moe_intermediate_size
            dense_ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
            params['dense_ffn'] = dense_ffn_params * first_k_dense_replace
        
        return params
    
    def _calculate_encoder_decoder_params(self, config: Dict[str, Any]) -> int:
        """Calculate encoder-decoder model parameters."""
        hidden_size = config.get('hidden_size', 768)
        vocab_size = config.get('vocab_size', 32000)
        
        encoder_layers = config.get('encoder_layers', config.get('num_encoder_layers', 12))
        decoder_layers = config.get('decoder_layers', config.get('num_decoder_layers', 12))
        
        # Embeddings
        embedding_params = self._calculate_embedding_params(config, vocab_size, hidden_size)
        
        # Attention params
        num_heads = config.get('num_attention_heads', 12)
        num_kv_heads = config.get('num_key_value_heads', num_heads)
        attention_params = self._calculate_attention_params(config, hidden_size, num_heads, num_kv_heads)
        
        # FFN params
        intermediate_size = config.get('intermediate_size', hidden_size * 4)
        ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
        
        # Norm params
        ln_params = self._calculate_norm_params(config, hidden_size, 2)
        
        # Encoder
        encoder_params = encoder_layers * (attention_params + ffn_params + ln_params)
        
        # Decoder (includes cross-attention)
        cross_attention_params = attention_params + self._calculate_norm_params(config, hidden_size, 1)
        decoder_params = decoder_layers * (attention_params + cross_attention_params + ffn_params + ln_params)
        
        # Final norms
        final_norms = self._calculate_norm_params(config, hidden_size, 2)
        
        return embedding_params + encoder_params + decoder_params + final_norms
    
    def _calculate_task_heads(self, config: Dict[str, Any], hidden_size: int) -> int:
        """Calculate task-specific head parameters."""
        params = 0
        
        # Classification head
        if config.get('num_labels', 0) > 0 and not config.get('is_decoder', True):
            params += hidden_size * config['num_labels']
            
            # Pooler
            if config.get('use_pooler', True):
                params += hidden_size * hidden_size + hidden_size
        
        # Vision components
        if 'image_size' in config and 'patch_size' in config:
            patch_size = config['patch_size']
            num_channels = config.get('num_channels', 3)
            params += num_channels * patch_size * patch_size * hidden_size + hidden_size
            
            if config.get('use_cls_token', True):
                params += hidden_size
        
        # Speech components
        if 'num_mel_bins' in config:
            num_mel_bins = config['num_mel_bins']
            params += num_mel_bins * hidden_size * 3 + hidden_size * hidden_size * 3
        
        return params
    
    def _calculate_sparse_attention_params(self, config: Dict[str, Any], hidden_size: int, num_layers: int) -> int:
        """Calculate sparse attention parameters."""
        params = 0
        attention_type = config.get('attention_type', 'full').lower()
        
        if attention_type == 'bigbird':
            num_global_tokens = config.get('num_global_tokens', 0)
            if num_global_tokens > 0:
                params += num_global_tokens * hidden_size
                params += num_layers * num_global_tokens * hidden_size * 2
        
        elif attention_type == 'longformer':
            global_attention_layers = config.get('global_attention_layers', [])
            if global_attention_layers:
                params += len(global_attention_layers) * 3 * hidden_size * hidden_size
        
        return params
    
    def _calculate_adapter_params(self, config: Dict[str, Any], hidden_size: int, num_layers: int) -> int:
        """Calculate adapter and parameter-efficient tuning parameters."""
        params = 0
        
        # Bottleneck adapters
        if 'adapter_size' in config or 'adapter_config' in config:
            adapter_size = config.get('adapter_size', config.get('adapter_config', {}).get('hidden_size', 64))
            num_adapters = config.get('num_adapters_per_layer', 2)
            adapter_params = hidden_size * adapter_size + adapter_size * hidden_size + hidden_size
            params += num_layers * num_adapters * adapter_params
        
        # LoRA
        if 'lora_r' in config or 'lora_config' in config:
            lora_r = config.get('lora_r', config.get('lora_config', {}).get('r', 8))
            lora_target_modules = config.get('lora_target_modules', ['q_proj', 'v_proj'])
            
            for module in lora_target_modules:
                if module in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    params += num_layers * 2 * hidden_size * lora_r
                elif module in ['gate_proj', 'up_proj', 'down_proj']:
                    intermediate_size = config.get('intermediate_size', hidden_size * 4)
                    if module == 'down_proj':
                        params += num_layers * (intermediate_size * lora_r + lora_r * hidden_size)
                    else:
                        params += num_layers * (hidden_size * lora_r + lora_r * intermediate_size)
        
        # Prefix tuning
        if 'prefix_length' in config:
            prefix_length = config['prefix_length']
            num_heads = config.get('num_attention_heads', 12)
            params += prefix_length * num_layers * num_heads * hidden_size * 2
        
        return params
    
    def _calculate_mamba_params(self, config: Dict[str, Any]) -> int:
        """Calculate Mamba/SSM model parameters."""
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_layers = config.get('num_hidden_layers', config.get('n_layer', 24))
        vocab_size = config.get('vocab_size', 50280)
        
        # State space dimensions
        state_size = config.get('state_size', config.get('d_state', 16))
        expand_factor = config.get('expand', config.get('expand_factor', 2))
        dt_rank = config.get('dt_rank', 'auto')
        conv_kernel = config.get('conv_kernel', config.get('d_conv', 4))
        
        # Compute inner dimension
        d_inner = int(expand_factor * hidden_size)
        
        # Auto dt_rank
        if dt_rank == 'auto':
            dt_rank = max(1, d_inner // 16)
        
        # Embeddings
        embedding_params = self._calculate_embedding_params(config, vocab_size, hidden_size)
        
        # Per-layer Mamba parameters
        mamba_layer_params = (
            hidden_size * d_inner * 2 +  # Input projection (gated)
            d_inner * conv_kernel +      # Convolution
            dt_rank * d_inner +          # Δ projection
            d_inner +                    # Δ bias
            d_inner * state_size +       # A matrix
            dt_rank * state_size +       # B projection
            dt_rank * state_size +       # C projection
            d_inner +                    # D parameter
            d_inner * hidden_size +      # Output projection
            hidden_size                  # Norm
        )
        
        # Total
        total_params = embedding_params + num_layers * mamba_layer_params + hidden_size
        
        return int(total_params)
    
    def _calculate_hybrid_params(self, config: Dict[str, Any]) -> int:
        """Calculate hybrid architecture parameters (e.g., Jamba)."""
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_hidden_layers', 32)
        vocab_size = config.get('vocab_size', 50280)
        
        # Layer configuration
        layer_types = config.get('layer_types', [])
        if not layer_types:
            mamba_ratio = config.get('mamba_ratio', 0.5)
            num_mamba_layers = int(num_layers * mamba_ratio)
            num_transformer_layers = num_layers - num_mamba_layers
        else:
            num_mamba_layers = sum(1 for t in layer_types if t == 'mamba')
            num_transformer_layers = sum(1 for t in layer_types if t == 'transformer')
        
        # Embeddings
        embedding_params = self._calculate_embedding_params(config, vocab_size, hidden_size)
        
        # Calculate per-layer params
        # Mamba layers
        mamba_config = {**config, 'num_hidden_layers': 1}
        single_mamba = self._calculate_mamba_params(mamba_config) - embedding_params - hidden_size
        
        # Transformer layers
        transformer_config = {**config, 'num_hidden_layers': 1}
        single_transformer = self._calculate_transformer_params(transformer_config) - embedding_params - hidden_size
        
        # Total
        total_params = (
            embedding_params + 
            num_mamba_layers * single_mamba +
            num_transformer_layers * single_transformer +
            hidden_size  # Final norm
        )
        
        return int(total_params)
    
    def _calculate_diffusion_params(self, config: Dict[str, Any]) -> int:
        """Calculate diffusion U-Net parameters."""
        in_channels = config.get('in_channels', 4)
        out_channels = config.get('out_channels', 4)
        
        # Channel configuration
        if 'block_out_channels' in config:
            channels = config['block_out_channels']
        else:
            base_channels = config.get('base_channels', 320)
            channel_mult = config.get('channel_mult', [1, 2, 4, 4])
            channels = [base_channels * m for m in channel_mult]
        
        # Architecture config
        num_res_blocks = config.get('num_res_blocks', config.get('layers_per_block', 2))
        attention_resolutions = config.get('attention_resolutions', [16, 8])
        num_heads = config.get('num_attention_heads', 8)
        
        # Time embedding
        time_embed_dim = config.get('time_embed_dim', channels[0] * 4)
        total_params = channels[0] + time_embed_dim + time_embed_dim * channels[0] * 2
        
        # Initial conv
        total_params += in_channels * channels[0] * 3 * 3 + channels[0]
        
        # Down path
        current_resolution = config.get('image_size', 64)
        ch = channels[0]
        
        for i, ch_out in enumerate(channels):
            for j in range(num_res_blocks):
                total_params += self._calculate_resnet_block_params(ch, ch_out, time_embed_dim)
                ch = ch_out
                
                if current_resolution in attention_resolutions:
                    total_params += self._calculate_attention_block_params(ch, num_heads)
            
            if i < len(channels) - 1:
                total_params += ch * channels[i+1] * 3 * 3 + channels[i+1]
                current_resolution //= 2
        
        # Middle
        middle_channels = channels[-1]
        total_params += self._calculate_resnet_block_params(middle_channels, middle_channels, time_embed_dim)
        total_params += self._calculate_attention_block_params(middle_channels, num_heads)
        total_params += self._calculate_resnet_block_params(middle_channels, middle_channels, time_embed_dim)
        
        # Up path (symmetric)
        for i in reversed(range(len(channels))):
            ch = channels[i]
            for j in range(num_res_blocks + 1):
                ch_in = ch * 2 if j == num_res_blocks else ch
                total_params += self._calculate_resnet_block_params(ch_in, ch, time_embed_dim)
                
                if current_resolution in attention_resolutions:
                    total_params += self._calculate_attention_block_params(ch, num_heads)
            
            if i > 0:
                total_params += ch * channels[i-1] * 3 * 3 + channels[i-1]
                current_resolution *= 2
        
        # Final layers
        total_params += channels[0] * 2  # Final norm
        total_params += channels[0] * out_channels * 3 * 3 + out_channels
        
        # Cross-attention for conditioning
        if 'cross_attention_dim' in config:
            cross_attn_dim = config['cross_attention_dim']
            num_cross_attention_blocks = sum(1 for r in attention_resolutions for _ in range(len(channels)))
            
            for _ in range(num_cross_attention_blocks):
                total_params += (
                    channels[0] * channels[0] +      # Q
                    cross_attn_dim * channels[0] * 2 +  # K, V
                    channels[0] * channels[0] +      # Output
                    channels[0] * 2                  # Norm
                )
        
        return int(total_params)
    
    def _calculate_resnet_block_params(self, in_channels: int, out_channels: int, temb_channels: int) -> int:
        """Calculate ResNet block parameters."""
        params = 0
        
        # First conv
        params += in_channels * out_channels * 3 * 3 + out_channels
        
        # Time embedding projection
        params += temb_channels * out_channels + out_channels
        
        # Second conv
        params += out_channels * out_channels * 3 * 3 + out_channels
        
        # Skip connection
        if in_channels != out_channels:
            params += in_channels * out_channels * 1 * 1 + out_channels
        
        # Group norms
        params += in_channels + out_channels
        
        return params
    
    def _calculate_attention_block_params(self, channels: int, num_heads: int) -> int:
        """Calculate attention block parameters."""
        # Group norm
        params = channels
        
        # Q, K, V, O projections
        params += 4 * channels * channels
        
        return params
    
    def _fallback_calculation(self, config: Dict[str, Any]) -> int:
        """Fallback calculation for unknown architectures."""
        # Try to identify basic components
        params = 0
        
        # Look for embeddings
        if 'vocab_size' in config:
            hidden_size = config.get('hidden_size', config.get('d_model', 768))
            params += config['vocab_size'] * hidden_size
        
        # Look for layers
        num_layers = config.get('num_hidden_layers', config.get('num_layers', 0))
        if num_layers > 0:
            hidden_size = config.get('hidden_size', 768)
            # Rough estimate: attention + FFN + norms
            params += num_layers * (12 * hidden_size * hidden_size + 2 * hidden_size)
        
        return int(params) if params > 0 else 1000000  # Default 1M params
    
    def get_parameter_breakdown(self, config: Union[Dict[str, Any], Any]) -> Dict[str, int]:
        """Get detailed parameter breakdown by component."""
        if hasattr(config, '__dict__'):
            config = vars(config)
        
        breakdown = {}
        
        # Basic architecture info
        vocab_size = config.get('vocab_size', 32000)
        hidden_size = config.get('hidden_size', 768)
        num_layers = config.get('num_hidden_layers', 12)
        num_attention_heads = config.get('num_attention_heads', 12)
        num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
        intermediate_size = config.get('intermediate_size', hidden_size * 4)
        
        # Embeddings
        breakdown['embeddings'] = self._calculate_embedding_params(config, vocab_size, hidden_size)
        
        # Check if this is an MoE model
        n_routed_experts = config.get('n_routed_experts', config.get('num_experts', config.get('num_local_experts', 1)))
        is_moe = n_routed_experts > 1
        
        if is_moe:
            # MoE model - calculate layer-specific parameters
            moe_components = self._calculate_moe_layer_params(
                config, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size
            )
            
            # Get layer configuration
            first_k_dense_replace = config.get('first_k_dense_replace', 0)
            moe_layer_freq = config.get('moe_layer_freq', 1)
            
            # Count different layer types
            num_moe_layers = 0
            num_dense_layers = 0
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace:
                    num_dense_layers += 1
                elif moe_layer_freq == 1 or (layer_idx - first_k_dense_replace) % moe_layer_freq == 0:
                    num_moe_layers += 1
                else:
                    num_dense_layers += 1
            
            # Attention for all layers
            attention_params = self._calculate_attention_params(config, hidden_size, num_attention_heads, num_key_value_heads)
            breakdown['attention'] = attention_params * num_layers
            
            # Normalization for all layers
            norm_params = self._calculate_norm_params(config, hidden_size, 2)
            breakdown['normalization'] = norm_params * num_layers + self._calculate_norm_params(config, hidden_size, 1)
            
            # Dense FFN layers
            if num_dense_layers > 0:
                dense_ffn = self._calculate_ffn_params(config, hidden_size, intermediate_size)
                breakdown['dense_ffn'] = dense_ffn * num_dense_layers
            
            # MoE components
            if 'routed_experts' in moe_components and num_moe_layers > 0:
                breakdown['routed_experts'] = moe_components['routed_experts'] * num_moe_layers
            
            if 'shared_experts' in moe_components and num_moe_layers > 0:
                breakdown['shared_experts'] = moe_components['shared_experts'] * num_moe_layers
            
            if 'router' in moe_components and num_moe_layers > 0:
                breakdown['routers'] = moe_components['router'] * num_moe_layers
        else:
            # Standard model breakdown
            attention_params = self._calculate_attention_params(config, hidden_size, num_attention_heads, num_key_value_heads)
            ffn_params = self._calculate_ffn_params(config, hidden_size, intermediate_size)
            norm_params = self._calculate_norm_params(config, hidden_size, 2)
            
            breakdown['attention'] = attention_params * num_layers
            breakdown['ffn'] = ffn_params * num_layers
            breakdown['normalization'] = norm_params * num_layers + self._calculate_norm_params(config, hidden_size, 1)
        
        # Auxiliary sequence heads if present
        aux_params = self._aux_seq_head_params(config, hidden_size)
        if aux_params > 0:
            breakdown['auxiliary_heads'] = aux_params
        
        # Multi-token prediction (MTP) layers
        num_mtp_layers = config.get('num_nextn_predict_layers', 0)
        if num_mtp_layers > 0:
            mtp_params = 0
            for _ in range(num_mtp_layers):
                # Projection matrix: 2 * hidden -> hidden
                mtp_params += 2 * hidden_size * hidden_size
                
                # Transformer block
                mtp_attention = self._calculate_attention_params(config, hidden_size, num_attention_heads, num_key_value_heads)
                mtp_ffn = self._calculate_ffn_params(config, hidden_size, intermediate_size)
                mtp_norms = self._calculate_norm_params(config, hidden_size, 2)
                
                mtp_params += mtp_attention + mtp_ffn + mtp_norms
            
            breakdown['mtp_layers'] = mtp_params
        
        # Total
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown
    
     

qwen_8b = {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 131072,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "attn_factor": 0.8782488562869419
  },
  "rope_theta": 1000000,
  "sliding_window": None,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 151936
}

config_llama_70b = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 28672,
  "max_position_embeddings": 131072,
  "mlp_bias": False,
  "model_type": "llama",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.47.0.dev0",
  "use_cache": True,
  "vocab_size": 128256
}


config_Deepseek_R1 = {
  "architectures": [
    "DeepseekV3ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    "AutoModel": "modeling_deepseek.DeepseekV3Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
  },
  "bos_token_id": 0,
  "eos_token_id": 1,
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 18432,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v3",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": True,
  "num_attention_heads": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 61,
  "num_key_value_heads": 128,
  "num_nextn_predict_layers": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": False,
  "topk_group": 4,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.46.3",
  "use_cache": True,
  "v_head_dim": 128,
  "vocab_size": 129280
}

config_v2_5_236_b = {
  "architectures": [
    "DeepseekV2ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"
  },
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1536,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 160,
  "n_shared_experts": 2,
  "norm_topk_prob": False,
  "num_attention_heads": 128,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 60,
  "num_key_value_heads": 128,
  "pretraining_tp": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 16.0,
  "scoring_func": "softmax",
  "seq_aux": True,
  "tie_word_embeddings": False,
  "topk_group": 3,
  "topk_method": "group_limited_greedy",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.39.3",
  "use_cache": True,
  "v_head_dim": 128,
  "vocab_size": 102400
}

nano_config = {
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 128000,
  "max_window_layers": 70,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "mrope_section": [
      16,
      24,
      24
    ],
    "rope_type": "default",
    "type": "default"
  },
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "text_config": {
    "architectures": [
      "Qwen2_5_VLForConditionalGeneration"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "image_token_id": None,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 128000,
    "max_window_layers": 70,
    "model_type": "qwen2_5_vl_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 36,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_section": [
        16,
        24,
        24
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "video_token_id": None,
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "vision_token_id": 151654,
    "vocab_size": 151936
  },
  "torch_dtype": "bfloat16",
  "transformers_version": "4.52.4",
  "use_cache": True,
  "use_sliding_window": False,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "hidden_act": "silu",
    "hidden_size": 1280,
    "in_channels": 3,
    "in_chans": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3420,
    "model_type": "qwen2_5_vl",
    "num_heads": 16,
    "out_hidden_size": 2048,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "tokens_per_second": 2,
    "torch_dtype": "bfloat16",
    "window_size": 112
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 151936
}

mini_max_config = {
  "architectures": [
    "MiniMaxM1ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "attn_type_list": [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
  ],
  "auto_map": {
    "AutoConfig": "configuration_minimax_m1.MiniMaxM1Config",
    "AutoModelForCausalLM": "modeling_minimax_m1.MiniMaxM1ForCausalLM"
  },
  "bos_token_id": None,
  "eos_token_id": None,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 6144,
  "initializer_range": 0.02,
  "intermediate_size": 9216,
  "layernorm_full_attention_alpha": 3.5565588200778455,
  "layernorm_full_attention_beta": 1.0,
  "layernorm_linear_attention_alpha": 3.5565588200778455,
  "layernorm_linear_attention_beta": 1.0,
  "layernorm_mlp_alpha": 3.5565588200778455,
  "layernorm_mlp_beta": 1.0,
  "max_position_embeddings": 10240000,
  "model_type": "minimax_m1",
  "num_attention_heads": 64,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "output_router_logits": False,
  "postnorm": True,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000000,
  "rotary_dim": 64,
  "router_aux_loss_coef": 0.001,
  "router_jitter_noise": 0.0,
  "shared_intermediate_size": 0,
  "shared_moe_mode": "sigmoid",
  "sliding_window": None,
  "tie_word_embeddings": False,
  "transformers_version": "4.45.2",
  "use_cache": True,
  "vocab_size": 200064
}


def test_deepseek_v3_counting():
    """Test parameter counting for DeepSeek V3 model."""
    
    # Initialize counter
    counter = UniversalParameterCounter()
    
    # DeepSeek V3 config
    deepseek_v3_config ={
  "architectures": [
    "DeepseekForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekConfig",
    "AutoModel": "modeling_deepseek.DeepseekModel",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekForCausalLM"
  },
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10944,
  "max_position_embeddings": 4096,
  "model_type": "deepseek",
  "moe_intermediate_size": 1408,
  "moe_layer_freq": 1,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "norm_topk_prob": False,
  "num_attention_heads": 16,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 28,
  "num_key_value_heads": 16,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "rope_theta": 10000,
  "scoring_func": "softmax",
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.0",
  "use_cache": True,
  "vocab_size": 102400
}
    
    # Count total parameters
    total_params = counter.count_parameters(deepseek_v3_config)
    print(f"DeepSeek V2 16B real MoE Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

    print("Qwen 8B")
    total_params = counter.count_parameters(qwen_8b)
    print(f"Qwen 8B real Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")
    
    print("Llama 70B")
    total_params = counter.count_parameters(config_llama_70b)
    print(f"Llama 70B real Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

    print("Deepseek R1")
    total_params = counter.count_parameters(config_Deepseek_R1)
    print(f"Deepseek R1  685B real Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

    print("Deepseek V2.5.236B")
    total_params = counter.count_parameters(config_v2_5_236_b)
    print(f"Deepseek V2.5.236B real 265B Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

    print("Nano")
    total_params = counter.count_parameters(nano_config)
    print(f"Nano real 3.75B Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

    print("MiniMax")
    total_params = counter.count_parameters(mini_max_config)
    print(f"MiniMax real 456B Total Parameters, and predicted parameters: {total_params:,} ({total_params/1e9:.1f}B)")

if __name__ == "__main__":
    test_deepseek_v3_counting()
