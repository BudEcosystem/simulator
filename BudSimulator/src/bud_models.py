import math
from typing import Dict, Any, Optional, Tuple, List, Union
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    from huggingface_hub import HfApi, hf_hub_download, ModelCard
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    raise ImportError(
        "Please install huggingface_hub: pip install huggingface_hub"
    )


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
        # Extract dimensions with fallbacks and validation
        vocab_size = config.get('vocab_size', config.get('n_vocab', 32000))
        hidden_size = config.get('hidden_size', config.get('d_model', config.get('n_embd', 768)))
        num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 12)))
        intermediate_size = config.get('intermediate_size', config.get('ffn_dim', config.get('d_ff', hidden_size * 4)))
        
        # Validate and sanitize inputs
        vocab_size = max(1000, abs(vocab_size)) if vocab_size else 32000
        hidden_size = max(256, abs(hidden_size)) if hidden_size else 768
        num_layers = max(1, abs(num_layers)) if num_layers else 12
        intermediate_size = max(hidden_size, abs(intermediate_size)) if intermediate_size else hidden_size * 4
        
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
        # Try to identify basic components with validation
        params = 0
        
        # Look for embeddings with validation
        vocab_size = config.get('vocab_size', 32000)
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        
        # Ensure positive values
        vocab_size = max(1000, abs(vocab_size)) if vocab_size else 32000
        hidden_size = max(256, abs(hidden_size)) if hidden_size else 768
        
        params += vocab_size * hidden_size
        
        # Look for layers with validation
        num_layers = config.get('num_hidden_layers', config.get('num_layers', 12))
        num_layers = max(1, abs(num_layers)) if num_layers else 12
        
        # Rough estimate: attention + FFN + norms
        params += num_layers * (12 * hidden_size * hidden_size + 2 * hidden_size)
        
        return max(1000000, int(params))  # Ensure at least 1M params

@dataclass
class MemoryReport:
    """Detailed memory breakdown for model inference."""
    model_type: str
    attention_type: Optional[str]
    precision: str
    parameter_count: int
    weight_memory_bytes: float
    kv_cache_bytes: float
    activation_memory_bytes: float
    state_memory_bytes: float
    image_memory_bytes: float
    extra_work_bytes: float
    
    @property
    def total_memory_bytes(self) -> float:
        return (
            self.weight_memory_bytes +
            self.kv_cache_bytes +
            self.activation_memory_bytes +
            self.state_memory_bytes +
            self.image_memory_bytes +
            self.extra_work_bytes
        )
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / 1e9
    
    # Convenience properties for GB values
    @property
    def weight_memory_gb(self) -> float:
        return self.weight_memory_bytes / 1e9
    
    @property
    def kv_cache_gb(self) -> float:
        return self.kv_cache_bytes / 1e9
    
    @property
    def activation_memory_gb(self) -> float:
        return self.activation_memory_bytes / 1e9
    
    @property
    def state_memory_gb(self) -> float:
        return self.state_memory_bytes / 1e9
    
    @property
    def image_memory_gb(self) -> float:
        return self.image_memory_bytes / 1e9
    
    @property
    def extra_work_gb(self) -> float:
        return self.extra_work_bytes / 1e9
    
    @property
    def recommended_gpu_memory_gb(self) -> float:
        """Recommended GPU memory size (rounded up to nearest 8GB)."""
        return math.ceil(self.total_memory_gb / 8) * 8
    
    @property
    def can_fit_24gb_gpu(self) -> bool:
        """Check if model can fit in 24GB GPU."""
        return self.total_memory_gb < 24
    
    @property
    def can_fit_80gb_gpu(self) -> bool:
        """Check if model can fit in 80GB GPU."""
        return self.total_memory_gb < 80
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'attention_type': self.attention_type,
            'precision': self.precision,
            'parameter_count': self.parameter_count,
            'weight_memory_gb': round(self.weight_memory_gb, 3),
            'kv_cache_gb': round(self.kv_cache_gb, 3),
            'activation_memory_gb': round(self.activation_memory_gb, 3),
            'state_memory_gb': round(self.state_memory_gb, 3),
            'image_memory_gb': round(self.image_memory_gb, 3),
            'extra_work_gb': round(self.extra_work_gb, 3),
            'total_memory_gb': round(self.total_memory_gb, 3),
            'memory_per_token_mb': round((self.kv_cache_bytes / 1024**2) / max(1, self.kv_cache_bytes / self.weight_memory_bytes * 1000), 3) if self.kv_cache_bytes > 0 else 0,
            'recommended_gpu_memory_gb': self.recommended_gpu_memory_gb,
            'can_fit_24gb_gpu': self.can_fit_24gb_gpu,
            'can_fit_80gb_gpu': self.can_fit_80gb_gpu,
        }

class ModelMemoryCalculator:
    """
    Comprehensive memory calculator for various model architectures during inference.
    Handles all edge cases from production deployments.
    
    Supported attention mechanisms:
    - MHA (Multi-Head Attention): Standard attention with Q, K, V projections
    - MQA (Multi-Query Attention): Single K, V head shared across all Q heads
    - GQA (Grouped-Query Attention): Groups of Q heads share K, V heads
    - MLA (Multi-head Latent Attention): Compresses K, V into low-rank latent space (DeepSeek V2)
    
    MLA provides superior KV cache compression by projecting keys and values into a 
    lower-dimensional latent space before attention computation, reducing memory by 80-90%.
    """
    
    # Precision to bytes mapping
    PRECISION_BYTES = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2, 
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
        'int4': 0.5, 'uint4': 0.5, 'nf4': 0.5,
        'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1,
        'auto': 2,  # safe default
    }
    
    # Model type patterns for detection
    ENCODER_MODELS = ['bert', 'roberta', 'deberta', 'electra', 'albert', 'xlm', 'distilbert', 'xlnet']
    DECODER_MODELS = ['gpt2', 'gpt', 'gptj', 'gpt_neo', 'gpt_neox', 'llama', 'mistral', 'mixtral', 'opt', 'bloom', 'falcon', 'qwen', 'yi', 'gemma', 'phi', 'deepseek']
    VISION_MODELS = ['clip', 'vit', 'deit', 'swin', 'convnext']
    SPEECH_MODELS = ['whisper', 'wav2vec2', 'hubert', 'wavlm']
    DIFFUSION_MODELS = ['unet', 'dit', 'diffusion', 'stable-diffusion', 'sdxl']
    SSM_MODELS = ['mamba', 's4', 'ssm']
    TTS_MODELS = ['tts', 'text_to_speech', 'dia', 'sesame', 'parler']
    
    def __init__(self):
        self.config = {}
        self.model_type = None
        self.attention_type = None
        # Initialize the UniversalParameterCounter for improved parameter calculations
        self._param_counter = UniversalParameterCounter()
        
    def detect_model_type(self, config: Dict[str, Any]) -> str:
        """Detect the model architecture type from config."""
        model_type = config.get('model_type', '').lower()
        architectures = config.get('architectures', [])
        arch_str = ' '.join(architectures).lower() if architectures else ''
        
        # Priority 1: Specific model types
        if any(speech in model_type for speech in self.SPEECH_MODELS):
            return 'speech_to_text'

        # Check for specific model types
        if any(diff in model_type or diff in arch_str for diff in self.DIFFUSION_MODELS):
            return 'diffusion'
            
        if any(ssm in model_type for ssm in self.SSM_MODELS) or config.get('state_size'):
            return 'mamba'
            
        if any(tts in model_type for tts in self.TTS_MODELS):
            return 'text_to_speech'
            
        if any(enc in model_type for enc in self.ENCODER_MODELS):
            if config.get('is_decoder', False) or config.get('is_encoder_decoder', False):
                return 'encoder-decoder'
            return 'encoder-only'
            
        if any(dec in model_type for dec in self.DECODER_MODELS):
            # Check for MoE variants
            if 'num_experts' in config or 'num_local_experts' in config:
                return 'moe'
            return 'decoder-only'
            
        if any(vis in model_type or vis in arch_str for vis in self.VISION_MODELS):
            if 'text_config' in config or 'decoder_config' in config:
                return 'multimodal'
            return 'vision'
            
        if config.get('is_encoder_decoder', False):
            return 'encoder-decoder'
            
        # Check for MoE
        if 'num_experts' in config or 'num_local_experts' in config:
            if any(ssm in model_type for ssm in self.SSM_MODELS):
                return 'hybrid'  # Like Jamba
            return 'moe'
            
        # Check for multimodal indicators
        if 'vision_config' in config or 'image_size' in config or 'clip_config' in config:
            return 'multimodal'
            
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
        # Check for MLA first (DeepSeek uses this)
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
    
    def calculate_model_weights(self, config: Dict[str, Any], precision: str) -> Tuple[int, float]:
        """Calculate model weight memory in GB and return (param_count, memory_gb)."""
        bytes_per_param = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # Priority 1: Check for direct num_parameters field (most accurate)
        if 'num_parameters' in config:
            param_count = int(config['num_parameters'])
        else:
            # Calculate based on architecture
            param_count = self._calculate_transformer_params(config)
            
        # Add extra parameters if specified
        param_count += config.get('extra_parameters', 0)
        param_count += config.get('audio_decoder_parameters', 0)
        param_count += config.get('clip_parameters', 0)
        param_count += config.get('vae_parameters', 0)
        param_count += config.get('vision_parameters', 0)
        
        # Handle refiner for SDXL
        if config.get('refiner'):
            refiner_params = config.get('refiner_parameters', config.get('num_parameters', param_count))
            param_count += refiner_params
        
        memory_gb = (param_count * bytes_per_param) / 1e9
        return param_count, memory_gb
    
    def _calculate_transformer_params(self, config: Dict[str, Any]) -> int:
        """Calculate transformer model parameters using UniversalParameterCounter."""
        try:
            # Use the sophisticated UniversalParameterCounter for accurate calculations
            return self._param_counter.count_parameters(config)
        except Exception as e:
            # Fallback to simplified calculation if UniversalParameterCounter fails
            warnings.warn(f"UniversalParameterCounter failed: {e}. Using fallback calculation.")
            return self._fallback_transformer_calculation(config)
    
    def _fallback_transformer_calculation(self, config: Dict[str, Any]) -> int:
        """Fallback transformer parameter calculation for backward compatibility."""
        vocab_size = config.get('vocab_size', 32000)
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 12)))
        intermediate_size = config.get('intermediate_size', config.get('ffn_dim', config.get('d_ff', hidden_size * 4)))
        max_position_embeddings = config.get('max_position_embeddings', 512)
        
        # Get attention dimensions with kv_channels support (e.g., Qwen models)
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        num_key_value_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
        
        # Handle custom head dimensions and kv_channels
        head_dim = config.get('head_dim', hidden_size // max(1, num_attention_heads))
        kv_channels = config.get('kv_channels')
        if kv_channels:
            kv_dim = kv_channels
        else:
            kv_dim = head_dim * num_key_value_heads
        
        # Embeddings
        embedding_params = vocab_size * hidden_size
        if not config.get('tie_word_embeddings', True):
            embedding_params *= 2  # Input and output embeddings
        embedding_params += max_position_embeddings * hidden_size
        
        # Attention projections with proper dimensions
        if 'q_lora_rank' in config or 'kv_lora_rank' in config:
            # MLA uses low-rank projections
            q_lora_rank = config.get('q_lora_rank', 1536)
            kv_lora_rank = config.get('kv_lora_rank', 512)
            qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
            qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
            v_head_dim = config.get('v_head_dim', qk_nope_head_dim)
            
            # MLA attention params: down projection + up projection for Q, KV
            attention_params = (
                hidden_size * q_lora_rank +  # Q down projection
                q_lora_rank * hidden_size +  # Q up projection
                hidden_size * kv_lora_rank + # KV down projection
                kv_lora_rank * (qk_rope_head_dim + qk_nope_head_dim + v_head_dim) * num_attention_heads +  # KV up projection
                hidden_size * hidden_size    # O projection
            )
        else:
            # Standard attention
            # Q projection: hidden_size -> hidden_size
            # K, V projections: hidden_size -> kv_dim (might be smaller)
            # O projection: hidden_size -> hidden_size
            attention_params = (
                hidden_size * hidden_size +    # Q projection
                hidden_size * kv_dim +         # K projection  
                hidden_size * kv_dim +         # V projection
                hidden_size * hidden_size      # O projection
            )
        
        # FFN parameters depend on activation function
        ffn_mult = self._get_activation_function_multiplier(config)
        ffn_params = ffn_mult * hidden_size * intermediate_size
        
        # Layer norms: typically 2 per layer (pre-attention, pre-ffn)
        ln_params = 4 * hidden_size
        
        # Total for transformer layers
        layer_params = attention_params + ffn_params + ln_params
        total_params = embedding_params + num_layers * layer_params + hidden_size  # Final layer norm
        
        # Handle MoE if present
        if 'num_experts' in config or 'num_local_experts' in config:
            num_experts = config.get('num_experts', config.get('num_local_experts', 8))
            # For MoE, replace FFN with expert FFNs
            expert_ffn_params = ffn_params * num_experts
            # Subtract original FFN and add expert FFNs
            total_params = total_params - (num_layers * ffn_params) + (num_layers * expert_ffn_params)
        
        return int(total_params)
    
    def calculate_kv_cache(self, config: Dict[str, Any], batch_size: int, 
                          seq_length: int, precision: str) -> float:
        """Calculate KV cache memory in GB."""
        model_type = self.detect_model_type(config)
        
        # Only decoder models and some encoder-decoder models use KV cache
        if model_type not in ['decoder-only', 'multimodal', 'hybrid', 'speech_to_text', 'encoder-decoder', 'moe', 'text_to_speech']:
            return 0.0
        
        # For Mamba/SSM models, return state memory instead
        if model_type == 'mamba':
            return self.calculate_state_memory(config, batch_size, precision)
        
        attention_type = self.detect_attention_type(config)
        if attention_type is None:
            return 0.0
        
        bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        # Get model dimensions
        num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 12)))
        hidden_size = config.get('hidden_size', config.get('d_model', 768))
        num_attention_heads = config.get('num_attention_heads', config.get('n_head', 12))
        
        # For hybrid models, only count attention layers
        if model_type == 'hybrid':
            num_attention_layers = config.get('num_attention_layers', num_layers // 8)
            num_layers = num_attention_layers
        
        # Calculate based on attention type
        if attention_type == 'mha':
            # Standard multi-head attention
            kv_elements = 2 * batch_size * num_layers * seq_length * hidden_size
        elif attention_type == 'mqa':
            # Multi-query attention (single KV head)
            head_dim = hidden_size // max(1, num_attention_heads)
            kv_elements = 2 * batch_size * num_layers * seq_length * head_dim
        elif attention_type == 'gqa':
            # Grouped-query attention
            num_kv_heads = config.get('num_key_value_heads', config.get('num_kv_heads', num_attention_heads))
            head_dim = hidden_size // max(1, num_attention_heads)
            kv_elements = 2 * batch_size * num_layers * seq_length * num_kv_heads * head_dim
        elif attention_type == 'mla':
            # Multi-latent attention (DeepSeek style)
            # MLA compresses KV into a latent space
            if 'kv_lora_rank' in config:
                # DeepSeek V2 style MLA
                kv_lora_rank = config.get('kv_lora_rank', 512)
                # Compressed KV cache stores the latent representations
                kv_elements = 2 * batch_size * num_layers * seq_length * kv_lora_rank
            else:
                # Generic MLA with latent_attention_dim
                latent_dim = config.get('latent_attention_dim', config.get('compressed_kv_dim', 128))
                kv_elements = 2 * batch_size * num_layers * seq_length * latent_dim
        else:
            kv_elements = 0
        
        # Handle sliding window attention (e.g., Mistral)
        if 'sliding_window' in config:
            window_size = config['sliding_window']
            if window_size is not None:
                seq_length = min(seq_length, window_size)
                kv_elements = kv_elements * (seq_length / max(seq_length, window_size))
        
        return (kv_elements * bytes_per_value) / 1e9
    
    def calculate_activations(self, config: Dict[str, Any], batch_size: int, 
                            seq_length: int, precision: str, 
                            decode_length: int = 0) -> float:
        """
        Calculate activation memory in GB.
        
        Args:
            config: Model configuration
            batch_size: Batch size
            seq_length: Prefill/prompt sequence length
            precision: Precision format
            decode_length: Number of tokens to generate (0 for prefill only)
        """
        model_type = self.detect_model_type(config)
        bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        if model_type == 'encoder-only':
            # Encoder processes all tokens in parallel
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_hidden_layers', 12)
            
            # Peak activation estimate: ~36x hidden size per layer per token
            activation_elements = batch_size * seq_length * hidden_size * 36 * num_layers
            
        elif model_type in ['decoder-only', 'moe']:
            # Decoder has different memory for prefill vs decode
            hidden_size = config.get('hidden_size', 768)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            
            # For MoE, account for multiple active experts
            if model_type == 'moe':
                top_k = config.get('top_k', config.get('num_experts_per_tok', 2))
                intermediate_size *= top_k
            
            # Prefill: process all prompt tokens at once
            prefill_elements = batch_size * seq_length * intermediate_size * 2
            
            # Decode: process one token at a time
            if decode_length > 0:
                decode_elements = batch_size * decode_length * intermediate_size * 2
                # Use max of prefill and per-token decode
                activation_elements = max(prefill_elements, decode_elements / decode_length)
            else:
                activation_elements = prefill_elements
            
        elif model_type == 'multimodal':
            # Vision tokens + text tokens
            text_config = config.get('text_config', config)
            vision_config = config.get('vision_config', {})
            
            text_hidden = text_config.get('hidden_size', 768)
            vision_hidden = vision_config.get('hidden_size', 768)
            
            # Vision tokens
            if 'image_size' in vision_config:
                patch_size = vision_config.get('patch_size', 16)
                num_patches = (vision_config['image_size'] // patch_size) ** 2
                vision_seq_length = num_patches
            else:
                vision_seq_length = 197  # Default for 224x224 with patch 16
            
            # Combined activation
            activation_elements = (batch_size * seq_length * text_hidden * 4 + 
                                 batch_size * vision_seq_length * vision_hidden * 4)
            
        elif model_type == 'diffusion':
            # U-Net feature maps - use simplified calculation
            if 'sample_size' in config:
                latent_size = config['sample_size']
            else:
                # Assume 512x512 image -> 64x64 latent
                latent_size = 64
            
            # Latent channels
            in_channels = config.get('in_channels', 4)
            
            # Feature maps at different scales - simplified estimate
            activation_elements = batch_size * latent_size * latent_size * in_channels * 100  # Rough multiplier
            
        elif model_type in ['speech_to_text', 'encoder-decoder']:
            # Similar to encoder
            hidden_size = config.get('d_model', config.get('hidden_size', 1024))
            max_source_positions = config.get('max_source_positions', 1500)
            
            activation_elements = batch_size * max_source_positions * hidden_size * 4
            
        elif model_type == 'mamba':
            # State space models have minimal activation memory
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_hidden_layers', 24)
            
            # Only current token activations
            activation_elements = batch_size * num_layers * hidden_size * 4
            
        elif model_type == 'text_to_speech':
            # TTS models often have large intermediate activations
            hidden_size = config.get('hidden_size', 1024)
            # Rough estimate for TTS activations
            activation_elements = batch_size * seq_length * hidden_size * 10
            
        else:
            # Default estimation
            hidden_size = config.get('hidden_size', 768)
            activation_elements = batch_size * seq_length * hidden_size * 10
        
        return (activation_elements * bytes_per_value) / 1e9
    
    def calculate_state_memory(self, config: Dict[str, Any], batch_size: int, precision: str) -> float:
        """Calculate state memory for SSM/Mamba models in GB."""
        model_type = self.detect_model_type(config)
        
        if model_type not in ['mamba', 'hybrid']:
            return 0.0
        
        bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
        
        state_size = config.get('state_size', config.get('d_state', config.get('n_state', 16)))
        
        if model_type == 'mamba':
            num_layers = config.get('num_hidden_layers', config.get('n_layer', 24))
        else:  # hybrid
            num_ssm_layers = config.get('num_ssm_layers', config.get('mamba_layers', 20))
            num_layers = num_ssm_layers
        
        # Mamba state is much smaller than KV cache - constant size regardless of sequence length
        state_elements = batch_size * num_layers * state_size
        
        return (state_elements * bytes_per_value) / 1e9
    
    def calculate_total_memory(self, config: Dict[str, Any], 
                              batch_size: int = 1,
                              seq_length: int = 2048,
                              num_images: int = 0,
                              precision: str = 'fp16',
                              include_gradients: bool = False,
                              decode_length: int = 0,
                              image_resolution: int = 512,
                              framework_overhead: float = 1.2) -> MemoryReport:
        """
        Calculate total memory requirements for model inference.
        
        Args:
            config: HuggingFace model config dictionary
            batch_size: Batch size for inference
            seq_length: Sequence length (input + output tokens for prefill, or prompt length if decode_length > 0)
            num_images: Number of images (for multimodal models)
            precision: Precision format (fp32, fp16, bf16, int8, int4)
            include_gradients: Whether to include gradient memory (for training)
            decode_length: Number of tokens to generate (0 for prefill only)
            image_resolution: Resolution for diffusion models
            framework_overhead: Overhead multiplier (default 1.2 = 20%)
            
        Returns:
            MemoryReport with detailed breakdown
        """
        self.config = config
        self.model_type = self.detect_model_type(config)
        self.attention_type = self.detect_attention_type(config)
        
        # Calculate components
        param_count, model_weights = self.calculate_model_weights(config, precision)
        kv_cache = self.calculate_kv_cache(config, batch_size, seq_length + decode_length, precision)
        activations = self.calculate_activations(config, batch_size, seq_length, precision, decode_length)
        state_memory = self.calculate_state_memory(config, batch_size, precision)
        
        # Add image memory for multimodal models
        image_memory = 0.0
        if self.model_type == 'multimodal' and num_images > 0:
            vision_config = config.get('vision_config', {})
            image_size = vision_config.get('image_size', 224)
            patch_size = vision_config.get('patch_size', 16)
            hidden_size = vision_config.get('hidden_size', 768)
            
            patches_per_image = (image_size // patch_size) ** 2
            bytes_per_value = self.PRECISION_BYTES.get(precision.lower(), 2)
            image_memory = (num_images * patches_per_image * hidden_size * bytes_per_value) / 1e9
        
        # Extra work buffers (important for TTS, diffusion, etc.)
        extra_work_bytes = config.get('extra_work_bytes', 0)
        
        # Default work buffer for TTS models
        if self.model_type == 'text_to_speech' and 'extra_work_bytes' not in config:
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
        weight_bytes = model_weights * 1e9
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
            extra_work_bytes=extra_work_bytes + (runtime_bytes_with_overhead - runtime_bytes),  # Include overhead and work buffers
        )


# Utility functions
def estimate_memory(config: Dict[str, Any], **kwargs) -> MemoryReport:
    """Convenience function matching the reference API."""
    calculator = ModelMemoryCalculator()
    return calculator.calculate_total_memory(config, **kwargs)


def analyze_attention_efficiency(config: Dict[str, Any], seq_lengths: List[int] = [1024, 4096, 16384, 32768]) -> Dict[str, Any]:
    """Analyze KV cache memory efficiency for different sequence lengths."""
    calculator = ModelMemoryCalculator()
    attention_type = calculator.detect_attention_type(config)
    
    results = {}
    for seq_len in seq_lengths:
        report = calculator.calculate_total_memory(config, seq_length=seq_len)
        results[seq_len] = {
            'kv_cache_gb': report.kv_cache_bytes / 1e9,
            'total_memory_gb': report.total_memory_gb,
            'kv_cache_percent': (report.kv_cache_bytes / report.total_memory_bytes) * 100
        }
    
    return {
        'attention_type': attention_type,
        'sequence_lengths': results,
        'memory_per_token_bytes': report.kv_cache_bytes / seq_lengths[-1] if seq_lengths else 0
    }


def estimate_max_sequence_length(config: Dict[str, Any], 
                                gpu_memory_gb: float,
                                batch_size: int = 1,
                                precision: str = 'fp16') -> int:
    """Estimate maximum sequence length that fits in given GPU memory."""
    calculator = ModelMemoryCalculator()
    
    # Binary search for max sequence length
    low, high = 1, 1_000_000
    best_length = 0
    
    while low <= high:
        mid = (low + high) // 2
        result = calculator.calculate_total_memory(config, batch_size, mid, precision=precision)
        
        if result.total_memory_gb <= gpu_memory_gb * 0.9:  # Leave 10% buffer
            best_length = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_length


def estimate_max_batch_size(config: Dict[str, Any],
                           gpu_memory_gb: float,
                           seq_length: int = 2048,
                           precision: str = 'fp16') -> int:
    """Estimate maximum batch size that fits in given GPU memory."""
    calculator = ModelMemoryCalculator()
    
    # Try increasing batch sizes
    batch_size = 1
    while True:
        result = calculator.calculate_total_memory(config, batch_size, seq_length, precision=precision)
        if result.total_memory_gb > gpu_memory_gb * 0.9:  # Leave 10% buffer
            return max(1, batch_size - 1)
        batch_size += 1
        if batch_size > 1024:  # Reasonable upper limit
            return batch_size


class HuggingFaceConfigLoader:
    """Load and analyze model configs from HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            token: Optional HuggingFace API token for private models.
                   Can also be set via HF_TOKEN environment variable.
        """
        self.api = HfApi(token=token)
        self.token = token
        
    def fetch_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Fetch model configuration from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            
        Returns:
            Raw config dictionary from HuggingFace
            
        Raises:
            RepositoryNotFoundError: If model doesn't exist
            GatedRepoError: If model requires access request
            Exception: For other errors
        """
        # List of possible config file names in order of preference
        config_filenames = [
            "config.json",      # Standard HuggingFace format
            "params.json",      # Mistral format (e.g., Pixtral models)
            "model.json",       # Alternative format
            "configuration.json"  # Less common alternative
        ]
        
        last_error = None
        
        for filename in config_filenames:
            try:
                # Try to download the config file
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=self.token
                )
                
                # Read and parse the config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Add metadata about which file was used
                config['_config_filename'] = filename
                
                return config
                
            except GatedRepoError:
                # Re-raise gated repo errors with a specific indicator
                raise Exception(
                    f"MODEL_GATED: Model '{model_id}' is gated. You need to:\n"
                    f"1. Request access at https://huggingface.co/{model_id}\n"
                    f"2. Use a token with access: HuggingFaceConfigLoader(token='your_token')"
                )
            except RepositoryNotFoundError:
                # Re-raise if model doesn't exist
                raise Exception(f"Model '{model_id}' not found on HuggingFace Hub")
            except Exception as e:
                # For other errors (like file not found), try the next filename
                last_error = e
                if "404" in str(e) or "Not Found" in str(e):
                    continue
                else:
                    # For non-404 errors, raise immediately
                    raise
        
        # If we've tried all filenames and none worked, raise the last error
        if last_error:
            raise Exception(
                f"Could not find configuration file for model '{model_id}'. "
                f"Tried: {', '.join(config_filenames)}. "
                f"Last error: {str(last_error)}"
            )
        
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get additional model information from the API.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Model information including tags, downloads, etc.
        """
        try:
            return self.api.model_info(model_id, token=self.token)
        except Exception:
            return {}
            
    def extract_parameter_count(self, model_id: str, config: Dict[str, Any]) -> Optional[int]:
        """
        Try to extract parameter count from various sources.
        
        Args:
            model_id: HuggingFace model identifier
            config: Model configuration
            
        Returns:
            Parameter count if found, None otherwise
        """
        # Method 1: Check if already in config
        if 'num_parameters' in config:
            return int(config['num_parameters'])
            
        # Method 2: Try to get from model info
        try:
            model_info = self.get_model_info(model_id)
            
            # Check safetensors metadata
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                safetensors = model_info.safetensors
                if 'total' in safetensors:
                    return safetensors['total']
                elif 'parameters' in safetensors:
                    return safetensors['parameters']
            
            # Check model card data
            if hasattr(model_info, 'card_data') and model_info.card_data:
                card_data = model_info.card_data
                if isinstance(card_data, dict):
                    # Look for parameter count in various fields
                    for field in ['num_parameters', 'parameters', 'model_size']:
                        if field in card_data:
                            value = card_data[field]
                            if isinstance(value, (int, float)):
                                return int(value)
                            elif isinstance(value, str):
                                # Parse strings like "7B", "175B", etc.
                                return self._parse_param_string(value)
            
            # Check tags
            if hasattr(model_info, 'tags'):
                for tag in model_info.tags:
                    if isinstance(tag, str) and ('parameters:' in tag or 'params:' in tag):
                        parts = tag.split(':')
                        if len(parts) == 2:
                            return self._parse_param_string(parts[1])
                            
        except Exception:
            pass
            
        # Method 3: Try to parse from model card
        try:
            card = ModelCard.load(model_id, token=self.token)
            if card.content:
                # Look for parameter count in the model card text
                import re
                
                # Patterns to match parameter counts
                patterns = [
                    r'(\d+\.?\d*)\s*[Bb]illion\s*param',
                    r'(\d+\.?\d*)[Bb]\s*param',
                    r'(\d+\.?\d*)\s*billion\s*param',
                    r'parameters:\s*(\d+\.?\d*)[Bb]',
                    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*parameters',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, card.content, re.IGNORECASE)
                    if match:
                        value = match.group(1).replace(',', '')
                        if 'illion' in pattern or '[Bb]' in pattern:
                            return int(float(value) * 1e9)
                        else:
                            return int(float(value))
                            
        except Exception:
            pass
            
        return None
        
    def _parse_param_string(self, param_str: str) -> Optional[int]:
        """Parse parameter count from strings like '7B', '175B', '1.3M'."""
        param_str = param_str.strip()
        
        try:
            if param_str.endswith('B') or param_str.endswith('b'):
                return int(float(param_str[:-1]) * 1e9)
            elif param_str.endswith('M') or param_str.endswith('m'):
                return int(float(param_str[:-1]) * 1e6)
            elif param_str.endswith('K') or param_str.endswith('k'):
                return int(float(param_str[:-1]) * 1e3)
            else:
                # Try to parse as raw number
                return int(float(param_str.replace(',', '')))
        except:
            return None
            
    def map_config_to_memory_format(self, hf_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map HuggingFace config format to the format expected by the memory calculator.
        
        Args:
            hf_config: Raw config from HuggingFace
            
        Returns:
            Config dictionary compatible with memory calculator
        """
        # Start with a copy of the original config
        config = hf_config.copy()
        
        # Check if this is a Mistral params.json format
        if '_config_filename' in config and config['_config_filename'] == 'params.json':
            # Handle Mistral params.json specific mappings
            if 'dim' in config and 'hidden_size' not in config:
                config['hidden_size'] = config['dim']
            if 'n_layers' in config and 'num_hidden_layers' not in config:
                config['num_hidden_layers'] = config['n_layers']
            if 'head_dim' in config and 'n_heads' in config:
                config['num_attention_heads'] = config['n_heads']
            if 'n_kv_heads' in config and 'num_key_value_heads' not in config:
                config['num_key_value_heads'] = config['n_kv_heads']
            if 'hidden_dim' in config and 'intermediate_size' not in config:
                config['intermediate_size'] = config['hidden_dim']
            if 'norm_eps' in config and 'rms_norm_eps' not in config:
                config['rms_norm_eps'] = config['norm_eps']
            
            # Set model type if not present
            if 'model_type' not in config:
                # Try to infer from model structure
                if 'vision_encoder' in config or 'vision_config' in config:
                    config['model_type'] = 'multimodal'
                else:
                    config['model_type'] = 'mistral'
        
        # Ensure model_type is set
        if 'model_type' not in config and 'architectures' in config:
            # Try to infer from architectures
            arch = config['architectures'][0].lower() if config['architectures'] else ''
            
            model_type_mapping = {
                'llama': 'llama',
                'mistral': 'mistral',
                'mixtral': 'mixtral',
                'pixtral': 'multimodal',  # Add Pixtral mapping
                'gpt2': 'gpt2',
                'gpt_neo': 'gpt_neo',
                'gptj': 'gptj',
                'gpt_neox': 'gpt_neox',
                'bert': 'bert',
                'roberta': 'roberta',
                'deberta': 'deberta',
                'whisper': 'whisper',
                'clip': 'clip',
                't5': 't5',
                'bloom': 'bloom',
                'falcon': 'falcon',
                'phi': 'phi',
                'qwen': 'qwen',
                'mamba': 'mamba',
            }
            
            for key, model_type in model_type_mapping.items():
                if key in arch:
                    config['model_type'] = model_type
                    break
        
        # Map common HF config keys to our expected format
        key_mappings = {
            # GPT-style models
            'n_positions': 'max_position_embeddings',
            'n_embd': 'hidden_size',
            'n_layer': 'num_hidden_layers',
            'n_head': 'num_attention_heads',
            'n_inner': 'intermediate_size',
            
            # Mistral params.json style
            'dim': 'hidden_size',
            'n_layers': 'num_hidden_layers',
            'n_heads': 'num_attention_heads',
            'n_kv_heads': 'num_key_value_heads',
            'hidden_dim': 'intermediate_size',
            'norm_eps': 'rms_norm_eps',
            
            # T5/encoder-decoder style
            'd_model': 'hidden_size',
            'd_ff': 'intermediate_size',
            'd_kv': 'kv_channels',
            'num_layers': 'num_hidden_layers',
            'num_heads': 'num_attention_heads',
            
            # Whisper style
            'encoder_layers': 'num_hidden_layers',
            'decoder_layers': 'num_hidden_layers',
            'encoder_attention_heads': 'num_attention_heads',
            'decoder_attention_heads': 'num_attention_heads',
            'encoder_ffn_dim': 'intermediate_size',
            'decoder_ffn_dim': 'intermediate_size',
            
            # Other common mappings
            'num_heads': 'num_attention_heads',
            'num_kv_heads': 'num_key_value_heads',
            'embed_dim': 'hidden_size',
            'ffn_dim': 'intermediate_size',
            'hidden_act': 'activation_function',
            'activation': 'activation_function',
            'multi_query_attention': 'num_key_value_heads',  # If True, set to 1
        }
        
        # Apply mappings
        for hf_key, our_key in key_mappings.items():
            if hf_key in config and our_key not in config:
                config[our_key] = config[hf_key]
        
        # Handle special cases
        
        # 1. Multi-query attention flag
        if config.get('multi_query_attention') is True and 'num_key_value_heads' not in config:
            config['num_key_value_heads'] = 1
            
        # 2. Whisper models
        if config.get('model_type') == 'whisper':
            if 'd_model' in config:
                config['hidden_size'] = config['d_model']
            if 'encoder_layers' in config:
                config['num_hidden_layers'] = config['encoder_layers']
            if 'encoder_attention_heads' in config:
                config['num_attention_heads'] = config['encoder_attention_heads']
        
        # 3. GPT-style models
        if config.get('model_type') in ['gpt2', 'gpt_neo', 'gptj', 'gpt_neox']:
            if 'n_embd' in config:
                config['hidden_size'] = config['n_embd']
            if 'n_layer' in config:
                config['num_hidden_layers'] = config['n_layer']
            if 'n_head' in config:
                config['num_attention_heads'] = config['n_head']
                
        # 4. Phi models
        if config.get('model_type') == 'phi':
            if 'n_embd' in config:
                config['hidden_size'] = config['n_embd']
            if 'n_layer' in config:
                config['num_hidden_layers'] = config['n_layer']
            if 'n_head' in config:
                config['num_attention_heads'] = config['n_head']
            if 'n_head_kv' in config:
                config['num_key_value_heads'] = config['n_head_kv']
        
        # 5. T5-style models
        if config.get('model_type') == 't5':
            if 'd_model' in config:
                config['hidden_size'] = config['d_model']
            if 'num_layers' in config:
                config['num_hidden_layers'] = config['num_layers']
            if 'num_heads' in config:
                config['num_attention_heads'] = config['num_heads']
            if 'd_ff' in config:
                config['intermediate_size'] = config['d_ff']
        
        # 6. MoE models (Mixtral, DeepSeek)
        if 'num_local_experts' in config and 'num_experts' not in config:
            config['num_experts'] = config['num_local_experts']
        if 'num_experts_per_tok' not in config and 'top_k' in config:
            config['num_experts_per_tok'] = config['top_k']
        
        # 7. Multimodal models (LLaVA, CLIP, Pixtral)
        if 'text_config' in config and isinstance(config['text_config'], dict):
            # Recursively map text config
            config['text_config'] = self.map_config_to_memory_format(config['text_config'])
        if 'vision_config' in config and isinstance(config['vision_config'], dict):
            # Recursively map vision config
            config['vision_config'] = self.map_config_to_memory_format(config['vision_config'])
        if 'vision_encoder' in config and isinstance(config['vision_encoder'], dict):
            # Handle Pixtral-style vision encoder
            config['vision_config'] = self.map_config_to_memory_format(config['vision_encoder'])
        
        # 8. Handle activation functions
        if 'activation_function' not in config:
            if 'hidden_act' in config:
                config['activation_function'] = config['hidden_act']
            elif 'activation' in config:
                config['activation_function'] = config['activation']
        
        # 9. Ensure intermediate_size is set
        if 'intermediate_size' not in config and 'hidden_size' in config:
            # Model-specific defaults
            if config.get('model_type') == 'gpt2':
                config['intermediate_size'] = config['hidden_size'] * 4
            elif config.get('model_type') in ['llama', 'mistral']:
                # LLaMA uses ~2.7x
                config['intermediate_size'] = int(config['hidden_size'] * 2.7)
            else:
                # Default to 4x
                config['intermediate_size'] = config['hidden_size'] * 4
        
        # 10. Add any missing critical fields with sensible defaults
        if 'vocab_size' not in config:
            config['vocab_size'] = 32000  # Common default
            
        # 11. Handle sliding window attention
        if 'sliding_window_size' in config and 'sliding_window' not in config:
            config['sliding_window'] = config['sliding_window_size']
            
        # 12. Handle RoPE scaling
        if 'rope_scaling' in config and isinstance(config['rope_scaling'], dict):
            if 'factor' in config['rope_scaling']:
                # Adjust max position embeddings
                base_max_pos = config.get('max_position_embeddings', 2048)
                config['max_position_embeddings'] = int(base_max_pos * config['rope_scaling']['factor'])
        
        # 13. Handle max_seq_len for Mistral models
        if 'max_seq_len' in config and 'max_position_embeddings' not in config:
            config['max_position_embeddings'] = config['max_seq_len']
        
        return config
        
    def get_model_config(
        self, 
        model_id: str,
        add_param_count: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch and prepare model config from HuggingFace for memory estimation.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            add_param_count: Whether to try to find and add parameter count
            
        Returns:
            Config dictionary ready for memory estimation
        """
        # Fetch raw config
        hf_config = self.fetch_model_config(model_id)
        
        # Map to our format
        config = self.map_config_to_memory_format(hf_config)
        
        # Try to add parameter count if requested
        if add_param_count and 'num_parameters' not in config:
            param_count = self.extract_parameter_count(model_id, config)
            if param_count:
                config['num_parameters'] = param_count
        
        # Add source info
        config['_source'] = f"huggingface:{model_id}"
        
        return config
        
    def analyze_model(
        self,
        model_id: str,
        seq_length: int = 2048,
        batch_size: int = 1,
        precision: str = 'fp16',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete analysis of a HuggingFace model's memory requirements.
        
        Args:
            model_id: HuggingFace model identifier
            seq_length: Sequence length for analysis
            batch_size: Batch size for analysis
            precision: Precision format
            **kwargs: Additional arguments for estimate_memory
            
        Returns:
            Dictionary with model info and memory analysis
        """
        # Get config
        config = self.get_model_config(model_id)
        
        # Run memory estimation
        result = estimate_memory(
            config,
            seq_length=seq_length,
            batch_size=batch_size,
            precision=precision,
            **kwargs
        )
        
        # Get model info
        try:
            model_info = self.get_model_info(model_id)
            downloads = getattr(model_info, 'downloads', 0)
            likes = getattr(model_info, 'likes', 0)
        except:
            downloads = 0
            likes = 0
        
        # Create analysis report
        analysis = {
            'model_id': model_id,
            'model_type': result.model_type,
            'attention_type': result.attention_type,
            'parameter_count': result.parameter_count,
            'config': config,
            'memory_analysis': result.as_dict(),
            'deployment_recommendations': {
                'min_gpu_memory_gb': result.recommended_gpu_memory_gb,
                'can_fit_24gb': result.can_fit_24gb_gpu,
                'can_fit_80gb': result.can_fit_80gb_gpu,
            },
            'model_popularity': {
                'downloads': downloads,
                'likes': likes,
            }
        }
        
        # Add attention-specific insights
        if result.attention_type:
            if result.attention_type == 'mha':
                analysis['notes'] = "Consider using a GQA variant for better memory efficiency with long contexts"
            elif result.attention_type == 'gqa':
                kv_compression = config.get('num_attention_heads', 1) / config.get('num_key_value_heads', 1)
                analysis['notes'] = f"GQA provides {kv_compression:.1f}x KV cache compression"
            elif result.attention_type == 'mla':
                analysis['notes'] = "MLA provides excellent memory efficiency for long contexts"
        
        return analysis
        
    def compare_models(
        self,
        model_ids: List[str],
        seq_length: int = 2048,
        precision: str = 'fp16',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Compare memory requirements of multiple HuggingFace models.
        
        Args:
            model_ids: List of HuggingFace model identifiers
            seq_length: Sequence length for comparison
            precision: Precision format
            **kwargs: Additional arguments for estimate_memory
            
        Returns:
            List of analysis results
        """
        results = []
        
        for model_id in model_ids:
            try:
                analysis = self.analyze_model(
                    model_id, 
                    seq_length=seq_length, 
                    precision=precision,
                    **kwargs
                )
                results.append(analysis)
            except Exception as e:
                results.append({
                    'model_id': model_id,
                    'error': str(e)
                })
                
        return results


# Convenience functions for backward compatibility
def get_model_config_from_hf(
    model_id: str, 
    token: Optional[str] = None,
    add_param_count: bool = True
) -> Dict[str, Any]:
    """
    Fetch and prepare model config from HuggingFace for memory estimation.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        token: Optional HuggingFace API token for private models
        add_param_count: Whether to try to find and add parameter count
        
    Returns:
        Config dictionary ready for memory estimation
    """
    loader = HuggingFaceConfigLoader(token=token)
    return loader.get_model_config(model_id, add_param_count=add_param_count)


def analyze_hf_model(
    model_id: str,
    seq_length: int = 2048,
    batch_size: int = 1,
    precision: str = 'fp16',
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete analysis of a HuggingFace model's memory requirements.
    
    Args:
        model_id: HuggingFace model identifier
        seq_length: Sequence length for analysis
        batch_size: Batch size for analysis
        precision: Precision format
        token: Optional HuggingFace API token
        **kwargs: Additional arguments for estimate_memory
        
    Returns:
        Dictionary with model info and memory analysis
    """
    loader = HuggingFaceConfigLoader(token=token)
    return loader.analyze_model(
        model_id,
        seq_length=seq_length,
        batch_size=batch_size,
        precision=precision,
        **kwargs
    )


def compare_hf_models(
    model_ids: List[str],
    seq_length: int = 2048,
    precision: str = 'fp16',
    token: Optional[str] = None,
    print_results: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Compare memory requirements of multiple HuggingFace models.
    
    Args:
        model_ids: List of HuggingFace model identifiers
        seq_length: Sequence length for comparison
        precision: Precision format
        token: Optional HuggingFace API token
        print_results: Whether to print a comparison table
        **kwargs: Additional arguments for estimate_memory
        
    Returns:
        List of analysis results
    """
    loader = HuggingFaceConfigLoader(token=token)
    results = loader.compare_models(
        model_ids,
        seq_length=seq_length,
        precision=precision,
        **kwargs
    )
    
    if print_results:
        print(f"\nComparing models at {seq_length} token context, {precision} precision:")
        print("=" * 90)
        print(f"{'Model':<40} {'Type':<15} {'Attention':<10} {'Memory':<10} {'GPU':<15}")
        print("-" * 90)
        
        for result in results:
            if 'error' in result:
                print(f"{result['model_id']:<40} Error: {result['error'][:40]}...")
            else:
                model_name = result['model_id'].split('/')[-1][:35]
                model_type = result['model_type'][:12]
                attention = result['attention_type'] or 'n/a'
                memory = result['memory_analysis']['total_memory_gb']
                gpu = f"{result['deployment_recommendations']['min_gpu_memory_gb']}GB"
                
                print(f"{model_name:<40} {model_type:<15} {attention:<10} {memory:<10.1f} {gpu:<15}")
        
        print("=" * 90)
    
    return results

if __name__ == "__main__":
    print("HuggingFace Model Config Loader (using HfApi)")
    print("=" * 50)
    
    # Initialize loader (can also use environment variable HF_TOKEN)
    loader = HuggingFaceConfigLoader()
    
    # Example 1: Load a single model
    print("\n1. Loading model config:")
    try:
        config = loader.get_model_config("mistralai/Magistral-Small-2506")
        print(f"✓ Loaded config for mistralai/Magistral-Small-2506")
        print(f"  Model type: {config.get('model_type')}")
        print(f"  Parameters: {config.get('num_parameters', 'not found'):,}")
        
        # Estimate memory
        try:
            result = estimate_memory(config, seq_length=2048, batch_size=8)
            print(f"  Memory at 2K context: {result.total_memory_gb:.1f} GB")
            print(result)
        except Exception as mem_error:
            print(f"  Memory estimation error: {mem_error}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Analyze a model
    print("\n2. Analyzing a model:")
    try:
        analysis = loader.analyze_model("mistralai/Pixtral-12B-Base-2409", seq_length=4096)
        print(analysis)
        print(f"✓ Analysis complete for {analysis['model_id']}")
        print(f"  Type: {analysis['model_type']}")
        print(f"  Attention: {analysis['attention_type']}")
        print(f"  Memory: {analysis['memory_analysis']['total_memory_gb']:.1f} GB")
        print(f"  Downloads: {analysis['model_popularity']['downloads']:,}")
        
    except Exception as e:
        print(f"Error: {e}")
    
