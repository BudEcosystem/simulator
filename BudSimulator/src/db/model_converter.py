"""
Model converter between HuggingFace and GenZ formats.
"""

import sys
import os
from typing import Dict, Any, Optional, List
import logging

from llm_memory_calculator.genz.Models.default_models import ModelConfig
from llm_memory_calculator.genz.Models.get_language_model import (
    encoder_decoder_fields,
    graph_layer_kinds,
    resolve_decoder_config,
)
from llm_memory_calculator.layer_plan import (
    attention_head_dim,
    attention_output_gate,
    decoder_streams,
    depth,
    ffn_width,
    learned_positions,
    routed_expert_count,
    routed_expert_width,
    shared_expert_width,
    vocab_size as vocab_size_of,
)
from llm_memory_calculator.genz.Models.model_quality import QualityMetricsCollection, MMLU, MATH, GSM8K, Hellaswag, TriviaQA


logger = logging.getLogger(__name__)


class ModelConverter:
    """Convert between HuggingFace and GenZ model formats."""
    
    # Mapping of HuggingFace model types to GenZ FFN implementations
    FFN_IMPLEMENTATIONS = {
        'mistral': 'default',
        'llama': 'default',
        'mixtral': 'default',
        'gpt2': 'default',
        'phi': 'default',
        'qwen': 'default',
    }
    
    # Mapping of activation functions
    ACTIVATION_MAPPING = {
        'silu': 'silu',
        'swiglu': 'silu',
        'gelu': 'gelu',
        'relu': 'relu',
        'gelu_new': 'gelu',
    }
    
    def hf_to_genz(self, hf_config: Dict[str, Any], 
                   quality_metrics: Optional[List[Dict[str, Any]]] = None) -> ModelConfig:
        """Convert HuggingFace config to GenZ ModelConfig.
        
        Args:
            hf_config: HuggingFace configuration dictionary
            quality_metrics: Optional list of quality metrics
            
        Returns:
            GenZ ModelConfig instance
        """
        # Extract model ID
        model_id = hf_config.get('_source', '').replace('huggingface:', '')
        if not model_id:
            model_id = hf_config.get('model_id', 'unknown')
        
        # An encoder-decoder's encoder depth, cross-attention cache and encoder stack, from the
        # config as stated (each stack's own key or block: T5's `num_layers` is the encoder's,
        # Dia nests each stack, BLIP-2 nests a whole flan-T5 under `text_config`) -- the HF
        # loader's own reader.
        enc_dec = encoder_decoder_fields(hf_config)

        # A multimodal config nests the decoder (`text_config`, ...), an encoder-decoder states it
        # under its own spellings (`decoder_layers`, T5's `num_heads` / `d_model`, Dia's
        # `decoder_config`): read its geometry, as the HF loader does. A config whose decoder
        # cannot be found is read as it stands.
        # an audio decoder's codebooks: as many embedding tables and output heads (Dia, MusicGen)
        streams = decoder_streams(hf_config)
        try:
            hf_config = resolve_decoder_config(hf_config)
        except ValueError:
            pass

        # Basic parameters
        vocab_size = vocab_size_of(hf_config, 32000) * streams
        hidden_size = hf_config.get('hidden_size', 4096)
        intermediate_size = ffn_width(hf_config, 11008)
        # The depth the layer-kind table reads: `num_hidden_layers`, or the length of a
        # per-layer list when that is all the config states (transformers 5.12's Nemotron-H)
        num_hidden_layers = depth(hf_config) or 32
        num_attention_heads = hf_config.get('num_attention_heads', 32)
        num_key_value_heads = hf_config.get('num_key_value_heads', num_attention_heads)
        
        # Head dimension: the one reader the calculator and GenZ use (`head_dim` or its other
        # spellings, e.g. Zamba2's `attention_head_dim`, before hidden_size / heads).
        head_dim = attention_head_dim(hf_config, num_attention_heads, hidden_size)
        if head_dim is None and num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads
        
        # Maximum sequence length
        max_model_len = hf_config.get('max_position_embeddings', 2048)
        if 'max_seq_len' in hf_config:
            max_model_len = hf_config['max_seq_len']
        
        # Activation function
        hidden_act = hf_config.get('activation_function', 
                                  hf_config.get('hidden_act', 'silu'))
        hidden_act = self.ACTIVATION_MAPPING.get(hidden_act.lower(), hidden_act)
        
        # FFN implementation
        model_type = hf_config.get('model_type', '').lower()
        ffn_implementation = self.FFN_IMPLEMENTATIONS.get(model_type, 'default')
        
        # Number of FFI (feed forward intermediate)
        # For models with gated activations (SwiGLU), we have 3 matrices
        num_ffi = 2  # Default
        if 'swiglu' in hidden_act.lower() or 'silu' in hidden_act.lower():
            num_ffi = 2  # GenZ uses 2 for gated activations
        
        # MoE parameters, through the readers the HF loader and the calculator use: the routed
        # experts under any spelling (DeepSeek's `n_routed_experts` was read as a dense model),
        # one expert's width, and the shared expert(s) beside them
        num_experts = routed_expert_count(hf_config) or 1
        expert_top_k = hf_config.get('num_experts_per_tok', 
                                    hf_config.get('top_k', 1))
        moe_intermediate_size = (routed_expert_width(hf_config) or intermediate_size) \
            if num_experts > 1 else intermediate_size
        shared_width = shared_expert_width(hf_config) if num_experts > 1 else 0
        if shared_width:
            ffn_implementation = 'deepseek'
        
        # Determine if this is an MoE model
        is_moe = num_experts > 1
        moe_layer_freq = None
        if is_moe:
            # For Mixtral-style models, every layer is MoE
            moe_layer_freq = 1
        
        # Encoder-decoder models: the depth read above is already the decoder's. The encoder's is
        # its own stack's, never a split of the decoder's (which halved Dia's and T5's decoder).
        num_encoder_layers = enc_dec["num_encoder_layers"]
        num_decoder_layers = num_hidden_layers
        
        # Mamba parameters (for SSM models)
        mamba_d_state = hf_config.get('state_size', hf_config.get('d_state'))
        mamba_d_conv = hf_config.get('conv_kernel', hf_config.get('d_conv'))
        mamba_expand = hf_config.get('expand', hf_config.get('mamba_expand'))
        
        # Quality metrics
        quality_collection = None
        if quality_metrics:
            metrics = []
            for metric in quality_metrics:
                metric_name = metric.get('metric_name', '').upper()
                metric_value = metric.get('metric_value', 0.0)
                shots = metric.get('shots', 0)
                
                # Map to GenZ quality metric classes
                if metric_name == 'MMLU':
                    metrics.append(MMLU(accuracy=metric_value, shots=shots))
                elif metric_name == 'MATH':
                    metrics.append(MATH(accuracy=metric_value, shots=shots))
                elif metric_name == 'GSM8K':
                    metrics.append(GSM8K(accuracy=metric_value, shots=shots))
                elif metric_name == 'HELLASWAG':
                    metrics.append(Hellaswag(accuracy=metric_value, shots=shots))
                elif metric_name == 'TRIVIAQA':
                    metrics.append(TriviaQA(accuracy=metric_value, shots=shots))
            
            if metrics:
                quality_collection = QualityMetricsCollection(metrics)
        
        # Per-layer kinds from the config's own keys (hybrids, interleaved windows,
        # KV-free layers), the windowed layers' window under any spelling, and whether GenZ's
        # all-Mamba stack applies -- through the same reader the HF loader uses, so a model
        # synced from the database is graphed exactly like one loaded from a config.
        plan = graph_layer_kinds(hf_config, num_decoder_layers, num_experts, hidden_size,
                                 mamba_d_state)
        layer_configs = plan["layer_configs"]
        sliding_window = plan["sliding_window"]
        mamba_d_state = plan["mamba_d_state"]
        # The state of an all-recurrent stack's layer, from the config's own mechanism (a
        # Mamba-2 conv buffer carries its B/C groups; without it GenZ falls back to Mamba-1's).
        recurrent_state = plan["recurrent_state"]

        # Create GenZ ModelConfig
        config = ModelConfig(
            model=model_id,
            vocab_size=vocab_size,
            max_model_len=max_model_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_ffi=num_ffi,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            sliding_window=sliding_window,
            ffn_implementation=ffn_implementation,
            # MoE parameters
            moe_layer_freq=moe_layer_freq,
            num_experts=num_experts,
            expert_top_k=expert_top_k,
            moe_intermediate_size=moe_intermediate_size,
            n_shared_experts=1 if shared_width else 0,
            shared_expert_intermediate_size=shared_width or None,
            # Mamba parameters
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            # MLA: one latent of kv_lora_rank + qk_rope_head_dim per layer, not K and V per head
            kv_lora_rank=hf_config.get('kv_lora_rank'),
            q_lora_rank=hf_config.get('q_lora_rank'),
            qk_rope_head_dim=hf_config.get('qk_rope_head_dim'),
            qk_nope_head_dim=hf_config.get('qk_nope_head_dim'),
            v_head_dim=hf_config.get('v_head_dim'),
            # DeepSeek Sparse Attention's indexer, and a query-width attention output gate
            index_n_heads=hf_config.get('index_n_heads') if hf_config.get('kv_lora_rank') else None,
            index_head_dim=hf_config.get('index_head_dim') if hf_config.get('kv_lora_rank') else None,
            attention_output_gate=attention_output_gate(hf_config),
            position_embeddings=learned_positions(hf_config),
            # Quality metrics
            model_quality=quality_collection,
            layer_configs=layer_configs,
            recurrent_state=recurrent_state,
            # An encoder-decoder's cross-attention cache and encoder stack
            cross_attention=enc_dec["cross_attention"],
            encoder=enc_dec["encoder"],
        )
        
        return config
    
    def genz_to_hf(self, genz_config: ModelConfig) -> Dict[str, Any]:
        """Convert GenZ ModelConfig to HuggingFace format.
        
        Args:
            genz_config: GenZ ModelConfig instance
            
        Returns:
            HuggingFace configuration dictionary
        """
        # Basic configuration
        hf_config = {
            'model_id': genz_config.model,
            'vocab_size': genz_config.vocab_size,
            'hidden_size': genz_config.hidden_size,
            'intermediate_size': genz_config.intermediate_size,
            'num_hidden_layers': genz_config.num_decoder_layers,
            'num_attention_heads': genz_config.num_attention_heads,
            'num_key_value_heads': genz_config.num_key_value_heads,
            'max_position_embeddings': genz_config.max_model_len,
            'hidden_act': genz_config.hidden_act,
            'head_dim': genz_config.head_dim,
        }
        
        # Add sliding window if present
        if genz_config.sliding_window:
            hf_config['sliding_window'] = genz_config.sliding_window
        
        # Add encoder-decoder info if applicable
        if genz_config.num_encoder_layers > 0:
            hf_config['is_encoder_decoder'] = True
            hf_config['encoder_layers'] = genz_config.num_encoder_layers
            hf_config['decoder_layers'] = genz_config.num_decoder_layers
        
        # Add MoE parameters if applicable
        if genz_config.num_experts > 1:
            hf_config['num_experts'] = genz_config.num_experts
            hf_config['num_local_experts'] = genz_config.num_experts
            hf_config['num_experts_per_tok'] = genz_config.expert_top_k
            hf_config['top_k'] = genz_config.expert_top_k
            if genz_config.moe_intermediate_size != genz_config.intermediate_size:
                hf_config['moe_intermediate_size'] = genz_config.moe_intermediate_size
        
        # Add Mamba parameters if applicable
        if genz_config.mamba_d_state:
            hf_config['state_size'] = genz_config.mamba_d_state
            hf_config['d_state'] = genz_config.mamba_d_state
        if genz_config.mamba_d_conv:
            hf_config['conv_kernel'] = genz_config.mamba_d_conv
            hf_config['d_conv'] = genz_config.mamba_d_conv
        if genz_config.mamba_expand and genz_config.mamba_expand != 1:
            hf_config['expand'] = genz_config.mamba_expand
            hf_config['mamba_expand'] = genz_config.mamba_expand
        
        # Infer model type
        if genz_config.mamba_d_state:
            hf_config['model_type'] = 'mamba'
        elif genz_config.num_experts > 1:
            if 'mixtral' in genz_config.model.lower():
                hf_config['model_type'] = 'mixtral'
            else:
                hf_config['model_type'] = 'moe'
        elif 'mistral' in genz_config.model.lower():
            hf_config['model_type'] = 'mistral'
        elif 'llama' in genz_config.model.lower():
            hf_config['model_type'] = 'llama'
        elif 'gpt' in genz_config.model.lower():
            hf_config['model_type'] = 'gpt2'
        else:
            hf_config['model_type'] = 'unknown'
        
        return hf_config
    
    def validate_conversion(self, original: Dict[str, Any], 
                          converted: Any, direction: str = "hf_to_genz") -> Dict[str, Any]:
        """Validate that conversion preserved essential parameters.
        
        Args:
            original: Original configuration
            converted: Converted configuration
            direction: Conversion direction ("hf_to_genz" or "genz_to_hf")
            
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if direction == "hf_to_genz":
            # Validate HF to GenZ conversion
            if isinstance(converted, ModelConfig):
                # Check essential parameters
                if original.get('vocab_size') != converted.vocab_size:
                    validation['warnings'].append(
                        f"vocab_size mismatch: {original.get('vocab_size')} vs {converted.vocab_size}"
                    )
                
                if original.get('hidden_size') != converted.hidden_size:
                    validation['errors'].append(
                        f"hidden_size mismatch: {original.get('hidden_size')} vs {converted.hidden_size}"
                    )
                    validation['valid'] = False
                
                # Check attention heads
                orig_heads = original.get('num_attention_heads', 0)
                if orig_heads != converted.num_attention_heads:
                    validation['warnings'].append(
                        f"num_attention_heads mismatch: {orig_heads} vs {converted.num_attention_heads}"
                    )
            else:
                validation['errors'].append("Converted object is not a ModelConfig")
                validation['valid'] = False
        
        else:
            # Validate GenZ to HF conversion
            if isinstance(original, ModelConfig) and isinstance(converted, dict):
                # Check essential parameters
                if original.vocab_size != converted.get('vocab_size'):
                    validation['warnings'].append(
                        f"vocab_size mismatch: {original.vocab_size} vs {converted.get('vocab_size')}"
                    )
                
                if original.hidden_size != converted.get('hidden_size'):
                    validation['errors'].append(
                        f"hidden_size mismatch: {original.hidden_size} vs {converted.get('hidden_size')}"
                    )
                    validation['valid'] = False
            else:
                validation['errors'].append("Invalid types for validation")
                validation['valid'] = False
        
        return validation 