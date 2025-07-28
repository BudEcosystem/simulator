"""
Model converter between HuggingFace and GenZ formats.
"""

import sys
import os
from typing import Dict, Any, Optional, List
import logging

# Add GenZ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'GenZ'))

from Models.default_models import ModelConfig
from Models.model_quality import QualityMetricsCollection, MMLU, MATH, GSM8K, Hellaswag, TriviaQA


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
        
        # Basic parameters
        vocab_size = hf_config.get('vocab_size', 32000)
        hidden_size = hf_config.get('hidden_size', 4096)
        intermediate_size = hf_config.get('intermediate_size', 11008)
        num_hidden_layers = hf_config.get('num_hidden_layers', 32)
        num_attention_heads = hf_config.get('num_attention_heads', 32)
        num_key_value_heads = hf_config.get('num_key_value_heads', num_attention_heads)
        
        # Head dimension
        head_dim = hf_config.get('head_dim')
        if head_dim is None and num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads
        
        # Maximum sequence length
        max_model_len = hf_config.get('max_position_embeddings', 2048)
        if 'max_seq_len' in hf_config:
            max_model_len = hf_config['max_seq_len']
        
        # Sliding window
        sliding_window = hf_config.get('sliding_window')
        
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
        
        # MoE parameters
        num_experts = hf_config.get('num_experts', 
                                   hf_config.get('num_local_experts', 1))
        expert_top_k = hf_config.get('num_experts_per_tok', 
                                    hf_config.get('top_k', 1))
        moe_intermediate_size = hf_config.get('moe_intermediate_size', intermediate_size)
        
        # Determine if this is an MoE model
        is_moe = num_experts > 1
        moe_layer_freq = None
        if is_moe:
            # For Mixtral-style models, every layer is MoE
            moe_layer_freq = 1
        
        # Handle encoder-decoder models
        num_encoder_layers = 0
        num_decoder_layers = num_hidden_layers
        if hf_config.get('is_encoder_decoder'):
            # Split layers between encoder and decoder
            if 'encoder_layers' in hf_config:
                num_encoder_layers = hf_config['encoder_layers']
                num_decoder_layers = hf_config.get('decoder_layers', num_hidden_layers)
            else:
                # Default split
                num_encoder_layers = num_hidden_layers // 2
                num_decoder_layers = num_hidden_layers - num_encoder_layers
        
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
            # Mamba parameters
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            # Quality metrics
            model_quality=quality_collection
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