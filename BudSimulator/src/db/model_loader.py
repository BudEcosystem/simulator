"""
Dynamic model loader that integrates database models with existing MODEL_DICT.
"""

import sys
import os
from typing import Dict, Any, Optional, Union, List
import logging

# Add GenZ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'GenZ'))

from Models.default_models import ModelConfig, ModelCollection
from .model_manager import ModelManager
from .model_converter import ModelConverter
from .hf_integration import HuggingFaceModelImporter


logger = logging.getLogger(__name__)


class DynamicModelCollection(ModelCollection):
    """Extended ModelCollection that loads from both static definitions and database."""
    
    def __init__(self, base_models: Optional[Dict[str, ModelConfig]] = None,
                 model_manager: Optional[ModelManager] = None,
                 auto_import: bool = True):
        """Initialize dynamic model collection.
        
        Args:
            base_models: Base models from static definitions
            model_manager: Model manager instance
            auto_import: If True, automatically import from HF when model not found
        """
        super().__init__(base_models)
        self.model_manager = model_manager or ModelManager()
        self.converter = ModelConverter()
        self.hf_importer = HuggingFaceModelImporter(self.model_manager)
        self.auto_import = auto_import
        self._db_models_cache = {}
        
    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get a model by name, checking database if not in static definitions.
        
        Args:
            model_name: Model name or ID
            
        Returns:
            ModelConfig instance or None
        """
        # First check static models
        model = super().get_model(model_name)
        if model:
            return model
        
        # Check cache
        if model_name in self._db_models_cache:
            return self._db_models_cache[model_name]
        
        # Try to load from database
        model = self._load_from_database(model_name)
        if model:
            self._db_models_cache[model_name] = model
            return model
        
        # If auto_import is enabled and model looks like HF ID, try to import
        if self.auto_import and '/' in model_name:
            logger.info(f"Model {model_name} not found, attempting to import from HuggingFace")
            if self.hf_importer.import_model(model_name):
                # Try loading again
                model = self._load_from_database(model_name)
                if model:
                    self._db_models_cache[model_name] = model
                    return model
        
        return None
    
    def _load_from_database(self, model_name: str) -> Optional[ModelConfig]:
        """Load a model from the database.
        
        Args:
            model_name: Model name or ID
            
        Returns:
            ModelConfig instance or None
        """
        try:
            # Get model from database
            db_model = self.model_manager.get_model(model_name)
            if not db_model:
                # Try alternative names
                if '/' not in model_name:
                    # Try with common prefixes
                    for prefix in ['mistralai/', 'meta-llama/', 'google/', 'microsoft/']:
                        full_id = prefix + model_name
                        db_model = self.model_manager.get_model(full_id)
                        if db_model:
                            break
                
                if not db_model:
                    return None
            
            # Get configuration
            config = self.model_manager.get_model_config(db_model['model_id'])
            if not config:
                return None
            
            # Get quality metrics
            metrics = self.model_manager.get_quality_metrics(db_model['model_id'])
            
            # Convert to GenZ format
            genz_config = self.converter.hf_to_genz(config, metrics)
            
            return genz_config
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from database: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List all available models (static + database).
        
        Returns:
            List of model names/IDs
        """
        # Get static models
        static_models = super().list_models()
        
        # Get database models
        db_models = self.model_manager.list_models()
        db_model_ids = [m['model_id'] for m in db_models]
        
        # Combine and deduplicate
        all_models = list(set(static_models + db_model_ids))
        all_models.sort()
        
        return all_models
    
    def refresh_cache(self):
        """Clear the database models cache."""
        self._db_models_cache.clear()
    
    def sync_static_models_to_db(self):
        """Sync all static models to the database."""
        synced = 0
        for model_name, model_config in self.models.items():
            try:
                # Convert to HF format
                hf_config = self.converter.genz_to_hf(model_config)
                
                # Add to database
                existing = self.model_manager.get_model(model_config.model)
                if not existing:
                    self.model_manager.add_model(
                        model_id=model_config.model,
                        config=hf_config,
                        source="manual",
                        model_type=hf_config.get('model_type'),
                        attention_type=self._detect_attention_type(model_config),
                        parameter_count=self._estimate_parameters(model_config)
                    )
                    
                    # Add quality metrics if available
                    if model_config.model_quality:
                        for metric in model_config.model_quality.metrics:
                            metric_name = metric.__class__.__name__
                            self.model_manager.add_quality_metric(
                                model_id=model_config.model,
                                metric_name=metric_name,
                                metric_value=metric.accuracy,
                                shots=getattr(metric, 'shots', None)
                            )
                    
                    synced += 1
                    logger.info(f"Synced model {model_config.model} to database")
                    
            except Exception as e:
                logger.error(f"Failed to sync model {model_name}: {e}")
        
        return synced
    
    def _detect_attention_type(self, config: ModelConfig) -> str:
        """Detect attention type from GenZ config."""
        if config.num_key_value_heads == 1:
            return "mqa"
        elif config.num_key_value_heads < config.num_attention_heads:
            return "gqa"
        else:
            return "mha"
    
    def _estimate_parameters(self, config: ModelConfig) -> int:
        """Accurate parameter count from GenZ config using unified calculation."""
        try:
            # Convert GenZ config to HF format for unified calculation
            hf_config = self.converter.genz_to_hf(config)
            
            # Use the same accurate calculation as ModelMemoryCalculator
            from src.bud_models import ModelMemoryCalculator
            calculator = ModelMemoryCalculator()
            
            # Check if we have direct parameter count
            if hasattr(config, 'num_parameters') and config.num_parameters:
                return int(config.num_parameters)
            
            # Use the sophisticated calculation method
            param_count = calculator._calculate_transformer_params(hf_config)
            return int(param_count)
            
        except Exception as e:
            logger.warning(f"Failed to use unified calculation for {config.model}, falling back to simplified: {e}")
            # Fallback to improved simplified calculation
            return self._fallback_parameter_estimation(config)
    
    def _fallback_parameter_estimation(self, config: ModelConfig) -> int:
        """Improved fallback parameter estimation for GenZ config."""
        # Embeddings (with tie_word_embeddings consideration)
        vocab_params = config.vocab_size * config.hidden_size
        if not getattr(config, 'tie_word_embeddings', True):
            vocab_params *= 2  # Input and output embeddings
        
        # Calculate KV dimensions based on attention type
        attention_type = self._detect_attention_type(config)
        if attention_type == "mqa":
            # Multi-Query: single KV head
            head_dim = config.hidden_size // max(1, config.num_attention_heads)
            kv_dim = head_dim
        elif attention_type == "gqa":
            # Grouped-Query: fewer KV heads
            head_dim = config.hidden_size // max(1, config.num_attention_heads) 
            kv_dim = head_dim * config.num_key_value_heads
        else:
            # Multi-Head: full dimension
            kv_dim = config.hidden_size
        
        # Attention parameters with proper KV dimensions
        # Q: hidden_size -> hidden_size
        # K, V: hidden_size -> kv_dim  
        # O: hidden_size -> hidden_size
        attention_params = (
            config.hidden_size * config.hidden_size +  # Q projection
            config.hidden_size * kv_dim +              # K projection
            config.hidden_size * kv_dim +              # V projection  
            config.hidden_size * config.hidden_size   # O projection
        )
        
        # FFN parameters with activation function multiplier
        ffn_mult = 3 if config.num_ffi == 2 else 2  # SwiGLU vs GELU
        ffn_params = ffn_mult * config.hidden_size * config.intermediate_size
        
        # Layer norm parameters (2 per layer typically)
        ln_params = 4 * config.hidden_size
        
        # Per-layer parameters
        layer_params = attention_params + ffn_params + ln_params
        total_layers = config.num_encoder_layers + config.num_decoder_layers
        
        # MoE scaling (multiply FFN by number of experts)
        if config.num_experts > 1:
            expert_ffn_params = ffn_params * config.num_experts
            layer_params = attention_params + expert_ffn_params + ln_params
        
        # Total parameters
        total_params = vocab_params + (total_layers * layer_params) + config.hidden_size  # Final layer norm
        
        return int(total_params)

    def _detect_model_type(self, config: ModelConfig) -> str:
        """Detect model type from GenZ config."""
        if config.num_experts > 1:
            return "moe"
        elif config.num_encoder_layers > 0 and config.num_decoder_layers > 0:
            return "encoder-decoder"
        elif config.num_encoder_layers > 0:
            return "encoder-only"
        else:
            return "decoder-only"

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get model metadata for API responses."""
        model_config = self.get_model(model_id)
        if not model_config:
            return {}
        
        attention_type = self._detect_attention_type(model_config)
        model_type = self._detect_model_type(model_config)
        parameter_count = self._estimate_parameters(model_config)
        
        metadata = {
            'model_id': model_config.model,
            'model_name': model_config.model.split('/')[-1] if '/' in model_config.model else model_config.model,
            'parameter_count': parameter_count,
            'attention_type': attention_type,
            'model_type': model_type,
            'source': 'genz',
        }
        
        return metadata


def create_dynamic_model_dict(base_model_dict: Optional[ModelCollection] = None,
                             auto_import: bool = True) -> DynamicModelCollection:
    """Create a dynamic MODEL_DICT that includes database models.
    
    Args:
        base_model_dict: Base MODEL_DICT to extend
        auto_import: If True, automatically import from HF when model not found
        
    Returns:
        DynamicModelCollection instance
    """
    base_models = base_model_dict.models if base_model_dict else {}
    return DynamicModelCollection(base_models, auto_import=auto_import)


# Monkey patch function to replace MODEL_DICT in GenZ
def patch_genz_model_dict():
    """Replace the static MODEL_DICT in GenZ with dynamic version."""
    try:
        from Models import default_models
        
        # Get the current MODEL_DICT
        current_dict = getattr(default_models, 'MODEL_DICT', None)
        
        if current_dict:
            # Create dynamic version
            dynamic_dict = create_dynamic_model_dict(current_dict)
            
            # Replace it
            default_models.MODEL_DICT = dynamic_dict
            
            logger.info("Successfully patched GenZ MODEL_DICT with dynamic version")
            return True
            
    except Exception as e:
        logger.error(f"Failed to patch GenZ MODEL_DICT: {e}")
        return False 