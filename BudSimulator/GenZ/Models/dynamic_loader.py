"""
Dynamic model loader integration for GenZ.
This module patches the MODEL_DICT to support loading models from the database.
"""

import sys
import os
import logging

# Add src to path for database modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

try:
    from db.model_loader import create_dynamic_model_dict, DynamicModelCollection
    from db.model_manager import ModelManager
    from db.hf_integration import HuggingFaceModelImporter
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    DynamicModelCollection = None

from .default_models import MODEL_DICT as ORIGINAL_MODEL_DICT


logger = logging.getLogger(__name__)


# Global variable to store the dynamic MODEL_DICT
_DYNAMIC_MODEL_DICT = None


def get_dynamic_model_dict(auto_import: bool = True):
    """Get or create the dynamic MODEL_DICT.
    
    Args:
        auto_import: If True, automatically import from HF when model not found
        
    Returns:
        DynamicModelCollection or original MODEL_DICT if DB not available
    """
    global _DYNAMIC_MODEL_DICT
    
    if not DB_AVAILABLE:
        logger.warning("Database modules not available, using static MODEL_DICT")
        return ORIGINAL_MODEL_DICT
    
    if _DYNAMIC_MODEL_DICT is None:
        _DYNAMIC_MODEL_DICT = create_dynamic_model_dict(
            ORIGINAL_MODEL_DICT, 
            auto_import=auto_import
        )
        logger.info("Created dynamic MODEL_DICT with database support")
    
    return _DYNAMIC_MODEL_DICT


def import_model_from_hf(model_id: str, force_update: bool = False) -> bool:
    """Import a model from HuggingFace into the database.
    
    Args:
        model_id: HuggingFace model identifier
        force_update: If True, update existing model
        
    Returns:
        True if successful, False otherwise
    """
    if not DB_AVAILABLE:
        logger.error("Database modules not available")
        return False
    
    try:
        manager = ModelManager()
        importer = HuggingFaceModelImporter(manager)
        return importer.import_model(model_id, force_update)
    except Exception as e:
        logger.error(f"Failed to import model {model_id}: {e}")
        return False


def sync_static_models_to_db() -> int:
    """Sync all static models to the database.
    
    Returns:
        Number of models synced
    """
    if not DB_AVAILABLE:
        logger.error("Database modules not available")
        return 0
    
    dynamic_dict = get_dynamic_model_dict()
    if isinstance(dynamic_dict, DynamicModelCollection):
        return dynamic_dict.sync_static_models_to_db()
    
    return 0


# Replace the MODEL_DICT in default_models with dynamic version
def enable_dynamic_models(auto_import: bool = True):
    """Enable dynamic model loading by replacing MODEL_DICT.
    
    Args:
        auto_import: If True, automatically import from HF when model not found
    """
    if DB_AVAILABLE:
        try:
            # Try different import paths
            import Models.default_models as default_models
        except ImportError:
            from . import default_models
        
        default_models.MODEL_DICT = get_dynamic_model_dict(auto_import)
        logger.info("Enabled dynamic model loading")
    else:
        logger.warning("Cannot enable dynamic models - database modules not available")


# Auto-enable on import if database is available
if DB_AVAILABLE:
    enable_dynamic_models(auto_import=True) 