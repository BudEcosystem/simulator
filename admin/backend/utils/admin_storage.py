"""
Admin-specific storage for managing models, hardware, and usecases
when the main API doesn't support full CRUD operations.
"""
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

class AdminStorage:
    """Simple file-based storage for admin operations"""
    
    def __init__(self, storage_dir: str = "./admin_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize storage files
        self.models_file = os.path.join(storage_dir, "models.json")
        self.hardware_file = os.path.join(storage_dir, "hardware.json")
        self.usecases_file = os.path.join(storage_dir, "usecases.json")
        
        # Load existing data or initialize empty
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage files if they don't exist"""
        for file_path in [self.models_file, self.hardware_file, self.usecases_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({}, f)
    
    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _write_file(self, file_path: str, data: Dict[str, Any]):
        """Write JSON data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Model operations
    def get_model_overrides(self) -> Dict[str, Any]:
        """Get all model overrides"""
        return self._read_file(self.models_file)
    
    def get_model_override(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get specific model override"""
        overrides = self.get_model_overrides()
        return overrides.get(model_id)
    
    def save_model_override(self, model_id: str, data: Dict[str, Any]):
        """Save model override (for updates)"""
        overrides = self.get_model_overrides()
        overrides[model_id] = {
            **data,
            "model_id": model_id,
            "updated_at": datetime.utcnow().isoformat(),
            "is_override": True
        }
        self._write_file(self.models_file, overrides)
    
    def delete_model_override(self, model_id: str):
        """Delete model override"""
        overrides = self.get_model_overrides()
        if model_id in overrides:
            del overrides[model_id]
            self._write_file(self.models_file, overrides)
    
    # Hardware operations
    def get_hardware_overrides(self) -> Dict[str, Any]:
        """Get all hardware overrides"""
        return self._read_file(self.hardware_file)
    
    def get_hardware_override(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific hardware override"""
        overrides = self.get_hardware_overrides()
        return overrides.get(name)
    
    def save_hardware_override(self, name: str, data: Dict[str, Any]):
        """Save hardware override"""
        overrides = self.get_hardware_overrides()
        overrides[name] = {
            **data,
            "name": name,
            "updated_at": datetime.utcnow().isoformat(),
            "is_override": True
        }
        self._write_file(self.hardware_file, overrides)
    
    def delete_hardware_override(self, name: str):
        """Delete hardware override"""
        overrides = self.get_hardware_overrides()
        if name in overrides:
            del overrides[name]
            self._write_file(self.hardware_file, overrides)
    
    # Usecase operations
    def get_usecase_overrides(self) -> Dict[str, Any]:
        """Get all usecase overrides"""
        return self._read_file(self.usecases_file)
    
    def get_usecase_override(self, unique_id: str) -> Optional[Dict[str, Any]]:
        """Get specific usecase override"""
        overrides = self.get_usecase_overrides()
        return overrides.get(unique_id)
    
    def save_usecase_override(self, unique_id: str, data: Dict[str, Any]):
        """Save usecase override"""
        overrides = self.get_usecase_overrides()
        overrides[unique_id] = {
            **data,
            "unique_id": unique_id,
            "updated_at": datetime.utcnow().isoformat(),
            "is_override": True
        }
        self._write_file(self.usecases_file, overrides)
    
    def delete_usecase_override(self, unique_id: str):
        """Delete usecase override"""
        overrides = self.get_usecase_overrides()
        if unique_id in overrides:
            del overrides[unique_id]
            self._write_file(self.usecases_file, overrides)

# Global instance
admin_storage = AdminStorage()