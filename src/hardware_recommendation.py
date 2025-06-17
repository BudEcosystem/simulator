"""Hardware recommendation module for BudSimulator."""
import math
from typing import List, Dict, Optional, Any
from .hardware import BudHardware


class HardwareRecommendation:
    """Recommend hardware based on model memory requirements."""
    
    def __init__(self):
        """Initialize with BudHardware instance."""
        self.hardware = BudHardware()
    
    def recommend_hardware(self, 
                         total_memory_gb: float, 
                         model_params_b: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Recommend hardware based on memory requirements.
        
        Args:
            total_memory_gb: Total memory needed in GB (from frontend)
            model_params_b: Model parameters in billions (for CPU filtering)
            
        Returns:
            List of dicts with:
            - hardware_name: Name of the hardware
            - nodes_required: Number of nodes needed
            - memory_per_chip: Memory per chip in GB
            - manufacturer: Intel, NVIDIA, AMD, etc.
            - type: cpu, gpu, accelerator, asic
        """
        recommendations = []
        all_hardware = self.hardware.get_all_hardwares()
        
        for hw in all_hardware:
            # Skip CPUs for models > 13B
            if hw['type'] == 'cpu' and model_params_b and model_params_b > 13:
                continue
            
            # Get memory per chip (already in GB)
            memory_per_chip = hw.get('Memory_size', 0)
            
            # Skip hardware with zero or invalid memory
            if memory_per_chip <= 0:
                continue
            
            # Calculate nodes required
            nodes_required = math.ceil(total_memory_gb / memory_per_chip)
            
            recommendations.append({
                'hardware_name': hw['name'],
                'nodes_required': nodes_required,
                'memory_per_chip': memory_per_chip,
                'manufacturer': hw.get('manufacturer', 'Unknown'),
                'type': hw.get('type', 'Unknown')
            })
        
        # Sort by nodes required (ascending)
        recommendations.sort(key=lambda x: x['nodes_required'])
        
        return recommendations
    
    def filter_hardware(self,
                       type: Optional[str] = None,
                       manufacturer: Optional[str] = None,
                       min_memory: Optional[float] = None,
                       max_memory: Optional[float] = None,
                       min_flops: Optional[float] = None,
                       max_flops: Optional[float] = None,
                       sort_by: str = 'name',
                       sort_order: str = 'asc') -> List[Dict[str, Any]]:
        """
        Filter and sort hardware based on criteria.
        
        Args:
            type: Hardware type filter (cpu, gpu, accelerator, asic)
            manufacturer: Manufacturer filter
            min_memory: Minimum memory in GB
            max_memory: Maximum memory in GB
            min_flops: Minimum FLOPS
            max_flops: Maximum FLOPS
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Filtered and sorted list of hardware
        """
        # Use BudHardware's search functionality
        return self.hardware.search_hardware(
            type=type,
            manufacturer=manufacturer,
            min_memory=min_memory,
            max_memory=max_memory,
            min_flops=min_flops,
            max_flops=max_flops,
            sort_by=sort_by,
            sort_order=sort_order
        ) 