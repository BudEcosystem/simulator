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
        Recommend hardware based on memory requirements with intelligent sorting.
        
        New Logic:
        - CPUs first if compatible (model < 14B params OR total memory < 35GB)
        - Sort by memory size descending (24GB, 40GB, 80GB)
        - Add optimality indicators for visual coding
        
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
            - optimality: 'optimal', 'good', 'ok' for visual indicators
            - utilization: Memory utilization percentage
        """
        recommendations = []
        all_hardware = self.hardware.get_all_hardwares()
        
        # Determine CPU compatibility: CPUs suitable for smaller models AND memory requirements
        cpu_compatible = (
            (model_params_b is None or model_params_b < 14) and  # Model size check
            total_memory_gb < 35  # Memory requirement check
        )
        
        cpu_recommendations = []
        gpu_recommendations = []
        
        for hw in all_hardware:
            # Get memory per chip (already in GB)
            memory_per_chip = hw.get('Memory_size', 0)
            
            # Skip hardware with zero or invalid memory
            if memory_per_chip <= 0:
                continue
            
            # Calculate nodes required
            nodes_required = math.ceil(total_memory_gb / memory_per_chip)
            
            # Calculate utilization percentage
            total_available_memory = memory_per_chip * nodes_required  
            utilization = (total_memory_gb / total_available_memory) * 100
            
            hardware_rec = {
                'hardware_name': hw['name'],
                'nodes_required': nodes_required,
                'memory_per_chip': memory_per_chip,
                'manufacturer': hw.get('manufacturer', 'Unknown'),
                'type': hw.get('type', 'Unknown'),
                'utilization': round(utilization, 1)
            }
            
            # Separate CPUs and GPUs/accelerators
            if hw['type'] == 'cpu':
                if cpu_compatible:  # Only include CPUs if compatible
                    cpu_recommendations.append(hardware_rec)
            else:
                gpu_recommendations.append(hardware_rec)
        
        # Sort CPUs by memory size descending
        cpu_recommendations.sort(key=lambda x: x['memory_per_chip'], reverse=True)
        
        # Sort GPUs/accelerators by memory size descending  
        gpu_recommendations.sort(key=lambda x: x['memory_per_chip'], reverse=True)
        
        # Combine: CPUs first (if compatible), then GPUs/accelerators
        if cpu_compatible and cpu_recommendations:
            recommendations = cpu_recommendations + gpu_recommendations
        else:
            recommendations = gpu_recommendations
        
        # Add optimality indicators
        for i, rec in enumerate(recommendations):
            if i < 2:
                rec['optimality'] = 'optimal'    # First 2: Green
            elif i < 5:
                rec['optimality'] = 'good'       # Next 3: Yellow  
            else:
                rec['optimality'] = 'ok'         # Rest: Orange
        
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