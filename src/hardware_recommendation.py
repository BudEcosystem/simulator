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
                         model_params_b: Optional[float] = None) -> Dict[str, Any]:
        """
        Recommend hardware based on memory requirements with intelligent sorting.
        
        Enhanced Logic:
        - Sort hardware by utilization in descending order
        - Separate CPU and GPU recommendations
        - Add batch recommendations for small models on CPUs
        - Model size detection (<14B considered "small")
        
        Args:
            total_memory_gb: Total memory needed in GB (from frontend)
            model_params_b: Model parameters in billions (for CPU filtering)
            
        Returns:
            Dict with:
            - cpu_recommendations: List of CPU hardware recommendations
            - gpu_recommendations: List of GPU/accelerator recommendations
            - model_info: Information about the model and compatibility
            - total_recommendations: Total count of recommendations
        """
        all_hardware = self.hardware.get_all_hardwares()
        
        # Handle edge case of zero memory
        if total_memory_gb <= 0:
            return {
                'cpu_recommendations': [],
                'gpu_recommendations': [],
                'model_info': {
                    'is_small_model': model_params_b is not None and model_params_b < 14,
                    'cpu_compatible': False,
                    'total_memory_gb': total_memory_gb,
                    'model_params_b': model_params_b
                },
                'total_recommendations': 0
            }
        
        # Determine if this is a small model
        is_small_model = model_params_b is not None and model_params_b < 14
        cpu_compatible = is_small_model and total_memory_gb < 40
        
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
                'utilization': round(utilization, 1),
                'total_memory_available': total_available_memory
            }
            
            # Add batch recommendations for CPUs if it's a small model
            if hw['type'] == 'cpu' and is_small_model and total_memory_gb < 40:
                batch_recommendations = []
                # Recommend 2, 4, and 8 node configurations
                for num_nodes in [2, 4, 8]:
                    multi_node_memory = memory_per_chip * num_nodes
                    # Calculate how many models can fit with optimal batch sizing
                    models_per_node = int(memory_per_chip / total_memory_gb)
                    total_models = models_per_node * num_nodes
                    
                    # For small models, recommend batch size of 32 or number of models
                    recommended_batch = min(32, max(1, total_models))
                    
                    # Calculate utilization at this batch size
                    batch_utilization = min(100, (total_memory_gb * recommended_batch / multi_node_memory) * 100)
                    
                    batch_recommendations.append({
                        'nodes': num_nodes,
                        'total_memory': multi_node_memory,
                        'recommended_batch_size': recommended_batch,
                        'utilization_at_batch': round(batch_utilization, 1)
                    })
                
                hardware_rec['batch_recommendations'] = batch_recommendations
            else:
                hardware_rec['batch_recommendations'] = None
            
            # Separate CPUs and GPUs/accelerators
            if hw['type'] == 'cpu':
                cpu_recommendations.append(hardware_rec)
            else:
                gpu_recommendations.append(hardware_rec)
        
        # Sort by utilization in descending order (highest utilization first)
        cpu_recommendations.sort(key=lambda x: x['utilization'], reverse=True)
        gpu_recommendations.sort(key=lambda x: x['utilization'], reverse=True)
        
        # Add optimality indicators based on utilization
        def add_optimality(recommendations):
            for rec in recommendations:
                if rec['utilization'] >= 80:
                    rec['optimality'] = 'optimal'
                elif rec['utilization'] >= 50:
                    rec['optimality'] = 'good'
                else:
                    rec['optimality'] = 'ok'
        
        add_optimality(cpu_recommendations)
        add_optimality(gpu_recommendations)
        
        return {
            'cpu_recommendations': cpu_recommendations,
            'gpu_recommendations': gpu_recommendations,
            'model_info': {
                'is_small_model': is_small_model,
                'cpu_compatible': cpu_compatible,
                'total_memory_gb': total_memory_gb,
                'model_params_b': model_params_b
            },
            'total_recommendations': len(cpu_recommendations) + len(gpu_recommendations)
        }
    
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