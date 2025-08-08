"""
Hardware optimization algorithm using GenZ for performance modeling
Finds the most cost-effective hardware configurations that meet SLO requirements
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# GenZ imports
try:
    from llm_memory_calculator import (
        estimate_prefill_performance,
        estimate_decode_performance,
        get_best_parallelization_strategy,
        get_minimum_system_size,
        get_hardware_config
    )
except ImportError:
    # For testing without GenZ installed
    estimate_prefill_performance = None
    estimate_decode_performance = None
    get_best_parallelization_strategy = None
    get_minimum_system_size = None
    get_hardware_config = None

# Local imports
try:
    from ..db.connection import DatabaseConnection
except ImportError:
    DatabaseConnection = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class OptimizationResult:
    """Result of hardware optimization for a configuration"""
    model_id: str
    model_size: str
    hardware_type: str
    num_nodes: int
    parallelism: str
    batch_size: int
    achieved_ttft: float
    achieved_e2e: float
    required_ttft: float
    required_e2e: float
    meets_slo: bool
    cost_per_hour: float
    throughput: float
    utilization: float
    cost_per_request: Optional[float] = None
    efficiency_score: Optional[float] = None


class HardwareOptimizer:
    """Main optimizer class for hardware selection"""
    
    def __init__(self):
        """Initialize the optimizer with model registry"""
        self.model_registry = {
            "1B": "facebook/OPT-1.3b",  # GenZ has this model
            "3B": "meta-llama/Llama-3.2-3B",  # GenZ has this model
            "8B": "meta-llama/Llama-2-7B",  # GenZ has this model
            "32B": "meta-llama/Llama-2-13B",  # GenZ has this model
            "70B": "meta-llama/Llama-3.1-70B",  # GenZ has this model
            "100B+": "meta-llama/meta-llama-3.1-405b"  # GenZ has this model
        }
        
        # Hardware cost registry (hourly rates in USD)
        self.hardware_costs = {
            "A100_40GB": 2.5,
            "A100_80GB": 3.0,
            "H100_80GB": 4.0,
            "GH200": 5.0,
            "B100": 6.0,
            "GB200": 7.0,
            "TPUv4": 3.5,
            "TPUv5": 5.0,
            "MI300X": 3.5,
            "Gaudi3": 2.8
        }
    
    def get_representative_model(self, size_category: str) -> str:
        """Get representative model for a size category"""
        if size_category not in self.model_registry:
            raise ValueError(f"Unknown model size category: {size_category}")
        return self.model_registry[size_category]
    
    def get_genz_model_name(self, model_id: str) -> str:
        """Convert HuggingFace model ID to GenZ model name"""
        # GenZ's MODEL_DICT actually contains the full HuggingFace names
        # and matches them case-insensitively, so we should just pass
        # the original model_id and let GenZ handle the mapping
        return model_id


def find_best_hardware_for_usecase(
    usecase: Dict[str, Any],
    batch_size: int,
    model_sizes: List[str]
) -> List[Dict[str, Any]]:
    """
    Find the best hardware configurations for a usecase across different model sizes
    
    Args:
        usecase: Usecase requirements with SLO constraints
        batch_size: Fixed batch size to evaluate
        model_sizes: List of model size categories to test
    
    Returns:
        List of configurations that meet SLO requirements, ranked by cost
    """
    optimizer = HardwareOptimizer()
    results = []
    
    # Get all available hardware types
    hardware_types = _get_hardware_types()
    
    for model_size in model_sizes:
        try:
            model_id = optimizer.get_representative_model(model_size)
            
            for hardware_type in hardware_types:
                # Find optimal configuration for this model-hardware pair
                config = find_optimal_configuration(
                    model_id=model_id,
                    hardware_type=hardware_type,
                    batch_size=batch_size,
                    usecase=usecase
                )
                
                if config and config['meets_slo']:
                    config['model_size'] = model_size
                    results.append(config)
                    
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_size} model: {str(e)}")
            continue
    
    # Rank by cost effectiveness
    return rank_by_cost_effectiveness(results)


def find_optimal_configuration(
    model_id: str,
    hardware_type: str,
    batch_size: int,
    usecase: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find the optimal configuration for a specific model and hardware type
    
    Returns None if no configuration meets SLO requirements
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if not model_id or '/' not in model_id:
        raise ValueError(f"Invalid model ID: {model_id}")
    
    # Skip GenZ calls if not available (for testing)
    if get_hardware_config is None:
        return _mock_configuration(model_id, hardware_type, batch_size, usecase)
    
    logger.info(f"Finding optimal config for {model_id} on {hardware_type}")
    
    # Get GenZ-compatible model name
    optimizer = HardwareOptimizer()
    genz_model_name = optimizer.get_genz_model_name(model_id)
    logger.debug(f"Using GenZ model name: {genz_model_name} for {model_id}")
    
    try:
        # Get hardware system configuration
        logger.debug(f"Getting hardware config for {hardware_type}")
        system = get_hardware_config(hardware_type)
        logger.debug(f"Got system config: {system}")
        logger.debug(f"System type: {type(system)}")
        
        # Skip get_minimum_system_size call and use heuristic instead
        logger.debug(f"Using heuristic to estimate minimum nodes for {genz_model_name}")
        model_size_gb = _estimate_model_size(genz_model_name)
        logger.debug(f"Model size estimate: {model_size_gb} GB")
        
        # Safely access memory size
        try:
            if isinstance(system, dict):
                memory_per_node = system.get('Memory_size', system.get('memory_size', 80))
            else:
                # If system is an object, try to access attribute
                memory_per_node = getattr(system, 'Memory_size', getattr(system, 'memory_size', 80))
            logger.debug(f"Memory per node: {memory_per_node} GB")
        except Exception as e:
            logger.error(f"Error accessing system memory: {e}")
            logger.debug(f"System object attributes: {dir(system) if hasattr(system, '__dir__') else 'N/A'}")
            memory_per_node = 80  # Default fallback
        min_nodes = max(1, math.ceil(model_size_gb * 2 / memory_per_node))  # 2x for weights + activations
        logger.debug(f"Estimated minimum nodes required: {min_nodes}")
        
        # Try different node counts with simplified parallelism strategies
        logger.debug(f"Generating node options starting from {min_nodes}")
        node_options = _generate_node_options(min_nodes)
        logger.debug(f"Node options to try: {node_options}")
        
        for num_nodes in node_options:
            try:
                logger.debug(f"Testing {num_nodes} nodes")
                
                # Try common parallelism strategies instead of using the hanging function
                parallelism_strategies = _generate_parallelism_strategies(num_nodes)
                
                for parallelism in parallelism_strategies:
                    try:
                        logger.debug(f"Testing parallelism: {parallelism}")
                        
                        # Evaluate performance directly
                        perf_metrics = evaluate_with_genz_direct(
                            model_id=genz_model_name,
                            system=system,  # Pass the system dict we got earlier
                            parallelism=parallelism,
                            batch_size=batch_size,
                            usecase=usecase,
                            num_nodes=num_nodes
                        )
                        
                        if perf_metrics is None:
                            logger.debug(f"Performance evaluation failed for {parallelism}")
                            continue
                        
                        # Check if meets SLO
                        if (perf_metrics['ttft'] <= usecase['ttft_max'] and 
                            perf_metrics['e2e'] <= usecase['e2e_max']):
                            
                            logger.info(f"Found solution: {num_nodes} nodes with {parallelism}")
                            return {
                                'model_id': model_id,
                                'hardware_type': hardware_type,
                                'num_nodes': num_nodes,
                                'parallelism': parallelism,
                                'batch_size': batch_size,
                                'achieved_ttft': perf_metrics['ttft'],
                                'achieved_e2e': perf_metrics['e2e'],
                                'required_ttft': usecase['ttft_max'],
                                'required_e2e': usecase['e2e_max'],
                                'meets_slo': True,
                                'cost_per_hour': _calculate_cost(hardware_type, num_nodes),
                                'throughput': batch_size / perf_metrics['e2e'] if perf_metrics['e2e'] > 0 else 0,
                                'utilization': perf_metrics['utilization']
                            }
                            
                    except Exception as e:
                        logger.debug(f"Parallelism {parallelism} failed: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Node count {num_nodes} failed: {str(e)}")
                continue
                    
    except Exception as e:
        logger.error(f"Error finding configuration: {str(e)}")
    
    return None


def evaluate_with_genz(
    model_id: str,
    system: Any,
    parallelism: str,
    batch_size: int,
    usecase: Dict[str, Any]
) -> Dict[str, float]:
    """
    Use GenZ to evaluate performance metrics
    
    Returns:
        Dictionary with ttft, e2e, tpot, and utilization metrics
    """
    # Skip if GenZ not available (for testing)
    if estimate_prefill_performance is None:
        return _mock_performance_metrics(usecase)
    
    logger.debug(f"Evaluating performance for {model_id} with {parallelism}")
    
    # Parse parallelism config
    tp, pp, ep = 1, 1, 1
    if isinstance(parallelism, str):
        # Parse format like "TP{4}_PP{2}_EP{1}"
        import re
        tp_match = re.search(r'TP\{(\d+)\}', parallelism)
        pp_match = re.search(r'PP\{(\d+)\}', parallelism)
        ep_match = re.search(r'EP\{(\d+)\}', parallelism)
        if tp_match:
            tp = int(tp_match.group(1))
        if pp_match:
            pp = int(pp_match.group(1))
        if ep_match:
            ep = int(ep_match.group(1))
    
    try:
        # Prefill phase (TTFT)
        logger.debug("Calling estimate_prefill_performance...")
        prefill_results = estimate_prefill_performance(
            model=model_id,
            batch_size=batch_size,
            input_tokens=usecase['input_tokens_max'],
            system_name=system,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep
        )
        logger.debug(f"Prefill results: {prefill_results}")
        
        # Decode phase (for E2E)
        logger.debug("Calling estimate_decode_performance...")
        decode_results = estimate_decode_performance(
            model=model_id,
            batch_size=batch_size,
            input_tokens=usecase['input_tokens_max'],
            output_tokens=usecase['output_tokens_max'],
            system_name=system,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            beam_size=usecase.get('beam_size', 1)
        )
        logger.debug(f"Decode results: {decode_results}")
    except Exception as e:
        logger.error(f"Error in GenZ performance estimation: {e}")
        raise
    
    # Handle different key names from GenZ
    # Prefill results have 'TTFT' and 'Latency' keys (capitalized)
    ttft = prefill_results.get('TTFT', prefill_results.get('Latency', 0))
    
    # Decode results have 'TPOT' and 'Latency' keys  
    tpot = decode_results.get('TPOT', decode_results.get('Latency', 0) / usecase['output_tokens_max'] if usecase['output_tokens_max'] > 0 else 0)
    
    # Calculate e2e latency (in seconds)
    e2e = ttft + (usecase['output_tokens_max'] * tpot)
    
    # Extract utilization - GenZ doesn't provide this directly, so use a reasonable default
    compute_util = 0.7  # Default 70% utilization
    
    return {
        'ttft': ttft,
        'e2e': e2e,
        'tpot': tpot,
        'utilization': compute_util
    }


def rank_by_cost_effectiveness(
    configurations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rank configurations by cost effectiveness
    
    Considers:
    - Cost per request served
    - SLO headroom (how close to limits)
    """
    for config in configurations:
        # Calculate cost per request
        throughput_per_hour = config['throughput'] * 3600
        if throughput_per_hour > 0:
            cost_per_request = config['cost_per_hour'] / throughput_per_hour
        else:
            cost_per_request = 999999.99  # Very high cost if no throughput (avoid JSON inf)
        config['cost_per_request'] = cost_per_request
        
        # Calculate SLO headroom (0-1, higher is more headroom)
        ttft_headroom = (config['required_ttft'] - config['achieved_ttft']) / config['required_ttft'] if config['required_ttft'] > 0 else 0
        e2e_headroom = (config['required_e2e'] - config['achieved_e2e']) / config['required_e2e'] if config['required_e2e'] > 0 else 0
        avg_headroom = (ttft_headroom + e2e_headroom) / 2
        
        # Efficiency score: lower is better
        # Penalize configurations that are too close to SLO limits
        # Avoid inf values for JSON serialization
        efficiency_score = cost_per_request * (1 - avg_headroom * 0.5)
        config['efficiency_score'] = min(efficiency_score, 999999.99)  # Cap at large value to avoid inf
    
    # Sort by efficiency score (lower is better)
    return sorted(configurations, key=lambda x: x['efficiency_score'])


# Helper functions

def _get_hardware_types() -> List[str]:
    """Get list of available hardware types"""
    # Return hardware types available in GenZ
    return [
        "A100_40GB", "A100_80GB", "H100_80GB", "MI300X", "TPUv4", "TPUv5e"
    ]


def _generate_node_options(min_nodes: int, max_nodes: int = 16) -> List[int]:
    """Generate node count options to try"""
    options = []
    current = min_nodes
    
    while current <= max_nodes:
        options.append(current)
        # Use powers of 2 for better network topology
        if current < 2:
            current = 2
        elif current < 4:
            current = 4
        elif current < 8:
            current = 8
        else:
            next_current = current * 2
            if next_current > max_nodes:
                break  # Exit loop if we can't go higher
            current = next_current
            
    return options


def _calculate_cost(hardware_type: str, num_nodes: int) -> float:
    """Calculate hourly cost for hardware configuration"""
    optimizer = HardwareOptimizer()
    base_cost = optimizer.hardware_costs.get(hardware_type, 3.0)  # Default $3/hour
    return base_cost * num_nodes


def _mock_configuration(
    model_id: str,
    hardware_type: str,
    batch_size: int,
    usecase: Dict[str, Any]
) -> Dict[str, Any]:
    """Mock configuration for testing without GenZ"""
    # Simple heuristic for testing
    model_size = _estimate_model_size(model_id)
    num_nodes = max(1, model_size // 40)  # Assume 40GB per node
    
    # Mock performance based on model size
    base_ttft = 0.1 * (model_size / 7)
    base_e2e = base_ttft + usecase['output_tokens_max'] * 0.02
    
    meets_slo = base_ttft <= usecase['ttft_max'] and base_e2e <= usecase['e2e_max']
    
    return {
        'model_id': model_id,
        'hardware_type': hardware_type,
        'num_nodes': num_nodes,
        'parallelism': f'TP{{{min(num_nodes, 8)}}}_PP{{1}}',
        'batch_size': batch_size,
        'achieved_ttft': base_ttft,
        'achieved_e2e': base_e2e,
        'required_ttft': usecase['ttft_max'],
        'required_e2e': usecase['e2e_max'],
        'meets_slo': meets_slo,
        'cost_per_hour': _calculate_cost(hardware_type, num_nodes),
        'throughput': batch_size / base_e2e if base_e2e > 0 else 0,
        'utilization': 0.7
    } if meets_slo else None


def _mock_performance_metrics(usecase: Dict[str, Any]) -> Dict[str, float]:
    """Mock performance metrics for testing"""
    return {
        'ttft': usecase['ttft_max'] * 0.8,  # 80% of limit
        'e2e': usecase['e2e_max'] * 0.8,
        'tpot': 0.02,
        'utilization': 0.75
    }


def _generate_parallelism_strategies(num_nodes: int) -> List[str]:
    """Generate common parallelism strategies for given node count"""
    strategies = []
    
    # For single node, only tensor parallelism
    if num_nodes == 1:
        strategies.append("TP{1}_PP{1}")
        return strategies
    
    # For multiple nodes, try different TP/PP combinations
    for tp in [1, 2, 4, 8, min(num_nodes, 16)]:
        pp = num_nodes // tp
        if tp * pp == num_nodes:
            strategies.append(f"TP{{{tp}}}_PP{{{pp}}}")
    
    return strategies


def evaluate_with_genz_direct(
    model_id: str,
    system: Any,
    parallelism: str,
    batch_size: int,
    usecase: Dict[str, Any],
    num_nodes: int
) -> Optional[Dict[str, float]]:
    """
    Direct evaluation using GenZ performance estimation without the hanging optimization
    """
    if estimate_prefill_performance is None:
        return None
    
    logger.debug(f"Direct evaluation for {model_id} with {parallelism}")
    logger.debug(f"System: {system}, type: {type(system)}")
    
    # Parse parallelism config
    tp, pp, ep = 1, 1, 1
    if isinstance(parallelism, str):
        import re
        tp_match = re.search(r'TP\{(\d+)\}', parallelism)
        pp_match = re.search(r'PP\{(\d+)\}', parallelism)
        ep_match = re.search(r'EP\{(\d+)\}', parallelism)
        if tp_match:
            tp = int(tp_match.group(1))
        if pp_match:
            pp = int(pp_match.group(1))
        if ep_match:
            ep = int(ep_match.group(1))
    
    try:
        # Both functions expect system_name parameter with a dict
        logger.debug(f"System passed: {system}")
        
        # Prefill phase (TTFT)
        logger.debug(f"Calling estimate_prefill_performance for {model_id}...")
        prefill_results = estimate_prefill_performance(
            model=model_id,
            batch_size=batch_size,
            input_tokens=usecase['input_tokens_max'],
            system_name=system,  # Pass the system dict directly
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep
        )
        logger.debug(f"Prefill completed: {prefill_results}")
        
        # Decode phase (for E2E)
        logger.debug(f"Calling estimate_decode_performance for {model_id}...")
        decode_results = estimate_decode_performance(
            model=model_id,
            batch_size=batch_size,
            input_tokens=usecase['input_tokens_max'],
            output_tokens=usecase['output_tokens_max'],
            system_name=system,  # Pass the system dict directly
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            beam_size=usecase.get('beam_size', 1)
        )
        logger.debug(f"Decode completed: {decode_results}")
        
    except Exception as e:
        logger.error(f"Error in direct GenZ evaluation: {e}")
        return None
    
    # Handle different key names from GenZ
    # Prefill results have 'TTFT' and 'Latency' keys (capitalized)
    # GenZ returns latencies in milliseconds
    ttft_ms = prefill_results.get('TTFT', prefill_results.get('Latency', 0))
    ttft = ttft_ms / 1000.0  # Convert to seconds
    
    # Decode results have 'TPOT' and 'Latency' keys  
    # TPOT is already in milliseconds per token
    tpot_ms = decode_results.get('TPOT', decode_results.get('Latency', 0) / usecase['output_tokens_max'] if usecase['output_tokens_max'] > 0 else 0)
    tpot = tpot_ms / 1000.0  # Convert to seconds
    
    # Calculate e2e latency (in seconds)
    e2e = ttft + (usecase['output_tokens_max'] * tpot)
    
    # Extract utilization - GenZ doesn't provide this directly, so use a reasonable default
    compute_util = 0.7  # Default 70% utilization
    
    logger.debug(f"Extracted metrics - TTFT: {ttft:.3f}s, TPOT: {tpot:.6f}s, E2E: {e2e:.3f}s")
    
    return {
        'ttft': ttft,
        'e2e': e2e,
        'tpot': tpot,
        'utilization': compute_util
    }


def _estimate_model_size(model_id: str) -> int:
    """Estimate model size in GB from model ID"""
    model_lower = model_id.lower()
    
    if '405b' in model_lower:
        return 810
    elif '70b' in model_lower or '72b' in model_lower:
        return 140
    elif '33b' in model_lower or '32b' in model_lower:
        return 66
    elif '13b' in model_lower:
        return 26
    elif '8b' in model_lower or '7b' in model_lower:
        return 16
    elif '3b' in model_lower or 'phi-2' in model_lower:
        return 6
    elif '1b' in model_lower or 'tiny' in model_lower:
        return 2
    else:
        return 10  # Default