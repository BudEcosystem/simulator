"""
Node Selector - Select optimal nodes from heterogeneous hardware pool.

This module provides algorithms to select the best subset of nodes
from a mixed hardware pool for training, considering:
- GPU capability matching
- Network topology locality
- Cost efficiency

Main entry points:
- select_optimal_nodes(): Select best nodes from pool
- evaluate_node_combination(): Evaluate a specific node combination
- find_homogeneous_groups(): Group nodes by GPU type

Example usage:
    >>> from llm_memory_calculator.training import select_optimal_nodes, NodeSpec
    >>>
    >>> nodes = [
    ...     NodeSpec(node_id='node-1', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
    ...     NodeSpec(node_id='node-2', gpu_type='H100', num_gpus=8, rack_id='rack-1'),
    ...     NodeSpec(node_id='node-3', gpu_type='A100_80GB', num_gpus=8, rack_id='rack-2'),
    ... ]
    >>>
    >>> result = select_optimal_nodes(
    ...     model='Llama-3.1-70B',
    ...     available_nodes=nodes,
    ...     max_nodes=2,
    ...     optimization_target='throughput',
    ... )
    >>>
    >>> print(f"Selected: {[n.node_id for n in result.selected_nodes]}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput
from ..genz.LLM_training.training_parallelization import (
    get_best_training_parallelization,
    TrainingParallelismConfig,
)
from .cluster_optimizer import _get_gpu_memory, _get_system_name
from .tco_calculator import get_gpu_pricing


@dataclass
class NodeSpec:
    """
    Specification for a single compute node.

    Represents a node in a heterogeneous cluster with its
    hardware capabilities and network topology information.
    """
    node_id: str
    gpu_type: str
    num_gpus: int
    gpu_memory_gb: Optional[float] = None  # Auto-detected if None
    intra_node_bw_gbps: float = 600.0  # NVLink bandwidth within node
    inter_node_bw_gbps: float = 400.0  # InfiniBand/RoCE to other nodes

    # Topology information
    rack_id: Optional[str] = None
    datacenter_id: Optional[str] = None
    availability_zone: Optional[str] = None

    # Cost
    hourly_cost: float = 0.0

    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect GPU memory if not provided."""
        if self.gpu_memory_gb is None:
            self.gpu_memory_gb = _get_gpu_memory(self.gpu_type)

    @property
    def total_hourly_cost(self) -> float:
        """Total hourly cost for this node."""
        return self.hourly_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'gpu_type': self.gpu_type,
            'num_gpus': self.num_gpus,
            'gpu_memory_gb': self.gpu_memory_gb,
            'intra_node_bw_gbps': self.intra_node_bw_gbps,
            'inter_node_bw_gbps': self.inter_node_bw_gbps,
            'rack_id': self.rack_id,
            'datacenter_id': self.datacenter_id,
            'hourly_cost': self.hourly_cost,
        }


@dataclass
class NodeSelectionResult:
    """
    Result from node selection algorithm.

    Contains selected nodes and their optimal parallelism configuration.
    """
    selected_nodes: List[NodeSpec]
    total_gpus: int
    total_nodes: int
    parallelism_config: TrainingParallelismConfig

    # Performance estimates
    estimated_throughput: float
    step_time_ms: float
    memory_per_gpu_gb: float
    mfu: float

    # Cost
    estimated_cost_per_hour: float
    cost_per_million_tokens: float

    # Network efficiency
    network_efficiency: float
    same_rack_fraction: float

    # Selection metadata
    homogeneous: bool
    gpu_types_used: List[str]
    rejection_reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selected_nodes': [n.to_dict() for n in self.selected_nodes],
            'total_gpus': self.total_gpus,
            'total_nodes': self.total_nodes,
            'parallelism': self.parallelism_config.to_dict() if self.parallelism_config else None,
            'performance': {
                'throughput': self.estimated_throughput,
                'step_time_ms': self.step_time_ms,
                'memory_per_gpu_gb': self.memory_per_gpu_gb,
                'mfu': self.mfu,
            },
            'cost': {
                'hourly': self.estimated_cost_per_hour,
                'per_million_tokens': self.cost_per_million_tokens,
            },
            'network': {
                'efficiency': self.network_efficiency,
                'same_rack_fraction': self.same_rack_fraction,
            },
            'metadata': {
                'homogeneous': self.homogeneous,
                'gpu_types': self.gpu_types_used,
            },
        }

    def summary(self) -> str:
        """Human-readable summary."""
        node_ids = [n.node_id for n in self.selected_nodes]
        lines = [
            "=" * 60,
            "NODE SELECTION RESULT",
            "=" * 60,
            "",
            f"Selected Nodes: {len(self.selected_nodes)}",
            f"  {', '.join(node_ids[:5])}{'...' if len(node_ids) > 5 else ''}",
            f"Total GPUs: {self.total_gpus}",
            f"GPU Types: {', '.join(self.gpu_types_used)}",
            f"Homogeneous: {self.homogeneous}",
            "",
            f"Parallelism: TP={self.parallelism_config.tensor_parallel}, "
            f"PP={self.parallelism_config.pipeline_parallel}, "
            f"DP={self.parallelism_config.data_parallel}",
            "",
            f"Throughput: {self.estimated_throughput:,.0f} tokens/sec",
            f"MFU: {self.mfu:.1%}",
            f"Memory/GPU: {self.memory_per_gpu_gb:.1f} GB",
            "",
            f"Network Efficiency: {self.network_efficiency:.1%}",
            f"Same-Rack Fraction: {self.same_rack_fraction:.1%}",
            "",
            f"Cost/hour: ${self.estimated_cost_per_hour:.2f}",
            f"Cost/M tokens: ${self.cost_per_million_tokens:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def select_optimal_nodes(
    model: str,
    available_nodes: List[NodeSpec],
    max_nodes: Optional[int] = None,
    max_gpus: Optional[int] = None,
    max_cost_per_hour: Optional[float] = None,
    optimization_target: str = 'throughput',
    prefer_homogeneous: bool = True,
    prefer_same_rack: bool = True,
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
) -> NodeSelectionResult:
    """
    Select optimal subset of nodes from heterogeneous pool.

    Args:
        model: Model name
        available_nodes: List of available nodes
        max_nodes: Maximum number of nodes to select
        max_gpus: Maximum total GPUs to use
        max_cost_per_hour: Maximum hourly cost constraint
        optimization_target: What to optimize:
            - 'throughput': Maximum tokens/sec
            - 'cost': Minimum cost/token
            - 'efficiency': Best hardware utilization
        prefer_homogeneous: Prefer same GPU type (default True)
        prefer_same_rack: Prefer nodes in same rack (default True)
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method
        optimizer: Optimizer type
        bits: Precision

    Returns:
        NodeSelectionResult with selected nodes and configuration
    """
    if not available_nodes:
        raise ValueError("No nodes provided")

    # Default max_nodes to all available
    if max_nodes is None:
        max_nodes = len(available_nodes)

    # Group nodes by GPU type
    groups = find_homogeneous_groups(available_nodes)

    candidates = []

    # Strategy 1: Try homogeneous groups first
    if prefer_homogeneous:
        for gpu_type, nodes in groups.items():
            # Try different node counts
            for n in range(1, min(len(nodes), max_nodes) + 1):
                # Sort by rack affinity if preferred
                if prefer_same_rack:
                    sorted_nodes = _sort_by_rack_affinity(nodes)
                else:
                    sorted_nodes = nodes

                selected = sorted_nodes[:n]
                result = _evaluate_node_selection(
                    model=model,
                    nodes=selected,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    training_stage=training_stage,
                    method=method,
                    optimizer=optimizer,
                    bits=bits,
                )

                if result is not None:
                    # Check constraints
                    if max_gpus and result.total_gpus > max_gpus:
                        continue
                    if max_cost_per_hour and result.estimated_cost_per_hour > max_cost_per_hour:
                        continue

                    candidates.append(result)

    # Strategy 2: If not strictly homogeneous, try mixed configurations
    if not prefer_homogeneous or not candidates:
        # Sort all nodes by capability and try combinations
        sorted_all = sorted(available_nodes, key=lambda n: -n.num_gpus)

        if prefer_same_rack:
            sorted_all = _sort_by_rack_affinity(sorted_all)

        for n in range(1, min(len(sorted_all), max_nodes) + 1):
            selected = sorted_all[:n]

            # Only try if homogeneous or mixed is allowed
            gpu_types = set(node.gpu_type for node in selected)
            if len(gpu_types) > 1 and prefer_homogeneous:
                continue

            result = _evaluate_node_selection(
                model=model,
                nodes=selected,
                batch_size=batch_size,
                seq_length=seq_length,
                training_stage=training_stage,
                method=method,
                optimizer=optimizer,
                bits=bits,
            )

            if result is not None:
                if max_gpus and result.total_gpus > max_gpus:
                    continue
                if max_cost_per_hour and result.estimated_cost_per_hour > max_cost_per_hour:
                    continue

                candidates.append(result)

    if not candidates:
        raise ValueError(
            f"No valid node configuration found for {model}. "
            "Try relaxing constraints or using a smaller model."
        )

    # Sort by optimization target
    if optimization_target == 'throughput':
        candidates.sort(key=lambda x: -x.estimated_throughput)
    elif optimization_target == 'cost':
        candidates.sort(key=lambda x: x.cost_per_million_tokens)
    elif optimization_target == 'efficiency':
        candidates.sort(key=lambda x: (-x.mfu, -x.network_efficiency))
    else:
        candidates.sort(key=lambda x: -x.estimated_throughput)

    return candidates[0]


def _evaluate_node_selection(
    model: str,
    nodes: List[NodeSpec],
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
    optimizer: str = 'adamw',
    bits: str = 'bf16',
) -> Optional[NodeSelectionResult]:
    """Evaluate a specific node selection."""
    if not nodes:
        return None

    # Calculate totals
    total_gpus = sum(n.num_gpus for n in nodes)
    total_cost_per_hour = sum(n.hourly_cost for n in nodes)

    # Check if all same GPU type
    gpu_types = list(set(n.gpu_type for n in nodes))
    is_homogeneous = len(gpu_types) == 1

    # For now, only support homogeneous clusters
    # Heterogeneous requires more complex handling
    if not is_homogeneous:
        # Use the most common GPU type for simulation
        type_counts = defaultdict(int)
        for n in nodes:
            type_counts[n.gpu_type] += n.num_gpus
        primary_gpu_type = max(type_counts, key=type_counts.get)
    else:
        primary_gpu_type = gpu_types[0]

    gpu_memory = _get_gpu_memory(primary_gpu_type)
    system_name = _get_system_name(primary_gpu_type)

    # Find best parallelism
    try:
        config, sim_result = get_best_training_parallelization(
            model=model,
            total_gpus=total_gpus,
            batch_size=batch_size,
            seq_length=seq_length,
            system_name=system_name,
            training_stage=training_stage,
            method=method,
            optimizer=optimizer,
            bits=bits,
            optimize_for='throughput',
            gpu_memory_gb=gpu_memory,
        )
    except Exception:
        return None

    # Calculate network efficiency
    network_efficiency = _calculate_network_efficiency(nodes)
    same_rack_fraction = _calculate_same_rack_fraction(nodes)

    # Apply network penalty to throughput estimate
    # Cross-rack communication reduces effective bandwidth
    if same_rack_fraction < 1.0:
        effective_throughput = sim_result.tokens_per_second * (0.85 + 0.15 * same_rack_fraction)
    else:
        effective_throughput = sim_result.tokens_per_second

    # Calculate cost per million tokens
    tokens_per_hour = effective_throughput * 3600
    cost_per_mtok = (total_cost_per_hour / tokens_per_hour) * 1_000_000 if tokens_per_hour > 0 else float('inf')

    return NodeSelectionResult(
        selected_nodes=nodes,
        total_gpus=total_gpus,
        total_nodes=len(nodes),
        parallelism_config=config,
        estimated_throughput=effective_throughput,
        step_time_ms=sim_result.step_time_ms,
        memory_per_gpu_gb=sim_result.memory_per_gpu_gb,
        mfu=sim_result.model_flops_utilization,
        estimated_cost_per_hour=total_cost_per_hour,
        cost_per_million_tokens=cost_per_mtok,
        network_efficiency=network_efficiency,
        same_rack_fraction=same_rack_fraction,
        homogeneous=is_homogeneous,
        gpu_types_used=gpu_types,
    )


def find_homogeneous_groups(nodes: List[NodeSpec]) -> Dict[str, List[NodeSpec]]:
    """
    Group nodes by GPU type.

    Args:
        nodes: List of nodes to group

    Returns:
        Dict mapping GPU type to list of nodes
    """
    groups = defaultdict(list)
    for node in nodes:
        # Normalize GPU type
        normalized = node.gpu_type.lower().replace('-', '_').replace(' ', '_')
        groups[normalized].append(node)

    return dict(groups)


def _sort_by_rack_affinity(nodes: List[NodeSpec]) -> List[NodeSpec]:
    """Sort nodes to maximize rack locality."""
    # Group by rack
    racks = defaultdict(list)
    no_rack = []

    for node in nodes:
        if node.rack_id:
            racks[node.rack_id].append(node)
        else:
            no_rack.append(node)

    # Sort racks by size (largest first)
    sorted_racks = sorted(racks.values(), key=len, reverse=True)

    # Flatten, keeping rack groups together
    result = []
    for rack_nodes in sorted_racks:
        result.extend(rack_nodes)
    result.extend(no_rack)

    return result


def _calculate_network_efficiency(nodes: List[NodeSpec]) -> float:
    """
    Calculate network efficiency score for node selection.

    Returns 0-1 score based on:
    - Same rack: 1.0
    - Same datacenter, different rack: 0.85
    - Different datacenter: 0.6
    """
    if len(nodes) <= 1:
        return 1.0

    # Count pairs by locality
    same_rack = 0
    same_dc = 0
    different_dc = 0
    total_pairs = 0

    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            total_pairs += 1

            if n1.rack_id and n2.rack_id and n1.rack_id == n2.rack_id:
                same_rack += 1
            elif n1.datacenter_id and n2.datacenter_id and n1.datacenter_id == n2.datacenter_id:
                same_dc += 1
            else:
                different_dc += 1

    if total_pairs == 0:
        return 1.0

    # Weighted efficiency
    efficiency = (
        (same_rack * 1.0) +
        (same_dc * 0.85) +
        (different_dc * 0.6)
    ) / total_pairs

    return efficiency


def _calculate_same_rack_fraction(nodes: List[NodeSpec]) -> float:
    """Calculate fraction of nodes in the same rack."""
    if len(nodes) <= 1:
        return 1.0

    # Find dominant rack
    rack_counts = defaultdict(int)
    no_rack_count = 0

    for node in nodes:
        if node.rack_id:
            rack_counts[node.rack_id] += 1
        else:
            no_rack_count += 1

    if not rack_counts:
        return 0.0  # No rack information

    max_same_rack = max(rack_counts.values())
    return max_same_rack / len(nodes)


def evaluate_node_combination(
    model: str,
    nodes: List[NodeSpec],
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
) -> Dict[str, Any]:
    """
    Evaluate a specific combination of nodes.

    Args:
        model: Model name
        nodes: List of nodes to evaluate
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method

    Returns:
        Dict with evaluation metrics
    """
    result = _evaluate_node_selection(
        model=model,
        nodes=nodes,
        batch_size=batch_size,
        seq_length=seq_length,
        training_stage=training_stage,
        method=method,
    )

    if result is None:
        return {
            'success': False,
            'error': 'Evaluation failed',
        }

    return {
        'success': True,
        'total_gpus': result.total_gpus,
        'total_nodes': result.total_nodes,
        'throughput': result.estimated_throughput,
        'cost_per_hour': result.estimated_cost_per_hour,
        'cost_per_million_tokens': result.cost_per_million_tokens,
        'memory_per_gpu_gb': result.memory_per_gpu_gb,
        'mfu': result.mfu,
        'network_efficiency': result.network_efficiency,
        'parallelism': {
            'tp': result.parallelism_config.tensor_parallel,
            'pp': result.parallelism_config.pipeline_parallel,
            'dp': result.parallelism_config.data_parallel,
            'zero': result.parallelism_config.zero_stage,
        },
    }


def rank_node_selections(
    model: str,
    available_nodes: List[NodeSpec],
    node_counts: List[int],
    optimization_target: str = 'throughput',
    batch_size: int = 4,
    seq_length: int = 4096,
    training_stage: str = 'sft',
    method: str = 'full',
) -> List[Dict[str, Any]]:
    """
    Rank different node count selections.

    Args:
        model: Model name
        available_nodes: Available nodes
        node_counts: List of node counts to evaluate
        optimization_target: Optimization target
        batch_size: Per-GPU batch size
        seq_length: Sequence length
        training_stage: Training type
        method: Training method

    Returns:
        List of results sorted by optimization target
    """
    results = []

    # Group by GPU type
    groups = find_homogeneous_groups(available_nodes)

    for gpu_type, nodes in groups.items():
        for n in node_counts:
            if n > len(nodes):
                continue

            sorted_nodes = _sort_by_rack_affinity(nodes)
            selected = sorted_nodes[:n]

            result = _evaluate_node_selection(
                model=model,
                nodes=selected,
                batch_size=batch_size,
                seq_length=seq_length,
                training_stage=training_stage,
                method=method,
            )

            if result:
                results.append({
                    'gpu_type': gpu_type,
                    'num_nodes': n,
                    'total_gpus': result.total_gpus,
                    'throughput': result.estimated_throughput,
                    'cost_per_hour': result.estimated_cost_per_hour,
                    'cost_per_mtok': result.cost_per_million_tokens,
                    'mfu': result.mfu,
                    'network_efficiency': result.network_efficiency,
                    'parallelism': result.parallelism_config.to_dict(),
                    'node_ids': [node.node_id for node in selected],
                })

    # Sort by optimization target
    if optimization_target == 'throughput':
        results.sort(key=lambda x: -x['throughput'])
    elif optimization_target == 'cost':
        results.sort(key=lambda x: x['cost_per_mtok'])
    elif optimization_target == 'efficiency':
        results.sort(key=lambda x: -x['mfu'])

    return results
