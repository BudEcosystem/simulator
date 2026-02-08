"""
Node Selector Validation Tests.

Validates that the node selection algorithm correctly selects
optimal nodes from heterogeneous hardware pools.
"""

import pytest
from typing import List, Dict, Any

from llm_memory_calculator.training import (
    select_optimal_nodes,
    find_homogeneous_groups,
    evaluate_node_combination,
    rank_node_selections,
    NodeSpec,
    NodeSelectionResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def homogeneous_h100_nodes():
    """Create homogeneous H100 nodes."""
    return [
        NodeSpec(
            node_id='h100-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-2',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-3',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-4',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=24.0,
        ),
    ]


@pytest.fixture
def heterogeneous_nodes():
    """Create heterogeneous nodes with different GPU types."""
    return [
        NodeSpec(
            node_id='h100-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='h100-2',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='a100-1',
            gpu_type='A100_80GB',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=16.0,
        ),
        NodeSpec(
            node_id='a100-2',
            gpu_type='A100_80GB',
            num_gpus=8,
            rack_id='rack-2',
            hourly_cost=16.0,
        ),
    ]


@pytest.fixture
def nodes_with_topology():
    """Create nodes with detailed topology information."""
    return [
        NodeSpec(
            node_id='node-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            datacenter_id='dc-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='node-2',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            datacenter_id='dc-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='node-3',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-2',
            datacenter_id='dc-1',
            hourly_cost=24.0,
        ),
        NodeSpec(
            node_id='node-4',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-3',
            datacenter_id='dc-2',
            hourly_cost=24.0,
        ),
    ]


# =============================================================================
# Test NodeSpec
# =============================================================================

class TestNodeSpec:
    """Tests for NodeSpec dataclass."""

    def test_node_spec_creation(self):
        """NodeSpec should be created with correct attributes."""
        node = NodeSpec(
            node_id='test-1',
            gpu_type='H100',
            num_gpus=8,
            hourly_cost=24.0,
        )

        assert node.node_id == 'test-1'
        assert node.gpu_type == 'H100'
        assert node.num_gpus == 8
        assert node.hourly_cost == 24.0

    def test_node_spec_auto_memory_detection(self):
        """GPU memory should be auto-detected if not provided."""
        node = NodeSpec(
            node_id='test-1',
            gpu_type='H100',
            num_gpus=8,
        )

        # H100 should have 80GB memory
        assert node.gpu_memory_gb is not None
        assert node.gpu_memory_gb > 0

    def test_node_spec_to_dict(self):
        """NodeSpec.to_dict() should return valid dictionary."""
        node = NodeSpec(
            node_id='test-1',
            gpu_type='H100',
            num_gpus=8,
            rack_id='rack-1',
            hourly_cost=24.0,
        )

        result = node.to_dict()
        assert isinstance(result, dict)
        assert result['node_id'] == 'test-1'
        assert result['gpu_type'] == 'H100'
        assert result['num_gpus'] == 8


# =============================================================================
# Test select_optimal_nodes
# =============================================================================

class TestSelectOptimalNodes:
    """Tests for select_optimal_nodes function."""

    def test_returns_valid_result(self, homogeneous_h100_nodes):
        """Should return valid NodeSelectionResult."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=2,
        )

        assert isinstance(result, NodeSelectionResult)
        assert len(result.selected_nodes) <= 2
        assert result.total_gpus > 0

    def test_respects_max_nodes(self, homogeneous_h100_nodes):
        """Should respect max_nodes constraint."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=1,
        )

        assert len(result.selected_nodes) <= 1

    def test_respects_max_gpus(self, homogeneous_h100_nodes):
        """Should respect max_gpus constraint."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_gpus=16,
        )

        assert result.total_gpus <= 16

    def test_respects_cost_constraint(self, homogeneous_h100_nodes):
        """Should respect max_cost_per_hour constraint."""
        max_cost = 30.0  # Should select ~1 node

        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_cost_per_hour=max_cost,
        )

        assert result.estimated_cost_per_hour <= max_cost

    def test_homogeneous_preference_true(self, heterogeneous_nodes):
        """prefer_homogeneous=True should select same GPU types."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=heterogeneous_nodes,
            max_nodes=2,
            prefer_homogeneous=True,
        )

        assert result.homogeneous
        assert len(result.gpu_types_used) == 1

    def test_rack_preference(self, homogeneous_h100_nodes):
        """prefer_same_rack=True should prefer same-rack nodes."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=2,
            prefer_same_rack=True,
        )

        if result.total_nodes == 2:
            # Should prefer rack-1 which has 2 nodes
            assert result.same_rack_fraction > 0

    def test_optimization_target_throughput(self, homogeneous_h100_nodes):
        """optimization_target='throughput' should maximize tokens/sec."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=4,
            optimization_target='throughput',
        )

        assert result.estimated_throughput > 0

    def test_optimization_target_cost(self, homogeneous_h100_nodes):
        """optimization_target='cost' should minimize cost/token."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=4,
            optimization_target='cost',
        )

        assert result.cost_per_million_tokens > 0

    def test_result_has_parallelism_config(self, homogeneous_h100_nodes):
        """Result should include valid parallelism config."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=2,
        )

        assert result.parallelism_config is not None
        product = (result.parallelism_config.tensor_parallel *
                  result.parallelism_config.pipeline_parallel *
                  result.parallelism_config.data_parallel)
        assert product == result.total_gpus


# =============================================================================
# Test find_homogeneous_groups
# =============================================================================

class TestFindHomogeneousGroups:
    """Tests for find_homogeneous_groups function."""

    def test_groups_homogeneous_correctly(self, heterogeneous_nodes):
        """Should group nodes by GPU type."""
        groups = find_homogeneous_groups(heterogeneous_nodes)

        assert len(groups) == 2  # H100 and A100

        # Check counts
        h100_count = sum(1 for n in heterogeneous_nodes if 'h100' in n.gpu_type.lower())
        a100_count = sum(1 for n in heterogeneous_nodes if 'a100' in n.gpu_type.lower())

        # Find the groups (keys are normalized)
        h100_key = [k for k in groups.keys() if 'h100' in k][0]
        a100_key = [k for k in groups.keys() if 'a100' in k][0]

        assert len(groups[h100_key]) == h100_count
        assert len(groups[a100_key]) == a100_count

    def test_single_gpu_type(self, homogeneous_h100_nodes):
        """Should return single group for homogeneous nodes."""
        groups = find_homogeneous_groups(homogeneous_h100_nodes)

        assert len(groups) == 1

    def test_normalizes_gpu_names(self):
        """Should normalize GPU type names."""
        # Test case sensitivity normalization
        nodes = [
            NodeSpec(node_id='1', gpu_type='H100', num_gpus=8),
            NodeSpec(node_id='2', gpu_type='h100', num_gpus=8),  # Lowercase
            NodeSpec(node_id='3', gpu_type='H100', num_gpus=8),  # Same as first
        ]

        groups = find_homogeneous_groups(nodes)

        # All should be in same group after normalization (case-insensitive)
        assert len(groups) == 1

    def test_normalizes_gpu_names_with_variations(self):
        """GPU names with different separators should normalize correctly."""
        # Note: 'H-100' normalizes to 'h_100' (dash becomes underscore)
        # So 'H100' and 'H-100' are treated as different GPUs
        nodes_different = [
            NodeSpec(node_id='1', gpu_type='H100', num_gpus=8),
            NodeSpec(node_id='2', gpu_type='H-100', num_gpus=8),  # Dash becomes underscore
        ]

        groups = find_homogeneous_groups(nodes_different)

        # These are treated as different GPU types (h100 vs h_100)
        assert len(groups) == 2


# =============================================================================
# Test evaluate_node_combination
# =============================================================================

class TestEvaluateNodeCombination:
    """Tests for evaluate_node_combination function."""

    def test_returns_valid_evaluation(self, homogeneous_h100_nodes):
        """Should return valid evaluation for node combination."""
        selected = homogeneous_h100_nodes[:2]

        result = evaluate_node_combination(
            model='meta-llama/meta-llama-3.1-8b',
            nodes=selected,
        )

        assert result['success']
        assert 'throughput' in result
        assert 'cost_per_hour' in result
        assert 'memory_per_gpu_gb' in result

    def test_calculates_total_gpus(self, homogeneous_h100_nodes):
        """Should correctly calculate total GPUs."""
        selected = homogeneous_h100_nodes[:2]  # 2 nodes x 8 GPUs = 16 GPUs

        result = evaluate_node_combination(
            model='meta-llama/meta-llama-3.1-8b',
            nodes=selected,
        )

        assert result['success']
        assert result['total_gpus'] == 16

    def test_calculates_total_nodes(self, homogeneous_h100_nodes):
        """Should correctly calculate total nodes."""
        selected = homogeneous_h100_nodes[:3]

        result = evaluate_node_combination(
            model='meta-llama/meta-llama-3.1-8b',
            nodes=selected,
        )

        assert result['success']
        assert result['total_nodes'] == 3

    def test_includes_parallelism(self, homogeneous_h100_nodes):
        """Should include parallelism configuration."""
        result = evaluate_node_combination(
            model='meta-llama/meta-llama-3.1-8b',
            nodes=homogeneous_h100_nodes[:2],
        )

        assert result['success']
        assert 'parallelism' in result
        assert 'tp' in result['parallelism']
        assert 'pp' in result['parallelism']
        assert 'dp' in result['parallelism']


# =============================================================================
# Test rank_node_selections
# =============================================================================

class TestRankNodeSelections:
    """Tests for rank_node_selections function."""

    def test_returns_ranked_list(self, homogeneous_h100_nodes):
        """Should return list of ranked selections."""
        results = rank_node_selections(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            node_counts=[1, 2],
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_sorted_by_throughput(self, homogeneous_h100_nodes):
        """Results should be sorted by throughput when target='throughput'."""
        results = rank_node_selections(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            node_counts=[1, 2],
            optimization_target='throughput',
        )

        if len(results) >= 2:
            # Should be sorted descending by throughput
            for i in range(len(results) - 1):
                assert results[i]['throughput'] >= results[i+1]['throughput']

    def test_sorted_by_cost(self, homogeneous_h100_nodes):
        """Results should be sorted by cost when target='cost'."""
        results = rank_node_selections(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            node_counts=[1, 2],
            optimization_target='cost',
        )

        if len(results) >= 2:
            # Should be sorted ascending by cost per token
            for i in range(len(results) - 1):
                assert results[i]['cost_per_mtok'] <= results[i+1]['cost_per_mtok']


# =============================================================================
# Test Network Efficiency
# =============================================================================

class TestNetworkEfficiency:
    """Tests for network topology and efficiency calculations."""

    def test_same_rack_high_efficiency(self, homogeneous_h100_nodes):
        """Same-rack nodes should have high network efficiency."""
        # Select 2 nodes from same rack
        same_rack_nodes = [n for n in homogeneous_h100_nodes if n.rack_id == 'rack-1']

        if len(same_rack_nodes) >= 2:
            result = select_optimal_nodes(
                model='meta-llama/meta-llama-3.1-8b',
                available_nodes=same_rack_nodes[:2],
                max_nodes=2,
            )

            assert result.same_rack_fraction == 1.0
            assert result.network_efficiency >= 0.9

    def test_cross_rack_lower_efficiency(self, nodes_with_topology):
        """Cross-rack nodes should have lower network efficiency."""
        # Force selection of nodes from different racks
        cross_rack = [n for n in nodes_with_topology if n.rack_id in ('rack-1', 'rack-3')]

        if len(cross_rack) >= 2:
            result = select_optimal_nodes(
                model='meta-llama/meta-llama-3.1-8b',
                available_nodes=cross_rack,
                max_nodes=2,
                prefer_same_rack=False,  # Allow cross-rack
            )

            # Should have less than perfect efficiency
            if result.total_nodes > 1 and result.same_rack_fraction < 1.0:
                assert result.network_efficiency < 1.0


# =============================================================================
# Test Result Properties
# =============================================================================

class TestNodeSelectionResultProperties:
    """Tests for NodeSelectionResult properties and methods."""

    def test_to_dict(self, homogeneous_h100_nodes):
        """to_dict() should return valid dictionary."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=2,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'selected_nodes' in result_dict
        assert 'performance' in result_dict
        assert 'cost' in result_dict
        assert 'network' in result_dict

    def test_summary(self, homogeneous_h100_nodes):
        """summary() should generate readable text."""
        result = select_optimal_nodes(
            model='meta-llama/meta-llama-3.1-8b',
            available_nodes=homogeneous_h100_nodes,
            max_nodes=2,
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'NODE SELECTION RESULT' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
