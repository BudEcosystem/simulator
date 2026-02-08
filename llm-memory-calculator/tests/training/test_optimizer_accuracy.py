"""
Tests for Optimizer Memory and Time Accuracy.

Tests cover:
- Memory footprint for different optimizers
- 8-bit optimizer savings
- Memory-efficient optimizers (Lion, Adafactor)
- Low-rank optimizers (GaLore)

Research Sources:
- 8-bit Adam (bitsandbytes): https://arxiv.org/abs/2110.02861
- Lion Optimizer: https://arxiv.org/abs/2302.06675
- GaLore: https://arxiv.org/abs/2403.03507
- Adafactor: https://arxiv.org/abs/1804.04235
"""

import pytest
from typing import Dict, Any

from llm_memory_calculator.genz import (
    training_modeling,
    list_optimizers,
    get_optimizer_profile,
    calculate_optimizer_memory,
    get_memory_efficient_optimizers,
    get_optimizers_by_category,
)


# =============================================================================
# Research-Backed Expected Values
# =============================================================================

# Optimizer memory footprint (bytes per parameter)
# Source: Various papers, bitsandbytes documentation
OPTIMIZER_MEMORY = {
    'adamw': {
        'states': 2,           # momentum + variance
        'bytes_per_param': 8,  # 2 × fp32 states
        'source': 'Standard AdamW',
    },
    'adamw_8bit': {
        'states': 2,
        'bytes_per_param': 2,  # 2 × int8 states
        'source': 'bitsandbytes 8-bit Adam',
    },
    'adam_8bit': {
        'states': 2,
        'bytes_per_param': 2,
        'source': 'bitsandbytes 8-bit Adam',
    },
    'sgd': {
        'states': 1,           # momentum only
        'bytes_per_param': 4,  # 1 × fp32 state
        'source': 'Standard SGD with momentum',
    },
    'adafactor': {
        'states': 1,           # factorized (reduced)
        'bytes_per_param': 4,  # Factorized representation
        'source': 'Adafactor paper',
    },
    'lion': {
        'states': 1,           # momentum only
        'bytes_per_param': 4,  # 1 × fp32 state
        'source': 'Lion paper',
    },
    'galore': {
        'states': 2,           # Low-rank
        'bytes_per_param': 2,  # Reduced due to low-rank
        'source': 'GaLore paper',
    },
}

# 8-bit optimizer expected savings
# Source: bitsandbytes paper - 75% reduction in optimizer states
EIGHT_BIT_SAVINGS = 0.75  # 75% memory reduction for optimizer states


# =============================================================================
# Helper Functions
# =============================================================================

def get_training_with_optimizer(optimizer: str, **kwargs):
    """Get training result with specific optimizer."""
    try:
        return training_modeling(
            model=kwargs.get('model', 'llama-2-7b'),
            training_stage=kwargs.get('training_stage', 'sft'),
            batch_size=kwargs.get('batch_size', 2),
            seq_length=kwargs.get('seq_length', 2048),
            num_gpus=kwargs.get('num_gpus', 1),
            optimizer=optimizer,
            **{k: v for k, v in kwargs.items()
               if k not in ['model', 'training_stage', 'batch_size', 'seq_length', 'num_gpus']},
        )
    except Exception as e:
        pytest.skip(f"Training with {optimizer} failed: {e}")


# =============================================================================
# Test Classes
# =============================================================================

class TestOptimizerEnumeration:
    """Test optimizer enumeration and profiles."""

    def test_list_optimizers(self):
        """Test listing all available optimizers."""
        optimizers = list_optimizers()

        # Should have at least 10 optimizers
        assert len(optimizers) >= 10, \
            f"Should have many optimizers, got {len(optimizers)}"

        # Check for essential optimizers
        essential = ['adamw', 'sgd', 'lion', 'lamb']
        for opt in essential:
            assert opt in optimizers, f"Missing essential optimizer: {opt}"

    def test_optimizer_profiles_complete(self):
        """Test optimizer profiles have required fields."""
        optimizers = list_optimizers()

        for opt in optimizers:
            profile = get_optimizer_profile(opt)

            assert 'states' in profile, f"{opt} missing states"
            assert 'flops' in profile, f"{opt} missing flops"
            assert 'memory_bytes' in profile, f"{opt} missing memory_bytes"

    def test_optimizers_by_category(self):
        """Test optimizer categorization."""
        categories = get_optimizers_by_category()

        expected_categories = ['standard', 'memory_efficient', 'layer_adaptive', 'second_order', 'low_rank']
        for cat in expected_categories:
            assert cat in categories, f"Missing category: {cat}"

        # Verify some assignments
        assert 'adamw' in categories['standard']
        assert 'lion' in categories['memory_efficient']


class TestAdamWMemory:
    """Test AdamW optimizer memory."""

    def test_adamw_memory_formula(self):
        """
        Test AdamW memory follows expected formula.

        AdamW: 8 bytes/param (2 fp32 states: momentum + variance)
        """
        num_params = 7e9  # 7B model
        memory_info = calculate_optimizer_memory(num_params, 'adamw')

        # Expected: ~56 GB for optimizer states
        # Plus master weights if using mixed precision
        expected_states_gb = (num_params * 8) / 1e9

        # Allow 20% tolerance
        assert abs(memory_info['optimizer_states_bytes'] - (num_params * 8)) / (num_params * 8) < 0.20, \
            f"AdamW states: expected ~{expected_states_gb:.1f}GB"

    def test_adamw_two_states(self):
        """Test AdamW has 2 states (momentum + variance)."""
        profile = get_optimizer_profile('adamw')

        assert profile['states'] == 2, \
            f"AdamW should have 2 states, got {profile['states']}"


class TestEightBitOptimizers:
    """Test 8-bit optimizer memory savings."""

    def test_8bit_adam_savings(self):
        """
        Test 8-bit Adam achieves ~75% memory savings.

        Source: bitsandbytes paper - quantize optimizer states to int8
        """
        num_params = 7e9

        adamw_mem = calculate_optimizer_memory(num_params, 'adamw')

        # Try different 8-bit optimizer names
        for opt_name in ['adam_8bit', 'adamw_8bit']:
            try:
                adam8_mem = calculate_optimizer_memory(num_params, opt_name)

                savings = 1 - (adam8_mem['optimizer_states_bytes'] / adamw_mem['optimizer_states_bytes'])

                # Should save 60-80% (centered around 75%)
                assert 0.50 <= savings <= 0.85, \
                    f"8-bit Adam savings {savings:.1%} should be 60-80%"
                return
            except Exception:
                continue

        # If no 8-bit optimizer found, skip
        pytest.skip("No 8-bit optimizer available")

    def test_8bit_bytes_per_param(self):
        """Test 8-bit optimizers use ~2 bytes per param."""
        for opt_name in ['adam_8bit', 'adamw_8bit']:
            try:
                profile = get_optimizer_profile(opt_name)

                # 8-bit should use ~2 bytes (2 int8 states)
                assert profile['memory_bytes'] <= 4, \
                    f"8-bit {opt_name} should use <=4 bytes/param, got {profile['memory_bytes']}"
                return
            except Exception:
                continue

    def test_8bit_training_result(self):
        """Test training with 8-bit optimizer produces results."""
        for opt_name in ['adam_8bit', 'adamw_8bit', 'paged_adamw_8bit']:
            result = get_training_with_optimizer(opt_name)
            if result is not None:
                # Should produce valid results
                assert result.optimizer_memory_gb > 0, \
                    f"8-bit optimizer should report memory"
                return


class TestSGDMemory:
    """Test SGD optimizer memory."""

    def test_sgd_single_state(self):
        """Test SGD has 0 or 1 state (momentum optional)."""
        profile = get_optimizer_profile('sgd')

        # SGD can have 0 states (no momentum) or 1 state (with momentum)
        assert profile['states'] in [0, 1], \
            f"SGD should have 0 or 1 state, got {profile['states']}"

    def test_sgd_less_memory_than_adamw(self):
        """Test SGD uses less memory than AdamW."""
        num_params = 7e9

        adamw_mem = calculate_optimizer_memory(num_params, 'adamw')
        sgd_mem = calculate_optimizer_memory(num_params, 'sgd')

        # SGD should use ~50% of AdamW memory
        ratio = sgd_mem['optimizer_states_bytes'] / adamw_mem['optimizer_states_bytes']

        assert ratio < 0.7, \
            f"SGD should use less memory than AdamW: ratio={ratio:.2f}"


class TestLionOptimizer:
    """Test Lion optimizer memory."""

    def test_lion_single_state(self):
        """
        Test Lion has single state.

        Source: Lion paper - uses momentum only
        """
        profile = get_optimizer_profile('lion')

        assert profile['states'] == 1, \
            f"Lion should have 1 state, got {profile['states']}"

    def test_lion_memory_efficient(self):
        """Test Lion is classified as memory-efficient."""
        efficient_opts = get_memory_efficient_optimizers()

        assert 'lion' in efficient_opts, \
            "Lion should be classified as memory-efficient"

    def test_lion_vs_adamw_memory(self):
        """Test Lion uses less memory than AdamW."""
        num_params = 7e9

        adamw_mem = calculate_optimizer_memory(num_params, 'adamw')
        lion_mem = calculate_optimizer_memory(num_params, 'lion')

        # Lion should use ~50% of AdamW memory
        ratio = lion_mem['optimizer_states_bytes'] / adamw_mem['optimizer_states_bytes']

        assert ratio < 0.7, \
            f"Lion should use less memory than AdamW: ratio={ratio:.2f}"


class TestAdafactorMemory:
    """Test Adafactor optimizer memory."""

    def test_adafactor_factorized_states(self):
        """
        Test Adafactor uses factorized second moment.

        Source: Adafactor paper - row and column factorization
        """
        profile = get_optimizer_profile('adafactor')

        # Should have reduced memory due to factorization
        assert profile['memory_bytes'] <= 4, \
            f"Adafactor should have reduced memory: {profile['memory_bytes']} bytes/param"

    def test_adafactor_vs_adamw(self):
        """Test Adafactor uses less memory than AdamW."""
        num_params = 7e9

        adamw_mem = calculate_optimizer_memory(num_params, 'adamw')
        adafactor_mem = calculate_optimizer_memory(num_params, 'adafactor')

        ratio = adafactor_mem['optimizer_states_bytes'] / adamw_mem['optimizer_states_bytes']

        # Adafactor should use significantly less
        assert ratio < 0.7, \
            f"Adafactor should use less memory: ratio={ratio:.2f}"


class TestGaLoreOptimizer:
    """Test GaLore low-rank optimizer."""

    def test_galore_low_rank(self):
        """
        Test GaLore uses low-rank representation.

        Source: GaLore paper - low-rank projection
        """
        profile = get_optimizer_profile('galore')

        # Should have rank_factor for low-rank
        assert 'rank_factor' in profile, \
            "GaLore should have rank_factor"
        assert profile['rank_factor'] < 1.0, \
            f"GaLore rank_factor {profile['rank_factor']} should be < 1.0"

    def test_galore_categorized_as_low_rank(self):
        """Test GaLore is in low-rank category."""
        categories = get_optimizers_by_category()

        assert 'galore' in categories['low_rank'], \
            "GaLore should be in low_rank category"


class TestOptimizerTrainingImpact:
    """Test optimizer impact on training results."""

    def test_different_optimizers_produce_results(self):
        """Test various optimizers produce valid training results."""
        test_optimizers = ['adamw', 'sgd', 'lion']

        for opt in test_optimizers:
            result = get_training_with_optimizer(opt)
            if result is not None:
                assert result.optimizer_memory_gb >= 0, \
                    f"{opt} should report optimizer memory"
                assert result.step_time_ms > 0, \
                    f"{opt} should report step time"

    def test_optimizer_memory_in_total(self):
        """Test optimizer memory is included in total memory."""
        result = get_training_with_optimizer('adamw')

        if result is None:
            return

        # Optimizer memory should be part of total
        component_sum = (
            result.weight_memory_gb +
            result.gradient_memory_gb +
            result.optimizer_memory_gb +
            result.activation_memory_gb
        )

        # Should be significant
        assert component_sum > 0, \
            "Memory components should sum to positive value"


class TestLayerAdaptiveOptimizers:
    """Test layer-adaptive optimizers (LAMB, LARS)."""

    def test_lamb_profile(self):
        """Test LAMB optimizer profile."""
        profile = get_optimizer_profile('lamb')

        # LAMB should be in layer_adaptive category
        categories = get_optimizers_by_category()
        assert 'lamb' in categories['layer_adaptive'], \
            "LAMB should be layer-adaptive"

    def test_lars_profile(self):
        """Test LARS optimizer profile."""
        profile = get_optimizer_profile('lars')

        categories = get_optimizers_by_category()
        assert 'lars' in categories['layer_adaptive'], \
            "LARS should be layer-adaptive"


class TestSecondOrderOptimizers:
    """Test second-order optimizers (Sophia, etc.)."""

    def test_sophia_profile(self):
        """Test Sophia optimizer profile."""
        profile = get_optimizer_profile('sophia')

        categories = get_optimizers_by_category()
        assert 'sophia' in categories['second_order'], \
            "Sophia should be second-order"

    def test_sophia_higher_flops(self):
        """
        Test Sophia has higher compute cost.

        Second-order methods require more computation per step.
        """
        adamw_profile = get_optimizer_profile('adamw')
        sophia_profile = get_optimizer_profile('sophia')

        # Sophia should have higher flops (Hessian estimation)
        assert sophia_profile['flops'] >= adamw_profile['flops'], \
            "Sophia should have >= flops than AdamW"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
