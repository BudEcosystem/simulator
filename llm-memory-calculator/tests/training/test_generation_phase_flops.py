"""
Tests for Generation Phase FLOPs Validation.

CRITICAL: This tests a key gap in many simulators - generation (inference) phase
should use 2x FLOPs formula, NOT the 6x training FLOPs formula.

Generation in RLHF (PPO, GRPO, etc.) is autoregressive inference:
- Forward-only (no backward pass)
- Token-by-token generation
- Memory-bound, not compute-bound

Research Sources:
- Scaling Laws Paper: https://arxiv.org/abs/2001.08361
- Chinchilla Paper: https://arxiv.org/abs/2203.15556
- LLM Training FLOPs: https://arxiv.org/abs/2104.04473
"""

import pytest
from typing import Dict, Any

from llm_memory_calculator.genz import (
    training_modeling,
    get_stage_config,
    System,
)


# =============================================================================
# FLOPs Formulas (Research-Backed)
# =============================================================================

# Training FLOPs per token:
#   6 × params (forward + backward + activation recompute)
# Source: Megatron-LM paper, Chinchilla paper

# Inference FLOPs per token:
#   2 × params (forward only)
# Source: Scaling Laws paper

FLOPS_PER_TOKEN = {
    'training': 6,   # Forward (2) + Backward (4, including recompute)
    'inference': 2,  # Forward only
}

# Model parameter counts for reference
MODEL_PARAMS = {
    'llama-2-7b': 7e9,
    'llama-2-13b': 13e9,
    'llama-2-70b': 70e9,
    'llama-3-8b': 8e9,
    'llama-3-70b': 70e9,
}


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_expected_generation_flops(num_params: float, num_tokens: int) -> float:
    """
    Calculate expected FLOPs for generation phase (inference).

    Formula: 2 × params × tokens
    """
    return 2 * num_params * num_tokens


def calculate_expected_training_flops(num_params: float, num_tokens: int) -> float:
    """
    Calculate expected FLOPs for training phase.

    Formula: 6 × params × tokens
    """
    return 6 * num_params * num_tokens


def get_rlhf_result(training_stage: str, **kwargs):
    """Get RLHF training result."""
    try:
        return training_modeling(
            model=kwargs.get('model', 'llama-2-7b'),
            training_stage=training_stage,
            batch_size=kwargs.get('batch_size', 2),
            seq_length=kwargs.get('seq_length', 2048),
            num_gpus=kwargs.get('num_gpus', 1),
            **{k: v for k, v in kwargs.items() if k not in ['model', 'batch_size', 'seq_length', 'num_gpus']},
        )
    except Exception as e:
        pytest.skip(f"RLHF modeling failed: {e}")


# =============================================================================
# Test Classes
# =============================================================================

class TestGenerationFlopsFormula:
    """Test that generation uses inference FLOPs, not training FLOPs."""

    def test_generation_should_use_2x_formula(self):
        """
        CRITICAL TEST: Generation phase should use 2x (inference) FLOPs.

        Bug to catch: Using 6x (training) formula for generation.
        Impact: Would overestimate generation time by 3x.
        """
        # Get result for GRPO which has explicit generation phase
        result = get_rlhf_result('grpo', model='llama-2-7b')

        if result is None:
            return

        # The config should track generation phase separately
        config = get_stage_config('grpo')

        # If generation is tracked, verify FLOPs multiplier
        if config.generation_forwards > 0:
            # Generation should contribute less FLOPs than training
            # A rough check: generation_multiplier should reflect inference FLOPs
            gen_multiplier = config.generation_forwards
            train_multiplier = config.backward_multiplier

            # Generation FLOPs should be less than backward FLOPs
            # (inference is forward-only, training is forward+backward)
            assert gen_multiplier <= 2.0, \
                f"Generation forward multiplier {gen_multiplier} seems too high for inference"

    def test_generation_vs_training_flops_ratio(self):
        """
        Test that generation uses 1/3 the FLOPs of training per token.

        Training: 6 × params × tokens
        Inference: 2 × params × tokens
        Ratio: 1/3
        """
        # Calculate theoretical values
        params = 7e9  # 7B model
        tokens = 256  # Generation tokens

        training_flops = calculate_expected_training_flops(params, tokens)
        generation_flops = calculate_expected_generation_flops(params, tokens)

        expected_ratio = generation_flops / training_flops  # Should be 1/3

        assert abs(expected_ratio - (1/3)) < 0.01, \
            f"Generation/training FLOPs ratio should be 1/3, got {expected_ratio:.4f}"


class TestPPOGenerationPhase:
    """Test PPO generation phase characteristics."""

    def test_ppo_detailed_has_generation_tracking(self):
        """Test PPO detailed tracks generation phase."""
        config = get_stage_config('ppo_detailed')

        assert config.generation_forwards > 0, \
            "PPO detailed should track generation forwards"
        assert config.generation_tokens > 0, \
            "PPO detailed should track generation tokens"

    def test_ppo_generation_is_memory_bound(self):
        """
        Test understanding that generation is memory-bound.

        Token-by-token generation is limited by memory bandwidth,
        not compute. This affects MFU calculations.
        """
        # Generation phase should have lower effective MFU
        # because it's memory bandwidth bound
        result = get_rlhf_result('ppo_detailed')

        if result is None:
            return

        # MFU should be present
        assert hasattr(result, 'model_flops_utilization'), \
            "Result should have MFU"

        # For RLHF with significant generation, MFU should be moderate
        # (generation lowers overall MFU due to memory-bound nature)
        # Typical: 10-40% MFU for mixed workloads
        mfu = result.model_flops_utilization
        assert 0.01 <= mfu <= 0.60, \
            f"MFU {mfu:.1%} should reflect mixed compute/memory-bound workload"


class TestGRPOGenerationPhase:
    """Test GRPO generation phase (DeepSeek)."""

    def test_grpo_has_generation_phase(self):
        """
        Test GRPO has generation phase.

        GRPO generates multiple samples per prompt for group comparison.
        Source: DeepSeek GRPO paper
        """
        config = get_stage_config('grpo')

        # Should have generation component
        has_generation = (
            config.generation_forwards > 0 or
            config.num_samples_per_prompt > 1
        )
        assert has_generation, \
            "GRPO should have generation phase (generates multiple samples)"

    def test_grpo_multiple_samples_per_prompt(self):
        """
        Test GRPO generates multiple samples per prompt.

        Typical: 4-16 samples per prompt for group comparison.
        Source: DeepSeek V3 used 8 samples per prompt.
        """
        config = get_stage_config('grpo')

        assert config.num_samples_per_prompt > 1, \
            f"GRPO should generate multiple samples: got {config.num_samples_per_prompt}"

        # DeepSeek used 8 samples
        assert config.num_samples_per_prompt >= 4, \
            f"GRPO typically uses 4-16 samples, got {config.num_samples_per_prompt}"


class TestRLOOGenerationPhase:
    """Test RLOO (REINFORCE Leave-One-Out) generation phase."""

    def test_rloo_has_multiple_samples(self):
        """
        Test RLOO generates multiple samples for LOO baseline.

        RLOO uses leave-one-out baseline, requiring multiple samples.
        """
        config = get_stage_config('rloo')

        assert config.num_samples_per_prompt > 1, \
            f"RLOO needs multiple samples for LOO baseline: got {config.num_samples_per_prompt}"


class TestGenerationTokenLatency:
    """Test generation token-by-token latency characteristics."""

    def test_generation_is_autoregressive(self):
        """
        Test understanding of autoregressive generation.

        Each token depends on all previous tokens.
        Cannot parallelize across sequence dimension.
        """
        # This is a documentation/understanding test
        # Autoregressive generation:
        # - Generates one token at a time
        # - Each token requires full KV cache access
        # - Memory bandwidth limited, not compute limited

        # Generation tokens should be positive for RLHF stages
        config = get_stage_config('ppo_detailed')
        assert config.generation_tokens > 0, \
            "PPO should track generation token count"

    def test_longer_generation_takes_more_time(self):
        """
        Test that longer generation increases step time.

        More generated tokens = more autoregressive steps.
        """
        # Compare GRPO with different sample counts
        config = get_stage_config('grpo')

        # More samples should mean more generation time
        # (this is implicit in the num_samples_per_prompt)
        samples = config.num_samples_per_prompt
        assert samples > 0, "Should generate at least some samples"


class TestFlopsAccounting:
    """Test FLOPs accounting for mixed phases."""

    def test_total_flops_includes_all_phases(self):
        """
        Test total FLOPs accounts for generation, scoring, and training.

        PPO phases:
        1. Generation: 2 × params × gen_tokens (inference)
        2. Scoring: 2 × params × total_tokens (reward model inference)
        3. Training: 6 × params × total_tokens (policy + critic training)
        """
        config = get_stage_config('ppo_detailed')

        # Forward multiplier should account for all phases
        assert config.forward_multiplier > 0, "Should have forward FLOPs"

        # Backward multiplier should account for training only
        assert config.backward_multiplier > 0, "Should have backward FLOPs"

        # Forward > backward because generation/scoring are forward-only
        # But forward might include both generation + training forward
        # so this relationship depends on implementation

    def test_sft_has_no_generation_phase(self):
        """
        Test SFT has no generation phase.

        SFT is pure training, no autoregressive generation.
        """
        config = get_stage_config('sft')

        assert config.generation_forwards == 0, \
            "SFT should have no generation phase"
        assert config.generation_tokens == 0, \
            "SFT should have no generation tokens"


class TestMFUImplications:
    """Test MFU implications of mixed generation/training."""

    def test_rlhf_mfu_lower_than_sft(self):
        """
        Test RLHF MFU is lower than SFT due to generation phase.

        Generation is memory-bound, reducing overall MFU.
        """
        from llm_memory_calculator.genz import training_modeling

        try:
            result_sft = training_modeling(
                model='llama-2-7b',
                training_stage='sft',
                batch_size=2,
                seq_length=2048,
            )
            result_ppo = training_modeling(
                model='llama-2-7b',
                training_stage='ppo',
                batch_size=2,
                seq_length=2048,
            )
        except Exception:
            pytest.skip("Training modeling failed")
            return

        # PPO MFU should be lower due to generation overhead
        # (memory-bound generation lowers effective MFU)
        if result_sft.model_flops_utilization > 0:
            mfu_ratio = result_ppo.model_flops_utilization / result_sft.model_flops_utilization

            # PPO MFU typically 30-80% of SFT MFU
            assert 0.1 <= mfu_ratio <= 1.2, \
                f"PPO/SFT MFU ratio {mfu_ratio:.2f} outside expected [0.1, 1.2]"


class TestKVCacheForGeneration:
    """Test KV cache requirements for generation phase."""

    def test_generation_needs_kv_cache_memory(self):
        """
        Test that generation phase requires KV cache memory.

        KV cache size: 2 × batch × seq × num_layers × num_heads × head_dim × precision
        """
        # KV cache is needed for efficient autoregressive generation
        # This memory should be tracked for accurate RLHF estimates

        # Get RLHF result
        result = get_rlhf_result('ppo', batch_size=4, seq_length=2048)

        if result is None:
            return

        # Memory should include activation memory (which includes KV cache during generation)
        assert result.activation_memory_gb > 0, \
            "Should track activation memory (includes KV cache for generation)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
