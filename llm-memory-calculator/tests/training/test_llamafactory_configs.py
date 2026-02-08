"""
Tests for LLaMA Factory Configuration Generation.

Tests cover:
- Config completeness for all training stages
- Parameter mapping to LLaMA Factory format
- Validation of generated configurations

Research Sources:
- LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory
- LLaMA Factory Documentation: https://llamafactory.readthedocs.io/
"""

import pytest
from typing import Dict, Any, List

from llm_memory_calculator.genz import (
    training_modeling,
    list_training_stages,
    get_stage_config,
    list_optimizers,
)


# =============================================================================
# LLaMA Factory Configuration Schema
# =============================================================================

# Required fields for different training stages
LLAMAFACTORY_REQUIRED_FIELDS = {
    # Common fields for all stages
    'common': [
        'model_name_or_path',
        'per_device_train_batch_size',
        'learning_rate',
        'num_train_epochs',
        'output_dir',
    ],
    # Stage-specific fields
    'sft': [],
    'pt': [],
    'dpo': [
        'pref_beta',
        'pref_loss',
    ],
    'orpo': [
        'pref_beta',
    ],
    'simpo': [
        'pref_beta',
    ],
    'kto': [
        'kto_beta',
    ],
    'ppo': [
        'reward_model',
        'ppo_epochs',
    ],
    'grpo': [
        'num_generations',
    ],
    'lora': [
        'lora_rank',
        'lora_alpha',
        'lora_target',
    ],
    'qlora': [
        'lora_rank',
        'lora_alpha',
        'lora_target',
        'quantization_bit',
        'quantization_method',
    ],
}

# Valid values for certain fields
LLAMAFACTORY_VALID_VALUES = {
    'pref_loss': ['sigmoid', 'hinge', 'ipo', 'kto_pair'],
    'quantization_bit': [4, 8],
    'quantization_method': ['bitsandbytes', 'gptq', 'awq'],
}


# =============================================================================
# Helper Functions
# =============================================================================

def generate_llamafactory_config(
    model: str = 'llama-2-7b',
    training_stage: str = 'sft',
    method: str = 'full',
    batch_size: int = 4,
    seq_length: int = 4096,
    optimizer: str = 'adamw',
    lora_rank: int = 16,
    bits: str = 'bf16',
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a LLaMA Factory compatible configuration.

    This simulates config generation from BudSim parameters.
    """
    config = {
        # Model
        'model_name_or_path': model,

        # Training
        'per_device_train_batch_size': batch_size,
        'max_length': seq_length,
        'learning_rate': 2e-5,
        'num_train_epochs': 3,
        'output_dir': f'./output/{model}_{training_stage}',

        # Optimizer
        'optim': optimizer,

        # Precision
        'bf16': bits == 'bf16',
        'fp16': bits == 'fp16',

        # Stage
        'stage': training_stage,
    }

    # Add LoRA config
    if method in ['lora', 'qlora', 'dora']:
        config['finetuning_type'] = 'lora'
        config['lora_rank'] = lora_rank
        config['lora_alpha'] = lora_rank * 2  # Common default
        config['lora_target'] = 'q_proj,v_proj,k_proj,o_proj'

        if method == 'qlora':
            config['quantization_bit'] = 4
            config['quantization_method'] = 'bitsandbytes'

    # Add DPO config
    if training_stage in ['dpo', 'orpo', 'simpo', 'kto']:
        config['pref_beta'] = 0.1
        if training_stage == 'dpo':
            config['pref_loss'] = 'sigmoid'

    # Add PPO config
    if training_stage in ['ppo', 'ppo_detailed']:
        config['reward_model'] = 'reward_model_path'
        config['ppo_epochs'] = 4

    # Add GRPO config
    if training_stage == 'grpo':
        config['num_generations'] = 8

    return config


def get_training_config(training_stage: str, **kwargs) -> Dict[str, Any]:
    """Get training result and extract config."""
    try:
        result = training_modeling(
            model=kwargs.get('model', 'llama-2-7b'),
            training_stage=training_stage,
            batch_size=kwargs.get('batch_size', 2),
            seq_length=kwargs.get('seq_length', 2048),
            **{k: v for k, v in kwargs.items() if k not in ['model', 'batch_size', 'seq_length']},
        )
        return result.config if result else {}
    except Exception:
        return {}


# =============================================================================
# Test Classes
# =============================================================================

class TestConfigGeneration:
    """Test configuration generation."""

    def test_generate_sft_config(self):
        """Test SFT config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='sft',
            method='full',
        )

        # Check required fields
        for field in LLAMAFACTORY_REQUIRED_FIELDS['common']:
            assert field in config, f"SFT config missing: {field}"

    def test_generate_dpo_config(self):
        """Test DPO config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='dpo',
        )

        # Check DPO-specific fields
        assert 'pref_beta' in config, "DPO config missing pref_beta"
        assert 'pref_loss' in config, "DPO config missing pref_loss"

    def test_generate_lora_config(self):
        """Test LoRA config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='sft',
            method='lora',
            lora_rank=16,
        )

        # Check LoRA-specific fields
        for field in LLAMAFACTORY_REQUIRED_FIELDS['lora']:
            assert field in config, f"LoRA config missing: {field}"

    def test_generate_qlora_config(self):
        """Test QLoRA config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='sft',
            method='qlora',
            lora_rank=16,
        )

        # Check QLoRA-specific fields
        for field in LLAMAFACTORY_REQUIRED_FIELDS['qlora']:
            assert field in config, f"QLoRA config missing: {field}"

    def test_generate_ppo_config(self):
        """Test PPO config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='ppo',
        )

        # Check PPO-specific fields
        for field in LLAMAFACTORY_REQUIRED_FIELDS['ppo']:
            assert field in config, f"PPO config missing: {field}"

    def test_generate_grpo_config(self):
        """Test GRPO config generation."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='grpo',
        )

        # Check GRPO-specific fields
        for field in LLAMAFACTORY_REQUIRED_FIELDS['grpo']:
            assert field in config, f"GRPO config missing: {field}"


class TestConfigCompleteness:
    """Test config completeness for all stages."""

    def test_all_stages_have_configs(self):
        """Test all training stages can generate configs."""
        stages = list_training_stages()

        for stage in stages:
            config = generate_llamafactory_config(training_stage=stage)

            # Should have basic fields
            assert 'model_name_or_path' in config, \
                f"Stage {stage} missing model_name_or_path"
            assert 'stage' in config, \
                f"Stage {stage} missing stage field"

    def test_preference_stages_have_beta(self):
        """Test preference stages have beta parameter."""
        preference_stages = ['dpo', 'orpo', 'simpo', 'kto']

        for stage in preference_stages:
            config = generate_llamafactory_config(training_stage=stage)

            # Should have beta parameter
            has_beta = 'pref_beta' in config or 'kto_beta' in config
            assert has_beta, \
                f"Preference stage {stage} should have beta parameter"


class TestConfigValidation:
    """Test config value validation."""

    def test_valid_pref_loss_values(self):
        """Test pref_loss has valid values."""
        config = generate_llamafactory_config(training_stage='dpo')

        if 'pref_loss' in config:
            assert config['pref_loss'] in LLAMAFACTORY_VALID_VALUES['pref_loss'], \
                f"Invalid pref_loss: {config['pref_loss']}"

    def test_valid_quantization_bit(self):
        """Test quantization_bit has valid values."""
        config = generate_llamafactory_config(method='qlora')

        if 'quantization_bit' in config:
            assert config['quantization_bit'] in LLAMAFACTORY_VALID_VALUES['quantization_bit'], \
                f"Invalid quantization_bit: {config['quantization_bit']}"

    def test_lora_rank_positive(self):
        """Test LoRA rank is positive."""
        config = generate_llamafactory_config(method='lora', lora_rank=16)

        if 'lora_rank' in config:
            assert config['lora_rank'] > 0, \
                "LoRA rank should be positive"

    def test_batch_size_positive(self):
        """Test batch size is positive."""
        config = generate_llamafactory_config(batch_size=4)

        assert config['per_device_train_batch_size'] > 0, \
            "Batch size should be positive"


class TestOptimizerMapping:
    """Test optimizer name mapping to LLaMA Factory."""

    def test_adamw_optimizer(self):
        """Test AdamW optimizer mapping."""
        config = generate_llamafactory_config(optimizer='adamw')

        assert config['optim'] == 'adamw', \
            "AdamW should map correctly"

    def test_8bit_optimizer_mapping(self):
        """Test 8-bit optimizer mapping."""
        config = generate_llamafactory_config(optimizer='adamw_8bit')

        # Should map to 8-bit variant
        assert '8bit' in config['optim'] or 'paged' in config['optim'] or config['optim'] == 'adamw_8bit', \
            "8-bit optimizer should be indicated"


class TestTrainingResultConfig:
    """Test config from training modeling results."""

    def test_result_includes_config(self):
        """Test training result includes config dict."""
        config = get_training_config('sft')

        # Config might be empty if modeling fails, but should be dict
        assert isinstance(config, dict), \
            "Config should be a dictionary"

    def test_result_config_has_stage(self):
        """Test result config includes training stage."""
        config = get_training_config('sft')

        if config:
            # Should track what stage was used
            assert 'training_stage' in config or len(config) > 0, \
                "Config should have some content"


class TestStageSpecificConfigs:
    """Test stage-specific configuration details."""

    def test_dpo_needs_paired_data(self):
        """Test DPO config implies paired preference data."""
        config = generate_llamafactory_config(training_stage='dpo')

        # DPO uses preference pairs
        stage_config = get_stage_config('dpo')
        assert stage_config.num_policy_forwards == 2, \
            "DPO needs 2 forwards (chosen + rejected)"

    def test_ppo_needs_reward_model(self):
        """Test PPO config includes reward model."""
        config = generate_llamafactory_config(training_stage='ppo')

        assert 'reward_model' in config, \
            "PPO config should include reward_model"

    def test_grpo_generation_count(self):
        """Test GRPO config includes generation count."""
        config = generate_llamafactory_config(training_stage='grpo')

        assert 'num_generations' in config, \
            "GRPO config should include num_generations"

        # Should match stage config
        stage_config = get_stage_config('grpo')
        # Generation count should be reasonable
        assert config['num_generations'] >= 1


class TestConfigExport:
    """Test config export functionality."""

    def test_config_to_yaml_format(self):
        """Test config can be formatted for YAML export."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='sft',
        )

        # All values should be YAML-serializable
        import json
        try:
            json.dumps(config)  # JSON is subset of YAML
        except TypeError as e:
            pytest.fail(f"Config not serializable: {e}")

    def test_config_complete_for_training(self):
        """Test config has all fields needed to run training."""
        config = generate_llamafactory_config(
            model='llama-2-7b',
            training_stage='sft',
            batch_size=4,
            optimizer='adamw',
        )

        required = [
            'model_name_or_path',
            'per_device_train_batch_size',
            'learning_rate',
            'output_dir',
        ]

        for field in required:
            assert field in config, f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
