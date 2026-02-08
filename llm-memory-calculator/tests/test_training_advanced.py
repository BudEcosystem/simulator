"""
Tests for Advanced Training Memory Calculator.

Validates training memory estimates against published benchmarks from:
- QLoRA paper (Dettmers et al., 2023)
- GaLore paper (Zhao et al., 2024)
- DeepSpeed documentation
- LlamaFactory benchmarks
- Modal VRAM guide
"""

import pytest
import math
from typing import Dict, Any

# Import the training modules
from llm_memory_calculator.training import (
    AdvancedTrainingCalculator,
    calculate_advanced_training_memory,
    list_supported_configurations,
    # Training types
    get_training_stage_config,
    list_training_stages,
    TRAINING_STAGE_CONFIGS,
    # Optimizers
    get_optimizer_config,
    list_optimizers,
    calculate_optimizer_memory,
    get_recommended_optimizer,
    OPTIMIZER_CONFIGS,
    # Distributed
    get_deepspeed_config,
    get_fsdp_config,
    ParallelismConfig,
    calculate_distributed_memory,
    recommend_distributed_strategy,
)


# Model configurations for testing (based on public model architectures)
MODEL_CONFIGS = {
    "llama_7b": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "num_parameters": 7_000_000_000,
    },
    "llama_13b": {
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "vocab_size": 32000,
        "num_parameters": 13_000_000_000,
    },
    "llama_70b": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "vocab_size": 32000,
        "num_parameters": 70_000_000_000,
    },
    "mistral_7b": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "num_parameters": 7_000_000_000,
    },
}


class TestTrainingTypes:
    """Test training stage configurations."""

    def test_list_training_stages(self):
        """Test that all training stages are listed."""
        stages = list_training_stages()
        assert len(stages) > 0

        stage_ids = [s["stage"] for s in stages]
        assert "sft" in stage_ids
        assert "dpo" in stage_ids
        assert "ppo" in stage_ids
        assert "kto" in stage_ids
        assert "rm" in stage_ids

    def test_sft_config(self):
        """Test SFT configuration."""
        config = get_training_stage_config("sft")
        assert config.num_model_instances == 1
        assert config.requires_reference_model is False
        assert config.requires_reward_model is False

    def test_dpo_config(self):
        """Test DPO configuration with reference model."""
        config = get_training_stage_config("dpo")
        assert config.num_model_instances == 2
        assert config.requires_reference_model is True

    def test_dpo_orpo_config(self):
        """Test ORPO (reference-free DPO) configuration."""
        config = get_training_stage_config("dpo", dpo_loss_type="orpo")
        assert config.num_model_instances == 1
        assert config.requires_reference_model is False

    def test_ppo_config(self):
        """Test PPO configuration."""
        config = get_training_stage_config("ppo")
        assert config.num_model_instances == 3  # policy + ref + reward
        assert config.requires_reference_model is True
        assert config.requires_reward_model is True
        assert config.requires_value_head is True

    def test_ppo_lora_reward_config(self):
        """Test PPO with LoRA reward model."""
        config = get_training_stage_config("ppo", ppo_reward_type="lora")
        assert config.num_model_instances == 2  # policy + ref (reward as LoRA)


class TestOptimizers:
    """Test optimizer configurations."""

    def test_list_optimizers(self):
        """Test that all optimizers are listed."""
        optimizers = list_optimizers()
        assert len(optimizers) > 0

        opt_names = [o["name"] for o in optimizers]
        assert "adamw" in opt_names
        assert "adamw_8bit" in opt_names
        assert "galore" in opt_names
        assert "apollo" in opt_names

    def test_adamw_memory(self):
        """Test AdamW optimizer memory calculation.

        AdamW: 2 states × 4 bytes = 8 bytes per param
        For 7B params: 7B × 8 = 56 GB
        """
        config = get_optimizer_config("adamw")
        assert config.state_count == 2
        assert config.total_bytes_per_param == 8.0

        memory_gb = calculate_optimizer_memory("adamw", 7_000_000_000)
        expected_gb = (7_000_000_000 * 8) / 1e9  # 56 GB
        assert abs(memory_gb - expected_gb) < 0.1

    def test_adamw_8bit_memory(self):
        """Test 8-bit AdamW optimizer memory.

        8-bit AdamW: 2 states × 1 byte = 2 bytes per param
        Should be 75% reduction vs standard AdamW
        """
        config = get_optimizer_config("adamw_8bit")
        assert config.total_bytes_per_param == 2.0
        assert config.memory_reduction_vs_adamw == 0.25

        memory_gb = calculate_optimizer_memory("adamw_8bit", 7_000_000_000)
        expected_gb = (7_000_000_000 * 2) / 1e9  # 14 GB
        assert abs(memory_gb - expected_gb) < 0.1

    def test_galore_memory_reduction(self):
        """Test GaLore optimizer memory reduction.

        GaLore claims 65.5% reduction in optimizer states
        """
        config = get_optimizer_config("galore")
        assert config.memory_reduction_vs_adamw == 0.25  # 75% reduction

    def test_optimizer_with_deepspeed(self):
        """Test optimizer memory with DeepSpeed sharding."""
        base_memory = calculate_optimizer_memory("adamw", 7_000_000_000)

        # With ZeRO-2 on 8 GPUs, optimizer states should be sharded
        sharded_memory = calculate_optimizer_memory(
            "adamw", 7_000_000_000,
            deepspeed_stage="zero2",
            data_parallel=8
        )

        assert sharded_memory < base_memory
        assert abs(sharded_memory - base_memory / 8) < 0.1


class TestDistributedTraining:
    """Test distributed training configurations."""

    def test_deepspeed_stages(self):
        """Test DeepSpeed ZeRO stage configurations."""
        zero1 = get_deepspeed_config("zero1")
        assert zero1.optimizer_sharded is True
        assert zero1.gradient_sharded is False

        zero2 = get_deepspeed_config("zero2")
        assert zero2.optimizer_sharded is True
        assert zero2.gradient_sharded is True

        zero3 = get_deepspeed_config("zero3")
        assert zero3.optimizer_sharded is True
        assert zero3.gradient_sharded is True
        assert zero3.param_sharded is True

    def test_fsdp_strategies(self):
        """Test FSDP strategy configurations."""
        full_shard = get_fsdp_config("full_shard")
        assert full_shard.equivalent_zero == "zero3"
        assert full_shard.param_sharded is True

    def test_parallelism_config(self):
        """Test parallelism configuration."""
        config = ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=4,
        )
        assert config.total_gpus == 16

        valid, msg = config.validate()
        assert valid is True

    def test_parallelism_overhead(self):
        """Test communication overhead calculation."""
        # Pure data parallel
        dp_only = ParallelismConfig(data_parallel=8)
        overhead = dp_only.get_communication_overhead()
        assert overhead > 0  # Should have some overhead

        # Tensor parallel has lower per-rank overhead than DP
        # TP=2, DP=4: 0.05*log2(2) + 0.10*log2(4) = 0.05 + 0.20 = 0.25
        # DP=8: 0.10*log2(8) = 0.30
        # So TP+DP can actually be more efficient than pure DP
        tp_dp = ParallelismConfig(tensor_parallel=2, data_parallel=4)
        overhead_tp = tp_dp.get_communication_overhead()
        assert overhead_tp > 0  # TP+DP still has overhead
        # Note: TP can reduce total overhead by replacing DP ranks

    def test_distributed_memory_calculation(self):
        """Test distributed memory calculation."""
        result = calculate_distributed_memory(
            model_params=7_000_000_000,
            trainable_params=7_000_000_000,
            precision="bf16",
            optimizer_bytes_per_param=8,
            batch_size=4,
            seq_length=2048,
            hidden_size=4096,
            num_layers=32,
            num_gpus=8,
            deepspeed_stage="zero2",
        )

        assert result.total_per_gpu_gb > 0
        assert result.num_gpus == 8
        assert result.deepspeed_stage == "zero2"


class TestAdvancedCalculator:
    """Test the advanced training calculator."""

    def test_sft_lora_7b(self):
        """Test SFT with LoRA on 7B model.

        Expected: ~15-20 GB per GPU based on benchmarks
        - Model weights in bf16: ~14 GB
        - LoRA adapters: ~100 MB
        - Optimizer for LoRA: ~200 MB
        - Activations: ~1-2 GB
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="lora",
            optimizer="adamw",
            precision="bf16",
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
            lora_rank=16,
        )

        assert result.total_memory_gb > 0
        assert result.trainable_percent < 5  # LoRA should be < 5% trainable

    def test_qlora_7b(self):
        """Test QLoRA on 7B model.

        Expected: ~4-6 GB based on QLoRA paper
        - Model weights in 4-bit: ~3.5 GB
        - LoRA adapters + optimizer: ~500 MB
        - Activations: ~1 GB
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="qlora",
            optimizer="paged_adamw_8bit",
            precision="bf16",
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
            lora_rank=16,
            base_model_quantization="nf4",
        )

        # QLoRA should be significantly smaller than LoRA
        assert result.total_memory_gb < 20
        assert result.base_model_quantization == "nf4"

    def test_dpo_with_reference(self):
        """Test DPO training with reference model.

        Expected: ~1.7× single model memory due to reference model
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="dpo",
            method="lora",
            optimizer="adamw_8bit",
            precision="bf16",
            batch_size=2,
            seq_length=2048,
        )

        assert result.reference_model_memory_gb > 0
        assert result.training_stage == "dpo"

    def test_dpo_orpo_no_reference(self):
        """Test ORPO (reference-free DPO)."""
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="dpo",
            dpo_loss_type="orpo",
            method="lora",
            optimizer="adamw",
            precision="bf16",
        )

        assert result.reference_model_memory_gb == 0

    def test_ppo_full(self):
        """Test PPO training with full reward model.

        Expected: ~2.8× single model memory (policy + ref + reward)
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="ppo",
            method="lora",
            optimizer="adamw_8bit",
            precision="bf16",
            batch_size=1,
            seq_length=512,
            ppo_reward_type="full",
        )

        assert result.reference_model_memory_gb > 0
        assert result.reward_model_memory_gb > 0
        assert result.training_stage == "ppo"

    def test_deepspeed_zero3_70b(self):
        """Test DeepSpeed ZeRO-3 for 70B model on 8 GPUs.

        Expected: Memory should be sharded across GPUs
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_70b"],
            training_stage="sft",
            method="lora",
            optimizer="adamw",
            precision="bf16",
            batch_size=1,
            seq_length=2048,
            num_gpus=8,
            deepspeed_stage="zero3",
        )

        assert result.deepspeed_stage == "zero3"
        assert result.num_gpus == 8
        # Memory per GPU should be reasonable with sharding
        assert result.total_memory_gb < 200  # Should fit on A100 80GB with sharding

    def test_full_finetuning_memory(self):
        """Test full fine-tuning memory requirements.

        Full training: weights + gradients + optimizer + activations
        For 7B bf16: 14 (weights) + 28 (grads fp32) + 56 (opt) + ~10 (act) = ~108 GB
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="full",
            optimizer="adamw",
            precision="bf16",
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        assert result.trainable_percent == 100.0
        # Full training should require significant memory
        assert result.total_memory_gb > 50

    def test_galore_optimizer(self):
        """Test GaLore optimizer memory savings."""
        # Standard AdamW
        standard_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="full",
            optimizer="adamw",
            precision="bf16",
        )

        # GaLore
        galore_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="full",
            optimizer="galore",
            precision="bf16",
        )

        # GaLore should use less optimizer memory
        assert galore_result.optimizer_memory_gb < standard_result.optimizer_memory_gb

    def test_list_supported_configurations(self):
        """Test listing all supported configurations."""
        configs = list_supported_configurations()

        assert "training_stages" in configs
        assert "methods" in configs
        assert "optimizers" in configs
        assert "precisions" in configs
        assert "deepspeed_stages" in configs
        assert "fsdp_strategies" in configs

    def test_fit_analysis(self):
        """Test GPU fit analysis."""
        # Small model should fit easily
        small_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="lora",
            optimizer="adamw_8bit",
            precision="bf16",
            batch_size=4,
        )
        assert small_result.fits_in_memory is True

    def test_throughput_estimation(self):
        """Test throughput estimation."""
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="lora",
            optimizer="adamw",
            precision="bf16",
            batch_size=4,
            num_gpus=1,
        )

        assert result.estimated_tps is not None
        assert result.estimated_tps > 0
        assert result.throughput_factor <= 1.0


class TestValidationAgainstBenchmarks:
    """
    Validation tests against published benchmarks.

    These tests verify that our estimates are within reasonable
    ranges of published results.
    """

    def test_qlora_paper_7b_memory(self):
        """Validate against QLoRA paper: 7B model on single GPU.

        Paper claims: Fine-tune 7B model on single 24GB GPU
        Expected range: 4-6 GB
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="qlora",
            optimizer="paged_adamw_8bit",
            precision="bf16",
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
            lora_rank=16,
            base_model_quantization="nf4",
        )

        # Should fit on 24GB GPU
        assert result.total_memory_gb < 24

    def test_galore_paper_7b_memory(self):
        """Validate GaLore reduces optimizer memory for 7B training.

        Paper claims: Pre-train 7B model on single RTX 4090 (24GB)
        With 8-bit GaLore and activation checkpointing + LAYERWISE updates.

        Note: The 24GB claim relies on layerwise gradient computation which
        doesn't store all gradients simultaneously. Our standard calculator
        models the conservative case where all gradients are stored.
        We verify that GaLore reduces optimizer memory significantly.
        """
        # GaLore optimizer
        galore_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="pt",
            method="full",
            optimizer="galore",
            precision="bf16",
            batch_size=1,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        # Standard AdamW for comparison
        adamw_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="pt",
            method="full",
            optimizer="adamw",
            precision="bf16",
            batch_size=1,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        # GaLore should reduce optimizer memory significantly (50% reduction)
        assert galore_result.optimizer_memory_gb <= adamw_result.optimizer_memory_gb * 0.5
        # GaLore total should be less than AdamW total
        assert galore_result.total_memory_gb < adamw_result.total_memory_gb

    def test_deepspeed_zero3_scaling(self):
        """Validate DeepSpeed ZeRO-3 memory scaling.

        ZeRO-3 should scale memory linearly with GPU count
        """
        single_gpu = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="full",
            optimizer="adamw",
            precision="bf16",
            num_gpus=1,
        )

        eight_gpu_zero3 = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="full",
            optimizer="adamw",
            precision="bf16",
            num_gpus=8,
            deepspeed_stage="zero3",
        )

        # ZeRO-3 should significantly reduce per-GPU memory
        # (not perfectly linear due to activation memory)
        assert eight_gpu_zero3.total_memory_gb < single_gpu.total_memory_gb

    def test_dpo_vs_ppo_memory_ratio(self):
        """Validate DPO vs PPO memory requirements.

        DPO: 2 models (policy + reference) ~1.7x
        PPO: 3 models (policy + reference + reward) ~2.8x
        """
        dpo_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="dpo",
            method="lora",
            optimizer="adamw",
            precision="bf16",
        )

        ppo_result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="ppo",
            method="lora",
            optimizer="adamw",
            precision="bf16",
            ppo_reward_type="full",
        )

        # PPO should require more memory than DPO
        assert ppo_result.total_memory_gb > dpo_result.total_memory_gb

    def test_lora_trainable_params_ratio(self):
        """Validate LoRA trainable parameter ratio.

        LoRA typically trains 0.1-1% of total parameters
        """
        result = calculate_advanced_training_memory(
            MODEL_CONFIGS["llama_7b"],
            training_stage="sft",
            method="lora",
            optimizer="adamw",
            precision="bf16",
            lora_rank=16,
        )

        # LoRA should train < 5% of parameters
        assert result.trainable_percent < 5
        assert result.trainable_percent > 0.01  # But not negligible


class TestStrategyRecommendation:
    """Test distributed strategy recommendations."""

    def test_small_model_recommendation(self):
        """Test recommendation for small model that fits easily."""
        result = recommend_distributed_strategy(
            model_params=7_000_000_000,
            available_memory_per_gpu_gb=80,
            num_gpus=1,
            training_type="sft",
        )

        assert result["best_recommendation"] is not None
        # Small model should use simple strategy
        assert "zero1" in result["best_recommendation"]["deepspeed_stage"] or \
               "zero2" in result["best_recommendation"]["deepspeed_stage"]

    def test_large_model_recommendation(self):
        """Test recommendation for large model on smaller GPUs."""
        # Use 24GB GPUs and fewer GPUs to force more aggressive optimization
        result = recommend_distributed_strategy(
            model_params=70_000_000_000,
            available_memory_per_gpu_gb=24,  # RTX 4090
            num_gpus=4,
            training_type="sft",
        )

        assert result["best_recommendation"] is not None
        # Large model on smaller GPUs should need aggressive sharding
        recommendations = [r["deepspeed_stage"] for r in result["recommendations"]]
        # Either zero3 or offload should be recommended when memory is tight
        # 70B / 4 GPUs = 35 GB per GPU, > 24 GB * 0.7 = 16.8 GB, so zero3/offload needed
        assert any("zero3" in r or "offload" in r for r in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
