"""
Regression tests for verified bug fixes in the training calculator subsystem.

Each test corresponds to a specific bug fix that was validated and merged.
These tests ensure the fixes do not regress in future changes.

Bug fixes covered:
1. ZeRO-1 optimizer state sharding (was missing, added to calculator)
2. LoRA param consistency between basic and advanced calculators
3. DPO reference model memory (was 0.8x, corrected to 1.0x)
4. DPO activation doubling for pairwise data
5. FP8 precision support (1 byte per param)
6. Adafactor factorized memory (~1 byte/param, not 4)
7. Pipeline bubble 1F1B formula (was flat 15%, now (pp-1)/(pp+m-1))
8. Sequence parallelism activation reduction
9. GaLore effective_params cap at 100% (was 50%)
10. Gradient checkpointing recompute FLOPs == forward FLOPs
"""

import math
import pytest
from typing import Dict, Any

from llm_memory_calculator.training import (
    TrainingMemoryCalculator,
    AdvancedTrainingCalculator,
    OPTIMIZER_CONFIGS,
    ParallelismConfig,
)
from llm_memory_calculator.training.optimizers import OptimizerCategory
from llm_memory_calculator.genz.LLM_training.training_modeling import (
    calculate_total_training_flops,
)


# =============================================================================
# Shared model config fixture
# =============================================================================

LLAMA_8B_CONFIG: Dict[str, Any] = {
    'hidden_size': 4096,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
    'intermediate_size': 14336,
    'vocab_size': 128256,
    'max_position_embeddings': 8192,
    'hidden_act': 'silu',
    'num_parameters': 8_030_261_248,
}


@pytest.fixture
def llama_config() -> Dict[str, Any]:
    """Return a copy of the Llama-8B config to avoid cross-test mutation."""
    return dict(LLAMA_8B_CONFIG)


@pytest.fixture
def basic_calc() -> TrainingMemoryCalculator:
    """Return a TrainingMemoryCalculator instance."""
    return TrainingMemoryCalculator()


@pytest.fixture
def advanced_calc() -> AdvancedTrainingCalculator:
    """Return an AdvancedTrainingCalculator instance."""
    return AdvancedTrainingCalculator()


# =============================================================================
# 1. ZeRO-1 optimizer sharding
# =============================================================================

class TestZero1OptimizerSharding:
    """ZeRO-1 should shard optimizer states across DP ranks."""

    def test_zero1_optimizer_sharding(self, basic_calc, llama_config):
        """ZeRO-1 with DP=8 should reduce optimizer memory by ~8x vs no ZeRO."""
        estimate_no_zero = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            deepspeed_stage=None,
            data_parallel=8,
            gradient_checkpointing=True,
        )

        estimate_zero1 = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            deepspeed_stage='zero1',
            data_parallel=8,
            gradient_checkpointing=True,
        )

        # ZeRO-1 optimizer memory must be strictly less than unsharded
        assert estimate_zero1.optimizer_memory_gb < estimate_no_zero.optimizer_memory_gb, (
            f"ZeRO-1 optimizer memory ({estimate_zero1.optimizer_memory_gb:.2f} GB) "
            f"should be less than no-ZeRO ({estimate_no_zero.optimizer_memory_gb:.2f} GB)"
        )

        # Should be roughly 8x reduction (within tolerance for rounding)
        ratio = estimate_no_zero.optimizer_memory_gb / estimate_zero1.optimizer_memory_gb
        assert 7.0 < ratio < 9.0, (
            f"Expected ~8x reduction, got {ratio:.2f}x "
            f"(no_zero={estimate_no_zero.optimizer_memory_gb:.2f} GB, "
            f"zero1={estimate_zero1.optimizer_memory_gb:.2f} GB)"
        )

    def test_zero1_only_shards_optimizer_not_gradients(self, basic_calc, llama_config):
        """ZeRO-1 should only shard optimizer states, NOT gradients."""
        estimate_no_zero = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            deepspeed_stage=None,
            data_parallel=4,
            gradient_checkpointing=True,
        )

        estimate_zero1 = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            deepspeed_stage='zero1',
            data_parallel=4,
            gradient_checkpointing=True,
        )

        # Gradient memory should remain the same (ZeRO-1 does NOT shard gradients)
        assert abs(estimate_zero1.gradient_memory_gb - estimate_no_zero.gradient_memory_gb) < 0.01, (
            f"ZeRO-1 should not change gradient memory: "
            f"zero1={estimate_zero1.gradient_memory_gb:.4f} vs "
            f"no_zero={estimate_no_zero.gradient_memory_gb:.4f}"
        )


# =============================================================================
# 2. LoRA calculators consistency
# =============================================================================

class TestLoraCalculatorsConsistent:
    """Basic and advanced calculators should agree on LoRA trainable params."""

    def test_lora_calculators_consistent(self, basic_calc, advanced_calc, llama_config):
        """Both calculators should produce LoRA trainable params within 5%."""
        basic_result = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            gradient_checkpointing=True,
        )

        advanced_result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='sft',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        basic_params = basic_result.trainable_params
        advanced_params = advanced_result.trainable_params

        # Both must be positive
        assert basic_params > 0, "Basic calculator returned 0 trainable params"
        assert advanced_params > 0, "Advanced calculator returned 0 trainable params"

        # Within 5% of each other
        max_params = max(basic_params, advanced_params)
        relative_diff = abs(basic_params - advanced_params) / max_params
        assert relative_diff < 0.05, (
            f"LoRA trainable params differ by {relative_diff*100:.1f}%: "
            f"basic={basic_params:,} vs advanced={advanced_params:,}"
        )

    def test_lora_calculators_consistent_rank_64(self, basic_calc, advanced_calc, llama_config):
        """Consistency check at higher rank (64)."""
        basic_result = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='lora',
            lora_rank=64,
            optimizer='adamw',
            gradient_checkpointing=True,
        )

        advanced_result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='sft',
            method='lora',
            lora_rank=64,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        max_params = max(basic_result.trainable_params, advanced_result.trainable_params)
        relative_diff = abs(basic_result.trainable_params - advanced_result.trainable_params) / max_params
        assert relative_diff < 0.05, (
            f"LoRA r=64 trainable params differ by {relative_diff*100:.1f}%: "
            f"basic={basic_result.trainable_params:,} vs "
            f"advanced={advanced_result.trainable_params:,}"
        )


# =============================================================================
# 3. DPO reference model uses full weight memory
# =============================================================================

class TestDpoReferenceModelFullMemory:
    """DPO reference model should use full weight memory (not 0.8x)."""

    def test_reference_model_full_memory(self, advanced_calc, llama_config):
        """Reference model memory should equal the policy weight memory."""
        result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='dpo',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        # Reference model memory should equal weight memory (same model in eval mode)
        assert result.reference_model_memory_gb > 0, (
            "DPO must have non-zero reference model memory"
        )
        # The reference model is the same architecture, loaded in eval mode with
        # the same weight footprint as the policy model's weight memory.
        assert abs(result.reference_model_memory_gb - result.weight_memory_gb) < 0.5, (
            f"Reference model memory ({result.reference_model_memory_gb:.2f} GB) "
            f"should equal weight memory ({result.weight_memory_gb:.2f} GB), not 0.8x"
        )

    def test_sft_has_no_reference_model(self, advanced_calc, llama_config):
        """SFT should NOT have reference model memory."""
        result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='sft',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )
        assert result.reference_model_memory_gb == 0.0, (
            f"SFT should have 0 reference model memory, got {result.reference_model_memory_gb:.2f} GB"
        )


# =============================================================================
# 4. DPO activation doubling for pairwise data
# =============================================================================

class TestDpoActivationDoubled:
    """DPO should use 2x effective batch size for activations (pairwise data)."""

    def test_dpo_activation_doubled(self, advanced_calc, llama_config):
        """DPO activation memory should be ~2x SFT activation memory."""
        sft_result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='sft',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        dpo_result = advanced_calc.calculate_advanced_training_memory(
            config=llama_config,
            training_stage='dpo',
            method='lora',
            lora_rank=16,
            optimizer='adamw',
            precision='bf16',
            batch_size=4,
            seq_length=2048,
            gradient_checkpointing=True,
        )

        ratio = dpo_result.activation_memory_gb / sft_result.activation_memory_gb
        assert 1.8 < ratio < 2.2, (
            f"DPO activation should be ~2x SFT activation. "
            f"Got ratio {ratio:.2f} "
            f"(DPO={dpo_result.activation_memory_gb:.2f} GB, "
            f"SFT={sft_result.activation_memory_gb:.2f} GB)"
        )


# =============================================================================
# 5. FP8 precision supported
# =============================================================================

class TestFp8PrecisionSupported:
    """FP8 should be recognized and use 1 byte per parameter."""

    def test_fp8_in_basic_calculator(self):
        """TrainingMemoryCalculator.PRECISION_BYTES should have fp8 == 1."""
        assert 'fp8' in TrainingMemoryCalculator.PRECISION_BYTES, (
            "fp8 not found in TrainingMemoryCalculator.PRECISION_BYTES"
        )
        assert TrainingMemoryCalculator.PRECISION_BYTES['fp8'] == 1, (
            f"fp8 should be 1 byte, got {TrainingMemoryCalculator.PRECISION_BYTES['fp8']}"
        )

    def test_fp8_in_advanced_calculator(self):
        """AdvancedTrainingCalculator.PRECISION_BYTES should have fp8 == 1."""
        assert 'fp8' in AdvancedTrainingCalculator.PRECISION_BYTES, (
            "fp8 not found in AdvancedTrainingCalculator.PRECISION_BYTES"
        )
        assert AdvancedTrainingCalculator.PRECISION_BYTES['fp8'] == 1, (
            f"fp8 should be 1 byte, got {AdvancedTrainingCalculator.PRECISION_BYTES['fp8']}"
        )

    def test_fp8_weight_memory_is_half_bf16(self, basic_calc, llama_config):
        """FP8 weight memory should be half of BF16 (1 byte vs 2 bytes)."""
        # fp8 does not trigger fp32 master copy (precision not in the fp16/bf16 set)
        estimate_fp8 = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='fp8',
            method='full',
            optimizer='adamw',
            gradient_checkpointing=True,
        )
        # fp8: 8B params * 1 byte = 8 GB (no fp32 master for fp8 precision)
        estimate_fp32 = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='fp32',
            method='full',
            optimizer='adamw',
            gradient_checkpointing=True,
        )
        # fp8 weights should be 1/4 of fp32 weights (1 byte vs 4 bytes)
        ratio = estimate_fp32.weight_memory_gb / estimate_fp8.weight_memory_gb
        assert 3.5 < ratio < 4.5, (
            f"fp8 weight memory should be ~1/4 of fp32. Got ratio {ratio:.2f} "
            f"(fp32={estimate_fp32.weight_memory_gb:.2f}, fp8={estimate_fp8.weight_memory_gb:.2f})"
        )


# =============================================================================
# 6. Adafactor factorized memory
# =============================================================================

class TestAdafactorMemoryFactorized:
    """Adafactor should use ~1 byte/param due to factorization, not 4."""

    def test_adafactor_total_bytes_per_param(self):
        """Adafactor config should specify ~1 byte/param."""
        assert 'adafactor' in OPTIMIZER_CONFIGS, "adafactor not in OPTIMIZER_CONFIGS"
        adafactor = OPTIMIZER_CONFIGS['adafactor']
        assert adafactor.total_bytes_per_param == 1.0, (
            f"Adafactor should use 1.0 byte/param (factorized), "
            f"got {adafactor.total_bytes_per_param}"
        )

    def test_adafactor_less_than_half_adamw(self, basic_calc, llama_config):
        """Adafactor optimizer memory should be less than half of AdamW."""
        estimate_adamw = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            gradient_checkpointing=True,
        )

        estimate_adafactor = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adafactor',
            gradient_checkpointing=True,
        )

        assert estimate_adafactor.optimizer_memory_gb < estimate_adamw.optimizer_memory_gb / 2, (
            f"Adafactor optimizer memory ({estimate_adafactor.optimizer_memory_gb:.2f} GB) "
            f"should be less than half of AdamW ({estimate_adamw.optimizer_memory_gb:.2f} GB)"
        )

    def test_adafactor_approximately_one_eighth_adamw(self, basic_calc, llama_config):
        """Adafactor (1 byte/param) vs AdamW (8 bytes/param) = 1/8 ratio."""
        estimate_adamw = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adamw',
            gradient_checkpointing=True,
        )

        estimate_adafactor = basic_calc.calculate_training_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            method='full',
            optimizer='adafactor',
            gradient_checkpointing=True,
        )

        ratio = estimate_adamw.optimizer_memory_gb / estimate_adafactor.optimizer_memory_gb
        assert 7.0 < ratio < 9.0, (
            f"AdamW/Adafactor optimizer ratio should be ~8x, got {ratio:.2f}x "
            f"(adamw={estimate_adamw.optimizer_memory_gb:.2f}, "
            f"adafactor={estimate_adafactor.optimizer_memory_gb:.2f})"
        )


# =============================================================================
# 7. Pipeline bubble 1F1B formula
# =============================================================================

class TestPipelineBubble1f1bFormula:
    """Pipeline bubble should follow 1F1B formula: (pp-1)/(pp+m-1)."""

    def test_bubble_varies_with_pp(self):
        """Bubble fraction should increase with more pipeline stages."""
        parallel_4 = ParallelismConfig(pipeline_parallel=4)
        parallel_8 = ParallelismConfig(pipeline_parallel=8)

        overhead_4 = parallel_4.get_communication_overhead()
        overhead_8 = parallel_8.get_communication_overhead()

        # pp=8 should have higher bubble than pp=4
        assert overhead_8 > overhead_4, (
            f"PP=8 overhead ({overhead_8:.4f}) should exceed PP=4 overhead ({overhead_4:.4f})"
        )

    def test_bubble_pp4_matches_1f1b(self):
        """PP=4: bubble = (4-1)/(4+m-1) with m=4*4=16 -> 3/19 ~ 0.158."""
        parallel = ParallelismConfig(pipeline_parallel=4)
        overhead = parallel.get_communication_overhead()

        # The implementation uses m = 4 * pp (standard micro-batch count)
        pp = 4
        m = 4 * pp  # = 16
        expected_bubble = (pp - 1) / (pp + m - 1)  # 3/19 = 0.1579

        # Overhead should include the bubble fraction
        assert abs(overhead - expected_bubble) < 0.01, (
            f"PP=4 overhead ({overhead:.4f}) should match 1F1B bubble "
            f"({expected_bubble:.4f}) with m={m}"
        )

    def test_bubble_pp8_not_flat_15_percent(self):
        """PP=8 bubble should NOT be a flat 15%."""
        parallel = ParallelismConfig(pipeline_parallel=8)
        overhead = parallel.get_communication_overhead()

        pp = 8
        m = 4 * pp  # = 32
        expected_bubble = (pp - 1) / (pp + m - 1)  # 7/39 = 0.1795

        # The 1F1B formula gives ~17.9% for pp=8, NOT flat 15%
        assert abs(overhead - 0.15) > 0.01, (
            f"PP=8 overhead ({overhead:.4f}) should NOT be flat 15% "
            f"(expected 1F1B: {expected_bubble:.4f})"
        )
        assert abs(overhead - expected_bubble) < 0.01, (
            f"PP=8 overhead ({overhead:.4f}) should match 1F1B formula "
            f"({expected_bubble:.4f})"
        )

    def test_bubble_pp2_small(self):
        """PP=2: bubble = (2-1)/(2+8-1) = 1/9 ~ 0.111."""
        parallel = ParallelismConfig(pipeline_parallel=2)
        overhead = parallel.get_communication_overhead()

        pp = 2
        m = 4 * pp  # = 8
        expected_bubble = (pp - 1) / (pp + m - 1)  # 1/9 = 0.1111

        assert abs(overhead - expected_bubble) < 0.01, (
            f"PP=2 overhead ({overhead:.4f}) should match 1F1B formula "
            f"({expected_bubble:.4f})"
        )


# =============================================================================
# 8. Sequence parallelism activation reduction
# =============================================================================

class TestSequenceParallelActivationReduction:
    """SP should divide all activation terms by TP degree."""

    def test_sp_reduces_activation_by_tp(self, basic_calc, llama_config):
        """With SP + TP=4, activation memory should be ~1/4 of without SP."""
        # Without SP (TP=1, no sequence parallel)
        act_no_sp = basic_calc._calculate_activation_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            gradient_checkpointing=True,
            tensor_parallel=1,
            flash_attention=True,
            sequence_parallel=False,
        )

        # With SP (TP=4, sequence parallel enabled)
        act_with_sp = basic_calc._calculate_activation_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            gradient_checkpointing=True,
            tensor_parallel=4,
            flash_attention=True,
            sequence_parallel=True,
        )

        # SP divides ALL activation terms by TP degree
        # TP alone only divides the TP-parallelized terms, so SP should
        # provide additional reduction on top of TP
        ratio = act_no_sp / act_with_sp
        # Expect close to 4x reduction since SP divides the full activation
        # by the TP degree (non-TP terms also get divided)
        assert ratio > 3.0, (
            f"SP+TP=4 should reduce activation by ~4x, got {ratio:.2f}x "
            f"(no_sp={act_no_sp:.4f} GB, with_sp={act_with_sp:.4f} GB)"
        )

    def test_sp_requires_tp_greater_than_1(self, basic_calc, llama_config):
        """SP with TP=1 should have no effect (no parallelism to exploit)."""
        act_no_sp = basic_calc._calculate_activation_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            gradient_checkpointing=True,
            tensor_parallel=1,
            flash_attention=True,
            sequence_parallel=False,
        )

        act_sp_tp1 = basic_calc._calculate_activation_memory(
            config=llama_config,
            batch_size=4,
            seq_length=2048,
            precision='bf16',
            gradient_checkpointing=True,
            tensor_parallel=1,
            flash_attention=True,
            sequence_parallel=True,
        )

        # SP with TP=1 should not change anything
        assert abs(act_no_sp - act_sp_tp1) < 0.001, (
            f"SP with TP=1 should have no effect: "
            f"no_sp={act_no_sp:.4f} vs sp_tp1={act_sp_tp1:.4f}"
        )


# =============================================================================
# 9. GaLore effective_params capped at 100%
# =============================================================================

class TestGaloreCapAtFullParams:
    """GaLore effective_params should be capped at 100%, not 50%."""

    def test_galore_config_exists(self):
        """GaLore should be in OPTIMIZER_CONFIGS."""
        assert 'galore' in OPTIMIZER_CONFIGS, "galore not in OPTIMIZER_CONFIGS"

    def test_galore_is_low_rank_category(self):
        """GaLore should be categorized as LOW_RANK."""
        galore = OPTIMIZER_CONFIGS['galore']
        assert galore.category == OptimizerCategory.LOW_RANK, (
            f"GaLore should be LOW_RANK, got {galore.category}"
        )

    def test_galore_cap_100_percent(self):
        """GaLore calculate_memory_gb should cap effective_params at 100%."""
        galore = OPTIMIZER_CONFIGS['galore']

        trainable_params = 1_000_000
        default_rank = galore.extra_params.get('default_rank', 16)

        # With a very large rank (e.g., 256), effective_params would exceed 100%
        # without the cap: rank / default_rank = 256 / 16 = 16x -> capped to 1.0
        large_rank = default_rank * 20  # 320
        memory_large_rank = galore.calculate_memory_gb(trainable_params, rank=large_rank)

        # Max memory should be trainable_params * bytes_per_param (capped at 1.0x)
        max_memory = (trainable_params * galore.total_bytes_per_param) / 1e9
        assert memory_large_rank <= max_memory + 1e-12, (
            f"GaLore memory ({memory_large_rank:.6f} GB) should be capped at "
            f"{max_memory:.6f} GB (100% of trainable params)"
        )

    def test_galore_not_capped_at_50_percent(self):
        """GaLore with rank > default_rank should exceed 50% of full memory."""
        galore = OPTIMIZER_CONFIGS['galore']

        trainable_params = 10_000_000
        default_rank = galore.extra_params.get('default_rank', 16)

        # With rank == default_rank, effective_params == trainable_params * 1.0
        memory_at_default = galore.calculate_memory_gb(trainable_params, rank=default_rank)
        full_memory = (trainable_params * galore.total_bytes_per_param) / 1e9

        # At default rank, memory should be at or near 100% (not capped at 50%)
        ratio = memory_at_default / full_memory
        assert ratio > 0.9, (
            f"GaLore at default rank should use ~100% of trainable params memory, "
            f"got {ratio*100:.1f}%"
        )


# =============================================================================
# 10. Gradient checkpointing recompute FLOPs == forward FLOPs
# =============================================================================

class TestRecomputeFlopsFullForward:
    """Gradient checkpointing recompute FLOPs should equal total forward FLOPs."""

    def test_recompute_equals_forward(self):
        """With gradient_checkpointing=True, recompute FLOPs == forward FLOPs."""
        result = calculate_total_training_flops(
            batch_size=4,
            seq_length=2048,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            vocab_size=128256,
            num_heads=32,
            num_kv_heads=8,
            gradient_checkpointing=True,
        )

        assert result['recompute'] == result['forward'], (
            f"Recompute FLOPs ({result['recompute']:,}) should equal "
            f"forward FLOPs ({result['forward']:,}) with gradient checkpointing"
        )

    def test_no_recompute_without_checkpointing(self):
        """Without gradient_checkpointing, recompute FLOPs should be 0."""
        result = calculate_total_training_flops(
            batch_size=4,
            seq_length=2048,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            vocab_size=128256,
            num_heads=32,
            num_kv_heads=8,
            gradient_checkpointing=False,
        )

        assert result['recompute'] == 0, (
            f"Recompute FLOPs should be 0 without checkpointing, "
            f"got {result['recompute']:,}"
        )

    def test_total_is_4f_with_checkpointing(self):
        """With checkpointing, total = F + F + 2F = 4F (approx, depends on backward ratio)."""
        result = calculate_total_training_flops(
            batch_size=4,
            seq_length=2048,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            vocab_size=128256,
            num_heads=32,
            num_kv_heads=8,
            gradient_checkpointing=True,
        )

        # total = forward + backward + recompute
        assert result['total'] == result['forward'] + result['backward'] + result['recompute'], (
            "total FLOPs should be sum of forward + backward + recompute"
        )

        # With checkpointing: recompute == forward, so total = 2*forward + backward
        # Backward is typically ~2x forward (for linear layers), so total ~ 4*forward
        expected_total = result['forward'] + result['forward'] + result['backward']
        assert result['total'] == expected_total, (
            f"With checkpointing: total ({result['total']:,}) should be "
            f"forward ({result['forward']:,}) + recompute ({result['recompute']:,}) + "
            f"backward ({result['backward']:,}) = {expected_total:,}"
        )

    def test_checkpointing_increases_total_flops(self):
        """Gradient checkpointing should increase total FLOPs (trade compute for memory)."""
        result_no_gc = calculate_total_training_flops(
            batch_size=4,
            seq_length=2048,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            vocab_size=128256,
            num_heads=32,
            num_kv_heads=8,
            gradient_checkpointing=False,
        )

        result_gc = calculate_total_training_flops(
            batch_size=4,
            seq_length=2048,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            vocab_size=128256,
            num_heads=32,
            num_kv_heads=8,
            gradient_checkpointing=True,
        )

        assert result_gc['total'] > result_no_gc['total'], (
            f"Checkpointing total FLOPs ({result_gc['total']:,}) should exceed "
            f"non-checkpointing ({result_no_gc['total']:,})"
        )

        # The additional cost should equal forward FLOPs (recomputation)
        flops_difference = result_gc['total'] - result_no_gc['total']
        assert flops_difference == result_gc['forward'], (
            f"FLOP increase ({flops_difference:,}) should equal "
            f"forward FLOPs ({result_gc['forward']:,})"
        )


# =============================================================================
# 11. GenZ LoRA formula: GQA-aware, 7 targets, DoRA magnitude
# =============================================================================

class TestGenZLoRAFormula:
    """GenZ LoRA formula should match the basic calculator formula."""

    def test_genz_lora_gqa_aware(self):
        """GenZ _calculate_trainable_params should produce fewer params for GQA than MHA."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        lora_rank = 16
        num_layers = 32
        hidden_size = 4096
        intermediate_size = 14336
        total_params = 8_000_000_000

        # MHA: num_kv_heads == num_heads
        mha_params = _calculate_trainable_params(
            total_params, 'lora', lora_rank, num_layers, hidden_size, intermediate_size,
            num_attention_heads=32, num_key_value_heads=32,
        )

        # GQA: num_kv_heads < num_heads (Llama-3 style)
        gqa_params = _calculate_trainable_params(
            total_params, 'lora', lora_rank, num_layers, hidden_size, intermediate_size,
            num_attention_heads=32, num_key_value_heads=8,
        )

        assert gqa_params < mha_params, (
            f"GQA LoRA params ({gqa_params:,}) should be less than MHA ({mha_params:,})"
        )

    def test_genz_lora_matches_basic_calculator(self, basic_calc, llama_config):
        """GenZ LoRA formula should match basic calculator within 1%."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        lora_rank = 16

        # GenZ formula
        genz_params = _calculate_trainable_params(
            LLAMA_8B_CONFIG['num_parameters'], 'lora', lora_rank,
            LLAMA_8B_CONFIG['num_hidden_layers'],
            LLAMA_8B_CONFIG['hidden_size'],
            LLAMA_8B_CONFIG['intermediate_size'],
            num_attention_heads=LLAMA_8B_CONFIG['num_attention_heads'],
            num_key_value_heads=LLAMA_8B_CONFIG['num_key_value_heads'],
        )

        # Basic calculator formula
        basic_params = basic_calc._calculate_trainable_params(
            llama_config, 'lora', lora_rank,
            freeze_layers=0,
            total_params=LLAMA_8B_CONFIG['num_parameters'],
        )

        ratio = genz_params / basic_params if basic_params > 0 else float('inf')
        assert 0.99 <= ratio <= 1.01, (
            f"GenZ LoRA params ({genz_params:,}) should match basic calc ({basic_params:,}), "
            f"ratio={ratio:.4f}"
        )

    def test_genz_lora_7_targets(self):
        """LoRA should target all 7 projections: q, k, v, o, gate, up, down."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        lora_rank = 16
        hidden_size = 4096
        intermediate_size = 14336
        num_layers = 1  # Single layer for clarity
        num_heads = 32
        num_kv_heads = 32  # MHA for simplicity
        total_params = 1_000_000_000

        result = _calculate_trainable_params(
            total_params, 'lora', lora_rank, num_layers, hidden_size, intermediate_size,
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads,
        )

        # Expected: 4 attn targets × r×(d_in+d_out) + 3 mlp targets × r×(h+ffn)
        expected_attn = 4 * lora_rank * (hidden_size + hidden_size)
        expected_mlp = 3 * lora_rank * (hidden_size + intermediate_size)
        expected = num_layers * (expected_attn + expected_mlp)

        assert result == expected, (
            f"LoRA params ({result:,}) should equal 7-target formula ({expected:,})"
        )

    def test_genz_dora_magnitude_vectors(self):
        """DoRA should add magnitude vectors on top of LoRA params."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        lora_rank = 16
        hidden_size = 4096
        intermediate_size = 14336
        total_params = 8_000_000_000

        lora_params = _calculate_trainable_params(
            total_params, 'lora', lora_rank, 32, hidden_size, intermediate_size,
            num_attention_heads=32, num_key_value_heads=8,
        )

        dora_params = _calculate_trainable_params(
            total_params, 'dora', lora_rank, 32, hidden_size, intermediate_size,
            num_attention_heads=32, num_key_value_heads=8,
        )

        assert dora_params > lora_params, (
            f"DoRA ({dora_params:,}) should have more params than LoRA ({lora_params:,})"
        )


class TestGenZParallelizationLoRA:
    """training_parallelization.py LoRA formula should be GQA-aware."""

    def test_parallelization_gqa_attention(self):
        """_estimate_memory_per_gpu should use GQA-aware attention params."""
        from types import SimpleNamespace
        from llm_memory_calculator.genz.LLM_training.training_parallelization import (
            _estimate_memory_per_gpu,
            TrainingParallelismConfig,
        )

        config = TrainingParallelismConfig(
            tensor_parallel=1, pipeline_parallel=1,
            data_parallel=1, expert_parallel=1,
        )

        # MHA model
        mha_model = SimpleNamespace(
            num_decoder_layers=32, hidden_size=4096,
            intermediate_size=14336, vocab_size=128256,
            num_attention_heads=32, num_key_value_heads=32,
        )

        # GQA model (same except KV heads)
        gqa_model = SimpleNamespace(
            num_decoder_layers=32, hidden_size=4096,
            intermediate_size=14336, vocab_size=128256,
            num_attention_heads=32, num_key_value_heads=8,
        )

        mha_mem = _estimate_memory_per_gpu(mha_model, config, 1, 512, 'lora', 'adamw', 'bf16')
        gqa_mem = _estimate_memory_per_gpu(gqa_model, config, 1, 512, 'lora', 'adamw', 'bf16')

        assert gqa_mem < mha_mem, (
            f"GQA LoRA memory ({gqa_mem:.2f} GB) should be less than MHA ({mha_mem:.2f} GB)"
        )


# =============================================================================
# 12. ZeRO-1 in optimizers.py calculate_optimizer_memory
# =============================================================================

class TestZero1InOptimizerFunction:
    """calculate_optimizer_memory should shard for ZeRO-1."""

    def test_zero1_shards_optimizer_state(self):
        """ZeRO-1 with DP=8 should reduce optimizer memory by 8x."""
        from llm_memory_calculator.training.optimizers import calculate_optimizer_memory

        trainable_params = 8_000_000_000

        no_zero = calculate_optimizer_memory('adamw', trainable_params, data_parallel=1)
        zero1 = calculate_optimizer_memory(
            'adamw', trainable_params, deepspeed_stage='zero1', data_parallel=8,
        )

        ratio = no_zero / zero1
        assert 7.9 <= ratio <= 8.1, (
            f"ZeRO-1 DP=8 should reduce by 8x, got ratio={ratio:.2f}"
        )

    def test_zero1_vs_zero2_same_reduction(self):
        """ZeRO-1 and ZeRO-2 should give the same optimizer state reduction."""
        from llm_memory_calculator.training.optimizers import calculate_optimizer_memory

        trainable_params = 8_000_000_000
        dp = 4

        zero1 = calculate_optimizer_memory(
            'adamw', trainable_params, deepspeed_stage='zero1', data_parallel=dp,
        )
        zero2 = calculate_optimizer_memory(
            'adamw', trainable_params, deepspeed_stage='zero2', data_parallel=dp,
        )

        assert zero1 == zero2, (
            f"ZeRO-1 ({zero1:.4f} GB) should equal ZeRO-2 ({zero2:.4f} GB) "
            f"for optimizer state sharding"
        )


# =============================================================================
# 13. Sequence parallelism: no double-division of TP terms
# =============================================================================

class TestSequenceParallelNoDblDiv:
    """SP should divide ALL terms by T once, not divide TP terms by T twice."""

    def test_sp_reduces_all_terms_equally(self, basic_calc, llama_config):
        """With SP, non-TP terms get divided by T (not just TP terms)."""
        no_sp = basic_calc._calculate_activation_memory(
            llama_config, batch_size=1, seq_length=2048, precision='bf16',
            gradient_checkpointing=False, tensor_parallel=4,
            flash_attention=True, sequence_parallel=False,
        )
        with_sp = basic_calc._calculate_activation_memory(
            llama_config, batch_size=1, seq_length=2048, precision='bf16',
            gradient_checkpointing=False, tensor_parallel=4,
            flash_attention=True, sequence_parallel=True,
        )

        # Without SP: non_tp terms are full, tp terms are /T
        # With SP: ALL terms are /T
        # So ratio should be > 1 but less than T
        ratio = no_sp / with_sp
        assert 1.0 < ratio < 4.0, (
            f"SP reduction ratio should be between 1 and T=4, got {ratio:.3f}"
        )

    def test_sp_tp_terms_not_double_divided(self, basic_calc, llama_config):
        """TP-parallelized terms should be divided by T once, never T^2."""
        # With T=8, if TP terms were double-divided, SP activation would be
        # unreasonably small
        no_sp = basic_calc._calculate_activation_memory(
            llama_config, batch_size=1, seq_length=2048, precision='bf16',
            gradient_checkpointing=False, tensor_parallel=8,
            flash_attention=True, sequence_parallel=False,
        )
        with_sp = basic_calc._calculate_activation_memory(
            llama_config, batch_size=1, seq_length=2048, precision='bf16',
            gradient_checkpointing=False, tensor_parallel=8,
            flash_attention=True, sequence_parallel=True,
        )

        # SP should reduce by a factor between 1 and T.
        # If TP terms were double-divided, the reduction would be much larger
        ratio = no_sp / with_sp
        assert ratio < 8.0, (
            f"SP reduction ratio ({ratio:.3f}) should be < T=8 "
            f"(double-division would give ratio >> T)"
        )


# =============================================================================
# 14. Hopper Flops: BF16 dense convention consistency
# =============================================================================

class TestHopperFlopsConvention:
    """All Hopper-class GPUs should use BF16 dense TFLOPS (not TF32)."""

    def test_h100_bf16_dense(self):
        """H100 SXM should report 1979 BF16 dense TFLOPS."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        assert HARDWARE_CONFIGS['H100_GPU']['Flops'] == 1979

    def test_h200_bf16_dense(self):
        """H200 should report 1979 BF16 dense TFLOPS (same die as H100 SXM)."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        assert HARDWARE_CONFIGS['H200_GPU']['Flops'] == 1979

    def test_gh200_bf16_dense(self):
        """GH200 should report 1979 BF16 dense TFLOPS."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        assert HARDWARE_CONFIGS['GH200_GPU']['Flops'] == 1979

    def test_h100_pcie_bf16_dense(self):
        """H100 PCIe should report 1513 BF16 dense TFLOPS."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        assert HARDWARE_CONFIGS['H100_PCIe_GPU']['Flops'] == 1513

    def test_b200_bf16_consistent(self):
        """B200 BF16 dense TFLOPS should be 2250 (already correct)."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        assert HARDWARE_CONFIGS['B200_GPU']['Flops'] == 2250

    def test_all_hopper_consistent(self):
        """All Hopper-class GPUs should use the same Flops convention."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        h100_sxm = HARDWARE_CONFIGS['H100_GPU']['Flops']
        h200 = HARDWARE_CONFIGS['H200_GPU']['Flops']
        gh200 = HARDWARE_CONFIGS['GH200_GPU']['Flops']
        # H100 SXM, H200, and GH200 share the same GPU die
        assert h100_sxm == h200 == gh200, (
            f"H100 SXM ({h100_sxm}), H200 ({h200}), GH200 ({gh200}) "
            f"should all use same BF16 dense value"
        )


# =============================================================================
# 15. LoRA parameter counts validated against known reference values
# =============================================================================

class TestLoRAAgainstReferenceValues:
    """Validate LoRA parameter counts against externally confirmed values.

    Sources:
    - Llama-3-8B rank=32 all targets: 83,886,080 (AWS Neuron docs)
    - Llama-2-7B rank=8 q+v only: 4,194,304 (Intel guide, Levanter docs)
    """

    def test_llama3_8b_rank32_all_targets(self):
        """Llama-3-8B with rank=32, all 7 targets = 83,886,080 (AWS Neuron confirmed)."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        # Llama-3-8B config
        result = _calculate_trainable_params(
            total_params=8_030_000_000,
            method='lora',
            lora_rank=32,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )

        assert result == 83_886_080, (
            f"Llama-3-8B rank=32 all targets should be 83,886,080 (AWS Neuron), got {result:,}"
        )

    def test_llama3_8b_rank8_all_targets(self):
        """Llama-3-8B with rank=8, all 7 targets = 83,886,080 / 4 = 20,971,520."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        result = _calculate_trainable_params(
            total_params=8_030_000_000,
            method='lora',
            lora_rank=8,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )

        assert result == 20_971_520, (
            f"Llama-3-8B rank=8 all targets should be 20,971,520, got {result:,}"
        )

    def test_llama2_7b_rank8_all_targets(self):
        """Llama-2-7B (MHA) with rank=8, all 7 targets = 19,988,480."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params,
        )

        # Llama-2-7B is MHA (num_kv_heads = num_heads = 32)
        result = _calculate_trainable_params(
            total_params=6_738_000_000,
            method='lora',
            lora_rank=8,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_key_value_heads=32,
        )

        assert result == 19_988_480, (
            f"Llama-2-7B rank=8 all targets should be 19,988,480, got {result:,}"
        )

    def test_basic_calculator_matches_genz(self, basic_calc):
        """Basic calculator and GenZ should produce identical LoRA params."""
        from llm_memory_calculator.genz.LLM_training.training_modeling import (
            _calculate_trainable_params as genz_calc,
        )

        config = dict(LLAMA_8B_CONFIG)
        lora_rank = 16

        basic_result = basic_calc._calculate_trainable_params(
            config, 'lora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )

        genz_result = genz_calc(
            config['num_parameters'], 'lora', lora_rank,
            config['num_hidden_layers'], config['hidden_size'],
            config['intermediate_size'],
            num_attention_heads=config['num_attention_heads'],
            num_key_value_heads=config['num_key_value_heads'],
        )

        assert basic_result == genz_result, (
            f"Basic calc ({basic_result:,}) != GenZ ({genz_result:,})"
        )

    def test_advanced_calculator_matches_basic(self, basic_calc, advanced_calc):
        """Advanced and basic calculators should produce identical LoRA params."""
        config = dict(LLAMA_8B_CONFIG)
        lora_rank = 16

        basic_result = basic_calc._calculate_trainable_params(
            config, 'lora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )

        advanced_result = advanced_calc._calculate_trainable_params(
            config, 'lora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )

        assert basic_result == advanced_result, (
            f"Basic ({basic_result:,}) != Advanced ({advanced_result:,})"
        )

    def test_dora_advanced_matches_basic(self, basic_calc, advanced_calc):
        """DoRA should include magnitude vectors in both calculators."""
        config = dict(LLAMA_8B_CONFIG)
        lora_rank = 16

        basic_result = basic_calc._calculate_trainable_params(
            config, 'dora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )

        advanced_result = advanced_calc._calculate_trainable_params(
            config, 'dora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )

        assert basic_result == advanced_result, (
            f"DoRA basic ({basic_result:,}) != advanced ({advanced_result:,})"
        )

        # DoRA should have more params than LoRA (magnitude vectors)
        lora_basic = basic_calc._calculate_trainable_params(
            config, 'lora', lora_rank, freeze_layers=0,
            total_params=config['num_parameters'],
        )
        assert basic_result > lora_basic, (
            f"DoRA ({basic_result:,}) should exceed LoRA ({lora_basic:,})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
