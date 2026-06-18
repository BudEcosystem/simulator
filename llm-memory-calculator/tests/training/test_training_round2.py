"""Round-2 training-memory fixes (solutions_round2.md §3).

TR1  API path (TrainingMemoryCalculator) shards weights by TP*PP*EP and full-FT grad/opt by the
     resident-trainable model-parallel degree; defaults (TP=PP=EP=1) byte-identical.
TR2  low-rank optimizers (GaLore/APOLLO) scale optimizer memory with rank (rank 16 == default;
     rank 256 > rank 16); non-low-rank optimizers unchanged.
TR4  the GenZ training path reads the sourced OPTIMIZER_CONFIGS, not the inferior OPTIMIZER_STATE_BYTES.
"""
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llm_memory_calculator.training.calculator import TrainingMemoryCalculator

LLAMA_8B = {
    "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
    "vocab_size": 128256, "head_dim": 128, "tie_word_embeddings": False,
    "num_parameters": 8_030_261_248,
}
LLAMA_70B = {
    "model_type": "llama", "hidden_size": 8192, "intermediate_size": 28672,
    "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8,
    "vocab_size": 128256, "head_dim": 128, "tie_word_embeddings": False,
    "num_parameters": 70_553_706_496,
}


def _calc(**kw):
    c = TrainingMemoryCalculator()
    base = dict(config=LLAMA_8B, batch_size=1, seq_length=2048, precision="bf16")
    base.update(kw)
    return c.calculate_training_memory(**base)


# ----------------------------------------------------------------- TR1
def test_default_lora_byte_identical_with_pp_ep_1():
    """Adding pipeline_parallel=1/expert_parallel=1 to the default LoRA path must not change anything."""
    a = _calc(method="lora")
    b = _calc(method="lora", pipeline_parallel=1, expert_parallel=1)
    assert a.weight_memory_gb == pytest.approx(b.weight_memory_gb, rel=1e-12)
    assert a.optimizer_memory_gb == pytest.approx(b.optimizer_memory_gb, rel=1e-12)


def test_pp_ep_shard_weights():
    """Weight memory must drop by exactly PP*EP (3D parallelism)."""
    base = _calc(method="full")
    sharded = _calc(method="full", pipeline_parallel=4, expert_parallel=2)
    assert base.weight_memory_gb / sharded.weight_memory_gb == pytest.approx(8.0, rel=1e-6)


def test_full_ft_grad_opt_shard_by_model_parallel():
    """Full-FT gradients + optimizer must shard by TP*PP (resident trainable params)."""
    base = _calc(method="full", optimizer="adamw")
    sharded = _calc(method="full", optimizer="adamw", tensor_parallel=2, pipeline_parallel=4)
    assert base.gradient_memory_gb / sharded.gradient_memory_gb == pytest.approx(8.0, rel=1e-6)
    assert base.optimizer_memory_gb / sharded.optimizer_memory_gb == pytest.approx(8.0, rel=1e-6)


def test_70b_tp8_pp8_full_ft_is_physically_feasible():
    """70B full-FT on TP8.PP8 must report tens of GB/GPU (not the old impossible ~hundreds)."""
    c = TrainingMemoryCalculator()
    e = c.calculate_training_memory(
        config=LLAMA_70B, batch_size=1, seq_length=2048, precision="bf16",
        method="full", optimizer="adamw", tensor_parallel=8, pipeline_parallel=8,
    )
    per_gpu = e.weight_memory_gb + e.gradient_memory_gb + e.optimizer_memory_gb
    assert per_gpu < 80.0, f"70B TP8.PP8 per-GPU model memory = {per_gpu:.1f} GB (must fit an 80GB GPU)"


# ----------------------------------------------------------------- TR2
def test_galore_optimizer_scales_with_rank():
    """GaLore optimizer state scales with the projection rank (capped at the full-rank default=16),
    so a lower rank uses strictly less memory. The old code bypassed calculate_memory_gb -> all ranks
    identical."""
    r4 = _calc(method="full", optimizer="galore", lora_rank=4)
    r16 = _calc(method="full", optimizer="galore", lora_rank=16)
    assert r4.optimizer_memory_gb < r16.optimizer_memory_gb, \
        "GaLore optimizer memory must scale with projection rank (was rank-independent)"
    assert r4.optimizer_memory_gb == pytest.approx(r16.optimizer_memory_gb * 0.25, rel=1e-6)


def test_adamw_optimizer_rank_independent():
    """A standard (non-low-rank) optimizer must NOT change with lora_rank."""
    a = _calc(method="full", optimizer="adamw", lora_rank=16)
    b = _calc(method="full", optimizer="adamw", lora_rank=256)
    assert a.optimizer_memory_gb == pytest.approx(b.optimizer_memory_gb, rel=1e-12)


# ----------------------------------------------------------------- TR4
def test_genz_training_uses_sourced_optimizer_table():
    """The GenZ training path must resolve per-param optimizer bytes from the sourced OPTIMIZER_CONFIGS
    (sgd=4 momentum, adafactor=1 factorized), not the inferior OPTIMIZER_STATE_BYTES (sgd=0, adafactor=4)."""
    from llm_memory_calculator.training.optimizers import get_optimizer_config
    assert get_optimizer_config('sgd').total_bytes_per_param == pytest.approx(4.0)
    assert get_optimizer_config('adafactor').total_bytes_per_param == pytest.approx(1.0)
    from llm_memory_calculator.genz.LLM_training.training_modeling import _optimizer_bytes_per_param
    assert _optimizer_bytes_per_param('sgd') == pytest.approx(4.0), "GenZ path still uses sgd=0"
    assert _optimizer_bytes_per_param('adafactor') == pytest.approx(1.0), "GenZ path still uses adafactor=4"
    assert _optimizer_bytes_per_param('adamw') == pytest.approx(8.0)


# ----------------------------------------------------------------- TR3
def test_cluster_selector_no_magic_fudge_ddp_does_not_fit():
    """R2-TR2: pure DDP (no ZeRO) must NOT pretend data-parallel shards weights. A 70B full-FT estimate
    fed to check_fit with a real component breakdown must require the physically-correct per-GPU memory
    (DP replicates), not total/(tp*pp*dp)*0.7."""
    from llm_memory_calculator.training.calculator import TrainingMemoryCalculator
    from llm_memory_calculator.training.cluster_selector import TrainingClusterSelector
    est = TrainingMemoryCalculator().calculate_training_memory(
        config=LLAMA_70B, batch_size=1, seq_length=2048, precision="bf16",
        method="full", optimizer="adamw",  # no deepspeed_stage -> pure DDP
    ).to_dict()
    sel = TrainingClusterSelector()
    # On 8x A100-80GB with NO ZeRO, a 70B full-FT (~1.1 TB) cannot fit (DP replicates everything).
    res = sel.check_fit(est, "A100_80GB_GPU", num_gpus=8)
    assert res.fits is False, "pure-DDP 70B full-FT must NOT fit 8xA100 (the 0.7 fudge falsely passed it)"


def test_cluster_selector_per_gpu_matches_component_sharding():
    """The selector's per-GPU footprint must equal the calculator's component breakdown under the same
    sharding contract (single source of truth), within rounding."""
    from llm_memory_calculator.training.cluster_selector import TrainingClusterSelector
    est = {
        'total_memory_gb': 100.0, 'weight_memory_gb': 40.0, 'gradient_memory_gb': 20.0,
        'optimizer_memory_gb': 30.0, 'activation_memory_gb': 10.0, 'deepspeed_stage': 'zero3',
    }
    # tp=2, pp=2 (mp=4), dp=2, zero3: (w+a)/4 + (g/4)/2 + (o/4)/2 = 12.5 + 2.5 + 3.75 = 18.75
    got = TrainingClusterSelector._per_gpu_footprint(est, 100.0, tp=2, pp=2, dp=2)
    assert got == pytest.approx(18.75, rel=1e-6)


# ----------------------------------------------------------------- TR1 re-review: activation + PP
def test_activation_memory_shards_by_pipeline_parallel():
    """R2 re-review: activation memory was sharded by TP but NOT by PP, over-stating per-GPU memory
    for PP>1 and disagreeing with the cluster selector (which divides activation by tp*pp). Under 1F1B
    (~PP in-flight microbatches of size batch/PP across L/PP layers) per-GPU activation ~= /PP."""
    c = TrainingMemoryCalculator()
    base = c.calculate_training_memory(config=LLAMA_70B, batch_size=2, seq_length=4096, precision="bf16",
                                       method="full", optimizer="adamw", gradient_checkpointing=False)
    pp4 = c.calculate_training_memory(config=LLAMA_70B, batch_size=2, seq_length=4096, precision="bf16",
                                      method="full", optimizer="adamw", gradient_checkpointing=False,
                                      pipeline_parallel=4)
    assert base.activation_memory_gb / pp4.activation_memory_gb == pytest.approx(4.0, rel=1e-6), \
        f"activation did not shard by PP: base={base.activation_memory_gb} pp4={pp4.activation_memory_gb}"
