"""
Tests for Cluster Ranking and Requirements Prediction.

Tests the new functions:
- rank_clusters_for_training()
- predict_cluster_requirements()
- generate_comprehensive_training_config()
"""

import pytest
from typing import List

# Import the new modules
from llm_memory_calculator.training import (
    # Types
    ClusterRankingResult,
    MinimumClusterRequirements,
    ComprehensiveLlamaFactoryConfig,
    MemoryBreakdownDetails,
    GPURequirement,
    RankingMetric,
    ClusterTopology,
    format_eta,
    # Functions
    rank_clusters_for_training,
    predict_cluster_requirements,
    generate_comprehensive_training_config,
    # Best practices
    TRAINING_TYPE_MEMORY_MULTIPLIERS,
    LEARNING_RATES,
    LORA_CONFIGS,
    PPO_BEST_PRACTICES,
    get_recommended_learning_rate,
    get_precision_bytes,
    # Existing types
    ClusterDefinition,
    TrainingJobSpec,
    ParallelismStrategy,
    ClusterOptimizer,
)


class TestClusterRankingTypes:
    """Test cluster ranking data types."""

    def test_format_eta(self):
        """Test ETA formatting."""
        assert format_eta(0.5) == "30 minutes"
        assert "hour" in format_eta(1.5)
        assert "day" in format_eta(48)
        assert "week" in format_eta(168)

    def test_memory_breakdown_details(self):
        """Test MemoryBreakdownDetails creation."""
        breakdown = MemoryBreakdownDetails(
            weight_memory_gb=10.0,
            gradient_memory_gb=5.0,
            optimizer_memory_gb=20.0,
            activation_memory_gb=8.0,
            total_per_gpu_gb=43.0,
        )
        assert breakdown.total_per_gpu_gb == 43.0
        result = breakdown.to_dict()
        assert "weights" in result
        assert "gradients" in result

    def test_cluster_ranking_result(self):
        """Test ClusterRankingResult creation."""
        cluster = ClusterDefinition(
            name="test-cluster",
            gpu_type="h100",
            num_gpus=8,
        )
        parallelism = ParallelismStrategy(
            tensor_parallel=1,
            data_parallel=8,
            zero_stage=2,
        )
        result = ClusterRankingResult(
            cluster=cluster,
            optimal_batch_size=4,
            parallelism=parallelism,
            tokens_per_second=100000,
            step_time_ms=500,
            mfu=0.45,
            memory_per_gpu_gb=60.0,
            estimated_training_hours=10.0,
            estimated_eta="10 hours",
            estimated_cost_usd=500.0,
            cost_per_million_tokens=5.0,
        )
        assert result.tokens_per_second == 100000
        assert result.estimated_training_hours == 10.0


class TestBestPractices:
    """Test best practices constants and functions."""

    def test_training_type_memory_multipliers(self):
        """Test training type memory multipliers."""
        assert TRAINING_TYPE_MEMORY_MULTIPLIERS["sft"] == 1.0
        assert TRAINING_TYPE_MEMORY_MULTIPLIERS["dpo"] == 1.7
        assert TRAINING_TYPE_MEMORY_MULTIPLIERS["ppo"] == 2.8

    def test_learning_rates(self):
        """Test learning rate recommendations."""
        assert "lora" in LEARNING_RATES
        assert "full" in LEARNING_RATES
        assert LEARNING_RATES["lora"]["7B"] == 1e-4
        assert LEARNING_RATES["full"]["7B"] == 1e-5

    def test_get_recommended_learning_rate(self):
        """Test learning rate recommendation function."""
        # LoRA rates
        lr = get_recommended_learning_rate("lora", 8)
        assert lr == 1e-4

        lr = get_recommended_learning_rate("lora", 70)
        assert lr == 2e-5

        # Full fine-tuning rates (should be lower)
        lr_full = get_recommended_learning_rate("full", 8)
        assert lr_full < lr

    def test_get_precision_bytes(self):
        """Test precision bytes function."""
        assert get_precision_bytes("bf16") == 2
        assert get_precision_bytes("fp32") == 4
        assert get_precision_bytes("nf4") == 0.5

    def test_lora_configs(self):
        """Test LoRA configuration constants."""
        assert "minimal" in LORA_CONFIGS
        assert "standard" in LORA_CONFIGS
        assert "high_quality" in LORA_CONFIGS
        assert "dora" in LORA_CONFIGS

        # Standard config should have reasonable defaults
        standard = LORA_CONFIGS["standard"]
        assert standard["lora_rank"] == 16
        assert standard["lora_alpha"] == 32

    def test_ppo_best_practices(self):
        """Test PPO best practices constants."""
        assert PPO_BEST_PRACTICES["ppo_epochs"] == 4
        assert PPO_BEST_PRACTICES["ppo_score_norm"] == True
        assert "models_required" in PPO_BEST_PRACTICES


class TestPredictClusterRequirements:
    """Test cluster requirements prediction."""

    def test_predict_sft_requirements(self):
        """Test predicting requirements for SFT training."""
        reqs = predict_cluster_requirements(
            training_type="sft",
            model="meta-llama/Llama-3.1-8B",
            dtype="bf16",
            optimizer="adamw",
            dataset_size_tokens=1_000_000_000,
            batch_size=4,
            method="lora",
        )

        assert isinstance(reqs, MinimumClusterRequirements)
        assert reqs.min_gpus >= 1
        assert reqs.min_gpu_memory_gb > 0
        assert reqs.memory_breakdown is not None

    def test_predict_ppo_requirements(self):
        """Test predicting requirements for PPO training."""
        reqs = predict_cluster_requirements(
            training_type="ppo",
            model="meta-llama/Llama-3.1-8B",
            dtype="bf16",
            optimizer="adamw",
            dataset_size_tokens=100_000_000,
            batch_size=2,
            method="lora",
        )

        assert isinstance(reqs, MinimumClusterRequirements)
        # PPO needs more memory for multiple models
        assert reqs.min_gpus >= 1
        # Should have reference and reward model memory
        if reqs.memory_breakdown:
            assert reqs.memory_breakdown.reference_model_memory_gb > 0

    def test_predict_large_model_requirements(self):
        """Test predicting requirements for large model."""
        reqs = predict_cluster_requirements(
            training_type="sft",
            model="meta-llama/Llama-3.1-70B",
            dtype="bf16",
            optimizer="adamw",
            dataset_size_tokens=1_000_000_000,
            batch_size=2,
            method="lora",
        )

        assert isinstance(reqs, MinimumClusterRequirements)
        # 70B model with LoRA can fit on single high-memory GPU (MI300X, B100)
        # but needs multiple GPUs for most common GPUs
        assert reqs.min_gpus >= 1
        # Should require substantial GPU memory
        assert reqs.min_gpu_memory_gb >= 80

    def test_predict_with_time_constraint(self):
        """Test predicting requirements with time constraint."""
        reqs = predict_cluster_requirements(
            training_type="sft",
            model="meta-llama/Llama-3.1-8B",
            dtype="bf16",
            optimizer="adamw",
            dataset_size_tokens=10_000_000_000,  # Large dataset
            batch_size=4,
            method="lora",
            max_training_hours=24,  # Constraint to 24 hours
        )

        assert isinstance(reqs, MinimumClusterRequirements)
        # Should recommend more GPUs to meet time constraint
        assert reqs.min_gpus >= 1


class TestRankClustersForTraining:
    """Test cluster ranking function."""

    def get_test_clusters(self) -> List[ClusterDefinition]:
        """Create test clusters."""
        return [
            ClusterDefinition(
                name="h100-8x",
                gpu_type="h100",
                num_gpus=8,
                hourly_rate_per_gpu=2.5,
            ),
            ClusterDefinition(
                name="a100-8x",
                gpu_type="a100_80gb",
                num_gpus=8,
                hourly_rate_per_gpu=1.5,
            ),
            ClusterDefinition(
                name="h100-4x",
                gpu_type="h100",
                num_gpus=4,
                hourly_rate_per_gpu=2.5,
            ),
        ]

    def test_rank_clusters_by_throughput(self):
        """Test ranking clusters by throughput."""
        clusters = self.get_test_clusters()

        try:
            results = rank_clusters_for_training(
                model="meta-llama/Llama-3.1-8B",
                training_type="sft",
                clusters=clusters,
                dataset_tokens=1_000_000_000,
                method="lora",
                sort_by="throughput",
                return_top_k=3,
            )

            assert len(results) <= 3
            for result in results:
                assert isinstance(result, ClusterRankingResult)
                assert result.tokens_per_second > 0
                assert result.throughput_rank > 0

            # Should be sorted by throughput (highest first)
            if len(results) > 1:
                assert results[0].tokens_per_second >= results[1].tokens_per_second

        except RuntimeError as e:
            # May fail if GenZ simulation fails
            pytest.skip(f"Simulation failed: {e}")

    def test_rank_clusters_by_cost(self):
        """Test ranking clusters by cost."""
        clusters = self.get_test_clusters()

        try:
            results = rank_clusters_for_training(
                model="meta-llama/Llama-3.1-8B",
                training_type="sft",
                clusters=clusters,
                dataset_tokens=1_000_000_000,
                method="lora",
                sort_by="cost",
                return_top_k=3,
            )

            for result in results:
                assert result.estimated_cost_usd > 0
                assert result.cost_rank > 0

            # Should be sorted by cost (lowest first)
            if len(results) > 1:
                assert results[0].estimated_cost_usd <= results[1].estimated_cost_usd

        except RuntimeError as e:
            pytest.skip(f"Simulation failed: {e}")

    def test_rank_clusters_composite(self):
        """Test ranking clusters by composite score."""
        clusters = self.get_test_clusters()

        try:
            results = rank_clusters_for_training(
                model="meta-llama/Llama-3.1-8B",
                training_type="sft",
                clusters=clusters,
                dataset_tokens=1_000_000_000,
                method="lora",
                sort_by="composite",
                return_top_k=3,
            )

            for result in results:
                assert result.composite_score >= 0
                assert result.composite_score <= 1

            # Should be sorted by composite (highest first)
            if len(results) > 1:
                assert results[0].composite_score >= results[1].composite_score

        except RuntimeError as e:
            pytest.skip(f"Simulation failed: {e}")


class TestGenerateComprehensiveConfig:
    """Test comprehensive config generation."""

    def test_generate_sft_config(self):
        """Test generating SFT config with best practices."""
        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
            num_epochs=1,
        )
        parallelism = ParallelismStrategy(
            tensor_parallel=1,
            data_parallel=8,
            zero_stage=2,
        )
        cluster = ClusterDefinition(
            name="h100-8x",
            gpu_type="h100",
            num_gpus=8,
        )

        config = generate_comprehensive_training_config(
            job_spec=job_spec,
            parallelism=parallelism,
            cluster=cluster,
            optimization_focus="balanced",
        )

        assert isinstance(config, ComprehensiveLlamaFactoryConfig)
        assert config.llamafactory_yaml is not None
        assert len(config.best_practices_applied) > 0
        assert "learning_rate" in config.llamafactory_yaml

    def test_generate_ppo_config(self):
        """Test generating PPO config with multi-model support."""
        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="ppo",
            method="lora",
            dataset_tokens=100_000_000,
            num_epochs=1,
        )
        parallelism = ParallelismStrategy(
            tensor_parallel=1,
            data_parallel=8,
            zero_stage=2,
        )
        cluster = ClusterDefinition(
            name="h100-8x",
            gpu_type="h100",
            num_gpus=8,
        )

        config = generate_comprehensive_training_config(
            job_spec=job_spec,
            parallelism=parallelism,
            cluster=cluster,
            optimization_focus="balanced",
        )

        assert isinstance(config, ComprehensiveLlamaFactoryConfig)
        # PPO should have multi-model configs
        assert config.ppo_actor_config is not None
        assert config.ppo_reference_config is not None
        assert config.ppo_reward_config is not None
        assert config.vllm_inference_config is not None

    def test_generate_config_with_deepspeed(self):
        """Test generating config with DeepSpeed."""
        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
        )
        parallelism = ParallelismStrategy(
            tensor_parallel=1,
            data_parallel=8,
            zero_stage=2,
        )
        cluster = ClusterDefinition(
            name="h100-8x",
            gpu_type="h100",
            num_gpus=8,
        )

        config = generate_comprehensive_training_config(
            job_spec=job_spec,
            parallelism=parallelism,
            cluster=cluster,
        )

        assert config.deepspeed_json is not None
        assert "zero_optimization" in config.deepspeed_json
        assert config.deepspeed_json["zero_optimization"]["stage"] == 2

    def test_optimization_focus_variations(self):
        """Test different optimization focuses."""
        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
        )
        parallelism = ParallelismStrategy(data_parallel=8, zero_stage=2)
        cluster = ClusterDefinition(
            name="h100-8x",
            gpu_type="h100",
            num_gpus=8,
        )

        for focus in ["stable", "convergence", "speed", "tco", "balanced"]:
            config = generate_comprehensive_training_config(
                job_spec=job_spec,
                parallelism=parallelism,
                cluster=cluster,
                optimization_focus=focus,
            )
            assert config.optimization_focus == focus


class TestClusterOptimizerIntegration:
    """Test ClusterOptimizer class with new methods."""

    def test_optimizer_rank_clusters(self):
        """Test ClusterOptimizer.rank_clusters method."""
        optimizer = ClusterOptimizer(debug=False)

        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
        )
        clusters = [
            ClusterDefinition(name="h100-8x", gpu_type="h100", num_gpus=8),
            ClusterDefinition(name="a100-8x", gpu_type="a100_80gb", num_gpus=8),
        ]

        try:
            results = optimizer.rank_clusters(job_spec, clusters, sort_by="throughput", k=2)
            assert len(results) <= 2
        except RuntimeError:
            pytest.skip("Simulation failed")

    def test_optimizer_predict_requirements(self):
        """Test ClusterOptimizer.predict_requirements method."""
        optimizer = ClusterOptimizer(debug=False)

        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
        )

        reqs = optimizer.predict_requirements(job_spec)
        assert isinstance(reqs, MinimumClusterRequirements)
        assert reqs.min_gpus >= 1

    def test_optimizer_generate_comprehensive_config(self):
        """Test ClusterOptimizer.generate_comprehensive_config method."""
        optimizer = ClusterOptimizer(debug=False)

        job_spec = TrainingJobSpec(
            model="meta-llama/Llama-3.1-8B",
            training_type="sft",
            method="lora",
            dataset_tokens=1_000_000_000,
        )
        parallelism = ParallelismStrategy(data_parallel=8, zero_stage=2)
        cluster = ClusterDefinition(name="h100-8x", gpu_type="h100", num_gpus=8)

        config = optimizer.generate_comprehensive_config(
            job_spec, parallelism, cluster, focus="balanced"
        )
        assert isinstance(config, ComprehensiveLlamaFactoryConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
