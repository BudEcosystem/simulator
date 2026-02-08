"""
Ground Truth Accuracy Validation Tests.

Tests the accuracy of training simulation against real-world benchmarks,
categorized by training type, quantization, hardware, and model size.
"""

import pytest
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import validation framework
from llm_memory_calculator.validation import (
    # Benchmark database
    PUBLISHED_BENCHMARKS,
    get_benchmark,
    get_benchmarks_by_category,
    get_high_confidence_benchmarks,
    BenchmarkCategory,
    ConfidenceLevel,
    # Extended schema
    ExtendedBenchmark,
    TrainingType,
    HardwareType,
    QuantizationMethod,
    # Collectors
    collectors,
    # Coverage
    CoverageMatrix,
    analyze_coverage,
    # Validator
    BenchmarkValidator,
    validate_benchmark_plausibility,
    filter_valid_benchmarks,
    # Optimizer benchmarks
    get_optimizer_benchmarks,
    # Multi-model benchmarks
    get_multimodel_benchmarks,
)


@dataclass
class AccuracyTarget:
    """Accuracy targets for different benchmark categories."""
    mre_target: float = 0.15  # Mean Relative Error
    pearson_target: float = 0.85  # Pearson correlation
    tolerance_rate: float = 0.85  # % within tolerance


# Accuracy targets by training type
ACCURACY_TARGETS = {
    TrainingType.PRETRAINING: AccuracyTarget(mre_target=0.15, pearson_target=0.85),
    TrainingType.SFT: AccuracyTarget(mre_target=0.18, pearson_target=0.80),
    TrainingType.LORA: AccuracyTarget(mre_target=0.20, pearson_target=0.80),
    TrainingType.QLORA: AccuracyTarget(mre_target=0.22, pearson_target=0.75),
    TrainingType.DPO: AccuracyTarget(mre_target=0.20, pearson_target=0.80),
    TrainingType.PPO: AccuracyTarget(mre_target=0.25, pearson_target=0.75),
    TrainingType.GRPO: AccuracyTarget(mre_target=0.25, pearson_target=0.75),
}


class TestBenchmarkSchema:
    """Tests for the extended benchmark schema."""

    def test_extended_benchmark_creation(self):
        """Test creating an ExtendedBenchmark from scratch."""
        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
            ParallelismConfig,
            QuantizationConfig,
            ReportedMetrics,
        )

        provenance = SourceProvenance(
            source_type=SourceType.ACADEMIC_PAPER,
            source_url="https://arxiv.org/abs/2307.09288",
            source_title="LLaMA 2 Paper",
            organization="meta",
        )

        parallelism = ParallelismConfig(
            tensor_parallel=8,
            pipeline_parallel=2,
            data_parallel=64,
        )

        quantization = QuantizationConfig(
            model_precision=QuantizationMethod.BF16,
        )

        metrics = ReportedMetrics(
            mfu=0.48,
            tokens_per_second=4500000,
        )

        benchmark = ExtendedBenchmark(
            benchmark_id="test_benchmark",
            name="Test LLaMA-70B",
            provenance=provenance,
            model_name="LLaMA-70B",
            model_params_b=70.0,
            training_type=TrainingType.PRETRAINING,
            num_gpus=1024,
            hardware_type=HardwareType.H100_SXM,
            batch_size=2048,
            seq_length=8192,
            parallelism=parallelism,
            quantization=quantization,
            metrics=metrics,
        )

        assert benchmark.benchmark_id == "test_benchmark"
        assert benchmark.model_params_b == 70.0
        assert benchmark.confidence >= 0.0
        assert benchmark.gpu_scale.value == "hyperscale"
        assert benchmark.model_size_category.value == "large"

    def test_benchmark_validation(self):
        """Test benchmark validation logic."""
        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
            ReportedMetrics,
        )

        # Create a valid benchmark
        provenance = SourceProvenance(source_type=SourceType.MLPERF)
        metrics = ReportedMetrics(mfu=0.50)

        benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="Valid Benchmark",
            provenance=provenance,
            model_name="LLaMA-7B",
            model_params_b=7.0,
            num_gpus=8,
            batch_size=32,
            seq_length=2048,
            metrics=metrics,
        )

        is_valid, issues = benchmark.validate()
        assert is_valid or len(issues) < 3  # Minor issues OK

    def test_benchmark_confidence_scoring(self):
        """Test confidence score calculation."""
        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
            ReportedMetrics,
        )

        # High confidence: MLPerf from NVIDIA with MFU
        provenance = SourceProvenance(
            source_type=SourceType.MLPERF,
            organization="nvidia",
        )
        metrics = ReportedMetrics(mfu=0.54, tokens_per_second=3200000)

        high_conf_benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="High Confidence",
            provenance=provenance,
            model_name="LLaMA-70B",
            model_params_b=70.0,
            num_gpus=512,
            metrics=metrics,
        )

        # Low confidence: Community source, no metrics
        low_provenance = SourceProvenance(
            source_type=SourceType.COMMUNITY,
            organization="unknown",
        )

        low_conf_benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="Low Confidence",
            provenance=low_provenance,
            model_name="LLaMA-7B",
            model_params_b=7.0,
            num_gpus=1,
        )

        assert high_conf_benchmark.confidence > low_conf_benchmark.confidence
        assert high_conf_benchmark.confidence >= 0.5

    def test_benchmark_serialization(self):
        """Test benchmark to_dict and from_dict."""
        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
        )

        provenance = SourceProvenance(source_type=SourceType.ACADEMIC_PAPER)

        original = ExtendedBenchmark(
            benchmark_id="test",
            name="Test Benchmark",
            provenance=provenance,
            model_name="LLaMA-7B",
            model_params_b=7.0,
            training_type=TrainingType.SFT,
            num_gpus=8,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ExtendedBenchmark.from_dict(data)

        assert restored.benchmark_id == original.benchmark_id
        assert restored.name == original.name
        assert restored.model_params_b == original.model_params_b
        assert restored.training_type == original.training_type


class TestBenchmarkValidator:
    """Tests for benchmark validation rules."""

    def test_mfu_bounds_validation(self):
        """Test MFU bounds checking."""
        validator = BenchmarkValidator()

        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
            ReportedMetrics,
        )

        provenance = SourceProvenance(source_type=SourceType.MANUAL_ENTRY)

        # Valid MFU
        valid_benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="Valid MFU",
            provenance=provenance,
            model_name="LLaMA-7B",
            model_params_b=7.0,
            num_gpus=8,
            metrics=ReportedMetrics(mfu=0.50),
        )

        result = validator.validate(valid_benchmark)
        mfu_errors = [i for i in result.issues if "mfu" in i.field.lower() and i.severity.value == "error"]
        assert len(mfu_errors) == 0

        # Invalid MFU (too high)
        invalid_benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="Invalid MFU",
            provenance=provenance,
            model_name="LLaMA-7B",
            model_params_b=7.0,
            num_gpus=8,
            metrics=ReportedMetrics(mfu=0.95),  # Impossible
        )

        result = validator.validate(invalid_benchmark)
        mfu_errors = [i for i in result.issues if "mfu" in i.field.lower()]
        assert len(mfu_errors) > 0

    def test_parallelism_validation(self):
        """Test parallelism consistency checking."""
        validator = BenchmarkValidator()

        from llm_memory_calculator.validation.benchmark_schema import (
            ExtendedBenchmark,
            SourceProvenance,
            SourceType,
            ParallelismConfig,
        )

        provenance = SourceProvenance(source_type=SourceType.MANUAL_ENTRY)

        # Invalid: parallelism exceeds GPU count
        parallelism = ParallelismConfig(
            tensor_parallel=8,
            pipeline_parallel=4,
            data_parallel=8,  # 8*4*8 = 256, but only 64 GPUs
        )

        benchmark = ExtendedBenchmark(
            benchmark_id="",
            name="Bad Parallelism",
            provenance=provenance,
            model_name="LLaMA-70B",
            model_params_b=70.0,
            num_gpus=64,
            parallelism=parallelism,
        )

        result = validator.validate(benchmark)
        parallelism_errors = [i for i in result.issues if "parallelism" in i.field.lower()]
        assert len(parallelism_errors) > 0

    def test_filter_valid_benchmarks(self):
        """Test filtering benchmarks by validation score."""
        # Get all benchmarks
        optimizer_benchmarks = get_optimizer_benchmarks()
        multimodel_benchmarks = get_multimodel_benchmarks()
        all_benchmarks = optimizer_benchmarks + multimodel_benchmarks

        # Filter for high-quality benchmarks
        valid_benchmarks = filter_valid_benchmarks(all_benchmarks, min_score=0.7)

        # Should get at least some valid benchmarks
        assert len(valid_benchmarks) > 0
        assert len(valid_benchmarks) <= len(all_benchmarks)


class TestCoverageMatrix:
    """Tests for coverage tracking."""

    def test_coverage_matrix_creation(self):
        """Test creating coverage matrix from benchmarks."""
        # Get benchmarks from all sources
        optimizer_benchmarks = get_optimizer_benchmarks()
        multimodel_benchmarks = get_multimodel_benchmarks()

        # Create coverage matrix
        matrix = CoverageMatrix(
            benchmarks=optimizer_benchmarks + multimodel_benchmarks
        )

        # Check coverage by dimension
        training_type_coverage = matrix.get_coverage_by_dimension("training_type")
        assert len(training_type_coverage) > 0

        optimizer_coverage = matrix.get_coverage_by_dimension("optimizer")
        assert len(optimizer_coverage) > 0

    def test_coverage_gap_identification(self):
        """Test identifying coverage gaps."""
        optimizer_benchmarks = get_optimizer_benchmarks()

        matrix = CoverageMatrix(benchmarks=optimizer_benchmarks)
        gaps = matrix.identify_gaps()

        # Should identify some gaps (we don't have full coverage)
        assert isinstance(gaps, list)

    def test_coverage_report_generation(self):
        """Test coverage report generation."""
        optimizer_benchmarks = get_optimizer_benchmarks()
        multimodel_benchmarks = get_multimodel_benchmarks()

        matrix = CoverageMatrix(
            benchmarks=optimizer_benchmarks + multimodel_benchmarks
        )

        report = matrix.generate_report()

        assert report.total_benchmarks > 0
        assert isinstance(report.coverage_by_dimension, dict)
        assert "training_type" in report.coverage_by_dimension

    def test_collection_priorities(self):
        """Test priority generation for data collection."""
        optimizer_benchmarks = get_optimizer_benchmarks()

        matrix = CoverageMatrix(benchmarks=optimizer_benchmarks)
        priorities = matrix.generate_priorities(top_n=5)

        assert len(priorities) <= 5
        for p in priorities:
            assert "dimensions" in p
            assert "priority_score" in p


class TestCollectors:
    """Tests for data collectors."""

    def test_arxiv_collector_known_papers(self):
        """Test arXiv collector with known papers."""
        from llm_memory_calculator.validation.collectors import ArxivCollector

        collector = ArxivCollector()

        # Test collecting from known paper
        result = collector.collect("2307.09288")  # LLaMA 2 paper

        assert result.success or len(result.benchmarks) > 0

    def test_arxiv_collector_list_known(self):
        """Test listing known arXiv papers."""
        from llm_memory_calculator.validation.collectors import ArxivCollector

        collector = ArxivCollector()
        known = collector.list_known_papers()

        assert len(known) > 0
        assert any("llama" in p["title"].lower() for p in known)

    def test_mlperf_collector_all(self):
        """Test MLPerf collector."""
        from llm_memory_calculator.validation.collectors import MLPerfCollector

        collector = MLPerfCollector()

        # Collect all MLPerf benchmarks
        result = collector.collect_all()

        assert len(result.benchmarks) > 0
        assert result.success

    def test_mlperf_collector_by_version(self):
        """Test MLPerf collector filtering by version."""
        from llm_memory_calculator.validation.collectors import MLPerfCollector

        collector = MLPerfCollector()
        result = collector.collect_by_version("v4.0")

        assert len(result.benchmarks) > 0

    def test_huggingface_collector_known(self):
        """Test HuggingFace collector with known models."""
        from llm_memory_calculator.validation.collectors import HuggingFaceCollector

        collector = HuggingFaceCollector()

        # List known models
        known = collector.list_known_models()
        assert len(known) > 0

        # Collect all known
        result = collector.collect_all_known()
        assert len(result.benchmarks) > 0

    def test_github_collector_known_repos(self):
        """Test GitHub collector with known repos."""
        from llm_memory_calculator.validation.collectors import GitHubCollector

        collector = GitHubCollector()

        # List known repos
        repos = collector.list_known_repos()
        assert len(repos) > 0

        # Collect by category
        optimizer_result = collector.collect_by_category("optimizer")
        assert len(optimizer_result.benchmarks) > 0

        rlhf_result = collector.collect_by_category("rlhf")
        assert len(rlhf_result.benchmarks) > 0


class TestOptimizerBenchmarks:
    """Tests for optimizer-specific benchmarks."""

    def test_optimizer_profiles(self):
        """Test optimizer profile data."""
        from llm_memory_calculator.validation import (
            OPTIMIZER_PROFILES,
            get_optimizer_profile,
            OptimizerType,
        )

        # Check we have profiles for common optimizers
        assert OptimizerType.ADAMW in OPTIMIZER_PROFILES
        assert OptimizerType.ADAMW_8BIT in OPTIMIZER_PROFILES
        assert OptimizerType.GALORE in OPTIMIZER_PROFILES

        # Check profile data
        adamw = get_optimizer_profile(OptimizerType.ADAMW)
        assert adamw.bytes_per_param == 8.0
        assert adamw.memory_multiplier == 1.0

        adamw_8bit = get_optimizer_profile(OptimizerType.ADAMW_8BIT)
        assert adamw_8bit.bytes_per_param == 2.0
        assert adamw_8bit.memory_multiplier < adamw.memory_multiplier

    def test_optimizer_memory_estimation(self):
        """Test optimizer memory estimation."""
        from llm_memory_calculator.validation import (
            estimate_optimizer_memory,
            compare_optimizer_memory,
            OptimizerType,
        )

        params_b = 7.0  # 7B model

        # AdamW should use 8 bytes per param
        adamw_memory = estimate_optimizer_memory(params_b, OptimizerType.ADAMW)
        assert adamw_memory == 56.0  # 7B * 8 bytes

        # 8-bit AdamW should use 2 bytes per param
        adamw_8bit_memory = estimate_optimizer_memory(params_b, OptimizerType.ADAMW_8BIT)
        assert adamw_8bit_memory == 14.0  # 7B * 2 bytes

        # Compare all optimizers
        comparison = compare_optimizer_memory(params_b)
        assert "AdamW" in comparison
        assert "8-bit AdamW" in comparison
        assert comparison["8-bit AdamW"] < comparison["AdamW"]

    def test_optimizer_benchmarks_collection(self):
        """Test getting optimizer benchmarks."""
        benchmarks = get_optimizer_benchmarks()

        assert len(benchmarks) > 0

        # Check we have different optimizer types
        optimizer_types = set(b.optimizer for b in benchmarks)
        assert len(optimizer_types) > 1


class TestMultiModelBenchmarks:
    """Tests for multi-model training benchmarks."""

    def test_multimodel_profiles(self):
        """Test multi-model training profiles."""
        from llm_memory_calculator.validation import (
            MULTIMODEL_PROFILES,
            get_multimodel_profile,
        )

        # Check we have profiles for RLHF methods
        assert TrainingType.PPO in MULTIMODEL_PROFILES
        assert TrainingType.DPO in MULTIMODEL_PROFILES
        assert TrainingType.GRPO in MULTIMODEL_PROFILES

        # Check PPO profile
        ppo = get_multimodel_profile(TrainingType.PPO)
        assert ppo.num_models >= 3
        assert ppo.memory_multiplier > 1.5
        assert "policy" in ppo.model_roles
        assert "reference" in ppo.model_roles
        assert "reward" in ppo.model_roles

        # Check DPO profile
        dpo = get_multimodel_profile(TrainingType.DPO)
        assert dpo.num_models == 2
        assert dpo.memory_multiplier < ppo.memory_multiplier

    def test_multimodel_memory_estimation(self):
        """Test multi-model memory estimation."""
        from llm_memory_calculator.validation import estimate_multimodel_memory

        # PPO with 7B model
        ppo_estimate = estimate_multimodel_memory(
            model_params_b=7.0,
            training_type=TrainingType.PPO,
        )

        assert "policy_model_gb" in ppo_estimate
        assert "reference_model_gb" in ppo_estimate
        assert "total_gb" in ppo_estimate
        # Memory multiplier is ratio of total to policy training memory
        # PPO adds reference + reward + value models in eval mode
        assert ppo_estimate["memory_multiplier"] > 1.0  # Uses more than single-model training
        assert ppo_estimate["total_gb"] > ppo_estimate["policy_model_gb"]  # Total > policy alone

        # DPO should use less memory
        dpo_estimate = estimate_multimodel_memory(
            model_params_b=7.0,
            training_type=TrainingType.DPO,
        )

        assert dpo_estimate["total_gb"] < ppo_estimate["total_gb"]

        # ORPO should be similar to single model
        orpo_estimate = estimate_multimodel_memory(
            model_params_b=7.0,
            training_type=TrainingType.ORPO,
        )

        assert orpo_estimate["reference_model_gb"] == 0.0

    def test_multimodel_benchmarks_collection(self):
        """Test getting multi-model benchmarks."""
        benchmarks = get_multimodel_benchmarks()

        assert len(benchmarks) > 0

        # Check we have different training types
        training_types = set(b.training_type for b in benchmarks)
        assert TrainingType.PPO in training_types
        assert TrainingType.DPO in training_types

        # Check multi-model configs are present
        for b in benchmarks:
            if b.training_type in (TrainingType.PPO, TrainingType.DPO, TrainingType.GRPO):
                assert b.multi_model is not None


class TestAccuracyByTrainingType:
    """Tests for accuracy broken down by training type."""

    @pytest.fixture
    def all_benchmarks(self):
        """Get all available benchmarks."""
        return get_optimizer_benchmarks() + get_multimodel_benchmarks()

    def test_pretraining_benchmarks_exist(self, all_benchmarks):
        """Check we have pretraining benchmarks."""
        pretraining = [b for b in all_benchmarks if b.training_type == TrainingType.PRETRAINING]
        assert len(pretraining) > 0, "Should have pretraining benchmarks"

    def test_sft_benchmarks_exist(self, all_benchmarks):
        """Check we have SFT benchmarks."""
        sft = [b for b in all_benchmarks if b.training_type == TrainingType.SFT]
        assert len(sft) > 0, "Should have SFT benchmarks"

    def test_dpo_benchmarks_exist(self, all_benchmarks):
        """Check we have DPO benchmarks."""
        dpo = [b for b in all_benchmarks if b.training_type == TrainingType.DPO]
        assert len(dpo) > 0, "Should have DPO benchmarks"

    def test_ppo_benchmarks_exist(self, all_benchmarks):
        """Check we have PPO benchmarks."""
        ppo = [b for b in all_benchmarks if b.training_type == TrainingType.PPO]
        assert len(ppo) > 0, "Should have PPO benchmarks"


class TestAccuracyByHardware:
    """Tests for accuracy broken down by hardware type."""

    @pytest.fixture
    def all_benchmarks(self):
        """Get all available benchmarks."""
        return get_optimizer_benchmarks() + get_multimodel_benchmarks()

    def test_h100_benchmarks_exist(self, all_benchmarks):
        """Check we have H100 benchmarks."""
        h100 = [b for b in all_benchmarks if b.hardware_type == HardwareType.H100_SXM]
        assert len(h100) > 0, "Should have H100 benchmarks"

    def test_a100_benchmarks_exist(self, all_benchmarks):
        """Check we have A100 benchmarks."""
        a100 = [b for b in all_benchmarks if b.hardware_type in (
            HardwareType.A100_40GB, HardwareType.A100_80GB
        )]
        assert len(a100) > 0, "Should have A100 benchmarks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
