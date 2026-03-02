"""CPU serving integration tests.

Verifies that the entire serving simulation pipeline works with CPU hardware,
including ServingSimulator, BatchScheduler, PowerModel, DisaggregationAnalyzer,
ClusterAnalyzer, and API-compatible routing.
"""
import pytest
import warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU_HARDWARE = "SapphireRapids_CPU"
MODEL = "meta-llama/Meta-Llama-3.1-8B"


def _suppress_genz_warnings():
    """Suppress GenZ fallback warnings during tests."""
    warnings.filterwarnings("ignore", message=".*GenZ.*")
    warnings.filterwarnings("ignore", message=".*not found.*")
    warnings.filterwarnings("ignore", message=".*heuristic.*")


# ---------------------------------------------------------------------------
# 1. ServingSimulator with CPU hardware
# ---------------------------------------------------------------------------

class TestServingSimulatorCPU:
    """ServingSimulator should run end-to-end with CPU hardware."""

    def test_simulate_completes(self):
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import (
            ServingSimulator, WorkloadConfig, SchedulerConfig,
        )
        sim = ServingSimulator(
            model=MODEL, hardware=CPU_HARDWARE, precision="bf16",
        )
        wl = WorkloadConfig(
            arrival_rate_rps=2.0, num_requests=5,
            model=MODEL, random_seed=42,
        )
        sc = SchedulerConfig(max_batch_size=4, max_num_batched_tokens=2048)
        result = sim.simulate(workload_config=wl, scheduler_config=sc)
        assert result.total_requests_completed > 0
        assert result.overall_throughput_rps > 0

    def test_default_tier_is_device_dram(self):
        """Default memory tier for CPU should be DEVICE_DRAM."""
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import ServingSimulator
        from llm_memory_calculator.genz.serving.constants import MemoryTier
        sim = ServingSimulator(model=MODEL, hardware=CPU_HARDWARE)
        tiers = sim._default_memory_tiers()
        assert len(tiers) >= 1
        assert tiers[0].tier == MemoryTier.DEVICE_DRAM


# ---------------------------------------------------------------------------
# 2. BatchScheduler with CPU hardware
# ---------------------------------------------------------------------------

class TestBatchSchedulerCPU:
    """BatchScheduler should produce positive latency estimates for CPU."""

    def test_prefill_latency_positive(self):
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import (
            BatchScheduler, SchedulerConfig, MemoryModel, MemoryTierConfig,
        )
        from llm_memory_calculator.genz.serving.constants import MemoryTier, GB_TO_BYTES
        from llm_memory_calculator.genz.serving.request import Request, Batch

        tier = MemoryTierConfig(
            tier=MemoryTier.DEVICE_DRAM,
            capacity_bytes=300 * GB_TO_BYTES,
            bandwidth_gbps=180.0,
        )
        from types import SimpleNamespace
        model_config = SimpleNamespace(
            num_key_value_heads=8, head_dim=128,
            num_decoder_layers=32, num_attention_heads=32,
            hidden_size=4096, vocab_size=32000,
            intermediate_size=11008,
        )
        mm = MemoryModel(model_config=model_config, tier_configs=[tier])
        sc = SchedulerConfig(max_batch_size=4)
        bs = BatchScheduler(
            model=MODEL, hardware=CPU_HARDWARE,
            memory_model=mm, config=sc,
        )

        req = Request(
            request_id=0, model=MODEL, input_tokens=128,
            max_output_tokens=32, arrival_time_ns=0,
        )
        bs.add_request(req)
        batch = bs.schedule(0)
        assert batch is not None
        latency = bs.estimate_batch_latency_ms(batch)
        assert latency > 0


# ---------------------------------------------------------------------------
# 3. PowerModel for CPU
# ---------------------------------------------------------------------------

class TestPowerModelCPU:
    """CPU power model should have reasonable power values."""

    def test_from_cpu_hardware_creates_model(self):
        from llm_memory_calculator.genz.serving import PowerModel
        pm = PowerModel.from_cpu_hardware(CPU_HARDWARE)
        assert pm is not None

    def test_idle_power_positive(self):
        from llm_memory_calculator.genz.serving import PowerModel
        pm = PowerModel.from_cpu_hardware(CPU_HARDWARE)
        idle = pm.get_base_power_w()
        assert idle > 0

    def test_active_power_greater_than_idle(self):
        from llm_memory_calculator.genz.serving import PowerModel
        from llm_memory_calculator.genz.serving.constants import PowerComponent
        pm = PowerModel.from_cpu_hardware(CPU_HARDWARE)
        accel = pm._config.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w > accel.idle_power_w


# ---------------------------------------------------------------------------
# 4. DisaggregationAnalyzer with CPU
# ---------------------------------------------------------------------------

class TestDisaggregationCPU:
    """DisaggregationAnalyzer should work with CPU hardware."""

    def test_analyze_returns_results(self):
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import DisaggregationAnalyzer
        da = DisaggregationAnalyzer(model=MODEL, hardware=CPU_HARDWARE)
        result = da.analyze(
            prefill_instances=2, decode_instances=2,
            input_tokens=128, output_tokens=32,
        )
        assert "system_throughput_rps" in result
        assert result["system_throughput_rps"] > 0


# ---------------------------------------------------------------------------
# 5. ClusterAnalyzer with CPU
# ---------------------------------------------------------------------------

class TestClusterCPU:
    """ClusterAnalyzer should work with CPU and limit TP to socket count."""

    def test_optimize_parallelism_limits_tp(self):
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import ClusterAnalyzer
        ca = ClusterAnalyzer(model=MODEL, hardware=CPU_HARDWARE)
        result = ca.optimize_parallelism(
            num_devices=4, input_tokens=128, output_tokens=32,
        )
        # CPU mode limits TP to [1,2] and PP to [1]
        for cfg in result["all_configs"]:
            assert cfg["tensor_parallel"] <= 2
            assert cfg["pipeline_parallel"] <= 1

    def test_scaling_analysis_works(self):
        _suppress_genz_warnings()
        from llm_memory_calculator.genz.serving import ClusterAnalyzer
        ca = ClusterAnalyzer(model=MODEL, hardware=CPU_HARDWARE)
        result = ca.analyze_scaling(
            instance_counts=[1, 2], input_tokens=128, output_tokens=32,
        )
        assert len(result["scaling_results"]) == 2


# ---------------------------------------------------------------------------
# 6. Routing helper
# ---------------------------------------------------------------------------

class TestGetModelingFunctions:
    """get_modeling_functions should return CPU-aware functions for CPU hardware."""

    def test_cpu_hardware_returns_cpu_functions(self):
        from llm_memory_calculator.genz.serving import get_modeling_functions
        from llm_memory_calculator.genz.cpu import (
            cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling,
        )
        pfn, dfn = get_modeling_functions(CPU_HARDWARE)
        assert pfn is cpu_aware_prefill_moddeling
        assert dfn is cpu_aware_decode_moddeling

    def test_gpu_hardware_returns_standard_functions(self):
        from llm_memory_calculator.genz.serving import get_modeling_functions
        from llm_memory_calculator.genz import prefill_moddeling, decode_moddeling
        pfn, dfn = get_modeling_functions("A100_80GB_GPU")
        assert pfn is prefill_moddeling
        assert dfn is decode_moddeling


# ---------------------------------------------------------------------------
# 7. is_cpu_hardware helper
# ---------------------------------------------------------------------------

class TestIsCpuHardware:
    """is_cpu_hardware should correctly identify CPU entries."""

    def test_cpu_entries(self):
        from llm_memory_calculator.hardware.configs import is_cpu_hardware
        assert is_cpu_hardware("SapphireRapids_CPU") is True
        assert is_cpu_hardware("Genoa_CPU") is True

    def test_gpu_entries(self):
        from llm_memory_calculator.hardware.configs import is_cpu_hardware
        assert is_cpu_hardware("A100_80GB_GPU") is False
        assert is_cpu_hardware("H100_GPU") is False

    def test_unknown_entry(self):
        from llm_memory_calculator.hardware.configs import is_cpu_hardware
        assert is_cpu_hardware("nonexistent_hardware_xyz") is False


# ---------------------------------------------------------------------------
# 8. DEVICE_DRAM tier enum
# ---------------------------------------------------------------------------

class TestDeviceDramTier:
    """MemoryTier.DEVICE_DRAM should exist and work in MemoryModel."""

    def test_enum_exists(self):
        from llm_memory_calculator.genz.serving.constants import MemoryTier
        assert hasattr(MemoryTier, "DEVICE_DRAM")
        assert MemoryTier.DEVICE_DRAM.value == "device_dram"

    def test_memory_model_accepts_dram_tier(self):
        from llm_memory_calculator.genz.serving import MemoryModel, MemoryTierConfig
        from llm_memory_calculator.genz.serving.constants import MemoryTier, GB_TO_BYTES
        from types import SimpleNamespace
        config = SimpleNamespace(
            num_key_value_heads=8, head_dim=128,
            num_decoder_layers=32, num_attention_heads=32,
        )
        tier = MemoryTierConfig(
            tier=MemoryTier.DEVICE_DRAM,
            capacity_bytes=300 * GB_TO_BYTES,
            bandwidth_gbps=180.0,
        )
        mm = MemoryModel(model_config=config, tier_configs=[tier])
        snap = mm.memory_snapshot()
        assert "device_dram" in snap
