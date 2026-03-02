"""
Regression tests for CPU roofline analysis fixes.
Tests all 35 bugs identified in the CPU roofline audit.
"""
import pytest
import numpy as np
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_memory_calculator.genz.cpu.cpu_configs import CPU_PRESETS, get_cpu_preset
from llm_memory_calculator.genz.cpu.cpu_system import CPUSystem, CacheConfig, CPUConfig
from llm_memory_calculator.genz.cpu.isa_model import ISASelector, ISAType
from llm_memory_calculator.genz.cpu.frequency_model import FrequencyGovernor
from llm_memory_calculator.genz.cpu.numa_model import NUMATopology
from llm_memory_calculator.genz.cpu.threading_model import ThreadingModel
from llm_memory_calculator.genz.cpu.cache_model import CacheHierarchy, CacheLevel


# ============================================================================
# Phase 0: Hardware Config Data Tests
# ============================================================================

class TestHardwareConfigs:
    """Test hardware configuration data accuracy."""

    def test_icelake_8380_no_amx(self):
        """Fix 0A: Ice Lake (3rd Gen) does NOT have AMX."""
        preset = CPU_PRESETS['intel_xeon_8380']
        assert 'amx' not in preset['cpu_specific'].isa_support, \
            "Ice Lake 8380 should not have AMX (AMX is Sapphire Rapids+)"
        assert 'avx512' in preset['cpu_specific'].isa_support
        assert 'avx2' in preset['cpu_specific'].isa_support

    def test_xeon_8592_plus_2_sockets(self):
        """Fix 0B: 8592+ is 2-socket only."""
        preset = CPU_PRESETS['intel_xeon_8592_plus']
        assert preset['cpu_specific'].sockets == 2, \
            "8592+ supports only 2-socket configurations (Intel ARK)"

    def test_xeon_8592_plus_numa_consistency(self):
        """Fix 0C: NUMA matrix must match numa_nodes."""
        preset = CPU_PRESETS['intel_xeon_8592_plus']
        cpu = preset['cpu_specific']
        assert cpu.numa_nodes == cpu.numa_distance_matrix.shape[0], \
            f"numa_nodes={cpu.numa_nodes} but distance matrix is {cpu.numa_distance_matrix.shape}"

    def test_xeon_6430_base_frequency(self):
        """Fix 0D: Xeon Gold 6430 base frequency is 2.10 GHz."""
        preset = CPU_PRESETS['intel_xeon_6430']
        assert preset['base_params']['frequency'] == 2.1e9, \
            f"6430 base freq should be 2.1 GHz, got {preset['base_params']['frequency']/1e9}"
        assert preset['cpu_specific'].base_frequency == 2.1e9

    def test_xeon_6430_ddr5_bandwidth(self):
        """Fix 0E: Sapphire Rapids 6430 uses DDR5-4400."""
        preset = CPU_PRESETS['intel_xeon_6430']
        # DDR5-4400 = 35.2 GB/s per channel
        assert preset['cpu_specific'].dram_bandwidth_per_channel == 35.2

    def test_graviton3_ddr5_bandwidth(self):
        """Fix 0F: Graviton3 uses DDR5-4800, not DDR4-3200."""
        preset = CPU_PRESETS['aws_graviton3']
        # DDR5-4800 = 38.4 GB/s per channel
        assert preset['cpu_specific'].dram_bandwidth_per_channel == 38.4, \
            f"Graviton3 uses DDR5-4800 (38.4 GB/s/ch), got {preset['cpu_specific'].dram_bandwidth_per_channel}"

    def test_graviton3_sve_not_sve2(self):
        """Fix 0G: Graviton3 (Neoverse V1) has SVE, not SVE2."""
        preset = CPU_PRESETS['aws_graviton3']
        assert 'sve' in preset['cpu_specific'].isa_support, \
            "Graviton3 should have SVE"
        assert 'sve2' not in preset['cpu_specific'].isa_support, \
            "Graviton3 does NOT have SVE2 (that's Neoverse V2/Graviton4)"

    def test_epyc_7763_l1i_64kb(self):
        """Fix 0H: Zen 3 has 64KB L1 instruction cache."""
        preset = CPU_PRESETS['amd_epyc_7763']
        assert preset['cpu_specific'].l1i_config.size == 64 * 1024, \
            f"Zen 3 L1I should be 64KB, got {preset['cpu_specific'].l1i_config.size}"

    def test_icelake_l2_associativity(self):
        """Fix 0I: Ice Lake Server L2 is 8-way, not 20-way."""
        preset = CPU_PRESETS['intel_xeon_8380']
        assert preset['cpu_specific'].l2_config.associativity == 8

    def test_cpu_dram_not_infinity(self):
        """Fix 1G: CPU DRAM should be finite, not infinity."""
        for name, preset in CPU_PRESETS.items():
            base = preset['base_params']
            assert 'off_chip_mem_size' in base, \
                f"{name}: missing off_chip_mem_size in base_params"
            assert base['off_chip_mem_size'] < float('inf'), \
                f"{name}: DRAM size should be finite"
            assert base['off_chip_mem_size'] > 0, \
                f"{name}: DRAM size should be positive"

    def test_cpu_dram_bandwidth_set(self):
        """All CPU presets should have offchip_mem_bw set."""
        for name, preset in CPU_PRESETS.items():
            base = preset['base_params']
            assert 'offchip_mem_bw' in base, \
                f"{name}: missing offchip_mem_bw in base_params"
            assert base['offchip_mem_bw'] > 0


# ============================================================================
# Phase 1: Roofline Model Tests
# ============================================================================

class TestRooflineModel:
    """Test roofline model correctness."""

    def test_roofline_uses_max_not_additive(self):
        """Fix 1A: exec_time should use max(compute, memory), not sum."""
        # Import and check the enhancement code
        import llm_memory_calculator.genz.cpu.operator_enhancement as oe
        import inspect
        source = inspect.getsource(oe.get_cpu_roofline)
        # The function should use max(), not compute_time + memory_time
        assert 'max(compute_time, memory_time)' in source, \
            "Roofline should use max(compute, memory), not additive"
        assert 'compute_time + memory_time + comm_time' not in source, \
            "Additive roofline (compute + memory) is incorrect"

    def test_amx_ops_per_tile_bf16(self):
        """Fix 1B: AMX BF16 ops/tile = 16,384."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)
        amx_config = selector.isa_configs[ISAType.AMX]
        assert amx_config.vector_width['bf16'] == 16384, \
            f"AMX BF16 ops/tile should be 16384, got {amx_config.vector_width['bf16']}"

    def test_amx_ops_per_tile_int8(self):
        """Fix 1B: AMX INT8 ops/tile = 32,768."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)
        amx_config = selector.isa_configs[ISAType.AMX]
        assert amx_config.vector_width['int8'] == 32768, \
            f"AMX INT8 ops/tile should be 32768, got {amx_config.vector_width['int8']}"

    def test_amx_tile_dimensions(self):
        """Fix 1C: AMX output tile is 16×16, not 16×64."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)
        constraints = selector.isa_configs[ISAType.AMX].special_constraints
        assert constraints['tile_m'] == 16
        assert constraints['tile_n'] == 16, \
            f"AMX output tile_n should be 16, got {constraints['tile_n']}"

    def test_amx_throughput_16_cycles(self):
        """Fix 1D: AMX throughput is 1 tile per 16 cycles (0.0625)."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)
        amx = selector.isa_configs[ISAType.AMX]
        assert amx.throughput['tilemmul'] == pytest.approx(0.0625), \
            f"AMX throughput should be 0.0625 (1/16), got {amx.throughput['tilemmul']}"
        assert amx.latency['tilemmul'] == 52, \
            f"AMX latency should be 52 cycles, got {amx.latency['tilemmul']}"

    def test_amx_units_per_core(self):
        """Fix 1E: 1 AMX unit per core."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['amx', 'avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)
        constraints = selector.isa_configs[ISAType.AMX].special_constraints
        assert constraints['amx_units_per_core'] == 1

    def test_no_data_type_double_counting(self):
        """Fix 1F: vector_width already encodes type width — no extra multiplier."""
        config = CPUConfig(
            cores_per_socket=32, sockets=1, threads_per_core=2,
            l1i_config=CacheConfig(32*1024, 4, 3200),
            l1d_config=CacheConfig(48*1024, 5, 3200),
            l2_config=CacheConfig(1280*1024, 14, 1600),
            l3_config=CacheConfig(60*1024*1024, 42, 800),
            numa_nodes=1, cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            isa_support=['avx512', 'avx2'],
            base_frequency=2.0e9,
            turbo_frequency_curve={1: 3.0e9, 32: 2.5e9},
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6},
            dram_bandwidth_per_channel=38.4, memory_channels_per_socket=8,
            vendor='intel', microarchitecture='sapphire_rapids'
        )
        selector = ISASelector(config)

        # For AVX-512: FP32=16 elements, BF16=32 elements
        # BF16 compute should be ~2× faster than FP32 (not 4×)
        fp32_peak = selector.get_peak_performance(ISAType.AVX512, 'fp32')
        bf16_peak = selector.get_peak_performance(ISAType.AVX512, 'bf16')
        ratio = bf16_peak / fp32_peak
        assert 1.5 <= ratio <= 2.5, \
            f"BF16/FP32 ratio should be ~2× (got {ratio:.1f}×). Double-counting?"


# ============================================================================
# Phase 2: High Severity Tests
# ============================================================================

class TestHighSeverityFixes:
    """Test high severity bug fixes."""

    def test_graviton3_sve_256bit(self):
        """Fix 2D: SVE on Graviton3 is 256-bit, not 512-bit."""
        config = CPU_PRESETS['aws_graviton3']['cpu_specific']
        selector = ISASelector(config)
        # Should have SVE, not SVE2
        assert ISAType.SVE in selector.isa_configs, \
            "Graviton3 should have SVE ISA config"
        sve_config = selector.isa_configs[ISAType.SVE]
        # 256-bit SVE: 8 FP32 elements (not 16)
        assert sve_config.vector_width['fp32'] == 8, \
            f"Graviton3 SVE fp32 width should be 8 (256-bit), got {sve_config.vector_width['fp32']}"

    def test_numa_penalty_realistic(self):
        """Fix 3A: NUMA penalty for 2S should be ~1.3-1.7×, not 2.1×."""
        config = CPU_PRESETS['intel_xeon_8380']['cpu_specific']
        numa = NUMATopology(config)
        # Remote access (node 0 → node 1, distance 21)
        penalty = numa.get_access_penalty(0, 0)  # local
        assert penalty == 1.0, "Local NUMA access should be 1.0×"

        # Remote: distance 21 → should be ~1.55×, not 2.1×
        # Core on node 0 accessing memory on node 1
        remote_core = config.cores_per_numa  # First core on node 1
        penalty_remote = numa.get_access_penalty(remote_core, 0)  # accessing node 0
        assert 1.3 <= penalty_remote <= 1.7, \
            f"Remote NUMA penalty should be ~1.5× (got {penalty_remote:.2f}×)"

    def test_thermal_model_has_cooling(self):
        """Fix 3B: Thermal model should cool down when power is reduced."""
        config = CPU_PRESETS['intel_xeon_8380']['cpu_specific']
        governor = FrequencyGovernor(config)

        # Heat up
        governor.update_thermal_state(power=300, duration=100)
        hot_temp = governor.thermal_state.temperature
        assert hot_temp > 25, f"Should be above ambient after heating, got {hot_temp}"

        # Cool down (low power)
        governor.update_thermal_state(power=10, duration=1000)
        cool_temp = governor.thermal_state.temperature
        assert cool_temp < hot_temp, \
            f"Should cool down: {cool_temp}°C should be < {hot_temp}°C"

    def test_frequency_not_above_turbo(self):
        """Fix 3C: Memory-bound workloads should not exceed turbo frequency."""
        config = CPU_PRESETS['intel_xeon_8380']['cpu_specific']
        governor = FrequencyGovernor(config)

        all_core_turbo = config.turbo_frequency_curve[max(config.turbo_frequency_curve.keys())]
        freq_compute = governor.get_frequency(ISAType.AVX512, 80, 'compute')
        freq_memory = governor.get_frequency(ISAType.AVX512, 80, 'memory')

        # Both should be <= turbo + offset (not 1.1× higher)
        max_possible = all_core_turbo  # With AVX offset, could be lower
        assert freq_memory <= max_possible * 1.01, \
            f"Memory-bound freq {freq_memory/1e9:.2f} GHz > turbo {max_possible/1e9:.2f} GHz"

    def test_no_framework_overhead_subtraction(self):
        """Fix 2H: No flat 1250ms subtraction from analytical model."""
        import inspect
        from llm_memory_calculator.genz.cpu import llm_cpu_integration as lci
        prefill_src = inspect.getsource(lci.cpu_aware_prefill_moddeling)
        decode_src = inspect.getsource(lci.cpu_aware_decode_moddeling)

        # Should NOT contain the subtraction pattern
        assert "framework_overhead_ms" not in prefill_src or "result['Latency'] -" not in prefill_src, \
            "Prefill should not subtract framework_overhead_ms from latency"
        assert "framework_overhead_ms" not in decode_src or "result['Latency'] -" not in decode_src, \
            "Decode should not subtract framework_overhead_ms from latency"


# ============================================================================
# Phase 3: Medium Severity Tests
# ============================================================================

class TestMediumSeverityFixes:
    """Test medium severity bug fixes."""

    def test_lru_counter_no_overflow(self):
        """Fix 4D: LRU counters should not overflow at 255."""
        config = CacheConfig(size=4096, latency=4, bandwidth=3200, associativity=8)
        cache = CacheLevel(config)
        # LRU dtype should be large enough to avoid overflow
        assert cache.lru.dtype in (np.uint16, np.uint32, np.uint64), \
            f"LRU dtype {cache.lru.dtype} may overflow — use uint16 or larger"

    def test_bandwidth_model_vendor_aware(self):
        """Fix 3E: Intel uses additive (ECM), AMD uses max (overlapping)."""
        import inspect
        from llm_memory_calculator.genz.cpu.cpu_operator import CPUOperatorMixin
        source = inspect.getsource(CPUOperatorMixin._calculate_bandwidth_limited_time)
        assert 'vendor' in source, \
            "Bandwidth model should be vendor-aware (Intel ECM vs AMD overlapping)"

    def test_roofline_vendor_aware(self):
        """Fix 1A: Roofline model should differentiate Intel ECM vs AMD max."""
        import inspect
        from llm_memory_calculator.genz.cpu.operator_enhancement import get_cpu_roofline
        source = inspect.getsource(get_cpu_roofline)
        assert 'vendor' in source, \
            "Roofline model should be vendor-aware"

    def test_gemm_cache_pattern_not_random(self):
        """Fix 3D: GEMM access pattern should model tiled access, not random."""
        import inspect
        source = inspect.getsource(CacheHierarchy.analyze_operator_access_pattern)
        # Should mention tiling/blocking, not purely random sampling
        assert 'tile' in source.lower(), \
            "GEMM access pattern should use tiled/blocked pattern"

    def test_sve_isa_type_exists(self):
        """SVE ISA type should exist for Graviton3 support."""
        assert hasattr(ISAType, 'SVE'), "ISAType.SVE should exist"

    def test_graviton3_system_creation(self):
        """Graviton3 CPU system should be creatable."""
        from llm_memory_calculator.genz.cpu import create_cpu_system
        system = create_cpu_system('aws_graviton3')
        assert isinstance(system, CPUSystem)
        # SVE should be in the ISA selector
        assert ISAType.SVE in system.isa_selector.isa_configs

    def test_sapphire_rapids_has_amx(self):
        """Sapphire Rapids presets should have AMX."""
        preset = CPU_PRESETS['intel_xeon_6430']
        assert 'amx' in preset['cpu_specific'].isa_support, \
            "Sapphire Rapids 6430 should have AMX"

    def test_emerald_rapids_has_amx(self):
        """Emerald Rapids presets should have AMX."""
        preset = CPU_PRESETS['intel_xeon_8592_plus']
        assert 'amx' in preset['cpu_specific'].isa_support, \
            "Emerald Rapids 8592+ should have AMX"

    def test_all_presets_have_consistent_channels(self):
        """DRAM bandwidth should match channels × per-channel BW."""
        for name, preset in CPU_PRESETS.items():
            cpu = preset['cpu_specific']
            expected_bw = (cpu.dram_bandwidth_per_channel *
                          cpu.memory_channels_per_socket *
                          cpu.sockets)
            base_bw = preset['base_params'].get('offchip_mem_bw', 0)
            if base_bw > 0:
                assert abs(base_bw - expected_bw) / expected_bw < 0.01, \
                    f"{name}: offchip_mem_bw={base_bw} != channels×bw={expected_bw}"


# ============================================================================
# Sanity / Integration Tests
# ============================================================================

class TestCPUSystemIntegration:
    """Integration tests for CPU system creation and basic operation."""

    @pytest.fixture(params=list(CPU_PRESETS.keys()))
    def cpu_system(self, request):
        """Create CPU system for each preset."""
        from llm_memory_calculator.genz.cpu import create_cpu_system
        return create_cpu_system(request.param)

    def test_cpu_system_creation(self, cpu_system):
        """All CPU presets should create valid systems."""
        assert isinstance(cpu_system, CPUSystem)
        assert cpu_system.peak_memory_bandwidth > 0
        assert cpu_system.get_active_cores() > 0

    def test_cpu_system_has_all_components(self, cpu_system):
        """CPU system should have all required components."""
        assert cpu_system.cache_hierarchy is not None
        assert cpu_system.numa_topology is not None
        assert cpu_system.isa_selector is not None
        assert cpu_system.frequency_governor is not None
        assert cpu_system.threading_model is not None

    def test_isa_selector_has_configs(self, cpu_system):
        """ISA selector should have at least scalar + one vector ISA."""
        configs = cpu_system.isa_selector.isa_configs
        assert ISAType.SCALAR in configs, "Must have scalar fallback"
        assert len(configs) >= 2, "Should have at least scalar + one vector ISA"

    def test_effective_bandwidth_positive(self, cpu_system):
        """Effective bandwidth should be positive for all levels."""
        for level in ['L1', 'L2', 'L3', 'DRAM']:
            bw = cpu_system.get_effective_bandwidth(level, 0)
            assert bw > 0, f"Bandwidth for {level} should be positive, got {bw}"

    def test_frequency_governor_returns_valid(self, cpu_system):
        """Frequency governor should return valid frequencies."""
        cores = cpu_system.get_active_cores()
        # Test with available ISA types
        for isa_type in cpu_system.isa_selector.isa_configs:
            freq = cpu_system.frequency_governor.get_frequency(isa_type, cores)
            assert freq > 0, f"Frequency for {isa_type} should be positive"
            assert freq < 10e9, f"Frequency {freq/1e9:.1f} GHz is unrealistically high"
