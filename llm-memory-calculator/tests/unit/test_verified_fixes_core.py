"""Regression tests for verified bug fixes in core engine and hardware configurations.

Each test targets a specific bug fix to prevent regressions. Tests cover:
- Hardware config accuracy (GH200, H100, H200, B200, H100 PCIe, MI300X)
- System memory management (on-chip memory release capping)
- Precision multiplier correctness (INT4, FP8)
- Attention layer expert-parallel independence
"""

import warnings

import pytest

from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
from llm_memory_calculator.genz.system import System
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.genz.Models.attention import (
    mha_flash_attention_prefill,
    mha_flash_attention_decode,
)
from llm_memory_calculator.genz.Models import ModelConfig
from llm_memory_calculator.genz.parallelism import ParallelismConfig


# ---------------------------------------------------------------------------
# 1. GH200 Flops should match H100 Flops (both 989.5 dense BF16 TFLOPS)
# ---------------------------------------------------------------------------
def test_gh200_flops_matches_h100():
    """GH200 (Grace Hopper Superchip) uses the same H100 GPU die.

    Bug fix: GH200 Flops was previously set to a different value than H100.
    Both should report 989 dense BF16 TFLOPS.
    """
    gh200_flops = HARDWARE_CONFIGS['GH200_GPU']['Flops']
    h100_flops = HARDWARE_CONFIGS['H100_GPU']['Flops']
    assert gh200_flops == h100_flops, (
        f"GH200 Flops ({gh200_flops}) should equal H100 Flops ({h100_flops})"
    )
    # Accuracy remediation (F1): dense bf16 per-chip is 989.5 TFLOPS. The prior 1979 was the 2:4
    # SPARSE figure, which is not achievable by dense LLM GEMMs (it made H100 prefill imply ~160% MFU).
    assert gh200_flops == 989.5, f"Expected 989.5 dense BF16 TFLOPS, got {gh200_flops}"


# ---------------------------------------------------------------------------
# 2. H100 SXM Memory_BW should be 3350 GB/s
# ---------------------------------------------------------------------------
def test_h100_memory_bandwidth():
    """H100 SXM uses HBM3 at 3350 GB/s.

    Bug fix: Memory bandwidth was previously reported incorrectly.
    """
    mem_bw = HARDWARE_CONFIGS['H100_GPU']['Memory_BW']
    assert mem_bw == 3350, f"H100 SXM Memory_BW should be 3350, got {mem_bw}"


# ---------------------------------------------------------------------------
# 3. H200 config should exist with correct specs
# ---------------------------------------------------------------------------
def test_h200_config_exists():
    """H200 uses the Hopper architecture with HBM3e (141 GB, 4800 GB/s, 989 TFLOPS).

    Bug fix: H200 was previously missing from hardware configs.
    """
    assert 'H200_GPU' in HARDWARE_CONFIGS, "H200_GPU missing from HARDWARE_CONFIGS"

    h200 = HARDWARE_CONFIGS['H200_GPU']
    assert h200['Memory_size'] == 141, (
        f"H200 Memory_size should be 141 GB, got {h200['Memory_size']}"
    )
    assert h200['Memory_BW'] == 4800, (
        f"H200 Memory_BW should be 4800 GB/s, got {h200['Memory_BW']}"
    )
    assert h200['Flops'] == 989.5, (  # F1: dense bf16 (H200 = H100 die); 1979 was the 2:4 sparse figure
        f"H200 Flops should be 989.5 BF16 dense TFLOPS, got {h200['Flops']}"
    )


# ---------------------------------------------------------------------------
# 4. B200 config should exist with correct specs
# ---------------------------------------------------------------------------
def test_b200_config_exists():
    """B200 (Blackwell) should have 192 GB HBM3e, 8000 GB/s, 2250 BF16 TFLOPS.

    Bug fix: B200 was previously missing from hardware configs.
    """
    assert 'B200_GPU' in HARDWARE_CONFIGS, "B200_GPU missing from HARDWARE_CONFIGS"

    b200 = HARDWARE_CONFIGS['B200_GPU']
    assert b200['Memory_size'] == 192, (
        f"B200 Memory_size should be 192 GB, got {b200['Memory_size']}"
    )
    assert b200['Memory_BW'] == 8000, (
        f"B200 Memory_BW should be 8000 GB/s, got {b200['Memory_BW']}"
    )
    assert b200['Flops'] == 2250, (
        f"B200 Flops should be 2250 BF16 TFLOPS, got {b200['Flops']}"
    )


# ---------------------------------------------------------------------------
# 5. H100 PCIe variant should exist with correct specs
# ---------------------------------------------------------------------------
def test_h100_pcie_config_exists():
    """H100 PCIe has lower compute and memory bandwidth than SXM variant.

    Bug fix: H100 PCIe was previously missing or had incorrect specs.
    """
    assert 'H100_PCIe_GPU' in HARDWARE_CONFIGS, "H100_PCIe_GPU missing from HARDWARE_CONFIGS"

    h100_pcie = HARDWARE_CONFIGS['H100_PCIe_GPU']
    assert h100_pcie['Flops'] == 756, (  # F1: dense bf16 (NVIDIA H100 PCIe datasheet); 1513 was 2:4 sparse
        f"H100 PCIe Flops should be 756 BF16 dense TFLOPS, got {h100_pcie['Flops']}"
    )
    assert h100_pcie['Memory_BW'] == 2000, (
        f"H100 PCIe Memory_BW should be 2000 GB/s, got {h100_pcie['Memory_BW']}"
    )


# ---------------------------------------------------------------------------
# 6. MI300X ICN should be 896 GB/s (7 IF links x 128 GB/s)
# ---------------------------------------------------------------------------
def test_mi300x_interconnect():
    """MI300X uses 7 Infinity Fabric links at 128 GB/s each = 896 GB/s.

    Bug fix: MI300X ICN was previously set to an incorrect value.
    """
    assert 'MI300X' in HARDWARE_CONFIGS, "MI300X missing from HARDWARE_CONFIGS"
    icn = HARDWARE_CONFIGS['MI300X']['ICN']
    assert icn == 896, f"MI300X ICN should be 896 GB/s, got {icn}"


# ---------------------------------------------------------------------------
# 7. On-chip memory release should be capped at total capacity
# ---------------------------------------------------------------------------
def test_release_onchip_mem_capped():
    """After releasing memory, available on-chip memory must not exceed total capacity.

    Bug fix: release_onchip_mem previously did not clamp to on_chip_mem_size,
    allowing available memory to exceed total capacity.
    """
    unit = Unit()
    sys = System(
        unit,
        frequency=1000,
        flops=100,
        on_chip_mem_size=1000,
        off_chip_mem_size=80 * 1024,
        compute_efficiency=1.0,
        memory_efficiency=1.0,
    )

    # Claim 500 then release 500: should return to original state
    sys.claim_onchip_mem(500)
    sys.release_onchip_mem(500)
    assert sys.on_chip_mem_left_size <= sys.on_chip_mem_size, (
        f"After balanced claim/release, available ({sys.on_chip_mem_left_size}) "
        f"should not exceed total ({sys.on_chip_mem_size})"
    )

    # Releasing more than was ever claimed should still be capped
    sys.release_onchip_mem(99999)
    assert sys.on_chip_mem_left_size == sys.on_chip_mem_size, (
        f"After excessive release, available ({sys.on_chip_mem_left_size}) "
        f"should equal total ({sys.on_chip_mem_size})"
    )


# ---------------------------------------------------------------------------
# 8. INT4 compute multiplier should be 0.5 (W4A16), not 0.25
# ---------------------------------------------------------------------------
def test_int4_compute_multiplier():
    """INT4 weights are dequantized to FP16 for compute (W4A16 pattern).

    Bug fix: INT4 compute multiplier was previously 0.25 (as if both weights
    and activations were 4-bit), but standard W4A16 quantization only halves
    compute relative to FP16 baseline.
    """
    assert System.compute_multiplier['int4'] == 0.5, (
        f"INT4 compute multiplier should be 0.5, got {System.compute_multiplier['int4']}"
    )


# ---------------------------------------------------------------------------
# 9. FP8 precision should produce a hardware compatibility warning
# ---------------------------------------------------------------------------
def test_fp8_hardware_warning():
    """FP8 hardware-compatibility warning is ARCHITECTURE-GATED (accuracy remediation L1).

    The warning moved from System.__init__ (which has no architecture and fired unconditionally — noise
    even on FP8-capable H100) to get_inference_system, where the device arch is known. So:
      - A bare System(bits='fp8') must NOT warn (no arch context).
      - get_inference_system on a NON-FP8 arch (A100/Ampere) with fp8 MUST warn.
      - get_inference_system on an FP8-capable arch (H100/Hopper) must NOT warn.
    """
    from llm_memory_calculator.genz.LLM_inference.utils import get_inference_system

    unit = Unit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        System(unit, frequency=1000, flops=100, off_chip_mem_size=80 * 1024, bits='fp8')
        assert not [w for w in caught if 'fp8' in str(w.message).lower()], \
            "Bare System(fp8) should NOT warn — it has no architecture to judge FP8 support"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_inference_system(system_name='A100_80GB_GPU', bits='fp8', phase='decode')
        assert [w for w in caught if 'fp8' in str(w.message).lower()], \
            "Ampere (A100) lacks FP8 tensor cores — get_inference_system(fp8) should warn"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_inference_system(system_name='H100_GPU', bits='fp8', phase='decode')
        assert not [w for w in caught if 'fp8' in str(w.message).lower()], \
            "Hopper (H100) supports FP8 — get_inference_system(fp8) should NOT warn"


# ---------------------------------------------------------------------------
# 10. Expert parallelism should NOT affect attention layer dimensions
# ---------------------------------------------------------------------------
def test_attention_ep_independent():
    """Attention dimensions depend on tensor_parallel but are independent of
    expert_parallel (EP only affects MoE FFN layers).

    Bug fix: EP was previously factored into attention head partitioning,
    producing incorrect QKV/logit/attend dimensions.
    """
    model_cfg = ModelConfig(
        model='test_ep_attention',
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        num_decoder_layers=32,
        intermediate_size=11008,
        vocab_size=32000,
    )

    pc_no_ep = ParallelismConfig(tensor_parallel=4, expert_parallel=1)
    pc_with_ep = ParallelismConfig(tensor_parallel=4, expert_parallel=2)

    # Prefill layers should be identical regardless of EP
    layers_no_ep = mha_flash_attention_prefill(model_cfg, pc_no_ep, 2048)
    layers_with_ep = mha_flash_attention_prefill(model_cfg, pc_with_ep, 2048)
    assert layers_no_ep == layers_with_ep, (
        "Prefill attention layers should be identical regardless of expert_parallel.\n"
        f"Without EP: {layers_no_ep}\n"
        f"With EP:    {layers_with_ep}"
    )

    # Decode layers should also be identical regardless of EP
    decode_no_ep = mha_flash_attention_decode(model_cfg, pc_no_ep, 2048, 128)
    decode_with_ep = mha_flash_attention_decode(model_cfg, pc_with_ep, 2048, 128)
    assert decode_no_ep == decode_with_ep, (
        "Decode attention layers should be identical regardless of expert_parallel.\n"
        f"Without EP: {decode_no_ep}\n"
        f"With EP:    {decode_with_ep}"
    )
