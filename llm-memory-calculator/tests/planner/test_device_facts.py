"""FRD-027 §6, T-DEV-01…06: device facts are the predictor's one hardware input."""

import pytest

from llm_memory_calculator.device import (
    DeviceFacts,
    HardwareMode,
    allows_dtype,
    catalogue,
    facts_from_node,
    host_register_allowed,
    kernel_path,
    max_tp,
)

GIB = 1024**3


def test_vgpu_is_its_own_sku_with_visible_memory():
    vgpu = catalogue.get("h100xm-80c-vgpu")
    bare = catalogue.get("h100-sxm5-80gb")
    assert vgpu.sku == "h100xm-80c-vgpu" and bare.sku == "h100-sxm5-80gb"
    assert vgpu.architecture == bare.architecture == "hopper"
    # 79.67 GiB visible of 80 nameplate (FRD §6); not the nameplate.
    assert abs(vgpu.memory_visible_bytes / GIB - 79.67) < 0.01
    assert vgpu.memory_nameplate_bytes == 80 * GIB
    assert vgpu.virtualization.vgpu and not vgpu.virtualization.vmm
    assert vgpu.virtualization.klass == "vgpu_no_vmm"
    assert bare.virtualization.klass == "bare_metal"
    assert bare.interconnect.kind == "nvlink" and vgpu.interconnect.kind == "none"
    assert bare.digest() != vgpu.digest()


def test_catalogue_lookup_by_product_strings():
    for product in ("NVIDIA-H100XM-80C", "GRID H100XM-80C", "H100XM-80C", "nvidia h100xm-80c"):
        assert catalogue.lookup(product).sku == "h100xm-80c-vgpu", product
    assert catalogue.lookup("NVIDIA-H100-80GB-HBM3").sku == "h100-sxm5-80gb"
    assert catalogue.lookup("NVIDIA A100-SXM4-80GB").sku == "a100-80gb"
    assert catalogue.lookup("NVIDIA-GeForce-RTX-3050-Laptop-GPU").sku == "rtx3050-laptop-4gb"
    assert catalogue.lookup("no such card") is None


def test_hardware_configs_records_map_onto_facts():
    """FR-DEV-1: every hardware/configs.py GPU record becomes a catalogue entry (additive)."""
    facts = catalogue.from_hardware_config("L40S_48GB_GPU")
    assert facts.architecture == "ada"
    assert facts.hbm_bandwidth == pytest.approx(864e9, rel=0.01)
    assert facts.peak_flops["bf16"] > 0
    assert allows_dtype(facts, "fp8") == "native"  # Ada has fp8 tensor cores


def test_dtype_support_gates_precision():
    a100 = catalogue.get("a100-80gb")
    assert allows_dtype(a100, "fp8") is None
    assert allows_dtype(a100, "fp8", grounded_paths={"fp8:weight_only"}) == "weight_only"
    b200 = catalogue.get("b200-180gb")
    assert allows_dtype(b200, "mxfp4") == "native"
    vgpu = catalogue.get("h100xm-80c-vgpu")
    assert allows_dtype(vgpu, "fp8") == "native"
    assert allows_dtype(vgpu, "mxfp4") is None
    assert allows_dtype(vgpu, "mxfp4", record_features={"mxfp4_triton"}) == "emulated"
    assert kernel_path(vgpu, {"mxfp4_triton"}, "mxfp4") == "mxfp4:triton"
    assert kernel_path(b200, set(), "mxfp4") == "mxfp4:native"
    assert kernel_path(vgpu, set(), "fp8") == "fp8:native"
    assert kernel_path(vgpu, set(), "bf16") == "bf16:native"
    assert allows_dtype(vgpu, "bf16") == "native"


def test_unknown_sku_is_derived_from_architecture():
    facts = facts_from_node(
        {"product": "NVIDIA-H999-94GB", "gpu_memory_mb": 94 * 1024, "compute_capability": "9.0"}
    )
    assert facts.derived is True
    assert facts.architecture == "hopper"
    assert facts.sku.startswith("derived:hopper")
    hopper = catalogue.architecture_defaults("hopper")
    # bandwidth scaled by the reported memory against the architecture's reference part
    assert facts.hbm_bandwidth == pytest.approx(hopper.hbm_bandwidth * 94 / 80, rel=0.01)
    assert facts.sources["hbm_bandwidth"] == "derived"
    assert facts.sources["memory_visible_bytes"] == "live"
    # nothing at all known: still plans (NFR-5)
    blind = facts_from_node({"product": "mystery accelerator", "gpu_memory_mb": 16384})
    assert (
        blind.derived and blind.architecture == "unknown" and blind.memory_visible_bytes == 16 * GIB
    )


def test_virtualization_gates_tp():
    vgpu = catalogue.get("h100xm-80c-vgpu")
    bare = catalogue.get("h100-sxm5-80gb")
    limits = {"max_tp": 8, "max_tp_by_virtualization": {"vgpu_no_vmm": 1}}
    assert max_tp(vgpu, limits) == 1
    assert max_tp(bare, limits) == 8
    assert max_tp(bare, {"max_tp": 4}) == 4
    # without a record override, a vGPU with no VMM still cannot do TP > 1
    assert max_tp(vgpu, {"max_tp": 8}) == 1
    assert host_register_allowed(vgpu) is False and host_register_allowed(bare) is True


def test_live_overrides_catalogue_and_records_sources():
    facts = facts_from_node({"product": "NVIDIA-H100XM-80C", "gpu_memory_mb": 81559})
    cat = catalogue.get("h100xm-80c-vgpu")
    assert facts.sku == "h100xm-80c-vgpu" and not facts.derived
    assert facts.memory_visible_bytes == 81559 * 1024**2
    assert facts.hbm_bandwidth == cat.hbm_bandwidth
    assert facts.sources["memory_visible_bytes"] == "live"
    assert facts.sources["hbm_bandwidth"] == "catalogue"


def test_budsim_device_record_shape():
    """budsim's cluster_info device rows (name, mem_per_gpu_in_gb, type) are accepted as is."""
    facts = facts_from_node(
        {"name": "H100XM-80C", "type": "cuda", "mem_per_gpu_in_gb": 79.67, "available_count": 2}
    )
    assert facts.sku == "h100xm-80c-vgpu"
    assert abs(facts.memory_visible_bytes / GIB - 79.67) < 0.01


def test_shared_mode_uses_the_slice():
    vgpu = catalogue.get("h100xm-80c-vgpu")
    shared = vgpu.with_mode(HardwareMode(kind="shared", slice_bytes=66956 * 1024**2))
    assert shared.usable_memory_bytes == 66956 * 1024**2
    assert vgpu.usable_memory_bytes == vgpu.memory_visible_bytes
    assert shared.digest() != vgpu.digest()


def test_round_trip_json():
    vgpu = catalogue.get("h100xm-80c-vgpu")
    again = DeviceFacts.from_dict(vgpu.to_dict())
    assert again == vgpu and again.digest() == vgpu.digest()


def test_cpu_device():
    facts = facts_from_node({"name": "Xeon", "type": "cpu", "mem_per_gpu_in_gb": 256})
    assert facts.device_class == "cpu"
    assert facts.derived
