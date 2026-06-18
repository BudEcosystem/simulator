"""F2/F4 + C5/C6/C7: hardware single source of truth.

With the DB enabled in the manager: (1) every device with a static config must resolve to its
F1-corrected DENSE bf16 value (STATIC-WINS — immune to DB drift, the anti-inversion guard);
(2) no static device resolves to a sparse value; (3) all catalog devices are simulatable (C5);
(4) DB-only devices resolve to their migrated dense values.
"""
import os
import warnings
import pytest

warnings.filterwarnings("ignore")

from llm_memory_calculator.hardware import set_hardware_db_path, get_hardware_config, get_all_hardware
from llm_memory_calculator.hardware.manager import HardwareManager
from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
from llm_memory_calculator.genz.LLM_inference.utils import get_inference_system

# Locate the BudSimulator prepopulated DB (the catalog source).
_DB = os.environ.get("BUDSIM_HARDWARE_DB")
if not _DB:
    _here = os.path.dirname(os.path.abspath(__file__))
    _cand = os.path.normpath(os.path.join(_here, "..", "..", "BudSimulator", "data", "prepopulated.db"))
    _DB = _cand if os.path.exists(_cand) else None

pytestmark = pytest.mark.skipif(_DB is None, reason="prepopulated.db not found")


@pytest.fixture(scope="module", autouse=True)
def _enable_db():
    set_hardware_db_path(_DB)   # rebuilds the module-level manager with the DB
    yield
    set_hardware_db_path(_DB)   # leave it enabled for the rest of the session


# F1-corrected dense bf16 values that MUST survive the DB merge (static-wins).
F1_DENSE = {
    "H100_GPU": 989.5, "H200_GPU": 989.5, "GH200_GPU": 989.5, "H100_PCIe_GPU": 756,
    "RTX4090_GPU": 330, "B100": 1750, "GB200": 2250, "B200_GPU": 2250, "MI300X": 1307,
}


@pytest.mark.parametrize("name,dense", list(F1_DENSE.items()))
def test_static_dense_values_survive_db_merge(name, dense):
    """STATIC-WINS: the F1 dense value must be what resolves, even though the DB may hold a different
    (possibly sparse) number for the same name."""
    cfg = get_hardware_config(name)
    assert cfg, f"{name} did not resolve"
    assert cfg["Flops"] == pytest.approx(dense, rel=1e-6), \
        f"{name} resolved to {cfg['Flops']} (expected static dense {dense}) — DB overrode static!"


def test_rtx4090_inversion_sentinel():
    """The verified inversion: DB RTX4090_GPU=330.3 (now migrated) but static=330. Static-wins → 330.
    Critically it must NOT be the old sparse 661."""
    cfg = get_hardware_config("RTX4090_GPU")
    assert cfg["Flops"] != 661, "RTX4090 resolved to the sparse 661 — inversion!"
    assert cfg["Flops"] == pytest.approx(330, rel=0.02)


def test_no_static_device_resolves_to_a_sparse_value():
    """Every device that has a static config must resolve to EXACTLY its static Flops."""
    for name, scfg in HARDWARE_CONFIGS.items():
        if not isinstance(scfg, dict) or "Flops" not in scfg:
            continue
        resolved = get_hardware_config(name)
        assert resolved and resolved["Flops"] == pytest.approx(scfg["Flops"], rel=1e-6), \
            f"{name}: resolved {resolved.get('Flops') if resolved else None} != static {scfg['Flops']}"


def test_all_catalog_devices_are_simulatable():
    """C5: every catalog device, addressed by its CANONICAL KEY (the id used to select it, not the
    free-text display 'name'), must build a System via the engine path — no ValueError 'not in
    predefined systems'. Previously the 18 DB-only GPU/accelerator devices 404'd."""
    mgr = HardwareManager(_DB)
    keys = list(mgr.get_all_hardware_dict().keys())
    assert len(keys) >= 90, f"expected ~92 devices, got {len(keys)}"
    unresolved = []
    for key in keys:
        try:
            sysobj = get_inference_system(system_name=key, bits="bf16", phase="decode")
            assert sysobj is not None
        except Exception as e:
            unresolved.append((key, type(e).__name__))
    assert not unresolved, f"{len(unresolved)}/{len(keys)} catalog devices NOT simulatable: {unresolved[:10]}"


@pytest.mark.parametrize("name,dense", [
    ("H100_SXM_GPU", 989.5), ("H800_GPU", 989.5), ("MI300X_GPU", 1307),
    ("V100_GPU", 125), ("TPU_v2", 46), ("TPU_v3", 123),
])
def test_db_only_devices_resolve_to_dense(name, dense):
    """DB-only devices (no static config) must resolve to their migrated dense bf16 value."""
    cfg = get_hardware_config(name)
    assert cfg, f"{name} did not resolve"
    assert cfg["Flops"] == pytest.approx(dense, rel=1e-6), f"{name}={cfg['Flops']} (expected {dense})"
