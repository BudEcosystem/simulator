"""FRD-027 Revision 2: where step profiles live and how a prediction finds one (FR-PROF-1, §20.2 L2).

A profile is keyed by the device SKU, the engine record, the model family and the *kernel class* of the settings:
the ones that change which kernels run (the served weight format itself, KV and state dtype, speculative method and
depth, LoRA). Capacity settings (sequences, context, utilization, chunk) do not key a profile; the scheduler simulation
applies them. A lookup is exact: another build's kernels differ (the fork's GDN and MoE paths), so a missing profile
means the fitted predictor serves, as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .profile import StepProfile

DATA_DIR = Path(__file__).parent / "data"


def kernel_class(r: Any, model: Any) -> Dict[str, str]:
    """The settings that change which kernels a step runs, from a resolved configuration."""
    # the served format itself: a family's FP8 checkpoint runs other kernels than its bf16 one
    weights = f"{r.fmt}-online" if getattr(r, "online", False) else str(r.fmt)
    state = str(getattr(r, "state_dtype", "auto") or "auto")
    return {
        "weights": weights,
        "kv": str(getattr(r, "kv_dtype", "auto") or "auto"),
        "state": "auto" if state in ("float32", "auto") else state,
        "spec": f"{r.spec_method}:{int(r.spec_k)}" if getattr(r, "spec_method", None) else "none",
        "lora": "1" if getattr(r, "lora_adapters", 0) else "0",
    }


def kernel_class_of_args(checkpoint_format: str, extra: Mapping[str, Any]) -> Dict[str, str]:
    """The kernel class a profiling run selected: the checkpoint's format and the engine kwargs it ran with
    (``--extra`` of ``perf.profiler``: ``quantization``, ``kv_cache_dtype``, ``speculative_config``, ``enable_lora``)."""
    extra = dict(extra or {})
    fmt = str(checkpoint_format or "bf16")
    quant = str(extra.get("quantization") or "")
    weights = f"{quant}-online" if quant and quant != fmt else fmt
    kv = str(extra.get("kv_cache_dtype") or "auto")
    spec = extra.get("speculative_config") or {}
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = {}
    return {
        "weights": weights,
        "kv": "fp8" if kv.startswith("fp8") else kv,
        "state": str(extra.get("mamba_ssm_cache_dtype") or "auto"),
        "spec": f"{spec['method']}:{int(spec.get('num_speculative_tokens') or 1)}" if spec.get("method") else "none",
        "lora": "1" if extra.get("enable_lora") else "0",
    }


#: the defaults of a profile's kernel class; its weights format has none (a profile that does not state the format it
#: was measured in matches no lookup)
BASE_CLASS = {"kv": "auto", "state": "auto", "spec": "none", "lora": "0"}


def _key(sku: str, engine: str, model: str, cls: Mapping[str, str]) -> Tuple[str, str, str, Tuple[Tuple[str, str], ...]]:
    return (sku, engine, model, tuple(sorted({**BASE_CLASS, **dict(cls)}.items())))


class ProfileStore:
    def __init__(self, profiles: Iterable[StepProfile] = ()):
        self._index: Dict[Tuple[Any, ...], StepProfile] = {}
        for p in profiles:
            self.add(p)

    def add(self, profile: StepProfile) -> None:
        self._index[_key(profile.sku, profile.engine, profile.model, profile.settings)] = profile

    def __len__(self) -> int:
        return len(self._index)

    def find(self, sku: str, engine: str, model: str, cls: Mapping[str, str]) -> Optional[StepProfile]:
        return self._index.get(_key(sku, engine, model, cls))

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "ProfileStore":
        root = Path(root or DATA_DIR)
        store = cls()
        if root.exists():
            for path in sorted(root.rglob("*.json")):
                data = json.loads(path.read_text())
                if data.get("schema") == "bud-perf-profile/1":
                    store.add(StepProfile.from_dict(data))
        return store


_DEFAULT: Optional[ProfileStore] = None


def default_profiles() -> ProfileStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ProfileStore.load()
    return _DEFAULT


def register_profile(profile: StepProfile) -> None:
    """Add a profile measured on this cluster (FR-PROF-2): it replaces the package's for the same key, since it ran the
    cluster's own image on its own GPUs (FRD-027 §20.8 open question 1)."""
    default_profiles().add(profile)
