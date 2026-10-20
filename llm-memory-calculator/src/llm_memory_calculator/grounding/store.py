"""Layer 3 of the FRD-027 predictor: the grounding store (§9).

Measured corrections keyed by ``(hardware, engine, model)`` plus a per-kind detail key (phase, batch bucket,
weight format, speculative method and k, sampling class...). Each of the three dimensions has its own
fallback ladder, applied independently (FR-GND-1):

- hardware: ``sku:<id>`` -> ``arch:<architecture>`` -> ``prior``
- engine:   ``record:<id>`` -> ``minor:<engine@x.y:runner>`` -> ``family:<engine>`` -> ``prior``
- model:    ``family:<lineage>`` -> ``class:<architecture class>`` -> ``pooled`` -> ``prior``

An entry is stored at the level it was fitted at; the fitter writes the aggregate levels (architecture,
minor, class, pooled) next to the measured ones, with their residuals inflated (FR-GND-7: vGPU data grounds
the Hopper level, never the bare-metal SKU). A lookup takes the most specific entry on every dimension and
reports the level it used per dimension; the band is the entry's residual times the inflation of every level
it fell to. Boot outcomes (FR-GND-6) widen the memory band for their key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_VERSION = "2026-10-19.3"

HARDWARE_LEVELS = ("sku", "arch", "prior")
ENGINE_LEVELS = ("record", "minor", "family", "prior")
MODEL_LEVELS = ("family", "class", "pooled", "prior")

#: Residual inflation per level fallen (multiplies the entry's own residual).
INFLATION = {
    "hardware": {"sku": 1.0, "arch": 1.5, "prior": 3.0},
    "engine": {"record": 1.0, "minor": 1.3, "family": 2.0, "prior": 3.0},
    "model": {"family": 1.0, "class": 1.4, "pooled": 1.8, "prior": 3.0},
}

#: Batch buckets for decode (FRD-027 §9.1).
BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 10**9))


def bucket_of(batch: float) -> str:
    b = max(int(round(batch)), 1)
    for lo, hi in BUCKETS:
        if lo <= b <= hi:
            return f"{lo}" if lo == hi else (f">{lo - 1}" if hi >= 10**9 else f"{lo}-{hi}")
    return ">256"


BUCKET_NAMES = tuple(bucket_of(lo) for lo, _ in BUCKETS)


@dataclass(frozen=True)
class Key:
    hardware: str  # "sku:h100xm-80c-vgpu" | "arch:hopper" | "prior"
    engine: str  # "record:vllm@0.30.0" | "minor:vllm@0.30:v2" | "family:vllm" | "prior"
    model: str  # "family:Qwen3-8B" | "class:dense-GQA" | "pooled" | "prior"

    def levels(self) -> Dict[str, str]:
        return {
            "hardware": self.hardware.split(":", 1)[0],
            "engine": self.engine.split(":", 1)[0],
            "model": self.model.split(":", 1)[0],
        }


@dataclass
class Entry:
    kind: str  # memory.overhead | decode.kappa | prefill.kappa | decode.tau | precision.ratio | spec.acceptance | boot.time ...
    key: Key
    detail: Dict[str, Any]
    value: float
    residual: float  # relative (or absolute bytes for memory kinds), p90
    samples: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = asdict(self.key)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entry":
        return cls(
            kind=str(data["kind"]),
            key=Key(**data["key"]),
            detail=dict(data.get("detail") or {}),
            value=float(data["value"]),
            residual=float(data.get("residual") or 0.0),
            samples=int(data.get("samples") or 0),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass
class BootOutcome:
    key: Key
    settings: Dict[str, Any]
    predicted_margin_bytes: float
    outcome: str  # booted | oom | refused | crashloop | timeout
    message: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = asdict(self.key)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BootOutcome":
        return cls(
            key=Key(**data["key"]),
            settings=dict(data.get("settings") or {}),
            predicted_margin_bytes=float(data.get("predicted_margin_bytes") or 0.0),
            outcome=str(data.get("outcome") or "booted"),
            message=str(data.get("message") or ""),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class Found:
    value: float
    residual: float  # inflated by the levels fallen
    levels: Dict[str, str]
    samples: int = 0

    @property
    def level_names(self) -> Tuple[str, str, str]:
        return self.levels["hardware"], self.levels["engine"], self.levels["model"]


def hardware_ladder(device: Any) -> List[str]:
    out = []
    sku = getattr(device, "sku", None)
    if sku and not getattr(device, "derived", False):
        out.append(f"sku:{sku}")
    arch = getattr(device, "architecture", None)
    if arch and arch != "unknown":
        out.append(f"arch:{arch}")
    out.append("prior")
    return out


def engine_ladder(record: Any) -> List[str]:
    out = []
    if getattr(record, "id", None):
        out.append(f"record:{record.id}")
    if getattr(record, "minor_key", None):
        out.append(f"minor:{record.minor_key}")
    if getattr(record, "engine", None):
        out.append(f"family:{record.engine}")
    out.append("prior")
    return out


def model_ladder(model: Any) -> List[str]:
    out = []
    if getattr(model, "recognised", True) and getattr(model, "family", None):
        out.append(f"family:{model.family}")
    if getattr(model, "recognised", True) and getattr(model, "arch_class", None):
        out.append(f"class:{model.arch_class}")
    if getattr(model, "recognised", True):
        out.append("pooled")
    out.append("prior")
    return out


class GroundingStore:
    """A loaded, versioned store. Pure data plus lookups; nothing here touches the network or a clock."""

    def __init__(
        self,
        entries: Iterable[Entry] = (),
        outcomes: Iterable[BootOutcome] = (),
        *,
        version: str = "empty",
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.version = version
        self.metadata = dict(metadata or {})
        self.entries: List[Entry] = list(entries)
        self.outcomes: List[BootOutcome] = list(outcomes)
        self._index: Dict[Tuple[str, str, str, str], List[Entry]] = {}
        # exact detail -> entry: a lookup whose detail has no wildcard matches only an entry with the same detail
        self._exact: Dict[Tuple[str, str, str, str, Tuple[Tuple[str, Any], ...]], Entry] = {}
        self._families = {
            e.key.model.split(":", 1)[1] for e in self.entries if e.key.model.startswith("family:")
        }
        for entry in self.entries:
            self._index.setdefault(
                (entry.kind, entry.key.hardware, entry.key.engine, entry.key.model), []
            ).append(entry)
            self._exact.setdefault(
                (
                    entry.kind,
                    entry.key.hardware,
                    entry.key.engine,
                    entry.key.model,
                    tuple(sorted(entry.detail.items())),
                ),
                entry,
            )

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, version: Optional[str] = None, root: Optional[Path] = None) -> "GroundingStore":
        root = Path(root or DATA_DIR)
        version = version or _latest(root)
        if version is None:
            return cls(version="empty")
        folder = root / version
        entries: List[Entry] = []
        for path in sorted(folder.glob("entries*.json")):
            entries.extend(Entry.from_dict(item) for item in json.loads(path.read_text()))
        outcomes: List[BootOutcome] = []
        outcome_path = folder / "boot_outcomes.json"
        if outcome_path.exists():
            outcomes = [
                BootOutcome.from_dict(item) for item in json.loads(outcome_path.read_text())
            ]
        meta_path = folder / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(entries, outcomes, version=version, metadata=metadata)

    def save(self, root: Optional[Path] = None) -> Path:
        folder = Path(root or DATA_DIR) / self.version
        folder.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self.entries,
            key=lambda e: (
                e.kind,
                e.key.hardware,
                e.key.engine,
                e.key.model,
                json.dumps(e.detail, sort_keys=True),
            ),
        )
        (folder / "entries.json").write_text(
            json.dumps([e.to_dict() for e in ordered], indent=1, sort_keys=True) + "\n"
        )
        (folder / "boot_outcomes.json").write_text(
            json.dumps([o.to_dict() for o in self.outcomes], indent=1, sort_keys=True) + "\n"
        )
        (folder / "metadata.json").write_text(
            json.dumps(self.metadata, indent=1, sort_keys=True) + "\n"
        )
        return folder

    def digest(self) -> str:
        blob = json.dumps(
            [e.to_dict() for e in self.entries] + [o.to_dict() for o in self.outcomes],
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ lookups

    def find(
        self,
        kind: str,
        detail: Mapping[str, Any],
        device: Any,
        record: Any,
        model: Any,
        *,
        prior: Optional[float] = None,
        prior_residual: float = 0.5,
    ) -> Found:
        """Most specific entry for ``kind`` + ``detail`` on each ladder; the prior when nothing matches."""
        best: Optional[Tuple[int, Entry, Dict[str, str]]] = None
        hw_l, en_l, mo_l = hardware_ladder(device), engine_ladder(record), model_ladder(model)
        family = getattr(model, "family", None)
        stranger = family is None or family not in self._families  # a hold-out or unknown family
        exact = None
        if all(v not in (None, "*") for v in detail.values()):
            exact = tuple(sorted(detail.items()))
        for hi, hw in enumerate(hw_l):
            for ei, en in enumerate(en_l):
                for mi, mo in enumerate(mo_l):
                    if exact is not None:
                        hit = self._exact.get((kind, hw, en, mo, exact))
                        candidates = [hit] if hit is not None else None
                    else:
                        candidates = self._index.get((kind, hw, en, mo))
                    if not candidates:
                        continue
                    for entry in candidates:
                        if not _detail_matches(entry.detail, detail):
                            continue
                        if mo == "pooled" and stranger and len(entry.provenance.get("families") or ()) == 1:
                            # one family's aggregate (bud-vllm 0.10.0's 27B) has no between-family spread: it made the
                            # held-out 8B look grounded there (NFR-5)
                            continue
                        # model specificity first, then engine, then hardware (a measured family on an
                        # unmeasured SKU keeps its family entry at the architecture level, FRD-027 §9.1)
                        score = mi * 100 + ei * 10 + hi
                        if stranger and mo.startswith("class:") and len(entry.provenance.get("families") or ()) == 1:
                            # nor does one family's class entry hold the class transfer: a family the store lacks reads
                            # a class of two families first, at any engine or hardware level (the fork's MoE class is
                            # gpt-oss alone, upstream's has Qwen1.5-MoE too), then this one, then the pooled level
                            score += 50
                        if best is None or score < best[0]:
                            best = (score, entry, {"hardware": hw, "engine": en, "model": mo})
        if best is None:
            if prior is None:
                raise KeyError(kind)
            levels = {"hardware": "prior", "engine": "prior", "model": "prior"}
            return Found(
                value=float(prior), residual=float(prior_residual), levels=levels, samples=0
            )
        _, entry, keys = best
        levels = {dim: keys[dim].split(":", 1)[0] for dim in keys}
        inflation = 1.0
        for dim, level in levels.items():
            inflation *= INFLATION[dim][level]
        return Found(
            value=entry.value,
            residual=entry.residual * inflation,
            levels=levels,
            samples=entry.samples,
        )

    def outcomes_for(self, device: Any, record: Any, model: Any) -> List[BootOutcome]:
        keys = {
            (hw, en, mo)
            for hw in hardware_ladder(device)[:1]
            for en in engine_ladder(record)[:1]
            for mo in model_ladder(model)[:1]
        }
        return [o for o in self.outcomes if (o.key.hardware, o.key.engine, o.key.model) in keys]

    # ------------------------------------------------------------------ FR-GND-6

    def record_boot_outcome(
        self,
        device: Any,
        record: Any,
        model: Any,
        settings: Mapping[str, Any],
        predicted_margin_bytes: float,
        outcome: str,
        message: str = "",
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> BootOutcome:
        key = Key(hardware_ladder(device)[0], engine_ladder(record)[0], model_ladder(model)[0])
        item = BootOutcome(
            key=key,
            settings=dict(settings),
            predicted_margin_bytes=float(predicted_margin_bytes),
            outcome=str(outcome),
            message=str(message)[:400],
            provenance=dict(provenance or {}),
        )
        self.outcomes.append(item)
        return item

    def memory_band_bytes(self, device: Any, record: Any, model: Any) -> float:
        """Extra memory headroom the next plan for this key reserves: the largest margin the predictor gave a
        configuration that then failed to boot (FR-GND-6). Several failures do not add up."""
        worst = 0.0
        for o in self.outcomes_for(device, record, model):
            if o.outcome in ("oom", "refused", "crashloop") and o.predicted_margin_bytes > 0:
                worst = max(worst, o.predicted_margin_bytes)
        return worst


def _detail_matches(have: Mapping[str, Any], want: Mapping[str, Any]) -> bool:
    for k, v in want.items():
        if k in have and have[k] != v:
            return False
        if k not in have and v not in (None, "*"):
            return False
    for k in have:
        if k not in want:
            return False
    return True


def _latest(root: Path) -> Optional[str]:
    if not root.exists():
        return None
    versions = sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "entries.json").exists()
    )
    return versions[-1] if versions else None


_DEFAULT: Optional[GroundingStore] = None


def default_store() -> GroundingStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GroundingStore.load()
    return _DEFAULT


def set_default_store(store: Optional[GroundingStore]) -> None:
    global _DEFAULT
    _DEFAULT = store


class OverrideStore(GroundingStore):
    """A store with some kinds pinned to a value (the fitter's inversions; what-if analysis)."""

    def __init__(self, base: GroundingStore, overrides: Mapping[str, float]):
        super().__init__(version=base.version, metadata=base.metadata)
        self.base = base
        self.overrides = dict(overrides)
        self.entries = base.entries
        self.outcomes = base.outcomes
        self._index = base._index
        self._exact = base._exact
        self._families = base._families

    def find(self, kind, detail, device, record, model, *, prior=None, prior_residual=0.5):  # type: ignore[override]
        if kind in self.overrides:
            found = self.base.find(
                kind,
                detail,
                device,
                record,
                model,
                prior=prior if prior is not None else 1.0,
                prior_residual=prior_residual,
            )
            return Found(
                value=float(self.overrides[kind]),
                residual=found.residual,
                levels=found.levels,
                samples=found.samples,
            )
        return self.base.find(
            kind, detail, device, record, model, prior=prior, prior_residual=prior_residual
        )
