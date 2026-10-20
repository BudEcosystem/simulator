"""FRD-027 Revision 2, L5: output grounding (§20.2; FR-RES-1…3).

The simulation (L2–L4) predicts each quantity from measured steps, with no fitted constant. Some things it does not
see, for example:
- a serving path the offline steps miss (gpt-oss serves 15–25% slower than its steps at c ≥ 16);
- the spread between devices;
- the API path under load.

These show up as a ratio r = measured / predicted that repeats within a slice. L5 learns r per slice and widens the
band by whatever is left:
- **Slices:** (SKU, engine build, family, quantity), then the load bucket within it (concurrency band × workload
  class). The correction is applied to the simulation's output only (FR-RES-1).
- **Partial pooling:** a slice's log-correction is shrunk toward its parent by a normal-normal posterior. A bucket's
  parent is its family's correction, and a family's is 1 (no correction). A slice with two noisy points stays near its
  parent; a slice with fifty keeps its own. The prior spreads are fixed, not estimated across families, so one family's
  runs never move another family's correction (T-RES-01: the §19.5 coupling cannot recur).
- **Bands:** the split-conformal 90% quantile of the leave-one-configuration-out residuals after correction, floored
  by the replicate noise (FR-RES-2).
- **Do no harm:** a slice keeps its correction only when that lowers its out-of-sample error. Where the residual
  varies from configuration to configuration rather than sharing a bias (gpt-oss's TTFT), a correction learned
  from the other configurations moves a new one further off, and the slice is left uncorrected with its uncorrected
  band.
- **The cap:** a slice whose correction would pass ln 1.3 is a model gap. Its correction is not applied and its
  predictions are reported ungrounded (FR-RES-3).
- **Builds:** corrections are keyed per engine build. A new build of the same minor starts from that minor's
  corrections with a wider band; it never starts from another minor's (T-RES-04).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

QUANTITIES = ("tpot", "throughput", "ttft")
#: a slice needing more than this is a model gap (FR-RES-3)
CAP = math.log(1.3)
#: prior spread of a family's ln r around 1 (NFR-6 expects L1–L4 within ~15%)
FAMILY_PRIOR = 0.15
#: prior spread of a load bucket's ln r around its family's
BUCKET_PRIOR = 0.10
#: the replicate spread no band is narrower than, per quantity (§16's noise amendment), until replicates set it
NOISE_FLOOR = {"tpot": 0.03, "throughput": 0.03, "ttft": 0.08}
#: a new build of a known minor carries the minor's corrections with this much wider band
MINOR_WIDEN = 1.5
SCHEMA = "bud-perf-corrections/1"
#: telemetry points a load bucket needs before production supersedes the corpus there (FRD-027 §20.8 open question 4)
AGE_OUT = 20
RESIDUALS_FILE = "residuals.jsonl"
FAMILY_LEVEL = "*"


def load_bucket(concurrency: int, workload_class: str) -> str:
    """The load slice a point or a plan falls in: its concurrency band and whether its text is real."""
    c = int(concurrency)
    band = "c1-4" if c <= 4 else ("c5-32" if c <= 32 else "c33+")
    return f"{band}/{'synthetic' if workload_class == 'synthetic' else 'real'}"


def minor_of(engine: str) -> str:
    """``vllm@0.30.0`` → ``vllm@0.30``; ``bud-vllm@0.9.0-481e481be`` → ``bud-vllm@0.9``."""
    name, _, version = str(engine).partition("@")
    parts = [p for p in re.split(r"[.\-+]", version) if p]
    return f"{name}@{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(engine)


@dataclass(frozen=True)
class Residual:
    """One replayed point: the simulation's value and the deployment's, for one quantity."""

    sku: str
    engine: str
    family: str
    quantity: str
    bucket: str
    predicted: float
    measured: float
    config: str = ""
    queueing: bool = False

    @property
    def ln_r(self) -> float:
        return math.log(self.measured / self.predicted)


@dataclass(frozen=True)
class Correction:
    """What L5 does to one quantity of one prediction."""

    factor: float  # the simulation's value is multiplied by this
    band: float  # the band after correction: the p90 of |ln(measured / corrected)|
    n: int  # the slice's own points
    level: str  # bucket | family | build (an unseen family: no factor, the build's band) | minor:<build>
    gap: bool = False  # the slice needs more than the cap: not applied, and the prediction is ungrounded
    needed: float = 1.0  # the correction the slice asks for

    @property
    def rel(self) -> float:
        """The band as the relative half-width an ``Estimate`` carries (its upper edge)."""
        return math.exp(self.band) - 1.0


_SliceKey = Tuple[str, str, str, str, str]  # sku, engine, family, quantity, bucket (FAMILY_LEVEL for the family)
_BuildKey = Tuple[str, str, str]  # sku, engine, quantity


@dataclass
class Corrections:
    entries: Dict[_SliceKey, Correction] = field(default_factory=dict)
    #: an unseen family on a known build: no factor, and the band L1–L4 alone earned across the build's families
    builds: Dict[_BuildKey, Correction] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def merge(self, other: "Corrections") -> None:
        self.entries.update(other.entries)
        self.builds.update(other.builds)

    def _exact(self, sku: str, engine: str, family: str, quantity: str, bucket: str) -> Optional[Correction]:
        hit = self.entries.get((sku, engine, family, quantity, bucket))
        if hit is not None:
            return hit
        hit = self.entries.get((sku, engine, family, quantity, FAMILY_LEVEL))
        if hit is not None:
            return hit
        return self.builds.get((sku, engine, quantity))

    def lookup(self, sku: str, engine: str, family: str, quantity: str, bucket: str) -> Optional[Correction]:
        """The correction for one quantity of a prediction: the bucket's, else the family's, else the build's band.

        A build with no corrections borrows the newest build of its minor, with a wider band; another minor's never.
        """
        hit = self._exact(sku, engine, family, quantity, bucket)
        if hit is not None:
            return hit
        minor = minor_of(engine)
        known = sorted({e for (s, e, *_rest) in self.entries if s == sku and e != engine and minor_of(e) == minor})
        for other in reversed(known):
            hit = self._exact(sku, other, family, quantity, bucket)
            if hit is not None:
                return Correction(
                    factor=hit.factor,
                    band=hit.band * MINOR_WIDEN,
                    n=hit.n,
                    level=f"minor:{other}",
                    gap=hit.gap,
                    needed=hit.needed,
                )
        return None

    # ------------------------------------------------------------------ storage

    def to_dict(self) -> Dict[str, object]:
        def row(key: Tuple[str, ...], c: Correction, names: Sequence[str]) -> Dict[str, object]:
            out: Dict[str, object] = dict(zip(names, key))
            out.update(
                factor=round(c.factor, 5), band=round(c.band, 5), n=c.n, level=c.level, gap=c.gap,
                needed=round(c.needed, 5),
            )
            return out

        return {
            "schema": SCHEMA,
            "entries": [
                row(k, c, ("sku", "engine", "family", "quantity", "bucket")) for k, c in sorted(self.entries.items())
            ],
            "builds": [row(k, c, ("sku", "engine", "quantity")) for k, c in sorted(self.builds.items())],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Corrections":
        def corr(d: Mapping[str, object]) -> Correction:
            return Correction(
                factor=float(d["factor"]), band=float(d["band"]), n=int(d["n"]), level=str(d["level"]),
                gap=bool(d.get("gap")), needed=float(d.get("needed", 1.0)),
            )

        out = cls()
        for d in data.get("entries") or []:  # type: ignore[union-attr]
            out.entries[(d["sku"], d["engine"], d["family"], d["quantity"], d["bucket"])] = corr(d)
        for d in data.get("builds") or []:  # type: ignore[union-attr]
            out.builds[(d["sku"], d["engine"], d["quantity"])] = corr(d)
        return out

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "Corrections":
        from .store import DATA_DIR

        out = cls()
        root = Path(root or DATA_DIR)
        if root.exists():
            for path in sorted(root.rglob("*.json")):
                data = json.loads(path.read_text())
                if isinstance(data, dict) and data.get("schema") == SCHEMA:
                    out.merge(cls.from_dict(data))
        return out


# ---------------------------------------------------------------------- fitting


def _posterior(values: Sequence[float], prior_mean: float, prior_sd: float, spread2: float) -> float:
    """The normal-normal posterior mean of a slice's ln r: its mean, shrunk toward the parent by its count and spread."""
    if not values:
        return prior_mean
    w = prior_sd**2 / (prior_sd**2 + spread2 / len(values))
    return prior_mean + w * (statistics.fmean(values) - prior_mean)


def _family_fit(points: Sequence[Residual], floor: float) -> Tuple[float, Dict[str, float]]:
    """A family slice's correction and its buckets', in ln r."""
    values = [p.ln_r for p in points]
    total2 = max(statistics.pvariance(values) if len(values) > 1 else 0.0, floor**2)
    family = _posterior(values, 0.0, FAMILY_PRIOR, total2)
    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for p in points:
        by_bucket[p.bucket].append(p.ln_r)
    # the within-bucket spread, pooled over the family's buckets: a two-point bucket's own variance means nothing
    dof = len(values) - len(by_bucket)
    within = sum((v - statistics.fmean(vs)) ** 2 for vs in by_bucket.values() for v in vs)
    within2 = max(within / dof if dof > 0 else total2, floor**2)
    buckets = {b: _posterior(vs, family, BUCKET_PRIOR, within2) for b, vs in by_bucket.items()}
    return family, buckets


def _conformal(values: Sequence[float]) -> float:
    from ..grounding.fit import conformal_p90

    return conformal_p90(values)


def out_of_sample(points: Sequence[Residual], floor: float) -> List[Tuple[Residual, float]]:
    """Each point's residual after a correction fitted without its configuration (leave one configuration out):
    ln(measured / corrected)."""
    out: List[Tuple[Residual, float]] = []
    configs = sorted({p.config for p in points})
    for cfg in configs:
        train = [p for p in points if p.config != cfg]
        family, buckets = _family_fit(train, floor) if train else (0.0, {})
        for p in points:
            if p.config == cfg:
                out.append((p, p.ln_r - buckets.get(p.bucket, family)))
    return out


@dataclass
class _SliceFit:
    """One family slice's fit: its corrections, which of them help out of sample, and the residuals each leaves."""

    family: float
    buckets: Dict[str, float]
    applied: Dict[str, bool]  # bucket (or FAMILY_LEVEL) -> its correction lowers the out-of-sample error
    residuals: List[Tuple[Residual, float]]  # every point's residual as served: corrected (out of sample) or not


def _slices(residuals: Iterable[Residual]) -> Dict[Tuple[str, str, str, str], List[Residual]]:
    slices: Dict[Tuple[str, str, str, str], List[Residual]] = defaultdict(list)
    for r in residuals:
        if r.predicted > 0 and r.measured > 0:
            slices[(r.sku, r.engine, r.family, r.quantity)].append(r)
    return slices


def _fit_slice(points: Sequence[Residual], floor: float) -> _SliceFit:
    family, buckets = _family_fit(points, floor)
    oos = out_of_sample(points, floor)
    applied: Dict[str, bool] = {}
    served: List[Tuple[Residual, float]] = []

    def helps(group: Sequence[Tuple[Residual, float]]) -> bool:
        return statistics.fmean(abs(e) for _, e in group) < statistics.fmean(abs(p.ln_r) for p, _ in group)

    applied[FAMILY_LEVEL] = helps(oos)
    for bucket in buckets:
        group = [(p, e) for p, e in oos if p.bucket == bucket]
        applied[bucket] = helps(group)
        served += group if applied[bucket] else [(p, p.ln_r) for p, _ in group]
    return _SliceFit(family, buckets, applied, served)


def fit(residuals: Iterable[Residual], *, noise: Optional[Mapping[str, float]] = None) -> Corrections:
    """L5 corrections and bands from replayed points."""
    floors = {**NOISE_FLOOR, **dict(noise or {})}
    out = Corrections()
    build_raw: Dict[_BuildKey, List[float]] = defaultdict(list)
    for (sku, engine, family, quantity), points in sorted(_slices(residuals).items()):
        floor = floors.get(quantity, 0.03)
        build_raw[(sku, engine, quantity)].extend(abs(p.ln_r) for p in points)
        sf = _fit_slice(points, floor)
        fam_ln = sf.family if sf.applied[FAMILY_LEVEL] else 0.0
        fam_band = max(_conformal([abs(e) for _, e in sf.residuals]), floor)
        out.entries[(sku, engine, family, quantity, FAMILY_LEVEL)] = Correction(
            factor=math.exp(fam_ln), band=fam_band, n=len(points), level="family", gap=abs(fam_ln) > CAP,
            needed=math.exp(sf.family),
        )
        for bucket, ln_c in sorted(sf.buckets.items()):
            mine = [abs(e) for p, e in sf.residuals if p.bucket == bucket]
            # a bucket too small for its own quantile keeps the family's band when that is wider
            band = _conformal(mine) if len(mine) >= 9 else max(_conformal(mine), fam_band)
            used = ln_c if sf.applied[bucket] else 0.0
            out.entries[(sku, engine, family, quantity, bucket)] = Correction(
                factor=math.exp(used), band=max(band, floor), n=len(mine),
                level="bucket" if sf.applied[bucket] else "bucket-uncorrected", gap=abs(used) > CAP,
                needed=math.exp(ln_c),
            )
    for key, raw in build_raw.items():
        out.builds[key] = Correction(
            factor=1.0, band=max(_conformal(raw), floors.get(key[2], 0.03)), n=len(raw), level="build"
        )
    return out


def cross_validated_errors(residuals: Iterable[Residual], *, noise: Optional[Mapping[str, float]] = None):
    """Per (family, quantity, workload class, queueing): each point's |served / measured - 1|, where served is its
    prediction after a correction fitted without its own configuration (NFR-7's grade), and before any (NFR-6's).

    A point whose slice keeps no correction (do no harm) or is a model gap (past the cap, so served uncorrected and
    ungrounded) is graded uncorrected; ``gaps`` counts the latter.
    """
    floors = {**NOISE_FLOOR, **dict(noise or {})}
    after: Dict[Tuple[str, str, str, bool], List[float]] = defaultdict(list)
    before: Dict[Tuple[str, str, str, bool], List[float]] = defaultdict(list)
    gaps: Dict[Tuple[str, str, str, bool], int] = defaultdict(int)
    for (_sku, _engine, family, quantity), points in _slices(residuals).items():
        sf = _fit_slice(points, floors.get(quantity, 0.03))
        for p, e in sf.residuals:
            key = (family, quantity, p.bucket.split("/")[1], p.queueing)
            gap = sf.applied.get(p.bucket) and abs(sf.buckets.get(p.bucket, 0.0)) > CAP
            if gap:
                gaps[key] += 1
                e = p.ln_r
            after[key].append(abs(math.exp(-e) - 1.0))  # served / measured = exp(-e)
            before[key].append(abs(p.predicted / p.measured - 1.0))
    return after, before, gaps


def refit(
    corpus: Iterable[Residual], telemetry: Iterable[Residual], *, noise: Optional[Mapping[str, float]] = None
) -> Corrections:
    """L5 refitted with production points (FR-RES-4): telemetry joins the corpus, and a load bucket with ``AGE_OUT``
    or more telemetry points is fitted from telemetry alone. A refit changes later plans only, never a running
    deployment."""
    telemetry = list(telemetry)
    counts: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    for r in telemetry:
        counts[(r.sku, r.engine, r.family, r.quantity, r.bucket)] += 1
    aged = {k for k, n in counts.items() if n >= AGE_OUT}
    kept = [r for r in corpus if (r.sku, r.engine, r.family, r.quantity, r.bucket) not in aged]
    return fit(kept + telemetry, noise=noise)


def corpus_residuals(root: Optional[Path] = None) -> List[Residual]:
    """The replayed corpus points the package's corrections were fitted from (``perf.replay --residuals``)."""
    from .store import DATA_DIR

    out: List[Residual] = []
    for path in sorted(Path(root or DATA_DIR).rglob(RESIDUALS_FILE)):
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(Residual(**json.loads(line)))
    return out


_DEFAULT: Optional[Corrections] = None


def default_corrections() -> Corrections:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Corrections.load()
    return _DEFAULT


def set_default_corrections(corrections: Corrections) -> None:
    """Replace the corrections plans use (a telemetry refit, FR-RES-4)."""
    global _DEFAULT
    _DEFAULT = corrections
