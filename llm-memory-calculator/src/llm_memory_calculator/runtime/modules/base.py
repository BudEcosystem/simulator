"""Module base class and registry (FRD-025 §6.1)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple, Type

from ..context import Fragment, PlanContext

REGISTRY: Dict[Tuple[str, str], Type["Module"]] = {}


class Module:
    name: str = ""
    version: str = "v1"
    stage: str = "capacity"
    after: Tuple[str, ...] = ()
    overrides: Tuple[str, ...] = ()

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:  # pragma: no cover - abstract
        raise NotImplementedError

    def frag(self) -> Fragment:
        return Fragment(overrides=self.overrides)

    def __repr__(self) -> str:
        return f"<{self.name}@{self.version}>"


def register(cls: Type[Module]) -> Type[Module]:
    key = (cls.name, cls.version)
    if key in REGISTRY:
        raise ValueError(f"module {key} registered twice")
    REGISTRY[key] = cls
    return cls


#: The module set per version selector. "v1" is today's behaviour ported (FRD-025 M2); FRD-027's modules are "v2".
V1_DEFAULT = "v1"


def build(versions: Optional[Mapping[str, str]] = None) -> List[Module]:
    """Instantiate one module per name at the version ``versions`` selects (default v1; "*" sets every module's
    default). A module that exists only in v2 runs only when selected; ``off`` disables a module."""
    versions = dict(versions or {})
    default = versions.get("*", V1_DEFAULT)
    names = sorted({name for name, _ in REGISTRY})
    out = []
    for name in names:
        want = versions.get(name, default)
        if want == "off":
            continue
        cls = REGISTRY.get((name, want))
        if cls is None and want != V1_DEFAULT and name not in versions:
            cls = REGISTRY.get((name, V1_DEFAULT))  # "*": v2 falls back to v1 where no v2 exists
        if cls is None and want == V1_DEFAULT:
            continue  # a v2-only module is off unless selected
        if cls is None:
            raise ValueError(f"module {name!r} has no version {want!r}")
        out.append(cls())
    return out
