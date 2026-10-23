"""Compose modules into a runtime spec (FRD-025 §6.2; FRD-027 §10.0).

- Two stages: ``numerics`` modules run once per plan before budsim's search and produce the model profile;
  ``capacity`` modules run per placement. Within a stage the order is topological in ``after``, ties broken by
  name; every module runs exactly once and reads earlier outputs from the context.
- No silent overrides: a module setting a key another module set must declare ``overrides``; otherwise the plan
  fails here, at simulation time, never at deploy time.
- Modules emit engine-neutral settings; the composer resolves them to arguments and environment through the
  record's ``settings`` mapping. A direct argument may only cover a key no setting resolves to.
- The spec is deterministic: the same inputs give the byte-identical spec (FRD-027 NFR-3).
"""

from __future__ import annotations

import copy

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..engine.record import EngineRecord
from .context import Draft, Fragment, PlanContext, Value
from .spec import CHECK_KINDS, RuntimeSpec, SPEC_VERSION

META_SETTINGS = {
    "weights.online",
    "weights.variant",
}  # decisions without an engine argument of their own


class CompositionError(ValueError):
    """A plan that cannot be composed: an undeclared override, an unsupported setting, a cycle."""


@dataclass
class ModuleRun:
    module: str
    stage: str
    applied: bool
    reason: str = ""


@dataclass
class Composition:
    settings: Dict[str, Value] = field(default_factory=dict)
    args: Dict[str, Value] = field(default_factory=dict)
    env: Dict[str, Value] = field(default_factory=dict)
    resources: Dict[str, Dict[str, Value]] = field(default_factory=dict)
    volumes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mounts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    probes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    containers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pod: Dict[str, Value] = field(default_factory=dict)
    engine: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    mutable: Dict[str, Any] = field(default_factory=dict)
    predicted: Optional[Dict[str, Any]] = None
    decisions: List[str] = field(default_factory=list)
    log: List[ModuleRun] = field(default_factory=list)
    stage1: Dict[str, Any] = field(default_factory=dict)

    def draft(self) -> Draft:
        return Draft(
            settings=dict(self.settings),
            args=dict(self.args),
            env=dict(self.env),
            pod=dict(self.pod),
            alternatives=list(self.alternatives),
            checks=list(self.checks),
            decisions=list(self.decisions),
            stage1=dict(self.stage1),
        )


# ====================================================================== ordering


def order(modules: Sequence[Any]) -> List[Any]:
    """Topological order of ``after`` edges among the given modules, ties broken by name (deterministic)."""
    by_name = {m.name: m for m in modules}
    if len(by_name) != len(modules):
        raise CompositionError("two modules share a name")
    remaining = {m.name: {a for a in m.after if a in by_name} for m in modules}
    out: List[Any] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise CompositionError(f"module dependency cycle among {sorted(remaining)}")
        name = ready[0]
        out.append(by_name[name])
        del remaining[name]
        for deps in remaining.values():
            deps.discard(name)
    return out


# ====================================================================== merging


def _merge_keyed(
    target: Dict[str, Value], incoming: Mapping[str, Value], module: Any, what: str
) -> None:
    for key, value in incoming.items():
        have = target.get(key)
        if (
            have is not None
            and have.module != module.name
            and have.module not in getattr(module, "overrides", ())
        ):
            if have.value == value.value:
                continue  # agreement is not a clash
            raise CompositionError(
                f"{module.name} sets {what} {key!r}={value.value!r} already set by {have.module} "
                f"({have.value!r}) without declaring overrides=({have.module!r},)"
            )
        target[key] = value


def merge(comp: Composition, frag: Fragment, module: Any) -> None:
    _merge_keyed(comp.settings, frag.settings, module, "setting")
    _merge_keyed(comp.args, frag.args, module, "argument")
    _merge_keyed(comp.env, frag.env, module, "env")
    _merge_keyed(comp.pod, frag.pod, module, "pod field")
    for kind, items in frag.resources.items():
        _merge_keyed(comp.resources.setdefault(kind, {}), items, module, f"resource {kind}")
    for vol in frag.volumes:
        comp.volumes[vol["name"]] = vol
    for mount in frag.mounts:
        comp.mounts[mount["volume"] + ":" + mount["path"]] = mount
    for port in frag.ports:
        comp.ports[port["name"]] = port
    comp.probes.update(frag.probes)
    for c in frag.containers:
        comp.containers[c["name"]] = c
    comp.engine.update(frag.engine)
    comp.checks.extend(frag.checks)
    comp.alternatives.extend(frag.alternatives)
    comp.mutable.update(frag.mutable)
    if frag.predicted is not None:
        comp.predicted = frag.predicted
    comp.decisions.extend(
        f"{module.name}: {d}" if not d.startswith(module.name) else d for d in frag.decisions
    )


def run_stage(
    stage: str, ctx: PlanContext, modules: Iterable[Any], comp: Optional[Composition] = None
) -> Composition:
    """Run every module of ``stage`` once, in order, merging fragments; returns the composition."""
    comp = comp or Composition()
    for module in order([m for m in modules if m.stage == stage]):
        ctx.draft = comp.draft()
        ok, reason = module.applies(ctx)
        if not ok:
            comp.log.append(ModuleRun(module.name, stage, False, reason))
            if reason:
                comp.decisions.append(f"{module.name}: not applied ({reason})")
            continue
        frag = module.plan(ctx)
        merge(comp, frag, module)
        comp.log.append(ModuleRun(module.name, stage, True))
        output = getattr(module, "stage1_output", None)
        if callable(output):
            comp.stage1[module.name] = output(ctx, frag)
    ctx.draft = comp.draft()
    return comp


# ====================================================================== settings -> args


def resolve_settings(
    settings: Mapping[str, Value], record: EngineRecord
) -> Tuple[Dict[str, Value], Dict[str, Value]]:
    """Resolve engine-neutral settings to ``(args, env)`` through the record's ``settings`` mapping."""
    args: Dict[str, Value] = {}
    env: Dict[str, Value] = {}
    json_args: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
    online = bool(settings.get("weights.online").value) if settings.get("weights.online") else False
    for name, val in settings.items():
        if name in META_SETTINGS:
            continue
        mapping = record.settings.get(name)
        if not mapping:
            raise CompositionError(f"setting {name!r} has no mapping in record {record.id}")
        if mapping.get("unsupported"):
            raise CompositionError(
                f"setting {name!r} is unsupported by record {record.id}; the module should not apply"
            )
        if mapping.get("when") == "online" and not online:
            continue
        value = val.value
        if mapping.get("invert"):
            value = not value
        if "json_key" in mapping:
            obj, modules = json_args.setdefault(mapping["arg"], ({}, []))
            obj[mapping["json_key"]] = value
            modules.append(f"{val.module}: {val.reason}" if val.reason else val.module)
            continue
        if isinstance(value, bool) and value is False and mapping.get("negated_arg"):
            args[mapping["negated_arg"]] = Value(True, val.module, val.reason)
            continue
        if "env" in mapping:
            env[mapping["env"]] = Value(str(value), val.module, val.reason)
        else:
            args[mapping["arg"]] = Value(value, val.module, val.reason)
        for implied, implied_value in (mapping.get("implies") or {}).items():
            args.setdefault(implied, Value(implied_value, val.module, f"implied by {name}"))
    for arg, (obj, modules) in json_args.items():
        args[arg] = Value(dict(sorted(obj.items())), modules[0].split(":")[0], "; ".join(modules))
    return args, env


# ====================================================================== assembly


def _requires(spec: Dict[str, Any]) -> List[str]:
    req = set()
    for c in spec["containers"]:
        if c.get("args") or c.get("raw_args"):
            req.add("container.args")
        if c.get("env"):
            req.add("container.env")
        if c.get("command"):
            req.add("container.command")
        if c["role"] == "sidecar":
            req.add("sidecar")
        for kind in ("requests", "limits"):
            for res in (c.get("resources") or {}).get(kind, {}):
                req.add(
                    {
                        "nvidia.com/gpu": "resources.gpu",
                        "nvidia.com/gpumem": "resources.gpumem",
                        "memory_mib": "resources.memory",
                        "cpu": "resources.cpu",
                    }.get(res, f"resources.{res}")
                )
    for vol in spec["pod"].get("volumes", []):
        req.add(
            {
                "emptyDir": "volume.emptyDir",
                "pvc": "volume.pvc",
                "model_store": "volume.model_store",
                "configMap": "volume.configMap",
                "hostPath": "volume.hostPath",
                "secret": "volume.secret",
            }[vol["kind"]]
        )
        if vol["kind"] == "emptyDir" and vol.get("medium") == "Memory":
            req.add("volume.emptyDir.memory")
    if spec["pod"].get("scheduler") not in (None, "default"):
        req.add("pod.scheduler")
    if spec["pod"].get("runtime_class"):
        req.add("pod.runtime_class")
    if spec["pod"].get("affinity"):
        req.add("pod.affinity")
    for check in spec["checks"]:
        req.add(f"check.{check['kind']}")
    if any(a.get("when", {}).get("check") == "boot_failed" for a in spec["alternatives"]):
        req.add("check.boot_failed")
    return sorted(req)


def inputs_digest(ctx: PlanContext) -> str:
    placement = ctx.placement.__dict__ if ctx.placement else None
    blob = json.dumps(
        {
            "intent": ctx.intent.__dict__,
            "model": ctx.model.digest(),
            "record": ctx.record.id,
            "features": sorted(ctx.record.features),
            "device": ctx.device.digest(),
            "placement": placement,
            "kv_plan": ctx.kv_plan,
            "modules": dict(sorted(ctx.module_versions.items())),
            "versions": ctx.predictor.versions(),
        },
        sort_keys=True,
        default=lambda o: getattr(o, "__dict__", str(o)),
    )
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def assemble(ctx: PlanContext, comp: Composition) -> RuntimeSpec:
    record = ctx.record
    resolved_args, resolved_env = resolve_settings(comp.settings, record)
    for key, val in resolved_args.items():
        if key in comp.args and comp.args[key].value != val.value:
            raise CompositionError(
                f"argument {key!r} is set directly by {comp.args[key].module} and resolved from a setting by {val.module}"
            )
    runtime = dict(record.runtime or {})
    base_args = dict((runtime.get("base_args") or {}))
    args: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for key, value in base_args.items():
        args[key] = value
        provenance[f"containers.engine.args.{key}"] = "engine.base: record base_args binding"
    for source in (comp.args, resolved_args):
        for key, val in source.items():
            if key in base_args and source is comp.args:
                args[key] = val.value
            else:
                args.setdefault(key, val.value) if key in args else args.__setitem__(key, val.value)
            provenance[f"containers.engine.args.{key}"] = f"{val.module}: {val.reason}".rstrip(": ")
    env = {k: v.value for k, v in sorted(comp.env.items())}
    for key, val in resolved_env.items():
        env[key] = val.value
        provenance[f"containers.engine.env.{key}"] = f"{val.module}: {val.reason}".rstrip(": ")
    for key, val in comp.env.items():
        provenance.setdefault(
            f"containers.engine.env.{key}", f"{val.module}: {val.reason}".rstrip(": ")
        )
    for name, val in comp.settings.items():
        provenance[f"settings.{name}"] = f"{val.module}: {val.reason}".rstrip(": ")
    resources = {
        kind: {k: v.value for k, v in items.items()} for kind, items in comp.resources.items()
    }
    for kind, items in comp.resources.items():
        for k, v in items.items():
            provenance[f"containers.engine.resources.{kind}.{k}"] = (
                f"{v.module}: {v.reason}".rstrip(": ")
            )
    engine_container = {
        "role": "engine",
        "name": comp.engine.get("container_name", "budruntime-container"),
        "image": comp.engine.get("image") or record.images.get(ctx.device.device_class, ""),
        "command": list(
            comp.engine.get("command") or (runtime.get("launch") or {}).get("single_node") or []
        ),
        "args": args,
        "env": env,
        "ports": list(comp.ports.values()),
        "resources": {
            "requests": resources.get("requests", {}),
            "limits": resources.get("limits", {}),
        },
        "mounts": list(comp.mounts.values()),
        "probes": dict(comp.probes),
        "image_pull_policy": comp.engine.get("pull_policy"),
    }
    pod: Dict[str, Any] = {
        "labels": {},
        "annotations": {},
        "node_selector": {},
        "volumes": list(comp.volumes.values()),
    }
    for path, val in comp.pod.items():
        head, _, rest = path.partition(".")
        if head in ("labels", "annotations", "node_selector") and rest:
            pod[head][rest] = val.value
        else:
            pod[path] = val.value
        provenance[f"pod.{path}"] = f"{val.module}: {val.reason}".rstrip(": ")
    containers = [engine_container] + [comp.containers[name] for name in sorted(comp.containers)]
    placement = ctx.placement
    arg_cfg = runtime.get("args") or {}
    spec_dict: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "engine": {
            "name": record.engine,
            "version": record.version,
            "record": f"bud-connect:{record.id}",
            "image": engine_container["image"],
            "pull_policy": comp.engine.get("pull_policy") or "IfNotPresent",
            "arg_style": arg_cfg.get("style", "--{key}={value}"),
            "bool_style": arg_cfg.get("bool", "flag"),
            "trailing_args": list(comp.engine.get("trailing_args", arg_cfg.get("trailing", []))),
            "boot_failure_patterns": copy.deepcopy(runtime.get("boot_failure_patterns")) or None,
        },
        "topology": {"kind": "single" if not placement or placement.pp <= 1 else "multi_node"},
        "replicas": int(placement.replicas if placement else 1),
        "settings": {k: v.value for k, v in comp.settings.items()},
        "predictor": ctx.predictor.versions(),
        "predicted": comp.predicted,
        "pod": pod,
        "containers": containers,
        "autoscaling": comp.mutable.pop("_autoscaling", None) or {"enabled": False},
        "checks": [c for c in comp.checks if c.get("kind") in CHECK_KINDS],
        "alternatives": comp.alternatives,
        "mutable": comp.mutable,
        "decisions": comp.decisions,
        "provenance": dict(sorted(provenance.items())),
        "inputs_digest": inputs_digest(ctx),
    }
    spec_dict["requires"] = _requires(spec_dict)
    if spec_dict["predicted"] is None:
        spec_dict.pop("predicted")
    return RuntimeSpec.model_validate(spec_dict)


def compose(
    ctx: PlanContext, modules: Iterable[Any], *, numerics: Optional[Composition] = None
) -> RuntimeSpec:
    """Run the capacity stage after a finished numerics stage and assemble the spec."""
    mods = list(modules)
    comp = numerics or run_stage("numerics", _stage(ctx, "numerics"), mods)
    capacity_ctx = _stage(ctx, "capacity")
    comp = run_stage("capacity", capacity_ctx, mods, comp)
    for module in mods:
        post = getattr(module, "post_compose", None)
        if callable(post) and module.stage == "capacity":
            post(capacity_ctx, comp)
    return assemble(capacity_ctx, comp)


def _stage(ctx: PlanContext, stage: str) -> PlanContext:
    ctx.stage = stage
    return ctx
