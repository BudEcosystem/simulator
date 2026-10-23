"""Print the runtime spec a plan composes, and the engine argv it renders (FRD-025 §11 "before deploy").

    python -m llm_memory_calculator.runtime.render plan.json [--modules legacy|planner-capacity|planner] [--argv]

``plan.json``: ``{"model_config": {...} | "model_config_path": "...", "family": "...", "record": "vllm@0.30.0" |
{BudConnect row}, "device": {cluster_info device row} | "sku": "h100xm-80c-vgpu", "intent": {Intent fields},
"placement": {Placement fields incl. "group": {NodeGroupConfiguration fields}}, "kv_plan": {...}}``.
A planner change is reviewed by diffing this output before and after.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ..device import HardwareMode, catalogue, facts_from_node
from ..model import ModelFacts
from . import plan
from .context import Intent, Placement, PrecisionCandidate, SpeculativeIntent
from .modules import LEGACY, PLANNER, PLANNER_CAPACITY
from .records import record as vendored_record
from .records import record_from_budconnect
from .spec import flatten_args

PRESETS = {"legacy": LEGACY, "planner-capacity": PLANNER_CAPACITY, "planner": PLANNER}
#: Earlier preset names, still accepted.
PRESET_ALIASES = {"frd027": "planner", "frd027-capacity": "planner-capacity"}


def build(data: Dict[str, Any], preset: str):
    cfg = data.get("model_config") or json.loads(Path(data["model_config_path"]).read_text())
    model = ModelFacts.from_config(cfg, family=data.get("family"))
    rec = data.get("record") or "vllm@0.30.0"
    record = vendored_record(rec) if isinstance(rec, str) else record_from_budconnect(rec)
    if record is None:
        raise SystemExit("no engine record for that BudConnect row")
    if "sku" in data:
        device = catalogue.get(data["sku"])
    else:
        device = facts_from_node(data.get("device") or {})
    intent_data = dict(data.get("intent") or {})
    spec = intent_data.pop("speculative", None)
    cands = [PrecisionCandidate(**c) for c in intent_data.pop("precision_candidates", [])]
    intent = Intent(
        **intent_data,
        precision_candidates=cands,
        speculative=SpeculativeIntent(**spec) if spec else SpeculativeIntent(),
    )
    placement = (
        Placement(**data["placement"])
        if data.get("placement")
        else Placement(device_type=device.device_class, concurrency=intent.concurrency)
    )
    if placement.hardware_mode == "shared":
        slice_mib = int((data.get("placement") or {}).get("slice_mib") or 0) or int(
            device.memory_visible_bytes // 2**20
        )
        device = device.with_mode(HardwareMode("shared", slice_mib * 2**20))
    ctx = plan.context(
        intent,
        model,
        cfg,
        record,
        device,
        placement=placement,
        versions=PRESETS[PRESET_ALIASES.get(preset, preset)],
        kv_plan=data.get("kv_plan"),
    )
    return plan.spec(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--modules", choices=sorted(PRESETS) + sorted(PRESET_ALIASES), default="planner"
    )
    parser.add_argument("--argv", action="store_true", help="print only the engine argv")
    args = parser.parse_args(argv)
    spec = build(json.loads(args.plan.read_text()), args.modules)
    if args.argv:
        eng = spec.engine_container()
        print(
            " ".join(
                eng.command
                + flatten_args(eng.args, style=spec.engine.arg_style)
                + spec.engine.trailing_args
            )
        )
    else:
        json.dump(spec.to_wire(), sys.stdout, indent=1, sort_keys=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
