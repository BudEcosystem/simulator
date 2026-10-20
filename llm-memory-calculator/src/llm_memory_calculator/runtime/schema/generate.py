from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..spec import RuntimeSpec
from ..vocabulary import MAPPING_KEYS, VERSION, VOCABULARY

ROOT = Path(__file__).resolve().parent.parent


def spec_schema() -> Dict[str, Any]:
    schema = RuntimeSpec.model_json_schema()
    schema["$id"] = "https://bud.studio/schemas/runtime-spec/v1.json"
    schema["title"] = "Bud runtime spec v1 (FRD-025)"
    return schema


def settings_schema() -> Dict[str, Any]:
    """The settings vocabulary and the shape of an engine record's ``settings`` mapping and ``conflicts``
    (FRD-027 §11): what BudConnect validates every record against."""
    mapping = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "arg": {"type": "string"},
            "env": {"type": "string"},
            "json_key": {"type": "string"},
            "implies": {"type": "object"},
            "invert": {"type": "boolean"},
            "negated_arg": {"type": "string"},
            "when": {"type": "string", "enum": ["online"]},
            "unsupported": {"type": "boolean"},
            "transform": {"type": "string"},
            "note": {"type": "string"},
        },
    }
    assert set(mapping["properties"]) == MAPPING_KEYS
    names = sorted(VOCABULARY)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://bud.studio/schemas/runtime-settings/v{VERSION}.json",
        "title": f"Engine-neutral settings v{VERSION} (FRD-027 §5.1) and the engine record mapping",
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "propertyNames": {"enum": names},
                "additionalProperties": mapping,
            },
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "when"],
                    "properties": {
                        "id": {"type": "string"},
                        "when": {"type": "object", "propertyNames": {"enum": names}},
                        "requires": {"type": "object", "propertyNames": {"enum": names}},
                        "unless_feature": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        "$defs": {"vocabulary": {name: VOCABULARY[name] for name in names}},
    }


def write_all() -> List[Path]:
    out = []
    for path, schema in (
        (ROOT / "schema" / "v1.json", spec_schema()),
        (ROOT / "settings" / "v1.json", settings_schema()),
    ):
        path.write_text(json.dumps(schema, indent=1, sort_keys=True) + "\n")
        out.append(path)
    return out
