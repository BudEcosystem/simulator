#!/usr/bin/env python3
"""Write the config-class defaults the readers take for a key a config leaves out.

What a config does not state is its config class's default: Llama4TextConfig's untied head and
"silu", Mllama's vision tower of 8 global layers and a 7,680-wide output, RwkvConfig's untied
head, a class's rotary or learned positions. ``layer_plan.config_class_default`` reads them from
``src/llm_memory_calculator/config_class_defaults.json``, which this script writes from the
installed transformers (5.12 for the shipped file): every model type its ``CONFIG_MAPPING`` knows,
each key of ``layer_plan.CLASS_DEFAULT_KEYS`` its default class states (a key the class leaves
None is left out). Shipping them makes every reader's figure the same whether transformers is
installed, absent, or a release that predates the class.

    python scripts/config_class_defaults.py          # rewrite the shipped file
"""

import json
import logging
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
OUT = os.path.join(SRC, "llm_memory_calculator", "config_class_defaults.json")


def _plain(v):
    """A JSON value for a default: numbers, strings, lists and dicts as they are; anything else
    (a nested config object) as its dict, else its string."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    to_dict = getattr(v, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict())
        except Exception:  # noqa: BLE001
            pass
    return str(v)


def snapshot():
    """``{"transformers": version, "keys": [...], "defaults": {model_type: {key: value}}}``."""
    sys.path.insert(0, SRC)
    try:
        from llm_memory_calculator.layer_plan import CLASS_DEFAULT_KEYS
    finally:
        sys.path.remove(SRC)
    import transformers
    from transformers import CONFIG_MAPPING

    logging.getLogger("transformers").setLevel(logging.ERROR)
    defaults = {}
    for mt in sorted(CONFIG_MAPPING.keys()):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = CONFIG_MAPPING[mt]()
        except Exception:  # noqa: BLE001  a class that cannot be built bare states nothing
            continue
        row = {}
        for k in CLASS_DEFAULT_KEYS:
            try:
                v = getattr(obj, k, None)
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                row[k] = _plain(v)
        row["model_type"] = mt
        defaults[mt] = row
    return {"about": "Config-class defaults (scripts/config_class_defaults.py): what a config that "
                     "leaves a key out means, read by layer_plan.config_class_default.",
            "transformers": transformers.__version__, "keys": list(CLASS_DEFAULT_KEYS),
            "defaults": defaults}


def main():
    snap = snapshot()
    with open(OUT, "w") as f:
        json.dump(snap, f, indent=0, sort_keys=True)
        f.write("\n")
    print(f"{len(snap['defaults'])} model types, transformers {snap['transformers']} -> {OUT}")


if __name__ == "__main__":
    main()
