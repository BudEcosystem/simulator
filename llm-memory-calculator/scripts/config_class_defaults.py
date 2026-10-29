#!/usr/bin/env python3
"""Write the config-class defaults the readers take for a key a config leaves out.

What a config does not state is its config class's default: Llama4TextConfig's untied head and
"silu", Mllama's vision tower of 8 global layers and a 7,680-wide output, RwkvConfig's untied
head, a class's rotary or learned positions. ``layer_plan.config_class_default`` reads them from
``src/llm_memory_calculator/config_class_defaults.json``, which this script writes from the
installed transformers (5.12 for the shipped file): every model type its ``CONFIG_MAPPING`` knows,
each key of ``layer_plan.CLASS_DEFAULT_KEYS`` its default class states (a key the class leaves
None is left out), and the geometry its default class builds (``geometry``: each key of
``layer_plan.CLASS_GEOMETRY_KEYS`` its ``to_dict()`` states under the class's own name), which
``layer_plan.with_class_geometry`` gives a config block that leaves a role of it out. A class
that ties its head whatever the config says (``tie_forced``) is recorded untied. Shipping them makes every reader's figure the same whether transformers is
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
        from llm_memory_calculator.layer_plan import (
            CLASS_DEFAULT_KEYS, CLASS_GEOMETRY_KEYS, geometry_value)
    finally:
        sys.path.remove(SRC)
    import transformers
    from transformers import CONFIG_MAPPING

    logging.getLogger("transformers").setLevel(logging.ERROR)
    defaults, geometry, tie_forced, sub_configs, auto_sub_configs = {}, {}, [], {}, {}
    width_derived = {}
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
        if row.get("tie_word_embeddings") is True and _forces_tying(CONFIG_MAPPING[mt]):
            # The class ties the head whatever the config says (transformers 5.12's T5, mT5 and
            # UMT5: "we tie weights always"), so its default does not say what a checkpoint of
            # the class holds -- T5 v1.1, flan-T5 and mT5 ship their own `lm_head`, and
            # transformers 4.x's MT5Config defaulted to an untied head. A config that leaves
            # the flag out is read untied: the head is counted, never left out.
            row["tie_word_embeddings"] = False
            tie_forced.append(mt)
        row["model_type"] = mt
        defaults[mt] = row
        try:
            stated = obj.to_dict()
        except Exception:  # noqa: BLE001
            stated = {}
        geo = {k: _plain(stated[k]) for k in CLASS_GEOMETRY_KEYS
               if k in stated and geometry_value(stated[k])}
        if geo:
            geometry[mt] = geo
            derived = _width_derived(CONFIG_MAPPING[mt], stated, geo)
            if derived:
                width_derived[mt] = derived
        # the class each nested block is built as, where the class fixes it (Blip2Config's
        # `vision_config` is a Blip2VisionConfig whatever model type the block names; a
        # `text_config` read through AutoConfig follows the block's own)
        subs, auto = {}, []
        for key, sub in (getattr(CONFIG_MAPPING[mt], "sub_configs", None) or {}).items():
            sub_mt = sub if isinstance(sub, str) else getattr(sub, "model_type", None)
            if not sub_mt:
                # read through AutoConfig: the class the composite builds when the block names
                # none (Florence2Config's `text_config`: a BartConfig)
                sub_mt = getattr(getattr(obj, key, None), "model_type", None)
                auto.append(key)
            if isinstance(sub_mt, str) and sub_mt:
                subs[key] = sub_mt
        if subs:
            sub_configs[mt] = subs
        if auto:
            auto_sub_configs[mt] = auto
    return {"about": "Config-class defaults (scripts/config_class_defaults.py): what a config that "
                     "leaves a key out means, read by layer_plan.config_class_default.",
            "transformers": transformers.__version__, "keys": list(CLASS_DEFAULT_KEYS),
            "defaults": defaults,
            "geometry_keys": list(CLASS_GEOMETRY_KEYS),
            "geometry": geometry,
            "tie_forced": tie_forced,
            "sub_configs": sub_configs,
            "auto_sub_configs": auto_sub_configs,
            "width_derived": width_derived}


def _width_derived(cls, stated, geo):
    """The geometry keys the class sizes from its width (FalconConfig's `ffn_hidden_size`, a
    Llama `head_dim`): those that change when the class is built twice as wide."""
    from llm_memory_calculator.layer_plan import CLASS_GEOMETRY_ROLES

    width_keys = [k for k in CLASS_GEOMETRY_ROLES[1] if isinstance(stated.get(k), int)
                  and not isinstance(stated.get(k), bool)]
    if not width_keys:
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wider = cls(**{width_keys[0]: 2 * stated[width_keys[0]]}).to_dict()
    except Exception:  # noqa: BLE001
        return []
    return sorted(k for k in geo if k not in CLASS_GEOMETRY_ROLES[1] and wider.get(k) != geo[k])


def _forces_tying(cls):
    """Whether the class ties its head even when built with ``tie_word_embeddings=False``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(tie_word_embeddings=False).tie_word_embeddings is True
    except Exception:  # noqa: BLE001
        return False


def main():
    snap = snapshot()
    with open(OUT, "w") as f:
        json.dump(snap, f, indent=0, sort_keys=True)
        f.write("\n")
    print(f"{len(snap['defaults'])} model types, transformers {snap['transformers']} -> {OUT}")


if __name__ == "__main__":
    main()
