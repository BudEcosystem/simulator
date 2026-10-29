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
that ties its head whatever the config says (``tie_forced``) is recorded untied. How the class
lays out its layers when a config states a window but no layout (``layouts``: the per-layer list
transformers 5.12 derives, as a rule -- no layout, a fixed period, a period whose last layer is
another kind, or windowed from `max_window_layers` on -- verified against the class at several
depths and windows) is what ``layer_plan.stated_layer_types`` reads for such a config. Shipping
them makes every reader's figure the same whether transformers is installed, absent, or a release
that predates the class.

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
    width_derived, layouts, head_dims = {}, {}, {}
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
        rule = _layout_rule(CONFIG_MAPPING[mt], obj)
        if rule is not None:
            layouts[mt] = rule
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
                rule = _head_dim_rule(CONFIG_MAPPING[mt], stated, derived)
                if rule is not None:
                    head_dims[mt] = rule
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
            "width_derived": width_derived,
            "layouts": layouts,
            "head_dims": head_dims}


def _head_dim_rule(cls, stated, derived):
    """A head width the class derives from its width under another name (``head_dims``):
    ``{"key": name, "ratio": r}`` -- the width is ``int(r x hidden_size) // num_attention_heads``
    (ZambaConfig's `attention_head_dim = 2 x hidden_size / heads`: its attention reads the hidden
    state concatenated with the embeddings) -- checked against the class at several widths and
    head counts, which ``layer_plan.attention_head_dim`` reads for a config that states neither
    the width nor its name. None for any other class."""
    from llm_memory_calculator.layer_plan import HEAD_DIM_KEYS

    key = next((k for k in HEAD_DIM_KEYS[1:] if k in derived), None)
    h, n = stated.get("hidden_size"), stated.get("num_attention_heads")
    if not (key and isinstance(h, int) and isinstance(n, int) and h > 0 and n > 0):
        return None
    ratio = stated[key] * n / h
    try:
        for hh, nn in ((1024, 8), (2048, 16), (3072, 24), (4096, 32), (5120, 40)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = getattr(cls(hidden_size=hh, num_attention_heads=nn), key, None)
            if got != int(ratio * hh) // nn:
                return None
    except Exception:  # noqa: BLE001
        return None
    return {"key": key, "ratio": ratio}


#: Depths and windows a layout rule is checked at: it must give the class's own list at each.
_RULE_DEPTHS = (5, 12, 13, 24, 37, 48)
_RULE_WINDOWS = (512, 4096)


def _class_names(obj):
    from llm_memory_calculator.layer_plan import per_layer_names

    names = getattr(obj, "layer_types", None)
    if isinstance(names, (list, tuple)) and names:
        return [str(x) for x in names]
    try:
        names = per_layer_names(obj.to_dict())[0]
    except Exception:  # noqa: BLE001
        names = None
    return [str(x) for x in names] if names else None


def _layout_rule(cls, obj):
    """How the class lays out a windowed config's layers, as ``layer_plan.class_layout_names``
    applies it, or None when no rule reproduces the class (then the installed transformers is
    asked): ``{"rule": "none"}`` -- the class derives no per-layer list (Mistral: the window is
    on every layer); ``{"rule": "period", "names": [...], "last": name?}`` -- layer i is
    ``names[i % len(names)]``, the last layer ``last`` where the class forces it (Gemma 2: sliding,
    full; Gemma 4: five sliding, one full, the last full); ``{"rule": "max_window_layers",
    "default": n}`` -- the window from layer `max_window_layers` on, the class's ``n`` where the
    config states none (Qwen2). A class with a `use_sliding_window` switch records its default
    (``switch``): a config that does not state it takes it, and switched off every layer is full.
    Every rule is checked against the class at several depths, windows, switches and (for the
    last) `max_window_layers` values."""
    from llm_memory_calculator.layer_plan import class_layout_names

    stated = obj.to_dict()
    if not _class_names(obj):
        return {"rule": "none"}

    def build(n, window, **extra):
        kwargs = {"num_hidden_layers": n, **extra}
        if "use_sliding_window" in stated:
            kwargs["use_sliding_window"] = True
        if "sliding_window" in stated:
            kwargs["sliding_window"] = window
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _class_names(cls(**kwargs))

    try:
        probes = {(n, w): build(n, w) for n in _RULE_DEPTHS for w in _RULE_WINDOWS}
    except Exception:  # noqa: BLE001
        return None
    if any(not names or len(names) != n for (n, _), names in probes.items()):
        return None
    candidates = []
    longest = probes[(max(_RULE_DEPTHS), _RULE_WINDOWS[-1])]
    for p in range(1, 9):
        candidates.append({"rule": "period", "names": longest[:p]})
        candidates.append({"rule": "period", "names": longest[:p], "last": longest[-1]})
    if isinstance(stated.get("max_window_layers"), int):
        candidates.append({"rule": "max_window_layers", "default": stated["max_window_layers"]})
    if isinstance(stated.get("use_sliding_window"), bool):
        # the class's own switch: a config that does not state it takes the class's default
        # (Qwen2Config's False: every layer full, whatever window the config states)
        switch = stated["use_sliding_window"]
        candidates = [{**rule, "switch": switch} for rule in candidates]
    for rule in candidates:
        if rule.get("last") is not None and rule["last"] == rule["names"][
                (max(_RULE_DEPTHS) - 1) % len(rule["names"])]:
            continue  # no different last layer: the plain period
        fits = all(class_layout_names(rule, {"num_hidden_layers": n, "use_sliding_window": True})
                   == names for (n, _), names in probes.items())
        if fits and "switch" in rule:
            try:
                fits = all(class_layout_names(rule, {"num_hidden_layers": n, **flag}) == _class_names(
                    cls(num_hidden_layers=n, sliding_window=4096, **flag))
                    for n in _RULE_DEPTHS for flag in ({}, {"use_sliding_window": False}))
            except Exception:  # noqa: BLE001
                fits = False
        if fits and rule["rule"] == "max_window_layers":
            try:
                fits = all(class_layout_names(rule, {"num_hidden_layers": n, "max_window_layers": m,
                                                     "use_sliding_window": True})
                           == build(n, 4096, max_window_layers=m)
                           for n in _RULE_DEPTHS for m in (3, 7, 60))
            except Exception:  # noqa: BLE001
                fits = False
        if fits:
            return rule
    return None


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
