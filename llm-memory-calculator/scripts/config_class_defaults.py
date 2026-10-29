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
    _record_dependencies(layouts)
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
_RULE_DEPTHS = (5, 12, 13, 24, 37, 48, 64, 96, 130)
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


#: The __init__ parameters a layout rule already reads (the depth, the window, its switch) or that
#: the readers take before any rule (an explicit per-layer list). Every other parameter is probed.
_RULE_INPUTS = ("num_hidden_layers", "sliding_window", "use_sliding_window", "layer_types")
#: The depths a dependency is probed at, and the depths an accepted value is checked at (deeper
#: than any class's default `max_window_layers`, 80, so a rule a deep stack breaks is not shipped).
_DEP_DEPTHS = (24, 96)
_DEP_CHECK_DEPTHS = (12, 13, 24, 37, 64, 96, 130)


def _record_dependencies(layouts):
    """Each rule's ``depends`` (``_layout_dependencies``), in two passes: the second probes every
    class again with the keys the first found to matter for any class and the readers' own layout
    keys (``layer_plan._LAYOUT_KEYS``), which a class may read from its keyword arguments alone. A
    class that cannot be probed ships no rule (the installed transformers is asked)."""
    from transformers import CONFIG_MAPPING

    from llm_memory_calculator.layer_plan import _LAYOUT_KEYS

    extra = set(_LAYOUT_KEYS)
    for rnd in (0, 1):
        found = set()
        for mt in list(layouts):
            rule = {k: v for k, v in layouts[mt].items() if k != "depends"}
            cls = CONFIG_MAPPING[mt]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stated = cls().to_dict()
                deps = _layout_dependencies(cls, rule, stated, sorted(extra) if rnd else ())
            except Exception:  # noqa: BLE001
                layouts.pop(mt)
                continue
            found.update(deps)
            layouts[mt] = {**rule, "depends": deps} if deps else rule
        extra |= found


def _trial_values(default, n):
    """Values to build the class with for a parameter whose default is ``default``, at depth ``n``:
    other numbers, the other flag, per-layer lists, index lists and pattern strings -- whatever
    type the parameter takes, the shapes a config may state for it."""
    lists = [[0, 1] * (n // 2), [1, 0] * (n // 2), [1] * n, [0] * n,
             [1, 1, 1, 0] * (n // 4), [3, 7, 11], [0], [n - 1],
             ["attention" if i % 3 == 0 else "mamba" for i in range(n)]]
    strings = ["LG", "LLLG", "M*M*M*M*", "full_attention"]
    ints = list(range(9))
    if isinstance(default, bool):
        return [not default]
    if isinstance(default, int):
        return [v for v in ints + [default + 1, 2 * default] if v != default]
    if isinstance(default, float):
        return [default / 2, default * 2] if default else [0.5, 1.0]
    if isinstance(default, str):
        return [s for s in strings if s != default]
    if isinstance(default, (list, tuple)):
        return lists + ([] if not default else [[]])
    if default is None:
        return ints + [True] + lists + strings
    return []


def _same(a, b):
    """Equal as JSON values (``True`` is not ``1``; a list is not a tuple's string)."""
    try:
        return json.dumps(_plain(a), sort_keys=True) == json.dumps(_plain(b), sort_keys=True)
    except Exception:  # noqa: BLE001
        return False


def _layout_dependencies(cls, rule, stated, extra_keys=()):
    """``{parameter: {"values": [...], "period": [...]?}}``: every parameter of the class other than
    the rule's own inputs that changes the layout the class derives -- built at another value, the
    class's list is not the rule's -- with the values at which the rule still holds: the
    parameter's default, the value the class stores for it when it is left out (where that is the
    same at every depth), each probed value that gives the rule's list at every depth checked, or,
    for a per-layer list the class derives, the period it repeats. ``layer_plan`` applies the rule
    only to a config whose stated value of each is one of those (AfmoeConfig's
    `global_attn_every_n_layers`, Llama4TextConfig's `no_rope_layer_interval`, a Qwen2-VL
    `max_window_layers` under the window); any other value is the class's to answer.

    The parameters probed are the class's __init__ parameters, the keys its default build states,
    and ``extra_keys`` -- keys a class reads from its keyword arguments alone (Cohere2Config's
    `sliding_window_pattern`), which ``snapshot`` gathers as every key found to matter for any
    class plus the readers' own layout keys."""
    import inspect

    from llm_memory_calculator.layer_plan import class_layout_names

    try:
        sig = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        sig = {}
    params = {}
    for key, prm in sig.items():
        if key == "self" or prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        params[key] = None if prm.default is inspect.Parameter.empty else prm.default
    for key in list(stated) + list(extra_keys):
        if key not in params and not str(key).startswith("_"):
            params[key] = None
    skip = set(_RULE_INPUTS)
    if rule["rule"] == "max_window_layers":
        skip.add("max_window_layers")
    switches = [{}]
    if "switch" in rule or isinstance(stated.get("use_sliding_window"), bool):
        switches = [{"use_sliding_window": True}, {"use_sliding_window": False}]

    def build(n, extra):
        kwargs = {"num_hidden_layers": n, **extra}
        if "sliding_window" in stated:
            kwargs["sliding_window"] = 4096
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _class_names(cls(**kwargs))

    def holds(n, switch, key, value, strict=False):
        """Whether the class built with ``key=value`` gives the rule's list (a build that raises
        says nothing about the layout; ``strict``, for a value the rule is to accept, it fails)."""
        try:
            got = build(n, {**switch, key: value})
        except Exception:  # noqa: BLE001
            return not strict
        return got == class_layout_names(rule, {"num_hidden_layers": n, **switch, key: value})

    deps = {}
    for key, default in params.items():
        if key in skip or isinstance(default, dict):
            continue
        if all(holds(n, s, key, v) for n in _DEP_DEPTHS for v in _trial_values(default, n)
               for s in switches):
            continue
        entry = {"values": [_plain(default)]}
        # a value at which the class gives the rule's list at every depth (a key read from the
        # keyword arguments alone has no default to read: Cohere2Config's pattern of 4)
        # (small values alone, below every depth checked: a larger one may hold at the depths
        # checked and not at a deeper stack's)
        for v in _trial_values(default, _DEP_CHECK_DEPTHS[0]):
            if isinstance(v, (list, tuple)) or (isinstance(v, (int, float)) and not isinstance(
                    v, bool) and v >= _DEP_CHECK_DEPTHS[0]):
                continue
            if not any(_same(v, x) for x in entry["values"]) and all(
                    holds(n, s, key, v, strict=True) for n in _DEP_CHECK_DEPTHS for s in switches):
                entry["values"].append(_plain(v))
        # what the class stores for the key when a config leaves it out: a hub config written by
        # `save_pretrained` states it (Llama 4's `no_rope_layers`, a list over its 48 layers)
        try:
            stored = {}
            for n in _DEP_CHECK_DEPTHS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stored[n] = getattr(cls(num_hidden_layers=n), key, None)
        except Exception:  # noqa: BLE001
            stored = {}
        if stored:
            first = stored[_DEP_CHECK_DEPTHS[0]]
            if all(_same(v, first) for v in stored.values()):
                if not any(_same(first, x) for x in entry["values"]) and all(
                        holds(n, s, key, first, strict=True)
                        for n in _DEP_CHECK_DEPTHS for s in switches):
                    entry["values"].append(_plain(first))
            elif all(isinstance(v, (list, tuple)) and len(v) == n for n, v in stored.items()):
                longest = list(stored[max(_DEP_CHECK_DEPTHS)])
                for p in range(1, 9):
                    unit = longest[:p]
                    if all(list(v) == [unit[i % p] for i in range(n)] for n, v in stored.items()):
                        if all(holds(n, s, key, [unit[i % p] for i in range(n)], strict=True)
                               for n in _DEP_CHECK_DEPTHS for s in switches):
                            entry["period"] = _plain(unit)
                        break
        deps[key] = entry
    return deps


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
