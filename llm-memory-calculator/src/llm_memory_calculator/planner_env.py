"""Planner settings from the environment.

Each setting is ``PLANNER_<NAME>``; the earlier ``FRD027_<NAME>`` spelling is still read when the new one is unset,
so a deployment configured before the rename keeps its settings.
"""

from __future__ import annotations

import os

#: Read in this order: the current prefix first, then the one it replaced.
PREFIXES = ("PLANNER_", "FRD027_")


def planner_env(name: str, default: str = "") -> str:
    """Return the planner setting ``name`` (without its prefix), or ``default`` when neither spelling is set."""
    for prefix in PREFIXES:
        value = os.environ.get(prefix + name)
        if value is not None:
            return value
    return default
