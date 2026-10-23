"""Planner settings keep working under the names they had before the rename (``FRD027_*``, preset ``frd027``)."""

import pytest

from llm_memory_calculator.perf.rank import slo_confidence
from llm_memory_calculator.planner_env import planner_env
from llm_memory_calculator.runtime import render
from llm_memory_calculator.runtime.modules import PLANNER, PLANNER_CAPACITY


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("PLANNER_SLO_CONFIDENCE", "FRD027_SLO_CONFIDENCE"):
        monkeypatch.delenv(name, raising=False)


def test_the_current_name_is_read():
    """``PLANNER_<NAME>`` is the setting."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PLANNER_SLO_CONFIDENCE", "0.8")
        assert slo_confidence() == 0.8


def test_the_earlier_name_is_still_read():
    """A deployment configured with ``FRD027_<NAME>`` keeps its setting."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FRD027_SLO_CONFIDENCE", "0.7")
        assert slo_confidence() == 0.7


def test_the_current_name_wins_over_the_earlier_one():
    """Both set: the current name decides."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PLANNER_SLO_CONFIDENCE", "0.8")
        mp.setenv("FRD027_SLO_CONFIDENCE", "0.7")
        assert planner_env("SLO_CONFIDENCE") == "0.8"


def test_neither_set_is_the_default():
    """Unset under both names: the default."""
    assert planner_env("SLO_CONFIDENCE", "0.9") == "0.9"


@pytest.mark.parametrize(("name", "versions"), [("planner", PLANNER), ("planner-capacity", PLANNER_CAPACITY)])
def test_presets_and_their_earlier_names(name, versions):
    """Each preset is reachable by its current name and by the one it had before."""
    assert render.PRESETS[name] is versions
    earlier = {v: k for k, v in render.PRESET_ALIASES.items()}[name]
    assert render.PRESETS[render.PRESET_ALIASES[earlier]] is versions
