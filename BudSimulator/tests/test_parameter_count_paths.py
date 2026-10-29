"""The parameter count BudSimulator stores for a model it is given as a config.

``ModelManager.get_model_config`` (behind ``/models/{id}/config``, ``/models/analyze`` and the
model loader), the gated-config submission and the database setup script asked the calculator for
``_calculate_transformer_params``, which it does not have: the AttributeError was caught and no
count was stored. ``/models/add/config`` estimated 4h^2 + 2hi a layer (no gated FFN, no experts).
Each now stores one count (``src.db.model_manager.estimate_parameter_count``): the calculator's
own (``ModelMemoryCalculator.calculate_model_weights``), never below the route's older estimate.
"""

import asyncio
import json
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator

FIXTURES = Path(__file__).resolve().parents[2] / "llm-memory-calculator" / "tests" / "fixtures"


def _cfg(name):
    return json.loads((FIXTURES / "hybrid_configs" / f"{name}.json").read_text())


def _count(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelMemoryCalculator().calculate_model_weights(json.loads(json.dumps(cfg)),
                                                               "bf16")[0]


def _facts(cfg):
    """The FRD-027 facts' count (every stack, tower and adapter): round 20 holds the stored count
    to it as well."""
    from llm_memory_calculator.model.facts import ModelFacts

    f = ModelFacts.from_config(json.loads(json.dumps(cfg)))
    return int(f.params_total + f.vision_params)


def _older_estimate(cfg):
    h, n = cfg["hidden_size"], cfg["num_hidden_layers"]
    v, i = cfg.get("vocab_size", 50000), cfg.get("intermediate_size", h * 4)
    return v * h * 2 + n * (4 * h * h + 2 * h * i + 4 * h)


def _one_config_manager(cfg):
    from src.db import ModelManager

    class OneConfig(ModelManager):
        def __init__(self, config):  # no database: the config is the user's
            self.config, self.added = config, {}

        def get_user_model_config(self, model_id):
            return json.loads(json.dumps(self.config))

        def get_model(self, model_id):
            return None

        def add_model(self, **kwargs):
            self.added = kwargs
            return kwargs.get("model_id")

    return OneConfig(cfg)


@pytest.mark.parametrize("name", ["LiquidAI_LFM2-2.6B", "Qwen_Qwen3-Next-80B-A3B-Instruct"])
def test_the_model_manager_counts_a_user_config_s_parameters(name):
    cfg = _cfg(name)
    got = _one_config_manager(cfg).get_model_config("m").get("num_parameters")
    assert got == max(_count(cfg), _facts(cfg), _older_estimate(cfg)) and got > 0


@pytest.mark.parametrize("name", ["LiquidAI_LFM2-2.6B", "Qwen_Qwen3-Next-80B-A3B-Instruct"])
def test_add_model_from_config_stores_the_calculator_s_count(name, monkeypatch):
    """LFM2-2.6B's gated FFN and Qwen3-Next's 512 experts: 4h^2 + 2hi a layer alone was 2.09B of
    2.57B and 4.8B of 80B."""
    pytest.importorskip("fastapi")
    from apis.routers import models as routes
    from apis.schemas import AddModelFromConfigRequest

    cfg = _cfg(name)
    manager = _one_config_manager(cfg)
    monkeypatch.setattr(routes, "model_manager", manager)
    asyncio.run(routes.add_model_from_config(AddModelFromConfigRequest(model_id="m", config=cfg)))
    assert manager.added["parameter_count"] == max(_count(cfg), _facts(cfg), _older_estimate(cfg))
    assert manager.added["parameter_count"] >= _count(cfg)


def test_the_stored_count_never_falls_below_the_older_estimate():
    """Zamba2-2.7B: the calculator's count (2.58B) sits below the checkpoint (2.84B: its shared
    blocks), the older estimate (4.41B) above it -- the larger is stored."""
    from src.db.model_manager import estimate_parameter_count

    cfg = _cfg("Zyphra_Zamba2-2.7B")
    assert estimate_parameter_count(cfg) == max(_count(cfg), _facts(cfg), _older_estimate(cfg)) >= 2_844_404_640
