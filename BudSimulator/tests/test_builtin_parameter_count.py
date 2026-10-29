"""The parameter count BudSimulator lists and stores for GenZ's built-in models (round 20).

``DynamicModelCollection.get_model_metadata`` (behind ``/models/list``, ``/models/filter`` and
``/models/popular``) and ``sync_static_models_to_db`` (whose stored count
``ModelManager.get_model_config`` hands the calculator as ``num_parameters``) counted a built-in
ModelConfig from the HF config ``ModelConverter.genz_to_hf`` makes of it -- which states no gated
FFN for a GeGLU model, no MLA rank, no shared expert and no dense leading layer -- or, on
origin/main, from a fallback that read every MoE layer's experts at the dense width. The training
routes mapped a built-in by hand the same way. The count is now the one the HF config states
(``ModelConverter.genz_parameter_count``): GenZ's own count of the ModelConfig's fields, never
below the calculator's count of the HF config, and the training routes use the same conversion.
"""

import asyncio
import contextlib
import io
import warnings

import pytest

from llm_memory_calculator.genz.Models.default_models import MODEL_DICT
from llm_memory_calculator.genz.serving.memory_model import model_parameter_count

#: transformers 5.12 builds of each model's hub config (the parameters its module holds), or the
#: config arithmetic where transformers does not build the model type; the GenZ built-ins named.
TRUTHS = {
    # GemmaForCausalLM on google/gemma-7b-it's config (the unsloth mirror of the gated repo)
    "google/gemma-7b-it": 8_537_680_896,
    # GlmForCausalLM on THUDM/glm-4-9b-chat's shape (padded vocabulary 151,552, 2 KV heads of 128,
    # a bias on Q, K and V, an untied head)
    "THUDM/glm-4-9b-chat": 9_399_951_360,
    # DeepseekV2ForCausalLM on deepseek-ai/DeepSeek-V2's config
    "deepseek-ai/DeepSeek-V2": 235_741_434_880,
    # deepseek-moe-16b-base (model_type deepseek): 2 x 102,400 x 2,048 + 28 x 4 x 2,048^2 + one
    # dense layer's 3 x 2,048 x 10,944 + 27 x (66 experts x 3 x 2,048 x 1,408 + 64 x 2,048) +
    # (2 x 28 + 1) x 2,048
    "deepseek-ai/deepseek-moe-16b-base": 16_375_728_128,
}


def _quiet(f):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


def _loader(models, manager=None):
    from src.db.model_loader import DynamicModelCollection

    return DynamicModelCollection(base_models=models, model_manager=manager, auto_import=False)


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    """A database of its own, never the tree's prepopulated.db."""
    from src.db import connection

    monkeypatch.setattr(connection.DatabaseConnection, "_instance", None)
    db = connection.DatabaseConnection(str(tmp_path / "models.db"))
    yield db
    monkeypatch.setattr(connection.DatabaseConnection, "_instance", None)


@pytest.mark.parametrize("name", sorted(TRUTHS))
def test_the_listed_count_is_at_least_the_model(name):
    """Gemma-7B's GeGLU read as two matrices (6.86B), GLM-4-9B (8.41B), DeepSeek-V2 without its
    MLA ranks and shared experts (234.0B), DeepSeek-MoE-16B (15.99B): each below its model."""
    mc = MODEL_DICT.get_model(name)
    listed = _quiet(lambda: _loader({name: mc}).get_model_metadata(name)["parameter_count"])
    assert listed >= TRUTHS[name], (name, listed, TRUTHS[name])


def test_every_builtin_is_listed_at_genz_s_count_or_above():
    """No built-in name (aliases too) is listed below GenZ's own count of its ModelConfig, nor
    below the calculator's count of the HF config BudSimulator stores for it."""
    from src.db.model_converter import ModelConverter
    from src.db.model_manager import estimate_parameter_count

    low = []
    for name, mc in sorted(MODEL_DICT.models.items()):
        listed = _quiet(lambda: _loader({name: mc}).get_model_metadata(name)["parameter_count"])
        hf = _quiet(lambda: ModelConverter().genz_to_hf(mc))
        hf.pop("num_parameters", None)
        floor = max(model_parameter_count(mc), _quiet(lambda: estimate_parameter_count(hf)) or 0)
        if listed < floor:
            low.append((name, listed, floor))
    assert not low, low[:10]


def test_the_stored_count_reaches_the_calculator(scratch_db):
    """sync_static_models_to_db stores the count; the model manager hands it to the calculator as
    num_parameters, and the weights follow it: Gemma-7B's were 13.7 GB at bf16 (17.1 GB held)."""
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from src.db.model_manager import ModelManager

    name = "google/gemma-7b-it"
    mc = MODEL_DICT.get_model(name)
    manager = ModelManager()
    _quiet(lambda: _loader({name: mc, mc.model: mc}, manager).sync_static_models_to_db())
    cfg = _quiet(lambda: manager.get_model_config(mc.model))
    assert cfg["num_parameters"] >= TRUTHS[name]
    rep = _quiet(lambda: ModelMemoryCalculator().calculate_total_memory(
        dict(cfg), batch_size=1, seq_length=2048, precision="bf16"))
    assert rep.weight_memory_gb >= 2 * TRUTHS[name] / 1e9


def test_the_stored_config_states_the_gated_ffn_and_the_count():
    """genz_to_hf states a num_ffi = 2 FFN as gated (`is_gated_act`) and the count."""
    from src.db.model_converter import ModelConverter

    mc = MODEL_DICT.get_model("google/gemma-7b-it")
    hf = _quiet(lambda: ModelConverter().genz_to_hf(mc))
    assert hf["is_gated_act"] is True
    assert hf["num_parameters"] >= TRUTHS["google/gemma-7b-it"]


def test_the_training_route_counts_a_builtin_as_the_loader_does(monkeypatch):
    """/training/estimate-training on a built-in name (the GenZ registry fallback) counted
    Gemma-7B at 7.24B; it reads the same conversion the loader stores."""
    from apis.routers import training

    class Offline:
        def get_model_config(self, *a, **k):
            raise RuntimeError("no hub in the test")

    monkeypatch.setattr(training, "HuggingFaceConfigLoader", Offline)

    r = _quiet(lambda: asyncio.run(training.estimate_training(training.TrainingEstimateRequest(
        model="google/gemma-7b-it", method="full", batch_size=1, seq_length=2048,
        precision="bf16", optimizer="adamw"))))
    assert r.total_params >= TRUTHS["google/gemma-7b-it"]
