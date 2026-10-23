import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "planner"


def load_model(name: str) -> dict:
    return json.loads((FIXTURES / "models" / f"{name}.json").read_text())


@pytest.fixture
def model_config():
    return load_model


def load_record(name: str) -> dict:
    return json.loads((FIXTURES / "records" / f"{name}.json").read_text())
