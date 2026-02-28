"""Tests for serving simulation API endpoints."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow testserver host for TestClient
os.environ.setdefault("BUD_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")

# Add the BudSimulator to path (matches existing test pattern)
budsimulator_path = Path(__file__).parent.parent
if str(budsimulator_path) not in sys.path:
    sys.path.insert(0, str(budsimulator_path))

try:
    from apis.main import app
    APP_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Import error: {e}")
    APP_AVAILABLE = False
    app = None


@pytest.fixture
def client():
    if not APP_AVAILABLE:
        pytest.skip("BudSimulator app could not be imported")
    return TestClient(app)


class TestMemoryTiersEndpoint:
    def test_basic_request(self, client):
        resp = client.post("/api/v2/memory/tiers", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "A100_80GB_GPU",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "meta-llama/Meta-Llama-3.1-8B"
        assert data["bytes_per_token_kv"] > 0
        assert "device_hbm" in data["tiers"]

    def test_with_tensor_parallel(self, client):
        resp = client.post("/api/v2/memory/tiers", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "A100_80GB_GPU",
            "tensor_parallel": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["bytes_per_token_kv"] > 0

    def test_multi_tier(self, client):
        resp = client.post("/api/v2/memory/tiers", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "A100_80GB_GPU",
            "host_ddr_gb": 256,
            "nvme_gb": 1024,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "device_hbm" in data["tiers"]
        assert "host_ddr" in data["tiers"]
        assert "nvme" in data["tiers"]

    def test_unknown_hardware(self, client):
        resp = client.post("/api/v2/memory/tiers", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "NONEXISTENT_GPU",
        })
        assert resp.status_code in (404, 500)


class TestPowerEstimateEndpoint:
    def test_basic_request(self, client):
        resp = client.post("/api/v2/power/estimate", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "A100_80GB_GPU",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["base_power_w"] > 0
        assert "estimated_power" in data
        assert "energy_summary" in data

    def test_unknown_hardware(self, client):
        resp = client.post("/api/v2/power/estimate", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "NONEXISTENT_GPU",
        })
        assert resp.status_code in (404, 500)

    def test_with_batch_size(self, client):
        resp = client.post("/api/v2/power/estimate", json={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "A100_80GB_GPU",
            "batch_size": 32,
            "input_tokens": 2048,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["energy_summary"]["total_energy_j"] > 0
