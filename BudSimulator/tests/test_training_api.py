"""
Test suite for Training API endpoints.

Tests the /api/simulator/* endpoints for training memory estimation,
cluster recommendations, and fit checking.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the app
import sys
from pathlib import Path

# Add the BudSimulator to path
budsimulator_path = Path(__file__).parent.parent
if str(budsimulator_path) not in sys.path:
    sys.path.insert(0, str(budsimulator_path))

try:
    from apis.main import app
    client = TestClient(app)
    APP_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    APP_AVAILABLE = False
    app = None
    client = None
except Exception as e:
    print(f"Error loading app: {e}")
    APP_AVAILABLE = False
    app = None
    client = None


# Mock model config for testing
MOCK_LLAMA_8B_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "head_dim": 128,
    "num_parameters": 8_030_261_248,
}


@pytest.fixture
def mock_model_config():
    """Mock HuggingFaceConfigLoader.get_model_config."""
    with patch("apis.routers.training.HuggingFaceConfigLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_instance.get_model_config.return_value = MOCK_LLAMA_8B_CONFIG
        mock_loader.return_value = mock_instance
        yield mock_instance


class TestTrainingEstimateEndpoint:
    """Test /api/simulator/estimate-training endpoint."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_training_lora(self, mock_model_config):
        """Test LoRA training memory estimation."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "batch_size": 4,
                "seq_length": 2048,
                "precision": "bf16",
                "optimizer": "adamw",
                "gradient_checkpointing": True,
                "lora_rank": 16,
            }
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "memory_breakdown" in data
        assert "total_params" in data
        assert "trainable_params" in data
        assert "trainable_percent" in data

        # LoRA should have small trainable percentage
        assert data["trainable_percent"] < 5.0

        # Memory breakdown
        breakdown = data["memory_breakdown"]
        assert breakdown["weight_memory_gb"] > 0
        assert breakdown["gradient_memory_gb"] > 0
        assert breakdown["optimizer_memory_gb"] > 0
        assert breakdown["activation_memory_gb"] > 0
        assert breakdown["total_memory_gb"] > 0

        # LoRA on 8B should fit on 80GB GPU
        assert data["fits_single_gpu_80gb"] is True

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_training_full(self, mock_model_config):
        """Test full fine-tuning memory estimation."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "full",
                "batch_size": 4,
                "seq_length": 2048,
                "precision": "bf16",
                "optimizer": "adamw",
                "gradient_checkpointing": True,
            }
        )
        assert response.status_code == 200
        data = response.json()

        # Full fine-tuning should train all parameters
        assert data["trainable_percent"] > 99.0

        # Full 8B training needs more memory
        breakdown = data["memory_breakdown"]
        assert breakdown["gradient_memory_gb"] > 30  # ~32GB for gradients
        assert breakdown["optimizer_memory_gb"] > 50  # ~64GB for optimizer states

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_training_qlora(self, mock_model_config):
        """Test QLoRA training memory estimation."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "qlora",
                "batch_size": 4,
                "seq_length": 2048,
                "precision": "int4",  # 4-bit quantization
                "optimizer": "adamw",
                "gradient_checkpointing": True,
                "lora_rank": 16,
            }
        )
        assert response.status_code == 200
        data = response.json()

        # QLoRA should fit on 24GB GPU
        assert data["fits_single_gpu_24gb"] is True

        # Weight memory should be ~4GB (4-bit)
        breakdown = data["memory_breakdown"]
        assert breakdown["weight_memory_gb"] < 6

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_training_invalid_method(self, mock_model_config):
        """Test error for invalid training method."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "invalid_method",
                "batch_size": 4,
                "seq_length": 2048,
            }
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_training_deepspeed(self, mock_model_config):
        """Test DeepSpeed ZeRO stage estimation."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "full",
                "batch_size": 4,
                "seq_length": 2048,
                "precision": "bf16",
                "optimizer": "adamw",
                "deepspeed_stage": "zero2",
                "data_parallel": 4,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deepspeed_stage"] == "zero2"


class TestClusterRecommendEndpoint:
    """Test /api/simulator/recommend-cluster endpoint."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_recommend_cluster_basic(self, mock_model_config):
        """Test basic cluster recommendation."""
        response = client.post(
            "/api/simulator/recommend-cluster",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "batch_size": 4,
                "prefer_cost": True,
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        assert "total_options" in data
        assert len(data["recommendations"]) > 0

        # Check first recommendation structure
        rec = data["recommendations"][0]
        assert "hardware_name" in rec
        assert "total_gpus" in rec
        assert "parallelism" in rec
        assert "estimated_cost_per_hour" in rec
        assert "fits" in rec
        assert rec["fits"] is True

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_recommend_cluster_speed_preference(self, mock_model_config):
        """Test speed-optimized cluster recommendation."""
        response = client.post(
            "/api/simulator/recommend-cluster",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "full",
                "batch_size": 4,
                "prefer_cost": False,  # Prefer speed
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) > 0

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_recommend_cluster_budget_constraint(self, mock_model_config):
        """Test cluster recommendation with budget constraint."""
        response = client.post(
            "/api/simulator/recommend-cluster",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "batch_size": 4,
                "prefer_cost": True,
                "max_budget_per_hour": 5.0,
            }
        )
        assert response.status_code == 200
        data = response.json()

        # All recommendations should be under budget
        for rec in data["recommendations"]:
            assert rec["estimated_cost_per_hour"] <= 5.0


class TestCheckFitEndpoint:
    """Test /api/simulator/check-fit endpoint."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_check_fit_success(self, mock_model_config):
        """Test fit check for LoRA on A100."""
        response = client.post(
            "/api/simulator/check-fit",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "batch_size": 4,
                "hardware": "A100_80GB_GPU",
                "num_gpus": 1,
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert data["fits"] is True
        assert "memory_per_gpu_gb" in data
        assert "utilization_percent" in data

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_check_fit_with_parallelism(self, mock_model_config):
        """Test fit check returns parallelism strategy."""
        response = client.post(
            "/api/simulator/check-fit",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "full",
                "batch_size": 4,
                "hardware": "A100_80GB_GPU",
                "num_gpus": 4,
            }
        )
        assert response.status_code == 200
        data = response.json()

        if data["fits"]:
            assert "parallelism" in data
            assert data["parallelism"] is not None


class TestTimeEstimateEndpoint:
    """Test /api/simulator/estimate-time endpoint."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_time_basic(self, mock_model_config):
        """Test basic time estimation."""
        response = client.post(
            "/api/simulator/estimate-time",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "dataset_tokens": 100_000_000,  # 100M tokens
                "batch_size": 4,
                "gradient_accumulation": 4,
                "epochs": 1.0,
                "hardware": "A100_80GB_GPU",
                "num_gpus": 1,
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert "total_steps" in data
        assert "tokens_per_second" in data
        assert "estimated_hours" in data
        assert "estimated_cost" in data

        assert data["total_steps"] > 0
        assert data["tokens_per_second"] > 0
        assert data["estimated_hours"] > 0
        assert data["estimated_cost"] > 0

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_estimate_time_multi_gpu(self, mock_model_config):
        """Test time estimation with multiple GPUs."""
        response_1 = client.post(
            "/api/simulator/estimate-time",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "dataset_tokens": 100_000_000,
                "batch_size": 4,
                "gradient_accumulation": 4,
                "epochs": 1.0,
                "hardware": "A100_80GB_GPU",
                "num_gpus": 1,
            }
        )
        response_4 = client.post(
            "/api/simulator/estimate-time",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "dataset_tokens": 100_000_000,
                "batch_size": 4,
                "gradient_accumulation": 4,
                "epochs": 1.0,
                "hardware": "A100_80GB_GPU",
                "num_gpus": 4,
            }
        )
        assert response_1.status_code == 200
        assert response_4.status_code == 200

        data_1 = response_1.json()
        data_4 = response_4.json()

        # 4 GPUs should be faster
        assert data_4["estimated_hours"] < data_1["estimated_hours"]


class TestHardwareListEndpoint:
    """Test /api/simulator/hardware endpoint."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_list_hardware(self):
        """Test hardware list endpoint."""
        response = client.get("/api/simulator/hardware")
        assert response.status_code == 200
        data = response.json()

        assert "hardware" in data
        assert "total_count" in data
        assert len(data["hardware"]) > 0

        # Check hardware structure
        hw = data["hardware"][0]
        assert "name" in hw
        assert "type" in hw
        assert "memory_gb" in hw
        assert "flops_tflops" in hw
        assert "cost_per_hour" in hw

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_hardware_includes_common_gpus(self):
        """Test that common GPUs are included."""
        response = client.get("/api/simulator/hardware")
        assert response.status_code == 200
        data = response.json()

        hardware_names = [hw["name"] for hw in data["hardware"]]

        # Check for common hardware
        common_hardware = ["A100_80GB_GPU", "H100_GPU", "V100_32GB_GPU"]
        for hw in common_hardware:
            assert hw in hardware_names, f"{hw} not found in hardware list"


class TestErrorHandling:
    """Test error handling in training API."""

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_invalid_precision(self, mock_model_config):
        """Test error for invalid precision."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "precision": "invalid_precision",
            }
        )
        assert response.status_code == 422

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_invalid_optimizer(self, mock_model_config):
        """Test error for invalid optimizer."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "optimizer": "invalid_optimizer",
            }
        )
        assert response.status_code == 422

    @pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
    def test_negative_batch_size(self, mock_model_config):
        """Test error for negative batch size."""
        response = client.post(
            "/api/simulator/estimate-training",
            json={
                "model": "meta-llama/Llama-3.1-8B",
                "method": "lora",
                "batch_size": -1,
            }
        )
        assert response.status_code == 422
