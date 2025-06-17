import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# It's good practice to ensure all necessary modules can be found
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.model_manager import ModelManager
from src.db.connection import DatabaseConnection
from src.db.hf_integration import HuggingFaceModelImporter

# Sample data for testing
SAMPLE_MODEL_ID = "test/test-model-v1"
SAMPLE_CONFIG = {"key": "value"}
SAMPLE_LOGO_PATH = "logos/test_model.png"
SAMPLE_ANALYSIS = {
    "description": "A test model.",
    "advantages": ["It works."],
    "disadvantages": ["It's fictional."],
    "usecases": ["Testing."],
    "evals": [{"name": "test_metric", "score": 1.0}]
}

# Mock HTML content for the scraper test
MOCK_HTML_CONTENT = """
<html>
<body>
    <div class="main">
        <header>
            <div>
                <div class="group">
                    <img src="/path/to/logo.png" />
                </div>
            </div>
        </header>
    </div>
</body>
</html>
"""

@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def db_connection(temp_db_dir):
    """Fixture to create a test database."""
    # Reset singleton instance
    DatabaseConnection.reset_instance()
    
    # Create a new database in temp directory
    db_path = os.path.join(temp_db_dir, "test_models.db")
    conn = DatabaseConnection(db_path=db_path)
    
    yield conn
    
    # Reset singleton after test
    DatabaseConnection.reset_instance()

@pytest.fixture
def model_manager(db_connection):
    """Fixture to create a ModelManager with the test database."""
    return ModelManager(db_connection)

@pytest.fixture
def hf_importer(model_manager):
    """Fixture for the HuggingFaceModelImporter."""
    # We pass a model_manager to satisfy the constructor
    return HuggingFaceModelImporter(model_manager=model_manager)

def test_add_and_get_model_with_extended_fields(model_manager):
    """
    Tests that a model can be added with the new logo and analysis fields
    and that these fields are correctly retrieved.
    """
    # Act: Add a model with the new fields
    model_manager.add_model(
        model_id=SAMPLE_MODEL_ID,
        config=SAMPLE_CONFIG,
        logo=SAMPLE_LOGO_PATH,
        model_analysis=SAMPLE_ANALYSIS
    )

    # Assert: Retrieve the model and verify the new fields
    model = model_manager.get_model(SAMPLE_MODEL_ID)

    assert model is not None
    assert model['logo'] == SAMPLE_LOGO_PATH
    assert model['model_analysis'] is not None
    
    # The analysis is stored as a JSON string, so we parse it back
    retrieved_analysis = json.loads(model['model_analysis'])
    assert retrieved_analysis == SAMPLE_ANALYSIS 

@patch('requests.get')
def test_scrap_hf_logo_success(mock_requests_get, hf_importer):
    """Tests that the logo scraper correctly parses HTML."""
    # Arrange: Mock the requests call to return our sample HTML
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = MOCK_HTML_CONTENT.encode('utf-8')
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    # Act
    logo_url = hf_importer.scrap_hf_logo("some/model")

    # Assert
    assert logo_url == "https://huggingface.co/path/to/logo.png"
    mock_requests_get.assert_called_once()

@patch('requests.get')
@patch('builtins.open', new_callable=MagicMock)
def test_save_logo(mock_open, mock_requests_get, hf_importer, tmp_path):
    """Tests that the _save_logo method downloads and saves a file."""
    # Arrange
    # Point the logos directory to a temporary path for this test
    hf_importer.logos_dir = tmp_path

    mock_image_data = b'imagedata'
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [mock_image_data]
    mock_requests_get.return_value = mock_response

    # Act
    local_path = hf_importer._save_logo("test/model", "http://example.com/logo.png")

    # Assert
    expected_filename = "test_model.png"
    expected_path = tmp_path / expected_filename
    
    assert local_path == f"logos/{expected_filename}"
    mock_open.assert_called_with(expected_path, 'wb')
    mock_open().__enter__().write.assert_called_with(mock_image_data) 