"""
Test model analysis integration with LLM.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from BudSimulator.src.db.connection import DatabaseConnection
from BudSimulator.src.db.model_manager import ModelManager
from BudSimulator.src.db.hf_integration import HuggingFaceModelImporter
from BudSimulator.src.utils.text_extraction import extract_text_from_huggingface
from BudSimulator.src.utils.llm_integration import parse_model_analysis, validate_analysis, get_empty_analysis
from BudSimulator.src.utils.bud_llm import call_bud_LLM


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_models.db"
    
    # Create connection and schema
    conn = DatabaseConnection(str(db_path))
    conn.create_schema()
    
    yield conn
    
    # Cleanup
    conn.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_manager(temp_db):
    """Create a model manager with temporary database."""
    return ModelManager(temp_db)


@pytest.fixture
def hf_importer(model_manager):
    """Create HuggingFace importer with mocked dependencies."""
    return HuggingFaceModelImporter(model_manager)


class TestLLMIntegration:
    """Test LLM integration utilities."""
    
    def test_parse_model_analysis_with_tags(self):
        """Test parsing model analysis from LLM response with JSON tags."""
        llm_response = """
        Here's the analysis:
        <json>
        {
            "TestModel": {
                "model_analysis": {
                    "description": "Test model description",
                    "advantages": ["Advantage 1", "Advantage 2"],
                    "disadvantages": ["Disadvantage 1"],
                    "usecases": ["Use case 1", "Use case 2"],
                    "evals": [
                        {"name": "MMLU", "score": 75.5},
                        {"name": "GSM8K", "score": 82.3}
                    ]
                }
            }
        }
        </json>
        """
        
        analysis = parse_model_analysis(llm_response)
        
        assert analysis['description'] == "Test model description"
        assert len(analysis['advantages']) == 2
        assert len(analysis['disadvantages']) == 1
        assert len(analysis['usecases']) == 2
        assert len(analysis['evals']) == 2
        assert analysis['evals'][0]['name'] == "MMLU"
        assert analysis['evals'][0]['score'] == 75.5
    
    def test_parse_model_analysis_direct_format(self):
        """Test parsing model analysis in direct format."""
        llm_response = """
        <json>
        {
            "description": "Direct format description",
            "advantages": ["Direct advantage"],
            "disadvantages": [],
            "usecases": ["Direct use case"],
            "evals": []
        }
        </json>
        """
        
        analysis = parse_model_analysis(llm_response)
        
        assert analysis['description'] == "Direct format description"
        assert len(analysis['advantages']) == 1
        assert len(analysis['disadvantages']) == 0
    
    def test_parse_model_analysis_invalid(self):
        """Test parsing invalid model analysis."""
        llm_response = "This is not JSON"
        
        analysis = parse_model_analysis(llm_response)
        
        # Should return empty analysis
        assert analysis == get_empty_analysis()
    
    def test_validate_analysis(self):
        """Test analysis validation."""
        raw_analysis = {
            "description": "Valid description",
            "advantages": ["Adv 1", None, "Adv 2"],  # Contains None
            "disadvantages": "Not a list",  # Wrong type
            "usecases": ["Use 1"],
            "evals": [
                {"name": "Test", "score": 90},
                {"name": "Invalid"},  # Missing score
                {"name": "String Score", "score": "80.5"}  # String score
            ]
        }
        
        validated = validate_analysis(raw_analysis)
        
        assert validated['description'] == "Valid description"
        assert validated['advantages'] == ["Adv 1", "Adv 2"]  # None removed
        assert validated['disadvantages'] == []  # Invalid type replaced
        assert len(validated['evals']) == 2  # Invalid eval removed
        assert validated['evals'][1]['score'] == 80.5  # String converted to float


class TestModelAnalysisIntegration:
    """Test full model analysis integration."""
    
    @patch('BudSimulator.src.db.hf_integration.extract_text_from_huggingface')
    @patch('BudSimulator.src.db.hf_integration.call_bud_LLM')
    def test_analyze_model_with_llm(self, mock_llm, mock_extract, hf_importer):
        """Test analyzing a model with LLM."""
        # Mock text extraction
        mock_extract.return_value = "Model description from HuggingFace"
        
        # Mock LLM response
        mock_llm.return_value = """
        <json>
        {
            "model": {
                "model_analysis": {
                    "description": "Analyzed model description",
                    "advantages": ["Fast inference", "Low memory usage"],
                    "disadvantages": ["Limited context length"],
                    "usecases": ["Text generation", "Code completion"],
                    "evals": [{"name": "Benchmark", "score": 85.0}]
                }
            }
        }
        </json>
        """
        
        # Test the analysis
        analysis = hf_importer._analyze_model_with_llm("test/model")
        
        assert analysis is not None
        assert analysis['description'] == "Analyzed model description"
        assert len(analysis['advantages']) == 2
        assert "Fast inference" in analysis['advantages']
        assert len(analysis['evals']) == 1
        
        # Verify calls
        mock_extract.assert_called_once_with("https://huggingface.co/test/model")
        assert mock_llm.called
    
    @patch('BudSimulator.src.db.hf_integration.HuggingFaceConfigLoader')
    @patch('BudSimulator.src.db.hf_integration.extract_text_from_huggingface')
    @patch('BudSimulator.src.db.hf_integration.call_bud_LLM')
    def test_import_model_with_analysis(self, mock_llm, mock_extract, mock_loader, hf_importer):
        """Test importing a model with LLM analysis."""
        # Mock HuggingFace config loader
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.get_model_config.return_value = {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'vocab_size': 50257
        }
        mock_loader_instance.get_model_info.return_value = MagicMock(cardData={})
        
        # Mock text extraction
        mock_extract.return_value = "Model card content"
        
        # Mock LLM response
        mock_llm.return_value = """
        <json>
        {
            "model": {
                "model_analysis": {
                    "description": "Test model for integration",
                    "advantages": ["Good performance"],
                    "disadvantages": ["High memory usage"],
                    "usecases": ["Testing"],
                    "evals": []
                }
            }
        }
        </json>
        """
        
        # Import model
        success = hf_importer.import_model("test/integration-model")
        
        assert success
        
        # Check if model was added with analysis
        model = hf_importer.model_manager.get_model("test/integration-model")
        assert model is not None
        
        # Check if analysis was saved
        if model.get('model_analysis'):
            analysis = json.loads(model['model_analysis'])
            assert analysis['description'] == "Test model for integration"
            assert len(analysis['advantages']) == 1


class TestTextExtraction:
    """Test text extraction from HuggingFace."""
    
    @patch('requests.Session')
    def test_extract_text_success(self, mock_session):
        """Test successful text extraction."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <body>
                <div class="model-card-content">
                    <h1>Model Title</h1>
                    <p>This is a test model.</p>
                    <ul>
                        <li>Feature 1</li>
                        <li>Feature 2</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        
        # Test extraction
        text = extract_text_from_huggingface("https://huggingface.co/test/model")
        
        assert "Model Title" in text
        assert "test model" in text
        assert "Feature 1" in text
    
    @patch('requests.Session')
    def test_extract_text_error(self, mock_session):
        """Test text extraction with error."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        
        # Test extraction
        text = extract_text_from_huggingface("https://huggingface.co/nonexistent")
        
        assert "Error" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 