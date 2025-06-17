"""Tests for BudSimulator setup verification."""

import os
import sys
import sqlite3
from pathlib import Path
import pytest


class TestSetupVerification:
    """Verify BudSimulator setup components."""
    
    def test_python_version(self):
        """Test Python version requirement."""
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
        
    def test_database_exists(self):
        """Test that database is properly set up."""
        db_path = Path.home() / ".genz_simulator" / "db" / "models.db"
        assert db_path.exists(), "Database file not found"
        
        # Check database has models
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM models")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0, "Database has no models"
        
    def test_prepopulated_database(self):
        """Test that prepopulated database exists."""
        prepop_db = Path(__file__).parent.parent / "data" / "prepopulated.db"
        assert prepop_db.exists(), "Prepopulated database not found"
        
    def test_requirements_file(self):
        """Test that requirements.txt exists."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"
        
    def test_frontend_directory(self):
        """Test that frontend directory exists."""
        frontend_dir = Path(__file__).parent.parent / "frontend"
        assert frontend_dir.exists(), "Frontend directory not found"
        
    def test_api_endpoints(self):
        """Test that API module can be imported."""
        try:
            from apis import main
            assert hasattr(main, 'app'), "FastAPI app not found"
        except ImportError:
            pytest.fail("Cannot import API module")
            
    def test_setup_script(self):
        """Test that setup script exists and is valid."""
        setup_script = Path(__file__).parent.parent / "setup.py"
        assert setup_script.exists(), "setup.py not found"
        
        # Check it's a valid Python file
        with open(setup_script, 'r') as f:
            content = f.read()
            assert "BudSimulatorSetup" in content, "Setup class not found"
            assert "def run(self)" in content, "Run method not found" 