"""Pytest configuration for budevolve tests."""
import pytest


def pytest_configure(config):
    """Register custom markers for budevolve tests."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with: -m 'not integration')")
    config.addinivalue_line("markers", "slow: marks tests as slow-running (deselect with: -m 'not slow')")
