"""
Pytest configuration for AFINE tests.
"""

import pytest


@pytest.fixture
def mock_rental_secret(monkeypatch):
    """Set BASILICA_RENTAL_SECRET for tests."""
    monkeypatch.setenv("BASILICA_RENTAL_SECRET", "test-secret")


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set BASILICA_API_KEY for tests."""
    monkeypatch.setenv("BASILICA_API_KEY", "test-api-key")
