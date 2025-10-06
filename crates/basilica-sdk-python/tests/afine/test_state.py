"""
Unit tests for StatePersistence.
"""

import tempfile
from pathlib import Path

import pytest

from basilica.afine import StatePersistence


class TestObject:
    """Simple test object for state persistence."""

    def __init__(self):
        self.value = 42
        self.name = "test"


def test_state_persistence_save_and_load():
    """Test saving and loading state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = StatePersistence(state_dir=Path(tmpdir))

        obj = TestObject()
        obj.value = 100

        persistence.save_state(obj, name="test")

        loaded_state = persistence.load_state(name="test")
        assert loaded_state["value"] == 100
        assert loaded_state["name"] == "test"


def test_state_persistence_load_nonexistent():
    """Test loading nonexistent state returns empty dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = StatePersistence(state_dir=Path(tmpdir))

        state = persistence.load_state(name="nonexistent")
        assert state == {}


def test_state_persistence_list_checkpoints():
    """Test listing checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = StatePersistence(state_dir=Path(tmpdir))

        obj = TestObject()

        persistence.save_state(obj, name="checkpoint_1")
        persistence.save_state(obj, name="checkpoint_2")

        checkpoints = persistence.list_checkpoints()
        assert "checkpoint_1" in checkpoints
        assert "checkpoint_2" in checkpoints
        assert len(checkpoints) == 2


def test_state_persistence_invalid_object():
    """Test that invalid objects raise warnings and exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = StatePersistence(state_dir=Path(tmpdir))

        class UnpicklableObject:
            """Object with unpicklable attribute."""

            def __init__(self):
                import threading
                self.lock = threading.Lock()

        obj = UnpicklableObject()

        with pytest.warns(RuntimeWarning):
            with pytest.raises(Exception):
                persistence.save_state(obj, name="invalid")
