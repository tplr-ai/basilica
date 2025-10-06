"""
State persistence for Basilica AFINE SDK.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List

import cloudpickle


class StatePersistence:
    """Handles state persistence for Service instances."""

    def __init__(self, state_dir: Path = Path("/app/state")) -> None:
        """
        Initialize state persistence.

        Args:
            state_dir: Directory for storing state files
        """
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, service_instance: Any, name: str = "service_state") -> None:
        """
        Save service instance state.

        Args:
            service_instance: The service instance to save
            name: State file name

        Warnings:
            Warns if state contains unpicklable objects
        """
        state_file = self._state_dir / f"{name}.pkl"

        try:
            with state_file.open('wb') as f:
                cloudpickle.dump(service_instance.__dict__, f)
        except Exception as e:
            warnings.warn(
                f"Failed to save state: {e}. "
                "State may contain unpicklable objects (GPU tensors, file handles, locks, etc.). "
                "Implement __getstate__/__setstate__ to handle custom serialization.",
                RuntimeWarning
            )
            raise

    def load_state(self, name: str = "service_state") -> Dict[str, Any]:
        """
        Load service instance state.

        Args:
            name: State file name

        Returns:
            State dictionary (empty if no saved state)
        """
        state_file = self._state_dir / f"{name}.pkl"
        if not state_file.exists():
            return {}

        try:
            with state_file.open('rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            warnings.warn(
                f"Failed to load state: {e}. Starting with fresh state.",
                RuntimeWarning
            )
            return {}

    def list_checkpoints(self) -> List[str]:
        """
        List available state checkpoints.

        Returns:
            List of checkpoint names
        """
        return [f.stem for f in self._state_dir.glob("*.pkl")]
