"""
Basilica Training SDK - REST client for checkpoint management.

This module provides the RestClient for managing training runs and checkpoints.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

from .types import APIFuture
from .exceptions import TrainingError


class RestClient:
    """REST client for checkpoint and run management.

    Example:
        >>> rest = client.create_rest_client()
        >>> runs = rest.list_training_runs().result()
        >>> checkpoints = rest.list_checkpoints(run_id="ts-abc123").result()
        >>> url = rest.get_checkpoint_archive_url("cp-xyz").result()
    """

    def __init__(self, client: httpx.Client):
        """Initialize REST client.

        Args:
            client: HTTP client for API requests
        """
        self._client = client
        self._executor = ThreadPoolExecutor(max_workers=4)

    # --- Training Runs ---

    def list_training_runs(self, limit: int = 20, offset: int = 0) -> APIFuture:
        """List training runs (paginated).

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            APIFuture resolving to List[Dict] of training runs
        """

        def _call():
            resp = self._client.get(
                "/training_runs", params={"limit": limit, "offset": offset}
            )
            if not resp.is_success:
                raise TrainingError(f"list_training_runs failed: {resp.text}")
            return resp.json().get("runs", [])

        return APIFuture(self._executor.submit(_call), list)

    def get_training_run(self, run_id: str) -> APIFuture:
        """Get training run by ID.

        Args:
            run_id: Training run ID

        Returns:
            APIFuture resolving to Dict with run details
        """

        def _call():
            resp = self._client.get(f"/training_runs/{run_id}")
            if not resp.is_success:
                raise TrainingError(f"get_training_run failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def get_training_run_by_path(self, path: str) -> APIFuture:
        """Get training run by checkpoint path.

        Args:
            path: Checkpoint path

        Returns:
            APIFuture resolving to Dict with run details
        """

        def _call():
            resp = self._client.get("/training_runs/by_path", params={"path": path})
            if not resp.is_success:
                raise TrainingError(f"get_training_run_by_path failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    # --- Checkpoints ---

    def list_checkpoints(
        self, run_id: Optional[str] = None, limit: int = 100
    ) -> APIFuture:
        """List checkpoints.

        Args:
            run_id: Filter by training run ID (optional)
            limit: Maximum number of checkpoints to return

        Returns:
            APIFuture resolving to List[Dict] of checkpoints
        """

        def _call():
            params: Dict[str, Any] = {"limit": limit}
            if run_id:
                params["run_id"] = run_id
            resp = self._client.get("/checkpoints", params=params)
            if not resp.is_success:
                raise TrainingError(f"list_checkpoints failed: {resp.text}")
            return resp.json().get("checkpoints", [])

        return APIFuture(self._executor.submit(_call), list)

    def get_checkpoint_archive_url(self, checkpoint_id: str) -> APIFuture:
        """Get signed download URL for checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            APIFuture resolving to presigned URL string
        """

        def _call():
            resp = self._client.get(f"/checkpoints/{checkpoint_id}/download_url")
            if not resp.is_success:
                raise TrainingError(f"get_checkpoint_archive_url failed: {resp.text}")
            return resp.json()["url"]

        return APIFuture(self._executor.submit(_call), str)

    def delete_checkpoint(self, checkpoint_id: str) -> APIFuture:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            APIFuture resolving when deletion completes
        """

        def _call():
            resp = self._client.delete(f"/checkpoints/{checkpoint_id}")
            if not resp.is_success:
                raise TrainingError(f"delete_checkpoint failed: {resp.text}")

        return APIFuture(self._executor.submit(_call))

    def get_weights_info_by_path(self, path: str) -> APIFuture:
        """Get checkpoint metadata by path.

        Args:
            path: Checkpoint path

        Returns:
            APIFuture resolving to Dict with checkpoint info
        """

        def _call():
            resp = self._client.get("/checkpoints/info", params={"path": path})
            if not resp.is_success:
                raise TrainingError(f"get_weights_info_by_path failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    # --- Publishing ---

    def publish_checkpoint(self, path: str) -> APIFuture:
        """Make checkpoint publicly accessible.

        Args:
            path: Checkpoint path to publish

        Returns:
            APIFuture resolving to public URL string
        """

        def _call():
            resp = self._client.post("/checkpoints/publish", json={"path": path})
            if not resp.is_success:
                raise TrainingError(f"publish_checkpoint failed: {resp.text}")
            return resp.json()["public_url"]

        return APIFuture(self._executor.submit(_call), str)

    def unpublish_checkpoint(self, path: str) -> APIFuture:
        """Revert checkpoint to private.

        Args:
            path: Checkpoint path to unpublish

        Returns:
            APIFuture resolving when unpublish completes
        """

        def _call():
            resp = self._client.post("/checkpoints/unpublish", json={"path": path})
            if not resp.is_success:
                raise TrainingError(f"unpublish_checkpoint failed: {resp.text}")

        return APIFuture(self._executor.submit(_call))

    # --- Sessions ---

    def list_sessions(self, limit: int = 20) -> APIFuture:
        """List active training sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            APIFuture resolving to List[Dict] of sessions
        """

        def _call():
            resp = self._client.get("/sessions", params={"limit": limit})
            if not resp.is_success:
                raise TrainingError(f"list_sessions failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), list)

    def get_session(self, session_id: str) -> APIFuture:
        """Get session details.

        Args:
            session_id: Session ID

        Returns:
            APIFuture resolving to Dict with session details
        """

        def _call():
            resp = self._client.get(f"/sessions/{session_id}")
            if not resp.is_success:
                raise TrainingError(f"get_session failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    # --- Async Variants ---

    async def list_training_runs_async(
        self, limit: int = 20, offset: int = 0
    ) -> List[Dict]:
        """List training runs (async)."""
        return await self.list_training_runs(limit, offset).result_async()

    async def get_training_run_async(self, run_id: str) -> Dict:
        """Get training run (async)."""
        return await self.get_training_run(run_id).result_async()

    async def list_checkpoints_async(
        self, run_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """List checkpoints (async)."""
        return await self.list_checkpoints(run_id, limit).result_async()

    async def get_checkpoint_archive_url_async(self, checkpoint_id: str) -> str:
        """Get checkpoint URL (async)."""
        return await self.get_checkpoint_archive_url(checkpoint_id).result_async()

    async def delete_checkpoint_async(self, checkpoint_id: str):
        """Delete checkpoint (async)."""
        return await self.delete_checkpoint(checkpoint_id).result_async()

    async def publish_checkpoint_async(self, path: str) -> str:
        """Publish checkpoint (async)."""
        return await self.publish_checkpoint(path).result_async()

    async def unpublish_checkpoint_async(self, path: str):
        """Unpublish checkpoint (async)."""
        return await self.unpublish_checkpoint(path).result_async()

    async def list_sessions_async(self, limit: int = 20) -> List[Dict]:
        """List sessions (async)."""
        return await self.list_sessions(limit).result_async()

    async def get_session_async(self, session_id: str) -> Dict:
        """Get session (async)."""
        return await self.get_session(session_id).result_async()

    def close(self):
        """Close the REST client."""
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# === Export ===

__all__ = ["RestClient"]
