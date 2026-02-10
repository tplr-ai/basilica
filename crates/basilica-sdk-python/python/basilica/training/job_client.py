"""
Basilica Training SDK - Job management client.

Provides high-level APIs for:
- Creating and cancelling training jobs
- Querying job status and metrics
- Using pre-configured training templates
- Managing checkpoint metadata endpoints
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

from .exceptions import TrainingError
from .types import APIFuture


class JobClient:
    """Client for the `/training` job management API."""

    def __init__(self, client: httpx.Client, base_path: str = "/training"):
        self._client = client
        self._base_path = base_path.rstrip("/")
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _url(self, path: str) -> str:
        return f"{self._base_path}/{path.lstrip('/')}"

    # --- Jobs ---

    def create_job(
        self,
        name: str,
        config: Dict[str, Any],
        wandb_api_key: Optional[str] = None,
    ) -> APIFuture:
        def _call():
            payload: Dict[str, Any] = {"name": name, "config": config}
            if wandb_api_key is not None:
                payload["wandb_api_key"] = wandb_api_key
            resp = self._client.post(self._url("/jobs"), json=payload)
            if not resp.is_success:
                raise TrainingError(f"create_job failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> APIFuture:
        def _call():
            params: Dict[str, Any] = {}
            if status is not None:
                params["status"] = status
            if limit is not None:
                params["limit"] = limit
            if offset is not None:
                params["offset"] = offset
            resp = self._client.get(self._url("/jobs"), params=params or None)
            if not resp.is_success:
                raise TrainingError(f"list_jobs failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), list)

    def get_job(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.get(self._url(f"/jobs/{job_id}"))
            if not resp.is_success:
                raise TrainingError(f"get_job failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def cancel_job(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.delete(self._url(f"/jobs/{job_id}"))
            if not resp.is_success:
                raise TrainingError(f"cancel_job failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def get_job_metrics(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.get(self._url(f"/jobs/{job_id}/metrics"))
            if not resp.is_success:
                raise TrainingError(f"get_job_metrics failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    # --- Templates ---

    def list_job_templates(self) -> APIFuture:
        def _call():
            resp = self._client.get(self._url("/job_templates"))
            if not resp.is_success:
                raise TrainingError(f"list_job_templates failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), list)

    def get_job_template(self, template_id: str) -> APIFuture:
        def _call():
            resp = self._client.get(self._url(f"/job_templates/{template_id}"))
            if not resp.is_success:
                raise TrainingError(f"get_job_template failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def create_job_from_template(
        self,
        template_id: str,
        name: str,
        dataset_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> APIFuture:
        def _call():
            payload: Dict[str, Any] = {"name": name}
            if dataset_path is not None:
                payload["dataset_path"] = dataset_path
            if model_name is not None:
                payload["model_name"] = model_name
            resp = self._client.post(
                self._url(f"/job_templates/{template_id}/jobs"),
                json=payload,
            )
            if not resp.is_success:
                raise TrainingError(f"create_job_from_template failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    # --- Checkpoints ---

    def get_checkpoint(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.get(self._url(f"/checkpoints/{job_id}"))
            if not resp.is_success:
                raise TrainingError(f"get_checkpoint failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def get_checkpoint_download_url(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.get(self._url(f"/checkpoints/{job_id}/download_url"))
            if not resp.is_success:
                raise TrainingError(f"get_checkpoint_download_url failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def publish_checkpoint(self, job_id: str, make_public: bool = True) -> APIFuture:
        def _call():
            resp = self._client.post(
                self._url(f"/checkpoints/{job_id}/publish"),
                json={"make_public": make_public},
            )
            if not resp.is_success:
                raise TrainingError(f"publish_checkpoint failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def delete_checkpoint(self, job_id: str) -> APIFuture:
        def _call():
            resp = self._client.delete(self._url(f"/checkpoints/{job_id}"))
            if not resp.is_success:
                raise TrainingError(f"delete_checkpoint failed: {resp.text}")
            return resp.json()

        return APIFuture(self._executor.submit(_call), dict)

    def close(self):
        self._executor.shutdown(wait=False)


