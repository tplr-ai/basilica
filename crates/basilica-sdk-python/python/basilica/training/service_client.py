"""
Basilica Training SDK - Service client.

This module provides the ServiceClient, the main entry point for the SDK.
"""

import os
import time
from typing import Dict, List, Optional

import httpx

from .types import GetServerCapabilitiesResponse
from .training_client import TrainingClient
from .sampling_client import SamplingClient
from .rest_client import RestClient
from .exceptions import (
    TrainingError,
    SessionTimeoutError,
    AuthenticationError,
    ValidationError,
)


class ServiceClient:
    """Main entry point for Basilica Training API.

    Example:
        >>> client = ServiceClient()
        >>> training = client.create_lora_training_client(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     rank=32,
        ... )
        >>> result = training.forward_backward(data).result()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize client.

        Args:
            api_key: API key (default: BASILICA_API_KEY or BASILICA_API_TOKEN env var)
            endpoint: API endpoint (default: BASILICA_ENDPOINT or BASILICA_API_URL)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If no API key provided
        """
        self.api_key = (
            api_key
            or os.environ.get("BASILICA_API_KEY")
            or os.environ.get("BASILICA_API_TOKEN")
        )
        self.endpoint = (
            endpoint
            or os.environ.get("BASILICA_ENDPOINT")
            or os.environ.get("BASILICA_API_URL", "https://api.basilica.ai")
        )

        if not self.api_key:
            raise ValueError(
                "API key required. Set BASILICA_API_KEY or pass api_key="
            )

        self._client = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        self._async_client = None  # Lazy init

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        """Query available models and limits.

        Returns:
            GetServerCapabilitiesResponse with models and limits
        """
        resp = self._client.get("/capabilities")
        resp.raise_for_status()
        return GetServerCapabilitiesResponse(**resp.json())

    async def get_server_capabilities_async(self) -> GetServerCapabilitiesResponse:
        """Query capabilities (async)."""
        client = await self._get_async_client()
        resp = await client.get("/capabilities")
        resp.raise_for_status()
        return GetServerCapabilitiesResponse(**resp.json())

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.05,
        seed: Optional[int] = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        user_metadata: Optional[Dict[str, str]] = None,
        gpu_count: int = 1,
        gpu_type: Optional[List[str]] = None,
        wait_timeout: float = 300.0,
    ) -> TrainingClient:
        """Create LoRA fine-tuning session.

        Args:
            base_model: HuggingFace model ID
            rank: LoRA rank (default 32)
            alpha: LoRA alpha (default 2 * rank)
            dropout: LoRA dropout (default 0.05)
            seed: Random seed for reproducibility
            train_mlp: Apply LoRA to MLP layers
            train_attn: Apply LoRA to attention layers
            train_unembed: Apply LoRA to unembedding layer
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization
            user_metadata: Custom metadata for tracking
            gpu_count: Number of GPUs (0 for CPU)
            gpu_type: Acceptable GPU types
            wait_timeout: Seconds to wait for session ready

        Returns:
            TrainingClient for training operations

        Raises:
            ValidationError: If configuration is invalid
            SessionTimeoutError: If session doesn't become ready
            TrainingError: If session creation fails

        Example:
            >>> training = client.create_lora_training_client(
            ...     "meta-llama/Llama-3.1-8B-Instruct",
            ...     rank=64,
            ...     train_mlp=True,
            ...     train_attn=True,
            ...     train_unembed=False,
            ... )
        """
        # Build target modules from flags
        target_modules = []
        if train_attn:
            target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
        if train_mlp:
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        if train_unembed:
            target_modules.append("lm_head")

        if not target_modules:
            raise ValidationError(
                "At least one of train_mlp, train_attn, train_unembed must be True"
            )

        # Default alpha to 2 * rank
        if alpha is None:
            alpha = rank * 2

        # Create K8s session via Basilica API
        resp = self._client.post(
            "/sessions",
            json={
                "baseModel": base_model,
                "checkpointStorage": {"backend": "r2", "bucket": "", "path": ""},
                "loraConfig": {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "targetModules": target_modules,
                    "trainMlp": train_mlp,
                    "trainAttn": train_attn,
                    "trainUnembed": train_unembed,
                },
                "optimizerConfig": {
                    "learningRate": learning_rate,
                    "weightDecay": weight_decay,
                    "gradClip": 1.0,
                },
                "gpuResources": {"count": gpu_count, "model": gpu_type or []},
                "seed": seed,
                "userMetadata": user_metadata or {},
                "ttlSeconds": 86400,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"Failed to create session: {resp.text}")

        session_id = resp.json()["sessionId"]

        # Wait for ready
        start = time.time()
        while time.time() - start < wait_timeout:
            resp = self._client.get(f"/sessions/{session_id}")
            if not resp.is_success:
                raise TrainingError(f"Failed to get session: {resp.text}")

            data = resp.json()
            phase = data.get("phase", "pending")

            if phase == "ready":
                break
            elif phase == "failed":
                raise TrainingError(f"Session failed: {data.get('error')}")

            time.sleep(5)
        else:
            raise SessionTimeoutError(session_id, wait_timeout)

        # Create internal training session
        internal_id = f"train-{session_id}"
        resp = self._client.post(
            f"/sessions/{session_id}/internal",
            json={
                "session_id": internal_id,
                "base_model": base_model,
                "lora_config": {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "target_modules": target_modules,
                },
                "optimizer_config": {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                "seed": seed,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"Failed to create internal session: {resp.text}")

        return TrainingClient(
            client=self._client,
            session_id=session_id,
            internal_id=internal_id,
            base_model=base_model,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )

    def create_training_client_from_state(
        self,
        path: str,
        user_metadata: Optional[Dict[str, str]] = None,
    ) -> TrainingClient:
        """Resume training from checkpoint (weights only, optimizer resets).

        Args:
            path: Checkpoint path
            user_metadata: Custom metadata for tracking

        Returns:
            TrainingClient for continued training
        """
        resp = self._client.post(
            "/sessions/from_state",
            json={
                "path": path,
                "userMetadata": user_metadata or {},
                "loadOptimizer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return TrainingClient(
            client=self._client,
            session_id=data["sessionId"],
            internal_id=data["internalId"],
            base_model=data["baseModel"],
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        user_metadata: Optional[Dict[str, str]] = None,
    ) -> TrainingClient:
        """Resume training from checkpoint (weights + optimizer state).

        Args:
            path: Checkpoint path
            user_metadata: Custom metadata for tracking

        Returns:
            TrainingClient for continued training
        """
        resp = self._client.post(
            "/sessions/from_state",
            json={
                "path": path,
                "userMetadata": user_metadata or {},
                "loadOptimizer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return TrainingClient(
            client=self._client,
            session_id=data["sessionId"],
            internal_id=data["internalId"],
            base_model=data["baseModel"],
        )

    def create_sampling_client(
        self,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> SamplingClient:
        """Create client for text generation.

        Args:
            model_path: Path to fine-tuned weights
            base_model: Base model name (if no fine-tuned weights)

        Returns:
            SamplingClient for text generation

        Raises:
            ValueError: If neither model_path nor base_model specified
        """
        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model required")

        return SamplingClient(
            client=self._client,
            model_path=model_path,
            base_model=base_model,
        )

    def create_rest_client(self) -> RestClient:
        """Create REST client for checkpoint and run management.

        Returns:
            RestClient for managing training runs and checkpoints

        Example:
            >>> rest = client.create_rest_client()
            >>> runs = rest.list_training_runs().result()
            >>> checkpoints = rest.list_checkpoints(run_id="ts-abc").result()
            >>> url = rest.get_checkpoint_archive_url("cp-xyz").result()
        """
        return RestClient(client=self._client)

    def get_session(self, session_id: str) -> TrainingClient:
        """Get existing training session.

        Args:
            session_id: Session ID

        Returns:
            TrainingClient for the session

        Raises:
            TrainingError: If session not found or not ready
        """
        resp = self._client.get(f"/sessions/{session_id}")

        if resp.status_code == 404:
            raise TrainingError(f"Session {session_id} not found")
        if not resp.is_success:
            raise TrainingError(f"Failed to get session: {resp.text}")

        data = resp.json()
        if data.get("phase") != "ready":
            raise TrainingError(f"Session not ready: {data.get('phase')}")

        return TrainingClient(
            client=self._client,
            session_id=session_id,
            internal_id=f"train-{session_id}",
            base_model=data.get("baseModel", "unknown"),
        )

    def list_sessions(self) -> List[Dict]:
        """List all sessions.

        Returns:
            List of session status dicts
        """
        resp = self._client.get("/sessions")

        if not resp.is_success:
            raise TrainingError(f"Failed to list sessions: {resp.text}")

        return resp.json()

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=300.0,
            )
        return self._async_client

    def close(self):
        """Close the client."""
        self._client.close()
        if self._async_client:
            # Note: async client should be closed in async context
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# === Export ===

__all__ = ["ServiceClient"]
