"""
Basilica Training SDK

High-level Python SDK for training LLMs with LoRA on Basilica's GPU cloud.

Quick Start:
    >>> from basilica.training import TrainingClient
    >>> client = TrainingClient()
    >>>
    >>> # Create a training session
    >>> session = client.create_session(
    ...     base_model="meta-llama/Llama-3.1-8B-Instruct",
    ...     rank=32,
    ...     learning_rate=1e-4,
    ... )
    >>>
    >>> # Training loop
    >>> for batch in dataloader:
    ...     result = session.forward_backward(batch)
    ...     print(f"Loss: {result.loss:.4f}")
    ...     session.optim_step()
    >>>
    >>> # Save checkpoint
    >>> session.save_state("checkpoint-final")

Authentication:
    Set the BASILICA_API_TOKEN environment variable:
        export BASILICA_API_TOKEN="basilica_..."

    Or pass directly:
        client = TrainingClient(api_key="basilica_...")
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

__version__ = "0.1.0"


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: Optional[float] = 1.0


@dataclass
class CheckpointStorage:
    """Checkpoint storage configuration."""

    backend: str = "r2"
    bucket: str = ""
    path: str = ""
    credentials_secret: Optional[str] = None


@dataclass
class GpuResources:
    """GPU resource requirements."""

    count: int = 1
    model: List[str] = field(default_factory=list)
    min_memory_gb: Optional[int] = None


@dataclass
class Datum:
    """Training example."""

    input_ids: List[int]
    labels: List[int]
    loss_weights: Optional[List[float]] = None


@dataclass
class ForwardBackwardResult:
    """Result of forward-backward pass."""

    loss: float
    logprobs: List[List[float]]
    tokens_processed: int


@dataclass
class Sample:
    """Generated sample."""

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"


class TrainingError(Exception):
    """Base exception for training errors."""

    pass


class SessionNotFoundError(TrainingError):
    """Session not found."""

    pass


class TrainingServiceError(TrainingError):
    """Error from training service."""

    pass


class TrainingSession:
    """
    A training session for fine-tuning LLMs with LoRA.

    Do not instantiate directly. Use TrainingClient.create_session().

    Example:
        >>> session = client.create_session(
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct",
        ...     rank=32,
        ... )
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     result = session.forward_backward(batch)
        ...     session.optim_step()
        >>>
        >>> # Generate sample
        >>> sample = session.sample("Hello!", max_tokens=50)
        >>> print(sample.text)
    """

    def __init__(
        self,
        session_id: str,
        client: httpx.Client,
        base_model: str,
    ):
        self._session_id = session_id
        self._client = client
        self._base_model = base_model
        self._step_count = 0

    @property
    def session_id(self) -> str:
        """Session ID."""
        return self._session_id

    @property
    def base_model(self) -> str:
        """Base model name."""
        return self._base_model

    @property
    def step_count(self) -> int:
        """Current training step count."""
        return self._step_count

    def forward_backward(
        self,
        data: List[Datum],
        loss_fn: str = "cross_entropy",
    ) -> ForwardBackwardResult:
        """
        Compute forward pass and gradients.

        Args:
            data: List of training examples
            loss_fn: Loss function (currently only "cross_entropy")

        Returns:
            ForwardBackwardResult with loss and logprobs
        """
        # Batch the data
        input_ids = [d.input_ids for d in data]
        labels = [d.labels for d in data]
        loss_weights = (
            [d.loss_weights for d in data] if data[0].loss_weights else None
        )

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
        padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels]
        attention_mask = [
            [1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids
        ]

        padded_loss_weights = None
        if loss_weights:
            padded_loss_weights = [
                w + [0.0] * (max_len - len(w)) for w in loss_weights
            ]

        response = self._client.post(
            f"/sessions/{self._session_id}/forward_backward",
            json={
                "inputIds": padded_input_ids,
                "attentionMask": attention_mask,
                "labels": padded_labels,
                "lossWeights": padded_loss_weights,
            },
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Forward-backward failed: {response.text}")

        data = response.json()
        return ForwardBackwardResult(
            loss=data["loss"],
            logprobs=data["logprobs"],
            tokens_processed=data["tokensProcessed"],
        )

    def optim_step(self) -> int:
        """
        Apply gradients and update weights.

        Returns:
            Current step count
        """
        response = self._client.post(
            f"/sessions/{self._session_id}/optim_step",
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Optim step failed: {response.text}")

        data = response.json()
        self._step_count = data["step"]
        return self._step_count

    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        include_logprobs: bool = False,
    ) -> Sample:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (0 = disabled)
            include_logprobs: Whether to return log probabilities

        Returns:
            Generated sample
        """
        response = self._client.post(
            f"/sessions/{self._session_id}/sample",
            json={
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "includeLogprobs": include_logprobs,
            },
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Sample failed: {response.text}")

        data = response.json()
        return Sample(
            text=data["text"],
            token_ids=data["tokenIds"],
            logprobs=data.get("logprobs"),
            finish_reason=data["finishReason"],
        )

    def save_state(
        self,
        checkpoint_name: str,
        include_optimizer: bool = True,
    ) -> str:
        """
        Save checkpoint.

        Args:
            checkpoint_name: Name for the checkpoint
            include_optimizer: Whether to save optimizer state

        Returns:
            Checkpoint path
        """
        response = self._client.post(
            f"/sessions/{self._session_id}/save",
            json={
                "checkpointName": checkpoint_name,
                "includeOptimizer": include_optimizer,
            },
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Save failed: {response.text}")

        data = response.json()
        return data["checkpointPath"]

    def load_state(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
    ) -> None:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        response = self._client.post(
            f"/sessions/{self._session_id}/load",
            json={
                "checkpointPath": checkpoint_path,
                "loadOptimizer": load_optimizer,
            },
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Load failed: {response.text}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get session status.

        Returns:
            Status dictionary
        """
        response = self._client.get(
            f"/sessions/{self._session_id}",
        )

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {self._session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Get status failed: {response.text}")

        return response.json()

    def close(self) -> None:
        """Delete the training session."""
        response = self._client.delete(
            f"/sessions/{self._session_id}",
        )
        # Ignore 404 - session may already be deleted
        if response.status_code != 404 and not response.is_success:
            raise TrainingServiceError(f"Delete failed: {response.text}")


class TrainingClient:
    """
    Client for training LLMs on Basilica's GPU cloud.

    Example:
        >>> from basilica.training import TrainingClient
        >>> client = TrainingClient()
        >>>
        >>> session = client.create_session(
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct",
        ...     rank=32,
        ...     learning_rate=1e-4,
        ... )
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     result = session.forward_backward(batch)
        ...     print(f"Loss: {result.loss:.4f}")
        ...     session.optim_step()
        >>>
        >>> session.save_state("checkpoint-final")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the training client.

        Args:
            api_key: API key for authentication.
                    Defaults to BASILICA_API_TOKEN env var.
            endpoint: API endpoint URL.
                     Defaults to BASILICA_API_URL env var or https://api.basilica.ai
            timeout: Request timeout in seconds

        Raises:
            ValueError: If no API key is provided
        """
        self._api_key = api_key or os.environ.get("BASILICA_API_TOKEN")
        self._endpoint = endpoint or os.environ.get(
            "BASILICA_API_URL", "https://api.basilica.ai"
        )
        self._timeout = timeout

        if not self._api_key:
            raise ValueError(
                "API key required. Set BASILICA_API_TOKEN env var or pass api_key parameter."
            )

        self._client = httpx.Client(
            base_url=self._endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout,
        )

    def create_session(
        self,
        base_model: str,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: Optional[float] = 1.0,
        gpu_count: int = 1,
        gpu_models: Optional[List[str]] = None,
        seed: Optional[int] = None,
        ttl_seconds: int = 86400,
        checkpoint_bucket: str = "",
        checkpoint_path: str = "",
    ) -> TrainingSession:
        """
        Create a new training session with LoRA.

        Args:
            base_model: HuggingFace model ID
                       (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha scaling factor (default: 64)
            dropout: LoRA dropout rate (default: 0.05)
            target_modules: Modules to apply LoRA to
            learning_rate: Initial learning rate (default: 1e-4)
            weight_decay: Weight decay (default: 0.01)
            grad_clip: Gradient clipping (default: 1.0)
            gpu_count: Number of GPUs (default: 1)
            gpu_models: Acceptable GPU models (e.g., ["H100", "A100"])
            seed: Random seed for reproducibility
            ttl_seconds: Session TTL in seconds (default: 24 hours)
            checkpoint_bucket: Storage bucket for checkpoints
            checkpoint_path: Path prefix for checkpoints

        Returns:
            TrainingSession for the new session

        Example:
            >>> session = client.create_session(
            ...     base_model="meta-llama/Llama-3.1-8B-Instruct",
            ...     rank=64,
            ...     alpha=128,
            ...     learning_rate=2e-4,
            ...     gpu_count=1,
            ...     gpu_models=["H100"],
            ... )
        """
        response = self._client.post(
            "/sessions",
            json={
                "baseModel": base_model,
                "checkpointStorage": {
                    "backend": "r2",
                    "bucket": checkpoint_bucket,
                    "path": checkpoint_path,
                },
                "loraConfig": {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "targetModules": target_modules
                    or ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "optimizerConfig": {
                    "learningRate": learning_rate,
                    "weightDecay": weight_decay,
                    "gradClip": grad_clip,
                },
                "gpuResources": {
                    "count": gpu_count,
                    "model": gpu_models or [],
                },
                "seed": seed,
                "ttlSeconds": ttl_seconds,
            },
        )

        if not response.is_success:
            raise TrainingServiceError(f"Failed to create session: {response.text}")

        data = response.json()
        return TrainingSession(
            session_id=data["sessionId"],
            client=self._client,
            base_model=base_model,
        )

    def get_session(self, session_id: str) -> TrainingSession:
        """
        Get an existing training session.

        Args:
            session_id: Session ID

        Returns:
            TrainingSession for the existing session

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        response = self._client.get(f"/sessions/{session_id}")

        if response.status_code == 404:
            raise SessionNotFoundError(f"Session {session_id} not found")
        if not response.is_success:
            raise TrainingServiceError(f"Failed to get session: {response.text}")

        data = response.json()
        return TrainingSession(
            session_id=session_id,
            client=self._client,
            base_model=data.get("baseModel", "unknown"),
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all training sessions.

        Returns:
            List of session status dictionaries
        """
        response = self._client.get("/sessions")

        if not response.is_success:
            raise TrainingServiceError(f"Failed to list sessions: {response.text}")

        return response.json()

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self) -> "TrainingClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# Export all public symbols
__all__ = [
    # Client
    "TrainingClient",
    "TrainingSession",
    # Config types
    "LoraConfig",
    "OptimizerConfig",
    "CheckpointStorage",
    "GpuResources",
    # Data types
    "Datum",
    "ForwardBackwardResult",
    "Sample",
    # Exceptions
    "TrainingError",
    "SessionNotFoundError",
    "TrainingServiceError",
]
