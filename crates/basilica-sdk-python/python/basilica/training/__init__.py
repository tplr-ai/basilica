"""
Basilica Training SDK

Fine-tune LLMs with LoRA on Basilica's GPU cloud.

Quick Start:
    >>> from basilica.training import Client
    >>>
    >>> client = Client()
    >>> with client.training("meta-llama/Llama-3.1-8B-Instruct", rank=32) as session:
    ...     loss = session.forward_backward([{"input_ids": [1, 2, 3]}])
    ...     session.optim_step()
    ...     print(session.sample("Hello!"))

Loss Functions:
    >>> # Standard cross-entropy (default)
    >>> loss = session.forward_backward(data)
    >>>
    >>> # Importance sampling (policy gradient)
    >>> loss = session.forward_backward(data, loss_fn="importance_sampling")
    >>>
    >>> # PPO with clipping
    >>> loss = session.forward_backward(data, loss_fn="ppo")
    >>>
    >>> # DPO (Direct Preference Optimization)
    >>> loss = session.forward_backward(data, loss_fn="dpo")

RL Training:
    >>> data = [{"input_ids": tokens, "loss_inputs": {"old_logprobs": lp, "rewards": r}}]
    >>> loss = session.forward_backward(data, loss_fn="importance_sampling")

Authentication:
    export BASILICA_API_TOKEN="basilica_..."
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx

__version__ = "0.1.0"


@dataclass
class Datum:
    """
    Training example.

    Attributes:
        input_ids: Token IDs for the input sequence
        labels: Target token IDs (defaults to input_ids for causal LM)
        weights: Per-token loss weights
        loss_inputs: Additional inputs for loss functions (e.g., rewards, old_logprobs)

    Example:
        >>> # Simple supervised learning
        >>> Datum(input_ids=[1, 2, 3])

        >>> # With custom loss weights
        >>> Datum(input_ids=[1, 2, 3], weights=[0.0, 1.0, 1.0])

        >>> # For RL (importance sampling)
        >>> Datum(input_ids=[1, 2, 3], loss_inputs={"old_logprobs": [-1.2, -0.8], "rewards": [1.0]})
    """
    input_ids: List[int]
    labels: Optional[List[int]] = None
    weights: Optional[List[float]] = None
    loss_inputs: Optional[Dict[str, Any]] = None


class TrainingError(Exception):
    """Training operation failed."""
    pass


class TrainingSession:
    """
    Training session for fine-tuning LLMs with LoRA.

    Created via Client.training(). All operations are synchronous
    and route through the Basilica API.

    Example:
        >>> session = client.training("meta-llama/Llama-3.1-8B-Instruct")
        >>>
        >>> for batch in dataloader:
        ...     loss = session.forward_backward(batch)
        ...     session.optim_step()
        >>>
        >>> print(session.sample("Hello!"))
    """

    def __init__(
        self,
        client: httpx.Client,
        session_id: str,
        internal_id: str,
        model: str,
    ):
        self._client = client
        self._session_id = session_id
        self._internal_id = internal_id
        self._model = model
        self._step = 0

    @property
    def id(self) -> str:
        """Session ID."""
        return self._session_id

    @property
    def model(self) -> str:
        """Base model name."""
        return self._model

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    def _proxy(self, op: str = "") -> str:
        """Build proxy path."""
        base = f"/sessions/{self._session_id}/internal/{self._internal_id}"
        return f"{base}/{op}" if op else base

    def forward_backward(
        self,
        data: Union[List[Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
    ) -> float:
        """
        Compute forward pass and gradients.

        Args:
            data: List of training examples (Datum objects or dicts with
                  'input_ids' and 'labels' keys)
            loss_fn: Loss function to use. Options:
                - "cross_entropy": Standard NLL loss (default)
                - "importance_sampling": Policy gradient with importance weighting
                - "ppo": Proximal Policy Optimization with clipping
                - "dpo": Direct Preference Optimization

        Returns:
            Loss value

        Example:
            >>> # Standard supervised learning
            >>> loss = session.forward_backward([{"input_ids": [1, 2, 3]}])

            >>> # RL with importance sampling
            >>> loss = session.forward_backward(data, loss_fn="importance_sampling")
        """
        # Normalize input
        examples = []
        for d in data:
            if isinstance(d, Datum):
                examples.append(d)
            else:
                examples.append(Datum(
                    input_ids=d["input_ids"],
                    labels=d.get("labels"),
                    weights=d.get("weights"),
                    loss_inputs=d.get("loss_inputs"),
                ))

        # Default labels to input_ids (causal LM)
        for ex in examples:
            if ex.labels is None:
                ex.labels = ex.input_ids.copy()

        # Pad sequences
        max_len = max(len(ex.input_ids) for ex in examples)
        input_ids = [ex.input_ids + [0] * (max_len - len(ex.input_ids)) for ex in examples]
        labels = [ex.labels + [-100] * (max_len - len(ex.labels)) for ex in examples]
        attention_mask = [[1] * len(ex.input_ids) + [0] * (max_len - len(ex.input_ids)) for ex in examples]

        weights = None
        if examples[0].weights:
            weights = [ex.weights + [0.0] * (max_len - len(ex.weights)) for ex in examples]

        # Collect loss function inputs
        loss_inputs = None
        if examples[0].loss_inputs:
            loss_inputs = [ex.loss_inputs for ex in examples]

        resp = self._client.post(
            self._proxy("forward_backward"),
            json={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_weights": weights,
                "loss_fn": loss_fn,
                "loss_inputs": loss_inputs,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"forward_backward failed: {resp.text}")

        return resp.json()["loss"]

    def optim_step(self) -> int:
        """
        Apply gradients and update weights.

        Returns:
            Current step count
        """
        resp = self._client.post(self._proxy("optim_step"))

        if not resp.is_success:
            raise TrainingError(f"optim_step failed: {resp.text}")

        self._step = resp.json()["step"]
        return self._step

    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text

        Example:
            >>> print(session.sample("Once upon a time"))
        """
        resp = self._client.post(
            self._proxy("sample"),
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"sample failed: {resp.text}")

        return resp.json()["text"]

    def save(self, name: str) -> str:
        """
        Save checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Checkpoint path
        """
        resp = self._client.post(
            self._proxy("save"),
            json={"checkpoint_name": name, "include_optimizer": True},
        )

        if not resp.is_success:
            raise TrainingError(f"save failed: {resp.text}")

        return resp.json()["checkpoint_path"]

    def load(self, path: str) -> None:
        """
        Load checkpoint.

        Args:
            path: Checkpoint path
        """
        resp = self._client.post(
            self._proxy("load"),
            json={"checkpoint_path": path, "load_optimizer": True},
        )

        if not resp.is_success:
            raise TrainingError(f"load failed: {resp.text}")

    def status(self) -> Dict[str, Any]:
        """Get session status."""
        resp = self._client.get(self._proxy())

        if not resp.is_success:
            raise TrainingError(f"status failed: {resp.text}")

        return resp.json()

    def close(self) -> None:
        """Delete the session."""
        resp = self._client.delete(f"/sessions/{self._session_id}")
        if resp.status_code != 404 and not resp.is_success:
            raise TrainingError(f"close failed: {resp.text}")

    def __enter__(self) -> "TrainingSession":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class Client:
    """
    Basilica API client.

    Example:
        >>> import basilica
        >>>
        >>> client = basilica.Client()
        >>> session = client.training("meta-llama/Llama-3.1-8B-Instruct")
        >>>
        >>> loss = session.forward_backward(data)
        >>> session.optim_step()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize client.

        Args:
            api_key: API key (default: BASILICA_API_TOKEN env var)
            endpoint: API endpoint (default: BASILICA_API_URL or https://api.basilica.ai)
            timeout: Request timeout in seconds
        """
        self._api_key = api_key or os.environ.get("BASILICA_API_TOKEN")
        self._endpoint = endpoint or os.environ.get("BASILICA_API_URL", "https://api.basilica.ai")

        if not self._api_key:
            raise ValueError("API key required. Set BASILICA_API_TOKEN or pass api_key=")

        self._client = httpx.Client(
            base_url=self._endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout,
        )

    def training(
        self,
        model: str,
        rank: int = 32,
        alpha: int = 64,
        lr: float = 1e-4,
        dropout: float = 0.05,
        gpu_count: int = 1,
        gpu_type: Optional[List[str]] = None,
        seed: Optional[int] = None,
        wait: float = 300.0,
    ) -> TrainingSession:
        """
        Create a training session.

        Args:
            model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha (default: 64)
            lr: Learning rate (default: 1e-4)
            dropout: Dropout rate (default: 0.05)
            gpu_count: Number of GPUs (0 for CPU)
            gpu_type: Acceptable GPU types (e.g., ["H100", "A100"])
            seed: Random seed
            wait: Seconds to wait for session ready

        Returns:
            TrainingSession

        Example:
            >>> session = client.training(
            ...     "meta-llama/Llama-3.1-8B-Instruct",
            ...     rank=64,
            ...     lr=2e-4,
            ... )
        """
        # Create K8s session
        resp = self._client.post(
            "/sessions",
            json={
                "baseModel": model,
                "checkpointStorage": {"backend": "r2", "bucket": "", "path": ""},
                "loraConfig": {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "optimizerConfig": {
                    "learningRate": lr,
                    "weightDecay": 0.01,
                    "gradClip": 1.0,
                },
                "gpuResources": {"count": gpu_count, "model": gpu_type or []},
                "seed": seed,
                "ttlSeconds": 86400,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"Failed to create session: {resp.text}")

        session_id = resp.json()["sessionId"]

        # Wait for ready
        start = time.time()
        while time.time() - start < wait:
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
            raise TrainingError(f"Session not ready after {wait}s")

        # Create internal session
        internal_id = f"train-{session_id}"
        resp = self._client.post(
            f"/sessions/{session_id}/internal",
            json={
                "session_id": internal_id,
                "base_model": model,
                "lora_config": {"rank": rank, "alpha": alpha, "dropout": dropout},
                "optimizer_config": {"learning_rate": lr, "weight_decay": 0.01},
                "seed": seed,
            },
        )

        if not resp.is_success:
            raise TrainingError(f"Failed to create internal session: {resp.text}")

        return TrainingSession(self._client, session_id, internal_id, model)

    def get_session(self, session_id: str) -> TrainingSession:
        """
        Get existing session.

        Args:
            session_id: Session ID

        Returns:
            TrainingSession
        """
        resp = self._client.get(f"/sessions/{session_id}")

        if resp.status_code == 404:
            raise TrainingError(f"Session {session_id} not found")
        if not resp.is_success:
            raise TrainingError(f"Failed to get session: {resp.text}")

        data = resp.json()
        if data.get("phase") != "ready":
            raise TrainingError(f"Session not ready: {data.get('phase')}")

        return TrainingSession(
            self._client,
            session_id,
            f"train-{session_id}",
            data.get("baseModel", "unknown"),
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        resp = self._client.get("/sessions")

        if not resp.is_success:
            raise TrainingError(f"Failed to list sessions: {resp.text}")

        return resp.json()

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# Convenience alias
TrainingClient = Client

# Export all public symbols
__all__ = [
    "Client",
    "TrainingClient",
    "TrainingSession",
    "TrainingError",
    "Datum",
]
