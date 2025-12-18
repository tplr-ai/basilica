"""
Basilica Training SDK - Training client.

This module provides the TrainingClient for training operations.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from .types import (
    APIFuture,
    Datum,
    ForwardBackwardResult,
    ForwardResult,
)
from .sampling_client import SamplingClient
from .exceptions import TrainingError


class TrainingClient:
    """Client for training operations.

    Example:
        >>> training = client.create_lora_training_client("meta-llama/Llama-3.1-8B")
        >>> for batch in dataloader:
        ...     result = training.forward_backward(batch).result()
        ...     training.optim_step().result()
        >>> training.save_state("checkpoint-final").result()
    """

    def __init__(
        self,
        client: httpx.Client,
        session_id: str,
        internal_id: str,
        base_model: str,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ):
        """Initialize training client.

        Args:
            client: HTTP client for API requests
            session_id: K8s session ID
            internal_id: Internal training session ID
            base_model: Base model name
            train_mlp: Whether MLP layers have LoRA
            train_attn: Whether attention layers have LoRA
            train_unembed: Whether unembedding layer has LoRA
        """
        self._client = client
        self._session_id = session_id
        self._internal_id = internal_id
        self._base_model = base_model
        self._train_mlp = train_mlp
        self._train_attn = train_attn
        self._train_unembed = train_unembed
        self._step = 0
        self._executor = ThreadPoolExecutor(max_workers=4)

    @property
    def session_id(self) -> str:
        """K8s session ID."""
        return self._session_id

    @property
    def base_model(self) -> str:
        """Base model name."""
        return self._base_model

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    def _proxy(self, op: str = "") -> str:
        """Build proxy path to training service."""
        base = f"/sessions/{self._session_id}/internal/{self._internal_id}"
        return f"{base}/{op}" if op else base

    def _normalize_data(self, data: Union[List[Datum], List[Dict]]) -> List[Dict]:
        """Convert data to list of dicts."""
        result = []
        for d in data:
            if isinstance(d, Datum):
                result.append(d.to_dict())
            else:
                result.append(d)
        return result

    # --- Training Operations ---

    def forward(self, data: List[Datum]) -> APIFuture:
        """Forward pass without gradient computation.

        Args:
            data: Training examples

        Returns:
            APIFuture resolving to ForwardResult
        """

        def _call():
            normalized = self._normalize_data(data)

            # Pad sequences
            max_len = max(len(d["input_ids"]) for d in normalized)
            input_ids = []
            attention_mask = []

            for d in normalized:
                ids = d["input_ids"]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

            resp = self._client.post(
                self._proxy("forward"),
                json={"input_ids": input_ids, "attention_mask": attention_mask},
            )
            if not resp.is_success:
                raise TrainingError(f"forward failed: {resp.text}")
            r = resp.json()
            return ForwardResult(
                logprobs=r["logprobs"], tokens_processed=r["tokens_processed"]
            )

        return APIFuture(self._executor.submit(_call), ForwardResult)

    def forward_backward(
        self,
        data: Union[List[Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
    ) -> APIFuture:
        """Compute forward pass and gradients.

        Args:
            data: Training examples (Datum objects or dicts with input_ids)
            loss_fn: Loss function ("cross_entropy", "importance_sampling", "ppo", "dpo")

        Returns:
            APIFuture resolving to ForwardBackwardResult

        Example:
            >>> data = [Datum(input_ids=[1, 2, 3])]
            >>> result = training.forward_backward(data).result()
            >>> print(f"Loss: {result.loss:.4f}")
        """

        def _call():
            normalized = self._normalize_data(data)

            # Default labels to input_ids (causal LM)
            for d in normalized:
                if "labels" not in d or d["labels"] is None:
                    d["labels"] = d["input_ids"].copy()

            # Pad sequences
            max_len = max(len(d["input_ids"]) for d in normalized)
            input_ids = []
            labels = []
            attention_mask = []
            loss_weights = []

            for d in normalized:
                ids = d["input_ids"]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                labels.append(d["labels"] + [-100] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                if d.get("loss_weights"):
                    loss_weights.append(d["loss_weights"] + [0.0] * pad_len)

            payload = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_fn": loss_fn,
            }
            if loss_weights:
                payload["loss_weights"] = loss_weights

            resp = self._client.post(self._proxy("forward_backward"), json=payload)
            if not resp.is_success:
                raise TrainingError(f"forward_backward failed: {resp.text}")

            r = resp.json()
            return ForwardBackwardResult(
                loss=r["loss"],
                logprobs=r.get("logprobs", []),
                tokens_processed=r.get("tokens_processed", 0),
            )

        return APIFuture(self._executor.submit(_call), ForwardBackwardResult)

    def forward_backward_custom(
        self,
        data: List[Datum],
        loss_fn: Callable,
    ) -> APIFuture:
        """Compute gradients with custom loss function.

        The custom loss function receives logprobs and should return a scalar loss.

        Args:
            data: Training examples
            loss_fn: Custom loss function operating on logprobs

        Note:
            This requires server-side support for custom loss computation.
        """
        raise NotImplementedError(
            "forward_backward_custom requires server-side support"
        )

    def optim_step(
        self,
        learning_rate: Optional[float] = None,
        betas: Optional[tuple] = None,
        eps: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> APIFuture:
        """Update weights using accumulated gradients (Adam).

        Args:
            learning_rate: Override learning rate
            betas: Adam beta parameters (beta1, beta2)
            eps: Adam epsilon
            weight_decay: L2 regularization

        Returns:
            APIFuture resolving to current step count
        """

        def _call():
            payload = {}
            if learning_rate is not None:
                payload["learning_rate"] = learning_rate
            if betas is not None:
                payload["beta1"], payload["beta2"] = betas
            if eps is not None:
                payload["eps"] = eps
            if weight_decay is not None:
                payload["weight_decay"] = weight_decay

            resp = self._client.post(
                self._proxy("optim_step"), json=payload if payload else None
            )
            if not resp.is_success:
                raise TrainingError(f"optim_step failed: {resp.text}")

            self._step = resp.json()["step"]
            return self._step

        return APIFuture(self._executor.submit(_call), int)

    # --- State Management ---

    def save_state(self, name: str) -> APIFuture:
        """Save checkpoint (weights + optimizer state).

        Args:
            name: Checkpoint name

        Returns:
            APIFuture resolving to checkpoint path
        """

        def _call():
            resp = self._client.post(
                self._proxy("save"),
                json={"checkpoint_name": name, "include_optimizer": True},
            )
            if not resp.is_success:
                raise TrainingError(f"save_state failed: {resp.text}")
            return resp.json()["checkpoint_path"]

        return APIFuture(self._executor.submit(_call), str)

    def load_state(self, path: str) -> APIFuture:
        """Load weights only (optimizer resets).

        Args:
            path: Checkpoint path

        Returns:
            APIFuture resolving when load completes
        """

        def _call():
            resp = self._client.post(
                self._proxy("load"),
                json={"checkpoint_path": path, "load_optimizer": False},
            )
            if not resp.is_success:
                raise TrainingError(f"load_state failed: {resp.text}")

        return APIFuture(self._executor.submit(_call))

    def load_state_with_optimizer(self, path: str) -> APIFuture:
        """Load weights and optimizer state.

        Args:
            path: Checkpoint path

        Returns:
            APIFuture resolving when load completes
        """

        def _call():
            resp = self._client.post(
                self._proxy("load"),
                json={"checkpoint_path": path, "load_optimizer": True},
            )
            if not resp.is_success:
                raise TrainingError(f"load_state_with_optimizer failed: {resp.text}")

        return APIFuture(self._executor.submit(_call))

    def save_weights_for_sampler(self, name: str) -> APIFuture:
        """Export weights formatted for sampling.

        Args:
            name: Export name

        Returns:
            APIFuture resolving to export path
        """

        def _call():
            resp = self._client.post(
                self._proxy("save_for_sampler"),
                json={"name": name},
            )
            if not resp.is_success:
                raise TrainingError(f"save_weights_for_sampler failed: {resp.text}")
            return resp.json()["path"]

        return APIFuture(self._executor.submit(_call), str)

    def save_weights_and_get_sampling_client(self, name: str) -> SamplingClient:
        """Save weights and return SamplingClient.

        Args:
            name: Export name

        Returns:
            SamplingClient configured with exported weights
        """
        path = self.save_weights_for_sampler(name).result()
        return SamplingClient(
            client=self._client,
            model_path=path,
            session_id=self._session_id,
            internal_id=self._internal_id,
        )

    # --- Utilities ---

    def get_tokenizer(self):
        """Get the model's tokenizer.

        Returns a HuggingFace tokenizer for encoding/decoding.
        Requires `transformers` package.
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self._base_model)

    def get_info(self) -> Dict[str, Any]:
        """Get session configuration.

        Returns:
            Dict with session metadata
        """
        resp = self._client.get(self._proxy())
        if not resp.is_success:
            raise TrainingError(f"get_info failed: {resp.text}")

        data = resp.json()
        return {
            "session_id": self._session_id,
            "internal_id": self._internal_id,
            "base_model": self._base_model,
            "train_mlp": self._train_mlp,
            "train_attn": self._train_attn,
            "train_unembed": self._train_unembed,
            "step": data.get("step_count", self._step),
            "tokens_processed": data.get("tokens_processed", 0),
        }

    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate text completion (convenience method).

        For full control, use save_weights_and_get_sampling_client().

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
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

    def close(self):
        """Delete the training session."""
        resp = self._client.delete(f"/sessions/{self._session_id}")
        if resp.status_code != 404 and not resp.is_success:
            raise TrainingError(f"close failed: {resp.text}")
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Async Variants ---

    async def forward_async(self, data: List[Datum]) -> ForwardResult:
        """Forward pass (async)."""
        return await self.forward(data).result_async()

    async def forward_backward_async(
        self, data: List[Datum], loss_fn: str = "cross_entropy"
    ) -> ForwardBackwardResult:
        """Compute gradients (async)."""
        return await self.forward_backward(data, loss_fn).result_async()

    async def optim_step_async(self, **kwargs) -> int:
        """Update weights (async)."""
        return await self.optim_step(**kwargs).result_async()

    async def save_state_async(self, name: str) -> str:
        """Save checkpoint (async)."""
        return await self.save_state(name).result_async()

    async def load_state_async(self, path: str):
        """Load weights (async)."""
        return await self.load_state(path).result_async()


# === Export ===

__all__ = ["TrainingClient"]
