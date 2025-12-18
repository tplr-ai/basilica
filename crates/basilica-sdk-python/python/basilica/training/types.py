"""
Basilica Training SDK - Type definitions.

This module contains all data types used by the training SDK.
"""

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import asyncio


# === Training Data Types ===


@dataclass
class Datum:
    """Training example.

    Attributes:
        input_ids: Token IDs for the input sequence
        labels: Target token IDs (defaults to input_ids for causal LM)
        loss_weights: Per-token loss weights

    Example:
        >>> # Simple supervised learning
        >>> Datum(input_ids=[1, 2, 3])

        >>> # With custom loss weights
        >>> Datum(input_ids=[1, 2, 3], loss_weights=[0.0, 1.0, 1.0])
    """

    input_ids: List[int]
    labels: Optional[List[int]] = None
    loss_weights: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {"input_ids": self.input_ids}
        if self.labels is not None:
            d["labels"] = self.labels
        if self.loss_weights is not None:
            d["loss_weights"] = self.loss_weights
        return d


@dataclass
class ModelInput:
    """Input tokens for sampling.

    Example:
        >>> prompt = ModelInput.from_ints([1, 2, 3])
        >>> prompt = ModelInput.from_string("Hello!", tokenizer)
    """

    token_ids: List[int]

    @classmethod
    def from_ints(cls, token_ids: List[int]) -> "ModelInput":
        """Create from token ID list."""
        return cls(token_ids=token_ids)

    @classmethod
    def from_string(cls, text: str, tokenizer) -> "ModelInput":
        """Create from text string using tokenizer."""
        return cls(token_ids=tokenizer.encode(text))


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Example:
        >>> params = SamplingParams(max_tokens=100, temperature=0.7)
    """

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: Optional[List[str]] = None
    include_logprobs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences or [],
            "include_logprobs": self.include_logprobs,
        }


# === Response Types ===


@dataclass
class SampleResponse:
    """Generated sample from the model."""

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"


@dataclass
class ForwardBackwardResult:
    """Result of forward-backward pass."""

    loss: float
    logprobs: List[List[float]]
    tokens_processed: int


@dataclass
class ForwardResult:
    """Result of forward-only pass."""

    logprobs: List[List[float]]
    tokens_processed: int


@dataclass
class GetServerCapabilitiesResponse:
    """Server capabilities response."""

    models: List[str]
    max_batch_tokens: int
    max_sequence_length: int


# === Async Support ===


class APIFuture:
    """Async handle for training operations.

    Supports both sync and async access patterns:
        # Sync
        result = future.result(timeout=30)

        # Async
        result = await future.result_async(timeout=30)

        # Or just await directly
        result = await future

    Example:
        >>> future = training.forward_backward(data)
        >>> result = future.result()  # Blocking
        >>> # Or
        >>> result = await future.result_async()  # Non-blocking
    """

    def __init__(self, future: Future, result_type: type = None):
        self._future = future
        self._result = None
        self._result_type = result_type

    def result(self, timeout: Optional[float] = None):
        """Block until operation completes (sync)."""
        if self._result is None:
            self._result = self._future.result(timeout=timeout)
        return self._result

    async def result_async(self, timeout: Optional[float] = None):
        """Wait for operation to complete (async)."""
        loop = asyncio.get_event_loop()
        if timeout is not None:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._future.result), timeout=timeout
            )
        return await loop.run_in_executor(None, self._future.result)

    def done(self) -> bool:
        """Check if operation is complete."""
        return self._future.done()

    def __await__(self):
        """Allow: result = await future"""
        return self.result_async().__await__()


# === Export ===

__all__ = [
    "Datum",
    "ModelInput",
    "SamplingParams",
    "SampleResponse",
    "ForwardBackwardResult",
    "ForwardResult",
    "GetServerCapabilitiesResponse",
    "APIFuture",
]
