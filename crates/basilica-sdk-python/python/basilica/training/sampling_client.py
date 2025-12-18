"""
Basilica Training SDK - Sampling client.

This module provides the SamplingClient for text generation.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import httpx

from .types import APIFuture, ModelInput, SampleResponse, SamplingParams
from .exceptions import TrainingError


class SamplingClient:
    """Client for text generation.

    Example:
        >>> sampling = client.create_sampling_client(base_model="Qwen/Qwen3-8B")
        >>> prompt = ModelInput.from_ints([1, 2, 3])
        >>> result = sampling.sample(prompt, SamplingParams(max_tokens=100)).result()
        >>> print(result[0].text)
    """

    def __init__(
        self,
        client: httpx.Client,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
        session_id: Optional[str] = None,
        internal_id: Optional[str] = None,
    ):
        """Initialize sampling client.

        Args:
            client: HTTP client for API requests
            model_path: Path to fine-tuned weights
            base_model: Base model name (if no fine-tuned weights)
            session_id: K8s session ID (for session-bound sampling)
            internal_id: Internal training session ID
        """
        self._client = client
        self._model_path = model_path
        self._base_model = base_model
        self._session_id = session_id
        self._internal_id = internal_id
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_endpoint(self) -> str:
        """Get the sampling endpoint."""
        if self._session_id and self._internal_id:
            return f"/sessions/{self._session_id}/internal/{self._internal_id}/sample"
        return "/sample"

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: Optional[SamplingParams] = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: Optional[int] = None,
    ) -> APIFuture:
        """Generate text completions.

        Args:
            prompt: Input tokens
            sampling_params: Generation parameters
            num_samples: Number of independent samples
            include_prompt_logprobs: Include logprobs for prompt tokens
            topk_prompt_logprobs: Top-k logprobs per position

        Returns:
            APIFuture resolving to List[SampleResponse]

        Example:
            >>> prompt = ModelInput.from_ints([1, 2, 3])
            >>> results = sampling.sample(prompt, SamplingParams(max_tokens=50)).result()
            >>> print(results[0].text)
        """
        params = sampling_params or SamplingParams()

        def _call():
            payload = {
                "token_ids": prompt.token_ids,
                "num_samples": num_samples,
                "include_prompt_logprobs": include_prompt_logprobs,
                **params.to_dict(),
            }
            if topk_prompt_logprobs is not None:
                payload["topk_prompt_logprobs"] = topk_prompt_logprobs
            if self._model_path:
                payload["model_path"] = self._model_path
            if self._base_model:
                payload["base_model"] = self._base_model

            resp = self._client.post(self._get_endpoint(), json=payload)
            if not resp.is_success:
                raise TrainingError(f"sample failed: {resp.text}")

            data = resp.json()
            samples = data.get("samples", [data])  # Handle single vs batch
            return [SampleResponse(**s) for s in samples]

        return APIFuture(self._executor.submit(_call), list)

    def sample_text(
        self,
        text: str,
        tokenizer,
        sampling_params: Optional[SamplingParams] = None,
        num_samples: int = 1,
    ) -> APIFuture:
        """Generate completions from text prompt.

        Args:
            text: Input text string
            tokenizer: HuggingFace tokenizer for encoding
            sampling_params: Generation parameters
            num_samples: Number of independent samples

        Returns:
            APIFuture resolving to List[SampleResponse]
        """
        prompt = ModelInput.from_string(text, tokenizer)
        return self.sample(prompt, sampling_params, num_samples)

    def compute_logprobs(self, prompt: ModelInput) -> APIFuture:
        """Compute log probabilities for prompt tokens.

        Args:
            prompt: Input tokens

        Returns:
            APIFuture resolving to List[Optional[float]]
            First token has None (no prior context).
        """

        def _call():
            payload = {"token_ids": prompt.token_ids}
            if self._model_path:
                payload["model_path"] = self._model_path
            if self._base_model:
                payload["base_model"] = self._base_model

            # Determine endpoint
            if self._session_id and self._internal_id:
                endpoint = f"/sessions/{self._session_id}/internal/{self._internal_id}/compute_logprobs"
            else:
                endpoint = "/compute_logprobs"

            resp = self._client.post(endpoint, json=payload)
            if not resp.is_success:
                raise TrainingError(f"compute_logprobs failed: {resp.text}")

            return resp.json()["logprobs"]

        return APIFuture(self._executor.submit(_call), list)

    # === Async Variants ===

    async def sample_async(
        self,
        prompt: ModelInput,
        sampling_params: Optional[SamplingParams] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> List[SampleResponse]:
        """Generate completions (async)."""
        return await self.sample(prompt, sampling_params, num_samples, **kwargs).result_async()

    async def compute_logprobs_async(
        self, prompt: ModelInput
    ) -> List[Optional[float]]:
        """Compute logprobs (async)."""
        return await self.compute_logprobs(prompt).result_async()

    def close(self):
        """Close the sampling client."""
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# === Export ===

__all__ = ["SamplingClient"]
