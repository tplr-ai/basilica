#!/usr/bin/env python3
"""Basilica Training Example - Fine-tune LLMs with LoRA

This example demonstrates the Basilica Training SDK for fine-tuning LLMs.

Usage:
    python training_example.py

API Pattern:
    client = ServiceClient()
    training = client.create_lora_training_client(...)
    result = training.forward_backward(data).result()
    training.optim_step().result()
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "crates", "basilica-sdk-python", "python"),
)

from basilica.training import (
    ServiceClient,
    Datum,
    ModelInput,
    SamplingParams,
)


def main():
    """Main training example."""
    # Create client (auto-detects BASILICA_API_KEY environment variable)
    api_key = None
    if os.path.exists("build/api-token.txt"):
        api_key = open("build/api-token.txt").read().strip()

    # Use local endpoint if BASILICA_API_URL not set
    endpoint = os.environ.get("BASILICA_API_URL", "http://localhost:8000")

    client = ServiceClient(api_key=api_key, endpoint=endpoint)

    # Create training session with LoRA module selection
    with client.create_lora_training_client(
        base_model="facebook/opt-125m",
        rank=8,
        train_mlp=True,       # Apply LoRA to MLP layers
        train_attn=True,      # Apply LoRA to attention layers
        train_unembed=False,  # Don't apply LoRA to unembedding
        learning_rate=1e-4,
        gpu_count=0,          # CPU for testing
    ) as training:

        print(f"Session: {training.session_id}")
        print(f"Model: {training.base_model}")

        # Training loop
        # Using 10 tokens for stable gradients: "The quick brown fox jumps over the lazy dog"
        train_tokens = [2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]
        for _ in range(5):
            # forward_backward returns APIFuture - call .result() to get value
            result = training.forward_backward(
                [Datum(input_ids=train_tokens)],
                loss_fn="cross_entropy",
            ).result()

            # optim_step also returns APIFuture
            step = training.optim_step().result()

            print(f"Step {step}: loss={result.loss:.4f}")

        # Save checkpoint
        checkpoint_path = training.save_state("checkpoint-final").result()
        print(f"\nCheckpoint saved: {checkpoint_path}")

        # === Phase 2: Logprobs Demo ===
        print("\n--- Logprobs Demo ---")

        # Forward-only pass (no gradients)
        forward_result = training.forward(
            [Datum(input_ids=train_tokens)]
        ).result()
        print(f"Forward logprobs shape: {len(forward_result.logprobs)}x{len(forward_result.logprobs[0]) if forward_result.logprobs else 0}")
        print(f"Tokens processed: {forward_result.tokens_processed}")

        # Compute per-token logprobs for a sequence
        logprobs = training.compute_logprobs(train_tokens).result()
        print(f"Per-token logprobs: {[f'{lp:.3f}' if lp else 'None' for lp in logprobs]}")

        # Calculate sequence probability
        valid_logprobs = [lp for lp in logprobs if lp is not None]
        if valid_logprobs:
            seq_logprob = sum(valid_logprobs)
            print(f"Sequence log-probability: {seq_logprob:.4f}")

        # Generate sample using convenience method
        sample = training.sample("The quick brown", max_tokens=10)
        print(f"\nSample: {sample}")


def sampling_example():
    """Example using SamplingClient separately."""
    api_key = None
    if os.path.exists("build/api-token.txt"):
        api_key = open("build/api-token.txt").read().strip()

    endpoint = os.environ.get("BASILICA_API_URL", "http://localhost:8000")
    client = ServiceClient(api_key=api_key, endpoint=endpoint)

    # Create sampling client for a base model
    sampling = client.create_sampling_client(base_model="facebook/opt-125m")

    # Create prompt from token IDs
    prompt = ModelInput.from_ints([2, 133, 2119])  # "The quick"

    # Generate with custom parameters
    params = SamplingParams(
        max_tokens=20,
        temperature=0.8,
        top_p=0.95,
    )

    results = sampling.sample(prompt, sampling_params=params, num_samples=2).result()

    for i, result in enumerate(results):
        print(f"Sample {i + 1}: {result.text}")


def async_example():
    """Example using async API."""
    import asyncio

    async def train_async():
        api_key = None
        if os.path.exists("build/api-token.txt"):
            api_key = open("build/api-token.txt").read().strip()

        endpoint = os.environ.get("BASILICA_API_URL", "http://localhost:8000")
        client = ServiceClient(api_key=api_key, endpoint=endpoint)

        training = client.create_lora_training_client(
            base_model="facebook/opt-125m",
            rank=8,
            gpu_count=0,
        )

        try:
            # Use async variants
            # 10 tokens for stable gradients: "The quick brown fox jumps over the lazy dog"
            train_tokens = [2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]
            for _ in range(3):
                result = await training.forward_backward_async(
                    [Datum(input_ids=train_tokens)]
                )
                step = await training.optim_step_async()
                print(f"Step {step}: loss={result.loss:.4f}")

            checkpoint = await training.save_state_async("async-checkpoint")
            print(f"Saved: {checkpoint}")
        finally:
            training.close()

    asyncio.run(train_async())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basilica Training SDK Example")
    parser.add_argument(
        "--sampling", action="store_true", help="Run sampling example"
    )
    parser.add_argument(
        "--async", dest="run_async", action="store_true", help="Run async example"
    )
    args = parser.parse_args()

    if args.sampling:
        sampling_example()
    elif args.run_async:
        async_example()
    else:
        main()
