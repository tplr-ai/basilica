#!/usr/bin/env python3
"""Basilica Training Example - Fine-tune LLMs with LoRA

This example demonstrates the Basilica Training SDK for fine-tuning LLMs.

Usage:
    python training_example.py
    python training_example.py --rest      # RestClient demo
    python training_example.py --sampling  # SamplingClient demo
    python training_example.py --async     # Async API demo

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

    # === Phase 4: Check Server Capabilities ===
    print("=== Server Capabilities ===")
    try:
        caps = client.get_server_capabilities()
        print(f"Available models: {caps.models[:3]}...")
        # print(f"Max batch tokens: {caps.max_batch_tokens}")
        print(f"GPU types: {caps.gpu_types}")
        print(f"LoRA ranks: {caps.lora_ranks}")
    except Exception as e:
        print(f"Could not fetch capabilities: {e}")
    print()

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

        # === R2 Checkpoint Round-Trip Test ===
        print("\n--- R2 Checkpoint Round-Trip Test ---")

        # Save checkpoint to R2 via FUSE
        checkpoint_path = training.save_state("checkpoint-final").result()
        print(f"Checkpoint saved: {checkpoint_path}")

        # Record loss before modifying weights
        pre_load_result = training.forward_backward(
            [Datum(input_ids=train_tokens)],
            loss_fn="cross_entropy",
        ).result()
        loss_before_changes = pre_load_result.loss
        print(f"Loss before additional training: {loss_before_changes:.4f}")

        # Train more steps to change the weights
        print("Training 3 more steps to modify weights...")
        for i in range(3):
            result = training.forward_backward(
                [Datum(input_ids=train_tokens)],
                loss_fn="cross_entropy",
            ).result()
            training.optim_step().result()
            print(f"  Step {i+1}: loss={result.loss:.4f}")

        # Check loss after additional training (should be different)
        post_train_result = training.forward_backward(
            [Datum(input_ids=train_tokens)],
            loss_fn="cross_entropy",
        ).result()
        loss_after_changes = post_train_result.loss
        print(f"Loss after additional training: {loss_after_changes:.4f}")

        # Load checkpoint from R2 via FUSE
        print(f"\nLoading checkpoint from R2: {checkpoint_path}")
        training.load_state(checkpoint_path).result()
        print("Checkpoint loaded successfully!")

        # Verify loss matches pre-save value (proves R2 round-trip worked)
        # Retry a few times as service may need a moment after load
        import time
        loss_after_load = None
        for attempt in range(5):
            try:
                time.sleep(1)
                post_load_result = training.forward_backward(
                    [Datum(input_ids=train_tokens)],
                    loss_fn="cross_entropy",
                ).result()
                loss_after_load = post_load_result.loss
                break
            except Exception as e:
                if attempt < 4:
                    print(f"  Retry {attempt + 1}/5 after error: {e}")
                else:
                    raise

        print(f"Loss after loading checkpoint: {loss_after_load:.4f}")

        # Verify the checkpoint restored the weights correctly
        if abs(loss_after_load - loss_before_changes) < 0.01:
            print("✓ R2 checkpoint round-trip PASSED - weights restored correctly!")
        else:
            print(f"✗ R2 checkpoint round-trip FAILED - loss mismatch")
            print(f"  Expected: {loss_before_changes:.4f}, Got: {loss_after_load:.4f}")

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


def rest_client_example():
    """Example using RestClient for checkpoint and run management."""
    api_key = None
    if os.path.exists("build/api-token.txt"):
        api_key = open("build/api-token.txt").read().strip()

    endpoint = os.environ.get("BASILICA_API_URL", "http://localhost:8000")
    client = ServiceClient(api_key=api_key, endpoint=endpoint)

    # Create RestClient for checkpoint management
    rest = client.create_rest_client()

    print("=== RestClient Demo ===\n")

    # List training runs
    print("--- Training Runs ---")
    runs = rest.list_training_runs(limit=10).result()
    print(f"Found {len(runs)} training runs:")
    for run in runs:
        print(f"  - {run.get('runId', 'N/A')}: {run.get('baseModel', 'N/A')} ({run.get('status', 'N/A')})")
    print()

    # List active sessions
    print("--- Active Sessions ---")
    sessions = rest.list_sessions(limit=10).result()
    print(f"Found {len(sessions)} active sessions")
    print()

    # List checkpoints
    print("--- Checkpoints ---")
    checkpoints = rest.list_checkpoints(limit=10).result()
    print(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"  - {cp.get('checkpointId', 'N/A')}: step {cp.get('step', 'N/A')}")
    print()

    # If we have a training run, get its details
    if runs:
        run_id = runs[0].get("runId")
        if run_id:
            print(f"--- Training Run Details: {run_id} ---")
            run_detail = rest.get_training_run(run_id).result()
            print(f"  Base model: {run_detail.get('baseModel', 'N/A')}")
            print(f"  Status: {run_detail.get('status', 'N/A')}")
            print(f"  Steps completed: {run_detail.get('stepsCompleted', 0)}")
            print(f"  Tokens processed: {run_detail.get('tokensProcessed', 0)}")

    rest.close()
    print("\n=== RestClient Demo Complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basilica Training SDK Example")
    parser.add_argument(
        "--sampling", action="store_true", help="Run sampling example"
    )
    parser.add_argument(
        "--async", dest="run_async", action="store_true", help="Run async example"
    )
    parser.add_argument(
        "--rest", action="store_true", help="Run RestClient example"
    )
    args = parser.parse_args()

    if args.sampling:
        sampling_example()
    elif args.run_async:
        async_example()
    elif args.rest:
        rest_client_example()
    else:
        main()
