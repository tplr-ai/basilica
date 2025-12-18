#!/usr/bin/env python3
"""
Example training script using Basilica Training SDK.

This example demonstrates:
1. Creating a training session with LoRA
2. Running forward-backward passes
3. Applying optimizer updates
4. Generating samples with the fine-tuned model
5. Saving and loading checkpoints

Usage:
    export BASILICA_API_TOKEN="your-api-token"
    python examples/training_example.py
"""

import os
import sys

# Add the SDK to path for local development
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "crates",
        "basilica-sdk-python",
        "python",
    ),
)

from basilica.training import (
    Datum,
    TrainingClient,
    TrainingError,
)


def main():
    """Run the training example."""
    # Check for API key
    api_key = os.getenv("BASILICA_API_TOKEN")
    if not api_key:
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("Set it with: export BASILICA_API_TOKEN='your-api-token'")
        sys.exit(1)

    # Initialize client
    endpoint = os.getenv("BASILICA_API_URL", "http://localhost:8080")
    print(f"Connecting to {endpoint}...")

    try:
        client = TrainingClient(
            api_key=api_key,
            endpoint=endpoint,
        )
    except Exception as e:
        print(f"Error: Failed to initialize client: {e}")
        sys.exit(1)

    # Create training session
    print("\n=== Creating Training Session ===")
    print("Model: meta-llama/Llama-3.1-8B-Instruct")
    print("LoRA Rank: 32")
    print("Learning Rate: 1e-4")

    try:
        session = client.create_session(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            rank=32,
            alpha=64,
            dropout=0.05,
            learning_rate=1e-4,
            weight_decay=0.01,
            gpu_count=1,
            gpu_models=["H100", "A100"],
            seed=42,
        )
        print(f"Session created: {session.session_id}")
    except TrainingError as e:
        print(f"Error: Failed to create session: {e}")
        sys.exit(1)

    # Create example training data
    # In a real scenario, you would use a tokenizer to create these
    print("\n=== Preparing Training Data ===")

    # Example: Training on simple text completion
    # Input: "The capital of France is"
    # Label: "The capital of France is Paris"
    example_data = [
        Datum(
            # Simplified token IDs (in practice, use a tokenizer)
            input_ids=[1, 450, 7483, 310, 3444, 338],  # "The capital of France is"
            labels=[1, 450, 7483, 310, 3444, 338, 3681],  # "The capital of France is Paris"
            loss_weights=[0, 0, 0, 0, 0, 0, 1],  # Only train on "Paris"
        ),
        Datum(
            input_ids=[1, 450, 7483, 310, 9556, 338],  # "The capital of Germany is"
            labels=[1, 450, 7483, 310, 9556, 338, 5765],  # "The capital of Germany is Berlin"
            loss_weights=[0, 0, 0, 0, 0, 0, 1],  # Only train on "Berlin"
        ),
    ]

    print(f"Training examples: {len(example_data)}")

    # Training loop
    print("\n=== Training Loop ===")
    num_steps = 10
    accumulation_steps = 2

    for step in range(num_steps):
        # Forward-backward pass (accumulate gradients)
        total_loss = 0.0
        for i in range(accumulation_steps):
            try:
                result = session.forward_backward([example_data[i % len(example_data)]])
                total_loss += result.loss
            except TrainingError as e:
                print(f"Error in forward_backward: {e}")
                session.close()
                sys.exit(1)

        avg_loss = total_loss / accumulation_steps

        # Optimizer step (apply gradients)
        try:
            current_step = session.optim_step()
        except TrainingError as e:
            print(f"Error in optim_step: {e}")
            session.close()
            sys.exit(1)

        print(f"Step {current_step}: loss={avg_loss:.4f}")

    # Save checkpoint
    print("\n=== Saving Checkpoint ===")
    try:
        checkpoint_path = session.save_state("checkpoint-final", include_optimizer=True)
        print(f"Checkpoint saved to: {checkpoint_path}")
    except TrainingError as e:
        print(f"Warning: Failed to save checkpoint: {e}")

    # Generate samples
    print("\n=== Generating Samples ===")
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "Machine learning is",
    ]

    for prompt in prompts:
        try:
            sample = session.sample(
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
            )
            print(f"Prompt: {prompt}")
            print(f"Generated: {sample.text}")
            print(f"Finish reason: {sample.finish_reason}")
            print()
        except TrainingError as e:
            print(f"Warning: Sample generation failed: {e}")

    # Get session status
    print("\n=== Session Status ===")
    try:
        status = session.get_status()
        print(f"Phase: {status.get('phase', 'unknown')}")
        print(f"Steps completed: {status.get('stepsCompleted', 0)}")
        print(f"Tokens processed: {status.get('tokensProcessed', 0)}")
    except TrainingError as e:
        print(f"Warning: Failed to get status: {e}")

    # Cleanup
    print("\n=== Cleaning Up ===")
    try:
        session.close()
        print("Session deleted successfully")
    except TrainingError as e:
        print(f"Warning: Failed to delete session: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
