#!/usr/bin/env python3
"""Basilica Training Example - Fine-tune LLMs with LoRA"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crates", "basilica-sdk-python", "python"))

from basilica.training import Client

# Create client (auto-detects BASILICA_API_TOKEN)
client = Client(
    api_key=open("build/api-token.txt").read().strip() if os.path.exists("build/api-token.txt") else None
)

# Create training session
with client.training("facebook/opt-125m", rank=8, gpu_count=0) as session:

    # Training loop
    for i in range(5):
        loss = session.forward_backward(
            [{"input_ids": [2, 133, 2119, 6219, 23602]}],
            loss_fn="cross_entropy",
        )
        step = session.optim_step()
        print(f"Step {step}: loss={loss:.4f}")

    # Generate sample
    print(f"\nSample: {session.sample('The quick brown', max_tokens=10)}")
