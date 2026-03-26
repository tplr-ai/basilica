#!/usr/bin/env python3
"""
Deploy Clawdbot with Kimi-K2.5 inference on Basilica.

Connects Clawdbot to a running Kimi-K2.5 vLLM deployment on Basilica,
giving the agent platform access to K2.5's reasoning capabilities.

Prerequisites:
    - A running Kimi-K2.5 deployment on Basilica (see 26_kimi_k2_5_multimodal.py)

Usage:
    export BASILICA_API_TOKEN="your-token"
    export KIMI_K2_5_DEPLOYMENT_URL="https://<id>.deployments.basilica.ai"
    python3 27_clawdbot_kimi_k2_5.py

See: https://github.com/clawdbot/clawdbot
"""
import os
import re
import sys

from basilica import BasilicaClient

k2_5_url = os.getenv("KIMI_K2_5_DEPLOYMENT_URL")
if not k2_5_url:
    sys.exit(
        "Set KIMI_K2_5_DEPLOYMENT_URL to your Kimi-K2.5 deployment URL.\n"
        "Example: export KIMI_K2_5_DEPLOYMENT_URL=https://<id>.deployments.basilica.ai"
    )

# Ensure the base URL ends with /v1 for OpenAI-compatible API
base_url = k2_5_url.rstrip("/")
if not base_url.endswith("/v1"):
    base_url = f"{base_url}/v1"

print("Deploying Clawdbot with Kimi-K2.5 backend...")
print(f"  K2.5 endpoint: {base_url}")
print()

client = BasilicaClient()

deployment = client.deploy(
    name="clawdbot-kimi-k2-5",
    image="ghcr.io/one-covenant/basilica-clawdbot:kimi-k2.5",
    port=18789,
    env={
        "KIMI_K2_5_BASE_URL": base_url,
    },
    cpu="2",
    memory="4Gi",
    timeout=600,
)

print(f"Clawdbot deployed: {deployment.url}")
print()

# Extract gateway token from logs
match = re.search(r"CLAWDBOT_GATEWAY_TOKEN=([a-f0-9]{64})", deployment.logs(tail=200))
if match:
    token = match.group(1)
    print(f"Control UI: {deployment.url}/chat?session=main&token={token}")
    print()
    print(f"Gateway token: {token}")
else:
    print(f"Get token: basilica deploy logs {deployment.name} | grep CLAWDBOT_GATEWAY_TOKEN")
