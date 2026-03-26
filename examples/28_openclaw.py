#!/usr/bin/env python3
"""
Deploy OpenClaw gateway on Basilica.

Usage:
    export OPENCLAW_BASE_URL="https://your-openai-compatible-backend/v1"
    export OPENAI_API_KEY="your-key"  # optional if backend requires
    python3 28_openclaw.py

See: https://github.com/openclaw/openclaw
"""
import os
import re
import sys

from basilica import BasilicaClient

backend_url = os.getenv("OPENCLAW_BASE_URL") or os.getenv("OPENAI_BASE_URL")
if not backend_url:
    sys.exit("Set OPENCLAW_BASE_URL or OPENAI_BASE_URL")

client = BasilicaClient()

env = {
    "OPENCLAW_BASE_URL": backend_url,
    "OPENAI_BASE_URL": backend_url,
}
if os.getenv("OPENAI_API_KEY"):
    env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

deployment = client.deploy(
    name="openclaw",
    image="ghcr.io/one-covenant/basilica-openclaw:latest",
    port=18789,
    env=env,
    cpu="2",
    memory="4Gi",
    timeout=600,
)

print(f"OpenClaw deployed: {deployment.url}")

match = re.search(
    r"(?:CLAWDBOT|OPENCLAW)_GATEWAY_TOKEN=([a-f0-9]{64})",
    deployment.logs(tail=200),
)
if match:
    print(f"Control UI: {deployment.url}/chat?session=main&token={match.group(1)}")
else:
    print(f"Get token: basilica deploy logs {deployment.name} | grep GATEWAY_TOKEN")
