#!/usr/bin/env python3
"""
Deploy Clawdbot AI agent platform on Basilica.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python3 24_clawdbot.py

See: https://github.com/clawdbot/clawdbot
"""
import os
import re
import sys

from basilica import BasilicaClient

api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY")

client = BasilicaClient()

deployment = client.deploy(
    name="clawdbot",
    image="ghcr.io/one-covenant/basilica-clawdbot:latest",
    port=18789,
    env={k: os.environ[k] for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"] if k in os.environ},
    cpu="2",
    memory="4Gi",
    timeout=600,
)

print(f"Clawdbot deployed: {deployment.url}")

# Extract gateway token from logs
match = re.search(r"CLAWDBOT_GATEWAY_TOKEN=([a-f0-9]{64})", deployment.logs(tail=200))
if match:
    print(f"Control UI: {deployment.url}/chat?session=main&token={match.group(1)}")
else:
    print(f"Get token: basilica deploy logs {deployment.name} | grep CLAWDBOT_GATEWAY_TOKEN")
