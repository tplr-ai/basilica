#!/usr/bin/env python3
"""
Deploy Tau agent on Basilica.

Usage:
    export TAU_BOT_TOKEN="your-telegram-token"
    export CURSOR_API_KEY="your-cursor-key"
    export OPENAI_API_KEY="your-openai-key"
    python3 30_tau.py
"""
import os
import sys

from basilica import BasilicaClient

bot_token = os.getenv("TAU_BOT_TOKEN")
cursor_key = os.getenv("CURSOR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not bot_token:
    sys.exit("Set TAU_BOT_TOKEN")
if not cursor_key:
    sys.exit("Set CURSOR_API_KEY")
if not openai_key:
    sys.exit("Set OPENAI_API_KEY")

env = {
    "TAU_BOT_TOKEN": bot_token,
    "CURSOR_API_KEY": cursor_key,
    "OPENAI_API_KEY": openai_key,
    "TAU_CHAT_BACKEND": os.getenv("TAU_CHAT_BACKEND", "cursor"),
    "TAU_CURSOR_CHAT_MODEL": os.getenv("TAU_CURSOR_CHAT_MODEL", "composer-1"),
    "TAU_OPENAI_CHAT_MODEL": os.getenv("TAU_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
}

client = BasilicaClient()

deployment = client.deploy(
    name="tau",
    image="ghcr.io/one-covenant/basilica-tau:latest",
    port=8080,
    env=env,
    cpu="2",
    memory="4Gi",
    timeout=600,
    storage=True,
)

print(f"Tau deployed: {deployment.url}")
print("Send a message to your Telegram bot to initialize chat_id.txt")

