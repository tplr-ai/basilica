#!/usr/bin/env python3
"""
Deploy Tau agent on Basilica.

Usage:
    export TAU_BOT_TOKEN="your-telegram-token"
    export CHUTES_API_TOKEN="your-chutes-token"
    export TAU_CHAT_MODEL="optional-model-override"
    python3 30_tau.py
"""
import os
import sys

from basilica import (
    BasilicaClient,
    Deployment,
    HealthCheckConfig,
    PersistentStorageSpec,
    ProbeConfig,
    StorageBackend,
    StorageSpec,
)

bot_token = os.getenv("TAU_BOT_TOKEN")
chutes_token = os.getenv("CHUTES_API_TOKEN")
chat_model = os.getenv("TAU_CHAT_MODEL")

if not bot_token:
    sys.exit("Set TAU_BOT_TOKEN")
if not chutes_token:
    sys.exit("Set CHUTES_API_TOKEN")

env = {
    "TAU_BOT_TOKEN": bot_token,
    "CHUTES_API_TOKEN": chutes_token,
}
if chat_model:
    env["TAU_CHAT_MODEL"] = chat_model

client = BasilicaClient()

health_probe = ProbeConfig(
    path="/health",
    port=8080,
    initial_delay_seconds=30,
    period_seconds=10,
    timeout_seconds=5,
    failure_threshold=3,
)

storage = StorageSpec(
    persistent=PersistentStorageSpec(
        enabled=True,
        backend=StorageBackend.R2,
        bucket="",
        region="auto",
        credentials_secret="basilica-r2-credentials",
        sync_interval_ms=1000,
        cache_size_mb=2048,
        mount_path="/data",
    )
)

response = client.create_deployment(
    instance_name="tau",
    image="ghcr.io/one-covenant/basilica-tau:latest",
    port=8080,
    command=["/usr/local/bin/basilica-entrypoint.sh"],
    env=env,
    cpu="2",
    memory="16Gi",
    public=False,
    storage=storage,
    health_check=HealthCheckConfig(
        liveness=health_probe,
        readiness=health_probe,
    ),
)

deployment = Deployment._from_response(client, response)
deployment.wait_until_ready(timeout=600)
deployment.refresh()

print(f"Tau deployed: {deployment.url}")
print("Send a message to your Telegram bot to initialize chat_id.txt")
