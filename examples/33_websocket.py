#!/usr/bin/env python3
"""
WebSocket Deployment - Deploy apps with long-lived WebSocket connections.

Enables WebSocket support with configurable idle timeout for real-time
applications like chat servers, live dashboards, and streaming APIs.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 33_websocket.py
"""
from basilica import BasilicaClient, WebSocketConfig

client = BasilicaClient()

# Deploy with WebSocket support (default 1800s idle timeout)
deployment = client.create_deployment(
    instance_name="ws-demo",
    image="hashicorp/http-echo:latest",
    replicas=1,
    port=5678,
    websocket=WebSocketConfig(enabled=True),
    ttl_seconds=600,
)

print(f"Deployed: {deployment.instance_name}")
print(f"  URL: {deployment.url}")
print()
print("WebSocket connections are now supported on this deployment.")
print("Idle timeout: 1800s (default)")

# Deploy with custom idle timeout (e.g. 3600s for long sessions)
deployment2 = client.create_deployment(
    instance_name="ws-demo-custom",
    image="hashicorp/http-echo:latest",
    replicas=1,
    port=5678,
    websocket=WebSocketConfig(enabled=True, idle_timeout_seconds=3600),
    ttl_seconds=600,
)

print()
print(f"Deployed: {deployment2.instance_name}")
print(f"  URL: {deployment2.url}")
print("  Idle timeout: 3600s (custom)")

# Cleanup
client.delete_deployment(deployment.instance_name)
client.delete_deployment(deployment2.instance_name)
print()
print("Deployments deleted.")
