#!/usr/bin/env python3
"""
Topology Spread - Deploy pods across different nodes.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 22_topology_spread.py
"""
from basilica import BasilicaClient, TopologySpreadConfig

client = BasilicaClient()

deployment = client.create_deployment(
    instance_name="spread-demo",
    image="hashicorp/http-echo:latest",
    replicas=2,
    port=5678,
    topology_spread=TopologySpreadConfig.unique_nodes(),
)

print(f"Live at: {deployment.url}")
print("Each pod runs on a different node.")
