#!/usr/bin/env python3
"""
Example: List available nodes with typed responses
"""

from basilica import BasilicaClient
import os

def main():
    # Create client (uses BASILICA_API_TOKEN from environment)
    # Create a token using: basilica tokens create
    client = BasilicaClient()

    # Check API health - returns typed HealthCheckResponse
    print("Checking API health...")
    health = client.health_check()
    print(f"API Status: {health.status}")
    print(f"Version: {health.version}")
    print(f"Healthy validators: {health.healthy_validators}/{health.total_validators}")
    print()

    # List available nodes - returns typed List[AvailableNode]
    print("Listing available nodes...")
    nodes = client.list_nodes(available=True)

    print(f"Found {len(nodes)} available nodes")

    # Display node details using typed attributes
    for node_info in nodes[:5]:  # Show first 5
        node = node_info.node
        availability = node_info.availability

        print(f"\nNode: {node.id}")
        print(f"  Location: {node.location or 'Not specified'}")

        # GPU specs (typed access)
        for gpu in node.gpu_specs:
            print(f"  GPU: {gpu.name} - {gpu.memory_gb} GB")
            print(f"    Compute capability: {gpu.compute_capability}")

        # CPU specs (typed access)
        cpu = node.cpu_specs
        print(f"  CPU: {cpu.cores} cores, {cpu.memory_gb} GB RAM")
        print(f"    Model: {cpu.model}")

        # Availability info (typed access)
        print(f"  Verification score: {availability.verification_score:.2f}")
        print(f"  Uptime: {availability.uptime_percentage:.1f}%")
        if availability.available_until:
            print(f"  Available until: {availability.available_until}")

if __name__ == "__main__":
    main()
