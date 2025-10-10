#!/usr/bin/env python3
"""
Start Rental Example for Basilica SDK

Demonstrates how to start GPU rentals with various configurations.
"""

from basilica import BasilicaClient
from ssh_utils import print_ssh_instructions


def main():
    print("Starting Basilica GPU rental...")
    print("Initializing client...")

    # Initialize client (uses BASILICA_API_URL and BASILICA_API_TOKEN from environment)
    # Create a token using: basilica tokens create
    client = BasilicaClient()

    print("\nConfiguration:")
    print("  GPU Type: h100")
    print("  Container: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime")
    print("  Ports: 8888 (Jupyter), 6006 (TensorBoard), 5000 (API)")
    print("\nRequesting GPU rental...")

    # Start a rental with all available configuration options
    rental = client.start_rental(
        # Container configuration
        container_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",  # Default: basilica default image

        # GPU selection - choose one method:

        # Method 1: Specify GPU type
        gpu_type="b200",  # Options: h100, a100, etc.

        # Method 2: Target a specific node by ID (find the id manually or by using list_rentals method)
        # node_id="node-uuid-here",  # Use specific node

        # SSH configuration
        # ssh_pubkey_path="~/.ssh/id_rsa.pub",  # Explicit SSH key path
        # Auto-uses ~/.ssh/basilica_ed25519.pub if not specified

        # Set custom Environment variables that will be set in the container
        environment={
            "CUDA_VISIBLE_DEVICES": "0,1",
        },

        # Port mappings for services
        ports=[
            {"container_port": 8888, "host_port": 8888, "protocol": "tcp"},  # Jupyter
            {"container_port": 6006, "host_port": 6006, "protocol": "tcp"},  # TensorBoard
            {"container_port": 5000, "host_port": 5000, "protocol": "tcp"},  # API server
        ],

        command=["/bin/bash"],
    )

    print("\nRental started successfully!")

    # Access rental details
    print(f"Rental ID: {rental.rental_id}")
    print(f"Container: {rental.container_name}")
    print(f"Status: {rental.status}")

    # Print SSH connection instructions
    print_ssh_instructions(rental.ssh_credentials, rental.rental_id)

    # Get updated rental status
    status = client.get_rental(rental.rental_id)
    print(f"Node ID: {status.node.id}")
    print(f"Created at: {status.created_at}")

    # List GPU details
    for gpu in status.node.gpu_specs:
        print(f"GPU: {gpu.name} - {gpu.memory_gb} GB")

    # Stop rental when done
    # client.stop_rental(rental.rental_id)


if __name__ == "__main__":
    main()
