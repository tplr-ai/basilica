#!/usr/bin/env python3
"""
Start CPU Rental Example for Basilica SDK

Demonstrates how to start and stop CPU-only rentals (no GPU) with secure cloud providers.
This script is interactive - it waits for the rental to be ready and lets you terminate it.
"""

import sys
import time
from pathlib import Path
from typing import Optional

from basilica import BasilicaClient
from rental_utils import format_ssh_command, find_private_key_for_public_key, wait_for_rental_ready


def main():
    print("=" * 60)
    print("  Basilica CPU Rental - Interactive Example")
    print("=" * 60)

    print("\nInitializing client...")

    # Initialize client connecting to local API server
    # Create a token using: basilica tokens create
    client = BasilicaClient()

    # Step 1: Ensure SSH key is registered
    print("\n[Step 1] Checking SSH key registration...")
    ssh_key = client.get_ssh_key()
    if ssh_key is None:
        print("No SSH key registered. Registering from ~/.ssh/id_ed25519.pub...")
        ssh_key = client.register_ssh_key("my-key")
        print(f"Registered SSH key: {ssh_key.name} (ID: {ssh_key.id})")
    else:
        print(f"Using existing SSH key: {ssh_key.name} (ID: {ssh_key.id})")

    # Step 2: List available CPU offerings
    print("\n[Step 2] Fetching available CPU offerings...")
    offerings = client.list_cpu_offerings()

    if not offerings:
        print("No CPU offerings available at this time.")
        return

    print(f"\nFound {len(offerings)} CPU offerings:")
    for i, o in enumerate(offerings):
        print(
            f"  [{i}] {o.id}: {o.vcpu_count} vCPUs, {o.system_memory_gb}GB RAM, "
            f"{o.storage_gb}GB storage @ ${o.hourly_rate}/hr ({o.region})"
        )

    # Step 3: Start a rental with the first available offering
    offering = offerings[0]
    print(f"\n[Step 3] Starting CPU rental with offering: {offering.id}")
    print(f"  - vCPUs: {offering.vcpu_count}")
    print(f"  - RAM: {offering.system_memory_gb}GB")
    print(f"  - Storage: {offering.storage_gb}GB")
    print(f"  - Hourly rate: ${offering.hourly_rate}")

    rental = client.start_cpu_rental(offering_id=offering.id)

    print("\nRental request submitted!")
    print(f"  Rental ID: {rental.rental_id}")
    print(f"  Provider: {rental.provider}")
    print(f"  Hourly cost: ${rental.hourly_cost:.4f}")

    # Step 4: Wait for rental to be ready
    print("\n[Step 4] Waiting for rental to be ready...")
    try:
        ready_rental = wait_for_rental_ready(client, rental.rental_id, rental_type="cpu")
    except (TimeoutError, RuntimeError) as e:
        print(f"\nError: {e}")
        print("Attempting to stop the rental...")
        try:
            client.stop_cpu_rental(rental.rental_id)
            print("Rental stopped.")
        except Exception as stop_error:
            print(f"Failed to stop rental: {stop_error}")
        return

    # Step 5: Display SSH credentials
    print("\n" + "=" * 60)
    print("  RENTAL READY")
    print("=" * 60)

    # Display all rental information
    print(f"\n  Rental ID: {ready_rental.rental_id}")
    print(f"  Status: {ready_rental.status}")
    print(f"  Provider: {ready_rental.provider}")

    if ready_rental.vcpu_count:
        print(f"  vCPUs: {ready_rental.vcpu_count}")
    if ready_rental.system_memory_gb:
        print(f"  RAM: {ready_rental.system_memory_gb}GB")
    if ready_rental.ip_address:
        print(f"  IP Address: {ready_rental.ip_address}")
    if ready_rental.hourly_cost:
        print(f"  Hourly Cost: ${ready_rental.hourly_cost:.4f}")

    # Find the matching private key from ~/.ssh/
    private_key_path = None
    if ssh_key and ssh_key.public_key:
        private_key_path = find_private_key_for_public_key(ssh_key.public_key)

    # Display SSH connection info
    # Note: ssh_command can be in various formats:
    # - "ssh user@host" (e.g., "ssh ubuntu@1.2.3.4")
    # - "user@host:port" (e.g., "root@1.2.3.4:22")
    # We use format_ssh_command to parse and build the actual command.
    print(f"\n  SSH Connection:")
    if ready_rental.ssh_command:
        ssh_cmd = format_ssh_command(ready_rental.ssh_command, private_key_path)
        print(f"    {ssh_cmd}")
        if not private_key_path:
            print(f"\n  Note: Could not find matching private key in ~/.ssh/")
            print(f"        Update the -i flag with your private key path.")
    else:
        print(f"    No SSH access available yet")

    if private_key_path:
        print(f"\n  Private Key: {private_key_path}")

    print("\n" + "-" * 60)

    # Step 6: Interactive - wait for user to terminate
    print("\nThe rental is now running. You can SSH into the machine using the command above.")
    print("Press Enter to terminate the rental (or Ctrl+C to exit without terminating)...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nExiting without terminating the rental.")
        print(f"To manually stop later, run: client.stop_cpu_rental('{rental.rental_id}')")
        return

    # Step 7: Stop the rental
    print("\n[Step 7] Stopping rental...")
    try:
        result = client.stop_cpu_rental(rental.rental_id)
        print("\n" + "=" * 60)
        print("  RENTAL STOPPED")
        print("=" * 60)
        print(f"\n  Rental ID: {result.rental_id}")
        print(f"  Status: {result.status}")
        print(f"  Duration: {result.duration_hours:.4f} hours")
        print(f"  Total cost: ${result.total_cost:.4f}")
    except Exception as e:
        print(f"Error stopping rental: {e}")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
