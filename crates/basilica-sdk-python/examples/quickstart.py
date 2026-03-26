#!/usr/bin/env python3
"""
Basilica SDK Quickstart - Minimal example
"""

from basilica import BasilicaClient
from ssh_utils import format_ssh_command

print("Starting Basilica rental with default configuration...")

# That's it! The client auto-configures from BASILICA_API_TOKEN environment variable
# Create a token using: basilica tokens create
client = BasilicaClient()

print("Requesting b200 GPU rental...")

# Start a rental with all defaults - returns typed RentalResponse with SSH credentials
rental = client.start_rental(gpu_type="b200")
print(f"Rental started with ID: {rental.rental_id}")

# Print SSH command if available - using typed attributes
if rental.ssh_credentials:
    ssh_command = format_ssh_command(rental.ssh_credentials)
    print(f"\nConnect with: {ssh_command}")
else:
    print("No SSH access (not yet provisioned)")

# When done, stop the rental
# client.stop_rental(rental.rental_id)