#!/usr/bin/env python3
"""
Public Deployment Metadata - Enroll deployments for validator verification.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 31_public_metadata.py
"""
from basilica import BasilicaClient

client = BasilicaClient()

# Create a deployment with public metadata enabled
deployment = client.create_deployment(
    instance_name="metadata-demo",
    image="hashicorp/http-echo:latest",
    replicas=1,
    port=5678,
    public_metadata=True,
    ttl_seconds=600,
)
print(f"Deployed: {deployment.instance_name}")
print(f"  URL:             {deployment.url}")
print(f"  Public Metadata: {deployment.public_metadata}")

# Check enrollment status
status = client.get_enrollment_status(deployment.instance_name)
print(f"\nEnrollment status: {'Enrolled' if status.public_metadata else 'Not enrolled'}")

# Query public metadata (no auth required - validators use this)
metadata = client.get_public_deployment_metadata(deployment.instance_name)
print(f"\nPublic metadata for '{metadata.instance_name}':")
print(f"  Image:    {metadata.image}:{metadata.image_tag}")
print(f"  State:    {metadata.state}")
print(f"  Replicas: {metadata.replicas.ready}/{metadata.replicas.desired}")
print(f"  Uptime:   {metadata.uptime_seconds}s")

# Disable enrollment
client.enroll_metadata(deployment.instance_name, enabled=False)
print("\nMetadata enrollment disabled.")

# Re-enable enrollment
client.enroll_metadata(deployment.instance_name, enabled=True)
print("Metadata enrollment re-enabled.")

# Cleanup
client.delete_deployment(deployment.instance_name)
print(f"\nDeployment '{deployment.instance_name}' deleted.")
