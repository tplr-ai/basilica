#!/usr/bin/env python3
"""
Example: List available executors with typed responses
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
    
    # List available executors - returns typed List[AvailableExecutor]
    print("Listing available executors...")
    executors = client.list_executors(available=True)
    
    print(f"Found {len(executors)} available executors")
    
    # Display executor details using typed attributes
    for executor_info in executors[:5]:  # Show first 5
        executor = executor_info.executor
        availability = executor_info.availability
        
        print(f"\nExecutor: {executor.id}")
        print(f"  Location: {executor.location or 'Not specified'}")
        
        # GPU specs (typed access)
        for gpu in executor.gpu_specs:
            print(f"  GPU: {gpu.name} - {gpu.memory_gb} GB")
            print(f"    Compute capability: {gpu.compute_capability}")
        
        # CPU specs (typed access)
        cpu = executor.cpu_specs
        print(f"  CPU: {cpu.cores} cores, {cpu.memory_gb} GB RAM")
        print(f"    Model: {cpu.model}")
        
        # Availability info (typed access)
        print(f"  Verification score: {availability.verification_score:.2f}")
        print(f"  Uptime: {availability.uptime_percentage:.1f}%")
        if availability.available_until:
            print(f"  Available until: {availability.available_until}")

if __name__ == "__main__":
    main()