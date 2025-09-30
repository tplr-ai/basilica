#!/usr/bin/env python3
"""
Health Check Example for Basilica SDK
"""

from basilica import BasilicaClient


def main():
    # Initialize client using environment variables
    # BASILICA_API_URL and BASILICA_API_TOKEN
    client = BasilicaClient()

    # Or initialize with explicit configuration
    # client = BasilicaClient(
    #     base_url="https://api.basilica.ai",
    #     api_key="basilica_..."  # Your token from 'basilica tokens create'
    # )
    
    # Perform health check
    response = client.health_check()
    
    # Access response fields
    print(f"Status: {response.status}")
    print(f"Version: {response.version}")
    print(f"Timestamp: {response.timestamp}")
    print(f"Healthy validators: {response.healthy_validators}/{response.total_validators}")


if __name__ == "__main__":
    main()