"""
End-to-end integration test for DNS propagation fix.

This test creates a real Basilica deployment and verifies that
wait_until_ready() waits for DNS resolution before returning.

Requirements:
- BASILICA_API_TOKEN environment variable must be set
- Network access to Basilica API
"""

import os
import sys
import time
import socket
import httpx
from urllib.parse import urlparse

# Add the SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from basilica import BasilicaClient, Deployment


def test_dns_resolves_after_wait_until_ready():
    """
    End-to-end test: Verify DNS is resolvable immediately after wait_until_ready() returns.

    This test:
    1. Creates a real deployment
    2. Calls wait_until_ready()
    3. Immediately checks DNS resolution
    4. Makes an HTTP request to the deployment URL
    5. Cleans up the deployment

    If the fix works, steps 3 and 4 should succeed without DNS errors.
    """
    api_token = os.environ.get("BASILICA_API_TOKEN")
    if not api_token:
        print("SKIP: BASILICA_API_TOKEN not set")
        return False

    # Generate unique deployment name
    deployment_name = f"dns-test-{int(time.time())}"
    client = BasilicaClient()
    deployment = None

    try:
        print(f"\n{'='*60}")
        print("DNS Propagation Fix - End-to-End Test")
        print(f"{'='*60}")

        # Step 1: Create deployment
        print(f"\n[1/5] Creating deployment: {deployment_name}")
        response = client.create_deployment(
            instance_name=deployment_name,
            image="hashicorp/http-echo:latest",
            port=5678,
            cpu="100m",
            memory="128Mi",
            ttl_seconds=300,
            public=True,
            env={"ECHO_TEXT": "dns-test-success"},
        )
        print(f"      Deployment created, URL: {response.url}")

        # Step 2: Wait for deployment to be ready
        print(f"\n[2/5] Calling wait_until_ready()...")
        deployment = Deployment._from_response(client, response)
        start_time = time.time()
        status = deployment.wait_until_ready(timeout=180, poll_interval=3, silent=True)
        wait_duration = time.time() - start_time
        print(f"      wait_until_ready() returned after {wait_duration:.1f}s")
        print(f"      Status: state={status.state}, replicas={status.replicas_ready}/{status.replicas_desired}")

        # Step 3: Immediately check DNS resolution
        print(f"\n[3/5] Checking DNS resolution immediately after wait_until_ready()...")
        hostname = urlparse(deployment.url).hostname

        dns_start = time.time()
        try:
            ip_address = socket.gethostbyname(hostname)
            dns_duration = time.time() - dns_start
            print(f"      DNS resolved: {hostname} -> {ip_address} ({dns_duration*1000:.1f}ms)")
            dns_success = True
        except socket.gaierror as e:
            dns_duration = time.time() - dns_start
            print(f"      DNS FAILED: {hostname} -> {e} ({dns_duration*1000:.1f}ms)")
            dns_success = False

        # Step 4: Make HTTP request
        print(f"\n[4/5] Making HTTP request to {deployment.url}...")
        http_start = time.time()
        try:
            with httpx.Client(timeout=30) as http_client:
                # Try health endpoint or root
                response = http_client.get(deployment.url)
                http_duration = time.time() - http_start
                print(f"      HTTP {response.status_code} ({http_duration*1000:.1f}ms)")
                http_success = response.status_code < 500
        except httpx.ConnectError as e:
            http_duration = time.time() - http_start
            print(f"      HTTP FAILED: ConnectError: {e} ({http_duration*1000:.1f}ms)")
            http_success = False
        except Exception as e:
            http_duration = time.time() - http_start
            print(f"      HTTP FAILED: {type(e).__name__}: {e} ({http_duration*1000:.1f}ms)")
            http_success = False

        # Step 5: Report results
        print(f"\n[5/5] Test Results")
        print(f"      {'='*50}")
        print(f"      DNS Resolution:  {'PASS' if dns_success else 'FAIL'}")
        print(f"      HTTP Request:    {'PASS' if http_success else 'FAIL'}")
        print(f"      {'='*50}")

        if dns_success and http_success:
            print(f"\n      SUCCESS: DNS propagation fix is working!")
            print(f"      The deployment URL was usable immediately after wait_until_ready() returned.")
            return True
        else:
            print(f"\n      FAILURE: DNS propagation issue still exists!")
            if not dns_success:
                print(f"      DNS resolution failed immediately after wait_until_ready().")
            if not http_success:
                print(f"      HTTP request failed - could be DNS or other network issue.")
            return False

    except Exception as e:
        print(f"\n      ERROR: Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if deployment:
            print(f"\n[Cleanup] Deleting deployment: {deployment_name}")
            try:
                deployment.delete()
                print(f"          Deployment deleted")
            except Exception as e:
                print(f"          Failed to delete: {e}")


if __name__ == "__main__":
    success = test_dns_resolves_after_wait_until_ready()
    sys.exit(0 if success else 1)
