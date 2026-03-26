# Basilica SDK - Complete Getting Started Guide

This guide walks you through everything you need to start using the Basilica SDK, from generating your API token to running your first GPU rental or deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Generating Your API Token](#generating-your-api-token)
3. [Installing the Python SDK](#installing-the-python-sdk)
4. [Your First API Call](#your-first-api-call)
5. [Complete Working Example](#complete-working-example)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** installed
- **Basilica CLI** installed (for token generation)
- **SSH key pair** for GPU rental access (optional, can be generated)
- **Internet connection** to access the Basilica API

### Installing the Basilica CLI

If you don't have the CLI installed yet:

```bash
# From the repository root
cargo install --path crates/basilica-cli

# Or build from source
cd crates/basilica-cli
cargo build --release
sudo cp target/release/basilica /usr/local/bin/
```

Verify installation:

```bash
basilica --version
```

---

## Generating Your API Token

There are three methods to generate a `BASILICA_API_TOKEN`:

### Method 1: Using the CLI (Recommended)

This is the **recommended** method for production use.

#### Step 1: Authenticate with Basilica

First, you need to authenticate to get a JWT token. The CLI uses file-based authentication:

```bash
# The CLI will prompt you for credentials or use existing JWT token
# stored in ~/.basilica/credentials
basilica tokens create
```

If you don't have credentials set up yet, you'll need to authenticate first. The authentication method depends on your Basilica deployment configuration.

#### Step 2: Create Your API Token

```bash
# Create a token with a descriptive name
basilica tokens create my-dev-token

# Or create interactively (will prompt for name)
basilica tokens create
```

**Output:**

```
API token created successfully!

Token: basilica_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z

IMPORTANT: This token will only be shown once. Save it securely!

Set it as an environment variable:
  export BASILICA_API_TOKEN="basilica_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z"

Token details:
  Name: my-dev-token
  Created: 2025-11-01 12:34:56 UTC
```

#### Step 3: Set the Environment Variable

```bash
# For current shell session
export BASILICA_API_TOKEN="basilica_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z"

# Or add to your shell profile for persistence
echo 'export BASILICA_API_TOKEN="basilica_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 4: Verify Token is Set

```bash
echo $BASILICA_API_TOKEN
# Should output: basilica_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z
```

### Method 2: Using the Bootstrap Script (Development/Testing)

For development and testing environments, you can use the bootstrap script:

```bash
cd scripts/e2e

# Generate token for a test user
./bootstrap-api-key.sh --user test-user --name my-test-token --scopes "rentals:* jobs:*"
```

**Output:**

```
Generated API token for user: test-user
Token: basilica_abc123...

Run this command to set it:
  export BASILICA_API_TOKEN="basilica_abc123..."
```

### Method 3: Programmatic Generation (Advanced)

For automated systems, you can generate tokens programmatically using the Rust SDK:

```rust
use basilica_sdk::ClientBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First authenticate with JWT
    let client = ClientBuilder::default()
        .base_url("https://api.basilica.ai")
        .with_file_based_auth()  // Uses ~/.basilica/credentials
        .build()?;

    // Create API token
    let response = client.create_api_key("my-automation-token").await?;

    println!("Token: {}", response.token);
    println!("Name: {}", response.name);

    Ok(())
}
```

### Managing Your Tokens

```bash
# List all your tokens
basilica tokens list

# Output:
# ID                                Name              Created              Last Used
# 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p  my-dev-token      2025-11-01 12:34:56  2025-11-01 14:22:10
# 2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q  my-test-token     2025-11-01 10:15:32  Never

# Revoke a token
basilica tokens revoke my-dev-token

# Or with auto-confirmation
basilica tokens revoke my-dev-token --yes
```

**Token Limits:**

- Maximum 10 tokens per user
- Token names: 1-100 characters, alphanumeric + hyphens + underscores
- Tokens are shown only once at creation

---

## Installing the Python SDK

### Option 1: Install from PyPI (Recommended for Users)

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from PyPI
pip install basilica-sdk
```

### Option 2: Install from Source (For Development)

If you're contributing to the SDK or need the latest development version:

#### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin (Python-Rust bridge)
pip install maturin
```

#### Build and Install

```bash
# Clone the repository
git clone https://github.com/your-org/basilica.git
cd basilica

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Navigate to Python SDK directory
cd crates/basilica-sdk-python

# Build and install in development mode
maturin develop

# Or build a wheel for distribution
maturin build --release
pip install target/wheels/basilica_sdk-*.whl
```

#### Verify Installation

```bash
python3 -c "from basilica import BasilicaClient; print('✓ SDK installed successfully')"
```

**Expected output:**

```
✓ SDK installed successfully
```

---

## Your First API Call

Let's verify everything is set up correctly with a simple health check:

### Create a Test Script

```bash
# Create test script
cat > test_basilica.py << 'EOF'
#!/usr/bin/env python3
"""Quick test to verify Basilica SDK is working"""

from basilica import BasilicaClient

def main():
    # SDK auto-detects BASILICA_API_TOKEN from environment
    client = BasilicaClient()

    # Health check
    health = client.health_check()

    print("=== Basilica API Health Check ===")
    print(f"Status: {health.status}")
    print(f"Version: {health.version}")
    print(f"Healthy Validators: {health.healthy_validators}/{health.total_validators}")
    print("\n✓ SDK is working correctly!")

if __name__ == "__main__":
    main()
EOF

chmod +x test_basilica.py
```

### Run the Test

```bash
# Ensure token is set
export BASILICA_API_TOKEN="your-token-here"

# Run test
python3 test_basilica.py
```

**Expected Output:**

```
=== Basilica API Health Check ===
Status: ok
Version: 0.1.0
Healthy Validators: 5/5

✓ SDK is working correctly!
```

---

## Complete Working Example

Here's a comprehensive example demonstrating the full workflow:

### Setup SSH Key (One-time setup)

```bash
# Generate SSH key for GPU access
ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519 -N ""

# Verify key was created
ls -la ~/.ssh/basilica_ed25519*
```

### Complete Example Script

```bash
cat > basilica_complete_example.py << 'EOF'
#!/usr/bin/env python3
"""
Complete Basilica SDK Example
Demonstrates: health check, node listing, rental creation, and deployments
"""

import os
import sys
import time
from basilica import BasilicaClient

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    # Verify environment
    api_token = os.environ.get("BASILICA_API_TOKEN")
    if not api_token:
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("Generate one with: basilica tokens create")
        sys.exit(1)

    # Initialize client
    client = BasilicaClient()
    print(f"✓ Client initialized with token: {api_token[:20]}...")

    # 1. Health Check
    print_section("1. API Health Check")
    health = client.health_check()
    print(f"API Status: {health.status}")
    print(f"API Version: {health.version}")
    print(f"Validators: {health.healthy_validators}/{health.total_validators} healthy")

    # 2. List Available Nodes
    print_section("2. Available GPU Nodes")
    nodes = client.list_nodes(available=True)
    print(f"Found {len(nodes)} available nodes")

    for i, node_info in enumerate(nodes[:3], 1):
        node = node_info.node
        availability = node_info.availability
        print(f"\nNode {i}: {node.id}")
        print(f"  Location: {node.location}")
        print(f"  Uptime: {availability.uptime_percentage:.1f}%")

        for gpu in node.gpu_specs:
            print(f"  GPU: {gpu.name} ({gpu.memory_gb} GB VRAM)")

        cpu = node.cpu_specs
        print(f"  CPU: {cpu.cores} cores, {cpu.memory_gb} GB RAM")

    # 3. Start a GPU Rental (Example - commented out to avoid charges)
    print_section("3. GPU Rental Example (Demo)")
    print("To start a rental, uncomment the code below:")
    print("""
    rental = client.start_rental(
        container_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
        gpu_type="b200",
        environment={"MY_VAR": "value"},
        ports=[
            {"container_port": 8888, "host_port": 8888, "protocol": "tcp"}
        ]
    )

    print(f"Rental ID: {rental.rental_id}")
    print(f"Status: {rental.status}")

    if rental.ssh_credentials:
        print(f"SSH: {rental.ssh_credentials}")
        print(f"Connect: ssh -i ~/.ssh/basilica_ed25519 {rental.ssh_credentials}")
    """)

    # 4. Create a Deployment (Example - commented out)
    print_section("4. K8s Deployment Example (Demo)")
    print("To create a deployment, uncomment the code below:")
    print("""
    deployment = client.create_deployment(
        instance_name="my-nginx-test",
        image="nginx:latest",
        replicas=2,
        port=80,
        cpu="500m",
        memory="512Mi",
        ttl_seconds=3600  # Auto-delete after 1 hour
    )

    print(f"Deployment: {deployment.instance_name}")
    print(f"URL: {deployment.url}")
    print(f"State: {deployment.state}")
    print(f"Replicas: {deployment.replicas.desired}")

    # Wait for deployment to be ready
    while True:
        status = client.get_deployment("my-nginx-test")
        print(f"Status: {status.state}, Ready: {status.replicas.ready}/{status.replicas.desired}")

        if status.replicas.ready == status.replicas.desired:
            print("✓ Deployment is ready!")
            break

        time.sleep(5)

    # Cleanup
    client.delete_deployment("my-nginx-test")
    print("✓ Deployment deleted")
    """)

    # 5. List Existing Rentals
    print_section("5. Your Existing Rentals")
    rentals = client.list_rentals()

    if isinstance(rentals, dict) and 'rentals' in rentals:
        rental_list = rentals['rentals']
        print(f"You have {len(rental_list)} rental(s)")

        for rental in rental_list[:5]:
            print(f"\nRental: {rental.get('rental_id', 'N/A')}")
            print(f"  Status: {rental.get('status', 'N/A')}")
            print(f"  Node: {rental.get('node_id', 'N/A')}")
    else:
        print("No active rentals")

    # 6. List Existing Deployments
    print_section("6. Your Existing Deployments")
    deployments = client.list_deployments()

    print(f"You have {deployments.total} deployment(s)")

    for dep in deployments.deployments[:5]:
        print(f"\nDeployment: {dep.instance_name}")
        print(f"  State: {dep.state}")
        print(f"  URL: {dep.url}")
        print(f"  Replicas: {dep.replicas.ready}/{dep.replicas.desired}")

    print_section("Complete!")
    print("✓ All API calls successful")
    print("\nNext steps:")
    print("  - Uncomment rental example to start a GPU")
    print("  - Uncomment deployment example to deploy a container")
    print("  - Check examples/ directory for more samples")
    print("  - Read docs/GETTING-STARTED.md for details")

if __name__ == "__main__":
    main()
EOF

chmod +x basilica_complete_example.py
```

### Run the Example

```bash
# Ensure environment is set
export BASILICA_API_TOKEN="your-token-here"

# Run the example
python3 basilica_complete_example.py
```

**Expected Output:**

```
✓ Client initialized with token: basilica_1a2b3c4d5e...

============================================================
  1. API Health Check
============================================================

API Status: ok
API Version: 0.1.0
Validators: 5/5 healthy

============================================================
  2. Available GPU Nodes
============================================================

Found 12 available nodes

Node 1: node-abc123
  Location: us-east-1
  Uptime: 99.8%
  GPU: NVIDIA B200 (192 GB VRAM)
  CPU: 64 cores, 512 GB RAM
...
```

---

## Troubleshooting

### Issue: "BASILICA_API_TOKEN environment variable not set"

**Solution:**

```bash
# Check if variable is set
echo $BASILICA_API_TOKEN

# If empty, set it
export BASILICA_API_TOKEN="your-token-here"

# Or add to shell profile
echo 'export BASILICA_API_TOKEN="..."' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "Authentication error" or "Invalid API key"

**Solutions:**

1. Verify token is correct:

   ```bash
   echo $BASILICA_API_TOKEN | wc -c
   # Should be around 50-70 characters
   ```

2. Check token hasn't been revoked:

   ```bash
   basilica tokens list
   ```

3. Generate a new token:

   ```bash
   basilica tokens create fresh-token
   ```

### Issue: "Failed to create runtime" or "Connection refused"

**Solutions:**

1. Check API endpoint is reachable:

   ```bash
   curl https://api.basilica.ai/health
   ```

2. Check your network/firewall:

   ```bash
   ping api.basilica.ai
   ```

3. Try with explicit API URL:

   ```python
   client = BasilicaClient(base_url="https://api.basilica.ai")
   ```

### Issue: "SSH connection failed" when connecting to rental

**Solutions:**

1. Verify SSH key exists:

   ```bash
   ls -la ~/.ssh/basilica_ed25519*
   ```

2. Generate SSH key if missing:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519 -N ""
   ```

3. Check SSH key permissions:

   ```bash
   chmod 600 ~/.ssh/basilica_ed25519
   chmod 644 ~/.ssh/basilica_ed25519.pub
   ```

4. Use explicit SSH key path:

   ```python
   rental = client.start_rental(
       ssh_pubkey_path="~/.ssh/basilica_ed25519.pub",
       gpu_type="b200"
   )
   ```

### Issue: "Module not found: basilica"

**Solutions:**

1. Verify SDK is installed:

   ```bash
   pip list | grep basilica
   ```

2. Reinstall SDK:

   ```bash
   pip uninstall basilica-sdk
   pip install basilica-sdk
   ```

3. Check Python path:

   ```bash
   python3 -c "import sys; print('\n'.join(sys.path))"
   ```

### Issue: "Deployment stuck in Pending state"

**Solutions:**

1. Check operator logs:

   ```bash
   kubectl logs -f deployment/basilica-operator -n basilica-system
   ```

2. Check pod status:

   ```bash
   # Get your namespace from deployment response
   kubectl get pods -n user-<your-namespace>
   ```

3. Describe the pod:

   ```bash
   kubectl describe pod <pod-name> -n user-<your-namespace>
   ```

### Issue: "Too many API keys" (10 limit reached)

**Solution:**

```bash
# List all tokens
basilica tokens list

# Revoke unused tokens
basilica tokens revoke old-token-name

# Or revoke without confirmation
basilica tokens revoke old-token-name --yes
```

---

## Next Steps

### Explore Examples

Check out the `examples/` directory for more advanced usage:

```bash
cd crates/basilica-sdk-python/examples

# Start a GPU rental with custom configuration
python3 start_rental.py

# List available GPU nodes with filtering
python3 list_executors.py

# Deploy a containerized application
python3 ../../../examples/deploy_container.py
```

### Read Documentation

- **SDK Reference**: `crates/basilica-sdk-python/README.md`
- **Deployment Guide**: `examples/README-deployments.md`
- **Integration Tests**: `tests/integration/README.md`
- **Architecture**: `prompts/k3s-basilica-sdk.architecture.md`

### Join the Community

- GitHub: <https://github.com/your-org/basilica>
- Discord: <https://discord.gg/Cy7c9vPsNK>
- Documentation: <https://docs.basilica.ai>

---

## Quick Reference

### Essential Commands

```bash
# Token Management
basilica tokens create [name]          # Create API token
basilica tokens list                   # List all tokens
basilica tokens revoke [name]          # Revoke token

# Environment Setup
export BASILICA_API_TOKEN="..."        # Set token
export BASILICA_API_URL="https://..."  # Optional: custom API URL

# Python SDK
pip install basilica-sdk               # Install from PyPI
maturin develop                        # Build from source
python3 -c "from basilica import BasilicaClient"  # Verify install

# SSH Setup
ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519 -N ""  # Generate key
```

### SDK Quick Reference

```python
from basilica import BasilicaClient

client = BasilicaClient()

# Health & Discovery
health = client.health_check()
nodes = client.list_nodes(available=True, gpu_type="b200")

# Rentals
rental = client.start_rental(gpu_type="b200")
status = client.get_rental(rental.rental_id)
client.stop_rental(rental.rental_id)
rentals = client.list_rentals()

# Deployments
deployment = client.create_deployment(
    instance_name="my-app",
    image="nginx:latest",
    replicas=2
)
status = client.get_deployment("my-app")
client.delete_deployment("my-app")
deployments = client.list_deployments()
```

---

**You're all set!** Start building with Basilica's decentralized GPU network.
