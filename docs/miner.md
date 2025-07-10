# Miner Guide

This guide covers running a Basilica miner node with GPU executors to provide compute resources to the network.

## Overview

The miner component manages a fleet of GPU executor machines, handling:

- Registration on the Bittensor network
- Executor fleet management via gRPC
- Task distribution and monitoring
- Serving compute requests through the Axon server
- GPU verification through Proof-of-Work challenges

## Prerequisites

- Bittensor wallet with TAO tokens
- Linux system with Docker support
- One or more GPU machines for executors
- Network connectivity between miner and executors

## Quick Start

### 1. Set Up Your Wallet

Ensure you have a Bittensor wallet configured:

```bash
# Create wallet if needed (skip if you already have one)
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

**Note**: The wallet file format has changed. The miner now supports both JSON wallet files (new format) and raw seed phrases (old format). The JSON format includes fields like `secretPhrase`, `publicKey`, `accountId`, etc.

### 2. Configure the Miner

Create a `miner.toml` configuration file:

```toml
[server]
host = "0.0.0.0"
port = 8092

[database]
url = "sqlite:./data/miner.db"
max_connections = 5
min_connections = 1
run_migrations = true

[bittensor]
wallet_name = "miner"
hotkey_name = "default"
network = "finney"  # Options: "finney", "test", or "local"
netuid = 39  # Basilica subnet (use 387 for test network)
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"  # Critical for metadata compatibility
# Network endpoints:
# finney: wss://entrypoint-finney.opentensor.ai:443
# test: wss://test.finney.opentensor.ai:443
# local: ws://127.0.0.1:9944
coldkey_name = "default"
axon_port = 8091
# external_ip = "your.external.ip.here"  # Optional

[executor_management]
health_check_interval = { secs = 30, nanos = 0 }
health_check_timeout = { secs = 10, nanos = 0 }
max_retry_attempts = 3
auto_recovery = true

# Define your executor machines
[[executor_management.executors]]
id = "executor-1"
name = "GPU Executor 1"
grpc_address = "127.0.0.1:50051"

[[executor_management.executors]]
id = "executor-2"
name = "GPU Executor 2"
grpc_address = "executor2.example.com:50051"

[validator_comms]
request_timeout = { secs = 30, nanos = 0 }
max_concurrent_sessions = 100

[validator_comms.auth]
enabled = true
method = "bittensor_signature"

[validator_comms.rate_limit]
enabled = true
requests_per_second = 10
burst_capacity = 20
window_duration = { secs = 60, nanos = 0 }

[security]
enable_mtls = false
jwt_secret = "your-secure-secret-key"
token_expiration = { secs = 3600, nanos = 0 }
allowed_validators = []
verify_signatures = true

[logging]
level = "info"
format = "pretty"

[metrics]
enabled = true
[metrics.prometheus]
enabled = true
port = 9091
```

### 3. Set Up Executors

On each GPU machine, run the executor:

```bash
# Download and run executor
./executor --server --config executor.toml
```

Executor configuration (`executor.toml`):

```toml
[server]
grpc_port = 50051
health_port = 8082

[bittensor]
enable_auth = true
jwt_secret = "your-secure-secret-key"

[system]
container_runtime = "docker"
gpu_enabled = true

[storage]
data_dir = "/opt/basilica/data"
attestation_dir = "/opt/basilica/attestations"

[logging]
level = "info"
```

### 4. Production Deployment (Recommended)

The easiest way to run a miner in production is using Docker Compose:

```bash
# Navigate to miner scripts directory
cd scripts/miner

# Copy and customize the production config
cp ../../config/miner.correct.toml ../../config/miner.toml
# Edit miner.toml with your specific settings:
# - Update external_ip and advertised_host to your public IP
# - Set your wallet_name and hotkey_name
# - Configure your executor fleet with correct IPs and SSH access
# - Choose network: "finney" for mainnet or "test" for testnet

# Ensure your Bittensor wallet exists
ls ~/.bittensor/wallets/your_miner_wallet/hotkeys/

# Create required directories
mkdir -p /var/log/basilica

# Deploy with Docker Compose (includes auto-updates and monitoring)
docker compose -f compose.prod.yml up -d

# Check status
docker compose -f compose.prod.yml ps
docker logs basilica-miner
```

This production setup includes:
- **Automatic updates** via Watchtower
- **Health monitoring** with automatic restarts
- **Persistent data storage** with named volumes
- **Proper logging** to `/var/log/basilica`
- **Network isolation** with dedicated Docker network

### 5. Alternative Deployment Methods

#### Using Build Script and Remote Deployment

```bash
# Build and deploy to remote server (see BASILICA-DEPLOYMENT-GUIDE.md)
./scripts/deploy.sh -s miner -m user@your-server:port

# Deploy with wallet sync and health checks
./scripts/deploy.sh -s miner -m user@your-server:port -w -c
```

#### Building from Source

```bash
# First, ensure metadata is up to date (recommended for production)
./scripts/generate-metadata.sh --network finney

# Build the miner using the build script
./scripts/miner/build.sh

# Or build manually
cargo build --release -p miner
```

#### Running with Docker Directly

```bash
# Build Docker image
docker build -f scripts/miner/Dockerfile -t basilica/miner .

# Run container
docker run -d \
  --name basilica-miner \
  --restart unless-stopped \
  -v ~/.bittensor:/home/basilica/.bittensor \
  -v ./config/miner.toml:/app/miner.toml:ro \
  -v miner-data:/app/data \
  -p 50051:50051 \
  -p 8091:8091 \
  -p 8080:8080 \
  basilica/miner:latest --config /app/miner.toml
```

**Important Notes**:

- The miner automatically discovers its UID from the Bittensor metagraph based on its hotkey
- UIDs are no longer hardcoded in configuration files
- Chain endpoint is auto-detected based on network type if not explicitly specified
- Ensure proper firewall configuration for ports 50051 (gRPC), 8091 (Axon), and 8080 (metrics)
- For production, use the compose.prod.yml for automatic updates and monitoring
- You must have at least one executor configured and accessible

## Advanced Configuration

### Fleet Management

The miner supports multiple deployment modes:

- **SSH Mode**: Direct SSH deployment to executor machines
- **Manual Mode**: Pre-deployed executors managed externally
- **Kubernetes Mode**: Kubernetes-based executor orchestration

### Monitoring

Monitor your miner's health and performance:

```bash
# Check miner health
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check executor status
curl http://localhost:8080/api/v1/executors
```

### Security Best Practices

1. **Secure Communication**
   - Use TLS for gRPC connections between miner and executors
   - Configure JWT authentication for executor access
   - Restrict network access with firewalls

2. **Key Management**
   - Keep hotkey secure with proper file permissions
   - Use separate wallets for different miners
   - Regularly rotate JWT secrets

3. **Resource Limits**
   - Set appropriate container resource limits
   - Monitor GPU memory usage
   - Configure task timeouts

## Troubleshooting

### Common Issues

**Executor Connection Failed**

```text
Error: Failed to connect to executor at gpu-1:50051
```

- Verify executor is running and accessible
- Check firewall rules allow port 50051
- Ensure gRPC address is correct in config

**Registration Failed**

```text
Error: Failed to serve axon on network
```

- Ensure wallet has sufficient TAO for registration
- Verify you're connected to the correct network
- Check if hotkey is already registered

**Metadata Compatibility Error**

```text
Error: failed to fetch metadata for netuid 39: RPC method error: get_metagraph - Metadata error: the generated code is not compatible with the node
```

- Regenerate metadata: `./scripts/generate-metadata.sh --network finney`
- Ensure `chain_endpoint` is specified in `[bittensor]` section
- Rebuild the miner after metadata update

**Wallet Loading Error**

```text
Error: Failed to load hotkey: Invalid format
```

- Ensure wallet file exists at `~/.bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}`
- Check if the wallet is in the correct format (JSON with secretPhrase field or raw seed phrase)
- Verify file permissions allow reading

**Database Connection Error**

```text
Error: unable to open database file
```

- Ensure the data directory exists (e.g., `mkdir -p data`)
- Check file permissions on the data directory
- Verify the database URL in config uses proper format: `sqlite:./data/miner.db`

**Executor Configuration Error**

```text
Error: At least one executor must be configured
```

- Ensure at least one executor is defined in the `[[executor_management.executors]]` section
- Verify the executor configuration syntax is correct

**Hardware Attestation Failed**

```text
Error: GPU attestation failed: No NVIDIA driver found
```

- Install NVIDIA drivers on executor machine
- Run executor with `--privileged` if using Docker
- Verify GPU is properly detected with `nvidia-smi`

**GPU PoW Challenge Failed**

```text
Error: Failed to initialize GPU PRNG kernel - CUDA kernels required
```

- Ensure CUDA is properly installed on the executor
- Verify gpu-attestor binary has access to CUDA libraries
- Check that PTX files are compiled correctly during build
- No CPU fallback is available - actual GPU hardware is required

### Logs and Debugging

Enable debug logging for detailed troubleshooting:

```toml
[logging]
level = "debug"
format = "json"
```

View logs:

```bash
# Miner logs
tail -f ./logs/miner.log

# Executor logs (on executor machine)
tail -f /opt/basilica/logs/executor.log
```

## GPU Verification

Miners must prove they possess the GPUs they claim through the GPU Proof-of-Work system:

1. **How It Works**
   - Validators send computational challenges to miners
   - Challenges require generating large matrices using a random seed
   - Miners must multiply specific matrices and compute checksums
   - Results are verified by validators with matching GPU models

2. **Requirements**
   - NVIDIA GPU with CUDA support
   - Sufficient VRAM for matrix operations (~90% utilization)
   - GPU kernels compiled during build (no CPU fallback)
   - Fast execution times (typically 50-200ms for H100)

3. **Testing GPU PoW**

   ```bash
   # Test GPU detection and challenge execution
   ./scripts/test_gpu_pow.sh
   ```

For detailed information, see [GPU Proof-of-Work Documentation](gpu_pow.md).

## Performance Optimization

1. **Network Optimization**
   - Place miner geographically close to executors
   - Use dedicated network connections
   - Enable gRPC compression for large payloads

2. **Resource Allocation**
   - Balance executor count with miner capacity
   - Monitor CPU/memory usage on miner
   - Tune verification intervals based on load

3. **GPU Performance**
   - Ensure GPUs have adequate cooling
   - Monitor VRAM usage during challenges
   - Keep CUDA drivers updated
   - Use latest gpu-attestor binary

4. **Scaling Considerations**
   - Use load balancing for multiple miners
   - Implement executor pooling for efficiency
   - Consider horizontal scaling for large fleets

## Next Steps

- Review the [Architecture Guide](architecture.md) to understand the system design
- Check the [Validator Guide](validator.md) to understand how your miner is evaluated
- Join the Basilica community for support and updates
