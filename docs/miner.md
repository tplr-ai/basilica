# Miner Guide

This guide covers running a Basilica miner node with GPU executors to provide compute resources to the network.

## Overview

The miner component manages a fleet of GPU executor machines, handling:

- Registration on the Bittensor network
- Executor fleet management via gRPC
- Task distribution and monitoring
- Serving compute requests through the Axon server
- GPU verification through Proof-of-Work challenges
- Automatic executor identity management with dual identifier system (UUID + HUID)

## Prerequisites

- **Bittensor wallet** with TAO tokens registered on the subnet
- **Miner server**: Linux system with healthy network (32+ CPU cores, 64GB+ RAM recommended, no GPU required)
- **Executor machines**: One or more servers with:
  - NVIDIA GPU: H100, H200, or B200
  - NVIDIA CUDA drivers version >12.8
  - Docker support
- **Network connectivity** between miner and executors

## Quick Start

### 1. Set Up Your Wallet

Ensure you have a Bittensor wallet configured:

```bash
# Create wallet if needed (skip if you already have one)
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### 2. Configure the Miner

Copy and customize the miner configuration:

```bash
# Copy the example configuration
cp config/miner.correct.toml miner.toml
# For production deployment, you can also use:
# cp config/miner.prod.toml miner.toml

# Edit the configuration with your settings
vim miner.toml
```

Key parameters to customize in your `miner.toml`:

- **[bittensor]**:
  - `wallet_name` and `hotkey_name`: Your Bittensor wallet credentials
  - `network`: Choose "finney" for mainnet or "test" for testnet
  - `netuid`: Use 39 for mainnet, 387 for testnet
  - `external_ip`: Your public IP address

- **[[executor_management.executors]]**: Configure your executor machines
  - `grpc_address`: IP and port of each executor (e.g., "192.168.1.10:50051")
  - `host`, `port`: Individual settings for the executor
  - `ssh_username`, `ssh_port`: SSH access credentials

- **[validator_assignment]**: Choose your validator assignment strategy
  - `strategy`: "highest_stake" (recommended) or "round_robin"
  - `min_stake_threshold`: Minimum stake in TAO (default: 6000)
  - `validator_hotkey`: Optional - specify a preferred validator

For complete configuration options and documentation, see the example files:
- `config/miner.correct.toml` - Standard configuration template
- `config/miner.prod.toml` - Production-ready configuration

### Validator Assignment Strategy

The miner supports assignment of executors to validators based on these strategies:

- **`highest_stake`** (default): Assigns executors to a single validator based on stake
  - If `validator_hotkey` is specified in config, assigns to that validator (if above threshold and online)
  - Otherwise, assigns to the highest staked validator above the threshold
  - Configurable minimum stake threshold (default: 6000 TAO)
  - Only considers validators that are online (have axon endpoints)
  - Increases security by working only with the most invested validators

- **`round_robin`**: Distributes executors evenly across all eligible validators
  - Useful for testing or when stake-based assignment is not desired

Configuration example for specific validator selection:

```toml
[validator_assignment]
enabled = true
strategy = "highest_stake"
min_stake_threshold = 6000.0
validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"  # Optional
```

When validator assignment is disabled (`enabled = false`), all executors are made available to all validators. For production deployments, it's recommended to enable assignment with the `highest_stake` strategy to ensure your resources are allocated to the most reliable validators.

### 3. Set Up Executors

**IMPORTANT**: Executors must be started BEFORE the miner.

On each GPU machine, run the executor with sudo:

```bash
# Run executor (requires sudo)
sudo ./basilica-executor --server --config executor.toml
```

Copy and customize the executor configuration:

```bash
# Copy the executor configuration template
cp config/executor.correct.toml executor.toml

# Edit with your settings
vim executor.toml
```

Key parameters to customize in `executor.toml`:
- `jwt_secret`: Must match the miner's JWT secret for authentication
- `grpc_port`: Port for gRPC communication (default: 50051)
- Ensure Docker and GPU settings match your hardware setup

See `config/executor.correct.toml` for all configuration options.

### 4. Start the Miner

After executors are running, start the miner with sudo:

```bash
# Run miner (requires sudo, run AFTER executors are started)
sudo ./basilica-miner --config miner.toml
```

You should see the miner sending health heartbeats to your executors. Wait approximately 2 hours for discovery on the network.

Monitor your deployment at: https://basilica-grafana.tplr.ai/

### 5. Production Deployment (Recommended)

The easiest way to run a miner in production is using Docker Compose:

```bash
# Navigate to miner scripts directory
cd scripts/miner

# Copy and customize the production config
cp ../../config/miner.correct.toml /opt/basilica/config/miner.toml
# Edit miner.toml with your specific settings:
# - Update external_ip and advertised_host to your public IP
# - Set your wallet_name and hotkey_name
# - Configure your executor fleet with correct IPs and SSH access
# - Choose network: "finney" for mainnet or "test" for testnet

# Ensure your Bittensor wallet exists
ls ~/.bittensor/wallets/your_miner_wallet/hotkeys/

# Create required directories
mkdir -p /opt/basilica/config
mkdir -p /opt/basilica/data
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

### 6. Alternative Deployment Methods

#### Using Deploy Script

The deploy.sh script simplifies deployment to remote servers:

```bash
# Deploy miner with wallet sync
./scripts/deploy.sh -s miner -m user@host -w

# Deploy executor
./scripts/deploy.sh -s executor -e user@host
```

#### Building from Source (Recommended)

```bash
# Build miner binary
./scripts/miner/build.sh

# Build executor binary
./scripts/executor/build.sh

# Note: The build scripts handle all necessary steps including metadata generation
```

#### Running with Docker Directly

```bash
# Build Docker image
docker build -f scripts/miner/Dockerfile -t basilica/miner .

# Run container
docker run -d \
  --name basilica-miner \
  --restart unless-stopped \
  -v ~/.bittensor:/root/.bittensor \
  -v /opt/basilica/config/miner.toml:/opt/basilica/config/miner.toml:ro \
  -v /opt/basilica/data:/opt/basilica/data \
  -v ~/.ssh:/opt/basilica/keys:ro \
  -v /var/log/basilica:/var/log/basilica \
  -p 8080:8080 \
  -p 9090:9090 \
  basilica/miner:latest --config /opt/basilica/config/miner.toml
```

**Important Notes**:

- Ensure proper firewall configuration for ports 8080 (server/API) and 9090 (metrics)
- For production, use the compose.prod.yml for automatic updates and monitoring
- You must have at least one executor configured and accessible

## Advanced Configuration

### Fleet Management

The miner supports multiple deployment modes:

- **SSH Mode**: Direct SSH deployment to executor machines
- **Manual Mode**: Pre-deployed executors managed externally

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
