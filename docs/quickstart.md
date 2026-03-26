# Quick Start Guide

This guide provides step-by-step instructions for quickly getting started with Basilica network participation.

## Deployment Options

Basilica supports two primary roles with multiple deployment methods:

**Roles:**

- **Validator** - Verifies GPU availability and performance, sets weights
- **Miner** - Orchestrates validator access to GPU nodes via SSH

**Deployment Methods:**

1. **Production Docker Compose** (Recommended) - Fully automated with monitoring
2. **Manual Build and Deploy** - For development and customization
3. **Remote Deployment** - Automated deployment to remote servers

## Key Features

- **Dynamic UID Discovery**: Services automatically discover their UID from the Bittensor metagraph
- **Auto Network Detection**: Chain endpoints are automatically configured based on network type
- **Flexible Wallet Support**: Works with both JSON wallet files and raw seed phrases
- **Production Ready**: Includes monitoring, auto-updates, and health checks

## Prerequisites

- **Docker and Docker Compose** (for production deployment)
- **Bittensor wallet** with sufficient TAO for staking
- **Linux server** with internet connectivity
- **Hardware requirements** vary by role (see individual guides)

## Option 1: Production Deployment (Recommended)

This is the fastest way to get started with production-ready deployment.

### Validator

```bash
# 1. Navigate to validator scripts
cd scripts/validator

# 2. Prepare configuration
cp ../../config/validator.toml.example /opt/basilica/config/validator.toml
# Edit /opt/basilica/config/validator.toml with your settings:
# - wallet_name and hotkey_name
# - external_ip (your public IP)
# - network ("finney" for mainnet, "test" for testnet)
# - netuid (39 for mainnet, 387 for testnet)

# 3. Ensure wallet exists and create directories
ls ~/.bittensor/wallets/your_wallet/hotkeys/
mkdir -p /opt/basilica/config /opt/basilica/data /var/log/basilica

# 4. Deploy with auto-updates and monitoring
docker compose -f compose.prod.yml up -d

# 5. Check status
docker logs basilica-validator
```

### Miner

```bash
# 1. Navigate to miner scripts
cd scripts/miner

# 2. Prepare configuration
cp ../../config/miner.toml.example /opt/basilica/config/miner.toml
# Edit /opt/basilica/config/miner.toml with your settings:
# - wallet_name and hotkey_name
# - external_ip (your public IP)
# - node_management.nodes (GPU node SSH endpoints)
# - network ("finney" for mainnet, "test" for testnet)
# - netuid (39 for mainnet, 387 for testnet)

# 3. Create directories and set up SSH key for GPU node access
mkdir -p /opt/basilica/config /opt/basilica/data /var/log/basilica
ssh-keygen -t ed25519 -f ~/.ssh/miner_node_key -N ""

# 4. Deploy key to your GPU nodes
ssh-copy-id -i ~/.ssh/miner_node_key.pub basilica@<gpu_node_ip>

# 5. Deploy with auto-updates and monitoring
docker compose -f compose.prod.yml up -d

# 6. Check status
docker logs basilica-miner
```

**Note**: GPU nodes require SSH access configuration. See [Miner Guide](miner.md) for detailed GPU node setup.

## Option 2: Remote Deployment

Deploy to remote servers using the automated deployment script:

```bash
# Deploy individual services to remote servers
./scripts/validator/deploy.sh -s user@validator-server:port -w --health-check
./scripts/miner/deploy.sh -s user@miner-server:port -w --health-check
```

## Option 3: Development Build

For development and customization:

```bash
# 1. Build components using the build scripts
./scripts/validator/build.sh
./scripts/miner/build.sh

# 2. Prepare configuration
cp config/validator.toml.example config/validator.toml
cp config/miner.toml.example config/miner.toml
# Edit configurations with your settings

# 3. Run services
./validator --config config/validator.toml start
./miner --config config/miner.toml
```

## Network Configuration

### Mainnet (Finney)

```toml
[bittensor]
network = "finney"
netuid = 39
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
```

### Testnet

```toml
[bittensor]
network = "test"
netuid = 387
chain_endpoint = "wss://test.finney.opentensor.ai:443"
```

## Monitoring Your Deployment

### Check Service Status

```bash
# Check if containers are running
docker ps

# View logs
docker logs basilica-validator
docker logs basilica-miner

# Check health endpoints
curl http://localhost:8080/health    # validator API
curl http://localhost:9090/metrics   # miner metrics
```

### Access Monitoring Dashboard

If monitoring is enabled (automatic with production compose files):

- **Grafana**: <http://localhost:3000> (admin/admin)
- **Prometheus**: <http://localhost:9090>

### Metrics Endpoints

```bash
# Validator metrics (default port 9090)
curl http://localhost:9090/metrics

# Miner metrics (default port 9090)
curl http://localhost:9090/metrics
```

## Common Issues

### Container Won't Start

```bash
# Check logs for specific errors
docker logs container-name

# Common fixes:
# 1. Check configuration file syntax
# 2. Ensure wallet files exist
# 3. Check port conflicts
# 4. Verify permissions on mounted volumes
```

### Wallet Not Found

```bash
# Ensure wallet exists
ls ~/.bittensor/wallets/your_wallet/hotkeys/

# Verify wallet name matches config
grep wallet_name /opt/basilica/config/validator.toml
```

### Network Connection Issues

```bash
# Test network connectivity
ping test.finney.opentensor.ai

# Check firewall rules
sudo ufw status

# Verify port configuration matches your setup
# Validator: 8080 (API), 9090 (metrics), axon_port (Bittensor)
# Miner: 50051 (gRPC), 8091 (axon), 9090 (metrics)
netstat -tlnp | grep -E "8080|50051|8091|9090"
```

### SSH Connection Issues (Miner)

```bash
# Test SSH access to GPU nodes
ssh -i ~/.ssh/miner_node_key basilica@<gpu_node_ip>

# Check SSH key permissions
chmod 600 ~/.ssh/miner_node_key
chmod 644 ~/.ssh/miner_node_key.pub
```

## Next Steps

Choose your role and dive deeper:

- **[Validator Guide](validator.md)** - Detailed validator setup and operation
- **[Miner Guide](miner.md)** - Comprehensive miner management and GPU node operations
- **[Architecture Guide](architecture.md)** - Understand the system design
- **[Monitoring Guide](monitoring.md)** - Advanced monitoring and alerting setup
- **[Scoring and Weights](scoring-and-weights.md)** - Understand incentive mechanisms

## Support

- Check the individual component guides for detailed troubleshooting
- [GitHub](https://github.com/one-covenant/basilica)
- [Discord](https://discord.gg/Cy7c9vPsNK)
- [Website](https://www.basilica.ai/)
