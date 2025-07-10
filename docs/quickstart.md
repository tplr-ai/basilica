# Quick Start Guide

This guide provides step-by-step instructions for quickly getting started with Basilica network participation.

## Deployment Options

Basilica supports multiple deployment methods:

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
cp ../../config/validator.correct.toml ../../config/validator.toml
# Edit config/validator.toml with your settings:
# - wallet_name and hotkey_name
# - external_ip (your public IP)
# - network ("finney" for mainnet, "test" for testnet)

# 3. Ensure wallet exists
ls ~/.bittensor/wallets/your_wallet/hotkeys/

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
cp ../../config/miner.correct.toml ../../config/miner.toml
# Edit config/miner.toml with your settings:
# - wallet_name and hotkey_name
# - external_ip (your public IP)
# - executor fleet configuration
# - network ("finney" for mainnet, "test" for testnet)

# 3. Deploy with auto-updates and monitoring
docker compose -f compose.prod.yml up -d

# 4. Check status
docker logs basilica-miner
```

### Executor

```bash
# 1. Navigate to executor scripts
cd scripts/executor

# 2. Prepare configuration
cp ../../config/executor.correct.toml ../../config/executor.toml
# Edit config/executor.toml with your settings:
# - managing_miner_hotkey (your miner's hotkey)
# - advertised_host (your public IP)

# 3. Deploy with GPU access and auto-updates
docker compose -f compose.prod.yml up -d

# 4. Check status
docker logs basilica-executor
```

## Option 2: Remote Deployment

Deploy to remote servers using the automated deployment script:

```bash
# Deploy all services to different servers
./scripts/deploy.sh -s all \
  -v user@validator-server:port \
  -m user@miner-server:port \
  -e user@executor-server:port \
  -w -c

# Or deploy individual services
./scripts/deploy.sh -s validator -v user@server:port -w
```

See [BASILICA-DEPLOYMENT-GUIDE.md](../BASILICA-DEPLOYMENT-GUIDE.md) for detailed deployment instructions.

## Option 3: Development Build

For development and customization:

```bash
# 1. Build components
./scripts/validator/build.sh
./scripts/miner/build.sh
./scripts/executor/build.sh

# 2. Prepare configuration
cp config/validator.correct.toml config/validator.toml
cp config/miner.correct.toml config/miner.toml
cp config/executor.correct.toml config/executor.toml
# Edit configurations as needed

# 3. Run services
./validator --config config/validator.toml
./miner --config config/miner.toml
./executor --config config/executor.toml
```

## Monitoring Your Deployment

### Check Service Status

```bash
# Check if containers are running
docker ps

# View logs
docker logs basilica-validator
docker logs basilica-miner
docker logs basilica-executor

# Check health endpoints
curl http://localhost:8080/health  # validator
curl http://localhost:8080/health  # miner
curl http://localhost:8080/health  # executor
```

### Access Monitoring Dashboard

If monitoring is enabled (automatic with production compose files):

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

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

# Copy from existing wallet if needed
cp ~/.bittensor/wallets/source/hotkeys/default ~/.bittensor/wallets/target/hotkeys/default
```

### Network Connection Issues

```bash
# Test network connectivity
ping test.finney.opentensor.ai

# Check firewall rules
sudo ufw status

# Verify port configuration matches your setup
```

## Next Steps

Choose your role and dive deeper:

- **[Validator Guide](validator.md)** - Detailed validator setup and operation
- **[Miner Guide](miner.md)** - Comprehensive miner management and fleet operations  
- **[Executor Guide](executor.md)** - GPU executor deployment and monitoring
- **[Monitoring Guide](monitoring.md)** - Advanced monitoring and alerting setup
- **[Architecture Guide](architecture.md)** - Understand the system design

## Support

- Check the individual component guides for detailed troubleshooting
- Review the [BASILICA-DEPLOYMENT-GUIDE.md](../BASILICA-DEPLOYMENT-GUIDE.md) for production deployment best practices
- Join the Basilica community for support and updates