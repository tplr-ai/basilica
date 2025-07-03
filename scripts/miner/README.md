# Basilica Miner

Bittensor neuron that manages executor fleets and handles validator communications.

## Quick Start

```bash
# Build
./build.sh

# Deploy
./deploy.sh user@host port

# Run locally
docker-compose -f compose.dev.yml up
```

## Files

- `Dockerfile` - Container image definition
- `build.sh` - Build script for Docker image
- `deploy.sh` - Remote deployment script
- `compose.prod.yml` - Production docker-compose with watchtower
- `compose.dev.yml` - Development docker-compose with local build
- `.env.example` - Environment variables template

## Configuration

Copy and edit the configuration:
```bash
cp ../../config/miner.toml.example ../../config/miner.toml
```

Key settings:
- `[bittensor]` - Wallet name, hotkey, network (finney/test/local)
- `[server]` - Port 50051 for gRPC
- `[executor_management]` - Configure executor fleet
- `[database]` - PostgreSQL or SQLite connection

## Ports

- `50051` - gRPC server for validator requests
- `8091` - Bittensor axon port
- `8080` - Metrics endpoint

## Environment Variables

Create `.env` from `.env.example`:
```bash
BITTENSOR_WALLET_NAME=your_wallet
BITTENSOR_HOTKEY_NAME=your_hotkey
POSTGRES_PASSWORD=secure_password
JWT_SECRET=your_secret
MINER_PUBLIC_IP=your_ip
```

## Commands

```bash
# Status
docker exec basilica-miner miner status

# List executors
docker exec basilica-miner miner executor list

# Database operations
docker exec basilica-miner miner database status
```

## Deployment

```bash
# Deploy to production
./deploy.sh root@51.159.160.71 55960

# Check logs
ssh root@51.159.160.71 -p 55960 'cd /opt/basilica && docker-compose logs -f miner'
```