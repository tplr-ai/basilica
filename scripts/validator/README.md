# Basilica Validator

Bittensor neuron for verification and scoring of miners and executors.

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
cp ../../config/validator.toml.example ../../config/validator.toml
```

Key settings:
- `[bittensor]` - Wallet name, hotkey, network (finney/test/local)
- `[server]` - Port 50053 for gRPC
- `[verification]` - GPU attestation and benchmark settings
- `[ssh_validation]` - SSH connection parameters

## Ports

- `50053` - gRPC server
- `3000` - Public API
- `8080` - Metrics endpoint

## Environment Variables

Create `.env` from `.env.example`:
```bash
BITTENSOR_WALLET_NAME=your_wallet
BITTENSOR_HOTKEY_NAME=your_hotkey
POSTGRES_PASSWORD=secure_password
JWT_SECRET=your_secret
VALIDATOR_PUBLIC_IP=your_ip
```

## Commands

```bash
# Status
docker exec basilica-validator validator status

# List miners
docker exec basilica-validator validator miner list

# Run verification
docker exec basilica-validator validator verify --miner-uid 123
```

## Deployment

```bash
# Deploy to production
./deploy.sh root@51.159.130.131 41199

# Check logs
ssh root@51.159.130.131 -p 41199 'cd /opt/basilica && docker-compose logs -f validator'
```

## SSH Keys

Generate validator SSH keys:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/validator_ssh -N ""
```

The public key must be added to executor machines for verification access.