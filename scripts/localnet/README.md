# Basilica Localnet Setup

This directory contains a complete Docker Compose setup for running Basilica services in a local test environment.

## Quick Start

```bash
# One-time setup (creates wallets and SSH keys)
./setup.sh

# Check service status
./test-services.sh

# Stop all services
docker compose down
```

## Development Workflow

When making code changes:

```bash
# Rebuild and restart all services
./restart.sh

# Restart specific service(s)
./restart.sh miner
./restart.sh miner validator

# Restart without rebuilding (config changes only)
./restart.sh --no-build

# Only rebuild without starting
./restart.sh --build-only
```

## Services

| Service | Port(s) | Description |
|---------|---------|-------------|
| Subtensor (Alice) | 9944 | Local Bittensor blockchain |
| Executor | 50052 (gRPC), 8082 (metrics) | GPU machine agent |
| Miner | 8092 (gRPC), 8090 (metrics) | Manages executor fleet |
| Validator | 50053 (gRPC), 3002 (API) | Verification service |
| Redis | 6379 | Cache and session storage |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics visualization |

## Configuration

All services are pre-configured to work without Bittensor registration:

- **Miner**: `skip_registration = true` in `config/miner-localnet.toml`
- **Validator**: `--local-test` flag in `compose.yml`
- **Network**: All services use `netuid = 1` (default subnet)
- **Chain**: `ws://subtensor-alice:9944`

## Directory Structure

```
scripts/localnet/
├── compose.yml           # Docker Compose configuration
├── setup.sh             # Main setup script
├── setup-wallets.sh     # Wallet creation script
├── test-services.sh     # Service health check script
├── ssh-keys/            # Static SSH keys (auto-generated)
│   └── validator_keys/
└── wallets/             # Bittensor wallets (auto-generated)
```

## Scripts

- **setup.sh**: Complete setup including wallets, SSH keys, and service startup
- **restart.sh**: Rebuild and restart services after code changes
- **setup-wallets.sh**: Creates all required Bittensor wallets
- **test-services.sh**: Checks health of all running services
- **fund-and-register.sh**: Reference script (not needed for localnet)

## Known Limitations

1. **No Registration**: Services bypass Bittensor registration for local testing
2. **No Funding**: Wallets have no balance (not needed with skip_registration)
3. **Public API**: Currently has issues with localnet (metadata compatibility and route conflicts)

## GPU Support

The executor is configured to use GPU if available. If you have NVIDIA GPU:
- Install NVIDIA drivers and NVIDIA Container Toolkit
- Docker will automatically pass through GPU access
- If no GPU is available, executor will log an info message and continue without GPU

## Troubleshooting

If services fail to start:

1. Check Docker logs: `docker logs <container-name>`
2. Ensure wallets exist: `ls -la wallets/`
3. Verify SSH keys: `ls -la ssh-keys/validator_keys/`
4. Check service configs in `../../config/*-localnet.toml`