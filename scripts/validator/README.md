# Basilica Validator

Bittensor neuron for verification and scoring of miners and executors.

## Quick Start

```bash
# Build locally
./build.sh

# Deploy with binary mode (default)
./deploy.sh -s root@validator.example.com:9001

# Deploy with systemd
./deploy.sh -s root@validator.example.com:9001 -d systemd

# Deploy with docker
./deploy.sh -s root@validator.example.com:9001 -d docker

# Run locally for development
docker-compose -f compose.dev.yml up
```

## Deployment Modes

The validator supports three deployment modes:

### Binary Mode (Default)
Deploys the validator binary and runs it with nohup:
```bash
./deploy.sh -s root@validator.example.com:9001
./deploy.sh -s root@validator.example.com:9001 -d binary
```

### Systemd Mode
Deploys the validator binary and manages it as a systemd service:
```bash
./deploy.sh -s root@validator.example.com:9001 -d systemd
```

### Docker Mode
Deploys using docker-compose with public registry images:
```bash
./deploy.sh -s root@validator.example.com:9001 -d docker
```

## Files

- `Dockerfile` - Container image definition
- `build.sh` - Build script for Docker image and binary
- `deploy.sh` - Multi-mode deployment script
- `systemd/basilica-validator.service` - Systemd service definition
- `compose.prod.yml` - Production docker-compose with watchtower
- `compose.dev.yml` - Development docker-compose with local build

## Configuration

The validator uses configuration files from the `config/` directory. Default is `config/validator.correct.toml`.

Specify custom config:
```bash
./deploy.sh -s root@validator.example.com:9001 -c config/validator.prod.toml
```

Key settings:
- `[bittensor]` - Wallet name, hotkey, network (finney/test/local), netuid 387
- `[server]` - Port 8081 for API server
- `[verification]` - GPU attestation and benchmark settings
- `[ssh_validation]` - SSH connection parameters for executor validation

## Ports

- `8081` - HTTP API server
- `9091` - Bittensor Axon port
- `9090` - Prometheus metrics endpoint

## Deployment Options

```bash
Usage: ./deploy.sh [OPTIONS]

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/validator.correct.toml)
    -w, --sync-wallets               Sync local wallets to remote server
    -f, --follow-logs                Stream logs after deployment
    --health-check                   Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -b, --veritas-binaries DIR       Directory containing veritas binaries to deploy
    -h, --help                       Show this help
```

## Examples

```bash
# Deploy with wallet sync and health checks
./deploy.sh -s root@validator.example.com:9001 -w --health-check

# Deploy with custom config and follow logs
./deploy.sh -s root@validator.example.com:9001 -c config/validator.staging.toml -f

# Deploy with veritas binaries
./deploy.sh -s root@validator.example.com:9001 -b ../veritas/binaries

# Deploy with systemd and health monitoring
./deploy.sh -s root@validator.example.com:9001 -d systemd --health-check -f
```

## Service Management

### Binary Mode
```bash
# Check status
ssh root@validator.example.com -p 9001 "pgrep -f validator"

# View logs
ssh root@validator.example.com -p 9001 "tail -f /opt/basilica/validator.log"

# Stop service
ssh root@validator.example.com -p 9001 "pkill -f validator"
```

### Systemd Mode
```bash
# Check status
ssh root@validator.example.com -p 9001 "systemctl status basilica-validator"

# View logs
ssh root@validator.example.com -p 9001 "journalctl -u basilica-validator -f"

# Restart service
ssh root@validator.example.com -p 9001 "systemctl restart basilica-validator"
```

### Docker Mode
```bash
# Check status
ssh root@validator.example.com -p 9001 "cd /opt/basilica && docker-compose ps"

# View logs
ssh root@validator.example.com -p 9001 "cd /opt/basilica && docker-compose logs -f"

# Restart containers
ssh root@validator.example.com -p 9001 "cd /opt/basilica && docker-compose restart"
```

## SSH Keys

Generate validator SSH keys for executor access:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/validator_ssh -N ""
```

The public key must be added to executor machines for verification access.