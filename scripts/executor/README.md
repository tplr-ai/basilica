# Basilica Executor

GPU machine agent for container task execution with validator SSH access.

## Quick Start

```bash
# Build locally
./build.sh

# Deploy with binary mode (default)
./deploy.sh -s user@executor.example.com:22

# Deploy with systemd
./deploy.sh -s user@executor.example.com:22 -d systemd

# Deploy with docker
./deploy.sh -s user@executor.example.com:22 -d docker

# Run locally for development
docker-compose -f compose.dev.yml up
```

## Deployment Modes

The executor supports three deployment modes:

### Binary Mode (Default)

Deploys the executor binary and runs it with nohup (requires root):

```bash
./deploy.sh -s user@executor.example.com:22
./deploy.sh -s user@executor.example.com:22 -d binary
```

### Systemd Mode

Deploys the executor binary and manages it as a systemd service (runs as root):

```bash
./deploy.sh -s user@executor.example.com:22 -d systemd
```

### Docker Mode

Deploys using docker-compose with public registry images:

```bash
./deploy.sh -s user@executor.example.com:22 -d docker
```

## Files

- `Dockerfile` - Container image with NVIDIA GPU support
- `build.sh` - Build script for Docker image and binary
- `deploy.sh` - Multi-mode deployment script
- `systemd/basilica-executor.service` - Systemd service definition (runs as root)
- `compose.prod.yml` - Production docker-compose with watchtower
- `compose.dev.yml` - Development docker-compose with local build

## Configuration

The executor uses configuration files from the `config/` directory. Default is `config/executor.correct.toml`.

Specify custom config:

```bash
./deploy.sh -s user@executor.example.com:22 -c config/executor.prod.toml
```

Key settings:

- `[server]` - Port 50051 for gRPC server
- `[docker]` - Container runtime settings
- `[validator]` - SSH access configuration for validators
- `[gpu_attestor]` - GPU verification settings

## Ports

- `50051` - gRPC server for miner communication
- `50052` - Health check endpoint
- `9090` - Prometheus metrics endpoint
- `22` - SSH for validator access

## GPU Requirements

- NVIDIA GPU with CUDA support
- nvidia-docker2 installed  
- NVIDIA driver 470.57.02+

## Deployment Options

```bash
Usage: ./deploy.sh [OPTIONS]

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/executor.correct.toml)
    -f, --follow-logs                Stream logs after deployment
    --health-check                   Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -b, --veritas-binaries DIR       Directory containing veritas binaries to deploy
    -h, --help                       Show this help
```

## Examples

```bash
# Deploy with health checks and follow logs
./deploy.sh -s user@executor.example.com:22 --health-check -f

# Deploy with custom config and veritas binaries
./deploy.sh -s user@executor.example.com:22 -c config/executor.staging.toml -b ../veritas/binaries

# Deploy with systemd and health monitoring
./deploy.sh -s user@executor.example.com:22 -d systemd --health-check -f
```

## Service Management

### Binary Mode

```bash
# Check status
ssh user@executor.example.com "pgrep -f executor"

# View logs
ssh user@executor.example.com "tail -f /opt/basilica/executor.log"

# Stop service
ssh user@executor.example.com "pkill -f executor"
```

### Systemd Mode

```bash
# Check status
ssh user@executor.example.com "systemctl status basilica-executor"

# View logs
ssh user@executor.example.com "journalctl -u basilica-executor -f"

# Restart service
ssh user@executor.example.com "systemctl restart basilica-executor"
```

### Docker Mode

```bash
# Check status
ssh user@executor.example.com "cd /opt/basilica && docker-compose ps"

# View logs
ssh user@executor.example.com "cd /opt/basilica && docker-compose logs -f"

# Restart containers
ssh user@executor.example.com "cd /opt/basilica && docker-compose restart"
```

## Security Notes

- **Runs as root** for Docker container management and GPU access
- Manages validator SSH access via authorized_keys
- GPU attestation verifies hardware authenticity
- Requires privileged access for container orchestration