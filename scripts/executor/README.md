# Basilica Executor

GPU machine agent for container task execution with validator SSH access.

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

- `Dockerfile` - Container image with NVIDIA GPU support
- `build.sh` - Build script for Docker image and binary extraction
- `deploy.sh` - Remote deployment script
- `compose.prod.yml` - Production docker-compose with watchtower
- `compose.dev.yml` - Development docker-compose with local build

## Configuration

Copy and edit the configuration:
```bash
cp ../../config/executor.toml.example ../../config/executor.toml
```

Key settings:
- `[server]` - Port 50052 for gRPC
- `[docker]` - Container runtime settings
- `[validator]` - SSH access configuration
- `[gpu_attestor]` - GPU verification settings

## Ports

- `50052` - gRPC server
- `8080` - Metrics endpoint
- `22222` - SSH for validator access

## GPU Requirements

- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- NVIDIA driver 470.57.02+

## Commands

```bash
# System info
docker exec basilica-executor executor system info

# Validator access
docker exec basilica-executor executor validator list

# Container management
docker exec basilica-executor executor container list
```

## Deployment

```bash
# Deploy to GPU server
./deploy.sh root@64.247.196.98 9001

# Verify GPU
ssh root@64.247.196.98 -p 9001 'nvidia-smi'

# Check logs
ssh root@64.247.196.98 -p 9001 'cd /opt/basilica && docker-compose logs -f executor'
```

## Security Notes

- Runs privileged for Docker-in-Docker
- Manages validator SSH access via system users
- GPU attestation verifies hardware authenticity