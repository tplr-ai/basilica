# Docker Support for Basilica

This directory contains Docker-related files for building and running Basilica services.

## Files

- `miner.Dockerfile` - Dockerfile for building the miner service
- `executor.Dockerfile` - Dockerfile for building the executor service (with GPU support)
- `build.sh` - Build script for creating Docker images
- `prometheus.yml` - Prometheus configuration for metrics collection

## Quick Build

```bash
# Build all images
./docker/build.sh

# Build specific service
./docker/build.sh miner
./docker/build.sh executor

# Build with custom tag
./docker/build.sh --tag v1.0.0 all
```

## Running with Docker Compose

### Production
```bash
# Uses docker-compose.yml in project root
docker-compose up -d
```

### Development
```bash
# Uses docker-compose.dev.yml with additional services
docker-compose -f docker-compose.dev.yml up -d
```

## Individual Container Commands

### Miner
```bash
docker run -d \
  --name basilica-miner \
  -p 8080:8080 \
  -v $(pwd)/config:/config \
  -v miner-data:/var/lib/basilica \
  -e BASILCA_CONFIG_FILE=/config/miner.toml \
  basilica-miner:latest
```

### Executor (with GPU)
```bash
docker run -d \
  --name basilica-executor \
  --gpus all \
  --privileged \
  -p 50051:50051 \
  -v $(pwd)/config:/config \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e BASILCA_CONFIG_FILE=/config/executor.toml \
  basilica-executor:latest
```

## GPU Support

The executor Dockerfile is based on NVIDIA CUDA runtime image and requires:
- NVIDIA GPU drivers installed on host
- NVIDIA Container Toolkit
- Docker configured with GPU support

Verify GPU support:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
```

## Development Features

The development compose file includes:
- PostgreSQL database
- Prometheus metrics server
- Grafana for visualization
- gRPC UI for API testing
- Volume mounts for source code
- Debug logging enabled

Access development services:
- Miner API: http://localhost:8080
- Executor gRPC: localhost:50051
- gRPC UI: http://localhost:8081
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- PostgreSQL: localhost:5432

## Troubleshooting

### Build Issues
- Ensure you have generated the validator key: `just gen-key`
- Check Docker daemon is running
- Verify sufficient disk space

### GPU Issues
- Check NVIDIA drivers: `nvidia-smi`
- Verify Docker GPU support: `docker info | grep nvidia`
- Ensure executor container has `--gpus all` flag

### Network Issues
- Check port availability before running
- Verify firewall rules allow Docker networks
- Use `docker network ls` to inspect networks