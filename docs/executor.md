# Executor Guide

This guide covers deploying and managing Basilica executor nodes that provide GPU compute resources to the network.

## Overview

The executor component is the workhouse of the Basilica network, providing:

- **GPU Computing**: Executes computational tasks using NVIDIA GPUs
- **Container Management**: Isolated task execution using Docker containers
- **Hardware Attestation**: Cryptographic proof of GPU capabilities via gpu-attestor
- **SSH Access**: Secure remote access for validators to perform verification
- **System Monitoring**: Real-time GPU, CPU, and memory monitoring

## Prerequisites

- **NVIDIA GPU** with CUDA support (required for GPU PoW challenges)
- **Docker** with GPU runtime support (nvidia-container-toolkit)
- **SSH server** running for validator access
- **Linux server** with sufficient resources
- **Network connectivity** for gRPC communication

## Hardware Requirements

### Minimum Specifications
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **Network**: 1Gbps connection

### Recommended Specifications
- **GPU**: NVIDIA H100/A100 with 40GB+ VRAM
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps connection

## Quick Start

### 1. Prepare Configuration

Create your executor configuration:

```bash
# Copy the production-ready config template
cp config/executor.correct.toml config/executor.toml

# Edit configuration for your environment
nano config/executor.toml
```

Key configuration parameters to customize:

```toml
# Update with your miner's hotkey that will manage this executor
managing_miner_hotkey = "YOUR_MINER_HOTKEY"

[server]
host = "0.0.0.0"
port = 50051
advertised_host = "YOUR_PUBLIC_IP"
advertised_port = 50051

[docker]
enable_gpu_passthrough = true
max_concurrent_containers = 10

[validator]
enabled = true
ssh_port = 22  # SSH port for validator access

[advertised_endpoint]
grpc_endpoint = "http://YOUR_PUBLIC_IP:50051"
ssh_endpoint = "ssh://YOUR_PUBLIC_IP:22"
health_endpoint = "http://YOUR_PUBLIC_IP:50052/health"
```

### 2. System Setup

Install required dependencies:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu20.04 nvidia-smi
```

### 3. Production Deployment (Recommended)

The recommended way to run an executor in production is using Docker Compose:

```bash
# Navigate to executor scripts directory
cd scripts/executor

# Copy and customize the production config
cp ../../config/executor.correct.toml ../../config/executor.toml
# Edit executor.toml with your specific settings:
# - Update advertised_host to your public IP
# - Set managing_miner_hotkey to your miner's hotkey
# - Configure resource limits based on your hardware

# Create required directories
mkdir -p /var/log/basilica

# Deploy with Docker Compose (includes auto-updates and monitoring)
docker compose -f compose.prod.yml up -d

# Check status
docker compose -f compose.prod.yml ps
docker logs basilica-executor
```

This production setup includes:
- **Automatic updates** via Watchtower
- **GPU access** with full device passthrough
- **Privileged container** for Docker-in-Docker operations
- **Health monitoring** with automatic restarts
- **Persistent data storage** with named volumes
- **SSH access** on port 22222 for validators

### 4. Alternative Deployment Methods

#### Using Build Script and Remote Deployment

```bash
# Build and deploy to remote server (see BASILICA-DEPLOYMENT-GUIDE.md)
./scripts/deploy.dev.sh -s executor -e user@your-server:port

# Deploy with health checks
./scripts/deploy.dev.sh -s executor -e user@your-server:port -c
```

#### Building from Source

```bash
# Build the executor using the build script
./scripts/executor/build.sh

# Or build manually
cargo build --release -p executor
```

#### Running with Docker Directly

```bash
# Build Docker image
docker build -f scripts/executor/Dockerfile -t basilica/executor .

# Run container (requires privileged mode for GPU and Docker access)
docker run -d \
  --name basilica-executor \
  --restart unless-stopped \
  --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ./config/executor.toml:/app/executor.toml:ro \
  -v executor-data:/app/data \
  -v /dev:/dev:rw \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 50052:50052 \
  -p 8080:8080 \
  -p 22222:22 \
  basilica/executor:latest --config /app/executor.toml
```

**Important Notes**:

- Executor must run in **privileged mode** for container management and GPU access
- Ensure proper firewall configuration for ports 50052 (gRPC), 8080 (metrics), and 22222 (SSH)
- For production, use the compose.prod.yml for automatic updates and monitoring
- GPU drivers and Docker GPU runtime must be properly configured

## Configuration Details

### GPU Configuration

Enable GPU monitoring and passthrough:

```toml
[system]
enable_gpu_monitoring = true
max_gpu_memory_usage = 90.0

[docker]
enable_gpu_passthrough = true
```

### Resource Limits

Configure container resource limits:

```toml
[docker.resource_limits]
memory_bytes = 8589934592      # 8GB RAM
cpu_cores = 4.0                # 4 CPU cores
gpu_memory_bytes = 4294967296  # 4GB GPU VRAM
disk_io_bps = 104857600        # 100MB/s disk I/O
network_bps = 104857600        # 100MB/s network
```

### Security Configuration

Configure validator access and security:

```toml
[validator]
enabled = true
strict_ssh_restrictions = false
ssh_port = 22

[validator.access_config.hotkey_verification]
enabled = true
challenge_timeout_seconds = 300
max_signature_attempts = 3

[validator.access_config.rate_limits]
ssh_requests_per_minute = 10
api_requests_per_minute = 100
```

### Network Configuration

Configure container network isolation:

```toml
[docker.network_config]
enable_isolation = true
allow_internet = false
dns_servers = ["8.8.8.8", "8.8.4.4"]
```

## Monitoring

### Health Checks

Monitor executor health:

```bash
# Check executor health
curl http://localhost:8080/health

# Check gRPC server status
grpcurl -plaintext localhost:50052 health.v1.Health/Check

# View system metrics
curl http://localhost:8080/metrics
```

### GPU Monitoring

Monitor GPU utilization:

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor running containers
docker ps --filter "ancestor=basilica/executor"
```

### Logs

View executor logs:

```bash
# Container logs
docker logs basilica-executor -f

# System logs
tail -f /var/log/basilica/executor.log

# Debug mode
docker run --rm -it basilica/executor --config /app/executor.toml --log-level debug
```

## Verification Process

### How Validators Verify Executors

1. **SSH Connection**: Validators connect via SSH using provided credentials
2. **Hardware Attestation**: Execute gpu-attestor to verify GPU specifications
3. **GPU PoW Challenge**: Run computational challenges to prove GPU capabilities
4. **Performance Testing**: Measure task execution times and accuracy
5. **Availability Check**: Monitor uptime and responsiveness

### GPU Proof-of-Work

Executors must pass GPU PoW challenges:

```bash
# Test GPU PoW manually
./gpu-attestor --seed 12345 --matrix-dim 256 --num-matrices 1000

# Expected output:
# GPU Model: NVIDIA H100
# Challenge completed in 150ms
# Checksum: a1b2c3d4e5f6...
```

See [GPU Proof-of-Work Documentation](gpu_pow.md) for details.

## Troubleshooting

### Common Issues

**GPU Not Detected**

```text
Error: No NVIDIA GPU found
```

- Install NVIDIA drivers: `sudo apt install nvidia-driver-525`
- Install Docker GPU runtime: `sudo apt install nvidia-container-toolkit`
- Restart Docker: `sudo systemctl restart docker`
- Verify: `docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu20.04 nvidia-smi`

**Container Startup Failed**

```text
Error: Failed to start container
```

- Check privileged mode is enabled
- Verify Docker socket access: `ls -la /var/run/docker.sock`
- Check resource limits in configuration
- Review executor logs for specific errors

**SSH Access Denied**

```text
Error: SSH connection refused
```

- Ensure SSH server is running: `sudo systemctl status sshd`
- Check firewall rules: `sudo ufw status`
- Verify SSH port mapping in Docker: `-p 22222:22`
- Check SSH keys and authorized_keys

**GPU PoW Challenge Failed**

```text
Error: CUDA kernel execution failed
```

- Install CUDA runtime: `sudo apt install nvidia-cuda-toolkit`
- Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Verify PTX compilation during build
- No CPU fallback available - GPU required

**Permission Denied Errors**

```text
Error: Permission denied accessing /dev/nvidia0
```

- Run container in privileged mode: `--privileged`
- Check device access: `-v /dev:/dev:rw`
- Verify nvidia-container-runtime: `docker info | grep nvidia`

### Performance Optimization

**GPU Memory Management**

```toml
[docker.resource_limits]
gpu_memory_bytes = 8589934592  # Adjust based on GPU VRAM
```

**Container Limits**

```toml
[docker]
max_concurrent_containers = 20  # Increase for high-throughput
```

**Network Optimization**

```toml
[server]
max_connections = 2000  # Increase for more concurrent validators
```

## Security Considerations

### Network Security

- **Firewall**: Only expose required ports (50052, 8080, 22222)
- **SSH**: Use key-based authentication, disable password auth
- **VPN**: Consider VPN access for additional security
- **Rate Limiting**: Configure rate limits for validator requests

### Container Security

- **Isolation**: Enable network isolation for containers
- **Resource Limits**: Set appropriate CPU/memory/GPU limits
- **Registry Security**: Only pull from trusted registries
- **Signature Verification**: Enable container signature verification

### Data Protection

- **Encryption**: Use encrypted storage for sensitive data
- **Backup**: Regular backups of configuration and data
- **Log Management**: Rotate and secure log files
- **Audit**: Enable audit logging for SSH access

## Advanced Configuration

### Multi-GPU Setup

Configure multiple GPU access:

```toml
[docker]
enable_gpu_passthrough = true

# GPU-specific resource allocation
[docker.gpu_config]
visible_devices = "0,1,2,3"  # Use specific GPUs
memory_fraction = 0.8        # Use 80% of GPU memory
```

### Custom Images

Configure allowed container images:

```toml
[docker.registry]
url = "docker.io"
verify_signatures = true
allowed_registries = ["docker.io", "ghcr.io", "quay.io"]
```

### Load Balancing

For multiple executors behind a load balancer:

```toml
[server]
advertised_host = "load-balancer.example.com"
advertised_port = 443
advertised_tls = true
```

## Next Steps

- Review the [Architecture Guide](architecture.md) to understand system design
- Read the [Miner Guide](miner.md) to understand how miners manage executors
- Check the [GPU PoW Documentation](gpu_pow.md) for verification details
- Join the Basilica executor community for support and updates