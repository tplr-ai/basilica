# Basilica Documentation

This directory contains comprehensive documentation for the Basilica decentralized GPU compute network built on Bittensor.

## Quick Navigation

### Getting Started

- **[Quick Start Guide](quickstart.md)** - Get up and running quickly with production deployments
- **[Architecture Guide](architecture.md)** - Understand the system design and components

### Component Guides

- **[Validator Guide](validator.md)** - Deploy and manage validator nodes for network verification
- **[Miner Guide](miner.md)** - Set up miners to orchestrate GPU node access via SSH

### Operations

- **[Monitoring Guide](monitoring.md)** - Set up metrics and monitoring with Prometheus/Grafana
- **[Scoring and Weights](scoring-and-weights.md)** - Understand how nodes are scored and weights are calculated

## Documentation Overview

### Architecture Guide

Comprehensive overview of Basilica's system design, including:

- Core components (Validator, Miner, GPU Nodes, Basilica API)
- Communication protocols (Bittensor, gRPC, SSH, REST)
- Security architecture with cryptographic framework
- Direct SSH-based verification model and deployment patterns

### Quick Start Guide

Step-by-step instructions for rapid deployment:

- Production Docker Compose setup (recommended)
- Remote deployment automation
- Development builds from source
- Monitoring and troubleshooting

### Validator Guide

Complete validator deployment and operation:

- Hardware and software requirements
- Production deployment with Docker Compose
- Verification process and scoring algorithms
- SSH-based remote verification setup
- Performance monitoring and maintenance
- Billing telemetry configuration

### Miner Guide

Comprehensive miner setup and GPU node orchestration:

- GPU node (executor) configuration and management
- Validator SSH key deployment and access control
- GPU verification through direct SSH access
- Security best practices and troubleshooting

### Monitoring Guide

Observability and monitoring setup:

- Prometheus and Grafana configuration
- Key metrics and alerts
- Production monitoring best practices
- Troubleshooting monitoring issues

## System Requirements

### Validator

- Linux system with stable internet connection
- Bittensor wallet with sufficient stake
- SSH access for remote verification
- CUDA Toolkit 12.8 (for GPU verification kernels)

### Miner

- Linux system with stable internet connection (8+ CPU cores, 16GB+ RAM recommended)
- Bittensor wallet with TAO tokens
- One or more GPU nodes with SSH access
- SSH key management for validator access control

### GPU Node (Executor)

GPU nodes are the compute resources managed by miners:

- NVIDIA GPU with CUDA support (A100, H100, or B200 recommended)
- CUDA Toolkit 12.8 (for GPU verification kernels)
- Docker with GPU runtime support (nvidia-container-toolkit)
- SSH server configured for validator access
- Linux server with sufficient resources (1TB+ disk recommended)
- All ports accessible for validator SSH connections

## Key Features

- **Dynamic UID Discovery**: Services automatically discover their network position
- **Auto Network Detection**: Chain endpoints configured based on network type
- **Flexible Wallet Support**: JSON wallet files and raw seed phrases
- **Production Ready**: Monitoring, auto-updates, and health checks included
- **GPU Proof-of-Work**: Cryptographic verification of GPU capabilities
- **Hardware Attestation**: P256 ECDSA signatures for hardware verification
- **Billing Telemetry**: Usage-based charging with streaming telemetry

## Deployment Options

1. **Production Docker Compose**
   - Fully automated with monitoring
   - Auto-updates via Watchtower
   - Health checks and persistent storage

2. **Remote Deployment**
   - Automated deployment to remote servers
   - Wallet synchronization and health checks
   - Support for distributed architectures

3. **Development Builds**
   - Source compilation and customization
   - Debug configurations and local testing

## Configuration Files

All configuration examples are in the `config/` directory:

- `validator.toml.example` - Validator configuration template
- `miner.toml.example` - Miner configuration template
- `api.toml.example` - API gateway configuration template
- `billing.toml.example` - Billing service configuration template
- `cli.toml.example` - CLI tool configuration template
- `gpu-attestor.toml.example` - GPU attestor configuration template

See `config/README.md` for detailed configuration instructions.

## Additional Resources

- **Configuration Examples**: Production-ready config templates in `config/`
- **Deployment Scripts**: Automated deployment tools in `scripts/`
- **GitHub Repository**: https://github.com/one-covenant/basilica
- **Discord**: https://discord.gg/Cy7c9vPsNK
- **Website**: https://www.basilica.ai/
