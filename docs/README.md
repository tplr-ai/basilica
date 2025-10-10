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

### Miner Guide

Comprehensive miner setup and GPU node orchestration:

- GPU node SSH endpoint configuration and management
- Validator SSH key deployment and access control
- GPU verification through Proof-of-Work challenges
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

- Linux system with stable internet connection
- Bittensor wallet with TAO tokens
- One or more GPU nodes with SSH access
- SSH key management for validator access control

### GPU Node (formerly Executor)

**Note**: The executor binary is deprecated. GPU nodes now require:

- NVIDIA GPU with CUDA support (8.7 CUDA GPU Compute Capability)
- CUDA Toolkit 12.8 (for GPU verification kernels)
- Docker with GPU runtime support (nvidia-container-toolkit)
- SSH server configured for validator access
- Linux server with sufficient resources

## Key Features

- **Dynamic UID Discovery**: Services automatically discover their network position
- **Auto Network Detection**: Chain endpoints configured based on network type
- **Flexible Wallet Support**: JSON wallet files and raw seed phrases
- **Production Ready**: Monitoring, auto-updates, and health checks included
- **GPU Proof-of-Work**: Cryptographic verification of GPU capabilities
- **Hardware Attestation**: P256 ECDSA signatures for hardware verification

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

## Additional Resources

- **Configuration Examples**: Production-ready config templates in `config/`
- **Deployment Scripts**: Automated deployment tools in `scripts/`
