# Basilica Documentation

This directory contains comprehensive documentation for the Basilica decentralized GPU compute network built on Bittensor.

## Quick Navigation

### Getting Started

- **[Quick Start Guide](quickstart.md)** - Get up and running quickly with production deployments
- **[Architecture Guide](architecture.md)** - Understand the system design and components

### Component Guides

- **[Validator Guide](validator.md)** - Deploy and manage validator nodes for network verification
- **[Miner Guide](miner.md)** - Set up miners to manage GPU executor fleets
- **[Executor Guide](executor.md)** - Configure GPU executors for computational tasks

### Operations

- **[Monitoring Guide](monitoring.md)** - Set up metrics and monitoring with Prometheus/Grafana

## Documentation Overview

### Architecture Guide

Comprehensive overview of Basilica's system design, including:

- Core components (Validator, Miner, Executor, Public API)
- Communication protocols (Bittensor, gRPC, SSH, REST)
- Security architecture with cryptographic framework
- Data flow and deployment patterns

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

Comprehensive miner setup and fleet management:

- Executor fleet configuration and management
- GPU verification through Proof-of-Work challenges
- Multiple deployment modes (SSH, Manual, Kubernetes)
- Security best practices and troubleshooting

### Executor Guide

GPU executor deployment and configuration:

- NVIDIA GPU requirements and Docker setup
- Container management and resource limits
- Hardware attestation and verification
- Security considerations and performance optimization

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

- Linux system with Docker support
- Bittensor wallet with TAO tokens
- One or more GPU machines for executors
- Network connectivity between miner and executors

### Executor

- NVIDIA GPU with CUDA support (8.7 CUDA GPU Compute Capability)
- CUDA Toolkit 12.8 (for GPU verification kernels)
- Docker with GPU runtime support
- SSH server for validator access
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
