# Basilica Provisioning System

Comprehensive end-to-end provisioning solution for the Basilica decentralized GPU rental marketplace.

## Overview

The Basilica provisioning system automates the complete setup and management of a three-tier architecture:

- **Validator**: Verifies hardware attestations and manages consensus
- **Miner**: Manages executor fleets and serves validator requests  
- **Executor**: Provides GPU compute resources with secure attestation

## Quick Start

### 1. Configure Your Environment

Copy and edit the appropriate environment configuration:

```bash
# For production deployment
cp scripts/provision/environments/production.conf.example scripts/provision/environments/production.conf
vim scripts/provision/environments/production.conf

# For development setup
cp scripts/provision/environments/development.conf.example scripts/provision/environments/development.conf
vim scripts/provision/environments/development.conf
```

### 2. Complete End-to-End Provisioning

```bash
# Complete provisioning for production
./scripts/basilica.sh provision all --env production

# Or step by step
./scripts/basilica.sh provision servers   # Setup server dependencies
./scripts/basilica.sh provision configure # Generate configurations
./scripts/basilica.sh deploy production   # Deploy services
./scripts/basilica.sh provision validate  # Validate setup
```

### 3. Manage Services

```bash
# Check service status
./scripts/basilica.sh manage status

# Start/stop services
./scripts/basilica.sh manage start
./scripts/basilica.sh manage stop validator

# View logs
./scripts/basilica.sh manage logs executor --follow
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BASILICA NETWORK                                     │
├─────────────────┬─────────────────┬───────────────────────────────────┬─────────────────┤
│    VALIDATOR    │      MINER      │              EXECUTOR             │   GPU-ATTESTOR  │
│   (Verifier)    │ (Fleet Manager) │           (GPU Machine)           │   (Attestation) │
│                 │                 │                                   │                 │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐  │ ┌─────────────┐ │
│ │   HTTP API  │ │ │    gRPC     │ │ │    gRPC     │  │ Container   │  │ │  Hardware   │ │
│ │   Server    │ │ │   Server    │ │ │   Server    │  │ Manager     │  │ │  Detection  │ │
│ │   :8080     │ │ │   :8092     │ │ │   :50051    │  │ (Docker)    │  │ │  (NVIDIA)   │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────┘  └─────────────┘  │ └─────────────┘ │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐  │ ┌─────────────┐ │
│ │ SSH Client  │◄┼►│ Executor    │◄┼►│ Validation  │  │ System      │  │ │  GPU PoW    │ │
│ │ Validation  │ │ │ Fleet Mgr   │ │ │ Sessions    │  │ Monitor     │  │ │  Computer   │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────┘  └─────────────┘  │ └─────────────┘ │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐  │ ┌─────────────┐ │
│ │  Database   │ │ │ Bittensor   │ │ │   Config    │  │   Journal   │  │ │   Crypto    │ │
│ │  (SQLite)   │ │ │ Integration │ │ │   Manager   │  │   Logger    │  │ │   Signer    │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────┘  └─────────────┘  │ └─────────────┘ │
└─────────────────┴─────────────────┴───────────────────────────────────┴─────────────────┘
```

## Provisioning Workflow

### 1. Server Setup Phase (`provision servers`)

- Install system dependencies (Docker, Rust, NVIDIA drivers)
- Configure SSH connectivity between all machines
- Set up system users and directories
- Configure firewall rules
- Install and configure monitoring tools

### 2. Configuration Phase (`provision configure`)

- Generate environment-specific service configurations
- Create systemd service definitions
- Deploy configurations to target servers
- Set up cryptographic keys and certificates
- Configure service discovery endpoints

### 3. Deployment Phase (`deploy <environment>`)

- Build Basilica binaries from source
- Deploy binaries to respective servers
- Install systemd services
- Configure database connections
- Set up log rotation and monitoring

### 4. Startup Phase (`manage start`)

- Start services in dependency order:
  1. Executor (provides GPU resources)
  2. Miner (registers with executor)
  3. Validator (connects to miner fleet)

### 5. Validation Phase (`provision validate`)

- Network connectivity validation
- Service health checks
- End-to-end workflow testing
- Security configuration validation
- Performance baseline testing

## Command Reference

### Main CLI Commands

```bash
# Provisioning commands
./scripts/basilica.sh provision all              # Complete end-to-end provisioning
./scripts/basilica.sh provision servers          # Setup servers with dependencies
./scripts/basilica.sh provision configure        # Generate service configurations
./scripts/basilica.sh provision validate         # Validate infrastructure

# Deployment commands  
./scripts/basilica.sh deploy production          # Deploy to production
./scripts/basilica.sh deploy staging             # Deploy to staging
./scripts/basilica.sh deploy development         # Deploy to development

# Management commands
./scripts/basilica.sh manage status              # Check service status
./scripts/basilica.sh manage start [service]     # Start services
./scripts/basilica.sh manage stop [service]      # Stop services
./scripts/basilica.sh manage restart [service]   # Restart services
./scripts/basilica.sh manage logs [service]      # View service logs
```

### Direct Script Access

```bash
# Provisioning scripts
./scripts/provision/provision.sh all --env production
./scripts/provision/configure.sh generate --env production
./scripts/provision/services.sh deploy production
./scripts/provision/network.sh setup --env production
./scripts/provision/validate.sh all --env production

# Individual operations
./scripts/provision/configure.sh generate --service validator
./scripts/provision/services.sh start executor
./scripts/provision/network.sh test
./scripts/provision/validate.sh workflow --verbose
```

## Environment Configuration

### Production Environment

- **Purpose**: Live network deployment with full security
- **Bittensor Network**: Finney mainnet
- **Security**: Full TLS, firewall restrictions, secure keys
- **Performance**: Optimized for throughput and reliability
- **Monitoring**: Comprehensive metrics and alerting

### Staging Environment

- **Purpose**: Pre-production testing with production-like setup
- **Bittensor Network**: Testnet
- **Security**: Production security with staging certificates
- **Performance**: Production-like settings with debug logging
- **Monitoring**: Detailed metrics for testing validation

### Development Environment

- **Purpose**: Local development and testing
- **Bittensor Network**: Testnet or local mock
- **Security**: Relaxed for development convenience
- **Performance**: Optimized for fast iteration
- **Monitoring**: Verbose logging and debug endpoints

## Configuration Schema

Each environment configuration includes:

### Server Configuration
- Host addresses, ports, and user accounts
- SSH connectivity settings
- Service endpoint definitions

### Bittensor Integration
- Network selection (finney/test)
- Subnet ID and chain endpoints  
- Wallet and key configuration

### Service Settings
- Resource limits and timeouts
- Concurrency and rate limiting
- Performance tuning parameters

### Security Configuration
- TLS/mTLS settings
- Firewall rules and access control
- Key management and rotation

### Monitoring & Logging
- Log levels and formats
- Metrics collection and export
- Health check configuration

## Network Topology

### Service Communication

- **Validator → Miner**: gRPC calls for executor availability
- **Validator → Executor**: SSH-based hardware validation
- **Miner → Executor**: gRPC calls for fleet management
- **All Services**: Prometheus metrics export on port 9090

### Port Configuration

| Service   | API Port | gRPC Port | Metrics | SSH |
|-----------|----------|-----------|---------|-----|
| Validator | 8080     | -         | 9090    | 22  |
| Miner     | -        | 8092      | 9090    | 22  |
| Executor  | -        | 50051     | 9090    | 22  |

### Firewall Rules

Automatically configured by the provisioning system:

- **Validator**: Inbound 8080 (API), 9090 (metrics), 22 (SSH)
- **Miner**: Inbound 8091 (Bittensor), 8092 (gRPC), 9090 (metrics), 22 (SSH)  
- **Executor**: Inbound 50051 (gRPC), 9090 (metrics), 22 (SSH)

## Monitoring & Observability

### Metrics Collection

- **Prometheus**: Metrics aggregation on port 9090
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification

### Logging

- **Structured Logging**: JSON format for production
- **Log Aggregation**: Centralized via rsyslog or ELK stack
- **Log Rotation**: Automatic cleanup and compression

### Health Monitoring

- **Service Health**: Built-in health endpoints
- **Resource Monitoring**: CPU, memory, disk, GPU utilization
- **Network Monitoring**: Connectivity and latency tracking

## Security Features

### Authentication & Authorization

- **SSH Key Management**: Automated distribution and rotation
- **Service Authentication**: Bittensor signature verification
- **API Security**: JWT tokens and rate limiting

### Network Security

- **Firewall Configuration**: UFW rules with service-specific ports
- **TLS Encryption**: Optional TLS/mTLS for service communication
- **Network Isolation**: VPC/subnet configuration support

### Operational Security

- **Principle of Least Privilege**: Service-specific system users
- **Security Hardening**: systemd security features enabled
- **Audit Logging**: Comprehensive operation tracking

## Troubleshooting

### Common Issues

1. **SSH Connectivity Issues**
   ```bash
   # Regenerate and redistribute SSH keys
   ./scripts/provision/network.sh keys --env production
   
   # Test connectivity
   ./scripts/provision/network.sh test --env production
   ```

2. **Service Startup Failures**
   ```bash
   # Check service logs
   ./scripts/basilica.sh manage logs validator
   
   # Validate configuration
   ./scripts/provision/validate.sh services --env production
   ```

3. **Network Connectivity Problems**
   ```bash
   # Test network connectivity
   ./scripts/provision/network.sh test --env production
   
   # Check firewall rules
   ./scripts/provision/network.sh firewall --env production
   ```

### Validation and Diagnostics

```bash
# Complete infrastructure validation
./scripts/provision/validate.sh all --env production --verbose --report

# Network-specific validation
./scripts/provision/validate.sh network --env production

# Service-specific validation  
./scripts/provision/validate.sh services --env production

# Security validation
./scripts/provision/validate.sh security --env production
```

### Log Analysis

```bash
# View real-time logs
./scripts/basilica.sh manage logs executor --follow

# Search logs for errors
./scripts/basilica.sh manage logs validator | grep ERROR

# Check systemd service status
ssh user@host 'sudo systemctl status basilica-validator'
```

## Advanced Usage

### Custom Environments

Create custom environment configurations:

```bash
# Copy template
cp scripts/provision/environments/production.conf scripts/provision/environments/custom.conf

# Edit configuration
vim scripts/provision/environments/custom.conf

# Use custom environment
./scripts/basilica.sh provision all --env custom
```

### Partial Deployments

Deploy specific services:

```bash
# Deploy only validator
./scripts/provision/services.sh deploy production validator

# Configure only executor
./scripts/provision/configure.sh generate --service executor --env production
```

### Rolling Updates

Perform rolling updates with zero downtime:

```bash
# Stop services in reverse order
./scripts/basilica.sh manage stop validator
./scripts/basilica.sh manage stop miner

# Deploy new version
./scripts/basilica.sh deploy production

# Start services in dependency order
./scripts/basilica.sh manage start executor
./scripts/basilica.sh manage start miner
./scripts/basilica.sh manage start validator
```

## Integration with Existing Scripts

The provisioning system integrates with existing Basilica scripts:

- **setup-ops/**: Used for initial server setup and dependency installation
- **production/**: Extended with enhanced deployment and configuration features  
- **localtest/**: Compatible with local testing and development workflows

## Support and Documentation

For additional help:

- Use `--help` option with any command for detailed usage information
- Check the `SYSTEM_ARCHITECTURE_OPERATIONS_GUIDE.md` for detailed architecture documentation
- Review `TODO.md` for current implementation status and known issues
- Generate validation reports for detailed infrastructure analysis

## Contributing

When extending the provisioning system:

1. Follow the modular design principles
2. Add appropriate validation tests
3. Update environment configurations as needed
4. Maintain backward compatibility with existing scripts
5. Document new features and configuration options