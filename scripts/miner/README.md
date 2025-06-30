# Basilica Miner Docker Deployment

This directory contains Docker build and deployment scripts for the Basilica Miner component, which serves as a Bittensor neuron that manages executor fleets and handles validator communications.

## Overview

The Basilica Miner is responsible for:
- **Fleet Management**: Managing local and remote executor instances
- **Validator Communications**: Providing gRPC server for validator requests
- **Bittensor Integration**: Participating in the Bittensor network as a neuron
- **SSH Access Coordination**: Managing secure access for validator verification
- **Health Monitoring**: Continuous monitoring of executor health and availability

## Architecture

The miner operates as a central coordinator in the Basilica ecosystem:

```
┌─────────────────────────────────────────────────────────────┐
│                    BASILICA MINER                           │
├─────────────────────────────────────────────────────────────┤
│  gRPC Server (8092)  │  Bittensor Axon (8091)  │ Metrics   │
│  ┌─────────────────┐ │ ┌─────────────────────┐ │ (9092)    │
│  │ Validator Comms │ │ │ Network Integration │ │           │
│  └─────────────────┘ │ └─────────────────────┘ │           │
│  ┌─────────────────────────────────────────────┐ │           │
│  │         Executor Fleet Manager             │ │           │
│  │  ┌─────────────┐  ┌─────────────────────┐  │ │           │
│  │  │  Local      │  │  Remote Deployment  │  │ │           │
│  │  │  Executors  │  │  (SSH-based)        │  │ │           │
│  │  └─────────────┘  └─────────────────────┘  │ │           │
│  └─────────────────────────────────────────────┘ │           │
│  ┌─────────────────────────────────────────────┐ │           │
│  │               SQLite Database               │ │           │
│  │    (Health Tracking & Audit Logging)       │ │           │
│  └─────────────────────────────────────────────┘ │           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Building the Image

```bash
# Build with default settings (release mode)
./build.sh

# Build with custom image name and tag
./build.sh --image-name my-registry/basilica-miner --image-tag v1.0.0

# Build in debug mode
./build.sh --debug

# Build without extracting binary
./build.sh --no-extract
```

### Running with Docker Compose

```bash
# Start miner service
docker compose up -d

# View logs
docker compose logs -f miner

# Stop service
docker compose down
```

## Configuration

### Required Configuration File

The miner requires a `miner.toml` configuration file. Generate a template:

```bash
# Generate sample configuration
docker run --rm basilica/miner:latest --gen-config > miner.toml
```

### Key Configuration Sections

```toml
[bittensor]
wallet_name = "your_miner_wallet"
hotkey_name = "your_hotkey"
network = "finney"
netuid = 27  # Basilca subnet ID
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
external_ip = "YOUR_PUBLIC_IP"  # Required for production

[database]
url = "sqlite:./data/miner.db"
run_migrations = true

[server]
host = "0.0.0.0"
port = 8092  # gRPC server port

[executor_management]
# Static executor configuration
executors = [
    { id = "executor-1", grpc_address = "10.0.1.100:50051", name = "GPU Machine 1" },
]
health_check_interval = { secs = 60 }
auto_recovery = true

[security]
jwt_secret = "CHANGE_THIS_IN_PRODUCTION"
verify_signatures = true
```

## Networking

### Port Configuration

- **8092**: Miner gRPC server for validator communications
- **8091**: Bittensor axon server port for network participation
- **9092**: Prometheus metrics endpoint

### Network Integration

The miner connects to:
- **Validators**: Receives verification requests and provides executor access
- **Executors**: Manages health monitoring and deployment
- **Bittensor Network**: Participates as a neuron with UID registration

## Executor Fleet Management

### Static Executor Configuration

Configure known executors in `miner.toml`:

```toml
[executor_management]
executors = [
    { 
        id = "gpu-server-1", 
        grpc_address = "10.0.1.100:50051", 
        name = "Primary GPU Server",
        gpu_count = 8
    },
    { 
        id = "gpu-server-2", 
        grpc_address = "10.0.1.101:50051", 
        name = "Secondary GPU Server",
        gpu_count = 4
    },
]
```

### Remote Deployment (Optional)

Enable automatic executor deployment:

```toml
[remote_executor_deployment]
auto_deploy = true
auto_start = true

[[remote_executor_deployment.remote_machines]]
id = "gpu-machine-1"
name = "Primary GPU Server"
gpu_count = 8
executor_port = 50051

[remote_executor_deployment.remote_machines.ssh]
host = "10.0.1.100"
port = 22
username = "ubuntu"
private_key_path = "/etc/basilica/keys/miner_key"
```

## CLI Commands

The miner provides comprehensive CLI management:

```bash
# Service management
docker exec basilica-miner miner status
docker exec basilica-miner miner migrate

# Executor fleet management
docker exec basilica-miner miner executor list
docker exec basilica-miner miner executor health
docker exec basilica-miner miner executor show <executor-id>

# Validator interaction management
docker exec basilica-miner miner validator list --limit 10
docker exec basilica-miner miner validator show-access <validator-hotkey>

# Database management
docker exec basilica-miner miner database status
docker exec basilica-miner miner database stats
docker exec basilica-miner miner database cleanup --days 30 --force

# Remote deployment
docker exec basilica-miner miner deploy-executors --dry-run
docker exec basilica-miner miner deploy-executors --status-only
```

## Health Monitoring

### Service Health Check

```bash
# Container health check
docker compose ps

# Application health check
curl http://localhost:9092/metrics
```

### Executor Fleet Health

```bash
# Check all executor health
docker exec basilica-miner miner executor health

# View fleet statistics
docker exec basilica-miner miner database stats
```

## Production Deployment

### Security Considerations

1. **Change Default Secrets**:
   ```toml
   [security]
   jwt_secret = "your-secure-random-secret"
   ```

2. **Configure Firewall**:
   ```bash
   # Allow only necessary ports
   ufw allow 8092/tcp  # Miner gRPC
   ufw allow 8091/tcp  # Bittensor axon
   ```

3. **Secure SSH Keys**:
   ```bash
   # Store SSH keys securely
   chmod 600 /etc/basilica/keys/miner_key
   chown root:basilica /etc/basilica/keys/miner_key
   ```

### Monitoring Setup

```bash
# Metrics collection
curl http://localhost:9092/metrics

# Key metrics to monitor:
# - basilica_miner_executor_health_total
# - basilica_miner_validator_requests_total
# - basilica_miner_ssh_sessions_active
# - basilica_miner_bittensor_registration_status
```

### Backup and Recovery

```bash
# Database backup
docker exec basilica-miner miner database backup --output /backup/miner-backup.db

# Configuration backup
docker cp basilica-miner:/app/miner.toml ./miner-backup.toml

# Restore database
docker exec basilica-miner miner database restore --input /backup/miner-backup.db --force
```

## Troubleshooting

### Common Issues

1. **Bittensor Registration Failed**:
   ```bash
   # Check network connectivity
   docker exec basilica-miner miner status
   
   # Verify wallet configuration
   docker exec basilica-miner ls -la /home/basilica/.bittensor/
   ```

2. **Executor Health Failures**:
   ```bash
   # Check executor connectivity
   docker exec basilica-miner miner executor list
   
   # Test specific executor
   docker exec basilica-miner miner executor show <executor-id>
   ```

3. **Database Connection Issues**:
   ```bash
   # Check database status
   docker exec basilica-miner miner database status
   
   # Run migrations
   docker exec basilica-miner miner migrate
   ```

### Log Analysis

```bash
# View miner logs
docker compose logs -f miner

# Filter for specific components
docker compose logs miner | grep "executor_manager"
docker compose logs miner | grep "validator_comms"
docker compose logs miner | grep "bittensor"
```

## Integration Testing

See the `scripts/localtest/` directory for comprehensive integration testing with validator and executor components.

```bash
# Run complete integration test
cd ../localtest
./test-workflow.sh all
```

## Support

For issues and documentation:
- **Architecture Guide**: `../../SYSTEM_ARCHITECTURE_OPERATIONS_GUIDE.md`
- **Configuration Reference**: Use `--gen-config` to generate example configurations
- **Integration Testing**: `../localtest/test-workflow.sh`