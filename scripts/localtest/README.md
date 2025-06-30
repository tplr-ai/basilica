# Basilica Local Testing Environment

This directory provides a comprehensive integration testing framework for the Basilica ecosystem, including automated testing for validator, miner, executor, and GPU attestor components. The framework supports both Docker-based and native testing with full production-ready integration tests.

## Quick Start

```bash
cd scripts/localtest

# 1. Setup test environment and configuration
cp test.conf.example test.conf
./test-workflow.sh setup

# 2. Build all required Docker images
./test-workflow.sh build

# 3. Run complete test suite
./test-workflow.sh all
```

## Directory Structure

```
scripts/localtest/
├── docker-compose.yml        # Main orchestration
├── docker-compose.build.yml  # Build configuration
├── test-workflow.sh          # Automated testing framework
├── test.conf                 # Test configuration (copy from .example)
├── test.conf.example         # Example test configuration
├── setup-test-env.sh         # Environment setup
├── validator.toml            # Validator config
├── miner.toml               # Miner config
├── executor.toml             # Executor config
├── prometheus.yml            # Metrics config
├── grafana-datasources.yml   # Grafana config
├── data/                     # Generated test data (gitignored)
└── keys/                     # Generated test keys (gitignored)

crates/miner/tests/integration/
├── bittensor_chain_integration_test.rs    # Bittensor chain integration
├── executor_deployment_integration_test.rs # Executor fleet management
├── validator_grpc_integration_test.rs     # gRPC communication testing
├── metrics_integration_test.rs             # Prometheus metrics testing
└── end_to_end_flow_test.rs               # Complete workflow testing
```

## Test Environment Setup

The setup script automatically generates:

```bash
./setup-test-env.sh
```

Creates (not tracked in git):
- `data/` - Runtime data and SQLite databases
- `keys/id_rsa` - SSH keys for validation testing
- `keys/private_key.pem` - P256 certificates for attestation
- `keys/wallets/` - Test Bittensor wallet files

## Test Configuration

The test framework uses `test.conf` for all configuration. Copy and customize:

```bash
cp test.conf.example test.conf
```

Key configuration sections:
- **Service Endpoints** - Host/port settings for all services
- **Test Parameters** - Timeouts, retries, delays
- **Load Test Settings** - Concurrency, duration, rate limits
- **Feature Flags** - Enable/disable test categories
- **Paths** - Database, attestor, and work directories

## Test Workflow Commands

```bash
./test-workflow.sh [command]
```

| Command | Description |
|---------|-------------|
| `setup` | Setup test environment (keys, directories) |
| `prerequisites` | Check Docker, GPU runtime, dependencies |
| `build` | Build all Docker images |
| `individual` | Test individual service compose files |
| `core` | Start and test core services (validator, miner, executor) |
| `miner` | Test miner-specific functionality |
| `ssh` | Test SSH validation with mock target |
| `gpu` | Test GPU attestation (requires NVIDIA runtime) |
| `e2e` | End-to-end validation workflow |
| `load` | Run load performance tests |
| `failover` | Test failover and recovery scenarios |
| `security` | Test security validations |
| `production-ready` | Test production readiness |
| `monitoring` | Start Prometheus + Grafana |
| `status` | Show system status and logs |
| `cleanup` | Stop services and clean up |
| `all` | Run complete test workflow (default) |

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Validator    │    │      Miner      │    │    Executor     │
│   Port: 8080    │◄──►│   Port: 8092    │◄──►│   Port: 50051   │
│   Port: 8081    │    │   Port: 9092    │    │   Port: 9090    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SQLite DB     │    │   SQLite DB     │    │  GPU-Attestor   │
│   (validator)   │    │    (miner)      │    │   (on-demand)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Service Profiles

### Default Services
- **Validator** - HTTP API (8080), Metrics (8081), SQLite storage
- **Miner** - gRPC server (8092), Metrics (9092), Executor fleet management
- **Executor** - gRPC server (50051), Metrics (9090), Docker integration

### Profile-based Services

#### SSH Testing (`--profile ssh-test`)
```bash
docker compose --profile ssh-test up -d ssh-target
./test-workflow.sh ssh
```

#### GPU Attestation (`--profile attestor`)
```bash
docker compose --profile attestor up gpu-attestor
./test-workflow.sh gpu
```

#### Monitoring (`--profile monitoring`)
```bash
docker compose --profile monitoring up -d prometheus grafana
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9091
```

## Testing Scenarios

### 1. Build Testing
```bash
# Test modular builds
./test-workflow.sh build

# Verify images
docker images | grep basilica
```

### 2. Service Integration
```bash
# Test core services
./test-workflow.sh core

# Check health
curl http://localhost:8080/health
```

### 3. SSH Validation
```bash
# End-to-end SSH validation
docker compose --profile ssh-test up -d
docker compose exec validator validator connect --host ssh-target --username testuser --private-key /etc/basilica/keys/id_rsa
```

### 4. GPU Attestation
```bash
# Hardware attestation (requires NVIDIA GPU)
./test-workflow.sh gpu
docker compose exec executor cat /shared/attestor/attestation.json | jq .
```

## Configuration

### Environment Setup
The test environment uses:
- **Test SSH keys** - Generated for validation testing
- **Mock certificates** - P256 keys for hardware attestation
- **Test wallets** - Bittensor wallet files with dummy data
- **SQLite database** - Local file storage for validation logs

### Service Configs
- `validator.toml` - SQLite database, HTTP server, SSH client config
- `executor.toml` - gRPC server, Docker integration, metrics
- `prometheus.yml` - Metrics scraping configuration

## Debugging

### Build Issues
```bash
# Individual service builds
cd ../validator && docker compose build --no-cache
cd ../executor && docker compose build --no-cache
cd ../gpu-attestor && docker compose build --no-cache
```

### Service Issues
```bash
# Check logs
docker compose logs validator
docker compose logs executor

# Test connectivity
docker compose exec validator ping executor
```

### Test Environment Issues
```bash
# Recreate test environment
rm -rf data/ keys/
./setup-test-env.sh

# Check permissions
ls -la keys/
```

## Running Integration Tests

### 1. Native Cargo Tests

Run the comprehensive integration test suite:

```bash
# From project root
cd crates/miner

# Run all integration tests
cargo test --test '*' -- --nocapture

# Run specific test files
cargo test --test bittensor_chain_integration_test -- --nocapture
cargo test --test executor_deployment_integration_test -- --nocapture
cargo test --test validator_grpc_integration_test -- --nocapture
cargo test --test metrics_integration_test -- --nocapture
cargo test --test end_to_end_flow_test -- --nocapture

# Run with specific test patterns
cargo test --test metrics_integration_test test_executor_fleet -- --nocapture
```

### 2. Docker-based Testing

If you encounter SSL linking issues, use Docker:

```bash
cd scripts/localtest

# Build images first
docker compose -f docker-compose.build.yml build

# Run services
docker compose up -d

# Execute tests inside container
docker compose exec miner cargo test --test '*' -- --nocapture

# View test output
docker compose logs -f miner
```

### 3. Test Categories

The integration tests cover:

- **Chain Integration** - Bittensor registration, UID discovery, weight updates
- **Executor Deployment** - Fleet management, health monitoring, auto-recovery
- **gRPC Communication** - Authentication, session management, rate limiting
- **Metrics Collection** - Prometheus integration, custom labels, aggregation
- **End-to-End Flows** - Complete validation workflows, failure scenarios

## Endpoints

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Validator API | http://localhost:8080 | REST API |
| Validator Metrics | http://localhost:8081/metrics | Prometheus metrics |
| Miner gRPC | localhost:8092 | gRPC service |
| Miner Metrics | http://localhost:9092/metrics | Prometheus metrics |
| Executor gRPC | localhost:50051 | gRPC service |
| Executor Metrics | http://localhost:9090/metrics | Prometheus metrics |
| SSH Target | localhost:2222 | Test SSH server |
| Prometheus | http://localhost:9091 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboard |

## Production Notes

⚠️ **Development/Testing Only**

This setup includes:
- Test keys and certificates
- Debug logging
- Relaxed security settings
- Local-only networking

For production deployment, see [SYSTEM_ARCHITECTURE_OPERATIONS_GUIDE.md](../../SYSTEM_ARCHITECTURE_OPERATIONS_GUIDE.md).

## Troubleshooting

### Common Issues

#### Missing Test Files
```bash
# Regenerate test environment
./test-workflow.sh setup
```

#### Image Not Found
```bash
# Build images
./test-workflow.sh build
docker images | grep basilica
```

#### Service Health Failures
```bash
# Check logs
docker compose logs validator
./test-workflow.sh status
```

#### GPU Runtime Issues
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Reset Environment
```bash
# Complete reset
./test-workflow.sh cleanup
rm -rf data/ keys/
./test-workflow.sh setup
```

## Verification Steps

### Quick Verification

```bash
# 1. Verify all test files exist
ls -la crates/miner/tests/integration/

# 2. Check test compilation
cd crates/miner && cargo check --tests

# 3. List available tests
cargo test --test '*' -- --list

# 4. Run a quick smoke test
cd scripts/localtest && ./test-workflow.sh prerequisites build
```

### Complete Verification

```bash
# 1. Full test suite execution
cd scripts/localtest
./test-workflow.sh all

# 2. Check service health
curl http://localhost:8080/health
curl http://localhost:8081/metrics | grep validator_
curl http://localhost:9092/metrics | grep miner_

# 3. Verify test results
docker compose exec validator sqlite3 /var/lib/basilica/validator/validator.db \
  "SELECT COUNT(*) FROM verification_logs;"

# 4. Review test logs
docker compose logs --tail=100
```

## Benefits

- **Configuration-Driven** - No hardcoded values, all settings in test.conf
- **Production-Ready Tests** - Zero mocks, stubs, or fake implementations
- **Comprehensive Coverage** - Tests all major components and workflows
- **Automated Setup** - No manual key generation or configuration
- **Modular Testing** - Test individual components separately
- **Reproducible** - Consistent test environment across systems
- **Fast Development** - Quick iteration and debugging
- **Docker & Native** - Support both Docker and native Cargo testing