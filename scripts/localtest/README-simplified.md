# Basilica Local Testing Environment

This directory provides a **comprehensive integration testing framework** for the Basilica ecosystem. It's designed specifically for testing the complete stack including validator, miner, executor, and blockchain integration.

## Purpose & Differences

| Environment | Purpose | Use Case |
|------------|---------|----------|
| `docker/` | Development & Production | Running services for development or deployment |
| `localtest/` | Integration Testing | Full-stack testing with blockchain, validators, and test automation |

## Quick Start

```bash
cd scripts/localtest

# Build all images using existing build scripts
./build-localtest.sh

# Run the test environment
./run-localtest.sh up

# Run automated tests
./test-workflow.sh all

# Stop everything
./run-localtest.sh down
```

## What's Simplified

1. **Build Process**: Now reuses existing build scripts from `scripts/*/build.sh`
2. **Docker Compose**: Separated test-specific services from core services
3. **Integration**: Can work with existing running services using `--use-existing`

## Architecture

### Core Services (can use from `docker/`)
- **Miner**: Manages executor fleet
- **Executor**: Runs GPU workloads
- **Monitoring**: Prometheus + Grafana

### Test-Specific Services (unique to localtest)
- **Validator**: Full validator implementation for testing
- **Subtensor**: Local Bittensor blockchain
- **SSH Target**: For validation testing
- **Test Automation**: Comprehensive test workflows

## New Commands

### Build Images
```bash
# Build all images using existing build infrastructure
./build-localtest.sh
```

### Run Tests
```bash
# Option 1: Run complete isolated test stack
./run-localtest.sh up

# Option 2: Use with existing miner/executor from docker/
./run-localtest.sh up --use-existing

# View logs
./run-localtest.sh logs

# Check status
./run-localtest.sh ps

# Stop services
./run-localtest.sh down
```

### Test Workflows
```bash
# Run specific test suite
./test-workflow.sh security
./test-workflow.sh performance
./test-workflow.sh integration

# Run all tests
./test-workflow.sh all
```

## Configuration

1. **Test Configuration**: `test.conf` - Controls test parameters
2. **Service Configs**: `*.toml` - Service-specific settings
3. **Environment**: Automatically sets up keys, wallets, and databases

## Integration with CI/CD

The localtest environment is designed to be CI-friendly:

```yaml
# Example GitHub Actions
- name: Build test images
  run: cd scripts/localtest && ./build-localtest.sh

- name: Run integration tests
  run: |
    cd scripts/localtest
    ./run-localtest.sh up
    ./test-workflow.sh all
    ./run-localtest.sh down
```

## Key Benefits of Simplification

1. **Reuses Build Scripts**: No duplicate Dockerfile builds
2. **Modular Architecture**: Can run standalone or integrate with existing services  
3. **Clear Separation**: Test infrastructure vs development/production
4. **Maintained Functionality**: All testing capabilities preserved
5. **Easier Maintenance**: Single source of truth for builds

## Directory Structure

```
scripts/localtest/
├── build-localtest.sh         # NEW: Builds using existing scripts
├── run-localtest.sh          # NEW: Simplified runner
├── docker-compose.yml        # Full test stack
├── docker-compose-simplified.yml # NEW: Test-specific services only
├── test-workflow.sh          # Automated testing (unchanged)
├── test.conf                 # Test configuration
├── *.toml                    # Service configs
└── keys/                     # Generated test keys
```

## Migration Notes

- Old: `docker compose -f docker-compose.build.yml build`
- New: `./build-localtest.sh`

- Old: `docker compose up -d`  
- New: `./run-localtest.sh up`

- Old: Complex monolithic compose files
- New: Modular approach with reusable components