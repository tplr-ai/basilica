# Basilica Scripts

Scripts for building, testing, and deploying Basilica components.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU drivers and Docker runtime (for GPU attestation)
- Rust toolchain (for native builds)
- SSH access to deployment servers (for production)

### Main CLI

```bash
./scripts/basilica.sh <command> [options]
```

## Building Components

### Build All Components

```bash
# Individual builds
cd scripts/
./executor/build.sh
./miner/build.sh
./validator/build.sh

# Unified build
cd scripts/localtest
docker compose -f docker-compose.build.yml build
```

### Build Individual Components

#### Executor
```bash
cd scripts/executor
./build.sh --release
```

#### GPU Attestor
```bash
cd scripts/gpu-attestor
export VALIDATOR_PUBLIC_KEY=$(cat ../../public_key.hex)
./build.sh --release
```

#### Miner
```bash
cd scripts/miner
./build.sh --release
```

#### Validator
```bash
cd scripts/validator
./build.sh --release
```

## Running Services Locally

### Local Development Setup

```bash
# 1. Generate development keys
./scripts/gen-key.sh

# 2. Setup test environment
cd scripts/localtest
./setup-test-env.sh

# 3. Run all services
docker compose up -d

# 4. Check status
docker compose ps
```

### Running Individual Services

#### Run Executor with GPU Support
```bash
docker run -d \
  --name basilica-executor \
  --gpus all \
  -p 50051:50051 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/data:/opt/basilica/data \
  basilica/executor:localtest \
  --config /opt/basilica/config/executor.toml
```

#### Run Miner
```bash
docker run -d \
  --name basilica-miner \
  -p 8080:8080 \
  -v $(pwd)/data:/opt/basilica/data \
  basilica/miner:localtest \
  --config /opt/basilica/config/miner.toml
```

#### Run Validator
```bash
docker run -d \
  --name basilica-validator \
  -p 8081:8081 \
  -p 9090:9090 \
  -v $(pwd)/data:/opt/basilica/data \
  basilica/validator:localtest \
  --config /opt/basilica/config/validator.toml
```

## Production Deployment

### Production Deployment

#### Step 1: Configure Environment
```bash
cd scripts/provision

# Copy template and configure your servers
cp environments/template.conf environments/production.conf
# Edit production.conf with your server details
```

#### Step 2: Deploy Everything
```bash
# Build and deploy all components
./scripts/basilica.sh provision all
./scripts/basilica.sh deploy all
./scripts/basilica.sh manage start all

# Verify deployment
./scripts/basilica.sh manage status
```

### Verify Deployment

```bash
./scripts/basilica.sh manage status
```

Expected output:
- Executor: RUNNING, gRPC port 50051 listening, GPU detected
- Miner: RUNNING, HTTP port 8080 listening, executors discovered
- Validator: RUNNING, HTTP port 8081 listening, SSH validation ready

## Testing GPU Attestation

### Run GPU Attestation Test
```bash
# SSH to executor server
ssh root@EXECUTOR_HOST -p EXECUTOR_PORT

# Run attestation
/opt/basilica/bin/gpu-attestor \
  --executor-id prod-executor-1 \
  --output /opt/basilica/data/attestations/test

# Expected output:
# GPU detection, attestation completion, and file paths for:
# - Report: /opt/basilica/data/attestations/test.json
# - Signature: /opt/basilica/data/attestations/test.sig
# - Public Key: /opt/basilica/data/attestations/test.pub
```

### Verify Attestation
```bash
# Check attestation content
cat /opt/basilica/data/attestations/test.json | grep -A5 gpu_info
```

## Container Rental Deployment

### Deploy Rental Container on Executor
```bash
# SSH to executor
ssh root@EXECUTOR_HOST -p EXECUTOR_PORT

# Deploy a rental container
docker run -d \
  --name gpu-rental-001 \
  --gpus all \
  --restart unless-stopped \
  nvidia/cuda:12.2.0-base-ubuntu22.04 \
  bash -c "nvidia-smi && sleep infinity"

# Verify deployment
docker ps | grep gpu-rental
```

## Monitoring and Management

### Check Service Health
```bash
# Executor health
curl http://EXECUTOR_HOST:8091/health

# Miner API
curl http://MINER_HOST:8080/health

# Validator metrics
curl http://VALIDATOR_HOST:9090/metrics
```

### View Service Logs
```bash
# Using basilica CLI
./scripts/basilica.sh manage logs executor
./scripts/basilica.sh manage logs miner
./scripts/basilica.sh manage logs validator

# Or directly on servers
ssh root@EXECUTOR_HOST -p EXECUTOR_PORT "tail -f /opt/basilica/logs/executor.log"
```

### Restart Services
```bash
# Restart individual service
./scripts/basilica.sh manage restart executor

# Restart all services
./scripts/basilica.sh manage restart all
```

## Development Workflow

### Local Development Setup

```bash
# 1. Install dependencies
./scripts/install-deps.sh

# 2. Generate keys
./scripts/gen-key.sh

# 3. Build components
cd scripts/localtest
docker compose -f docker-compose.build.yml build

# 4. Start services
docker compose up -d

# 5. Verify status
docker compose ps

# 6. Run tests
./test-workflow.sh all
```

### Testing Individual Components

#### Test GPU Detection
```bash
docker compose exec executor /opt/basilica/bin/executor --health-check
```

#### Test GPU Attestation
```bash
docker compose exec executor /opt/basilica/bin/gpu-attestor \
  --executor-id local-test \
  --skip-vdf \
  --output /tmp/test-attestation
```

#### Test Service Communication
```bash
docker compose exec miner curl http://executor:50051/health
```

#### Test Validator SSH Validation
```bash
docker compose exec validator curl -X POST http://localhost:8080/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"executor_id": "test-executor", "challenge": "echo test"}'
```

## Directory Structure

```
scripts/
├── basilica.sh              # Main CLI entry point
├── executor/                # Executor Docker build
│   ├── Dockerfile
│   ├── build.sh
│   └── docker-compose.yml
├── miner/                   # Miner Docker build
│   ├── Dockerfile
│   ├── build.sh
│   └── docker-compose.yml
├── validator/               # Validator Docker build
│   ├── Dockerfile
│   ├── build.sh
│   └── docker-compose.yml
├── gpu-attestor/            # GPU attestor Docker build
│   ├── Dockerfile
│   └── build.sh
├── localtest/               # Local testing environment
│   ├── docker-compose.yml
│   ├── test-workflow.sh
│   └── configs/
├── provision/               # Production deployment
│   ├── deploy.sh
│   ├── config-generator.sh
│   ├── service-manager.sh
│   └── environments/
└── utilities/               # Helper scripts
    ├── gen-key.sh
    ├── extract-pubkey.sh
    └── install-deps.sh
```

## Common Operations

### Building Services

```bash
# Build specific service
./scripts/executor/build.sh --release
./scripts/miner/build.sh --release
./scripts/validator/build.sh --release

# Build all services
cd scripts/localtest
docker compose -f docker-compose.build.yml build

# Build with custom tag
docker build -t myregistry/basilica/executor:v1.0 executor/
```

### Running Services

```bash
# Start all services locally
cd scripts/localtest
docker compose up -d

# Start individual service
docker compose up -d executor
docker compose up -d miner
docker compose up -d validator

# View logs
docker compose logs -f executor
docker compose logs -f miner
docker compose logs -f validator
```

### Testing Services

```bash
# Run complete test suite
cd scripts/localtest
./test-workflow.sh all

# Test specific component
./test-workflow.sh gpu      # GPU attestation
./test-workflow.sh ssh      # SSH validation
./test-workflow.sh core     # Core services
```

## Troubleshooting

### Common Issues and Solutions

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Solution: Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Service Connection Failed
```bash
# Check status
docker compose ps

# Check logs
docker compose logs executor | tail -50

# Test connectivity
docker compose exec miner ping executor

# Restart if needed
docker compose restart
```

#### Configuration Errors
```bash
# Duration format error
# Wrong: timeout = 60
# Right: timeout = { secs = 60 }

# SQLite URL error
# Wrong: sqlite:./data/executor.db
# Right: sqlite:///opt/basilica/data/executor.db
```

#### Wallet Issues
```bash
# Error: "Failed to load hotkey seed"
# Note: Miner/Validator require real Bittensor wallets
# For testing without wallets, use executor and GPU attestation
```

### Useful Commands

```bash
# Check Docker resources
docker system df
docker system prune -af

# Monitor service resources
docker stats

# Debug container networking
docker network ls
docker network inspect basilica_default

# Check service endpoints
curl http://localhost:50051/health  # Executor
curl http://localhost:8080/health   # Miner
curl http://localhost:8081/health   # Validator
```

## Additional Resources

### Key Scripts
- `gen-key.sh` - Generate P256 key pairs
- `extract-pubkey.sh` - Extract public key from private key
- `install-deps.sh` - Install system dependencies
- `quick-start.sh` - One-command development setup

### Configuration
- Templates: `scripts/provision/configs/*.toml.example`
- Working examples: `scripts/localtest/configs/`

## Summary

Basilica scripts provide:
- Building all components (executor, miner, validator, gpu-attestor)
- Running services locally with Docker Compose
- Deploying to production servers with automated provisioning
- Managing services with health checks and monitoring

Note: Executor and GPU attestation components can be tested without Bittensor wallets.
