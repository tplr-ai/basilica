# Basilica Scripts

This directory contains build and deployment scripts for each Basilica component.

## Structure

Each component has its own directory with:

- `build.sh` - Build the component
- `deploy.sh` - Deploy to remote servers
- `Dockerfile` - Container definition
- `compose.dev.yml` - Local development setup
- `compose.prod.yml` - Production deployment
- `README.md` - Component-specific documentation

## Metadata Management

**Important**: `generate-metadata.sh` - Regenerates Bittensor metadata for all services

```bash
# Generate metadata for all networks (test, finney)
./scripts/generate-metadata.sh

# Generate for specific network
./scripts/generate-metadata.sh --network finney

# Generate for multiple networks
./scripts/generate-metadata.sh test finney
```

**When to regenerate metadata:**

- Before production deployments
- When encountering "metadata compatibility" errors
- After Bittensor network upgrades
- When switching between networks

## Components

### Core Services

- **validator/** - Bittensor validator neuron for verification and scoring
- **miner/** - Bittensor miner neuron for GPU node orchestration
- **api/** - External HTTP API service

### Supporting Services

- **billing/** - Billing and payment processing
- **payments/** - Payment gateway integration

### Infrastructure & Tools

- **cloud/** - Terraform infrastructure as code (AWS deployments)
- **localtest/** - Local testing environment
- **provision/** - Provisioning and configuration management

## Usage

### Building

```bash
# For production builds, regenerate metadata first
./scripts/generate-metadata.sh --network finney

cd scripts/{component}
./build.sh
```

### Deploying

```bash
cd scripts/{component}
./deploy.sh user@host [port]
```

### Running Locally

```bash
cd scripts/{component}
docker compose -f compose.dev.yml up -d
```

### Running in Production

```bash
cd scripts/{component}
docker compose -f compose.prod.yml up -d
```

## Main CLI

Use `just` commands from the project root:

- `just test-run` - Run tests
- `just test-verify` - Verify test implementation
- `just test-stats` - Show test statistics
- `just build` - Build all components
- `just check` - Check code quality (format, clippy, test compilation)
- `just deploy-{component}` - Deploy individual components

For local development and production deployment, use the individual component docker-compose files.
