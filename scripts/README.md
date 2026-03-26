# Basilica Scripts

This directory contains build and deployment scripts for Basilica public components.

> **Note**: Backend services (api, billing, payments, operator, storage, autoscaler) have been moved to the private `basilica-backend` repository.

## Structure

Each component has its own directory with:

- `build.sh` - Build the component
- `deploy.sh` - Deploy to remote servers
- `Dockerfile` - Container definition
- `compose.local.yml` - Local development setup
- `README.md` - Component-specific documentation

## Components

### Core Services (This Repo)

- **validator/** - Bittensor validator neuron for verification and scoring
- **miner/** - Bittensor miner neuron for GPU node orchestration
- **cli/** - Command-line interface for users

### Development Tools

- **subtensor-local/** - Local Bittensor devnet (Alice/Bob)
- **localnet/** - Full local network setup
- **test/** - Test utilities

### Backend Services (Moved to `basilica-backend`)

The following have been moved to the private repo:
- api/ - External HTTP API service
- billing/ - Billing and payment processing
- payments/ - Payment gateway integration
- operator/ - Kubernetes operator
- storage-daemon/ - Storage service
- autoscaler/ - Auto-scaling service
- cloud/ - Terraform infrastructure

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

## Usage

### Building

```bash
# Build all public images
./scripts/build-images.sh

# Or use just
just docker-build-miner
just docker-build-validator
just docker-build-cli
```

### Local Development

```bash
# Start local Subtensor network
just local-subtensor-up

# Start local validator
just local-validator-up

# Start local miner
just local-miner-up

# Or start everything
just local-dev-up
```

### Deploying

```bash
cd scripts/{component}
./deploy.sh user@host [port]
```

## Main CLI

Use `just` commands from the project root:

- `just build` - Build all components
- `just test` - Run tests
- `just fix` - Fix linting issues
- `just local-dev-up` - Start full local dev environment
- `just local-subtensor-up` - Start local Subtensor only
- `just docker-build-all` - Build all Docker images

For backend services, see the `basilica-backend` repository.
