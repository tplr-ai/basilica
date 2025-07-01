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

## Components

- **executor/** - GPU machine agent
- **miner/** - Bittensor miner neuron
- **validator/** - Bittensor validator neuron
- **gpu-attestor/** - GPU verification tool
- **public-api/** - External HTTP API

## Usage

### Building
```bash
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
docker-compose -f compose.dev.yml up
```

### Running in Production
```bash
cd scripts/{component}
docker-compose -f compose.prod.yml up -d
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