# Executor Build Pipeline

Build basilica executor with Docker and extract local binaries.

## Quick Start

```bash
# Build everything (Docker image + local binary)
./scripts/executor/build.sh

# Run with Docker
docker run --rm -p 8080:8080 -p 8081:8081 -v /var/run/docker.sock:/var/run/docker.sock basilica/executor:latest

# Run locally (after extraction)
./executor --gen-config  # Generate sample config
./executor --config executor.toml
```

## Build Options

```bash
./build.sh --image-name <name>   # Custom image name (default: basilica/executor)
./build.sh --image-tag <tag>     # Custom image tag (default: latest)
./build.sh --no-extract          # Don't extract binary to local filesystem
./build.sh --no-image            # Skip Docker image creation (local build only)
./build.sh --debug               # Debug build instead of release
./build.sh --features <features> # Additional cargo features to enable
```

## Examples

```bash
# Build debug version with custom name
./scripts/executor/build.sh --debug --image-name my-executor --image-tag dev

# Build with custom features (if any are added to executor)
./scripts/executor/build.sh --features "some-feature"

# Only extract binary (don't rebuild image)
./scripts/executor/build.sh --no-image

# Only build image (don't extract binary)
./scripts/executor/build.sh --no-extract
```

## Docker Usage

The built Docker image includes:

- Executor binary at `/usr/local/bin/executor`
- Docker daemon access (requires mounting socket)
- Required system dependencies

### Running with Docker

```bash
# Basic run
docker run --rm basilica/executor:latest --help

# Simple run with Docker access
docker run -d \
  --name basilica-executor \
  -p 8080:8080 \
  -p 8081:8081 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  basilica/executor:latest

# With custom config
docker run -d \
  -v $(pwd)/executor.toml:/app/executor.toml:ro \
  basilica/executor:latest --config /app/executor.toml
```

### Docker Compose for Host Management

For full host management capabilities, use Docker Compose:

```bash
# Build the image first
./scripts/executor/build.sh

# Start with Docker Compose
cd scripts/executor
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The Docker Compose setup provides:
- **Host network access** - Full network visibility
- **Privileged mode** - System-level operations
- **Docker socket** - Container management
- **Host filesystem** - SSH user management
- **System directories** - Process and hardware monitoring

## Local Binary Usage

After extraction, the binary supports all executor commands:

```bash
# Generate sample configuration
./executor --gen-config

# Run executor daemon
./executor --config executor.toml

# Validator access management
sudo ./executor validator grant --hotkey "5ABC..." --ssh-public-key "ssh-rsa ..."
sudo ./executor validator list
sudo ./executor validator revoke --hotkey "5ABC..."

# System monitoring
./executor system info
./executor system health
```

## Build Requirements

- Docker (for containerized builds)
- Rust 1.87+ (for local development)
- OpenSSL development libraries
- pkg-config

## Security Notes

- The Docker image runs SSH server on port 22
- Validator SSH access creates system users dynamically
- Requires Docker socket access for container management
- Uses sudo for system-level operations
