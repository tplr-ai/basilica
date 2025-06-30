# GPU Attestor Build Pipeline

Build gpu-attestor with Docker and local binaries.

## Quick Start

```bash
# Generate keys
./scripts/gen-key.sh

# Build everything
./scripts/gpu-attestor/build.sh

# Run with Docker
docker run --rm -v $(pwd)/output:/attestation basilica/gpu-attestor:latest

# Run locally
./gpu-attestor
```

## Build Options

```bash
./build.sh --key <hex>           # Specify key
./build.sh --no-extract          # Don't extract binary to local
./build.sh --no-image            # Skip Docker image creation
./build.sh --debug               # Debug build
./build.sh --image-name <name>   # Custom image name
./build.sh --features <features> # Cargo features
```

Build always happens in Docker. `--no-extract` skips copying the binary to local filesystem.

## Key Detection

Auto-detects from: `private_key.pem` → `public_key.hex` → `public_key.pem`
