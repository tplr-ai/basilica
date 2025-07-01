# Basilica GPU Attestor

GPU verification tool that validates hardware authenticity and performance.

## Quick Start

```bash
# Generate validator key
../gen-key.sh

# Build with validator key
./build.sh

# Run attestation
docker run --rm --gpus all basilica/gpu-attestor:latest
```

## Files

- `Dockerfile` - Container image with CUDA/OpenCL support
- `build.sh` - Build script with validator key embedding
- `gen-key.sh` - Key generation script (in parent directory)

## Configuration

Copy and edit the configuration:
```bash
cp ../../config/gpu-attestor.toml.example ../../config/gpu-attestor.toml
```

Key settings:
- `[attestor]` - Validator key path and output directory
- `[gpu]` - Device selection and benchmark settings
- `[vdf]` - Verifiable delay function parameters
- `[benchmarks]` - CUDA/OpenCL test configuration

## Build Requirements

- Validator public key (generated with gen-key.sh)
- CUDA toolkit (for CUDA support)
- OpenCL headers (for OpenCL support)

## Build Options

```bash
# Build with specific key
./build.sh --key <hex>

# Debug build
./build.sh --debug

# Custom image name
./build.sh --image-name my-attestor --image-tag v1.0
```

## Running Attestation

```bash
# Run on all GPUs
docker run --rm --gpus all \
  -v $(pwd)/output:/attestation \
  basilica/gpu-attestor:latest

# Run on specific GPU
docker run --rm --gpus '"device=0"' \
  -v $(pwd)/output:/attestation \
  basilica/gpu-attestor:latest
```

## Output

Attestation results are saved as JSON:
- GPU specifications
- Performance benchmarks
- VDF computation proof
- Cryptographic signature

## Integration

The attestor is deployed to executor machines and run by validators via SSH to verify GPU authenticity.