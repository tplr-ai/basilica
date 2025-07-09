#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="basilica/validator"
IMAGE_TAG="latest"
EXTRACT_BINARY=true
BUILD_IMAGE=true
RELEASE_MODE=true
FEATURES=""
VERITAS_BINARIES_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-extract)
            EXTRACT_BINARY=false
            shift
            ;;
        --no-image)
            BUILD_IMAGE=false
            shift
            ;;
        --debug)
            RELEASE_MODE=false
            shift
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --veritas-binaries)
            VERITAS_BINARIES_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--image-name NAME] [--image-tag TAG] [--no-extract] [--no-image] [--debug] [--features FEATURES] [--veritas-binaries DIR]"
            echo ""
            echo "Options:"
            echo "  --image-name NAME         Docker image name (default: basilica/validator)"
            echo "  --image-tag TAG           Docker image tag (default: latest)"
            echo "  --no-extract              Don't extract binary to local filesystem"
            echo "  --no-image                Skip Docker image creation"
            echo "  --debug                   Build in debug mode"
            echo "  --features FEATURES       Additional cargo features to enable"
            echo "  --veritas-binaries DIR    Directory containing executor-binary and validator-binary"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Copy veritas binaries to build context if specified
if [[ -n "$VERITAS_BINARIES_DIR" ]]; then
    if [[ ! -d "$VERITAS_BINARIES_DIR" ]]; then
        echo "Error: Veritas binaries directory does not exist: $VERITAS_BINARIES_DIR"
        exit 1
    fi

    EXECUTOR_BINARY_PATH="$VERITAS_BINARIES_DIR/executor-binary/executor-binary"
    VALIDATOR_BINARY_PATH="$VERITAS_BINARIES_DIR/validator-binary/validator-binary"

    if [[ ! -f "$EXECUTOR_BINARY_PATH" ]]; then
        echo "Error: executor-binary not found at: $EXECUTOR_BINARY_PATH"
        exit 1
    fi

    if [[ ! -f "$VALIDATOR_BINARY_PATH" ]]; then
        echo "Error: validator-binary not found at: $VALIDATOR_BINARY_PATH"
        exit 1
    fi

    echo "Copying veritas binaries to build context..."
    cp "$EXECUTOR_BINARY_PATH" ./executor-binary
    cp "$VALIDATOR_BINARY_PATH" ./validator-binary
    echo "  - executor-binary: copied to ./executor-binary"
    echo "  - validator-binary: copied to ./validator-binary"
else
    # Create empty placeholder files for Docker COPY
    touch ./executor-binary
    touch ./validator-binary
fi

BUILD_ARGS=""
if [[ "$RELEASE_MODE" == "true" ]]; then
    BUILD_ARGS="--build-arg BUILD_MODE=release"
else
    BUILD_ARGS="--build-arg BUILD_MODE=debug"
fi

if [[ -n "$FEATURES" ]]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg FEATURES=$FEATURES"
fi


# Pass Bittensor network configuration if set
if [[ -n "$BITTENSOR_NETWORK" ]]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BITTENSOR_NETWORK=$BITTENSOR_NETWORK"
    echo "Building with BITTENSOR_NETWORK=$BITTENSOR_NETWORK"
fi

if [[ -n "$METADATA_CHAIN_ENDPOINT" ]]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg METADATA_CHAIN_ENDPOINT=$METADATA_CHAIN_ENDPOINT"
    echo "Building with METADATA_CHAIN_ENDPOINT=$METADATA_CHAIN_ENDPOINT"
fi


if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"

    docker build \
        $BUILD_ARGS \
        -f scripts/validator/Dockerfile \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
    echo "Docker image built successfully"
fi

if [[ "$EXTRACT_BINARY" == "true" ]]; then
    echo "Extracting validator binary..."
    container_id=$(docker create "$IMAGE_NAME:$IMAGE_TAG")
    docker cp "$container_id:/usr/local/bin/validator" ./validator
    docker rm "$container_id"
    chmod +x ./validator
    echo "Binary extracted to: ./validator"
fi

echo "Build completed successfully!"
