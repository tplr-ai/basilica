#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="basilica/miner"
IMAGE_TAG="latest"
EXTRACT_BINARY=true
BUILD_IMAGE=true
RELEASE_MODE=true
FEATURES=""
NO_CACHE=false

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
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--image-name NAME] [--image-tag TAG] [--no-extract] [--no-image] [--debug] [--features FEATURES] [--no-cache]"
            echo ""
            echo "Options:"
            echo "  --image-name NAME     Docker image name (default: basilica/miner)"
            echo "  --image-tag TAG       Docker image tag (default: latest)"
            echo "  --no-extract          Don't extract binary to local filesystem"
            echo "  --no-image            Skip Docker image creation"
            echo "  --debug               Build in debug mode"
            echo "  --features FEATURES   Additional cargo features to enable"
            echo "  --no-cache            Rebuild without using Docker cache"
            echo "  --help                Show this help message"
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

    DOCKER_BUILD_FLAGS="--platform linux/amd64"
    if [[ "$NO_CACHE" == "true" ]]; then
        echo "Building without cache..."
        DOCKER_BUILD_FLAGS="$DOCKER_BUILD_FLAGS --no-cache"
    fi

    docker build \
        $DOCKER_BUILD_FLAGS \
        $BUILD_ARGS \
        -f scripts/miner/Dockerfile \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
    echo "Docker image built successfully"
fi

if [[ "$EXTRACT_BINARY" == "true" ]]; then
    echo "Extracting miner binary..."
    container_id=$(docker create "$IMAGE_NAME:$IMAGE_TAG")
    docker cp "$container_id:/usr/local/bin/basilica-miner" ./basilica-miner
    docker rm "$container_id"
    chmod +x ./basilica-miner
    echo "Binary extracted to: ./basilica-miner"
fi

echo "Build completed successfully!"