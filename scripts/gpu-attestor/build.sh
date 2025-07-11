#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="basilica/gpu-attestor"
IMAGE_TAG="latest"
EXTRACT_BINARY=true
BUILD_IMAGE=true
RELEASE_MODE=true
FEATURES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)
            # Validator key no longer required, ignore for backward compatibility
            shift 2
            ;;
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
        --help)
            echo "Usage: $0 [--image-name NAME] [--image-tag TAG] [--no-extract] [--no-image] [--debug] [--features FEATURES]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"
if [[ "$BUILD_IMAGE" == "true" ]]; then
    docker build \
        --build-arg FEATURES="$FEATURES" \
        -f scripts/gpu-attestor/Dockerfile \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
fi
if [[ "$EXTRACT_BINARY" == "true" ]]; then
    container_id=$(docker create "$IMAGE_NAME:$IMAGE_TAG")
    docker cp "$container_id:/usr/local/bin/gpu-attestor" ./gpu-attestor
    docker rm "$container_id"
    chmod +x ./gpu-attestor
fi