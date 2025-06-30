#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="basilica/gpu-attestor"
IMAGE_TAG="latest"
VALIDATOR_PUBLIC_KEY=""
EXTRACT_BINARY=true
BUILD_IMAGE=true
RELEASE_MODE=true
FEATURES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)
            VALIDATOR_PUBLIC_KEY="$2"
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
            echo "Usage: $0 [--key KEY] [--image-name NAME] [--image-tag TAG] [--no-extract] [--no-image] [--debug] [--features FEATURES]"
            exit 0
            ;;
        *)
            if [[ -z "$VALIDATOR_PUBLIC_KEY" ]]; then
                VALIDATOR_PUBLIC_KEY="$1"
            fi
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"
if [[ -z "$VALIDATOR_PUBLIC_KEY" ]]; then
    if [[ -f "private_key.pem" ]]; then
        VALIDATOR_PUBLIC_KEY=$(scripts/extract-pubkey.sh private_key.pem 2>/dev/null || true)
    elif [[ -f "public_key.hex" ]]; then
        VALIDATOR_PUBLIC_KEY=$(cat public_key.hex | tr -d '\n\r ' || true)
    elif [[ -f "public_key.pem" ]]; then
        VALIDATOR_PUBLIC_KEY=$(openssl ec -in public_key.pem -pubin -inform PEM -conv_form compressed -outform DER 2>/dev/null | tail -c 33 | xxd -p -c 33 2>/dev/null || true)
    fi
    
    if [[ -z "$VALIDATOR_PUBLIC_KEY" ]]; then
        echo "Error: No key found. Generate with: ./scripts/gen-key.sh" >&2
        exit 1
    fi
fi
if [[ ${#VALIDATOR_PUBLIC_KEY} -ne 66 ]] || ! echo "$VALIDATOR_PUBLIC_KEY" | grep -qE '^[0-9a-fA-F]{66}$'; then
    echo "Error: Invalid key format" >&2
    exit 1
fi

export VALIDATOR_PUBLIC_KEY
if [[ "$BUILD_IMAGE" == "true" ]]; then
    docker build \
        --build-arg VALIDATOR_PUBLIC_KEY="$VALIDATOR_PUBLIC_KEY" \
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