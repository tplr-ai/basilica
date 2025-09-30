#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_IMAGE="basilica/validator"
TARGET_IMAGE="ghcr.io/one-covenant/basilica/validator"
IMAGE_TAG="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --source-image)
            SOURCE_IMAGE="$2"
            shift 2
            ;;
        --target-image)
            TARGET_IMAGE="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--source-image SOURCE] [--target-image TARGET] [--tag TAG]"
            echo ""
            echo "Options:"
            echo "  --source-image SOURCE     Source Docker image (default: basilica/validator)"
            echo "  --target-image TARGET     Target Docker image (default: ghcr.io/one-covenant/basilica/validator)"
            echo "  --tag TAG                 Image tag (default: latest)"
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

SOURCE_IMAGE_FULL="${SOURCE_IMAGE}:${IMAGE_TAG}"
TARGET_IMAGE_FULL="${TARGET_IMAGE}:${IMAGE_TAG}"

echo "Tagging and pushing validator image..."
echo "  Source: $SOURCE_IMAGE_FULL"
echo "  Target: $TARGET_IMAGE_FULL"

# Check if source image exists
if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^${SOURCE_IMAGE_FULL}$"; then
    echo "Error: Source image $SOURCE_IMAGE_FULL not found"
    echo "Please build the image first using: ./scripts/validator/build.sh"
    exit 1
fi

# Tag the image
echo "Tagging image..."
docker tag "$SOURCE_IMAGE_FULL" "$TARGET_IMAGE_FULL"

# Push the image
echo "Pushing image to registry..."
docker push "$TARGET_IMAGE_FULL"

echo "Successfully pushed $TARGET_IMAGE_FULL"
