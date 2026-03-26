#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="${REGISTRY:-ghcr.io/one-covenant}"
TAG="${TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"

echo "Building Basilica Public Docker images"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"
echo "Platform: $PLATFORM"
echo "----------------------------------------"

cd "$PROJECT_ROOT"

# Public components only
COMPONENTS=(
  "miner"
  "validator"
  "cli"
)

FAILED=()
SUCCEEDED=()

for component in "${COMPONENTS[@]}"; do
  dockerfile="scripts/${component}/Dockerfile"

  if [ ! -f "$dockerfile" ]; then
    echo "⚠️  Skipping $component - Dockerfile not found"
    continue
  fi

  image_name="${REGISTRY}/basilica-${component}:${TAG}"
  echo ""
  echo "🔨 Building $component..."
  echo "   Image: $image_name"

  if docker build \
    --platform "$PLATFORM" \
    --build-arg BUILD_MODE=release \
    -f "$dockerfile" \
    -t "$image_name" \
    . ; then
    echo "✅ Successfully built $component"
    SUCCEEDED+=("$component")
  else
    echo "❌ Failed to build $component"
    FAILED+=("$component")
  fi
done

echo ""
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "Succeeded: ${#SUCCEEDED[@]}"
for comp in "${SUCCEEDED[@]}"; do
  echo "  ✅ $comp"
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "Failed: ${#FAILED[@]}"
  for comp in "${FAILED[@]}"; do
    echo "  ❌ $comp"
  done
  exit 1
else
  echo ""
  echo "🎉 All public images built successfully!"
  echo ""
  echo "To push images to registry:"
  for comp in "${SUCCEEDED[@]}"; do
    echo "  docker push ${REGISTRY}/basilica-${comp}:${TAG}"
  done
fi

