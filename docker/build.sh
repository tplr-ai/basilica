#!/bin/bash
# Docker build script for Basilica services

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Function to build an image
build_image() {
    local service=$1
    local dockerfile=$2
    local image_name=$3
    
    echo -e "${YELLOW}Building $service...${NC}"
    
    if docker build -f "$dockerfile" -t "$image_name" .; then
        echo -e "${GREEN}Successfully built $service image: $image_name${NC}"
    else
        echo -e "${RED}Failed to build $service${NC}"
        exit 1
    fi
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [SERVICE]"
    echo ""
    echo "SERVICE:"
    echo "  miner      Build miner image"
    echo "  executor   Build executor image"
    echo "  all        Build all images (default)"
    echo ""
    echo "OPTIONS:"
    echo "  --push     Push images to registry after building"
    echo "  --tag TAG  Additional tag for the images (default: latest)"
    echo "  --help     Show this help message"
}

# Parse arguments
PUSH=false
TAG="latest"
SERVICE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        miner|executor|all)
            SERVICE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Ensure we have the validator key
if [ ! -f "public_key.hex" ]; then
    echo -e "${YELLOW}Generating validator public key...${NC}"
    if [ -f "scripts/gen-key.sh" ]; then
        bash scripts/gen-key.sh
    else
        echo -e "${RED}Cannot find gen-key.sh script${NC}"
        exit 1
    fi
fi

# Build images
echo -e "${GREEN}Starting Docker builds...${NC}"

if [ "$SERVICE" = "miner" ] || [ "$SERVICE" = "all" ]; then
    build_image "miner" "docker/miner.Dockerfile" "basilica-miner:$TAG"
fi

if [ "$SERVICE" = "executor" ] || [ "$SERVICE" = "all" ]; then
    build_image "executor" "docker/executor.Dockerfile" "basilica-executor:$TAG"
fi

# Push images if requested
if [ "$PUSH" = true ]; then
    echo -e "${YELLOW}Pushing images...${NC}"
    
    if [ "$SERVICE" = "miner" ] || [ "$SERVICE" = "all" ]; then
        docker push "basilica-miner:$TAG"
    fi
    
    if [ "$SERVICE" = "executor" ] || [ "$SERVICE" = "all" ]; then
        docker push "basilica-executor:$TAG"
    fi
    
    echo -e "${GREEN}Images pushed successfully${NC}"
fi

echo -e "${GREEN}Build complete!${NC}"

# Show next steps
echo ""
echo "Next steps:"
echo "1. Create configuration files in ./config/"
echo "2. Run with docker-compose: docker-compose up -d"
echo "3. Or run individually:"
echo "   - Miner: docker run -p 8080:8080 -v \$(pwd)/config:/config basilica-miner:$TAG"
echo "   - Executor: docker run --gpus all -p 50051:50051 -v \$(pwd)/config:/config basilica-executor:$TAG"