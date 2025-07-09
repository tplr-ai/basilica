#!/bin/bash
# Build all Basilica images for localtest using existing build scripts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}Building Basilica images for localtest...${NC}"
echo "========================================"

# Function to build and tag image
build_and_tag() {
    local service=$1
    local tag="basilica/${service}:localtest"
    
    echo -e "\n${YELLOW}Building ${service}...${NC}"
    
    # Use the existing build script
    if [ -f "$PROJECT_ROOT/scripts/${service}/build.sh" ]; then
        # Build using the service's build script
        cd "$PROJECT_ROOT"
        ./scripts/${service}/build.sh
        
        # The build script creates a binary, now we need to build the Docker image
        if [ -f "$PROJECT_ROOT/scripts/${service}/Dockerfile" ]; then
            echo -e "${YELLOW}Creating Docker image for ${service}...${NC}"
            docker build -t "$tag" -f "scripts/${service}/Dockerfile" \
                --build-arg BUILD_MODE=release \
                .
            echo -e "${GREEN}✓ Built ${tag}${NC}"
        else
            echo -e "${RED}✗ No Dockerfile found for ${service}${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ No build script found for ${service}${NC}"
        return 1
    fi
}

# Generate key for gpu-attestor if needed
if [ ! -f "$PROJECT_ROOT/public_key.hex" ]; then
    echo -e "${YELLOW}Generating key for gpu-attestor...${NC}"
    cd "$PROJECT_ROOT"
    chmod +x scripts/gen-key.sh
    ./scripts/gen-key.sh
fi


# Build all services
services=("validator" "miner" "executor")
failed=0

for service in "${services[@]}"; do
    if build_and_tag "$service"; then
        echo -e "${GREEN}✓ Successfully built ${service}${NC}"
    else
        echo -e "${RED}✗ Failed to build ${service}${NC}"
        ((failed++))
    fi
done

# Build additional test-specific images
echo -e "\n${YELLOW}Building test-specific images...${NC}"

# SSH target for testing
echo -e "${YELLOW}Building SSH target...${NC}"
docker build -t basilica/ssh-target:localtest -f - <<EOF
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y openssh-server sudo && \
    mkdir /var/run/sshd && \
    echo 'root:test' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
EOF

# Summary
echo -e "\n========================================"
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All images built successfully!${NC}"
    echo -e "\nImages created:"
    for service in "${services[@]}"; do
        echo -e "  - basilica/${service}:localtest"
    done
    echo -e "  - basilica/ssh-target:localtest"
    echo -e "\nYou can now run: ${YELLOW}docker compose up${NC}"
else
    echo -e "${RED}Build failed for $failed service(s)${NC}"
    exit 1
fi