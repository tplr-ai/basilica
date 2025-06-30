#!/bin/bash
# Install system dependencies for Basilica

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Installing system dependencies for Basilica...${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo -e "${GREEN}Detected Debian/Ubuntu${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            pkg-config \
            libssl-dev \
            protobuf-compiler \
            curl \
            git \
            xxd
        
        # Optional: Docker
        if ! command -v docker &> /dev/null; then
            echo -e "${YELLOW}Docker not found. Install Docker? (y/n)${NC}"
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                curl -fsSL https://get.docker.com | sh
                sudo usermod -aG docker $USER
                echo -e "${GREEN}Docker installed. Please log out and back in for group changes.${NC}"
            fi
        fi
        
        # Optional: NVIDIA drivers for GPU
        if ! command -v nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}NVIDIA drivers not found. Install? (y/n)${NC}"
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo apt-get install -y nvidia-driver-525
                echo -e "${GREEN}NVIDIA drivers installed. Reboot required.${NC}"
            fi
        fi
        
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS/Fedora
        echo -e "${GREEN}Detected RHEL/CentOS/Fedora${NC}"
        sudo yum install -y \
            gcc \
            gcc-c++ \
            make \
            pkgconfig \
            openssl-devel \
            protobuf-compiler \
            curl \
            git
            
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo -e "${GREEN}Detected Arch Linux${NC}"
        sudo pacman -Syu --noconfirm \
            base-devel \
            pkg-config \
            openssl \
            protobuf \
            curl \
            git
    else
        echo -e "${RED}Unsupported Linux distribution${NC}"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo -e "${GREEN}Detected macOS${NC}"
    
    # Install Homebrew if not present
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install \
        pkg-config \
        openssl \
        protobuf \
        xxd
        
    echo -e "${YELLOW}Note: GPU attestation is not supported on macOS${NC}"
    
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Update Rust
echo -e "${YELLOW}Updating Rust toolchain...${NC}"
rustup update stable
rustup component add rustfmt clippy

# Install cargo tools
echo -e "${YELLOW}Installing useful cargo tools...${NC}"
cargo install cargo-watch cargo-edit cargo-audit || true

echo -e "${GREEN}Dependencies installed successfully!${NC}"
echo
echo "Next steps:"
echo "1. Generate validator key: just gen-key"
echo "2. Build the project: just build"
echo "3. Run tests: just test"

# Check if reboot is needed
if [[ -f /var/run/reboot-required ]]; then
    echo -e "${YELLOW}System reboot required for some changes to take effect${NC}"
fi