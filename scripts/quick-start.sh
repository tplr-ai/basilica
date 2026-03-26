#!/bin/bash
set -e

# Quick start script for Basilica miner

echo "Basilica Quick Start Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi

    # Check if public key exists
    if [ ! -f "public_key.hex" ]; then
        echo -e "${YELLOW}Generating validator public key...${NC}"
        just gen-key
    fi

    echo -e "${GREEN}Prerequisites check complete!${NC}"
}

# Create default configuration
create_default_config() {
    local service=$1
    local config_dir="config"
    
    mkdir -p $config_dir
    
    if [[ "$service" == "miner" ]]; then
        if [ ! -f "$config_dir/miner.toml" ]; then
            echo -e "${YELLOW}Creating default miner configuration...${NC}"
            cat > "$config_dir/miner.toml" << EOF
# Miner Configuration
network = "local"
netuid = 1

[server]
host = "0.0.0.0"
port = 8080

[database]
url = "sqlite:///var/lib/basilica/miner.db"
run_migrations = true

[logging]
level = "info"
format = "pretty"

[bittensor]
network = "local"
netuid = 1
wallet_name = "default"
wallet_hotkey = "default"
EOF
            echo -e "${GREEN}Created $config_dir/miner.toml${NC}"
        fi
    fi
}

# Run miner
run_miner() {
    echo -e "${YELLOW}Starting Basilica Miner...${NC}"
    
    create_default_config "miner"
    
    # Create data directory
    mkdir -p /var/lib/basilica
    
    # Build if needed
    if [ ! -f "target/release/basilica-miner" ]; then
        echo -e "${YELLOW}Building miner...${NC}"
        cargo build --release -p basilica-miner
    fi
    
    # Run miner
    export BASILCA_CONFIG_FILE=config/miner.toml
    echo -e "${GREEN}Starting miner on port 8080...${NC}"
    ./target/release/basilica-miner
}

# Run with Docker
run_docker() {
    local service=$1
    
    if [[ "$service" == "miner" ]]; then
        echo -e "${YELLOW}Building and running miner with Docker...${NC}"
        
        create_default_config "miner"
        
        # Build Docker image
        docker build -f docker/miner.Dockerfile -t basilica-miner .
        
        # Stop existing container if any
        docker stop basilica-miner 2>/dev/null || true
        docker rm basilica-miner 2>/dev/null || true
        
        # Run container
        docker run -d \
            --name basilica-miner \
            -p 8080:8080 \
            -v $(pwd)/config:/config \
            -v /var/lib/basilica:/var/lib/basilica \
            -e BASILCA_CONFIG_FILE=/config/miner.toml \
            basilica-miner
        
        echo -e "${GREEN}Miner started! Check logs with: docker logs basilica-miner${NC}"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] <SERVICE>"
    echo ""
    echo "SERVICE:"
    echo "  miner      Run the miner service"
    echo ""
    echo "OPTIONS:"
    echo "  --docker   Run services in Docker containers"
    echo "  --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 miner                    # Run miner directly"
    echo "  $0 --docker miner          # Run miner in Docker"
}

# Main script logic
main() {
    local use_docker=false
    local service=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                use_docker=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            miner)
                service=$1
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                usage
                exit 1
                ;;
        esac
    done

    if [ -z "$service" ]; then
        echo -e "${RED}Error: No service specified${NC}"
        usage
        exit 1
    fi

    # Check prerequisites
    check_prerequisites $service

    # Run services
    if [ "$use_docker" = true ]; then
        run_docker $service
    else
        if [[ "$service" == "miner" ]]; then
            run_miner
        fi
    fi
}

# Run main function
main "$@"