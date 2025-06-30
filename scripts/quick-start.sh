#!/bin/bash
set -e

# Quick start script for Basilica miner and executor

echo "Basilica Quick Start Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root when needed
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}This script must be run as root for executor setup${NC}"
        echo "Please run: sudo $0 $@"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check NVIDIA GPU (for executor)
    if [[ "$1" == "executor" ]] || [[ "$1" == "both" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not be available.${NC}"
        else
            echo -e "${GREEN}GPU detected:${NC}"
            nvidia-smi --query-gpu=name --format=csv,noheader
        fi
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
    
    if [[ "$service" == "executor" ]]; then
        if [ ! -f "$config_dir/executor.toml" ]; then
            echo -e "${YELLOW}Creating default executor configuration...${NC}"
            cat > "$config_dir/executor.toml" << EOF
# Executor Configuration
[server]
host = "0.0.0.0"
port = 50051

managing_miner_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

[system]
enable_gpu_monitoring = true
max_cpu_usage = 90.0
max_memory_usage = 90.0

[docker]
socket_path = "/var/run/docker.sock"
enable_gpu_support = true

[validator]
enabled = true

[logging]
level = "info"
format = "pretty"
EOF
            echo -e "${GREEN}Created $config_dir/executor.toml${NC}"
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
    if [ ! -f "target/release/miner" ]; then
        echo -e "${YELLOW}Building miner...${NC}"
        cargo build --release -p miner
    fi
    
    # Run miner
    export BASILCA_CONFIG_FILE=config/miner.toml
    echo -e "${GREEN}Starting miner on port 8080...${NC}"
    ./target/release/miner
}

# Run executor
run_executor() {
    check_root
    
    echo -e "${YELLOW}Starting Basilica Executor...${NC}"
    
    create_default_config "executor"
    
    # Build if needed
    if [ ! -f "target/release/executor" ]; then
        echo -e "${YELLOW}Building executor...${NC}"
        cargo build --release -p executor
    fi
    
    # Set environment variables
    export BASILCA_CONFIG_FILE=config/executor.toml
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    
    echo -e "${GREEN}Starting executor on port 50051...${NC}"
    ./target/release/executor
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
    
    if [[ "$service" == "executor" ]]; then
        check_root
        
        echo -e "${YELLOW}Building and running executor with Docker...${NC}"
        
        create_default_config "executor"
        
        # Build Docker image
        docker build -f docker/executor.Dockerfile -t basilica-executor .
        
        # Stop existing container if any
        docker stop basilica-executor 2>/dev/null || true
        docker rm basilica-executor 2>/dev/null || true
        
        # Run container with GPU support
        docker run -d \
            --name basilica-executor \
            --gpus all \
            --privileged \
            -p 50051:50051 \
            -v $(pwd)/config:/config \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -e BASILCA_CONFIG_FILE=/config/executor.toml \
            -e VALIDATOR_PUBLIC_KEY=$(cat public_key.hex) \
            basilica-executor
        
        echo -e "${GREEN}Executor started! Check logs with: docker logs basilica-executor${NC}"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] <SERVICE>"
    echo ""
    echo "SERVICE:"
    echo "  miner      Run the miner service"
    echo "  executor   Run the executor service (requires root)"
    echo "  both       Run both services (requires root)"
    echo ""
    echo "OPTIONS:"
    echo "  --docker   Run services in Docker containers"
    echo "  --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 miner                    # Run miner directly"
    echo "  $0 executor                 # Run executor directly (needs sudo)"
    echo "  $0 --docker miner          # Run miner in Docker"
    echo "  sudo $0 --docker executor   # Run executor in Docker with GPU"
    echo "  sudo $0 --docker both       # Run both services in Docker"
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
            miner|executor|both)
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
        if [[ "$service" == "both" ]]; then
            run_docker "miner"
            run_docker "executor"
            echo -e "${GREEN}Both services started!${NC}"
            echo "Miner: http://localhost:8080"
            echo "Executor: localhost:50051 (gRPC)"
        else
            run_docker $service
        fi
    else
        if [[ "$service" == "both" ]]; then
            echo -e "${RED}Running both services directly is not supported.${NC}"
            echo "Please run them in separate terminals or use --docker"
            exit 1
        elif [[ "$service" == "miner" ]]; then
            run_miner
        elif [[ "$service" == "executor" ]]; then
            run_executor
        fi
    fi
}

# Run main function
main "$@"