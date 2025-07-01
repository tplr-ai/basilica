#!/bin/bash
# Run localtest environment using a combination of existing and test-specific services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
OPERATION="up"
USE_EXISTING_SERVICES=false
COMPOSE_FILES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        up|down|logs|ps|build)
            OPERATION="$1"
            ;;
        --use-existing)
            USE_EXISTING_SERVICES=true
            ;;
        -h|--help)
            echo "Usage: $0 [OPERATION] [OPTIONS]"
            echo ""
            echo "Operations:"
            echo "  up        Start all services (default)"
            echo "  down      Stop all services"
            echo "  logs      Show logs"
            echo "  ps        Show running services"
            echo "  build     Build all images"
            echo ""
            echo "Options:"
            echo "  --use-existing  Use existing miner/executor from docker/"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
    shift
done

# Function to check if services are already running
check_existing_services() {
    local running=false
    if docker ps --format "{{.Names}}" | grep -q "basilica-miner"; then
        echo -e "${BLUE}ℹ️  Found running miner service${NC}"
        running=true
    fi
    if docker ps --format "{{.Names}}" | grep -q "basilica-executor"; then
        echo -e "${BLUE}ℹ️  Found running executor service${NC}"
        running=true
    fi
    
    if [ "$running" = true ] && [ "$USE_EXISTING_SERVICES" = false ]; then
        echo -e "${YELLOW}⚠️  Existing services detected. Use --use-existing to integrate with them.${NC}"
        echo -e "${YELLOW}   Or stop them first with: cd docker && docker compose -f docker-compose.dev-sqlite.yml down${NC}"
        exit 1
    fi
}

# Function to setup test environment
setup_test_env() {
    echo -e "${YELLOW}Setting up test environment...${NC}"
    
    # Ensure keys exist
    if [ ! -f "$SCRIPT_DIR/keys/id_rsa" ]; then
        echo -e "${YELLOW}Generating test SSH keys...${NC}"
        mkdir -p "$SCRIPT_DIR/keys"
        ssh-keygen -t rsa -b 2048 -f "$SCRIPT_DIR/keys/id_rsa" -N "" -q
    fi
    
    # Setup test configuration
    if [ ! -f "$SCRIPT_DIR/test.conf" ]; then
        if [ -f "$SCRIPT_DIR/test.conf.example" ]; then
            cp "$SCRIPT_DIR/test.conf.example" "$SCRIPT_DIR/test.conf"
            echo -e "${GREEN}✓ Created test.conf from example${NC}"
        fi
    fi
}

# Main operations
case "$OPERATION" in
    build)
        echo -e "${GREEN}Building localtest images...${NC}"
        cd "$SCRIPT_DIR"
        ./build-localtest.sh
        ;;
        
    up)
        check_existing_services
        setup_test_env
        
        echo -e "${GREEN}Starting localtest environment...${NC}"
        
        if [ "$USE_EXISTING_SERVICES" = true ]; then
            # Use only test-specific services
            echo -e "${BLUE}Using existing miner/executor services${NC}"
            cd "$SCRIPT_DIR"
            docker compose -f docker-compose-simplified.yml up -d validator subtensor ssh-target db-init
            
            # Wait for services
            echo -e "${YELLOW}Waiting for services to start...${NC}"
            sleep 10
            
            # Register test neurons if subtensor is ready
            if docker exec basilica-subtensor curl -s http://localhost:9933/health > /dev/null 2>&1; then
                echo -e "${YELLOW}Registering test neurons...${NC}"
                ./register-test-neurons.sh || echo -e "${YELLOW}⚠️  Neuron registration failed (may already be registered)${NC}"
            fi
        else
            # Start everything including miner/executor
            echo -e "${BLUE}Starting full test stack...${NC}"
            cd "$SCRIPT_DIR"
            docker compose up -d
            
            # Wait and register
            echo -e "${YELLOW}Waiting for services to start...${NC}"
            sleep 15
            
            if docker exec basilica-subtensor curl -s http://localhost:9933/health > /dev/null 2>&1; then
                echo -e "${YELLOW}Registering test neurons...${NC}"
                ./register-test-neurons.sh || echo -e "${YELLOW}⚠️  Neuron registration failed${NC}"
            fi
        fi
        
        echo -e "${GREEN}✓ Localtest environment is ready!${NC}"
        echo -e "\nServices:"
        docker compose ps
        echo -e "\nYou can now run tests with: ${YELLOW}./test-workflow.sh${NC}"
        ;;
        
    down)
        echo -e "${YELLOW}Stopping localtest environment...${NC}"
        cd "$SCRIPT_DIR"
        
        if [ "$USE_EXISTING_SERVICES" = true ]; then
            docker compose -f docker-compose-simplified.yml down
        else
            docker compose down
        fi
        
        echo -e "${GREEN}✓ Stopped${NC}"
        ;;
        
    logs)
        cd "$SCRIPT_DIR"
        if [ "$USE_EXISTING_SERVICES" = true ]; then
            docker compose -f docker-compose-simplified.yml logs -f
        else
            docker compose logs -f
        fi
        ;;
        
    ps)
        cd "$SCRIPT_DIR"
        if [ "$USE_EXISTING_SERVICES" = true ]; then
            echo -e "${BLUE}Test-specific services:${NC}"
            docker compose -f docker-compose-simplified.yml ps
            echo -e "\n${BLUE}Existing services (from docker/):${NC}"
            docker ps --filter "name=basilica-miner" --filter "name=basilica-executor"
        else
            docker compose ps
        fi
        ;;
esac