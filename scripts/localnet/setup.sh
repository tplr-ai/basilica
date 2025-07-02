#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Basilica Localnet Setup ==="
echo ""

# Check dependencies
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Setup wallets
echo "1. Setting up wallets..."
./setup-wallets.sh

# Generate SSH keys
echo ""
echo "2. Generating SSH keys..."
if [ -f "./ssh-keys/generate-static-keys.sh" ]; then
    ./ssh-keys/generate-static-keys.sh
else
    mkdir -p ssh-keys
    cd ssh-keys && ../../../localnet/ssh-keys/generate-static-keys.sh && cd ..
fi

# Start Subtensor (only Alice node to avoid equivocation)
echo ""
echo "3. Starting Subtensor..."
docker compose -f compose.yml up -d alice

# Wait for chain
echo "   Waiting for chain to be ready..."
while ! nc -z localhost 9944 2>/dev/null; do 
    echo -n "."
    sleep 2
done
echo " Ready!"

# Build and start Basilica services
echo ""
echo "4. Starting Basilica services..."
docker compose -f compose.yml build executor miner validator public-api
docker compose -f compose.yml up -d executor redis prometheus grafana
sleep 5  # Wait for executor to be healthy
docker compose -f compose.yml up -d miner validator public-api

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Services running:"
echo "  - Subtensor (Alice): ws://localhost:9944"
echo "  - Executor: localhost:50052 (gRPC), http://localhost:8082/metrics"
echo "  - Miner: localhost:8092 (gRPC), http://localhost:8090/metrics"
echo "  - Validator: localhost:50053 (gRPC), http://localhost:3002 (API)"
echo "  - Public API: http://localhost:8000 (may have issues with localnet)"
echo "  - Redis: localhost:6379"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "Note: Services run without registration (skip_registration=true for miner, --local-test for validator)"
echo ""
echo "To check status: ./test-services.sh"
echo "To stop: docker compose down"