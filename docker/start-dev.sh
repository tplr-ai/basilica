#!/bin/bash
# One-command startup for local development with remote executors

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Basilica Development Environment"
echo "==========================================="

# Step 1: Build the executor binary that will be deployed
echo "📦 Building executor binary..."
cd "$PROJECT_ROOT"
cargo build --release -p executor

# Step 2: Check if we need to configure remote machines
if grep -q "YOUR_VAST_AI_HOST" "$SCRIPT_DIR/configs/miner-local.toml"; then
    echo ""
    echo "⚠️  Remote machine configuration needed!"
    echo "Please update $SCRIPT_DIR/configs/miner-local.toml with your remote machine details."
    echo ""
    echo "Or run: ./setup-vast-gpu.sh"
    exit 1
fi

# Step 3: Start services with docker-compose
echo "🐳 Starting Docker services..."
cd "$SCRIPT_DIR"
docker compose -f docker-compose.dev-remote.yml up -d

# Step 4: Wait for miner to be healthy
echo "⏳ Waiting for miner to start..."
sleep 5

# Step 5: Deploy executors to remote machines
echo "🚀 Deploying executors to remote machines..."
docker exec basilica-miner-dev miner deploy-executors || {
    echo "❌ Failed to deploy executors. Checking miner logs..."
    docker logs basilica-miner-dev --tail 50
    exit 1
}

# Step 6: Show status
echo ""
echo "✅ Development environment started!"
echo ""
echo "📊 Services:"
echo "  - Miner API: http://localhost:8092"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - gRPC UI: http://localhost:8081"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: docker compose -f docker-compose.dev-remote.yml logs -f"
echo "  - Check executor status: docker exec basilica-miner-dev miner deploy-executors --status-only"
echo "  - Stop everything: docker compose -f docker-compose.dev-remote.yml down"
echo ""

# Optional: Follow logs
read -p "Follow logs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose -f docker-compose.dev-remote.yml logs -f
fi