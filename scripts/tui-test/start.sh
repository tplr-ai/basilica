#!/bin/bash
# Start local API for TUI testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Basilica API for TUI testing..."
echo "   This runs the API in dev mode (no Bittensor required)"
echo ""

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

# Start services
docker compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."

# Wait for API to be healthy
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Waiting... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ API failed to start. Check logs with: docker compose logs api"
    exit 1
fi

echo ""
echo "🎉 Services are running!"
echo ""
echo "Available endpoints:"
echo "  - API:     http://localhost:8000"
echo "  - Health:  http://localhost:8000/health"
echo "  - Metrics: http://localhost:9401/metrics"
echo ""
echo "To run the TUI against this API:"
echo "  BASILICA_API_URL=http://localhost:8000 cargo run -p basilica-tui"
echo ""
echo "Or run TUI in dev mode (mock data, no API needed):"
echo "  cargo run -p basilica-tui -- --dev"
echo ""
echo "To stop services:"
echo "  cd scripts/tui-test && docker compose down"

