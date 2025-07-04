#!/bin/bash
set -euo pipefail

# Restart script for Basilica localnet services
# This script rebuilds and restarts services after code changes

echo "=== Restarting Basilica Localnet Services ==="
echo ""

cd "$(dirname "$0")"

# Parse command line arguments
SERVICES=""
BUILD_ONLY=false
NO_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        *)
            SERVICES="$SERVICES $1"
            shift
            ;;
    esac
done

# If no specific services provided, restart all Basilica services
if [ -z "$SERVICES" ]; then
    SERVICES="executor miner validator public-api"
fi

echo "Services to restart: $SERVICES"
echo ""

# Stop the services
echo "1. Stopping services..."
for service in $SERVICES; do
    echo "   Stopping $service..."
    docker compose stop $service 2>/dev/null || true
done

# Remove old containers to ensure clean restart
echo ""
echo "2. Removing old containers..."
for service in $SERVICES; do
    docker compose rm -f $service 2>/dev/null || true
done

# Rebuild the services if not skipped
if [ "$NO_BUILD" = false ]; then
    echo ""
    echo "3. Rebuilding services..."
    docker compose build $SERVICES
fi

# Exit if build-only mode
if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "Build complete. Services not started (--build-only mode)"
    exit 0
fi

# Start the services
echo ""
echo "4. Starting services..."

# Special handling for service dependencies
if [[ " $SERVICES " =~ " executor " ]]; then
    echo "   Starting executor first..."
    docker compose up -d executor

    # Wait for executor to be healthy
    echo "   Waiting for executor to be healthy..."
    for i in {1..30}; do
        if docker compose ps executor | grep -q "healthy"; then
            echo "   Executor is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "   Executor health check timeout"
        fi
        sleep 2
    done
fi

# Start remaining services
for service in $SERVICES; do
    if [ "$service" != "executor" ]; then
        echo "   Starting $service..."
        docker compose up -d $service
    fi
done

# Wait a bit for services to initialize
echo ""
echo "5. Waiting for services to initialize..."
sleep 5

# Check service status
echo ""
echo "6. Checking service status..."
docker compose ps $SERVICES

# Show logs for any services that aren't running
echo ""
for service in $SERVICES; do
    if ! docker compose ps $service | grep -q "Up"; then
        echo "$service is not running. Recent logs:"
        docker compose logs --tail=20 $service
        echo ""
    fi
done
