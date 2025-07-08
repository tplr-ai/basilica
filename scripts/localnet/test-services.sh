#!/bin/bash
set -euo pipefail

echo "1. Checking Subtensor node..."
curl -s http://localhost:9944/health | jq . || echo "Alice node not responding"

echo ""
echo "2. Checking Executor service..."
nc -zv localhost 50052 2>&1 || echo "Executor gRPC not accessible"
curl -s http://localhost:8082/metrics | grep -q "basilica" && echo "Executor metrics accessible" || echo "Executor metrics not accessible"

echo ""
echo "3. Checking Redis..."
docker exec basilica-redis-localnet redis-cli ping 2>&1 || echo "Redis not responding"

echo ""
echo "4. Checking monitoring..."
curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy" && echo "Prometheus healthy" || echo "Prometheus not healthy"
curl -s http://localhost:3000/api/health | jq . || echo "Grafana not responding"

echo ""
echo "5. Container status..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep basilica
