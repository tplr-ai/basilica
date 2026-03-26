#!/bin/bash
# Basilica Localnet - Health Check Script
# Checks connectivity and health of all services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
    echo "Basilica Localnet - Health Check"
    echo ""
    echo "Usage: ./test.sh [-h|--help]"
    echo ""
    echo "Checks connectivity and health of all localnet services."
    echo ""
    echo "Services checked:"
    echo "  1. Subtensor    - HTTP health + WebSocket port (9944)"
    echo "  2. PostgreSQL   - pg_isready via Docker"
    echo "  3. Validator    - HTTP health (8080), metrics (9090)"
    echo "  4. Miner        - Metrics (9091), gRPC (8092), Axon (8091)"
    echo "  5. Monitoring   - Prometheus (9099), Grafana (3000)"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Exit codes:"
    echo "  0  All services healthy"
    echo "  1  One or more services unhealthy"
    echo ""
    echo "See also: ./start.sh, ./restart.sh"
}

[[ "${1:-}" =~ ^(-h|--help)$ ]] && show_help && exit 0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

check_service() {
    local name=$1
    local url=$2
    local expected=${3:-""}

    printf "  %-15s " "${name}:"

    response=$(curl -sf "${url}" 2>/dev/null) && status=0 || status=1

    if [ $status -eq 0 ]; then
        if [ -n "${expected}" ]; then
            if echo "${response}" | grep -q "${expected}"; then
                echo -e "${GREEN}OK${NC}"
            else
                echo -e "${YELLOW}WARN${NC} (unexpected response)"
                OVERALL_STATUS=1
            fi
        else
            echo -e "${GREEN}OK${NC}"
        fi
    else
        echo -e "${RED}FAIL${NC}"
        OVERALL_STATUS=1
    fi
}

check_port() {
    local name=$1
    local host=$2
    local port=$3

    printf "  %-15s " "${name}:"

    if nc -z "${host}" "${port}" 2>/dev/null; then
        echo -e "${GREEN}OK${NC} (port ${port})"
    else
        echo -e "${RED}FAIL${NC} (port ${port} not accessible)"
        OVERALL_STATUS=1
    fi
}

echo "========================================"
echo "  Basilica Localnet Health Check"
echo "========================================"
echo ""

# =============================================================================
# Subtensor
# =============================================================================
echo "[1/5] Subtensor (Local Chain)"
check_service "Health" "http://localhost:9944/health" "Healthy"
check_port "WebSocket" "localhost" "9944"
echo ""

# =============================================================================
# PostgreSQL
# =============================================================================
echo "[2/5] PostgreSQL Database"
printf "  %-15s " "Connection:"
if docker exec basilica-postgres pg_isready -U basilica -d validator > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} (container may not be running)"
    OVERALL_STATUS=1
fi
echo ""

# =============================================================================
# Validator
# =============================================================================
echo "[3/5] Validator Service"
check_service "Health" "http://localhost:8080/health"
check_service "Metrics" "http://localhost:9090/metrics" "basilica"
check_port "API" "localhost" "8080"
echo ""

# =============================================================================
# Miner
# =============================================================================
echo "[4/5] Miner Service"
check_service "Metrics" "http://localhost:9091/metrics" "basilica"
check_port "gRPC" "localhost" "8092"
check_port "Axon" "localhost" "8091"
echo ""

# =============================================================================
# Monitoring
# =============================================================================
echo "[5/5] Monitoring Stack"
check_service "Prometheus" "http://localhost:9099/-/healthy" "Healthy"
check_service "Grafana" "http://localhost:3000/api/health"
echo ""

# =============================================================================
# Container Status
# =============================================================================
echo "========================================"
echo "  Container Status"
echo "========================================"
echo ""
docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "No containers found"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "  Status: ${GREEN}ALL HEALTHY${NC}"
else
    echo -e "  Status: ${RED}SOME SERVICES UNHEALTHY${NC}"
    echo ""
    echo "  Troubleshooting:"
    echo "    - View logs: docker compose logs -f [service]"
    echo "    - Restart:   docker compose restart [service]"
    echo "    - Rebuild:   ./start.sh --build"
fi
echo "========================================"
echo ""

exit $OVERALL_STATUS
