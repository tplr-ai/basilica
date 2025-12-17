#!/bin/bash
# Refresh miner revenue summary for a given time period
#
# Usage:
#   ./refresh_miner_revenue.sh --start "2024-01-01" --end "2024-01-31"
#   ./refresh_miner_revenue.sh --start "2024-01-01" --end "2024-01-31" --endpoint localhost:50051
#
# Prerequisites:
#   - grpcurl installed (brew install grpcurl on macOS)
#   - Access to the billing gRPC service

set -euo pipefail

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
ENDPOINT="${BILLING_GRPC_ENDPOINT:-localhost:50051}"
COMPUTATION_VERSION=1
PROTO_PATH="$REPO_ROOT/crates/basilica-protocol/proto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) PERIOD_START="$2"; shift 2 ;;
        --end) PERIOD_END="$2"; shift 2 ;;
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --version) COMPUTATION_VERSION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --start <YYYY-MM-DD> --end <YYYY-MM-DD> [--endpoint host:port] [--version N]"
            echo ""
            echo "Options:"
            echo "  --start     Period start date (YYYY-MM-DD, e.g., 2024-01-01)"
            echo "  --end       Period end date (YYYY-MM-DD, must be before today, e.g., 2024-01-31)"
            echo "  --endpoint  gRPC endpoint (default: localhost:50051, or BILLING_GRPC_ENDPOINT env var)"
            echo "  --version   Computation version (default: 1)"
            echo ""
            echo "Example:"
            echo "  $0 --start '2024-01-01' --end '2024-01-31'"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "${PERIOD_START:-}" || -z "${PERIOD_END:-}" ]]; then
    echo "Error: --start and --end are required"
    echo "Usage: $0 --start <YYYY-MM-DD> --end <YYYY-MM-DD> [--endpoint host:port] [--version N]"
    echo "Example: $0 --start '2024-01-01' --end '2024-01-31'"
    exit 1
fi

# Check if grpcurl is installed
if ! command -v grpcurl &> /dev/null; then
    echo "Error: grpcurl is not installed"
    echo "Install with: brew install grpcurl"
    exit 1
fi

# Build JSON request
REQUEST=$(cat <<EOF
{
  "period_start": "$PERIOD_START",
  "period_end": "$PERIOD_END",
  "computation_version": $COMPUTATION_VERSION
}
EOF
)

echo "Calling RefreshMinerRevenueSummary on $ENDPOINT..."
echo "Request: $REQUEST"
echo ""

# Call gRPC endpoint
grpcurl \
    -import-path "$PROTO_PATH" \
    -proto billing.proto \
    -d "$REQUEST" \
    "$ENDPOINT" \
    basilica.billing.v1.BillingService/RefreshMinerRevenueSummary
