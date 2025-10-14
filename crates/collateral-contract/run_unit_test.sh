#!/bin/bash

# Set environment variables
export BITTENSOR_NETWORK=local
export OPEN_EVM_PRIVATE_KEY=5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133
export LOCAL_CHAIN_ID=31337

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Collateral Contract Test Runner${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to run a specific test
run_test() {
    local test_name=$1
    local flags=$2
    echo -e "${GREEN}Running test: ${test_name}${NC}"
    echo -e "${YELLOW}Environment:${NC}"
    echo "  BITTENSOR_NETWORK=${BITTENSOR_NETWORK}"
    echo "  LOCAL_CHAIN_ID=${LOCAL_CHAIN_ID}"
    echo ""
    cargo test --package collateral-contract --lib -- tests::${test_name} --exact --show-output --include-ignored
}

# Display menu if no arguments provided
if [ $# -eq 0 ]; then
    echo "Available test cases:"
    echo "  1) test_collateral_deposit_reclaim_finalize"
    echo "  2) test_deploy_deposit_reclaim_deny"
    echo "  3) test_deploy_deposit_slash"
    echo "  4) test_deploy_upgrade"
    echo "  5) test_convert_h160_to_public_key"
    echo ""
    read -p "Select test to run (1-5): " choice
    
    case $choice in
        1) run_test "test_collateral_deposit_reclaim_finalize" ;;
        2) run_test "test_deploy_deposit_reclaim_deny" ;;
        3) run_test "test_deploy_deposit_slash" ;;
        4) run_test "test_deploy_upgrade" ;;
        5) run_test "test_convert_h160_to_public_key" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    # Command line argument handling
    case $1 in
        --help|-h)
            echo "Usage: $0 [test_name|--all|--sequential]"
            echo ""
            echo "Options:"
            echo "  --help, -h           Show this help message"
            echo "  test_name            Run specific test (e.g., test_collateral_deposit_reclaim_finalize)"
            echo ""
            echo "Available tests:"
            echo "  - test_collateral_deposit_reclaim_finalize"
            echo "  - test_deploy_deposit_reclaim_deny"
            echo "  - test_deploy_deposit_slash"
            echo "  - test_deploy_upgrade"
            echo "  - test_convert_h160_to_public_key"
            ;;
        --sequential)
            run_all_tests "--test-threads=1"
            ;;
        *)
            run_test "$1" "${@:2}"
            ;;
    esac
fi
