#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NETWORKS=("test" "finney")

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [NETWORKS...]

Generate fresh metadata for Bittensor networks.

OPTIONS:
    -n, --network NETWORK       Generate for specific network (test, finney, local)
    -a, --all                   Generate for all networks (default)
    -h, --help                  Show this help

NETWORKS:
    test     - Test network
    finney   - Mainnet
    local    - Local development

EXAMPLES:
    $0                          # Generate for test and finney
    $0 --network finney         # Generate for finney only
    $0 test finney              # Generate for specific networks

EOF
    exit 1
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

validate_network() {
    local network="$1"
    case "$network" in
        test|finney|local)
            return 0
            ;;
        *)
            echo "ERROR: Invalid network '$network'. Must be: test, finney, local"
            return 1
            ;;
    esac
}

generate_metadata_for_network() {
    local network="$1"

    log "Generating metadata for $network"

    cd "$PROJECT_ROOT/crates/bittensor"

    if ! cargo run --bin generate-metadata --features="generate-metadata" -- "$network"; then
        echo "ERROR: Failed to generate metadata for $network"
        return 1
    fi

    local metadata_file="metadata/${network}.rs"
    if [[ ! -f "$metadata_file" ]]; then
        echo "ERROR: Metadata file not created: $metadata_file"
        return 1
    fi

    local file_size=$(stat -c%s "$metadata_file" 2>/dev/null || stat -f%z "$metadata_file" 2>/dev/null || echo "0")
    if [[ "$file_size" -lt 1000000 ]]; then
        echo "ERROR: Metadata file appears incomplete (size: $file_size bytes)"
        return 1
    fi

    log "Generated metadata for $network ($file_size bytes)"
    return 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--network)
            if ! validate_network "$2"; then
                exit 1
            fi
            NETWORKS=("$2")
            shift 2
            ;;
        -a|--all)
            NETWORKS=("test" "finney")
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
        *)
            if ! validate_network "$1"; then
                exit 1
            fi
            NETWORKS=("$1")
            shift
            ;;
    esac
done

# If networks were passed as positional arguments, use those
if [[ $# -gt 0 ]]; then
    NETWORKS=()
    for network in "$@"; do
        if ! validate_network "$network"; then
            exit 1
        fi
        NETWORKS+=("$network")
    done
fi

log "Generating metadata for networks: ${NETWORKS[*]}"

if [[ ! -d "$PROJECT_ROOT/crates/bittensor" ]]; then
    echo "ERROR: Bittensor crate not found at crates/bittensor"
    exit 1
fi

failed_networks=()
for network in "${NETWORKS[@]}"; do
    if ! generate_metadata_for_network "$network"; then
        failed_networks+=("$network")
    fi
done

if [[ ${#failed_networks[@]} -gt 0 ]]; then
    echo "ERROR: Failed to generate metadata for: ${failed_networks[*]}"
    exit 1
fi

log "Metadata generation completed for: ${NETWORKS[*]}"
