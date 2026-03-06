#!/bin/bash
# Common functions and utilities for Basilica scripts

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export PURPLE='\033[0;35m'
export NC='\033[0m' # No Color

# Project paths
export BASILICA_ROOT="${BASILICA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export SCRIPTS_DIR="$BASILICA_ROOT/scripts"
export CRATES_DIR="$BASILICA_ROOT/crates"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}=== $1 ===${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're in the correct directory
ensure_basilica_root() {
    if [ ! -f "$BASILICA_ROOT/Cargo.toml" ]; then
        log_error "Not in Basilica root directory"
        log_info "Expected root: $BASILICA_ROOT"
        return 1
    fi
    cd "$BASILICA_ROOT" || return 1
}

# Get list of crates
get_crates() {
    find "$CRATES_DIR" -name "Cargo.toml" -type f | while read -r cargo_file; do
        dirname "$cargo_file" | xargs basename
    done | sort | uniq
}

# Check if a crate exists
crate_exists() {
    local crate=$1
    [ -d "$CRATES_DIR/$crate" ] && [ -f "$CRATES_DIR/$crate/Cargo.toml" ]
}

# Read a substrate hotkey (bytes32) from a wallet pubkey JSON file.
# Returns 0x-prefixed hex string suitable for use as a Solidity bytes32 argument.
read_hotkey() {
    local pubfile="$1"
    if [[ ! -f "$pubfile" ]]; then
        log_error "Hotkey file not found: $pubfile"
        return 1
    fi
    echo "0x$(awk -F'"' '/"publicKey"/ {print $4; exit}' "$pubfile" | sed 's/^0x//')"
}

# Compute the on-chain bytes16 node ID from a seed string (e.g. an IP address).
# Uses the same NodeId::new() derivation as the validator and miner.
compute_node_id() {
    local seed="$1"
    cargo run -q --example compute_node_id -p basilica-common -- "$seed"
}