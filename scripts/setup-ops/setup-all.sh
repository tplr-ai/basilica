#!/bin/bash
set -e

# Complete Basilica server setup automation
# Usage: ./scripts/setup-ops/setup-all.sh [config_file]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/servers.conf}"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Create $CONFIG_FILE with your server details:"
    echo "VALIDATOR=user@host:port"
    echo "MINER=user@host:port"
    echo "EXECUTOR=user@host:port"
    exit 1
fi

# Load server configuration
source "$CONFIG_FILE"

# Parse server details
parse_server() {
    local server_def="$1"
    local user_host="${server_def%:*}"
    local port="${server_def##*:}"
    echo "$user_host -p $port"
}

echo "=== Basilica Complete Server Setup ==="
echo "Using configuration: $CONFIG_FILE"
echo ""

# Setup all servers
echo "Step 1: Setting up all servers..."
echo ""

echo "Setting up validator ($VALIDATOR)..."
"$SCRIPT_DIR/remote.sh" validator $(parse_server "$VALIDATOR")

echo ""
echo "Setting up miner ($MINER)..."
"$SCRIPT_DIR/remote.sh" miner $(parse_server "$MINER")

echo ""
echo "Setting up executor ($EXECUTOR)..."
"$SCRIPT_DIR/remote.sh" executor $(parse_server "$EXECUTOR")

echo ""
echo "Step 2: Setting up SSH connectivity..."
"$SCRIPT_DIR/setup-ssh.sh" "$CONFIG_FILE"

echo ""
echo "=== All Setup Complete ==="
echo "✅ All three servers configured"
echo "✅ SSH keys distributed"
echo "✅ Cross-machine connectivity working"
echo "✅ Ready for Basilica development!"
echo ""
echo "Server details:"
echo "  Validator: $VALIDATOR"
echo "  Miner:     $MINER"
echo "  Executor:  $EXECUTOR"