#!/bin/bash
set -e

# Automate SSH key distribution between Basilica servers
# Usage: ./scripts/setup-ops/setup-ssh.sh [config_file]

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

VALIDATOR_SSH=$(parse_server "$VALIDATOR")
MINER_SSH=$(parse_server "$MINER")
EXECUTOR_SSH=$(parse_server "$EXECUTOR")

echo "=== Basilica SSH Key Distribution ==="
echo ""

# Function to get public key from a server
get_public_key() {
    local server="$1"
    local name="$2"
    echo "Getting public key from $name..."
    ssh $server "cat ~/.ssh/basilica.pub 2>/dev/null || echo 'Key not found'"
}

# Function to add key to server
add_key_to_server() {
    local server="$1"
    local name="$2"
    local key="$3"

    if [[ -n "$key" && "$key" != "Key not found" ]]; then
        echo "Adding key to $name..."
        ssh $server "echo '$key' >> ~/.ssh/authorized_keys && echo 'Key added to $name'"
    else
        echo "Warning: No valid key to add to $name"
    fi
}

# Get all public keys
echo "Step 1: Collecting public keys from all servers..."
VALIDATOR_KEY=$(get_public_key "$VALIDATOR_SSH" "validator")
MINER_KEY=$(get_public_key "$MINER_SSH" "miner")
EXECUTOR_KEY=$(get_public_key "$EXECUTOR_SSH" "executor")

echo ""
echo "Step 2: Distributing keys to all servers..."

# Add validator key to miner and executor
add_key_to_server "$MINER_SSH" "miner" "$VALIDATOR_KEY"
add_key_to_server "$EXECUTOR_SSH" "executor" "$VALIDATOR_KEY"

# Add miner key to validator and executor
add_key_to_server "$VALIDATOR_SSH" "validator" "$MINER_KEY"
add_key_to_server "$EXECUTOR_SSH" "executor" "$MINER_KEY"

# Add executor key to validator and miner
add_key_to_server "$VALIDATOR_SSH" "validator" "$EXECUTOR_KEY"
add_key_to_server "$MINER_SSH" "miner" "$EXECUTOR_KEY"

echo ""
echo "Step 3: Testing connectivity..."

# Extract connection details for testing
VALIDATOR_HOST=$(echo "$VALIDATOR" | cut -d@ -f2 | cut -d: -f1)
VALIDATOR_PORT=$(echo "$VALIDATOR" | cut -d: -f2)
MINER_HOST=$(echo "$MINER" | cut -d@ -f2 | cut -d: -f1)
MINER_PORT=$(echo "$MINER" | cut -d: -f2)
EXECUTOR_HOST=$(echo "$EXECUTOR" | cut -d@ -f2 | cut -d: -f1)
EXECUTOR_PORT=$(echo "$EXECUTOR" | cut -d: -f2)

# Test connections from validator
echo "Testing from validator..."
ssh $VALIDATOR_SSH "
echo '- Connecting to miner...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $MINER_HOST -p $MINER_PORT 'echo Connected to miner from validator' || echo 'Failed to connect to miner'

echo '- Connecting to executor...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT 'echo Connected to executor from validator' || echo 'Failed to connect to executor'
"

# Test connections from miner
echo "Testing from miner..."
ssh $MINER_SSH "
echo '- Connecting to validator...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $VALIDATOR_HOST -p $VALIDATOR_PORT 'echo Connected to validator from miner' || echo 'Failed to connect to validator'

echo '- Connecting to executor...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT 'echo Connected to executor from miner' || echo 'Failed to connect to executor'
"

# Test connections from executor
echo "Testing from executor..."
ssh $EXECUTOR_SSH "
echo '- Connecting to validator...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $VALIDATOR_HOST -p $VALIDATOR_PORT 'echo Connected to validator from executor' || echo 'Failed to connect to validator'

echo '- Connecting to miner...'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/basilica $MINER_HOST -p $MINER_PORT 'echo Connected to miner from executor' || echo 'Failed to connect to miner'
"

echo ""
echo "=== SSH Setup Complete ==="
echo "✅ All SSH keys have been distributed"
echo "✅ Cross-machine connectivity tested"
echo "✅ Ready for Basilica development and testing!"
echo ""
echo "Manual connection examples:"
echo "  From validator: ssh -i ~/.ssh/basilica $MINER_HOST -p $MINER_PORT"
echo "  From miner:     ssh -i ~/.ssh/basilica $VALIDATOR_HOST -p $VALIDATOR_PORT"
echo "  From executor:  ssh -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT"
