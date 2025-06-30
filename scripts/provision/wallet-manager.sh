#!/bin/bash
# Wallet management for Bittensor integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Default wallet paths
DEFAULT_WALLET_PATH="/root/.bittensor/wallets"

# Create development wallet structure
create_dev_wallet() {
    local wallet_name="$1"
    local hotkey_name="${2:-default}"
    local wallet_path="${3:-$DEFAULT_WALLET_PATH}"
    
    log_info "Creating development wallet: $wallet_name"
    
    # Create wallet directory structure
    local wallet_dir="$wallet_path/$wallet_name"
    local hotkeys_dir="$wallet_dir/hotkeys"
    
    mkdir -p "$hotkeys_dir"
    
    # Generate development keypair
    local coldkey_path="$wallet_dir/coldkey"
    local hotkey_path="$hotkeys_dir/$hotkey_name"
    
    # Create development coldkey (32 bytes of deterministic data for development)
    echo -n "${wallet_name}-coldkey-dev-$(date +%s)" | sha256sum | head -c 64 | xxd -r -p > "$coldkey_path"
    chmod 600 "$coldkey_path"
    
    # Create development hotkey (32 bytes of deterministic data for development)
    echo -n "${wallet_name}-hotkey-${hotkey_name}-dev-$(date +%s)" | sha256sum | head -c 64 | xxd -r -p > "$hotkey_path"
    chmod 600 "$hotkey_path"
    
    # Create encrypted versions (for development, just copy with .encrypted suffix)
    cp "$coldkey_path" "${coldkey_path}.encrypted"
    cp "$hotkey_path" "${hotkey_path}.encrypted"
    
    # Create a hotkey info file for compatibility
    cat > "${hotkey_path}.info" << EOF
{
    "type": "development",
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "wallet": "$wallet_name",
    "hotkey": "$hotkey_name"
}
EOF
    
    log_success "Created development wallet at: $wallet_dir"
}

# Create wallet on remote server
create_remote_wallet() {
    local server_host="$1"
    local server_port="$2"
    local server_user="$3"
    local wallet_name="$4"
    local hotkey_name="${5:-default}"
    
    log_info "Creating wallet on $server_host for $wallet_name"
    
    # Create wallet creation script
    local script_content=$(cat << 'SCRIPT'
#!/bin/bash
set -e

WALLET_NAME="$1"
HOTKEY_NAME="$2"
WALLET_PATH="/root/.bittensor/wallets"

# Create directory structure
mkdir -p "$WALLET_PATH/$WALLET_NAME/hotkeys"

# Generate development keys
COLDKEY_PATH="$WALLET_PATH/$WALLET_NAME/coldkey"
HOTKEY_PATH="$WALLET_PATH/$WALLET_NAME/hotkeys/$HOTKEY_NAME"

# Create deterministic development keys
echo -n "${WALLET_NAME}-coldkey-dev-$(hostname)-$(date +%s)" | sha256sum | head -c 64 | xxd -r -p > "$COLDKEY_PATH"
echo -n "${WALLET_NAME}-hotkey-${HOTKEY_NAME}-dev-$(hostname)-$(date +%s)" | sha256sum | head -c 64 | xxd -r -p > "$HOTKEY_PATH"

# Set permissions
chmod 600 "$COLDKEY_PATH" "$HOTKEY_PATH"

# Create encrypted versions
cp "$COLDKEY_PATH" "${COLDKEY_PATH}.encrypted"
cp "$HOTKEY_PATH" "${HOTKEY_PATH}.encrypted"

# Create info file
cat > "${HOTKEY_PATH}.info" << EOF
{
    "type": "development",
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "wallet": "$WALLET_NAME",
    "hotkey": "$HOTKEY_NAME",
    "host": "$(hostname)"
}
EOF

echo "Wallet created successfully"
ls -la "$WALLET_PATH/$WALLET_NAME/"
SCRIPT
)
    
    # Execute on remote server
    ssh -p "$server_port" "$server_user@$server_host" "bash -s" -- "$wallet_name" "$hotkey_name" <<< "$script_content" || {
        log_error "Failed to create wallet on $server_host"
        return 1
    }
    
    log_success "Wallet created on $server_host"
}

# Verify wallet exists on server
verify_remote_wallet() {
    local server_host="$1"
    local server_port="$2"
    local server_user="$3"
    local wallet_name="$4"
    local hotkey_name="${5:-default}"
    
    log_info "Verifying wallet on $server_host"
    
    local check_result
    check_result=$(ssh -p "$server_port" "$server_user@$server_host" "
        if [[ -f '/root/.bittensor/wallets/$wallet_name/coldkey' ]] && \
           [[ -f '/root/.bittensor/wallets/$wallet_name/hotkeys/$hotkey_name' ]]; then
            echo 'EXISTS'
        else
            echo 'NOT_FOUND'
        fi
    ")
    
    if [[ "$check_result" == "EXISTS" ]]; then
        log_success "Wallet verified on $server_host"
        return 0
    else
        log_warn "Wallet not found on $server_host"
        return 1
    fi
}

# Create wallets for all services
create_all_wallets() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    source "$env_file"
    
    log_header "Creating wallets for $env environment"
    
    # Create miner wallet
    if ! verify_remote_wallet "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "${MINER_WALLET_NAME:-miner}" "${MINER_HOTKEY_NAME:-default}"; then
        create_remote_wallet "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "${MINER_WALLET_NAME:-miner}" "${MINER_HOTKEY_NAME:-default}" || {
            log_error "Failed to create miner wallet"
            return 1
        }
    else
        log_info "Miner wallet already exists"
    fi
    
    # Create validator wallet
    if ! verify_remote_wallet "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "${VALIDATOR_WALLET_NAME:-validator}" "${VALIDATOR_HOTKEY_NAME:-default}"; then
        create_remote_wallet "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "${VALIDATOR_WALLET_NAME:-validator}" "${VALIDATOR_HOTKEY_NAME:-default}" || {
            log_error "Failed to create validator wallet"
            return 1
        }
    else
        log_info "Validator wallet already exists"
    fi
    
    log_success "All wallets created successfully"
}

# Import existing wallet
import_wallet() {
    local server_host="$1"
    local server_port="$2"
    local server_user="$3"
    local wallet_name="$4"
    local coldkey_path="$5"
    local hotkey_path="$6"
    local hotkey_name="${7:-default}"
    
    if [[ ! -f "$coldkey_path" ]] || [[ ! -f "$hotkey_path" ]]; then
        log_error "Wallet files not found"
        return 1
    fi
    
    log_info "Importing wallet to $server_host"
    
    # Create remote directory
    ssh -p "$server_port" "$server_user@$server_host" "mkdir -p /root/.bittensor/wallets/$wallet_name/hotkeys"
    
    # Copy wallet files
    scp -P "$server_port" "$coldkey_path" "$server_user@$server_host:/root/.bittensor/wallets/$wallet_name/coldkey" || {
        log_error "Failed to copy coldkey"
        return 1
    }
    
    scp -P "$server_port" "$hotkey_path" "$server_user@$server_host:/root/.bittensor/wallets/$wallet_name/hotkeys/$hotkey_name" || {
        log_error "Failed to copy hotkey"
        return 1
    }
    
    # Set permissions
    ssh -p "$server_port" "$server_user@$server_host" "
        chmod 600 /root/.bittensor/wallets/$wallet_name/coldkey
        chmod 600 /root/.bittensor/wallets/$wallet_name/hotkeys/$hotkey_name
    "
    
    log_success "Wallet imported successfully"
}

# Get wallet address
get_wallet_address() {
    local wallet_path="$1"
    
    if [[ ! -f "$wallet_path" ]]; then
        log_error "Wallet file not found: $wallet_path"
        return 1
    fi
    
    # For development wallets, generate a mock SS58 address
    local key_data=$(xxd -p -c 32 "$wallet_path" | head -n 1)
    local mock_address="5$(echo -n "$key_data" | sha256sum | head -c 47)"
    
    echo "$mock_address"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        create)
            local wallet_name="${1:-miner}"
            local hotkey_name="${2:-default}"
            create_dev_wallet "$wallet_name" "$hotkey_name"
            ;;
        create-all)
            create_all_wallets "$@"
            ;;
        verify)
            local env="${1:-production}"
            source "$SCRIPT_DIR/environments/${env}.conf"
            verify_remote_wallet "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "${MINER_WALLET_NAME:-miner}"
            verify_remote_wallet "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "${VALIDATOR_WALLET_NAME:-validator}"
            ;;
        import)
            if [[ $# -lt 6 ]]; then
                echo "Usage: wallet-manager.sh import <host> <port> <user> <wallet_name> <coldkey_path> <hotkey_path>"
                exit 1
            fi
            import_wallet "$@"
            ;;
        help|*)
            cat << EOF
Basilica Wallet Manager

Usage: wallet-manager.sh <command> [options]

Commands:
    create <name> [hotkey]     Create local development wallet
    create-all [env]           Create wallets on all servers
    verify [env]               Verify wallets exist on servers
    import <host> <port> <user> <name> <coldkey> <hotkey>
                              Import existing wallet to server
    
Examples:
    wallet-manager.sh create miner
    wallet-manager.sh create-all production
    wallet-manager.sh verify production
    
Note: Creates development wallets for testing. For production,
      use btcli to create proper Bittensor wallets.
EOF
            ;;
    esac
}

main "$@"