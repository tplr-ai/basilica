#!/bin/bash
# Binary deployment module for Basilica

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# SSH wrapper not needed for dynamic discovery
# If ssh-wrapper.sh exists, source it for backward compatibility
if [[ -f "$SCRIPT_DIR/../lib/ssh-wrapper.sh" ]]; then
    source "$SCRIPT_DIR/../lib/ssh-wrapper.sh"
fi

# Default configuration
DEFAULT_BINARY_DIR="/opt/basilica/bin"
DEFAULT_CONFIG_DIR="/opt/basilica/config"
DEFAULT_DATA_DIR="/opt/basilica/data"
DEFAULT_LOG_DIR="/opt/basilica/logs"

# Load environment configuration
load_environment() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    source "$env_file"
    log_info "Loaded environment: $env"
}

# Deploy single binary to server
deploy_binary() {
    local binary_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    local binary_path="${5:-$BASILICA_ROOT/$binary_name}"
    
    if [[ ! -f "$binary_path" ]]; then
        log_error "Binary not found: $binary_path"
        return 1
    fi
    
    log_info "Deploying $binary_name to $server_host:$server_port"
    
    # Create binary directory on remote
    ssh -p "$server_port" "$server_user@$server_host" '
        if [ "$EUID" -eq 0 ]; then
            mkdir -p '"$DEFAULT_BINARY_DIR"'
        elif command -v sudo >/dev/null 2>&1; then
            sudo mkdir -p '"$DEFAULT_BINARY_DIR"' && sudo chown -R $USER:$USER '"$DEFAULT_BINARY_DIR"'
        else
            echo "Error: Need root privileges to create directory '"$DEFAULT_BINARY_DIR"'"
            exit 1
        fi
    ' || {
        log_error "Failed to create binary directory on $server_host"
        return 1
    }
    
    # Stop the service if running (to avoid "Text file busy" error)
    ssh -p "$server_port" "$server_user@$server_host" "pkill -f $binary_name || true" 2>/dev/null
    
    # Wait a moment for the process to stop
    sleep 2
    
    # Copy binary
    scp -P "$server_port" "$binary_path" "$server_user@$server_host:$DEFAULT_BINARY_DIR/$binary_name" || {
        log_error "Failed to copy $binary_name to $server_host"
        return 1
    }
    
    # Set executable permissions
    ssh -p "$server_port" "$server_user@$server_host" "chmod 755 $DEFAULT_BINARY_DIR/$binary_name" || {
        log_error "Failed to set permissions for $binary_name"
        return 1
    }
    
    log_success "Deployed $binary_name successfully"
}

# Deploy configuration file to server
deploy_config() {
    local config_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    local config_path="${5:-$BASILICA_ROOT/$config_name}"
    
    if [[ ! -f "$config_path" ]]; then
        log_error "Configuration not found: $config_path"
        return 1
    fi
    
    log_info "Deploying $config_name to $server_host:$server_port"
    
    # Create config directory on remote
    ssh -p "$server_port" "$server_user@$server_host" '
        if [ "$EUID" -eq 0 ]; then
            mkdir -p '"$DEFAULT_CONFIG_DIR"'
        elif command -v sudo >/dev/null 2>&1; then
            sudo mkdir -p '"$DEFAULT_CONFIG_DIR"' && sudo chown -R $USER:$USER '"$DEFAULT_CONFIG_DIR"'
        else
            echo "Error: Need root privileges to create directory '"$DEFAULT_CONFIG_DIR"'"
            exit 1
        fi
    ' || {
        log_error "Failed to create config directory on $server_host"
        return 1
    }
    
    # Copy configuration
    scp -P "$server_port" "$config_path" "$server_user@$server_host:$DEFAULT_CONFIG_DIR/$config_name" || {
        log_error "Failed to copy $config_name to $server_host"
        return 1
    }
    
    log_success "Deployed $config_name successfully"
}

# Create required directories on server
create_directories() {
    local server_host="$1"
    local server_port="$2"
    local server_user="$3"
    
    log_info "Creating directories on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" << 'EOF'
        set -e
        
        # Check if running as root or if sudo is available
        if [ "$EUID" -eq 0 ]; then
            # Running as root, no sudo needed
            SUDO_CMD=""
        elif command -v sudo >/dev/null 2>&1; then
            # Not root but sudo is available
            SUDO_CMD="sudo"
        else
            # Not root and no sudo available
            echo "Error: Need root privileges to create directories in /opt/basilica"
            exit 1
        fi
        
        # Create all required directories
        for dir in /opt/basilica/{bin,config,data,logs,ssh_keys}; do
            $SUDO_CMD mkdir -p "$dir"
            $SUDO_CMD chmod 755 "$dir"
        done
        
        # Set secure permissions for SSH keys directory
        $SUDO_CMD chmod 700 /opt/basilica/ssh_keys
        
        # Set ownership to current user if not root
        if [ "$EUID" -ne 0 ] && [ -n "$SUDO_CMD" ]; then
            $SUDO_CMD chown -R $USER:$USER /opt/basilica
        fi
        
        echo "Directories created successfully"
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Directories created on $server_host"
    else
        log_error "Failed to create directories on $server_host"
        return 1
    fi
}

# Deploy executor
deploy_executor() {
    local env="${1:-production}"
    load_environment "$env" || return 1
    
    log_header "Deploying Executor to $EXECUTOR_HOST"
    
    # Create directories
    create_directories "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    
    # Deploy binaries
    deploy_binary "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    deploy_binary "gpu-attestor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    
    # Deploy configuration
    deploy_config "executor.toml" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    
    log_success "Executor deployment complete"
}

# Deploy miner
deploy_miner() {
    local env="${1:-production}"
    load_environment "$env" || return 1
    
    log_header "Deploying Miner to $MINER_HOST"
    
    # Create directories
    create_directories "$MINER_HOST" "$MINER_PORT" "$MINER_USER" || return 1
    
    # Deploy binary
    deploy_binary "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" || return 1
    
    # Deploy configuration
    deploy_config "miner.toml" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" || return 1
    
    log_success "Miner deployment complete"
}

# Deploy validator
deploy_validator() {
    local env="${1:-production}"
    load_environment "$env" || return 1
    
    log_header "Deploying Validator to $VALIDATOR_HOST"
    
    # Create directories
    create_directories "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" || return 1
    
    # Deploy binary
    deploy_binary "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" || return 1
    
    # Deploy configuration
    deploy_config "validator.toml" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" || return 1
    
    log_success "Validator deployment complete"
}

# Deploy all services
deploy_all() {
    local env="${1:-production}"
    
    log_header "Deploying all services to $env environment"
    
    deploy_executor "$env" || return 1
    deploy_miner "$env" || return 1
    deploy_validator "$env" || return 1
    
    log_success "All services deployed successfully"
}

# Build binaries locally
build_binaries() {
    log_header "Building Basilica binaries"
    
    # Check if we're in the project root
    if [[ ! -f "$BASILICA_ROOT/Cargo.toml" ]]; then
        log_error "Not in Basilica project root. Please run from project root."
        return 1
    fi
    
    # Build with Docker for compatibility
    log_info "Building validator..."
    "$SCRIPT_DIR/../validator/build.sh" --release || {
        log_error "Failed to build validator"
        return 1
    }
    
    log_info "Building miner..."
    "$SCRIPT_DIR/../miner/build.sh" --release || {
        log_error "Failed to build miner"
        return 1
    }
    
    log_info "Building executor..."
    "$SCRIPT_DIR/../executor/build.sh" --release || {
        log_error "Failed to build executor"
        return 1
    }
    
    log_info "Building gpu-attestor..."
    "$SCRIPT_DIR/../gpu-attestor/build.sh" --release || {
        log_error "Failed to build gpu-attestor"
        return 1
    }
    
    log_success "All binaries built successfully"
}

# Main deployment function
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        binaries)
            local env="${1:-production}"
            deploy_all "$env"
            ;;
        executor)
            deploy_executor "$@"
            ;;
        miner)
            deploy_miner "$@"
            ;;
        validator)
            deploy_validator "$@"
            ;;
        build)
            build_binaries
            ;;
        all)
            local env="${1:-production}"
            build_binaries || exit 1
            deploy_all "$env" || exit 1
            ;;
        help|*)
            cat << EOF
Basilica Binary Deployment Tool

Usage: deploy.sh <command> [environment]

Commands:
    binaries    Deploy all binaries to servers
    executor    Deploy executor only
    miner       Deploy miner only
    validator   Deploy validator only
    build       Build all binaries locally
    all         Build and deploy everything
    
Environment:
    production  Production servers (default)
    staging     Staging servers
    
Examples:
    deploy.sh build                    # Build binaries
    deploy.sh binaries production      # Deploy to production
    deploy.sh all                      # Build and deploy everything
EOF
            ;;
    esac
}

main "$@"