#!/bin/bash
# Basilica Network Connectivity and Discovery Setup
# Manages SSH connectivity, service discovery, and network topology

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASILICA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

print_usage() {
    cat << EOF
network.sh - Basilica Network Connectivity and Discovery Management

USAGE:
    network.sh <COMMAND> [OPTIONS]

COMMANDS:
    setup       Complete network setup (SSH + discovery)
    ssh         Setup SSH connectivity between all machines
    discovery   Configure service discovery endpoints
    firewall    Configure firewall rules for Basilica services
    test        Test network connectivity and service discovery
    keys        Distribute SSH keys between machines

OPTIONS:
    -e, --env ENV       Environment (production, staging, development)
    -n, --dry-run       Show what would be done without executing
    -v, --verbose       Verbose output
    -h, --help          Show this help message

EXAMPLES:
    network.sh setup --env production    # Complete network setup
    network.sh ssh                       # Setup SSH connectivity only
    network.sh test                      # Test all connections
    network.sh firewall                  # Configure firewall rules
    network.sh discovery                 # Configure service discovery

NETWORK FEATURES:
    - Automated SSH key distribution and management
    - Service discovery endpoint configuration
    - Firewall rule automation for Basilica ports
    - Network connectivity validation
    - Cross-machine authentication setup
    - Security hardening for production
EOF
}

# Default values
ENVIRONMENT="production"
DRY_RUN=false
VERBOSE=false

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
}

# Load environment configuration
load_env_config() {
    local env_file="$SCRIPT_DIR/environments/${ENVIRONMENT}.conf"
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment config not found: $env_file"
        exit 1
    fi
    source "$env_file"
}

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would execute: $description"
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Command: $cmd"
        fi
    else
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Executing: $description"
            log_info "Command: $cmd"
        fi
        eval "$cmd"
    fi
}

# Execute SSH command on a server
ssh_exec() {
    local host="$1"
    local port="$2"
    local user="$3"
    local command="$4"
    local description="${5:-Executing SSH command}"
    
    local ssh_cmd="ssh -p $port $user@$host '$command'"
    execute_cmd "$ssh_cmd" "$description on $user@$host:$port"
}

# Command: Complete network setup
cmd_setup() {
    log_header "Basilica Network Setup"
    load_env_config
    
    # Execute all network setup steps
    cmd_ssh
    cmd_discovery
    cmd_firewall
    cmd_test
    
    log_success "Network setup completed successfully"
}

# Command: Setup SSH connectivity
cmd_ssh() {
    log_header "Setting Up SSH Connectivity"
    load_env_config
    
    # Use existing setup-ssh.sh script but with our config format
    local temp_config=$(mktemp)
    cat > "$temp_config" << EOF
VALIDATOR=${VALIDATOR_USER}@${VALIDATOR_HOST}:${VALIDATOR_PORT}
MINER=${MINER_USER}@${MINER_HOST}:${MINER_PORT}
EXECUTOR=${EXECUTOR_USER}@${EXECUTOR_HOST}:${EXECUTOR_PORT}
EOF
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would setup SSH connectivity using config:"
        cat "$temp_config"
        rm -f "$temp_config"
        return
    fi
    
    # Use existing SSH setup script
    local ssh_script="$SCRIPT_DIR/../setup-ops/setup-ssh.sh"
    if [[ -f "$ssh_script" ]]; then
        log_info "Using existing SSH setup script"
        "$ssh_script" "$temp_config"
    else
        log_warning "SSH setup script not found, implementing basic SSH setup"
        setup_ssh_basic "$temp_config"
    fi
    
    rm -f "$temp_config"
    log_success "SSH connectivity setup completed"
}

# Basic SSH setup if setup-ops script not available
setup_ssh_basic() {
    local config_file="$1"
    source "$config_file"
    
    # Extract connection details
    local validator_ssh="${VALIDATOR%:*} -p ${VALIDATOR##*:}"
    local miner_ssh="${MINER%:*} -p ${MINER##*:}"
    local executor_ssh="${EXECUTOR%:*} -p ${EXECUTOR##*:}"
    
    log_info "Setting up SSH keys for cross-machine access"
    
    # Generate SSH keys on each machine if they don't exist
    for server_ssh in "$validator_ssh" "$miner_ssh" "$executor_ssh"; do
        ssh $server_ssh "
            mkdir -p ~/.ssh
            chmod 700 ~/.ssh
            if [[ ! -f ~/.ssh/basilica ]]; then
                ssh-keygen -t ed25519 -f ~/.ssh/basilica -N ''
                echo 'Generated SSH key for Basilica'
            fi
        "
    done
    
    # Collect and distribute public keys
    log_info "Distributing SSH keys between machines"
    
    # Get all public keys
    local validator_key=$(ssh $validator_ssh "cat ~/.ssh/basilica.pub 2>/dev/null || echo 'Key not found'")
    local miner_key=$(ssh $miner_ssh "cat ~/.ssh/basilica.pub 2>/dev/null || echo 'Key not found'")
    local executor_key=$(ssh $executor_ssh "cat ~/.ssh/basilica.pub 2>/dev/null || echo 'Key not found'")
    
    # Distribute keys to all machines
    for server_ssh in "$validator_ssh" "$miner_ssh" "$executor_ssh"; do
        ssh $server_ssh "
            mkdir -p ~/.ssh
            chmod 700 ~/.ssh
            touch ~/.ssh/authorized_keys
            chmod 600 ~/.ssh/authorized_keys
            
            # Add all keys (idempotent)
            echo '$validator_key' | grep -v 'Key not found' | while read key; do
                grep -qF \"\$key\" ~/.ssh/authorized_keys || echo \"\$key\" >> ~/.ssh/authorized_keys
            done
            echo '$miner_key' | grep -v 'Key not found' | while read key; do
                grep -qF \"\$key\" ~/.ssh/authorized_keys || echo \"\$key\" >> ~/.ssh/authorized_keys
            done
            echo '$executor_key' | grep -v 'Key not found' | while read key; do
                grep -qF \"\$key\" ~/.ssh/authorized_keys || echo \"\$key\" >> ~/.ssh/authorized_keys
            done
        "
    done
}

# Command: Configure service discovery
cmd_discovery() {
    log_header "Configuring Service Discovery"
    load_env_config
    
    # Create service discovery configuration
    local discovery_config="/tmp/basilica_discovery.conf"
    
    cat > "$discovery_config" << EOF
# Basilica Service Discovery Configuration
# Generated on $(date) for environment: $ENVIRONMENT

[services]
validator_api = "${VALIDATOR_HOST}:${VALIDATOR_API_PORT:-8080}"
validator_metrics = "${VALIDATOR_HOST}:${VALIDATOR_METRICS_PORT:-9090}"
miner_grpc = "${MINER_HOST}:${MINER_GRPC_PORT:-8092}"
miner_axon = "${MINER_HOST}:${MINER_AXON_PORT:-8091}"
miner_metrics = "${MINER_HOST}:${MINER_METRICS_PORT:-9090}"
executor_grpc = "${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}"
executor_metrics = "${EXECUTOR_HOST}:${EXECUTOR_METRICS_PORT:-9090}"

[networks]
bittensor_network = "${BITTENSOR_NETWORK:-finney}"
bittensor_netuid = ${BITTENSOR_NETUID:-39}
bittensor_endpoint = "${BITTENSOR_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"

[topology]
validator_to_miner = "${VALIDATOR_HOST} -> ${MINER_HOST}:${MINER_GRPC_PORT:-8092}"
validator_to_executor = "${VALIDATOR_HOST} -> ${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}"
miner_to_executor = "${MINER_HOST} -> ${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}"
EOF
    
    # Deploy discovery configuration to all servers
    for server_info in \
        "${VALIDATOR_HOST}:${VALIDATOR_PORT}:${VALIDATOR_USER}" \
        "${MINER_HOST}:${MINER_PORT}:${MINER_USER}" \
        "${EXECUTOR_HOST}:${EXECUTOR_PORT}:${EXECUTOR_USER}"
    do
        IFS=':' read -r host port user <<< "$server_info"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would deploy discovery config to $user@$host:$port"
            continue
        fi
        
        scp -P "$port" "$discovery_config" "$user@$host:/tmp/"
        ssh_exec "$host" "$port" "$user" \
            "sudo mv /tmp/basilica_discovery.conf /etc/basilica/ && sudo chown root:basilica /etc/basilica/basilica_discovery.conf && sudo chmod 644 /etc/basilica/basilica_discovery.conf" \
            "Deploying discovery configuration"
    done
    
    rm -f "$discovery_config"
    log_success "Service discovery configuration completed"
}

# Command: Configure firewall rules
cmd_firewall() {
    log_header "Configuring Firewall Rules"
    load_env_config
    
    # Configure validator firewall
    configure_firewall_for_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" \
        "8080:tcp" "9090:tcp" "22:tcp"
    
    # Configure miner firewall
    configure_firewall_for_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" \
        "8091:tcp" "8092:tcp" "9090:tcp" "22:tcp"
    
    # Configure executor firewall
    configure_firewall_for_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" \
        "50051:tcp" "9090:tcp" "22:tcp"
    
    log_success "Firewall configuration completed"
}

# Configure firewall for a specific service
configure_firewall_for_service() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    shift 4
    local ports=("$@")
    
    log_info "Configuring firewall for $service on $host"
    
    local firewall_script="
        # Enable UFW if not already enabled
        if ! ufw status | grep -q 'Status: active'; then
            ufw --force enable
        fi
        
        # Set default policies
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH from anywhere (be careful in production)
        ufw allow ssh
        
        # Allow Basilica service ports
    "
    
    # Add port rules
    for port_spec in "${ports[@]}"; do
        IFS=':' read -r port_num proto <<< "$port_spec"
        firewall_script+="\n        ufw allow $port_num/$proto comment 'Basilica $service'"
    done
    
    # Add inter-service communication rules
    case "$service" in
        validator)
            firewall_script+="\n        # Allow connections to miner and executor"
            firewall_script+="\n        ufw allow out to ${MINER_HOST} port ${MINER_GRPC_PORT:-8092}"
            firewall_script+="\n        ufw allow out to ${EXECUTOR_HOST} port ${EXECUTOR_GRPC_PORT:-50051}"
            ;;
        miner)
            firewall_script+="\n        # Allow connections from validator and to executor"
            firewall_script+="\n        ufw allow from ${VALIDATOR_HOST} to any port ${MINER_GRPC_PORT:-8092}"
            firewall_script+="\n        ufw allow out to ${EXECUTOR_HOST} port ${EXECUTOR_GRPC_PORT:-50051}"
            ;;
        executor)
            firewall_script+="\n        # Allow connections from validator and miner"
            firewall_script+="\n        ufw allow from ${VALIDATOR_HOST} to any port ${EXECUTOR_GRPC_PORT:-50051}"
            firewall_script+="\n        ufw allow from ${MINER_HOST} to any port ${EXECUTOR_GRPC_PORT:-50051}"
            ;;
    esac
    
    firewall_script+="\n        
        # Show status
        ufw status numbered
    "
    
    ssh_exec "$host" "$port" "$user" \
        "$firewall_script" \
        "Configuring firewall rules for $service"
}

# Command: Test network connectivity
cmd_test() {
    log_header "Testing Network Connectivity"
    load_env_config
    
    local errors=0
    
    # Test SSH connectivity
    log_info "Testing SSH connectivity..."
    if ! test_ssh_connectivity; then
        ((errors++))
    fi
    
    # Test service endpoints
    log_info "Testing service endpoints..."
    if ! test_service_endpoints; then
        ((errors++))
    fi
    
    # Test inter-service communication
    log_info "Testing inter-service communication..."
    if ! test_inter_service_communication; then
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "All network connectivity tests passed"
    else
        log_error "Network connectivity tests failed with $errors errors"
        return 1
    fi
}

# Test SSH connectivity between all machines
test_ssh_connectivity() {
    local errors=0
    
    # Extract connection details
    local validator_host="${VALIDATOR_HOST}"
    local miner_host="${MINER_HOST}"
    local executor_host="${EXECUTOR_HOST}"
    
    # Test from validator
    if ssh -p "$VALIDATOR_PORT" "$VALIDATOR_USER@$VALIDATOR_HOST" \
        "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $miner_host -p $MINER_PORT 'echo Success' && 
         ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $executor_host -p $EXECUTOR_PORT 'echo Success'" >/dev/null 2>&1; then
        log_success "Validator can connect to miner and executor"
    else
        log_error "Validator cannot connect to other services"
        ((errors++))
    fi
    
    # Test from miner
    if ssh -p "$MINER_PORT" "$MINER_USER@$MINER_HOST" \
        "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $validator_host -p $VALIDATOR_PORT 'echo Success' && 
         ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $executor_host -p $EXECUTOR_PORT 'echo Success'" >/dev/null 2>&1; then
        log_success "Miner can connect to validator and executor"
    else
        log_error "Miner cannot connect to other services"
        ((errors++))
    fi
    
    # Test from executor
    if ssh -p "$EXECUTOR_PORT" "$EXECUTOR_USER@$EXECUTOR_HOST" \
        "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $validator_host -p $VALIDATOR_PORT 'echo Success' && 
         ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $miner_host -p $MINER_PORT 'echo Success'" >/dev/null 2>&1; then
        log_success "Executor can connect to validator and miner"
    else
        log_error "Executor cannot connect to other services"
        ((errors++))
    fi
    
    return $errors
}

# Test service endpoints
test_service_endpoints() {
    local errors=0
    
    # Test validator API
    if ssh -p "$VALIDATOR_PORT" "$VALIDATOR_USER@$VALIDATOR_HOST" \
        "timeout 10 bash -c '</dev/tcp/localhost/${VALIDATOR_API_PORT:-8080}'" >/dev/null 2>&1; then
        log_success "Validator API port is accessible"
    else
        log_warning "Validator API port is not accessible (service may not be running)"
    fi
    
    # Test miner gRPC
    if ssh -p "$MINER_PORT" "$MINER_USER@$MINER_HOST" \
        "timeout 10 bash -c '</dev/tcp/localhost/${MINER_GRPC_PORT:-8092}'" >/dev/null 2>&1; then
        log_success "Miner gRPC port is accessible"
    else
        log_warning "Miner gRPC port is not accessible (service may not be running)"
    fi
    
    # Test executor gRPC
    if ssh -p "$EXECUTOR_PORT" "$EXECUTOR_USER@$EXECUTOR_HOST" \
        "timeout 10 bash -c '</dev/tcp/localhost/${EXECUTOR_GRPC_PORT:-50051}'" >/dev/null 2>&1; then
        log_success "Executor gRPC port is accessible"
    else
        log_warning "Executor gRPC port is not accessible (service may not be running)"
    fi
    
    return $errors
}

# Test inter-service communication
test_inter_service_communication() {
    local errors=0
    
    # Test validator to miner connection
    if ssh -p "$VALIDATOR_PORT" "$VALIDATOR_USER@$VALIDATOR_HOST" \
        "timeout 10 bash -c '</dev/tcp/${MINER_HOST}/${MINER_GRPC_PORT:-8092}'" >/dev/null 2>&1; then
        log_success "Validator can reach miner gRPC endpoint"
    else
        log_error "Validator cannot reach miner gRPC endpoint"
        ((errors++))
    fi
    
    # Test validator to executor connection
    if ssh -p "$VALIDATOR_PORT" "$VALIDATOR_USER@$VALIDATOR_HOST" \
        "timeout 10 bash -c '</dev/tcp/${EXECUTOR_HOST}/${EXECUTOR_GRPC_PORT:-50051}'" >/dev/null 2>&1; then
        log_success "Validator can reach executor gRPC endpoint"
    else
        log_error "Validator cannot reach executor gRPC endpoint"
        ((errors++))
    fi
    
    # Test miner to executor connection
    if ssh -p "$MINER_PORT" "$MINER_USER@$MINER_HOST" \
        "timeout 10 bash -c '</dev/tcp/${EXECUTOR_HOST}/${EXECUTOR_GRPC_PORT:-50051}'" >/dev/null 2>&1; then
        log_success "Miner can reach executor gRPC endpoint"
    else
        log_error "Miner cannot reach executor gRPC endpoint"
        ((errors++))
    fi
    
    return $errors
}

# Command: Distribute SSH keys
cmd_keys() {
    log_header "Distributing SSH Keys"
    load_env_config
    
    # This is essentially the same as cmd_ssh but focused on key management
    cmd_ssh
    
    log_success "SSH key distribution completed"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    parse_args "$@"
    
    case "$command" in
        setup)
            cmd_setup
            ;;
        ssh)
            cmd_ssh
            ;;
        discovery)
            cmd_discovery
            ;;
        firewall)
            cmd_firewall
            ;;
        test)
            cmd_test
            ;;
        keys)
            cmd_keys
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: $command"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"