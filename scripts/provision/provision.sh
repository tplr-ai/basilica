#!/bin/bash
# Basilica End-to-End Provisioning Orchestrator
# Comprehensive solution for provisioning the entire Basilica architecture

set -e

# Get script directory and load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASILICA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Configuration
DEFAULT_CONFIG="$SCRIPT_DIR/environments/production.conf"
PROVISION_LOG="$BASILICA_ROOT/provision.log"

# Logging function with timestamps
provision_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROVISION_LOG"
}

print_usage() {
    cat << EOF
basilica provision - End-to-End Basilica Infrastructure Provisioning

USAGE:
    provision.sh <COMMAND> [OPTIONS]

COMMANDS:
    all         Complete end-to-end provisioning workflow
    servers     Setup servers with dependencies and SSH
    deploy      Build and deploy Basilica services
    configure   Generate service configurations
    start       Start all services in correct order
    validate    Validate complete infrastructure setup

OPTIONS:
    -c, --config FILE    Use custom configuration file
    -e, --env ENV        Environment (production, staging, development)
    -n, --dry-run        Show what would be done without executing
    -v, --verbose        Verbose output
    -h, --help           Show this help message

EXAMPLES:
    provision.sh all                         # Complete provisioning
    provision.sh all --env production        # Production environment
    provision.sh servers --dry-run           # Test server setup
    provision.sh deploy --config custom.conf # Custom configuration
    provision.sh validate                    # Test infrastructure

WORKFLOW:
    1. Setup servers with dependencies (Docker, Rust, NVIDIA)
    2. Configure SSH connectivity between all machines
    3. Build Basilica binaries and Docker images
    4. Generate service configurations for each role
    5. Deploy services to respective servers
    6. Start services in correct dependency order
    7. Validate end-to-end functionality (miner → executor → validator)

For detailed help: provision.sh <command> --help
EOF
}

# Parse command line arguments
parse_args() {
    CONFIG_FILE=""
    ENVIRONMENT="production"
    DRY_RUN=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
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
    
    # Set config file if not specified
    if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$SCRIPT_DIR/environments/${ENVIRONMENT}.conf"
    fi
    
    # Validate config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Create configuration file or use: provision.sh configure"
        exit 1
    fi
    
    # Load configuration
    source "$CONFIG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        provision_log "Using configuration: $CONFIG_FILE"
        provision_log "Environment: $ENVIRONMENT"
        provision_log "Dry run: $DRY_RUN"
    fi
}

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would execute: $description"
        log_info "Command: $cmd"
    else
        provision_log "Executing: $description"
        if [[ "$VERBOSE" == "true" ]]; then
            provision_log "Command: $cmd"
        fi
        eval "$cmd"
    fi
}

# Command: Complete end-to-end provisioning
cmd_all() {
    log_header "Basilica End-to-End Provisioning"
    provision_log "Starting complete provisioning workflow"
    
    # Execute all steps in correct order
    cmd_servers "$@"
    cmd_deploy "$@"
    cmd_configure "$@"
    cmd_start "$@"
    cmd_validate "$@"
    
    log_success "Complete provisioning finished successfully!"
    provision_log "Provisioning completed successfully"
    
    # Show final status
    echo ""
    log_header "Provisioning Summary"
    echo "Environment: $ENVIRONMENT"
    echo "Configuration: $CONFIG_FILE"
    echo "Services provisioned: validator, miner, executor"
    echo "Log file: $PROVISION_LOG"
    echo ""
    echo "Next steps:"
    echo "1. Run 'basilica manage status' to check service health"
    echo "2. Test end-to-end workflow with 'basilica provision validate'"
    echo "3. Monitor logs with 'basilica manage logs'"
}

# Command: Setup servers with dependencies
cmd_servers() {
    log_header "Setting Up Servers"
    provision_log "Starting server setup phase"
    
    # Use existing setup-ops scripts but with enhanced logging
    local setup_script="$SCRIPT_DIR/../setup-ops/setup-all.sh"
    if [[ ! -f "$setup_script" ]]; then
        log_error "Setup script not found: $setup_script"
        exit 1
    fi
    
    # Create temporary config file for setup-ops
    local temp_config=$(mktemp)
    cat > "$temp_config" << EOF
VALIDATOR=${VALIDATOR_USER}@${VALIDATOR_HOST}:${VALIDATOR_PORT}
MINER=${MINER_USER}@${MINER_HOST}:${MINER_PORT}
EXECUTOR=${EXECUTOR_USER}@${EXECUTOR_HOST}:${EXECUTOR_PORT}
EOF
    
    execute_cmd "$setup_script $temp_config" "Setup all servers with dependencies"
    rm -f "$temp_config"
    
    provision_log "Server setup completed"
}

# Command: Build and deploy services
cmd_deploy() {
    log_header "Building and Deploying Services"
    provision_log "Starting deployment phase"
    
    # Build binaries using existing build system
    execute_cmd "cd '$BASILICA_ROOT' && cargo build --release --workspace" "Build Basilica binaries"
    
    # Use network.sh to setup connectivity
    execute_cmd "$SCRIPT_DIR/network.sh setup" "Setup network connectivity"
    
    # Use services.sh to deploy
    execute_cmd "$SCRIPT_DIR/services.sh deploy $ENVIRONMENT" "Deploy services to servers"
    
    provision_log "Deployment completed"
}

# Command: Generate configurations
cmd_configure() {
    log_header "Generating Service Configurations"
    provision_log "Starting configuration generation"
    
    execute_cmd "$SCRIPT_DIR/configure.sh generate --env $ENVIRONMENT" "Generate service configurations"
    execute_cmd "$SCRIPT_DIR/configure.sh deploy --env $ENVIRONMENT" "Deploy configurations to servers"
    
    provision_log "Configuration completed"
}

# Command: Start services
cmd_start() {
    log_header "Starting Services"
    provision_log "Starting services in correct order"
    
    # Start services in dependency order: executor → miner → validator
    execute_cmd "$SCRIPT_DIR/services.sh start executor" "Start executor service"
    execute_cmd "sleep 10" "Wait for executor to initialize"
    
    execute_cmd "$SCRIPT_DIR/services.sh start miner" "Start miner service"
    execute_cmd "sleep 15" "Wait for miner to register with executor"
    
    execute_cmd "$SCRIPT_DIR/services.sh start validator" "Start validator service"
    execute_cmd "sleep 10" "Wait for validator to initialize"
    
    provision_log "All services started"
}

# Command: Validate infrastructure
cmd_validate() {
    log_header "Validating Infrastructure"
    provision_log "Starting infrastructure validation"
    
    execute_cmd "$SCRIPT_DIR/validate.sh all" "Run complete infrastructure validation"
    
    provision_log "Validation completed"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    # Parse arguments first
    parse_args "$@"
    
    case "$command" in
        all)
            cmd_all
            ;;
        servers|setup)
            cmd_servers
            ;;
        deploy)
            cmd_deploy
            ;;
        configure|config)
            cmd_configure
            ;;
        start)
            cmd_start
            ;;
        validate|test)
            cmd_validate
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

# Initialize logging
provision_log "Basilica provisioning started with args: $*"

main "$@"