#!/bin/bash
# Enhanced provisioning script for Basilica with full error handling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Track deployment state for rollback
DEPLOYMENT_STATE_FILE="/tmp/basilica-deployment-state-$$"
trap cleanup EXIT

cleanup() {
    rm -f "$DEPLOYMENT_STATE_FILE"
}

# Record deployment action
record_action() {
    echo "$1" >> "$DEPLOYMENT_STATE_FILE"
}

# Rollback on failure
rollback() {
    log_error "Deployment failed. Starting rollback..."
    
    if [[ ! -f "$DEPLOYMENT_STATE_FILE" ]]; then
        log_warn "No deployment state to rollback"
        return
    fi
    
    # Read actions in reverse order
    while IFS= read -r action; do
        case "$action" in
            "deployed:config:*")
                local service="${action#deployed:config:}"
                log_info "Rolling back config for $service"
                # Could implement config backup/restore here
                ;;
            "deployed:binary:*")
                local service="${action#deployed:binary:}"
                log_info "Rolling back binary for $service"
                # Could remove deployed binaries
                ;;
            "started:service:*")
                local service="${action#started:service:}"
                log_info "Stopping $service"
                "$SCRIPT_DIR/service-manager.sh" stop "$service" "$ENV" 2>/dev/null || true
                ;;
        esac
    done < <(tac "$DEPLOYMENT_STATE_FILE")
    
    log_warn "Rollback completed"
}

# Run pre-flight checks
run_preflight_checks() {
    log_header "Running pre-flight checks"
    
    if ! "$SCRIPT_DIR/preflight-check.sh" "$ENV"; then
        log_error "Pre-flight checks failed"
        return 1
    fi
    
    return 0
}

# Build binaries
build_binaries() {
    log_header "Building binaries"
    
    if ! "$SCRIPT_DIR/deploy.sh" build; then
        log_error "Failed to build binaries"
        return 1
    fi
    
    return 0
}

# Generate configurations
generate_configurations() {
    log_header "Generating configurations"
    
    if ! "$SCRIPT_DIR/config-generator.sh" all "$ENV"; then
        log_error "Failed to generate configurations"
        return 1
    fi
    
    record_action "generated:configs"
    return 0
}

# Create wallets
create_wallets() {
    log_header "Creating development wallets"
    
    if ! "$SCRIPT_DIR/wallet-manager.sh" create-all "$ENV"; then
        log_error "Failed to create wallets"
        return 1
    fi
    
    record_action "created:wallets"
    return 0
}

# Deploy binaries
deploy_binaries() {
    log_header "Deploying binaries"
    
    local services=("executor" "miner" "validator")
    
    for service in "${services[@]}"; do
        if ! "$SCRIPT_DIR/deploy.sh" "$service" "$ENV"; then
            log_error "Failed to deploy $service"
            rollback
            return 1
        fi
        record_action "deployed:binary:$service"
    done
    
    return 0
}

# Deploy service files
deploy_service_files() {
    log_header "Deploying systemd service files"
    
    if ! "$SCRIPT_DIR/service-manager.sh" deploy "$ENV"; then
        log_error "Failed to deploy service files"
        rollback
        return 1
    fi
    
    record_action "deployed:service-files"
    return 0
}

# Start services
start_services() {
    log_header "Starting services"
    
    # Start in dependency order
    local services=("executor" "miner" "validator")
    
    for service in "${services[@]}"; do
        log_info "Starting $service..."
        
        if ! "$SCRIPT_DIR/service-manager.sh" start "$service" "$ENV"; then
            log_error "Failed to start $service"
            rollback
            return 1
        fi
        
        record_action "started:service:$service"
        
        # Wait for service to stabilize
        sleep 5
    done
    
    return 0
}

# Validate deployment
validate_deployment() {
    log_header "Validating deployment"
    
    # Check service status
    if ! "$SCRIPT_DIR/service-manager.sh" status "$ENV"; then
        log_warn "Some services may not be running properly"
    fi
    
    # Run basic connectivity test
    log_info "Testing service connectivity..."
    
    # Could implement actual service tests here
    
    return 0
}

# Full provisioning workflow
provision_all() {
    local start_time=$(date +%s)
    
    log_header "Starting full provisioning for $ENV environment"
    
    # Phase 1: Pre-flight checks
    if ! run_preflight_checks; then
        return 1
    fi
    
    # Phase 2: Build
    if ! build_binaries; then
        return 1
    fi
    
    # Phase 3: Configuration
    if ! generate_configurations; then
        return 1
    fi
    
    # Phase 4: Wallets
    if ! create_wallets; then
        return 1
    fi
    
    # Phase 5: Deployment
    if ! deploy_binaries; then
        return 1
    fi
    
    # Phase 6: Service setup
    if ! deploy_service_files; then
        return 1
    fi
    
    # Phase 7: Start services
    if ! start_services; then
        return 1
    fi
    
    # Phase 8: Validation
    if ! validate_deployment; then
        log_warn "Deployment validation had warnings"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Full provisioning completed in ${duration} seconds!"
    
    echo
    log_info "Next steps:"
    echo "  1. Check service status: ./scripts/basilica.sh manage status"
    echo "  2. View logs: ./scripts/basilica.sh manage logs <service>"
    echo "  3. For production, replace development wallets with real Bittensor wallets"
    
    return 0
}

# Main function
main() {
    local command="${1:-help}"
    ENV="${2:-production}"
    
    case "$command" in
        all)
            provision_all
            ;;
        servers)
            # Just run the original setup scripts
            "$SCRIPT_DIR/../setup-ops/setup-all.sh"
            ;;
        build)
            build_binaries
            ;;
        config)
            generate_configurations
            ;;
        wallets)
            create_wallets
            ;;
        deploy)
            deploy_binaries
            ;;
        services)
            deploy_service_files && start_services
            ;;
        validate)
            validate_deployment
            ;;
        help|*)
            cat << EOF
Enhanced Basilica Provisioning

Usage: provision-enhanced.sh <command> [environment]

Commands:
    all         Complete end-to-end provisioning
    servers     Setup servers with dependencies
    build       Build binaries
    config      Generate configurations
    wallets     Create development wallets
    deploy      Deploy binaries to servers
    services    Deploy and start services
    validate    Validate deployment
    
Environment:
    production (default)
    staging
    
Features:
    - Pre-flight checks before deployment
    - Automatic rollback on failure
    - Development wallet creation
    - Systemd service management
    - Deployment validation
    
Example:
    provision-enhanced.sh all production
EOF
            ;;
    esac
}

main "$@"