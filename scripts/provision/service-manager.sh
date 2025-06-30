#!/bin/bash
# Service management for Basilica components

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

TEMPLATE_DIR="$SCRIPT_DIR/templates"

# Deploy systemd service file
deploy_service_file() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    local service_file="$TEMPLATE_DIR/basilica-${service_name}.service"
    
    if [[ ! -f "$service_file" ]]; then
        log_error "Service file not found: $service_file"
        return 1
    fi
    
    log_info "Deploying systemd service for $service_name to $server_host"
    
    # Copy service file
    scp -P "$server_port" "$service_file" "$server_user@$server_host:/tmp/basilica-${service_name}.service" || {
        log_error "Failed to copy service file"
        return 1
    }
    
    # Install service file
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo mv /tmp/basilica-${service_name}.service /etc/systemd/system/
        sudo systemctl daemon-reload
    " || {
        log_error "Failed to install service file"
        return 1
    }
    
    log_success "Service file deployed for $service_name"
}

# Start service
start_service() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    
    log_info "Starting $service_name on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo systemctl start basilica-${service_name}
        sleep 2
        sudo systemctl is-active basilica-${service_name}
    " || {
        log_error "Failed to start $service_name"
        return 1
    }
    
    log_success "$service_name started successfully"
}

# Stop service
stop_service() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    
    log_info "Stopping $service_name on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo systemctl stop basilica-${service_name} || true
    "
    
    log_success "$service_name stopped"
}

# Enable service
enable_service() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    
    log_info "Enabling $service_name on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo systemctl enable basilica-${service_name}
    " || {
        log_error "Failed to enable $service_name"
        return 1
    }
    
    log_success "$service_name enabled for auto-start"
}

# Get service status
service_status() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    
    log_info "Checking status of $service_name on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo systemctl status basilica-${service_name} --no-pager || true
    "
}

# Get service logs
service_logs() {
    local service_name="$1"
    local server_host="$2"
    local server_port="$3"
    local server_user="$4"
    local lines="${5:-50}"
    
    log_info "Getting logs for $service_name on $server_host"
    
    ssh -p "$server_port" "$server_user@$server_host" "
        sudo journalctl -u basilica-${service_name} -n $lines --no-pager
    "
}

# Deploy all service files
deploy_all_services() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    source "$env_file"
    
    log_header "Deploying service files for $env environment"
    
    deploy_service_file "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    deploy_service_file "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" || return 1
    deploy_service_file "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" || return 1
    
    log_success "All service files deployed"
}

# Start all services
start_all_services() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    source "$env_file"
    
    log_header "Starting all services in $env environment"
    
    # Start in dependency order
    start_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" || return 1
    sleep 5
    start_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" || return 1
    sleep 5
    start_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" || return 1
    
    log_success "All services started"
}

# Stop all services
stop_all_services() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    source "$env_file"
    
    log_header "Stopping all services in $env environment"
    
    stop_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
    stop_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
    stop_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
    
    log_success "All services stopped"
}

# Get status of all services
status_all_services() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    source "$env_file"
    
    log_header "Service Status for $env environment"
    
    echo "=== Executor Service ==="
    service_status "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
    echo
    echo "=== Miner Service ==="
    service_status "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
    echo
    echo "=== Validator Service ==="
    service_status "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        deploy)
            deploy_all_services "$@"
            ;;
        start)
            if [[ "$1" == "all" ]]; then
                shift
                start_all_services "$@"
            else
                # Start specific service
                local service="$1"
                local env="${2:-production}"
                source "$SCRIPT_DIR/environments/${env}.conf"
                
                case "$service" in
                    executor)
                        start_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
                        ;;
                    miner)
                        start_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
                        ;;
                    validator)
                        start_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
                        ;;
                    *)
                        log_error "Unknown service: $service"
                        exit 1
                        ;;
                esac
            fi
            ;;
        stop)
            if [[ "$1" == "all" ]]; then
                shift
                stop_all_services "$@"
            else
                # Stop specific service
                local service="$1"
                local env="${2:-production}"
                source "$SCRIPT_DIR/environments/${env}.conf"
                
                case "$service" in
                    executor)
                        stop_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
                        ;;
                    miner)
                        stop_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
                        ;;
                    validator)
                        stop_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
                        ;;
                    *)
                        log_error "Unknown service: $service"
                        exit 1
                        ;;
                esac
            fi
            ;;
        status)
            status_all_services "$@"
            ;;
        enable)
            local env="${1:-production}"
            source "$SCRIPT_DIR/environments/${env}.conf"
            
            enable_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            enable_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            enable_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        logs)
            local service="$1"
            local env="${2:-production}"
            local lines="${3:-50}"
            source "$SCRIPT_DIR/environments/${env}.conf"
            
            case "$service" in
                executor)
                    service_logs "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "$lines"
                    ;;
                miner)
                    service_logs "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "$lines"
                    ;;
                validator)
                    service_logs "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "$lines"
                    ;;
                *)
                    log_error "Unknown service: $service"
                    exit 1
                    ;;
            esac
            ;;
        help|*)
            cat << EOF
Basilica Service Manager

Usage: service-manager.sh <command> [options]

Commands:
    deploy               Deploy systemd service files
    start <service|all>  Start service(s)
    stop <service|all>   Stop service(s)
    status              Get status of all services
    enable              Enable services for auto-start
    logs <service>      View service logs
    
Services:
    executor, miner, validator
    
Examples:
    service-manager.sh deploy production
    service-manager.sh start all production
    service-manager.sh logs executor production 100
EOF
            ;;
    esac
}

main "$@"