#!/bin/bash
# Basilica Service Lifecycle Management
# Deploy, start, stop, and manage Basilica services across the infrastructure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASILICA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

print_usage() {
    cat << EOF
services.sh - Basilica Service Lifecycle Management

USAGE:
    services.sh <COMMAND> [SERVICE] [OPTIONS]

COMMANDS:
    deploy      Deploy services to servers
    start       Start services
    stop        Stop services
    restart     Restart services
    status      Check service status
    logs        View service logs
    enable      Enable services for auto-start
    disable     Disable services from auto-start

SERVICES:
    all         All services (default)
    validator   Validator service only
    miner       Miner service only
    executor    Executor service only

OPTIONS:
    -e, --env ENV       Environment (production, staging, development)
    -f, --follow        Follow logs (for logs command)
    -n, --lines NUM     Number of log lines to show (default: 50)
    -h, --help          Show this help message

EXAMPLES:
    services.sh deploy production        # Deploy all services to production
    services.sh start all                # Start all services
    services.sh status validator         # Check validator status
    services.sh logs executor --follow   # Follow executor logs
    services.sh restart miner            # Restart miner service
    services.sh stop all                 # Stop all services

SERVICE MANAGEMENT:
    - Builds and deploys binaries to target servers
    - Manages systemd services for each component
    - Handles service dependencies and startup order
    - Provides health monitoring and log access
    - Supports rolling deployments and updates
EOF
}

# Default values
ENVIRONMENT="production"
FOLLOW_LOGS=false
LOG_LINES=50

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--follow)
                FOLLOW_LOGS=true
                shift
                ;;
            -n|--lines)
                LOG_LINES="$2"
                shift 2
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

# Execute SSH command on a server
ssh_exec() {
    local host="$1"
    local port="$2"
    local user="$3"
    local command="$4"
    local description="${5:-Executing command}"
    
    log_info "$description on $user@$host:$port"
    ssh -p "$port" "$user@$host" "$command"
}

# Build binaries using existing build system
build_binaries() {
    log_header "Building Basilica Binaries"
    
    # Use existing production build system
    local production_script="$SCRIPT_DIR/../production/deploy.sh"
    if [[ -f "$production_script" ]]; then
        log_info "Using existing production build system"
        cd "$(dirname "$production_script")"
        ./deploy.sh build
    else
        # Fallback to direct cargo build
        log_info "Building with cargo"
        cd "$BASILICA_ROOT"
        cargo build --release --workspace
        
        # Create production build directory
        mkdir -p "$BASILICA_ROOT/build/production"
        cd "$BASILICA_ROOT/build/production"
        
        # Copy binaries
        for binary in validator miner executor gpu-attestor; do
            if [[ -f "$BASILICA_ROOT/target/release/$binary" ]]; then
                cp "$BASILICA_ROOT/target/release/$binary" .
                log_success "Copied $binary binary"
            else
                log_warning "Binary not found: $binary"
            fi
        done
    fi
}

# Deploy binary to a specific server
deploy_binary() {
    local host="$1"
    local port="$2"
    local user="$3"
    local binary="$4"
    local binary_path="$BASILICA_ROOT/build/production/$binary"
    
    if [[ ! -f "$binary_path" ]]; then
        log_error "Binary not found: $binary_path"
        return 1
    fi
    
    log_info "Deploying $binary to $user@$host:$port"
    
    # Create directories on remote server
    ssh_exec "$host" "$port" "$user" \
        "sudo mkdir -p /opt/basilica/bin /var/lib/basilica/$binary /etc/basilica" \
        "Creating directories"
    
    # Upload binary
    scp -P "$port" "$binary_path" "$user@$host:/tmp/$binary"
    
    # Install binary with proper permissions
    ssh_exec "$host" "$port" "$user" \
        "sudo mv /tmp/$binary /opt/basilica/bin/ && sudo chmod +x /opt/basilica/bin/$binary && sudo chown root:root /opt/basilica/bin/$binary" \
        "Installing binary"
    
    # Create symlink in /usr/local/bin
    ssh_exec "$host" "$port" "$user" \
        "sudo ln -sf /opt/basilica/bin/$binary /usr/local/bin/$binary" \
        "Creating symlink"
    
    log_success "Successfully deployed $binary"
}

# Command: Deploy services
cmd_deploy() {
    local environment="${1:-$ENVIRONMENT}"
    
    log_header "Deploying Basilica Services to $environment"
    load_env_config
    
    # Build binaries first
    build_binaries
    
    # Deploy validator
    log_info "Deploying validator service"
    deploy_binary "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "validator"
    deploy_binary "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "gpu-attestor"
    
    # Deploy miner
    log_info "Deploying miner service"
    deploy_binary "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "miner"
    
    # Deploy executor
    log_info "Deploying executor service"
    deploy_binary "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "executor"
    deploy_binary "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "gpu-attestor"
    
    # Setup basilica user and directories on all servers
    for server_info in \
        "$VALIDATOR_HOST:$VALIDATOR_PORT:$VALIDATOR_USER" \
        "$MINER_HOST:$MINER_PORT:$MINER_USER" \
        "$EXECUTOR_HOST:$EXECUTOR_PORT:$EXECUTOR_USER"
    do
        IFS=':' read -r host port user <<< "$server_info"
        
        ssh_exec "$host" "$port" "$user" \
            "sudo useradd --system --home /var/lib/basilica --shell /bin/false basilica 2>/dev/null || true" \
            "Creating basilica user"
        
        ssh_exec "$host" "$port" "$user" \
            "sudo chown -R basilica:basilica /var/lib/basilica" \
            "Setting directory ownership"
    done
    
    log_success "Service deployment completed"
}

# Command: Start services
cmd_start() {
    local service="${1:-all}"
    
    log_header "Starting Basilica Services"
    load_env_config
    
    case "$service" in
        all)
            # Start in dependency order: executor → miner → validator
            start_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            sleep 10
            start_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            sleep 15
            start_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        validator)
            start_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        miner)
            start_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            ;;
        executor)
            start_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
    
    log_success "Service start completed"
}

# Start a specific service
start_service() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    log_info "Starting $service service on $host"
    
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl start basilica-$service" \
        "Starting basilica-$service"
    
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl enable basilica-$service" \
        "Enabling basilica-$service"
    
    # Wait a moment and check status
    sleep 3
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl is-active basilica-$service" \
        "Checking $service status"
}

# Command: Stop services
cmd_stop() {
    local service="${1:-all}"
    
    log_header "Stopping Basilica Services"
    load_env_config
    
    case "$service" in
        all)
            # Stop in reverse dependency order: validator → miner → executor
            stop_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            stop_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            stop_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        validator)
            stop_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        miner)
            stop_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            ;;
        executor)
            stop_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
    
    log_success "Service stop completed"
}

# Stop a specific service
stop_service() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    log_info "Stopping $service service on $host"
    
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl stop basilica-$service" \
        "Stopping basilica-$service"
}

# Command: Restart services
cmd_restart() {
    local service="${1:-all}"
    
    log_header "Restarting Basilica Services"
    cmd_stop "$service"
    sleep 5
    cmd_start "$service"
}

# Command: Check service status
cmd_status() {
    local service="${1:-all}"
    
    log_header "Basilica Service Status"
    load_env_config
    
    case "$service" in
        all)
            check_service_status "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            check_service_status "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            check_service_status "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        validator)
            check_service_status "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        miner)
            check_service_status "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            ;;
        executor)
            check_service_status "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

# Check status of a specific service
check_service_status() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    echo ""
    log_info "Checking $service service on $host"
    
    # Check systemd service status
    if ssh -p "$port" "$user@$host" "sudo systemctl is-active basilica-$service" >/dev/null 2>&1; then
        log_success "$service service is running"
    else
        log_error "$service service is not running"
    fi
    
    # Check if binary exists
    if ssh -p "$port" "$user@$host" "test -f /usr/local/bin/$service" >/dev/null 2>&1; then
        log_success "$service binary is installed"
    else
        log_error "$service binary is missing"
    fi
    
    # Check service-specific endpoints
    case "$service" in
        validator)
            if ssh -p "$port" "$user@$host" "curl -sf http://localhost:8080/health" >/dev/null 2>&1; then
                log_success "Validator API is responding"
            else
                log_warning "Validator API is not responding"
            fi
            ;;
        executor)
            if ssh -p "$port" "$user@$host" "timeout 5 bash -c '</dev/tcp/localhost/50051'" >/dev/null 2>&1; then
                log_success "Executor gRPC is accepting connections"
            else
                log_warning "Executor gRPC is not accepting connections"
            fi
            ;;
        miner)
            if ssh -p "$port" "$user@$host" "timeout 5 bash -c '</dev/tcp/localhost/8092'" >/dev/null 2>&1; then
                log_success "Miner gRPC is accepting connections"
            else
                log_warning "Miner gRPC is not accepting connections"
            fi
            ;;
    esac
}

# Command: View service logs
cmd_logs() {
    local service="${1:-all}"
    
    load_env_config
    
    case "$service" in
        all)
            log_info "Use 'services.sh logs <service>' to view specific service logs"
            log_info "Available services: validator, miner, executor"
            ;;
        validator)
            view_service_logs "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            ;;
        miner)
            view_service_logs "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            ;;
        executor)
            view_service_logs "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

# View logs for a specific service
view_service_logs() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    log_info "Viewing $service logs on $host (last $LOG_LINES lines)"
    
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        ssh -p "$port" "$user@$host" "sudo journalctl -u basilica-$service -f"
    else
        ssh -p "$port" "$user@$host" "sudo journalctl -u basilica-$service -n $LOG_LINES --no-pager"
    fi
}

# Command: Enable services for auto-start
cmd_enable() {
    local service="${1:-all}"
    
    log_header "Enabling Basilica Services"
    load_env_config
    
    case "$service" in
        all)
            enable_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            enable_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            enable_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        validator|miner|executor)
            local host port user
            eval "host=\$${service^^}_HOST"
            eval "port=\$${service^^}_PORT"
            eval "user=\$${service^^}_USER"
            enable_service "$service" "$host" "$port" "$user"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

# Enable a specific service
enable_service() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl enable basilica-$service" \
        "Enabling $service service for auto-start"
}

# Command: Disable services from auto-start
cmd_disable() {
    local service="${1:-all}"
    
    log_header "Disabling Basilica Services"
    load_env_config
    
    case "$service" in
        all)
            disable_service "validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER"
            disable_service "miner" "$MINER_HOST" "$MINER_PORT" "$MINER_USER"
            disable_service "executor" "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
            ;;
        validator|miner|executor)
            local host port user
            eval "host=\$${service^^}_HOST"
            eval "port=\$${service^^}_PORT"
            eval "user=\$${service^^}_USER"
            disable_service "$service" "$host" "$port" "$user"
            ;;
        *)
            log_error "Unknown service: $service"
            exit 1
            ;;
    esac
}

# Disable a specific service
disable_service() {
    local service="$1"
    local host="$2"
    local port="$3"
    local user="$4"
    
    ssh_exec "$host" "$port" "$user" \
        "sudo systemctl disable basilica-$service" \
        "Disabling $service service from auto-start"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    parse_args "$@"
    
    case "$command" in
        deploy)
            cmd_deploy "$@"
            ;;
        start)
            cmd_start "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        enable)
            cmd_enable "$@"
            ;;
        disable)
            cmd_disable "$@"
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