#!/bin/bash
# Pre-flight checks for Basilica deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Track check results
FAILED_CHECKS=()
WARNINGS=()

# Check SSH connectivity
check_ssh_connectivity() {
    local host="$1"
    local port="$2"
    local user="$3"
    local service="$4"
    
    log_info "Checking SSH connectivity to $service ($host:$port)"
    
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -p "$port" "$user@$host" "echo 'SSH OK'" &>/dev/null; then
        log_success "SSH connectivity OK for $service"
        return 0
    else
        log_error "Cannot connect to $service via SSH"
        FAILED_CHECKS+=("SSH connection to $service failed")
        return 1
    fi
}

# Check required software on server
check_server_software() {
    local host="$1"
    local port="$2"
    local user="$3"
    local service="$4"
    
    log_info "Checking required software on $service"
    
    local check_script='
    MISSING=()
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        MISSING+=("docker")
    else
        if ! docker info >/dev/null 2>&1; then
            MISSING+=("docker-daemon-not-running")
        fi
    fi
    
    # Check systemctl
    if ! command -v systemctl >/dev/null 2>&1; then
        MISSING+=("systemctl")
    fi
    
    # Check basic tools
    for cmd in curl wget tar gzip; do
        if ! command -v $cmd >/dev/null 2>&1; then
            MISSING+=("$cmd")
        fi
    done
    
    if [ ${#MISSING[@]} -eq 0 ]; then
        echo "ALL_OK"
    else
        echo "MISSING: ${MISSING[*]}"
    fi
    '
    
    local result
    result=$(ssh -p "$port" "$user@$host" "$check_script" 2>/dev/null) || {
        log_error "Failed to check software on $service"
        FAILED_CHECKS+=("Software check failed on $service")
        return 1
    }
    
    if [[ "$result" == "ALL_OK" ]]; then
        log_success "All required software present on $service"
        return 0
    else
        log_error "$service: $result"
        FAILED_CHECKS+=("$service: $result")
        return 1
    fi
}

# Check disk space
check_disk_space() {
    local host="$1"
    local port="$2"
    local user="$3"
    local service="$4"
    local min_gb="${5:-10}"
    
    log_info "Checking disk space on $service"
    
    local available_gb
    available_gb=$(ssh -p "$port" "$user@$host" "df -BG /opt | tail -1 | awk '{print \$4}' | sed 's/G//'" 2>/dev/null) || {
        log_warn "Could not check disk space on $service"
        WARNINGS+=("Could not verify disk space on $service")
        return 0
    }
    
    if [[ "$available_gb" -lt "$min_gb" ]]; then
        log_error "$service has only ${available_gb}GB available (minimum: ${min_gb}GB)"
        FAILED_CHECKS+=("Insufficient disk space on $service")
        return 1
    else
        log_success "$service has ${available_gb}GB available"
        return 0
    fi
}

# Check network connectivity between servers
check_network_connectivity() {
    local from_host="$1"
    local from_port="$2"
    local from_user="$3"
    local to_host="$4"
    local to_port="$5"
    local from_name="$6"
    local to_name="$7"
    
    log_info "Checking network connectivity from $from_name to $to_name"
    
    local result
    result=$(ssh -p "$from_port" "$from_user@$from_host" "nc -zv -w 5 $to_host $to_port 2>&1" 2>/dev/null) || {
        log_warn "Cannot connect from $from_name to $to_name on port $to_port"
        WARNINGS+=("Network connectivity issue: $from_name -> $to_name:$to_port")
        return 0  # Warning only, not a failure
    }
    
    log_success "Network connectivity OK: $from_name -> $to_name"
    return 0
}

# Check if ports are available
check_port_availability() {
    local host="$1"
    local port="$2"
    local user="$3"
    local service="$4"
    local check_port="$5"
    
    log_info "Checking if port $check_port is available on $service"
    
    local in_use
    in_use=$(ssh -p "$port" "$user@$host" "ss -tlnp 2>/dev/null | grep -c ':$check_port ' || echo 0" 2>/dev/null) || {
        log_warn "Could not check port availability on $service"
        WARNINGS+=("Could not verify port $check_port on $service")
        return 0
    }
    
    if [[ "$in_use" -gt 0 ]]; then
        log_warn "Port $check_port is already in use on $service"
        WARNINGS+=("Port $check_port already in use on $service")
        return 0
    else
        log_success "Port $check_port is available on $service"
        return 0
    fi
}

# Check binaries exist
check_binaries() {
    log_info "Checking if binaries exist locally"
    
    local binaries=("executor" "miner" "validator" "gpu-attestor")
    local missing=()
    
    for binary in "${binaries[@]}"; do
        if [[ ! -f "$BASILICA_ROOT/$binary" ]]; then
            missing+=("$binary")
        fi
    done
    
    if [[ ${#missing[@]} -eq 0 ]]; then
        log_success "All binaries found"
        return 0
    else
        log_error "Missing binaries: ${missing[*]}"
        FAILED_CHECKS+=("Missing binaries: ${missing[*]}")
        return 1
    fi
}

# Check configurations exist
check_configurations() {
    log_info "Checking if configurations exist"
    
    local configs=("executor.toml" "miner.toml" "validator.toml")
    local missing=()
    
    for config in "${configs[@]}"; do
        if [[ ! -f "$BASILICA_ROOT/$config" ]]; then
            missing+=("$config")
        fi
    done
    
    if [[ ${#missing[@]} -eq 0 ]]; then
        log_success "All configurations found"
        return 0
    else
        log_error "Missing configurations: ${missing[*]}"
        FAILED_CHECKS+=("Missing configurations: ${missing[*]}")
        return 1
    fi
}

# Check GPU on executor
check_gpu_availability() {
    local host="$1"
    local port="$2"
    local user="$3"
    
    log_info "Checking GPU availability on executor"
    
    local gpu_info
    gpu_info=$(ssh -p "$port" "$user@$host" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NO_GPU'" 2>/dev/null) || {
        log_warn "Could not check GPU on executor"
        WARNINGS+=("Could not verify GPU availability")
        return 0
    }
    
    if [[ "$gpu_info" == "NO_GPU" ]]; then
        log_warn "No GPU detected on executor"
        WARNINGS+=("No GPU detected on executor - attestation may fail")
        return 0
    else
        log_success "GPU detected: $gpu_info"
        return 0
    fi
}

# Run all checks
run_all_checks() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    source "$env_file"
    
    log_header "Running pre-flight checks for $env environment"
    
    # Check binaries and configs
    check_binaries
    check_configurations
    
    # Check SSH connectivity
    check_ssh_connectivity "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "executor"
    check_ssh_connectivity "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "miner"
    check_ssh_connectivity "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "validator"
    
    # Check software
    check_server_software "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "executor"
    check_server_software "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "miner"
    check_server_software "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "validator"
    
    # Check disk space
    check_disk_space "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "executor" 20
    check_disk_space "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "miner" 10
    check_disk_space "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "validator" 10
    
    # Check port availability
    check_port_availability "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER" "executor" "${EXECUTOR_GRPC_PORT:-50051}"
    check_port_availability "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "miner" "${MINER_HTTP_PORT:-8092}"
    check_port_availability "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "validator" "${VALIDATOR_HTTP_PORT:-8080}"
    
    # Check GPU
    check_gpu_availability "$EXECUTOR_HOST" "$EXECUTOR_PORT" "$EXECUTOR_USER"
    
    # Check network connectivity
    check_network_connectivity "$MINER_HOST" "$MINER_PORT" "$MINER_USER" "$EXECUTOR_HOST" "${EXECUTOR_GRPC_PORT:-50051}" "miner" "executor"
    check_network_connectivity "$VALIDATOR_HOST" "$VALIDATOR_PORT" "$VALIDATOR_USER" "$EXECUTOR_HOST" 22 "validator" "executor"
    
    # Summary
    echo
    log_header "Pre-flight Check Summary"
    
    if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
        log_success "All critical checks passed!"
    else
        log_error "Failed checks:"
        for check in "${FAILED_CHECKS[@]}"; do
            echo "  ❌ $check"
        done
    fi
    
    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo
        log_warn "Warnings:"
        for warning in "${WARNINGS[@]}"; do
            echo "  ⚠️  $warning"
        done
    fi
    
    if [[ ${#FAILED_CHECKS[@]} -gt 0 ]]; then
        echo
        log_error "Pre-flight checks failed. Please fix the issues before deployment."
        return 1
    else
        echo
        log_success "System ready for deployment!"
        return 0
    fi
}

# Main function
main() {
    local command="${1:-run}"
    shift || true
    
    case "$command" in
        run|check)
            run_all_checks "$@"
            ;;
        help|*)
            cat << EOF
Basilica Pre-flight Check

Usage: preflight-check.sh [environment]

Checks:
    - Binary and configuration files exist
    - SSH connectivity to all servers
    - Required software installed
    - Sufficient disk space
    - Port availability
    - GPU availability on executor
    - Network connectivity between services
    
Environment:
    production (default)
    staging
    
Example:
    preflight-check.sh production
EOF
            ;;
    esac
}

main "$@"