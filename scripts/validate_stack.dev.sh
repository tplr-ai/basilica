#!/bin/bash
# validate_stack.dev.sh - Basilica Development Stack Validation Script
#
# PURPOSE:
#   Validates the health and discovery mechanisms of deployed Basilica components
#   including miner, validator, and executor nodes. Performs comprehensive checks
#   on process health, network discovery, and inter-component communication.
#
# USAGE:
#   ./validate_stack.dev.sh \
#       -m root@miner:port \
#       -v root@validator:port \
#       -e user@executor \
#       -n <netuid> \
#       -N <network_name>
#
# REQUIREMENTS:
#   - SSH access to all component hosts
#   - sqlite3 installed on miner host
#   - Basilica components deployed with standard paths (/opt/basilica)

set -euo pipefail

MINER_HOST=""
MINER_PORT=""
VALIDATOR_HOST=""
VALIDATOR_PORT=""
EXECUTOR_HOST=""
NETUID=""
NETWORK=""
VERBOSE=false
CHECK_INTERVAL=10
MAX_RETRIES=3

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Validates Basilica development stack health and discovery mechanisms.

OPTIONS:
    -m, --miner HOST:PORT        Miner SSH connection (e.g., root@10.0.1.10:22)
    -v, --validator HOST:PORT    Validator SSH connection (e.g., root@10.0.1.20:22)
    -e, --executor HOST          Executor SSH connection (e.g., user@10.0.1.30)
    -n, --netuid ID              Network UID (required)
    -N, --network NAME           Network name (required)
    -i, --interval SECONDS       Check interval in seconds (default: 10)
    -r, --retries COUNT          Max retry attempts (default: 3)
    -V, --verbose                Enable verbose output
    -h, --help                   Display this help message

EXAMPLE:
    $0 -m root@10.0.1.10:22 -v root@10.0.1.20:22 -e user@10.0.1.30 -n 387 -N test

EOF
    exit 1
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--miner)
                IFS=':' read -r MINER_HOST MINER_PORT <<< "$2"
                shift 2
                ;;
            -v|--validator)
                IFS=':' read -r VALIDATOR_HOST VALIDATOR_PORT <<< "$2"
                shift 2
                ;;
            -e|--executor)
                EXECUTOR_HOST="$2"
                shift 2
                ;;
            -n|--netuid)
                NETUID="$2"
                shift 2
                ;;
            -N|--network)
                NETWORK="$2"
                shift 2
                ;;
            -i|--interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            -r|--retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            -V|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
}

validate_args() {
    if [[ -z "$MINER_HOST" ]] || [[ -z "$MINER_PORT" ]]; then
        echo "Error: Miner connection details required"
        usage
    fi
    if [[ -z "$VALIDATOR_HOST" ]] || [[ -z "$VALIDATOR_PORT" ]]; then
        echo "Error: Validator connection details required"
        usage
    fi
    if [[ -z "$EXECUTOR_HOST" ]]; then
        echo "Error: Executor connection details required"
        usage
    fi
    if [[ -z "$NETUID" ]]; then
        echo "Error: Network UID required"
        usage
    fi
    if [[ -z "$NETWORK" ]]; then
        echo "Error: Network name required"
        usage
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG: $*"
    fi
}

log_success() {
    log "[OK] $*"
}

log_error() {
    log "[ERROR] $*"
}

log_warning() {
    log "[WARN] $*"
}

ssh_cmd() {
    local host="$1"
    local port="${2:-22}"
    local cmd="$3"
    local retries=0

    while [[ $retries -lt $MAX_RETRIES ]]; do
        if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$port" "$host" "$cmd" 2>/dev/null; then
            return 0
        fi
        retries=$((retries + 1))
        if [[ $retries -lt $MAX_RETRIES ]]; then
            log_verbose "SSH command failed, retry $retries/$MAX_RETRIES"
            sleep 2
        fi
    done
    return 1
}

check_component_status() {
    local name="$1"
    local host="$2"
    local port="$3"
    local process_name="$4"

    log "Checking $name status..."

    if ssh_cmd "$host" "$port" "ps aux | grep -v grep | grep -q '$process_name'"; then
        log_success "$name process is running"
        return 0
    else
        log_error "$name process is not running"
        return 1
    fi
}

check_miner() {
    log "=== Miner Health Check ==="

    if ! check_component_status "Miner" "$MINER_HOST" "$MINER_PORT" "basilica.*miner"; then
        return 1
    fi

    log_verbose "Checking miner logs..."
    local recent_logs=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "tail -20 /opt/basilica/logs/miner-testnet.log 2>/dev/null | grep -E '(ERROR|WARN|discovered|validators|executors)' || true")

    if [[ -n "$recent_logs" ]]; then
        log_verbose "Recent miner log entries:"
        echo "$recent_logs" | while IFS= read -r line; do
            log_verbose "  $line"
        done
    fi

    local validators_found=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "tail -100 /opt/basilica/logs/miner-testnet.log | grep -o 'Found [0-9]* validators' | tail -1 | grep -o '[0-9]*' || echo 0")
    if [[ "$validators_found" -gt 0 ]]; then
        log_success "Miner discovered $validators_found validator(s)"
    else
        log_warning "Miner has not discovered any validators"
    fi

    local executors_found=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "tail -100 /opt/basilica/logs/miner-testnet.log | grep -o 'Found [0-9]* available executors' | tail -1 | grep -o '[0-9]*' || echo 0")
    if [[ "$executors_found" -gt 0 ]]; then
        log_success "Miner has $executors_found available executor(s)"
    else
        log_warning "Miner has no available executors"
    fi

    local executor_health=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "sqlite3 /opt/basilica/data/miner.db 'SELECT executor_id, is_healthy FROM executor_health ORDER BY updated_at DESC LIMIT 1;' 2>/dev/null || echo 'DB_ERROR'")
    if [[ "$executor_health" != "DB_ERROR" ]] && [[ -n "$executor_health" ]]; then
        local executor_id=$(echo "$executor_health" | cut -d'|' -f1)
        local is_healthy=$(echo "$executor_health" | cut -d'|' -f2)
        if [[ "$is_healthy" == "1" ]]; then
            log_success "Executor $executor_id is healthy in database"
        else
            log_error "Executor $executor_id is unhealthy in database"
        fi
    else
        log_warning "Could not query executor health from database"
    fi

    local grpc_running=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "grep -c 'Validator communications server started successfully' /opt/basilica/logs/miner-testnet.log || echo 0")
    if [[ "$grpc_running" -gt 0 ]]; then
        log_success "Miner gRPC server is running"
    else
        log_error "Miner gRPC server status unknown"
    fi

    return 0
}

check_validator() {
    log "=== Validator Health Check ==="

    if ! check_component_status "Validator" "$VALIDATOR_HOST" "$VALIDATOR_PORT" "basilica.*validator"; then
        return 1
    fi

    log_verbose "Checking validator logs..."
    local log_size=$(ssh_cmd "$VALIDATOR_HOST" "$VALIDATOR_PORT" "wc -l < /opt/basilica/logs/validator-testnet.log 2>/dev/null || echo 0")
    log_verbose "Validator log has $log_size lines"

    local registered=$(ssh_cmd "$VALIDATOR_HOST" "$VALIDATOR_PORT" "grep -c 'registered with discovered UID' /opt/basilica/logs/validator-testnet.log 2>/dev/null || echo 0")
    if [[ "$registered" -gt 0 ]]; then
        local uid=$(ssh_cmd "$VALIDATOR_HOST" "$VALIDATOR_PORT" "grep 'registered with discovered UID' /opt/basilica/logs/validator-testnet.log | grep -o 'UID: [0-9]*' | grep -o '[0-9]*' | tail -1 || echo 'unknown'")
        log_success "Validator registered with UID: $uid"
    else
        log_warning "Validator registration status unknown"
    fi

    return 0
}

check_executor() {
    log "=== Executor Health Check ==="

    if ! check_component_status "Executor" "$EXECUTOR_HOST" "22" "basilica.*executor"; then
        return 1
    fi

    log_verbose "Checking executor health check logs..."
    local health_checks=$(ssh_cmd "$EXECUTOR_HOST" "22" "tail -50 /opt/basilica/logs/executor-testnet.log | grep -c 'Health check requested by: miner' || echo 0")
    if [[ "$health_checks" -gt 0 ]]; then
        log_success "Executor received $health_checks health check(s) from miner recently"

        local last_check=$(ssh_cmd "$EXECUTOR_HOST" "22" "tail -50 /opt/basilica/logs/executor-testnet.log | grep 'Health check requested by: miner' | tail -1 | grep -o '^[^Z]*' || echo 'unknown'")
        log_verbose "Last health check: $last_check"
    else
        log_warning "No recent health checks from miner"
    fi

    local containers=$(ssh_cmd "$EXECUTOR_HOST" "22" "tail -20 /opt/basilica/logs/executor-testnet.log | grep 'Health check:.*containers found' | tail -1 | grep -o '[0-9]* containers' || echo 'unknown'")
    if [[ "$containers" != "unknown" ]]; then
        log_verbose "Container status: $containers"
    fi

    return 0
}

check_discovery() {
    log "=== Discovery Chain Verification ==="

    log_verbose "Checking Bittensor network connectivity..."
    local metagraph_fetch=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "tail -100 /opt/basilica/logs/miner-testnet.log | grep -c 'Metagraph fetched successfully for netuid: $NETUID' || echo 0")
    if [[ "$metagraph_fetch" -gt 0 ]]; then
        log_success "Miner successfully fetching metagraph from netuid $NETUID"
    else
        log_error "Miner not fetching metagraph successfully"
    fi

    local assignments=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "tail -100 /opt/basilica/logs/miner-testnet.log | grep 'assignment for validator' | tail -1 || echo 'none'")
    if [[ "$assignments" != "none" ]]; then
        log_success "Executor assignments found: ${assignments#*: }"
    else
        log_warning "No executor assignments to validators found"
    fi

    local interactions=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "sqlite3 /opt/basilica/data/miner.db 'SELECT COUNT(*) FROM validator_interactions;' 2>/dev/null || echo 0")
    if [[ "$interactions" -gt 0 ]]; then
        log_success "Found $interactions validator interaction(s) in database"
    else
        log_warning "No validator interactions recorded yet"
    fi

    return 0
}

generate_report() {
    log "=== Health Report Summary ==="

    local overall_status="HEALTHY"
    local issues=()

    if ! ssh_cmd "$MINER_HOST" "$MINER_PORT" "ps aux | grep -v grep | grep -q 'basilica.*miner'"; then
        overall_status="DEGRADED"
        issues+=("Miner process not running")
    fi

    if ! ssh_cmd "$VALIDATOR_HOST" "$VALIDATOR_PORT" "ps aux | grep -v grep | grep -q 'basilica.*validator'"; then
        overall_status="DEGRADED"
        issues+=("Validator process not running")
    fi

    if ! ssh_cmd "$EXECUTOR_HOST" "22" "ps aux | grep -v grep | grep -q 'basilica.*executor'"; then
        overall_status="DEGRADED"
        issues+=("Executor process not running")
    fi

    local executor_healthy=$(ssh_cmd "$MINER_HOST" "$MINER_PORT" "sqlite3 /opt/basilica/data/miner.db 'SELECT is_healthy FROM executor_health LIMIT 1;' 2>/dev/null || echo 0")
    if [[ "$executor_healthy" != "1" ]]; then
        overall_status="DEGRADED"
        issues+=("Executor marked unhealthy in miner database")
    fi

    echo
    echo "Stack Status: $overall_status"
    echo "Network: $NETWORK (netuid: $NETUID)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S UTC' -u)"

    if [[ ${#issues[@]} -gt 0 ]]; then
        echo
        echo "Issues Found:"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
    fi

    echo
    log "Validation complete"
}

# Main execution
main() {
    parse_args "$@"
    validate_args

    log "Starting Basilica stack validation"
    log_verbose "Configuration:"
    log_verbose "  Miner: $MINER_HOST:$MINER_PORT"
    log_verbose "  Validator: $VALIDATOR_HOST:$VALIDATOR_PORT"
    log_verbose "  Executor: $EXECUTOR_HOST"
    log_verbose "  Network: $NETWORK (netuid: $NETUID)"

    # Run health checks
    check_miner
    echo
    check_validator
    echo
    check_executor
    echo
    check_discovery
    echo

    # Generate final report
    generate_report
}

# Execute main function
main "$@"
