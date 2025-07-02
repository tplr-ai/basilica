#!/bin/bash
# Basilica End-to-End Validation System
# Comprehensive validation of the complete Basilica infrastructure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASILICA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Validation results tracking
VALIDATION_LOG="$BASILICA_ROOT/validation_results.log"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

print_usage() {
    cat << EOF
validate.sh - Basilica End-to-End Infrastructure Validation

USAGE:
    validate.sh <COMMAND> [OPTIONS]

COMMANDS:
    all         Complete end-to-end validation workflow
    network     Network connectivity and discovery validation
    services    Service health and functionality validation
    workflow    End-to-end business workflow validation
    security    Security configuration validation
    performance Basic performance validation

OPTIONS:
    -e, --env ENV       Environment (production, staging, development)
    -v, --verbose       Verbose output with detailed test results
    -f, --fail-fast     Stop on first failure
    -r, --report        Generate detailed validation report
    -h, --help          Show this help message

EXAMPLES:
    validate.sh all --env production     # Complete validation
    validate.sh network                  # Network tests only
    validate.sh workflow --verbose       # Detailed workflow testing
    validate.sh services --fail-fast     # Stop on first service failure
    validate.sh all --report             # Generate validation report

VALIDATION COVERAGE:
    - Network connectivity between all services
    - Service health and API responsiveness
    - SSH validation workflow (validator → executor)
    - GPU attestation and verification process
    - Miner fleet management and discovery
    - End-to-end rental workflow simulation
    - Security configuration compliance
    - Performance baseline validation
EOF
}

# Default values
ENVIRONMENT="production"
VERBOSE=false
FAIL_FAST=false
GENERATE_REPORT=false

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--fail-fast)
                FAIL_FAST=true
                shift
                ;;
            -r|--report)
                GENERATE_REPORT=true
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

# Validation helper functions
validation_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$VALIDATION_LOG"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local test_description="$3"
    
    ((TOTAL_TESTS++))
    
    validation_log "TEST: $test_name - $test_description"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Running: $test_name"
        log_info "Command: $test_command"
    fi
    
    if eval "$test_command" >/dev/null 2>&1; then
        validation_log "PASS: $test_name"
        ((PASSED_TESTS++))
        if [[ "$VERBOSE" == "true" ]]; then
            log_success "$test_name passed"
        fi
        return 0
    else
        validation_log "FAIL: $test_name"
        ((FAILED_TESTS++))
        log_error "$test_name failed"
        
        if [[ "$FAIL_FAST" == "true" ]]; then
            log_error "Stopping validation due to --fail-fast option"
            exit 1
        fi
        return 1
    fi
}

# Command: Complete end-to-end validation
cmd_all() {
    log_header "Basilica Complete Infrastructure Validation"
    validation_log "Starting complete validation for environment: $ENVIRONMENT"
    
    load_env_config
    
    # Run all validation suites
    cmd_network
    cmd_services
    cmd_security
    cmd_workflow
    cmd_performance
    
    # Generate summary
    generate_validation_summary
    
    if [[ "$GENERATE_REPORT" == "true" ]]; then
        generate_validation_report
    fi
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log_success "All validation tests passed!"
        validation_log "VALIDATION COMPLETE: SUCCESS"
        return 0
    else
        log_error "Validation failed: $FAILED_TESTS/$TOTAL_TESTS tests failed"
        validation_log "VALIDATION COMPLETE: FAILED"
        return 1
    fi
}

# Command: Network validation
cmd_network() {
    log_header "Network Connectivity Validation"
    load_env_config
    
    # SSH connectivity tests
    run_test "ssh_validator_to_miner" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $MINER_HOST -p $MINER_PORT echo Success'" \
        "SSH connection from validator to miner"
    
    run_test "ssh_validator_to_executor" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT echo Success'" \
        "SSH connection from validator to executor"
    
    run_test "ssh_miner_to_executor" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT echo Success'" \
        "SSH connection from miner to executor"
    
    # Service endpoint connectivity
    run_test "validator_api_port" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'timeout 10 bash -c \"</dev/tcp/localhost/${VALIDATOR_API_PORT:-8080}\"'" \
        "Validator API port accessibility"
    
    run_test "miner_grpc_port" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'timeout 10 bash -c \"</dev/tcp/localhost/${MINER_GRPC_PORT:-8092}\"'" \
        "Miner gRPC port accessibility"
    
    run_test "executor_grpc_port" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'timeout 10 bash -c \"</dev/tcp/localhost/${EXECUTOR_GRPC_PORT:-50051}\"'" \
        "Executor gRPC port accessibility"
    
    # Inter-service connectivity
    run_test "validator_to_miner_grpc" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'timeout 10 bash -c \"</dev/tcp/$MINER_HOST/${MINER_GRPC_PORT:-8092}\"'" \
        "Validator to miner gRPC connectivity"
    
    run_test "validator_to_executor_grpc" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'timeout 10 bash -c \"</dev/tcp/$EXECUTOR_HOST/${EXECUTOR_GRPC_PORT:-50051}\"'" \
        "Validator to executor gRPC connectivity"
    
    run_test "miner_to_executor_grpc" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'timeout 10 bash -c \"</dev/tcp/$EXECUTOR_HOST/${EXECUTOR_GRPC_PORT:-50051}\"'" \
        "Miner to executor gRPC connectivity"
}

# Command: Service validation
cmd_services() {
    log_header "Service Health Validation"
    load_env_config
    
    # Service binary existence
    run_test "validator_binary_exists" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'test -f /usr/local/bin/validator && test -x /usr/local/bin/validator'" \
        "Validator binary exists and is executable"
    
    run_test "miner_binary_exists" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'test -f /usr/local/bin/miner && test -x /usr/local/bin/miner'" \
        "Miner binary exists and is executable"
    
    run_test "executor_binary_exists" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'test -f /usr/local/bin/executor && test -x /usr/local/bin/executor'" \
        "Executor binary exists and is executable"
    
    run_test "gpu_attestor_binary_exists" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'test -f /usr/local/bin/gpu-attestor && test -x /usr/local/bin/gpu-attestor'" \
        "GPU-attestor binary exists and is executable"
    
    # Configuration files
    run_test "validator_config_exists" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'test -f /etc/basilica/validator.toml'" \
        "Validator configuration file exists"
    
    run_test "miner_config_exists" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'test -f /etc/basilica/miner.toml'" \
        "Miner configuration file exists"
    
    run_test "executor_config_exists" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'test -f /etc/basilica/executor.toml'" \
        "Executor configuration file exists"
    
    # Systemd services
    run_test "validator_service_enabled" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'systemctl is-enabled basilica-validator'" \
        "Validator systemd service is enabled"
    
    run_test "miner_service_enabled" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'systemctl is-enabled basilica-miner'" \
        "Miner systemd service is enabled"
    
    run_test "executor_service_enabled" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'systemctl is-enabled basilica-executor'" \
        "Executor systemd service is enabled"
    
    # Service health (if running)
    run_test "validator_service_healthy" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'curl -sf http://localhost:${VALIDATOR_API_PORT:-8080}/health'" \
        "Validator service health endpoint responds"
    
    # Database connectivity
    run_test "validator_database_accessible" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'test -f /var/lib/basilica/validator/validator.db'" \
        "Validator database file exists"
    
    run_test "miner_database_accessible" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'test -f /var/lib/basilica/miner/miner.db'" \
        "Miner database file exists"
}

# Command: Security validation
cmd_security() {
    log_header "Security Configuration Validation"
    load_env_config
    
    # SSH key security
    run_test "ssh_keys_secure_permissions" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'stat -c \"%a\" ~/.ssh/basilica | grep -q \"^600$\"'" \
        "SSH private keys have secure permissions (600)"
    
    run_test "ssh_directory_secure" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'stat -c \"%a\" ~/.ssh | grep -q \"^700$\"'" \
        "SSH directory has secure permissions (700)"
    
    # Configuration file security
    run_test "config_files_secure" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'find /etc/basilica -name \"*.toml\" -exec stat -c \"%a\" {} \\; | grep -q \"^640$\"'" \
        "Configuration files have secure permissions (640)"
    
    # Firewall configuration
    run_test "firewall_enabled" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ufw status | grep -q \"Status: active\"'" \
        "UFW firewall is enabled on validator"
    
    # Service user security
    run_test "basilica_user_exists" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'id basilica'" \
        "Basilica system user exists"
    
    run_test "basilica_user_no_shell" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'getent passwd basilica | grep -q \"/bin/false\"'" \
        "Basilica user has no shell access"
}

# Command: Workflow validation
cmd_workflow() {
    log_header "End-to-End Workflow Validation"
    load_env_config
    
    # GPU attestation workflow
    run_test "gpu_attestor_can_run" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST '/usr/local/bin/gpu-attestor --executor-id test-validation --output /tmp/validation-test --help'" \
        "GPU-attestor can execute and show help"
    
    # Test basic attestation (if GPU available)
    run_test "gpu_attestation_basic" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'timeout 60 /usr/local/bin/gpu-attestor --executor-id validation-$(date +%s) --output /tmp/validation-attestation --skip-pow --skip-network-benchmark'" \
        "Basic GPU attestation can complete"
    
    # SSH validation workflow simulation
    run_test "ssh_validation_workflow" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT \"echo Validation workflow success\"'" \
        "SSH validation workflow can execute"
    
    # Service discovery validation
    run_test "service_discovery_config" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'test -f /etc/basilica/basilica_discovery.conf'" \
        "Service discovery configuration is deployed"
    
    # Network topology validation
    run_test "network_topology_validator_miner" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ping -c 1 -W 5 $MINER_HOST'" \
        "Validator can ping miner host"
    
    run_test "network_topology_validator_executor" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ping -c 1 -W 5 $EXECUTOR_HOST'" \
        "Validator can ping executor host"
    
    run_test "network_topology_miner_executor" \
        "ssh -p $MINER_PORT $MINER_USER@$MINER_HOST 'ping -c 1 -W 5 $EXECUTOR_HOST'" \
        "Miner can ping executor host"
}

# Command: Performance validation
cmd_performance() {
    log_header "Performance Validation"
    load_env_config
    
    # Basic performance tests
    run_test "validator_response_time" \
        "ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'timeout 10 time curl -sf http://localhost:${VALIDATOR_API_PORT:-8080}/health'" \
        "Validator API responds within timeout"
    
    run_test "ssh_connection_speed" \
        "timeout 30 ssh -p $VALIDATOR_PORT $VALIDATOR_USER@$VALIDATOR_HOST 'ssh -o ConnectTimeout=5 -i ~/.ssh/basilica $EXECUTOR_HOST -p $EXECUTOR_PORT hostname'" \
        "SSH connections complete within reasonable time"
    
    # System resource checks
    run_test "executor_gpu_available" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'nvidia-smi --query-gpu=name --format=csv,noheader,nounits'" \
        "GPU is available on executor"
    
    run_test "executor_docker_functional" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'docker info'" \
        "Docker is functional on executor"
    
    run_test "executor_disk_space" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'df / | awk \"NR==2 {if(\\$4 > 10000000) exit 0; else exit 1}\"'" \
        "Executor has sufficient disk space (>10GB)"
    
    # Memory and CPU checks
    run_test "system_memory_adequate" \
        "ssh -p $EXECUTOR_PORT $EXECUTOR_USER@$EXECUTOR_HOST 'free -m | awk \"NR==2 {if(\\$2 > 4000) exit 0; else exit 1}\"'" \
        "System has adequate memory (>4GB)"
}

# Generate validation summary
generate_validation_summary() {
    log_header "Validation Summary"
    
    echo ""
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""
    
    validation_log "SUMMARY: $PASSED_TESTS/$TOTAL_TESTS tests passed ($(( PASSED_TESTS * 100 / TOTAL_TESTS ))%)"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo "Failed tests:"
        grep "^FAIL:" "$VALIDATION_LOG" | sed 's/^.*FAIL: /- /'
    fi
}

# Generate detailed validation report
generate_validation_report() {
    local report_file="$BASILICA_ROOT/validation_report_$(date +%Y%m%d_%H%M%S).md"
    
    log_info "Generating detailed validation report: $report_file"
    
    cat > "$report_file" << EOF
# Basilica Infrastructure Validation Report

**Date**: $(date)
**Environment**: $ENVIRONMENT
**Total Tests**: $TOTAL_TESTS
**Passed Tests**: $PASSED_TESTS
**Failed Tests**: $FAILED_TESTS
**Success Rate**: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

## Environment Configuration

- **Validator**: $VALIDATOR_USER@$VALIDATOR_HOST:$VALIDATOR_PORT
- **Miner**: $MINER_USER@$MINER_HOST:$MINER_PORT
- **Executor**: $EXECUTOR_USER@$EXECUTOR_HOST:$EXECUTOR_PORT

## Test Results

### Passed Tests
EOF
    
    grep "^PASS:" "$VALIDATION_LOG" | sed 's/^.*PASS: /- /' >> "$report_file"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        cat >> "$report_file" << EOF

### Failed Tests
EOF
        grep "^FAIL:" "$VALIDATION_LOG" | sed 's/^.*FAIL: /- /' >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Recommendations

EOF
    
    # Add recommendations based on failures
    if [[ $FAILED_TESTS -gt 0 ]]; then
        cat >> "$report_file" << EOF
Based on the failed tests, consider the following actions:

1. **Network Issues**: Check firewall configurations and network connectivity
2. **Service Issues**: Verify service configurations and restart if necessary
3. **SSH Issues**: Regenerate and redistribute SSH keys if needed
4. **Performance Issues**: Check system resources and optimize configurations

For detailed troubleshooting, review the validation log: $VALIDATION_LOG
EOF
    else
        cat >> "$report_file" << EOF
All validation tests passed successfully. The Basilica infrastructure is properly configured and operational.

**Next Steps**:
1. Begin production operations
2. Set up monitoring and alerting
3. Schedule regular validation runs
4. Implement backup and disaster recovery procedures
EOF
    fi
    
    log_success "Validation report generated: $report_file"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    parse_args "$@"
    
    # Initialize validation log
    echo "=== Basilica Validation Started at $(date) ===" > "$VALIDATION_LOG"
    
    case "$command" in
        all)
            cmd_all
            ;;
        network)
            load_env_config
            cmd_network
            generate_validation_summary
            ;;
        services)
            load_env_config
            cmd_services
            generate_validation_summary
            ;;
        security)
            load_env_config
            cmd_security
            generate_validation_summary
            ;;
        workflow)
            load_env_config
            cmd_workflow
            generate_validation_summary
            ;;
        performance)
            load_env_config
            cmd_performance
            generate_validation_summary
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