#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load test configuration
if [ -f "test.conf" ]; then
    source test.conf
else
    log_error "test.conf not found. Copy test.conf.example to test.conf and configure it."
    exit 1
fi

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    command -v docker >/dev/null || { log_error "Docker not installed"; exit 1; }
    command -v docker compose >/dev/null || { log_error "Docker Compose not installed"; exit 1; }
    
    # Setup test environment
    if [ ! -d "data" ] || [ ! -d "keys" ]; then
        log_info "Setting up test environment..."
        ./setup-test-env.sh
    fi
    
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &>/dev/null; then
        GPU_AVAILABLE=true
    else
        log_warn "NVIDIA Docker runtime not available - GPU testing will be skipped"
        GPU_AVAILABLE=false
    fi
    
    
    log_success "Prerequisites check completed"
}

build_images() {
    log_info "Building Docker images..."
    
    docker compose -f docker-compose.build.yml build || { log_error "Failed to build images"; return 1; }
    
    for image in validator miner executor gpu-attestor; do
        docker image inspect "basilica/${image}:localtest" &>/dev/null || { log_error "Image basilica/${image}:localtest not found"; return 1; }
    done
    
    log_success "Images built successfully"
}

start_core_services() {
    log_info "Starting core services..."
    
    docker compose up -d validator miner executor
    sleep 45  # Extended wait time for all services
    
    log_success "Core services started"
}

test_service_health() {
    log_info "Testing service health..."
    
    curl -f -s http://${VALIDATOR_HOST}:${VALIDATOR_PORT}/health >/dev/null || { log_error "Validator health check failed"; docker compose logs ${VALIDATOR_SERVICE} | tail -10; return 1; }
    log_success "Validator health check passed"
    
    docker compose exec ${EXECUTOR_SERVICE} executor service health >/dev/null || { log_error "Executor health check failed"; docker compose logs ${EXECUTOR_SERVICE} | tail -10; return 1; }
    log_success "Executor health check passed"
    
    curl -f -s http://${MINER_HOST}:${MINER_METRICS_PORT}/metrics >/dev/null || { log_error "Miner health check failed"; docker compose logs ${MINER_SERVICE} | tail -10; return 1; }
    log_success "Miner health check passed"
    
    if docker compose exec ${VALIDATOR_SERVICE} test -r ${VALIDATOR_DB_PATH}; then
        log_success "SQLite database accessible"
    else
        log_warn "SQLite database not found"
    fi
}

test_api_endpoints() {
    log_info "Testing API endpoints..."
    
    if curl -f -s -H "Authorization: Bearer ${VALIDATOR_API_KEY}" http://${VALIDATOR_HOST}:${VALIDATOR_API_PORT}/capacity/available >/dev/null; then
        log_success "Validator API accessible"
    else
        log_warn "Validator API endpoint failed"
    fi
    
    if curl -f -s http://${VALIDATOR_HOST}:${VALIDATOR_METRICS_PORT}/metrics >/dev/null; then
        log_success "Validator metrics accessible"
    else
        log_warn "Validator metrics not accessible"
    fi
    
    if curl -f -s http://${EXECUTOR_HOST}:${EXECUTOR_METRICS_PORT}/metrics >/dev/null; then
        log_success "Executor metrics accessible"
    else
        log_warn "Executor metrics not accessible"
    fi
    
    if curl -f -s http://${MINER_HOST}:${MINER_METRICS_PORT}/metrics >/dev/null; then
        log_success "Miner metrics accessible"
    else
        log_warn "Miner metrics not accessible"
    fi
}

test_miner_functionality() {
    log_info "Testing miner-specific functionality..."
    
    # Test miner CLI commands
    if docker compose exec miner miner --config /etc/basilica/miner.toml executor list >/dev/null 2>&1; then
        log_success "Miner executor list command works"
    else
        log_warn "Miner executor list command failed"
    fi
    
    if docker compose exec miner miner --config /etc/basilica/miner.toml executor health >/dev/null 2>&1; then
        log_success "Miner executor health check works"
    else
        log_warn "Miner executor health check failed"
    fi
    
    if docker compose exec miner miner --config /etc/basilica/miner.toml database stats >/dev/null 2>&1; then
        log_success "Miner database accessible"
    else
        log_warn "Miner database connection failed"
        docker compose logs miner | tail -10
    fi
    
    # Test gRPC connectivity between miner and executor
    if docker compose exec ${MINER_SERVICE} test -f ${MINER_DB_PATH}; then
        log_success "Miner database file exists"
    else
        log_warn "Miner database file not found"
    fi
    
    # Check if miner can communicate with executor
    log_info "Testing miner-executor communication..."
    sleep 10  # Give services time to establish connections
    
    # Check miner logs for executor connectivity
    if docker compose logs miner 2>/dev/null | grep -q "executor.*healthy\|executor.*connected"; then
        log_success "Miner-executor communication established"
    else
        log_warn "Miner-executor communication not detected"
    fi
}

test_ssh_validation() {
    log_info "Testing SSH validation..."
    
    docker compose --profile ssh-test up -d ${SSH_TARGET_SERVICE}
    sleep 15
    
    if docker compose exec ${VALIDATOR_SERVICE} validator connect --host ${SSH_TARGET_SERVICE} --username ${SSH_TEST_USER} --private-key ${SSH_TEST_KEY} --timeout ${TEST_TIMEOUT_SECONDS} >/dev/null 2>&1; then
        log_success "SSH connection test passed"
    else
        log_error "SSH connection test failed"
        docker compose logs ${SSH_TARGET_SERVICE} | tail -10
        return 1
    fi
}

test_gpu_attestation() {
    if [ "${ENABLE_GPU_TESTS}" != "true" ] || [ "$GPU_AVAILABLE" != "true" ]; then
        log_warn "Skipping GPU attestation test - GPU tests disabled or no GPU available"
        return 0
    fi
    
    log_info "Testing GPU attestation..."
    
    if docker compose --profile attestor up gpu-attestor; then
        if docker compose exec ${EXECUTOR_SERVICE} test -f ${SHARED_ATTESTOR_PATH}/attestation.json; then
            log_success "Attestation files created"
        else
            log_warn "Attestation files not found"
        fi
    else
        log_error "GPU attestation failed"
        docker compose logs gpu-attestor | tail -10
        return 1
    fi
}

test_end_to_end() {
    log_info "Testing end-to-end workflow..."
    
    docker compose --profile ssh-test up -d
    
    if docker compose exec gpu-attestor test -f /usr/local/bin/gpu-attestor; then
        docker compose exec gpu-attestor cp /usr/local/bin/gpu-attestor ${SHARED_ATTESTOR_PATH}/ 2>/dev/null || log_warn "Could not copy gpu-attestor binary"
    fi
    
    if docker compose exec ${VALIDATOR_SERVICE} validator verify --host ${SSH_TARGET_SERVICE} --username ${SSH_TEST_USER} --private-key ${SSH_TEST_KEY} --gpu-attestor-path ${SHARED_ATTESTOR_PATH}/gpu-attestor --remote-work-dir ${TEST_WORK_DIR} --skip-cleanup --verbose >/tmp/verification.log 2>&1; then
        log_success "Hardware verification completed"
        
        if docker compose exec ${VALIDATOR_SERVICE} sqlite3 ${VALIDATOR_DB_PATH} "SELECT COUNT(*) FROM verification_logs;" 2>/dev/null | grep -q "[0-9]"; then
            log_success "Verification results in database"
        else
            log_warn "No verification results in database"
        fi
    else
        log_error "Hardware verification failed"
        cat /tmp/verification.log | tail -10
        return 1
    fi
}

start_monitoring() {
    log_info "Starting monitoring services..."
    
    if [ "${ENABLE_MONITORING_TESTS}" != "true" ]; then
        log_warn "Monitoring tests disabled"
        return 0
    fi
    
    docker compose --profile monitoring up -d prometheus grafana
    sleep 20
    
    if curl -f -s http://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}/api/v1/targets >/dev/null; then
        log_success "Prometheus accessible at http://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}"
    else
        log_warn "Prometheus not accessible"
    fi
    
    if curl -f -s http://${GRAFANA_HOST}:${GRAFANA_PORT}/api/health >/dev/null; then
        log_success "Grafana accessible at http://${GRAFANA_HOST}:${GRAFANA_PORT}"
    else
        log_warn "Grafana not accessible"
    fi
}

show_status() {
    log_info "System status:"
    
    echo "=== Docker Images ==="
    docker images | grep basilica || echo "No Basilica images found"
    
    echo -e "\n=== Service Status ==="
    docker compose ps
    
    echo -e "\n=== Endpoints ==="
    echo "Validator API: http://${VALIDATOR_HOST}:${VALIDATOR_API_PORT}"
    echo "Validator Metrics: http://${VALIDATOR_HOST}:${VALIDATOR_METRICS_PORT}/metrics"
    echo "Miner gRPC: ${MINER_HOST}:${MINER_GRPC_PORT}"
    echo "Miner Metrics: http://${MINER_HOST}:${MINER_METRICS_PORT}/metrics"
    echo "Executor gRPC: ${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT}"
    echo "Executor Metrics: http://${EXECUTOR_HOST}:${EXECUTOR_METRICS_PORT}/metrics"
    echo "Prometheus: http://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}"
    echo "Grafana: http://${GRAFANA_HOST}:${GRAFANA_PORT}"
    
    echo -e "\n=== Recent Validation Logs ==="
    docker compose exec ${VALIDATOR_SERVICE} sqlite3 ${VALIDATOR_DB_PATH} "SELECT executor_id, score, success, created_at FROM verification_logs ORDER BY created_at DESC LIMIT 5;" 2>/dev/null || echo "No validation data available"
}

test_individual_services() {
    log_info "Testing individual compose files..."
    
    cd ../validator
    docker compose config >/dev/null || { log_error "Validator compose file has errors"; return 1; }
    log_success "Validator compose valid"
    
    cd ../miner
    docker compose config >/dev/null || { log_error "Miner compose file has errors"; return 1; }
    log_success "Miner compose valid"
    
    cd ../executor
    docker compose config >/dev/null || { log_error "Executor compose file has errors"; return 1; }
    log_success "Executor compose valid"
    
    cd ../gpu-attestor
    docker compose config >/dev/null || { log_error "GPU-attestor compose file has errors"; return 1; }
    log_success "GPU-attestor compose valid"
    
    cd ../localtest
}

cleanup() {
    log_info "Cleaning up..."
    
    docker compose --profile ssh-test --profile attestor --profile monitoring down
    
    # Clean test data (optional, uncomment if needed)
    # rm -rf data/ keys/
    
    log_success "Cleanup completed"
}

test_load_performance() {
    log_info "Running load performance tests..."
    
    if [ "${ENABLE_LOAD_TESTS}" != "true" ]; then
        log_warn "Load tests disabled"
        return 0
    fi
    
    # Ensure services are up
    docker compose ps ${VALIDATOR_SERVICE} | grep -q "Up" || { log_error "Validator not running"; return 1; }
    docker compose ps ${MINER_SERVICE} | grep -q "Up" || { log_error "Miner not running"; return 1; }
    docker compose ps ${EXECUTOR_SERVICE} | grep -q "Up" || { log_error "Executor not running"; return 1; }
    
    # Test concurrent validation workflows
    log_info "Testing concurrent validation workflows..."
    local CONCURRENT_VALIDATIONS=${LOAD_TEST_CONCURRENCY}
    local pids=()
    
    for i in $(seq 1 $CONCURRENT_VALIDATIONS); do
        (
            docker compose exec ${VALIDATOR_SERVICE} validator verify \
                --host ${SSH_TARGET_SERVICE} \
                --username ${SSH_TEST_USER} \
                --private-key ${SSH_TEST_KEY} \
                --gpu-attestor-path ${SHARED_ATTESTOR_PATH}/gpu-attestor \
                --executor-id "load-test-$i" \
                --remote-work-dir "${TEST_WORK_DIR}_load_$i" >/dev/null 2>&1
            echo "[LOAD] Validation $i completed with exit code $?"
        ) &
        pids+=($!)
    done
    
    # Wait for all validations to complete
    local failed=0
    for pid in "${pids[@]}"; do
        wait $pid || ((failed++))
    done
    
    if [ $failed -eq 0 ]; then
        log_success "All $CONCURRENT_VALIDATIONS concurrent validations completed successfully"
    else
        log_warn "$failed out of $CONCURRENT_VALIDATIONS validations failed"
    fi
    
    # Test miner executor management under load
    log_info "Testing miner executor fleet management under load..."
    for i in {1..10}; do
        docker compose exec miner miner executor health >/dev/null 2>&1 &
    done
    wait
    log_success "Executor fleet health checks completed"
    
    # Check system resource usage
    log_info "System resource usage during load test:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep -E "NAME|validator|miner|executor" || true
}

test_failover_recovery() {
    log_info "Testing failover and recovery scenarios..."
    
    if [ "${ENABLE_FAILOVER_TESTS}" != "true" ]; then
        log_warn "Failover tests disabled"
        return 0
    fi
    
    # Test executor failure and recovery
    log_info "Testing executor failure and recovery..."
    docker compose exec ${MINER_SERVICE} miner executor list > /tmp/executors_before.txt
    
    # Stop executor
    docker compose stop ${EXECUTOR_SERVICE}
    sleep 10
    
    # Check if miner detects executor failure
    if docker compose exec ${MINER_SERVICE} miner executor health 2>&1 | grep -q "unhealthy\|offline\|failed"; then
        log_success "Miner detected executor failure"
    else
        log_warn "Miner may not have detected executor failure"
    fi
    
    # Restart executor
    docker compose start ${EXECUTOR_SERVICE}
    sleep 20
    
    # Check if miner detects executor recovery
    if docker compose exec ${MINER_SERVICE} miner executor health 2>&1 | grep -q "healthy\|online"; then
        log_success "Miner detected executor recovery"
    else
        log_error "Miner did not detect executor recovery"
    fi
    
    # Test database recovery
    log_info "Testing database recovery..."
    docker compose exec ${VALIDATOR_SERVICE} cp ${VALIDATOR_DB_PATH} ${VALIDATOR_DB_PATH}.backup
    log_success "Database backup created"
    
    # Restore from backup
    docker compose exec ${VALIDATOR_SERVICE} cp ${VALIDATOR_DB_PATH}.backup ${VALIDATOR_DB_PATH}
    log_success "Database restored from backup"
}

test_security_validation() {
    log_info "Testing security validations..."
    
    if [ "${ENABLE_SECURITY_TESTS}" != "true" ]; then
        log_warn "Security tests disabled"
        return 0
    fi
    
    # Test invalid authentication
    log_info "Testing invalid authentication..."
    if curl -s -f -H "Authorization: Bearer invalid_token" http://${VALIDATOR_HOST}:${VALIDATOR_API_PORT}/capacity/available 2>&1 | grep -q "401\|403\|Unauthorized"; then
        log_success "Invalid authentication properly rejected"
    else
        log_warn "Invalid authentication may not be properly handled"
    fi
    
    # Test rate limiting
    log_info "Testing rate limiting..."
    local RATE_LIMIT_REQUESTS=${LOAD_TEST_RATE_LIMIT_REQUESTS}
    local rate_limited=false
    
    for i in $(seq 1 $RATE_LIMIT_REQUESTS); do
        if ! curl -s -f http://${VALIDATOR_HOST}:${VALIDATOR_PORT}/health >/dev/null 2>&1; then
            rate_limited=true
            break
        fi
    done
    
    if [ "$rate_limited" = true ]; then
        log_success "Rate limiting is active"
    else
        log_warn "Rate limiting may not be configured"
    fi
    
    # Test input validation
    log_info "Testing input validation..."
    if docker compose exec ${VALIDATOR_SERVICE} validator verify \
        --host "'; DROP TABLE verification_logs; --" \
        --username ${SSH_TEST_USER} \
        --private-key ${SSH_TEST_KEY} 2>&1 | grep -q "invalid\|error"; then
        log_success "SQL injection attempt properly blocked"
    else
        log_warn "Input validation needs verification"
    fi
}

test_production_readiness() {
    log_info "Testing production readiness..."
    
    # Check configuration from environment
    if [ -f ".env" ]; then
        source .env
    fi
    
    # Test with production configuration if available
    if [ -n "$BASILICA_PRODUCTION_CONFIG" ] && [ -f "$BASILICA_PRODUCTION_CONFIG" ]; then
        log_info "Using production configuration from $BASILICA_PRODUCTION_CONFIG"
        
        # Validate production config
        docker compose exec validator validator --config "$BASILICA_PRODUCTION_CONFIG" --validate-config >/dev/null 2>&1 || \
            { log_error "Production configuration validation failed"; return 1; }
        log_success "Production configuration valid"
    else
        log_warn "No production configuration found - using test configuration"
    fi
    
    # Test metrics collection
    log_info "Testing metrics collection..."
    local metrics_endpoints=("${VALIDATOR_HOST}:${VALIDATOR_METRICS_PORT}/metrics" "${EXECUTOR_HOST}:${EXECUTOR_METRICS_PORT}/metrics" "${MINER_HOST}:${MINER_METRICS_PORT}/metrics")
    
    for endpoint in "${metrics_endpoints[@]}"; do
        if curl -s -f "http://$endpoint" | grep -q "basilica_"; then
            log_success "Metrics available at $endpoint"
        else
            log_warn "No basilica metrics found at $endpoint"
        fi
    done
    
    # Test logging
    log_info "Testing structured logging..."
    if docker compose logs validator 2>&1 | grep -q '"level":\|"timestamp":\|"msg":'; then
        log_success "Structured JSON logging enabled"
    else
        log_warn "Structured logging may not be configured"
    fi
    
    # Test health endpoints
    log_info "Testing health check endpoints..."
    local health_checks_passed=0
    local health_checks_total=3
    
    curl -f -s http://${VALIDATOR_HOST}:${VALIDATOR_PORT}/health >/dev/null && ((health_checks_passed++)) || log_warn "Validator health check failed"
    curl -f -s http://${MINER_HOST}:${MINER_METRICS_PORT}/metrics >/dev/null && ((health_checks_passed++)) || log_warn "Miner health check failed"
    docker compose exec ${EXECUTOR_SERVICE} executor service health >/dev/null 2>&1 && ((health_checks_passed++)) || log_warn "Executor health check failed"
    
    log_info "Health checks passed: $health_checks_passed/$health_checks_total"
}

main() {
    case "${1:-all}" in
        "prerequisites") check_prerequisites ;;
        "build") check_prerequisites; build_images ;;
        "individual") test_individual_services ;;
        "core") check_prerequisites; build_images; start_core_services; test_service_health; test_api_endpoints ;;
        "miner") test_miner_functionality ;;
        "ssh") test_ssh_validation ;;
        "gpu") test_gpu_attestation ;;
        "e2e") test_end_to_end ;;
        "monitoring") start_monitoring ;;
        "status") show_status ;;
        "cleanup") cleanup ;;
        "setup") ./setup-test-env.sh ;;
        "load") test_load_performance ;;
        "failover") test_failover_recovery ;;
        "security") test_security_validation ;;
        "production-ready") test_production_readiness ;;
        "all") 
            check_prerequisites
            build_images
            test_individual_services
            start_core_services
            test_service_health
            test_api_endpoints
            test_miner_functionality
            test_ssh_validation
            test_gpu_attestation
            test_end_to_end
            test_load_performance
            test_failover_recovery
            test_security_validation
            test_production_readiness
            start_monitoring
            show_status
            log_success "All tests completed"
            ;;
        *)
            echo "Usage: $0 [prerequisites|build|individual|core|miner|ssh|gpu|e2e|monitoring|status|setup|cleanup|load|failover|security|production-ready|all]"
            echo "Commands:"
            echo "  prerequisites    - Check system prerequisites"
            echo "  build           - Build Docker images"
            echo "  individual      - Test individual compose files"
            echo "  core            - Start and test core services"
            echo "  miner           - Test miner-specific functionality"
            echo "  ssh             - Test SSH validation"
            echo "  gpu             - Test GPU attestation"
            echo "  e2e             - Test end-to-end workflow"
            echo "  monitoring      - Start monitoring services"
            echo "  status          - Show system status"
            echo "  setup           - Setup test environment (keys, dirs)"
            echo "  cleanup         - Clean up test environment"
            echo "  load            - Run load performance tests"
            echo "  failover        - Test failover and recovery scenarios"
            echo "  security        - Test security validations"
            echo "  production-ready - Test production readiness"
            echo "  all             - Run complete test workflow (default)"
            exit 1
            ;;
    esac
}

trap cleanup INT TERM
main "$@"