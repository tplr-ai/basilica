#!/bin/bash
# Run tests for Basilica

set -e

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test_utils.sh"

# Configuration
PARALLEL=${PARALLEL:-false}
VERBOSE=${VERBOSE:-false}
CAPTURE_OUTPUT=${CAPTURE_OUTPUT:-true}

# Functions
run_crate_tests() {
    local crate=$1
    local test_type=${2:-""}
    local extra_args=""
    
    # Build cargo test arguments
    [ "$VERBOSE" = "true" ] && extra_args="$extra_args -v"
    [ "$CAPTURE_OUTPUT" = "false" ] && extra_args="$extra_args -- --nocapture"
    [ -n "$test_type" ] && extra_args="$extra_args --test $test_type"
    
    log_header "Running tests for $crate"
    
    if run_cargo_command "$crate" test $extra_args; then
        log_success "$crate tests passed"
        return 0
    else
        log_error "$crate tests failed"
        return 1
    fi
}

run_parallel_tests() {
    local pids=()
    local failed=0
    
    for crate in miner validator executor; do
        if crate_exists "$crate"; then
            run_crate_tests "$crate" &
            pids+=($!)
        fi
    done
    
    # Wait for all tests to complete
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    
    return $failed
}

run_sequential_tests() {
    local failed=0
    
    for crate in miner validator executor; do
        if crate_exists "$crate"; then
            if ! run_crate_tests "$crate"; then
                ((failed++))
            fi
        fi
    done
    
    return $failed
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [CRATE|all]

Run Basilica tests with various options.

OPTIONS:
    -p, --parallel       Run tests in parallel
    -v, --verbose        Verbose output
    -n, --no-capture     Don't capture test output
    -t, --type TYPE      Test type (unit, integration, all)
    -h, --help           Show this help message

ARGUMENTS:
    CRATE               Name of crate to test (miner, validator, executor)
    all                 Run all tests (default)

EXAMPLES:
    $0                   # Run all tests sequentially
    $0 -p                # Run all tests in parallel
    $0 miner             # Run only miner tests
    $0 -t integration    # Run only integration tests
    $0 -pn               # Run parallel without output capture
EOF
}

# Parse arguments
parse_args() {
    local crate=""
    local test_type=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--no-capture)
                CAPTURE_OUTPUT=false
                shift
                ;;
            -t|--type)
                test_type="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                crate="$1"
                shift
                ;;
        esac
    done
    
    echo "${crate:-all}:${test_type}"
}

# Main
main() {
    ensure_basilica_root || exit 1
    
    # Parse command line arguments
    IFS=':' read -r target test_type <<< "$(parse_args "$@")"
    
    log_header "Basilica Test Runner"
    log_info "Target: $target"
    log_info "Parallel: $PARALLEL"
    log_info "Verbose: $VERBOSE"
    log_info "Capture output: $CAPTURE_OUTPUT"
    [ -n "$test_type" ] && log_info "Test type: $test_type"
    echo
    
    local failed=0
    
    case "$target" in
        all)
            if [ "$PARALLEL" = "true" ]; then
                run_parallel_tests
                failed=$?
            else
                run_sequential_tests
                failed=$?
            fi
            ;;
        miner|validator|executor)
            if crate_exists "$target"; then
                run_crate_tests "$target" "$test_type"
                failed=$?
            else
                log_error "Crate '$target' not found"
                exit 1
            fi
            ;;
        *)
            log_error "Unknown target: $target"
            usage
            exit 1
            ;;
    esac
    
    echo
    if [ $failed -eq 0 ]; then
        log_success "All tests completed successfully!"
    else
        log_error "$failed test suite(s) failed"
        exit 1
    fi
}

main "$@"