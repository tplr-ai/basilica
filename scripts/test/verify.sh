#!/bin/bash
# Verify test implementation

set -e

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test_utils.sh"

# Functions
verify_crate_tests() {
    local crate=$1
    local test_type=${2:-integration}
    
    log_header "Verifying $crate $test_type tests"
    
    local test_dir="$CRATES_DIR/$crate/tests/$test_type"
    if [ ! -d "$test_dir" ]; then
        log_warning "No $test_type tests found for $crate"
        return 1
    fi
    
    # Get statistics
    local stats=$(get_crate_test_stats "$crate" "$test_type")
    IFS=':' read -r file_count test_count line_count <<< "$stats"
    
    log_info "Found $file_count test files with $test_count tests ($line_count lines)"
    
    # List test files with details
    find "$test_dir" -name "*.rs" -not -name "mod.rs" | while read -r file; do
        local basename=$(basename "$file")
        local tests=$(grep -c "^\s*#\[\(test\|tokio::test\)\]" "$file" || true)
        local lines=$(wc -l < "$file")
        echo "  - $basename: $tests tests, $lines lines"
    done
    
    # Check for placeholders
    log_info "Checking for placeholder code..."
    local placeholders=$(check_placeholders "$test_dir")
    if [ -z "$placeholders" ]; then
        log_success "No placeholders found"
    else
        log_error "Found placeholders:"
        echo "$placeholders"
    fi
    
    # Check compilation
    log_info "Checking compilation..."
    if check_test_compilation "$crate"; then
        log_success "Tests compile successfully"
    else
        log_warning "Compilation check failed (may be due to missing dependencies)"
    fi
    
    echo
}

verify_all_tests() {
    local test_type=${1:-integration}
    local total_files=0
    local total_tests=0
    local total_lines=0
    
    log_header "Basilica Test Verification Report"
    echo
    
    # Verify each crate
    for crate in miner validator executor; do
        if crate_exists "$crate"; then
            verify_crate_tests "$crate" "$test_type"
            
            # Add to totals
            local stats=$(get_crate_test_stats "$crate" "$test_type")
            if [ $? -eq 0 ]; then
                IFS=':' read -r files tests lines <<< "$stats"
                total_files=$((total_files + files))
                total_tests=$((total_tests + tests))
                total_lines=$((total_lines + lines))
            fi
        fi
    done
    
    # Summary
    log_header "Summary"
    echo "Total test files: $total_files"
    echo "Total test functions: $total_tests"
    echo "Total lines of code: $total_lines"
    echo
    
    # Overall placeholder check
    log_info "Overall placeholder check..."
    local all_placeholders=$(find "$CRATES_DIR"/*/tests -name "*.rs" -exec grep -l "todo!\|unimplemented!" {} \; 2>/dev/null | wc -l)
    if [ "$all_placeholders" -eq 0 ]; then
        log_success "No TODO/unimplemented placeholders in any tests"
    else
        log_error "Found $all_placeholders files with placeholders"
    fi
}

# Main
main() {
    ensure_basilica_root || exit 1
    
    case "${1:-all}" in
        all)
            verify_all_tests "integration"
            ;;
        unit)
            verify_all_tests "unit"
            ;;
        *)
            log_error "Usage: $0 [all|unit]"
            exit 1
            ;;
    esac
}

main "$@"