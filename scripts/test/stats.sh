#!/bin/bash
# Generate test statistics and reports

set -e

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test_utils.sh"

# Functions
print_crate_stats() {
    local crate=$1
    local test_type=$2
    local stats=$(get_crate_test_stats "$crate" "$test_type")
    
    if [ $? -eq 0 ]; then
        IFS=':' read -r files tests lines <<< "$stats"
        printf "%-12s %-12s %6d %8d %8d\n" "$crate" "$test_type" "$files" "$tests" "$lines"
    fi
}

generate_detailed_report() {
    local test_type=${1:-integration}
    
    log_header "Detailed Test Report - $test_type tests"
    echo
    
    for crate in miner validator executor; do
        if ! crate_exists "$crate"; then
            continue
        fi
        
        local test_dir="$CRATES_DIR/$crate/tests/$test_type"
        if [ ! -d "$test_dir" ]; then
            continue
        fi
        
        echo "## $crate"
        echo
        
        find "$test_dir" -name "*.rs" -not -name "mod.rs" | sort | while read -r file; do
            local basename=$(basename "$file" .rs)
            local tests=$(grep -c "^\s*#\[\(test\|tokio::test\)\]" "$file" || true)
            local lines=$(wc -l < "$file")
            
            echo "### $basename"
            echo "- Lines: $lines"
            echo "- Tests: $tests"
            
            # List test function names
            if [ "$tests" -gt 0 ]; then
                echo "- Test functions:"
                grep -h "^\s*\(async \)\?fn test_" "$file" | sed 's/^\s*/    - /' | sed 's/(.*$//'
            fi
            echo
        done
    done
}

generate_summary_report() {
    log_header "Test Statistics Summary"
    echo
    
    # Header
    printf "%-12s %-12s %6s %8s %8s\n" "Crate" "Type" "Files" "Tests" "Lines"
    printf "%s\n" "--------------------------------------------------------"
    
    # Stats for each crate and type
    local total_files=0
    local total_tests=0
    local total_lines=0
    
    for crate in miner validator executor; do
        if ! crate_exists "$crate"; then
            continue
        fi
        
        for test_type in unit integration; do
            local stats=$(get_crate_test_stats "$crate" "$test_type")
            if [ $? -eq 0 ]; then
                print_crate_stats "$crate" "$test_type"
                
                IFS=':' read -r files tests lines <<< "$stats"
                total_files=$((total_files + files))
                total_tests=$((total_tests + tests))
                total_lines=$((total_lines + lines))
            fi
        done
    done
    
    # Total
    printf "%s\n" "--------------------------------------------------------"
    printf "%-12s %-12s %6d %8d %8d\n" "TOTAL" "" "$total_files" "$total_tests" "$total_lines"
    echo
}

generate_coverage_estimate() {
    log_header "Test Coverage Estimate"
    echo
    
    for crate in miner validator executor; do
        if ! crate_exists "$crate"; then
            continue
        fi
        
        local src_files=$(find "$CRATES_DIR/$crate/src" -name "*.rs" 2>/dev/null | wc -l)
        local test_files=$(find "$CRATES_DIR/$crate/tests" -name "*.rs" -not -name "mod.rs" 2>/dev/null | wc -l)
        local test_ratio=0
        
        if [ "$src_files" -gt 0 ]; then
            test_ratio=$(echo "scale=2; $test_files * 100 / $src_files" | bc)
        fi
        
        printf "%-12s %3d source files, %3d test files (%.0f%% coverage by file count)\n" \
            "$crate:" "$src_files" "$test_files" "$test_ratio"
    done
    echo
}

# Main
main() {
    ensure_basilica_root || exit 1
    
    case "${1:-summary}" in
        summary)
            generate_summary_report
            ;;
        detailed)
            generate_detailed_report "${2:-integration}"
            ;;
        coverage)
            generate_coverage_estimate
            ;;
        all)
            generate_summary_report
            generate_coverage_estimate
            generate_detailed_report "integration"
            ;;
        *)
            log_error "Usage: $0 [summary|detailed|coverage|all]"
            exit 1
            ;;
    esac
}

main "$@"