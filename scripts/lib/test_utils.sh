#!/bin/bash
# Test-related utility functions

# Source common functions
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Test types
export TEST_TYPES=("unit" "integration" "doc")

# Count test files in a crate
count_test_files() {
    local crate=$1
    local test_type=${2:-integration}
    local test_dir="$CRATES_DIR/$crate/tests/$test_type"
    
    if [ -d "$test_dir" ]; then
        find "$test_dir" -name "*.rs" -not -name "mod.rs" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

# Count test functions in files
count_test_functions() {
    local files=$1
    grep -h "^\s*#\[\(test\|tokio::test\)\]" $files 2>/dev/null | wc -l
}

# Get test statistics for a crate
get_crate_test_stats() {
    local crate=$1
    local test_type=${2:-integration}
    local test_dir="$CRATES_DIR/$crate/tests/$test_type"
    
    if [ ! -d "$test_dir" ]; then
        return 1
    fi
    
    local files=$(find "$test_dir" -name "*.rs" -not -name "mod.rs" 2>/dev/null)
    local file_count=$(echo "$files" | wc -l)
    local test_count=0
    local line_count=0
    
    if [ -n "$files" ]; then
        test_count=$(grep -h "^\s*#\[\(test\|tokio::test\)\]" $files 2>/dev/null | wc -l)
        line_count=$(wc -l $files 2>/dev/null | tail -1 | awk '{print $1}')
    fi
    
    echo "$file_count:$test_count:$line_count"
}

# Check for placeholder code
check_placeholders() {
    local path=$1
    local patterns="todo!|unimplemented!|unreachable!|panic!.*TODO"
    
    grep -n -E "$patterns" "$path"/*.rs 2>/dev/null
}

# Run cargo command with proper error handling
run_cargo_command() {
    local crate=$1
    shift
    local cargo_args="$@"
    
    cd "$CRATES_DIR/$crate" || return 1
    
    if cargo $cargo_args; then
        cd - >/dev/null
        return 0
    else
        cd - >/dev/null
        return 1
    fi
}

# Check if tests compile
check_test_compilation() {
    local crate=$1
    local test_type=${2:-""}
    
    if [ -n "$test_type" ]; then
        run_cargo_command "$crate" test --no-run --test "$test_type"
    else
        run_cargo_command "$crate" test --no-run --all-targets
    fi
}

# Format test output
format_test_output() {
    local crate=$1
    local status=$2
    local details=$3
    
    if [ "$status" -eq 0 ]; then
        log_success "$crate: $details"
    else
        log_error "$crate: $details"
    fi
}