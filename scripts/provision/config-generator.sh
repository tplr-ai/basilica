#!/bin/bash
# Fixed configuration generator for Basilica services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# SSH wrapper not needed for dynamic discovery
# If ssh-wrapper.sh exists, source it for backward compatibility
if [[ -f "$SCRIPT_DIR/../lib/ssh-wrapper.sh" ]]; then
    source "$SCRIPT_DIR/../lib/ssh-wrapper.sh"
fi

TEMPLATE_DIR="$SCRIPT_DIR/templates"
CONFIG_OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-$BASILICA_ROOT}"

# Generate configuration from template using envsubst
generate_config() {
    local template_file="$1"
    local output_file="$2"
    local env_file="$3"
    
    if [[ ! -f "$template_file" ]]; then
        log_error "Template not found: $template_file"
        return 1
    fi
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    log_info "Generating config from $(basename "$template_file")"
    
    # Load environment variables
    source "$env_file"
    
    # Export all variables for envsubst
    export $(grep -v '^#' "$env_file" | grep '=' | cut -d= -f1)
    
    # Use envsubst to replace variables
    envsubst < "$template_file" > "$output_file"
    
    # Validate the generated config
    validate_toml "$output_file" || {
        log_error "Generated configuration is invalid"
        rm -f "$output_file"
        return 1
    }
    
    log_success "Generated: $output_file"
}

# Validate TOML file
validate_toml() {
    local file="$1"
    
    # Skip validation if python3 or toml module not available
    if ! command -v python3 &> /dev/null; then
        log_warn "Python3 not found, skipping TOML validation"
        return 0
    fi
    
    # Try to parse the TOML file
    python3 -c "
import sys
try:
    import toml
    with open('$file', 'r') as f:
        toml.load(f)
    sys.exit(0)
except ImportError:
    print('TOML module not installed, skipping validation', file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f'TOML validation error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1

    return $?
}

# Generate all configurations
generate_all_configs() {
    local env="${1:-production}"
    local env_file="$SCRIPT_DIR/environments/${env}.conf"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    log_header "Generating configurations for $env environment"
    
    # Generate executor config
    generate_config \
        "$TEMPLATE_DIR/executor.toml.template" \
        "$CONFIG_OUTPUT_DIR/executor.toml" \
        "$env_file" || return 1
    
    # Generate miner config
    generate_config \
        "$TEMPLATE_DIR/miner.toml.template" \
        "$CONFIG_OUTPUT_DIR/miner.toml" \
        "$env_file" || return 1
    
    # Generate validator config
    generate_config \
        "$TEMPLATE_DIR/validator.toml.template" \
        "$CONFIG_OUTPUT_DIR/validator.toml" \
        "$env_file" || return 1
    
    log_success "All configurations generated successfully"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        all)
            generate_all_configs "$@"
            ;;
        executor)
            generate_config \
                "$TEMPLATE_DIR/executor.toml.template" \
                "$CONFIG_OUTPUT_DIR/executor.toml" \
                "$SCRIPT_DIR/environments/${1:-production}.conf"
            ;;
        miner)
            generate_config \
                "$TEMPLATE_DIR/miner.toml.template" \
                "$CONFIG_OUTPUT_DIR/miner.toml" \
                "$SCRIPT_DIR/environments/${1:-production}.conf"
            ;;
        validator)
            generate_config \
                "$TEMPLATE_DIR/validator.toml.template" \
                "$CONFIG_OUTPUT_DIR/validator.toml" \
                "$SCRIPT_DIR/environments/${1:-production}.conf"
            ;;
        help|*)
            cat << EOF
Basilica Configuration Generator

Usage: config-generator.sh <command> [environment]

Commands:
    all         Generate all service configurations
    executor    Generate executor configuration only
    miner       Generate miner configuration only
    validator   Generate validator configuration only

Environment:
    production  Production environment (default)
    staging     Staging environment
    testnet     Testnet environment

Examples:
    config-generator.sh all production
    config-generator.sh miner staging
    config-generator.sh validator testnet
EOF
            ;;
    esac
}

main "$@"