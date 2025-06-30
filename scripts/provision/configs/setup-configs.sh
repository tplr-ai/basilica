#!/bin/bash
# Basilica Configuration Setup Script
# Helps users create initial configuration files from examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_usage() {
    cat << EOF
Basilica Configuration Setup

USAGE:
    setup-configs.sh [OPTIONS] <COMMAND>

COMMANDS:
    init        Initialize configuration files from examples
    validate    Validate existing configuration files
    clean       Remove generated configuration files
    help        Show this help message

OPTIONS:
    --env ENV       Environment (production, staging, development) [default: development]
    --force         Overwrite existing configuration files
    --secrets       Also copy secrets.toml.example (with warnings)
    --dry-run       Show what would be done without executing
    -h, --help      Show this help message

EXAMPLES:
    setup-configs.sh init                    # Initialize development configs
    setup-configs.sh init --env production   # Initialize production configs  
    setup-configs.sh init --force            # Overwrite existing configs
    setup-configs.sh validate                # Check configuration syntax
    setup-configs.sh clean                   # Remove generated configs

SECURITY NOTE:
    Configuration files contain sensitive values and are automatically
    excluded from version control via .gitignore entries.
EOF
}

# Parse command line arguments
ENVIRONMENT="development"
FORCE=false
INCLUDE_SECRETS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --secrets)
            INCLUDE_SECRETS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        init|validate|clean|help)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

COMMAND="${COMMAND:-help}"

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would execute: $description"
        log_info "Command: $cmd"
    else
        log_info "$description"
        eval "$cmd"
    fi
}

# Command: Initialize configuration files
cmd_init() {
    log_info "Initializing Basilica configuration files for environment: $ENVIRONMENT"
    
    # List of configuration files to create
    local config_files=(
        "executor.toml"
        "miner.toml"
        "validator.toml"
    )
    
    # Check if files exist and handle force option
    if [[ "$FORCE" != "true" ]]; then
        for config in "${config_files[@]}"; do
            if [[ -f "$config" ]]; then
                log_warning "Configuration file $config already exists"
                log_warning "Use --force to overwrite existing files"
                return 1
            fi
        done
    fi
    
    # Copy example files
    for config in "${config_files[@]}"; do
        local example_file="${config}.example"
        if [[ -f "$example_file" ]]; then
            execute_cmd "cp '$example_file' '$config'" "Creating $config from example"
            if [[ "$DRY_RUN" != "true" ]]; then
                log_success "Created $config"
            fi
        else
            log_error "Example file $example_file not found"
            return 1
        fi
    done
    
    # Handle secrets file with special warning
    if [[ "$INCLUDE_SECRETS" == "true" ]]; then
        if [[ -f "secrets.toml.example" ]]; then
            log_warning "Creating secrets.toml - CONTAINS SENSITIVE DATA!"
            log_warning "Remember to:"
            log_warning "1. Fill in actual secret values"
            log_warning "2. Never commit this file to version control"
            log_warning "3. Set restrictive file permissions (600)"
            
            execute_cmd "cp 'secrets.toml.example' 'secrets.toml'" "Creating secrets.toml"
            if [[ "$DRY_RUN" != "true" ]]; then
                execute_cmd "chmod 600 'secrets.toml'" "Setting secure permissions"
                log_success "Created secrets.toml with secure permissions"
            fi
        fi
    fi
    
    # Set appropriate permissions
    for config in "${config_files[@]}"; do
        if [[ -f "$config" && "$DRY_RUN" != "true" ]]; then
            execute_cmd "chmod 600 '$config'" "Setting secure permissions for $config"
        fi
    done
    
    log_success "Configuration initialization complete!"
    echo
    log_info "Next steps:"
    echo "1. Edit configuration files with your actual values:"
    for config in "${config_files[@]}"; do
        echo "   vim $config"
    done
    if [[ "$INCLUDE_SECRETS" == "true" ]]; then
        echo "   vim secrets.toml  # Add your actual secrets"
    fi
    echo "2. Deploy configurations: ./scripts/basilica.sh deploy config"
    echo "3. Validate setup: ./scripts/basilica.sh provision validate"
}

# Command: Validate configuration files
cmd_validate() {
    log_info "Validating Basilica configuration files"
    
    local config_files=(
        "executor.toml"
        "miner.toml"
        "validator.toml"
    )
    
    local errors=0
    
    for config in "${config_files[@]}"; do
        if [[ -f "$config" ]]; then
            log_info "Validating $config..."
            
            # Check TOML syntax using Python (if available)
            if command -v python3 &> /dev/null; then
                if python3 -c "import toml; toml.load('$config')" 2>/dev/null; then
                    log_success "$config syntax is valid"
                else
                    log_error "$config has invalid TOML syntax"
                    ((errors++))
                fi
            else
                log_warning "Python3 not available, skipping TOML syntax validation"
            fi
            
            # Check for placeholder values
            if grep -q "YOUR_.*_HERE" "$config" 2>/dev/null; then
                log_warning "$config contains placeholder values that need to be replaced"
                grep "YOUR_.*_HERE" "$config" | head -3
                ((errors++))
            fi
            
            # Check file permissions
            local perms=$(stat -c %a "$config" 2>/dev/null || echo "unknown")
            if [[ "$perms" != "600" ]]; then
                log_warning "$config has permissions $perms (should be 600 for security)"
            fi
            
        else
            log_error "$config not found"
            ((errors++))
        fi
    done
    
    # Check secrets file if it exists
    if [[ -f "secrets.toml" ]]; then
        log_info "Validating secrets.toml..."
        local perms=$(stat -c %a "secrets.toml" 2>/dev/null || echo "unknown")
        if [[ "$perms" != "600" ]]; then
            log_error "secrets.toml has permissions $perms (MUST be 600 for security)"
            ((errors++))
        else
            log_success "secrets.toml has correct permissions"
        fi
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "All configuration files are valid!"
        return 0
    else
        log_error "Found $errors validation errors"
        return 1
    fi
}

# Command: Clean configuration files
cmd_clean() {
    log_info "Cleaning generated Basilica configuration files"
    
    local config_files=(
        "executor.toml"
        "miner.toml"
        "validator.toml"
        "secrets.toml"
    )
    
    for config in "${config_files[@]}"; do
        if [[ -f "$config" ]]; then
            execute_cmd "rm '$config'" "Removing $config"
            if [[ "$DRY_RUN" != "true" ]]; then
                log_success "Removed $config"
            fi
        fi
    done
    
    log_success "Cleanup complete!"
}

# Command: Help
cmd_help() {
    print_usage
}

# Main execution
case "$COMMAND" in
    init)
        cmd_init
        ;;
    validate)
        cmd_validate
        ;;
    clean)
        cmd_clean
        ;;
    help)
        cmd_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac