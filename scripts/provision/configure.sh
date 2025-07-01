#!/bin/bash
# Basilica Service Configuration Generator
# Generates production-ready configurations for validator, miner, and executor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASILICA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

print_usage() {
    cat << EOF
configure.sh - Generate and deploy Basilica service configurations

USAGE:
    configure.sh <COMMAND> [OPTIONS]

COMMANDS:
    generate    Generate configurations for all services
    deploy      Deploy configurations to servers
    validate    Validate generated configurations
    keys        Generate and distribute cryptographic keys

OPTIONS:
    -e, --env ENV       Environment (production, staging, development)
    -s, --service SVC   Specific service (validator, miner, executor)
    -o, --output DIR    Output directory for configurations
    -n, --dry-run       Show generated configs without writing
    -h, --help          Show this help message

EXAMPLES:
    configure.sh generate --env production       # Generate all configs
    configure.sh generate --service validator    # Generate validator config only
    configure.sh deploy --env production         # Deploy configs to servers
    configure.sh keys --env production           # Generate and distribute keys
    configure.sh validate                        # Validate all configurations

CONFIGURATION FEATURES:
    - Environment-specific settings (dev/staging/production)
    - Automatic service discovery configuration
    - Cryptographic key management
    - Network topology optimization
    - Security hardening settings
    - Monitoring and logging setup
EOF
}

# Default values
ENVIRONMENT="production"
SERVICE=""
OUTPUT_DIR="$SCRIPT_DIR/configs"
DRY_RUN=false

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
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

# Generate validator configuration
generate_validator_config() {
    log_info "Generating validator configuration"
    
    local config_file="$OUTPUT_DIR/validator.toml"
    mkdir -p "$OUTPUT_DIR"
    
    cat > "$config_file" << EOF
# Basilica Validator Configuration
# Generated on $(date) for environment: $ENVIRONMENT

[database]
url = "sqlite:/var/lib/basilica/validator/validator.db"
max_connections = 10
min_connections = 1
run_migrations = true

[database.connect_timeout]
secs = 30
nanos = 0

[database.max_lifetime]
secs = 3600
nanos = 0

[server]
host = "0.0.0.0"
port = 8080
max_connections = 1000
tls_enabled = ${VALIDATOR_TLS_ENABLED:-false}

[server.request_timeout]
secs = 30
nanos = 0

[bittensor]
wallet_name = "${VALIDATOR_WALLET_NAME:-validator}"
hotkey_name = "${VALIDATOR_HOTKEY_NAME:-default}"
network = "${BITTENSOR_NETWORK:-finney}"
netuid = ${BITTENSOR_NETUID:-39}
chain_endpoint = "${BITTENSOR_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
weight_interval_secs = 300

[verification]
max_concurrent_verifications = ${VALIDATOR_MAX_CONCURRENT:-50}
min_score_threshold = ${VALIDATOR_MIN_SCORE:-0.1}
min_stake_threshold = ${VALIDATOR_MIN_STAKE:-1000.0}
max_miners_per_round = ${VALIDATOR_MAX_MINERS:-20}
netuid = ${BITTENSOR_NETUID:-39}

[verification.verification_interval]
secs = ${VALIDATOR_VERIFY_INTERVAL:-600}
nanos = 0

[verification.challenge_timeout]
secs = ${VALIDATOR_CHALLENGE_TIMEOUT:-120}
nanos = 0

[verification.min_verification_interval]
secs = ${VALIDATOR_MIN_VERIFY_INTERVAL:-1800}
nanos = 0

[api]
max_body_size = 1048576
bind_address = "0.0.0.0:8080"

[storage]
data_dir = "/var/lib/basilica/validator"

[logging]
level = "${LOG_LEVEL:-info}"
format = "${LOG_FORMAT:-json}"
stdout = true

[metrics]
enabled = true

[metrics.prometheus]
host = "0.0.0.0"
port = 9090
path = "/metrics"

# SSH validation settings
[ssh]
private_key_path = "/etc/basilica/keys/validator_ssh"
timeout_seconds = 30
max_concurrent_sessions = 10

# Network discovery
[discovery]
miner_endpoints = [
    "${MINER_HOST}:${MINER_GRPC_PORT:-8092}",
]
executor_endpoints = [
    "${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}",
]
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would write validator config to: $config_file"
        cat "$config_file"
    else
        log_success "Generated validator config: $config_file"
    fi
}

# Generate miner configuration
generate_miner_config() {
    log_info "Generating miner configuration"
    
    local config_file="$OUTPUT_DIR/miner.toml"
    mkdir -p "$OUTPUT_DIR"
    
    cat > "$config_file" << EOF
# Basilica Miner Configuration
# Generated on $(date) for environment: $ENVIRONMENT

[bittensor]
wallet_name = "${MINER_WALLET_NAME:-miner}"
hotkey_name = "${MINER_HOTKEY_NAME:-default}"
network = "${BITTENSOR_NETWORK:-finney}"
netuid = ${BITTENSOR_NETUID:-39}
chain_endpoint = "${BITTENSOR_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
weight_interval_secs = 300
uid = 0  # Auto-discovered from chain
coldkey_name = "${MINER_COLDKEY_NAME:-default}"
axon_port = ${MINER_AXON_PORT:-8091}
external_ip = "${MINER_EXTERNAL_IP:-${MINER_HOST}}"
max_weight_uids = 256

[database]
url = "sqlite:/var/lib/basilica/miner/miner.db"
max_connections = 10
min_connections = 1
run_migrations = true

[server]
host = "0.0.0.0"
port = ${MINER_GRPC_PORT:-8092}
max_connections = 1000
tls_enabled = ${MINER_TLS_ENABLED:-false}

[validator_comms]
max_concurrent_sessions = ${MINER_MAX_SESSIONS:-100}
auth.enabled = true
auth.method = "bittensor_signature"
rate_limit.enabled = true
rate_limit.requests_per_second = ${MINER_RATE_LIMIT:-10}
rate_limit.burst_capacity = ${MINER_BURST_CAPACITY:-20}

[executor_management]
# Static executor configuration
executors = [
    { id = "executor-1", grpc_address = "${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}", name = "GPU Machine 1" },
]
health_check_interval = { secs = 60 }
health_check_timeout = { secs = 10 }
max_retry_attempts = 3
auto_recovery = true

[security]
enable_mtls = ${MINER_MTLS_ENABLED:-false}
jwt_secret = "${MINER_JWT_SECRET:-$(openssl rand -hex 32)}"
allowed_validators = []  # Empty = allow all
verify_signatures = true
token_expiration = { secs = 3600 }

[logging]
level = "${LOG_LEVEL:-info}"
format = "${LOG_FORMAT:-json}"

[metrics]
enabled = true
port = 9090

# Network discovery
[discovery]
validator_endpoints = [
    "${VALIDATOR_HOST}:${VALIDATOR_API_PORT:-8080}",
]
executor_endpoints = [
    "${EXECUTOR_HOST}:${EXECUTOR_GRPC_PORT:-50051}",
]
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would write miner config to: $config_file"
        cat "$config_file"
    else
        log_success "Generated miner config: $config_file"
    fi
}

# Generate executor configuration
generate_executor_config() {
    log_info "Generating executor configuration"
    
    local config_file="$OUTPUT_DIR/executor.toml"
    mkdir -p "$OUTPUT_DIR"
    
    cat > "$config_file" << EOF
# Basilica Executor Configuration
# Generated on $(date) for environment: $ENVIRONMENT

[server]
host = "0.0.0.0"
port = ${EXECUTOR_GRPC_PORT:-50051}

[logging]
level = "${LOG_LEVEL:-info}"
format = "${LOG_FORMAT:-json}"

[metrics]
enabled = true
port = 9090

[system]
update_interval = 10
enable_gpu_monitoring = true
enable_network_monitoring = true
max_cpu_usage = 90.0
max_memory_usage = 90.0
max_gpu_memory_usage = 90.0
min_disk_space_gb = 10

[docker]
socket_path = "/var/run/docker.sock"
network_name = "basilica"
enable_gpu_support = true
pull_timeout = 300
stop_timeout = 30

[validator]
enabled = ${EXECUTOR_VALIDATOR_ACCESS:-true}
ssh_port = 22
max_sessions = ${EXECUTOR_MAX_SESSIONS:-10}
session_timeout = ${EXECUTOR_SESSION_TIMEOUT:-3600}

# Managing miner hotkey for authentication
managing_miner_hotkey = "${MINER_HOTKEY:-5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY}"

# Security settings
[security]
allowed_validator_ips = [
    "${VALIDATOR_HOST}",
]
allowed_miner_ips = [
    "${MINER_HOST}",
]
ssh_key_path = "/etc/basilica/keys/executor_ssh"
require_authentication = true
session_encryption = true

# Resource limits
[resources]
max_containers = ${EXECUTOR_MAX_CONTAINERS:-10}
max_memory_per_container = "${EXECUTOR_MAX_MEMORY:-8g}"
max_cpu_per_container = ${EXECUTOR_MAX_CPU:-4.0}
gpu_allocation_strategy = "exclusive"

# Network discovery
[discovery]
miner_endpoints = [
    "${MINER_HOST}:${MINER_GRPC_PORT:-8092}",
]
validator_endpoints = [
    "${VALIDATOR_HOST}:${VALIDATOR_API_PORT:-8080}",
]
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would write executor config to: $config_file"
        cat "$config_file"
    else
        log_success "Generated executor config: $config_file"
    fi
}

# Generate systemd service files
generate_systemd_services() {
    log_info "Generating systemd service files"
    
    local services_dir="$OUTPUT_DIR/systemd"
    mkdir -p "$services_dir"
    
    # Validator service
    cat > "$services_dir/basilica-validator.service" << 'EOF'
[Unit]
Description=Basilica Validator Service
Documentation=https://github.com/spacejar/basilica
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/basilica

[Service]
Type=simple
User=basilica
Group=basilica
WorkingDirectory=/var/lib/basilica/validator
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
ExecStart=/usr/local/bin/validator start --config /etc/basilica/validator.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=basilica-validator

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/basilica /var/log/basilica
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictRealtime=yes
RestrictNamespaces=yes
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    # Miner service
    cat > "$services_dir/basilica-miner.service" << 'EOF'
[Unit]
Description=Basilica Miner Service
Documentation=https://github.com/spacejar/basilica
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/basilica

[Service]
Type=simple
User=basilica
Group=basilica
WorkingDirectory=/var/lib/basilica/miner
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
ExecStart=/usr/local/bin/miner --config /etc/basilica/miner.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=basilica-miner

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/basilica /var/log/basilica
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictRealtime=yes
RestrictNamespaces=yes
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    # Executor service
    cat > "$services_dir/basilica-executor.service" << 'EOF'
[Unit]
Description=Basilica Executor Service
Documentation=https://github.com/spacejar/basilica
After=network-online.target docker.service
Wants=network-online.target
RequiresMountsFor=/var/lib/basilica

[Service]
Type=simple
User=basilica
Group=basilica
SupplementaryGroups=docker
WorkingDirectory=/var/lib/basilica/executor
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
ExecStart=/usr/local/bin/executor --server --config /etc/basilica/executor.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=basilica-executor

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/basilica /var/log/basilica /var/run/docker.sock
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictRealtime=yes
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    log_success "Generated systemd services in: $services_dir"
}

# Command: Generate configurations
cmd_generate() {
    log_header "Generating Service Configurations"
    load_env_config
    
    if [[ -n "$SERVICE" ]]; then
        case "$SERVICE" in
            validator)
                generate_validator_config
                ;;
            miner)
                generate_miner_config
                ;;
            executor)
                generate_executor_config
                ;;
            *)
                log_error "Unknown service: $SERVICE"
                exit 1
                ;;
        esac
    else
        generate_validator_config
        generate_miner_config
        generate_executor_config
        generate_systemd_services
    fi
    
    log_success "Configuration generation completed"
}

# Command: Deploy configurations to servers
cmd_deploy() {
    log_header "Deploying Configurations"
    load_env_config
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would deploy configurations to servers"
        return
    fi
    
    # Deploy validator config
    log_info "Deploying validator configuration"
    scp -P "$VALIDATOR_PORT" "$OUTPUT_DIR/validator.toml" \
        "$VALIDATOR_USER@$VALIDATOR_HOST:/tmp/validator.toml"
    ssh -p "$VALIDATOR_PORT" "$VALIDATOR_USER@$VALIDATOR_HOST" \
        "sudo mv /tmp/validator.toml /etc/basilica/validator.toml && sudo chown root:basilica /etc/basilica/validator.toml && sudo chmod 640 /etc/basilica/validator.toml"
    
    # Deploy miner config
    log_info "Deploying miner configuration"
    scp -P "$MINER_PORT" "$OUTPUT_DIR/miner.toml" \
        "$MINER_USER@$MINER_HOST:/tmp/miner.toml"
    ssh -p "$MINER_PORT" "$MINER_USER@$MINER_HOST" \
        "sudo mv /tmp/miner.toml /etc/basilica/miner.toml && sudo chown root:basilica /etc/basilica/miner.toml && sudo chmod 640 /etc/basilica/miner.toml"
    
    # Deploy executor config
    log_info "Deploying executor configuration"
    scp -P "$EXECUTOR_PORT" "$OUTPUT_DIR/executor.toml" \
        "$EXECUTOR_USER@$EXECUTOR_HOST:/tmp/executor.toml"
    ssh -p "$EXECUTOR_PORT" "$EXECUTOR_USER@$EXECUTOR_HOST" \
        "sudo mv /tmp/executor.toml /etc/basilica/executor.toml && sudo chown root:basilica /etc/basilica/executor.toml && sudo chmod 640 /etc/basilica/executor.toml"
    
    # Deploy systemd services
    if [[ -d "$OUTPUT_DIR/systemd" ]]; then
        log_info "Deploying systemd service files"
        
        for service_file in "$OUTPUT_DIR/systemd"/*.service; do
            local service_name=$(basename "$service_file")
            local target_host target_port target_user
            
            case "$service_name" in
                basilica-validator.service)
                    target_host="$VALIDATOR_HOST"
                    target_port="$VALIDATOR_PORT"
                    target_user="$VALIDATOR_USER"
                    ;;
                basilica-miner.service)
                    target_host="$MINER_HOST"
                    target_port="$MINER_PORT"
                    target_user="$MINER_USER"
                    ;;
                basilica-executor.service)
                    target_host="$EXECUTOR_HOST"
                    target_port="$EXECUTOR_PORT"
                    target_user="$EXECUTOR_USER"
                    ;;
            esac
            
            scp -P "$target_port" "$service_file" "$target_user@$target_host:/tmp/"
            ssh -p "$target_port" "$target_user@$target_host" \
                "sudo mv /tmp/$service_name /etc/systemd/system/ && sudo systemctl daemon-reload"
        done
    fi
    
    log_success "Configuration deployment completed"
}

# Command: Validate configurations
cmd_validate() {
    log_header "Validating Configurations"
    load_env_config
    
    local errors=0
    
    # Validate config files exist
    for config in validator.toml miner.toml executor.toml; do
        if [[ ! -f "$OUTPUT_DIR/$config" ]]; then
            log_error "Missing configuration file: $config"
            ((errors++))
        else
            log_success "Found configuration: $config"
        fi
    done
    
    # Validate environment variables are set
    local required_vars=("VALIDATOR_HOST" "MINER_HOST" "EXECUTOR_HOST")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Missing required environment variable: $var"
            ((errors++))
        else
            log_success "Environment variable set: $var"
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log_success "All configuration validations passed"
    else
        log_error "Configuration validation failed with $errors errors"
        exit 1
    fi
}

# Command: Generate and distribute cryptographic keys
cmd_keys() {
    log_header "Managing Cryptographic Keys"
    load_env_config
    
    # Generate validator P256 keys if not exists
    local keys_dir="$OUTPUT_DIR/keys"
    mkdir -p "$keys_dir"
    
    if [[ ! -f "$keys_dir/validator_private.pem" ]]; then
        log_info "Generating P256 key pair for validator"
        "$BASILICA_ROOT/scripts/gen-key.sh" "$keys_dir/validator"
        log_success "Generated validator P256 keys"
    fi
    
    # Generate SSH keys for each service
    for service in validator miner executor; do
        local ssh_key="$keys_dir/${service}_ssh"
        if [[ ! -f "$ssh_key" ]]; then
            log_info "Generating SSH key for $service"
            ssh-keygen -t ed25519 -f "$ssh_key" -N "" -C "basilica-$service"
            log_success "Generated SSH key for $service"
        fi
    done
    
    log_success "Key generation completed"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    parse_args "$@"
    
    case "$command" in
        generate)
            cmd_generate
            ;;
        deploy)
            cmd_deploy
            ;;
        validate)
            cmd_validate
            ;;
        keys)
            cmd_keys
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