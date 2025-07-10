#!/bin/bash
set -euo pipefail

SERVICE="validator"
SERVER_USER=""
SERVER_HOST=""
SERVER_PORT=""
DEPLOY_MODE="binary"
CONFIG_FILE="config/validator.correct.toml"
SYNC_WALLETS=false
FOLLOW_LOGS=false
HEALTH_CHECK=false
TIMEOUT=60
VERITAS_BINARIES_DIR=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy Basilica validator component to remote server.

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/validator.correct.toml)
    -w, --sync-wallets               Sync local wallets to remote server
    -f, --follow-logs                Stream logs after deployment
    --health-check                   Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -b, --veritas-binaries DIR       Directory containing veritas binaries to deploy
    -h, --help                       Show this help

DEPLOYMENT MODES:
    binary   - Deploy binary with nohup (default)
    systemd  - Deploy binary with systemd service management
    docker   - Deploy using docker compose with public images

EXAMPLES:
    # Deploy validator with binary mode (default)
    $0 -s root@64.247.196.98:9001

    # Deploy validator with systemd
    $0 -s root@64.247.196.98:9001 -d systemd

    # Deploy validator with docker
    $0 -s root@64.247.196.98:9001 -d docker

    # Deploy with custom config
    $0 -s root@64.247.196.98:9001 -c config/validator.prod.toml

    # Deploy with wallet sync and health checks
    $0 -s root@64.247.196.98:9001 -w --health-check

    # Deploy with veritas binaries
    $0 -s root@64.247.196.98:9001 -b ../veritas/binaries
EOF
    exit 1
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

ssh_cmd() {
    local cmd="$1"
    timeout "$TIMEOUT" ssh -o ConnectTimeout=30 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "$cmd"
}

scp_file() {
    local src="$1"
    local dest="$2"
    timeout "$TIMEOUT" scp -o ConnectTimeout=30 -P "$SERVER_PORT" "$src" "$SERVER_USER@$SERVER_HOST:$dest"
}

rsync_wallet() {
    local wallet_name="$1"
    local wallet_path="$HOME/.bittensor/wallets/$wallet_name"

    if [[ ! -d "$wallet_path" ]]; then
        log "ERROR: Wallet $wallet_name not found at $wallet_path"
        exit 1
    fi

    log "Syncing wallet $wallet_name to validator server"
    ssh_cmd "mkdir -p ~/.bittensor/wallets/$wallet_name"
    timeout "$TIMEOUT" rsync -avz -e "ssh -p $SERVER_PORT -o ConnectTimeout=30" "$wallet_path/" "$SERVER_USER@$SERVER_HOST:~/.bittensor/wallets/$wallet_name/"
    ssh_cmd "chmod -R 700 ~/.bittensor/wallets/$wallet_name"
    ssh_cmd "find ~/.bittensor/wallets/$wallet_name -name '*.json' -exec chmod 600 {} +"
}

validate_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    log "Using config file: $CONFIG_FILE"
}

build_service() {
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        log "Docker mode: skipping local build"
        return
    fi
    
    log "Building validator..."
    if [[ ! -f "scripts/validator/build.sh" ]]; then
        log "ERROR: Build script scripts/validator/build.sh not found"
        exit 1
    fi
    
    # Build with veritas binaries if specified
    if [[ -n "$VERITAS_BINARIES_DIR" ]]; then
        ./scripts/validator/build.sh --veritas-binaries "$VERITAS_BINARIES_DIR"
    else
        ./scripts/validator/build.sh
    fi
    
    if [[ ! -f "./validator" ]]; then
        log "ERROR: Binary ./validator not found after build"
        exit 1
    fi
}

deploy_veritas_binaries() {
    if [[ -z "$VERITAS_BINARIES_DIR" ]]; then
        return
    fi
    
    if [[ ! -d "$VERITAS_BINARIES_DIR" ]]; then
        log "ERROR: Veritas binaries directory does not exist: $VERITAS_BINARIES_DIR"
        exit 1
    fi
    
    local executor_binary="$VERITAS_BINARIES_DIR/executor-binary/executor-binary"
    local validator_binary="$VERITAS_BINARIES_DIR/validator-binary/validator-binary"
    
    if [[ ! -f "$executor_binary" ]]; then
        log "ERROR: executor-binary not found at: $executor_binary"
        exit 1
    fi
    
    if [[ ! -f "$validator_binary" ]]; then
        log "ERROR: validator-binary not found at: $validator_binary"
        exit 1
    fi
    
    log "Deploying veritas binaries to validator"
    ssh_cmd "mkdir -p /opt/basilica/bin"
    
    scp_file "$executor_binary" "/opt/basilica/bin/executor-binary"
    scp_file "$validator_binary" "/opt/basilica/bin/validator-binary"
    
    ssh_cmd "chmod +x /opt/basilica/bin/executor-binary"
    ssh_cmd "chmod +x /opt/basilica/bin/validator-binary"
    
    log "Veritas binaries deployed successfully"
}

deploy_binary() {
    log "Deploying validator in binary mode"

    log "Stopping existing validator processes"
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -f '/opt/basilica/validator' 2>/dev/null || true" || log "WARNING: Could not connect to stop validator processes"

    sleep 2

    # Force kill with shorter timeout
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -9 -f '/opt/basilica/validator' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"

    sleep 3

    log "Removing old validator files"
    ssh_cmd "cp /opt/basilica/validator /opt/basilica/validator.backup 2>/dev/null || true"

    # Try to move the current binary out of the way to avoid "Text file busy"
    ssh_cmd "mv /opt/basilica/validator /opt/basilica/validator.old 2>/dev/null || true"

    scp_file "validator" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/validator"

    log "Creating directories for validator"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/validator.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    # Deploy veritas binaries if specified
    deploy_veritas_binaries

    local start_cmd="cd /opt/basilica && RUST_LOG=debug nohup ./validator start --config config/validator.toml > validator.log 2>&1 &"

    log "Starting validator"
    timeout 15 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "$start_cmd" || true

    sleep 5
    if ssh_cmd "pgrep -f validator > /dev/null"; then
        log "Validator started successfully"
    else
        log "ERROR: Validator failed to start"
        ssh_cmd "tail -10 /opt/basilica/validator.log"
        exit 1
    fi
}

deploy_systemd() {
    log "Deploying validator in systemd mode"

    log "Stopping existing validator service"
    ssh_cmd "systemctl stop basilica-validator 2>/dev/null || true"

    log "Removing old validator files"
    ssh_cmd "cp /opt/basilica/validator /opt/basilica/validator.backup 2>/dev/null || true"
    ssh_cmd "mv /opt/basilica/validator /opt/basilica/validator.old 2>/dev/null || true"

    scp_file "validator" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/validator"

    log "Creating directories for validator"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/validator.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    # Deploy veritas binaries if specified
    deploy_veritas_binaries

    log "Installing systemd service"
    if [[ ! -f "scripts/validator/systemd/basilica-validator.service" ]]; then
        log "ERROR: Systemd service file not found: scripts/validator/systemd/basilica-validator.service"
        exit 1
    fi
    
    scp_file "scripts/validator/systemd/basilica-validator.service" "/etc/systemd/system/"
    ssh_cmd "systemctl daemon-reload"
    ssh_cmd "systemctl enable basilica-validator"

    log "Starting validator service"
    ssh_cmd "systemctl start basilica-validator"

    sleep 5
    if ssh_cmd "systemctl is-active basilica-validator --quiet"; then
        log "Validator service started successfully"
    else
        log "ERROR: Validator service failed to start"
        ssh_cmd "journalctl -u basilica-validator --lines=20 --no-pager"
        exit 1
    fi
}

deploy_docker() {
    log "Deploying validator in docker mode"

    log "Stopping existing validator containers"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml down 2>/dev/null || true"

    log "Creating directories for validator"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/validator.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"

    log "Deploying docker compose files"
    if [[ ! -f "scripts/validator/compose.prod.yml" ]]; then
        log "ERROR: Docker compose file not found: scripts/validator/compose.prod.yml"
        exit 1
    fi
    
    scp_file "scripts/validator/compose.prod.yml" "/opt/basilica/"
    
    # Deploy .env file if it exists
    if [[ -f "scripts/validator/.env" ]]; then
        scp_file "scripts/validator/.env" "/opt/basilica/"
    fi

    log "Pulling and starting validator container"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml pull"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml up -d"

    sleep 5
    if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
        log "Validator container started successfully"
    else
        log "ERROR: Validator container failed to start"
        ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml logs --tail=20"
        exit 1
    fi
}

deploy_service() {
    case "$DEPLOY_MODE" in
        binary)
            deploy_binary
            ;;
        systemd)
            deploy_systemd
            ;;
        docker)
            deploy_docker
            ;;
        *)
            log "ERROR: Unknown deployment mode: $DEPLOY_MODE"
            exit 1
            ;;
    esac
}

health_check_service() {
    case "$DEPLOY_MODE" in
        binary)
            if ssh_cmd "pgrep -f validator > /dev/null"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/validator.toml | cut -d' ' -f3")
                log "Validator running (port $port)"
            else
                log "Validator not running"
            fi
            ;;
        systemd)
            if ssh_cmd "systemctl is-active basilica-validator --quiet"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/validator.toml | cut -d' ' -f3")
                log "Validator service active (port $port)"
            else
                log "Validator service not active"
            fi
            ;;
        docker)
            if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
                log "Validator container running"
            else
                log "Validator container not running"
            fi
            ;;
    esac
}

follow_logs_service() {
    log "Following logs for validator"
    case "$DEPLOY_MODE" in
        binary)
            ssh_cmd "tail -f /opt/basilica/validator.log"
            ;;
        systemd)
            ssh_cmd "journalctl -u basilica-validator -f"
            ;;
        docker)
            ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml logs -f"
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server)
            IFS='@' read -r SERVER_USER temp <<< "$2"
            IFS=':' read -r SERVER_HOST SERVER_PORT <<< "$temp"
            shift 2
            ;;
        -d|--deploy-mode)
            DEPLOY_MODE="$2"
            if [[ "$DEPLOY_MODE" != "binary" && "$DEPLOY_MODE" != "systemd" && "$DEPLOY_MODE" != "docker" ]]; then
                echo "ERROR: Invalid deployment mode: $DEPLOY_MODE. Must be binary, systemd, or docker"
                exit 1
            fi
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -w|--sync-wallets)
            SYNC_WALLETS=true
            shift
            ;;
        -f|--follow-logs)
            FOLLOW_LOGS=true
            shift
            ;;
        --health-check)
            HEALTH_CHECK=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -b|--veritas-binaries)
            VERITAS_BINARIES_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$SERVER_USER" || -z "$SERVER_HOST" || -z "$SERVER_PORT" ]]; then
    echo "ERROR: Server connection required (-s)"
    usage
fi

# Validate deployment mode specific requirements
if [[ "$DEPLOY_MODE" == "docker" && "$SYNC_WALLETS" == "true" ]]; then
    log "WARNING: Wallet sync not supported in docker mode"
    SYNC_WALLETS=false
fi

if [[ "$DEPLOY_MODE" == "docker" && -n "$VERITAS_BINARIES_DIR" ]]; then
    log "WARNING: Veritas binaries deployment not supported in docker mode"
    VERITAS_BINARIES_DIR=""
fi

log "Deployment mode: $DEPLOY_MODE"
validate_config

log "Building validator"
build_service

if [[ "$SYNC_WALLETS" == "true" ]]; then
    log "Syncing wallets to validator server"
    rsync_wallet "test_validator"
fi

deploy_service

if [[ "$HEALTH_CHECK" == "true" ]]; then
    log "Running health check on validator"
    health_check_service
fi

log "Deployment completed successfully"

if [[ "$FOLLOW_LOGS" == "true" ]]; then
    follow_logs_service
fi
