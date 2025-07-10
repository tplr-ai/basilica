#!/bin/bash
set -euo pipefail

SERVICE="executor"
SERVER_USER=""
SERVER_HOST=""
SERVER_PORT=""
DEPLOY_MODE="binary"
CONFIG_FILE="config/executor.correct.toml"
FOLLOW_LOGS=false
HEALTH_CHECK=false
TIMEOUT=60
VERITAS_BINARIES_DIR=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy Basilica executor component to remote server.

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/executor.correct.toml)
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
    # Deploy executor with binary mode (default)
    $0 -s shadeform@160.202.129.13:22

    # Deploy executor with systemd
    $0 -s shadeform@160.202.129.13:22 -d systemd

    # Deploy executor with docker
    $0 -s shadeform@160.202.129.13:22 -d docker

    # Deploy with custom config
    $0 -s shadeform@160.202.129.13:22 -c config/executor.prod.toml

    # Deploy with health checks and follow logs
    $0 -s shadeform@160.202.129.13:22 --health-check -f

    # Deploy with veritas binaries
    $0 -s shadeform@160.202.129.13:22 -b ../veritas/binaries
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
    
    log "Building executor..."
    if [[ ! -f "scripts/executor/build.sh" ]]; then
        log "ERROR: Build script scripts/executor/build.sh not found"
        exit 1
    fi
    
    ./scripts/executor/build.sh
    
    if [[ ! -f "./executor" ]]; then
        log "ERROR: Binary ./executor not found after build"
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
    
    log "Deploying veritas binaries to executor"
    ssh_cmd "mkdir -p /opt/basilica/bin"
    
    scp_file "$executor_binary" "/opt/basilica/bin/executor-binary"
    scp_file "$validator_binary" "/opt/basilica/bin/validator-binary"
    
    ssh_cmd "chmod +x /opt/basilica/bin/executor-binary"
    ssh_cmd "chmod +x /opt/basilica/bin/validator-binary"
    
    log "Veritas binaries deployed successfully"
}

deploy_binary() {
    log "Deploying executor in binary mode"

    log "Stopping existing executor processes"
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -f '/opt/basilica/executor' 2>/dev/null || true" || log "WARNING: Could not connect to stop executor processes"

    sleep 2

    # Force kill with shorter timeout
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -9 -f '/opt/basilica/executor' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"

    sleep 3

    log "Removing old executor files"
    ssh_cmd "cp /opt/basilica/executor /opt/basilica/executor.backup 2>/dev/null || true"

    # Try to move the current binary out of the way to avoid "Text file busy"
    ssh_cmd "mv /opt/basilica/executor /opt/basilica/executor.old 2>/dev/null || true"

    scp_file "executor" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/executor"

    log "Creating directories for executor"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/executor.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    # Deploy veritas binaries if specified
    deploy_veritas_binaries

    # CRITICAL: Executor requires sudo/root permissions for container management and system access
    local start_cmd="cd /opt/basilica && RUST_LOG=debug nohup sudo ./executor --server --config config/executor.toml > executor.log 2>&1 &"

    log "Starting executor"
    # IMPORTANT: Executor must be started with sudo for proper permissions
    timeout 15 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "$start_cmd" || true

    sleep 5
    if ssh_cmd "pgrep -f executor > /dev/null"; then
        log "Executor started successfully"

        # Special verification for executor requiring root permissions
        log "Verifying executor is running with proper permissions..."
        if ssh_cmd "ps aux | grep -v grep | grep 'root.*executor'" > /dev/null; then
            log "Executor confirmed running with root permissions"
        else
            log "WARNING: Executor may not be running with root permissions - check manually"
        fi

        # Verify executor is listening on gRPC port
        if ssh_cmd "ss -tlnp | grep :50051" > /dev/null; then
            log "Executor gRPC server listening on port 50051"
        else
            log "WARNING: Executor gRPC port 50051 not found - service may need restart"
        fi
    else
        log "ERROR: Executor failed to start"
        ssh_cmd "tail -10 /opt/basilica/executor.log"
        exit 1
    fi
}

deploy_systemd() {
    log "Deploying executor in systemd mode"

    log "Stopping existing executor service"
    ssh_cmd "systemctl stop basilica-executor 2>/dev/null || true"

    log "Removing old executor files"
    ssh_cmd "cp /opt/basilica/executor /opt/basilica/executor.backup 2>/dev/null || true"
    ssh_cmd "mv /opt/basilica/executor /opt/basilica/executor.old 2>/dev/null || true"

    scp_file "executor" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/executor"

    log "Creating directories for executor"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/executor.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    # Deploy veritas binaries if specified
    deploy_veritas_binaries

    log "Installing systemd service"
    if [[ ! -f "scripts/executor/systemd/basilica-executor.service" ]]; then
        log "ERROR: Systemd service file not found: scripts/executor/systemd/basilica-executor.service"
        exit 1
    fi
    
    scp_file "scripts/executor/systemd/basilica-executor.service" "/etc/systemd/system/"
    ssh_cmd "systemctl daemon-reload"
    ssh_cmd "systemctl enable basilica-executor"

    log "Starting executor service"
    ssh_cmd "systemctl start basilica-executor"

    sleep 5
    if ssh_cmd "systemctl is-active basilica-executor --quiet"; then
        log "Executor service started successfully"
        
        # Verify executor is listening on gRPC port
        if ssh_cmd "ss -tlnp | grep :50051" > /dev/null; then
            log "Executor gRPC server listening on port 50051"
        else
            log "WARNING: Executor gRPC port 50051 not found - service may need restart"
        fi
    else
        log "ERROR: Executor service failed to start"
        ssh_cmd "journalctl -u basilica-executor --lines=20 --no-pager"
        exit 1
    fi
}

deploy_docker() {
    log "Deploying executor in docker mode"

    log "Stopping existing executor containers"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml down 2>/dev/null || true"

    log "Creating directories for executor"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/executor.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"

    log "Deploying docker compose files"
    if [[ ! -f "scripts/executor/compose.prod.yml" ]]; then
        log "ERROR: Docker compose file not found: scripts/executor/compose.prod.yml"
        exit 1
    fi
    
    scp_file "scripts/executor/compose.prod.yml" "/opt/basilica/"
    
    # Deploy .env file if it exists
    if [[ -f "scripts/executor/.env" ]]; then
        scp_file "scripts/executor/.env" "/opt/basilica/"
    fi

    log "Pulling and starting executor container"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml pull"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml up -d"

    sleep 5
    if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
        log "Executor container started successfully"
    else
        log "ERROR: Executor container failed to start"
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
            if ssh_cmd "pgrep -f executor > /dev/null"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/executor.toml | cut -d' ' -f3")
                log "Executor running (port $port)"
            else
                log "Executor not running"
            fi
            ;;
        systemd)
            if ssh_cmd "systemctl is-active basilica-executor --quiet"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/executor.toml | cut -d' ' -f3")
                log "Executor service active (port $port)"
            else
                log "Executor service not active"
            fi
            ;;
        docker)
            if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
                log "Executor container running"
            else
                log "Executor container not running"
            fi
            ;;
    esac
}

follow_logs_service() {
    log "Following logs for executor"
    case "$DEPLOY_MODE" in
        binary)
            ssh_cmd "tail -f /opt/basilica/executor.log"
            ;;
        systemd)
            ssh_cmd "journalctl -u basilica-executor -f"
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
if [[ "$DEPLOY_MODE" == "docker" && -n "$VERITAS_BINARIES_DIR" ]]; then
    log "WARNING: Veritas binaries deployment not supported in docker mode"
    VERITAS_BINARIES_DIR=""
fi

log "Deployment mode: $DEPLOY_MODE"
validate_config

log "Building executor"
build_service

deploy_service

if [[ "$HEALTH_CHECK" == "true" ]]; then
    log "Running health check on executor"
    health_check_service
fi

log "Deployment completed successfully"

if [[ "$FOLLOW_LOGS" == "true" ]]; then
    follow_logs_service
fi
