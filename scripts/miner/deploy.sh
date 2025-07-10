#!/bin/bash
set -euo pipefail

SERVICE="miner"
SERVER_USER=""
SERVER_HOST=""
SERVER_PORT=""
DEPLOY_MODE="binary"
CONFIG_FILE="config/miner.correct.toml"
SYNC_WALLETS=false
FOLLOW_LOGS=false
HEALTH_CHECK=false
TIMEOUT=60

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy Basilica miner component to remote server.

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/miner.correct.toml)
    -w, --sync-wallets               Sync local wallets to remote server
    -f, --follow-logs                Stream logs after deployment
    --health-check                   Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -h, --help                       Show this help

DEPLOYMENT MODES:
    binary   - Deploy binary with nohup (default)
    systemd  - Deploy binary with systemd service management
    docker   - Deploy using docker compose with public images

EXAMPLES:
    # Deploy miner with binary mode (default)
    $0 -s root@51.159.160.71:46088

    # Deploy miner with systemd
    $0 -s root@51.159.160.71:46088 -d systemd

    # Deploy miner with docker
    $0 -s root@51.159.160.71:46088 -d docker

    # Deploy with custom config
    $0 -s root@51.159.160.71:46088 -c config/miner.prod.toml

    # Deploy with wallet sync and health checks
    $0 -s root@51.159.160.71:46088 -w --health-check
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

    log "Syncing wallet $wallet_name to miner server"
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
    
    log "Building miner..."
    if [[ ! -f "scripts/miner/build.sh" ]]; then
        log "ERROR: Build script scripts/miner/build.sh not found"
        exit 1
    fi
    
    ./scripts/miner/build.sh
    
    if [[ ! -f "./miner" ]]; then
        log "ERROR: Binary ./miner not found after build"
        exit 1
    fi
}

deploy_binary() {
    log "Deploying miner in binary mode"

    log "Stopping existing miner processes"
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -f '/opt/basilica/miner' 2>/dev/null || true" || log "WARNING: Could not connect to stop miner processes"

    sleep 2

    # Force kill with shorter timeout
    timeout 10 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "pkill -9 -f '/opt/basilica/miner' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"

    sleep 3

    log "Removing old miner files"
    ssh_cmd "cp /opt/basilica/miner /opt/basilica/miner.backup 2>/dev/null || true"

    # Try to move the current binary out of the way to avoid "Text file busy"
    ssh_cmd "mv /opt/basilica/miner /opt/basilica/miner.old 2>/dev/null || true"

    scp_file "miner" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/miner"

    log "Creating directories for miner"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/miner.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    log "Setting up data directories and permissions for miner"
    ssh_cmd "touch /opt/basilica/data/miner.db && chmod 644 /opt/basilica/data/miner.db"
    ssh_cmd "[ ! -f /root/.ssh/miner_executor_key ] && ssh-keygen -t ed25519 -f /root/.ssh/miner_executor_key -N '' || true"

    local start_cmd="cd /opt/basilica && RUST_LOG=debug nohup ./miner --config config/miner.toml > miner.log 2>&1 &"

    log "Starting miner"
    timeout 15 ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" -p "$SERVER_PORT" "$start_cmd" || true

    sleep 5
    if ssh_cmd "pgrep -f miner > /dev/null"; then
        log "Miner started successfully"
    else
        log "ERROR: Miner failed to start"
        ssh_cmd "tail -10 /opt/basilica/miner.log"
        exit 1
    fi
}

deploy_systemd() {
    log "Deploying miner in systemd mode"

    log "Stopping existing miner service"
    ssh_cmd "systemctl stop basilica-miner 2>/dev/null || true"

    log "Removing old miner files"
    ssh_cmd "cp /opt/basilica/miner /opt/basilica/miner.backup 2>/dev/null || true"
    ssh_cmd "mv /opt/basilica/miner /opt/basilica/miner.old 2>/dev/null || true"

    scp_file "miner" "/opt/basilica/"
    ssh_cmd "chmod +x /opt/basilica/miner"

    log "Creating directories for miner"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/miner.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"
    
    log "Setting up data directories and permissions for miner"
    ssh_cmd "touch /opt/basilica/data/miner.db && chmod 644 /opt/basilica/data/miner.db"
    ssh_cmd "[ ! -f /root/.ssh/miner_executor_key ] && ssh-keygen -t ed25519 -f /root/.ssh/miner_executor_key -N '' || true"

    log "Installing systemd service"
    if [[ ! -f "scripts/miner/systemd/basilica-miner.service" ]]; then
        log "ERROR: Systemd service file not found: scripts/miner/systemd/basilica-miner.service"
        exit 1
    fi
    
    scp_file "scripts/miner/systemd/basilica-miner.service" "/etc/systemd/system/"
    ssh_cmd "systemctl daemon-reload"
    ssh_cmd "systemctl enable basilica-miner"

    log "Starting miner service"
    ssh_cmd "systemctl start basilica-miner"

    sleep 5
    if ssh_cmd "systemctl is-active basilica-miner --quiet"; then
        log "Miner service started successfully"
    else
        log "ERROR: Miner service failed to start"
        ssh_cmd "journalctl -u basilica-miner --lines=20 --no-pager"
        exit 1
    fi
}

deploy_docker() {
    log "Deploying miner in docker mode"

    log "Stopping existing miner containers"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml down 2>/dev/null || true"

    log "Creating directories for miner"
    ssh_cmd "mkdir -p /opt/basilica/config"
    scp_file "$CONFIG_FILE" "/opt/basilica/config/miner.toml"

    ssh_cmd "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"

    log "Deploying docker compose files"
    if [[ ! -f "scripts/miner/compose.prod.yml" ]]; then
        log "ERROR: Docker compose file not found: scripts/miner/compose.prod.yml"
        exit 1
    fi
    
    scp_file "scripts/miner/compose.prod.yml" "/opt/basilica/"
    
    # Deploy .env file if it exists
    if [[ -f "scripts/miner/.env" ]]; then
        scp_file "scripts/miner/.env" "/opt/basilica/"
    fi

    log "Pulling and starting miner container"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml pull"
    ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml up -d"

    sleep 5
    if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
        log "Miner container started successfully"
    else
        log "ERROR: Miner container failed to start"
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
            if ssh_cmd "pgrep -f miner > /dev/null"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/miner.toml | cut -d' ' -f3")
                log "Miner running (port $port)"
            else
                log "Miner not running"
            fi
            ;;
        systemd)
            if ssh_cmd "systemctl is-active basilica-miner --quiet"; then
                local port=$(ssh_cmd "grep '^port = ' /opt/basilica/config/miner.toml | cut -d' ' -f3")
                log "Miner service active (port $port)"
            else
                log "Miner service not active"
            fi
            ;;
        docker)
            if ssh_cmd "cd /opt/basilica && docker-compose -f compose.prod.yml ps | grep -q 'Up'"; then
                log "Miner container running"
            else
                log "Miner container not running"
            fi
            ;;
    esac
}

follow_logs_service() {
    log "Following logs for miner"
    case "$DEPLOY_MODE" in
        binary)
            ssh_cmd "tail -f /opt/basilica/miner.log"
            ;;
        systemd)
            ssh_cmd "journalctl -u basilica-miner -f"
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

log "Deployment mode: $DEPLOY_MODE"
validate_config

log "Building miner"
build_service

if [[ "$SYNC_WALLETS" == "true" ]]; then
    log "Syncing wallets to miner server"
    rsync_wallet "test_miner"
fi

deploy_service

if [[ "$HEALTH_CHECK" == "true" ]]; then
    log "Running health check on miner"
    health_check_service
fi

log "Deployment completed successfully"

if [[ "$FOLLOW_LOGS" == "true" ]]; then
    follow_logs_service
fi
