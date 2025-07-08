#!/bin/bash
set -euo pipefail

SERVICES=""
VALIDATOR_USER=""
VALIDATOR_HOST=""
VALIDATOR_PORT=""
MINER_USER=""
MINER_HOST=""
MINER_PORT=""
EXECUTOR_USER=""
EXECUTOR_HOST=""
EXECUTOR_PORT=""
SYNC_WALLETS=false
FOLLOW_LOGS=false
HEALTH_CHECK=false
TIMEOUT=60

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy Basilica components to remote servers.

OPTIONS:
    -s, --services SERVICES      Comma-separated list: validator,miner,executor or 'all'
    -v, --validator USER@HOST:PORT    Validator server connection
    -m, --miner USER@HOST:PORT        Miner server connection
    -e, --executor USER@HOST:PORT     Executor server connection
    -w, --sync-wallets               Sync local wallets to remote servers
    -f, --follow-logs                Stream logs after deployment
    -c, --health-check               Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -h, --help                       Show this help

EXAMPLES:
    # Deploy all services
    $0 -s all -v root@64.247.196.98:9001 -m root@51.159.160.71:46088 -e shadeform@160.202.129.13:22

    # Deploy only miner with wallet sync
    $0 -s miner -m root@51.159.160.71:46088 -w

    # Deploy validator and miner with health checks
    $0 -s validator,miner -v root@64.247.196.98:9001 -m root@51.159.160.71:46088 -c

    # Deploy all services and follow logs
    $0 -s all -v root@64.247.196.98:9001 -m root@51.159.160.71:46088 -e shadeform@160.202.129.13:22 -f
EOF
    exit 1
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

ssh_cmd() {
    local service="$1"
    local cmd="$2"

    case $service in
        validator)
            timeout "$TIMEOUT" ssh -o ConnectTimeout=30 "$VALIDATOR_USER@$VALIDATOR_HOST" -p "$VALIDATOR_PORT" "$cmd"
            ;;
        miner)
            timeout "$TIMEOUT" ssh -o ConnectTimeout=30 "$MINER_USER@$MINER_HOST" -p "$MINER_PORT" "$cmd"
            ;;
        executor)
            timeout "$TIMEOUT" ssh -o ConnectTimeout=30 "$EXECUTOR_USER@$EXECUTOR_HOST" -p "$EXECUTOR_PORT" "$cmd"
            ;;
    esac
}

scp_file() {
    local service="$1"
    local src="$2"
    local dest="$3"

    case $service in
        validator)
            timeout "$TIMEOUT" scp -o ConnectTimeout=30 -P "$VALIDATOR_PORT" "$src" "$VALIDATOR_USER@$VALIDATOR_HOST:$dest"
            ;;
        miner)
            timeout "$TIMEOUT" scp -o ConnectTimeout=30 -P "$MINER_PORT" "$src" "$MINER_USER@$MINER_HOST:$dest"
            ;;
        executor)
            timeout "$TIMEOUT" scp -o ConnectTimeout=30 -P "$EXECUTOR_PORT" "$src" "$EXECUTOR_USER@$EXECUTOR_HOST:$dest"
            ;;
    esac
}

rsync_wallet() {
    local wallet_name="$1"
    local service="$2"
    local wallet_path="$HOME/.bittensor/wallets/$wallet_name"

    if [[ ! -d "$wallet_path" ]]; then
        log "ERROR: Wallet $wallet_name not found at $wallet_path"
        exit 1
    fi

    log "Syncing wallet $wallet_name to $service server"

    case $service in
        validator)
            ssh_cmd "$service" "mkdir -p ~/.bittensor/wallets/$wallet_name"
            timeout "$TIMEOUT" rsync -avz -e "ssh -p $VALIDATOR_PORT -o ConnectTimeout=30" "$wallet_path/" "$VALIDATOR_USER@$VALIDATOR_HOST:~/.bittensor/wallets/$wallet_name/"
            ;;
        miner)
            ssh_cmd "$service" "mkdir -p ~/.bittensor/wallets/$wallet_name"
            timeout "$TIMEOUT" rsync -avz -e "ssh -p $MINER_PORT -o ConnectTimeout=30" "$wallet_path/" "$MINER_USER@$MINER_HOST:~/.bittensor/wallets/$wallet_name/"
            ;;
    esac

    ssh_cmd "$service" "chmod -R 700 ~/.bittensor/wallets/$wallet_name"
    ssh_cmd "$service" "find ~/.bittensor/wallets/$wallet_name -name '*.json' -exec chmod 600 {} +"
}

build_service() {
    local service="$1"
    log "Building $service..."
    if [[ ! -f "scripts/$service/build.sh" ]]; then
        log "ERROR: Build script scripts/$service/build.sh not found"
        exit 1
    fi
    ./scripts/$service/build.sh
    if [[ ! -f "./$service" ]]; then
        log "ERROR: Binary ./$service not found after build"
        exit 1
    fi
}

deploy_service() {
    local service="$1"
    log "Deploying $service"

    log "Stopping existing $service processes"

    # Use shorter timeout for kill commands to avoid hanging
    case $service in
        validator)
            timeout 10 ssh -o ConnectTimeout=5 "$VALIDATOR_USER@$VALIDATOR_HOST" -p "$VALIDATOR_PORT" "pkill -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect to stop $service processes"
            ;;
        miner)
            timeout 10 ssh -o ConnectTimeout=5 "$MINER_USER@$MINER_HOST" -p "$MINER_PORT" "pkill -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect to stop $service processes"
            ;;
        executor)
            timeout 10 ssh -o ConnectTimeout=5 "$EXECUTOR_USER@$EXECUTOR_HOST" -p "$EXECUTOR_PORT" "pkill -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect to stop $service processes"
            ;;
    esac

    sleep 2

    # Force kill with shorter timeout
    case $service in
        validator)
            timeout 10 ssh -o ConnectTimeout=5 "$VALIDATOR_USER@$VALIDATOR_HOST" -p "$VALIDATOR_PORT" "pkill -9 -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"
            ;;
        miner)
            timeout 10 ssh -o ConnectTimeout=5 "$MINER_USER@$MINER_HOST" -p "$MINER_PORT" "pkill -9 -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"
            ;;
        executor)
            timeout 10 ssh -o ConnectTimeout=5 "$EXECUTOR_USER@$EXECUTOR_HOST" -p "$EXECUTOR_PORT" "pkill -9 -f '/opt/basilica/$service' 2>/dev/null || true" || log "WARNING: Could not connect for force kill"
            ;;
    esac

    sleep 3

    log "Removing old $service files"
    ssh_cmd "$service" "cp /opt/basilica/$service /opt/basilica/$service.backup 2>/dev/null || true"

    # Try to move the current binary out of the way to avoid "Text file busy"
    ssh_cmd "$service" "mv /opt/basilica/$service /opt/basilica/$service.old 2>/dev/null || true"

    scp_file "$service" "$service" "/opt/basilica/"
    ssh_cmd "$service" "chmod +x /opt/basilica/$service"

    log "Creating directories for $service"
    ssh_cmd "$service" "mkdir -p /opt/basilica/config"
    scp_file "$service" "config/$service.correct.toml" "/opt/basilica/config/$service.toml"

    ssh_cmd "$service" "mkdir -p /opt/basilica/data && chmod 755 /opt/basilica/data"

    log "Setting up data directories and permissions for $service"
    case $service in
        miner)
            ssh_cmd "$service" "touch /opt/basilica/data/miner.db && chmod 644 /opt/basilica/data/miner.db"
            ssh_cmd "$service" "[ ! -f /root/.ssh/miner_executor_key ] && ssh-keygen -t ed25519 -f /root/.ssh/miner_executor_key -N '' || true"
            ;;
    esac

    local start_cmd="cd /opt/basilica && RUST_LOG=debug nohup"
    case $service in
        validator)
            start_cmd="$start_cmd ./validator start --config config/validator.toml > validator.log 2>&1 &"
            ;;
        miner)
            start_cmd="$start_cmd ./miner --config config/miner.toml > miner.log 2>&1 &"
            ;;
        executor)
            # CRITICAL: Executor requires sudo/root permissions for container management and system access
            start_cmd="$start_cmd sudo ./executor --server --config config/executor.toml > executor.log 2>&1 &"
            ;;
    esac

    log "Starting $service"
    case $service in
        validator)
            timeout 15 ssh -o ConnectTimeout=5 "$VALIDATOR_USER@$VALIDATOR_HOST" -p "$VALIDATOR_PORT" "$start_cmd" || true
            ;;
        miner)
            timeout 15 ssh -o ConnectTimeout=5 "$MINER_USER@$MINER_HOST" -p "$MINER_PORT" "$start_cmd" || true
            ;;
        executor)
            # IMPORTANT: Executor must be started with sudo for proper permissions
            timeout 15 ssh -o ConnectTimeout=5 "$EXECUTOR_USER@$EXECUTOR_HOST" -p "$EXECUTOR_PORT" "$start_cmd" || true
            ;;
    esac

    sleep 5
    if ssh_cmd "$service" "pgrep -f $service > /dev/null"; then
        log "$service started successfully"

        # Special verification for executor requiring root permissions
        if [[ "$service" == "executor" ]]; then
            log "Verifying executor is running with proper permissions..."
            if ssh_cmd "$service" "ps aux | grep -v grep | grep 'root.*executor'" > /dev/null; then
                log "Executor confirmed running with root permissions"
            else
                log "WARNING: Executor may not be running with root permissions - check manually"
            fi

            # Verify executor is listening on gRPC port
            if ssh_cmd "$service" "ss -tlnp | grep :50051" > /dev/null; then
                log "Executor gRPC server listening on port 50051"
            else
                log "WARNING: Executor gRPC port 50051 not found - service may need restart"
            fi
        fi
    else
        log "ERROR: $service failed to start"
        ssh_cmd "$service" "tail -10 /opt/basilica/$service.log"
        exit 1
    fi
}

health_check_service() {
    local service="$1"
    if ssh_cmd "$service" "pgrep -f $service > /dev/null"; then
        local port=$(ssh_cmd "$service" "grep '^port = ' /opt/basilica/config/$service.toml | cut -d' ' -f3")
        log "$service running (port $port)"
    else
        log "$service not running"
    fi
}

follow_logs_service() {
    local service="$1"
    log "Following logs for $service"
    ssh_cmd "$service" "tail -f /opt/basilica/$service.log"
}

setup_ssh_access() {
    if [[ -z "$MINER_HOST" || -z "$EXECUTOR_HOST" ]]; then
        return
    fi

    log "Setting up SSH access from miner to executor"
    MINER_PUBKEY=$(ssh_cmd miner "cat /root/.ssh/miner_executor_key.pub")
    ssh_cmd executor "grep -qF '$MINER_PUBKEY' ~/.ssh/authorized_keys || echo '$MINER_PUBKEY' >> ~/.ssh/authorized_keys"

    log "Testing SSH connectivity"
    ssh_cmd miner "ssh -i /root/.ssh/miner_executor_key -o StrictHostKeyChecking=no -o ConnectTimeout=10 $EXECUTOR_USER@$EXECUTOR_HOST 'echo SSH access verified'"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--services)
            SERVICES="$2"
            shift 2
            ;;
        -v|--validator)
            IFS='@' read -r VALIDATOR_USER temp <<< "$2"
            IFS=':' read -r VALIDATOR_HOST VALIDATOR_PORT <<< "$temp"
            shift 2
            ;;
        -m|--miner)
            IFS='@' read -r MINER_USER temp <<< "$2"
            IFS=':' read -r MINER_HOST MINER_PORT <<< "$temp"
            shift 2
            ;;
        -e|--executor)
            IFS='@' read -r EXECUTOR_USER temp <<< "$2"
            IFS=':' read -r EXECUTOR_HOST EXECUTOR_PORT <<< "$temp"
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
        -c|--health-check)
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

if [[ -z "$SERVICES" ]]; then
    echo "ERROR: Services parameter required (-s)"
    usage
fi

if [[ "$SERVICES" == "all" ]]; then
    SERVICES="validator,miner,executor"
fi

IFS=',' read -ra SERVICE_LIST <<< "$SERVICES"

for service in "${SERVICE_LIST[@]}"; do
    case $service in
        validator)
            if [[ -z "$VALIDATOR_USER" || -z "$VALIDATOR_HOST" || -z "$VALIDATOR_PORT" ]]; then
                echo "ERROR: Validator connection required for validator deployment (-v)"
                exit 1
            fi
            ;;
        miner)
            if [[ -z "$MINER_USER" || -z "$MINER_HOST" || -z "$MINER_PORT" ]]; then
                echo "ERROR: Miner connection required for miner deployment (-m)"
                exit 1
            fi
            ;;
        executor)
            if [[ -z "$EXECUTOR_USER" || -z "$EXECUTOR_HOST" || -z "$EXECUTOR_PORT" ]]; then
                echo "ERROR: Executor connection required for executor deployment (-e)"
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown service: $service"
            exit 1
            ;;
    esac
done

log "Building services: ${SERVICE_LIST[*]}"
for service in "${SERVICE_LIST[@]}"; do
    build_service "$service"
done

if [[ "$SYNC_WALLETS" == "true" ]]; then
    log "Syncing wallets to remote servers"
    for service in "${SERVICE_LIST[@]}"; do
        case $service in
            validator)
                rsync_wallet "test_validator" "$service"
                ;;
            miner)
                rsync_wallet "test_miner" "$service"
                ;;
        esac
    done
fi

for service in "${SERVICE_LIST[@]}"; do
    deploy_service "$service"
done

if [[ " ${SERVICE_LIST[*]} " =~ " miner " ]] && [[ " ${SERVICE_LIST[*]} " =~ " executor " ]]; then
    setup_ssh_access
fi

if [[ "$HEALTH_CHECK" == "true" ]]; then
    log "Running health checks on deployed services"
    for service in "${SERVICE_LIST[@]}"; do
        health_check_service "$service"
    done
fi

log "Deployment completed successfully"

if [[ "$FOLLOW_LOGS" == "true" ]]; then
    if [[ ${#SERVICE_LIST[@]} -eq 1 ]]; then
        follow_logs_service "${SERVICE_LIST[0]}"
    else
        log "Multiple services deployed. Choose service to follow:"
        select service in "${SERVICE_LIST[@]}" "Exit"; do
            case $service in
                "Exit")
                    break
                    ;;
                *)
                    if [[ -n "$service" ]]; then
                        follow_logs_service "$service"
                        break
                    fi
                    ;;
            esac
        done
    fi
fi
