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
VERITAS_BINARIES_DIR=""

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
    -b, --veritas-binaries DIR       Directory containing veritas binaries to deploy
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
    
    # Deploy validator with veritas binaries
    $0 -s validator -v root@64.247.196.98:9001 -b ../veritas/binaries
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

deploy_service() {
    local service="$1"
    log "Deploying $service using service-specific script"
    
    local deploy_script="./scripts/$service/deploy.sh"
    if [[ ! -f "$deploy_script" ]]; then
        log "ERROR: Deploy script $deploy_script not found"
        exit 1
    fi
    
    local server_arg=""
    local deploy_args=()
    
    case $service in
        validator)
            server_arg="$VALIDATOR_USER@$VALIDATOR_HOST:$VALIDATOR_PORT"
            ;;
        miner)
            server_arg="$MINER_USER@$MINER_HOST:$MINER_PORT"
            ;;
        executor)
            server_arg="$EXECUTOR_USER@$EXECUTOR_HOST:$EXECUTOR_PORT"
            ;;
    esac
    
    deploy_args+=("-s" "$server_arg")
    deploy_args+=("-t" "$TIMEOUT")
    
    if [[ "$SYNC_WALLETS" == "true" && ("$service" == "validator" || "$service" == "miner") ]]; then
        deploy_args+=("-w")
    fi
    
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        deploy_args+=("-f")
    fi
    
    if [[ "$HEALTH_CHECK" == "true" ]]; then
        deploy_args+=("--health-check")
    fi
    
    if [[ -n "$VERITAS_BINARIES_DIR" && ("$service" == "validator" || "$service" == "executor") ]]; then
        deploy_args+=("-b" "$VERITAS_BINARIES_DIR")
    fi
    
    log "Executing: $deploy_script ${deploy_args[*]}"
    "$deploy_script" "${deploy_args[@]}"
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
    if [[ ! -f "scripts/$service/build.sh" ]]; then
        log "ERROR: Build script scripts/$service/build.sh not found"
        exit 1
    fi
    log "Building $service..."
    ./scripts/$service/build.sh
done

for service in "${SERVICE_LIST[@]}"; do
    deploy_service "$service"
done

if [[ " ${SERVICE_LIST[*]} " =~ " miner " ]] && [[ " ${SERVICE_LIST[*]} " =~ " executor " ]]; then
    setup_ssh_access
fi


log "Deployment completed successfully"
