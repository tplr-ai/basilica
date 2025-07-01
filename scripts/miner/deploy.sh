#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [port]"
    echo "Example: $0 root@192.168.1.10 22"
    exit 1
fi

MINER_HOST="$1"
MINER_PORT="${2:-22}"
REMOTE_DIR="/opt/basilica"

echo "Deploying Miner to $MINER_HOST:$MINER_PORT"

ssh -p "$MINER_PORT" "$MINER_HOST" "mkdir -p $REMOTE_DIR"

scp -P "$MINER_PORT" -r \
    scripts/miner/compose.prod.yml \
    scripts/miner/.env \
    config/miner.toml \
    "$MINER_HOST:$REMOTE_DIR/"

ssh -p "$MINER_PORT" "$MINER_HOST" << 'EOF'
    cd /opt/basilica
    docker-compose -f compose.prod.yml pull
    docker-compose -f compose.prod.yml up -d
    docker-compose -f compose.prod.yml ps
EOF

echo "Miner deployed successfully"
