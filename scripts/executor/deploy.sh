#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [port]"
    echo "Example: $0 root@192.168.1.10 22"
    exit 1
fi

EXECUTOR_HOST="$1"
EXECUTOR_PORT="${2:-22}"
REMOTE_DIR="/opt/basilica"

echo "Deploying Executor to $EXECUTOR_HOST:$EXECUTOR_PORT"

ssh -p "$EXECUTOR_PORT" "$EXECUTOR_HOST" "mkdir -p $REMOTE_DIR"

scp -P "$EXECUTOR_PORT" -r \
    scripts/executor/compose.prod.yml \
    config/executor.toml \
    "$EXECUTOR_HOST:$REMOTE_DIR/"

ssh -p "$EXECUTOR_PORT" "$EXECUTOR_HOST" << 'EOF'
    cd /opt/basilica
    docker-compose -f compose.prod.yml pull
    docker-compose -f compose.prod.yml up -d
    docker-compose -f compose.prod.yml ps
EOF

echo "Executor deployed successfully"
