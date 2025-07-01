#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [port]"
    echo "Example: $0 root@192.168.1.10 22"
    exit 1
fi

VALIDATOR_HOST="$1"
VALIDATOR_PORT="${2:-22}"
REMOTE_DIR="/opt/basilica"

echo "Deploying Validator to $VALIDATOR_HOST:$VALIDATOR_PORT"

ssh -p "$VALIDATOR_PORT" "$VALIDATOR_HOST" "mkdir -p $REMOTE_DIR"

scp -P "$VALIDATOR_PORT" -r \
    scripts/validator/compose.prod.yml \
    scripts/validator/.env \
    config/validator.toml \
    "$VALIDATOR_HOST:$REMOTE_DIR/"

ssh -p "$VALIDATOR_PORT" "$VALIDATOR_HOST" << 'EOF'
    cd /opt/basilica
    docker-compose -f compose.prod.yml pull
    docker-compose -f compose.prod.yml up -d
    docker-compose -f compose.prod.yml ps
EOF

echo "Validator deployed successfully"
