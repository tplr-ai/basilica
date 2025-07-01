#!/bin/bash
set -euo pipefail

# Simple Public API Deployment Script

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [port]"
    echo "Example: $0 root@192.168.1.10 22"
    exit 1
fi

PUBLIC_API_HOST="$1"
PUBLIC_API_PORT="${2:-22}"
REMOTE_DIR="/opt/basilica"

echo "Deploying Public API to $PUBLIC_API_HOST:$PUBLIC_API_PORT"

# Create remote directory
ssh -p "$PUBLIC_API_PORT" "$PUBLIC_API_HOST" "mkdir -p $REMOTE_DIR"

# Copy necessary files
scp -P "$PUBLIC_API_PORT" -r \
    scripts/public-api/compose.prod.yml \
    config/public-api.toml \
    "$PUBLIC_API_HOST:$REMOTE_DIR/"

# Deploy
ssh -p "$PUBLIC_API_PORT" "$PUBLIC_API_HOST" << 'EOF'
    cd /opt/basilica
    docker-compose -f compose.prod.yml pull
    docker-compose -f compose.prod.yml up -d
    docker-compose -f compose.prod.yml ps
EOF

echo "Public API deployed successfully"