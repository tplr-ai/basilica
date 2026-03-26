#!/bin/bash
#
# Basilica WebSocket Deployment - CLI Example
#
# Demonstrates deploying with WebSocket support using the basilica CLI:
#   1. Deploy with --websocket to enable WebSocket connections
#   2. Deploy with --ws-idle-timeout for custom idle timeout
#
# WebSocket support configures the gateway to allow long-lived bidirectional
# connections, useful for chat servers, live dashboards, and streaming APIs.
#
# Prerequisites:
#   - basilica CLI installed and authenticated (basilica login)
#
# Usage:
#   ./33_websocket.sh

set -e

INSTANCE_NAME="ws-demo-$(date +%s)"

echo "========================================================================"
echo "Basilica WebSocket Deployment - CLI Example"
echo "========================================================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Deploy with WebSocket support (default idle timeout)
# ------------------------------------------------------------------
echo "Step 1: Creating deployment with --websocket..."
echo "------------------------------------------------------------------------"

basilica deploy hashicorp/http-echo:latest \
    --name "$INSTANCE_NAME" \
    --port 5678 \
    --websocket \
    --ttl 600 \
    --json

echo ""

# ------------------------------------------------------------------
# Step 2: Deploy with custom idle timeout
# ------------------------------------------------------------------
INSTANCE_NAME_CUSTOM="ws-custom-$(date +%s)"

echo "Step 2: Creating deployment with custom idle timeout (3600s)..."
echo "------------------------------------------------------------------------"

basilica deploy hashicorp/http-echo:latest \
    --name "$INSTANCE_NAME_CUSTOM" \
    --port 5678 \
    --websocket \
    --ws-idle-timeout 3600 \
    --ttl 600 \
    --json

echo ""

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
echo "------------------------------------------------------------------------"
read -p "Delete the deployments? (y/N): " CONFIRM
if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    basilica summon delete "$INSTANCE_NAME"
    basilica summon delete "$INSTANCE_NAME_CUSTOM"
else
    echo "Deployments left running:"
    echo "  $INSTANCE_NAME"
    echo "  $INSTANCE_NAME_CUSTOM"
    echo "  Delete later: basilica summon delete <name>"
fi

echo ""
echo "========================================================================"
echo "CLI Commands Used:"
echo "  basilica deploy <image> --websocket                - Enable WebSocket"
echo "  basilica deploy <image> --websocket --ws-idle-timeout 3600"
echo "                                                     - Custom idle timeout"
echo "========================================================================"
