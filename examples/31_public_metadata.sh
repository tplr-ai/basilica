#!/bin/bash
#
# Basilica Public Deployment Metadata - CLI Example
#
# Demonstrates the full metadata enrollment lifecycle using the basilica CLI:
#   1. Deploy with --public-metadata to enable enrollment at creation
#   2. Check enrollment status with enroll-metadata subcommand
#   3. Query public metadata (no authentication required)
#   4. Toggle enrollment on/off
#
# This feature allows deployment owners to opt-in to exposing non-sensitive
# metadata publicly, enabling Bittensor subnet validators to verify what
# miners have deployed.
#
# Prerequisites:
#   - basilica CLI installed and authenticated (basilica login)
#
# Usage:
#   ./31_public_metadata.sh

set -e

INSTANCE_NAME="metadata-demo-$(date +%s)"

echo "========================================================================"
echo "Basilica Public Deployment Metadata - CLI Example"
echo "========================================================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Deploy with --public-metadata enabled at creation
# ------------------------------------------------------------------
echo "Step 1: Creating deployment with --public-metadata..."
echo "------------------------------------------------------------------------"

basilica deploy hashicorp/http-echo:latest \
    --name "$INSTANCE_NAME" \
    --port 5678 \
    --public-metadata \
    --ttl 600 \
    --json

echo ""

# ------------------------------------------------------------------
# Step 2: Check enrollment status (authenticated)
# ------------------------------------------------------------------
echo "Step 2: Checking enrollment status..."
echo "------------------------------------------------------------------------"

basilica deploy enroll-metadata "$INSTANCE_NAME"

echo ""

# ------------------------------------------------------------------
# Step 3: Query public metadata (no auth needed)
# ------------------------------------------------------------------
echo "Step 3: Querying public metadata (unauthenticated endpoint)..."
echo "------------------------------------------------------------------------"

basilica deploy metadata "$INSTANCE_NAME"

echo ""
echo "Same in JSON format:"
basilica deploy metadata "$INSTANCE_NAME" --json

echo ""

# ------------------------------------------------------------------
# Step 4: Disable enrollment
# ------------------------------------------------------------------
echo "Step 4: Disabling metadata enrollment..."
echo "------------------------------------------------------------------------"

basilica deploy enroll-metadata "$INSTANCE_NAME" --disable

echo ""

# ------------------------------------------------------------------
# Step 5: Verify public metadata is no longer visible
# ------------------------------------------------------------------
echo "Step 5: Querying public metadata after disabling..."
echo "------------------------------------------------------------------------"

basilica deploy metadata "$INSTANCE_NAME"

echo ""

# ------------------------------------------------------------------
# Step 6: Re-enable enrollment
# ------------------------------------------------------------------
echo "Step 6: Re-enabling metadata enrollment..."
echo "------------------------------------------------------------------------"

basilica deploy enroll-metadata "$INSTANCE_NAME" --enable

echo ""

# ------------------------------------------------------------------
# Step 7: Verify status shows enrolled again
# ------------------------------------------------------------------
echo "Step 7: Final status check..."
echo "------------------------------------------------------------------------"

basilica deploy enroll-metadata "$INSTANCE_NAME"

echo ""

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
echo "------------------------------------------------------------------------"
read -p "Delete the deployment? (y/N): " CONFIRM
if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    basilica summon delete "$INSTANCE_NAME"
else
    echo "Deployment left running: $INSTANCE_NAME"
    echo "  Delete later: basilica summon delete $INSTANCE_NAME"
fi

echo ""
echo "========================================================================"
echo "CLI Commands Used:"
echo "  basilica deploy <image> --public-metadata      - Deploy with enrollment"
echo "  basilica deploy enroll-metadata <name>          - Check enrollment status"
echo "  basilica deploy enroll-metadata <name> --enable - Enable enrollment"
echo "  basilica deploy enroll-metadata <name> --disable- Disable enrollment"
echo "  basilica deploy metadata <name>                 - Public lookup (no auth)"
echo "  basilica deploy metadata <name> --json          - Public lookup as JSON"
echo "========================================================================"
