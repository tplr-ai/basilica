#!/bin/bash
# SSH wrapper stub for backward compatibility
# With dynamic discovery, SSH configuration is handled automatically
# This file exists to prevent errors in legacy scripts

# Default SSH options for non-interactive operation
SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# SSH identity file (if needed)
SSH_IDENTITY="${SSH_IDENTITY:-~/.ssh/tplr}"

# Wrapper function for SSH commands
ssh_wrapper() {
    if [[ -f "$SSH_IDENTITY" ]]; then
        ssh -i "$SSH_IDENTITY" $SSH_OPTIONS "$@"
    else
        ssh $SSH_OPTIONS "$@"
    fi
}

# Export for use in other scripts
export -f ssh_wrapper
export SSH_OPTIONS
export SSH_IDENTITY