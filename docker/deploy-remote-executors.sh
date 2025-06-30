#!/bin/bash
# Deploy executors to remote GPU machines

# Build the executor binary first
echo "Building executor binary..."
cargo build --release -p executor

# Copy the binary to a staging location
cp ../target/release/executor ./executor-binary

# Create SSH keys if they don't exist
if [ ! -f ~/.ssh/gpu_server_key ]; then
    echo "Generating SSH key for GPU servers..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/gpu_server_key -N ""
fi

# Deploy to remote machines
echo "Deploying executors to remote machines..."
basilica-miner deploy-executors \
    --config configs/miner-local-remote.toml \
    "$@"

# Check status
echo "Checking executor status..."
basilica-miner deploy-executors \
    --config configs/miner-local-remote.toml \
    --status-only