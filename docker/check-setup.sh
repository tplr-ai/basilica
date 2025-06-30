#!/bin/bash
# Quick setup checker for remote executor development

echo "🔍 Checking Basilica setup..."
echo "=============================="

# Check if SSH key exists
if [ -f ~/.ssh/basilica_gpu_test ]; then
    echo "✅ SSH key exists"
else
    echo "❌ SSH key missing - run: ssh-keygen -t rsa -b 4096 -f ~/.ssh/basilica_gpu_test -N ''"
    exit 1
fi

# Check if config needs updating
if grep -q "YOUR_VAST_AI_HOST" docker/configs/miner-local.toml; then
    echo "❌ Remote machine not configured"
    echo "   Run: ./setup-vast-gpu.sh"
    echo "   Or edit: docker/configs/miner-local.toml"
    exit 1
else
    echo "✅ Remote machine configured"
fi

# Check if executor binary exists
if [ -f ../target/release/executor ]; then
    echo "✅ Executor binary exists"
else
    echo "⚠️  Executor binary missing - will be built"
fi

echo ""
echo "✅ Setup looks good! You can run: just dev"