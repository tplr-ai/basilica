#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Setting up test environment..."

# Create directories
mkdir -p data keys/wallets/test_validator/hotkeys

# Generate SSH keys for testing
if [ ! -f "keys/id_rsa" ]; then
    echo "[INFO] Generating SSH test keys..."
    ssh-keygen -t rsa -b 2048 -f keys/id_rsa -N "" -C "basilica-test-key"
fi

# Generate test certificates
if [ ! -f "keys/private_key.pem" ]; then
    echo "[INFO] Generating test certificates..."
    openssl genrsa -out keys/private_key.pem 2048
    openssl rsa -in keys/private_key.pem -pubout -out keys/public_key.pem
fi

# Generate test Bittensor wallet files
if [ ! -f "keys/wallets/test_validator/coldkey" ]; then
    echo "[INFO] Generating test Bittensor wallet..."
    # Create dummy wallet files for testing
    echo "test_coldkey_data_$(date +%s)" > keys/wallets/test_validator/coldkey
    echo "test_hotkey_data_$(date +%s)" > keys/wallets/test_validator/hotkeys/test_hotkey
fi

# Set appropriate permissions
chmod 600 keys/id_rsa
chmod 644 keys/id_rsa.pub
chmod 600 keys/private_key.pem
chmod 644 keys/public_key.pem
chmod 600 keys/wallets/test_validator/coldkey
chmod 600 keys/wallets/test_validator/hotkeys/test_hotkey

echo "[SUCCESS] Test environment setup completed"
echo "Generated files:"
echo "  - SSH keys: keys/id_rsa, keys/id_rsa.pub"
echo "  - Test certificates: keys/private_key.pem, keys/public_key.pem"
echo "  - Test wallet: keys/wallets/test_validator/"
echo "  - Data directory: data/"