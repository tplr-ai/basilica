#!/bin/bash
set -e

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Generating new P256 key pair..." >&2

# Generate private key
openssl ecparam -genkey -name prime256v1 -noout -out "$TEMP_DIR/private_key.pem"

PUBLIC_KEY_HEX=$(openssl ec -in "$TEMP_DIR/private_key.pem" -pubout -outform DER 2>/dev/null | \
                 openssl ec -pubin -inform DER -conv_form compressed -outform DER 2>/dev/null | \
                 tail -c 33 | xxd -p -c 33)

# Extract standard PEM public key
openssl ec -in "$TEMP_DIR/private_key.pem" -pubout -out "$TEMP_DIR/public_key.pem"

cp "$TEMP_DIR/private_key.pem" "./private_key.pem"
cp "$TEMP_DIR/public_key.pem" "./public_key.pem"
echo "$PUBLIC_KEY_HEX" > "./public_key.hex"

# Set appropriate permissions
chmod 600 "./private_key.pem"
chmod 644 "./public_key.pem"
chmod 644 "./public_key.hex"

echo "Generated files:" >&2
echo "  Private key: private_key.pem (permissions: 600)" >&2
echo "  Public key (PEM): public_key.pem" >&2
echo "  Public key (hex): public_key.hex" >&2
echo "" >&2
echo "Compressed public key:" >&2
echo "$PUBLIC_KEY_HEX"
