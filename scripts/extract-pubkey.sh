#!/bin/bash
set -e

PRIVATE_KEY=""
EXPORT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --export)
            EXPORT_FILE="$2"
            shift 2
            ;;
        *)
            PRIVATE_KEY="$1"
            shift
            ;;
    esac
done

PRIVATE_KEY="${PRIVATE_KEY:-private_key.pem}"

if [ ! -f "$PRIVATE_KEY" ]; then
    echo "Private key file not found: $PRIVATE_KEY" >&2
    exit 1
fi

extract_compressed_hex() {
    openssl ec -in "$1" -pubout -outform DER 2>/dev/null | \
    openssl ec -pubin -inform DER -conv_form compressed -outform DER 2>/dev/null | \
    tail -c 33 | xxd -p -c 33
}

if [ -n "$EXPORT_FILE" ]; then
    openssl ec -in "$PRIVATE_KEY" -pubout -out "$EXPORT_FILE"
    echo "Public key exported to: $EXPORT_FILE" >&2

    echo "Compressed hex format:" >&2
    extract_compressed_hex "$PRIVATE_KEY"
else
    extract_compressed_hex "$PRIVATE_KEY"
fi

