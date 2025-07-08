#!/usr/bin/env bash

set -euo pipefail

NVML_LIB=$(find /usr/lib -name "libnvidia-ml.so.*" -type f 2>/dev/null | head -1)

if [ -z "$NVML_LIB" ]; then
    echo "NVML library not found"
    exit 1
fi

NVML_DIR=$(dirname "$NVML_LIB")
ln -sf "$(basename "$NVML_LIB")" "$NVML_DIR/libnvidia-ml.so.1"
ln -sf "$NVML_DIR/libnvidia-ml.so.1" "$NVML_DIR/libnvidia-ml.so"

ldconfig

echo "done."
