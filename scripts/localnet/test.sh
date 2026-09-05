#!/usr/bin/env bash
# Read-only checks for the selected Compose profile; see healthcheck.py --help.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/healthcheck.py" "$@"
