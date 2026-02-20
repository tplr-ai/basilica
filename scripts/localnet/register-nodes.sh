#!/usr/bin/env bash
# Trigger and verify miner -> validator node registration on localnet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
LOG_TIMEOUT_SECS=120
RESTART_MINER=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--restart-miner] [--timeout <seconds>]

Checks localnet miner/validator containers, optionally restarts miner,
waits for registration log marker, then checks validator APIs:
- GET /miners
- GET /nodes
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

is_container_running() {
  local name="$1"
  docker ps --format '{{.Names}}' | grep -qx "$name"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-miner)
      RESTART_MINER=true
      ;;
    --timeout)
      shift
      [[ $# -gt 0 ]] || {
        echo "ERROR: --timeout requires a value" >&2
        exit 1
      }
      LOG_TIMEOUT_SECS="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

require_cmd docker
require_cmd curl

echo "Checking localnet containers..."
is_container_running "basilica-validator" || {
  echo "ERROR: basilica-validator is not running" >&2
  exit 1
}
is_container_running "basilica-miner" || {
  echo "ERROR: basilica-miner is not running" >&2
  exit 1
}

echo "validator/miner containers are running"

start_marker="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ "$RESTART_MINER" == "true" ]]; then
  echo "Restarting miner container via docker compose..."
  docker compose -f "$COMPOSE_FILE" restart miner
fi

echo "Watching miner logs for registration marker..."
found_marker=false
start_epoch="$(date +%s)"

while true; do
  if docker logs basilica-miner --since "$start_marker" 2>&1 | grep -q "Successfully registered with validator"; then
    found_marker=true
    break
  fi

  now_epoch="$(date +%s)"
  if (( now_epoch - start_epoch >= LOG_TIMEOUT_SECS )); then
    break
  fi

  sleep 2
done

if [[ "$found_marker" != "true" ]]; then
  echo "WARNING: registration marker was not observed within ${LOG_TIMEOUT_SECS}s"
else
  echo "Observed registration marker: Successfully registered with validator"
fi

echo "Checking validator API: /health"
curl -fsS "http://localhost:8080/health" >/dev/null
echo "  /health OK"

echo "Checking validator API: /miners"
miners_json="$(curl -fsS "http://localhost:8080/miners")"
echo "$miners_json"

echo "Checking validator API: /nodes"
nodes_json="$(curl -fsS "http://localhost:8080/nodes")"
echo "$nodes_json"

if command -v jq >/dev/null 2>&1; then
  miners_count="$(printf '%s\n' "$miners_json" | jq -r '.total_count // (.miners | length) // 0')"
  nodes_count="$(printf '%s\n' "$nodes_json" | jq -r '.total_count // (.available_nodes | length) // 0')"
  echo "Summary: miners=${miners_count}, nodes=${nodes_count}"
fi
