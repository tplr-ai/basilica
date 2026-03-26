#!/bin/bash
set -euo pipefail

# Basilica agent skills installer
# Usage:
#   curl -fsSL https://basilica.ai/agents/install.sh | bash
#   curl -fsSL https://basilica.ai/agents/install.sh | bash -s -- --cursor-only

BASE_URL="${BASILICA_AGENT_BASE_URL:-https://basilica.ai}"
INSTALL_CURSOR=1
INSTALL_CLAUDE=1
INSTALL_CODEX=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cursor-only)
      INSTALL_CURSOR=1
      INSTALL_CLAUDE=0
      INSTALL_CODEX=0
      shift
      ;;
    --claude-only)
      INSTALL_CURSOR=0
      INSTALL_CLAUDE=1
      INSTALL_CODEX=0
      shift
      ;;
    --codex-only)
      INSTALL_CURSOR=0
      INSTALL_CLAUDE=0
      INSTALL_CODEX=1
      shift
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--cursor-only|--claude-only|--codex-only] [--base-url URL]" >&2
      exit 1
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

SKILLS=(
  "basilica-cloud-operator"
  "basilica-account-ops"
  "basilica-rentals-ops"
  "basilica-serverless-ops"
  "basilica-sdk-ops"
)

fetch_file() {
  local url="$1"
  local out="$2"
  curl -fsSL "$url" -o "$out"
}

install_skill_set() {
  local root="$1"
  local kind="$2"

  mkdir -p "$root"

  for skill in "${SKILLS[@]}"; do
    local skill_dir="$root/$skill"
    mkdir -p "$skill_dir"
    fetch_file "$BASE_URL/agents/$skill/SKILL.md" "$skill_dir/SKILL.md"
  done

  fetch_file "$BASE_URL/agents/cloud-ops.md" "$root/BASILICA-CLOUD-OPS.md"

  echo "Installed Basilica skills for $kind at $root"
}

if [[ "$INSTALL_CURSOR" -eq 1 ]]; then
  install_skill_set "$HOME/.cursor/skills" "Cursor"
fi

if [[ "$INSTALL_CLAUDE" -eq 1 ]]; then
  install_skill_set "$HOME/.claude/skills" "Claude"
fi

if [[ "$INSTALL_CODEX" -eq 1 ]]; then
  install_skill_set "$HOME/.codex/skills" "Codex"
fi

cat <<EOF

Basilica agent bundle installed.

Primary skill:
  basilica-cloud-operator

Reference:
  $BASE_URL/agents/cloud-ops.md
  $BASE_URL/llms-full.txt

If your agent was already running, restart it so it reloads installed skills.
EOF
