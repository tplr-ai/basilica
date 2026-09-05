#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The CLI owns bundle identity, version, destinations,
# upgrade/migration and uninstall. This script downloads no independent skills.
agents=(--agent universal --agent claude-code)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cursor-only) agents=(--agent cursor); shift ;;
    --claude-only) agents=(--agent claude-code); shift ;;
    --codex-only) agents=(--agent codex); shift ;;
    -h|--help)
      echo 'Usage: install-agent-skills.sh [--cursor-only|--claude-only|--codex-only]'
      echo 'Delegates to basilica skills install -y; install the current Basilica CLI first.'
      exit 0 ;;
    --base-url)
      echo 'The standalone website bundle is retired. --base-url is no longer supported.' >&2
      echo 'Use BASILICA_SKILLS_TARBALL_URL only to mirror the CLI-pinned archive; its contents are verified.' >&2
      exit 1 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
if [[ -n "${BASILICA_AGENT_BASE_URL:-}" ]]; then
  echo 'BASILICA_AGENT_BASE_URL belongs to the retired standalone bundle; unset it and use the CLI installer.' >&2
  exit 1
fi
if ! command -v basilica >/dev/null 2>&1; then
  echo 'Install the Basilica CLI from https://basilica.ai/install.sh, then rerun this command.' >&2
  exit 1
fi
# Older CLI releases download an unversioned bundle. Require the manifest-aware
# contract rather than silently delegating to the deprecated implementation.
identity="$(basilica skills list --agent universal)"
case "$identity" in
  *'Bundle version:'*) ;;
  *) echo 'Upgrade Basilica: this CLI predates versioned skill bundles.' >&2; exit 1 ;;
esac
exec basilica skills install -y "${agents[@]}"
