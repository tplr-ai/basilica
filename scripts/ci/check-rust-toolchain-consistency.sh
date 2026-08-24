#!/usr/bin/env bash

# Keep duplicated Rust version declarations aligned with the repository
# toolchain. The migrations Dockerfile is a separate project with its own
# toolchain and is intentionally outside this check.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

expected="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' rust-toolchain.toml)"

if [[ -z "$expected" ]]; then
  echo "::error::could not parse [toolchain] channel from rust-toolchain.toml" >&2
  exit 2
fi

dockerfiles=(
  scripts/miner/Dockerfile
  scripts/validator/Dockerfile
)

mismatches=0
for dockerfile in "${dockerfiles[@]}"; do
  builder="$(sed -n 's/^FROM rust:\([0-9][0-9A-Za-z._]*\)-.*/\1/p' "$dockerfile" | head -n 1)"

  if [[ -z "$builder" ]]; then
    echo "::error file=$dockerfile::could not parse Rust builder version" >&2
    mismatches=$((mismatches + 1))
  elif [[ "$builder" != "$expected" ]]; then
    echo "::error file=$dockerfile::Rust builder is $builder; expected $expected" >&2
    mismatches=$((mismatches + 1))
  fi
done

release_workflow=".github/workflows/release-python-sdk.yml"
maturin_uses="$(awk '/^[[:space:]]*uses: PyO3\/maturin-action@/ { count++ } END { print count + 0 }' "$release_workflow")"
release_pins="$(sed -n 's/^[[:space:]]*rust-toolchain:[[:space:]]*"\([^"]*\)".*/\1/p' "$release_workflow")"
release_pin_count="$(printf '%s\n' "$release_pins" | awk 'NF { count++ } END { print count + 0 }')"

if [[ "$maturin_uses" -eq 0 ]]; then
  echo "::error file=$release_workflow::could not find any Maturin release actions" >&2
  mismatches=$((mismatches + 1))
elif [[ "$release_pin_count" -ne "$maturin_uses" ]]; then
  echo "::error file=$release_workflow::found $maturin_uses Maturin release actions but $release_pin_count Rust toolchain pins" >&2
  mismatches=$((mismatches + 1))
fi

while IFS= read -r release_pin; do
  [[ -z "$release_pin" ]] && continue
  if [[ "$release_pin" != "$expected" ]]; then
    echo "::error file=$release_workflow::Maturin release toolchain is $release_pin; expected $expected" >&2
    mismatches=$((mismatches + 1))
  fi
done <<< "$release_pins"

if [[ "$mismatches" -ne 0 ]]; then
  exit 1
fi

echo "Supported Docker builders and Python release jobs match Rust $expected."
