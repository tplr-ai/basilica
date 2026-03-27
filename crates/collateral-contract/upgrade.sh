#!/usr/bin/env bash
set -euo pipefail

# ── Required variables ────────────────────────────────────────────────
PROXY_ADDRESS="${PROXY_ADDRESS:?must be set to the existing proxy contract address}"
PRIVATE_KEY="${PRIVATE_KEY:?must be set to the admin (UPGRADER_ROLE) private key}"
RPC_URL="${RPC_URL:?must be set to the target RPC URL}"

# ── Derive admin address from private key ─────────────────────────────
admin_address="$(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Admin address: $admin_address"
echo "Proxy address: $PROXY_ADDRESS"
echo "RPC URL:       $RPC_URL"
echo

# ── Pre-flight checks ────────────────────────────────────────────────

# 1. Verify RPC is reachable
if ! cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
  echo "ERROR: RPC is not reachable at ${RPC_URL}" >&2
  exit 1
fi
echo "✓ RPC reachable"

# 2. Verify proxy has code (is a deployed contract)
proxy_code="$(cast code "$PROXY_ADDRESS" --rpc-url "$RPC_URL")"
if [ "$proxy_code" = "0x" ] || [ -z "$proxy_code" ]; then
  echo "ERROR: No contract found at proxy address ${PROXY_ADDRESS}" >&2
  exit 1
fi
echo "✓ Proxy contract exists"

# 3. Read current version from proxy
current_version="$(cast call "$PROXY_ADDRESS" "getVersion()(uint256)" --rpc-url "$RPC_URL" 2>/dev/null)" || {
  echo "WARNING: Could not read getVersion() — contract may not have this function yet" >&2
  current_version="unknown"
}
echo "✓ Current implementation version: $current_version"

# 4. Verify the caller has UPGRADER_ROLE
upgrader_role="$(cast call "$PROXY_ADDRESS" "UPGRADER_ROLE()(bytes32)" --rpc-url "$RPC_URL")"
has_role="$(cast call "$PROXY_ADDRESS" "hasRole(bytes32,address)(bool)" "$upgrader_role" "$admin_address" --rpc-url "$RPC_URL")"
if [ "$has_role" != "true" ]; then
  echo "ERROR: Address $admin_address does not have UPGRADER_ROLE on the proxy" >&2
  exit 1
fi
echo "✓ Admin has UPGRADER_ROLE"

# 5. Read current implementation address from ERC1967 slot
# ERC1967 implementation slot: 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc
current_impl="$(cast storage "$PROXY_ADDRESS" 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc --rpc-url "$RPC_URL")"
current_impl="0x$(echo "$current_impl" | sed 's/0x//' | sed 's/^0*//')"
echo "✓ Current implementation: $current_impl"

echo
echo "── Deploying new implementation ──────────────────────────────────"

impl_output="$(FOUNDRY_PROFILE=local forge create src/CollateralUpgradeable.sol:CollateralUpgradeable \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --legacy \
  --broadcast)"
echo "$impl_output"

NEW_IMPLEMENTATION="$(echo "$impl_output" | awk '/Deployed to:/ {print $3}')"
if [ -z "$NEW_IMPLEMENTATION" ]; then
  echo "ERROR: Failed to parse new implementation address from deploy output" >&2
  exit 1
fi
echo
echo "✓ New implementation deployed: $NEW_IMPLEMENTATION"

# 6. Verify new implementation has code
new_code="$(cast code "$NEW_IMPLEMENTATION" --rpc-url "$RPC_URL")"
if [ "$new_code" = "0x" ] || [ -z "$new_code" ]; then
  echo "ERROR: No code at new implementation address ${NEW_IMPLEMENTATION}" >&2
  exit 1
fi

# 7. Read new version from the new implementation directly
new_version="$(cast call "$NEW_IMPLEMENTATION" "getVersion()(uint256)" --rpc-url "$RPC_URL" 2>/dev/null)" || {
  echo "WARNING: Could not read getVersion() from new implementation" >&2
  new_version="unknown"
}
echo "✓ New implementation version: $new_version"

if [ "$current_version" != "unknown" ] && [ "$new_version" != "unknown" ]; then
  if [ "$new_version" -le "$current_version" ] 2>/dev/null; then
    echo "ERROR: New version ($new_version) must be greater than current version ($current_version)" >&2
    exit 1
  fi
fi

echo
echo "── Summary ───────────────────────────────────────────────────────"
echo "  Proxy:              $PROXY_ADDRESS"
echo "  Current impl:       $current_impl"
echo "  New impl:           $NEW_IMPLEMENTATION"
echo "  Version:            $current_version → $new_version"
echo "  Admin:              $admin_address"
echo

# Optional: pass MIGRATE_CALLDATA for re-initialization in the new version
# e.g. MIGRATE_CALLDATA="$(cast calldata 'migrateV2(uint256)' 42)"
MIGRATE_CALLDATA="${MIGRATE_CALLDATA:-0x}"

if [ "$MIGRATE_CALLDATA" != "0x" ]; then
  echo "  Migration calldata: $MIGRATE_CALLDATA"
  echo
fi

read -rp "Proceed with upgrade? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Upgrade cancelled."
  exit 0
fi

echo
echo "── Upgrading proxy ─────────────────────────────────────────────"

cast send "$PROXY_ADDRESS" \
  "upgradeToAndCall(address,bytes)" \
  "$NEW_IMPLEMENTATION" \
  "$MIGRATE_CALLDATA" \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --legacy

echo
echo "── Post-upgrade verification ─────────────────────────────────────"

# Verify the implementation slot was updated
updated_impl="$(cast storage "$PROXY_ADDRESS" 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc --rpc-url "$RPC_URL")"
updated_impl="0x$(echo "$updated_impl" | sed 's/0x//' | sed 's/^0*//')"

# Compare lowercase
new_impl_lower="$(echo "$NEW_IMPLEMENTATION" | tr '[:upper:]' '[:lower:]')"
updated_impl_lower="$(echo "$updated_impl" | tr '[:upper:]' '[:lower:]')"

if [ "$updated_impl_lower" != "$new_impl_lower" ]; then
  echo "ERROR: Implementation slot not updated! Expected $NEW_IMPLEMENTATION, got $updated_impl" >&2
  exit 1
fi
echo "✓ Implementation slot updated to $updated_impl"

# Verify version through proxy
post_version="$(cast call "$PROXY_ADDRESS" "getVersion()(uint256)" --rpc-url "$RPC_URL" 2>/dev/null)" || post_version="unknown"
echo "✓ Proxy now reports version: $post_version"

# Spot-check that state survived: read netuid (should be non-zero if initialized)
netuid="$(cast call "$PROXY_ADDRESS" "netuid()(uint16)" --rpc-url "$RPC_URL" 2>/dev/null)" || netuid="unknown"
echo "✓ State check — netuid: $netuid"

trustee="$(cast call "$PROXY_ADDRESS" "trustee()(address)" --rpc-url "$RPC_URL" 2>/dev/null)" || trustee="unknown"
echo "✓ State check — trustee: $trustee"

echo
echo "Upgrade complete."
echo "  Proxy:          $PROXY_ADDRESS"
echo "  Implementation: $NEW_IMPLEMENTATION"
echo "  Version:        $post_version"
