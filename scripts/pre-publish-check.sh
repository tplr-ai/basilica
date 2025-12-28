#!/bin/bash
# Pre-publish check script for crates.io
# Usage: ./scripts/pre-publish-check.sh [crate-name]
# Example: ./scripts/pre-publish-check.sh basilica-common

set -e

CRATE=${1:-basilica-common}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Pre-publish Check: $CRATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# All publishable crates
ALL_CRATES=(
    "basilica-common"
    "basilica-protocol"
    "basilica-sdk"
    "basilica-validator"
    "basilica-miner"
    "basilica-api"
    "basilica-cli"
    "basilica-aggregator"
    "basilica-autoscaler"
    "basilica-billing"
    "basilica-operator"
    "basilica-payments"
    "basilica-storage"
)

# Check if crate is in the list
is_valid_crate() {
    local crate=$1
    for c in "${ALL_CRATES[@]}"; do
        if [[ "$c" == "$crate" ]]; then
            return 0
        fi
    done
    return 1
}

if ! is_valid_crate "$CRATE"; then
    echo "❌ Unknown crate: $CRATE"
    echo ""
    echo "Available crates:"
    for c in "${ALL_CRATES[@]}"; do
        echo "  - $c"
    done
    exit 1
fi

# Step 1: Check required Cargo.toml fields
echo "1️⃣  Checking Cargo.toml metadata..."

check_field() {
    local field=$1
    if cargo read-manifest -p "$CRATE" 2>/dev/null | jq -e ".$field" > /dev/null 2>&1; then
        echo "   ✅ $field present"
    else
        echo "   ❌ Missing $field"
        return 1
    fi
}

check_field "license" || FAILED=1
check_field "repository" || FAILED=1
check_field "description" || FAILED=1
check_field "readme" || echo "   ⚠️  readme field not set (optional)"
check_field "keywords" || echo "   ⚠️  keywords not set (optional)"
check_field "categories" || echo "   ⚠️  categories not set (optional)"

if [[ -n "${FAILED:-}" ]]; then
    echo ""
    echo "❌ Missing required metadata fields"
    exit 1
fi
echo ""

# Step 2: Check README.md exists
echo "2️⃣  Checking README.md..."
if [[ -f "crates/$CRATE/README.md" ]]; then
    echo "   ✅ README.md exists"
    # Check it's not empty
    if [[ -s "crates/$CRATE/README.md" ]]; then
        LINES=$(wc -l < "crates/$CRATE/README.md")
        echo "   ✅ README.md has $LINES lines"
    else
        echo "   ⚠️  README.md is empty"
    fi
else
    echo "   ❌ README.md missing"
    echo "   Create at: crates/$CRATE/README.md"
    exit 1
fi
echo ""

# Step 3: Check CHANGELOG.md exists
echo "3️⃣  Checking CHANGELOG.md..."
if [[ -f "crates/$CRATE/CHANGELOG.md" ]]; then
    echo "   ✅ CHANGELOG.md exists"
else
    echo "   ⚠️  CHANGELOG.md missing (recommended)"
fi
echo ""

# Step 4: Check documentation builds
echo "4️⃣  Building documentation..."
if RUSTDOCFLAGS="-D warnings" cargo doc -p "$CRATE" --no-deps 2>&1 | tee /tmp/doc-output.txt; then
    echo "   ✅ Documentation builds without errors"
else
    echo "   ⚠️  Documentation has warnings/errors"
    cat /tmp/doc-output.txt
fi
echo ""

# Step 5: Run doc tests
echo "5️⃣  Running doc tests..."
if cargo test --doc -p "$CRATE" 2>&1 | tee /tmp/doctest-output.txt; then
    echo "   ✅ Doc tests pass"
else
    echo "   ⚠️  Some doc tests failed"
    tail -20 /tmp/doctest-output.txt
fi
echo ""

# Step 6: Check for git dependencies
echo "6️⃣  Checking for git dependencies..."
CARGO_TOML="crates/$CRATE/Cargo.toml"
if grep -q 'git = "' "$CARGO_TOML" 2>/dev/null; then
    echo "   ⚠️  Found git dependencies (these block crates.io publishing):"
    grep 'git = "' "$CARGO_TOML" | sed 's/^/      /'
else
    echo "   ✅ No direct git dependencies"
fi

# Check workspace dependencies
if grep -q "workspace = true" "$CARGO_TOML"; then
    # Check root Cargo.toml for git deps that might affect this crate
    ROOT_CARGO="Cargo.toml"
    if grep -q 'git = "' "$ROOT_CARGO" 2>/dev/null; then
        echo "   ⚠️  Workspace has git dependencies (may block publishing):"
        grep 'git = "' "$ROOT_CARGO" | head -5 | sed 's/^/      /'
    else
        echo "   ✅ No workspace git dependencies"
    fi
fi
echo ""

# Step 7: Dry-run publish
echo "7️⃣  Dry-run publish..."
if cargo publish -p "$CRATE" --dry-run 2>&1 | tee /tmp/publish-output.txt; then
    echo "   ✅ Dry-run publish succeeded"
else
    echo "   ❌ Dry-run publish failed"
    cat /tmp/publish-output.txt
    exit 1
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ $CRATE is ready for publishing!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To publish:"
echo "  cargo publish -p $CRATE"
echo ""
echo "Or use the CI workflow:"
echo "  gh workflow run publish.yml -f crate=$CRATE -f dry_run=false"

