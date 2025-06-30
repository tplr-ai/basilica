#!/bin/bash
# Setup Alice wallet for local testing

echo "Setting up Alice wallet for local testing..."

# Create wallet directory
mkdir -p ~/.bittensor/wallets/alice/hotkeys

# Alice's well-known development keys
# These are standard Substrate development keys
cat > ~/.bittensor/wallets/alice/coldkey << EOF
{
  "secretPhrase": "bottom drive obey lake curtain smoke basket hold race lonely fit walk",
  "secretSeed": "0xfac7959dbfe72f052e5a0c3c8d6530f202b02fd8f9f5ca3580ec8deb7797479e",
  "publicKey": "0x46ebddef8cd9bb167dc30878d7113b7e168e6f0646beffd77d69d39bad76b47a",
  "ss58Address": "5DfhGyQdFobKM8NsWvEeAKk5EQQgYe9AydgJ7rMB6E1EqRzV",
  "accountId": "0x46ebddef8cd9bb167dc30878d7113b7e168e6f0646beffd77d69d39bad76b47a",
  "ss58PublicKey": "5DfhGyQdFobKM8NsWvEeAKk5EQQgYe9AydgJ7rMB6E1EqRzV"
}
EOF

# Use same key for hotkey in dev
cp ~/.bittensor/wallets/alice/coldkey ~/.bittensor/wallets/alice/hotkeys/default

echo "Alice wallet created!"
echo "Coldkey: 5DfhGyQdFobKM8NsWvEeAKk5EQQgYe9AydgJ7rMB6E1EqRzV"