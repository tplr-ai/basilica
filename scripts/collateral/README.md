# Collateral CLI Ops (Localnet First)

This directory is CLI-first. Use `collateral-cli` directly for all collateral operations.

## Files

- `setup-localnet-env.sh`: deploy localnet contract and generate `.env.local`.
- `.env.local.example`: env template for direct `collateral-cli` usage.

## Setup

### Manual env setup

1. Copy env template:

```bash
cp scripts/collateral/.env.local.example scripts/collateral/.env.local
```

2. Fill required values in `scripts/collateral/.env.local`.
3. Ensure `collateral-cli` is installed and on `PATH`.

### E2E localnet setup (recommended)

1. Start localnet:

```bash
cd scripts/localnet
./start.sh network
```

2. Deploy contract + generate env:

```bash
cd ../..
scripts/collateral/setup-localnet-env.sh
```

`setup-localnet-env.sh` also creates/reuses:
`scripts/localnet/wallets/contract_deployer_evm.env`

## Run Commands (Direct `collateral-cli`)

From repo root:

```bash
source scripts/collateral/.env.local
cc() { collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" "$@"; }
```

Contract metadata:

```bash
cc query trustee
cc query min-collateral-increase
cc query decision-timeout
cc query netuid
cc query contract-hotkey
```

Deposit alpha:

```bash
cc tx deposit \
  --private-key "$PRIVATE_KEY" \
  --hotkey "$HOTKEY" \
  --node-id "$NODE_ID" \
  --alpha-hotkey "$ALPHA_HOTKEY" \
  --alpha-amount "$ALPHA_AMOUNT_WEI"

cc query node-to-miner --hotkey "$HOTKEY" --node-id "$NODE_ID"
cc query alpha-collaterals --hotkey "$HOTKEY" --node-id "$NODE_ID"
```

Reclaim flow:

```bash
cc tx reclaim-collateral \
  --private-key "$PRIVATE_KEY" \
  --hotkey "$HOTKEY" \
  --node-id "$NODE_ID" \
  --url "$URL" \
  --url-content-sha256 "$URL_CONTENT_SHA256"

cc events scan --from-block "$FROM_BLOCK" --to-block "$TO_BLOCK" --format json
# set RECLAIM_REQUEST_ID from the returned ReclaimProcessStarted event before finalize/deny

cc tx finalize-reclaim --private-key "$PRIVATE_KEY" --reclaim-request-id "$RECLAIM_REQUEST_ID"
# or trustee path:
cc tx deny-reclaim \
  --private-key "$PRIVATE_KEY" \
  --reclaim-request-id "$RECLAIM_REQUEST_ID" \
  --url "$URL" \
  --url-content-sha256 "$URL_CONTENT_SHA256"
```

Burn-register:

```bash
cc tx burn-register --private-key "$PRIVATE_KEY"
```

## Notes

- Keep private keys out of version control.
- This flow is localnet-first but works on other networks with the right env values.
