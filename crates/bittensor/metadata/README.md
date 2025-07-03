# Bittensor Metadata Files

This directory contains pre-generated metadata files for different Bittensor networks.
These files are used during compilation to provide type definitions without requiring
network access at build time.

## Files

- `finney.rs` - Mainnet metadata
- `test.rs` - Testnet metadata  
- `local.rs` - Local development network metadata

## Generating/Updating Metadata

To regenerate these files when the chain runtime changes:

```bash
# Generate all network metadata (except local)
cargo run --bin generate-metadata -- all

# Generate specific network metadata
cargo run --bin generate-metadata -- finney
cargo run --bin generate-metadata -- test

# Generate local network metadata (requires running local node)
LOCAL_ENDPOINT=ws://localhost:9944 cargo run --bin generate-metadata -- local
```

## Why Pre-generated?

Pre-generating metadata files:
- Eliminates network dependencies during builds
- Makes builds reproducible
- Works in isolated environments (Docker, CI)
- Speeds up compilation
- Prevents build failures due to network issues

## When to Update

Update these files when:
- Bittensor chain runtime is upgraded
- New pallets or types are added
- Breaking changes in the chain API