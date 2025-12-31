# Publishing to crates.io

This document describes the process and requirements for publishing Basilica crates to crates.io.

## Recommended: Unified `basilica` Crate

For most users, the unified `basilica` crate is the easiest way to get started:

```toml
[dependencies]
basilica = "0.1"
```

This meta-crate re-exports all components with feature flags. See [docs.rs/basilica](https://docs.rs/basilica) for unified documentation.

## Publishable Crates

The following crates are prepared for crates.io publication:

| Crate | Version | Status | Description |
|-------|---------|--------|-------------|
| **basilica** | 0.1.0 | ✅ Ready | Unified meta-crate (recommended) |
| basilica-common | 0.1.0 | ✅ Ready | Core types and utilities |
| basilica-protocol | 0.1.0 | ✅ Ready | gRPC protocol definitions |
| basilica-sdk | 0.10.0 | ✅ Ready | High-level client SDK |
| basilica-validator | 0.1.0 | ✅ Ready | Validator node |
| basilica-miner | 0.1.0 | ✅ Ready | Miner node |
| basilica-api | 0.1.0 | ✅ Ready | REST API server |
| basilica-cli | 0.10.1 | ✅ Ready | Command-line interface |
| basilica-aggregator | 0.1.0 | ✅ Ready | Metrics aggregation |
| basilica-autoscaler | 0.1.0 | ✅ Ready | Auto-scaling service |
| basilica-billing | 0.1.0 | ✅ Ready | Billing service |
| basilica-operator | 0.1.0 | ✅ Ready | Kubernetes operator |
| basilica-payments | 0.1.0 | ✅ Ready | Payment processing |
| basilica-storage | 0.1.0 | ✅ Ready | Storage backends |

## Dependencies

All dependencies are now available on crates.io:

```toml
# From Cargo.toml (workspace root)
bittensor = { version = "0.1.1", package = "bittensor-rs" }
```

The `bittensor-rs` crate is published at: https://crates.io/crates/bittensor-rs

## Publishing Order

Crates must be published in dependency order (7 tiers):

```
1. basilica-common, basilica-storage, basilica-operator (no internal deps)
2. basilica-protocol, basilica-autoscaler (depend on common)
3. basilica-billing, basilica-validator, basilica-miner, basilica-payments (depend on common + protocol)
4. basilica-aggregator (depends on billing)
5. basilica-sdk (depends on aggregator + validator)
6. basilica-api, basilica-cli (depend on sdk)
7. basilica (unified meta-crate - depends on all above)
```

## Pre-Publish Checklist

Before publishing any crate:

```bash
# Run the pre-publish check script
./scripts/pre-publish-check.sh basilica-common

# Or use just
just pre-publish-check basilica-common
```

The check verifies:
- ✅ Required Cargo.toml metadata (license, repository, description)
- ✅ README.md exists and is non-empty
- ✅ CHANGELOG.md exists
- ✅ Documentation builds without warnings
- ✅ Doc tests pass
- ✅ No blocking git dependencies
- ✅ Dry-run publish succeeds

## CI/CD Workflows

### Documentation Workflow (`.github/workflows/docs.yml`)

Runs on every push/PR:
- Doc tests for publishable crates
- Documentation build checks
- README and CHANGELOG validation
- Cargo metadata validation

### Publish Workflow (`.github/workflows/publish.yml`)

Manual workflow for publishing individual crates:
- Select crate from dropdown
- Enable/disable dry-run mode
- Automatic pre-publish verification
- Publishes to crates.io when approved

### Release Crate Workflow (`.github/workflows/release-crate.yml`)

**Recommended for releases.** Triggered by version tags:
- Automatically triggered when you push a tag like `basilica-common-v0.1.0`
- Verifies version matches Cargo.toml
- Runs pre-publish checks
- Publishes to crates.io
- Creates GitHub release with changelog

**Usage:**
```bash
# Create and push a release tag
just release basilica-common 0.1.0
git push origin basilica-common-v0.1.0
```

### Release All Workflow (`.github/workflows/release-all.yml`)

Publishes all crates in dependency order:
- Manual trigger with dry-run option
- Skips already-published versions
- Publishes in correct dependency order (6 tiers)
- Waits for crates.io index between publishes

**Required secrets:**
- `CRATES_IO_TOKEN`: API token from crates.io

## Justfile Commands

```bash
# Build and open documentation
just docs
just docs-open

# Check documentation builds
just docs-check

# Run doc tests
just doc-test

# Pre-publish checks
just pre-publish-check basilica-common
just pre-publish-check-all

# Publishing (manual)
just publish-dry basilica-common  # Dry run
just publish basilica-common       # Actual publish
just publish-all-dry               # Dry run all crates

# Release automation (recommended)
just release basilica-common 0.1.0     # Create release tag
just release-push basilica-common 0.1.0 # Push tag to trigger CI
just release-status                     # Show all crate versions
just release-list                       # List existing release tags
just release-delete basilica-common 0.1.0  # Delete a tag if needed
```

## Version Management

Each crate has independent versioning following [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes

Update versions in:
1. `crates/<name>/Cargo.toml` - package version
2. `crates/<name>/CHANGELOG.md` - add release notes
3. All dependent crates' `Cargo.toml` - version specifier

## Post-Publish

After publishing a crate:

1. Tag the release: `git tag <crate>-v<version>`
2. Push the tag: `git push origin <crate>-v<version>`
3. Update CHANGELOG.md to add link to the release
4. Update dependent crates to use new version

