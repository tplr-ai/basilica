# Publishing to crates.io

This document describes the process and requirements for publishing Basilica crates to crates.io.

## Publishable Crates

The following crates are prepared for crates.io publication:

| Crate | Version | Status |
|-------|---------|--------|
| basilica-common | 0.1.0 | ✅ Ready |
| basilica-protocol | 0.1.0 | ✅ Ready |
| basilica-sdk | 0.10.0 | ✅ Ready |
| basilica-validator | 0.1.0 | ✅ Ready |
| basilica-miner | 0.1.0 | ✅ Ready |
| basilica-api | 0.1.0 | ✅ Ready |
| basilica-cli | 0.10.1 | ✅ Ready |
| basilica-aggregator | 0.1.0 | ✅ Ready |
| basilica-autoscaler | 0.1.0 | ✅ Ready |
| basilica-billing | 0.1.0 | ✅ Ready |
| basilica-operator | 0.1.0 | ✅ Ready |
| basilica-payments | 0.1.0 | ✅ Ready |
| basilica-storage | 0.1.0 | ✅ Ready |

## Dependencies

All dependencies are now available on crates.io:

```toml
# From Cargo.toml (workspace root)
bittensor = { version = "0.1.1", package = "bittensor-rs" }
```

The `bittensor-rs` crate is published at: https://crates.io/crates/bittensor-rs

## Publishing Order

Crates must be published in dependency order:

```
1. basilica-common
2. basilica-protocol
3. basilica-aggregator, basilica-storage, basilica-operator
4. basilica-validator, basilica-miner, basilica-billing, basilica-payments, basilica-autoscaler
5. basilica-sdk
6. basilica-api, basilica-cli
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

Manual workflow for publishing:
- Select crate from dropdown
- Enable/disable dry-run mode
- Automatic pre-publish verification
- Publishes to crates.io when approved

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

# Publishing
just publish-dry basilica-common  # Dry run
just publish basilica-common       # Actual publish
just publish-all-dry               # Dry run all crates
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

